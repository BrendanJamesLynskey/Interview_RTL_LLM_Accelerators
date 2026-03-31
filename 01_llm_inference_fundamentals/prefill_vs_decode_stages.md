# Prefill vs. Decode Stages

## Overview

LLM inference is not a single uniform workload. It splits into two fundamentally different
computational phases with opposite hardware characteristics. Getting this distinction right is
central to every accelerator design decision: dataflow scheduling, buffer allocation, memory
controller design, power delivery, and SLA trade-offs.

**Terminology used throughout**

| Symbol | Meaning |
|---|---|
| $P$ | Prompt length (prefill tokens) |
| $G$ | Number of tokens to generate |
| $B$ | Batch size |
| $d$ | Model hidden dimension |
| $d_{ff}$ | FFN intermediate dimension |
| $L$ | Number of layers |
| $N$ | Non-embedding parameter count |
| $\Pi$ | Peak compute throughput [FLOP/s] |
| $\beta$ | Peak memory bandwidth [Byte/s] |
| $I$ | Arithmetic intensity [FLOP/Byte] |
| $I^*$ | Ridge-point intensity $= \Pi/\beta$ |

---

## Tier 1 — Fundamentals

### Q1. Describe the prefill phase. What is computed, and why is it done all at once?

**Answer**

**What happens**: The prefill phase processes the entire input prompt (all $P$ tokens)
in a single forward pass through the model. For every transformer layer it computes:
- Q, K, V projections for all $P$ positions simultaneously
- The full $P \times P$ attention score matrix
- The FFN for all $P$ positions
- Stores the resulting K and V tensors in the KV cache for later use during decode

**Why all at once**: Because the prompt tokens are all known at the start, there is no
causal dependency that prevents processing them in parallel. The attention mechanism
requires each token to attend to all previous tokens, but since all prompt tokens are
already available, the full attention matrix can be computed in a single batched matrix
operation.

**From a hardware perspective**: The compute engine sees large matrix-matrix multiply
(GEMM) problems. The weight matrices are reused across all $P$ tokens in the same operation,
amortising the cost of loading them from HBM:

$$I_{\text{prefill}} \approx P \quad \text{[FLOP/Byte, FP16]}$$

For $P = 512$ and a ridge point of $I^* = 93$, we have $I_{\text{prefill}} \gg I^*$:
the operation is **compute-bound** and the systolic array is well-utilised.

---

### Q2. Describe the decode phase. Why must tokens be generated one at a time?

**Answer**

**What happens**: After prefill, the model generates new tokens autoregressively. At each
step, it takes the single most recently generated token as input, runs it through all
transformer layers, and produces a probability distribution over the vocabulary from which
the next token is sampled or selected.

**Why sequential**: The causal attention mask means that token $t$ must attend to all
tokens $0 \ldots t-1$ (including the prompt and all previously generated tokens). Token
$t+1$ depends on the output of step $t$, which is only known after step $t$ completes.
This is a **data dependency** that cannot be parallelised away in standard autoregressive
decoding.

**From a hardware perspective**: Each decode step processes a single token vector through
the model. Every linear layer reduces to a matrix-vector multiply (GEMV):

$$y = Wx, \quad W \in \mathbb{R}^{d_{out} \times d_{in}}, \quad x \in \mathbb{R}^{d_{in}}$$

The entire weight matrix $W$ must be loaded from HBM to compute one output vector:

$$I_{\text{decode, single}} = \frac{2 \cdot d_{out} \cdot d_{in}}{2 \cdot d_{out} \cdot d_{in}} = 1 \quad \text{FLOP/Byte (FP16)}$$

This is $\sim 90\times$ below the ridge point — the accelerator is severely underutilised,
spending most of its time waiting for memory rather than computing.

---

### Q3. What is Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT)? Which phase determines each metric?

**Answer**

**TTFT (Time-To-First-Token)**: The latency from when the user submits a request to when
the first output token is returned. This is determined almost entirely by the **prefill phase**:
$$\text{TTFT} \approx \frac{2NP}{\Pi} + \frac{2LP \cdot P \cdot d}{\Pi} \quad \text{(compute-bound)}$$
The dominant cost is proportional to $P$ (prompt length). Long prompts dramatically increase TTFT.

**TPOT (Time-Per-Output-Token)**: The time to generate each subsequent token after the first.
This is determined by the **decode phase**:
$$\text{TPOT} \approx \frac{2N}{\beta} \quad \text{(memory-bandwidth-bound, single request)}$$
TPOT is approximately constant per token (for short contexts relative to $d$) and is governed
by how fast the model weights can be streamed from HBM.

**Example** (LLaMA-2 7B on an A100-80GB: $\Pi=312\text{ TFLOP/s}$ BF16, $\beta=2\text{ TB/s}$):
- $N \approx 6.7\text{B}$ parameters
- TPOT $\approx \frac{2 \times 6.7 \times 10^9 \times 2\text{B}}{2 \times 10^{12}\text{B/s}} \approx 13.4\text{ ms/token}$ (single request)
- At batch $B=64$: weights amortised over $B$ tokens, TPOT $\approx 0.21\text{ ms/token}$

**Design implication**: TTFT and TPOT often conflict. To minimise TTFT, you want maximum
compute throughput. To minimise TPOT at low batch sizes, you want maximum memory bandwidth.
These are different hardware optimisation targets.

---

### Q4. Why is the prefill phase described as "compute-bound" and the decode phase as "memory-bandwidth-bound"?

**Answer**

The roofline model gives the attainable performance as:
$$\text{Perf} = \min(\Pi, \; \beta \cdot I)$$

**Prefill is compute-bound** because with $P$ tokens processed together:
- All weight matrices are loaded once and reused $P$ times (once per token)
- Arithmetic intensity $\approx P \gg I^*$ (ridge point)
- The compute engines ($\Pi$) are the limiting resource
- Adding more memory bandwidth does not help; adding more MACs does

**Decode is memory-bandwidth-bound** because with 1 token per step:
- All weight matrices are loaded from HBM once but used for only 1 MAC per weight
- Arithmetic intensity $\approx 1 \ll I^*$
- The memory controller ($\beta$) is the limiting resource
- Adding more compute units does not help; adding more HBM bandwidth or reducing model size does

**Practical consequence**: An accelerator optimal for prefill (high $\Pi$, moderate $\beta$)
will be a poor fit for decode, and vice versa. This motivates disaggregated prefill/decode
architectures where different hardware handles each phase.

---

## Tier 2 — Intermediate

### Q5. Derive the minimum achievable TPOT for a single-request decode step on a given accelerator, and show how it relates to the weight byte count.

**Answer**

**Single-request decode bottleneck**: During a decode step, the model must load every
weight parameter from HBM at least once to produce one output token. There is no way to
avoid this: every linear layer must multiply its weights by the current token's hidden state.

**Minimum time to stream all weights**:
$$T_{\text{min}} = \frac{W_{\text{bytes}}}{\beta}$$

where $W_{\text{bytes}}$ is total weight size in bytes and $\beta$ is peak HBM bandwidth.

For FP16 ($2$ bytes per parameter):
$$T_{\text{min}} = \frac{2N}{\beta}$$

**Derived from first principles for one layer** (e.g., FFN down-projection,
$W \in \mathbb{R}^{d \times d_{ff}}$):
- FLOPs: $2 \cdot d \cdot d_{ff}$
- Bytes: $2 \cdot d \cdot d_{ff}$ (FP16, loaded once)
- Time at bandwidth $\beta$: $\frac{2d \cdot d_{ff}}{\beta}$
- Time at peak compute $\Pi$: $\frac{2d \cdot d_{ff}}{\Pi}$
- Since $I=1 \ll I^*$, we are bandwidth-bound: actual time $= \frac{2d \cdot d_{ff}}{\beta}$

**Summing over all layers and all weight matrices** gives:
$$\text{TPOT}_{\min} = \frac{2N}{\beta}$$

**This is a hard lower bound** regardless of compute throughput. To halve TPOT at fixed $\beta$,
you must halve $N$ (e.g., via quantisation or model distillation) or double $\beta$ (e.g., more
HBM stacks, higher-generation HBM).

**Numerical check** (LLaMA-2 7B, A100 SXM5 $\beta = 2\text{ TB/s}$, FP16):
$$\text{TPOT}_{\min} = \frac{2 \times 6.7 \times 10^9 \times 2}{2 \times 10^{12}} \approx 13.4\text{ ms}$$

---

### Q6. What is "continuous batching" (also called iteration-level scheduling)? How does it differ from static batching, and what problem does it solve?

**Answer**

**Static batching**: All requests in a batch must complete before any new requests are
admitted. If request A finishes generating at step 50 and request B is still generating
at step 200, the compute engine idles or wastes work on padding for those extra 150 steps.
GPU utilisation is low for variable-length workloads.

**Continuous batching**: At every decode iteration (every token step), completed sequences
are evicted from the batch and new sequences are inserted to fill the vacant slots. The
batch composition changes dynamically at every step.

**How it solves the problem**:
1. **Eliminates padding waste**: No sequence occupies a batch slot without contributing
   useful work.
2. **Reduces queue wait time**: New requests enter the system as soon as a slot frees,
   rather than waiting for the entire batch to complete.
3. **Improves throughput**: Sustained high batch size maintains high arithmetic intensity
   throughout the generation window.

**Hardware complications introduced**:
- **Non-uniform KV cache sizes**: Each sequence has a different context length at any given
  step. The KV cache manager must handle variable-length allocations efficiently (motivating
  PagedAttention-style virtual memory for KV).
- **Irregular memory access**: The attention engine must handle per-sequence context lengths
  that differ, requiring flexible addressing rather than fixed-stride access patterns.
- **Prefill/decode mixing**: A newly admitted request must prefill its prompt while other
  sequences are decoding. These have different arithmetic intensities and memory access
  patterns, making scheduling complex.

---

### Q7. Explain the KV cache growth problem during decode and its memory consequence. How does it interact with the choice of batch size?

**Answer**

**KV cache growth**: At each decode step, the model appends one new K,V vector per layer
per head to the KV cache. After generating $g$ tokens, the total KV cache size for one
sequence is:

$$M_{\text{KV}}(g) = 2 \cdot L \cdot S_{\text{total}} \cdot d \cdot \text{bytes}$$

where $S_{\text{total}} = P + g$ (prompt + generated tokens) and the factor 2 is for K and V.

For LLaMA-2 7B ($L=32$, $d=4096$, FP16):
$$M_{\text{KV}}(g) = 2 \times 32 \times (P+g) \times 4096 \times 2 = 524,288 \times (P+g)\text{ bytes}$$

At $P+g = 2048$: $M_{\text{KV}} \approx 1.07\text{GB}$ per sequence.
At $P+g = 8192$: $M_{\text{KV}} \approx 4.29\text{GB}$ per sequence.

**Interaction with batch size**: Total HBM = weights + activations + KV caches. With a
7B model at FP16 using ~14 GB for weights:
$$B_{\max}(S) = \frac{\text{HBM} - 14\text{GB}}{M_{\text{KV}}(S)}$$

For an 80 GB GPU at $S=2048$: $B_{\max} \approx (80-14)/1.07 \approx 61$ sequences.
For an 80 GB GPU at $S=8192$: $B_{\max} \approx (80-14)/4.29 \approx 15$ sequences.

**This creates a direct trade-off**: Supporting longer sequences shrinks maximum batch size,
reducing throughput and worsening memory-bandwidth utilisation. Hardware architects respond
with:
- KV cache quantisation (INT8 or FP8 KV cache reduces size by 2-4×)
- GQA/MQA to reduce KV head count
- KV cache offloading to cheaper DRAM with prefetching
- Sparse attention (not all tokens attend to all past tokens)

---

### Q8. What is speculative decoding and how does it change the prefill/decode balance?

**Answer**

**Problem it solves**: Standard autoregressive decode generates one token per forward pass
of the full model. The large model is memory-bandwidth-bound at batch size 1; the hardware
is under-utilised.

**Mechanism**:
1. A small **draft model** (or self-speculative method) generates $k$ candidate tokens quickly.
2. The full **target model** verifies all $k$ candidates in a single forward pass — this
   is effectively a prefill of length $k$, which is compute-bound and hardware-efficient.
3. The longest accepted prefix is kept; the first rejected token triggers re-generation.

**Change to the compute profile**:
- **Without speculative decoding**: $G$ decode steps, each memory-bandwidth-bound.
- **With speculative decoding**: $G/\alpha$ target-model forward passes, each processing
  $\sim k$ tokens (compute-bound if $k$ is large enough), plus $G$ draft-model forward passes
  (cheap). Here $\alpha$ is the average accepted token count per verification pass.

**Arithmetic intensity of the verification step**:
$$I_{\text{verify}} \approx k \quad \text{FLOP/Byte}$$

For $k=5$ and $I^*=93$, the verification pass is still memory-bound for single requests.
But for batched speculative decoding, the effective batch size seen by compute is $B \times k$,
pushing intensity toward the ridge point.

**Hardware trade-off**: Speculative decoding requires simultaneously running two model sizes
on the accelerator, increasing memory pressure. The draft model must be small enough that
its overhead is negligible, yet capable enough to achieve high acceptance rate $\alpha$.
RTL implications include: dual-model weight management, flexible batch scheduling between
draft and verify phases, and KV cache management for both models.

---

## Tier 3 — Advanced

### Q9. Design a disaggregated prefill/decode system at the hardware level. What interface does the prefill unit expose to the decode unit? What are the bandwidth requirements of that interface?

**Answer**

**Motivation for disaggregation**: Prefill and decode have opposing hardware affinities.
Prefill wants high compute density (systolic arrays, high FLOP/s per die area). Decode
wants high memory bandwidth (many HBM stacks, fast weight-streaming). Running them on the
same hardware means each phase runs at reduced efficiency.

**Disaggregated architecture**:
- **Prefill cluster**: Compute-optimised chips (e.g., high-FLOP/s ASICs with standard HBM).
  Handles all prompt processing.
- **Decode cluster**: Bandwidth-optimised chips (e.g., fewer MACs but more HBM stacks or
  on-chip SRAM). Handles token generation.

**Interface: KV cache transfer**

After prefill, the prefill unit must send the generated KV cache to the decode unit. This
is the only required inter-unit communication.

**Bandwidth requirement**: After processing a prompt of length $P$:

$$M_{\text{KV transfer}} = 2 \cdot L \cdot P \cdot d \cdot \text{bytes\_per\_elem}$$

For $L=32$, $d=4096$, $P=1024$, FP16:
$$M_{\text{KV transfer}} = 2 \times 32 \times 1024 \times 4096 \times 2 = 536\text{ MB}$$

If the prefill unit must achieve TTFT $\leq T_{\text{target}}$ and the transfer must complete
before decode can start, the required inter-chip bandwidth is:

$$\beta_{\text{link}} \geq \frac{536\text{ MB}}{T_{\text{target}} - T_{\text{prefill}}}$$

For $T_{\text{target}} = 200\text{ms}$ and $T_{\text{prefill}} = 100\text{ms}$:
$\beta_{\text{link}} \geq 5.36\text{ GB/s}$ — achievable with a single PCIe 4.0 x16 link
($\sim 32\text{ GB/s}$) or NVLink.

**Additional considerations**:
- **Token streaming**: To minimise TTFT, the decode unit should begin generating as soon as
  the last prefill layer's KV cache is sent. This requires pipelining the KV transfer
  layer-by-layer across the link.
- **KV quantisation at the boundary**: Sending FP8 or INT8 KV tensors halves/quarters
  link bandwidth requirements.
- **Routing**: In a multi-request system, the scheduler must route a request's KV cache to
  the specific decode chip where its slot resides.

---

### Q10. Analyse the interaction between prefill chunking (also called chunked prefill) and decode batching. How does this affect hardware scheduling and what does the RTL scheduler need to support?

**Answer**

**Chunked prefill**: Instead of processing the full prompt in one step, the prefill is
split into chunks of $C$ tokens. Each chunk is processed as a mini-prefill step, interleaved
with decode steps from other requests.

**Why this is beneficial**:
- **Latency fairness**: A long prefill (e.g., $P=8192$ tokens) would block all decode requests
  for a substantial time, increasing TPOT for already-running requests. Chunking limits the
  maximum latency spike.
- **Resource mixing**: By mixing prefill chunks and decode tokens in a single forward pass,
  we can craft a batch with a target total token count $T$ that keeps the compute engine
  at a chosen utilisation point.

**Arithmetic intensity of a mixed batch**:

Let the batch contain $C$ prefill tokens and $B_d$ decode tokens (one per decode sequence).
Total tokens processed: $T = C + B_d$.
For weight projections, the intensity is:
$$I_{\text{mixed}} = T = C + B_d \quad \text{FLOP/Byte (FP16)}$$

**RTL scheduling requirements**:

1. **Variable sequence length attention**: The attention engine must handle a batch where
   some sequences have full context (decode sequences) and one sequence is being filled
   in (the chunked prefill sequence). This requires per-sequence context-length registers
   and flexible mask generation.

2. **Causal mask heterogeneity**: Decode tokens use a causal mask against all prior tokens.
   The prefill chunk uses a triangular mask within the chunk. The hardware must apply
   different mask logic per token type within the same matrix operation.

3. **KV cache write vs. read arbitration**: During a mixed step, decode sequences read from
   existing KV cache entries while the prefill sequence writes new entries. The KV cache
   memory controller must handle simultaneous read and write addresses without bank conflicts.

4. **Priority scheduling registers**: The RTL scheduler needs configurable priority weights
   to decide how large the prefill chunk $C$ should be relative to batch size $B_d$,
   controlling the TTFT/TPOT trade-off at the hardware scheduler level.

---

### Q11. Quantify the impact of Key-Value cache bandwidth on total decode throughput. At what model size and sequence length does KV bandwidth dominate over weight-loading bandwidth?

**Answer**

**Two bandwidth consumers in decode**:

1. **Weight loading**: Load all $N$ parameters once per decode step.
   Bandwidth demand: $2N$ bytes per step (FP16).

2. **KV cache access**: For the attention computation, load all past K,V vectors for every
   sequence in the batch. Per decode step, per sequence of length $S$:
   $$M_{\text{KV/seq}} = 2 \cdot L \cdot S \cdot d \cdot 2\text{B} = 4LSd \text{ bytes}$$
   For a batch of $B_d$ sequences:
   $$M_{\text{KV/step}} = 4 L S d B_d \text{ bytes}$$

**Crossover condition**: KV bandwidth exceeds weight bandwidth when:
$$4 L S d B_d > 2N$$

Substituting $N \approx 24Ld^2$ (from the $\sim 24d^2$ per-layer approximation summed over $L$):
$$4 L S d B_d > 48 L d^2$$
$$S B_d > 12 d$$

**Crossover boundary**: $S B_d = 12d$.

For $d=4096$: crossover at $S \cdot B_d = 49,152$.

| Scenario | $S$ | $B_d$ | $SB_d$ | Dominant bandwidth source |
|---|---|---|---|---|
| Single request, short ctx | 512 | 1 | 512 | Weights |
| Small batch, medium ctx | 2048 | 8 | 16,384 | Weights |
| Medium batch, long ctx | 4096 | 16 | 65,536 | **KV cache** |
| Large batch, max ctx | 8192 | 32 | 262,144 | **KV cache** |

**RTL implication**: Once KV bandwidth dominates, the standard optimisation of reducing
weight precision (e.g., INT4 weights) has diminishing returns. The next lever is
**KV cache quantisation** — reducing KV from FP16 to INT8 or FP8 halves the KV bandwidth
demand and shifts the crossover to $S B_d = 24d$.

**Architectural response**: Some accelerators use a tiered memory architecture: hot KV
entries (recent tokens) in on-chip SRAM, cold entries in HBM, with speculative prefetching
of the KV cache for the expected next decode step. This hides KV latency behind the
compute of the current step.

---

## Phase Comparison Summary

| Property | Prefill | Decode |
|---|---|---|
| Tokens processed per step | $P$ (all prompt tokens) | 1 (or $B_d$ with batching) |
| Key operation type | GEMM (matrix-matrix) | GEMV (matrix-vector) |
| Arithmetic intensity | $\sim P$ FLOP/Byte | $\sim 1$ FLOP/Byte (single) |
| Hardware bottleneck | Compute ($\Pi$) | Memory bandwidth ($\beta$) |
| Systolic array utilisation | High | Low (unless large batch) |
| KV cache role | Write (populate cache) | Read (attend to context) |
| Latency sensitivity | Determines TTFT | Determines TPOT |
| Parallelisation | Across tokens (trivially) | Only across independent requests |
| Scaling with sequence length | $O(P^2)$ FLOPs total | $O(S)$ bandwidth per step |
| Optimised by | Compute throughput, tensor parallelism | Memory bandwidth, weight quantisation, batching |
