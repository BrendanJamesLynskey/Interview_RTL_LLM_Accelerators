# Memory Bandwidth Bottlenecks in LLM Inference

## Overview

Memory bandwidth is the primary constraint on LLM inference performance at realistic deployment
scales. This file examines the sources of bandwidth demand, the limits imposed by HBM technology,
the specific contribution of weight streaming and KV cache access, and the architectural
strategies hardware teams use to mitigate the "memory wall."

**Key notation**

| Symbol | Meaning |
|---|---|
| $N$ | Non-embedding parameter count |
| $L$ | Number of transformer layers |
| $d$ | Model hidden dimension |
| $d_{ff}$ | FFN intermediate dimension |
| $h$ | Number of attention heads |
| $d_h = d/h$ | Per-head dimension |
| $B$ | Batch size |
| $S$ | Sequence length in KV cache |
| $\beta$ | Peak HBM bandwidth [Byte/s] |
| $\Pi$ | Peak compute throughput [FLOP/s] |
| $I^*$ | Ridge point $= \Pi/\beta$ [FLOP/Byte] |
| $\text{bpe}$ | Bytes per element (2 for FP16, 1 for INT8, 0.5 for INT4) |

---

## Tier 1 — Fundamentals

### Q1. What is the "memory wall" problem, and why does it affect LLM inference more severely than most other deep learning workloads?

**Answer**

**The memory wall** refers to the growing gap between processor compute throughput (scaling
with transistor density and voltage) and memory bandwidth (constrained by pin counts, PCB
routing, and DRAM physics). Since the 1990s, compute throughput has grown faster than
bandwidth, meaning the ridge point $I^* = \Pi/\beta$ has been increasing over time.

**Why LLMs are especially affected**:

During decode (autoregressive token generation), every forward pass is a series of
matrix-vector products. The arithmetic intensity of a GEMV is:

$$I_{\text{GEMV}} = \frac{2 \cdot m \cdot n}{2 \cdot m \cdot n \cdot \text{bpe}} = \frac{1}{\text{bpe}} \approx \begin{cases} 0.5 & \text{INT8} \\ 1 & \text{FP16} \end{cases}$$

Modern accelerators have ridge points of $I^* \approx 50\text{–}200$ FLOP/Byte. A GEMV has
intensity of order 1. The decode step therefore wastes $\sim 99\%$ of the compute budget
waiting for memory.

**Contrast with training or prefill**: Matrix-matrix products (GEMM) have intensity scaling
with the batch / sequence dimension. Training with large minibatches routinely achieves
$I \gg I^*$, keeping the hardware compute-bound. It is specifically the **sequential,
single-token nature of autoregressive decoding** that creates the memory wall problem for LLMs.

---

### Q2. What is HBM (High Bandwidth Memory) and why is it used in LLM accelerators? What are its practical bandwidth limits?

**Answer**

**HBM** stacks multiple DRAM dies vertically using through-silicon vias (TSVs) and connects
them to the processor through a silicon interposer. The wide parallel bus between the stack
and the processor (typically 1024 bits per stack) provides far higher bandwidth than conventional
GDDR.

**Why HBM for LLMs**:
1. **Bandwidth**: HBM3 delivers $\sim 800\text{ GB/s}$ per stack. An accelerator with 6 stacks
   achieves $\sim 4.8\text{ TB/s}$ — impossible with GDDR5/6 which tops out at $\sim 20\text{ GB/s}$ per chip.
2. **Capacity**: Current HBM3E stacks offer up to 36 GB per stack, enabling 288 GB on a
   6-stack device — necessary to hold large model weights on-chip.
3. **Energy efficiency**: HBM consumes less energy per bit transferred than GDDR due to the
   shorter physical path and wider, lower-voltage I/O.

**Practical limits and generational comparison**:

| Standard | BW per stack | Max stacks (typical) | Total BW | Capacity |
|---|---|---|---|---|
| HBM2 | ~256 GB/s | 4 | ~1 TB/s | 16 GB |
| HBM2E | ~460 GB/s | 6 | ~2.4 TB/s | 48 GB |
| HBM3 | ~819 GB/s | 6 | ~4.9 TB/s | 96 GB |
| HBM3E | ~1.2 TB/s | 8 | ~9.6 TB/s | 288 GB |

**Current bottleneck**: Even with HBM3E, a 70B-parameter model in FP16 requires loading
$140\text{ GB}$ of weights per decode step. At $9.6\text{ TB/s}$:
$$\text{TPOT}_{\min} = \frac{140\text{ GB}}{9.6\text{ TB/s}} \approx 14.6\text{ ms}$$

Weight quantisation to INT4 ($35\text{ GB}$) gives TPOT $\approx 3.6\text{ ms}$.

---

### Q3. Name the three main categories of memory bandwidth demand in LLM inference and give the bytes required for each per decode step.

**Answer**

**1. Weight loading** — loading all model weight matrices to perform the linear projections.

Per decode step (1 token, batch size $B$), the weights must be loaded from HBM once per step.
For the full model at precision $\text{bpe}$ bytes per parameter:

$$M_{\text{weights}} = N \cdot \text{bpe}$$

For a 7B model in FP16: $M_{\text{weights}} = 7 \times 10^9 \times 2 = 14\text{ GB}$.

**2. KV cache access** — loading all cached K and V tensors to compute attention over the context.

Per decode step, for $B$ sequences each with context length $S$:

$$M_{\text{KV}} = 2 \cdot L \cdot S \cdot d \cdot B \cdot \text{bpe}_{\text{KV}}$$

For LLaMA-2 7B ($L=32$, $d=4096$, $S=2048$, $B=16$, FP16):
$$M_{\text{KV}} = 2 \times 32 \times 2048 \times 4096 \times 16 \times 2 = 17.2\text{ GB}$$

**3. Activations** — intermediate tensors flowing between layers.

Activations are generated and immediately consumed; they occupy a small high-bandwidth
SRAM buffer in well-designed hardware. The DRAM bandwidth for activations is typically
negligible compared to weights and KV cache because activations are reused within the
same layer and then discarded.

**Summary at typical operating point** (7B FP16, $B=16$, $S=2048$):
- Weights: $14\text{ GB}$
- KV cache: $17.2\text{ GB}$
- Activations: $\ll 1\text{ GB}$
- **Total per step**: $\approx 31\text{ GB}$

---

## Tier 2 — Intermediate

### Q4. Derive the bandwidth utilisation for a single-request FP16 decode step as a function of accelerator specs. Then show how batch size improves utilisation.

**Answer**

**Single request, $B=1$**:

Per decode step, weights must be streamed: $M = 2N$ bytes.
Time to stream at bandwidth $\beta$: $t_{\text{BW}} = 2N/\beta$.
Time if compute were the bottleneck: $t_{\text{compute}} = 2N/\Pi$ (using the $2N$ FLOPs/token rule).

Since $I_{\text{decode}} = 1 \text{ FLOP/Byte} \ll I^* = \Pi/\beta$, we have $t_{\text{BW}} \gg t_{\text{compute}}$:

$$\text{Compute utilisation} = \frac{t_{\text{compute}}}{t_{\text{BW}}} = \frac{2N/\Pi}{2N/\beta} = \frac{\beta}{\Pi} = \frac{1}{I^*}$$

For $I^* = 93$: utilisation = $1/93 \approx 1.1\%$.

**Batch size $B$**:

With $B$ sequences processed together, weights are loaded once but used $B$ times.
Compute time: $t_{\text{compute}} = 2NB/\Pi$.
Bandwidth time (weights still loaded once): $t_{\text{BW}} = 2N/\beta$.

$$\text{Compute utilisation} = \frac{t_{\text{compute}}}{t_{\text{BW}}} = \frac{B}{I^*}$$

The compute utilisation reaches $100\%$ (ridge point) at $B = I^* = 93$.

**KV cache complicates the picture**: At large $B$ and $S$, KV bandwidth $M_{\text{KV}} = 4LSdB\cdot\text{bpe}$
grows with $B$ and eventually exceeds weight bandwidth. The effective batch-size benefit saturates
when KV loading time matches or exceeds weight loading time.

---

### Q5. Explain why INT8 weight quantisation approximately doubles decode throughput, but INT8 KV cache quantisation has a different (and context-dependent) impact.

**Answer**

**INT8 weight quantisation effect on throughput**:

Weight bytes loaded per decode step: $N \cdot \text{bpe}$.
- FP16: $2N$ bytes $\Rightarrow$ TPOT $= 2N/\beta$
- INT8: $N$ bytes $\Rightarrow$ TPOT $= N/\beta$

$$\text{Speedup from INT8 weights} = \frac{2N/\beta}{N/\beta} = 2\times$$

This is **always a $2\times$ TPOT improvement** for memory-bandwidth-bound decode, regardless
of context length, batch size, or model architecture. The improvement is pure and predictable.

**INT8 KV cache effect**:

KV cache bytes per decode step: $4LSdB\cdot\text{bpe}$.
- FP16: $4LSdB \times 2$ bytes
- INT8: $4LSdB \times 1$ byte

The speedup from INT8 KV is $2\times$ on the KV cache bandwidth. But the KV cache is only
one component of total bandwidth.

**Total TPOT with mixed precision** (INT8 weights, INT8 KV):
$$\text{TPOT} = \frac{N + 2LSdB}{\beta}$$

**KV-dominated regime** ($2LSdB \gg N$, i.e. $SB > 12d$):
INT8 KV gives close to $2\times$ total improvement.

**Weight-dominated regime** ($2LSdB \ll N$, i.e. $SB < 12d$):
INT8 KV gives near-zero improvement.

**Numerical example** (7B model, $L=32$, $d=4096$, $B=4$, $S=512$):
- $2LSdB = 2 \times 32 \times 4096 \times 512 \times 4 \times 2\text{B} = 1.07\text{ GB}$
- $N_{\text{weights}} = 7\text{GB}$ (INT8)
- KV fraction: $1.07/(7+1.07) = 13\%$ of bandwidth
- INT8 KV reduces KV bytes by half, saving $0.535\text{GB}$: a $\sim 6.6\%$ total improvement

At $B=64$, $S=2048$: KV bytes $= 34.4\text{GB} \gg 7\text{GB}$. INT8 KV now dominates and nearly doubles throughput.

---

### Q6. What is the "working set" concept in memory hierarchy design, and how does it apply to LLM accelerator SRAM sizing?

**Answer**

**Working set**: The set of data items that must be simultaneously resident in fast memory
(on-chip SRAM / L1-L2 cache) to avoid repeated slow-memory (HBM/DRAM) accesses during a
computation.

**Applied to LLM accelerator layers**:

For an FFN down-projection $W \in \mathbb{R}^{d \times d_{ff}}$:
- Weight tensor size (FP16): $2 \times 4096 \times 11008 = 90.2\text{ MB}$ (7B LLaMA)
- If the SRAM can hold this entire tensor, the weight is loaded once and reused for every
  decode step in a persistent batch, effectively giving $R$ FLOP/Byte (where $R$ = number of steps).

**Tiled execution** (when working set does not fit):

Decompose $W$ into tiles $W_{ij} \in \mathbb{R}^{t_r \times t_c}$. The on-chip working set
per tile is $2 t_r t_c$ bytes (weights) plus $2 B t_r$ bytes (output accumulator) plus
$2 B t_c$ bytes (input activation slice).

For SRAM of size $M_{\text{SRAM}}$, the maximum tile area:
$$t_r \cdot t_c \leq \frac{M_{\text{SRAM}} - 2B(t_r + t_c)}{2}$$

Each tile is loaded once from HBM and used $B$ times: effective intensity of tile $= B$.
To achieve compute-bound execution: $B \geq I^*$, so tile-level execution requires the same
batch-size threshold regardless of tile size.

**Where larger SRAM helps**:
1. **KV cache for the active layer**: At $S=4096$ per sequence, per-layer KV is $64\text{MB}$ (FP16).
   If this fits in SRAM, the attention step requires no HBM access for KV.
2. **Activation buffering**: Holding all layer activations in SRAM eliminates
   intermediate activation saves/loads.
3. **Persistent weight residence**: If the entire model fits in SRAM (unrealistic today
   for large models), decode becomes trivially compute-bound.

**Rule of thumb**: SRAM should be sized to hold at minimum: one attention layer's KV cache
per sequence at the target context length, plus one weight tile per active compute engine.

---

### Q7. Compare the effective memory bandwidth requirements of three different attention mechanisms: standard MHA, MQA, and GQA-8 for a 70B-parameter model at long context.

**Answer**

**Model config** (LLaMA-like 70B): $L=80$, $d=8192$, $h=64$ heads, $d_h=128$, FP16.

**KV cache bytes per token per layer**:

- **MHA** ($h_K = h_V = 64$): $2 \times 64 \times 128 \times 2\text{B} = 32,768\text{B} = 32\text{KB}$
- **MQA** ($h_K = h_V = 1$): $2 \times 1 \times 128 \times 2\text{B} = 512\text{B}$
- **GQA-8** ($h_K = h_V = 8$): $2 \times 8 \times 128 \times 2\text{B} = 4,096\text{B} = 4\text{KB}$

**KV cache bandwidth per decode step**, $B=1$ sequence, $S=8192$ tokens:

$$M_{\text{KV/layer}} = \text{bytes/token/layer} \times S$$

Over $L=80$ layers:
- **MHA**: $32\text{KB} \times 8192 \times 80 = 20.97\text{GB}$
- **MQA**: $512\text{B} \times 8192 \times 80 = 327\text{MB}$
- **GQA-8**: $4\text{KB} \times 8192 \times 80 = 2.62\text{GB}$

**Weight bandwidth** (INT8, 70B): $70\text{GB}$.

**Total memory bandwidth comparison** (single request, $S=8192$):

| Mechanism | KV BW | Weight BW | Total | KV fraction |
|---|---|---|---|---|
| MHA | 21.0 GB | 70 GB | 91.0 GB | 23% |
| GQA-8 | 2.6 GB | 70 GB | 72.6 GB | 3.6% |
| MQA | 0.33 GB | 70 GB | 70.3 GB | 0.5% |

At this context length, MHA's KV overhead is significant (~23%). GQA-8 reduces it to 3.6%.
For very long contexts ($S=32768$) or large batches the KV fraction grows proportionally.

**RTL implication**: GQA/MQA dramatically simplify the KV cache memory controller — fewer
distinct K/V tensors to manage, smaller address spaces, and simpler broadcast logic from the
single K/V head to all Q heads. The price is a potential reduction in model quality for a
given parameter count.

---

## Tier 3 — Advanced

### Q8. Analyse the memory bandwidth required for a flash-attention style tiled attention computation. How does tiling affect the HBM bandwidth consumption for the attention mechanism?

**Answer**

**Standard (non-tiled) attention bandwidth** during prefill ($S$ tokens):

The naive implementation materialises the $S \times S$ attention score matrix:
- Write scores: $2 \times S^2 \times h$ bytes (FP16, $h$ heads)
- Read scores for softmax: $2 \times S^2 \times h$ bytes
- Total for scores: $4 S^2 h$ bytes

Plus Q, K, V loading and output O writing:
$$M_{\text{naive}} = 4S^2 h + 4 \times S \times d \times 2 = 4S^2 h + 8Sd$$

For $S=2048$, $h=32$, $d=4096$: $M_{\text{naive}} = 4 \times 2048^2 \times 32 + 8 \times 2048 \times 4096 = 536.9\text{MB} + 67.1\text{MB} \approx 604\text{MB}$.

The $S^2$ term dominates at long sequences.

**FlashAttention tiled bandwidth**:

Tile Q, K, V into blocks of size $B_r \times d_h$ and $B_c \times d_h$. Process each tile
completely on-chip, writing only the final output $O$ back to HBM. The $S \times S$ matrix
is **never materialised**.

HBM accesses:
- Read Q, K, V once each: $3 \times 2 \times S \times d = 6Sd$ bytes
- Write O once: $2 \times S \times d$ bytes
$$M_{\text{flash}} = 8Sd$$

For same parameters: $M_{\text{flash}} = 8 \times 2048 \times 4096 = 67.1\text{MB}$.

**Reduction factor**:
$$\frac{M_{\text{naive}}}{M_{\text{flash}}} = \frac{4S^2 h + 8Sd}{8Sd} = \frac{Sh}{2d} + 1$$

For large $S$: $\approx \frac{Sh}{2d} = \frac{2048 \times 32}{2 \times 4096} = 8\times$.

At $S=8192$: $\approx 32\times$ bandwidth reduction.

**RTL requirements for FlashAttention**:
1. **On-chip tile buffers** sized $\geq B_r \times d_h + B_c \times d_h$ per head, all in SRAM.
2. **Running softmax normaliser** registers — the online softmax algorithm requires
   maintaining running $m$ (max) and $\ell$ (normalisation factor) per query tile.
3. **Fused kernel execution** — score, softmax, and value-weighting in a single pass,
   requiring the datapath to feed back the normaliser without intermediate HBM stores.
4. **Tile scheduling logic** — outer loop over Q tiles, inner loop over K,V tiles, with
   configurable tile sizes based on available SRAM.

---

### Q9. A new HBM generation doubles bandwidth from 2 TB/s to 4 TB/s. For which inference configurations does this provide a 2× throughput improvement, and for which does it provide much less?

**Answer**

**The roofline tells us**: Throughput improvement $=2\times$ only when the workload was
**at the bandwidth roof** before the upgrade and **remains at the bandwidth roof** after.

**Case 1: Single-request decode (weight-bound)**

Before: TPOT $= 2N/\beta_1$, bandwidth-bound since $I=1 \ll I^*_1 = \Pi/\beta_1$.
After: TPOT $= 2N/\beta_2 = 2N/(2\beta_1)$ — exactly $2\times$ faster, still bandwidth-bound
since $I=1 \ll I^*_2 = \Pi/(2\beta_1)$ (note: $I^*$ is halved, so ridge point is now lower,
making more workloads compute-bound — but GEMV is still memory-bound).

**Result: $2\times$ throughput improvement.**

**Case 2: Large-batch decode already at ridge point**

Suppose $B = I^*_1 = \Pi/\beta_1$, so the workload is exactly at the ridge point (compute-bound).
After bandwidth doubling: $I^*_2 = \Pi/(2\beta_1) = I^*_1/2$. Now $B = 2I^*_2$, still above
the new ridge point — still compute-bound.

**Result: $0\times$ throughput improvement** (already compute-bound, bandwidth doubling is irrelevant).

**Case 3: Prefill**

Prefill is compute-bound ($I = S \gg I^*$). Doubling bandwidth does not change $\Pi$.

**Result: $0\times$ throughput improvement.**

**Case 4: KV-cache-bound large-batch long-context decode**

When KV bandwidth $\gg$ weight bandwidth ($S \cdot B > 12d$), total bandwidth demand scales
with both weight and KV. If the total demand is exactly $2\times$ the old $\beta$:

$$\text{TPOT} = \frac{2N + 4LSdB \cdot \text{bpe}}{\beta}$$

Doubling $\beta$ gives exactly $2\times$ improvement, provided the workload remains bandwidth-bound.

**Summary table**:

| Configuration | Bottleneck | Benefit of $2\times$ BW |
|---|---|---|
| Single-request decode | Weight BW | $2\times$ |
| Prefill | Compute | $0\times$ |
| Large-batch decode ($B \geq I^*$) | Compute | $0\times$ |
| Long-context, large-batch decode | KV+Weight BW | Up to $2\times$ |
| Decode with model fully in SRAM | Compute | $0\times$ |

---

### Q10. Describe three architectural strategies that hardware teams use to improve effective memory bandwidth utilisation in LLM decode, and evaluate the trade-offs of each.

**Answer**

**Strategy 1: Compression — Weight and KV quantisation**

Reduce the bytes-per-element of weights and/or KV cache.

- **How it helps**: Fewer bytes loaded per decode step, directly reducing $M_{\text{weights}}$
  and $M_{\text{KV}}$. INT4 weights give $4\times$ bandwidth reduction.
- **RTL implication**: Requires dequantisation hardware (lookup tables or multipliers for
  scale/zero-point) inline with the weight fetch path. The dequantisation latency must be
  hidden by pipelining.
- **Trade-offs**:
  - Accuracy loss, especially below INT8 for weights and below FP8 for KV.
  - Accumulator must be wider than the input precision to avoid overflow (e.g., INT4 inputs
    accumulate in INT32).
  - Mixed-precision datapaths (INT4 weights, FP16 activations) complicate the MAC array.

**Strategy 2: Compute reuse — Weight stationary dataflow with persistent batching**

Keep weight tiles resident in on-chip SRAM and process many decode steps per weight load.

- **How it helps**: If a weight tile of size $t_r \times t_c$ stays in SRAM for $R$ decode
  steps (batch of $R$ requests), effective intensity is $R$ rather than 1.
- **RTL implication**: Requires a large SRAM (potentially tens of MB per compute cluster),
  SRAM arbitration between weight residency and KV cache, and a scheduling engine that
  groups requests whose weight access patterns align.
- **Trade-offs**:
  - SRAM area is expensive (6T SRAM ~40× area of DRAM per bit).
  - Only helps if $R \geq I^*$; for small $I^*$ (bandwidth-efficient chips), this is easier to achieve.
  - Model must fit in the SRAM hierarchy or tiling degrades back to DRAM-bound.

**Strategy 3: Access scheduling — HBM bank-level parallelism and prefetching**

Exploit HBM's internal parallelism (multiple banks, channels, pseudochannels) to maximise
sustained bandwidth utilisation.

- **How it helps**: Naive sequential weight loads may under-utilise HBM due to row-buffer
  misses, bank conflicts, or command queue stalls. Intelligent scheduling can sustain
  bandwidth closer to the theoretical peak.
- **RTL implication**: Requires a specialised memory controller with:
  - Address interleaving across banks/channels (weight matrix stored in striped layout)
  - Command queue depth sufficient to pipeline row-activates and column-reads
  - Prefetch buffers that issue the next weight tile request while the current tile is being consumed
- **Trade-offs**:
  - Complex memory controller RTL; difficult to verify.
  - Benefits depend heavily on DRAM timing parameters and workload access patterns.
  - Interleaving layout must be known at compile time (model weights packed offline).
  - Modest improvement ceiling: can close the gap between achieved and theoretical peak
    bandwidth, but cannot exceed it (peak is a hard physical limit).

**Comparing the strategies**:

| Strategy | Bandwidth reduction | RTL complexity | Accuracy risk | Best for |
|---|---|---|---|---|
| INT8 quantisation | $2\times$ | Medium | Low | All decode |
| INT4 quantisation | $4\times$ | High | Medium | Throughput-critical |
| FP8 KV cache | $2\times$ KV | Low | Low | Long-context |
| Weight SRAM residency | Up to $R\times$ | High | None | Low-latency single-user |
| HBM scheduling | $1.1\text{–}1.3\times$ | Medium | None | All workloads |

In practice, production chips combine all three: quantised weights reduce bytes moved, SRAM
holds hot activations, and the memory controller is optimised for sequential streaming access.
