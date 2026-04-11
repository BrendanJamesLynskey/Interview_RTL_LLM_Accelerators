# End-to-End Inference Pipeline

## Overview

LLM inference is not a single monolithic computation. It consists of a prefill phase (processing
the prompt in parallel), a decode phase (generating tokens one at a time), and the scheduling
logic that batches, interleaves, and prioritises many concurrent requests. Understanding how these
phases map to hardware resources — and the tradeoffs between throughput and latency — is essential
for designing or evaluating an LLM accelerator system.

---

## Tier 1: Fundamentals

### Q1. Explain the prefill and decode phases of LLM inference. Why are they computationally
different?

**Answer.**

**Prefill phase:**
The model processes the input prompt tokens $[t_0, t_1, \ldots, t_{S-1}]$ in a single forward
pass. Because all $S$ tokens are available simultaneously, attention can be computed over the full
sequence in parallel (using a causal mask to prevent future tokens from attending to past ones).

The dominant computation is matrix-matrix multiplication (GEMM):
- For each transformer layer: $[B \times S, d] \times [d, 4d]$ for the FFN, where $B$ is batch size.
- Arithmetic intensity: $\frac{2 \times B \times S \times d \times 4d}{B \times S \times d \times 2 + d \times 4d \times 2} \approx \frac{8d^2}{d^2 \cdot 2} = 4S$ for large $S$.
- At $S = 512$, arithmetic intensity is ~2048 FLOPs/byte — **compute-bound** on a modern
  accelerator with 300 TOPS / 3.2 TB/s roofline at 94 FLOPs/byte.

**Decode phase:**
After prefill, the model generates one new token per step. At each step, only a single new token
(or $B$ tokens for a batch of $B$ sequences) is processed, with attention computed over all
previously generated tokens (via the KV cache).

The dominant computation is matrix-vector multiplication (GEMV):
$$Y = W \cdot x, \quad W \in \mathbb{R}^{d_{out} \times d_{in}},\quad x \in \mathbb{R}^{d_{in}}$$

- Arithmetic intensity: $\frac{2 \times d_{out} \times d_{in}}{d_{out} \times d_{in} \times 2} = 1$ FLOP/byte for $B = 1$.
- Even for $B = 32$: intensity = 32 FLOPs/byte.
- Far below the roofline crossover of 94 FLOPs/byte — **memory-bandwidth-bound**.

**Summary:**

| Property | Prefill | Decode |
|---|---|---|
| Compute pattern | GEMM | GEMV |
| Bottleneck | Compute (TOPS) | Memory bandwidth (TB/s) |
| Tokens processed per step | $B \times S$ | $B$ |
| Parallelism | Sequence parallelism | Batch parallelism |
| KV cache | Written | Read + extended |

---

### Q2. What is the KV cache, and why does it exist?

**Answer.**

During attention, each token computes Key and Value tensors:

$$K_i = x_i W_K, \quad V_i = x_i W_V$$

In autoregressive decode, at step $t$, the attention mechanism attends over tokens $\{0, 1, \ldots, t\}$:

$$\text{Attn}(Q_t, K_{0:t}, V_{0:t}) = \text{softmax}\!\left(\frac{Q_t K_{0:t}^T}{\sqrt{d_k}}\right) V_{0:t}$$

Without a KV cache, $K_{0:t-1}$ and $V_{0:t-1}$ would need to be **recomputed** at every decode
step. Recomputing all keys and values for a 512-token prefix at every step multiplies computation
by $(S_{\text{context}} + 1)$ relative to a single forward pass.

**With the KV cache:**
- During prefill: compute and store $K_i, V_i$ for all prompt tokens in an SRAM or HBM buffer.
- During decode: append only the new token's $K_t, V_t$ at each step; reuse all prior entries.
- Total computation per decode step is $O(1)$ in new FLOPs (only one new $K, V$ pair computed)
  plus $O(t)$ attention (reading the growing KV cache).

**KV cache memory cost:**

For each transformer layer, one token's KV cache requires:

$$2 \times n_{heads} \times d_{head} \times 2\ \text{B (BF16)} = 2 \times 32 \times 128 \times 2 = 16\ \text{KB (per layer, LLaMA-2 7B)}$$

For 32 layers and maximum context length 4096:

$$\text{KV cache per sequence} = 16\ \text{KB} \times 32\ \text{layers} \times 4096\ \text{tokens} = 2\ \text{GB}$$

This is substantial — a 80 GB HBM device can hold only ~40 KV caches at full context length,
limiting the maximum batch size under continuous batching.

---

### Q3. What is continuous batching and why does it improve GPU/accelerator utilisation?

**Answer.**

**Static batching** (the naive approach): Assemble a batch of $B$ requests; run all of them to
completion before starting the next batch. Because different requests have different output lengths,
some finish early and their slots sit **idle** while the longest request finishes. The accelerator
is underutilised during these idle periods.

**Continuous batching** (iteration-level scheduling): At every decode step, the scheduler can:
- Remove completed requests from the batch (free their KV cache slots).
- Insert new incoming requests into the freed slots for the **next decode step**.

```
Time -->
Slot 0: [req A: 20 tokens] [req E: 30 tokens] ...
Slot 1: [req B: 15 tokens] [req F: 25 tokens] ...
Slot 2: [req C: 40 tokens         ] [req G:...] ...
Slot 3: [req D: 10 tokens][req E already shown above] ...
         ▲ req D done; req E inserted immediately
```

**Hardware requirements for continuous batching:**

1. **Variable batch dimension:** The accelerator's GEMV/GEMM must handle $B$ that changes per
   decode step. This means the command queue descriptor format must include a runtime batch size
   field.

2. **KV cache slot manager:** On-chip memory must be organised into fixed-size pages (e.g.,
   16-token pages, as in vLLM's PagedAttention). The control processor maintains a free-list of
   pages and allocates/frees them as requests arrive and complete.

3. **Per-sequence context length tracking:** The attention kernel must know the current context
   length for each sequence in the batch (sequences have different history lengths). The control
   processor maintains a table of $(B_{max})$ context length registers, updated per step.

4. **Asynchronous prefill insertion:** A new request's prefill can be interleaved with ongoing
   decode steps. The hardware must support "chunked prefill" — processing part of a new prompt
   during a decode step for other sequences — without stalling the decode batch.

---

## Tier 2: Intermediate

### Q4. How is pipeline parallelism implemented across transformer layers on a multi-chip system?
What is the pipeline bubble, and how is it mitigated?

**Answer.**

**Pipeline parallelism (PP)** assigns consecutive groups of transformer layers to different
processing elements (chips or compute clusters). For a 4-chip PP with 32 layers total:

- Chip 0: layers 0–7
- Chip 1: layers 8–15
- Chip 2: layers 16–23
- Chip 3: layers 24–31

A single request flows through chips 0→1→2→3 sequentially. To keep all chips busy, multiple
**micro-batches** are in flight simultaneously:

```
Time →  T0   T1   T2   T3   T4   T5   T6   T7
Chip 0: [μ0][μ1][μ2][μ3][μ0][μ1][μ2][μ3]
Chip 1:      [μ0][μ1][μ2][μ3][μ0][μ1][μ2]
Chip 2:           [μ0][μ1][μ2][μ3][μ0][μ1]
Chip 3:                [μ0][μ1][μ2][μ3][μ0]
         ▲──bubble──▲
```

**Pipeline bubble:** At startup, chips 1–3 are idle while the first micro-batch propagates
through chip 0. The bubble fraction is:

$$\text{bubble fraction} = \frac{P - 1}{P + m - 1}$$

where $P$ is pipeline depth (number of chips) and $m$ is the number of micro-batches in flight.

For $P = 4$ and $m = 8$: bubble fraction $= 3/11 \approx 27\%$.
For $P = 4$ and $m = 32$: bubble fraction $= 3/35 \approx 8.6\%$.

**Mitigation strategies:**

1. **Increase micro-batch count ($m$):** More in-flight micro-batches amortise the startup bubble.
   Requires proportionally more KV cache memory.

2. **1F1B (one-forward-one-backward) schedule:** Used during training; for inference, the analogous
   strategy is interleaving prefill and decode micro-batches.

3. **Interleaved PP (virtual pipeline):** Assign non-consecutive layer groups (e.g., chip 0 handles
   layers 0–3 and 16–19). Each chip processes two non-adjacent stage slices per micro-batch pass,
   halving the bubble fraction at the cost of doubling inter-chip activation transfers.

4. **Asynchronous pipeline stages:** A chip can begin processing the next micro-batch's weights
   from HBM while the current micro-batch result is being transmitted over the chip-to-chip link.

---

### Q5. Describe tensor parallelism within a transformer layer. What are the column-parallel and
row-parallel GEMM strategies, and how do they affect the all-reduce communication?

**Answer.**

For a two-layer FFN with weight matrices $W_1 \in \mathbb{R}^{d \times 4d}$ and
$W_2 \in \mathbb{R}^{4d \times d}$, tensor parallelism with $N$ chips splits:

**Column-parallel (first layer, $W_1$):**

Split $W_1$ column-wise: chip $i$ holds $W_1^{(i)} \in \mathbb{R}^{d \times (4d/N)}$.

$$Y^{(i)} = X \cdot W_1^{(i)}, \quad Y^{(i)} \in \mathbb{R}^{B \times (4d/N)}$$

Each chip computes its shard of the intermediate activation. The non-linearity $\text{GELU}(Y^{(i)})$
is applied locally (no communication needed since GELU is elementwise).

**Row-parallel (second layer, $W_2$):**

Split $W_2$ row-wise: chip $i$ holds $W_2^{(i)} \in \mathbb{R}^{(4d/N) \times d}$.

$$Z^{(i)} = \text{GELU}(Y^{(i)}) \cdot W_2^{(i)}, \quad Z^{(i)} \in \mathbb{R}^{B \times d}$$

Each chip produces a partial sum $Z^{(i)}$ of size $[B, d]$. **All-reduce required:**

$$Z = \sum_{i=0}^{N-1} Z^{(i)}$$

**Communication volume per all-reduce:**

$$\text{bytes} = B \times d \times 2\ \text{B (BF16)} = B \times 4096 \times 2$$

For $B = 32$: $32 \times 4096 \times 2 = 256\ \text{KB}$.

For each layer with attention + FFN: **2 all-reduces per layer** (one after attention output
projection, one after FFN second layer).

Total all-reduce traffic for 32 layers at $B = 32$:

$$32 \times 2 \times 256\ \text{KB} = 16\ \text{MB per forward pass}$$

At NVLink Gen4 bandwidth of 900 GB/s: $16\ \text{MB} / 900\ \text{GB/s} \approx 18\ \mu\text{s}$.
This is negligible compared to the compute time — confirming that TP within a node (high-BW link)
is highly efficient.

---

### Q6. What is "chunked prefill" and how does it enable interleaving prefill and decode on the
same hardware?

**Answer.**

**The problem without chunked prefill:**

When a new request arrives with a 4096-token prompt, a naive system must complete the full prefill
(processing all 4096 tokens) before resuming decode for existing requests. Prefill for 4096 tokens
takes:

$$t_{prefill,4096} \approx \frac{2 \times 4096 \times 4096 \times 4096 \times 32\ \text{layers}}{300 \times 10^{12}\ \text{TOPS}} \approx 14.5\ \text{ms}$$

During these 14.5 ms, every ongoing decode request is stalled. At 5 tokens/second output rate,
this adds 72 ms latency to existing requests — unacceptable for interactive serving.

**Chunked prefill:**

Split the 4096-token prompt into $C$-token chunks (e.g., $C = 128$). At each decode step:
- Process one chunk of the new request's prefill ($C$ tokens) concurrently with decode of
  $B_{decode}$ existing sequences.

The combined "decode + prefill chunk" batch has dimensions:

$$B_{eff} = B_{decode} + C$$

For $B_{decode} = 16$ and $C = 128$: $B_{eff} = 144$ tokens, well within typical batch capacity.

**Hardware requirements:**

1. **Mixed-mode GEMM:** The GEMM kernel must handle a batch where some sequences are in
   "prefill mode" (causal attention over the chunk) and others in "decode mode" (single-vector
   attention over long KV history). This requires per-sequence mode flags in the command descriptor.

2. **Non-uniform sequence lengths:** The attention kernel processes each sequence with a different
   KV cache depth. Hardware must support a vector of context lengths $[l_0, l_1, \ldots, l_{B-1}]$
   rather than a scalar.

3. **KV cache write path active simultaneously with read path:** During chunked prefill, new KV
   entries are being written while decode sequences are reading existing entries. The KV cache SRAM
   must support simultaneous read and write — either by using separate read/write ports or by
   time-multiplexing access (write KV at the end of the decode step, after all reads complete).

**Tradeoff:**

Larger $C$ → fewer total prefill steps → lower time to serve the new request.
Smaller $C$ → less latency impact on existing decode requests per step.

Typical production values: $C = 64\text{–}512$ with $C$ chosen dynamically based on decode queue depth.

---

## Tier 3: Advanced

### Q7. Design the scheduling policy for a continuous batching system. What information must the
hardware expose to the scheduler, and how do KV cache capacity constraints affect admission control?

**Answer.**

**Scheduler inputs (hardware-exposed state):**

| Register / Memory | Content |
|---|---|
| `FREE_KV_PAGES` | Count of free KV cache pages (updated by accelerator after each decode step) |
| `BATCH_STATUS[B_max]` | Per-slot state: FREE, PREFILL, DECODE, DRAINING |
| `SEQ_LEN[B_max]` | Current context length per active sequence (KV entries used) |
| `DECODE_STEP_LATENCY` | Rolling average decode step time in µs (for SLO estimation) |
| `PREFILL_QUEUE_DEPTH` | Number of pending prefill chunks in the command queue |

**Scheduler algorithm (run once per decode step, in ~1 µs on control CPU):**

```python
def schedule_step(state):
    # 1. Free completed sequences
    for slot in range(B_max):
        if state.BATCH_STATUS[slot] == DONE:
            free_kv_pages(state.SEQ_LEN[slot])
            state.BATCH_STATUS[slot] = FREE

    # 2. Admit new requests (subject to KV page budget)
    for req in pending_queue:
        pages_needed = ceil(req.max_output_len / PAGE_SIZE)
        if state.FREE_KV_PAGES >= pages_needed:
            slot = find_free_slot()
            allocate_kv_pages(slot, pages_needed)
            enqueue_prefill_chunk(slot, req, chunk_size=C)
            state.BATCH_STATUS[slot] = PREFILL
            state.FREE_KV_PAGES -= pages_needed
        else:
            break  # cannot admit more without KV OOM

    # 3. Build decode command for next step
    active_slots = [s for s in range(B_max) if state.BATCH_STATUS[s] == DECODE]
    issue_decode_command(active_slots, state.SEQ_LEN[active_slots])
```

**KV cache admission control:**

The scheduler must not admit a request if the KV cache will overflow before output completes.
The KV pages needed per request:

$$\text{pages} = \left\lceil \frac{l_{prompt} + l_{max\_output}}{P_{page}}\right\rceil$$

where $P_{page}$ is the page size in tokens (e.g., 16). Optimistically reserve pages for
$l_{max\_output}$; if the actual output is shorter, pages are returned early.

**Preemption (when KV cache is exhausted mid-generation):**

If a long-running request consumes more KV pages than expected and the cache fills:
1. **Swapping:** Evict the KV cache of the lowest-priority sequence to host DRAM (via DMA), free
   its pages, and later reload it when capacity is available. Requires a DMA bandwidth budget for
   swapping (2 GB KV cache per sequence at DMA bandwidth of 63 GB/s → 32 ms swap time).
2. **Recomputation:** Discard the KV cache and recompute it from the stored token IDs. Cheaper in
   bandwidth but expensive in compute.
3. **Early termination:** Truncate the output at the current token if the request has already
   satisfied its SLO. Used as a last resort.

---

### Q8. Analyse the hardware implications of speculative decoding. What additional compute, memory,
and scheduling resources are required?

**Answer.**

**Speculative decoding** uses a small "draft" model to propose $K$ candidate tokens in $K$
forward passes, then verifies all $K$ tokens in a single forward pass of the large "target" model.
If all $K$ tokens are accepted, throughput improves by approximately $K\times$ with no output
quality degradation.

**Hardware resource additions:**

**1. Draft model compute:**

The draft model (e.g., 68M parameter LLaMA for a 7B target) requires its own GEMM/GEMV execution.
At each decode step:
- Run $K = 5$ draft model forward passes: $K \times T_{draft}$.
- Run 1 target model forward pass with a batch of $K+1$ tokens: $T_{target} \times (K+1)/K$
  relative overhead.

For $K = 5$, $T_{draft} = 0.5\ \text{ms}$, $T_{target} = 10\ \text{ms}$:

$$T_{spec} = 5 \times 0.5 + 10 \times \frac{6}{1} \times \frac{1}{6} \approx 2.5 + 10 = 12.5\ \text{ms per step}$$

Expected accepted tokens: $\alpha K$ where $\alpha \in [0, 1]$ is acceptance rate. For $\alpha = 0.8$:
$0.8 \times 5 = 4$ accepted tokens per step.

Effective throughput: $4\ \text{tokens} / 12.5\ \text{ms} = 320\ \text{tokens/s}$ vs.
$1\ \text{token} / 10\ \text{ms} = 100\ \text{tokens/s}$ for standard decode. 3.2× improvement.

**2. Draft model KV cache:**

The draft model also needs a KV cache, but it is much smaller:

$$\text{draft KV cache} \approx \frac{68M}{7B} \approx 1\%\ \text{of target KV cache size}$$

Negligible for memory budgeting.

**3. Verification kernel:**

The target model processes $K+1$ tokens in a single forward pass. This is a GEMM (not GEMV) of
size $[B \times (K+1), d]$, which is more compute-efficient than $K+1$ separate GEMVs. The
hardware scheduler must:

- Pass the $K$ draft token IDs alongside the original context to the target model.
- Run target model attention over the $K+1$-length extended context.
- Compare target model logits at positions $[t, t+1, \ldots, t+K-1]$ against draft tokens using
  a rejection sampling kernel.

The rejection sampling kernel is a vectorised comparison: for each position $i$:

$$\text{accept}(i) = \text{Uniform}(0,1) \leq \frac{p_{target}(draft_i)}{p_{draft}(draft_i)}$$

This is a pure arithmetic operation on the logit vectors — negligible hardware overhead.

**4. Scheduling implications:**

- The speculative step is not atomic from the scheduling perspective: the draft model and target
  model share the MAC array, requiring time-multiplexed scheduling.
- Draft model runs first (lower priority, preemptable), followed immediately by target model
  verification.
- If acceptance rate $\alpha$ drops below a threshold (e.g., $\alpha < 0.5$), the scheduler
  dynamically disables speculative decoding for that request to avoid wasting draft compute on
  mostly-rejected tokens.
- Hardware counter registers (`SPEC_ACCEPT_COUNT`, `SPEC_REJECT_COUNT`) feed back to the software
  scheduler for dynamic $K$ and $\alpha$ estimation.

---

### Q9. An operator graph compiler must schedule 32 transformer layers across 2 pipeline stages and
4 tensor-parallel chips. How does the compiler minimise communication overhead while respecting
data dependencies?

**Answer.**

**System topology:** 4 chips split into 2 PP stages × 2 TP groups.

```
Stage 0 (layers 0–15):  [Chip 0, Chip 1]  ─── TP all-reduce within stage
Stage 1 (layers 16–31): [Chip 2, Chip 3]  ─── TP all-reduce within stage
                         │
                         └── PP activation send: Stage 0 → Stage 1 (after layer 15)
```

**Compiler scheduling algorithm:**

**Step 1: Assign layers to stages.**
Minimise load imbalance: each stage gets equal compute. For uniform layers: 16 layers per stage.
If layers have unequal compute (e.g., the first/last layer with embedding lookup is heavier),
the compiler can shift the boundary by 1–2 layers.

**Step 2: Schedule TP communication.**
Within each stage, the two TP chips perform all-reduce after each layer's output projection and
FFN. The compiler inserts `ALL_REDUCE` barrier nodes in the graph. Key optimisation: **fuse**
adjacent all-reduce operations if they are for different attention heads and can be pipelined.

**Step 3: Schedule PP communication.**
After layer 15 on Stage 0, activations $[B, S, d]$ must be sent to Stage 1. The compiler inserts
a `SEND` node on Stage 0 and a `RECV` node on Stage 1. These are scheduled as:

```
Stage 0, last decode step of layer 15:
  GEMM(layer 15) → TP_ALL_REDUCE → SEND(activation to Stage 1)

Stage 1, first decode step of layer 16:
  RECV(activation from Stage 0) → GEMM(layer 16)
```

The compiler analyses the critical path: RECV on Stage 1 cannot start until SEND on Stage 0
completes. This serialisation cannot be eliminated, but can be overlapped by pre-fetching the
next micro-batch's weights during the RECV stall.

**Step 4: Overlap compute and communication.**
The compiler uses a "communication-computation overlap" transformation:

For each layer $l$ on Stage 0:
1. Start weight pre-fetch for layer $l+1$ (DMA descriptor issued at the beginning of layer $l$
   compute).
2. Compute layer $l$.
3. Issue TP all-reduce (non-blocking: post the descriptor, continue to step 4).
4. Compute any non-linearity or residual add that does not depend on the all-reduce output.
5. Block on all-reduce completion.

This overlaps steps 3 and 4, hiding ~50% of all-reduce latency.

**Formal representation — Directed Acyclic Graph (DAG) with resource annotations:**

```
Node:     (chip_id, layer_id, op_type, start_cycle, duration_cycles)
Edge:     data dependency or communication dependency
Resource: MAC_ARRAY, HBM_BW, LINK_BW (per chip)

Critical path = longest path through DAG weighted by duration_cycles
Compiler objective: minimise critical path length
```

The compiler runs a **list scheduling** algorithm:
- Priority = bottom-level (longest remaining path to graph exit).
- At each cycle, schedule the highest-priority ready node to an available resource.
- Iterate until all nodes scheduled.

This is polynomial (O(N log N) for N nodes) and produces near-optimal schedules for the
well-structured transformer operator graph.
