# Multi-Head Attention Hardware Implementation

## Background

Multi-head attention (MHA) is the computational core of every transformer-based LLM. At inference time the
operation is dominated by two distinct phases that have very different hardware characteristics:

- **Prefill (prompt processing)**: All prompt tokens processed in parallel. Compute-bound. QKV projections
  and attention score computation are matrix-matrix multiplies.
- **Decode (autoregressive generation)**: One new token per step. Memory-bandwidth-bound. The QKV projection
  for the new token is a matrix-vector multiply; the attention score computation reads the entire KV cache.

A hardware designer must understand both phases deeply to make good architectural decisions.

### Mathematical Recap

Given input X of shape [S, d_model]:

```
Q = X @ W_Q          # [S, d_model]  -> projected to [S, H, d_head]
K = X @ W_K
V = X @ W_V

# Per head h:
scores_h = Q_h @ K_h^T / sqrt(d_head)   # [S, S]
weights_h = softmax(scores_h)            # [S, S]
out_h     = weights_h @ V_h              # [S, d_head]

# Concatenate and project:
out = concat(out_0..out_{H-1}) @ W_O    # [S, d_model]
```

Typical values: d_model = 4096, H = 32, d_head = 128 (LLaMA-2 7B).

---

## Tier 1 — Fundamentals

### Q1: Why is the decode phase memory-bandwidth-bound rather than compute-bound?

**Answer**

During decode, only one new token is generated per step. The QKV projection for that token is:

```
q = x @ W_Q    # x is [1, 4096], W_Q is [4096, 4096]
```

This is a matrix-vector multiply (GEMV). The number of multiply-accumulate (MAC) operations is
`4096 * 4096 = 16.7M`. The weight matrix W_Q must be read from memory (or cache) to perform those MACs.
At FP16 that is `4096 * 4096 * 2 bytes = 32 MB` of weight data. Modern accelerators can sustain hundreds
of TFLOP/s of compute but only hundreds of GB/s of memory bandwidth, giving an arithmetic intensity
(FLOPs / byte) of roughly `33.5M FLOPs / 32 MB ≈ 1 FLOP/byte`. The hardware's roofline arithmetic
intensity is typically 10-100x higher, so the operation is heavily bandwidth-limited.

The KV cache attention read compounds this: for a sequence of S tokens and H=32 heads with d_head=128,
reading the full KV cache per decode step requires `2 * S * H * d_head * 2 bytes = S * 16 KB` of bandwidth
per step — this grows with sequence length, making long-context decode increasingly bandwidth-starved.

**Common mistake**: Confusing the prefill phase (which IS compute-bound, with GEMM operations) with decode.

---

### Q2: What is the purpose of the scaling factor `1/sqrt(d_head)` and does it need to be implemented exactly?

**Answer**

The scaling prevents the dot products Q_h @ K_h^T from growing large in magnitude as d_head increases.
Without scaling, the softmax input would have very large values, pushing the softmax into a saturated region
where gradients vanish — critical during training. At inference, large inputs produce extremely peaked
attention distributions, which may not be desirable.

**Hardware implementation**: The factor is a compile-time constant for a given model. It can be folded into
the Q projection weight matrix W_Q at model-loading time (multiply every element of W_Q by `1/sqrt(d_head)`),
eliminating any runtime overhead. Alternatively it can be applied as a right-shift in fixed-point arithmetic:
for d_head=128, `1/sqrt(128) = 1/11.31`, which is close to `2^{-3.5}`, allowing approximation by a
right-shift of 3 or 4 bits. The small approximation error is irrelevant at inference.

---

### Q3: Describe the dataflow for a single attention head in hardware, from receiving Q/K/V vectors to producing the head output.

**Answer**

**Stage 1 — Score computation**: For each query vector q_i (dimension d_head), compute dot products with
all key vectors k_0..k_{S-1}. Each dot product requires d_head MACs. Output: S scalar scores.

**Stage 2 — Scale**: Multiply scores by `1/sqrt(d_head)` (often pre-folded).

**Stage 3 — Softmax**: Numerically stable softmax over the S scores. Requires finding max, subtracting,
exponentiating, summing, dividing. See softmax_hardware_implementation.md for detail.

**Stage 4 — Weighted sum**: The S softmax weights are used to form a weighted sum of the S value vectors
v_0..v_{S-1} (each of dimension d_head). This is again a GEMV: weights [1,S] @ V [S, d_head] = out [1, d_head].

**Data dependencies**: Stage 2 can begin once all scores for one query are computed. Stage 4 can begin
immediately after softmax since weights are consumed one at a time and can be multiplied by v_j as they
arrive. In hardware, stages 2-4 can be pipelined across queries when scores stream in sequentially.

---

### Q4: How are multiple heads typically handled in hardware — time-multiplexed or spatially parallel?

**Answer**

Both strategies are used, and the choice depends on the area/throughput trade-off:

**Time-multiplexed (area-efficient)**: A single set of PE arrays computes one head at a time. After
completing head h, the same hardware is reused for head h+1. The weight matrices for all heads must
be stored or streamed. This minimises area but limits throughput to one head at a time.

**Spatially parallel (throughput-optimised)**: H identical head engines run concurrently. All H heads
read from the same Q/K/V projection outputs simultaneously and produce their outputs in parallel. The
head outputs are then concatenated. This H-fold speedup comes at the cost of H-fold area for the head
compute units. In practice, designs often choose a parallelism factor P where 1 < P <= H, processing
P heads simultaneously.

**Example calculation**: LLaMA-2 7B has H=32 heads, d_head=128. Each head's score computation for a
single query against a 2048-token sequence requires 2048 * 128 = 262K MACs. With P=8 parallel head
engines, throughput is 8x the single-head rate, but the area of score-compute PEs is also 8x. The
weight projection matrices are shared regardless of P.

---

### Q5: What data must be stored in the KV cache per generated token, and how large does this grow?

**Answer**

For each new token generated, one K vector and one V vector must be appended to the cache — one per
attention head. For MHA with H heads and d_head dimensions at FP16:

```
Per token storage = 2 (K and V) * H * d_head * 2 bytes
                  = 4 * H * d_head bytes

LLaMA-2 7B example (H=32, d_head=128, 32 layers):
  Per layer per token = 4 * 32 * 128 = 16,384 bytes = 16 KB
  All layers per token = 32 * 16 KB = 512 KB
  Sequence of 4096 tokens = 4096 * 512 KB = 2 GB
```

This is why KV cache is the dominant memory consumer at inference for long sequences, often exceeding
the model weight footprint at large batch sizes.

---

## Tier 2 — Intermediate

### Q6: Explain how FlashAttention's tiled computation principle applies to hardware design. What problem does tiling solve and how does online softmax enable it?

**Answer**

**The problem**: Standard attention materialises the full S x S score matrix in memory before computing
softmax. For S=4096, this is `4096^2 * 2 bytes = 32 MB` per head — far exceeding on-chip SRAM on most
accelerators. Materialising this off-chip causes enormous bandwidth overhead.

**Tiling solution**: FlashAttention divides the S queries into tiles of size B_r and the S keys/values
into tiles of size B_c, sized to fit in SRAM simultaneously. For each query tile, all key tiles are
streamed through one at a time; the attention output is accumulated incrementally. The S x S matrix
never exists in full — only B_r x B_c tiles are live at once.

**Online softmax** is what makes this numerically correct. Softmax requires the global maximum over all
S scores before exponentiation, but with tiling we see scores block by block. The online softmax
algorithm maintains two running statistics:

```
m_i   = running max of all scores seen so far for query i
l_i   = running sum of exp(score - m_i) corrected for updates to m_i
out_i = running weighted sum of values, also corrected for max updates
```

When a new block of scores arrives with local max m_new:
```
m_i_new = max(m_i, m_new)
l_i_new = exp(m_i - m_i_new) * l_i + sum(exp(scores_block - m_i_new))
out_i_new = exp(m_i - m_i_new) * out_i + exp(scores_block - m_i_new) @ V_block
```

**Hardware implication**: A hardware attention engine need never store more than one tile of scores on
chip. The SRAM requirement is O(B_r * d_head + B_c * d_head) rather than O(S^2). A 512 KB SRAM can
hold tiles with B_r = B_c = 64 for d_head=128 at FP16, versus being completely unable to hold a 32 MB
score matrix for S=4096.

---

### Q7: Design a block diagram for a pipelined MHA hardware engine for the decode phase. Identify the critical path and bandwidth bottleneck.

**Answer**

**Block diagram** (decode, single new token, S prior tokens in KV cache):

```
                   Weight SRAM
                   [W_Q, W_K, W_V, W_O]
                        |
          x_new ------> GEMV Unit (QKV Proj)
                        |
              +---------+---------+
              |         |         |
              q        k_new     v_new
              |         |         |
              |    KV Cache Controller
              |    (append k_new, v_new)
              |         |
              |    KV Cache SRAM
              |    [k_0..k_{S-1}, v_0..v_{S-1}]
              |         |
              +---> Score Unit (dot products: q @ K^T)
                        |
                   Scale + Softmax Unit
                        |
                   Weighted Sum Unit (weights @ V)
                        |
                   Head Output Buffer
                   [H * d_head outputs]
                        |
                   GEMV Unit (Output Proj W_O)
                        |
                   y_new (d_model output)
```

**Critical path per decode step**:
1. GEMV for QKV projection: 3 * d_model * d_head * H MACs — but this is actually one GEMV of
   x against [W_Q; W_K; W_V], so `3 * d_model^2` MACs total.
2. KV cache append: one write per head (latency-negligible).
3. Score computation: S * d_head MACs per head — grows with sequence length.
4. Softmax: S operations.
5. Weighted sum: S * d_head MACs per head.
6. Output projection: `d_model^2` MACs.

**Bandwidth bottleneck**: Reading the KV cache (step 3+5). Per decode step, all S * H * d_head * 2
key vectors and S * H * d_head * 2 value vectors must be read. For S=2048, this is `2*2048*32*128*2 =
33.5 MB` per layer per step. At 1 TB/s of HBM bandwidth, this takes 33.5 us per layer — 32 layers
gives 1.07 ms just for KV cache reads.

---

### Q8: How does the head concatenation and output projection work in hardware? Can the output projection start before all heads are complete?

**Answer**

After each head h produces its output out_h of shape [d_head], the full MHA output is:

```
concat(out_0, out_1, ..., out_{H-1}) @ W_O
```

where concat produces a vector of size H * d_head = d_model.

**Implementation options**:

**Option A — Wait for all heads**: Collect all H head outputs into a register file (H * d_head elements),
then perform the output GEMV against W_O. Latency = all_heads_compute + output_proj_compute.

**Option B — Partial accumulation (pipeline-friendly)**: Observe that the output projection is a GEMV:

```
y = concat * W_O  =  sum_{h=0}^{H-1} out_h @ W_O[h*d_head : (h+1)*d_head, :]
```

Each head contributes a rank-1 (outer product) update to y. As soon as head h completes, its partial
product `out_h @ W_O_slice_h` (producing a d_model vector) can be accumulated into a running sum.
This allows the output projection to overlap with later head computations, hiding latency.

**Hardware realisation of Option B**: A single accumulator register of width d_model. Each completed
head triggers a GEMV of [d_head] x [d_head, d_model] and the d_model result is added to the
accumulator. This reuses the same PE array for all heads and the output projection with no idle time.

**Memory layout implication**: W_O should be stored as H slices of shape [d_head, d_model] so that
slice h can be streamed as soon as head h completes, without seeking within the weight file.

---

### Q9: What is the arithmetic intensity of the attention score computation during decode, and how does it compare to the GEMV threshold for a typical HBM-attached accelerator?

**Answer**

**Score computation arithmetic intensity**:

For one decode step with sequence length S:
- MACs: S * d_head (one dot product per past token, each of length d_head)
- Bytes read: S * d_head * 2 (K cache) + d_head * 2 (q vector, reused)

Arithmetic intensity = `(2 * S * d_head) FLOPs / (S * d_head * 2 + d_head * 2) bytes`
                     = `(2S * d_head) / (2 * d_head * (S + 1))`
                     ≈ `1 FLOPs/byte` for large S

**GEMV threshold** (roofline crossover): For an accelerator with P TFLOP/s compute and B TB/s bandwidth:
```
threshold = P / B FLOPs/byte
```
A100: 77.6 TFLOP/s (BF16), 2 TB/s HBM => threshold = 38.8 FLOPs/byte
H100: 989 TFLOP/s (BF16 sparsity), 3.35 TB/s => threshold = 295 FLOPs/byte

The attention score computation has ~1 FLOPs/byte intensity, which is roughly 40-300x below the
roofline threshold. This confirms that decode attention is extremely bandwidth-limited — the compute
units are idle more than 97-99% of the time during KV cache reads.

**Implication for design**: Optimising PE array size or adding FP units has essentially no impact
on decode throughput. The only levers are increasing memory bandwidth (wider HBM, higher frequency,
near-memory compute) or reducing bytes read (quantisation of KV cache, GQA/MQA).

---

### Q10: Describe how a weight-stationary dataflow differs from an output-stationary dataflow for the QKV projection, and which is better suited for the decode phase.

**Answer**

**Weight-stationary**: Weight tiles are loaded into the PE array once and held fixed (stationary)
while input activations stream through. Each weight is reused as many times as there are input
rows. Best when the same weights are applied to many inputs (large batch size). During decode
with batch_size=1, each weight is used exactly once, so there is zero reuse — weight-stationary
provides no benefit and weights must be reloaded every step.

**Output-stationary**: Each PE accumulates one or more output values. Partial sums remain in the
PE register file (stationary) until complete. The PE reads input and weight data as they arrive.
For GEMV with one output vector, every output element accumulates d_model partial products.
This is natural for GEMV: each output element has high reuse of the input vector (one dot product).

**Recommendation for decode**: Output-stationary is better suited. The input vector x is broadcast
to all PEs; each PE accumulates one output element by reading its column of the weight matrix. The
input is reused d_model times (once per output element), while weights stream through once. This
matches the data movement pattern: reading all weights is unavoidable (bandwidth-limited), but the
input is tiny and fits in registers.

**Input-stationary variant**: The input vector x is held in registers (stationary) and the weight
matrix is streamed through. Each weight element w_{ij} is multiplied by x_j and accumulated into
output y_i. This is equivalent to output-stationary for GEMV and is the most natural mapping.

---

## Tier 3 — Advanced

### Q11: A hardware team proposes splitting the QKV projection into three independent GEMVs (Q, K, V separately) rather than one fused GEMV. Analyse the trade-offs in terms of bandwidth, latency, and hardware resources.

**Answer**

**Fused GEMV (single pass)**:
- Weight read: `3 * d_model * d_model * 2 bytes` total, read once in a single pass.
- Latency: one GEMV of x against a [d_model, 3*d_model] matrix.
- Hardware: one PE array of width 3*d_model outputs, or one array reused with width d_model three
  times sequentially. With sequential reuse, latency triples but area is 1/3.

**Three independent GEMVs**:
- If pipelined or run in parallel, same bandwidth and similar latency as fused.
- If run sequentially on the same hardware: identical to fused with sequential reuse — no difference.
- If on separate hardware (three PE arrays): 3x area but 3x throughput, allowing all three to
  execute in parallel.

**Critical insight for decode**: The dominant cost is weight data movement. Whether fused or split,
`3 * d_model^2 * 2` bytes must be read. The fused approach allows the memory controller to issue
sequential reads to [W_Q; W_K; W_V] as a single contiguous block, maximising DRAM burst efficiency.
Three separate GEMVs may incur address calculation overhead and burst restarts, slightly reducing
effective bandwidth.

**Latency hiding opportunity**: The key vector k_new is needed to update the KV cache before the
attention score computation begins. If K projection finishes first, the hardware can immediately
start the KV cache write and simultaneously continue computing Q and V projections, hiding some
latency. This is easier to schedule with separate GEMV modules that can signal completion independently.

**Recommendation**: A single fused weight read with three output accumulation units (one each for Q,
K, V) is optimal. The weight data is read once; the three accumulators work in parallel. This retains
the bandwidth efficiency of fused execution and the latency-hiding benefits of independent completion
signalling for K and V.

---

### Q12: How would you design a hardware unit to handle causal (autoregressive) masking efficiently? What would change in the dataflow compared to bidirectional attention?

**Answer**

**Causal masking requirement**: Token i can only attend to tokens j <= i. The upper triangle of the
attention score matrix (j > i) must be set to -infinity before softmax.

**Bidirectional dataflow**: Every query attends to all S keys. The score computation is a full S x S
matrix multiply. Symmetry can sometimes be exploited but is not required.

**Causal dataflow modification**:

1. **Score generation gating**: When computing scores for query i, the score unit generates scores
   s_{i,0}, s_{i,1}, ..., s_{i,S-1} sequentially (or in tiles). A simple comparator checks whether
   the key index j > i. If so, the score is replaced by a large negative constant (e.g., -65504 in
   FP16, the minimum finite value) before entering the softmax pipeline. This comparator adds no
   meaningful area or latency.

2. **Triangular tile skipping**: When processing query tile [i_start, i_end] against key tile
   [j_start, j_end], if j_start > i_end (the entire key tile is in the future), the tile is skipped
   entirely. This eliminates roughly half the compute for prefill, matching the theoretical 50%
   reduction in FLOPs from causal masking. Each tile produces either a full result, a masked result
   (all -inf), or a partial result (diagonal tiles) requiring element-wise masking.

3. **Online softmax adaptation**: Masked scores (set to -inf) produce exp(-inf) = 0, contributing
   nothing to the softmax sum. No special handling is needed in the online softmax accumulator
   beyond correctly representing -inf in the score precision. In FP16, -65504 gives
   exp(-65504/T) ≈ 0 for any reasonable temperature T.

4. **KV cache access pattern**: During decode, query i only needs keys k_0..k_i. The KV cache
   controller never needs to mask during decode because the cache only contains past tokens by
   construction. Causal masking is only an active concern during prefill.

**Hardware savings from causal masking**: During prefill, roughly half the score tiles are all-masked
(skipped) and the diagonal tiles are half-masked. A causal-aware tile scheduler reduces prefill
attention compute by ~50%, matching the 2x compute reduction in software implementations.

---

### Q13: Describe a complete pipeline hazard analysis for a pipelined MHA engine. Identify all data hazards and propose resolution strategies.

**Answer**

**Pipeline stages** (decode phase):
```
S1: QKV Projection GEMV
S2: KV Cache Append (write k_new, v_new)
S3: Score Computation (q @ K_cache^T)
S4: Online Softmax
S5: Weighted Sum (weights @ V_cache)
S6: Output Projection GEMV
```

**Hazard analysis**:

**RAW hazard S2->S3**: S3 reads the KV cache, which must include k_new appended in S2. If S3 begins
immediately after S2, it may read a stale KV cache that does not yet contain k_new.
Resolution: S3 reads k_new directly from S1's output registers (bypassing the cache for the current
token's key) and reads k_0..k_{S-1} from cache. The bypass path requires a multiplexer at the score
unit input: "if key_index == current_position, use k_new register; else use KV cache read data."

**RAW hazard S4->S5**: S5 needs softmax weights to accumulate the weighted sum. With online softmax,
weights are produced one at a time as scores arrive. S5 can consume weight w_j and simultaneously
compute `w_j * v_j` before w_{j+1} is ready — full pipeline overlap if V cache reads are scheduled
to deliver v_j at the same cycle as w_j.
Resolution: interleave the score computation order to match V cache access order. Score for key j is
computed, immediately enters the softmax unit, and the renormalised weight for j is forwarded to the
weighted sum unit while v_j is concurrently fetched from cache.

**Structural hazard on PE array**: If the same PE array handles both QKV projection (S1) and output
projection (S6), they cannot run simultaneously.
Resolution: Either (a) dedicate separate PE arrays to S1 and S6, or (b) schedule them sequentially.
For decode, S6 cannot start until all H head outputs from S5 are ready, so sequential scheduling
incurs no additional latency beyond the natural data dependency.

**Memory port contention (S2 vs S3)**: KV cache append (S2) writes to the same SRAM that S3 reads.
Dual-port SRAM (one read port, one write port) resolves this without arbitration. Single-port SRAM
requires time-multiplexing: write in cycle T, read in cycle T+1. Since S2 appends to position S
while S3 reads positions 0..S-1, there is no aliasing — the write address always differs from all
read addresses. This allows the SRAM controller to freely interleave accesses.

**Control hazard**: If the softmax unit detects all scores are -inf (degenerate masked query), the
weighted sum must output zero. A flag from the softmax unit to S5 handles this, either by gating
the accumulator or by outputting zeros directly.

---

### Q14: An accelerator targets a 1024-token context with 32 heads and d_head=128 using INT8 for weights and BF16 for activations. Estimate the KV cache SRAM size needed to keep all KV data on-chip, and propose a memory hierarchy if this exceeds available SRAM.

**Answer**

**On-chip KV cache size calculation**:

Per layer, per token, for K and V:
```
bytes = 2 (K+V) * H * d_head * precision_bytes
      = 2 * 32 * 128 * 2 (BF16)
      = 16,384 bytes = 16 KB
```

For 1024 tokens, one layer:
```
1024 * 16 KB = 16 MB
```

For 32 transformer layers:
```
32 * 16 MB = 512 MB
```

This significantly exceeds typical on-chip SRAM budgets (2-32 MB for AI accelerators). Options:

**Hierarchical memory strategy**:

**Level 1 (L1 SRAM, ~2-4 MB)**: Holds the KV cache for the current layer being computed plus the
KV tile currently being scored. Sized to hold `B_c * H * d_head * 2` bytes for the active key tile
and the corresponding value tile. For B_c=64, H=32, d_head=128 at BF16: `2 * 64 * 32 * 128 * 2 = 1 MB`.
This fits comfortably in L1.

**Level 2 (L2 SRAM or eDRAM, ~32-64 MB)**: Holds the full KV cache for one or two transformer layers.
This allows layer-by-layer streaming: load layer l's KV cache from HBM into L2 before computing layer l,
then stream tiles from L2 to L1 during attention computation. L2 -> L1 bandwidth is much higher than
HBM bandwidth (e.g., 8 TB/s internal vs 2 TB/s HBM).

**HBM (High Bandwidth Memory, multiple GB)**: Holds all 32 layers of KV cache. During each decode step,
the 512 MB KV cache is streamed from HBM through L2 to L1. At 2 TB/s HBM bandwidth, streaming 512 MB
takes 256 us — the dominant latency component of a long-context decode step.

**KV cache quantisation**: Quantising KV cache to INT8 (from BF16) halves the size to 256 MB and doubles
effective bandwidth. Recent models (LLaMA-3) use this routinely with negligible quality loss. INT4
quantisation (128 MB) is also feasible with careful quantisation scheme design.

**Prefetching**: The KV cache access pattern during attention is fully predictable (sequential key
indices 0..S-1 for each query). A hardware prefetch engine can issue HBM reads 10-50 cache lines
ahead of the current read pointer, hiding HBM latency (typically 200-500 ns) completely behind
bandwidth-limited transfer time.
