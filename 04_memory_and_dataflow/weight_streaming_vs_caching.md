# Weight Streaming vs. Caching in LLM Accelerators

## Overview

The decision of whether to stream weights from off-chip memory on every inference pass or to cache
them on-chip is one of the most consequential architectural choices for an LLM accelerator. It
determines the off-chip memory bandwidth requirement, the on-chip memory area budget, the latency
of each token, and the maximum model size the accelerator can serve. This section examines the
trade-offs from first principles and works through the quantitative analysis required to make the
right choice for a given deployment scenario.

---

## Tier 1 — Fundamentals

### Q1. What is weight streaming and what is weight caching?

**Question:** Define weight streaming and weight caching as applied to LLM inference. For each,
describe the data flow from storage to compute and identify the primary cost.

**Answer:**

**Weight Streaming:**

In weight streaming, model weights are stored entirely in off-chip memory (HBM, LPDDR5, or GDDR)
and are fetched into on-chip SRAM at the time they are needed for computation. After the
corresponding tile is computed, the weight data is either discarded or overwritten by the next
tile. Weights are not retained between forward passes.

Data flow:
```
HBM -> [DMA] -> On-chip SRAM tile buffer -> PE array -> (weight data discarded)
                                                         ^-- weight is used once then replaced
```

Primary cost: **HBM bandwidth**. Every forward pass re-reads the entire weight matrix. For a
70B parameter model at INT8, that is 70 GB of HBM reads per forward pass per generated token.

**Weight Caching:**

In weight caching, some or all model weights are retained in on-chip SRAM (or a large on-package
memory like HBM used as a software-managed weight cache) across multiple forward passes or across
multiple requests. The weight is loaded once and reused for many tokens.

Data flow:
```
HBM -> [one-time DMA load] -> On-chip SRAM cache -> PE array (weight reused across passes)
                                          ^-- weight stays here across multiple tokens
```

Primary cost: **On-chip SRAM area and power**. The cache must be large enough to hold the working
set of weights. SRAM at 28nm costs roughly 0.15 MB/mm^2, so caching 70 GB of weights is not
feasible on-chip; partial caching of hot layers or hot attention heads is more realistic.

**The central trade-off:**

Streaming trades off bandwidth (re-read cost per pass) against area (no large cache needed).
Caching trades off area (large on-chip buffer) against bandwidth (amortised load cost).
The break-even point depends on batch size and reuse frequency, as analysed in Q3.

---

### Q2. What determines whether a workload is bandwidth-bound or compute-bound?

**Question:** Define the roofline model. Using it, determine whether LLM decode and LLM prefill
are bandwidth-bound or compute-bound for a chip with 200 TOPS of INT8 compute and 3.2 TB/s HBM
bandwidth. State the implications for weight streaming vs. caching.

**Answer:**

**Roofline model:**

The roofline model plots achievable performance (FLOPS) against arithmetic intensity (FLOPs/byte):

```
Achievable FLOPS = min(Peak_FLOPS, AI * Peak_BW)
```

The "ridge point" (balance point) is:
```
AI_ridge = Peak_FLOPS / Peak_BW = 200e12 / 3.2e12 = 62.5 FLOPs/byte
```

Operations with AI < 62.5 F/B are bandwidth-bound; those with AI > 62.5 F/B are compute-bound.

**LLM decode (M=1):**

GEMV for one token: C(1 x N) = A(1 x K) * W(K x N), INT8 weights, FP16 activations:
```
FLOPs = 2 * 1 * K * N
Bytes from HBM = K * N * 1 (weights, INT8) + 1 * K * 2 (activations, FP16, small)
               ≈ K * N (weight-dominated)
AI ≈ 2 * K * N / (K * N) = 2 FLOPs/byte
```

AI = 2 << AI_ridge = 62.5: **strongly bandwidth-bound**.

Achievable throughput = 2 * 3.2e12 = 6.4 TOPS (only 3.2% of peak compute utilised).

**Implication for decode:** Compute is massively underutilised. Caching weights on-chip does
not help because the problem is not how fast weights can be supplied to compute — it is how
quickly the vast weight tensor can physically be read. The only levers are: (a) increase HBM BW,
(b) reduce bytes per weight (quantisation), (c) increase batch size to raise AI.

**LLM prefill (M=2048):**

GEMM: C(2048 x N) = A(2048 x K) * W(K x N):
```
FLOPs = 2 * 2048 * K * N
Bytes from HBM ≈ 2048 * K * 2 (activations) + K * N * 1 (weights) ≈ K*N for large K,N
AI ≈ 2 * 2048 * K * N / (K * N) = 4096 FLOPs/byte
```

AI = 4096 >> AI_ridge = 62.5: **strongly compute-bound**.

Achievable throughput ≈ 200 TOPS (near peak PE utilisation).

**Implication for prefill:** The bottleneck is compute, not bandwidth. Weight caching has little
benefit because weights are already being loaded efficiently and compute is the binding constraint.
Energy can be saved by reducing unnecessary weight re-reads (use weight caching if the same model
serves many prefill requests back-to-back), but throughput is unchanged.

---

### Q3. What is the weight reuse ratio and when does caching become beneficial?

**Question:** Define the weight reuse ratio for weight caching. Derive the break-even condition
that determines when it is worthwhile to cache weights on-chip rather than streaming them from
HBM. Apply the analysis to a scenario with a 70B model, 3.2 TB/s HBM, and a 20 MB on-chip SRAM
budget for weight caching.

**Answer:**

**Weight reuse ratio:**

```
Reuse ratio R = Total weight accesses / Number of unique weight elements
```

For a batch of B tokens processed together through one layer:
```
R = B    (each weight is multiplied against B different input tokens)
```

For streaming (no cache):
```
Cost per token = W_bytes / B    (HBM reads divided across the batch)
where W_bytes = weight matrix size in bytes
```

For caching (weights loaded once, reused for B tokens):
```
HBM cost per token = W_bytes / (B * N_requests_before_eviction)
```

where N_requests_before_eviction is how many batches use the weight before it must be reloaded
(due to context switches, model swaps, etc.).

**Break-even condition:**

Caching is beneficial when the amortised load cost falls below the streaming cost. For a fixed
SRAM budget S_cache bytes and model size W_bytes >> S_cache:

We can cache a fraction f = S_cache / W_bytes of the total weights. These cached weights are
never re-read from HBM. The remaining (1-f) fraction is streamed.

HBM traffic per token:
- Streaming: W_bytes per token
- Partial cache: (1-f) * W_bytes per token + f * W_bytes / N_reuse

Break-even when partial cache helps: always beneficial to cache the most-reused weights as long
as the SRAM is not wasted on cold weights.

**Concrete scenario:**

70B model at INT8 = 70 GB. On-chip cache = 20 MB.
Fraction cacheable = 20 MB / 70 GB = 0.029% -- negligible.

Conclusion: 20 MB on-chip SRAM cannot meaningfully cache 70B model weights.
The weight caching approach requires either:
- A much larger on-chip SRAM (e.g., Cerebras WSE: 40 GB SRAM on-chip), or
- A tiered scheme where frequently accessed layers (embedding, first/last layers) are cached
  while the bulk of the model streams from HBM.

**When caching IS beneficial:**

- Small models (7B at INT4 = 3.5 GB) on chips with large HBM used as a software-managed cache:
  the entire model fits in HBM (not streamed from LPDDR/NVMe), eliminating the slowest tier.
- Repeated inference on the same model: a dedicated weight prefetch warms up HBM caches in the
  memory controller, reducing effective read latency (though not bandwidth).
- Shared layers in MoE: the shared transformer blocks (non-expert FFN, attention) can be cached
  in fast SRAM while the expert weights stream from HBM. This is a common practical optimisation.

---

## Tier 2 — Intermediate

### Q4. How does batching change the weight streaming vs. caching trade-off?

**Question:** Explain quantitatively how increasing batch size during decode shifts the arithmetic
intensity and reduces the relative advantage of weight caching. At what batch size does the
workload transition from bandwidth-bound to compute-bound for the chip described in Q2
(200 TOPS, 3.2 TB/s HBM)?

**Answer:**

**Arithmetic intensity as a function of batch size B:**

For a linear layer with weight W of shape (K x N) at INT8:
```
FLOPs = 2 * B * K * N
HBM bytes (weight-dominated) = K * N * 1 + B * K * 2 (inputs) + B * N * 2 (outputs)
                               ≈ K*N + 2*B*(K+N)
AI(B) = 2*B*K*N / (K*N + 2*B*(K+N))
```

For large K=N=d (square weight, typical FFN):
```
AI(B) = 2*B*d^2 / (d^2 + 4*B*d) = 2*B*d / (d + 4*B) ≈ B/2 for B << d
                                                         ≈ d/2 for B >> d
```

For d=4096 (Llama-13B FFN hidden):
- B=1: AI = 2*1*4096^2 / (4096^2 + 4*1*4096) ≈ 2 FLOPs/byte
- B=64: AI ≈ 64 FLOPs/byte (approaching ridge point!)
- B=128: AI ≈ 128/2 = 64 FLOPs/byte (just above ridge point for this chip)

**Transition from bandwidth-bound to compute-bound:**

AI_ridge = 62.5 F/B (from Q2).
```
AI(B) = ridge => B/2 = 62.5 => B = 125 requests
```

At batch size B = 125, the decode phase transitions from bandwidth-bound to compute-bound.

**Implication for weight caching vs. streaming:**

For B < 125: bandwidth-bound. Weight caching reduces HBM traffic, but with only 20 MB cache on
a 70B model, almost no weights can be cached. The binding constraint (HBM bandwidth) cannot be
relieved by on-chip caching. Quantisation (fewer bytes per weight) is more effective.

For B > 125: compute-bound. Bandwidth is no longer the bottleneck. Weight caching is irrelevant
to throughput (the PE array is already saturated). Energy efficiency improves if weights are
cached (fewer HBM accesses), but throughput does not.

**Practical takeaway:**

For an LLM inference system targeting high throughput (large batch), neither weight streaming
nor weight caching is the critical variable — PE utilisation and compute efficiency dominate.
For low-latency single-request inference (small batch), weight streaming from the fastest
available memory (HBM) at maximum bandwidth is the only option; caching helps only if it places
weights in faster memory than they would otherwise be in (e.g., HBM vs. NVMe).

---

### Q5. What is weight compression and how does it interact with streaming?

**Question:** Describe three weight compression techniques applicable to streamed weights.
For each, quantify the effective bandwidth improvement for a chip with 3.2 TB/s HBM, and
describe the decompression hardware needed on-chip.

**Answer:**

**Technique 1: Post-training Quantisation (PTQ) — INT8 or INT4**

Reduces weight precision from FP16 (2 bytes) to INT8 (1 byte) or INT4 (0.5 bytes).

Effective bandwidth improvement:
- FP16 -> INT8: 2x bandwidth improvement (same weights, half the bytes)
- FP16 -> INT4: 4x bandwidth improvement

At 3.2 TB/s HBM with INT4: effective weight streaming bandwidth = 12.8 TB/s equivalent.

Decompression hardware:
- INT8: Each PE has a lookup table or scale/zero-point register. Dequantisation is:
  `fp16_val = (int8_val - zero_point) * scale`
  One multiply and one add per weight -- can be fused into the MAC pipeline at negligible cost.
- INT4: Two weights packed per byte. The dequantisation unit unpacks and scales:
  `w_hi = (byte >> 4) & 0xF; w_lo = byte & 0xF`
  Group quantisation (per 128 weights) stores one FP16 scale and one FP16 zero-point per group,
  adding ~3% storage overhead but maintaining accuracy.

Accuracy impact: INT8 -- essentially lossless for most LLMs (< 0.5% perplexity increase).
INT4 with GPTQ or AWQ calibration -- acceptable for most 7B-70B models.

**Technique 2: Sparse Weight Representation (Structured Sparsity)**

NVIDIA's 2:4 structured sparsity: for every 4 consecutive weights, exactly 2 are zero. The
non-zero 2 values plus a 2-bit index per pair are stored instead of 4 values.

Storage compression: 2 non-zero values + 4 bits metadata vs. 4 original values.
- At INT8: (2*1 byte + 4 bits) = 2.5 bytes per 4 weights vs. 4 bytes = 1.6x compression
- At FP16: (2*2 bytes + 4 bits) = 4.5 bytes vs. 8 bytes = 1.78x compression

Effective HBM bandwidth: 3.2 TB/s * 1.6 = 5.12 TB/s equivalent at INT8.

Decompression hardware:
- A small expansion unit reads the 2-bit indices and inserts zeros at the correct positions
  before feeding the weight register file. This is a 4-to-2 sparse decoder: maps 2 values to
  4 positions using 2-bit indices.
- NVIDIA Tensor Cores include a dedicated sparse decoder that performs this in hardware,
  achieving 2x MAC throughput for 2:4 sparse models.

Accuracy impact: Requires fine-tuning with sparsity masks applied during training (magnitude
pruning + fine-tuning). Typically 1-2% accuracy degradation without careful training.

**Technique 3: Delta Compression / Weight Sharing**

Weights within a layer often have correlated values. Delta coding stores the difference between
adjacent weights rather than absolute values, exploiting the fact that deltas are smaller and
compress better.

More commonly used: weight sharing (k-means quantisation). Cluster weights into 256 centroids
per group; store 8-bit centroid index per weight + a small codebook (256 * 2 bytes = 512 bytes
per group). For groups of 32 weights: 32 bytes + 512 bytes overhead = 32/32 * 512 overhead
(16x overhead) -- only beneficial for large groups.

For groups of 4096 weights: 4096 bytes -> 4096 indices (1 byte each) + 512 byte codebook
= 4608 bytes vs. 4096 * 2 = 8192 bytes (FP16) = 1.78x compression.

Decompression hardware:
- A lookup table indexed by the 8-bit centroid index, outputting the FP16 centroid value.
- 256-entry FP16 lookup table = 512 bytes per group -- must be loaded alongside the indices.
- Latency: 1-2 cycles for the table lookup, which must be hidden by weight prefetching.

**Comparison summary:**

| Technique | Compression | BW improvement | Hardware cost | Training needed |
|-----------|-------------|----------------|---------------|-----------------|
| INT8 PTQ | 2x | 2x | Minimal (scale registers) | PTQ calibration only |
| INT4 PTQ | 4x | 4x | Small (unpack + scale) | PTQ or QAT |
| 2:4 Sparse | 1.6x | 1.6x | Sparse decoder unit | Sparsity-aware fine-tuning |
| Weight sharing | ~1.8x | ~1.8x | Lookup table | PTQ or QAT |

---

### Q6. When is it better to cache the KV-cache rather than model weights on-chip?

**Question:** Contrast the reuse patterns of model weights vs. KV-cache entries during decode.
Argue for or against the proposition: "For single-request low-latency inference, on-chip SRAM
is better spent caching KV-cache entries than model weights."

**Answer:**

**Model weight reuse during single-request decode:**

For a single request (batch=1) generating tokens autoregressively:
- Weights are the same for every generated token (same model parameters).
- Each token generation requires reading all weights once (GEMV = one weight read per weight).
- Across T generated tokens, each weight is accessed T times.
- Reuse ratio = T (the number of generated tokens).

For T=100 tokens: weights are reused 100 times -- moderate reuse, worth caching IF they fit.

For a 70B model: 70 GB of weights. To cache the full model, need 70 GB of on-chip SRAM --
physically impossible on current silicon.

**KV-cache reuse during single-request decode:**

The KV-cache grows by 2 * n_layers * n_kv_heads * d_head bytes per token. For Llama-70B with
GQA (8 KV heads), d_head=128, 80 layers:
```
KV per token = 2 * 80 * 8 * 128 * 2 bytes = 327,680 bytes = 320 KB per token
```

For the current context of length L tokens, the KV-cache size = L * 320 KB.
For L=1000: 320 MB of KV-cache.

During decode token T+1, the attention operation reads ALL L previous K and V entries once.
Reuse ratio = 1 per decode step (each KV entry read once per new token generated).

**Comparison for a 32 MB on-chip SRAM budget:**

If used for weight caching:
- Can cache 32 MB / 70 GB = 0.045% of model weights.
- These cached weights eliminate 0.045% of HBM reads per token = negligible impact.
- Weight streaming latency reduction: near zero.

If used for KV-cache:
- Can hold 32 MB / 320 KB = 100 tokens of KV-cache entirely on-chip.
- For the first 100 tokens of a sequence, attention reads K and V from fast on-chip SRAM
  (bandwidth ~2 TB/s) rather than HBM (3.2 TB/s available but shared with weight streaming).
- Attention reads KV at up to 2 TB/s vs. 3.2 TB/s from HBM -- marginal improvement, but the
  key win is eliminating HBM contention between attention (KV reads) and the FFN (weight reads).
  Separating these onto different memory paths (KV on on-chip SRAM, weights on HBM) allows
  both to proceed at full bandwidth simultaneously.

**Verdict:**

For single-request decode with a 70B model, on-chip SRAM is better spent on KV-cache than on
weight caching, because:
1. The fraction of model weights that fit is too small to move the needle on HBM bandwidth.
2. KV-cache for short contexts can fit entirely on-chip, enabling full-bandwidth decoupled
   attention + FFN execution.
3. FlashAttention-style tiling for long contexts can keep the working KV tile on-chip even
   when the full KV-cache does not fit, reducing HBM pressure from the attention path.

The calculus changes for small models (< 2B parameters at INT4 fit in 1 GB) or future chips
with very large on-chip SRAM budgets.

---

## Tier 3 — Advanced

### Q7. Design a weight prefetch scheduler for a continuous batching LLM inference server.

**Question:** A production LLM inference server uses continuous batching: new requests join the
batch while others are still generating. This means the effective batch size varies dynamically.
The weight prefetch scheduler must adapt to changing batch sizes without stalling the PE array.
Design the scheduler's state machine, its interaction with the DMA engine, and its adaptation
strategy. Consider the cases of batch size increasing, decreasing, and remaining stable.

**Answer:**

**System model:**

The scheduler operates on a per-layer granularity. Each layer consists of:
1. Weight tile prefetch phase (DMA reads weight tiles from HBM into on-chip buffer)
2. Compute phase (PE array multiplies activation batch against weight tiles)
3. Activation writeback phase (DMA writes output activations to HBM)

With double-buffering, phases 1, 2, and 3 for consecutive tiles overlap.

**Dynamic variables:**

```
B_current   : current batch size (number of active sequences)
T_comp(B)   : compute time per weight tile = Tile_FLOPs / (B * TOPS_per_element)
T_dma       : DMA time per weight tile (fixed: Tile_bytes / HBM_BW)
T_stall(B)  : max(0, T_dma - T_comp(B))  -- stall if DMA is slower than compute
```

For the chip in Q2 (200 TOPS INT8, 3.2 TB/s HBM), a 1024x1024 INT8 weight tile:
```
T_dma = 1024*1024*1 / 3.2e12 = 330 ns
T_comp(B) = 2*B*1024*1024 / 200e12 = B * 10.5 ns
Balance: T_dma = T_comp => B_balance = 330 / 10.5 ≈ 31 requests
```

For B < 31: compute is faster than DMA -- stalls occur.
For B >= 31: DMA is fully hidden by compute.

**Scheduler state machine:**

```
States: PREFETCH_WARM, STEADY_STATE, BATCH_INCREASING, BATCH_DECREASING, STALL_RECOVERY

PREFETCH_WARM (initial state):
  Action: Issue DMA for tiles 0 and 1 (fill both double-buffer slots)
  -> STEADY_STATE after both DMAs complete

STEADY_STATE:
  Action: On compute_done(tile_i):
    - Issue DMA for tile i+2 (two tiles ahead = 2-deep prefetch pipeline)
    - Assert compute_start(tile_i+1) if buffer ready
  Transition: If B_new != B_current -> appropriate transition state

BATCH_INCREASING (new request admitted):
  Action: On next tile boundary:
    - Update T_comp(B_new) < T_comp(B_old)
    - Check if T_comp(B_new) >= T_dma (still DMA-hidden)
    - If yes: continue STEADY_STATE
    - If no (B_new > B_balance): reduce tile depth K_tile to shorten T_comp
      (split one deep tile into two shallower tiles, each taking half the compute time)
  -> STEADY_STATE

BATCH_DECREASING (request completes):
  Action: On next tile boundary:
    - Update T_comp(B_new) > T_comp(B_old)
    - Check if T_comp(B_new) >= T_dma
    - If yes: STEADY_STATE (still hidden)
    - If no: T_stall = T_dma - T_comp(B_new) > 0
      -> STALL_RECOVERY

STALL_RECOVERY:
  Action: Deepen prefetch pipeline (3-deep instead of 2-deep):
    - Issue DMA for tiles i+3 while computing i+1
    - Stalls occur between compute_done and next tile ready
    - Log stall cycles to perf counter register
    - If stalls > threshold: notify host scheduler to increase batch size
  -> STEADY_STATE (with stall tolerance)
```

**Adaptation strategies:**

**1. Tile depth adaptation (K_tile scaling):**

When B increases (more compute per tile but same DMA time), risk: T_comp > T_dma (DMA becomes
the new bottleneck). Solution: increase K_tile to process more inner-product depth per tile,
making T_comp larger and ensuring it stays >= T_dma.

When B decreases (less compute per tile), risk: T_comp < T_dma. Solution: decrease K_tile to
reduce compute per tile, reducing the stall gap. Minimum K_tile is bounded by PE array depth
(tiles too shallow cause pipeline flush overhead that exceeds the benefit).

**2. Prefetch depth adaptation:**

Standard: 2-deep prefetch (one tile in compute, one tile DMA-loading).
For B < B_balance: extend to 3-deep or 4-deep prefetch. This requires more SRAM (3-4 buffer
slots instead of 2) but allows the DMA to run ahead of compute. The stall is pushed to the
startup transient only, amortised across the deeper pipeline.

**3. Host-level admission control:**

The scheduler feeds back stall cycle counts per second to the host inference server. When stall
rate > 10% (indicating the batch is too small to hide DMA latency), the server either:
- Waits to admit new requests in larger groups (increasing average batch size), or
- Accepts the throughput penalty and prioritises single-request latency.

This creates a closed-loop system where hardware performance counters drive software scheduling
policy -- the key mechanism behind efficient continuous batching on real accelerators.

---

### Q8. Analyse the weight streaming bandwidth requirement for a 1-trillion-parameter MoE model.

**Question:** A mixture-of-experts model has 1 trillion total parameters but only 20 billion
active parameters per forward pass (top-2 routing over 128 experts). During decode with a batch
of 512 requests, each routed to potentially different experts: calculate the HBM bandwidth
required to sustain 100 tokens/second throughput. State whether expert weight caching is
beneficial and under what conditions.

**Answer:**

**Parameter breakdown:**

Assuming the MoE follows a standard architecture with:
- Shared parameters (attention + embedding + normalisation): ~10B params (INT8 = 10 GB)
- Expert FFN parameters: ~990B params across 128 experts = 7.74B params/expert (INT8 = 7.74 GB/expert)
- Active per token: 2 experts * 7.74 GB = 15.48 GB active expert weights + 10 GB shared = 25.5 GB

**HBM bandwidth requirement for 100 tokens/second:**

Per token, the unique weights loaded from HBM:
- Shared weights: Always loaded = 10 GB (same for every token)
- Expert weights: In a batch of 512 tokens, each token routes to 2 experts.
  Total distinct expert requests = 512 * 2 = 1024 expert invocations.
  Over 128 experts: average requests/expert = 1024/128 = 8 tokens per expert.
  Number of distinct experts activated in batch ≈ 128 (high probability all are hit with 512 tokens)
  Expert weights loaded = 128 * 7.74 GB = ~990 GB per batch step

Wait -- the tokens in a batch are processed simultaneously. All 128 experts must have their
weights loaded once to serve the batch:
```
Total HBM reads per batch = 10 GB (shared) + 990 GB (all experts) = 1000 GB per batch
```

For 100 tokens/second with batch=512:
```
Batch rate = 100 tokens/s / 512 tokens/batch = 0.195 batches/second
Required HBM bandwidth = 1000 GB * 0.195/s = 195 GB/s
```

This is within the range of a 4-stack HBM3 system (4 * 819 GB/s = 3.28 TB/s). The chip is
comfortably bandwidth-sufficient for this rate.

But let's check the per-token latency for a single request (batch=1):
```
HBM reads per token (batch=1) = 10 GB (shared) + 2 * 7.74 GB (2 active experts) = 25.5 GB
At 3.28 TB/s HBM: latency = 25.5 GB / 3.28 TB/s = 7.8 ms per token
Throughput: 128 ms/token, far below 100 tokens/second
```

For batch=1, this model is severely latency-limited unless weights are cached.

**Expert weight caching analysis:**

Can we cache the most popular experts on-chip to reduce HBM traffic?

Expert access frequency follows a Zipf distribution in practice: some experts are routed to
far more often than others. If the top-8 experts (6.25% of experts) handle 50% of all
routing decisions:

- Cache top-8 experts: 8 * 7.74 GB = 61.9 GB on fast memory.
- Requires 62 GB of on-chip or near-chip memory -- possible only with HBM used as L2 cache
  or very large on-chip SRAM (e.g., Cerebras WSE class).
- HBM traffic reduction: 50% of expert weight fetches eliminated = 990/2 = 495 GB saved per batch.

For a more practical 1 GB on-chip SRAM:
- Can cache 1 GB / 7.74 GB/expert = 0.13 experts -- not even one full expert fits.
- Caching is not viable at the expert level with typical on-chip SRAM.

**Practical optimisations for MoE streaming:**

1. **Expert grouping on HBM**: Store each expert's weights contiguously in HBM, sorted by
   expected activation frequency. Frequently-activated experts reside in lower-latency HBM
   channels. This reduces effective read latency without caching.

2. **Speculative expert prefetch**: Use the token's embedding vector to predict likely expert
   routes before the final routing decision (the routing is a simple linear layer -- its output
   can be estimated one step early). Prefetch the predicted expert weights while the routing
   computation completes. Works well when prediction accuracy > 70%.

3. **Expert replication**: In a multi-chip system, hot experts are replicated across more chips
   than cold experts. Routing sends tokens to the chip that already has the required expert
   loaded, achieving a distributed form of expert caching.

4. **Expert weight quantisation to INT4**: Reduces each expert from 7.74 GB to 3.87 GB,
   doubling the effective HBM bandwidth for expert weight streaming and enabling 2 experts to
   be cached in on-chip SRAM with a realistic 8 GB on-chip budget.

**Summary:**

| Scenario | HBM BW needed | Feasible at 3.28 TB/s? | Key optimisation |
|----------|--------------|------------------------|------------------|
| 100 tok/s, B=512 | 195 GB/s | Yes (6% utilisation) | None needed |
| 1000 tok/s, B=512 | 1.95 TB/s | Yes (59%) | Quantisation recommended |
| 10K tok/s, B=512 | 19.5 TB/s | No | Multi-chip + quantisation |
| Latency < 10ms, B=1 | 25.5 GB/token | 7.8ms -- borderline | Expert caching critical |
