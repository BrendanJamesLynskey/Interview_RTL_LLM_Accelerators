# Matrix-Vector vs. Matrix-Matrix Operations in LLM Inference

## Overview

LLM inference alternates between two fundamentally different compute regimes: GEMM (General Matrix-Matrix Multiplication) during prefill and GEMV (General Matrix-Vector Multiplication) during autoregressive decode. These two operations have different arithmetic intensity, different hardware bottlenecks, and require different hardware optimisation strategies. This topic appears frequently in interviews for inference accelerator roles.

---

## Tier 1 — Fundamentals

### Q1. What is the difference between GEMV and GEMM, and when does each occur in LLM inference?

**Question:** Define GEMM and GEMV, give the shape of each during inference, and explain which phase of LLM inference each corresponds to.

**Answer:**

**GEMM (General Matrix-Matrix Multiplication):**
- Operation: C = A x B, where A has shape (M, K) and B has shape (K, N).
- Both input operands are matrices.
- The output C has shape (M, N).
- Arithmetic: M x N x K multiply-accumulate operations.

**GEMV (General Matrix-Vector Multiplication):**
- Operation: y = W x, where W has shape (N, K) and x has shape (K,).
- One input is a matrix, the other is a vector.
- The output y has shape (N,).
- Equivalent to GEMM with M=1: (1, K) x (K, N) = (1, N).
- Arithmetic: N x K multiply-accumulate operations.

**When each occurs in LLM inference:**

*Prefill (prompt processing):*
All tokens in the input prompt are processed in parallel. If the prompt has S tokens, then the Q, K, V projections each perform a GEMM of shape (S, hidden_dim) x (hidden_dim, head_dim * n_heads). With S >> 1, this is a true GEMM and the compute is the bottleneck.

*Decode (token generation):*
Each new token is generated one at a time. At each step, there is only one new token, so all projections degrade to GEMV: (1, hidden_dim) x (hidden_dim, hidden_dim). With M=1, the operation is memory-bandwidth-bound — the weight matrix must be loaded from SRAM/HBM to produce only a single output vector.

**Summary:**

| Phase | M dimension | Operation | Bottleneck |
|---|---|---|---|
| Prefill | S (seq_len, often 128-8192) | GEMM | Compute |
| Decode, batch=1 | 1 | GEMV | Memory bandwidth |
| Decode, batch=B | B | GEMM (small M) | Memory bandwidth until B is large |

---

### Q2. What is arithmetic intensity and why does it matter for GEMV vs. GEMM?

**Question:** Define arithmetic intensity. Calculate the arithmetic intensity of GEMM and GEMV for a (4096 x 4096) weight matrix. What does this tell you about hardware bottlenecks?

**Answer:**

**Arithmetic Intensity (AI):**

AI = FLOPs / Bytes_of_memory_traffic

It measures how many arithmetic operations are performed per byte loaded from memory. It is the key metric for determining whether a workload is compute-bound or memory-bandwidth-bound.

**GEMV — y = Wx, W shape (4096, 4096), x shape (4096,):**

- FLOPs: 2 x 4096 x 4096 = 33.6 MFLOP (factor of 2 for multiply + add).
- Bytes read: W = 4096 x 4096 x 2 bytes (FP16) = 33.6 MB; x = 4096 x 2 = 8 KB; y = 4096 x 2 = 8 KB.
- Total bytes: ~33.6 MB.
- AI = 33.6 MFLOPs / 33.6 MB = **~1.0 FLOP/byte**.

**GEMM — C = AB, A shape (4096, 4096), B shape (4096, 4096):**

- FLOPs: 2 x 4096 x 4096 x 4096 = 137.4 GFLOP.
- Bytes read: A = 33.6 MB, B = 33.6 MB, C = 33.6 MB.
- Total bytes: ~100.7 MB.
- AI = 137.4 GFLOP / 100.7 MB = **~1,364 FLOP/byte**.

**Implication using the roofline model:**

On an H100 GPU: compute peak (FP16) = 989 TFLOPS, memory bandwidth = 3.35 TB/s.
Ridge point (compute/bandwidth crossover) = 989 TFLOPS / 3.35 TB/s = ~295 FLOP/byte.

- GEMV AI (~1) << ridge point (295): GEMV is **deeply memory-bandwidth-bound**. Adding more FLUs does nothing; you need more bandwidth.
- GEMM AI (~1364) >> ridge point (295): GEMM is **compute-bound**. More FLUs improve performance; bandwidth is not the limiter.

This is the mathematical basis for why decode is bandwidth-bound and prefill is compute-bound.

---

### Q3. Why does batching improve decode efficiency?

**Question:** A model has hidden_dim=4096. You have a hardware accelerator with 1 TFLOPS peak compute and 2 TB/s memory bandwidth. Show numerically why increasing batch size during decode improves utilisation.

**Answer:**

**Hardware ridge point:** 1 TFLOPS / 2 TB/s = 500 FLOP/byte. Any workload with AI < 500 is bandwidth-bound.

**For a single linear layer W of shape (4096, 4096), INT8:**

Weight size = 4096 x 4096 x 1 byte = 16 MB.

For batch size B, the GEMM is (B, 4096) x (4096, 4096):
- FLOPs: 2 x B x 4096 x 4096 = 2B x 16.8M = 33.6B MFLOP.
- Bytes: W (16 MB, loaded once) + activations (B x 4096 x 1 = 4B KB) + output (same).
  - For large W: bytes ≈ 16 MB (W dominates when B is small).
- AI ≈ 33.6B MFLOP / 16 MB = **2.1B FLOP/byte**.

At B=1: AI = 2.1. Compute used = min(1 TFLOPS, 2 TB/s x 2.1) = min(1 TFLOPS, 4.2 TFLOPS) = **limited to 4.2 TFLOPS equivalent** — but peak is 1 TFLOPS, so bandwidth delivers 2 TB/s x 2.1 FLOP/byte = 4.2 TFLOPS effective, but we are capped at 1 TFLOPS. Actually we are bandwidth-limited: throughput = BW x AI = 2e12 x 2.1 = 4.2e12 FLOPS which exceeds compute. So we are compute-bound at B=1? Let's recalculate: at B=1, FLOPs = 33.6M, bandwidth needed = 16 MB, time at peak bandwidth = 16e6/2e12 = 8 us, time at peak compute = 33.6e6/1e12 = 0.034 us. **Bandwidth is the bottleneck**: time is 8 us, compute is idle 99.6% of the time.

At B=238 (ridge point): AI = 2.1 x 238 = 500 = ridge point. Both compute and bandwidth are exactly balanced.

At B=1024: AI = 2.1 x 1024 = 2150. Compute-bound. Throughput scales with compute, not bandwidth.

**Practical conclusion:** For this hardware, batching beyond ~238 does not improve throughput per token further (you are now compute-bound). The "sweet spot" for batch size is approximately the ridge point AI divided by the per-sample AI.

---

## Tier 2 — Intermediate

### Q4. How do you design a flexible datapath that handles both GEMM and GEMV efficiently on the same hardware?

**Question:** You are designing an accelerator datapath that must handle both prefill (GEMM, large M) and decode (GEMV, M=1) for a transformer with hidden_dim=4096, n_heads=32. Describe the key design decisions and trade-offs.

**Answer:**

**The core tension:** GEMM favours large 2D arrays operating on wide tiles (high compute intensity); GEMV favours wide parallel dot products with broadcast activation (high memory bandwidth with minimal compute reuse).

**Design decision 1: Array shape and tile strategy**

A square N x N array is optimal for GEMM (equal utilisation in M and N). For GEMV, only one row is ever active (M=1), so utilisation is 1/N.

Two approaches:
- **Tall array (M > N tiles):** Better for GEMV since more rows can be packed. For example, a 4096x64 array with 64 weights per row. At decode, a single input vector drives all 4096 rows simultaneously, each producing one partial output. The 4096 outputs are produced in K/64 cycles. This functions as a broadcast-and-accumulate engine.
- **Square array with broadcast path:** A 64x64 array with an added broadcast bus. In GEMM mode, activations shift in row by row. In GEMV mode, the same activation is broadcast to all 64 rows simultaneously. This uses the same array area but adds routing for the broadcast bus (~10% area overhead).

**Design decision 2: Memory access pattern**

For GEMM: Weight tiles and activation tiles are both 2D; both benefit from double-buffered tile loading. Bandwidth to compute ratio favours pipelining tile loads.

For GEMV: The weight matrix is loaded column-by-column (or in tiles) but the activation is a single vector loaded once. The bottleneck is entirely weight bandwidth. The optimal memory strategy is to maximise weight load bandwidth and minimise activation overhead.

Implication: SRAM port configuration matters. For GEMV, you want a single wide read port serving the weight path. For GEMM, you want balanced bandwidth for both activation and weight paths. A common solution is a shared SRAM with a 2:1 bandwidth allocation (weights:activations) which slightly favours GEMV without harming GEMM significantly.

**Design decision 3: Accumulator handling**

GEMM: Partial sums accumulate across K-tiles, require 32-bit or FP32 accumulation, and are written to output SRAM at the end of K-tile iteration.

GEMV: Output has only N elements (4096 for a 4096x4096 projection). The entire output fits in a small accumulation register file (4096 x 4 bytes = 16 KB). This register file can be kept live across K-tile iterations without writing to SRAM, reducing output bandwidth.

**Unified design recommendation:**
- 256x256 array with output-stationary dataflow.
- Broadcast mux on the left input bus (GEMM: shift, GEMV: broadcast single row).
- 16 KB output accumulation register file that bypasses SRAM for GEMV.
- 2:1 weight:activation SRAM bandwidth.
- Estimated area overhead for GEMV support over GEMM-only: ~15%.

---

### Q5. In attention computation during decode, which operations are GEMV and which have higher M?

**Question:** For a transformer with hidden_dim=4096, n_heads=32, head_dim=128, KV cache length of 4096 tokens, and batch=1 decode step, identify each operation's shape and classify it.

**Answer:**

**Layer structure during a single decode step (batch=1, one new token):**

**(a) QKV projection:**
- Input: x shape (1, 4096).
- Weight: W_Q, W_K, W_V each (4096, 4096).
- Output: q, k, v each (1, 4096) = (1, n_heads x head_dim) = (1, 32 x 128).
- Classification: GEMV (M=1). Memory-bandwidth-bound.

**(b) Attention score computation — Q x K^T:**
- Q for new token: (1, head_dim) = (1, 128) per head.
- K from KV cache: (4096, 128) per head (all previous tokens).
- Score shape: (1, 4096) per head.
- This is: (1, 128) x (128, 4096) = GEMV with M=1, K=128, N=4096.
- Classification: GEMV. M=1 but K=128 is small so arithmetic intensity = 2*128*4096 / (128*4096*2 + 4096*2) ≈ 1 FLOP/byte. Very bandwidth-bound.

**(c) Attention score x V:**
- Softmax scores: (1, 4096) per head.
- V from KV cache: (4096, 128) per head.
- Output: (1, 128) per head.
- This is: (1, 4096) x (4096, 128) = GEMV with M=1, K=4096, N=128.
- Classification: GEMV. The KV cache must be read in full.

**(d) Output projection:**
- Input: (1, n_heads x head_dim) = (1, 4096).
- Weight: (4096, 4096).
- Output: (1, 4096).
- Classification: GEMV.

**(e) FFN layer (2 linear layers):**
- Input: (1, 4096). Weight: (4096, 16384) [4x expansion].
- Classification: GEMV (M=1), very large N.

**Summary:** During decode, every linear operation is GEMV. The attention score operations are also effectively GEMV, but the KV cache reads are the dominant bandwidth cost (scales with context length, not model width).

**Interview insight:** The KV cache attention operations become increasingly bandwidth-bound as context length grows. At 4096-token context, the KV cache for a single layer at INT8 is 2 * 4096 * 4096 * 1 byte = 32 MB per layer. For a 32-layer model, that is 1 GB of KV cache bandwidth per decode step — larger than the model weights for short sequences.

---

### Q6. How does multi-query attention (MQA) and grouped-query attention (GQA) change the GEMV vs. GEMM balance?

**Question:** Standard multi-head attention (MHA) uses n_heads Q heads and n_heads K/V heads. MQA uses 1 K/V head; GQA uses n_kv_groups K/V heads. Describe how this changes the compute and memory bandwidth characteristics.

**Answer:**

**MHA — standard multi-head attention:**
- n_heads Q, K, V = 32 each (for a typical 7B model).
- KV projection: W_K, W_V both shape (4096, 4096). GEMV: (1, 4096) x (4096, 4096).
- KV cache size per layer: 2 x seq_len x n_heads x head_dim bytes = 2 x seq_len x 4096 bytes.

**MQA — multi-query attention:**
- n_heads Q = 32, n_kv_heads = 1.
- W_K, W_V shape (4096, 128) instead of (4096, 4096). Projection GEMV is 32x smaller.
- KV cache per layer: 2 x seq_len x 1 x 128 = 2 x seq_len x 128 bytes. 32x reduction.
- Attention score computation: Q x K^T is now (32 heads of Q) x (1 head of K), requiring broadcast of K across all Q heads. The compute is the same but the K matrix is 32x smaller.
- **Hardware implication:** KV cache bandwidth in decode is reduced 32x. This dramatically improves memory-bandwidth-bound decode throughput.

**GQA — grouped query attention (used in LLaMA-2 70B, Mistral, Gemma):**
- n_kv_groups = g (e.g., g=8 for LLaMA-2 70B with 64 Q heads and 8 KV heads).
- KV cache: 2 x seq_len x g x head_dim bytes. Reduction factor = n_heads/g = 64/8 = 8x vs. MHA.
- W_K, W_V shape: (4096, g x head_dim). Projection GEMV is g x smaller.
- Quality closer to MHA than MQA at the same model size.

**Hardware design impact:**

For GEMV (decode), the bandwidth for KV projection and KV cache read is:

| Attention type | KV projection (GEMV) weight bytes | KV cache bandwidth/step (4096 ctx) |
|---|---|---|
| MHA (32 heads) | 33.6 MB per W_K + W_V | 32 MB/layer |
| GQA (8 groups) | 8.4 MB per W_K + W_V | 8 MB/layer |
| MQA (1 group) | 1.05 MB per W_K + W_V | 1 MB/layer |

A hardware design targeting MQA or GQA models can use a narrower KV cache SRAM with fewer read ports, freeing area for larger activation SRAM or wider weight load paths.

---

## Tier 3 — Advanced

### Q7. Design a unified datapath controller that switches between GEMM and GEMV modes. What state must be tracked and how are tiles dispatched differently?

**Question:** Describe the microarchitecture of a dispatch unit that sends tiles to a 256x256 systolic array. The unit must support GEMM mode (M >= 256) and GEMV mode (M < 16). What signals change, what buffers change, and what is the impact on back-pressure handling?

**Answer:**

**Common datapath elements (mode-independent):**
- Weight tile DMA: loads 256x256 INT8 tiles from HBM into weight SRAM double-buffer. Always active.
- Systolic array: 256x256 MAC array with output-stationary or weight-stationary dataflow.
- Output SRAM: stores completed output tiles.

**GEMM mode (M >= 256, M is a multiple of 256 for simplicity):**

Dispatch controller state:
- `m_tile_idx`: current row tile index (0 to M/256 - 1).
- `n_tile_idx`: current col tile index (0 to N/256 - 1).
- `k_iter`: current K-tile iteration (0 to K/256 - 1).

Dispatch sequence:
1. Load weight tile W[n][k] into weight SRAM (256x256, 64 KB at INT8).
2. Load activation tile A[m][k] into activation SRAM (256x256, 64 KB at INT8).
3. Send array start signal. Array runs for 256 + 2*(256-1) + fill cycles.
4. Accumulate outputs in output SRAM. After K-tiles complete, store output tile C[m][n].
5. Increment m, n, k indices.

Back-pressure: Weight DMA and activation DMA are double-buffered. If the next tile is not ready when the array completes, assert `stall` for that many cycles. Stalls are rare because weight tiles are large (64 KB) and HBM bandwidth is sufficient to preload within one array computation time.

**GEMV mode (M = 1 to 15):**

Key differences:
- The activation "tile" is a small vector: M x 256 bytes (at most 15 x 256 = 3.75 KB). This fits entirely in a small register file; no SRAM double-buffer needed for activations.
- The activation vector is broadcast to all M rows of the array. For M=1, a single row is replicated across all 256 rows of the array (or the array runs with only row 0 active and the result is gathered).
- Output is a small vector: M x 256 elements = at most 3840 elements. Fits in a 16 KB accumulation register file.
- Because the activation is tiny, the dispatch cycle time is dominated by weight DMA. The controller should pre-issue weight DMA requests aggressively.

Additional dispatch controller state for GEMV:
- `acc_regfile[N]`: 32-bit accumulators for each output element, kept live across K-tile iterations.
- `broadcast_en`: signal to the array to replicate row 0 activation to all rows.
- `k_accum_en`: signal to accumulate into `acc_regfile` rather than writing to output SRAM.
- After all K-tiles complete: write `acc_regfile` to output SRAM (single write, small).

**Back-pressure in GEMV:**

In GEMV, the array completes a K-tile in approximately 256 + fill cycles, but only produces M output rows (possibly M=1). The throughput is determined entirely by weight DMA bandwidth. Back-pressure manifests when the weight DMA cannot keep up — this occurs if HBM effective bandwidth < (256x256 bytes) / (256 + fill cycles).

At 1 GHz and INT8: array processes 256 K-elements in ~256 cycles (1 weight column per cycle). Weight DMA must deliver 256 x 256 = 64 KB every ~256 ns = 250 GB/s. HBM3 (819 GB/s) is sufficient for one 256x256 array; two arrays would require careful scheduling.

**Mode switching latency:**
Switching from GEMM to GEMV requires flushing the activation double-buffer (write back any partial state) and resetting the output SRAM pointer to use `acc_regfile` instead. This takes 2-4 cycles plus one pipeline flush (proportional to array fill depth). For a 256-deep array, mode switch overhead is approximately 256 + 4 cycles = 260 cycles. At 1 GHz, this is 260 ns — negligible compared to compute time for any realistically sized matrix.

---

### Q8. How does the GEMM-vs-GEMV distinction interact with quantisation strategy?

**Question:** Explain why INT8 weight quantisation benefits GEMV more than it benefits GEMM in terms of memory bandwidth efficiency. How does this inform which quantisation schemes are deployed for inference?

**Answer:**

**GEMV bandwidth analysis with quantisation:**

For a GEMV with weight matrix W of shape (N, K) = (4096, 4096):

Weight bytes at different precisions:
- FP32: 4 x 4096 x 4096 = 64 MB
- FP16/BF16: 2 x 4096 x 4096 = 32 MB
- INT8: 1 x 4096 x 4096 = 16 MB
- INT4: 0.5 x 4096 x 4096 = 8 MB

For a fixed memory bandwidth B GB/s, the time to load weights for one GEMV:
- FP16: 32 MB / B seconds. At 2 TB/s (H100): 16 us.
- INT8: 16 MB / B seconds. At 2 TB/s: 8 us. **2x throughput improvement.**
- INT4: 8 MB / B seconds. At 2 TB/s: 4 us. **4x throughput improvement.**

Compute time for GEMV at INT8: 2 x 4096 x 4096 ops / 1e12 ops/s = 0.034 us. Negligible vs. bandwidth time.

**Conclusion for GEMV:** Quantisation directly translates to proportional decode throughput improvement because the operation is 100% bandwidth-bound.

**GEMM bandwidth analysis with quantisation:**

For a GEMM (M=4096, K=4096, N=4096) at AI = 1364 FLOP/byte (FP16):
- At INT8: weight bytes halved, but FLOPs also halved. AI = 1364 / (32/16) = 1364 FLOP/byte (unchanged if compute counts INT8 ops with FP16 equivalent).
- Actually: INT8 FLOPs = 2 x M x K x N / (2 for mixed precision) — the arithmetic intensity stays approximately the same because both numerator and denominator scale together.
- For GEMM in the compute-bound regime, quantisation helps by doubling the compute throughput (same number of elements, half the bits, twice as many ops per cycle on a fixed-width array), but the throughput gain is bounded by compute, not bandwidth.

**Why W4A16 is a common production choice:**

Weight-only INT4 quantisation (W4A16) stores weights at 4-bit but multiplies in FP16:
- For GEMV: 4x bandwidth reduction = 4x decode throughput improvement.
- For GEMM: weights are dequantised to FP16 before multiplication, so compute throughput is unchanged from FP16 GEMM, but weight bandwidth is 4x lower, which helps fill the prefill pipeline faster.
- Accuracy: weight-only quantisation to 4-bit is well-tolerated by models with >7B parameters, especially with group quantisation (per-group scales).

**Hardware implication:** A chip targeting W4A16 must include a fast INT4-to-FP16 dequantisation unit in the weight load path, operating at full weight DMA bandwidth (e.g., 2 TB/s x 4 bits per element = 500 billion INT4 elements per second converted to FP16).

---
