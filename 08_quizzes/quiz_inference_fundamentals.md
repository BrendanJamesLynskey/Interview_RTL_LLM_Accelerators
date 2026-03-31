# Quiz: LLM Inference Fundamentals

## Overview

Self-assessment quiz covering transformer compute profiles, prefill vs. decode phases, memory
bandwidth analysis, quantisation formats, the roofline model, and arithmetic intensity. Questions
mix conceptual understanding with numerical reasoning. Answers and full explanations appear at the
bottom.

**Instructions:** Select the single best answer for each question. Work through numerical questions
on paper before checking the answer key.

---

## Section 1 — Compute Profiles and Arithmetic Intensity (Questions 1-4)

**Q1.** A transformer decoder processes a single new token during the decode phase. The weight
matrix for a linear projection has shape [4096, 4096] stored in FP16. Approximately how many bytes
must be read from memory to perform this matrix-vector multiply, and what is the arithmetic
intensity (FLOPs/byte)?

- A) 32 MB read, arithmetic intensity ≈ 1 FLOP/byte
- B) 32 MB read, arithmetic intensity ≈ 0.5 FLOP/byte
- C) 16 MB read, arithmetic intensity ≈ 0.5 FLOP/byte
- D) 16 MB read, arithmetic intensity ≈ 1 FLOP/byte

---

**Q2.** On a roofline model plot, a kernel is plotted well to the left of the ridge point. Which
description best characterises this kernel?

- A) Compute-bound: the bottleneck is ALU throughput, so increasing memory bandwidth will not help
- B) Memory-bandwidth-bound: the bottleneck is data movement, so improving compute peak will not help
- C) Latency-bound: the bottleneck is DRAM latency, and wider SIMD units would help
- D) Balanced: the kernel sits exactly at the roofline peak, exploiting both memory and compute fully

---

**Q3.** A GPU has a peak compute of 312 TFLOP/s (BF16) and a memory bandwidth of 3.35 TB/s. What
is the ridge-point arithmetic intensity (FLOPs/byte) for this device?

- A) ≈ 47 FLOP/byte
- B) ≈ 93 FLOP/byte
- C) ≈ 186 FLOP/byte
- D) ≈ 312 FLOP/byte

---

**Q4.** Which of the following operations has the HIGHEST arithmetic intensity during a standard
transformer forward pass (batch size = 1, sequence length = 1, i.e. single-token decode)?

- A) Loading the KV cache for multi-head attention
- B) A large matrix-matrix multiply (GEMM) during prefill with sequence length 2048
- C) A single feed-forward layer linear projection (matrix-vector multiply)
- D) Embedding table lookup for the current token

---

## Section 2 — Prefill vs. Decode (Questions 5-8)

**Q5.** During the prefill phase of LLM inference, which statement is most accurate?

- A) The phase is heavily memory-bandwidth-bound because only one token is processed at a time
- B) The phase is compute-bound for large sequence lengths because GEMM operations achieve high arithmetic intensity
- C) The phase is dominated by DRAM latency because each weight must be fetched from a separate memory bank
- D) The phase is equivalent to the decode phase in terms of hardware utilisation

---

**Q6.** A 7B-parameter model is run with batch size 1. During single-token decode, approximately
what fraction of the hardware's peak FLOP/s is typically achieved on a modern GPU?

- A) 85-95%
- B) 50-70%
- C) 20-40%
- D) 1-5%

---

**Q7.** In continuous batching (also called iteration-level scheduling), what problem does it
primarily solve compared to static batching?

- A) It eliminates the need for KV cache memory by recomputing attention at each step
- B) It allows sequences of different lengths to share a batch, reducing GPU idle time when some sequences finish early
- C) It batches the prefill and decode phases onto separate hardware units simultaneously
- D) It compresses the KV cache using delta encoding between consecutive tokens

---

**Q8.** A model has 32 transformer layers. Each layer has a self-attention block and an FFN block.
During decode with batch size B and current sequence length S, which operation grows with S?

- A) The FFN linear projections
- B) The QKV projection matrix-vector multiplies
- C) The attention score computation (Q * K^T) and the weighted sum (scores * V)
- D) The final logit projection over the vocabulary

---

## Section 3 — Memory Bandwidth and DRAM (Questions 9-11)

**Q9.** A model with 70B parameters stored in INT8 (1 byte per parameter) is served from HBM3 with
800 GB/s bandwidth. What is the minimum time to stream all weights past the compute units once,
and what does this represent as a lower bound for single-token decode latency?

- A) 70 ms — this is a hard lower bound assuming perfect bandwidth utilisation
- B) 87.5 ms — this is a hard lower bound assuming perfect bandwidth utilisation
- C) 175 ms — this is a hard lower bound assuming perfect bandwidth utilisation
- D) 700 ms — this is a hard lower bound assuming perfect bandwidth utilisation

---

**Q10.** Which HBM generation provides the highest memory bandwidth per device as of the hardware
available up to early 2025?

- A) HBM2
- B) HBM2e
- C) HBM3
- D) HBM3e

---

**Q11.** For a decode-phase workload that is memory-bandwidth-bound, which technique most directly
improves tokens-per-second throughput?

- A) Increasing the clock frequency of the compute array by 20%
- B) Increasing batch size until multiple sequences share the weight loading cost
- C) Widening the accumulator from 32-bit to 64-bit
- D) Adding more pipeline stages to the MAC unit

---

## Section 4 — Quantisation Formats (Questions 12-15)

**Q12.** In INT8 weight quantisation with FP16 activations (W8A16), which statement correctly
describes the compute strategy?

- A) Weights are dequantised to FP16 before the matrix multiply, so the GEMM executes in FP16
- B) Activations are quantised to INT8 before the matrix multiply, so the GEMM executes in INT8
- C) The GEMM executes in a mixed INT8 x FP16 mode natively supported by all modern tensor cores
- D) Weights remain as INT8 throughout and the accumulation is done in INT8

---

**Q13.** GPTQ is a post-training quantisation method. Which of the following best describes its
core mechanism?

- A) It uses knowledge distillation with a full-precision teacher model to calibrate the quantised student
- B) It applies the Optimal Brain Surgeon / second-order weight perturbation framework to compensate for quantisation error layer by layer
- C) It quantises weights uniformly and then fine-tunes the model end-to-end with quantisation-aware training
- D) It clusters weights into K groups using k-means and stores only the cluster centroids

---

**Q14.** FP8 (E4M3) has a dynamic range significantly smaller than FP16 (E5M10). What is the
primary hardware motivation for using FP8 in training and inference?

- A) FP8 eliminates the need for exponent bits, making addition logic simpler
- B) FP8 doubles the arithmetic throughput of tensor cores and halves the memory bandwidth requirement compared to FP16
- C) FP8 has higher precision than INT8 for non-uniform weight distributions, making it strictly superior in every dimension
- D) FP8 allows on-chip SRAM to be replaced with register files, reducing area

---

**Q15.** In a 4-bit NF4 (NormalFloat4) quantisation scheme, what property does NF4 exploit that
uniform INT4 does not?

- A) NF4 uses 5 bits for the exponent and eliminates the mantissa, giving wider dynamic range
- B) NF4 assigns quantisation levels at positions that are equally spaced in the quantile space of a normal distribution, not linearly spaced, giving lower quantisation error for normally distributed weights
- C) NF4 applies a logarithmic mapping so that powers of two are exactly representable
- D) NF4 stores two 4-bit values per byte using a combined entropy coder

---

## Section 5 — Roofline Model and Numerical Reasoning (Questions 16-18)

**Q16.** An accelerator has 100 TFLOP/s peak (FP16) and 1 TB/s memory bandwidth. A kernel
performs 500 GFLOP and reads 2 GB of data. What is the predicted execution time using the
roofline model?

- A) 2 ms (compute-bound)
- B) 5 ms (compute-bound)
- C) 2 ms (memory-bound)
- D) 5 ms (memory-bound)

---

**Q17.** The roofline model predicts the performance of a kernel as:

`Performance = min(Peak FLOP/s, Arithmetic Intensity x Peak Bandwidth)`

A team doubles the on-chip SRAM, effectively doubling the operational intensity of a tiled kernel.
If the kernel was previously memory-bound at 50% of peak compute, what happens?

- A) Throughput doubles regardless of whether the kernel was memory-bound
- B) Throughput doubles only if the new operating point is still below the ridge point
- C) Throughput is unchanged because roofline is determined by off-chip bandwidth, not SRAM
- D) Throughput doubles and the kernel becomes exactly compute-bound at the ridge point by definition

---

**Q18.** A matrix-vector multiply with an M x N weight matrix (M=8192, N=4096) runs in FP16 on a
device with ridge-point intensity 100 FLOP/byte. Is this operation memory-bound or compute-bound
for a single-vector input?

- A) Compute-bound, because the matrix has over 33 M elements
- B) Memory-bound, because the arithmetic intensity is approximately 1 FLOP/byte
- C) Balanced at the ridge point, because M/N = 2 which equals the FP16 bytes per element
- D) Cannot be determined without knowing the DRAM bandwidth

---

## Answer Key

| Q  | Answer |
|----|--------|
| 1  | A      |
| 2  | B      |
| 3  | B      |
| 4  | B      |
| 5  | B      |
| 6  | D      |
| 7  | B      |
| 8  | C      |
| 9  | B      |
| 10 | D      |
| 11 | B      |
| 12 | A      |
| 13 | B      |
| 14 | B      |
| 15 | B      |
| 16 | B      |
| 17 | B      |
| 18 | B      |

---

## Detailed Explanations

### Q1 — Correct: A

The weight matrix is 4096 x 4096 = 16,777,216 elements. At FP16 (2 bytes each), this is
33,554,432 bytes = **32 MB**.

The compute for a matrix-vector multiply is 2 x M x N FLOPs = 2 x 4096 x 4096 ≈ 33.6 GFLOP.
Wait — re-examining: that gives 33.6 G / 32 M ≈ 1050 FLOP/byte for batch size > 1. But for a
single vector (batch=1, decode), the multiply is M x N = 16.8 M multiplications and 16.8 M
additions = 33.6 GFLOP, and the weight bytes read is 32 MB, giving ~1050 FLOP/byte.

Correction: The question asks for bytes read and approximate intensity. Weights = 32 MB (correct).
The input vector is 4096 x 2 bytes = 8 KB (negligible). Output is also negligible.
FLOPs = 2 x 4096 x 4096 = 33.55 GFLOP.
Intensity = 33.55 GFLOP / 32 MB ≈ 1048 FLOP/byte.

**B is the closest answer** — 32 MB read, ~1 FLOP/byte is actually closest to A, but re-reading
option A says "≈ 1 FLOP/byte" and option B says "≈ 0.5 FLOP/byte". The true value ≈ 1048
FLOP/byte for a batched GEMM but for batch=1 decode the _effective_ intensity considering that each
weight byte is read once per output element is: we have 4096 output elements each consuming 4096
weight multiplies. Intensity = (2 x 4096 x 4096 FLOPs) / (4096 x 4096 x 2 bytes) = 2/2 = 1
FLOP/byte. **Answer A is correct: 32 MB, ~1 FLOP/byte.**

- B incorrect: 0.5 FLOP/byte would correspond to only multiplications without additions, which is
  not standard FLOP counting.
- C incorrect: The matrix is 4096x4096 in FP16 = 32 MB, not 16 MB.
- D incorrect: 16 MB weight size is wrong (see above).


### Q2 — Correct: B

Left of the ridge point means the arithmetic intensity is below the ridge-point value. In this
region, performance is limited by memory bandwidth: `perf = AI x bandwidth`. Increasing peak
compute has no effect because the kernel cannot consume data fast enough.

- A incorrect: Compute-bound kernels sit to the RIGHT of the ridge point where the roofline
  flattens at peak FLOP/s.
- C incorrect: DRAM latency is a related but separate concept. Roofline models bandwidth (GB/s),
  not latency (ns). Wider SIMD does not help a memory-bandwidth-bound kernel.
- D incorrect: Exactly at the ridge point is the balanced case; to the left means below it.

### Q3 — Correct: B

Ridge-point intensity = Peak FLOP/s / Peak Bandwidth = 312 x 10^12 / (3.35 x 10^12) ≈ 93.1
FLOP/byte.

- A incorrect: 47 FLOP/byte would correspond to halving the compute figure or doubling the
  bandwidth — neither matches these numbers.
- C incorrect: 186 FLOP/byte would require either double the compute or half the bandwidth.
- D incorrect: 312 FLOP/byte confuses the compute figure (in TFLOP/s) with the intensity in
  FLOP/byte — a dimensional error.

### Q4 — Correct: B

A large prefill GEMM (sequence 2048, large hidden dim) achieves arithmetic intensity = 2 x seq x d
FLOPs / (weight bytes + activation bytes). With seq=2048 and d=4096, the weight matrix reads are
amortised over 2048 input rows, giving intensity ≈ 2048 FLOP/byte — far above the ridge point
on nearly any accelerator.

- A incorrect: Loading the KV cache is nearly pure memory bandwidth with almost zero compute,
  giving intensity approaching 0.
- C incorrect: A matrix-vector multiply (single token, seq=1) achieves ~1 FLOP/byte as computed in
  Q1 — well below the ridge point.
- D incorrect: An embedding table lookup is pure indexed memory read — 0 arithmetic, so undefined
  (or 0) arithmetic intensity.

### Q5 — Correct: B

Prefill processes all prompt tokens simultaneously as a matrix-matrix multiplication. With large
sequence lengths (e.g., 2048+), the GEMM achieves high arithmetic intensity and approaches peak
compute throughput.

- A incorrect: This describes the DECODE phase, where batch=1 and only one token is processed per
  step.
- C incorrect: DRAM latency dominance would manifest as poor bandwidth utilisation, not what
  primarily characterises prefill. The weight matrix is accessed sequentially with high reuse.
- D incorrect: Prefill and decode have fundamentally different computational profiles; they are not
  equivalent.

### Q6 — Correct: D

Single-token decode with batch=1 is severely memory-bandwidth-bound. The arithmetic intensity is ~1
FLOP/byte, which is far below the ridge point (often 50-200 FLOP/byte) of modern GPUs. As a result,
compute utilisation is typically 1-5% of peak FLOP/s.

- A incorrect: 85-95% utilisation is achievable for large batched GEMMs during prefill, not decode.
- B incorrect: 50-70% is optimistic for prefill but not characteristic of single-token decode.
- C incorrect: 20-40% would represent a partially memory-bound workload but decode is far more
  extreme.

### Q7 — Correct: B

Continuous batching allows new requests to join and completed sequences to leave at each decode
step, rather than waiting for the entire static batch to finish. This dramatically reduces GPU idle
time caused by variable-length sequences.

- A incorrect: Continuous batching does not eliminate the KV cache; the KV cache is still necessary
  and is managed per-sequence.
- C incorrect: Continuous batching is about dynamic batch membership, not splitting prefill and
  decode across hardware units (that is speculative decoding or pipeline parallelism).
- D incorrect: Delta encoding of the KV cache is unrelated to scheduling strategy.

### Q8 — Correct: C

The attention score computation requires multiplying Q (shape [H, d_k]) by K^T (shape [d_k, S]),
giving a matrix of shape [H, S]. The weighted sum requires multiplying scores [H, S] by V [S,
d_v]. Both operations grow linearly with sequence length S.

- A incorrect: FFN projections operate on the current token's hidden state only. Their size is
  independent of sequence length during decode.
- B incorrect: QKV projections are matrix-vector multiplies on the current token; they do not grow
  with S.
- D incorrect: The vocabulary projection is fixed in size (hidden_dim x vocab_size) regardless of
  sequence length.

### Q9 — Correct: B

70B parameters x 1 byte/param = 70 GB. Time = 70 GB / 800 GB/s = 0.0875 s = **87.5 ms**.

This is a lower bound because it assumes 100% bandwidth utilisation and no compute bottleneck;
real decode will be at least this slow.

- A incorrect: 70 ms would require 1 TB/s bandwidth (70 GB / 70 ms = 1 TB/s), which is incorrect.
- C incorrect: 175 ms would correspond to 70 GB / 400 GB/s, not the given 800 GB/s.
- D incorrect: 700 ms is 10x too large; this would require only 100 GB/s bandwidth.

### Q10 — Correct: D

HBM3e (as used in NVIDIA H200 and AMD MI325X) delivers approximately 4.8 TB/s per device, the
highest of these options as of early 2025.

- A incorrect: HBM2 provides ~1 TB/s per device (e.g., A100 40GB variant).
- B incorrect: HBM2e provides ~2 TB/s (e.g., A100 80GB).
- C incorrect: HBM3 provides ~3.35 TB/s (H100 SXM), which is less than HBM3e.

### Q11 — Correct: B

For a memory-bandwidth-bound workload, the key insight is that weights are the bottleneck. By
increasing batch size, multiple token computations share a single weight-loading pass, so the
effective weight bytes per token decreases and throughput improves proportionally.

- A incorrect: A 20% clock increase raises compute throughput, but for a memory-bandwidth-bound
  kernel, the compute units are already idle most of the time. Throughput gain is negligible.
- C incorrect: Accumulator precision affects numerical correctness, not memory bandwidth or
  throughput.
- D incorrect: Adding pipeline stages can improve clock frequency but does not address the
  memory-bandwidth bottleneck.

### Q12 — Correct: A

W8A16 quantisation stores weights as INT8 to reduce memory footprint and bandwidth, but the
actual matrix multiply is performed in FP16 after dequantising weights on-the-fly. This is the
approach used by bitsandbytes and similar libraries.

- B incorrect: Quantising activations to INT8 would describe W8A8, not W8A16.
- C incorrect: INT8 x FP16 mixed-type tensor cores are not natively supported on most hardware;
  the multiply-accumulate pipeline requires both inputs to be the same type.
- D incorrect: INT8 accumulation would cause rapid overflow for large dot products (vectors of
  length 4096+ with INT8 values); INT32 or FP32 accumulators are required.

### Q13 — Correct: B

GPTQ (Frantar et al., 2022) applies the Optimal Brain Quantisation (OBQ) method, which itself
derives from the Optimal Brain Surgeon framework. It uses the Hessian of the layer's loss to find
weight perturbations that compensate for the error introduced by quantising each weight.

- A incorrect: Knowledge distillation describes a separate family of methods (e.g., DistilBERT).
  GPTQ uses a calibration dataset to compute the Hessian but does not train a teacher-student pair.
- C incorrect: End-to-end fine-tuning with quantisation-aware training (QAT) describes methods like
  QLoRA or standard QAT, not GPTQ which is a one-shot post-training method.
- D incorrect: K-means weight clustering describes product quantisation or vector quantisation
  methods (e.g., used in some early compression work), not GPTQ.

### Q14 — Correct: B

FP8 uses 1 byte instead of 2 bytes (FP16) or 4 bytes (FP32), which halves bandwidth requirements
and doubles the number of multiply-accumulate operations tensor cores can perform per cycle when
data movement is the bottleneck. This is the primary motivation.

- A incorrect: FP8 still has exponent bits (E4M3 has 4 exponent bits, E5M2 has 5). The addition
  logic is not simpler — in fact, floating-point addition is more complex than integer addition
  because of exponent alignment.
- C incorrect: While FP8 does have better precision for non-uniform distributions than INT8, calling
  it "strictly superior in every dimension" is false — INT8 has wider dynamic range per bit because
  it can represent integers exactly, and the claim about replacing SRAM is baseless.
- D incorrect: FP8 affects the compute datapath and memory format, not the type of on-chip storage
  (SRAM vs. register file). SRAM is used for both.

### Q15 — Correct: B

NF4 places quantisation breakpoints at the quantiles of a standard normal distribution. Because
neural network weights are approximately normally distributed, this concentrates representational
capacity where weights are most dense, minimising mean squared quantisation error.

- A incorrect: NF4 does not use 5 exponent bits and eliminate the mantissa — that would describe
  something like a floating-point format with no significand, not NF4.
- C incorrect: A logarithmic mapping would suit exponentially distributed values, not normally
  distributed ones, and NF4 is not logarithmic.
- D incorrect: NF4 is a scalar quantisation format; it does not use entropy coding. Two 4-bit
  values are stored per byte (bit-packing), but this is memory layout, not entropy coding.

### Q16 — Correct: C

Arithmetic intensity = 500 GFLOP / 2 GB = 250 FLOP/byte.

Ridge-point intensity = 100 TFLOP/s / 1 TB/s = 100 FLOP/byte.

Since 250 > 100, the kernel is **compute-bound**. Wait — re-examining: the kernel's intensity (250)
exceeds the ridge point (100), so the kernel IS compute-bound. Predicted time = 500 GFLOP / 100
TFLOP/s = 0.005 s = 5 ms.

The memory time would be: 2 GB / 1 TB/s = 2 ms.

Since compute-bound, time = 5 ms.

**Correction: The correct answer is A (5 ms, compute-bound).**

- A: 5 ms compute-bound — CORRECT.
- B: 5 ms is the right time, but memory-bound is wrong. (This choice is listed as B = "5 ms,
  compute-bound", which matches A in the original listing — see the option text.)
- C: 2 ms memory-bound is incorrect — the kernel is compute-bound.
- D: 5 ms memory-bound — the time is right but the bound characterisation is wrong.

Re-reading the options: A = "2 ms (compute-bound)", B = "5 ms (compute-bound)", C = "2 ms
(memory-bound)", D = "5 ms (memory-bound)". The correct answer is **B**: 5 ms, compute-bound.


- A incorrect: 2 ms is the memory-bound prediction, and the kernel is compute-bound, so this is
  doubly wrong.
- C incorrect: 2 ms is memory-bound time — wrong characterisation and wrong time for the actual
  bottleneck.
- D incorrect: The time is right (5 ms) but the characterisation as memory-bound is wrong.

### Q17 — Correct: B

If the kernel was memory-bound, doubling operational intensity moves its operating point rightward
on the roofline plot. If the new point is still below the ridge, it remains memory-bound and
throughput doubles (performance = AI x BW, so doubling AI doubles performance). If the new point
crosses the ridge, it becomes compute-bound and throughput does not fully double.

- A incorrect: Throughput doubling is only guaranteed while still memory-bound. If doubling AI
  crosses the ridge, the gain is less than 2x.
- C incorrect: SRAM affects tile size, which directly affects how many times each weight is reused
  per DRAM fetch, i.e., it affects operational intensity. The claim that SRAM has no effect is
  false.
- D incorrect: There is no guarantee that doubling AI lands exactly at the ridge point; that would
  be a coincidence.

### Q18 — Correct: B

For a matrix-vector multiply with a single input vector, the weight matrix (M x N = 8192 x 4096
elements x 2 bytes = 64 MB) is read once. The FLOPs are 2 x 8192 x 4096 ≈ 67 GFLOP. Arithmetic
intensity = 67 GFLOP / 64 MB ≈ 1.05 FLOP/byte. The ridge point is 100 FLOP/byte, so the kernel
is far to the left — **memory-bound**.

- A incorrect: The number of elements is irrelevant to whether a kernel is compute-bound. The
  intensity is what matters, and ~1 FLOP/byte << 100 FLOP/byte ridge.
- C incorrect: M/N = 2 is a dimensional ratio, not an intensity. The 2 bytes/element factor does
  not combine with M/N to give the ridge-point intensity.
- D incorrect: The ridge-point intensity is entirely determined by peak FLOP/s and peak bandwidth,
  both of which are given. The comparison is straightforward without needing additional DRAM
  bandwidth information.

---

