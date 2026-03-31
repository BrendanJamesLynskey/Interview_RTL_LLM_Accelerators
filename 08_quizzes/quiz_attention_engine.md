# Quiz: Attention Engine Hardware

## Overview

Self-assessment quiz covering multi-head attention hardware implementation, numerically stable
softmax, KV cache design, Rotary Position Embedding (RoPE) hardware, Grouped-Query and
Multi-Query Attention, FlashAttention memory implications, multi-head parallelism, and online
softmax algorithms. Questions target RTL and microarchitecture design decisions.

**Instructions:** Select the single best answer for each question. Answers and full explanations
appear at the bottom.

---

## Section 1 — Softmax Implementation (Questions 1-3)

**Q1.** Naive softmax computes `exp(x_i) / sum(exp(x_j))`. Why is this numerically unsafe for
hardware with limited floating-point range?

- A) The division operation is not supported in fixed-point arithmetic, causing silent errors
- B) For large positive logits, `exp(x_i)` overflows the floating-point range before the sum can
   normalise it, producing infinity or NaN
- C) The sum in the denominator may underflow to zero when all logits are very negative, making
   the division undefined
- D) Softmax requires sorting, which is O(N log N) and introduces data hazards in pipelined hardware

---

**Q2.** The safe softmax trick subtracts the maximum value: `exp(x_i - max) / sum(exp(x_j - max))`.
In hardware, computing the exact maximum requires a full pass over all logits. FlashAttention
avoids this two-pass approach using online softmax. What is the core idea of online softmax?

- A) Approximating the softmax with a piecewise-linear exponential to avoid needing the true maximum
- B) Maintaining a running maximum m and a running sum of shifted exponentials s, updating both as
   each new logit arrives, then rescaling accumulated partial outputs when the maximum changes
- C) Using a look-up table (LUT) indexed by the logit value to approximate exp() in a single cycle
- D) Processing logits in parallel using a butterfly reduction tree to find the maximum and sum
   simultaneously in O(log N) time

---

**Q3.** In a hardware softmax unit, the exponential function `exp(x)` is commonly approximated.
Which hardware implementation offers the best trade-off between area and accuracy for a production
accelerator?

- A) Taylor series expansion to 10 terms, computed in a combinational circuit for maximum speed
- B) A small lookup table for the fractional part combined with a shift for the integer part,
   exploiting `exp(x) = 2^(x / ln2)` decomposed into integer and fractional components
- C) An iterative Newton-Raphson solver that converges in 2-3 cycles
- D) A cordic rotation in hyperbolic mode, which naturally computes exponentials

---

## Section 2 — KV Cache (Questions 4-6)

**Q4.** During autoregressive decode, the KV cache stores key and value tensors for all previous
tokens. For a model with H attention heads, head dimension d_k, and a current sequence of S tokens,
what is the size of the KV cache for a single layer in bytes, assuming FP16?

- A) 2 * H * d_k * S * 1 byte (INT8 format)
- B) 2 * H * d_k * S * 2 bytes (FP16 format)
- C) H * d_k * S * 2 bytes (only keys, not values)
- D) 4 * H * d_k * S * 2 bytes (Q, K, V, and output projections)

---

**Q5.** PagedAttention (used in vLLM) addresses KV cache memory fragmentation. What is the
hardware-relevant insight?

- A) It stores the KV cache in a compressed format using run-length encoding to save memory
- B) It partitions the KV cache into fixed-size pages that can be allocated non-contiguously,
   allowing dynamic memory management similar to virtual memory paging in operating systems
- C) It moves the KV cache from HBM to the host CPU DRAM to save accelerator memory
- D) It quantises the KV cache to INT4, halving memory at the cost of attention accuracy

---

**Q6.** A 70B parameter model with 80 transformer layers, 64 attention heads, head dimension 128,
and FP16 precision is serving requests with a maximum sequence length of 4096 tokens. What is the
KV cache size per request (both K and V, all layers)?

- A) 5 GB
- B) 10 GB
- C) 20 GB
- D) 40 GB

---

## Section 3 — Rotary Position Embedding (RoPE) (Questions 7-9)

**Q7.** RoPE applies a rotation to query and key vectors based on their absolute position index.
What property of RoPE makes it particularly useful for extending context length beyond the
training window?

- A) RoPE is a learned embedding that can be fine-tuned on longer sequences with minimal parameters
- B) RoPE encodes relative position information through rotation, so the dot product Q_m * K_n
   depends only on the relative position (m - n), enabling positional generalisation
- C) RoPE adds a scalar position bias to each attention logit, which can be extrapolated linearly
   beyond the training range
- D) RoPE eliminates the need for position encoding altogether by encoding order in the attention mask

---

**Q8.** In hardware, applying RoPE to a query vector q of dimension d requires rotating pairs of
elements. For a 128-dimensional query vector at position m, how many complex rotations (or
equivalent 2x2 rotation matrix applications) are needed?

- A) 1 rotation (one global rotation of the entire vector)
- B) 64 rotations (one per pair of dimensions, each using a different frequency)
- C) 128 rotations (one per dimension)
- D) 256 rotations (two per dimension, for real and imaginary components)

---

**Q9.** When implementing RoPE in RTL, the rotation requires computing `cos(m * theta_i)` and
`sin(m * theta_i)` for each head dimension pair i and token position m. For inference with fixed
sequence lengths, what is the most hardware-efficient approach?

- A) Computing cos and sin in hardware using CORDIC for each token on every forward pass
- B) Pre-computing a table of cos/sin values for all (m, i) pairs up to the maximum sequence
   length and storing it in on-chip SRAM, then looking up values during inference
- C) Approximating the rotation using a first-order Taylor expansion: cos(x) ≈ 1, sin(x) ≈ x
- D) Absorbing the RoPE rotation into the weight matrices during model export, eliminating run-time
   computation entirely

---

## Section 4 — GQA and MQA (Questions 10-12)

**Q10.** In Multi-Query Attention (MQA), all query heads share a single set of key and value
projections. What is the primary motivation for MQA in inference hardware?

- A) MQA improves model accuracy by reducing overfitting in the key-value space
- B) MQA reduces the KV cache size proportional to the number of heads, substantially reducing
   memory bandwidth during decode since fewer K and V vectors need to be loaded
- C) MQA allows query vectors to be computed in parallel because they share the same projection
   matrix
- D) MQA simplifies the softmax computation by ensuring all heads have identical attention distributions

---

**Q11.** Grouped-Query Attention (GQA) generalises both standard MHA and MQA. If a model has
H=32 query heads and G=8 KV groups, how many KV heads does the model have?

- A) 32 KV heads (same as MHA)
- B) 8 KV heads
- C) 4 KV heads
- D) 1 KV head (same as MQA)

---

**Q12.** In hardware implementing GQA, queries are grouped into G groups of H/G heads each. What
is the key microarchitectural consequence for the attention engine?

- A) Each of the G KV groups can be loaded once and broadcast to H/G query heads for dot product
   computation, reducing KV cache bandwidth by a factor of H/G
- B) GQA requires H separate SRAM banks (one per query head) to avoid port conflicts
- C) GQA requires the softmax unit to be replicated G times to handle each group independently
- D) GQA forces the attention engine to process heads sequentially because the shared KV state
   creates data hazards

---

## Section 5 — FlashAttention Hardware Implications (Questions 13-15)

**Q13.** The fundamental bottleneck that FlashAttention addresses is not compute throughput but
memory bandwidth. Which specific memory access pattern does FlashAttention avoid?

- A) Repeatedly reading the weight matrices Q, K, V projections from DRAM
- B) Writing the full N x N attention score matrix (where N is sequence length) to HBM and reading
   it back for the softmax pass, causing O(N^2) HBM memory traffic
- C) Loading the same KV cache entries multiple times due to cache thrashing in the GPU L2
- D) Accessing the output projection matrix O in a non-coalesced access pattern

---

**Q14.** FlashAttention tiles the attention computation to fit Q, K, and V blocks in SRAM. For an
accelerator with on-chip SRAM capacity S_sram, head dimension d, and FP16 precision, what is the
approximate maximum block size (number of tokens per block) B_c for the key/value tiles?

- A) B_c = S_sram / (4 * d * 2) (dividing SRAM equally among Q, K, V, and O blocks)
- B) B_c = S_sram / (2 * d * 2) (K and V tiles together, with factor 2 for FP16)
- C) B_c = sqrt(S_sram / (d * 2)) (square root to balance row and column block sizes)
- D) B_c = S_sram / d (ignoring FP16 byte width)

---

**Q15.** FlashAttention v2 introduced a key change to the work partitioning compared to v1.
What was this change and why does it matter for hardware utilisation?

- A) FlashAttention v2 parallelises across the sequence length (Q blocks) rather than the batch
   and head dimensions, which reduces redundant work and improves GPU occupancy for long sequences
- B) FlashAttention v2 replaced the online softmax with a precomputed lookup table, halving the
   time spent in the softmax step
- C) FlashAttention v2 eliminated the use of SRAM tiling and instead used register files for all
   intermediate computations
- D) FlashAttention v2 added support for sparse attention masks, which is the primary source of its
   speedup over v1

---

## Section 6 — Multi-Head Parallelism and Online Softmax (Questions 16-18)

**Q16.** In a hardware attention engine with P parallel units, what is the most natural dimension
to parallelise across, and why?

- A) Parallelise across the batch dimension because batch elements are independent and have no
   shared state
- B) Parallelise across attention heads because heads are fully independent, each requiring
   separate Q, K, V vectors with no inter-head data dependencies during the forward pass
- C) Parallelise across the sequence length because this reduces the O(N^2) complexity to O(N^2/P)
- D) Parallelise across the hidden dimension because weight matrices are largest in that dimension

---

**Q17.** The online softmax algorithm maintains two running statistics. What are they, and what
update must be applied to the accumulated output when the running maximum changes?

- A) Running sum and running variance; when maximum changes, variance must be renormalised
- B) Running maximum m and running denominator sum s; when m increases to m_new, both s and the
   accumulated output O must be rescaled by `exp(m_old - m_new)` to remain consistent
- C) Running maximum m and running product of exponentials; when m changes, the product must be
   divided by the new maximum
- D) Running minimum and running sum; when the minimum changes, the sum is corrected additively

---

**Q18.** In hardware, the online softmax rescaling step requires multiplying the accumulated partial
output by `exp(m_old - m_new)`. Since m_new >= m_old, this factor is always in the range (0, 1].
Which hardware optimisation exploits this property?

- A) Using unsigned arithmetic for the rescaling factor, saving 1 bit of representation
- B) Representing the rescaling factor as a negative power of two for efficient barrel shifter
   implementation
- C) Storing the rescaling factor in a fixed-point format with a bias, since values are bounded to
   [0, 1]
- D) Skipping the rescaling step when `exp(m_old - m_new)` is within a small epsilon of 1.0,
   controlled by a configurable threshold register

---

## Answer Key

| Q  | Answer |
|----|--------|
| 1  | B      |
| 2  | B      |
| 3  | B      |
| 4  | B      |
| 5  | B      |
| 6  | B      |
| 7  | B      |
| 8  | B      |
| 9  | B      |
| 10 | B      |
| 11 | B      |
| 12 | A      |
| 13 | B      |
| 14 | A      |
| 15 | A      |
| 16 | B      |
| 17 | B      |
| 18 | D      |

---

## Detailed Explanations

### Q1 — Correct: B

IEEE 754 FP32 can represent values up to approximately 3.4 x 10^38. The exponential function grows
extremely rapidly: `exp(90) ≈ 1.2 x 10^39` overflows FP32. In transformer models, attention logits
can easily reach 50-100 before softmax normalisation, causing overflow before the normalisation
denominator is computed.

- A incorrect: Division is fully supported in floating-point arithmetic. The issue is overflow
  before division, not the division itself.
- C incorrect: Underflow would cause the sum to be zero (which is a separate but less critical
  issue). The primary concern is overflow of `exp(x_i)` for large positive x_i.
- D incorrect: Softmax does not require sorting. The maximum-finding step is O(N) linear scan,
  not a sort.

### Q2 — Correct: B

Online softmax (Milakov and Gimelshein, 2018) maintains a running maximum m_i and a running
scaled sum s_i. When token i+1 arrives with logit x_{i+1}:
- Update running max: m_{i+1} = max(m_i, x_{i+1})
- Rescale previous sum: s_{i+1} = s_i * exp(m_i - m_{i+1}) + exp(x_{i+1} - m_{i+1})
- Update output accumulator similarly

This allows single-pass processing without storing all logits.

- A incorrect: A piecewise-linear approximation reduces accuracy but does not solve the two-pass
  problem; you still need to know the maximum before computing exponentials.
- C incorrect: A LUT computes exp() efficiently but does not address the maximum-dependency
  problem — the LUT just replaces one computational step.
- D incorrect: A butterfly reduction is a valid approach for batch parallel systems but requires
  storing all logits simultaneously and is a parallel (not online/streaming) algorithm.

### Q3 — Correct: B

The decomposition `exp(x) = exp(n * ln2 + f) = 2^n * exp(f)` where n = floor(x / ln2) and f is
the remainder (|f| < ln2 / 2) allows: (1) the integer part 2^n is an exact shift operation, and
(2) exp(f) over a small bounded range is well-approximated by a small polynomial or LUT. This is
the basis of libm implementations and hardware exp units.

- A incorrect: A 10-term Taylor series for general exp(x) has poor convergence for large |x| and
  requires 10 multiplications and additions in a combinational path — too slow and inaccurate
  without range reduction.
- C incorrect: Newton-Raphson solves equations iteratively and is used for division/sqrt, not
  typically for exp(). The iterative nature also means variable latency, which is undesirable in
  a streaming pipeline.
- D incorrect: CORDIC in hyperbolic mode can compute exp() but requires many iterations (typically
  15-25) for hardware precision, resulting in high latency and area relative to the LUT+shift
  approach.

### Q4 — Correct: B

KV cache stores both keys and values (factor of 2), across all H heads, with head dimension d_k,
for all S sequence positions, using 2 bytes per FP16 value. Total = 2 x H x d_k x S x 2 bytes.

- A incorrect: This gives the size in INT8 (1 byte per value) and is therefore half the correct
  answer for FP16.
- C incorrect: Both K and V tensors are cached. Omitting V halves the count.
- D incorrect: Q and the output projection are not cached in the KV cache. Q is recomputed at each
  step. The factor of 4 is wrong.

### Q5 — Correct: B

PagedAttention maps logical KV cache blocks to non-contiguous physical memory pages, analogous to
OS virtual memory. This eliminates external fragmentation (wasted memory gaps between allocations)
that occurs in systems requiring contiguous KV cache buffers of variable lengths.

- A incorrect: Run-length encoding exploits repeated values, which are not characteristic of KV
  cache tensors (they are dense floating-point activations). PagedAttention is a memory management
  technique, not a compression technique.
- C incorrect: PagedAttention keeps KV cache on the accelerator's HBM; offloading to CPU DRAM
  would introduce severe bandwidth bottlenecks.
- D incorrect: KV cache quantisation (e.g., INT4 or INT8) is a separate and complementary
  technique. PagedAttention is specifically about memory layout management.

### Q6 — Correct: B

Per layer, per request: 2 (K and V) * 64 heads * 128 head_dim * 4096 tokens * 2 bytes/FP16
= 2 * 64 * 128 * 4096 * 2 = 134,217,728 bytes = 128 MB per layer.

Total for 80 layers: 80 * 128 MB = 10,240 MB ≈ **10 GB**.

- A incorrect: 5 GB would correspond to half the sequence length (2048) or half the head dimension,
  neither of which matches the given parameters.
- C incorrect: 20 GB would correspond to double the correct calculation — e.g., if BF32 (4 bytes)
  was assumed instead of FP16 (2 bytes).
- D incorrect: 40 GB would require 4 bytes per element (FP32) and full 4096 sequence, which does
  not match the given FP16 precision.

### Q7 — Correct: B

RoPE encodes position by multiplying query and key vectors by complex rotation matrices indexed by
position. The dot product Q_m * K_n = f(q, m)^T * f(k, n) = g(q, k, m-n), which depends only on
the relative offset. This relative encoding enables better generalisation to positions not seen
during training, unlike absolute positional embeddings.

- A incorrect: RoPE is a deterministic function of position index, not a learned embedding.
  Extending context with RoPE-based models requires techniques like YaRN or Position Interpolation,
  not simple fine-tuning of the RoPE parameters.
- C incorrect: RoPE does not add a scalar bias. It applies a rotation matrix to the query and key
  vectors, which is a multiplicative (rotation) operation.
- D incorrect: RoPE is a positional encoding scheme, not an alternative to positional encoding.
  It encodes position in the rotation angle applied to Q and K.

### Q8 — Correct: B

RoPE divides the d-dimensional query vector into d/2 pairs. Each pair (q_{2i}, q_{2i+1}) is
rotated by angle m * theta_i, where theta_i = 1 / (10000^(2i/d)). For d=128, there are 64 pairs,
requiring **64 rotations**, each with a different frequency theta_i.

- A incorrect: A single global rotation would use the same frequency for all dimensions, losing
  the multi-scale positional encoding property that makes RoPE effective.
- C incorrect: 128 individual rotations would imply rotating each scalar independently, which is
  not how rotation matrices work — they operate on pairs.
- D incorrect: 256 is double the correct answer and would imply operating on the real and imaginary
  parts separately in a way that is not the standard RoPE formulation.

### Q9 — Correct: B

For inference, the sequence positions are known ahead of time (up to some maximum). Pre-computing
a cos/sin table for all positions and head dimensions costs O(max_seq * d/2) values. This table
fits in on-chip SRAM (e.g., for max_seq=4096, d=128: 4096 * 64 * 2 * 2 bytes = 1 MB) and
provides single-cycle access with no numerical error from approximation.

- A incorrect: CORDIC is an iterative algorithm (typically 15+ cycles) and would be a latency
  bottleneck applied to every token during every decode step. It is appropriate for design-time
  table generation, not real-time inference.
- C incorrect: cos(x) ≈ 1, sin(x) ≈ x is a first-order approximation valid only for very small x.
  For m * theta_i with large m (e.g., m=4096), this approximation is wildly inaccurate.
- D incorrect: Q and K vectors change with input data; their projections cannot absorb a
  position-dependent rotation because the rotation depends on the token's position index, which
  is not known until inference time.

### Q10 — Correct: B

In MQA, there is only 1 KV head instead of H. The KV cache shrinks by a factor of H, and at each
decode step, far fewer K/V bytes need to be loaded from HBM. For H=32 heads, this is a 32x
reduction in KV cache bandwidth — directly improving decode throughput.

- A incorrect: MQA generally slightly reduces model quality because of reduced representational
  capacity in the key-value space. It is a performance trade-off, not an accuracy improvement.
- C incorrect: Query projection matrices are separate per head in both MHA and MQA. MQA shares
  the K and V projections, not Q.
- D incorrect: Different query heads apply their own attention distributions to the same shared K
  and V, so attention distributions are not identical across heads.

### Q11 — Correct: B

In GQA with G=8 groups and H=32 total query heads, each group has H/G = 4 query heads sharing one
set of K and V projections. Total KV heads = G = **8**.

- A incorrect: 32 KV heads would be standard Multi-Head Attention (MHA), not GQA.
- C incorrect: 4 would be the number of query heads per group, not the number of KV groups/heads.
- D incorrect: 1 KV head describes Multi-Query Attention (MQA), the extreme case of GQA with G=1.

### Q12 — Correct: A

With G KV groups, each K/V block is shared by H/G query heads. In hardware, this means loading
each KV block from SRAM once and multiplying it against H/G query vectors before evicting the
KV block. The KV cache bandwidth is reduced by the factor H/G compared to standard MHA.

- B incorrect: The number of SRAM banks needed is determined by the access parallelism required,
  not by the number of query heads. GQA reduces bandwidth requirements and does not increase the
  number of SRAM banks needed.
- C incorrect: The softmax unit processes one attention head at a time sequentially or per-group;
  it does not need to be replicated G times for GQA.
- D incorrect: Attention heads are independent of each other — there are no data hazards between
  heads. GQA groups process queries that share KV, but within a group, the queries are
  independent and can be processed in parallel.

### Q13 — Correct: B

Standard attention writes the N x N score matrix to HBM after computing QK^T, reads it back for
softmax, then reads it again to compute the weighted sum with V. For N=8192, the score matrix is
8192^2 * 2 bytes = 128 GB — far exceeding HBM capacity for a single sequence. FlashAttention
avoids materialising this matrix in HBM entirely.

- A incorrect: Q, K, V projection weights are part of the model and are loaded once per decode
  step regardless; FlashAttention does not address this.
- C incorrect: L2 cache thrashing is a secondary concern. The primary issue is the explicit
  software-visible write to HBM of the score matrix, not cache behaviour.
- D incorrect: The output projection O comes after the attention mechanism. Non-coalesced access
  to O is a separate concern and not what FlashAttention primarily targets.

### Q14 — Correct: A

In FlashAttention, SRAM must hold Q, K, V, and output O blocks simultaneously (plus scratch space).
With four blocks of size B_c * d * 2 bytes (FP16), total SRAM = 4 * B_c * d * 2. Setting this
equal to S_sram gives B_c = S_sram / (4 * d * 2) = S_sram / (8d). This is the standard
derivation in the FlashAttention paper.

- B incorrect: Dividing by 2 * d * 2 only accounts for K and V, omitting the Q and O blocks that
  must also reside in SRAM simultaneously.
- C incorrect: A square root relationship would apply to a 2D tiling where both row and column
  block sizes are equal; FlashAttention tiles in one dimension (tokens) for K/V.
- D incorrect: Ignoring the FP16 byte width (factor of 2) and the four blocks (factor of 4 * 2 = 8)
  overstates the block size by 8x, which would exceed SRAM capacity.

### Q15 — Correct: A

FlashAttention v1 parallelised across batch and head dimensions only. For long sequences with
small batch sizes (common in LLM serving), this left GPU SMs underutilised. FlashAttention v2
additionally parallelises across the query sequence length dimension, giving more independent
work units (one per Q tile) and improving SM occupancy.

- B incorrect: FlashAttention v2 still uses online softmax with running statistics. It did not
  replace online softmax with a precomputed LUT.
- C incorrect: SRAM tiling is the fundamental technique of FlashAttention; v2 did not eliminate it.
  Register files are used for inner loop accumulators, but tiling to SRAM remains essential.
- D incorrect: Sparse attention mask support was not the primary innovation of v2 over v1. The
  speedup came from improved parallelism and reduced non-matmul FLOPs.

### Q16 — Correct: B

Each attention head independently computes Q, K, V projections and the attention pattern from its
own subspace of the input. There are no data dependencies between heads within a single layer.
This makes heads the most natural parallelism dimension — each unit can work on a separate head
with no communication needed until the final concatenation.

- A incorrect: Batch parallelism is valid and commonly used (each batch element is independent),
  but it does not exploit the structure of multi-head attention specifically. More importantly,
  for single-request inference (batch=1), batch parallelism provides no benefit, while head
  parallelism is always applicable.
- C incorrect: Parallelising across sequence length reduces the O(N^2) compute to O(N^2/P) but
  requires care with the softmax normalisation (each partial result needs the global normaliser),
  introducing dependencies. This is the harder dimension to parallelise correctly.
- D incorrect: The hidden dimension is the reduction dimension for dot products, not a parallelism
  dimension for independent computation. Splitting the hidden dimension creates partial sums that
  must be reduced.

### Q17 — Correct: B

Online softmax maintains:
- m_i: running max of logits seen so far
- s_i: sum of exp(x_j - m_i) for j <= i
- O_i: accumulated weighted output = sum over j <= i of exp(x_j - m_i) * v_j / s_i

When a new logit x_{i+1} arrives and m_{i+1} = max(m_i, x_{i+1}) > m_i:
- s must be rescaled: s_{i+1} = s_i * exp(m_i - m_{i+1}) + exp(x_{i+1} - m_{i+1})
- O must be rescaled: O_{i+1} = O_i * exp(m_i - m_{i+1}) * s_i / s_{i+1} + ...

This ensures all accumulated values remain correctly shifted relative to the current maximum.

- A incorrect: Variance tracking is not part of softmax computation. Softmax requires only the
  maximum (for stability) and the sum of exponentials (for normalisation).
- C incorrect: A running product of exponentials is numerically equivalent to the sum (since
  log-sum-exp = log of product) but is not the standard formulation and does not correctly
  describe the rescaling operation.
- D incorrect: Tracking a running minimum is used in some min-max normalisation schemes but not
  in softmax. Additive correction for the sum does not correctly account for the exponential
  relationship.

### Q18 — Correct: D

When m_new is only slightly larger than m_old, the rescaling factor exp(m_old - m_new) is very
close to 1.0 (e.g., within 2^-10). In this case, multiplying the accumulated output by a value
nearly equal to 1 has negligible effect and the operation can be safely skipped. This reduces
the number of multiply operations in the rescaling path, saving energy and cycles. A configurable
epsilon threshold allows the hardware designer to trade off accuracy for performance.

- A incorrect: The rescaling factor can be very small (not just bounded to [0,1] in sign) and is
  a floating-point value that requires full floating-point arithmetic. Using unsigned format would
  not simplify the hardware significantly — FP already handles non-negative values in the mantissa.
- B incorrect: exp(m_old - m_new) is generally not a power of two. The exponent difference could
  be any real number, so a barrel shift representation would introduce quantisation error.
- C incorrect: While the rescaling factor is in (0, 1], a fixed-point representation with a bias
  would be lossy and add complexity. The rescaling factor is computed in floating-point and used
  in a floating-point multiply — there is no clear hardware advantage to reformatting it as
  fixed-point.
