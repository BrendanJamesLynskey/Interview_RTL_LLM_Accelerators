# RMSNorm and LayerNorm Hardware

## Overview

Normalisation layers appear at every transformer block — in LLaMA-3 70B there are 160 of them.
Each normalisation operation reads an entire hidden vector, computes statistics across it,
and rescales every element. This makes normalisation a memory-bandwidth-bound operation with
a reduction dependency: the scale factor cannot be computed until all elements have been
accumulated. This document covers the hardware implementation of both LayerNorm and RMSNorm,
with special focus on the inverse square root computation, pipeline structure, and fusion
with quantisation.

---

## Background Mathematics

**LayerNorm** (Ba et al., 2016):

```
y_i = gamma_i * (x_i - mu) / sqrt(sigma^2 + eps)

where:
  mu    = (1/d) * sum_{i=0}^{d-1} x_i          (mean)
  sigma^2 = (1/d) * sum_{i=0}^{d-1} (x_i - mu)^2  (variance)
  gamma_i, beta_i = learnable per-element scale and bias
```

**RMSNorm** (Zhang & Sennrich, 2019):

```
y_i = gamma_i * x_i / RMS(x)

where:
  RMS(x) = sqrt( (1/d) * sum_{i=0}^{d-1} x_i^2 )
```

RMSNorm omits the mean subtraction and the bias term `beta`. This is the version used in
LLaMA, Mistral, Gemma, and most modern open-weight LLMs.

---

## Tier 1 — Fundamentals

### Q1. What is the difference between LayerNorm and RMSNorm? Why do modern LLMs prefer RMSNorm?

**Answer.**

**Computational difference.** LayerNorm computes both mean `mu` and variance `sigma^2`, then
subtracts the mean and divides by the standard deviation. RMSNorm computes only the root mean
square (RMS), skipping mean computation and mean subtraction entirely.

**Flop count comparison** for a hidden vector of dimension d:

| Operation | LayerNorm | RMSNorm |
|---|---|---|
| Sum for mean | d additions | 0 |
| Mean subtraction | d subtractions | 0 |
| Sum of squared deviations | d multiply-adds | d multiply-adds (x_i^2) |
| Inverse sqrt | 1 | 1 |
| Scale by gamma | d multiplies | d multiplies |
| Shift by beta | d additions | 0 |
| **Total** | ~5d + 2 | ~2d + 1 |

RMSNorm is approximately 2.5x cheaper in operations.

**Why modern LLMs prefer RMSNorm:**

1. **Empirical parity.** Zhang & Sennrich (2019) showed that the re-centring (mean subtraction)
   in LayerNorm contributes little to its regularisation benefit in language models. The
   re-scaling (variance normalisation) is the essential operation. RMSNorm provides the
   re-scaling without re-centering, with negligible quality loss.

2. **Hardware simplicity.** Removing the mean subtraction eliminates a two-pass dependency.
   LayerNorm requires either two passes over the data (pass 1: compute mean; pass 2: compute
   variance) or an online Welford algorithm. RMSNorm needs only one pass: accumulate x_i^2.

3. **No beta parameter.** Halves the number of normalisation parameters loaded from HBM, which
   matters in weight-memory-bound inference.

4. **Numerical stability.** The mean subtraction in LayerNorm can cause catastrophic
   cancellation when `x_i ≈ mu` for all i (near-constant input). RMSNorm is immune to this.

---

### Q2. Describe the four stages of an RMSNorm computation pipeline.

**Answer.**

**Stage 1: Accumulate sum of squares.**

```
acc = 0
for i in range(d):
    acc += x[i] * x[i]
```

This is a sequential reduction. Hardware: a multiply-accumulate (MAC) unit running for `d`
cycles, or a tree of parallel adders for higher throughput.

Output: `sum_sq = sum(x_i^2)`, a scalar.

**Stage 2: Compute mean and add epsilon.**

```
mean_sq = sum_sq / d   (or equivalently, sum_sq * (1/d))
val = mean_sq + eps
```

Division by `d` is by a known constant power-of-2 (or near power-of-2), implemented as a
right shift (for exact powers of 2) or a multiply by the precomputed reciprocal `1/d`.

`eps` is a small constant (typically 1e-5 or 1e-6) added for numerical stability to prevent
division by zero.

Output: `val = RMS^2 + eps`, a scalar.

**Stage 3: Inverse square root.**

```
inv_rms = 1 / sqrt(val)
```

This is the expensive step. Methods: Newton-Raphson iteration, Goldschmidt iteration, or LUT.
See Q7 for detailed implementation. Output: `inv_rms`, a scalar.

**Stage 4: Scale by gamma and inv_rms.**

```
for i in range(d):
    y[i] = gamma[i] * x[i] * inv_rms
```

This is an element-wise multiply of each input `x[i]` by a per-element learnable weight
`gamma[i]` and the computed `inv_rms`. The vector `x` must be buffered while stages 1-3
complete.

---

### Q3. What is the role of `eps` in normalisation, and how do you choose its value for fixed-point hardware?

**Answer.**

**Purpose of eps.** `eps` is added to the denominator to prevent division by zero when the
input vector is all-zeros (or near-zero). Without it, a zero-valued input produces `RMS = 0`
and `1/0 = infinity`, causing a NaN or saturation in the output.

**Value in floating-point.** Common choices are `1e-5` or `1e-6`. These are small enough not
to affect normalisation when `RMS >> eps`, yet large enough to keep the denominator away from
zero.

**For fixed-point hardware**, the choice interacts with the number format:

1. **Format constraint.** `eps` must be representable in the format used for `mean_sq`. If
   `mean_sq` is in Q8.24, then `eps = 1e-6` requires at least 20 fractional bits to represent
   `(1e-6 * 2^24 ≈ 16.7)`. This is representable in Q8.24.

2. **Effective eps floor.** In fixed-point, the accumulator has a natural non-zero floor from
   the LSB of `sum_sq`. For a 16-bit input over `d=1024` elements, the minimum non-zero
   `sum_sq` is `1024 * 1^2 / 1024 = 1` in the squared-input units. An `eps` smaller than this
   minimum LSB is redundant.

3. **Practical fixed-point choice.** Set `eps` to the value of 1 LSB in the `mean_sq` format.
   This guarantees the denominator is always at least 1 LSB, avoiding division by zero, while
   being small enough to not affect results when the input is non-trivial.

---

### Q4. Compare pre-norm and post-norm transformer architectures from a hardware perspective.

**Answer.**

**Post-norm** (original Transformer, Vaswani et al. 2017):

```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

Normalisation is applied after the residual addition. The input to the normalisation layer
can have arbitrary magnitude (accumulated residuals over many layers). Range grows with depth.

**Pre-norm** (GPT-2, LLaMA, most modern LLMs):

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Normalisation is applied before the attention/FFN block. The input to each attention/FFN
operation is always normalised to a known range.

**Hardware implications:**

1. **Input range for linear layers.** In pre-norm, the input to GEMM is always normalised
   (mean ≈ 0, std ≈ 1 before `gamma` scaling). This tightens the input distribution for the
   GEMM, enabling smaller integer range and better quantisation. In post-norm, the pre-GEMM
   input is the raw residual, which can grow over layers — harder to quantise.

2. **Output range for normalisation.** In pre-norm, the normalisation input is the residual
   stream, which can accumulate large values over many layers. In post-norm, the normalisation
   input is the sum of residual + sublayer output, but the normalisation output is always
   bounded by `gamma`.

3. **Training stability vs. inference hardware.** Pre-norm is more training-stable at large
   scale (hence universal adoption in modern LLMs). For hardware, pre-norm simplifies
   activation quantisation because the normalised activations have predictable statistics.

4. **Dependency chain.** In pre-norm, the normalisation layer is on the critical path before
   each GEMM. This means normalisation latency directly adds to the end-to-end latency per
   layer. In post-norm, normalisation and the next layer's GEMM are on the critical path but
   can partially overlap in time (the GEMM can begin with the first available normalised
   element in a streaming design).

---

### Q5. Describe LayerNorm's two-pass algorithm and explain why it creates a hardware challenge.

**Answer.**

**Two-pass algorithm:**

```
Pass 1: mean = (1/d) * sum(x[i])         for i in 0..d-1
Pass 2: var  = (1/d) * sum((x[i]-mean)^2) for i in 0..d-1
        y[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i]
```

Pass 2 cannot begin until Pass 1 is complete, because `mean` is needed to compute each
`(x[i] - mean)^2`.

**Hardware challenge:**

1. **Latency.** The element-wise output cannot be produced until both passes complete. For
   `d = 4096` elements at one element per cycle, this is 8192 cycles of latency before the
   first output. For a 1 GHz clock and 4096-wide hidden dimension, this is 8.2 microseconds —
   a significant addition to layer latency.

2. **Input buffering.** All `d` input elements `x[i]` must be stored during Pass 1 and
   re-read during Pass 2. This requires `d * W` bits of on-chip buffer, where `W` is the
   element bitwidth. For d=4096, W=16: 64 Kbits = 8 KB. This is a non-trivial SRAM cost,
   especially if normalisation is instantiated many times.

3. **Memory bandwidth.** Data is read twice from the buffer. In a bandwidth-limited system,
   this doubles the memory access for the normalisation layer. Alternatives (Welford online
   algorithm) compute mean and variance in one pass but require more arithmetic per element.

**RMSNorm solves this.** With no mean subtraction, only one pass is needed:
```
sum_sq = sum(x[i]^2)
inv_rms = 1/sqrt(sum_sq/d + eps)
y[i] = x[i] * inv_rms * gamma[i]
```
The input still needs buffering until `inv_rms` is computed, but there is only one pass over
the data — halving buffer read bandwidth.

---

## Tier 2 — Intermediate

### Q6. Describe the Welford online algorithm for computing mean and variance in a single pass. When would you use it in hardware?

**Answer.**

**Welford's algorithm** (1962) maintains running estimates of mean and variance:

```
Initialize: m = 0, M2 = 0, count = 0

For each element x:
    count += 1
    delta  = x - m
    m     += delta / count
    delta2 = x - m
    M2    += delta * delta2

Result: mean = m, variance = M2 / count
```

At termination: `mean = m`, `variance = M2 / (count - 1)` (sample) or `M2 / count` (population).

**Why it avoids catastrophic cancellation.** The naive two-pass formula `E[x^2] - E[x]^2` can
suffer severe cancellation when `E[x]` is large and `var` is small (e.g., inputs near 1000
with variance 0.001). Welford maintains `M2 = sum of squared deviations from running mean`,
which is always non-negative and avoids large intermediate values.

**Hardware considerations:**

Pros:
- One pass over data — no need to store the vector, read it twice, or wait.
- Streaming architecture: process one element per cycle, emit normalised outputs immediately
  after the last element.

Cons:
- The update `delta / count` requires a division by a changing count. In hardware, `1/count`
  must be precomputed for each value of count (small table: max count is d, and d is fixed
  at design time) or computed with a reciprocal unit.
- Two multiplications per element (vs. one for the naive sum-of-squares approach) means
  higher arithmetic utilisation per cycle.
- The sequential dependency `m[k+1] = f(m[k])` creates a feedback loop with a one-cycle
  latency, limiting throughput to one element per clock unless the feedback loop is pipelined
  (requires careful retiming).

**When to use in hardware.** Welford is preferred when:
- Memory for input buffering is the scarce resource (streaming architecture).
- The input values are large (overflow risk with x^2 sum).
- Latency to first output is critical (post-normalisation GEMM can overlap with late elements).

RMSNorm hardware almost always uses the simpler `sum(x^2)` approach since mean computation
is eliminated. Welford's algorithm is most relevant for full LayerNorm implementations.

---

### Q7. Explain how Newton-Raphson iteration is used to compute the inverse square root in hardware. Derive the iteration formula.

**Answer.**

**Goal:** Compute `y = 1/sqrt(a)` given input `a > 0`.

**Reformulate.** Define `f(y) = 1/y^2 - a`. Finding the zero of `f(y)` gives `y = 1/sqrt(a)`.

**Newton-Raphson update rule:**

```
y_{n+1} = y_n - f(y_n) / f'(y_n)

f(y)  = y^{-2} - a
f'(y) = -2 * y^{-3}

y_{n+1} = y_n - (y_n^{-2} - a) / (-2 * y_n^{-3})
         = y_n + (y_n^{-2} - a) * y_n^3 / 2
         = y_n + (y_n - a * y_n^3) / 2
         = y_n * (3/2) - (a/2) * y_n^3
         = (y_n / 2) * (3 - a * y_n^2)
```

**Final iteration formula:**

```
y_{n+1} = (y_n / 2) * (3 - a * y_n^2)
```

This is the famous "Quake fast inverse square root" iteration.

**Hardware implementation per iteration:**

```
Step 1: sq   = y_n * y_n          (multiply)
Step 2: prod = a   * sq           (multiply)
Step 3: diff = 3   - prod         (subtract from constant)
Step 4: y_next = (y_n * diff) / 2 (multiply + right shift by 1)
```

Four operations per iteration, all independent of each other in terms of critical path
(the data dependency chain is 3 sequential multiplies + 1 subtract + 1 multiply = 5 ops).

**Convergence analysis.** Newton-Raphson for `1/sqrt(a)` has quadratic convergence: each
iteration approximately doubles the number of correct bits. Starting from a 4-bit accurate
initial estimate:
- After 1 iteration: ~8 correct bits
- After 2 iterations: ~16 correct bits
- After 3 iterations: ~32 correct bits (FP32 accuracy)

For FP16 (10-bit mantissa), two iterations from a 3-bit initial estimate suffice.
For INT8 inference (8 bits of precision needed), one iteration from a 4-bit seed is enough.

**Initial seed.** The initial estimate `y_0` is obtained from a small LUT indexed by the
leading bits of `a`. For a 4-bit seed LUT, the table has 16 entries. The LUT is computed
offline using `y_0 = 1/sqrt(a_midpoint)` for the midpoint of each 4-bit interval.

**Hardware cost summary:**
- 2 Newton-Raphson iterations: 6 multiplications total.
- 1 initial LUT: 16 x 8-bit = 128-bit ROM (negligible).
- Total latency (fully pipelined): 6 * 1 cycle multiply = 6+ cycles.

---

### Q8. How do you pipeline an RMSNorm unit for a vector of d elements where d is a design-time parameter?

**Answer.**

**Parameterisable pipeline design:**

The pipeline has three phases with a serialisation dependency at the boundary between Phase 1
and Phase 2:

```
Phase 1 (d cycles): Accumulate sum of squares
    Reads x[0], x[1], ..., x[d-1] sequentially.
    Feeds MAC: acc += x[i]^2

    Latency: d cycles + pipeline flush (~2 cycles for MAC pipeline)

Phase 2 (fixed latency, ~10 cycles): Compute inv_rms
    mean_sq = acc / d     (1 multiply by 1/d, precomputed constant)
    val     = mean_sq + eps
    y0      = seed_lut[val_msbs]     (1 cycle LUT read)
    y1      = (y0/2)*(3 - val*y0^2)  (3 cycles: multiply + subtract + multiply)
    y2      = (y1/2)*(3 - val*y1^2)  (3 cycles: second N-R iteration)
    inv_rms = y2

Phase 3 (d cycles): Scale outputs
    Reads x[i] and gamma[i] from buffers.
    y[i] = x[i] * gamma[i] * inv_rms
```

**Buffer requirements:**
- `x` buffer: d elements * W_in bits (e.g., 4096 * 16 = 64 Kbits).
- `gamma` buffer: d elements * W_gamma bits (typically 16-bit, same size).
- Both are read during Phase 3 while Phase 1/2 data has been consumed.

**Double-buffering for throughput.** To process consecutive vectors without stalls, use two
ping-pong sets of `x` buffers. While Phase 3 is reading buffer A for vector N, Phase 1 is
writing buffer B for vector N+1. This achieves one vector per `d + 12` cycles (approximately),
dominated by Phase 1.

**Parameterisation in SystemVerilog:**

```systemverilog
module rmsnorm #(
    parameter int D        = 4096,    // vector length
    parameter int W_IN     = 16,      // input bitwidth
    parameter int W_ACC    = 48,      // accumulator bitwidth (W_IN*2 + log2(D))
    parameter int W_OUT    = 16       // output bitwidth
)(
    input  logic              clk,
    input  logic              rst_n,
    input  logic [W_IN-1:0]   x_in,
    input  logic              x_valid,
    input  logic [W_IN-1:0]   gamma_in,
    output logic [W_OUT-1:0]  y_out,
    output logic              y_valid
);
```

Key parameter: `W_ACC = W_IN*2 + $clog2(D)` ensures no overflow during accumulation.
For W_IN=16 and D=4096: W_ACC = 32 + 12 = 44 bits minimum, use 48 for margin.

---

### Q9. Describe how to fuse RMSNorm with the subsequent quantisation step in INT8 inference. What are the benefits?

**Answer.**

**Unfused flow:**

```
x_fp16  -> RMSNorm -> y_fp16 -> Quantise(scale) -> y_int8 -> INT8 GEMM
```

Two separate memory round-trips: RMSNorm writes `y_fp16` to SRAM, then the quantisation
kernel reads it.

**Fused flow:**

```
x_fp16  -> [RMSNorm + Quantise fused] -> y_int8 -> INT8 GEMM
```

The `inv_rms` scalar computed in RMSNorm is combined with the quantisation scale factor `q_s`
into a single fused scaling:

```
y_int8[i] = round( x[i] * gamma[i] * inv_rms * (1 / q_s) )
           = round( x[i] * fused_scale[i] )

where fused_scale[i] = gamma[i] * inv_rms / q_s
```

`fused_scale[i]` is computed once per vector in floating-point, then applied with a single
integer multiply+round per element.

**Benefits:**

1. **Memory bandwidth.** Eliminates the intermediate FP16 tensor write and read — saves 2 *
   d * 2 bytes of memory traffic per RMSNorm layer. For d=4096 and a batch of 32 tokens:
   2 * 4096 * 2 * 32 = 512 KB per layer, per forward pass.

2. **Arithmetic efficiency.** Two multiplies (`x * gamma * inv_rms` and then `* 1/q_s`)
   collapse to one multiply (`x * fused_scale`). Halves the multiply count in Phase 3.

3. **Reduced on-chip buffering.** The FP16 intermediate does not need to be staged in SRAM
   — the pipeline can flow directly from the RMSNorm output multiply to the INT8 packing logic.

4. **Quantisation scale selection.** The quantisation scale `q_s` must be determined before
   Phase 3 begins. For static quantisation (scale is fixed at calibration time), `fused_scale`
   is computed in floating-point in the host CPU once per layer at model load time, then stored
   in the per-layer coefficient ROM. For dynamic quantisation (scale determined per-token),
   `q_s` must be computed from the RMSNorm output statistics, adding latency before Phase 3.

**Hardware structure for fused unit:**

```
Phase 1: acc = sum(x[i]^2)
Phase 2: inv_rms = 1/sqrt(acc/d + eps)
         Load q_s from configuration register (static quant)
         Compute fused_scale[i] = gamma[i] * inv_rms / q_s  (per element)
Phase 3: y_int8[i] = saturate_round(x[i] * fused_scale[i])
```

---

### Q10. What is the Goldschmidt algorithm for division/square root? When is it preferred over Newton-Raphson?

**Answer.**

**Goldschmidt algorithm** (1964) computes `a/b` or `sqrt(a)` through repeated multiplication
by factors that converge the denominator to 1.

**For inverse square root** `y = 1/sqrt(a)`:

Initialise `x_0 = a`, `y_0 = 1/sqrt(a_approx)` (from a seed LUT, same as Newton-Raphson).

At each iteration:

```
delta_n = (3 - x_n) / 2       (scale factor approaching 1 as x_n -> 1)
x_{n+1} = x_n * delta_n^2     (converges to 1)
y_{n+1} = y_n * delta_n        (converges to 1/sqrt(a))
```

More precisely, define `d_n = (3 - a * y_n^2) / 2`:

```
y_{n+1} = y_n * d_n
a_{n+1} = a * y_{n+1}^2       (used to compute next d_{n+2})
```

**Key difference from Newton-Raphson:**

| Property | Newton-Raphson | Goldschmidt |
|---|---|---|
| Operations per iteration | 4 multiplies + 1 sub | 3 multiplies + 1 sub |
| Parallelism per iteration | Sequential (3-deep dep chain) | Two multiplies are independent |
| Pipelining | Natural: y update is sequential | `y * d` and `a * d^2` can be parallel |
| Convergence | Quadratic (doubles bits each iter) | Quadratic |
| Fused multiply-add (FMA) friendly | Yes | Yes |

**Why Goldschmidt can be faster in hardware:**

In Goldschmidt, the operations `y_{n+1} = y_n * d_n` and `a_{n+1} = a_n * d_n^2` share the
same multiplier operand `d_n` but operate on different accumulators. On a processor with two
independent multiplier units (e.g., a systolic array with two PEs dedicated to the reciprocal
sqrt), both multiplies can execute simultaneously, halving wall-clock time per iteration.

**When to prefer each:**

- **Newton-Raphson** is preferred for single-multiplier datapaths and ASIC implementations
  where the iteration is serialised. The formula `y_{n+1} = (y_n/2)(3 - a*y_n^2)` maps
  cleanly to a pipeline with a small number of fixed-latency multipliers.

- **Goldschmidt** is preferred in designs with two parallel multiply units, or in
  implementations targeting FMA (fused multiply-add) instructions where `a*d^2` and `y*d`
  can both use FMA units simultaneously.

In practice, all modern high-performance hardware (NVIDIA tensor cores, Google TPU) uses
Newton-Raphson for its simpler dependency chain in a single-pipeline implementation.

---

## Tier 3 — Advanced

### Q11. Design a streaming RMSNorm unit that overlaps Phase 1 (accumulation) for vector N+1 with Phase 3 (output) of vector N. Describe the buffer and control logic required.

**Answer.**

**Goal.** Hide the Phase 1 accumulation latency (d cycles) by double-buffering so Phase 1
of the next vector runs concurrently with Phase 3 of the current vector.

**Buffer structure (double-buffered):**

```
Buffer A: x_buf_A[0..d-1]  (W_IN bits each)
Buffer B: x_buf_B[0..d-1]  (W_IN bits each)

State machine maintains:
  write_ptr: which buffer Phase 1 is filling
  read_ptr:  which buffer Phase 3 is reading
```

Initially: `write_ptr = A`, `read_ptr = invalid`.

**Timeline:**

```
Cycle range     | Phase 1 (accumulate)      | Phase 3 (output)
0 .. d-1        | Fill buffer A, acc_A       | Idle
d .. d+12       | Idle (Phase 2 on acc_A)    | Idle
d+12 .. 2d+12   | Fill buffer B, acc_B       | Output from buffer A
2d+12..2d+24    | Idle (Phase 2 on acc_B)    | Idle
2d+24..3d+24    | Fill buffer C (=A again)   | Output from buffer B
...
```

**Critical observation.** Phase 2 (12 cycles) is serial — during this time neither Phase 1
nor Phase 3 can make progress on the same data. If `d >> 12`, the Phase 2 stall is amortised
over the long Phase 1/3 windows.

**Overlap condition.** Full overlap (Phase 3 always busy) requires Phase 1 ≥ Phase 2 duration,
which is `d >= 12`. For d=4096 this is easily satisfied.

**Control logic FSM states:**

```
FILL_A:    Phase 1 writing to buffer A. Phase 3 idle or reading buffer B.
           Transition: when Phase 1 completes -> COMPUTE_INV_RMS_A
COMPUTE_A: Phase 2 computing inv_rms from acc_A. Phase 1 may simultaneously start on buffer B.
           Transition: when inv_rms ready -> DRAIN_A (Phase 3 active)
DRAIN_A:   Phase 3 reading buffer A, multiplying by inv_rms_A * gamma.
           Phase 1 may fill buffer B concurrently.
```

**Hazard.** Phase 3 reads `x_buf_A` while Phase 1 is writing to `x_buf_B`. No conflict as
long as `write_ptr != read_ptr`. The FSM must assert an error if a Phase 1 fill completion
occurs before Phase 3 has finished draining — this means the consumer (downstream GEMM) is
stalling. A `stall_out` signal propagates backpressure upstream.

**Area cost.** The double buffer costs `2 * d * W_IN` bits. For d=4096, W_IN=16: 128 Kbits
= 16 KB. This is typically two SRAM macros of 8 KB each, or one dual-port 16 KB SRAM with
port A for write and port B for read.

---

### Q12. Quantisation and normalisation interact critically in INT8 pipelines. Explain the scale accumulation problem and how per-token dynamic quantisation with RMSNorm fusion solves it.

**Answer.**

**Scale accumulation problem.**

In a multi-layer INT8 transformer, each layer performs:
```
y_int8 = quantise(GEMM(dequantise(x_int8, s_x), W_int8, s_w))
```

The dequantisation scale `s_x` and weight scale `s_w` must be tracked and applied correctly.
After K layers, the effective scale is the product of K individual scales:
```
s_effective = prod_{k=1}^{K} s_x_k * s_w_k
```

If these scales vary significantly across tokens (dynamic inputs), using a fixed
calibration-time scale causes clipping (when actual values exceed the calibrated range) or
wasted dynamic range (when actual values are much smaller).

**Per-token dynamic quantisation.** For each token independently:
1. Compute the actual max absolute value: `max_val = max(|y_i|)`.
2. Set `q_s = max_val / 127` (for INT8 symmetric quantisation).
3. Quantise: `y_int8[i] = round(y_i / q_s)`.

This ensures the INT8 range is fully utilised per token, eliminating scale drift.

**The problem without fusion.** To compute `q_s`, you need the full FP16 output of RMSNorm
before quantising. This requires:
1. RMSNorm pass (Phase 1: accumulate, Phase 2: inv_rms, Phase 3: FP16 output).
2. Separate max-abs scan over the FP16 output.
3. Quantisation pass.

Three separate passes over the data — three times the memory bandwidth.

**Solution: Fused RMSNorm + dynamic quantisation.**

The key insight is that the maximum absolute value of the RMSNorm output can be tracked
during Phase 3 with zero additional latency:

```
Phase 3 (modified):
    max_val = 0
    for i in 0..d-1:
        y_fp16[i] = x[i] * gamma[i] * inv_rms     // compute FP16 intermediate
        max_val    = max(max_val, |y_fp16[i]|)     // track max in parallel
        // DO NOT output yet -- buffer y_fp16[i]

    q_s = max_val / 127                            // compute quant scale

Phase 4:
    for i in 0..d-1:
        y_int8[i] = round(y_fp16[i] / q_s)        // quantise
```

This still requires two passes (Phase 3 + 4) and a buffer for `y_fp16`. But `y_fp16` is
only `d * 16` bits = 8 KB for d=4096, which stays on-chip. The HBM traffic is:
- Input: d * 16 bits (read once).
- Output: d * 8 bits (INT8).

One read, one write — no intermediate HBM traffic. Total memory traffic reduced by 2x vs.
unfused, and by 3x vs. the naive three-pass approach.

**Advanced: online max tracking in Phase 3 removes Phase 4.** If the quantisation can be
deferred and the FP16 buffer is available, Phase 3 and Phase 4 can be fused further using a
feedback technique: compute a provisional INT8 using the previous token's scale, then
re-scale at the end. This is sometimes called "smooth quantisation" and is used in
high-performance inference engines like TensorRT-LLM.

---

### Q13. Derive the required accumulator bitwidth for RMSNorm given input precision W_IN, vector length D, and explain what happens if the accumulator overflows.

**Answer.**

**Derivation.**

The accumulator holds `sum_{i=0}^{D-1} x_i^2`.

**Maximum value of a single squared term:**

For a W_IN-bit signed integer, the maximum magnitude is `2^(W_IN - 1) - 1 ≈ 2^(W_IN - 1)`.
The maximum squared value is:

```
max(x_i^2) = (2^(W_IN-1) - 1)^2 ≈ 2^(2*W_IN - 2)
```

**Maximum value of the accumulated sum:**

```
max(sum(x_i^2)) = D * max(x_i^2) ≈ D * 2^(2*W_IN - 2)
```

**Required accumulator bitwidth:**

```
W_ACC >= ceil(log2(D * 2^(2*W_IN - 2)))
       = ceil(log2(D)) + 2*W_IN - 2
       = ceil(log2(D)) + 2*(W_IN - 1)
```

**Concrete example.** W_IN=16, D=4096:

```
W_ACC >= log2(4096) + 2*(16-1)
       = 12 + 30
       = 42 bits
```

Using 48 bits provides 6 bits of headroom, ensuring no overflow.

**What happens on overflow?**

1. **Wraparound (unsigned accumulator, no saturation).** If the accumulator is 32-bit unsigned
   and the true sum exceeds 2^32, the value wraps to a small number. The `inv_rms` computed
   from this wrapped value is orders of magnitude larger than the true value. The output
   `x * inv_rms` saturates to the maximum INT8/INT16 value for every element — the normalised
   vector is all-maximum, which is a hard numerical error. This is silently wrong: the circuit
   produces outputs that look plausible (not NaN, not zero) but are completely incorrect.

2. **Saturation (saturating accumulator).** If the accumulator saturates at 2^(W_ACC - 1) - 1,
   `inv_rms` is computed from an underestimated `sum_sq`, producing an `inv_rms` that is too
   large. Again, the output saturates but at a less extreme wrong value. Still incorrect.

3. **Detection.** Add an overflow flag: if the carry-out of the MSB of the accumulator is
   asserted during accumulation, set an error register visible to the host CPU. The host can
   then recalibrate the input quantisation scale to bring inputs within range.

**Design rule.** Never truncate the accumulator to fit a power-of-2 SRAM width at the cost
of correctness. If the minimum safe W_ACC is 42 bits, use a 48-bit accumulator. The cost is
6 extra flip-flops in the MAC register — negligible compared to the surrounding SRAM and
multipliers.

---

### Q14. Describe numerical issues specific to RMSNorm in INT4/INT8 quantised inference, and propose hardware mitigation strategies.

**Answer.**

**Issue 1: Squared-input dynamic range mismatch.**

In INT8, inputs span [-128, 127]. Squaring gives [0, 16129]. Accumulating 4096 squared INT8
values gives a maximum sum of 4096 * 16129 = 66,060,000, which requires 27 bits. If the
squared terms are computed in INT16 (max 65535), squaring INT8 to INT16 is safe. Accumulating
4096 INT16 values into INT32 is safe (max 4096 * 65535 ≈ 268M < 2^31). This is a manageable
range if all intermediate formats are sized correctly.

For INT4 (range [-8, 7]): squared max is 64, sum of 4096 squares max is 262144 = 2^18.
Manageable in INT32.

**Issue 2: Small RMS leading to large inv_rms.**

If the input vector is nearly zero (e.g., after ReLU kills most activations in a layer above),
`sum_sq` is small and `inv_rms` is large. The output `x * inv_rms` is then very large (large
gain applied to small input), which can:
- Overflow the output format (INT8 range exceeded).
- Amplify quantisation noise: the 1 LSB noise of INT8 gets multiplied by inv_rms.

**Mitigation:** Saturate `inv_rms` to a maximum value `inv_rms_max = 1 / sqrt(eps)`. This
clips the gain for near-zero inputs. The `eps` term in `sum_sq/d + eps` provides this
automatically if `eps` is sized correctly relative to the input format's LSB^2.

**Issue 3: Gamma coefficient precision.**

`gamma` is typically stored as FP16 in the weight file. When fusing with the INT8 pipeline,
`gamma * inv_rms` is computed in FP16 or BF16, then quantised to a fixed-point scale factor
for the INT8 multiply. If the scale factor is itself quantised too coarsely (e.g., to 8 bits),
each element's effective normalisation scale has ~0.4% error. For most models this is
acceptable, but for extremely sensitive models (instruction-following tasks with precise token
probabilities), 16-bit scale factors are safer.

**Issue 4: Accumulator rounding mode.** Truncation vs. rounding in the sum accumulation.
Truncating `x^2` to W_ACC bits instead of rounding introduces a systematic negative bias in
the accumulated sum (each term is slightly underestimated). This makes `sum_sq` slightly too
small, `inv_rms` slightly too large, and the outputs slightly over-scaled. For d=4096 elements,
the bias accumulates as `d * 0.5 LSB` of the squared term. Using round-to-nearest in the MAC
eliminates the bias with negligible area cost (one XOR gate for rounding).

**Hardware mitigation table:**

| Issue | Root cause | Mitigation |
|---|---|---|
| Accumulator overflow | Insufficient W_ACC | Size W_ACC = 2*W_IN + ceil(log2(D)) |
| inv_rms blow-up | Near-zero input | Clamp sum_sq: sum_sq = max(sum_sq, eps_min) |
| Gamma precision loss | FP16 -> fixed-point conversion | Use 16-bit scale factors, not 8-bit |
| Accumulation bias | Truncation in MAC | Use round-to-nearest in squared-term MAC |
| Dynamic range of output | inv_rms * x may overflow | Saturating multiply in Phase 3 |
