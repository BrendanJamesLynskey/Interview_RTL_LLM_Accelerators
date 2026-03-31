# Activation Functions Hardware

## Overview

Modern LLMs have moved well beyond simple ReLU activations. The dominant nonlinearities in
production models — GELU, SiLU/Swish, and their gated variants like SwiGLU — are significantly
more expensive to compute in hardware than a comparator. This document covers how to implement
them efficiently on custom silicon and FPGA, the approximation techniques that trade accuracy
for area and latency, and the interview questions that probe whether a candidate truly understands
the design space.

---

## Tier 1 — Fundamentals

### Q1. What is ReLU and why was it popular in early neural networks? How is it trivially implemented in hardware?

**Answer.**

ReLU (Rectified Linear Unit) is defined as:

```
ReLU(x) = max(0, x)
```

In a fixed-point or floating-point datapath, this reduces to a single comparison: if the sign
bit is set (negative), output zero; otherwise pass the value through unchanged. In hardware this
is implemented as a multiplexer whose select signal is the sign bit — zero extra logic beyond
the sign-bit extraction already present in most datapaths. Latency is effectively zero (one
MUX delay), and the area cost is negligible.

ReLU became popular because:
1. It avoids the vanishing gradient problem that afflicts sigmoid and tanh.
2. Its derivative is either 0 or 1 — no expensive multiply in the backward pass.
3. Hardware cost approaches zero.

**Why it matters today.** Modern LLMs do not use bare ReLU because it produces sparse
activations that hurt representational capacity at large model scale. Understanding why we
moved away from it contextualises every subsequent activation question.

---

### Q2. Define GELU. Write its exact formula and describe what it computes intuitively.

**Answer.**

GELU (Gaussian Error Linear Unit) was introduced by Hendrycks & Gimpel (2016) and is used in
GPT-2, BERT, and many subsequent models.

**Exact definition:**

```
GELU(x) = x * Phi(x)
```

where `Phi(x)` is the CDF of the standard normal distribution:

```
Phi(x) = (1/2) * [1 + erf(x / sqrt(2))]
```

Expanding:

```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

**Intuition.** GELU gates the input `x` by the probability that a standard normal random
variable is less than `x`. For large positive x, `Phi(x) -> 1` and the function approaches
the identity. For large negative x, `Phi(x) -> 0` and the output is driven to zero. Unlike
ReLU, the transition through zero is smooth, which provides a nonzero gradient everywhere and
improves optimisation.

**Tanh approximation** (used in practice because `erf` is expensive):

```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

This approximation is accurate to within ~0.001 and is what most inference engines use.

---

### Q3. What is SiLU (Swish)? How does it relate to GELU?

**Answer.**

SiLU (Sigmoid Linear Unit), also called Swish, is defined as:

```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

Like GELU, it gates the identity `x` by a smooth probability-like function — here the logistic
sigmoid instead of the normal CDF. Both functions:
- Are smooth and differentiable everywhere.
- Are approximately linear for large positive x.
- Suppress large negative values smoothly.
- Have a small negative lobe (outputs slightly below zero for inputs near -1 to -3).

**Computational difference.** SiLU requires `sigmoid(x) = 1/(1+exp(-x))`, which demands either
an exponential unit or a sigmoid LUT. GELU with the tanh approximation requires `tanh`, which
can be computed via `exp` as well. In practice both have similar hardware cost. SiLU is
sometimes preferred because it avoids the polynomial pre-warp of the tanh GELU approximation.

---

### Q4. What is a gated activation function? Describe SwiGLU and explain why LLaMA uses it.

**Answer.**

A gated activation function splits the hidden dimension into two halves and uses one half to
gate the other:

```
Gated(x) = f(W1 * x) elementwise_multiply g(W2 * x)
```

where `f` is a nonlinearity and `g` is the gate branch.

**SwiGLU** (Shazeer 2020) uses SiLU as the nonlinearity:

```
SwiGLU(x, W, V, b, c) = SiLU(x * W + b) elementwise_multiply (x * V + c)
```

In practice (LLaMA architecture) the bias terms are dropped and the feed-forward block becomes:

```
FFN_SwiGLU(x) = (SiLU(x * W1) elementwise_multiply (x * W3)) * W2
```

**Why LLaMA uses SwiGLU:**
1. Empirically outperforms ReLU and GELU on language modelling benchmarks at the same parameter
   count (Touvron et al. 2023).
2. The gating mechanism allows the network to adaptively suppress or pass-through information
   per-dimension, providing richer representational capacity.
3. Requires ~50% more weight matrices (W1, W2, W3 vs W1, W2 for standard FFN) but the hidden
   dimension is scaled down by 2/3 to keep parameter count constant, so total FLOPs are similar.

**Hardware implication.** SwiGLU means the nonlinear unit sees the result of one matrix
multiply, not the raw input embedding. The input range is thus bounded by the weight
distribution and the previous LayerNorm, which enables tighter fixed-point range analysis.

---

### Q5. Why is computing `erf` or `tanh` expensive in hardware?

**Answer.**

Both `erf` and `tanh` are transcendental functions with no closed-form expression in terms of
basic arithmetic (+, -, *, /). Computing them requires one of:

1. **Taylor/polynomial expansion**: e.g., `tanh(x) ≈ x - x^3/3 + 2x^5/15 - ...`. Converges
   slowly; needs many multiply-accumulate operations; only valid near x=0.

2. **CORDIC**: Iterative shift-and-add algorithm. Accurate but slow (one iteration per bit of
   precision) and difficult to pipeline efficiently for very short latency targets.

3. **LUT**: Store precomputed values. Provides O(1) latency but consumes SRAM area and has
   limited precision without interpolation.

4. **Exponential-based identity**: `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`. Trades the
   problem to computing `exp`, which can be done efficiently by decomposing into integer and
   fractional parts, but still requires an `exp` unit.

The fundamental issue is that these functions have no single dominant term; they require global
approximation over the entire input domain, which demands either significant computation or
significant memory.

---

## Tier 2 — Intermediate

### Q6. Describe the piecewise linear (PWL) approximation method for GELU. What are its trade-offs?

**Answer.**

**Method.** Divide the input domain into N segments. For each segment `[x_i, x_{i+1}]`, store
a slope `m_i` and offset `b_i` such that:

```
GELU(x) ≈ m_i * x + b_i   for x in [x_i, x_{i+1}]
```

The segment boundaries and coefficients are computed offline using least-squares fitting or
by sampling the true GELU at the breakpoints.

**Hardware implementation:**

1. Register the input x.
2. Decode the segment index from the MSBs of x (if segments are uniform-width) or via a
   comparator tree (if non-uniform).
3. Read `m_i` and `b_i` from a small coefficient ROM.
4. Compute `m_i * x + b_i` with a multiply-accumulate unit.
5. Clamp output to valid range.

**Typical latency:** 2-4 clock cycles (decode + multiply + add).

**Trade-offs:**

| Factor | More segments | Fewer segments |
|---|---|---|
| Accuracy | Higher | Lower |
| ROM area | Larger | Smaller |
| Decode logic | More complex | Simpler |
| Max error | Smaller | Larger |

**Non-uniform segmentation** is important for GELU because the function's curvature is highest
near x=0 and nearly flat for |x| > 3. Placing more breakpoints near zero dramatically reduces
error for the same number of segments. For example, 8 non-uniform segments can match the
accuracy of 32 uniform segments.

**Saturation handling.** For |x| > 4 (approximately), GELU(x) ≈ x for positive and ≈ 0 for
negative. These can be handled by direct comparators before the PWL unit, eliminating the need
for segments in the tails.

**Common interview mistake.** Candidates often propose uniform segmentation. The follow-up
question is always: "Where is the error concentrated?" The answer is near x=0, where the
curvature is highest. Non-uniform segmentation is the correct approach.

---

### Q7. Describe the LUT with linear interpolation approach. How does it improve on a direct LUT?

**Answer.**

**Direct LUT.** Store one output value per input code. For a W-bit input and N output bits, the
LUT has 2^W entries of N bits. Accuracy is limited by the quantisation step size 1/2^W.

**LUT + linear interpolation.** Split the W-bit input into:
- Upper `U` bits: LUT address, selecting two adjacent entries `f[k]` and `f[k+1]`.
- Lower `L` bits: fractional offset within the segment `alpha = lower_bits / 2^L`.

Compute:

```
output = f[k] + alpha * (f[k+1] - f[k])
         = f[k] * (1 - alpha) + f[k+1] * alpha
```

This is a linear blend between adjacent LUT entries.

**Accuracy improvement.** With U address bits and L fraction bits (W = U + L total input bits):
- Direct LUT with U bits: quantisation error proportional to `1/2^U`.
- LUT+interp with U+L bits: error is the interpolation error, which for a smooth function like
  GELU is proportional to `(delta_x)^2 * f''(x) / 8` where `delta_x = 1/2^U`. This is the
  second-derivative error of linear interpolation — much smaller than the zeroth-order error
  of direct lookup.

**Hardware cost.** Versus a direct LUT with W=U+L bits:
- LUT size reduced from 2^(U+L) to 2^U entries — exponentially smaller.
- Added cost: one subtraction, one multiply (or shift-add), one addition.
- Net: significant area saving for equivalent accuracy, or better accuracy for same area.

**Practical sizing example.** GELU over [-4, 4] with 8-bit input:
- Direct LUT: 256 entries, ~8-bit accuracy.
- U=6, L=2 LUT+interp: 64 entries + small multiplier, effectively 10-11 bits of accuracy.

---

### Q8. You are designing a fixed-point SiLU unit for a Q4.12 (4 integer bits, 12 fraction bits) datapath. What are the key design decisions?

**Answer.**

**Step 1: Analyse the function domain.**

The input is Q4.12: range [-8.0, +7.999...], 16-bit signed. SiLU(x) = x * sigmoid(x).

For the sigmoid sub-function:
- sigmoid(-8) ≈ 0.000335 (negligible, output ≈ 0)
- sigmoid(+8) ≈ 0.99966 (nearly 1, output ≈ x)
- Saturation threshold: |x| > ~6, sigmoid ≈ 0 or 1.

This immediately tells us the effective input range for the sigmoid LUT is approximately [-6, +6].

**Step 2: Choose sigmoid implementation.**

Option A — LUT+interpolation: 64-entry LUT with 4-bit interpolation fraction.
Option B — Piecewise linear: 8-16 segments with stored slopes.
Option C — Polynomial: `sigmoid(x) ≈ 0.5 + 0.25x - x^3/48` (valid near x=0, needs range splitting).

For Q4.12, a 64-entry LUT over [-6, +6] with 3-bit linear interpolation provides about 15 bits
of accuracy — better than the 12-bit fraction width needed, so LUT+interp is a clean choice.

**Step 3: Multiplication.**

`SiLU(x) = x * sigmoid(x)`

Both are Q4.12. The product is Q8.24 (24-bit fraction). We need to round/truncate back to
Q4.12, discarding the lower 12 bits with rounding.

Watch for overflow: the integer part grows from 4 to 8 bits in the full product, but since
`|sigmoid(x)| <= 1`, the output magnitude is bounded by `|x|`, so the integer part of the
output is at most 4 bits. Truncate the full product back to Q4.12 safely.

**Step 4: Negative lobe.**

SiLU has a negative minimum at x ≈ -1.28 where SiLU(-1.28) ≈ -0.278. The Q4.12 format can
represent this exactly (it is within the signed range). This is a correctness check — formats
that cannot represent small negatives (e.g., unsigned or saturated unsigned) would clip this
incorrectly.

**Step 5: Pipeline structure.**

```
Cycle 1: Register input x, compute LUT address from bits[15:6], fraction from bits[5:2]
Cycle 2: LUT read latency; compute interpolation (f[k+1]-f[k])*alpha
Cycle 3: Add f[k] to interpolation result -> sigmoid_q4.12
Cycle 4: Multiply x * sigmoid -> Q8.24
Cycle 5: Round to Q4.12, register output
```

5-cycle latency, fully pipelined at one result per cycle.

---

### Q9. What is SwiGLU's hardware dataflow? How does it differ from a standard FFN nonlinearity?

**Answer.**

**Standard FFN with GELU:**

```
h = GELU(x * W1 + b1)   // [batch, d_ff]
y = h * W2 + b2          // [batch, d_model]
```

The nonlinearity is applied element-wise to the result of one matrix multiply. The hardware
dataflow is sequential: GEMM -> activation -> GEMM.

**FFN with SwiGLU (LLaMA style):**

```
gate = SiLU(x * W1)        // [batch, d_ff_reduced]
up   = x * W3              // [batch, d_ff_reduced]  (no activation)
h    = gate elementwise* up // [batch, d_ff_reduced]
y    = h * W2              // [batch, d_model]
```

where `d_ff_reduced = (2/3) * d_ff` to keep the total parameter count constant.

**Hardware implications:**

1. **Two parallel GEMMs.** W1 and W3 matrix multiplies can run concurrently because they have
   the same input `x`. This is a natural data-parallelism opportunity: schedule both on the
   same GEMM array with different weight tiles, or on two separate GEMM engines.

2. **Nonlinearity placement.** The SiLU activation applies only to the W1 branch (the gate).
   The W3 branch is linear. The activation unit only needs to process `d_ff_reduced` elements,
   not 2*`d_ff_reduced`.

3. **Element-wise multiply.** After both branches complete, a vector multiply of length
   `d_ff_reduced` is needed before W2. This is cheap — one cycle per element in a systolic
   array's accumulator stage.

4. **Memory bandwidth.** Three weight matrices (W1, W2, W3) instead of two. Weight loading
   is the dominant bottleneck in memory-bound inference; the 50% increase in weight matrices
   at 2/3 width is break-even in bytes but may stress the weight prefetch buffer geometry.

---

### Q10. Describe the accuracy vs. area trade-off curve for GELU approximation methods. How would you choose between them for an 8-bit INT8 inference accelerator?

**Answer.**

**Accuracy requirements for INT8.** The quantisation noise floor for INT8 is about 1/256 ≈ 0.004
in normalised units. There is no benefit to an activation approximation with error below this
floor — it is dominated by quantisation error from the surrounding linear layers.

**Method comparison:**

| Method | Max Error | Area (normalised) | Latency |
|---|---|---|---|
| Direct LUT (8-bit addr) | ~0.002 | 1.0x | 1 cycle |
| PWL 8 segments | ~0.01 | 0.3x | 2 cycles |
| PWL 16 non-uniform | ~0.002 | 0.5x | 2 cycles |
| LUT+linear interp (6+2) | ~0.0005 | 0.4x | 3 cycles |
| Tanh polynomial (degree 3) | ~0.001 | 0.6x | 4 cycles |
| Full tanh (CORDIC) | <0.0001 | 3.0x | 8+ cycles |

**Decision for INT8 inference.** The target error is ≤ 0.004 to avoid degrading INT8 accuracy:

- **8 non-uniform PWL segments** is a strong choice: 0.3-0.5x area, meets the error budget,
  2-cycle latency. The segment coefficients for GELU fit in a tiny ROM (~128 bits of slopes +
  offsets for 8 segments).

- **LUT+linear interp (64 entries)** also works and requires only a multiplier, which is
  already present in the datapath for the surrounding matrix multiplies. Reusing the multiplier
  is zero additional area.

- **Full CORDIC/tanh** is overkill — 3x area and 8x latency for accuracy beyond what INT8 can
  use. Reject on area grounds.

**Corner case consideration.** If the chip must also support FP16 inference (e.g., a mixed-mode
accelerator), the error requirement tightens to ~0.0001 (FP16 has ~3.9e-3 relative error but
for activation functions the absolute error budget shrinks). In this case LUT+quadratic
interpolation or a small CORDIC unit becomes justified.

---

## Tier 3 — Advanced

### Q11. How would you design a unified activation function unit that supports GELU, SiLU, ReLU, and SwiGLU without separate hardware for each?

**Answer.**

**Key insight.** All four functions can be expressed as:

```
output = x * gate(x)
```

where:
- ReLU:  gate(x) = step(x) = 1 if x>=0, else 0
- SiLU:  gate(x) = sigmoid(x) = 1/(1+exp(-x))
- GELU:  gate(x) = Phi(x) = 0.5*(1+erf(x/sqrt(2)))
- SwiGLU: output = SiLU(x_gate) * x_linear  (two separate inputs)

The hardware unification opportunity is the `gate(x)` function unit.

**Unified architecture:**

```
                   ┌─────────────────────────────┐
                   │   Gate Function Unit (GFU)  │
  x_in  ──────────►│                             │──► gate_val
                   │  Mode[1:0]:                  │
                   │  00 = step (ReLU gate)       │
                   │  01 = sigmoid (SiLU gate)    │
                   │  10 = normal CDF (GELU gate) │
                   │  11 = pass-through (=1.0)   │
                   └─────────────────────────────┘
                            │
  x_linear ────────────────►│
                   ┌────────▼────────────────────┐
                   │   Output Multiplier          │──► output
                   │   out = x_linear * gate_val  │
                   └─────────────────────────────┘
```

The GFU contains:
1. A LUT+interpolation unit (one hardware block, shared).
2. Three separate coefficient sets in ROM, selected by Mode.
3. A comparator for the step function (ReLU).

For SwiGLU, the mode register is set to sigmoid, x_in receives the gate branch, and x_linear
receives the up-projection branch. The output multiplier is already present.

**Mode switching.** In a pipelined accelerator, the mode bits travel with the data token. Since
activation function type changes only at layer granularity (not token granularity), the pipeline
is never in a mixed state — no flush is needed for mode changes between layers.

**Area overhead vs. dedicated units.** A dedicated SiLU unit and a dedicated GELU unit each
need their own LUT + multiplier. The unified GFU uses one LUT (with a muxed coefficient ROM
that is tiny) and one multiplier. Area saving is approximately 40% over dedicated units.

**Validation concern.** The LUT entries for sigmoid and the normal CDF must be computed to
sufficient accuracy for both INT8 and FP16 modes. If the ROM entries are 16-bit and the
interpolation is done at 16-bit, both precision targets are met.

---

### Q12. In a token-parallel LLM accelerator, activation functions are applied per-element to large vectors (e.g., 4096 elements). How do you structure the activation pipeline to avoid throughput bottlenecks?

**Answer.**

**The problem.** A matmul result for one layer has shape [batch * seq_len, d_ff]. For
LLaMA-7B, d_ff = 11008. Each element needs an activation function. The matmul engine may
produce 64-128 results per cycle. The activation unit must consume at the same rate.

**Throughput matching.** If the GEMM produces T results/cycle, the activation unit must also
process T elements/cycle. This forces either:
1. T parallel activation units (area scales linearly with T).
2. A deeper pipeline that accepts T inputs per cycle and produces T outputs after a fixed delay.

For T=64 with LUT+interp, option 1 (64 parallel units) is the standard approach. Each unit is
small (64-entry LUT + 1 multiplier + 1 adder) and the total area for 64 units is less than
one large multiplier.

**Critical path in the activation unit.** With pipelined LUT+interp:

```
Cycle 1: Latch input, decode LUT address and fraction
Cycle 2: LUT read (SRAM access, typically 1 cycle for small LUTs)
Cycle 3: Compute interpolation: delta * frac
Cycle 4: Add base value: f[k] + delta*frac
Cycle 5: Output multiply: x * gate (for SiLU/GELU)
```

5-cycle pipeline. All 64 units share the same structure. Since inputs are independent
(element-wise operation), there are no inter-element dependencies.

**Avoiding LUT port conflicts.** With 64 parallel units, each needs its own LUT. Options:
1. 64 separate copies of the 64-entry LUT: 64 * 64 * 16 = 65536 bits = 8 KB. This fits in
   a single SRAM macro with 64 read ports if the foundry provides it, or as 64 small
   distributed RAM blocks in FPGA.
2. One shared 64-entry LUT with 64 read ports: only feasible if the SRAM compiler supports
   wide-bus read. Many do not.
3. 8 copies of the LUT, serving 8 elements each in time-division: adds 8x latency, breaks
   throughput matching — generally unacceptable.

The practical choice is 64 small (64-entry) register-file-based LUTs in synthesised flip-flop
memory (for < 4KB) or 64 SRAM instances (for larger LUTs).

**Backpressure and flow control.** The activation unit sits between two matrix multiply engines.
Both sides should use a valid/ready handshake. The activation pipeline has fixed, known latency,
so the downstream GEMM can be pre-scheduled to accept results without bubble cycles.

---

### Q13. A polynomial approximation of GELU uses the tanh-based form. Describe how to implement the cubic polynomial `tanh(a*(x + b*x^3))` efficiently in fixed-point hardware, and identify the numerical precision pitfalls.

**Answer.**

**Target expression (tanh GELU approximation):**

```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Let c1 = sqrt(2/pi) ≈ 0.7978845608
Let c2 = 0.044715

inner = c1 * (x + c2 * x^3)
GELU(x) ≈ 0.5 * x * (1 + tanh(inner))
```

**Hardware dataflow for `inner`:**

```
x^2 = x * x                      (multiply, 1 cycle)
x^3 = x^2 * x                    (multiply, 1 cycle)
t1  = c2 * x^3                   (multiply by constant, 1 cycle)
t2  = x + t1                     (add, 1 cycle)
inner = c1 * t2                  (multiply by constant, 1 cycle)
```

Total for `inner`: 5 sequential multiplications/additions.
`c1` and `c2` are constants and can be absorbed into the multipliers as immediate operands
(no additional registers for them).

**Hardware dataflow for `tanh(inner)`:**

Use LUT+linear interpolation on the tanh LUT. Since `tanh` saturates to ±1 for |x| > 3,
the LUT only needs to cover [-4, +4]; outside this range clamp to ±1.

**Numerical pitfalls:**

1. **x^3 overflow.** In Q4.12, x^3 can reach 8^3 = 512. This does not fit in Q4.12 (max ~8).
   The intermediate `x^3` must be computed in a wider format, e.g., Q12.36. After multiplying
   by `c2 = 0.044715`, the result fits back in Q4.12 (512 * 0.044715 ≈ 22.8, which needs Q5.12
   at minimum). The intermediate precision must be planned explicitly.

2. **Cancellation in `x + c2*x^3`.** For small x, `x` and `c2*x^3` are both small, and the
   sum is dominated by x. No cancellation. For large x (near ±4), `c2*x^3` is ~28x larger than
   x, so x is the small term. Still no catastrophic cancellation (the terms add, not subtract).
   This is relatively benign — the sum is monotone increasing.

3. **c1 scaling error.** Multiplying by `c1 = 0.7978845608` in Q4.12 means the constant is
   represented as `round(0.7978845608 * 4096) = 3268` with a representational error of about
   4e-5. This error propagates to the `inner` argument of tanh. Since tanh is Lipschitz-1, the
   output error is at most 4e-5 — acceptable for INT8, borderline for FP16.

4. **0.5 * x multiply.** This is a right-shift by one bit, which is exact. No floating-point
   error.

5. **`(1 + tanh(inner))` range.** This lies in [0, 2]. If the format is Q1.15 for tanh output
   (range [-1, +1]), adding 1 requires Q2.14 for the sum. Then multiplying by 0.5 gives Q1.14.
   Then multiplying by x (Q4.12) gives Q5.26. Final truncation to Q4.12 loses 14 bits. This is
   a large truncation — you must round correctly to avoid bias.

**Recommended approach.** Keep all intermediates in Q8.24 to avoid overflow on x^3 and
maintain 24 bits of fraction throughout. The final truncation to the output precision (INT8 or
INT16) is done once at the output register.

---

### Q14. Compare direct LUT versus polynomial approximation for SiLU in terms of synthesisability, timing closure, and portability across process nodes.

**Answer.**

**Direct LUT (implemented as ROM or case statement):**

Synthesisability: Straightforward. `case(x)` statements with constant RHS synthesise to
decode logic or SRAM depending on the synthesis tool setting. Timing is deterministic.

Timing closure: A 256-entry x 16-bit ROM has a read delay of ~0.5-1.0 ns in 16nm. For a
2 GHz clock (0.5 ns period), this barely closes and may require pipelining the ROM read.
For 1 GHz, it closes comfortably.

Process portability: ROM timing scales with process node (smaller = faster). No re-tuning
of the approximation coefficients needed when porting.

**Polynomial approximation (multiplier-based):**

Synthesisability: Requires multiply-accumulate chain. Synthesis tools handle this well;
DSP blocks on FPGA or standard cells on ASIC both support it. The polynomial coefficients
are embedded as constants optimised by the synthesiser.

Timing closure: A 16x16-bit multiplier has a delay of ~0.8-1.2 ns in 16nm using
Booth-encoded Wallace tree. Multiple cascaded multiplies (for x^3 term) create a longer
combinational path — requires pipelining into 2-3 stages for >1 GHz.

Process portability: Multiplier delay scales aggressively with node (8nm is ~2x faster than
16nm). The approximation coefficients need no changes, but pipeline stage insertion may change.

**Trade-off summary:**

| Criterion | Direct LUT | Polynomial |
|---|---|---|
| Implementation effort | Low | Medium |
| Area (for equivalent accuracy) | Medium | Low |
| Timing at >2 GHz | Needs deep pipeline | Needs 2-3 stage pipeline |
| Process portability | High | High |
| FPGA block RAM efficiency | Good (uses BRAM) | Good (uses DSP slices) |
| Accuracy control | Exact (at LUT points) | Depends on coefficient precision |
| Input range extension | Hard (LUT size grows) | Easy (adjust saturation logic) |

**Recommendation for a production tape-out.** Use LUT+linear interpolation rather than either
pure approach. It combines the low-latency LUT read with a single multiplier for the fraction
correction, gives the best accuracy per area, and is straightforward to pipeline. The
polynomial approach is preferred when the hardware already contains a MAC unit that can be
shared (e.g., the same DSP block used for the surrounding linear layers in a low-area design).

---

### Q15. An RTL review flags that your GELU unit produces incorrect results for inputs in the range [-0.5, 0] at INT8 precision. Walk through your debugging methodology.

**Answer.**

This question tests systematic debugging, not just knowledge of GELU.

**Step 1: Characterise the failure.**

Run the unit against a software golden model for all 256 INT8 input codes in [-0.5, 0]:
approximately codes -64 to 0 for a Q4.4 mapping, or -128 to 0 for a raw signed-byte mapping.
Record: input code, hardware output, golden output, error. Plot the error. Is it:
- Systematic offset? Suggests a rounding or offset error in the LUT generation.
- Sign error? Suggests the negative-lobe representation is wrong.
- Monotone wrong trend? Suggests a coefficient scaling bug.
- Sporadic? Suggests a timing issue (race, hold violation) or LUT addressing bug.

**Step 2: Check the LUT generation script.**

GELU in the range [-0.5, 0] spans the negative lobe where GELU(x) < 0. If the LUT was
generated with `max(0, x*erf(x))` (accidentally inserting a ReLU), all negative outputs
would be clipped to zero — a common mistake when copying from ReLU-era codebases.

Check: `python3 -c "import scipy.special; x=-0.2; print(0.5*x*(1+scipy.special.erf(x/1.41421)))"` — should give approximately -0.046, not 0.

**Step 3: Check fixed-point format in the ROM.**

If the LUT stores values as unsigned integers (to save a sign bit), negative GELU values
wrap to large positive values. Verify: `$readmemh` loaded data should be in two's complement
for signed outputs. Check that the top bit of negative entries is 1 in the hex file.

**Step 4: Verify address mapping.**

For a 256-entry LUT with 8-bit signed input, the address lines must correctly handle
negative inputs. In two's complement, `-1` is `8'hFF = 255`. The LUT must be indexed such
that `address = unsigned_cast(signed_input)`. A common bug is:

```systemverilog
// WRONG: signed indexing
logic signed [7:0] x;
lut_out = lut[x];  // synthesis tool may warn; behaviour is undefined or truncated

// CORRECT: explicit unsigned cast
lut_out = lut[x[7:0]]; // or $unsigned(x)
```

If the designer used a signed index, the synthesiser may have mapped negative addresses to
zero or the last entry — producing a flat incorrect output for negative inputs.

**Step 5: Waveform inspection.**

Capture the simulation waveform for x = -32 (roughly -0.25 in Q4.4):
- Confirm the address to the ROM is `8'hE0` (224 in unsigned = -32 in signed).
- Confirm the ROM output is the correct two's complement negative value.
- Confirm no downstream sign extension errors.

**Step 6: Regression and sign-off.**

After fixing, run the full [-8, +8] input range at all precisions. Verify:
- Maximum absolute error < 1 LSB (for INT8 this is < 0.004 per unit step).
- No sign errors.
- Correct saturation at ±8.
