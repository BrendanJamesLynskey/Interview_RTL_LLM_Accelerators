# LUT and CORDIC Methods for Transcendental Functions

## Overview

Transcendental functions — exponential, logarithm, sine, cosine, and their relatives — appear
throughout neural network hardware: softmax requires `exp`, attention temperature scaling
requires `exp` and division, normalisation requires `sqrt` and its reciprocal, and rotary
positional embeddings (RoPE) require `sin` and `cos`. None of these functions can be
implemented as a fixed sequence of additions and multiplications without approximation. This
document covers the two dominant hardware approaches: lookup table (LUT) methods and the
CORDIC algorithm, along with hybrid approaches used in production accelerators.

---

## Tier 1 — Fundamentals

### Q1. What is a direct LUT for a transcendental function? What are its fundamental limitations?

**Answer.**

**Direct LUT method.** Precompute the function `f(x)` at every representable input value and
store the results in a read-only memory (ROM). At runtime, use `x` directly as the address
to look up the precomputed output:

```
output = ROM[x]
```

**Latency.** One memory access. For a small table fitting in flip-flop memory: ~1 cycle.
For a table in SRAM: 1-2 cycles. This is the minimum achievable latency for any method.

**Fundamental limitations:**

1. **Exponential area growth.** A W-bit input requires 2^W entries. For W=8: 256 entries —
   trivial. For W=16: 65536 entries of 16 bits = 128 KB of ROM. For W=24: 16 million entries
   of 24 bits = 48 MB — impractical on-chip. The area grows as O(2^W), which limits direct
   LUT to narrow inputs (≤ 10 bits typically).

2. **Output precision.** The output is quantised to the number of bits stored per entry. A
   16-entry LUT for a 4-bit input gives 4-bit addressing precision at the breakpoints; values
   between breakpoints are assigned to the nearest entry's value. The maximum error is
   half the function variation within one step, `(delta_x / 2) * |f'(x_max)|`.

3. **Input range vs. precision trade-off.** The full dynamic range of a W-bit input must be
   covered by 2^W entries. If the function has interesting behaviour only in a sub-range (e.g.,
   sigmoid saturates outside [-6, +6]), entries for the saturation regions waste table space.
   Preprocessing to clamp the input to the active region before indexing improves efficiency.

4. **Read port contention.** If many parallel units share one LUT, only one can read per
   cycle. Each unit needs its own copy of the LUT, which multiplies area by the parallelism
   factor.

---

### Q2. Describe LUT with linear interpolation. How does it improve accuracy without increasing the LUT size?

**Answer.**

**Method.** Split the W-bit input `x` into:
- Upper `U` bits: LUT address, selecting the interval `[x_k, x_{k+1}]`.
- Lower `L` bits: fractional position within the interval `t = lower_bits / 2^L`, in [0, 1).

Look up two adjacent entries `f_k = LUT[addr]` and `f_{k+1} = LUT[addr+1]`. Interpolate:

```
output = f_k + t * (f_{k+1} - f_k)
       = f_k * (1 - t) + f_{k+1} * t
```

**Why it improves accuracy.** A direct LUT with U address bits has a worst-case error of:

```
E_direct = max over all x of |f(x) - f(floor(x * 2^U / range))|
         ≈ (range / 2^U) * max|f'(x)|  (first-order)
```

Linear interpolation with U bits addressing and L bits for the fraction approximates the
function linearly within each interval. The error is the second-order interpolation error:

```
E_interp ≈ (delta_x)^2 / 8 * max|f''(x)|

where delta_x = range / 2^U (interval width)
```

This is second-order in `delta_x` vs. first-order for direct LUT. For the same U bits of
addressing (same LUT size), interpolation gives error proportional to `(range/2^U)^2` instead
of `(range/2^U)`. For `2^U = 64` intervals over a range of 8: `delta_x = 0.125`.

- Direct LUT error ∝ 0.125 * f'_max
- Interpolation error ∝ 0.015625 * f''_max

Typically an 8-16x accuracy improvement for the same LUT size, or equivalently, a 3-4x
reduction in address bits needed for equivalent accuracy.

**Hardware cost added.** Versus direct LUT:
- One additional LUT read (adjacent entry).
- One subtraction.
- One multiply (t * delta_f, where delta_f = f_{k+1} - f_k).
- One addition.

For an accelerator that already contains multipliers, the incremental area cost of the
multiply and adder is small. The LUT size reduction is the dominant saving.

---

### Q3. What is CORDIC? Give a one-sentence description of the core idea.

**Answer.**

CORDIC (Coordinate Rotation Digital Computer, Volder 1959) computes transcendental functions
by decomposing them into a sequence of shift-and-add operations that rotate a vector through
a target angle, converging iteratively to the desired function value.

**Core idea.** A rotation by angle `theta` in 2D cannot be performed in fixed-point hardware
without a multiply. But a rotation by `arctan(2^{-i})` for integer `i >= 0` can: the
rotation matrix becomes:

```
[x_{n+1}]   [1          -sigma_i * 2^{-i}] [x_n]
[y_{n+1}] = [sigma_i * 2^{-i}           1] [y_n]
```

where `sigma_i = +1 or -1` (the rotation direction). This is:
```
x_{n+1} = x_n - sigma_i * 2^{-i} * y_n   (one shift + one subtract)
y_{n+1} = y_n + sigma_i * 2^{-i} * x_n   (one shift + one add)
z_{n+1} = z_n - sigma_i * arctan(2^{-i}) (subtract from angle accumulator)
```

No multiplications — only shifts, additions, and subtractions.

---

### Q4. What functions can CORDIC compute, and what are the two operating modes?

**Answer.**

**Two operating modes:**

**Rotation mode.** Given `(x_0, y_0, z_0)`, rotate the vector `(x_0, y_0)` by angle `z_0`.
After N iterations, `z_N ≈ 0` and:
```
x_N = K * (x_0 * cos(z_0) - y_0 * sin(z_0))
y_N = K * (x_0 * sin(z_0) + y_0 * cos(z_0))
```
where `K = prod_{i=0}^{N-1} 1/cos(arctan(2^{-i})) ≈ 1.64676` (the CORDIC gain constant).

**Vectoring mode.** Given `(x_0, y_0, z_0)`, rotate the vector until `y_N ≈ 0`. The
accumulated angle `z_N = z_0 + arctan(y_0 / x_0)`. This computes `arctan`.

**Functions computable:**

Starting from `(x_0=1/K, y_0=0, z_0=theta)` in rotation mode:
- `x_N = cos(theta)`, `y_N = sin(theta)` — simultaneous sin and cos.

Starting from `(x_0=1, y_0=0, z_0=theta)` in rotation mode with hyperbolic CORDIC:
- `x_N = K_h * cosh(theta)`, `y_N = K_h * sinh(theta)`.

From `sinh` and `cosh`: `exp(theta) = cosh(theta) + sinh(theta)`, `tanh(theta) = sinh/cosh`.

From `cosh` and CORDIC identities: `sqrt(a^2 + b^2)` via vectoring mode (magnitude).

In vectoring mode: `arctan(y/x)`, `log(x)` (via hyperbolic CORDIC).

**Summary of directly computable functions:**
`sin`, `cos`, `arctan`, `sinh`, `cosh`, `tanh`, `arcsinh`, `sqrt`, `exp`, `log`.

---

### Q5. How many CORDIC iterations are needed for a given output precision?

**Answer.**

**Convergence rate.** Each CORDIC iteration adds approximately one bit of accuracy. After N
iterations:
```
error ≈ 2^{-N}
```

This is linear convergence — N iterations gives N bits of correct output.

**Required iterations by precision target:**

| Target precision | Iterations needed |
|---|---|
| 8 bits (INT8) | ~8-10 |
| 10 bits (FP16 mantissa) | ~12-14 |
| 16 bits (INT16) | ~18-20 |
| 23 bits (FP32 mantissa) | ~26-28 |

In practice, 2-4 extra iterations are added as margin for the CORDIC gain constant
correction and rounding errors.

**Comparison with Newton-Raphson.** Newton-Raphson doubles correct bits each iteration
(quadratic convergence). For FP32 accuracy: 5-6 N-R iterations vs. 28 CORDIC iterations.
CORDIC requires 5x more iterations for the same accuracy, but each CORDIC iteration costs
only shifts and adds (no multiplier), while each N-R iteration requires 2-3 multiplications.

**When CORDIC is faster.** CORDIC wins when:
- No hardware multiplier is available (e.g., FPGA without DSP blocks, microcontroller).
- The function is trigonometric (sin/cos) and must be evaluated at low latency with no
  LUT memory.
- The target precision is low (8-12 bits) — few iterations needed.

**When N-R or LUT wins.** N-R/LUT wins when:
- Hardware multipliers are available and underutilised.
- High precision (FP32+) is needed.
- The function is evaluated in large parallel batches (LUT is O(1) per element).

---

## Tier 2 — Intermediate

### Q6. Derive the CORDIC rotation mode update equations from first principles. Show how shifts replace multiplications.

**Answer.**

**Goal.** Rotate 2D vector `(x, y)` by angle `theta` to obtain `(x', y')`:
```
x' = x * cos(theta) - y * sin(theta)
y' = x * sin(theta) + y * cos(theta)
```

This requires four multiplications: `x*cos`, `y*sin`, `x*sin`, `y*cos`. Hardware cost is high.

**CORDIC approach.** Decompose theta into a sum of arctangent angles:
```
theta = sum_{i=0}^{N-1} sigma_i * arctan(2^{-i})

where sigma_i = +1 or -1 chosen to converge z -> 0
```

Apply N micro-rotations, each by angle `alpha_i = arctan(2^{-i})`:

**Single micro-rotation by `alpha_i`:**

```
x_{n+1} = x_n * cos(alpha_i) - sigma_i * y_n * sin(alpha_i)
y_{n+1} = sigma_i * x_n * sin(alpha_i) + y_n * cos(alpha_i)
```

Factor out `cos(alpha_i)`:

```
x_{n+1} = cos(alpha_i) * [x_n - sigma_i * y_n * tan(alpha_i)]
y_{n+1} = cos(alpha_i) * [sigma_i * x_n * tan(alpha_i) + y_n]
```

Since `alpha_i = arctan(2^{-i})`, we have `tan(alpha_i) = 2^{-i}` exactly.

Therefore:

```
x_{n+1} = cos(alpha_i) * [x_n - sigma_i * 2^{-i} * y_n]
y_{n+1} = cos(alpha_i) * [y_n + sigma_i * 2^{-i} * x_n]
```

The term `sigma_i * 2^{-i}` is a shift (by i positions) combined with a sign flip. The
`cos(alpha_i)` scaling accumulates as a product across all iterations:

```
K_N = prod_{i=0}^{N-1} cos(arctan(2^{-i}))
    = prod_{i=0}^{N-1} 1/sqrt(1 + 2^{-2i})
```

This constant `K_N ≈ 0.6073` (its reciprocal `1/K_N ≈ 1.6468` is the CORDIC gain). It can
be pre-applied to the initial vector: set `x_0 = K_N * x_in`, `y_0 = K_N * y_in`, and the
final output is unscaled.

**Key result.** The pseudo-rotation (ignoring the `cos(alpha_i)` factor) uses only:

```
x_{n+1} = x_n - sigma_i * (y_n >> i)   [shift + conditional negate + subtract]
y_{n+1} = y_n + sigma_i * (x_n >> i)   [shift + conditional negate + add]
z_{n+1} = z_n - sigma_i * arctan(2^{-i}) [subtract from precomputed table]
```

Zero multiplications per iteration.

---

### Q7. Compare the hardware cost of a CORDIC unit versus a LUT+interpolation unit for computing `exp(x)` over the range [-4, 4] with 12-bit output accuracy.

**Answer.**

**LUT+interpolation approach:**

Decompose: `exp(x) = exp(x_int) * exp(x_frac)` where `x_int` is the integer part and `x_frac`
is the fractional part in [0, 1).

- `exp(x_int)` for `x_int` in {-4, -3, -2, -1, 0, 1, 2, 3, 4}: 9 entries, trivial ROM.
- `exp(x_frac)` for `x_frac` in [0, 1): LUT with 64 entries (6-bit address), 12-bit values.
  Use 4-bit linear interpolation fraction for ~15-bit effective accuracy (exceeds 12-bit target).

**Hardware resources for LUT+interp:**
- ROM: 64 * 12 = 768 bits (fit in LUT-based ROM on FPGA or synthesised flip-flops on ASIC).
- 9-entry integer exp ROM: 9 * 12 = 108 bits.
- One 4-bit * 12-bit multiplier (for interpolation fraction * delta_f).
- One 12-bit adder.
- One 12-bit multiplier (for exp_int * exp_frac product).
- Pipeline stages: 3-4.
- Area estimate (28nm ASIC): ~400 gate equivalents (GE).

**CORDIC approach for exp(x):**

`exp(x) = cosh(x) + sinh(x)`. Use hyperbolic CORDIC.

For 12-bit output accuracy: need approximately 14 iterations.

Each iteration: 2 shifts + 2 adders + 1 adder (z update) = 3 adders, 2 shifters.
For 14 iterations in a pipelined CORDIC: 14 pipeline stages, each with 3 adders of 16-bit
width.

**Hardware resources for CORDIC:**
- 14 pipeline stages * 3 adders * 16-bit = 42 adders.
- Angle ROM: 14 * 16-bit = 224 bits.
- Pipeline registers: 14 * 3 * 16 = 672 flip-flops.
- Area estimate (28nm ASIC): ~1200 GE (adders cost ~28 GE each for 16-bit; 42 * 28 ≈ 1176 GE
  plus control).

**Summary comparison:**

| Metric | LUT+interp | CORDIC (pipelined) |
|---|---|---|
| Area (GE) | ~400 | ~1200 |
| Pipeline depth | 3-4 | 14 |
| Throughput | 1/cycle | 1/cycle |
| Latency (cycles) | 3-4 | 14 |
| Multiplier required? | Yes (small) | No |
| FPGA (Xilinx) | ~100 LUTs + 1 DSP | ~800 LUTs + 0 DSP |
| Accuracy | Exactly 12 bits | ~14 bits |
| Parameterisability | Easy (change ROM size) | Easy (change N_iter) |

**Verdict.** LUT+interpolation is strongly preferred for `exp` in neural network hardware:
3x lower area, 4x lower latency, and a hardware multiplier is almost always available
in an accelerator context. CORDIC for `exp` is only compelling when no multiplier exists
(pure shift-add datapath, e.g., tiny IoT microcontroller or a purely CORDIC-based engine).

---

### Q8. How is `exp(x)` computed efficiently in fixed-point hardware using range reduction?

**Answer.**

**Range reduction technique.** The key identity:

```
exp(x) = exp(n * ln(2) + r)
        = 2^n * exp(r)

where n = round(x / ln(2)) = round(x * log2(e))
      r = x - n * ln(2)   (remainder, |r| <= ln(2)/2 ≈ 0.347)
```

This decomposes the problem:
1. `2^n` is an exact integer power of 2 — implemented as a left shift of the exponent field
   in floating-point, or a range check in fixed-point.
2. `exp(r)` for `r` in `[-0.347, +0.347]` — a much smaller range than the original `x`.

**Hardware steps:**

```
Step 1: Compute n = round(x * 1.44269504) where 1.44269504 = log2(e)
        (multiply by constant, round to integer)

Step 2: Compute r = x - n * 0.69314718  where 0.69314718 = ln(2)
        (multiply constant + subtract; use two-step Cody-Waite for precision)

Step 3: Compute exp(r) for |r| <= 0.347
        Method A: Polynomial: exp(r) ≈ 1 + r + r^2/2 + r^3/6 (3 terms, <0.001 error)
        Method B: LUT with linear interp over [-0.4, 0.4]: 32-entry table
        Method C: Minimax polynomial fitted to the range

Step 4: Output = exp(r) * 2^n
        In fixed-point: left-shift the result of step 3 by n bits.
        Clamp if n > output format's max exponent.
```

**Precision in Step 2 (Cody-Waite reduction).** Subtracting `n * ln(2)` from `x` when both
are large can cause catastrophic cancellation. The Cody-Waite technique splits `ln(2)` into
two parts `ln2_hi + ln2_lo` where `ln2_hi` has few significant bits and `ln2_lo` carries the
high-precision remainder. This maintains accuracy without widening the arithmetic.

**Fixed-point sizing for Step 3.** The polynomial `1 + r + r^2/2 + r^3/6` for `r` in
`[-0.35, 0.35]`: the output ranges from `exp(-0.35) ≈ 0.705` to `exp(0.35) ≈ 1.419`. A
Q1.15 format covers this range with 15 bits of fraction. The polynomial evaluation requires:

```
r^2 = r * r           (Q2.30 if r is Q1.15)
r^3 = r^2 * r         (Q3.45; truncate to Q3.30 before use)
r^2/2 = r^2 >> 1      (exact right shift)
r^3/6: multiply r^3 by 1/6 ≈ 0.1667 (stored as Q0.15 constant)
sum all terms in Q2.30, truncate to Q1.15
```

---

### Q9. Describe LUT with quadratic (second-order) interpolation. When is it worth the extra hardware?

**Answer.**

**Method.** For each interval `[x_k, x_{k+2}]` spanning two LUT steps, fit a quadratic
through three points `f_{k-1}`, `f_k`, `f_{k+1}`:

```
output = a_k + b_k * t + c_k * t^2

where t = (x - x_k) / (x_{k+1} - x_k) in [0, 1)
      a_k = f_k
      b_k = f_{k+1} - f_{k-1}  (central difference, divided by 2)
            approximated as (f_{k+1} - f_{k-1}) / 2
      c_k = f_{k+1} - 2*f_k + f_{k-1}  (second difference)
```

Alternatively, store explicit `(a_k, b_k, c_k)` in a coefficient ROM, computed offline using
Lagrange polynomial fitting.

**Error analysis.** For linear interpolation over N uniform intervals:
```
E_linear ≈ (range/N)^2 * max|f''| / 8
```

For quadratic interpolation over N/2 intervals (two steps per interval):
```
E_quadratic ≈ (2*range/N)^3 * max|f'''| / 48
```

The error decreases as the cube of the step size — quadratic convergence in h. In practice,
quadratic interpolation allows 4x fewer LUT entries for equivalent accuracy vs. linear
interpolation.

**Hardware cost comparison (per element):**

| Method | Multiplies | Adds | ROM reads |
|---|---|---|---|
| Linear interp | 1 | 2 | 2 |
| Quadratic interp | 2 | 3 | 3 (or 2 if coefficients precomputed) |

**When quadratic interpolation is worth it:**

1. **LUT memory is the scarce resource.** If on-chip SRAM is limited but multipliers are
   available (common in ASIC designs with large MAC arrays), trading one extra multiply for
   4x LUT size reduction is a good deal.

2. **High precision required.** For FP32-equivalent accuracy (23 bits), linear interpolation
   requires 2^12 = 4096 LUT entries (excessive). Quadratic interpolation achieves the same
   accuracy with ~512 entries.

3. **Function has high curvature.** `exp(x)` has rapidly growing `f''`; linear interpolation
   requires many intervals to bound the second-derivative error. Quadratic interpolation's cubic
   error term involves `f'''`, which for `exp(x)` is also `exp(x)` — still large, but the 1/48
   factor and the cubic scaling make it much better than linear's 1/8 * quadratic scaling.

**When linear is sufficient:** INT8 inference (8-bit accuracy, linear interpolation with 64
entries more than sufficient), area-constrained designs where the extra multiply is expensive,
or functions with low curvature (nearly linear activation functions, smooth gating).

---

### Q10. Explain the CORDIC convergence condition and the "double iteration" trick for hyperbolic CORDIC.

**Answer.**

**Standard (circular) CORDIC convergence condition.**

For the angle accumulator `z` to converge to zero, the set of angles `{arctan(2^{-i})}` must
span all possible target angles. This requires:

```
arctan(2^{-(i+1)}) < sum_{j=i+1}^{inf} arctan(2^{-j})

which is satisfied for all i >= 0 because:
arctan(2^{-i}) > arctan(2^{-(i+1)}) for all i
```

The convergence domain for circular CORDIC is `|theta| < pi/2 ≈ 1.743` radians. Inputs
outside this range must be reduced using quarter-wave symmetry:
```
sin(theta + pi/2) = cos(theta)   etc.
```

**Hyperbolic CORDIC and the double iteration problem.**

Hyperbolic CORDIC uses rotation angles `arctanh(2^{-i})` for i = 1, 2, 3, ...

The convergence condition requires:
```
arctanh(2^{-(i+1)}) < sum_{j=i+1}^{inf} arctanh(2^{-j})
```

This fails at i = 4: `arctanh(2^{-4}) > sum_{j=5}^{inf} arctanh(2^{-j})`. The standard
iteration sequence fails to converge because step 4 is too large to be compensated by all
subsequent steps.

**The double iteration fix.** Repeat certain iterations. The standard fix: execute iteration
`i` twice for i in {4, 13, 40, 121, ...} = {i : i = 3k+1 for k = 1, 2, 3, ...}.

The repeated sequence: 1, 2, 3, 4, 4, 5, 6, ..., 12, 13, 13, 14, ...

With double iterations, convergence is restored. The convergence domain for hyperbolic
CORDIC is `|x| < 1.118` (for `sinh`/`cosh`). Range reduction is needed for larger inputs.

**Impact on hardware.** The double iteration means the total number of micro-rotation stages
for N effective iterations is slightly more than N. For N=14 effective iterations (12-bit
accuracy), the actual number of pipeline stages is 15 (one extra stage for the repeated
iteration). This is a minor overhead but must be accounted for in the pipeline stage count
and latency calculation.

---

## Tier 3 — Advanced

### Q11. Design a hybrid exp(x) unit that uses range reduction, a 64-entry LUT with linear interpolation, and a 2-iteration Newton-Raphson correction. Analyse its accuracy budget.

**Answer.**

**Architecture overview:**

```
x_in (Q8.16)
  |
  v
[Range Reduction]
  | n (integer), r (Q0.20)
  v
[LUT + Linear Interp for exp(r)]
  | exp_r (Q1.20)
  v
[Scale by 2^n]
  | exp_x_approx (Q8.20)
  v
[Newton-Raphson Refinement]  <-- optional, for high-precision modes
  | exp_x_refined
  v
[Output Truncation] -> Q8.16
```

**Step 1: Range reduction (see Q8).** Accuracy: Cody-Waite reduction introduces error
bounded by `eps_rr ≈ 2^{-28}` (well below 20-bit target). Cost: 1 multiply + 2 adds.

**Step 2: LUT + linear interpolation for `exp(r)`, `r` in [-0.347, +0.347].**

- 64 entries spanning [-0.35, +0.35]: interval width `delta = 0.7/64 = 0.0109`.
- Linear interpolation error: `delta^2 * max(exp''(r)) / 8`.
- `exp''(r) = exp(r)`, max at r=0.35: `exp(0.35) = 1.419`.
- Error: `0.0109^2 * 1.419 / 8 ≈ 2.1e-5 ≈ 2^{-15.5}`.

This is ~15.5 bits of accuracy — enough for Q1.15 output. For Q1.20 (20-bit fraction), this
falls short by ~4.5 bits.

**Step 3: Newton-Raphson refinement for exp.**

Newton-Raphson for `exp(x)`: given estimate `y ≈ exp(x)`, use:
```
y_{n+1} = y_n * (2 - y_n * exp(-x))
```

But `exp(-x)` is also the unknown. Alternative: since we know the true `x` and have `y ≈
exp(x)`, use the residual:

```
correction = x - log(y)   // compute log of current estimate
y_refined  = y * exp(correction)
```

This is circular (requires log and exp). A simpler first-order Newton approach exploits:

```
If y = exp(x) * (1 + eps) for small eps:
  log(y) ≈ x + eps
  x - log(y) ≈ -eps
  y_refined = y * exp(-eps) ≈ y * (1 - eps) = exp(x)
```

In practice, for a 4.5-bit correction, use the linear approximation:
```
correction = x - y_approx_log   // log computed via LUT
y_refined = y * (1 + correction) // first-order Taylor of exp(correction)
```

This requires one additional LUT (for log) and two multiplies, but reduces error from
2^{-15.5} to 2^{-20} (the dominant error is now the 20-bit fraction of the LUT outputs).

**Accuracy budget summary:**

| Error source | Magnitude | Bits accurate |
|---|---|---|
| Range reduction (Cody-Waite) | ~2^{-28} | 28 |
| LUT + linear interp (64 entries) | ~2^{-15.5} | 15.5 |
| After N-R correction | ~2^{-20} | 20 |
| Final Q8.16 truncation | 2^{-17} (half LSB) | 17 |
| **Composite (dominated by rounding)** | ~2^{-17} | **17** |

The Q8.16 output has at most 1 LSB of rounding error — correctly rounded to the output
format.

**Hardware cost:** 2 LUTs (64+16 entries), 3 multipliers, 4 adders. Pipeline depth: ~8 stages.

---

### Q12. Describe the accuracy implications of finite word-length effects in a pipelined CORDIC. How do accumulated rounding errors interact with the convergence guarantee?

**Answer.**

**Sources of finite-word-length error in CORDIC:**

1. **Register truncation.** After each iteration, `x_n`, `y_n`, `z_n` are stored in W-bit
   registers. The shift `y_n >> i` introduces a truncation error of at most `2^{-i}` in the
   least significant position. Over N iterations, the cumulative truncation error in `x` is:

   ```
   E_trunc ≤ sum_{i=0}^{N-1} 2^{-i} * (K correction factor) ≈ 2 * (2-bit-growth factor)
   ```

   This means the truncation error is bounded by approximately `2` times the value of the LSB
   after N shifts, which is `2 * 2^{-(W-1)}` for a W-bit register. To achieve M bits of
   output accuracy, the register must be at least M+2 bits wide (2 guard bits).

2. **Angle table quantisation.** The precomputed angles `arctan(2^{-i})` are stored with
   finite precision. If stored with P bits, each angle introduces at most `2^{-P}` error per
   step. Over N steps: cumulative angle error ≤ `N * 2^{-P}`. To bound this below `2^{-M}`:
   P ≥ M + log2(N). For M=16, N=18: P ≥ 16 + 4.17 → P = 21 bits.

3. **CORDIC gain constant K.** `K ≈ 1.64676` must be pre-applied to the input. If K is
   quantised to P_K bits, the gain error is `|K_true - K_approx| / K_true * output_magnitude`.
   For M-bit accuracy: P_K ≥ M+1 bits.

**Interaction with convergence guarantee.** The theoretical convergence guarantee assumes
exact arithmetic. With finite word length:

- The residual `z_N` does not exactly reach zero — it reaches a value within `2^{-N}` of zero
  plus the accumulated quantisation error. The final angle error is:

  ```
  |z_N| ≤ 2^{-N} + N * (quantisation error per step)
  ```

- If `N * 2^{-P} ≥ 2^{-N}`, the quantisation error dominates over the algorithmic convergence
  error. The crossover is at `N ≈ P - log2(N)`. For P=20 bits and N=18: the quantisation
  term is `18 * 2^{-20} ≈ 2^{-15.8}`, which is close to the algorithmic error `2^{-18}`.
  Increasing P to 24 pushes quantisation error to `18 * 2^{-24} ≈ 2^{-20.1}`, well below
  `2^{-18}`.

**Design rule.** For a CORDIC targeting M-bit output accuracy with N iterations:
- Data path width: W ≥ M + ceil(log2(N)) + 2 guard bits.
- Angle table precision: P ≥ M + ceil(log2(N)).
- Round (not truncate) during shift: replaces systematic truncation bias with random rounding
  error, halving the accumulated bias.

**Example.** 16-bit accuracy, N=20 iterations:
- W ≥ 16 + 5 + 2 = 23 bits → use 24-bit data path.
- P ≥ 16 + 5 = 21 bits → use 24-bit angle table.
- Use round-to-nearest in the shift: `x >> i` with round bit from the dropped LSB.

---

### Q13. An LLM accelerator must compute softmax, which requires `exp` and division. Describe an end-to-end hardware pipeline for numerically stable softmax with LUT-based exp.

**Answer.**

**Numerically stable softmax.** The naive formula `exp(x_i) / sum(exp(x_j))` overflows for
large x_i. The stable formulation subtracts the maximum:

```
softmax(x_i) = exp(x_i - x_max) / sum_j(exp(x_j - x_max))

where x_max = max(x_j)
```

`exp(x_i - x_max) ≤ 1` for all i, so no overflow.

**Three-pass algorithm:**

```
Pass 1: x_max = max(x_i) for i in 0..d-1            (reduction: max)
Pass 2: exp_i = exp(x_i - x_max) for i in 0..d-1    (element-wise exp)
        sum_exp = sum(exp_i)                          (reduction: sum)
Pass 3: y_i = exp_i / sum_exp for i in 0..d-1        (element-wise divide)
```

**Two-pass online algorithm (flash attention style).** For streaming architectures where all
elements cannot be buffered:

```
Initialise: m = -inf, l = 0

For each x_i:
    m_new = max(m, x_i)
    l_new = l * exp(m - m_new) + exp(x_i - m_new)
    m = m_new, l = l_new

After all elements: for each buffered exp_i, divide by l and rescale by exp(old_m - final_m)
```

This is the two-pass numerically stable softmax. It requires only one exp computation per
element (during the streaming pass) and one correction + division per element at the end.

**Pipeline for attention softmax (sequence length S, head dim H):**

For each attention head:

```
Stage 1 (S cycles): Stream Q*K^T scores x[0..S-1] through max-finder.
    Registers: running max m, pipeline reg for x_i.
    Output: x_max (scalar, after S cycles).

Stage 2 (S cycles): Stream x[i] again (from on-chip SRAM buffer).
    For each x_i: compute shifted = x_i - x_max, then exp(shifted) via LUT+interp.
    Accumulate: sum_exp += exp_i.
    Output: exp_i values (stream), sum_exp (scalar after S cycles).

Stage 3 (S cycles): Compute inv_sum = 1/sum_exp (reciprocal, ~5 cycles latency).
    For each exp_i: y_i = exp_i * inv_sum.
    Output: softmax probabilities y[0..S-1].
```

**Buffer requirement.** Stage 2 reads `x[i]` again (subtracted scores), needing `S * W` bits.
For S=2048 tokens, W=16: 32 Kbits = 4 KB. Stage 3 reads the `exp_i` stream: pipe directly
from Stage 2 into Stage 3 with a FIFO of depth = reciprocal latency (~5 entries). No
additional large buffer needed.

**LUT-based exp considerations for softmax.** Since inputs to exp are always `x_i - x_max ≤ 0`
(negative or zero), the effective input range is `(-inf, 0]`. In practice, attention logits
rarely exceed 20 (softmax with temperature 1), so the effective range is approximately
`[-20, 0]`. A 64-entry LUT covers this with step size `20/64 ≈ 0.31`. For values below -8:
`exp(-8) ≈ 3.4e-4`, small enough to saturate to zero without affecting softmax quality. So
the LUT only needs to cover `[-8, 0]` with step size `8/64 = 0.125` — better accuracy with
the same 64 entries.

**Reciprocal unit for division.** `inv_sum = 1/sum_exp` uses Newton-Raphson:

```
y0 = seed_lut[sum_exp_msbs]
y1 = y0 * (2 - sum_exp * y0)    // 1 iteration: ~8 bits accuracy
y2 = y1 * (2 - sum_exp * y1)    // 2 iterations: ~16 bits accuracy
```

For FP16 softmax (10-bit mantissa), two Newton-Raphson iterations suffice. Hardware: 4 multiplies.

---

### Q14. An interviewer asks you to optimise a combined sin/cos hardware unit for computing RoPE (rotary positional embeddings) in a transformer. Describe the architecture.

**Answer.**

**RoPE computation.** For a position `pos` and dimension `i`, RoPE applies a rotation matrix:

```
[y_{2i}  ]   [cos(theta_i)  -sin(theta_i)] [x_{2i}  ]
[y_{2i+1}] = [sin(theta_i)   cos(theta_i)] [x_{2i+1}]

where theta_i = pos / 10000^{2i/d}
```

For each position-dimension pair `(pos, i)`, we need both `sin(theta)` and `cos(theta)`.

**Key observation: precompute angles offline.**

The angles `theta_i = pos / base^{2i/d}` depend only on the position and dimension index,
not on the input tokens. In batch inference:
- Base `= 10000` (or 500000 in LLaMA-3).
- d = head_dim (e.g., 128).
- Number of unique angles per token: d/2 = 64.
- For sequence length S: S * d/2 unique `(pos, i)` pairs → up to S * 64 sin/cos pairs.

For a fixed context window (e.g., 4096 tokens), all `4096 * 64 = 262144` sin/cos values can
be precomputed at model load time and stored in SRAM as a "position embedding table":

```
cos_table[pos][i] and sin_table[pos][i] for pos in [0, S_max), i in [0, d/2)
```

Storage: `2 * 4096 * 64 * 16 bits = 16 MB` — too large for on-chip SRAM. Solution:

**Streaming architecture.** Compute sin/cos on-the-fly per token using a CORDIC unit or LUT
unit, indexed by `theta = pos * freq[i]` where `freq[i] = 1/base^{2i/d}` is precomputed and
stored in a d/2-entry coefficient ROM (128 entries * 32 bits = 4 Kbits — fits trivially).

**Hardware pipeline for RoPE:**

```
Inputs per cycle: pos (integer), i (dimension index)
Step 1: theta = pos * freq[i]          (multiply, 1 cycle)
         theta is bounded: freq[i] in (1/10000^1, 1], pos < 4096
         -> theta in [0, 4096) -- need range reduction to [0, 2*pi)

Step 2: theta_reduced = theta mod (2*pi)  (modulo by constant, via multiply+subtract)

Step 3: Simultaneous sin(theta) and cos(theta):
  Option A -- LUT+interp: quarter-wave table (pi/2 range, 256 entries per quadrant),
              use symmetry to cover all four quadrants. Provides both sin and cos from
              one table read + interpolation.
  Option B -- CORDIC: single unit produces both x=cos and y=sin simultaneously
              in rotation mode. N=14 iterations for 12-bit accuracy.

Step 4: Apply rotation to (x_{2i}, x_{2i+1}):
  y_{2i}   = cos(theta) * x_{2i}   - sin(theta) * x_{2i+1}
  y_{2i+1} = sin(theta) * x_{2i+1} + cos(theta) * x_{2i}
  (4 multiplies + 2 adds)
```

**LUT+interp approach advantage.** The quarter-wave table for sin over `[0, pi/2]`:
256 entries of 16 bits = 4 Kbits. From `sin(theta)` in the first quadrant:
- cos(theta) = sin(pi/2 - theta): look up at `(256 - addr)` — one LUT read for both.
- No extra computation for cos — the two values come from offset addresses in the same LUT.

**CORDIC approach advantage.** CORDIC naturally produces both `x_N ≈ cos(theta)` and
`y_N ≈ sin(theta)` simultaneously with no additional hardware — one CORDIC unit gives both.

**Comparison for RoPE:**

| Criterion | LUT+interp | CORDIC |
|---|---|---|
| Latency | 3-4 cycles | 14 cycles |
| Area | ~200 GE | ~1200 GE |
| Accuracy | 15-16 bits | 12-14 bits |
| Simultaneous sin+cos | Yes (two reads) | Yes (native) |
| Parallelism for d/2=64 dims | 64 copies: 64 * 200 = 12800 GE | 64 copies: 76800 GE |

For RoPE in a high-throughput LLM accelerator, 64 parallel LUT+interp units is the standard
approach — the 64x parallelism is mandatory for throughput, and at 200 GE per unit, the total
cost is ~13 K GE vs. ~77 K GE for 64 CORDIC units. The 6x area advantage decisively favours
LUT+interpolation for this application.
