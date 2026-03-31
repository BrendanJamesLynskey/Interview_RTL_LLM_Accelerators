# Softmax Hardware Implementation

## Background

The softmax function is applied to the attention score vector of length S (sequence length) for each
query. In hardware, it is a challenging operation because:

1. It requires a sequential scan to find the maximum value (for numerical stability).
2. Exponentiation is transcendental — not directly implementable with basic arithmetic.
3. Division by the normalisation sum is costly.
4. The pipeline cannot begin producing outputs until sufficient input data has been consumed.

At inference, S can range from dozens to tens of thousands of tokens. The hardware must handle this
range efficiently without area-prohibitive lookup tables.

**Mathematical definition**:
```
softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

The subtraction of `max(x)` is the numerically stable formulation, preventing overflow in exp().

---

## Tier 1 — Fundamentals

### Q1: Why must hardware subtract the maximum value before computing exp()? What goes wrong if you don't?

**Answer**

The raw attention scores `q @ k^T / sqrt(d_head)` can reach values of order +/-10 to +/-30 in
BF16/FP16 arithmetic. The exp() function grows extremely rapidly: `exp(30) ≈ 1.07e13` in FP64,
but FP16 only represents values up to 65504. `exp(11) = 59874` fits in FP16, but `exp(12) = 162755`
overflows to infinity. Once any score overflows to +inf, the entire softmax row becomes `inf / inf = NaN`,
corrupting the attention output completely.

**The fix**: Subtract `max(x)` before exponentiation. This shifts the largest score to 0, so
`exp(x_i - max(x)) <= exp(0) = 1` for all i. The result is always in [0, 1], well within
representable range. The softmax value is unchanged because:

```
exp(x_i - max(x)) / sum_j exp(x_j - max(x))
= [exp(x_i) / exp(max(x))] / [sum_j exp(x_j) / exp(max(x))]
= exp(x_i) / sum_j exp(x_j)
```

**Hardware implication**: A two-pass algorithm is needed: pass 1 finds max(x), pass 2 computes
exp() and accumulates the sum. For long sequences with tiled computation, the online softmax
algorithm (see Q7) avoids storing all scores between passes.

**Underflow consideration**: Very negative scores (e.g., x_i - max = -30) produce
`exp(-30) ≈ 9.4e-14`, which rounds to 0 in FP16 (smallest positive is ~5.96e-8). This is
harmless — the token simply receives zero attention weight, which is the correct behaviour
for a heavily negative score.

---

### Q2: List three hardware approaches to computing exp(x) and give the trade-offs of each.

**Answer**

**Approach 1: Lookup Table (LUT)**

Store pre-computed values of exp(x) at evenly spaced input points. For exp(x) over the range
[-8, 0] (post-subtraction range in FP16 attention), a table of 256 entries with linear interpolation
between adjacent entries gives about 8 bits of accuracy.

- Pros: Fixed 1-2 cycle latency regardless of input; no iterative convergence; simple hardware.
- Cons: Accuracy limited by table size; larger ranges require more entries or segmented tables;
  SRAM area for the table itself (256 entries * 16 bits = 512 bytes — small).

**Approach 2: CORDIC (Coordinate Rotation Digital Computer)**

Iterative algorithm using only shifts and additions. exp(x) can be computed via the identity
`exp(x) = 2^(x / ln2)`, with `2^y` computed iteratively. Each CORDIC iteration adds one bit
of precision.

- Pros: No multipliers needed; precision scales with iteration count; same hardware can
  compute multiple transcendental functions (sin, cos, ln).
- Cons: Latency proportional to precision (16 iterations for 16-bit result); area-intensive
  for high-throughput pipelined designs; slower than LUT at the same precision.

**Approach 3: Piecewise Linear Approximation**

Divide the input range into segments; within each segment, approximate exp(x) as a linear
function `a * x + b` where a and b are pre-computed constants stored in a small ROM. Use the
high bits of x as a segment index and the low bits as the interpolation offset.

- Pros: Can achieve >12 bits of accuracy with moderate hardware (32 segments, each needing
  two 16-bit coefficients = 128 bytes); single multiply-add operation; 2-3 cycle latency.
- Cons: Slightly more complex hardware than pure LUT; accuracy depends on segment count;
  implementation bugs can cause visible glitches at segment boundaries.

**Practical choice**: Most hardware accelerators use a piecewise linear or LUT-based approach
for exp() in softmax. CORDIC is more common for sin/cos where LUTs are less area-efficient.
Google TPUs and NVIDIA tensor cores use polynomial approximations.

---

### Q3: How is division implemented for the softmax normalisation step? Why is multiplying by the reciprocal preferred?

**Answer**

The softmax denominator is `D = sum_j exp(x_j - max(x))`. Each output is `exp(x_i - max(x)) / D`.
Dividing by D for each of S outputs would require S division operations — costly in hardware.

**Reciprocal approach**: Compute `R = 1/D` once, then multiply each numerator by R. This replaces
S divisions with 1 reciprocal computation + S multiplications. Multiplication is far cheaper in
hardware than division (a hardware divider typically has 20-50x more area than a multiplier of the
same precision).

**Computing reciprocal in hardware**:

Method 1 — LUT + Newton-Raphson refinement: A coarse LUT provides an initial estimate x_0 of 1/D,
then Newton-Raphson iteration refines it: `x_{n+1} = x_n * (2 - D * x_n)`. Each iteration roughly
doubles the number of correct bits. Starting with 8-bit LUT accuracy, one iteration gives ~16 bits,
two iterations give ~32 bits. For BF16 (7 mantissa bits), one iteration is sufficient.

Method 2 — Integer division unit: A dedicated hardware divider computing 1/D directly. More area
than the LUT+refinement approach but simpler to design and verify.

Method 3 — Float-point hardware: Modern FP units often include a reciprocal instruction. NVIDIA
CUDA uses `__frcp_rn()` (round-to-nearest reciprocal) which is a single-cycle hardware operation
on modern GPUs.

**Precision consideration**: The softmax sum D accumulates S exponentials. For S=2048, with values
in [0,1], the sum can reach 2048, requiring additional dynamic range in the accumulator. Using
FP32 accumulation for the sum (even with FP16 inputs) prevents overflow and ensures accurate
normalisation.

---

### Q4: Sketch a simple two-pass softmax hardware pipeline and identify its latency in terms of S.

**Answer**

**Two-pass pipeline**:

```
Pass 1 — Max Finding:
  Input scores x_0, x_1, ..., x_{S-1} arrive sequentially.
  Running max register: m = max(m, x_i) each cycle.
  After S cycles: m = max(x)

Pass 2 — Exp, Sum, Multiply:
  Scores must be re-read (requires buffering or second memory pass).
  For each x_i: compute e_i = exp(x_i - m), accumulate D = D + e_i.
  After S cycles: D = sum of all exp values.

Reciprocal:
  R = 1/D (a few cycles)

Pass 3 — Output (or folded into Pass 2 with buffering):
  For each buffered e_i: output e_i * R.
  After S cycles: all S outputs produced.
```

**Total latency**: 2S + reciprocal_latency + S = 3S + const cycles minimum (if exp is 1-cycle LUT).

**Problem**: Pass 2 requires re-reading the scores, which means either:
- Buffering all S scores in a register file (S * 16 bits = 4 KB for S=2048 — feasible for small S)
- Re-reading from SRAM (adds memory bandwidth and latency)
- Using online softmax to avoid the second pass (see Q7)

**Latency with buffering**: For S=512, 3*512 = 1536 cycles at say 1 GHz = 1.5 us per row. For S=2048,
6144 cycles = 6.1 us. This is often the critical path during prefill for long-context models.

---

## Tier 2 — Intermediate

### Q5: Describe how you would pipeline the exp() computation using a LUT with linear interpolation. How many pipeline stages are required?

**Answer**

**Input**: A fixed-point or floating-point value x in the range [-8, 0] (after max subtraction).
**Output**: exp(x), approximately, at the same precision.

**Step 1 — Range decomposition** (1 cycle): Split x into an integer part i = floor(x) and a
fractional part f = x - i. The integer part indexes a coarse LUT of powers of e:
`exp_coarse[i] = exp(i)`, requiring entries for i = -8, -7, ..., 0 (9 entries).
The fractional part f is in [0, 1).

**Step 2 — Fine LUT lookup** (1 cycle): The fractional part f is quantised to 8 bits and used to
index a fine LUT: `exp_fine[f_8bit] = exp(f)`, 256 entries covering [0, 1). Since exp maps [0,1)
to [1, e), all 256 entries fit in FP16 without overflow.

**Step 3 — Multiply** (1 cycle): The final result is `exp(x) = exp_coarse[i] * exp_fine[f_8bit]`.
One FP16 multiply.

**Linear interpolation variant** (avoids 256-entry fine LUT):
Use a 32-entry LUT covering [0,1) at 1/32 granularity, plus linear interpolation:
```
idx = upper 5 bits of f
rem = lower 3 bits of f (scaled remainder)
exp_approx = LUT[idx] + rem * (LUT[idx+1] - LUT[idx]) / 8
```
This requires one LUT access, one subtraction, one multiply-add (2 cycles), but uses only 32 entries.

**Full pipeline**:
```
Stage 1: Decompose x -> (i, f)
Stage 2: LUT lookup for exp_coarse[i] and exp_fine[f]  (parallel reads)
Stage 3: Multiply exp_coarse * exp_fine
```

Total pipeline depth: 3 stages. Throughput: 1 exp() per cycle after fill. For S=2048, all 2048
exp() values are produced in 2048 + 3 = 2051 cycles.

**Accuracy**: With 8-bit fine LUT, the error is bounded by the LUT quantisation error, typically
less than 0.5 LSB of the 8-bit fine table, giving about 8 bits of relative accuracy in exp(x).
This is sufficient for FP16 softmax (FP16 has 10-bit mantissa, but softmax weights are typically
accumulated in FP32 for stability).

---

### Q6: How would you implement the reciprocal `1/sum` using Newton-Raphson iteration in hardware? Show the convergence equations and cycle count.

**Answer**

**Newton-Raphson for reciprocal**: To find R = 1/D, find the root of f(R) = 1/R - D = 0.
Newton update: `R_{n+1} = R_n * (2 - D * R_n)`.

**Convergence**: If the initial estimate R_0 has relative error epsilon_0 (i.e., R_0 = (1 + epsilon_0)/D),
then epsilon_{n+1} = -epsilon_n^2. This is quadratic convergence — the number of correct bits doubles
each iteration.

```
Iteration 0: R_0 from 8-bit LUT  =>  ~8 correct bits
Iteration 1: R_1 = R_0*(2 - D*R_0) =>  ~16 correct bits  (sufficient for BF16)
Iteration 2: R_2 = R_1*(2 - D*R_1) =>  ~32 correct bits  (sufficient for FP32)
```

**Hardware cycle count for one iteration**:
```
Cycle 1: D * R_n          (FP multiply)
Cycle 2: 2 - (D * R_n)    (FP subtract, 2.0 is a constant)
Cycle 3: R_n * (2 - D*R_n) (FP multiply)
```
Each FP multiply-add takes 3-5 cycles in a typical FP16 pipeline. One Newton iteration therefore
takes 6-10 cycles. Two iterations: 12-20 cycles. This is negligible compared to the S-cycle exp
computation.

**Initial estimate LUT**: The sum D ranges from 1 (all scores except one are -inf) to S (all equal
scores). Store `1/D_quantised` for D values spanning the expected range in an 8-bit or 10-bit
indexed table. The table can be small because the initial estimate only needs ~8 bits of accuracy.
For FP16: D is represented as a 16-bit float; use the 8 most significant bits of D's mantissa as the
table index (the exponent provides range scaling separately).

---

### Q7: Explain the online softmax algorithm and derive the recurrence relations. Why is it important for hardware that processes attention in tiles?

**Answer**

**Motivation**: Standard softmax requires two passes over the data: one to find max, one to compute
exp and sum. In tiled hardware, storing all S scores between passes requires S * 2 bytes of buffer
(4 KB for S=2048 at FP16). Online softmax achieves numerically equivalent results in a single pass
with O(1) state.

**State variables** (maintained per query):
- `m`: running maximum of scores seen so far
- `d`: running sum of exp(score_j - m) for all j seen so far
- `o`: running weighted sum of values: sum_j(exp(score_j - m) * v_j) (if V is available)

**Initial state**: m = -inf, d = 0, o = 0 (zero vector)

**Update rule** when new score s arrives (with associated value v):

```python
m_new = max(m, s)
# The existing sum d was computed relative to old m.
# Rescale to new m: multiply by exp(m - m_new)
d_new = exp(m - m_new) * d + exp(s - m_new)
# Rescale accumulated output similarly:
o_new = exp(m - m_new) * o + exp(s - m_new) * v
# Update state:
m, d, o = m_new, d_new, o_new
```

**Final output**: `softmax_output = o / d`  (divide accumulated weighted sum by normalisation)

**Correctness**: After processing all S scores, `o / d` equals `sum_j(softmax(s_j) * v_j)`, which
is exactly the attention output for this query.

**Hardware implications**:

1. **Single-pass**: Scores and values are consumed in one sequential pass. No buffering of
   intermediate scores needed — the S * 2-byte score buffer is eliminated.

2. **Tile-compatible**: When scores arrive in tiles (as in FlashAttention), apply the same update
   rule tile-by-tile. The per-tile sum of exp() values is computed and merged into the global state
   at tile boundaries using the rescaling factors.

3. **Parallelism**: Multiple queries can share the same exp() hardware in a time-multiplexed manner.
   Each query maintains its own (m, d, o) state in a small register file.

4. **Critical path per score**: Each update requires: one max comparison, one subtraction
   (m - m_new), two exp() calls, two multiply-adds. Total: ~5-7 cycles per score, fully pipelineable.

---

### Q8: What is the impact of mixed-precision on softmax hardware? Specifically, discuss BF16 input, FP32 accumulation, and BF16 output.

**Answer**

**The problem with BF16-only accumulation**:

BF16 has 7 mantissa bits (~3 decimal digits of precision). When summing S=2048 exp() values each
in [0, 1], the sum can reach ~2048. Summing 2048 numbers of order 1 into a BF16 accumulator:
the ULP (unit in last place) at value 2048 is `2048 * 2^{-7} = 16`. An individual addend of
~0.001 (for a small exp value) would be completely lost — it is smaller than the accumulator ULP.
This causes systematic underestimation of the denominator, shifting softmax weights toward tokens
with large exp values, distorting the attention distribution.

**FP32 accumulation solution**:

The sum accumulator and the output accumulator (o in online softmax) are widened to FP32.
Individual exp() inputs (in BF16) are converted to FP32 before accumulation. At FP32, the ULP
at value 2048 is `2048 * 2^{-23} ≈ 2.4e-4`, far below any exp() value. No precision loss occurs.

**Hardware cost**: The accumulator register widens from 16 bits to 32 bits. The adder input
mux must handle BF16->FP32 conversion, which is free (BF16 is simply the top 16 bits of FP32;
zero-extend the mantissa). The area cost is minimal — one 32-bit register versus one 16-bit
register per query, plus a wider adder.

**Output conversion**:

After computing `o / d` (both FP32), the result is rounded to BF16 for output. This final
truncation loses ~7 bits of precision but is unavoidable given the downstream data format.
The key is that the accumulation was done correctly in FP32 before truncation.

**Summary**: The hardware pattern is:
```
Input scores: BF16 -> convert to FP32 for all accumulation
exp() unit: BF16 input, FP32 output
Sum accumulator: FP32
Output weighted sum: FP32
Final output: FP32 -> round to BF16
```

This is standard practice on TPUs, GPUs, and most neural network accelerators.

---

## Tier 3 — Advanced

### Q9: Design the microarchitecture of a pipelined online softmax unit for a single query. Specify all pipeline registers, mux selections, and the feedback path for the running state.

**Answer**

**Interface**:
- Input: one score per cycle (FP16), valid signal
- Input: associated value vector v_j (d_head FP16 values) — must arrive same cycle as score
- Output: o / d (FP32 vector of d_head), valid after last score processed

**Pipeline microarchitecture**:

```
Stage 1 — Score Register & State Read (1 cycle):
  Registers: score_s1 (FP16), v_s1 (FP16 * d_head)
  Read state registers: m_curr (FP32), d_curr (FP32), o_curr (FP32 * d_head)

Stage 2 — Max Update (1 cycle):
  m_new = max(m_curr, fp16_to_fp32(score_s1))
  delta = m_curr - m_new   (FP32 subtract; result in [-max_score, 0])
  Registers: m_new (FP32), delta (FP32), score_s2 (FP32), v_s2

Stage 3 — Exp Computations (3 cycles, pipelined):
  exp_delta = exp(delta)        [exp pipeline: stage 3a, 3b, 3c]
  exp_score = exp(score_s2 - m_new)
  Note: two exp() units running in parallel, or one unit alternating with 1-cycle slack

Stage 4 — Scale Old State (1 cycle):
  d_scaled = exp_delta * d_curr
  o_scaled = exp_delta * o_curr   (d_head parallel multiplies)

Stage 5 — Update (1 cycle):
  d_new = d_scaled + exp_score
  o_new = o_scaled + exp_score * v_s5   (d_head FMA operations)
  Registers: d_new, o_new, m_new

Stage 6 — State Write-Back (1 cycle):
  Write m_new, d_new, o_new back to state registers.
  Mux on state register write port: "if valid_in, write stage 6 result; else hold"
```

**Feedback path**: Stages 1 and 6 share state registers. A pipeline bubble occupies the 6 cycles
between a state read (stage 1) and its corresponding write-back (stage 6). During this window, a
new score must not be consumed without stalling. Solutions:

Option A — **Stall**: Insert 5 bubble cycles between consecutive scores. Throughput: 1 score per
6 cycles. For S=2048, total cycles = 6 * 2048 = 12,288. Latency ~12 us at 1 GHz.

Option B — **Deep state file**: Maintain 6 sets of (m, d, o) state — one per pipeline stage. Each
cycle a different query is in flight, interleaved round-robin. Only works if d_head * 6 = 768 FP32
registers are acceptable (~3 KB for d_head=128). Throughput: 1 score per cycle (6x improvement).

**Final division**: After all S scores, the output is computed as:
```
o_final / d_final
```
This is d_head parallel FP32 divides (or reciprocal + multiply), all using the same d_final.
One reciprocal R = 1/d_final, then d_head FP32 multiplies: total ~10-15 cycles for the division.

---

### Q10: How does the pipelining of softmax interact with the causal mask for variable-length sequences? What hardware mechanisms handle the "last valid token" boundary cleanly?

**Answer**

**Problem statement**: In a batch, different sequences have different lengths S_1, S_2, ..., S_B.
The softmax for sequence i operates over S_i scores, not the padded maximum S_max. Padding tokens
receive masked scores of -inf and must not affect the normalisation of real tokens.

**Mechanism 1 — Sequence length register**:
Each query processed by the softmax unit has an associated sequence length register S_i. A counter
counts valid scores. When the counter reaches S_i, a "last_valid" signal is asserted. Subsequent
scores (padding) are forced to -inf regardless of their computed value, simply by the counter
comparison. No changes to the softmax pipeline itself are needed.

**Mechanism 2 — Valid mask input**:
An alternative is to provide a per-score valid bit alongside each score. The score unit sets
valid=1 for real tokens and valid=0 (with score=-inf) for padding. The softmax unit processes both
identically — exp(-inf) = 0 adds nothing to the sum or weighted output.

**End-of-sequence flush**:
When the last valid score is consumed, the pipeline still has scores in flight in stages 1-5 (the
pipeline depth). The state write-back in stage 6 must happen for all in-flight valid scores before
the final `o/d` division can begin. A flush counter tracks how many additional cycles are needed
after the last valid input before the output is ready. This is exactly the pipeline depth (6 cycles
in the example above). The division is scheduled to begin after the flush completes.

**Zero-division guard**:
If S_i = 0 (degenerate empty sequence) or all scores are -inf (fully masked row), d_final = 0.
A comparator on d_final triggers a bypass: the output is set to the zero vector or a uniform
distribution (1/S_i) as specified by the model. In practice this should never occur in a
correctly constructed attention mask, but the hardware must handle it without producing NaN outputs
that would corrupt downstream computation.

**Batch interleaving**:
For multi-query batches, multiple softmax units run in parallel or one unit is time-multiplexed
across queries. Each query has its own (m, d, o) state. The state file is indexed by query_id,
which is a tag carried alongside each score through the pipeline. On write-back, query_id selects
the correct state row to update.
