# MAC Unit Design

## Overview

The multiply-accumulate (MAC) unit is the fundamental compute primitive in every ML accelerator. Efficient MAC design directly determines the area, power, and frequency of the entire chip. Interview questions in this area test knowledge of digital arithmetic, pipelining, number representation, and design trade-offs. This document covers questions from fundamentals to advanced.

---

## Tier 1 — Fundamentals

### Q1. What is a multiply-accumulate unit and what does it compute?

**Question:** Define the MAC operation. Write the equation. Describe the three components of a MAC unit and their roles.

**Answer:**

A MAC unit computes:

```
accumulator = accumulator + (A * B)
```

More precisely, given inputs A and B, and an existing accumulator value:

```
acc[n] = acc[n-1] + A[n] * B[n]
```

The three main hardware components:

**1. Multiplier:**
Takes two N-bit input operands (A and B) and produces a 2N-bit product. For INT8 x INT8 the product fits in INT16. The multiplier is typically the largest component by area and has the longest critical path.

**2. Adder:**
Adds the multiplier output (2N bits) to the accumulator register output. The adder must be wide enough to hold the running sum without overflow. For 256 accumulated INT8 x INT8 products, the adder must be at least 24 bits; in practice 32 bits is standard.

**3. Accumulator register:**
A flip-flop register that holds the running partial sum between clock cycles. It is connected in a feedback loop: the output feeds back as one input to the adder.

**Schematic (block diagram):**

```
A ───────┐
         ├──► [ Multiplier ] ──► 2N-bit product ──┐
B ───────┘                                         ├──► [ Adder ] ──► [ Reg ] ──► acc_out
                                          acc_in ──┘          feedback ──────────┘
```

**Common mistake:** Confusing MAC with FMA (Fused Multiply-Add). A MAC accumulates into a persistent register (multiple inputs accumulate into one output). An FMA computes a + b*c as a single-round operation and outputs a fresh result each cycle. FMA is used in floating-point units; MAC with a persistent accumulator is the standard structure in integer systolic arrays.

---

### Q2. Why must the accumulator be wider than the input operand width?

**Question:** You are accumulating 128 INT8 products. What is the minimum accumulator width required? Show your derivation.

**Answer:**

**Step 1: Product range.**

INT8 values range from -128 to +127 (signed two's complement).
The product of two INT8 values ranges from: -128 * 127 = -16,256 to +127 * +127 = +16,129.
The extreme is -128 * -128 = +16,384.
Required range: [-16,256, +16,384].
Bits required: ceil(log2(16,384)) + 1 sign bit = 14 + 1 = 15 bits. (INT16 is sufficient for a single product.)

**Step 2: Accumulated sum range.**

After accumulating N products, worst case = N * max_single_product.
For N=128, worst case = 128 * 16,384 = 2,097,152.
Bits required: ceil(log2(2,097,152)) + 1 sign bit = 21 + 1 = 22 bits.
**Minimum accumulator width: 22 bits.**

**Step 3: Practical choice.**

Designers always round up to a power of two or standard width to simplify downstream logic:
- 22 bits → round up to **32 bits** (standard INT32 accumulator for INT8 MACs).
- 32-bit also matches standard integer types and simplifies interfacing with reduction trees, normalisation units, and output buffers.
- For K=4096 accumulations: worst case = 4096 * 16,384 = 67,108,864. Requires 26 + 1 = 27 bits. Still fits in 32 bits.
- For K=65,536: 65,536 * 16,384 = ~1.07B. Requires 30 + 1 = 31 bits. Still fits in 32 bits, barely.

**Rule of thumb:** For INT8 x INT8 accumulation, a 32-bit accumulator is safe for up to K = 2^(32-15-1) = 2^16 = 65,536 accumulations.

**Common mistake:** Using INT16 accumulators for INT8 MACs. INT16 holds a single product but saturates after just two worst-case accumulations.

---

### Q3. What is the difference between a pipelined MAC and a non-pipelined MAC?

**Question:** A MAC unit has a multiplier critical path of 2.0 ns and an adder critical path of 0.5 ns. What is the maximum frequency of a non-pipelined MAC? A 2-stage pipelined MAC? Draw the pipeline stages.

**Answer:**

**Non-pipelined MAC:**

All operations occur in a single clock cycle. The critical path is:
multiplier (2.0 ns) + adder (0.5 ns) = 2.5 ns.
Maximum frequency = 1 / 2.5 ns = **400 MHz**.

One MAC result per cycle, but frequency is limited by the slowest path.

**2-stage pipelined MAC:**

Insert a pipeline register between the multiplier output and the adder input.

```
Cycle 1: Inputs A, B → Multiplier → [Pipeline Reg: product]
Cycle 2: [Pipeline Reg: product] + acc_reg → Adder → [Accumulator Reg: new acc]
```

Stage 1 critical path: 2.0 ns (multiplier only).
Stage 2 critical path: 0.5 ns (adder) + register read (assume 0.1 ns) = 0.6 ns.
Maximum frequency = 1 / max(2.0, 0.6) = 1 / 2.0 ns = **500 MHz**.

**Note on loop-carried dependency:** In a MAC with feedback (acc = acc + A*B), stage 2 reads the accumulator that was written by the same stage. The loop-carried dependency means the pipeline *cannot* be fully utilised for sequential accumulation — you must wait for the accumulator to be written before the next add can use it.

However, in a systolic array with output-stationary dataflow, each PE accumulates independently, so this is not an issue for throughput (each PE has its own accumulator loop at the same rate). The loop-carried dependency only affects the *latency* at which you can read back the final accumulator value, not the *throughput* of the array.

**3-stage pipelined MAC (for further frequency improvement):**

Split the multiplier into 2 stages (1.0 ns each) and keep adder as stage 3 (0.5 ns).
Maximum frequency = 1 / max(1.0, 0.5) = **1.0 GHz** — a 2.5x improvement over non-pipelined.

---

## Tier 2 — Intermediate

### Q4. Explain Booth encoding for integer multipliers. Why is it used?

**Question:** Explain Modified Booth Encoding (Radix-4 Booth). What problem does it solve, and what is the trade-off compared to an array multiplier?

**Answer:**

**The problem with a basic array multiplier:**

An N-bit x N-bit array multiplier computes N partial products (one per bit of the multiplier), then sums them. For INT8 x INT8: 8 partial products. For INT32 x INT32: 32 partial products. The partial product reduction tree grows linearly with N — more partial products means more adder area and longer critical path.

**Booth encoding principle:**

Instead of examining one bit of the multiplier at a time, Booth encoding examines 2 (or 3) bits at a time, reducing the number of partial products by 2x (or more).

**Modified Booth Encoding (Radix-4 / MBE):**

Examine overlapping groups of 3 bits of the multiplier: bits [2i+1, 2i, 2i-1] for i = 0, 1, ..., N/2-1.
Each 3-bit group maps to a partial product that is {0, ±1, ±2} times the multiplicand.

| b[2i+1] | b[2i] | b[2i-1] | Operation |
|---|---|---|---|
| 0 | 0 | 0 | 0 (no partial product) |
| 0 | 0 | 1 | +multiplicand |
| 0 | 1 | 0 | +multiplicand |
| 0 | 1 | 1 | +2*multiplicand (left shift by 1) |
| 1 | 0 | 0 | -2*multiplicand |
| 1 | 0 | 1 | -multiplicand |
| 1 | 1 | 0 | -multiplicand |
| 1 | 1 | 1 | 0 |

For INT8 x INT8: instead of 8 partial products, MBE produces 4 partial products. This halves the partial product reduction tree depth.

**Trade-offs:**

Advantages:
- Reduces partial products by 2x, shrinking the adder tree.
- Reduces critical path of the multiplier (fewer stages in the tree).
- Reduces dynamic power (fewer switching partial product bits).

Disadvantages:
- Requires a Booth encoder per group (small but adds a little area).
- Requires computation of ±2x multiplicand (one extra bit shift and invert path).
- More complex to design and verify.
- For small N (e.g., N=4), the overhead of the encoder may exceed the savings. Booth encoding is most beneficial for N >= 8.

**Common interview trap:** Asking whether Booth encoding changes the result. It does not — it is an algebraic reformulation of the same multiplication, guaranteed to produce the same bit-exact output.

---

### Q5. What is a Fused Multiply-Add (FMA) and how does it differ from a MAC unit?

**Question:** A floating-point FMA computes result = a + b*c with a single rounding at the end. Explain why this is preferable to computing mul(b,c) then add(a, result_of_mul). What are the hardware implications?

**Answer:**

**Without FMA (two separate operations):**

1. Compute p = b * c. Round p to FP precision (e.g., round to nearest FP32).
2. Compute r = a + p. Round again to FP32.

Two rounding errors are introduced. The intermediate rounding of p discards bits that were exactly representable in extended precision.

**With FMA (single operation):**

Compute a + b*c exactly in extended precision (the product b*c is kept at full double-width, typically 2x mantissa bits), then add a, then round *once* to FP32.

The mathematical result is: round(a + b*c), where b*c is computed with no intermediate rounding.

**Why this matters:**

- **Accuracy:** FMA is strictly more accurate for dot-product computations. The Kahan summation algorithm, for example, relies on exact computation of intermediate products. FMA enables this naturally.
- **Performance:** In dot-product loops, FMA computes `acc = acc + w*x` as a single instruction with one fewer operation issue, one fewer register, and half the rounding overhead.
- **Standards:** IEEE 754-2008 mandates FMA support in compliant floating-point units. All modern CPUs and GPU tensor cores expose FMA.

**Hardware differences:**

| | Separate Mul + Add | FMA |
|---|---|---|
| Multiplier output | Rounded to N bits | Full 2N-bit mantissa kept |
| Adder input width | N+1 bits | ~3N bits (to align a with 2N-bit product) |
| Rounding operations | 2 | 1 |
| Area | Smaller (narrower adder) | ~30% larger adder |
| Critical path | Mul → round → align → add | Mul (no round) → align → add → round |
| Common in | Integer systolic arrays | FP32/FP16/BF16 tensor cores |

**Practical note:** NVIDIA Tensor Cores use FMA. The A100/H100 FP16 Tensor Core computes D = A x B + C where C is a full FP32 or FP16 accumulator. This is a matrix-level FMA: no intermediate rounding between the matrix product and the accumulation.

---

### Q6. How do you pipeline a MAC unit to maximise frequency while minimising area? Walk through the design of a 4-stage INT8 MAC.

**Question:** Design a 4-stage pipelined INT8 x INT8 MAC unit with a 32-bit accumulator. Specify what logic goes in each stage, which registers are needed, and identify any forwarding or hazard issues.

**Answer:**

**Target:** INT8 x INT8 multiply, 32-bit accumulation, 4 pipeline stages, output-stationary (accumulator retains state across cycles).

**Stage partitioning:**

```
Stage 1: Partial product generation (Booth encoding)
  - Input: a[7:0], b[7:0]
  - Booth encode b → 4 partial product selectors
  - Generate 4 partial products (each 9 bits, including sign extension to 9 bits)
  - Register: 4 x 9-bit partial products → 36 bits of register

Stage 2: Partial product reduction (Wallace tree level 1+2)
  - Reduce 4 partial products using CSA (carry-save adder) tree
  - 4 → 2 in one CSA level (two 10-bit sum/carry pairs)
  - Register: 2 x 10-bit sum + 2 x 10-bit carry → 40 bits of register

Stage 3: Final product and sign-extension to 32 bits
  - Ripple-carry or CLA to resolve the final carry-save pair → 16-bit product
  - Sign-extend product from 16 to 32 bits
  - Register: 32-bit product

Stage 4: Accumulate
  - 32-bit product + 32-bit accumulator register → 32-bit adder
  - Clamp or saturate if overflow detection is needed
  - Register: 32-bit accumulator (feedback register)
```

**Hazard analysis:**

The accumulator in stage 4 has a loop-carried dependency: the accumulator output feeds back to stage 4's adder input. The loop is: stage 4 output → accumulator register → next cycle's stage 4 input. This is a 1-cycle loop (no inter-stage forwarding needed), meaning every new A*B that arrives at stage 4 can immediately use the accumulator updated by the previous cycle.

However: stages 1-3 feed their *own* pipeline data, and there is no way for the result computed at stage 4 in cycle N to be used as input at stage 1 in cycle N+1 (since stage 1 in cycle N+1 has already started). This is fine because the accumulation at stage 4 is self-contained — the accumulator is updated every cycle and is always current for the next cycle's accumulation.

**Clear signal handling:**

An accumulator clear (start of a new dot product) must be applied at stage 4 synchronously. The clear must be delayed by 3 cycles from when the first new A,B pair is presented at stage 1, so that the clear arrives at stage 4 exactly when the first product arrives. Failure to delay the clear correctly is a common RTL bug.

**Timing budget (example at 1 GHz TSMC 7nm):**

| Stage | Logic | Estimated delay | Slack |
|---|---|---|---|
| 1 | Booth encoder + PP mux | ~300 ps | ~700 ps |
| 2 | 3:2 CSA x 2 levels | ~250 ps | ~750 ps |
| 3 | 16-bit CLA + sign-extend | ~350 ps | ~650 ps |
| 4 | 32-bit adder + sat logic | ~400 ps | ~600 ps |

All stages well within 1 ns clock period.

---

### Q7. How does saturation logic work and when is it required in a MAC unit?

**Question:** Define overflow saturation. When should a MAC unit saturate vs. wrap around (modular overflow)? Describe how to implement saturation for a 32-bit accumulator feeding an 8-bit output.

**Answer:**

**Overflow vs. saturation:**

When an N-bit accumulator exceeds its range, two behaviours are possible:

- **Modular wrap-around (default in most HDL/C):** Bits above N are discarded. E.g., 0x7FFFFFFF + 1 = 0x80000000 (becomes a large negative number for signed types). This is catastrophically wrong for neural network activations — a large positive activation becoming a large negative number corrupts the output completely.

- **Saturation:** The result is clamped to the maximum (or minimum) representable value. 0x7FFFFFFF + 1 = 0x7FFFFFFF. The information is lost but the sign is preserved, and the magnitude error is bounded.

**When saturation is required:**

- In the accumulator itself: if K is very large or inputs are untypical, the 32-bit accumulator could theoretically overflow. For INT8 x INT8 with K=65,536: max value = 65,536 * 16,384 = 2^30 which fits in INT32. Overflow is extremely rare but possible if quantisation is not carefully controlled.
- **Always required** when converting a 32-bit accumulator output to INT8 output (for output activation quantisation). The 32-bit value must be scaled and clamped to [-128, 127] before writing back.

**Implementation of 32-bit to INT8 saturating output:**

```
// After scaling (divide by quantisation scale factor):
// scaled_value is a signed 32-bit value
// Output y is INT8 (signed 8-bit)

if (scaled_value > 127)        y = 8'sd127;   // positive saturation
else if (scaled_value < -128)  y = -8'sd128;  // negative saturation
else                           y = scaled_value[7:0]; // in-range: truncate
```

Hardware implementation: detect overflow by checking the upper bits of the scaled value:

```systemverilog
logic signed [31:0] scaled;
logic signed [7:0]  sat_out;

// Positive overflow: any bit above bit 7 is 1 when the value should be positive
// (i.e., sign bit is 0 but upper bits are non-zero)
// Negative overflow: sign bit is 1 but bits [31:8] are not all 1s

logic pos_overflow = (~scaled[31]) & (|scaled[30:7]);  // positive, too large
logic neg_overflow = (scaled[31]) & (~(&scaled[30:7])); // negative, too small

assign sat_out = pos_overflow ? 8'sd127 :
                 neg_overflow ? -8'sd128 :
                 scaled[7:0];
```

**Power implication:** Saturation logic adds approximately 3-5% area to the accumulator output stage but is architecturally mandatory for correct quantised inference.

---

## Tier 3 — Advanced

### Q8. Compare the area, power, and timing of INT8, FP16, and BF16 MAC units. How does precision choice affect chip-level design?

**Question:** For a fixed silicon area budget of 1 mm^2 in TSMC 7nm, estimate how many INT8 MAC units vs. FP16 MAC units vs. BF16 MAC units you can fit. Discuss the accuracy implications and why different precisions are used for different parts of a model.

**Answer:**

**Relative area of MAC units (normalised to INT8 MAC = 1.0x):**

Based on published data from academic and industry papers:

| Precision | Multiplier bits | Accumulator | Relative area | Relative power |
|---|---|---|---|---|
| INT4 x INT4 | 4x4 | 24-bit | ~0.4x | ~0.3x |
| INT8 x INT8 | 8x8 | 32-bit | 1.0x | 1.0x |
| FP16 x FP16 | 10x10 mantissa (11-bit with hidden) + exp logic | 32-bit | ~2.5x | ~2.8x |
| BF16 x BF16 | 7x7 mantissa (8-bit with hidden) + exp logic | 32-bit | ~1.8x | ~2.0x |
| FP32 x FP32 | 23x23 mantissa + exp logic | 32-bit | ~8.0x | ~10x |

Note: FP MAC units are larger because they include exponent alignment logic, leading-one detection for normalisation, rounding logic, and NaN/Inf handling. Integer MACs have none of this.

**At TSMC 7nm, estimated density for a simple MAC (not including SRAM, routing):**

- INT8 MAC: approximately 1,000 standard cells → ~0.002 mm^2 per MAC.
- FP16 MAC: approximately 2,500 standard cells → ~0.005 mm^2 per MAC.
- BF16 MAC: approximately 1,800 standard cells → ~0.0036 mm^2 per MAC.

**For 1 mm^2:**

| Precision | MACs per mm^2 | Theoretical TOPs at 1 GHz |
|---|---|---|
| INT8 | ~500 | 0.5 TOPS |
| BF16 | ~278 | 0.28 TOPS (in FP16 ops equivalent) |
| FP16 | ~200 | 0.2 TOPS |

(These are rough order-of-magnitude estimates; real chips also allocate significant area to SRAM, interconnect, and control logic.)

**Accuracy implications and precision selection strategy:**

*INT8 (W8A8):*
- Suitable for weights and activations in most transformer layers after quantisation-aware training or post-training quantisation (PTQ).
- Problematic for attention softmax (requires wide dynamic range), LayerNorm (requires high precision for mean/variance), and embedding lookup (usually kept at FP16 or BF16).

*BF16:*
- Same 8-bit exponent as FP32 → same dynamic range → no overflow in any layer that worked in FP32.
- Smaller mantissa (7 bits vs. 23 bits in FP32) → lower precision for small differences, but sufficient for most inference.
- Preferred for: attention score computation, softmax, LayerNorm, residual additions.

*FP16:*
- Narrower exponent (5 bits) → overflow/underflow in attention for long sequences (logits can exceed FP16 max of 65504). Requires careful scaling.
- Higher mantissa precision than BF16 (10 bits vs. 7 bits) but the dynamic range problem makes it less reliable than BF16 for training.
- Used in: CUDA tensor cores for mixed FP16/FP32 accumulation.

**Practical chip design decision:**

Most modern LLM inference chips use a heterogeneous precision strategy:
- INT8 or INT4 MACs for linear layers (GEMM/GEMV): maximum area efficiency.
- BF16/FP32 accumulation registers: prevents loss of precision in long dot products.
- BF16 special function units for softmax, LayerNorm, activation functions: small number of units, high precision.
- Dequantisation units (INT8 → BF16 scale-and-convert) in the weight load path.

This hybrid approach achieves near-INT8 area efficiency while maintaining FP-level accuracy for numerically sensitive operations.

---

### Q9. How does the choice of Carry-Ripple, Carry-Lookahead, and Carry-Save adder topologies affect MAC unit design at different pipeline depths?

**Question:** You are designing the accumulator adder of a 32-bit MAC unit. Compare a 32-bit Ripple-Carry Adder (RCA), a 32-bit Carry-Lookahead Adder (CLA), and a Carry-Save Adder (CSA) used in a reduction tree, for critical path, area, and when each is appropriate.

**Answer:**

**Ripple-Carry Adder (RCA):**

Structure: N full adders chained with carry out → carry in.
Critical path: proportional to N. For 32-bit: approximately 32 * t_FA where t_FA is one full adder delay (~50-80 ps in 7nm).
At 70 ps per stage: 32 * 70 ps = 2.24 ns. Maximum frequency: ~450 MHz.
Area: minimum (N full adders, no lookahead logic).
Use case: Appropriate only at low frequencies (< 500 MHz) or when area is the critical constraint. Not used in high-performance MAC accumulators.

**Carry-Lookahead Adder (CLA):**

Structure: Groups of k bits compute generate (G) and propagate (P) signals simultaneously, then compute carry into each group in O(log N) gate levels.
Critical path for 32-bit (4-bit groups, 2-level hierarchy): approximately 4-5 gate levels = ~150-200 ps.
Maximum frequency: ~5 GHz in 7nm — well above any practical MAC unit frequency.
Area: approximately 2-3x an RCA of the same width due to G/P logic and carry network.
Use case: The standard choice for the final accumulator adder in a MAC unit. Balances speed and area well.

**Carry-Save Adder (CSA):**

Structure: Takes 3 N-bit inputs, produces two N-bit outputs (sum vector and carry vector) with NO carry propagation. Each bit position is independent.
Critical path: 1 full adder delay (gate level ~50-70 ps) — independent of N.
Area: Same as one full adder array (N full adders).
Limitation: Does NOT produce a binary result. The sum and carry must be combined by a final CLA or RCA to get the actual value.
Use case: Intermediate stages of a partial product reduction tree (Wallace/Dadda tree). A sequence of CSAs reduces many partial products efficiently; the final CLA resolves the last carry-save pair.

**Choosing the right adder for a MAC stage:**

| Adder | Stage | Reason |
|---|---|---|
| CSA | Partial product reduction (stages 1-N-1) | Zero carry propagation; handles 3-input reduction per stage |
| CLA | Final product resolution (last stage of tree) | Must produce binary output from carry-save pair |
| CLA | Accumulator adder (feedback add) | 2-input add with binary inputs; CLA is optimal |
| RCA | Never in critical path | Too slow for any target above 500 MHz |

**Synthesis tool note:** Modern synthesis tools (Synopsys Design Compiler, Cadence Genus) automatically select adder topologies based on timing constraints. However, in interviews the question is about demonstrating conceptual understanding of why these topologies exist, not about hand-instantiating specific cells.

---

### Q10. What are the power optimisation techniques for a MAC unit operating in a large array?

**Question:** A 256x256 MAC array consumes 50W at full utilisation. Describe three distinct power optimisation techniques that can be applied at the MAC unit level. Estimate the power reduction from each.

**Answer:**

At 256x256 = 65,536 MACs x ~750 uW per MAC at 1 GHz 7nm ≈ 49W. This is realistic.

**Technique 1: Clock gating at PE granularity**

When a PE's inputs are zero (a common case in sparse models or padding), disable the clock to the pipeline registers within that PE. This eliminates switching activity (dynamic power) without affecting the result.

Implementation: A pre-compute zero-detect circuit checks if A == 0 or B == 0. If either is zero, assert clock_enable = 0 to the PE's pipeline registers.

Power model: Dynamic power P_dyn = alpha * C * V^2 * f, where alpha is activity factor. If 30% of activations are zero (typical for ReLU-activated models), alpha is reduced by ~30%.

Estimated power reduction: **~25-30%** (accounting for zero-detect logic overhead of ~2-3% of PE power).

**Technique 2: Operand isolation (input gating)**

Prevent unnecessary switching on the multiplier inputs when the output will be discarded (e.g., padding cycles, fill cycles during systolic array pipeline drain). Insert AND gates on the A and B inputs, controlled by a valid signal.

Implementation: `a_gated = a_in & {8{valid}}`. When valid = 0, multiplier inputs are forced to zero, preventing any partial product generation and switching.

Power model: The multiplier is typically 60% of the MAC dynamic power. Gating for padding cycles (which can be 10-20% of cycles in workloads with awkward shapes) saves: 0.15 * 0.60 = 9% of total MAC power.

Estimated power reduction: **~8-10%** with minimal area impact (8 AND gates per PE per input).

**Technique 3: Data encoding (bit-flip minimisation)**

Use Gray code or bus-invert encoding on the activation shift buses between PEs. In a systolic array, activations flow horizontally through 256 PEs per row, switching with every clock cycle. Each wire transition contributes to dynamic power.

Bus-invert: Before driving a new value onto the bus, compare it to the previous value. If the Hamming distance is > N/2, invert the entire bus and set an invert flag bit. This guarantees at most N/2 transitions per cycle instead of up to N.

For a 256-column array with 8-bit activation buses: 256 buses x 8 bits = 2048 wires. At typical switching activity of 25%, bus-invert reduces transitions to at most 12.5% → ~50% reduction in bus switching power.

Wire power is typically 20-30% of total dynamic power for a 256-wide array.

Estimated power reduction: **~10-15%** in total dynamic power. Cost: one comparison and invert unit per bus row (small area).

**Combined estimate:**

If the three techniques are applied together with partial overlap:
- Baseline: 50W.
- After clock gating: ~37W (26% reduction).
- After operand isolation: ~33W (10% on top).
- After bus-invert: ~28-29W (12% on top).
- **Combined: approximately 40-45% total power reduction** for typical sparse workloads.

**Interview note:** Power estimates vary widely based on workload, process node, and implementation quality. The key skill being tested is the ability to reason about *where* power is consumed (switching activity, clock trees, buses, multiplier, adder) and propose targeted techniques for each source.

---
