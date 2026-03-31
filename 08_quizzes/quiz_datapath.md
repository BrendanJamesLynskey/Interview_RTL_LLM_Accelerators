# Quiz: Datapath Design

## Overview

Self-assessment quiz covering systolic arrays, MAC units, accumulator design, mixed-precision
arithmetic, dataflow taxonomies (output-stationary, weight-stationary, row-stationary), multiplier
architectures, and pipeline stage design. Questions target RTL-level understanding relevant to
hardware design roles.

**Instructions:** Select the single best answer for each question. Answers and full explanations
appear at the bottom.

---

## Section 1 — Systolic Arrays (Questions 1-4)

**Q1.** In a weight-stationary (WS) systolic array, which data remains stationary in the
processing elements (PEs) throughout a tile computation?

- A) Input activations — they are pre-loaded and reused across multiple weight rows
- B) Partial sums — they accumulate inside each PE before being drained to an output buffer
- C) Weights — they are loaded into PEs once and activations are streamed through
- D) Both weights and activations — systolic arrays stall until both are pre-loaded

---

**Q2.** A 16x16 systolic array is used to compute a 16x16 matrix multiplication. How many cycles
does the systolic wavefront take to produce all output values (ignoring pipeline fill latency,
assuming one MAC per PE per cycle)?

- A) 16 cycles
- B) 31 cycles
- C) 256 cycles
- D) 512 cycles

---

**Q3.** In an output-stationary (OS) dataflow, which resource is conserved compared to
weight-stationary dataflow?

- A) On-chip SRAM reads of weight data
- B) On-chip register bandwidth for partial sum movement
- C) Input activation SRAM read bandwidth
- D) The number of multipliers required per PE

---

**Q4.** Systolic arrays are well-suited for which computational pattern, and why?

- A) Sparse matrix operations, because the regular data flow handles irregular access patterns gracefully
- B) Dense matrix-matrix multiplications, because data flows rhythmically through the array with high reuse
- C) Attention softmax, because the pipelined structure naturally implements the exponential function
- D) Memory-bound operations, because each PE contains a large local SRAM buffer

---

## Section 2 — MAC Units and Accumulator Design (Questions 5-7)

**Q5.** A MAC unit computes `acc = acc + a * b` where `a` and `b` are INT8 inputs. What is the
minimum bit-width required for the accumulator to guarantee no overflow when accumulating K terms?

- A) 8 bits
- B) 16 bits
- C) `8 + ceil(log2(K))` bits
- D) `16 + ceil(log2(K))` bits

---

**Q6.** In a dot product of two INT8 vectors of length N=4096, using a tree of adders to accumulate
INT16 products, the final accumulator must be at least how wide to prevent overflow in the worst
case?

- A) 16 bits
- B) 24 bits
- C) 28 bits
- D) 32 bits

---

**Q7.** A pipelined FP16 multiplier has a latency of 3 cycles. If 64 such multipliers are used in
parallel within a single cycle budget, and results must be accumulated before the next input
arrives, which microarchitectural technique allows the accumulation to absorb the multiplier
latency without stalling?

- A) Register renaming in the accumulator to eliminate write-after-read hazards
- B) A shift register (delay line) on the accumulator feedback path to match the multiplier latency
- C) Stalling the input pipeline for 3 cycles after each multiply
- D) Using a wider databus to transfer all 64 results in a single cycle

---

## Section 3 — Mixed Precision and Number Formats (Questions 8-10)

**Q8.** NVIDIA tensor cores support INT8 matrix multiply with INT32 accumulation. Why is the
accumulator wider than the inputs?

- A) INT32 provides backwards compatibility with FP32 code paths and is required by CUDA
- B) Multiplying two INT8 values produces up to a 16-bit product; summing many such products over a
   dot-product dimension can overflow 16 bits and requires 32-bit accumulators
- C) INT8 x INT8 arithmetic inherently loses 8 bits of precision, and INT32 recovers those bits
- D) NVIDIA chose INT32 arbitrarily; INT16 would have been sufficient for all practical purposes

---

**Q9.** When designing a BF16 multiplier compared to a FP16 multiplier, which component changes
most significantly?

- A) The exponent adder — BF16 has 8 exponent bits vs. FP16's 5, requiring a wider adder
- B) The mantissa multiplier — BF16 has 7 mantissa bits vs. FP16's 10, making the multiplier smaller
- C) The rounding logic — BF16 requires round-to-nearest-even while FP16 does not
- D) The sign bit logic — BF16 uses a two's complement sign representation

---

**Q10.** A designer proposes using posit arithmetic instead of IEEE 754 floating point for an
LLM accelerator. Which statement accurately describes a key trade-off of posit numbers?

- A) Posit arithmetic is simpler to implement in hardware than IEEE 754 because it eliminates NaN
   and infinity representations
- B) Posit numbers provide more precision near zero and less near large values compared to
   same-width IEEE 754, potentially reducing quantisation error, but the variable-length regime
   field makes hardware implementation more complex
- C) Posit numbers are byte-aligned with fixed exponent width, making memory layout identical to FP8
- D) Posit arithmetic achieves the same dynamic range as FP32 while using only 8 bits

---

## Section 4 — Dataflow Types (Questions 11-13)

**Q11.** In the Eyeriss dataflow paper, "row-stationary" (RS) dataflow is introduced. What is the
key insight of RS dataflow compared to WS or OS?

- A) RS dataflow eliminates the need for any DRAM access by fitting entire models in on-chip SRAM
- B) RS dataflow maps a single row of the filter convolution to each PE, allowing partial sums,
   weights, and activations all to be reused within the PE, minimising total data movement
- C) RS dataflow uses a row of output activations as the stationary element, equivalent to
   output-stationary but described differently
- D) RS dataflow requires no accumulator because each row produces exactly one scalar output

---

**Q12.** For a workload with very large weight matrices and small batch sizes (the LLM decode
regime), which dataflow minimises DRAM traffic for weights?

- A) Output-stationary, because outputs accumulate locally and weights are streamed
- B) Input-stationary, because inputs are small and can be broadcast while weights are reused
- C) Weight-stationary, because each weight is loaded once and activations are streamed through the array
- D) The dataflow choice has no impact on DRAM weight traffic; only tiling dimensions matter

---

**Q13.** A hardware team debates whether to implement weight-stationary or output-stationary
dataflow for an attention engine. The attention score matrix S = Q * K^T is computed where Q and K
both change every decode step. Which is the stronger argument?

- A) Weight-stationary is better because Q plays the role of "weights" and can be preloaded
- B) Output-stationary is better because the partial sums for each output score element can
   accumulate locally, and neither Q nor K is truly stationary across steps
- C) Weight-stationary is always better for matrix multiplies, regardless of operand characteristics
- D) The dataflow does not matter because attention is memory-bandwidth-bound in all configurations

---

## Section 5 — Multiplier Design and Pipeline Stages (Questions 14-17)

**Q14.** A Booth-encoded multiplier uses radix-4 recoding. Compared to a straightforward shift-and-add
multiplier for 8x8-bit multiplication, Booth encoding primarily reduces:

- A) The number of full adders in the final carry-propagate adder
- B) The number of partial products, halving the rows in the partial product reduction tree
- C) The width of the accumulator required after multiplication
- D) The number of pipeline registers needed to achieve timing closure

---

**Q15.** A designer implements a 16x16 integer multiplier using a Wallace tree. What is the
primary advantage of a Wallace tree over a simple ripple-carry addition of partial products?

- A) Wallace trees use fewer gates overall, reducing area
- B) Wallace trees reduce the critical path by performing carry-save addition in parallel stages,
   minimising gate levels before the final carry-propagate adder
- C) Wallace trees allow the result to be computed without a final carry-propagate adder
- D) Wallace trees handle signed multiplication natively without Booth encoding

---

**Q16.** In a pipelined datapath for a matrix multiply unit, the designer adds pipeline registers
between the multiplier array and the adder tree. What is the primary reason for this stage
boundary?

- A) Pipeline registers prevent metastability from propagating between multipliers and adders
- B) The stage boundary allows the multiplier and adder trees to operate at their own optimum
   clock frequency independently
- C) The pipeline register breaks the combinational path, allowing a higher clock frequency by
   reducing the critical path within each stage
- D) Pipeline registers convert carry-save format partial products into two's complement before
   addition

---

**Q17.** A multiply-accumulate unit is described by:

```
acc[n] = acc[n-1] + a[n] * b[n]
```

If the multiplier has a 4-cycle latency and outputs are available at cycle N+4, which RTL
technique most cleanly resolves the accumulation dependency without reducing throughput?

- A) Using a FIFO between the multiplier and accumulator to buffer 4 results, then adding them all at once
- B) Implementing 4 independent accumulators in a round-robin pattern, reducing the effective
   feedback path to 1 cycle, then combining at the end
- C) Reducing the multiplier latency to 1 cycle by removing pipeline stages, accepting lower clock frequency
- D) Using speculative execution to predict the accumulator value 4 cycles ahead

---

## Section 6 — Integration and RTL Questions (Questions 18-20)

**Q18.** In SystemVerilog RTL for a MAC array, you have:

```systemverilog
always_ff @(posedge clk) begin
    if (en) acc <= acc + (a * b);
end
```

`a` and `b` are 8-bit signed inputs. `acc` is 32-bit signed. Which statement about this code
is correct?

- A) This is correct as written; SystemVerilog automatically sign-extends 8-bit operands before
   multiplication
- B) This is incorrect because `a * b` produces a 16-bit result in SystemVerilog, and assigning
   to a 32-bit acc requires an explicit cast
- C) This is functionally correct for signed arithmetic because SystemVerilog sign-extends signed
   operands in expressions, but the multiply result is 16 bits and the implicit widening to 32
   bits before addition is implementation-defined in some tools
- D) The `en` signal causes a clock-gating inferred latch, which is a synthesis error

---

**Q19.** A designer instantiates a 256-PE systolic array in RTL and finds that post-synthesis
timing shows a critical path through the carry-propagate adder in each PE's 32-bit accumulator.
Which is the most effective RTL-level fix?

- A) Replace the 32-bit accumulator with two 16-bit accumulators operating in parallel
- B) Insert a pipeline register to split the accumulator addition into two stages (e.g., a
   carry-save stage followed by a carry-propagate stage)
- C) Reduce the input precision from INT8 to INT4 to shorten the multiplier critical path
- D) Increase the systolic array clock period to 2x to give the adder more time

---

**Q20.** In a weight-stationary systolic array for matrix multiplication, skewing is used for the
input activation feed. What does skewing refer to, and why is it needed?

- A) Skewing applies a delay of D cycles to the inputs of PE[row D], ensuring that the correct
   activation aligns in time with its corresponding weight as data flows through the array
- B) Skewing rotates the weight matrix by 45 degrees to reduce interconnect length between PEs
- C) Skewing staggers the pipeline enable signals to prevent power supply noise from all PEs
   switching simultaneously
- D) Skewing refers to the clock distribution strategy that compensates for clock skew across the
   array

---

## Answer Key

| Q  | Answer |
|----|--------|
| 1  | C      |
| 2  | B      |
| 3  | B      |
| 4  | B      |
| 5  | D      |
| 6  | C      |
| 7  | B      |
| 8  | B      |
| 9  | B      |
| 10 | B      |
| 11 | B      |
| 12 | C      |
| 13 | B      |
| 14 | B      |
| 15 | B      |
| 16 | C      |
| 17 | B      |
| 18 | C      |
| 19 | B      |
| 20 | A      |

---

## Detailed Explanations

### Q1 — Correct: C

Weight-stationary means weights are preloaded into PEs at the start of a tile and stay put.
Activations are fed in from the edge of the array and propagate through the PEs, picking up the
weight at each PE they visit.

- A incorrect: Stationary activations describe an input-stationary dataflow.
- B incorrect: Stationary partial sums describe output-stationary dataflow.
- D incorrect: Stalling until both are loaded is not a dataflow type; it would eliminate the
  pipelining benefit of systolic execution.

### Q2 — Correct: B

For an N x N systolic array computing an N x N matrix multiply, the wavefront requires N cycles to
fill the first output row and a further N-1 cycles for the wavefront to reach the last PE.
Total = N + (N-1) = 2N - 1 = 2(16) - 1 = **31 cycles**.

- A incorrect: 16 cycles would only cover the fill time without draining the far corner.
- C incorrect: 256 cycles would be correct for a fully serial approach with no pipelining.
- D incorrect: 512 cycles vastly overestimates; this number has no clear derivation from N=16.

### Q3 — Correct: B

In output-stationary dataflow, each partial sum accumulates inside a PE's register without being
moved over the interconnect. This minimises partial sum data movement. Weight-stationary dataflow,
by contrast, must route partial sums between PEs as activations flow through, consuming accumulator
bus bandwidth.

- A incorrect: Both OS and WS still read weight data from SRAM; OS does not specially reduce weight SRAM reads.
- C incorrect: Input activation reads are roughly equal across WS and OS; neither has a strong
  advantage here without knowing the specific tile dimensions.
- D incorrect: Dataflow type does not change the number of multipliers; it changes what data moves
  and what stays in place.

### Q4 — Correct: B

Systolic arrays are the canonical architecture for dense GEMM. Data flows rhythmically — one step
per cycle — in a regular pattern, and each weight is reused across many activation inputs, keeping
multiply-accumulate utilisation high.

- A incorrect: Sparse operations involve irregular access patterns (varying row lengths, arbitrary
  column indices) that do not fit the fixed-stride flow of a systolic array.
- C incorrect: Softmax requires an exponential function and a normalisation step; these are not
  naturally implemented by a multiply-accumulate array.
- D incorrect: Systolic arrays are compute-focused and typically have small per-PE storage. They are
  designed for high arithmetic throughput, not memory-bound operations.

### Q5 — Correct: D

Each product of two INT8 values is at most 127 x (-128) = -16256, which fits in 16 bits. After
accumulating K such products, the magnitude can reach 16256 x K. To represent this without
overflow requires ceil(log2(K)) additional bits beyond 16. The product of two 8-bit values is a
16-bit result, and accumulating K such products needs 16 + ceil(log2(K)) bits.

Correct answer is **D: 16 + ceil(log2(K))**.


- A incorrect: 8 bits cannot hold even a single product of two INT8 values in general (max product
  magnitude is 16384 which requires 15 bits signed).
- B incorrect: 16 bits holds one product but overflows when accumulating more than one term with
  maximum values.
- C incorrect: 8 + ceil(log2(K)) understates the required width by 8 bits, forgetting that the
  product itself is 16 bits wide.

### Q6 — Correct: C

INT8 x INT8 product fits in 16 bits. Accumulating 4096 such products: 2^ceil(log2(4096)) = 2^12 =
4096, so 12 additional bits are needed. Total = 16 + 12 = **28 bits**. The accumulator needs at
least 28 bits.

- A incorrect: 16 bits only holds one product and overflows immediately upon accumulation.
- B incorrect: 24 bits corresponds to log2(256) = 8 extra bits, implying N=256 not N=4096.
- D incorrect: 32 bits is safe and commonly used in hardware, but the question asks for the minimum
  required, which is 28 bits.

### Q7 — Correct: B

A delay line (shift register) of depth equal to the multiplier latency on the accumulator feedback
path ensures the partial sum from the previous cycle is available exactly when the current
multiplier result arrives, maintaining full throughput without stalls.

- A incorrect: Register renaming solves write-after-read data hazards in out-of-order processors,
  not pipeline latency mismatches in fixed-function arithmetic units.
- C incorrect: Stalling for 3 cycles per multiply would reduce throughput by 4x, which defeats the
  purpose.
- D incorrect: A wider databus addresses parallelism, not latency. The issue is not data transfer
  width but timing of availability.

### Q8 — Correct: B

INT8 inputs have values in [-128, 127]. Their product ranges up to ±16384, requiring 15 bits for
magnitude. Accumulating many such products over a dot-product length of N can produce values up to
N x 16384. For N=256, this requires ~23 bits; for practical dot-product lengths (1024-4096), INT32
is the safe choice.

- A incorrect: INT32 accumulation is chosen for correctness, not backwards compatibility. FP32
  compatibility is a separate consideration.
- C incorrect: INT8 x INT8 does not "lose" 8 bits — it is a valid product. The wider accumulator
  is for overflow prevention during summation.
- D incorrect: INT16 would overflow for dot products of significant length. Empirically, 4096-length
  dot products can produce values exceeding INT16 range.

### Q9 — Correct: B

BF16 has 1 sign bit, 8 exponent bits, and 7 mantissa bits. FP16 has 1 sign bit, 5 exponent bits,
and 10 mantissa bits. The mantissa multiplier size is dominated by the mantissa width. BF16's
7-bit mantissa multiplier (an 8x8 multiply of 1.mantissa) is substantially smaller than FP16's
11x11 multiply, reducing area and power.

- A incorrect: BF16 does have a wider exponent (8 vs. 5), so the exponent adder is wider — but the
  exponent adder is a small, fast adder and is not the "most significant" change in die area or
  critical path. The mantissa multiplier dominates.
- C incorrect: Both BF16 and FP16 use round-to-nearest-even as their default rounding mode per
  IEEE 754. This is not a differentiating factor.
- D incorrect: IEEE 754 floating-point formats use a sign-magnitude representation, not two's
  complement. Both BF16 and FP16 have the same 1-bit sign field.

### Q10 — Correct: B

Posit numbers use a variable-length "regime" field that shifts the representable range. Near zero
(small magnitude), they devote more bits to precision; near large values, precision decreases.
This matches the distribution of neural network weights. However, the variable-field-length
requires more complex decode/encode logic than fixed-format IEEE floats.

- A incorrect: While posit does eliminate NaN and infinity (replacing them with a single ±maxpos
  and NaR), the arithmetic is not simpler — the variable regime field makes addition and
  multiplication hardware more complex than IEEE 754.
- C incorrect: Posit numbers have a variable-length regime field, making them incompatible with
  fixed-layout formats like FP8.
- D incorrect: An 8-bit posit does have competitive dynamic range compared to FP8, but claiming
  parity with FP32 dynamic range in 8 bits overstates the capability.

### Q11 — Correct: B

The RS dataflow insight (from Chen et al., Eyeriss 2016) is that mapping one filter row to each
PE maximises reuse of all three data types — filter weights, input activations (shifted diagonally),
and partial sums — within the local PE, reducing data movement at every level of the memory
hierarchy.

- A incorrect: No practical accelerator fits entire LLMs in on-chip SRAM; this statement is false
  by orders of magnitude.
- C incorrect: RS dataflow is specifically distinct from OS; describing them as equivalent
  demonstrates a misunderstanding of the taxonomy.
- D incorrect: Convolution involves a sum over a filter window, producing a scalar per output
  position only when the filter has a single element; in general, the PE accumulates over the
  filter row and a partial sum does exist.

### Q12 — Correct: C

In weight-stationary dataflow, each weight is loaded into a PE once and activation data is
streamed through. For LLM decode, weights are enormous (billions of parameters) and the batch of
activations is tiny (often a single vector). Weight-stationary minimises the number of times each
weight byte is fetched from DRAM (ideally exactly once per inference step).

- A incorrect: Output-stationary holds partial sums in place; weights are streamed in from memory
  repeatedly for each output element, which is expensive when weights are large.
- B incorrect: "Input-stationary" is not a standard dataflow category in the same way; this
  description does not clearly reduce weight DRAM traffic.
- D incorrect: Dataflow choice absolutely affects DRAM traffic patterns. Weight-stationary
  specifically minimises weight fetches.

### Q13 — Correct: B

Since both Q and K change every decode step, neither qualifies as "stationary" in the WS sense.
Output-stationary allows the score accumulation (the dot product result) to remain local to a PE,
reducing partial sum movement. This is appropriate when both inputs are dynamic.

- A incorrect: Q does not stay constant across inference steps; it changes with each new token.
  WS requires the stationary operand to be reused across many operations.
- C incorrect: Dataflow choice depends on the access pattern of the operands. The claim that WS is
  always better for matrix multiplies regardless of operand characteristics is false.
- D incorrect: While attention can be memory-bound, this does not make the dataflow choice
  irrelevant — the dataflow determines how much weight traffic and partial sum traffic occurs,
  which matters even in memory-bound regimes.

### Q14 — Correct: B

Radix-4 Booth encoding recodes consecutive pairs of multiplier bits into a signed digit {-2, -1,
0, +1, +2}, reducing the number of partial products from N to N/2 for an N-bit multiplier. This
halves the rows in the partial product reduction tree, saving adder area and critical path.

- A incorrect: The final carry-propagate adder width is determined by the result width, not the
  number of partial products. Booth encoding does not change the final CPA.
- C incorrect: The accumulator width depends on the product width and accumulation depth, not
  whether Booth encoding is used.
- D incorrect: The number of pipeline registers is a microarchitectural choice independent of
  whether Booth encoding is used; it depends on timing closure requirements.

### Q15 — Correct: B

Wallace trees use carry-save adders (3-input, 2-output) in parallel to reduce all partial product
rows to two rows in O(log N) stages, before a single final carry-propagate adder. This minimises
the gate-level depth of the critical path.

- A incorrect: Wallace trees use more adder hardware than a serial approach because they add
  parallelism; area is not the primary benefit.
- C incorrect: A carry-propagate adder (CPA) is still required at the final stage to produce a
  single binary result. Wallace trees reduce intermediate rows but cannot eliminate the CPA.
- D incorrect: Handling signed multiplication requires Booth encoding or sign extension of partial
  products. A Wallace tree is an adder structure and is agnostic to sign handling.

### Q16 — Correct: C

A pipeline register breaks the combinational logic path. After insertion, each clock cycle only
needs to traverse one pipeline stage's logic (e.g., the multiplier array OR the adder tree, not
both). This allows the clock to run faster — the critical path is now the longer of the two
individual stage depths, which is shorter than their sum.

- A incorrect: Metastability is a synchroniser concern between asynchronous clock domains, not an
  intra-pipeline issue within a single synchronous design.
- B incorrect: In a synchronous design, all flip-flops share the same clock (or synchronised
  derived clocks). Having pipeline registers does not allow independent clock frequencies within
  a single pipeline.
- D incorrect: Carry-save format is a partial sum representation used within adder trees. A
  pipeline register does not inherently perform format conversion; that is an explicit design
  choice.

### Q17 — Correct: B

Four independent accumulators receive products in a round-robin assignment: acc0 gets cycle 0, 4,
8, ...; acc1 gets cycle 1, 5, 9, ...; and so on. Each individual accumulator only needs its result
4 cycles later, so the 4-cycle multiplier latency fits without any stall. At the end, acc0 through
acc3 are summed to produce the final result.

- A incorrect: A FIFO buffers results for later, but simply queuing results does not resolve the
  dependency — you still need to add them to the accumulator, which has the same latency issue.
- C incorrect: Removing pipeline stages reduces clock frequency (longer critical path), trading
  throughput for lower latency. This is often a worse engineering trade-off than option B.
- D incorrect: Speculative execution of the accumulator value is not practical for arithmetic
  circuits where the value is data-dependent and unpredictable.

### Q18 — Correct: C

In SystemVerilog, signed multiplication of two 8-bit signed values produces a 16-bit signed result
(the tool handles sign extension based on the signedness of operands). When this 16-bit result is
added to a 32-bit acc, the tool sign-extends the 16-bit value to 32 bits before adding. This is
functionally correct, but whether the intermediate result is exactly 16 bits or is promoted earlier
is technically tool-dependent and is a known source of subtle simulation-synthesis mismatches.

- A incorrect: While the functional result is often correct, saying SystemVerilog "automatically"
  handles this without caveats is imprecise; the promotion rules are defined but tool behaviour
  on expression width can vary.
- B incorrect: The assignment to a 32-bit acc does not require an explicit cast for correctness;
  sign extension happens implicitly. The concern is the intermediate product width, not the
  assignment.
- D incorrect: The `en` signal controls the flip-flop enable, not a latch. `always_ff` blocks are
  synthesised to flip-flops by definition, and a conditional assignment within `always_ff` with
  an enable is standard clock-gating inference, not an error.

### Q19 — Correct: B

The carry-propagate adder in the accumulator is on the critical path. Inserting a pipeline register
to split the addition into a carry-save stage (fast, carry-save output) followed by a
carry-propagate stage (in the next cycle) breaks the long chain. This is the standard "pipelined
adder" technique.

- A incorrect: Two 16-bit accumulators operating in parallel still require their results to be
  combined with a carry-propagate adder, which reproduces the original problem.
- C incorrect: Reducing input precision from INT8 to INT4 shortens the multiplier tree, not the
  accumulator adder. The critical path through a 32-bit carry-propagate adder is independent
  of the multiplier input precision.
- D incorrect: Doubling the clock period degrades the array's throughput (tokens/second) by 2x.
  This fixes timing but at an unacceptable performance cost.

### Q20 — Correct: A

In a systolic array, each PE sits at a unique (row, col) position. The activation for input row D
must arrive at PE[D] at the correct cycle to match the weight that has been preloaded there. Since
the activation is broadcast/fed from the left edge, it reaches PE[row=0] first. PE[row=1] would
see the activation one cycle later. Skewing deliberately delays the activation feed to row D by
D cycles, ensuring spatial and temporal alignment of data.

- B incorrect: The weight matrix is not physically rotated. Skewing is a temporal delay, not a
  geometric transformation.
- C incorrect: Power supply noise mitigation is a real design concern but is addressed through
  power grid design, decoupling capacitors, and clock gating — not input data skewing.
- D incorrect: Clock skew is a clocking concern (unequal clock arrival times at flip-flops). Input
  data skewing for systolic arrays is unrelated to clock distribution.
