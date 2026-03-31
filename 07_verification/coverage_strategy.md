# Coverage Strategy for LLM Accelerator Verification

## Overview

Coverage is the quantitative measure of how thoroughly a design has been verified. For LLM
accelerators, a naive "line coverage" metric is insufficient — a design can have 100% line
coverage while leaving entire operational modes completely untested. This document covers the
full spectrum of coverage strategy, from basic metrics to formal verification integration.

---

## Tier 1 — Fundamentals

### Q1. What is the difference between code coverage and functional coverage, and why do you need both?

**Answer.**

**Code coverage** measures which lines, branches, and expressions of the RTL source code were
exercised during simulation. It is generated automatically by the simulator with no additional
verification effort.

- **Line/statement coverage:** was this line of RTL executed?
- **Branch coverage:** was each branch of `if/else` and `case` taken in both directions?
- **Toggle coverage:** did each signal transition from 0→1 and 1→0?
- **Expression coverage:** were all sub-expressions of a boolean evaluated to both true and false?

**Functional coverage** measures whether the design was exercised in each operationally
meaningful configuration. It requires the verification engineer to define what "meaningful"
means for this design, encoded in `covergroup` constructs.

- Did the design operate in all supported precision modes (INT8, BF16, FP16)?
- Were all supported matrix dimensions exercised?
- Was the pipeline exercised when back-pressure was asserted on the output side?
- Was the accumulator tested when exactly at the saturation boundary?

**Why you need both:**

Code coverage answers: "did the simulator execute all of the RTL logic?"
Functional coverage answers: "did the simulator operate the hardware in all intended modes?"

They are orthogonal:

- 100% code coverage with 20% functional coverage means every gate was toggled, but most
  operational modes were never tested.
- 100% functional coverage with 80% code coverage means all modes were exercised, but some
  RTL corner cases (unreachable dead code, or rarely-taken error paths) were not simulated.

The standard target in industry is 100% functional coverage and >95% code coverage (with a
documented waiver for each uncovered line that explains why it is unreachable or untestable
in simulation).

---

### Q2. Define three functional coverage points that are specific to an LLM accelerator and explain why each is important.

**Answer.**

**Coverpoint 1 — Precision mode × accumulator utilisation.**

```systemverilog
covergroup precision_utilisation_cg;
  cp_precision: coverpoint cfg.precision_mode {
    bins INT8  = {PRECISION_INT8};
    bins INT4  = {PRECISION_INT4};
    bins BF16  = {PRECISION_BF16};
    bins FP16  = {PRECISION_FP16};
    bins FP8   = {PRECISION_FP8};
  }
  cp_utilisation: coverpoint monitor.acc_utilisation_pct {
    bins low    = {[0:25]};
    bins medium = {[26:75]};
    bins high   = {[76:99]};
    bins full   = {100};
  }
  cx_prec_util: cross cp_precision, cp_utilisation;
endgroup
```

*Why it matters:* An INT8 MAD produces a 16-bit result, accumulated into 32 bits. At full
utilisation the accumulator approaches overflow. At low utilisation the multiplier array is
partially idle and gating logic is exercised. Each precision mode has different overflow
thresholds, so this cross is important.

**Coverpoint 2 — Output valid handshake under back-pressure.**

```systemverilog
covergroup output_backpressure_cg;
  cp_valid_ready: coverpoint {dut.output_valid, dut.output_ready} {
    bins valid_and_ready     = {2'b11};  // normal transfer
    bins valid_no_ready      = {2'b10};  // back-pressure: downstream stalling
    bins neither             = {2'b00};  // idle
    // 2'b01 (ready but not valid) is legal but less interesting
  }
endgroup
```

*Why it matters:* The output FIFO full condition (back-pressure) requires the compute pipeline
to stall and resume correctly. If never tested, stale-output or dropped-output bugs will only
appear in a system-level simulation or silicon.

**Coverpoint 3 — Quantisation zero-point modes.**

```systemverilog
covergroup zero_point_mode_cg;
  cp_zp_activation: coverpoint cfg.zero_point_activation {
    bins symmetric   = {0};
    bins asymmetric  = {[1:127]};
    bins max_asymm   = {128};
  }
  cp_zp_weight: coverpoint cfg.zero_point_weight {
    bins symmetric  = {0};
    bins nonzero    = {[1:127]};
  }
  cx_zp_modes: cross cp_zp_activation, cp_zp_weight;
endgroup
```

*Why it matters:* Asymmetric quantisation introduces a correction term in the GEMM computation
(`K × zp_a × sum_of_weights + K × zp_w × sum_of_activations`) that is often added as a bias
pre-loaded into the accumulator. RTL bugs in this correction path are only exposed when zero
points are non-zero.

---

### Q3. What are typical code coverage targets for RTL verification of an accelerator, and what is the process for closing coverage gaps?

**Answer.**

**Typical targets:**

| Coverage Type | Target | Notes |
|---------------|--------|-------|
| Line / statement | 100% | Waiver required for each gap |
| Branch | 100% | Waiver required for each gap |
| Toggle | 95–100% | Constants (tied-off bits) are excluded |
| Expression | 90–100% | Some complex boolean sub-expressions are impractical |
| Functional | 100% | All coverpoints must be closed before tape-out |

**Process for closing gaps:**

1. **Generate a coverage report** after each regression run. Most EDA simulators (Questa,
   VCS, Xcelium) output a database that can be merged across parallel runs and reported as
   an HTML or text summary.

2. **Categorise each gap.** Gaps fall into one of three categories:
   - *Reachable and not yet exercised:* add a directed test or tighten constraints.
   - *Reachable only in an error/fault injection scenario:* add a fault injection test.
   - *Structurally unreachable (dead code):* raise a waiver with design team confirmation.

3. **Directed tests for stubborn gaps.** If constrained random testing has not closed a bin
   after N regression runs, write a directed test that specifically targets the uncovered
   condition. Document why random generation was insufficient.

4. **Waivers for legitimate exclusions.** Dead code (e.g. unused parameter combinations,
   hardwired tie-offs, debug-only state machine states) must be formally excluded from the
   coverage target with a written justification reviewed and signed off by the verification
   lead.

5. **Track trends across commits.** Coverage should increase (or stay flat) with every new
   commit. A coverage regression (coverage decreasing after a code change) is a sign that
   new RTL was added without corresponding tests.

---

## Tier 2 — Intermediate

### Q4. How do you design cross-coverage between precision modes and data patterns for an LLM accelerator?

**Answer.**

Cross-coverage captures the simultaneous occurrence of two or more independent conditions.
For an LLM accelerator, precision mode and data pattern interact because:

- INT8 with all-maximum inputs approaches overflow at much shorter K than BF16.
- FP8-E4M3 with very small values produces denormals; BF16 with the same values does not.
- INT4 with alternating max/min creates the worst-case carry propagation in the adder tree.

**Comprehensive cross-coverage specification:**

```systemverilog
typedef enum logic [2:0] {
  INT8  = 3'd0,
  INT4  = 3'd1,
  BF16  = 3'd2,
  FP16  = 3'd3,
  FP8   = 3'd4
} precision_e;

typedef enum logic [3:0] {
  PAT_RANDOM    = 4'd0,   // uniformly random
  PAT_ALL_ZERO  = 4'd1,   // all elements 0
  PAT_MAX_POS   = 4'd2,   // all elements at max positive
  PAT_MIN_NEG   = 4'd3,   // all elements at min negative
  PAT_ALT_MAXMIN= 4'd4,   // alternating max, min, max, min...
  PAT_SPARSE    = 4'd5,   // ~90% zeros, ~10% random
  PAT_DENORMAL  = 4'd6,   // FP only: values near the denormal boundary
  PAT_NAN       = 4'd7,   // FP only: NaN inputs
  PAT_INF       = 4'd8    // FP only: Inf inputs
} data_pattern_e;

covergroup precision_pattern_cross_cg
    @(posedge clk iff (transaction_valid));

  cp_precision: coverpoint cfg_precision_mode {
    bins INT8  = {INT8};
    bins INT4  = {INT4};
    bins BF16  = {BF16};
    bins FP16  = {FP16};
    bins FP8   = {FP8};
  }

  cp_data_pattern: coverpoint stimulus_data_pattern {
    bins random     = {PAT_RANDOM};
    bins all_zero   = {PAT_ALL_ZERO};
    bins max_pos    = {PAT_MAX_POS};
    bins min_neg    = {PAT_MIN_NEG};
    bins alt_maxmin = {PAT_ALT_MAXMIN};
    bins sparse     = {PAT_SPARSE};
    bins denormal   = {PAT_DENORMAL};
    bins nan_input  = {PAT_NAN};
    bins inf_input  = {PAT_INF};
  }

  // Cross: every precision × every applicable data pattern
  // Illegal bins: denormal, NaN, Inf are not applicable to integer modes
  cx_prec_pattern: cross cp_precision, cp_data_pattern {
    // Exclude FP-specific patterns for integer precision modes
    illegal_bins int_modes_no_fp_patterns =
      binsof(cp_precision.INT8) && binsof(cp_data_pattern.denormal);
    illegal_bins int4_no_denormal =
      binsof(cp_precision.INT4) && binsof(cp_data_pattern.denormal);
    illegal_bins int8_no_nan =
      binsof(cp_precision.INT8) && binsof(cp_data_pattern.nan_input);
    illegal_bins int4_no_nan =
      binsof(cp_precision.INT4) && binsof(cp_data_pattern.nan_input);
    illegal_bins int8_no_inf =
      binsof(cp_precision.INT8) && binsof(cp_data_pattern.inf_input);
    illegal_bins int4_no_inf =
      binsof(cp_precision.INT4) && binsof(cp_data_pattern.inf_input);
  }

endgroup
```

**Closing the cross-coverage bins:**

Most cross-bins will be hit by random tests naturally. The hard cases are:

- `BF16 × PAT_DENORMAL` — requires generating inputs near 2^-133; unlikely by chance.
- `FP8 × PAT_NAN` — FP8 has only specific NaN encodings; random generation rarely produces them.
- `INT8 × PAT_ALT_MAXMIN` with K > 1024 — random K values rarely land at large values.

These require directed tests or constraint overrides targeting each unfilled bin.

---

### Q5. When is formal verification appropriate for an LLM accelerator, and what properties are most valuable to prove formally?

**Answer.**

Formal verification (property checking / model checking) exhaustively proves or disproves
properties over all possible input sequences, rather than sampled test vectors. It is most
valuable when:

1. **The property space is small but the input space is large.** Proving that the accumulator
   clear logic fires on every valid tiling boundary is tractable for a formal tool even though
   the number of possible input tensor shapes is huge.

2. **The bug would be extremely rare in simulation.** A race condition between two control
   signals that only occurs at a specific pipeline depth may never be hit in a trillion random
   cycles, but formal can find it in seconds.

3. **Safety or correctness is a strong requirement.** If a quantisation overflow silently
   corrupts the output with no error flag, this is unacceptable; formal can prove it never
   happens for all inputs.

**High-value formal properties for an LLM accelerator:**

**Property 1 — Accumulator never overflows INT32 for valid input ranges.**
```systemverilog
// For INT8 inputs in [-128, 127] and K ≤ MAX_K:
// max accumulator value = 127 * 127 * MAX_K = 16129 * MAX_K
// must be < 2^31 - 1 = 2,147,483,647
assert property (@(posedge clk)
  disable iff (!reset_n)
  (input_valid && (precision_mode == INT8)) |->
  ##[1:PIPE_LATENCY] ($signed(accumulator) <= INT32_MAX)
);
```

**Property 2 — No output valid without prior input valid (no phantom outputs).**
```systemverilog
assert property (@(posedge clk)
  disable iff (!reset_n)
  output_valid |-> $past(input_valid, PIPE_LATENCY)
);
```

**Property 3 — Output FIFO never overflows when output ready is asserted.**
```systemverilog
assert property (@(posedge clk)
  disable iff (!reset_n)
  output_fifo_full |-> !output_valid || output_ready
);
```

**Property 4 — After reset, accumulator is zero.**
```systemverilog
assert property (@(posedge clk)
  $rose(reset_n) |-> ##1 (accumulator == '0)
);
```

**Limitations of formal for LLM accelerators:**

- State space explosion: a 256-deep systolic array with 8-bit inputs has a state space of
  2^(256×8) which no formal tool can handle monolithically. The solution is to bound-check
  smaller sub-modules and abstract larger ones.
- Formal tools work best on control logic (FSMs, handshake protocols, FIFO management).
  They are less suited to verifying the numerical correctness of large datapath operations.

The industry best practice is to use formal for control and safety properties, and simulation
(with a golden model) for datapath numerical correctness.

---

### Q6. How do you define assertion-based verification (ABV) for a pipelined GEMM datapath?

**Answer.**

Assertion-based verification embeds `assert` and `assume` properties directly in the RTL or
bind them via a separate assertion module. For a pipelined GEMM datapath, assertions protect
the control invariants that, if violated, would produce incorrect numerical results.

**Key assertions for a pipelined GEMM:**

```systemverilog
// Assert module: bound to the GEMM datapath
module gemm_assertions
  #(parameter PIPE_DEPTH = 8,
    parameter MAX_K      = 4096)
(
  input logic        clk,
  input logic        reset_n,
  input logic        input_valid,
  input logic        input_ready,
  input logic        output_valid,
  input logic        output_ready,
  input logic        acc_clear,       // accumulator clear signal
  input logic        last_k,          // last K-tile indicator from controller
  input logic [31:0] acc_value,       // accumulator output (for overflow check)
  input logic [11:0] tile_k_count     // number of K-tiles processed
);

  // 1. Output must arrive exactly PIPE_DEPTH cycles after input
  property output_latency;
    @(posedge clk) disable iff (!reset_n)
    (input_valid && input_ready) |-> ##PIPE_DEPTH output_valid;
  endproperty
  assert property (output_latency)
    else $error("Output latency violation: expected %0d cycles", PIPE_DEPTH);

  // 2. Accumulator must be cleared after last tile
  property acc_clear_on_last_tile;
    @(posedge clk) disable iff (!reset_n)
    last_k |-> ##1 acc_clear;
  endproperty
  assert property (acc_clear_on_last_tile)
    else $error("Accumulator clear not asserted after last K-tile");

  // 3. No output valid when pipeline is empty (no inflight transactions)
  // Tracked via a counter of in-flight transactions
  logic [$clog2(PIPE_DEPTH+1)-1:0] inflight_count;
  always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n)
      inflight_count <= '0;
    else begin
      case ({input_valid & input_ready, output_valid & output_ready})
        2'b10: inflight_count <= inflight_count + 1;
        2'b01: inflight_count <= inflight_count - 1;
        default: inflight_count <= inflight_count;
      endcase
    end
  end

  assert property (@(posedge clk) disable iff (!reset_n)
    output_valid |-> (inflight_count > 0))
    else $error("Spurious output_valid with no inflight transaction");

  // 4. K-tile counter must not exceed maximum supported K
  assert property (@(posedge clk) disable iff (!reset_n)
    tile_k_count <= (MAX_K / 32))    // assuming 32-element K tiles
    else $error("K-tile count %0d exceeds maximum %0d", tile_k_count, MAX_K/32);

  // 5. No deadlock: if input valid and pipeline not full, ready must assert
  // (This is an assumption on the upstream, not an assertion on the DUT,
  //  but models the expected interface contract)
  assume property (@(posedge clk) disable iff (!reset_n)
    !$past(output_valid && !output_ready, PIPE_DEPTH) |-> input_ready)
    else $error("Deadlock: input_ready deasserted without back-pressure cause");

endmodule
```

**Integration in simulation:**

Assertions are active during simulation and generate immediate error messages when violated,
with a timestamp and signal dump, eliminating the need to manually correlate waves. They also
serve as the formal tool's specification — the same SVA properties used in simulation are the
formal proof obligations.

---

## Tier 3 — Advanced

### Q7. How do you construct a coverage closure plan for a full-chip LLM accelerator tape-out?

**Answer.**

A coverage closure plan is a formal document (usually maintained in a spreadsheet or a
coverage management tool) that records:

- Every covergroup and coverpoint defined in the testbench
- The simulation target for each bin (100% required vs. N% acceptable with justification)
- The current coverage status (from the latest regression run)
- The method to close each uncovered bin (random, directed, formal, waiver)
- The sign-off owner for each waiver

**Structure for an LLM accelerator:**

```
Coverage Plan — LLM Accelerator Compute Core
Last updated: [date of regression run]
Overall closure: [XX%]

Section 1: Input Interface
  1.1 AXI burst length distribution (all burst lengths 1..256)     → 98% closed
  1.2 AXI back-pressure scenarios (output stall while reading)     → 100% closed
  1.3 Address alignment (64B, 32B, unaligned)                      → 95% closed [GAP: unaligned >128B]

Section 2: Compute Core Configuration
  2.1 Precision mode (INT8, INT4, BF16, FP16, FP8)                 → 100% closed
  2.2 Matrix dimensions (all bins: small/medium/large/off-tile)     → 97% closed
  2.3 Accumulation depth K (1, power-of-2, max, off-tile)          → 90% closed [GAP: K=1]
  2.4 Activation function (none, ReLU, GELU)                       → 100% closed
  2.5 Bias add (enabled/disabled)                                   → 100% closed

Section 3: Datapath Stress
  3.1 Accumulator saturation (INT8: value at ±127/±128 boundary)   → 85% closed
  3.2 Zero-point modes (symmetric vs. asymmetric, all combos)      → 100% closed
  3.3 Cross: precision × data pattern                              → 92% closed [see detail]
      - GAP: BF16 × denormal inputs        → directed test planned
      - GAP: FP8 × NaN input               → formal proof + waiver
      - GAP: INT8 × K=1 × max value        → directed test added run-5

Section 4: Power Management
  4.1 Clock gating enable/disable per sub-block                    → 88% closed
  4.2 Power state transitions (active/idle/sleep)                  → 100% closed

Section 5: Error Handling
  5.1 AXI SLVERR response handling                                 → 100% closed
  5.2 Configuration register out-of-range values                   → 100% closed
  5.3 Watchdog timer expiry during long GEMM                       → waivered [untestable in RTL sim]

Waivers:
  W-001: FP8 NaN input: NaN propagation formally proven correct via SVA proof.
         No simulation test required. Sign-off: [verification lead]
  W-002: Watchdog timer: requires 10^6 cycle simulation, impractical.
         Timer logic isolated and verified by inspection + directed unit test.
```

**Key principle for tape-out readiness:** Every open bin must have either a committed
simulation run that will close it or a signed waiver. "We will try to close it before tape-out"
is not an acceptable status in a tape-out checklist.

---

### Q8. How would you use mutation coverage to assess the quality of your testbench's ability to detect RTL bugs?

**Answer.**

Mutation testing systematically introduces small, single-line changes (mutations) into the
RTL and measures what percentage of mutations cause at least one test to fail. A mutation that
is not caught by any test represents a potential bug class that the testbench cannot detect.

**Mutation operators for RTL:**

| Operator | Example change | Bug modelled |
|----------|---------------|-------------|
| Arithmetic operator swap | `+` → `-` | Wrong operation |
| Relational operator swap | `>=` → `>` | Off-by-one in comparator |
| Bit-shift amount change | `>>4` → `>>3` | Requantisation scale error |
| Conditional inversion | `if (acc_clear)` → `if (!acc_clear)` | Accumulator not cleared |
| Register enable removal | Remove `if (enable)` | Always-writing register |
| Off-by-one in loop bound | `K_TILES-1` → `K_TILES` | One extra accumulation tile |

**Running mutation testing in practice:**

Commercial tools (Cadence vManager Mutation, Mentor Questa Propcheck, Synopsys VC Formal)
automate this. Open-source options include `mutmut` (for Python golden models) and manual
mutation with script automation for RTL.

```python
# Example: mutation analysis script (Python, for the golden model)
# Identifies which test fails on which mutation

mutations = {
    "wrong_rounding": lambda x: np.floor(x),        # truncation instead of RHAFZ
    "skip_clamp":     lambda x: x,                   # no INT8 clamp applied
    "wrong_shift":    lambda x: x / 32.0,            # shift by 5 instead of 4
    "skip_zp":        lambda x, zp: x,               # ignore zero point
}

results = {}
for mut_name, mut_fn in mutations.items():
    failures = 0
    for stimulus, expected in test_suite:
        mutated_output = apply_mutation(mut_fn, stimulus)
        if not np.array_equal(mutated_output, expected):
            failures += 1
    results[mut_name] = failures / len(test_suite)
    print(f"Mutation '{mut_name}': {results[mut_name]*100:.1f}% of tests caught it")
```

**Interpreting results:**

- Mutation score = (caught mutations) / (total mutations) × 100%.
- A mutation score below 90% indicates the testbench has structural gaps.
- Mutations that are not caught should either generate new directed tests or be documented
  as equivalent mutations (semantically equivalent changes that do not affect correctness).

**For LLM accelerators specifically**, the highest-value mutations to test are:
- The accumulator clear signal (does the testbench catch "never cleared" or "cleared too early"?)
- The rounding mode in the requantisation stage (does changing RHAFZ to truncation fail tests?)
- The saturation clamp limits (does using 126 instead of 127 get caught?)

If any of these mutations produce a score below 100%, the testbench is missing coverage of
the most safety-critical numerical correctness paths.
