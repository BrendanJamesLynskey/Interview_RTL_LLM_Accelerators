// =============================================================================
// Challenge 03 — Functional Coverage Plan for an LLM Accelerator Compute Core
// =============================================================================
//
// BACKGROUND
// ----------
// A coverage model documents WHAT must be observed in simulation to gain
// confidence that a design has been adequately exercised.  For an LLM
// accelerator the key axes of interest are:
//
//   - Matrix dimensions:   does the core handle small, large, and non-power-
//                          of-2 shapes that stress boundary logic?
//   - Precision modes:     does the datapath behave correctly across all
//                          supported numeric formats?
//   - Pipeline state:      do stall, flush, and bubble insertion work?
//   - Data patterns:       are special values (zeros, saturated, denormals)
//                          handled without silent corruption?
//   - Operation types:     is each operation independently verified?
//   - Cross-coverage:      do corner interactions get hit (e.g., INT4 GEMM
//                          with maximum-valued inputs)?
//
// HOW TO USE THIS FILE
// --------------------
// 1. Instantiate `llm_core_coverage` inside your UVM scoreboard or checker
//    component (or bind it to the DUT using `bind`).
// 2. Drive the interface signals from your UVM monitor.
// 3. Use `$get_coverage()` / your simulator's coverage merge flow to
//    report per-bin hit counts at the end of regression.
//
// INTERVIEW TASKS
// ---------------
//   1. Add a new covergroup for memory access patterns (sequential, strided,
//      broadcast) and explain why memory access patterns matter for a systolic
//      array versus a vector processor.
//   2. The cross-coverage bins will explode combinatorially for large enums.
//      Describe how you would use `binsof` and `intersect` to reduce the
//      cross-coverage space to the meaningful subset.
//   3. Write a constraint block in a UVM sequence that biases stimulus
//      towards uncovered bins (coverage-driven verification).
//   4. What is the difference between `illegal_bins` and `ignore_bins`?
//      Give an example where each is appropriate in this context.
//   5. How would you extend this coverage model to track temporal properties
//      (e.g., "stall must be de-asserted within 4 cycles of a flush")?
//
// =============================================================================

`timescale 1ns/1ps

// ---------------------------------------------------------------------------
// Enumerated types used across all covergroups
// ---------------------------------------------------------------------------

// Precision mode supported by the accelerator
typedef enum logic [2:0] {
    PREC_INT8  = 3'b000,
    PREC_INT4  = 3'b001,
    PREC_FP16  = 3'b010,
    PREC_BF16  = 3'b011,
    PREC_MIXED = 3'b100   // e.g., INT8 accum with INT4 weights
} precision_mode_e;

// Operation type dispatched to the compute core
typedef enum logic [2:0] {
    OP_GEMM       = 3'b000,  // General Matrix-Matrix multiply
    OP_GEMV       = 3'b001,  // General Matrix-Vector multiply
    OP_ATTENTION  = 3'b010,  // QK^T V attention kernel
    OP_SOFTMAX    = 3'b011,  // Row-wise softmax
    OP_NORM       = 3'b100   // RMSNorm / LayerNorm
} op_type_e;

// Pipeline state as reported by the DUT status signals
typedef enum logic [1:0] {
    PIPE_IDLE      = 2'b00,
    PIPE_COMPUTING = 2'b01,
    PIPE_STALLED   = 2'b10,
    PIPE_FLUSHING  = 2'b11
} pipe_state_e;

// Data pattern classification (set by the monitor, not driven by DUT)
typedef enum logic [2:0] {
    PAT_ALL_ZERO   = 3'b000,
    PAT_ALL_MAX    = 3'b001,  // saturated positive values
    PAT_ALL_MIN    = 3'b010,  // saturated negative values (or zero for uint)
    PAT_RANDOM     = 3'b011,
    PAT_DENORMAL   = 3'b100   // sub-normal FP values (only valid for FP modes)
} data_pattern_e;

// ---------------------------------------------------------------------------
// Interface carrying the signals sampled by the coverage model.
// In a real project this would be your testbench virtual interface.
// ---------------------------------------------------------------------------
interface llm_core_cov_if (input logic clk);
    logic         rst_n;

    // Dimension signals (M x K x N matrix problem size)
    logic [11:0]  dim_M;   // output rows    (0..4095)
    logic [11:0]  dim_K;   // inner dimension
    logic [11:0]  dim_N;   // output columns

    // Current precision mode
    precision_mode_e  precision;

    // Current operation
    op_type_e         op_type;

    // Current pipeline state
    pipe_state_e      pipe_state;

    // Data pattern identifiers (set externally by the UVM monitor)
    data_pattern_e    input_a_pattern;
    data_pattern_e    input_b_pattern;

    // Transaction valid strobe — sample coverage only on valid transactions
    logic             txn_valid;

    // Error / exception flags from DUT
    logic             overflow_flag;
    logic             underflow_flag;
    logic             nan_flag;        // only relevant for FP modes
endinterface

// ---------------------------------------------------------------------------
// Coverage collector module
//
// Bind this to the testbench or use it as an interface wrapper.
// All covergroups sample on the rising edge of clk when txn_valid is high.
// ---------------------------------------------------------------------------
module llm_core_coverage (
    llm_core_cov_if  cif
);

    // =======================================================================
    // CG 1 — Matrix Dimensions
    //
    // INTENT: Ensure the core is tested with matrices that stress:
    //   - Minimum size (1x1 edge case)
    //   - Small tensors (attention heads in LLMs are often 64 or 128 wide)
    //   - Medium tensors (typical FFN dimensions: 1024-4096)
    //   - Large tensors (max supported: 4096x4096)
    //   - Non-power-of-2 sizes that break naive tiling assumptions
    //
    // NOTES ON BINS:
    //   Bins are defined per dimension independently; cross-coverage between
    //   dimensions is captured in cg_dim_cross below.
    // =======================================================================
    covergroup cg_matrix_dimensions @(posedge cif.clk iff (cif.txn_valid && cif.rst_n));

        cp_dim_M: coverpoint cif.dim_M {
            bins dim_min        = {1};               // 1 row: edge case
            // One bin per size class (not dim_small[] / dim_non_pow2[]: an
            // array bin makes one bin per value, 252 bins that no regression
            // would ever fill)
            bins dim_small      = {[2:63]};           // typical small batch
            bins dim_head       = {64, 128};          // attention head sizes
            bins dim_medium     = {[256:1023]};       // mid-range
            bins dim_large      = {[1024:4095]};      // large FFN dims
            bins dim_max        = {4095};             // maximum
            // Non-power-of-2 sizes that expose tiling remainder logic
            bins dim_non_pow2   = {[65:127], [129:255]}; // odd sizes
            illegal_bins dim_zero = {0};             // zero dimension is illegal
        }

        cp_dim_K: coverpoint cif.dim_K {
            bins dim_min      = {1};
            bins dim_small    = {[2:63]};
            bins dim_head     = {64, 128};
            bins dim_medium   = {[256:1023]};
            bins dim_large    = {[1024:4095]};
            bins dim_max      = {4095};
            bins dim_non_pow2 = {[65:127], [129:255]};
            illegal_bins dim_zero = {0};
        }

        cp_dim_N: coverpoint cif.dim_N {
            bins dim_min      = {1};
            bins dim_small    = {[2:63]};
            bins dim_head     = {64, 128};
            bins dim_medium   = {[256:1023]};
            bins dim_large    = {[1024:4095]};
            bins dim_max      = {4095};
            bins dim_non_pow2 = {[65:127], [129:255]};
            illegal_bins dim_zero = {0};
        }

        // Cross: verify M and N hit each size category together.
        // Avoids the full 3-way explosion by crossing only M x N.
        cx_M_x_N: cross cp_dim_M, cp_dim_N {
            // Explicitly require the "square" cases
            bins square_small  = binsof(cp_dim_M.dim_head)   && binsof(cp_dim_N.dim_head);
            bins square_medium = binsof(cp_dim_M.dim_medium)  && binsof(cp_dim_N.dim_medium);
            bins square_large  = binsof(cp_dim_M.dim_large)   && binsof(cp_dim_N.dim_large);
            // Rectangular cases (tall and wide) stress output buffering
            bins tall_matrix   = binsof(cp_dim_M.dim_large)   && binsof(cp_dim_N.dim_small);
            bins wide_matrix   = binsof(cp_dim_M.dim_small)   && binsof(cp_dim_N.dim_large);
        }

    endgroup : cg_matrix_dimensions


    // =======================================================================
    // CG 2 — Precision Modes
    //
    // INTENT: Every supported numeric format must be exercised.  Additionally,
    // verify that transitions between modes (mode-switching) occur, which is
    // important for systems that reconfigure the datapath between layers.
    // =======================================================================
    covergroup cg_precision_modes @(posedge cif.clk iff (cif.txn_valid && cif.rst_n));

        cp_precision: coverpoint cif.precision {
            bins int8   = {PREC_INT8};
            bins int4   = {PREC_INT4};
            bins fp16   = {PREC_FP16};
            bins bf16   = {PREC_BF16};
            bins mixed  = {PREC_MIXED};
            // No illegal_bins here: all enum values are legal.
        }

    endgroup : cg_precision_modes

    // Precision mode transitions — sampled on every clock edge (not gated by
    // txn_valid) so that back-to-back reconfiguration is captured.
    covergroup cg_precision_transitions @(posedge cif.clk iff cif.rst_n);

        cp_prec_transition: coverpoint cif.precision {
            bins int8_to_fp16  = (PREC_INT8  => PREC_FP16);
            bins int8_to_bf16  = (PREC_INT8  => PREC_BF16);
            bins int4_to_int8  = (PREC_INT4  => PREC_INT8);
            bins fp16_to_bf16  = (PREC_FP16  => PREC_BF16);
            bins bf16_to_fp16  = (PREC_BF16  => PREC_FP16);
            bins int8_to_mixed = (PREC_INT8  => PREC_MIXED);
            bins any_to_int4   = (PREC_INT8, PREC_FP16, PREC_BF16, PREC_MIXED => PREC_INT4);
            // Wildcard: any->same (stays in same mode for multiple transactions)
            bins stay_same[]   = (PREC_INT8 => PREC_INT8),
                                 (PREC_FP16 => PREC_FP16),
                                 (PREC_BF16 => PREC_BF16);
        }

    endgroup : cg_precision_transitions


    // =======================================================================
    // CG 3 — Pipeline States
    //
    // INTENT: Every pipeline state must be reached, and all state transitions
    // must be observed at least once.  This ensures stall and flush control
    // logic has been exercised.
    //
    // Key transitions of interest:
    //   COMPUTING -> STALLED : downstream back-pressure
    //   STALLED   -> COMPUTING: back-pressure cleared
    //   COMPUTING -> FLUSHING : error or context switch
    //   FLUSHING  -> IDLE    : flush completion
    //   IDLE      -> COMPUTING: new operation dispatched
    // =======================================================================
    covergroup cg_pipeline_states @(posedge cif.clk iff cif.rst_n);

        cp_pipe_state: coverpoint cif.pipe_state {
            bins idle      = {PIPE_IDLE};
            bins computing = {PIPE_COMPUTING};
            bins stalled   = {PIPE_STALLED};
            bins flushing  = {PIPE_FLUSHING};
        }

        // State transition coverage
        cp_state_transitions: coverpoint cif.pipe_state {
            bins compute_to_stall  = (PIPE_COMPUTING => PIPE_STALLED);
            bins stall_to_compute  = (PIPE_STALLED   => PIPE_COMPUTING);
            bins compute_to_flush  = (PIPE_COMPUTING => PIPE_FLUSHING);
            bins flush_to_idle     = (PIPE_FLUSHING  => PIPE_IDLE);
            bins idle_to_compute   = (PIPE_IDLE      => PIPE_COMPUTING);
            bins stall_to_flush    = (PIPE_STALLED   => PIPE_FLUSHING);
            // Illegal: jump from IDLE directly to FLUSHING without computing
            illegal_bins idle_to_flush = (PIPE_IDLE => PIPE_FLUSHING);
        }

        // Sustained stall depth: observe 1, 2, 4, 8-16 consecutive stall cycles.
        // Implemented via a sequence of transitions. Transition repeat ranges
        // must be bounded ([*8:$] is not legal in a bin), so the deepest bin
        // uses an explicit upper limit.
        cp_stall_depth: coverpoint cif.pipe_state {
            bins stall_1_cycle  = (PIPE_COMPUTING => PIPE_STALLED => PIPE_COMPUTING);
            bins stall_2_cycles = (PIPE_COMPUTING => PIPE_STALLED[*2] => PIPE_COMPUTING);
            bins stall_4_cycles = (PIPE_COMPUTING => PIPE_STALLED[*4] => PIPE_COMPUTING);
            bins stall_8_cycles = (PIPE_COMPUTING => PIPE_STALLED[*8:16] => PIPE_COMPUTING);
        }

    endgroup : cg_pipeline_states


    // =======================================================================
    // CG 4 — Data Patterns
    //
    // INTENT: Stress the datapath with pathological numeric inputs.
    //
    //   all_zeros   : multiplier arrays must produce zero cleanly
    //   all_max     : verify saturation / overflow handling
    //   all_min     : verify negative saturation for signed types
    //   random      : general correctness (most directed tests use this)
    //   denormal    : FP denormals must flush-to-zero OR propagate correctly
    //
    // NOTES:
    //   PAT_DENORMAL is only meaningful for FP16 and BF16.  For integer
    //   modes it is an ignore_bin to avoid spurious coverage misses.
    //   The monitor classifies the data pattern by inspecting the input
    //   buffers before asserting txn_valid.
    // =======================================================================
    covergroup cg_data_patterns @(posedge cif.clk iff (cif.txn_valid && cif.rst_n));

        cp_input_a_pattern: coverpoint cif.input_a_pattern {
            bins all_zeros  = {PAT_ALL_ZERO};
            bins all_max    = {PAT_ALL_MAX};
            bins all_min    = {PAT_ALL_MIN};
            bins random     = {PAT_RANDOM};
            // Denormal is meaningless for integer modes, so only count it in
            // FP modes (rather than marking it illegal, which would flag
            // harmless stimuli). binsof() is only legal in cross bins, so the
            // condition goes in an iff guard on the bin itself.
            bins denormal   = {PAT_DENORMAL}
                              iff (!(cif.precision inside {PREC_INT8, PREC_INT4}));
        }

        cp_input_b_pattern: coverpoint cif.input_b_pattern {
            bins all_zeros  = {PAT_ALL_ZERO};
            bins all_max    = {PAT_ALL_MAX};
            bins all_min    = {PAT_ALL_MIN};
            bins random     = {PAT_RANDOM};
            bins denormal   = {PAT_DENORMAL}
                              iff (!(cif.precision inside {PREC_INT8, PREC_INT4}));
        }

        // Cross A x B patterns: corner combinations are the most interesting.
        // We constrain the cross to only the practically important pairs
        // rather than all 5x5=25 combinations: the named bins below, plus an
        // ignore_bins for every other pair (otherwise the remaining pairs
        // become automatic cross bins and still count towards coverage).
        cx_pattern_pair: cross cp_input_a_pattern, cp_input_b_pattern {
            // Zero x anything: output must be zero
            bins a_zero_b_any   = binsof(cp_input_a_pattern.all_zeros);
            bins b_zero_a_any   = binsof(cp_input_b_pattern.all_zeros);
            // Max x Max: must saturate without wrap-around
            bins both_max       = binsof(cp_input_a_pattern.all_max)  &&
                                  binsof(cp_input_b_pattern.all_max);
            // Max x Min: large magnitude result, check sign handling
            bins max_times_min  = binsof(cp_input_a_pattern.all_max)  &&
                                  binsof(cp_input_b_pattern.all_min);
            // Denormal x random: FP exception propagation
            bins denorm_x_rand  = binsof(cp_input_a_pattern.denormal) &&
                                  binsof(cp_input_b_pattern.random);
            // Random x random: general correctness
            bins both_random    = binsof(cp_input_a_pattern.random) &&
                                  binsof(cp_input_b_pattern.random);
            // Everything else. "!" only applies to a binsof() term, so the
            // complement of the bins above is written out by De Morgan.
            ignore_bins others  = !binsof(cp_input_a_pattern.all_zeros) &&
                                  !binsof(cp_input_b_pattern.all_zeros) &&
                                  (!binsof(cp_input_a_pattern.all_max)  || !binsof(cp_input_b_pattern.all_max)) &&
                                  (!binsof(cp_input_a_pattern.all_max)  || !binsof(cp_input_b_pattern.all_min)) &&
                                  (!binsof(cp_input_a_pattern.denormal) || !binsof(cp_input_b_pattern.random)) &&
                                  (!binsof(cp_input_a_pattern.random)   || !binsof(cp_input_b_pattern.random));
        }

        // Exception flags: verify they fire at least once
        cp_overflow:  coverpoint cif.overflow_flag  { bins seen = {1'b1}; }
        cp_underflow: coverpoint cif.underflow_flag { bins seen = {1'b1}; }
        cp_nan:       coverpoint cif.nan_flag {
            // Only a NaN in an FP mode counts. (A NaN in an integer mode is an
            // error: that is checked by a_no_int_nan below rather than an
            // "illegal_bins ... iff": illegal/ignore bins remove their values
            // from the other bins whatever the iff, and some simulators --
            // xsim 2025.2 -- ignore the iff, making every NaN illegal.)
            bins seen = {1'b1}
                        iff (!(cif.precision inside {PREC_INT8, PREC_INT4}));
        }

    endgroup : cg_data_patterns

    // NaN is illegal in integer modes
    a_no_int_nan: assert property (@(posedge cif.clk) disable iff (!cif.rst_n)
        (cif.txn_valid && cif.nan_flag) |-> !(cif.precision inside {PREC_INT8, PREC_INT4}))
        else $error("nan_flag asserted in an integer precision mode");


    // =======================================================================
    // CG 5 — Operation Types
    //
    // INTENT: Each of the five operation categories must be independently
    // exercised.  Beyond raw hit count we want to see each operation dispatch
    // from the IDLE state (cold start) and from COMPUTING (back-to-back ops).
    // =======================================================================
    covergroup cg_operation_types @(posedge cif.clk iff (cif.txn_valid && cif.rst_n));

        cp_op_type: coverpoint cif.op_type {
            bins gemm      = {OP_GEMM};
            bins gemv      = {OP_GEMV};
            bins attention = {OP_ATTENTION};
            bins softmax   = {OP_SOFTMAX};
            bins norm      = {OP_NORM};
        }

        // Back-to-back same operation: tests double-buffering / overlap
        cp_op_sequence: coverpoint cif.op_type {
            bins gemm_x2      = (OP_GEMM      => OP_GEMM);
            bins gemv_x2      = (OP_GEMV      => OP_GEMV);
            bins attention_x2 = (OP_ATTENTION => OP_ATTENTION);
            // Typical LLM layer pattern: GEMM -> GEMM -> SOFTMAX -> GEMM
            bins ffn_pattern  = (OP_GEMM => OP_GEMM => OP_NORM);
            bins attn_pattern = (OP_GEMM => OP_SOFTMAX => OP_GEMM);
        }

    endgroup : cg_operation_types


    // =======================================================================
    // CG 6 — Cross: Precision x Matrix Dimensions
    //
    // INTENT: Some precision modes interact with specific dimension sizes in
    // non-obvious ways.  For example, INT4 GEMM with dimension not a multiple
    // of 8 may require padding that is tricky to implement correctly.
    // =======================================================================
    covergroup cg_prec_x_dims @(posedge cif.clk iff (cif.txn_valid && cif.rst_n));

        // Classify K dimension (inner dimension drives the accumulation depth)
        cp_K_class: coverpoint cif.dim_K {
            bins k_tiny    = {[1:7]};     // less than one INT4 vector lane
            bins k_aligned = {[8:255]} with (item % 8 == 0);   // multiples of 8 (INT4 friendly)
            bins k_large   = {[256:4095]};
            bins k_unaligned = {[9:255]} with (item % 8 != 0); // non-8-multiple values
            illegal_bins k_zero = {0};
        }

        cp_prec_for_dim: coverpoint cif.precision {
            bins int8  = {PREC_INT8};
            bins int4  = {PREC_INT4};
            bins fp16  = {PREC_FP16};
            bins bf16  = {PREC_BF16};
            bins mixed = {PREC_MIXED};
        }

        cx_prec_K: cross cp_prec_for_dim, cp_K_class {
            // INT4 with misaligned K is the critical corner case
            bins int4_unaligned_K = binsof(cp_prec_for_dim.int4)  &&
                                    binsof(cp_K_class.k_unaligned);
            bins int4_tiny_K      = binsof(cp_prec_for_dim.int4)  &&
                                    binsof(cp_K_class.k_tiny);
            bins fp16_large_K     = binsof(cp_prec_for_dim.fp16)  &&
                                    binsof(cp_K_class.k_large);
            bins mixed_aligned_K  = binsof(cp_prec_for_dim.mixed) &&
                                    binsof(cp_K_class.k_aligned);
        }

    endgroup : cg_prec_x_dims


    // =======================================================================
    // CG 7 — Cross: Precision x Data Patterns
    //
    // INTENT: Verify that saturating arithmetic in INT8/INT4 is correct AND
    // that FP special values (NaN, Inf, denormal) are handled per spec.
    //
    // The most dangerous combination is max-valued INT4 inputs which can
    // overflow an INT8 accumulator if more than 256 accumulations occur.
    // =======================================================================
    covergroup cg_prec_x_data @(posedge cif.clk iff (cif.txn_valid && cif.rst_n));

        cp_prec_pd: coverpoint cif.precision {
            bins int8  = {PREC_INT8};
            bins int4  = {PREC_INT4};
            bins fp16  = {PREC_FP16};
            bins bf16  = {PREC_BF16};
            bins mixed = {PREC_MIXED};
        }

        cp_pat_pd: coverpoint cif.input_a_pattern {
            bins all_zeros = {PAT_ALL_ZERO};
            bins all_max   = {PAT_ALL_MAX};
            bins all_min   = {PAT_ALL_MIN};
            bins random    = {PAT_RANDOM};
            bins denormal  = {PAT_DENORMAL};
        }

        cx_prec_data: cross cp_prec_pd, cp_pat_pd {
            // Integer saturation corners
            bins int8_max   = binsof(cp_prec_pd.int8) && binsof(cp_pat_pd.all_max);
            bins int8_min   = binsof(cp_prec_pd.int8) && binsof(cp_pat_pd.all_min);
            bins int4_max   = binsof(cp_prec_pd.int4) && binsof(cp_pat_pd.all_max);
            bins int4_min   = binsof(cp_prec_pd.int4) && binsof(cp_pat_pd.all_min);
            // FP special-value corners
            bins fp16_denorm  = binsof(cp_prec_pd.fp16)  && binsof(cp_pat_pd.denormal);
            bins bf16_denorm  = binsof(cp_prec_pd.bf16)  && binsof(cp_pat_pd.denormal);
            bins mixed_max    = binsof(cp_prec_pd.mixed) && binsof(cp_pat_pd.all_max);
            // Ignore integer modes with denormal inputs (meaningless)
            ignore_bins int_denormal =
                (binsof(cp_prec_pd.int8) || binsof(cp_prec_pd.int4)) &&
                 binsof(cp_pat_pd.denormal);
        }

    endgroup : cg_prec_x_data


    // =======================================================================
    // CG 8 — Cross: Operation Type x Pipeline State
    //
    // INTENT: Verify that stall and flush handling is correct for every
    // operation.  A stall during softmax is different from a stall during
    // GEMM because softmax is not easily restartable mid-row.
    // =======================================================================
    covergroup cg_op_x_pipe @(posedge cif.clk iff cif.rst_n);

        cp_op_ps: coverpoint cif.op_type {
            bins gemm      = {OP_GEMM};
            bins gemv      = {OP_GEMV};
            bins attention = {OP_ATTENTION};
            bins softmax   = {OP_SOFTMAX};
            bins norm      = {OP_NORM};
        }

        cp_pipe_ps: coverpoint cif.pipe_state {
            bins idle      = {PIPE_IDLE};
            bins computing = {PIPE_COMPUTING};
            bins stalled   = {PIPE_STALLED};
            bins flushing  = {PIPE_FLUSHING};
        }

        cx_op_pipe: cross cp_op_ps, cp_pipe_ps {
            // Every operation must be observed in the COMPUTING state
            bins gemm_computing      = binsof(cp_op_ps.gemm)      && binsof(cp_pipe_ps.computing);
            bins gemv_computing      = binsof(cp_op_ps.gemv)      && binsof(cp_pipe_ps.computing);
            bins attention_computing = binsof(cp_op_ps.attention) && binsof(cp_pipe_ps.computing);
            bins softmax_computing   = binsof(cp_op_ps.softmax)   && binsof(cp_pipe_ps.computing);
            bins norm_computing      = binsof(cp_op_ps.norm)      && binsof(cp_pipe_ps.computing);

            // All operations must be stalled (back-pressure test)
            bins gemm_stalled        = binsof(cp_op_ps.gemm)      && binsof(cp_pipe_ps.stalled);
            bins attention_stalled   = binsof(cp_op_ps.attention) && binsof(cp_pipe_ps.stalled);
            bins softmax_stalled     = binsof(cp_op_ps.softmax)   && binsof(cp_pipe_ps.stalled);

            // All operations must be flushed (error-recovery test)
            bins gemm_flushing       = binsof(cp_op_ps.gemm)      && binsof(cp_pipe_ps.flushing);
            bins attention_flushing  = binsof(cp_op_ps.attention) && binsof(cp_pipe_ps.flushing);

            // Illegal: no operation should appear as "running" while IDLE
            // (the op_type signal should be don't-care / last valid value
            // while idle — this bin catches a monitor mis-sampling).
            // Use ignore_bins rather than illegal_bins here because the
            // op_type field may legally retain its previous value while idle.
            ignore_bins op_while_idle = binsof(cp_pipe_ps.idle);
        }

    endgroup : cg_op_x_pipe


    // =======================================================================
    // Covergroup instantiation
    // =======================================================================
    cg_matrix_dimensions   cg_dims_inst;
    cg_precision_modes     cg_prec_inst;
    cg_precision_transitions cg_prec_trans_inst;
    cg_pipeline_states     cg_pipe_inst;
    cg_data_patterns       cg_data_inst;
    cg_operation_types     cg_op_inst;
    cg_prec_x_dims         cg_prec_dims_inst;
    cg_prec_x_data         cg_prec_data_inst;
    cg_op_x_pipe           cg_op_pipe_inst;

    initial begin
        cg_dims_inst       = new();
        cg_prec_inst       = new();
        cg_prec_trans_inst = new();
        cg_pipe_inst       = new();
        cg_data_inst       = new();
        cg_op_inst         = new();
        cg_prec_dims_inst  = new();
        cg_prec_data_inst  = new();
        cg_op_pipe_inst    = new();
    end

    // =======================================================================
    // Coverage reporting task
    //
    // Call from a UVM final_phase or simulation end to print per-group
    // hit rates.  In a real flow your simulator's built-in report replaces
    // this, but having a human-readable summary task aids triage.
    // =======================================================================
    task automatic report_coverage();
        $display("========================================");
        $display("  LLM Accelerator Coverage Report");
        $display("========================================");
        $display("  cg_matrix_dimensions    : %.1f%%",
                 cg_dims_inst.get_coverage());
        $display("  cg_precision_modes      : %.1f%%",
                 cg_prec_inst.get_coverage());
        $display("  cg_precision_transitions: %.1f%%",
                 cg_prec_trans_inst.get_coverage());
        $display("  cg_pipeline_states      : %.1f%%",
                 cg_pipe_inst.get_coverage());
        $display("  cg_data_patterns        : %.1f%%",
                 cg_data_inst.get_coverage());
        $display("  cg_operation_types      : %.1f%%",
                 cg_op_inst.get_coverage());
        $display("  cg_prec_x_dims (cross)  : %.1f%%",
                 cg_prec_dims_inst.get_coverage());
        $display("  cg_prec_x_data (cross)  : %.1f%%",
                 cg_prec_data_inst.get_coverage());
        $display("  cg_op_x_pipe   (cross)  : %.1f%%",
                 cg_op_pipe_inst.get_coverage());
        $display("  ----------------------------------------");
        $display("  Aggregate coverage      : %.1f%%",
                 $get_coverage());
        $display("========================================");
    endtask

endmodule : llm_core_coverage


// =============================================================================
// Minimal testbench demonstrating how to drive the coverage interface
//
// In a production UVM environment, replace this with a proper UVM agent and
// monitor that connects to the real DUT interface.  This TB is purely for
// demonstrating coverage collection mechanics in a standalone simulation.
// =============================================================================
module coverage_plan_tb;

    logic clk;
    initial clk = 0;
    always #5 clk = ~clk;

    // Instantiate the coverage interface
    llm_core_cov_if cov_if (.clk(clk));

    // Instantiate the coverage collector
    llm_core_coverage cov_collect (.cif(cov_if));

    int cov_errors = 0;

    task automatic check_cov(input string name, input real got, input real exp);
        if (got < exp - 0.01 || got > exp + 0.01) begin
            $display("  MISMATCH %s: %.3f%%, expected %.3f%%", name, got, exp);
            cov_errors++;
        end
    endtask

    // -----------------------------------------------------------------------
    // Drive a variety of transactions to hit a representative sample of bins
    // -----------------------------------------------------------------------
    task automatic send_txn(
        input precision_mode_e  prec,
        input op_type_e         op,
        input pipe_state_e      ps,
        input data_pattern_e    pat_a,
        input data_pattern_e    pat_b,
        input logic [11:0]      m, k, n
    );
        @(negedge clk);
        cov_if.precision       = prec;
        cov_if.op_type         = op;
        cov_if.pipe_state      = ps;
        cov_if.input_a_pattern = pat_a;
        cov_if.input_b_pattern = pat_b;
        cov_if.dim_M           = m;
        cov_if.dim_K           = k;
        cov_if.dim_N           = n;
        cov_if.overflow_flag   = 1'b0;
        cov_if.underflow_flag  = 1'b0;
        cov_if.nan_flag        = 1'b0;
        cov_if.txn_valid       = 1'b1;
        // Hold for the posedge the covergroups sample on; drop at the next
        // negedge (clearing it right after the posedge would race the sampling)
        @(negedge clk);
        cov_if.txn_valid       = 1'b0;
    endtask

    initial begin
        // Reset
        cov_if.rst_n           = 1'b0;
        cov_if.txn_valid       = 1'b0;
        cov_if.precision       = PREC_INT8;
        cov_if.op_type         = OP_GEMM;
        cov_if.pipe_state      = PIPE_IDLE;
        cov_if.input_a_pattern = PAT_RANDOM;
        cov_if.input_b_pattern = PAT_RANDOM;
        cov_if.dim_M           = 12'd0;
        cov_if.dim_K           = 12'd0;
        cov_if.dim_N           = 12'd0;
        cov_if.overflow_flag   = 1'b0;
        cov_if.underflow_flag  = 1'b0;
        cov_if.nan_flag        = 1'b0;
        repeat(4) @(posedge clk);
        cov_if.rst_n = 1'b1;
        @(posedge clk);

        // ---- Precision mode sweep ----
        $display("\n[TB] Sweeping precision modes...");
        send_txn(PREC_INT8,  OP_GEMM, PIPE_COMPUTING, PAT_RANDOM,   PAT_RANDOM,   128, 128, 128);
        send_txn(PREC_INT4,  OP_GEMM, PIPE_COMPUTING, PAT_ALL_MAX,  PAT_ALL_MAX,   64,  64,  64);
        send_txn(PREC_FP16,  OP_GEMM, PIPE_COMPUTING, PAT_RANDOM,   PAT_RANDOM,   256, 256, 256);
        send_txn(PREC_BF16,  OP_GEMM, PIPE_COMPUTING, PAT_DENORMAL, PAT_RANDOM,   128, 128, 128);
        send_txn(PREC_MIXED, OP_GEMM, PIPE_COMPUTING, PAT_ALL_MAX,  PAT_RANDOM,   512, 512, 512);

        // ---- Operation type sweep ----
        $display("[TB] Sweeping operation types...");
        send_txn(PREC_FP16, OP_GEMM,      PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM, 64, 64, 64);
        send_txn(PREC_FP16, OP_GEMV,      PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM, 64, 64, 1);
        send_txn(PREC_BF16, OP_ATTENTION, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM, 128, 64, 128);
        send_txn(PREC_FP16, OP_SOFTMAX,   PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM, 1,  512, 1);
        send_txn(PREC_FP16, OP_NORM,      PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM, 1,  768, 1);

        // ---- Pipeline state sweep ----
        $display("[TB] Sweeping pipeline states...");
        send_txn(PREC_INT8, OP_GEMM, PIPE_IDLE,      PAT_RANDOM, PAT_RANDOM, 64, 64, 64);
        send_txn(PREC_INT8, OP_GEMM, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM, 64, 64, 64);
        send_txn(PREC_INT8, OP_GEMM, PIPE_STALLED,   PAT_RANDOM, PAT_RANDOM, 64, 64, 64);
        send_txn(PREC_INT8, OP_GEMM, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM, 64, 64, 64); // stall recovery
        send_txn(PREC_INT8, OP_GEMM, PIPE_FLUSHING,  PAT_RANDOM, PAT_RANDOM, 64, 64, 64);
        send_txn(PREC_INT8, OP_GEMM, PIPE_IDLE,      PAT_RANDOM, PAT_RANDOM,  1,  1,  1);

        // ---- Dimension sweep ----
        $display("[TB] Sweeping matrix dimensions...");
        send_txn(PREC_FP16, OP_GEMM, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM,    1,   1,   1); // min
        send_txn(PREC_FP16, OP_GEMM, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM,   64,  64,  64); // head
        send_txn(PREC_FP16, OP_GEMM, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM,  512, 512, 512); // medium
        send_txn(PREC_FP16, OP_GEMM, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM, 4095,4095,4095); // max
        send_txn(PREC_INT4, OP_GEMM, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM,   65,  65,  65); // non-pow2
        send_txn(PREC_INT8, OP_GEMM, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM, 2048,   7,  64); // tiny K
        send_txn(PREC_MIXED,OP_GEMM, PIPE_COMPUTING, PAT_RANDOM, PAT_RANDOM,   64, 100,  64); // K not a multiple of 8: must not count as aligned

        // ---- Data pattern corners ----
        $display("[TB] Sweeping data patterns...");
        send_txn(PREC_INT8, OP_GEMM, PIPE_COMPUTING, PAT_ALL_ZERO, PAT_ALL_MAX,  128, 128, 128);
        send_txn(PREC_INT8, OP_GEMM, PIPE_COMPUTING, PAT_ALL_MIN,  PAT_ALL_MAX,  128, 128, 128);
        // Trigger overflow flag
        @(negedge clk);
        cov_if.precision       = PREC_INT8;
        cov_if.op_type         = OP_GEMM;
        cov_if.pipe_state      = PIPE_COMPUTING;
        cov_if.input_a_pattern = PAT_ALL_MAX;
        cov_if.input_b_pattern = PAT_ALL_MAX;
        cov_if.dim_M           = 12'd256;
        cov_if.dim_K           = 12'd256;
        cov_if.dim_N           = 12'd256;
        cov_if.overflow_flag   = 1'b1; // synthetic overflow
        cov_if.txn_valid       = 1'b1;
        @(negedge clk);
        cov_if.txn_valid     = 1'b0;
        cov_if.overflow_flag = 1'b0;

        // FP denormal
        send_txn(PREC_FP16, OP_GEMM, PIPE_COMPUTING, PAT_DENORMAL, PAT_RANDOM, 64, 64, 64);

        repeat(4) @(posedge clk);

        // ---- Final coverage report ----
        cov_collect.report_coverage();

        // ---- Self-check ----
        // Expected coverage for exactly this stimulus, derived independently
        // of the simulator (a model of the sample stream -- one gated sample
        // per transaction, two ungated samples per transaction -- and of the
        // bins, auto cross bins and ignore bins above). A mismatch means the
        // model is not sampling or binning what this file says it does.
        check_cov("cg_matrix_dimensions",     cov_collect.cg_dims_inst.get_coverage(),       72.959);
        check_cov("cg_precision_modes",       cov_collect.cg_prec_inst.get_coverage(),      100.000);
        check_cov("cg_precision_transitions", cov_collect.cg_prec_trans_inst.get_coverage(), 90.000);
        check_cov("cg_pipeline_states",       cov_collect.cg_pipe_inst.get_coverage(),       69.444);
        check_cov("cg_data_patterns",         cov_collect.cg_data_inst.get_coverage(),       51.111);
        check_cov("cg_operation_types",       cov_collect.cg_op_inst.get_coverage(),         60.000);
        check_cov("cg_prec_x_dims",           cov_collect.cg_prec_dims_inst.get_coverage(),  85.000);
        check_cov("cg_prec_x_data",           cov_collect.cg_prec_data_inst.get_coverage(),  84.058);
        check_cov("cg_op_x_pipe",             cov_collect.cg_op_pipe_inst.get_coverage(),    82.222);

        if (cov_errors == 0) $display("\n=== Coverage plan TB complete: all coverage as expected ===\n");
        else                 $display("\n=== Coverage plan TB FAILED: %0d mismatch(es) ===\n", cov_errors);
        $finish;
    end

    // Timeout watchdog
    initial begin
        #100000;
        $display("ERROR: simulation timeout in coverage_plan_tb");
        $finish;
    end

    initial begin
        $dumpfile("coverage_plan.vcd");
        $dumpvars(0, coverage_plan_tb);
    end

endmodule : coverage_plan_tb
