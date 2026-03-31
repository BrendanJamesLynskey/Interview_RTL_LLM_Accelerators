// =============================================================================
// Challenge 03 — Exponential Function via LUT with Linear Interpolation
// =============================================================================
//
// BACKGROUND
// ----------
// The exponential function e^x appears everywhere in LLM inference:
//   - Softmax:          softmax(x)_i = e^{x_i} / sum_j(e^{x_j})
//   - GeLU activation:  GeLU(x) ≈ 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
//   - Sigmoid gate:     sigma(x) = 1 / (1 + e^{-x})
//
// A hardware LUT-based approximation provides predictable latency and
// area-efficient implementation compared to iterative or CORDIC methods.
//
// ALGORITHM
// ---------
//   Input:  x in fixed-point Q4.12 (signed), range approximately [-8, 8)
//   Output: e^x in fixed-point Q8.8 (unsigned), range [0, 255.996)
//
//   1. Decompose x into an integer-aligned LUT index and a fractional remainder:
//        idx  = x[IN_W-1 : IN_FRAC - LUT_ADDR_W]   (top address bits)
//        frac = x[IN_FRAC - LUT_ADDR_W - 1 : 0]     (sub-step fractional bits)
//
//   2. Look up two adjacent LUT entries: lut[idx] and lut[idx+1]
//
//   3. Linear interpolation:
//        out = lut[idx] + frac * (lut[idx+1] - lut[idx]) / STEP_SIZE
//
//   4. Clamp negative input results to zero (since e^x > 0 always, but
//      very negative inputs underflow to zero in fixed-point).
//
// FIXED-POINT CONVENTIONS
// -----------------------
//   IN_W    = 17   (1 sign + 4 integer + 12 fraction = Q4.12 signed)
//   IN_FRAC = 12
//   OUT_W   = 16   (8 integer + 8 fraction = Q8.8 unsigned)
//   OUT_FRAC = 8
//
//   LUT_DEPTH = 256  (8-bit index into the LUT; each entry covers 1/16 of a unit)
//   LUT_ADDR_W = 8
//
//   The LUT covers the range [LUT_X_MIN, LUT_X_MAX).
//   Inputs outside this range are clamped to the boundary values.
//
// LATENCY
// -------
//   The pipeline has 3 registered stages:
//     Cycle 1: Register input, extract index and fraction
//     Cycle 2: Read LUT (two reads); registered LUT outputs available
//     Cycle 3: Compute interpolation, produce output
//
// INTERVIEW TASKS
// ---------------
//   1. Populate the LUT contents.  The `generate` block initialises the ROM
//      using $rtoi() and the $exp() system function.  Explain what happens at
//      the LUT boundaries and how to handle overflow gracefully.
//   2. Replace the linear interpolation with quadratic (Hermite) interpolation
//      using the LUT derivative at each point.  What is the accuracy improvement?
//   3. Quantify the worst-case ULP error relative to IEEE 754 double precision.
//   4. Describe how you would split this into a range-reduction + small-LUT
//      design to support a wider input range (e.g., [-20, 20]) without
//      increasing LUT depth proportionally.
//   5. For softmax, you need exp() at very high throughput.  What architectural
//      changes enable one result per cycle (fully pipelined, no stalls)?
//
// =============================================================================

`timescale 1ns/1ps

// ---------------------------------------------------------------------------
// exp_lut_interpolation — pipelined e^x with LUT + linear interpolation
// ---------------------------------------------------------------------------
module exp_lut_interpolation #(
    // --- Input precision (signed Q4.12) ---
    parameter int unsigned IN_W      = 17,   // total input width (sign + int + frac)
    parameter int unsigned IN_FRAC   = 12,   // fractional bits of input

    // --- Output precision (unsigned Q8.8) ---
    parameter int unsigned OUT_W     = 16,   // total output width
    parameter int unsigned OUT_FRAC  = 8,    // fractional bits of output

    // --- LUT configuration ---
    // LUT_ADDR_W determines how many bits of the input address the LUT.
    // The LUT has 2^LUT_ADDR_W entries.  The remaining IN_FRAC-LUT_ADDR_W
    // bits form the interpolation fraction.
    parameter int unsigned LUT_ADDR_W = 8,   // 256-entry LUT
    parameter int unsigned LUT_DEPTH  = 256, // must equal 2^LUT_ADDR_W

    // Signed integer range covered by the LUT.
    // The LUT spans LUT_X_MIN to LUT_X_MAX in steps of STEP = range/LUT_DEPTH.
    parameter real LUT_X_MIN = -8.0,
    parameter real LUT_X_MAX =  8.0
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Input: signed fixed-point Q4.12
    input  logic signed [IN_W-1:0]  x_in,
    input  logic                    valid_in,

    // Output: unsigned fixed-point Q8.8 (e^x is always positive)
    output logic [OUT_W-1:0]        exp_out,
    output logic                    valid_out
);

    // -----------------------------------------------------------------------
    // Derived constants
    // -----------------------------------------------------------------------
    localparam real LUT_RANGE   = LUT_X_MAX - LUT_X_MIN;   // 16.0
    localparam real LUT_STEP    = LUT_RANGE / real'(LUT_DEPTH); // 16/256 = 0.0625
    // Number of sub-step fraction bits remaining after the LUT index
    localparam int unsigned FRAC_BITS = IN_FRAC - LUT_ADDR_W; // 12 - 8 = 4

    // Scale factor to convert LUT output to Q8.8
    localparam real OUT_SCALE   = 2.0 ** OUT_FRAC;          // 256.0

    // -----------------------------------------------------------------------
    // LUT ROM (synthesised as a register array; a real design uses BRAM)
    // -----------------------------------------------------------------------
    // Each entry stores e^(LUT_X_MIN + i * LUT_STEP) in Q8.8 unsigned.
    // We allocate LUT_DEPTH + 1 entries so that lut[idx+1] is always valid
    // for the last real index (idx = LUT_DEPTH-1 uses lut[LUT_DEPTH]).
    logic [OUT_W-1:0] lut_rom [0:LUT_DEPTH]; // +1 guard entry

    // Generate LUT contents at elaboration time
    generate
        for (genvar i = 0; i <= LUT_DEPTH; i++) begin : gen_lut
            // x value at this LUT entry
            localparam real x_val = LUT_X_MIN + real'(i) * LUT_STEP;
            // e^x in Q8.8; clamp to OUT_W max if it overflows
            localparam real exp_val_f = $exp(x_val) * OUT_SCALE;
            localparam real max_val_f = (2.0 ** OUT_W) - 1.0;
            // Use min() to clamp — $rtoi truncates towards zero
            localparam real clamped   = (exp_val_f < max_val_f) ? exp_val_f : max_val_f;
            initial lut_rom[i] = OUT_W'($rtoi(clamped));
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Stage 1 registers — decode input
    // -----------------------------------------------------------------------
    // We must map the signed Q4.12 input to an unsigned LUT index.
    // Step:
    //   1. Subtract LUT_X_MIN (add |LUT_X_MIN| since it is negative) scaled
    //      to the input's fixed-point representation.
    //   2. The resulting unsigned value's top LUT_ADDR_W bits are the index.
    //   3. The bottom FRAC_BITS bits are the interpolation fraction.
    //
    // LUT_X_MIN = -8.0 -> in Q4.12 = -8 * 4096 = -32768
    // After adding 32768 the minimum input maps to index 0.

    localparam int signed OFFSET_FP = $rtoi((-LUT_X_MIN) * (2.0 ** IN_FRAC));
    // OFFSET_FP = 8 * 4096 = 32768  (fits in 16 bits unsigned)

    // Clamp boundaries before indexing
    localparam int signed X_MIN_FP = $rtoi(LUT_X_MIN * (2.0 ** IN_FRAC)); // -32768
    localparam int signed X_MAX_FP = $rtoi((LUT_X_MAX - LUT_STEP) * (2.0 ** IN_FRAC)); // just below +8.0

    logic signed [IN_W-1:0]      s1_x_clamped;    // clamped input
    logic [LUT_ADDR_W-1:0]       s1_lut_idx;      // LUT base index
    logic [FRAC_BITS-1:0]        s1_frac;          // interpolation fraction
    logic                        s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_x_clamped <= '0;
            s1_lut_idx   <= '0;
            s1_frac      <= '0;
            s1_valid     <= 1'b0;
        end else begin
            s1_valid <= valid_in;
            if (valid_in) begin
                // Clamp to LUT range
                logic signed [IN_W-1:0] x_clamped;
                if      (x_in < IN_W'(X_MIN_FP)) x_clamped = IN_W'(X_MIN_FP);
                else if (x_in > IN_W'(X_MAX_FP)) x_clamped = IN_W'(X_MAX_FP);
                else                              x_clamped = x_in;

                // Shift to unsigned: x_unsigned = x_clamped + OFFSET_FP
                // OFFSET_FP is always positive so the result is unsigned.
                // Width: IN_W+1 to hold the addition without overflow.
                logic [IN_W:0] x_unsigned;
                x_unsigned = IN_W'(x_clamped) + IN_W'(OFFSET_FP);

                // Top LUT_ADDR_W bits of the fraction field are the index
                // The input fraction field occupies bits [IN_FRAC-1:0].
                // After adding OFFSET_FP the index sits at bits
                //   [IN_FRAC-1 + ceiling : IN_FRAC - LUT_ADDR_W]
                // For our parameters: index = bits [IN_FRAC-1 : FRAC_BITS]
                //                     frac  = bits [FRAC_BITS-1 : 0]
                s1_lut_idx   <= x_unsigned[IN_FRAC-1 -: LUT_ADDR_W];
                s1_frac      <= x_unsigned[FRAC_BITS-1:0];
                s1_x_clamped <= x_clamped;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Stage 2 registers — LUT read (two adjacent entries)
    // -----------------------------------------------------------------------
    logic [OUT_W-1:0]       s2_lut0;       // lut[idx]
    logic [OUT_W-1:0]       s2_lut1;       // lut[idx+1]
    logic [FRAC_BITS-1:0]   s2_frac;       // interpolation fraction (pipeline)
    logic                   s2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_lut0  <= '0;
            s2_lut1  <= '0;
            s2_frac  <= '0;
            s2_valid <= 1'b0;
        end else begin
            s2_valid <= s1_valid;
            if (s1_valid) begin
                s2_lut0 <= lut_rom[s1_lut_idx];
                s2_lut1 <= lut_rom[s1_lut_idx + 1]; // guard entry prevents OOB
                s2_frac <= s1_frac;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Stage 3 registers — linear interpolation
    //
    // interpolated = lut0 + frac * (lut1 - lut0) / STEP_SIZE
    //
    // frac    is FRAC_BITS wide, in units of 1/2^FRAC_BITS of a LUT step.
    // (lut1 - lut0) is signed, at most OUT_W+1 bits.
    // The product frac * delta needs FRAC_BITS + OUT_W + 1 bits before >> FRAC_BITS.
    //
    // After the shift the result is OUT_W bits wide and is added to lut0.
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            exp_out   <= '0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= s2_valid;
            if (s2_valid) begin
                // Compute delta = lut1 - lut0 (signed; lut1 >= lut0 for e^x)
                logic signed [OUT_W:0] delta;
                delta = $signed({1'b0, s2_lut1}) - $signed({1'b0, s2_lut0});

                // frac * delta: frac is unsigned FRAC_BITS; delta is signed OUT_W+1
                // Product is signed FRAC_BITS+OUT_W+1 bits
                logic signed [FRAC_BITS+OUT_W:0] interp_prod;
                interp_prod = $signed({1'b0, s2_frac}) * delta;

                // Divide by 2^FRAC_BITS (the interpolation denominator)
                logic signed [OUT_W:0] interp_term;
                interp_term = interp_prod >>> FRAC_BITS;

                // Add to base LUT entry and clamp to OUT_W unsigned
                logic signed [OUT_W+1:0] result_full;
                result_full = $signed({1'b0, s2_lut0}) + $signed({interp_term[OUT_W], interp_term});

                // Clamp: result should always be positive for e^x, but guard anyway
                if (result_full < 0)
                    exp_out <= '0;
                else if (result_full > $signed({{2{1'b0}}, {OUT_W{1'b1}}}))
                    exp_out <= {OUT_W{1'b1}};
                else
                    exp_out <= OUT_W'(result_full);
            end
        end
    end

endmodule


// =============================================================================
// Testbench — exp_lut_interpolation_tb
// =============================================================================
// Tests a sweep of input values and reports the absolute and relative error
// versus the ideal IEEE 754 double-precision exp().
//
// Expected accuracy: linear interpolation of a 256-entry LUT over [-8,8)
// gives a worst-case absolute error of roughly half the maximum slope
// times the step size: max_err ≈ 0.5 * e^8 * (16/256)^2 / 2 ≈ 4.1
// in real units.  In Q8.8 terms that is about 1050 LSBs near x=8.
// For small x the absolute error is much less (e^0=1, step error ~0.0002).
// The test uses a 2% relative tolerance which is generous for x near 0
// and tighter in absolute terms than the worst case at x=8.
// =============================================================================
module exp_lut_interpolation_tb;

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    localparam int unsigned IN_W       = 17;
    localparam int unsigned IN_FRAC    = 12;
    localparam int unsigned OUT_W      = 16;
    localparam int unsigned OUT_FRAC   = 8;
    localparam int unsigned LUT_ADDR_W = 8;
    localparam int unsigned LUT_DEPTH  = 256;
    localparam real         LUT_X_MIN  = -8.0;
    localparam real         LUT_X_MAX  =  8.0;

    localparam real IN_SCALE  = 2.0 ** IN_FRAC;   // 4096
    localparam real OUT_SCALE = 2.0 ** OUT_FRAC;  // 256

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    logic                     clk;
    logic                     rst_n;
    logic signed [IN_W-1:0]   x_in;
    logic                     valid_in;
    logic [OUT_W-1:0]         exp_out;
    logic                     valid_out;

    // -----------------------------------------------------------------------
    // DUT instantiation
    // -----------------------------------------------------------------------
    exp_lut_interpolation #(
        .IN_W      (IN_W),
        .IN_FRAC   (IN_FRAC),
        .OUT_W     (OUT_W),
        .OUT_FRAC  (OUT_FRAC),
        .LUT_ADDR_W(LUT_ADDR_W),
        .LUT_DEPTH (LUT_DEPTH),
        .LUT_X_MIN (LUT_X_MIN),
        .LUT_X_MAX (LUT_X_MAX)
    ) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .x_in     (x_in),
        .valid_in (valid_in),
        .exp_out  (exp_out),
        .valid_out(valid_out)
    );

    // -----------------------------------------------------------------------
    // Clock: 10 ns period
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Stimulus and checker
    // -----------------------------------------------------------------------
    // Test a set of evenly spaced x values including boundary cases.
    localparam int NUM_TESTS = 32;
    real   test_x_real [0:NUM_TESTS-1];
    real   expected_f  [0:NUM_TESTS-1];
    int    received_out[0:NUM_TESTS-1];
    int    send_ptr, recv_ptr, fail_count;

    // --- Error tracking ---
    real max_abs_err, max_rel_err;
    int  max_abs_err_idx, max_rel_err_idx;

    // Specific spot-check values for boundary / interesting points
    task automatic spot_check(real xval, string label);
        real ideal_f, got_f, abs_err, rel_err;
        logic signed [IN_W-1:0] x_fp;
        x_fp    = IN_W'($rtoi(xval * IN_SCALE));
        ideal_f = $exp(xval);
        @(negedge clk);
        x_in    = x_fp;
        valid_in = 1;
        @(negedge clk);
        valid_in = 0;
        // Wait 3 pipeline stages + margin
        repeat(5) @(posedge clk);
        @(posedge clk);
        if (valid_out) begin
            got_f   = real'(exp_out) / OUT_SCALE;
            abs_err = got_f - ideal_f; if (abs_err < 0.0) abs_err = -abs_err;
            rel_err = (ideal_f > 1e-6) ? abs_err / ideal_f : abs_err;
            $display("  %s: x=%6.3f  ideal=%8.4f  got=%8.4f  abs_err=%7.4f  rel_err=%5.2f%%",
                     label, xval, ideal_f, got_f, abs_err, rel_err*100.0);
        end
    endtask

    initial begin
        // Build sweep: -7.5 to +7.0 in steps of 0.5 (32 values)
        for (int i = 0; i < NUM_TESTS; i++) begin
            test_x_real[i] = LUT_X_MIN + 0.5 + real'(i) * 0.5;
            if (test_x_real[i] > LUT_X_MAX - 0.01)
                test_x_real[i] = LUT_X_MAX - 0.01;
            expected_f[i]  = $exp(test_x_real[i]);
        end

        // Reset
        rst_n     = 0;
        valid_in  = 0;
        x_in      = '0;
        send_ptr  = 0;
        recv_ptr  = 0;
        fail_count = 0;
        max_abs_err = 0.0;
        max_rel_err = 0.0;
        max_abs_err_idx = -1;
        max_rel_err_idx = -1;
        repeat(4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // --- Spot checks at notable points ---
        $display("\n--- Spot checks ---");
        spot_check( 0.0,  "x=0.0   ");
        spot_check( 1.0,  "x=1.0   ");
        spot_check(-1.0,  "x=-1.0  ");
        spot_check( 2.0,  "x=2.0   ");
        spot_check(-2.0,  "x=-2.0  ");
        spot_check( 0.5,  "x=0.5   ");
        spot_check(-0.5,  "x=-0.5  ");
        spot_check( 6.0,  "x=6.0   ");
        spot_check(-7.0,  "x=-7.0  ");
        spot_check( 7.9,  "x=7.9   ");

        // --- Streaming throughput test ---
        $display("\n--- Streaming sweep test (%0d values) ---", NUM_TESTS);
        fork
            // Sender
            begin
                for (int i = 0; i < NUM_TESTS; i++) begin
                    @(negedge clk);
                    x_in     = IN_W'($rtoi(test_x_real[i] * IN_SCALE));
                    valid_in = 1;
                    send_ptr = i;
                end
                @(negedge clk);
                valid_in = 0;
            end

            // Receiver
            begin
                // Wait for first valid output (pipeline latency = 3 cycles)
                for (int timeout = 0; timeout < 200; timeout++) begin
                    @(posedge clk);
                    if (valid_out && recv_ptr < NUM_TESTS) begin
                        received_out[recv_ptr] = int'(exp_out);
                        recv_ptr++;
                        if (recv_ptr == NUM_TESTS) break;
                    end
                end
            end
        join

        // Wait any remaining outputs
        for (int timeout = 0; timeout < 20 && recv_ptr < NUM_TESTS; timeout++) begin
            @(posedge clk);
            if (valid_out && recv_ptr < NUM_TESTS) begin
                received_out[recv_ptr] = int'(exp_out);
                recv_ptr++;
            end
        end

        // Evaluate accuracy
        $display("\n--- Accuracy report ---");
        $display("  %6s  %8s  %8s  %8s  %7s  %7s",
                 "x", "ideal", "ideal_q", "got_q", "abs_err", "rel_err%");
        for (int i = 0; i < recv_ptr; i++) begin
            real ideal_q, got_q, abs_err, rel_err;
            ideal_q = expected_f[i] * OUT_SCALE;  // ideal in Q8.8 integer units
            got_q   = real'(received_out[i]);
            abs_err = got_q - ideal_q; if (abs_err < 0.0) abs_err = -abs_err;
            rel_err = (ideal_q > 0.5) ? abs_err / ideal_q : abs_err;
            $display("  %6.2f  %8.4f  %8.0f  %8d  %7.1f  %6.2f%%",
                     test_x_real[i], expected_f[i], ideal_q,
                     received_out[i], abs_err, rel_err*100.0);
            if (abs_err > max_abs_err) begin
                max_abs_err     = abs_err;
                max_abs_err_idx = i;
            end
            if (rel_err > max_rel_err) begin
                max_rel_err     = rel_err;
                max_rel_err_idx = i;
            end
            // Fail if relative error > 5% (very generous for a 256-entry LUT)
            if (rel_err > 0.05 && expected_f[i] > 0.01) begin
                $display("    *** FAIL: rel_err exceeds 5%% threshold ***");
                fail_count++;
            end
        end

        $display("\n--- Summary ---");
        if (max_abs_err_idx >= 0)
            $display("  Max absolute error: %.1f LSBs at x=%.2f",
                     max_abs_err, test_x_real[max_abs_err_idx]);
        if (max_rel_err_idx >= 0)
            $display("  Max relative error: %.2f%% at x=%.2f",
                     max_rel_err*100.0, test_x_real[max_rel_err_idx]);

        if (recv_ptr < NUM_TESTS)
            $display("  WARNING: only %0d/%0d outputs received", recv_ptr, NUM_TESTS);

        if (fail_count == 0)
            $display("\n=== ALL TESTS PASSED ===\n");
        else
            $display("\n=== %0d TESTS FAILED ===\n", fail_count);

        $finish;
    end

    // -----------------------------------------------------------------------
    // Timeout watchdog
    // -----------------------------------------------------------------------
    initial begin
        #100000;
        $display("ERROR: simulation timeout");
        $finish;
    end

    // -----------------------------------------------------------------------
    // Waveform dump
    // -----------------------------------------------------------------------
    initial begin
        $dumpfile("exp_lut_interpolation.vcd");
        $dumpvars(0, exp_lut_interpolation_tb);
    end

endmodule
