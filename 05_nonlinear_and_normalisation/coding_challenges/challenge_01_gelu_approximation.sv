// =============================================================================
// Challenge 01: Pipelined GELU Approximation Unit
// =============================================================================
//
// Task
// ----
// Implement a hardware GELU activation function using piecewise linear (PWL)
// approximation over non-uniform segments, combined with direct saturation
// detection for the tails.
//
// Format
// ------
// Fixed-point Q4.12 (16-bit signed, 4 integer bits, 12 fractional bits).
//   - Range:        [-8.0, +7.999755859375]
//   - Resolution:   1/4096 ≈ 0.000244
//   - Representation: value = integer_bits / 4096
//
// GELU Reference
// --------------
// Exact:       GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// Tanh approx: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
//
// Approximation Strategy
// -----------------------
// 1. For x >= 4.0:  output = x          (GELU saturates to identity)
// 2. For x <= -4.0: output = 0          (GELU saturates to zero)
// 3. For x in (-4, +4): use 16-segment non-uniform PWL
//    Segment coefficients (slope m, offset b) stored in ROM.
//    output = clip(m * x + b, output_range)
//
// PWL coefficients are computed for the 16 non-uniform breakpoints listed below.
// All coefficients are in Q4.12 fixed-point.
//
// Pipeline
// --------
// 5 stages, fully pipelined (one output per cycle after latency):
//   Stage 1: Latch input, classify region (tail vs. active)
//   Stage 2: Decode segment index, read coefficients
//   Stage 3: Multiply: m * x
//   Stage 4: Add offset: m*x + b
//   Stage 5: Mux tail/active, register output
//
// Interface
// ---------
// Inputs:
//   clk     - clock
//   rst_n   - active-low synchronous reset
//   x_in    - Q4.12 signed input [15:0]
//   valid_i - input valid strobe
// Outputs:
//   y_out   - Q4.12 signed output [15:0]
//   valid_o - output valid strobe (valid_i delayed by 5 cycles)
//
// Verification
// ------------
// The included testbench sweeps every 16th Q4.12 input code (4096 codes over
// the full range) and compares against the tanh-form GELU, reporting max
// absolute error. With the fitted coefficients the max error over all codes
// is about 20 LSB (~0.005); the testbench threshold is 0.02 (82 LSB).
// =============================================================================

// ---------------------------------------------------------------------------
// PWL coefficient package
// ---------------------------------------------------------------------------
// Breakpoints (in Q4.12 integer representation, i.e., value * 4096):
//   -4.0  -3.0  -2.0  -1.5  -1.0  -0.75  -0.5  -0.25   0.0
//    0.25   0.5   0.75   1.0   1.5   2.0   3.0   (4.0)
// 16 segments, each [BP[i], BP[i+1]), plus the >= 4.0 identity region.
//
// Slopes and offsets are minimax fits of the tanh-form GELU over each
// segment (see SLOPE below). Values are Q4.12 signed integers.
// ---------------------------------------------------------------------------
package gelu_coeff_pkg;

    // Number of active (non-tail) PWL segments
    localparam int NUM_SEGS = 16;

    // Segment breakpoints in Q4.12 (signed 16-bit).
    // bp[i] is the lower boundary of segment i.
    // Segment i covers [bp[i], bp[i+1]).
    typedef logic signed [15:0] q4_12_t;

    // Breakpoints: -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
    //              0.0,  0.25,  0.5,  0.75,  1.0,  1.5,  2.0,  3.0
    // Encoded as Q4.12: value * 4096
    localparam logic signed [15:0] BP [0:NUM_SEGS-1] = '{
        16'sh_C000,  // -4.0   * 4096 = -16384
        16'sh_D000,  // -3.0   * 4096 = -12288
        16'sh_E000,  // -2.0   * 4096 = -8192
        16'sh_E800,  // -1.5   * 4096 = -6144
        16'sh_F000,  // -1.0   * 4096 = -4096
        16'sh_F400,  // -0.75  * 4096 = -3072
        16'sh_F800,  // -0.5   * 4096 = -2048
        16'sh_FC00,  // -0.25  * 4096 = -1024
        16'sh_0000,  //  0.0   * 4096 =  0
        16'sh_0400,  //  0.25  * 4096 =  1024
        16'sh_0800,  //  0.5   * 4096 =  2048
        16'sh_0C00,  //  0.75  * 4096 =  3072
        16'sh_1000,  //  1.0   * 4096 =  4096
        16'sh_1800,  //  1.5   * 4096 =  6144
        16'sh_2000,  //  2.0   * 4096 =  8192
        16'sh_3000   //  3.0   * 4096 =  12288
    };

    // Slopes m[i] for each segment, in Q4.12.
    // GELU(x) ≈ m[i]*x + b[i] for x in [BP[i], BP[i+1]).
    // Fitted offline as integer Q4.12 pairs minimising the maximum error
    // against the tanh-form GELU over every input code in the segment, using
    // a bit-exact model of this datapath (rounded m*x, then + b).
    // Note GELU is not monotonic: it dips to -0.17 near x = -0.75, so the
    // slopes on [-4, -0.75) are negative, and it overshoots slope 1 above x = 0.75.
    localparam logic signed [15:0] SLOPE [0:NUM_SEGS-1] = '{
        16'sh_FFF0, // seg  0: [-4,-3)       slope  = -0.0039
        16'sh_FF54, // seg  1: [-3,-2)       slope  = -0.0420
        16'sh_FE3A, // seg  2: [-2,-1.5)     slope  = -0.1108
        16'sh_FE19, // seg  3: [-1.5,-1)     slope  = -0.1189
        16'sh_FF3E, // seg  4: [-1,-0.75)    slope  = -0.0474
        16'sh_00FA, // seg  5: [-0.75,-0.5)  slope  = +0.0610
        16'sh_0365, // seg  6: [-0.5,-0.25)  slope  = +0.2122
        16'sh_0665, // seg  7: [-0.25,0)     slope  = +0.3997
        16'sh_098B, // seg  8: [0,0.25)      slope  = +0.5964
        16'sh_0C7E, // seg  9: [0.25,0.5)    slope  = +0.7808
        16'sh_0EF6, // seg 10: [0.5,0.75)    slope  = +0.9351
        16'sh_10B3, // seg 11: [0.75,1)      slope  = +1.0437
        16'sh_11D8, // seg 12: [1,1.5)       slope  = +1.1152
        16'sh_11C1, // seg 13: [1.5,2)       slope  = +1.1096
        16'sh_10A9, // seg 14: [2,3)         slope  = +1.0413
        16'sh_100E  // seg 15: [3,4)         slope  = +1.0034
    };

    // Offsets b[i] for each segment, in Q4.12.
    localparam logic signed [15:0] OFFSET [0:NUM_SEGS-1] = '{
        16'sh_FFC3, // seg  0: [-4,-3)       offset = -0.0149  (max err ~3 LSB)
        16'sh_FE01, // seg  1: [-3,-2)       offset = -0.1248  (max err ~20 LSB)
        16'sh_FBC0, // seg  2: [-2,-1.5)     offset = -0.2656  (max err ~6 LSB)
        16'sh_FB88, // seg  3: [-1.5,-1)     offset = -0.2793  (max err ~7 LSB)
        16'sh_FCB0, // seg  4: [-1,-0.75)    offset = -0.2070  (max err ~6 LSB)
        16'sh_FDFC, // seg  5: [-0.75,-0.5)  offset = -0.1260  (max err ~9 LSB)
        16'sh_FF32, // seg  6: [-0.5,-0.25)  offset = -0.0503  (max err ~12 LSB)
        16'sh_FFF3, // seg  7: [-0.25,0)     offset = -0.0032  (max err ~13 LSB)
        16'sh_FFF5, // seg  8: [0,0.25)      offset = -0.0027  (max err ~13 LSB)
        16'sh_FF3D, // seg  9: [0.25,0.5)    offset = -0.0476  (max err ~12 LSB)
        16'sh_FE06, // seg 10: [0.5,0.75)    offset = -0.1235  (max err ~9 LSB)
        16'sh_FCBD, // seg 11: [0.75,1)      offset = -0.2039  (max err ~6 LSB)
        16'sh_FB9A, // seg 12: [1,1.5)       offset = -0.2749  (max err ~7 LSB)
        16'sh_FBC9, // seg 13: [1.5,2)       offset = -0.2634  (max err ~6 LSB)
        16'sh_FE08, // seg 14: [2,3)         offset = -0.1230  (max err ~20 LSB)
        16'sh_FFCA  // seg 15: [3,4)         offset = -0.0132  (max err ~3 LSB)
    };

endpackage

// ---------------------------------------------------------------------------
// Segment decoder: maps Q4.12 input to a 4-bit segment index
// ---------------------------------------------------------------------------
// Compares input against the 16 breakpoints and returns the index of the
// highest breakpoint that is <= x.  Implemented as a priority encoder.
// ---------------------------------------------------------------------------
module gelu_seg_decoder
    import gelu_coeff_pkg::*;
(
    input  logic signed [15:0] x,       // Q4.12 input
    output logic        [3:0]  seg_idx  // segment index 0..15
);
    always_comb begin
        seg_idx = 4'd0;
        // Priority: last matching breakpoint wins
        for (int i = 0; i < NUM_SEGS; i++) begin
            if (x >= BP[i])
                seg_idx = 4'(i);
        end
    end
endmodule

// ---------------------------------------------------------------------------
// Main GELU approximation unit
// ---------------------------------------------------------------------------
module gelu_approximation
    import gelu_coeff_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [15:0] x_in,    // Q4.12 signed input
    input  logic        valid_i,
    output logic signed [15:0] y_out,   // Q4.12 signed output
    output logic        valid_o
);

    // -----------------------------------------------------------------------
    // Constants in Q4.12
    // -----------------------------------------------------------------------
    localparam logic signed [15:0] POS_SAT  = 16'sh4000;  //  +4.0 * 4096
    localparam logic signed [15:0] NEG_SAT  = 16'shC000;  //  -4.0 * 4096
    localparam logic signed [15:0] OUT_MAX  = 16'sh7FFF;  //  max Q4.12
    localparam logic signed [15:0] OUT_MIN  = 16'sh8000;  //  min Q4.12

    // -----------------------------------------------------------------------
    // Stage 1: Latch input, classify region, decode segment
    // -----------------------------------------------------------------------
    logic signed [15:0] s1_x;
    logic               s1_upper_tail;  // x >= +4.0 -> output = x
    logic               s1_lower_tail;  // x <= -4.0 -> output = 0
    logic        [3:0]  s1_seg;
    logic               s1_valid;

    logic [3:0] seg_idx_comb;  // combinational output from decoder

    gelu_seg_decoder u_decoder (
        .x       (x_in),
        .seg_idx (seg_idx_comb)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s1_x          <= '0;
            s1_upper_tail <= 1'b0;
            s1_lower_tail <= 1'b0;
            s1_seg        <= '0;
            s1_valid      <= 1'b0;
        end else begin
            s1_x          <= x_in;
            s1_upper_tail <= (x_in >= POS_SAT);
            s1_lower_tail <= (x_in <  NEG_SAT);
            s1_seg        <= seg_idx_comb;
            s1_valid      <= valid_i;
        end
    end

    // -----------------------------------------------------------------------
    // Stage 2: Read coefficients from combinational ROM
    // -----------------------------------------------------------------------
    logic signed [15:0] s2_x;
    logic signed [15:0] s2_slope;
    logic signed [15:0] s2_offset;
    logic               s2_upper_tail;
    logic               s2_lower_tail;
    logic               s2_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s2_x          <= '0;
            s2_slope      <= '0;
            s2_offset     <= '0;
            s2_upper_tail <= 1'b0;
            s2_lower_tail <= 1'b0;
            s2_valid      <= 1'b0;
        end else begin
            s2_x          <= s1_x;
            s2_slope      <= SLOPE [s1_seg];
            s2_offset     <= OFFSET[s1_seg];
            s2_upper_tail <= s1_upper_tail;
            s2_lower_tail <= s1_lower_tail;
            s2_valid      <= s1_valid;
        end
    end

    // -----------------------------------------------------------------------
    // Stage 3: Multiply m * x (Q4.12 * Q4.12 = Q8.24; keep upper 16 bits)
    // -----------------------------------------------------------------------
    // Full product is 32-bit signed.  We need to re-interpret as Q4.12:
    //   Q4.12 * Q4.12 = Q8.24, so the Q4.12 result sits in bits [27:12].
    // -----------------------------------------------------------------------
    logic signed [15:0] s3_mx;     // m * x in Q4.12
    logic signed [15:0] s3_offset;
    logic               s3_upper_tail;
    logic               s3_lower_tail;
    logic               s3_valid;
    logic signed [15:0] s3_x;

    logic signed [31:0] mul_full;   // full 32-bit product

    assign mul_full = s2_slope * s2_x;  // Q8.24

    // Round and extract Q4.12 result from bits [27:12] of the Q8.24 product.
    // Round-to-nearest: add 0.5 LSB at bit 11 before truncating.
    wire signed [31:0] mul_rounded = mul_full + 32'sh800;  // add 0.5 LSB
    wire signed [15:0] mx_q4_12    = mul_rounded[27:12];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s3_mx         <= '0;
            s3_offset     <= '0;
            s3_upper_tail <= 1'b0;
            s3_lower_tail <= 1'b0;
            s3_valid      <= 1'b0;
            s3_x          <= '0;
        end else begin
            s3_mx         <= mx_q4_12;
            s3_offset     <= s2_offset;
            s3_upper_tail <= s2_upper_tail;
            s3_lower_tail <= s2_lower_tail;
            s3_valid      <= s2_valid;
            s3_x          <= s2_x;
        end
    end

    // -----------------------------------------------------------------------
    // Stage 4: Add offset b: result = m*x + b
    // -----------------------------------------------------------------------
    // Both operands are Q4.12; sum is Q4.12 with potential 1-bit overflow.
    // Use 17-bit addition and saturate.
    // -----------------------------------------------------------------------
    logic signed [15:0] s4_pwl_out;
    logic               s4_upper_tail;
    logic               s4_lower_tail;
    logic               s4_valid;
    logic signed [15:0] s4_x;

    logic signed [16:0] sum_wide;  // 17-bit to catch overflow

    assign sum_wide = {s3_mx[15], s3_mx} + {s3_offset[15], s3_offset};

    // Saturate to Q4.12 representable range
    wire signed [15:0] pwl_saturated =
        (sum_wide > $signed(17'sh0_7FFF)) ? OUT_MAX :
        (sum_wide < $signed(17'sh1_8000)) ? OUT_MIN :
        sum_wide[15:0];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s4_pwl_out    <= '0;
            s4_upper_tail <= 1'b0;
            s4_lower_tail <= 1'b0;
            s4_valid      <= 1'b0;
            s4_x          <= '0;
        end else begin
            s4_pwl_out    <= pwl_saturated;
            s4_upper_tail <= s3_upper_tail;
            s4_lower_tail <= s3_lower_tail;
            s4_valid      <= s3_valid;
            s4_x          <= s3_x;
        end
    end

    // -----------------------------------------------------------------------
    // Stage 5: Select output based on region
    //   upper tail (x >= 4): pass x through (identity)
    //   lower tail (x <= -4): output 0
    //   active region: PWL approximation result
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            y_out   <= '0;
            valid_o <= 1'b0;
        end else begin
            valid_o <= s4_valid;
            if (s4_upper_tail)
                y_out <= s4_x;           // identity: GELU(x) ≈ x for x >> 0
            else if (s4_lower_tail)
                y_out <= 16'sh0000;      // zero: GELU(x) ≈ 0 for x << 0
            else
                y_out <= s4_pwl_out;     // piecewise linear approximation
        end
    end

endmodule


// =============================================================================
// Testbench
// =============================================================================
// Sweeps a representative set of Q4.12 input values, drives the DUT, and
// compares against a golden reference.
//
// Golden reference values for GELU are precomputed using the tanh approximation:
//   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// The testbench also reports the maximum absolute error across all tested inputs.
// =============================================================================

// ---------------------------------------------------------------------------
// Golden GELU function (combinational, uses tanh approximation in real math)
// Returns Q4.12 signed integer representing GELU(x_real).
// NOTE: This is a synthesis-excluded model only (uses $realtobits etc.)
// ---------------------------------------------------------------------------
module tb_gelu_golden (
    input  logic signed [15:0] x_in,
    output logic signed [15:0] gelu_golden
);
    // Convert Q4.12 to real, compute GELU, convert back
    real x_real, gelu_real;
    real tanh_arg, tanh_val;

    localparam real SQRT2OVERPI = 0.7978845608;
    localparam real COEFF       = 0.044715;

    always_comb begin
        x_real    = $itor($signed(x_in)) / 4096.0;
        tanh_arg  = SQRT2OVERPI * (x_real + COEFF * x_real * x_real * x_real);
        // tanh approximation: tanh(u) = (exp(2u)-1)/(exp(2u)+1)
        tanh_val  = (2.0 * tanh_arg > 40.0)  ?  1.0 :
                    (2.0 * tanh_arg < -40.0)  ? -1.0 :
                    ($exp(2.0 * tanh_arg) - 1.0) / ($exp(2.0 * tanh_arg) + 1.0);
        gelu_real = 0.5 * x_real * (1.0 + tanh_val);
        // Convert back to Q4.12 with rounding and saturation
        gelu_golden = (gelu_real >  7.999755) ? 16'sh7FFF :
                      (gelu_real < -8.0)       ? 16'sh8000 :
                      16'($rtoi(gelu_real * 4096.0 + 0.5));
    end
endmodule

module tb_gelu_approximation;

    // Absolute value of an integer. SystemVerilog has no $abs system function
    // (it is a simulator extension), so define one for portability.
    function automatic integer abs_int(integer x);
        return (x < 0) ? -x : x;
    endfunction


    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    logic        clk;
    logic        rst_n;
    logic signed [15:0] x_in;
    logic        valid_i;
    logic signed [15:0] y_out;
    logic        valid_o;

    // -----------------------------------------------------------------------
    // Clock generation: 10 ns period
    // -----------------------------------------------------------------------
    initial clk = 1'b0;
    always  #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // DUT instantiation
    // -----------------------------------------------------------------------
    gelu_approximation dut (
        .clk     (clk),
        .rst_n   (rst_n),
        .x_in    (x_in),
        .valid_i (valid_i),
        .y_out   (y_out),
        .valid_o (valid_o)
    );

    // -----------------------------------------------------------------------
    // Golden reference
    // -----------------------------------------------------------------------
    logic signed [15:0] golden_out;
    tb_gelu_golden u_golden (.x_in(x_in), .gelu_golden(golden_out));

    // -----------------------------------------------------------------------
    // Stimulus and checking
    // -----------------------------------------------------------------------
    // We feed inputs with a 5-cycle pipeline delay between input and output.
    // Use a shift register to align golden values with DUT outputs.
    // -----------------------------------------------------------------------
    localparam int PIPE_DEPTH = 5;

    logic signed [15:0] golden_pipe [0:PIPE_DEPTH-1];
    logic               valid_pipe  [0:PIPE_DEPTH-1];

    // Shift register for golden values (follows the pipeline)
    always_ff @(posedge clk) begin
        golden_pipe[0] <= golden_out;
        valid_pipe [0] <= valid_i;
        for (int i = 1; i < PIPE_DEPTH; i++) begin
            golden_pipe[i] <= golden_pipe[i-1];
            valid_pipe [i] <= valid_pipe [i-1];
        end
    end

    // -----------------------------------------------------------------------
    // Error tracking
    // -----------------------------------------------------------------------
    integer          max_abs_error;
    logic signed [15:0] worst_input;
    integer          num_checked;
    integer          num_errors;   // number of inputs with error > threshold
    localparam int   ERROR_THRESH = 82;  // ~0.02 in Q4.12 (0.02 * 4096 ≈ 82)

    // -----------------------------------------------------------------------
    // Test sequence
    // -----------------------------------------------------------------------
    integer i;
    integer abs_err;

    initial begin
        // Initialise
        rst_n       = 1'b0;
        x_in        = 16'sh0000;
        valid_i     = 1'b0;
        max_abs_error = 0;
        worst_input   = 16'sh0000;
        num_checked   = 0;
        num_errors    = 0;

        // Hold reset for 3 cycles
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // ----------------------------------------------------------------
        // Test 1: Spot-check key values
        // ----------------------------------------------------------------
        $display("--- Spot-check test ---");
        begin
            // x = 0.0 -> GELU(0) = 0
            valid_i = 1'b1;
            x_in    = 16'sh0000;
            @(posedge clk);
            // x = 1.0 -> GELU(1.0) ≈ 0.8413
            x_in = 16'sh1000;  // 4096 = 1.0 in Q4.12
            @(posedge clk);
            // x = -1.0 -> GELU(-1.0) ≈ -0.1587
            x_in = 16'shF000;  // -4096 = -1.0 in Q4.12
            @(posedge clk);
            // x = 2.0 -> GELU(2.0) ≈ 1.9545
            x_in = 16'sh2000;
            @(posedge clk);
            // x = -2.0 -> GELU(-2.0) ≈ -0.0455
            x_in = 16'shE000;
            @(posedge clk);
            valid_i = 1'b0;
            // Flush pipeline
            repeat (PIPE_DEPTH + 2) @(posedge clk);
        end

        // ----------------------------------------------------------------
        // Test 2: Sweep representative inputs
        // ----------------------------------------------------------------
        $display("--- Sweep test (step=16, covering full Q4.12 range) ---");
        @(posedge clk);

        fork
            // Driver: send inputs every cycle with step of 16 codes
            begin
                for (i = -32768; i <= 32767; i = i + 16) begin
                    valid_i = 1'b1;
                    x_in    = 16'(i);
                    @(posedge clk);
                end
                valid_i = 1'b0;
                // Extra cycles to flush pipeline
                repeat (PIPE_DEPTH + 2) @(posedge clk);
            end

            // Checker: compare DUT output vs. golden (pipeline-delayed)
            begin
                // Wait for first valid output
                while (!valid_o) @(posedge clk);

                // Check each output
                forever begin
                    @(posedge clk);
                    if (valid_o) begin
                        abs_err = abs_int($signed(y_out) - $signed(golden_pipe[PIPE_DEPTH-1]));
                        if (abs_err > max_abs_error) begin
                            max_abs_error = abs_err;
                            worst_input   = golden_pipe[PIPE_DEPTH-1];  // approx
                        end
                        if (abs_err > ERROR_THRESH) begin
                            num_errors++;
                            $display("  ERROR: x_in=%0d  dut=%0d  golden=%0d  err=%0d",
                                     $signed(x_in), $signed(y_out),
                                     $signed(golden_pipe[PIPE_DEPTH-1]), abs_err);
                        end
                        num_checked++;
                    end
                    // End checker when driver finishes
                    if (!valid_pipe[PIPE_DEPTH-1] && num_checked > 100) break;
                end
            end
        join

        // ----------------------------------------------------------------
        // Test 3: Tail saturation checks
        // ----------------------------------------------------------------
        $display("--- Tail saturation test ---");
        // Positive tail: x = 5.0 -> output should equal x (identity)
        valid_i = 1'b1;
        x_in    = 16'sh5000;  // 5.0 in Q4.12
        @(posedge clk);
        x_in    = 16'sh7000;  // 7.0
        @(posedge clk);
        // Negative tail: x = -6.0 -> output should be 0
        x_in    = 16'shA000;  // -6.0
        @(posedge clk);
        x_in    = 16'sh8000;  // min value
        @(posedge clk);
        valid_i = 1'b0;
        repeat (PIPE_DEPTH + 4) @(posedge clk);

        // Wait a few more cycles for checker to finish
        repeat (10) @(posedge clk);

        // ----------------------------------------------------------------
        // Report
        // ----------------------------------------------------------------
        $display("");
        $display("=== GELU Approximation Test Results ===");
        $display("  Inputs checked:     %0d", num_checked);
        $display("  Max abs error:      %0d LSBs (%.6f in real units)",
                 max_abs_error, real'(max_abs_error) / 4096.0);
        $display("  Error threshold:    %0d LSBs (%.4f)", ERROR_THRESH, 0.02);
        $display("  Inputs over thresh: %0d", num_errors);
        if (num_errors == 0)
            $display("  PASS: All errors within tolerance.");
        else
            $display("  FAIL: %0d inputs exceeded error threshold.", num_errors);
        $display("=======================================");

        $finish;
    end

    // -----------------------------------------------------------------------
    // Timeout watchdog
    // -----------------------------------------------------------------------
    initial begin
        #2_000_000;
        $display("TIMEOUT: Testbench exceeded time limit.");
        $finish;
    end

endmodule
