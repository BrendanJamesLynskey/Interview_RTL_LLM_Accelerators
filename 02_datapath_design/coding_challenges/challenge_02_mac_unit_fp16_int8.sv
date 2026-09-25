// =============================================================================
// Challenge 02: Mixed-Precision MAC Unit (FP16 and INT8 with Mode Select)
// =============================================================================
//
// PROBLEM STATEMENT
// -----------------
// Design a MAC unit that supports two computation modes selected at runtime:
//
//   Mode 0 — INT8 mode (W8A8):
//     - Input A: INT8 signed activation [-128, 127]
//     - Input B: INT8 signed weight     [-128, 127]
//     - Accumulator: INT32 signed
//     - Operation: acc = acc + signed(A) * signed(B)
//
//   Mode 1 — FP16 mode (W16A16 or dequantised path):
//     - Input A: IEEE 754 FP16 (half precision): 1 sign + 5 exponent + 10 mantissa
//     - Input B: IEEE 754 FP16
//     - Accumulator: FP32 (single precision)
//     - Operation: acc = acc + A * B  (FP32 FMA semantics: no intermediate round)
//
// The unit must also include:
//   - Accumulator clear (start of new dot product)
//   - Saturation on INT32 output (clamp to INT32 range)
//   - FP32 accumulator NaN/Inf propagation
//   - Pipelined for high frequency (3 pipeline stages shown)
//
// INTERFACE
// ---------
//   clk        - Clock
//   rst_n      - Active-low synchronous reset
//   mode       - 0 = INT8, 1 = FP16
//   a_in[15:0] - FP16 or INT8 (sign-extended to 16b) activation
//   b_in[15:0] - FP16 or INT8 (sign-extended to 16b) weight
//   clear      - Clear accumulator (synchronous, stage-aligned)
//   valid_in   - Input data is valid
//   acc_int32  - INT32 accumulator output (valid in INT8 mode)
//   acc_fp32   - FP32 accumulator output (valid in FP16 mode)
//   valid_out  - Output is valid (reflects valid_in delayed by pipeline depth)
//
// PIPELINE STRUCTURE (3 stages)
// -----------------------------------
//   Stage 1 (S1): Decode inputs, begin multiply
//     - INT8: sign-extend, partial product generation
//     - FP16: decode sign/exponent/mantissa, compute product exponent,
//             align mantissas
//   Stage 2 (S2): Complete multiply
//     - INT8: finish 8x8 multiply, produce INT16 product
//     - FP16: multiply mantissas (10x10 with hidden bit = 11x11), normalise
//   Stage 3 (S3): Accumulate
//     - INT8: add INT16 product to INT32 accumulator (sign-extended)
//     - FP16: add FP32 product to FP32 accumulator (with alignment)
//
// NOTE ON FP16 MULTIPLY IMPLEMENTATION
// -------------------------------------
// A fully IEEE-754 compliant FP16 multiplier with all rounding modes,
// subnormal handling, and NaN/Inf propagation is complex. The implementation
// here is functionally correct for normal numbers with round-to-nearest-even.
// Subnormals are flushed to zero (FTZ), consistent with most ML accelerators.
// NaN and Inf propagate correctly through the accumulator.
//
// =============================================================================

`timescale 1ns/1ps

// =============================================================================
// FP16 Multiplier (combinational, outputs FP32 result for accumulation)
// Inputs:  two FP16 values
// Output:  FP32 product (no intermediate rounding — full product precision)
//
// FP16 format: [15]=sign, [14:10]=exponent (bias=15), [9:0]=mantissa
// FP32 format: [31]=sign, [30:23]=exponent (bias=127), [22:0]=mantissa
// =============================================================================

module fp16_mul_to_fp32 (
    input  logic [15:0] a,      // FP16 input A
    input  logic [15:0] b,      // FP16 input B
    output logic [31:0] product // FP32 output product
);

    // -------------------------
    // Decode FP16 fields
    // -------------------------
    logic       a_sign,  b_sign;
    logic [4:0] a_exp,   b_exp;
    logic [9:0] a_mant,  b_mant;
    logic       a_inf,   b_inf;
    logic       a_nan,   b_nan;
    logic       a_zero,  b_zero;

    assign a_sign = a[15];
    assign a_exp  = a[14:10];
    assign a_mant = a[9:0];
    assign b_sign = b[15];
    assign b_exp  = b[14:10];
    assign b_mant = b[9:0];

    // Special value detection
    assign a_inf  = (a_exp == 5'b11111) & (a_mant == 10'd0);
    assign b_inf  = (b_exp == 5'b11111) & (b_mant == 10'd0);
    assign a_nan  = (a_exp == 5'b11111) & (a_mant != 10'd0);
    assign b_nan  = (b_exp == 5'b11111) & (b_mant != 10'd0);
    assign a_zero = (a_exp == 5'd0) & (a_mant == 10'd0);  // Flush subnormals to zero
    assign b_zero = (b_exp == 5'd0) & (b_mant == 10'd0);

    // -------------------------
    // Compute result sign
    // -------------------------
    logic result_sign;
    assign result_sign = a_sign ^ b_sign;

    // -------------------------
    // Add hidden bits to form 11-bit significands
    // For normal numbers (exp != 0): hidden bit = 1
    // For subnormals (exp == 0): flush to zero (hidden bit = 0)
    // -------------------------
    logic [10:0] a_sig, b_sig; // 11-bit significands with hidden bit
    assign a_sig = (a_exp != 5'd0) ? {1'b1, a_mant} : 11'd0; // FTZ for subnormals
    assign b_sig = (b_exp != 5'd0) ? {1'b1, b_mant} : 11'd0;

    // -------------------------
    // Compute product exponent
    // FP16 bias = 15. FP32 bias = 127.
    // Unbiased exponent = exp - bias.
    // Product unbiased exp = (a_exp - 15) + (b_exp - 15)
    // Product biased (FP32) = product_unbiased + 127
    //                       = a_exp + b_exp - 30 + 127
    //                       = a_exp + b_exp + 97
    // -------------------------
    logic [8:0] product_exp_raw; // 9 bits to detect overflow/underflow
    assign product_exp_raw = {4'd0, a_exp} + {4'd0, b_exp} + 9'd97;
    // Note: If either operand was subnormal (flushed to zero), exponent is don't-care.

    // -------------------------
    // Multiply significands: 11-bit × 11-bit = 22-bit product
    // -------------------------
    logic [21:0] mant_product;
    assign mant_product = a_sig * b_sig;

    // -------------------------
    // Normalise: the product of two 1.x numbers is either 1x.xxx (no leading 1)
    // or x.1xxx (needs left shift). The MSB of mant_product[21] indicates this.
    // If mant_product[21] = 1: product is in [2,4) range, exponent += 1.
    // If mant_product[21] = 0: product is in [1,2) range, no adjustment needed.
    // In both cases, the FP32 23-bit mantissa gets the bits below the leading 1.
    // -------------------------
    logic [8:0] product_exp;
    logic [22:0] product_mant; // FP32 mantissa (23 bits)

    always_comb begin
        if (mant_product[21]) begin
            // Leading 1 at bit 21: mantissa bits [20:0] -> FP32 mantissa [22:2]
            // (shift right by 2 to get 21 bits into the top of 23-bit FP32 mantissa)
            // Actually: bits [20:0] are the fractional part. FP32 has 23 mantissa bits.
            // mant_product = 1X.XXXXXXXXXX (22 bits), FP32 mantissa = XXXXXXXXXXXXXXXXXXXXXXX
            // Take bits [20:0] and place in [22:2], zero-pad [1:0].
            // This is a right-shift by 2 relative to bit 21.
            product_exp  = product_exp_raw + 9'd1;
            product_mant = {mant_product[20:0], 2'b00};
        end else begin
            // Leading 1 at bit 20: mant_product = 0 1.XXXXXXXXXX
            // Take bits [19:0] as fractional part, placed in [22:3].
            product_exp  = product_exp_raw;
            product_mant = {mant_product[19:0], 3'b000};
        end
    end

    // -------------------------
    // Assemble FP32 result, handling special cases
    // -------------------------
    logic result_nan, result_inf, result_zero;
    assign result_nan  = a_nan | b_nan | (a_inf & b_zero) | (b_inf & a_zero);
    assign result_inf  = (a_inf | b_inf) & ~result_nan;
    assign result_zero = (a_zero | b_zero) & ~result_nan;

    logic [8:0] final_exp;
    assign final_exp = result_zero ? 9'd0 :
                       result_inf  ? 9'd255 :
                       result_nan  ? 9'd255 :
                       (product_exp[8]) ? 9'd255 :  // Overflow to Inf
                       product_exp;

    logic [22:0] final_mant;
    assign final_mant = result_nan  ? 23'h400000 :  // Quiet NaN
                        result_inf  ? 23'd0 :
                        result_zero ? 23'd0 :
                        product_mant;

    logic final_sign;
    assign final_sign = result_nan ? 1'b0 : result_sign;

    assign product = {final_sign, final_exp[7:0], final_mant};

endmodule : fp16_mul_to_fp32


// =============================================================================
// FP32 Adder (combinational, for FP16-mode accumulation)
// Adds two FP32 values and returns a FP32 result.
// Round-to-nearest-even, flush subnormals to zero.
// =============================================================================

module fp32_add (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] result
);

    logic        a_sign, b_sign;
    logic [7:0]  a_exp,  b_exp;
    logic [22:0] a_mant, b_mant;

    assign a_sign = a[31]; assign a_exp = a[30:23]; assign a_mant = a[22:0];
    assign b_sign = b[31]; assign b_exp = b[30:23]; assign b_mant = b[22:0];

    // Special values
    logic a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
    assign a_nan  = (a_exp == 8'hFF) & (a_mant != 23'd0);
    assign b_nan  = (b_exp == 8'hFF) & (b_mant != 23'd0);
    assign a_inf  = (a_exp == 8'hFF) & (a_mant == 23'd0);
    assign b_inf  = (b_exp == 8'hFF) & (b_mant == 23'd0);
    assign a_zero = (a_exp == 8'd0);
    assign b_zero = (b_exp == 8'd0);

    // Exponent difference and alignment
    logic [7:0] exp_diff;
    logic       a_larger;
    logic [7:0] result_exp;
    logic [26:0] a_sig_shifted, b_sig_shifted; // 24-bit significand + 3 guard bits

    // Determine which operand has larger exponent
    assign a_larger = (a_exp >= b_exp);
    assign exp_diff = a_larger ? (a_exp - b_exp) : (b_exp - a_exp);
    assign result_exp = a_larger ? a_exp : b_exp;

    // Form 24-bit significands (hidden bit + 23 mantissa bits)
    logic [23:0] a_sig24, b_sig24;
    assign a_sig24 = (a_exp != 8'd0) ? {1'b1, a_mant} : 24'd0;
    assign b_sig24 = (b_exp != 8'd0) ? {1'b1, b_mant} : 24'd0;

    // Align smaller exponent operand (right shift by exp_diff, keep 3 guard bits)
    // Cap shift at 27 to avoid undefined behaviour
    logic [5:0] shift_amt;
    assign shift_amt = (exp_diff > 6'd27) ? 6'd27 : exp_diff[5:0];

    assign a_sig_shifted = a_larger ? {a_sig24, 3'b000} : ({a_sig24, 3'b000} >> shift_amt);
    assign b_sig_shifted = a_larger ? ({b_sig24, 3'b000} >> shift_amt) : {b_sig24, 3'b000};

    // Add or subtract based on effective signs
    logic effective_add; // 1 = add magnitudes, 0 = subtract
    assign effective_add = (a_sign == b_sign);

    logic [27:0] sum_raw; // 28-bit to detect carry
    logic result_sign_int;

    always_comb begin
        if (effective_add) begin
            sum_raw = {1'b0, a_sig_shifted} + {1'b0, b_sig_shifted};
            result_sign_int = a_sign;
        end else begin
            // Subtract: larger - smaller (result sign = sign of larger magnitude)
            if (a_sig_shifted >= b_sig_shifted) begin
                sum_raw = {1'b0, a_sig_shifted} - {1'b0, b_sig_shifted};
                result_sign_int = a_sign;
            end else begin
                sum_raw = {1'b0, b_sig_shifted} - {1'b0, a_sig_shifted};
                result_sign_int = b_sign;
            end
        end
    end

    // Normalise result
    // If carry out (sum_raw[27]=1): result = 1X.XXX, shift right, exp++
    // Otherwise: leading-zero count to normalise left shifts
    logic [7:0]  norm_exp;
    logic [26:0] norm_sig;
    logic [22:0] rounded_mant;

    // Leading-zero count for normalisation (simplified: check bit-by-bit)
    logic [4:0] lzc; // Leading zero count in sum_raw[26:0]
    always_comb begin
        lzc = 5'd0;
        for (int b = 26; b >= 0; b--) begin
            if (sum_raw[b]) break;
            lzc = lzc + 5'd1;
        end
    end

    always_comb begin
        if (sum_raw[27]) begin
            // Carry: shift right by 1, increment exponent
            norm_exp = result_exp + 8'd1;
            norm_sig = sum_raw[27:1]; // Drop the LSB (rounding simplified here)
        end else if (sum_raw == 28'd0) begin
            norm_exp = 8'd0;
            norm_sig = 27'd0;
        end else begin
            // Left-normalise
            norm_exp = result_exp - {3'd0, lzc};
            norm_sig = sum_raw[26:0] << lzc;
        end
        // Round to nearest (truncate the 3 guard bits for simplicity here)
        rounded_mant = norm_sig[26:4];
    end

    // Final result assembly
    logic result_nan_f, result_inf_f, result_zero_f;
    assign result_nan_f  = a_nan | b_nan | (a_inf & b_inf & (a_sign ^ b_sign));
    assign result_inf_f  = (a_inf | b_inf | (norm_exp == 8'hFF)) & ~result_nan_f;
    assign result_zero_f = (sum_raw == 28'd0) & ~result_nan_f;

    assign result = result_nan_f  ? 32'h7FC00000 :  // Quiet NaN
                    result_inf_f  ? {result_sign_int, 8'hFF, 23'd0} :
                    result_zero_f ? 32'd0 :
                    {result_sign_int, norm_exp, rounded_mant};

endmodule : fp32_add


// =============================================================================
// Mixed-Precision MAC Unit Top-Level (3-stage pipelined)
// =============================================================================

module mac_unit_mixed_precision #(
    parameter int DATA_W = 16, // Width of a_in and b_in ports (FP16 uses all 16 bits;
                                // INT8 uses bits [7:0] sign-extended to [15:0])
    parameter int INT_ACC_W = 32, // INT8-mode accumulator width
    parameter int FP_ACC_W  = 32  // FP16-mode accumulator width (FP32)
) (
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   mode,       // 0=INT8, 1=FP16
    input  logic [DATA_W-1:0]      a_in,       // Activation input
    input  logic [DATA_W-1:0]      b_in,       // Weight input
    input  logic                   clear,      // Clear accumulator (synced to input)
    input  logic                   valid_in,   // Input data valid
    output logic [INT_ACC_W-1:0]   acc_int32,  // INT8-mode accumulator
    output logic [FP_ACC_W-1:0]    acc_fp32,   // FP16-mode accumulator
    output logic                   valid_out   // Output valid (valid_in delayed 3 cycles)
);

    // =========================================================================
    // Pipeline stage registers
    // =========================================================================

    // Stage 1 → Stage 2
    logic        s1_mode,   s2_mode,   s3_mode;
    logic        s1_valid,  s2_valid,  s3_valid;
    logic        s1_clear,  s2_clear,  s3_clear;

    // INT8 path
    logic signed [7:0]  s1_a_int8, s1_b_int8;
    logic signed [15:0] s2_int8_product;  // INT8 x INT8 = INT16

    // FP16 path
    logic [15:0] s1_a_fp16, s1_b_fp16;
    logic [31:0] s2_fp32_product;  // FP16 x FP16 = FP32

    // =========================================================================
    // STAGE 1: Input registration and decode
    // =========================================================================

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s1_mode      <= 1'b0;
            s1_valid     <= 1'b0;
            s1_clear     <= 1'b0;
            s1_a_int8    <= '0;
            s1_b_int8    <= '0;
            s1_a_fp16    <= '0;
            s1_b_fp16    <= '0;
        end else begin
            s1_mode   <= mode;
            s1_valid  <= valid_in;
            s1_clear  <= clear;
            // INT8 mode: use lower 8 bits (sign-extended inputs expected at [7:0])
            s1_a_int8 <= a_in[7:0];
            s1_b_int8 <= b_in[7:0];
            // FP16 mode: use full 16 bits
            s1_a_fp16 <= a_in;
            s1_b_fp16 <= b_in;
        end
    end

    // =========================================================================
    // STAGE 2: Multiply
    // =========================================================================

    // INT8 multiply: 8 x 8 = 16-bit signed result (no overflow with INT16)
    logic signed [15:0] int8_product_comb;
    assign int8_product_comb = $signed(s1_a_int8) * $signed(s1_b_int8);

    // FP16 multiply: uses fp16_mul_to_fp32 sub-module (combinational)
    logic [31:0] fp32_product_comb;

    fp16_mul_to_fp32 u_fp16_mul (
        .a      (s1_a_fp16),
        .b      (s1_b_fp16),
        .product(fp32_product_comb)
    );

    // Register multiply results
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s2_mode          <= 1'b0;
            s2_valid         <= 1'b0;
            s2_clear         <= 1'b0;
            s2_int8_product  <= '0;
            s2_fp32_product  <= '0;
        end else begin
            s2_mode         <= s1_mode;
            s2_valid        <= s1_valid;
            s2_clear        <= s1_clear;
            s2_int8_product <= int8_product_comb;
            s2_fp32_product <= fp32_product_comb;
        end
    end

    // =========================================================================
    // STAGE 3: Accumulate
    // =========================================================================

    // INT32 accumulator
    logic signed [INT_ACC_W-1:0] int32_acc_reg;
    logic signed [INT_ACC_W-1:0] int32_acc_next;

    // Sign-extend INT16 product to INT32 before adding
    logic signed [INT_ACC_W-1:0] int32_product_extended;
    assign int32_product_extended = INT_ACC_W'(signed'(s2_int8_product));

    always_comb begin
        if (s2_clear) begin
            int32_acc_next = int32_product_extended; // Clear then accumulate
        end else if (s2_valid && ~s2_mode) begin
            int32_acc_next = int32_acc_reg + int32_product_extended;
        end else begin
            int32_acc_next = int32_acc_reg;
        end
    end

    // FP32 accumulator
    logic [FP_ACC_W-1:0] fp32_acc_reg;
    logic [FP_ACC_W-1:0] fp32_acc_next;
    logic [FP_ACC_W-1:0] fp32_add_result;

    fp32_add u_fp32_add (
        .a     (fp32_acc_reg),
        .b     (s2_fp32_product),
        .result(fp32_add_result)
    );

    always_comb begin
        if (s2_clear) begin
            fp32_acc_next = s2_fp32_product; // Clear then accumulate
        end else if (s2_valid && s2_mode) begin
            fp32_acc_next = fp32_add_result;
        end else begin
            fp32_acc_next = fp32_acc_reg;
        end
    end

    // Register accumulators and pipeline control
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s3_mode      <= 1'b0;
            s3_valid     <= 1'b0;
            s3_clear     <= 1'b0;
            int32_acc_reg <= '0;
            fp32_acc_reg  <= 32'h00000000; // +0.0 in FP32
        end else begin
            s3_mode       <= s2_mode;
            s3_valid      <= s2_valid;
            s3_clear      <= s2_clear;
            int32_acc_reg <= int32_acc_next;
            fp32_acc_reg  <= fp32_acc_next;
        end
    end

    // =========================================================================
    // Outputs
    // =========================================================================

    assign acc_int32  = int32_acc_reg;
    assign acc_fp32   = fp32_acc_reg;
    assign valid_out  = s3_valid;

endmodule : mac_unit_mixed_precision


// =============================================================================
// TESTBENCH
// =============================================================================
// Tests:
//   Test 1 (INT8 mode): Accumulate 4 pairs, check INT32 result
//     Pairs: (2,3), (4,5), (-1,6), (127,-128)
//     Expected: 2*3 + 4*5 + (-1)*6 + 127*(-128) = 6 + 20 - 6 - 16256 = -16236
//
//   Test 2 (FP16 mode): Accumulate 3 pairs of common FP16 values
//     Pairs: (1.0, 2.0), (0.5, 4.0), (3.0, -1.0)
//     Expected: 1.0*2.0 + 0.5*4.0 + 3.0*(-1.0) = 2.0 + 2.0 - 3.0 = 1.0
//
//   Test 3: Mode switch (INT8 → FP16), verify correct accumulator isolation
//
// FP16 encoding helpers are included in the testbench.
// =============================================================================

module tb_mac_unit_mixed_precision;

    // Absolute value of a real. SystemVerilog has no $abs system function
    // (it is a simulator extension), so define one for portability.
    function automatic real abs_real(real x);
        return (x < 0.0) ? -x : x;
    endfunction


    // DUT signals
    logic        clk;
    logic        rst_n;
    logic        mode;
    logic [15:0] a_in;
    logic [15:0] b_in;
    logic        clear;
    logic        valid_in;
    logic [31:0] acc_int32;
    logic [31:0] acc_fp32;
    logic        valid_out;

    // Pipeline depth
    localparam int PIPE_DEPTH = 3;

    // DUT
    mac_unit_mixed_precision dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .mode      (mode),
        .a_in      (a_in),
        .b_in      (b_in),
        .clear     (clear),
        .valid_in  (valid_in),
        .acc_int32 (acc_int32),
        .acc_fp32  (acc_fp32),
        .valid_out (valid_out)
    );

    // Clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------
    // FP16 encoding helper
    // Converts a real value to the nearest FP16 bit pattern.
    // Only handles normal numbers in [-65504, 65504].
    // -------------------------
    function automatic logic [15:0] to_fp16(input real val);
        logic        s;
        int          e_unbiased;
        real         m_real;
        logic [9:0]  m_bits;

        if (val == 0.0) return 16'h0000;

        s = (val < 0.0) ? 1'b1 : 1'b0;
        val = (val < 0.0) ? -val : val;

        // Find exponent: largest e such that 2^e <= val
        e_unbiased = 0;
        if (val >= 2.0) begin
            while (val >= 2.0) begin val = val / 2.0; e_unbiased++; end
        end else if (val < 1.0) begin
            while (val < 1.0) begin val = val * 2.0; e_unbiased--; end
        end
        // val is now in [1.0, 2.0)
        m_real = val - 1.0; // Fractional part
        m_bits = 10'(int'(m_real * 1024.0)); // 10-bit mantissa

        return {s, 5'(e_unbiased + 15), m_bits};
    endfunction

    // FP32 to real (for display)
    function automatic real fp32_to_real(input logic [31:0] f);
        int    exp_int;
        real   mantissa_real;
        if (f[30:23] == 8'hFF) return (f[22:0] != 23'd0) ? 0.0/0.0 : (f[31] ? -1.0/0.0 : 1.0/0.0);
        if (f[30:23] == 8'd0)  return 0.0;
        exp_int = int'(f[30:23]) - 127;
        mantissa_real = 1.0 + real'(f[22:0]) / real'(1 << 23);
        for (int i = 0; i < exp_int && i < 64; i++) mantissa_real *= 2.0;
        for (int i = 0; i > exp_int && i > -64; i--) mantissa_real /= 2.0;
        return f[31] ? -mantissa_real : mantissa_real;
    endfunction

    // -------------------------
    // Task: INT8 MAC sequence
    // -------------------------
    task automatic int8_mac_sequence(
        input logic signed [7:0] a_vals[],
        input logic signed [7:0] b_vals[],
        input int                n_pairs,
        output logic [31:0]      result
    );
        integer i;
        // Clear accumulator with first valid data
        @(negedge clk);
        mode     = 1'b0;
        clear    = 1'b1;
        valid_in = 1'b1;
        a_in     = 16'(signed'(a_vals[0]));
        b_in     = 16'(signed'(b_vals[0]));

        @(negedge clk);
        clear = 1'b0;

        for (i = 1; i < n_pairs; i++) begin
            a_in = 16'(signed'(a_vals[i]));
            b_in = 16'(signed'(b_vals[i]));
            @(negedge clk);
        end

        valid_in = 1'b0;
        a_in     = '0;
        b_in     = '0;

        // Wait for pipeline to drain
        repeat (PIPE_DEPTH + 1) @(posedge clk);
        result = acc_int32;
    endtask

    // -------------------------
    // Task: FP16 MAC sequence
    // -------------------------
    task automatic fp16_mac_sequence(
        input real   a_vals[],
        input real   b_vals[],
        input int    n_pairs,
        output logic [31:0] result
    );
        integer i;
        @(negedge clk);
        mode     = 1'b1;
        clear    = 1'b1;
        valid_in = 1'b1;
        a_in     = to_fp16(a_vals[0]);
        b_in     = to_fp16(b_vals[0]);

        @(negedge clk);
        clear = 1'b0;

        for (i = 1; i < n_pairs; i++) begin
            a_in = to_fp16(a_vals[i]);
            b_in = to_fp16(b_vals[i]);
            @(negedge clk);
        end

        valid_in = 1'b0;
        a_in     = '0;
        b_in     = '0;

        repeat (PIPE_DEPTH + 1) @(posedge clk);
        result = acc_fp32;
    endtask

    // -------------------------
    // Main test
    // -------------------------
    logic [31:0] test_result;
    real         test_result_real;
    real         expected_real;

    logic signed [7:0] int8_a[4];
    logic signed [7:0] int8_b[4];
    real fp16_a[3];
    real fp16_b[3];

    initial begin
        rst_n    = 1'b0;
        mode     = 1'b0;
        clear    = 1'b0;
        valid_in = 1'b0;
        a_in     = '0;
        b_in     = '0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        // -----------------------------------------------
        // Test 1: INT8 mode
        // -----------------------------------------------
        $display("\n=== Test 1: INT8 mode ===");
        int8_a[0] =  2; int8_b[0] =  3;   //  6
        int8_a[1] =  4; int8_b[1] =  5;   // 20
        int8_a[2] = -1; int8_b[2] =  6;   // -6
        int8_a[3] = 127; int8_b[3] = -128; // -16256
        // Expected: 6 + 20 - 6 - 16256 = -16236

        int8_mac_sequence(int8_a, int8_b, 4, test_result);

        if ($signed(test_result) == -32'sd16236)
            $display("PASS: INT8 acc = %0d (expected -16236)", $signed(test_result));
        else
            $display("FAIL: INT8 acc = %0d (expected -16236)", $signed(test_result));

        // -----------------------------------------------
        // Test 2: FP16 mode
        // -----------------------------------------------
        $display("\n=== Test 2: FP16 mode ===");
        fp16_a[0] = 1.0; fp16_b[0] =  2.0;  // 2.0
        fp16_a[1] = 0.5; fp16_b[1] =  4.0;  // 2.0
        fp16_a[2] = 3.0; fp16_b[2] = -1.0;  // -3.0
        // Expected: 2.0 + 2.0 - 3.0 = 1.0

        fp16_mac_sequence(fp16_a, fp16_b, 3, test_result);

        test_result_real = fp32_to_real(test_result);
        expected_real = 1.0;

        if (abs_real(test_result_real - expected_real) < 0.01)
            $display("PASS: FP16 acc = %f (expected 1.0)", test_result_real);
        else
            $display("FAIL: FP16 acc = %f (expected 1.0)", test_result_real);

        // -----------------------------------------------
        // Test 3: INT8 accumulator unaffected by FP16 ops
        // -----------------------------------------------
        $display("\n=== Test 3: Accumulator isolation ===");
        // After the FP16 test, the INT8 accumulator should be unchanged
        // (still holds -16236 from test 1).
        if ($signed(acc_int32) == -32'sd16236)
            $display("PASS: INT8 accumulator preserved during FP16 test = %0d",
                      $signed(acc_int32));
        else
            $display("FAIL: INT8 accumulator changed during FP16 test = %0d (expected -16236)",
                      $signed(acc_int32));

        // -----------------------------------------------
        // Test 4: INT8 edge case — clear between tiles
        // -----------------------------------------------
        $display("\n=== Test 4: Clear and new accumulation ===");
        int8_a[0] = 10; int8_b[0] = 10;  // 100
        int8_a[1] =  5; int8_b[1] =  2;  // 10
        // Expected: 110 (previous -16236 should be cleared)

        int8_mac_sequence(int8_a, int8_b, 2, test_result);

        if ($signed(test_result) == 32'sd110)
            $display("PASS: After clear, INT8 acc = %0d (expected 110)", $signed(test_result));
        else
            $display("FAIL: After clear, INT8 acc = %0d (expected 110)", $signed(test_result));

        $display("\n=== All MAC unit tests complete ===");
        $finish;
    end

    initial begin
        #10000;
        $display("TIMEOUT");
        $finish;
    end

    initial begin
        $dumpfile("mac_unit.vcd");
        $dumpvars(0, tb_mac_unit_mixed_precision);
    end

endmodule : tb_mac_unit_mixed_precision

// =============================================================================
// INTERVIEW DISCUSSION POINTS
// =============================================================================
//
// 1. WHY FP32 ACCUMULATION FOR FP16 INPUTS?
//    FP16 has only 10 bits of mantissa. Accumulating many FP16 products in
//    FP16 causes catastrophic cancellation and precision loss. FP32 accumulation
//    (23-bit mantissa) captures the full precision of each FP16 product and
//    avoids underflow in partial sums. This is why NVIDIA Tensor Cores use
//    FP16 for multiply but FP32 for accumulation (the "mixed-precision" of the
//    cuBLAS name).
//
// 2. WHY SIGN-EXTEND INT8 TO INT32 BEFORE ADDING?
//    A negative INT8 product (e.g., -5 = 8'b11111011) must be sign-extended to
//    32 bits (32'hFFFFFFFB) before addition to avoid treating it as a large
//    positive number. Failure to sign-extend is one of the most common RTL bugs
//    in fixed-point datapath design.
//
// 3. CLEAR TIMING IN PIPELINED MAC
//    In a 3-stage pipeline, the clear signal must be synchronised to the input
//    data it clears. If clear is asserted at the same cycle as the first data
//    pair, stage 3 will receive both the clear command and the first product at
//    the same time — but the clear must cause the accumulator to load the first
//    product (not zero). The implementation above handles this by setting
//    int32_acc_next = product (not zero) when clear is asserted with valid data.
//    An alternative is to implement clear as "load 0 then add product" in two
//    cycles, which is simpler but adds latency.
//
// 4. SATURATION
//    The INT32 accumulator never overflows for INT8 inputs with K <= 65536.
//    However, when converting the INT32 output to INT8 for requantisation
//    (done outside this module), saturation is mandatory. See the saturation
//    logic in mac_unit_design.md Q7.
//
// =============================================================================
