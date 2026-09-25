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
    assign a_zero = (a_exp == 5'd0);  // Zero or subnormal: flush to zero (FTZ)
    assign b_zero = (b_exp == 5'd0);

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

    // Align smaller exponent operand (right shift by exp_diff, keep 3 guard bits:
    // guard, round, sticky). Bits shifted out below the sticky position are ORed
    // into it so round-to-nearest-even sees them. Cap shift at 27 (everything
    // is then sticky).
    logic [5:0]  shift_amt;
    logic [26:0] a_ext, b_ext, small_ext, small_shifted;
    logic        small_sticky;
    assign shift_amt = (exp_diff > 8'd27) ? 6'd27 : exp_diff[5:0];
    assign a_ext     = {a_sig24, 3'b000};
    assign b_ext     = {b_sig24, 3'b000};
    assign small_ext = a_larger ? b_ext : a_ext;
    assign small_shifted = small_ext >> shift_amt;
    assign small_sticky  = |(small_ext & ~(27'h7FFFFFF << shift_amt));

    assign a_sig_shifted = a_larger ? a_ext : (small_shifted | {26'd0, small_sticky});
    assign b_sig_shifted = a_larger ? (small_shifted | {26'd0, small_sticky}) : b_ext;

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

    // Normalise result so the hidden bit sits at norm_sig[26]:
    // If carry out (sum_raw[27]=1): result = 1X.XXX, shift right, exp++
    // (the bit shifted out is folded into sticky).
    // Otherwise: leading-zero count to normalise left shifts. A left shift of
    // 2 or more only happens when exp_diff <= 1, when no bits were lost.
    logic signed [9:0] norm_exp;   // wide + signed to catch overflow/underflow
    logic [26:0] norm_sig;
    logic [23:0] rounded_sig;      // hidden bit + 23-bit mantissa after rounding
    logic        round_up;
    logic signed [9:0] final_exp_f;

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
            norm_exp = $signed({2'b00, result_exp}) + 10'sd1;
            norm_sig = {sum_raw[27:2], sum_raw[1] | sum_raw[0]};
        end else if (sum_raw == 28'd0) begin
            norm_exp = 10'sd0;
            norm_sig = 27'd0;
        end else begin
            // Left-normalise
            norm_exp = $signed({2'b00, result_exp}) - $signed({5'd0, lzc});
            norm_sig = sum_raw[26:0] << lzc;
        end
        // Round to nearest, ties to even: guard = norm_sig[2],
        // round|sticky = norm_sig[1:0], LSB of the kept mantissa = norm_sig[3]
        round_up    = norm_sig[2] & ((|norm_sig[1:0]) | norm_sig[3]);
        rounded_sig = norm_sig[26:3] + {23'd0, round_up};
        final_exp_f = norm_exp;
        // Rounding carried out of the significand (1.111..1 -> 10.000..0)
        if (round_up && norm_sig[26:3] == 24'hFFFFFF) begin
            rounded_sig = 24'h800000;
            final_exp_f = norm_exp + 10'sd1;
        end
    end

    // Final result assembly
    logic result_nan_f, result_inf_f, result_zero_f;
    assign result_nan_f  = a_nan | b_nan | (a_inf & b_inf & (a_sign ^ b_sign));
    assign result_inf_f  = (a_inf | b_inf | (final_exp_f >= 10'sd255)) & ~result_nan_f;
    // Exact zero, or underflow below the smallest normal (flush to zero)
    assign result_zero_f = ((sum_raw == 28'd0) | (final_exp_f <= 10'sd0)) & ~result_nan_f & ~result_inf_f;

    assign result = result_nan_f  ? 32'h7FC00000 :  // Quiet NaN
                    result_inf_f  ? {(a_inf ? a_sign : b_inf ? b_sign : result_sign_int), 8'hFF, 23'd0} :
                    result_zero_f ? 32'd0 :
                    {result_sign_int, final_exp_f[7:0], rounded_sig[22:0]};

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

    // Saturating add: one extra bit catches overflow, then clamp to the INT32
    // range (the result sticks at the rail rather than wrapping)
    logic signed [INT_ACC_W:0]   int32_sum_wide;
    logic signed [INT_ACC_W-1:0] int32_sum_sat;
    localparam logic signed [INT_ACC_W:0] INT_MAX_W = (INT_ACC_W+1)'((64'sd1 <<< (INT_ACC_W-1)) - 1);
    localparam logic signed [INT_ACC_W:0] INT_MIN_W = -(INT_MAX_W + 1);
    assign int32_sum_wide = {int32_acc_reg[INT_ACC_W-1], int32_acc_reg}
                          + {int32_product_extended[INT_ACC_W-1], int32_product_extended};
    assign int32_sum_sat  = (int32_sum_wide > INT_MAX_W) ? INT_MAX_W[INT_ACC_W-1:0] :
                            (int32_sum_wide < INT_MIN_W) ? INT_MIN_W[INT_ACC_W-1:0] :
                                                           int32_sum_wide[INT_ACC_W-1:0];

    // clear only affects the accumulator of the mode it arrives with, so a
    // tile in one mode never disturbs the other mode's accumulator
    always_comb begin
        if (s2_clear && ~s2_mode) begin
            // Clear then accumulate (or just clear if no valid data came with it)
            int32_acc_next = s2_valid ? int32_product_extended : '0;
        end else if (s2_valid && ~s2_mode) begin
            int32_acc_next = int32_sum_sat;
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
        if (s2_clear && s2_mode) begin
            // Clear then accumulate (or just clear to +0.0 if no valid data)
            fp32_acc_next = s2_valid ? s2_fp32_product : 32'h00000000;
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
//   Test 4: clear starts a new INT8 tile; FP32 accumulator untouched by it
//   Test 5: 50 random INT8 tiles vs an integer reference
//   Test 6: 50 random FP16 tiles, bit-exact vs an FP32 round-to-nearest-even
//           reference (ref_fp_acc)
//   Test 7: FP16 specials -- subnormal FTZ, Inf, Inf-Inf, Inf*0, NaN, and a
//           carry-out rounding case that needs the sticky bit
//   Test 8: INT32 saturation at both rails
// Every check is bit-exact and counted; the final verdict depends on the count.
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

    // FP16 bit pattern to real (subnormals flushed to zero, like the DUT;
    // only used for finite values)
    function automatic real fp16_to_real(input logic [15:0] h);
        real v;
        if (h[14:10] == 5'd0) return 0.0;
        v = 1.0 + real'(h[9:0]) / 1024.0;
        for (int i = 15; i < int'(h[14:10]); i++) v *= 2.0;
        for (int i = 15; i > int'(h[14:10]); i--) v /= 2.0;
        return h[15] ? -v : v;
    endfunction

    // Reference FP16-mode tile: each FP16 x FP16 product is exact in double
    // (22-bit significand); each accumulate is a double add rounded once to
    // FP32. Because 53 >= 2*24 + 2, rounding the double sum of two FP32
    // values to FP32 equals a single correctly-rounded (RNE) FP32 add.
    // The rounding is forced through $shortrealtobits: a simulator may hold
    // shortreal variables at double precision (xsim 2025.2 does), so a plain
    // shortreal accumulator would silently skip the per-step FP32 rounding.
    function automatic logic [31:0] ref_fp_acc(input logic [15:0] a_vals[], input logic [15:0] b_vals[]);
        logic [31:0] acc_bits;
        real         p;
        foreach (a_vals[i]) begin
            p        = fp16_to_real(a_vals[i]) * fp16_to_real(b_vals[i]);
            acc_bits = (i == 0) ? $shortrealtobits(shortreal'(p))
                                : $shortrealtobits(shortreal'(real'($bitstoshortreal(acc_bits)) + p));
        end
        return acc_bits;
    endfunction

    // -------------------------
    // Task: FP16 MAC sequence (raw FP16 bit patterns)
    // -------------------------
    task automatic fp16_mac_bits(
        input logic [15:0] a_vals[],
        input logic [15:0] b_vals[],
        input int          n_pairs,
        output logic [31:0] result
    );
        integer i;
        @(negedge clk);
        mode     = 1'b1;
        clear    = 1'b1;
        valid_in = 1'b1;
        a_in     = a_vals[0];
        b_in     = b_vals[0];

        @(negedge clk);
        clear = 1'b0;

        for (i = 1; i < n_pairs; i++) begin
            a_in = a_vals[i];
            b_in = b_vals[i];
            @(negedge clk);
        end

        valid_in = 1'b0;
        a_in     = '0;
        b_in     = '0;

        repeat (PIPE_DEPTH + 1) @(posedge clk);
        result = acc_fp32;
    endtask

    // -------------------------
    // Task: FP16 MAC sequence (real values, converted with to_fp16)
    // -------------------------
    task automatic fp16_mac_sequence(
        input real   a_vals[],
        input real   b_vals[],
        input int    n_pairs,
        output logic [31:0] result
    );
        logic [15:0] a_bits[], b_bits[];
        a_bits = new[n_pairs];
        b_bits = new[n_pairs];
        for (int i = 0; i < n_pairs; i++) begin
            a_bits[i] = to_fp16(a_vals[i]);
            b_bits[i] = to_fp16(b_vals[i]);
        end
        fp16_mac_bits(a_bits, b_bits, n_pairs, result);
    endtask

    // -------------------------
    // Main test
    // -------------------------
    logic [31:0] test_result;
    real         test_result_real;
    real         expected_real;
    logic [31:0] expected_bits;
    int          errors = 0;
    int          exp_int;

    logic signed [7:0] int8_a[4];
    logic signed [7:0] int8_b[4];
    real fp16_a[3];
    real fp16_b[3];
    logic signed [7:0] rnd_a[], rnd_b[];
    logic [15:0]       rnd_fa[], rnd_fb[];
    logic [15:0]       sp_a[], sp_b[];

    // Check helper: bit-exact compare, count failures
    task automatic check32(input string what, input logic [31:0] got, input logic [31:0] exp);
        if (got === exp)
            $display("PASS: %s = 0x%08h", what, got);
        else begin
            $display("FAIL: %s = 0x%08h (expected 0x%08h)", what, got, exp);
            errors++;
        end
    endtask

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
        check32("INT8 acc (-16236)", test_result, -32'sd16236);

        // -----------------------------------------------
        // Test 2: FP16 mode
        // -----------------------------------------------
        $display("\n=== Test 2: FP16 mode ===");
        fp16_a[0] = 1.0; fp16_b[0] =  2.0;  // 2.0
        fp16_a[1] = 0.5; fp16_b[1] =  4.0;  // 2.0
        fp16_a[2] = 3.0; fp16_b[2] = -1.0;  // -3.0
        // Expected: 2.0 + 2.0 - 3.0 = 1.0 exactly (FP32 0x3F800000)

        fp16_mac_sequence(fp16_a, fp16_b, 3, test_result);
        test_result_real = fp32_to_real(test_result);
        $display("      FP16 acc = %f", test_result_real);
        check32("FP16 acc (1.0)", test_result, 32'h3F800000);

        // -----------------------------------------------
        // Test 3: INT8 accumulator unaffected by FP16 ops
        // -----------------------------------------------
        $display("\n=== Test 3: Accumulator isolation (INT8 during FP16) ===");
        // After the FP16 test (including its clear), the INT8 accumulator
        // should be unchanged (still holds -16236 from test 1).
        check32("INT8 acc after FP16 tile", acc_int32, -32'sd16236);

        // -----------------------------------------------
        // Test 4: INT8 edge case — clear between tiles
        // -----------------------------------------------
        $display("\n=== Test 4: Clear and new accumulation ===");
        int8_a[0] = 10; int8_b[0] = 10;  // 100
        int8_a[1] =  5; int8_b[1] =  2;  // 10
        // Expected: 110 (previous -16236 should be cleared)

        int8_mac_sequence(int8_a, int8_b, 2, test_result);
        check32("INT8 acc after clear (110)", test_result, 32'sd110);
        // ...and the FP32 accumulator must have kept 1.0 through the INT8 tile
        check32("FP32 acc after INT8 tile", acc_fp32, 32'h3F800000);

        // -----------------------------------------------
        // Test 5: random INT8 tiles vs integer reference
        // -----------------------------------------------
        $display("\n=== Test 5: 50 random INT8 tiles of 16 ===");
        rnd_a = new[16]; rnd_b = new[16];
        for (int t = 0; t < 50; t++) begin
            exp_int = 0;
            for (int i = 0; i < 16; i++) begin
                rnd_a[i] = $urandom; rnd_b[i] = $urandom;
                exp_int += int'(rnd_a[i]) * int'(rnd_b[i]);
            end
            int8_mac_sequence(rnd_a, rnd_b, 16, test_result);
            if (test_result !== exp_int) begin
                $display("FAIL: random INT8 tile %0d = %0d (expected %0d)", t, $signed(test_result), exp_int);
                errors++;
            end
        end
        $display("      done (%0d errors so far)", errors);

        // -----------------------------------------------
        // Test 6: random FP16 tiles, bit-exact vs FP32 RNE reference
        // Exponent fields 5..25 (2^-10..2^10), random signs, so tiles exercise
        // alignment shifts, cancellation and inexact (rounded) sums.
        // -----------------------------------------------
        $display("\n=== Test 6: 50 random FP16 tiles of 16 (bit-exact) ===");
        rnd_fa = new[16]; rnd_fb = new[16];
        for (int t = 0; t < 50; t++) begin
            for (int i = 0; i < 16; i++) begin
                rnd_fa[i] = {1'($urandom), 5'(5 + $urandom_range(20)), 10'($urandom)};
                rnd_fb[i] = {1'($urandom), 5'(5 + $urandom_range(20)), 10'($urandom)};
            end
            expected_bits = ref_fp_acc(rnd_fa, rnd_fb);
            fp16_mac_bits(rnd_fa, rnd_fb, 16, test_result);
            if (t < 3) $display("      tile %0d: DUT 0x%08h ref 0x%08h (%g)", t, test_result, expected_bits,
                                $bitstoshortreal(expected_bits));
            if (test_result !== expected_bits) begin
                $display("FAIL: random FP16 tile %0d = 0x%08h (expected 0x%08h)", t, test_result, expected_bits);
                errors++;
            end
        end
        $display("      done (%0d errors so far)", errors);

        // -----------------------------------------------
        // Test 7: FP16 special values
        // -----------------------------------------------
        $display("\n=== Test 7: FP16 specials (FTZ, Inf, NaN) ===");
        sp_a = new[2]; sp_b = new[2];
        // 2.0*1.0 + subnormal(0x0001)*1.0 -> subnormal flushed, exactly 2.0
        sp_a = '{16'h4000, 16'h0001}; sp_b = '{16'h3C00, 16'h3C00};
        fp16_mac_bits(sp_a, sp_b, 2, test_result);
        check32("2.0 + subnormal*1.0 (FTZ)", test_result, 32'h40000000);
        // +Inf*1.0 + 1.0*1.0 -> +Inf
        sp_a = '{16'h7C00, 16'h3C00}; sp_b = '{16'h3C00, 16'h3C00};
        fp16_mac_bits(sp_a, sp_b, 2, test_result);
        check32("+Inf + 1.0", test_result, 32'h7F800000);
        // +Inf*1.0 + (-Inf)*1.0 -> NaN
        sp_a = '{16'h7C00, 16'hFC00}; sp_b = '{16'h3C00, 16'h3C00};
        fp16_mac_bits(sp_a, sp_b, 2, test_result);
        check32("+Inf + -Inf (NaN)", test_result, 32'h7FC00000);
        // 1.0*1.0 + Inf*0 -> NaN
        sp_a = '{16'h3C00, 16'h7C00}; sp_b = '{16'h3C00, 16'h0000};
        fp16_mac_bits(sp_a, sp_b, 2, test_result);
        check32("1.0 + Inf*0 (NaN)", test_result, 32'h7FC00000);
        // NaN*1.0 + 1.0*1.0 -> NaN propagates through the accumulator
        sp_a = '{16'h7E00, 16'h3C00}; sp_b = '{16'h3C00, 16'h3C00};
        fp16_mac_bits(sp_a, sp_b, 2, test_result);
        check32("NaN + 1.0", test_result, 32'h7FC00000);
        // Rounding with carry-out: 0x3FFF^2 = 0x407FC004 plus 0x38F6*0x1E73 =
        // 0x3B7FF410 is exactly 4 + 1.016 ulp(4). After the carry shift the
        // guard bit is 1, round is 0 and only the lowest sticky bit is set, so
        // RNE must round up to 0x40800001 (dropping that sticky bit on the
        // carry shift would turn it into a tie and round to even, 0x40800000).
        sp_a = '{16'h3FFF, 16'h38F6}; sp_b = '{16'h3FFF, 16'h1E73};
        fp16_mac_bits(sp_a, sp_b, 2, test_result);
        check32("carry + sticky rounding", test_result, 32'h40800001);

        // -----------------------------------------------
        // Test 8: INT32 saturation
        // (-128)*(-128) = 16384 = 2^14; 131072 = 2^17 of them reach 2^31, one
        // past INT32_MAX -> clamp to 0x7FFFFFFF. 127*(-128) = -16256;
        // ceil(2^31 / 16256) = 132105 of them pass INT32_MIN -> 0x80000000.
        // -----------------------------------------------
        $display("\n=== Test 8: INT32 saturation ===");
        rnd_a = new[131072]; rnd_b = new[131072];
        foreach (rnd_a[i]) begin rnd_a[i] = -8'sd128; rnd_b[i] = -8'sd128; end
        int8_mac_sequence(rnd_a, rnd_b, 131072, test_result);
        check32("131072 x 16384 (clamp high)", test_result, 32'h7FFFFFFF);
        rnd_a = new[132105]; rnd_b = new[132105];
        foreach (rnd_a[i]) begin rnd_a[i] = 8'sd127; rnd_b[i] = -8'sd128; end
        int8_mac_sequence(rnd_a, rnd_b, 132105, test_result);
        check32("132105 x -16256 (clamp low)", test_result, 32'h80000000);

        $display("\n=== All MAC unit tests complete: %0d error(s) ===", errors);
        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("TESTS FAILED");
        $finish;
    end

    initial begin
        #5ms;
        $display("FAIL: TIMEOUT");
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
//    The INT32 accumulator cannot overflow for INT8 inputs with K <= 131071
//    (|product| <= 2^14, and 131071 * 2^14 < 2^31); beyond that the saturating
//    add clamps it to the INT32 rails (Test 8). Converting the INT32 output to
//    INT8 for requantisation (done outside this module) also needs
//    saturation. See the saturation logic in mac_unit_design.md Q7.
//
// =============================================================================
