// =============================================================================
// Challenge 01: Pipelined Softmax Unit with Online Max Tracking
// =============================================================================
//
// PROBLEM STATEMENT
// -----------------
// Implement a pipelined hardware softmax unit for attention score normalisation.
// The unit must process a stream of BF16 attention scores and produce normalised
// softmax weights using the numerically stable online softmax algorithm.
//
// ALGORITHM (Online Softmax)
// --------------------------
// Maintain running state (m, d) where:
//   m = running maximum of all scores seen so far
//   d = running sum of exp(score_j - m), corrected for updates to m
//
// For each new score s_j:
//   m_new = max(m, s_j)
//   d_new = exp(m - m_new) * d + exp(s_j - m_new)
//
// Final output for score s_j:
//   weight_j = exp(s_j - m_final) / d_final
//
// Note: this implementation uses a two-pass approach:
//   Pass 1 streams scores in, accumulates (m, d) using online algorithm.
//   Pass 2 re-reads buffered scores to compute final weights.
//
// EXP APPROXIMATION
// -----------------
// exp(x) is approximated using a segmented LUT over the range [-8, 0].
// Inputs outside this range are clamped:
//   x > 0  : clamped to 0 (should not occur after max subtraction)
//   x < -8 : output ≈ 0 (underflow to zero, contributing nothing to sum)
//
// INTERFACE
// ---------
// Scores are presented one per cycle with in_valid.
// in_last marks the final score in the sequence.
// The unit buffers all scores internally (up to MAX_SEQ_LEN).
// After in_last, it enters pass 2 and outputs weights one per cycle.
// out_valid is asserted for each output weight; out_last on the final one.
//
// PARAMETERS
// ----------
// MAX_SEQ_LEN  : Maximum number of scores (sequence length). Must be power of 2.
// EXP_LUT_BITS : Number of address bits for the exp LUT (LUT has 2^EXP_LUT_BITS entries).
//
// NOTE ON BF16
// ------------
// BF16 is represented as a 16-bit value: sign[15], exponent[14:7], mantissa[6:0].
// It shares the same exponent format as FP32. For simplicity, accumulation in this
// design uses a 32-bit fixed-point representation scaled to the range [-16, 16].
// A production design would use IEEE BF16/FP32 arithmetic units.
// =============================================================================

`default_nettype none
`timescale 1ns / 1ps

// -----------------------------------------------------------------------------
// Exp LUT module: approximates exp(x) for x in [-8, 0] using 256 entries.
// Input  : 8-bit index representing x mapped to [0, 255] for x in [-8, 0].
//          index = round((x + 8) * 255 / 8)
//          so index=0 -> x=-8 (exp≈0.000335), index=255 -> x=0 (exp=1.0)
// Output : 16-bit unsigned fixed-point Q1.15 representing exp(x) in [0, 1].
// -----------------------------------------------------------------------------
module exp_lut (
    input  wire        clk,
    input  wire [7:0]  index,   // x mapped to [0..255] for x in [-8..0]
    output reg  [15:0] exp_out  // Q1.15 fixed-point, exp(x) in [0,1]
);
    // LUT contents: exp_lut_data[i] = round(exp(-8 + i*8/255) * 32767)
    // First few values: exp(-8)=0.000335 -> 11; exp(-7)=0.000912 -> 29; ...
    // Last value: exp(0)=1.0 -> 32767
    // This ROM is 256 entries x 16 bits = 512 bytes.
    reg [15:0] lut_mem [0:255];

    // Initialise the LUT with pre-computed exp values.
    // In a real ASIC this would be a synthesised ROM from a $readmemh file.
    integer k;
    initial begin
        for (k = 0; k < 256; k = k + 1) begin
            // exp(-8 + k * 8.0/255) scaled to Q1.15
            // We approximate with the formula: val = 32767 * exp(-8 + k*8/255)
            // Actual values computed offline and loaded; this init is illustrative.
            // For simulation purposes we use $exp (non-synthesisable).
            lut_mem[k] = $rtoi(32767.0 * $exp(-8.0 + (k * 8.0) / 255.0));
        end
    end

    // Registered output: 1-cycle latency
    always_ff @(posedge clk) begin
        exp_out <= lut_mem[index];
    end
endmodule


// -----------------------------------------------------------------------------
// Fixed-point exp approximation wrapper.
// Input  : 16-bit signed fixed-point Q4.11 representing x in [-8, 0].
//          (4 integer bits + sign + 11 fractional bits)
// Output : 16-bit Q1.15 representing exp(x).
// Latency: 1 cycle (LUT registered output).
// -----------------------------------------------------------------------------
module exp_fixed (
    input  wire        clk,
    input  wire signed [15:0] x_fp,    // Q4.11, represents x = x_fp / 2048.0
    output wire        [15:0] exp_out  // Q1.15 result
);
    // Clamp x to [-8, 0]:
    // x_fp in Q4.11: -8.0 = -16384, 0.0 = 0
    wire signed [15:0] x_clamped;
    assign x_clamped = (x_fp > 16'sd0)   ? 16'sd0    :  // clamp at 0
                       (x_fp < -16'sd16384) ? -16'sd16384 :  // clamp at -8
                       x_fp;

    // Map x_clamped to LUT index [0..255]:
    // x in [-8, 0] maps to index (x + 8) * 255 / 8
    // x_clamped is Q4.11, so x + 8 = x_clamped + 16384 (adding 8 in Q4.11)
    // Then multiply by 255/8 and take top 8 bits.
    wire [15:0] x_shifted;
    assign x_shifted = x_clamped + 16'sd16384; // Now in [0, 16384] representing [0, 8]

    // Scale to [0, 255]: index = x_shifted * 255 / 16384 = x_shifted >> 6 (approx)
    // More precisely: x_shifted * 255 / 16384 = (x_shifted * 255) >> 14
    // Use truncation (floor) - adequate for LUT indexing
    wire [7:0] lut_index;
    assign lut_index = x_shifted[14:7]; // Top 8 bits of 15-bit range [0, 16384]

    exp_lut u_lut (
        .clk     (clk),
        .index   (lut_index),
        .exp_out (exp_out)
    );
endmodule


// -----------------------------------------------------------------------------
// Reciprocal unit: computes R ≈ 1/D using one Newton-Raphson iteration.
// Input  : 32-bit unsigned Q16.16 fixed-point (sum of exp values)
// Output : 32-bit unsigned Q0.32 fixed-point (reciprocal, in [0, 1] after scaling)
// Latency: 4 cycles
//
// Algorithm:
//   1. Initial estimate R0 from 8-bit LUT indexed by top 8 bits of D.
//   2. R1 = R0 * (2 - D * R0)  [one Newton iteration, doubles bits of accuracy]
// -----------------------------------------------------------------------------
module reciprocal_unit (
    input  wire        clk,
    input  wire        valid_in,
    input  wire [31:0] D,         // Q16.16 fixed-point denominator
    output reg         valid_out,
    output reg  [31:0] R          // Q0.32 reciprocal approximation
);
    // Initial estimate LUT: maps top 8 bits of D to 8-bit approximation of 1/D
    // scaled to Q0.8. In hardware this is an 8-bit addressed, 8-bit data ROM.
    reg [7:0] recip_lut [0:255];
    integer m;
    initial begin
        recip_lut[0] = 8'hFF; // guard: D≈0 -> maximum reciprocal
        for (m = 1; m < 256; m = m + 1) begin
            // top 8 bits of Q16.16 -> D in [1, 256] in integer units
            // 1/D scaled to Q0.8: round(256 / m)
            recip_lut[m] = $rtoi(256.0 / m) > 255 ? 8'hFF : $rtoi(256.0 / m);
        end
    end

    // Pipeline registers
    reg        valid_s1, valid_s2, valid_s3;
    reg [31:0] D_s1, D_s2, D_s3;
    reg [15:0] R0_s1, R0_s2;
    reg [31:0] D_times_R0;

    // Stage 1: LUT lookup for initial estimate
    wire [7:0] D_top8 = D[31:24];

    always_ff @(posedge clk) begin
        valid_s1 <= valid_in;
        D_s1     <= D;
        R0_s1    <= {recip_lut[D_top8], 8'b0}; // extend to Q0.16
    end

    // Stage 2: Compute D * R0 (32-bit * 16-bit multiply, take upper bits)
    always_ff @(posedge clk) begin
        valid_s2    <= valid_s1;
        D_s2        <= D_s1;
        R0_s2       <= R0_s1;
        // D is Q16.16, R0 is Q0.16: product is Q16.32, we want Q16.16
        D_times_R0  <= (D_s1 * {16'b0, R0_s1}) >> 16;
    end

    // Stage 3: Compute 2 - D*R0 (in Q16.16: 2.0 = 32'h00020000)
    reg [31:0] two_minus_DR0;
    always_ff @(posedge clk) begin
        valid_s3    <= valid_s2;
        D_s3        <= D_s2;
        two_minus_DR0 <= 32'h0002_0000 - D_times_R0;
    end

    // Stage 4: R1 = R0 * (2 - D*R0)
    always_ff @(posedge clk) begin
        valid_out <= valid_s3;
        // R0_s2 is Q0.16, two_minus_DR0 is Q16.16; product is Q16.32 -> take [47:16]
        R <= ({16'b0, R0_s2} * two_minus_DR0) >> 16;
    end
endmodule


// =============================================================================
// Main softmax pipeline module
// =============================================================================
module softmax_pipeline #(
    parameter int MAX_SEQ_LEN  = 512,   // Max sequence length (power of 2)
    parameter int EXP_LUT_BITS = 8      // LUT address bits (256 entries)
) (
    input  wire        clk,
    input  wire        rst_n,

    // --- Input port (Pass 1: score ingestion) ---
    input  wire        in_valid,    // Score is valid this cycle
    input  wire [15:0] in_score,    // BF16 attention score
    input  wire        in_last,     // Marks the last score in the sequence
    output wire        in_ready,    // Asserted when unit can accept a score

    // --- Output port (Pass 2: weight output) ---
    output reg         out_valid,   // Softmax weight is valid this cycle
    output reg  [15:0] out_weight,  // BF16 softmax weight
    output reg         out_last     // Marks the last output weight
);
    // -------------------------------------------------------------------------
    // Local types and parameters
    // -------------------------------------------------------------------------
    localparam int PTR_BITS  = $clog2(MAX_SEQ_LEN);
    // Q4.11 fixed-point for score representation: range [-16, 16]
    // BF16 scores from attention are in roughly [-10, 10]; Q4.11 is sufficient.
    localparam int SCORE_BITS = 16;  // Q4.11 signed
    // Q1.15 for exp output (range [0, 1])
    localparam int EXP_BITS   = 16;
    // Q16.16 for sum accumulator (can hold up to ~65535 exp values summing to ~512 max)
    localparam int ACC_BITS   = 32;

    // -------------------------------------------------------------------------
    // FSM states
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE    = 2'd0,  // Waiting for first score
        S_PASS1   = 2'd1,  // Ingesting scores, computing online max and sum
        S_RECIP   = 2'd2,  // Computing 1/d_final (reciprocal unit latency)
        S_PASS2   = 2'd3   // Re-reading buffer, computing final weights
    } state_t;

    state_t state, state_next;

    // -------------------------------------------------------------------------
    // Score buffer: stores all scores for pass 2 re-read
    // -------------------------------------------------------------------------
    reg signed [SCORE_BITS-1:0] score_buf [0:MAX_SEQ_LEN-1];
    reg [PTR_BITS-1:0] wr_ptr;   // Next write address (pass 1)
    reg [PTR_BITS-1:0] rd_ptr;   // Next read address (pass 2)
    reg [PTR_BITS-1:0] seq_len;  // Length of current sequence

    // -------------------------------------------------------------------------
    // Online max-and-sum state (Q4.11 for max, Q16.16 for sum)
    // -------------------------------------------------------------------------
    reg signed [SCORE_BITS-1:0] m_running;   // Running maximum, Q4.11
    reg        [ACC_BITS-1:0]   d_running;   // Running sum of exp, Q16.16
    reg        [ACC_BITS-1:0]   d_final;     // Final sum after pass 1 completes
    reg signed [SCORE_BITS-1:0] m_final;     // Final max after pass 1

    // -------------------------------------------------------------------------
    // BF16 to Q4.11 conversion (simplified: use top bits only)
    // BF16: s[15] exp[14:7] frac[6:0]
    // This is a simplified conversion for simulation; production would use
    // a proper IEEE floating-point to fixed-point converter.
    // -------------------------------------------------------------------------
    function automatic signed [SCORE_BITS-1:0] bf16_to_q4_11;
        input [15:0] bf16;
        logic        sign;
        logic [7:0]  exp_bits;
        logic [6:0]  frac_bits;
        logic [31:0] val_fp32;
        real         real_val;
        begin
            sign      = bf16[15];
            exp_bits  = bf16[14:7];
            frac_bits = bf16[6:0];
            // Convert BF16 mantissa to real and scale to Q4.11
            // Simplified: return 0 for subnormals and infinities
            if (exp_bits == 8'hFF || exp_bits == 8'h00)
                bf16_to_q4_11 = '0;
            else begin
                real_val = (1.0 + frac_bits / 128.0) *
                           (2.0 ** ($signed({1'b0, exp_bits}) - 127));
                if (sign) real_val = -real_val;
                // Clamp to Q4.11 range [-16, 16)
                if      (real_val >  15.999) real_val =  15.999;
                else if (real_val < -16.0)   real_val = -16.0;
                bf16_to_q4_11 = $rtoi(real_val * 2048.0); // * 2^11
            end
        end
    endfunction

    // -------------------------------------------------------------------------
    // Q4.11 to BF16 conversion (simplified)
    // -------------------------------------------------------------------------
    function automatic [15:0] q0_15_to_bf16;
        input [15:0] q0_15;  // Unsigned Q0.15 in [0, 1)
        real         val;
        integer      exp_val;
        logic [7:0]  exp_field;
        logic [6:0]  frac_field;
        begin
            val = q0_15 / 32768.0; // Divide by 2^15
            if (val == 0.0) begin
                q0_15_to_bf16 = 16'h0000;
            end else begin
                // Normalise: find leading 1
                exp_val   = -1;
                while (val < 1.0 && exp_val > -127) begin
                    val     = val * 2.0;
                    exp_val = exp_val - 1;
                end
                exp_field  = exp_val + 127;
                frac_field = $rtoi((val - 1.0) * 128.0);
                q0_15_to_bf16 = {1'b0, exp_field, frac_field};
            end
        end
    endfunction

    // -------------------------------------------------------------------------
    // Exp unit instance: computes exp of the current online update delta
    // -------------------------------------------------------------------------
    // Two exp units needed simultaneously in pass 1:
    //   exp_delta: exp(m_running - m_new)   -- rescaling factor for existing sum
    //   exp_score: exp(score - m_new)        -- new term to add to sum
    // For simplicity this design uses one exp unit and pipelines with 1-cycle stall.
    // A production design would instantiate two exp_fixed units in parallel.

    reg signed [SCORE_BITS-1:0] exp_arg_reg;
    reg        [EXP_BITS-1:0]   exp_result;
    wire       [EXP_BITS-1:0]   exp_result_wire;

    exp_fixed u_exp (
        .clk     (clk),
        .x_fp    (exp_arg_reg),
        .exp_out (exp_result_wire)
    );

    always_ff @(posedge clk) begin
        exp_result <= exp_result_wire;
    end

    // -------------------------------------------------------------------------
    // Reciprocal unit instance
    // -------------------------------------------------------------------------
    wire        recip_valid_in;
    wire        recip_valid_out;
    wire [31:0] recip_R;

    assign recip_valid_in = (state == S_PASS1) && in_valid && in_last;

    reciprocal_unit u_recip (
        .clk       (clk),
        .valid_in  (recip_valid_in),
        .D         (d_final),
        .valid_out (recip_valid_out),
        .R         (recip_R)
    );

    reg [31:0] R_stored;  // Stored reciprocal for use in pass 2

    // -------------------------------------------------------------------------
    // Control logic: accept scores only in PASS1 state
    // -------------------------------------------------------------------------
    assign in_ready = (state == S_PASS1) || (state == S_IDLE);

    // -------------------------------------------------------------------------
    // Pass 2 read: re-read score buffer and compute final weights
    // -------------------------------------------------------------------------
    wire signed [SCORE_BITS-1:0] rd_score;
    assign rd_score = score_buf[rd_ptr];

    // Compute exp(score - m_final) for pass 2 output
    reg signed [SCORE_BITS-1:0] p2_exp_arg;

    // -------------------------------------------------------------------------
    // FSM: sequential state register
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= state_next;
    end

    // -------------------------------------------------------------------------
    // FSM: next-state logic
    // -------------------------------------------------------------------------
    always_comb begin
        state_next = state;
        case (state)
            S_IDLE  : if (in_valid)                         state_next = S_PASS1;
            S_PASS1 : if (in_valid && in_last)              state_next = S_RECIP;
            S_RECIP : if (recip_valid_out)                  state_next = S_PASS2;
            S_PASS2 : if (rd_ptr == seq_len - 1'b1)        state_next = S_IDLE;
            default : state_next = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Pass 1 datapath: score buffering and online max/sum computation
    // -------------------------------------------------------------------------
    wire signed [SCORE_BITS-1:0] score_q4_11;
    assign score_q4_11 = bf16_to_q4_11(in_score);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr    <= '0;
            m_running <= -16'sd16384; // -8.0 in Q4.11 (minimum representable useful value)
            d_running <= '0;
            seq_len   <= '0;
            m_final   <= '0;
            d_final   <= '0;
        end else begin
            // Reset accumulators when a new sequence starts
            if (state == S_IDLE && in_valid) begin
                m_running <= -16'sd16384;
                d_running <= 32'h0000_0000;
                wr_ptr    <= '0;
            end

            if ((state == S_PASS1 || state == S_IDLE) && in_valid) begin
                // Buffer the incoming score
                score_buf[wr_ptr] <= score_q4_11;
                wr_ptr            <= wr_ptr + 1'b1;

                // Update running max
                if (score_q4_11 > m_running)
                    m_running <= score_q4_11;

                // Update running sum (simplified: one-cycle online update).
                // Full implementation would pipeline the exp computation and stall
                // here; for correctness in simulation we compute combinatorially.
                // delta = m_running - max(m_running, score_q4_11)
                // exp_delta * d_running + exp(score - new_max)
                // Note: this combinatorial block is non-synthesisable as written;
                // a pipelined version would register the exp output from u_exp.
                begin
                    automatic logic signed [SCORE_BITS-1:0] new_max;
                    automatic logic signed [SCORE_BITS-1:0] delta_arg;
                    automatic logic signed [SCORE_BITS-1:0] score_arg;
                    automatic real  exp_delta_real, exp_score_real;
                    automatic real  d_new_real;

                    new_max = (score_q4_11 > m_running) ? score_q4_11 : m_running;
                    delta_arg = m_running - new_max; // <= 0
                    score_arg = score_q4_11 - new_max; // <= 0

                    // Use real arithmetic for simulation; hardware would use exp_fixed units
                    exp_delta_real = $exp($itor(delta_arg) / 2048.0);
                    exp_score_real = $exp($itor(score_arg) / 2048.0);

                    d_new_real = (exp_delta_real * $itor(d_running) / 65536.0)
                                 + exp_score_real;
                    d_running <= $rtoi(d_new_real * 65536.0); // Back to Q16.16
                end

                // Capture final state on last score
                if (in_last) begin
                    seq_len <= wr_ptr + 1'b1;
                    m_final <= (score_q4_11 > m_running) ? score_q4_11 : m_running;
                    // d_final updated one cycle later; use a flag
                end
            end

            // Latch d_final one cycle after last score processed
            if (state == S_PASS1 && in_valid && in_last) begin
                d_final <= d_running; // Captured in next cycle when state->S_RECIP
            end
        end
    end

    // -------------------------------------------------------------------------
    // Reciprocal storage
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (recip_valid_out)
            R_stored <= recip_R;
    end

    // -------------------------------------------------------------------------
    // Pass 2 datapath: read buffered scores, compute final weights
    // -------------------------------------------------------------------------
    // Pipeline: rd_ptr -> exp_fixed (1 cycle) -> multiply by R_stored (1 cycle) -> output

    reg [PTR_BITS-1:0] rd_ptr_d1;       // Delayed read pointer for output alignment
    reg                rd_valid_d1, rd_last_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr    <= '0;
            out_valid <= 1'b0;
            out_weight<= 16'h0;
            out_last  <= 1'b0;
        end else begin
            // Advance read pointer in pass 2
            if (state == S_PASS2) begin
                // Drive exp_arg for the exp unit (registered, 1-cycle latency)
                exp_arg_reg <= rd_score - m_final; // Q4.11 subtraction
                rd_ptr      <= rd_ptr + 1'b1;
                rd_ptr_d1   <= rd_ptr;
                rd_valid_d1 <= 1'b1;
                rd_last_d1  <= (rd_ptr == seq_len - 1'b1);
            end else begin
                rd_ptr      <= '0;
                rd_valid_d1 <= 1'b0;
                rd_last_d1  <= 1'b0;
            end

            // One cycle after exp_arg is registered, exp_result_wire is valid.
            // Multiply exp result (Q1.15) by reciprocal R_stored (Q0.32):
            // product is Q1.47; we take bits [46:31] for Q1.15 result (softmax weight in [0,1]).
            if (rd_valid_d1) begin
                out_valid  <= 1'b1;
                out_weight <= q0_15_to_bf16(
                    ({17'b0, exp_result_wire} * R_stored) >> 31
                );
                out_last   <= rd_last_d1;
            end else begin
                out_valid  <= 1'b0;
                out_last   <= 1'b0;
            end
        end
    end

endmodule


// =============================================================================
// TESTBENCH
// =============================================================================
// Drives a sequence of 4 attention scores through the softmax pipeline and
// checks that the output weights sum to approximately 1.0.
//
// Test vector:
//   Scores (BF16 approximation): [-2.0, 0.0, 1.0, -1.0]
//   Expected softmax (reference):
//     x_shift = [-3.0, -1.0, 0.0, -2.0]  (subtract max=1.0)
//     exp     = [0.0498, 0.3679, 1.0, 0.1353]
//     sum     = 1.5530
//     weights = [0.0321, 0.2369, 0.6439, 0.0871]
//     sum of weights ≈ 1.0 (check)
// =============================================================================
`ifdef SIMULATION
module tb_softmax_pipeline;
    // Parameters
    localparam int MAX_SEQ_LEN = 16;
    localparam real CLK_PERIOD = 10.0; // 100 MHz

    // DUT signals
    logic        clk, rst_n;
    logic        in_valid, in_last, in_ready;
    logic [15:0] in_score;
    logic        out_valid, out_last;
    logic [15:0] out_weight;

    // Instantiate DUT
    softmax_pipeline #(
        .MAX_SEQ_LEN (MAX_SEQ_LEN),
        .EXP_LUT_BITS(8)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .in_score  (in_score),
        .in_last   (in_last),
        .in_ready  (in_ready),
        .out_valid (out_valid),
        .out_weight(out_weight),
        .out_last  (out_last)
    );

    // Clock generation
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Helper function: pack a real value as approximate BF16
    // (for test stimulus only)
    function automatic [15:0] real_to_bf16;
        input real val;
        logic        sign;
        integer      exp_val;
        real         mantissa;
        logic [7:0]  exp_field;
        logic [6:0]  frac_field;
        begin
            if (val == 0.0) begin
                real_to_bf16 = 16'h0000;
            end else begin
                sign = (val < 0.0);
                if (sign) val = -val;
                exp_val = 0;
                mantissa = val;
                while (mantissa >= 2.0) begin mantissa = mantissa / 2.0; exp_val++; end
                while (mantissa <  1.0) begin mantissa = mantissa * 2.0; exp_val--; end
                exp_field    = exp_val + 127;
                frac_field   = $rtoi((mantissa - 1.0) * 128.0);
                real_to_bf16 = {sign, exp_field, frac_field};
            end
        end
    endfunction

    // Helper: decode BF16 to real
    function automatic real bf16_to_real;
        input [15:0] bf16;
        begin
            if (bf16[14:7] == 8'h00) begin
                bf16_to_real = 0.0;
            end else begin
                bf16_to_real = (bf16[15] ? -1.0 : 1.0) *
                               (1.0 + bf16[6:0] / 128.0) *
                               (2.0 ** ($signed({1'b0, bf16[14:7]}) - 127));
            end
        end
    endfunction

    // Test sequence: BF16 encoding of [-2.0, 0.0, 1.0, -1.0]
    localparam int NUM_SCORES = 4;
    logic [15:0] test_scores [0:NUM_SCORES-1];

    initial begin
        test_scores[0] = real_to_bf16(-2.0);
        test_scores[1] = real_to_bf16( 0.0);
        test_scores[2] = real_to_bf16( 1.0);
        test_scores[3] = real_to_bf16(-1.0);
    end

    // Collected output weights
    real collected_weights [0:NUM_SCORES-1];
    int  out_count;
    real weight_sum;

    // Test stimulus
    initial begin
        $display("=== Softmax Pipeline Testbench ===");
        $display("Input scores: -2.0, 0.0, 1.0, -1.0");
        $display("Expected weights (approx): 0.0321, 0.2369, 0.6439, 0.0871");
        $display("");

        // Reset
        rst_n    = 1'b0;
        in_valid = 1'b0;
        in_last  = 1'b0;
        in_score = 16'h0;
        out_count = 0;
        @(posedge clk);
        @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Send scores
        $display("[TB] Starting Pass 1: sending %0d scores", NUM_SCORES);
        for (int i = 0; i < NUM_SCORES; i++) begin
            @(negedge clk); // drive before rising edge
            in_valid = 1'b1;
            in_score = test_scores[i];
            in_last  = (i == NUM_SCORES - 1);
            $display("[TB] Sending score[%0d] = BF16 0x%04h (approx %.4f)",
                     i, test_scores[i], bf16_to_real(test_scores[i]));
            @(posedge clk);
        end
        @(negedge clk);
        in_valid = 1'b0;
        in_last  = 1'b0;

        // Wait for outputs
        $display("[TB] Waiting for Pass 2 outputs...");
        fork
            begin : timeout_block
                repeat (1000) @(posedge clk);
                $display("[TB] TIMEOUT waiting for outputs");
                disable wait_block;
            end
            begin : wait_block
                wait (out_count == NUM_SCORES);
                disable timeout_block;
            end
        join

        // Verify results
        $display("");
        $display("[TB] Collected softmax weights:");
        weight_sum = 0.0;
        for (int i = 0; i < NUM_SCORES; i++) begin
            $display("  weight[%0d] = %.6f", i, collected_weights[i]);
            weight_sum += collected_weights[i];
        end
        $display("  Sum of weights = %.6f (expected ≈ 1.0)", weight_sum);

        if (weight_sum > 0.95 && weight_sum < 1.05)
            $display("[TB] PASS: weights sum to approximately 1.0");
        else
            $display("[TB] FAIL: weight sum out of tolerance");

        $display("");
        $finish;
    end

    // Output collection
    always_ff @(posedge clk) begin
        if (out_valid) begin
            collected_weights[out_count] = bf16_to_real(out_weight);
            out_count++;
            $display("[TB] Output weight[%0d] = BF16 0x%04h (%.6f)",
                     out_count-1, out_weight, collected_weights[out_count-1]);
        end
    end

    // Simulation timeout guard
    initial begin
        #100000;
        $display("[TB] Global timeout");
        $finish;
    end

endmodule
`endif // SIMULATION
