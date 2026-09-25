// =============================================================================
// Challenge 02 — Pipelined RMSNorm Unit
// =============================================================================
//
// BACKGROUND
// ----------
// RMSNorm (Root Mean Square Layer Normalisation) is used in LLMs such as
// LLaMA and Mistral as a cheaper alternative to LayerNorm. It normalises
// each token vector x of dimension D by its root-mean-square:
//
//   RMSNorm(x)_i = (x_i / RMS(x)) * gamma_i
//
// where  RMS(x) = sqrt( (1/D) * sum_i( x_i^2 ) )
//
// This module implements that computation as a 4-stage pipeline using
// fixed-point Q8.16 arithmetic (1 sign bit, 8 integer bits, 16 fraction
// bits — 25 bits total, sign-magnitude stored as 2's-complement 25-bit).
//
// For clarity in this exercise the data width is kept as a parameter so
// reviewers can easily see how every stage maps to the algorithm.
//
// FIXED-POINT CONVENTION
// ----------------------
//   Q_INT.Q_FRAC  =>  value = stored_integer / 2^Q_FRAC
//
//   DATA_W = 1 + Q_INT + Q_FRAC  (sign + integer + fraction bits)
//
// The default parameters use Q8.16 (25-bit) for the input/output data and
// Q2.30 (33-bit) for intermediate high-precision accumulators.
//
// PIPELINE STAGES
// ---------------
//   Stage 1 (S1):  Accumulate sum-of-squares over the input vector.
//                  Runs over DIM clock cycles, then latches the result.
//   Stage 2 (S2):  Compute mean by right-shifting the sum by log2(DIM).
//                  (Requires DIM to be a power of two for a simple shift.)
//   Stage 3 (S3):  Approximate 1/sqrt(mean) via Newton-Raphson iteration.
//                  Range reduction: write mean = m * 4^k with m in [1, 4),
//                  seed y0 ~ 1/sqrt(m) from a 4-entry table (< 11% error),
//                  refine y_{n+1} = y_n * (1.5 - 0.5 * m * y_n^2), then
//                  scale the result by 2^-k.
//   Stage 4 (S4):  Multiply each original input element by inv_rms and gamma.
//                  Runs over DIM clock cycles to emit the output vector.
//
// INTERVIEW TASKS
// ---------------
//   1. Complete the Newton-Raphson iteration logic in the S3 block.
//   2. Implement the valid/ready handshake so back-pressure is handled.
//   3. Extend to non-power-of-2 DIM (hint: store a reciprocal of DIM in ROM).
//   4. Quantify the maximum fixed-point error versus a floating-point model.
//   5. Describe how you would pipeline Stage 1 to accept a new vector every
//      cycle rather than every DIM cycles (stream folding / ping-pong buffering).
//
// =============================================================================

`timescale 1ns/1ps

// ---------------------------------------------------------------------------
// Top-level RMSNorm pipeline
// ---------------------------------------------------------------------------
// Parameters
//   DIM       : vector length (must be a power of 2 for Stage 2 shift trick)
//   DATA_W    : total width of a Q8.16 sample (should be 25)
//   Q_FRAC    : fractional bits (16 for Q8.16)
//   ACC_W     : accumulator width for sum-of-squares (must hold DIM * max_sq)
//   NR_ITERS  : Newton-Raphson iterations (seed error < 11%; 2 iterations
//               give ~3e-4 relative error, 3 give ~1e-7)
// ---------------------------------------------------------------------------
module rmsnorm_pipeline #(
    parameter int unsigned DIM     = 8,    // vector dimension (power-of-2)
    parameter int unsigned DATA_W  = 25,   // Q8.16 fixed-point word width
    parameter int unsigned Q_FRAC  = 16,   // fractional bits in DATA_W
    parameter int unsigned ACC_W   = 48,   // sum-of-squares accumulator width
    parameter int unsigned NR_ITERS = 2    // Newton-Raphson refinement steps
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // --- Input vector interface (one element per cycle) ---
    input  logic signed [DATA_W-1:0]   in_data,    // x_i  (Q8.16)
    input  logic signed [DATA_W-1:0]   in_gamma,   // gamma_i (Q8.16, learned scale)
    input  logic                       in_valid,   // asserted for DIM consecutive cycles
    output logic                       in_ready,   // back-pressure to source

    // --- Output vector interface (one element per cycle) ---
    output logic signed [DATA_W-1:0]   out_data,   // normalised y_i (Q8.16)
    output logic                       out_valid
);

    // -----------------------------------------------------------------------
    // Local constants
    // -----------------------------------------------------------------------
    localparam int unsigned DIM_LOG2 = $clog2(DIM); // shift amount for mean

    // -----------------------------------------------------------------------
    // Input buffer — stores the full input vector so Stage 4 can re-read it
    // -----------------------------------------------------------------------
    logic signed [DATA_W-1:0]  vec_buf   [0:DIM-1]; // raw input samples
    logic signed [DATA_W-1:0]  gamma_buf [0:DIM-1]; // corresponding gammas
    logic [$clog2(DIM)-1:0]    in_ptr;              // write pointer

    // -----------------------------------------------------------------------
    // Pipeline state machine
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        ST_IDLE,     // waiting for first valid input
        ST_LOAD,     // Stage 1: accumulating sum-of-squares
        ST_MEAN,     // Stage 2: compute mean (1 cycle)
        ST_INVSQRT,  // Stage 3: Newton-Raphson (NR_ITERS * 2 cycles)
        ST_OUTPUT    // Stage 4: emit normalised output elements
    } state_t;

    state_t state, state_nxt;

    // -----------------------------------------------------------------------
    // Stage 1 signals — sum-of-squares accumulator
    // -----------------------------------------------------------------------
    logic signed [ACC_W-1:0]   s1_acc;          // running sum of x_i^2
    logic        [ACC_W-1:0]   s1_acc_latch;    // final accumulated value
    logic signed [2*DATA_W-1:0] s1_sq;          // x_i * x_i (full precision)

    // -----------------------------------------------------------------------
    // Stage 2 signals — mean of squares
    // -----------------------------------------------------------------------
    logic [ACC_W-1:0]          s2_mean_sq;      // sum_sq / DIM

    // -----------------------------------------------------------------------
    // Stage 3 signals — inverse square root (Newton-Raphson)
    //
    // We work in a wider fixed-point to keep precision through iterations.
    // The intermediate values are in Q2.30 (33-bit unsigned):
    //   y   approximates  1/sqrt(mean_sq)
    //   The iteration:  y_{n+1} = y_n * (3/2 - (mean_sq/2) * y_n^2)
    // -----------------------------------------------------------------------
    localparam int unsigned NR_W = 34;          // Q2.30 +  guard bit
    localparam int unsigned NR_FRAC = 30;       // fractional bits in NR domain

    // 1/RMS itself needs more integer bits than the NR domain: the smallest
    // non-zero mean (2^-16) gives 1/RMS = 2^8. INV_W holds Q9.30 unsigned.
    localparam int unsigned INV_W = NR_FRAC + 10;

    logic [NR_W-1:0]   s3_y;                    // current NR estimate of 1/sqrt(m)
    logic [NR_W-1:0]   s3_half_mean;            // 0.5 * m in NR domain, m in [1, 4)
    logic signed [6:0] s3_k;                    // range-reduction exponent: mean = m * 4^k
    logic [NR_ITERS:0] s3_iter;                 // one-hot iteration counter
    logic [INV_W-1:0]  s3_inv_rms;              // final 1/RMS result (Q9.30)

    // -----------------------------------------------------------------------
    // Stage 4 signals — output multiply
    // -----------------------------------------------------------------------
    logic [$clog2(DIM)-1:0]    s4_ptr;          // output element index
    logic signed [INV_W+DATA_W:0] s4_prod_tmp; // x_i * inv_rms (before truncation)
    logic signed [2*DATA_W-1:0] s4_out_tmp;    // * gamma_i (before truncation)

    // -----------------------------------------------------------------------
    // in_ready: accept input only during LOAD state
    // -----------------------------------------------------------------------
    assign in_ready = (state == ST_LOAD) || (state == ST_IDLE);

    // -----------------------------------------------------------------------
    // State register
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= ST_IDLE;
        else        state <= state_nxt;
    end

    // -----------------------------------------------------------------------
    // Next-state logic
    // -----------------------------------------------------------------------
    always_comb begin
        state_nxt = state;
        unique case (state)
            ST_IDLE:
                if (in_valid) state_nxt = ST_LOAD;

            ST_LOAD:
                // Transition to mean when the last element has been latched
                if (in_valid && (in_ptr == DIM_LOG2'(DIM - 1)))
                    state_nxt = ST_MEAN;

            ST_MEAN:
                state_nxt = ST_INVSQRT;

            ST_INVSQRT:
                // s3_iter is a shift register; done when the MSB reaches 1
                if (s3_iter[NR_ITERS]) state_nxt = ST_OUTPUT;

            ST_OUTPUT:
                if (s4_ptr == $clog2(DIM)'(DIM - 1))
                    state_nxt = ST_IDLE;

            default: state_nxt = ST_IDLE;
        endcase
    end

    // -----------------------------------------------------------------------
    // Stage 1 — accumulate sum of squares
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_acc        <= '0;
            in_ptr        <= '0;
            s1_acc_latch  <= '0;
        end else begin
            unique case (state)
                ST_IDLE: begin
                    s1_acc <= '0;
                    in_ptr <= '0;
                    if (in_valid) begin
                        // Capture first element immediately
                        vec_buf[0]   <= in_data;
                        gamma_buf[0] <= in_gamma;
                        // x_i^2 — multiply signed by itself; sign cancels
                        s1_sq = in_data * in_data;
                        s1_acc <= ACC_W'(s1_sq >>> Q_FRAC); // keep Q8.16 scaling
                        in_ptr <= 1;
                    end
                end

                ST_LOAD: begin
                    if (in_valid) begin
                        vec_buf[in_ptr]   <= in_data;
                        gamma_buf[in_ptr] <= in_gamma;
                        // Compute x_i^2 and accumulate.
                        // Squaring a Q8.16 number gives Q16.32; shift right by
                        // Q_FRAC (16) to normalise back to Q24.16 in the acc.
                        s1_sq = in_data * in_data;
                        s1_acc <= s1_acc + ACC_W'(s1_sq >>> Q_FRAC);
                        in_ptr <= in_ptr + 1;
                        // Latch when last element arrives
                        if (in_ptr == DIM_LOG2'(DIM - 1))
                            s1_acc_latch <= s1_acc + ACC_W'(s1_sq >>> Q_FRAC);
                    end
                end

                default: ; // hold
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Stage 2 — divide sum by DIM (power-of-2 right shift)
    //
    // s1_acc_latch holds  sum( x_i^2 )  in Q(8+DIM_LOG2).16 format.
    // Shifting right by DIM_LOG2 gives the mean in Q8.16. This is a pure
    // shift, so it is combinational: Stage 3 consumes it in ST_MEAN, the cycle
    // after s1_acc_latch is written.
    // -----------------------------------------------------------------------
    assign s2_mean_sq = s1_acc_latch >> DIM_LOG2;

    // -----------------------------------------------------------------------
    // Stage 3 — 1/sqrt(mean_sq) via Newton-Raphson
    //
    // The inverse-square-root iteration converges as:
    //   y_{n+1} = y_n * (3/2 - m/2 * y_n^2)
    // but only for 0 < y_0 < sqrt(3/m), and quadratically only once y_0 is
    // close. So we range-reduce first:
    //
    //   p    = position of the leading 1 in mean_sq (Q8.16)
    //   e    = p - Q_FRAC                  so mean_sq in [2^e, 2^(e+1))
    //   k    = floor(e / 2)                so m = mean_sq * 4^-k is in [1, 4)
    //   1/sqrt(mean_sq) = 1/sqrt(m) * 2^-k
    //
    // m is represented in Q2.30 (NR_W bits), where it always fits. The seed
    // comes from a 4-entry table indexed by the parity of e (m in [1,2) or
    // [2,4)) and the bit just below the leading 1 (lower/upper half of that
    // interval). Its worst-case relative error is under 11%, so two NR
    // iterations reach ~3e-4 and three reach ~1e-7.
    // -----------------------------------------------------------------------
    logic [5:0] s3_msb_pos; // position of leading 1 in mean_sq

    // Leading-one detector (combinational)
    always_comb begin
        s3_msb_pos = '0;
        for (int i = 0; i < ACC_W; i++) begin
            if (s2_mean_sq[i])
                s3_msb_pos = 6'(i);
        end
    end

    // Seed table: approx 1/sqrt(m) over each quarter-octave, in Q2.30
    function automatic logic [NR_W-1:0] nr_seed(input logic odd_e, input logic next_bit);
        unique case ({odd_e, next_bit})
            2'b00: return NR_W'(64'd966367642);   // 0.90  for m in [1.0, 1.5)
            2'b01: return NR_W'(64'd816043786);   // 0.76  for m in [1.5, 2.0)
            2'b10: return NR_W'(64'd687194767);   // 0.64  for m in [2.0, 3.0)
            default: return NR_W'(64'd579820584); // 0.54  for m in [3.0, 4.0)
        endcase
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_y         <= '0;
            s3_half_mean <= '0;
            s3_k         <= '0;
            s3_iter      <= 1; // start at bit 0 (iteration 0 = init)
            s3_inv_rms   <= '0;
        end else begin
            if (state == ST_MEAN) begin
                // Range reduction and seed
                int signed    e, k, shift;
                logic [63:0]  m_fp;     // m in Q2.30 (wide for the shift)
                logic         next_bit;

                e     = int'(s3_msb_pos) - int'(Q_FRAC);
                k     = e >>> 1;                               // floor(e/2)
                // m = mean_sq * 2^(NR_FRAC - Q_FRAC - 2k)
                shift = int'(NR_FRAC) - int'(Q_FRAC) - 2 * k;
                if (shift >= 0) m_fp = 64'(s2_mean_sq) << shift;
                else            m_fp = 64'(s2_mean_sq) >> (-shift);
                next_bit = (s3_msb_pos > 0) ? s2_mean_sq[s3_msb_pos - 1] : 1'b0;

                s3_y         <= nr_seed(1'(e - 2 * k), next_bit);
                s3_half_mean <= NR_W'(m_fp >> 1);
                s3_k         <= 7'(k);
                s3_iter      <= 1; // reset shift reg
            end else if (state == ST_INVSQRT && !s3_iter[NR_ITERS]) begin
                // -----------------------------------------------------------
                // One Newton-Raphson step per clock cycle.
                //
                // y_{n+1} = y_n * (3/2 - half_m * y_n^2)
                //
                // All values in Q2.30 (NR_W bits).
                // Multiplications produce 2*NR_W bits; we truncate back to
                // NR_W by shifting right by NR_FRAC.
                // -----------------------------------------------------------
                logic [2*NR_W-1:0] yn_sq;       // y_n^2    (Q4.60)
                logic [2*NR_W-1:0] hm_yn_sq;    // half_m * y_n^2
                logic [NR_W-1:0]   correction;  // 3/2 - hm_yn_sq  (Q2.30)
                logic [NR_W-1:0]   three_halves;
                logic [2*NR_W-1:0] y_next;      // y_n * correction (Q4.60)

                three_halves = NR_W'(3) << (NR_FRAC - 1); // 1.5 in Q2.30

                yn_sq      = s3_y * s3_y;                             // Q4.60
                hm_yn_sq   = (s3_half_mean * NR_W'(yn_sq >> NR_FRAC)) >> NR_FRAC; // back to NR_W
                correction = three_halves - NR_W'(hm_yn_sq);         // Q2.30
                // y_{n+1} in Q2.30. The product must be formed at full width
                // first: inside a cast, s3_y * correction would be
                // self-determined at NR_W bits and overflow.
                y_next     = s3_y * correction;
                s3_y       <= NR_W'(y_next >> NR_FRAC);
                s3_iter    <= s3_iter << 1; // advance iteration counter
            end else if (state == ST_INVSQRT && s3_iter[NR_ITERS]) begin
                // Undo the range reduction: 1/RMS = y * 2^-k
                if (s3_k >= 0) s3_inv_rms <= INV_W'(s3_y) >> s3_k;
                else           s3_inv_rms <= INV_W'(s3_y) << (-s3_k);
            end
        end
    end

    // -----------------------------------------------------------------------
    // Stage 4 — output element: y_i = x_i * inv_rms * gamma_i
    //
    // inv_rms is in Q2.30; x_i and gamma_i are in Q8.16.
    // x_i * inv_rms  -> Q10.46; truncate to Q8.16 by taking [46+8-1 : 30].
    // result * gamma -> Q16.32; truncate to Q8.16 by taking [32+8-1 : 16].
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s4_ptr    <= '0;
            out_data  <= '0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            if (state == ST_OUTPUT) begin
                // Retrieve buffered input and gamma for this element
                automatic logic signed [DATA_W-1:0] xi    = vec_buf[s4_ptr];
                automatic logic signed [DATA_W-1:0] gi    = gamma_buf[s4_ptr];
                logic signed [DATA_W-1:0]           normed;  // x_i * inv_rms in Q8.16

                // Step A: xi * inv_rms
                // xi is signed Q8.16 (25-bit), inv_rms is unsigned Q2.30 (34-bit)
                // Product is Q10.46, 59-bit; normalise back to Q8.16 by >> 30
                s4_prod_tmp = $signed({1'b0, s3_inv_rms}) * xi;  // treat inv_rms as unsigned
                normed = DATA_W'(s4_prod_tmp >>> NR_FRAC); // keep Q8.16 portion

                // Step B: normed * gamma_i
                // Both Q8.16 (25-bit signed); product Q16.32, 50-bit
                // Normalise back to Q8.16 by >> Q_FRAC
                s4_out_tmp = normed * gi;
                out_data  <= DATA_W'(s4_out_tmp >>> Q_FRAC);
                out_valid <= 1'b1;

                s4_ptr <= s4_ptr + 1;
            end else begin
                s4_ptr <= '0;
            end
        end
    end

endmodule


// =============================================================================
// Testbench — rmsnorm_pipeline_tb
// =============================================================================
// Test vector (DIM=4, Q8.16):
//   x = [1.0, 2.0, 3.0, 4.0]
//   gamma = [1.0, 1.0, 1.0, 1.0]  (identity scale)
//
// RMS(x) = sqrt( (1+4+9+16)/4 ) = sqrt(7.5) ≈ 2.7386
// Expected output:
//   y[0] = 1.0 / 2.7386 ≈ 0.3651
//   y[1] = 2.0 / 2.7386 ≈ 0.7303
//   y[2] = 3.0 / 2.7386 ≈ 1.0954
//   y[3] = 4.0 / 2.7386 ≈ 1.4606
//
// In Q8.16 integer representation:
//   1.0   -> 32'd65536   (1 << 16)
//   2.0   -> 32'd131072
//   3.0   -> 32'd196608
//   4.0   -> 32'd262144
//   0.3651 ~ 23924
//   0.7303 ~ 47849
//   1.0954 ~ 71774
//   1.4606 ~ 95699
// =============================================================================
module rmsnorm_pipeline_tb;

    // -----------------------------------------------------------------------
    // Parameters matching the DUT
    // -----------------------------------------------------------------------
    localparam int DIM    = 4;
    localparam int DATA_W = 25;
    localparam int Q_FRAC = 16;
    localparam real ONE   = 2.0 ** Q_FRAC; // Q8.16 scale factor = 65536

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    logic                       clk;
    logic                       rst_n;
    logic signed [DATA_W-1:0]   in_data;
    logic signed [DATA_W-1:0]   in_gamma;
    logic                       in_valid;
    logic                       in_ready;
    logic signed [DATA_W-1:0]   out_data;
    logic                       out_valid;

    // -----------------------------------------------------------------------
    // DUT instantiation
    // -----------------------------------------------------------------------
    rmsnorm_pipeline #(
        .DIM     (DIM),
        .DATA_W  (DATA_W),
        .Q_FRAC  (Q_FRAC),
        .NR_ITERS(2)
    ) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_data  (in_data),
        .in_gamma (in_gamma),
        .in_valid (in_valid),
        .in_ready (in_ready),
        .out_data (out_data),
        .out_valid(out_valid)
    );

    // -----------------------------------------------------------------------
    // Clock generation — 10 ns period
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Test stimulus
    // -----------------------------------------------------------------------
    // Each vector is driven one element per cycle, then the DIM outputs are
    // collected and compared against an IEEE 754 double-precision reference.
    // A result passes if it is within 4 LSB or 0.1% of the reference.
    localparam real TOL_LSB  = 4.0;
    localparam real TOL_FRAC = 0.001;

    int fail_count;

    task automatic run_vector(input string name, input real x[DIM], input real g[DIM]);
        real sum_sq, rms, ref_out, got, err_lsb, rel_err;
        int  out_idx;
        real collected[DIM];

        sum_sq = 0.0;
        for (int i = 0; i < DIM; i++) sum_sq += x[i] * x[i];
        rms = $sqrt(sum_sq / real'(DIM));
        $display("\n--- Vector '%s': RMS = %0.6f ---", name, rms);

        // Drive input vector — one element per cycle
        for (int i = 0; i < DIM; i++) begin
            @(negedge clk); // drive on falling edge, sample on rising
            in_data  = DATA_W'($rtoi(x[i] * ONE));
            in_gamma = DATA_W'($rtoi(g[i] * ONE));
            in_valid = 1;
        end
        @(negedge clk);
        in_valid = 0;

        // Collect outputs (generous timeout: DIM + pipeline overhead)
        out_idx = 0;
        for (int timeout = 0; timeout < 200 && out_idx < DIM; timeout++) begin
            @(posedge clk); #1;  // sample after the DUT's NBA updates
            if (out_valid) begin
                collected[out_idx] = real'($signed(out_data)) / ONE;
                out_idx++;
            end
        end
        if (out_idx < DIM) begin
            $display("  FAIL: only %0d/%0d outputs received", out_idx, DIM);
            fail_count++;
        end

        for (int i = 0; i < out_idx; i++) begin
            ref_out = x[i] / rms * g[i];
            got     = collected[i];
            err_lsb = (got - ref_out) * ONE; if (err_lsb < 0.0) err_lsb = -err_lsb;
            rel_err = (ref_out != 0.0) ? err_lsb / ONE / ((ref_out > 0.0) ? ref_out : -ref_out) : 0.0;
            if (err_lsb > TOL_LSB && rel_err > TOL_FRAC) begin
                $display("  FAIL [%0d]: got %9.6f  expected %9.6f  err=%6.1f LSB (%0.4f%%)",
                         i, got, ref_out, err_lsb, rel_err*100.0);
                fail_count++;
            end else begin
                $display("  PASS [%0d]: got %9.6f  expected %9.6f  err=%6.1f LSB (%0.4f%%)",
                         i, got, ref_out, err_lsb, rel_err*100.0);
            end
        end
    endtask

    initial begin
        // Reset sequence
        rst_n    = 0;
        in_valid = 0;
        in_data  = '0;
        in_gamma = '0;
        fail_count = 0;
        repeat(4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // mean = 7.5  -> k = 1 (odd exponent, upper half of the octave)
        run_vector("ramp",      '{1.0, 2.0, 3.0, 4.0},     '{1.0, 1.0, 1.0, 1.0});
        // mean ~ 3350 -> k = 5, mixed signs and non-unit gamma
        run_vector("large",     '{100.0, -50.0, 0.25, 30.0}, '{1.0, 0.5, 2.0, -1.0});
        // mean ~ 0.28 -> k = -1 (negative exponent)
        run_vector("small",     '{0.25, -0.5, 0.75, 0.5},  '{1.0, 1.0, 1.0, 1.0});
        // mean = 1.0 exactly -> k = 0, seed at the bottom of its interval
        run_vector("unit",      '{1.0, -1.0, 1.0, -1.0},   '{0.5, 0.5, 0.5, 0.5});

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
        #50000;
        $display("ERROR: simulation timeout");
        $finish;
    end

    // -----------------------------------------------------------------------
    // Optional waveform dump
    // -----------------------------------------------------------------------
    initial begin
        $dumpfile("rmsnorm_pipeline.vcd");
        $dumpvars(0, rmsnorm_pipeline_tb);
    end

endmodule
