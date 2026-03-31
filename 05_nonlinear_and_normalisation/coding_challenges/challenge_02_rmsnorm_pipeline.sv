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
//                  Uses the fast initial estimate:  y0 = magic_const >> (msb/2)
//                  Then refines: y_{n+1} = y_n * (1.5 - 0.5 * mean * y_n^2)
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
//   NR_ITERS  : Newton-Raphson iterations (2 gives ~24-bit accuracy)
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

    logic [NR_W-1:0]   s3_y;                    // current NR estimate
    logic [NR_W-1:0]   s3_half_mean;            // 0.5 * mean_sq in NR domain
    logic [NR_ITERS:0] s3_iter;                 // one-hot iteration counter
    logic [NR_W-1:0]   s3_inv_rms;              // final 1/RMS result

    // -----------------------------------------------------------------------
    // Stage 4 signals — output multiply
    // -----------------------------------------------------------------------
    logic [$clog2(DIM)-1:0]    s4_ptr;          // output element index
    logic signed [2*NR_W-1:0]  s4_prod_tmp;    // x_i * inv_rms (before truncation)
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
    // Shifting right by DIM_LOG2 gives the mean in Q8.16.
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) s2_mean_sq <= '0;
        else if (state == ST_MEAN)
            s2_mean_sq <= s1_acc_latch >> DIM_LOG2;
    end

    // -----------------------------------------------------------------------
    // Stage 3 — 1/sqrt(mean_sq) via Newton-Raphson
    //
    // The fast inverse-square-root iteration (Quake-style) converges as:
    //   y_{n+1} = y_n * (3/2 - x/2 * y_n^2)
    //
    // where x = mean_sq.  We represent everything in Q2.30 (NR_W bits) to
    // retain enough precision through several iterations.
    //
    // Initial estimate strategy:
    //   Find the position of the leading '1' bit in mean_sq (call it 'msb').
    //   Then  1/sqrt(mean_sq) ≈ 2^( -(msb-NR_FRAC)/2 )  as a starting point.
    //   This gives a relative error < 50% which NR iterations rapidly reduce.
    //
    // INTERVIEW NOTE: A production design would use a small seed LUT (e.g.,
    // index the top 6 bits of the mantissa) to get a much tighter initial
    // estimate and require only one NR iteration for sufficient accuracy.
    // -----------------------------------------------------------------------
    logic [5:0] s3_msb_pos; // position of leading 1 in mean_sq

    // Leading-one detector (combinational)
    always_comb begin
        s3_msb_pos = '0;
        for (int i = ACC_W-1; i >= 0; i--) begin
            if (s2_mean_sq[i] && (s3_msb_pos == 0))
                s3_msb_pos = 6'(i);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_y         <= '0;
            s3_half_mean <= '0;
            s3_iter      <= 1; // start at bit 0 (iteration 0 = init)
            s3_inv_rms   <= '0;
        end else begin
            if (state == ST_MEAN) begin
                // On the cycle we compute mean_sq, also set up the NR seed.
                // mean_sq is in Q(8+DIM_LOG2-DIM_LOG2).16 = Q8.16.
                // Convert to NR_FRAC domain: scale by 2^(NR_FRAC - Q_FRAC).
                //
                // Initial estimate: y0 = 2^( floor((NR_FRAC - msb_of_mean)/2) )
                // msb_of_mean is in terms of the Q8.16 representation.
                // After the >> DIM_LOG2 shift the mean is still Q8.16 aligned
                // inside s2_mean_sq; effective bit position = s3_msb_pos.
                logic [NR_W-1:0] y0;
                logic [5:0]      half_exp;
                // msb_pos is the integer bit position inside the ACC_W word.
                // The Q-point is at position Q_FRAC (16).
                // So the value magnitude ~ 2^(msb_pos - Q_FRAC).
                // 1/sqrt(val) ~ 2^( -(msb_pos - Q_FRAC)/2 )
                //             = 2^( (Q_FRAC - msb_pos)/2 ) in natural units.
                // In our NR_FRAC-bit representation this becomes bit position:
                //   NR_FRAC + (Q_FRAC - msb_pos)/2
                half_exp = 6'(NR_FRAC) + 6'((6'(Q_FRAC) - s3_msb_pos) >>> 1);
                y0 = NR_W'(1) << half_exp; // power-of-2 seed
                s3_y    <= y0;
                // Pre-compute 0.5 * mean_sq in NR domain.
                // mean_sq (Q8.16) -> NR domain by scaling up by 2^(NR_FRAC-Q_FRAC)=14
                // Then take half: >> 1
                s3_half_mean <= NR_W'(s2_mean_sq >> DIM_LOG2) << (NR_FRAC - Q_FRAC) >> 1;
                s3_iter <= 1; // reset shift reg
            end else if (state == ST_INVSQRT && !s3_iter[NR_ITERS]) begin
                // -----------------------------------------------------------
                // One Newton-Raphson step per clock cycle.
                //
                // y_{n+1} = y_n * (3/2 - half_mean * y_n^2)
                //
                // All values in Q2.30 (NR_W bits).
                // Multiplications produce 2*NR_W bits; we truncate back to
                // NR_W by taking bits [2*NR_W-1 : NR_FRAC].
                // -----------------------------------------------------------
                logic [2*NR_W-1:0] yn_sq;       // y_n^2    (Q4.60)
                logic [2*NR_W-1:0] hm_yn_sq;    // half_mean * y_n^2  (Q4.60 * scale)
                logic [NR_W-1:0]   correction;  // 3/2 - hm_yn_sq  (Q2.30)
                logic [NR_W-1:0]   three_halves;

                three_halves = NR_W'(3) << (NR_FRAC - 1); // 1.5 in Q2.30

                yn_sq      = s3_y * s3_y;                             // Q4.60
                hm_yn_sq   = (s3_half_mean * NR_W'(yn_sq >> NR_FRAC)) >> NR_FRAC; // back to NR_W
                correction = three_halves - NR_W'(hm_yn_sq);         // Q2.30
                // y_{n+1} in Q2.30
                s3_y       <= NR_W'((s3_y * correction) >> NR_FRAC);
                s3_iter    <= s3_iter << 1; // advance iteration counter
            end else if (state == ST_INVSQRT && s3_iter[NR_ITERS]) begin
                // Latch final result
                s3_inv_rms <= s3_y;
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

                // Step A: xi * inv_rms
                // xi is signed Q8.16 (25-bit), inv_rms is unsigned Q2.30 (34-bit)
                // Product is Q10.46, 59-bit; normalise back to Q8.16 by >> 30
                s4_prod_tmp = $signed({1'b0, s3_inv_rms}) * xi;  // treat inv_rms as unsigned
                automatic logic signed [DATA_W-1:0] normed =
                    DATA_W'(s4_prod_tmp >>> NR_FRAC); // keep Q8.16 portion

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
    // Input vector [1.0, 2.0, 3.0, 4.0] in Q8.16
    logic signed [DATA_W-1:0] test_vec[0:DIM-1];
    logic signed [DATA_W-1:0] test_gamma[0:DIM-1];

    // Reference expected outputs (float, for comparison)
    real ref_out[0:DIM-1];

    // Collected outputs
    real       collected[0:DIM-1];
    int        out_idx;
    int        fail_count;

    // Tolerance: accept up to 1% relative error from fixed-point rounding
    localparam real TOL_FRAC = 0.02; // 2% — Newton-Raphson may have ~1% error

    initial begin
        // Initialise test vector in Q8.16
        test_vec[0]   = DATA_W'($rtoi(1.0 * ONE));
        test_vec[1]   = DATA_W'($rtoi(2.0 * ONE));
        test_vec[2]   = DATA_W'($rtoi(3.0 * ONE));
        test_vec[3]   = DATA_W'($rtoi(4.0 * ONE));
        test_gamma[0] = DATA_W'($rtoi(1.0 * ONE));
        test_gamma[1] = DATA_W'($rtoi(1.0 * ONE));
        test_gamma[2] = DATA_W'($rtoi(1.0 * ONE));
        test_gamma[3] = DATA_W'($rtoi(1.0 * ONE));

        // Reference (IEEE 754 double precision)
        begin
            real sum_sq, rms;
            sum_sq = 0.0;
            for (int i = 0; i < DIM; i++)
                sum_sq += (i+1.0) * (i+1.0);
            rms = $sqrt(sum_sq / real'(DIM));
            $display("Reference RMS = %0.6f", rms);
            for (int i = 0; i < DIM; i++) begin
                ref_out[i] = (i+1.0) / rms * 1.0; // gamma = 1
                $display("  ref_out[%0d] = %0.6f  (Q8.16 int ~ %0d)",
                         i, ref_out[i], $rtoi(ref_out[i] * ONE));
            end
        end

        // Reset sequence
        rst_n    = 0;
        in_valid = 0;
        in_data  = '0;
        in_gamma = '0;
        out_idx  = 0;
        fail_count = 0;
        repeat(4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Drive input vector — one element per cycle
        $display("\n--- Driving input vector ---");
        for (int i = 0; i < DIM; i++) begin
            @(negedge clk); // drive on falling edge, sample on rising
            in_data  = test_vec[i];
            in_gamma = test_gamma[i];
            in_valid = 1;
            $display("  Sending x[%0d] = %0.4f  (raw = %0d)",
                     i, real'(in_data) / ONE, in_data);
        end
        @(negedge clk);
        in_valid = 0;

        // Wait for outputs
        $display("\n--- Collecting outputs ---");
        // Generous timeout: DIM + pipeline overhead
        for (int timeout = 0; timeout < 200; timeout++) begin
            @(posedge clk);
            if (out_valid) begin
                collected[out_idx] = real'($signed(out_data)) / ONE;
                $display("  out[%0d] = %0.6f  (Q8.16 raw = %0d)  ref = %0.6f",
                         out_idx, collected[out_idx], out_data, ref_out[out_idx]);
                out_idx++;
                if (out_idx == DIM) break;
            end
        end

        // Check results
        $display("\n--- Checking accuracy ---");
        for (int i = 0; i < DIM; i++) begin
            real err, rel_err;
            err     = collected[i] - ref_out[i];
            if (err < 0.0) err = -err;
            rel_err = (ref_out[i] > 0.0) ? err / ref_out[i] : err;
            if (rel_err > TOL_FRAC) begin
                $display("  FAIL [%0d]: got %0.6f  expected %0.6f  rel_err=%0.4f%%",
                         i, collected[i], ref_out[i], rel_err*100.0);
                fail_count++;
            end else begin
                $display("  PASS [%0d]: got %0.6f  expected %0.6f  rel_err=%0.4f%%",
                         i, collected[i], ref_out[i], rel_err*100.0);
            end
        end

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
