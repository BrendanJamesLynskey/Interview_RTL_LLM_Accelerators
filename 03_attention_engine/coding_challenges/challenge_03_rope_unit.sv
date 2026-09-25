// =============================================================================
// Challenge 03: RoPE (Rotary Position Embedding) Computation Unit
// =============================================================================
//
// PROBLEM STATEMENT
// -----------------
// Implement a hardware unit that applies Rotary Position Embeddings (RoPE) to
// a query or key vector. The unit uses a pre-computed sin/cos LUT indexed by
// (position, dimension_pair) and applies the rotation:
//
//   q'[2i]   = q[2i]   * cos(pos * theta_i) - q[2i+1] * sin(pos * theta_i)
//   q'[2i+1] = q[2i]   * sin(pos * theta_i) + q[2i+1] * cos(pos * theta_i)
//
// for i = 0 .. HEAD_DIM/2 - 1.
//
// DATAPATH SUMMARY
// ----------------
// The vector is stored in a register file with two banks:
//   - Bank A: even-indexed elements q[0], q[2], q[4], ..., q[HEAD_DIM-2]
//   - Bank B: odd-indexed  elements q[1], q[3], q[5], ..., q[HEAD_DIM-1]
//
// Both banks are indexed by pair_index i. PAIR_ENGINES rotation cells process
// PAIR_ENGINES pairs in parallel each cycle. The full HEAD_DIM/2 pairs are
// processed in (HEAD_DIM/2) / PAIR_ENGINES cycles (4 cycles for HEAD_DIM=128,
// PAIR_ENGINES=16).
//
// SIN/COS LUT
// -----------
// A single LUT is indexed by (position, pair_index). For a given position, all
// HEAD_DIM/2 (sin, cos) pairs for that position can be pre-computed and held in
// a row buffer. The LUT is pre-loaded with values for all supported positions.
//
// For simplicity this design pre-loads a single position's sin/cos values into
// a "position row buffer" at the start of each operation via the lut_load_*
// interface. In hardware, a larger LUT SRAM would be addressed automatically.
//
// ARITHMETIC
// ----------
// Input vector elements and sin/cos values are in signed INT16 (Q1.14 format):
//   value = bits / 16384.0   (range [-2, 2), sufficient for normalised vectors)
//
// The rotation multiply produces Q2.28 intermediates (32-bit signed).
// After the subtract/add and right-shifting by 14, the output is Q1.14 INT16.
// Overflow from the addition step (q[2i]^2 + q[2i+1]^2 = 1 for unit vectors)
// is prevented because the rotation preserves vector magnitude.
//
// PIPELINE STAGES (per batch of PAIR_ENGINES pairs)
// --------------------------------------------------
//   Cycle 1 (combinational, then registered in pipe_q_*): read the vector banks
//            and sin/cos row buffer, four multiplies per cell (q_e*cos, q_o*sin,
//            q_e*sin, q_o*cos), subtract/add, arithmetic right shift by 14
//            (truncating), saturate to INT16.
//   Cycle 2: write the registered batch into the output register file.
// One batch enters per cycle, so a rotation takes CYCLES_PER_ROTATE + 1 cycles
// of busy; done pulses the cycle after the last batch has been written.
// A deeper pipeline would split cycle 1 at the multipliers for timing.
//
// PARAMETERS
// ----------
// HEAD_DIM     : Dimension of the Q/K vector (must be even; e.g., 128)
// PAIR_ENGINES : Number of rotation cells (parallelism; must divide HEAD_DIM/2)
// MAX_SEQ_LEN  : Maximum supported position index
// =============================================================================

`default_nettype none
`timescale 1ns / 1ps

// =============================================================================
// One 2D rotation cell (combinatorial)
// Computes the rotation for one (q_even, q_odd) pair.
// Inputs/outputs are signed Q1.14 (INT16).
// Internal arithmetic uses Q2.28 (INT32).
// =============================================================================
module rotation_cell (
    // Inputs: signed Q1.14
    input  wire signed [15:0] q_even,       // q[2i]
    input  wire signed [15:0] q_odd,        // q[2i+1]
    input  wire signed [15:0] cos_val,      // cos(angle) in Q1.14
    input  wire signed [15:0] sin_val,      // sin(angle) in Q1.14
    // Outputs: signed Q1.14 (saturated)
    output wire signed [15:0] q_even_rot,   // q'[2i]   = q_e*cos - q_o*sin
    output wire signed [15:0] q_odd_rot     // q'[2i+1] = q_e*sin + q_o*cos
);
    // Q1.14 * Q1.14 = Q2.28 (32-bit signed)
    wire signed [31:0] prod_ee, prod_os, prod_es, prod_oe;

    assign prod_ee = q_even * cos_val;   // q_e * cos
    assign prod_os = q_odd  * sin_val;   // q_o * sin
    assign prod_es = q_even * sin_val;   // q_e * sin
    assign prod_oe = q_odd  * cos_val;   // q_o * cos

    // Sum and difference in Q2.28 (no overflow since |q|<=1, |sin/cos|<=1)
    wire signed [31:0] sum_even, sum_odd;
    assign sum_even = prod_ee - prod_os;  // q'_e in Q2.28
    assign sum_odd  = prod_es + prod_oe;  // q'_o in Q2.28

    // Convert back to Q1.14: right-shift by 14 bits.
    // Saturate to [-32768, 32767] in case of extreme values.
    wire signed [31:0] even_shifted, odd_shifted;
    assign even_shifted = sum_even >>> 14;
    assign odd_shifted  = sum_odd  >>> 14;

    // Saturating clamp to INT16 range
    assign q_even_rot = (even_shifted > 32'sd32767)  ? 16'sd32767  :
                        (even_shifted < -32'sd32768) ? -16'sd32768 :
                        even_shifted[15:0];

    assign q_odd_rot  = (odd_shifted  > 32'sd32767)  ? 16'sd32767  :
                        (odd_shifted  < -32'sd32768) ? -16'sd32768 :
                        odd_shifted[15:0];

endmodule


// =============================================================================
// Sin/Cos LUT Row Buffer
// Stores sin and cos values for all HEAD_DIM/2 pairs at one position.
// Loaded via a sequential write interface before the rotation begins.
// In a full design this would be a SRAM pre-loaded for each new position.
// =============================================================================
module sincos_row_buffer #(
    parameter int HEAD_DIM    = 128,
    parameter int PAIR_ENGINES = 16  // Number of pairs read per cycle
) (
    input  wire clk,
    input  wire rst_n,

    // Load interface: write one (sin, cos) pair per cycle
    input  wire        load_valid,               // Write enable for loading
    input  wire [$clog2(HEAD_DIM/2)-1:0] load_idx, // Pair index to load
    input  wire signed [15:0] load_sin,          // sin value (Q1.14)
    input  wire signed [15:0] load_cos,          // cos value (Q1.14)

    // Read interface: read PAIR_ENGINES consecutive pairs per cycle
    input  wire [$clog2(HEAD_DIM/2)-1:0] rd_base_idx, // First pair index to read
    output wire signed [15:0] rd_sin [0:PAIR_ENGINES-1],
    output wire signed [15:0] rd_cos [0:PAIR_ENGINES-1]
);
    localparam int NUM_PAIRS = HEAD_DIM / 2;

    reg signed [15:0] sin_mem [0:NUM_PAIRS-1];
    reg signed [15:0] cos_mem [0:NUM_PAIRS-1];

    // Sequential load (one pair per cycle during setup)
    always_ff @(posedge clk) begin
        if (load_valid) begin
            sin_mem[load_idx] <= load_sin;
            cos_mem[load_idx] <= load_cos;
        end
    end

    // Parallel read: PAIR_ENGINES entries starting from rd_base_idx
    // These are continuous assignments (combinatorial, same-cycle read)
    genvar g;
    generate
        for (g = 0; g < PAIR_ENGINES; g++) begin : gen_rd
            assign rd_sin[g] = sin_mem[rd_base_idx + g[$clog2(HEAD_DIM/2)-1:0]];
            assign rd_cos[g] = cos_mem[rd_base_idx + g[$clog2(HEAD_DIM/2)-1:0]];
        end
    endgenerate

endmodule


// =============================================================================
// Main RoPE unit
// =============================================================================
module rope_unit #(
    parameter int HEAD_DIM     = 128,
    parameter int PAIR_ENGINES = 16,    // Pairs processed per cycle
    parameter int MAX_SEQ_LEN  = 4096
) (
    input  wire        clk,
    input  wire        rst_n,

    // -----------------------------------------------------------------------
    // Sin/Cos LUT load interface
    // Before starting a rotation, load the (sin, cos) pairs for the target
    // position into the row buffer. Provide one pair per cycle.
    // -----------------------------------------------------------------------
    input  wire        lut_load_valid,
    input  wire [$clog2(HEAD_DIM/2)-1:0] lut_load_idx,
    input  wire signed [15:0] lut_load_sin,
    input  wire signed [15:0] lut_load_cos,

    // -----------------------------------------------------------------------
    // Vector input interface
    // Load the Q or K vector to be rotated, one element per cycle.
    // Elements are loaded in order: q[0], q[1], q[2], ..., q[HEAD_DIM-1].
    // -----------------------------------------------------------------------
    input  wire        vec_load_valid,
    input  wire [$clog2(HEAD_DIM)-1:0] vec_load_idx,
    input  wire signed [15:0] vec_load_data,  // Q1.14 signed

    // -----------------------------------------------------------------------
    // Operation control
    // Assert start_rotate for one cycle to trigger the rotation pipeline.
    // The unit processes all HEAD_DIM/2 pairs in (HEAD_DIM/2)/PAIR_ENGINES
    // pipeline cycles. busy is asserted during processing.
    // -----------------------------------------------------------------------
    input  wire        start_rotate,   // Pulse: begin rotation of loaded vector
    output wire        busy,           // Rotation in progress
    output wire        done,           // Pulsed for one cycle when rotation complete

    // -----------------------------------------------------------------------
    // Vector output
    // After done is asserted, the rotated vector is available in the output
    // register file, read one element per cycle via out_idx/out_data.
    // -----------------------------------------------------------------------
    input  wire [$clog2(HEAD_DIM)-1:0] out_idx,
    output wire signed [15:0]          out_data
);

    localparam int NUM_PAIRS = HEAD_DIM / 2;
    localparam int CYCLES_PER_ROTATE = NUM_PAIRS / PAIR_ENGINES; // e.g., 128/2/16 = 4

    // -----------------------------------------------------------------------
    // Input vector register file: two banks (even and odd elements)
    // -----------------------------------------------------------------------
    reg signed [15:0] q_bank_even [0:NUM_PAIRS-1]; // q[0], q[2], ..., q[HEAD_DIM-2]
    reg signed [15:0] q_bank_odd  [0:NUM_PAIRS-1]; // q[1], q[3], ..., q[HEAD_DIM-1]

    // Load from sequential interface (the caller must not load while busy)
    always_ff @(posedge clk) begin
        if (vec_load_valid) begin
            if (!vec_load_idx[0]) // Even index
                q_bank_even[vec_load_idx[$clog2(HEAD_DIM)-1:1]] <= vec_load_data;
            else                   // Odd index
                q_bank_odd [vec_load_idx[$clog2(HEAD_DIM)-1:1]] <= vec_load_data;
        end
    end

    // -----------------------------------------------------------------------
    // Output vector register file
    // -----------------------------------------------------------------------
    reg signed [15:0] out_bank_even [0:NUM_PAIRS-1];
    reg signed [15:0] out_bank_odd  [0:NUM_PAIRS-1];

    // Read port (combinatorial)
    assign out_data = out_idx[0] ?
                      out_bank_odd [out_idx[$clog2(HEAD_DIM)-1:1]] :
                      out_bank_even[out_idx[$clog2(HEAD_DIM)-1:1]];

    // -----------------------------------------------------------------------
    // Sin/Cos row buffer instance
    // -----------------------------------------------------------------------
    wire signed [15:0] rd_sin [0:PAIR_ENGINES-1];
    wire signed [15:0] rd_cos [0:PAIR_ENGINES-1];

    // Base index for current PAIR_ENGINES-wide read
    reg [$clog2(NUM_PAIRS)-1:0] pair_base;

    sincos_row_buffer #(
        .HEAD_DIM    (HEAD_DIM),
        .PAIR_ENGINES(PAIR_ENGINES)
    ) u_sincos (
        .clk         (clk),
        .rst_n       (rst_n),
        .load_valid  (lut_load_valid),
        .load_idx    (lut_load_idx),
        .load_sin    (lut_load_sin),
        .load_cos    (lut_load_cos),
        .rd_base_idx (pair_base),
        .rd_sin      (rd_sin),
        .rd_cos      (rd_cos)
    );

    // -----------------------------------------------------------------------
    // PAIR_ENGINES rotation cells (combinatorial)
    // -----------------------------------------------------------------------
    wire signed [15:0] rot_q_even [0:PAIR_ENGINES-1];
    wire signed [15:0] rot_q_odd  [0:PAIR_ENGINES-1];

    genvar gi;
    generate
        for (gi = 0; gi < PAIR_ENGINES; gi++) begin : gen_cells
            rotation_cell u_cell (
                .q_even    (q_bank_even[pair_base + gi[$clog2(NUM_PAIRS)-1:0]]),
                .q_odd     (q_bank_odd [pair_base + gi[$clog2(NUM_PAIRS)-1:0]]),
                .cos_val   (rd_cos[gi]),
                .sin_val   (rd_sin[gi]),
                .q_even_rot(rot_q_even[gi]),
                .q_odd_rot (rot_q_odd[gi])
            );
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Pipeline registers: capture rotation_cell outputs
    // (rotation_cell is combinatorial; add one register stage here)
    // -----------------------------------------------------------------------
    reg signed [15:0] pipe_q_even [0:PAIR_ENGINES-1];
    reg signed [15:0] pipe_q_odd  [0:PAIR_ENGINES-1];
    reg [$clog2(NUM_PAIRS)-1:0] pipe_base; // Which pair base index these outputs correspond to
    reg pipe_valid;                          // Outputs are valid this cycle

    // -----------------------------------------------------------------------
    // Control FSM
    // -----------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE    = 2'd0,
        S_ROTATE  = 2'd1,
        S_FLUSH   = 2'd2   // One extra cycle to capture last pipeline register
    } state_t;

    state_t ctrl_state;
    reg [$clog2(CYCLES_PER_ROTATE):0] cycle_count; // Counts 0..CYCLES_PER_ROTATE-1

    // done is registered: it rises the cycle after FLUSH, i.e. once the last
    // batch has been written into the output banks and can be read
    reg done_reg;
    assign busy = (ctrl_state != S_IDLE);
    assign done = done_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state  <= S_IDLE;
            pair_base   <= '0;
            cycle_count <= '0;
            pipe_valid  <= 1'b0;
            done_reg    <= 1'b0;
        end else begin
            pipe_valid <= 1'b0; // Default: no valid pipeline output
            done_reg   <= (ctrl_state == S_FLUSH);

            case (ctrl_state)
                // ---------------------------------------------------------
                S_IDLE : begin
                    if (start_rotate) begin
                        pair_base   <= '0;
                        cycle_count <= '0;
                        ctrl_state  <= S_ROTATE;
                    end
                end

                // ---------------------------------------------------------
                // ROTATE: each cycle, the PAIR_ENGINES rotation cells compute
                // results for pairs [pair_base .. pair_base+PAIR_ENGINES-1].
                // We register their outputs in the pipeline register stage and
                // write them to the output banks in the NEXT cycle.
                // ---------------------------------------------------------
                S_ROTATE : begin
                    // Capture rotation outputs from this cycle's pair_base
                    pipe_base  <= pair_base;
                    pipe_valid <= 1'b1;

                    // Advance to next batch of pairs
                    if (cycle_count == CYCLES_PER_ROTATE - 1) begin
                        // Last batch of pairs
                        ctrl_state  <= S_FLUSH;
                        cycle_count <= '0;
                    end else begin
                        pair_base   <= pair_base + PAIR_ENGINES[$clog2(NUM_PAIRS)-1:0];
                        cycle_count <= cycle_count + 1'b1;
                    end

                    // Register the rotation cell outputs
                    for (int p = 0; p < PAIR_ENGINES; p++) begin
                        pipe_q_even[p] <= rot_q_even[p];
                        pipe_q_odd [p] <= rot_q_odd[p];
                    end
                end

                // ---------------------------------------------------------
                // FLUSH: write the last pipeline-registered batch to output banks
                // pipe_valid is still asserted from the ROTATE->FLUSH transition
                // ---------------------------------------------------------
                S_FLUSH : begin
                    pipe_valid <= 1'b0;
                    ctrl_state <= S_IDLE;
                end

                default : ctrl_state <= S_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Write registered outputs to output bank
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (pipe_valid) begin
            for (int p = 0; p < PAIR_ENGINES; p++) begin
                out_bank_even[pipe_base + p[$clog2(NUM_PAIRS)-1:0]] <= pipe_q_even[p];
                out_bank_odd [pipe_base + p[$clog2(NUM_PAIRS)-1:0]] <= pipe_q_odd[p];
            end
        end
    end

endmodule


// =============================================================================
// TESTBENCH
// =============================================================================
// Tests the RoPE unit with a known input and verifiable output.
//
// Test vector (HEAD_DIM=8, 4 pairs, PAIR_ENGINES=2):
//   Input Q = [16384, 0, 0, 16384, -16384, 0, 0, -16384]
//             In Q1.14: [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]
//
//   Position = 0: all angles = 0 => cos=1, sin=0 for all pairs.
//   Expected output = input unchanged (rotation by 0 is identity).
//
//   Verification: out[i] == in[i] for all i.
//
// Every test compares all HEAD_DIM outputs with a bit-exact integer model of
// the datapath and with the ideal real-arithmetic rotation (tolerance derived
// in check_rotation); Test 4 uses real RoPE angles pos * 10000^(-2i/d) at
// several positions and Test 5 checks output saturation. done is checked to be
// a one-cycle pulse after which the whole result is readable.
//
// Extended test: position=1 with theta_0=pi/4 (pair 0 angle = pi/4):
//   q[0] = 1.0, q[1] = 0.0
//   cos(pi/4) = sin(pi/4) = 1/sqrt(2) ≈ 0.7071 -> Q1.14 = 11585
//   q'[0] = 1.0*cos - 0.0*sin = 1/sqrt(2) ≈ 0.7071 -> 11585
//   q'[1] = 1.0*sin + 0.0*cos = 1/sqrt(2) ≈ 0.7071 -> 11585
// =============================================================================
`ifdef SIMULATION
module tb_rope_unit;

    // Absolute value of a real. SystemVerilog has no $abs system function
    // (it is a simulator extension), so define one for portability.
    function automatic real abs_real(real x);
        return (x < 0.0) ? -x : x;
    endfunction

    // Parameters (small for simulation)
    localparam int HEAD_DIM     = 8;
    localparam int PAIR_ENGINES = 2;
    localparam int MAX_SEQ_LEN  = 16;
    localparam int NUM_PAIRS    = HEAD_DIM / 2;  // = 4
    localparam real CLK_PERIOD  = 10.0;

    // DUT signals
    logic        clk, rst_n;
    logic        lut_load_valid;
    logic [$clog2(NUM_PAIRS)-1:0] lut_load_idx;
    logic signed [15:0] lut_load_sin, lut_load_cos;
    logic        vec_load_valid;
    logic [$clog2(HEAD_DIM)-1:0] vec_load_idx;
    logic signed [15:0] vec_load_data;
    logic        start_rotate;
    logic        busy, done;
    logic [$clog2(HEAD_DIM)-1:0] out_idx;
    logic signed [15:0] out_data;

    // DUT instantiation
    rope_unit #(
        .HEAD_DIM    (HEAD_DIM),
        .PAIR_ENGINES(PAIR_ENGINES),
        .MAX_SEQ_LEN (MAX_SEQ_LEN)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .lut_load_valid (lut_load_valid),
        .lut_load_idx   (lut_load_idx),
        .lut_load_sin   (lut_load_sin),
        .lut_load_cos   (lut_load_cos),
        .vec_load_valid (vec_load_valid),
        .vec_load_idx   (vec_load_idx),
        .vec_load_data  (vec_load_data),
        .start_rotate   (start_rotate),
        .busy           (busy),
        .done           (done),
        .out_idx        (out_idx),
        .out_data       (out_data)
    );

    // Values actually loaded into the DUT (after Q1.14 conversion), the
    // output read back, and the last element as read in the done cycle
    logic signed [15:0] q_int   [0:HEAD_DIM-1];
    logic signed [15:0] sin_int [0:NUM_PAIRS-1];
    logic signed [15:0] cos_int [0:NUM_PAIRS-1];
    logic signed [15:0] out_int [0:HEAD_DIM-1];
    logic signed [15:0] last_at_done;
    int errors = 0;

    // Clock generation
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Helper: real to Q1.14 (saturating)
    function automatic signed [15:0] to_q1_14;
        input real val;
        real scaled;
        begin
            scaled = val * 16384.0;
            if      (scaled >  32767.0) to_q1_14 =  16'sd32767;
            else if (scaled < -32768.0) to_q1_14 = -16'sd32768;
            else                         to_q1_14 = $rtoi(scaled);
        end
    endfunction

    // Helper: Q1.14 to real
    function automatic real from_q1_14;
        input signed [15:0] val;
        begin
            from_q1_14 = $itor(val) / 16384.0;
        end
    endfunction

    // Task: load sin/cos for all NUM_PAIRS pairs
    task automatic load_sincos(
        input real angles [0:NUM_PAIRS-1]
    );
        for (int i = 0; i < NUM_PAIRS; i++) begin
            @(negedge clk);
            lut_load_valid <= 1'b1;
            lut_load_idx   <= i[$clog2(NUM_PAIRS)-1:0];
            lut_load_sin   <= to_q1_14($sin(angles[i]));
            lut_load_cos   <= to_q1_14($cos(angles[i]));
            sin_int[i]      = to_q1_14($sin(angles[i]));
            cos_int[i]      = to_q1_14($cos(angles[i]));
            @(posedge clk);
        end
        @(negedge clk);
        lut_load_valid <= 1'b0;
    endtask

    // Task: load input vector
    task automatic load_vector(input real vec [0:HEAD_DIM-1]);
        for (int i = 0; i < HEAD_DIM; i++) begin
            @(negedge clk);
            vec_load_valid <= 1'b1;
            vec_load_idx   <= i[$clog2(HEAD_DIM)-1:0];
            vec_load_data  <= to_q1_14(vec[i]);
            q_int[i]        = to_q1_14(vec[i]);
            @(posedge clk);
        end
        @(negedge clk);
        vec_load_valid <= 1'b0;
    endtask

    // Task: trigger rotation and wait for done
    task automatic do_rotate();
        @(negedge clk);
        start_rotate <= 1'b1;
        @(posedge clk);
        @(negedge clk);
        start_rotate <= 1'b0;
        // Wait for done signal
        fork
            begin : to_blk
                repeat(200) @(posedge clk);
                $display("[TB] FAIL: TIMEOUT waiting for done");
                errors++;
                disable w_blk;
            end
            begin : w_blk
                wait(done);
                disable to_blk;
                // The whole result must already be readable while done is high
                out_idx = HEAD_DIM - 1;
                #1 last_at_done = out_data;
            end
        join
        // done must be a one-cycle pulse, and the output banks are
        // complete from the cycle done is seen (read_output starts later)
        @(posedge clk);
        #1;
        if (done !== 1'b0) begin
            $display("[TB]   FAIL: done high for more than one cycle");
            errors++;
        end
    endtask

    // Task: read and display output vector
    task automatic read_output(output real out_vec [0:HEAD_DIM-1]);
        for (int i = 0; i < HEAD_DIM; i++) begin
            @(negedge clk);
            out_idx <= i[$clog2(HEAD_DIM)-1:0];
            @(posedge clk);
            // out_data is combinatorial from out_idx, available same cycle
            out_vec[i] = from_q1_14(out_data);
            out_int[i] = out_data;
        end
    endtask

    // -----------------------------------------------------------------------
    // Reference model and checking
    // -----------------------------------------------------------------------
    // Bit-exact model of rotation_cell: 64-bit products and sums, arithmetic
    // shift right by 14 (floor), saturate to INT16
    function automatic logic signed [15:0] sat_shift(input longint v);
        longint s;
        s = v >>> 14;
        if (s >  32767) return 16'sd32767;
        if (s < -32768) return -16'sd32768;
        return 16'(s);
    endfunction

    // Compare every output element with (a) the bit-exact model and (b) the
    // real-arithmetic rotation of the loaded input by the ideal angle.
    // (b) tolerance: sin/cos are truncated to Q1.14 (error < 2^-14 each), so
    // with |q_e|,|q_o| < 2 the products are off by < 2*2^-14 each, and the
    // output shift truncates by < 2^-14: total < 5 * 2^-14 = 3.1e-4.
    // Saturated outputs are only checked bit-exactly.
    task automatic check_rotation(input string name, input real angles [0:NUM_PAIRS-1]);
        logic signed [15:0] exp_e, exp_o;
        real  qe, qo, re, ro;
        int   bad;
        real  max_err;
        bad = 0;
        max_err = 0.0;
        for (int i = 0; i < NUM_PAIRS; i++) begin
            exp_e = sat_shift(longint'(q_int[2*i]) * cos_int[i] - longint'(q_int[2*i+1]) * sin_int[i]);
            exp_o = sat_shift(longint'(q_int[2*i]) * sin_int[i] + longint'(q_int[2*i+1]) * cos_int[i]);
            if (out_int[2*i] !== exp_e || out_int[2*i+1] !== exp_o) begin
                $display("[TB]   FAIL %s pair %0d: got (%0d, %0d), bit-exact model (%0d, %0d)",
                         name, i, out_int[2*i], out_int[2*i+1], exp_e, exp_o);
                bad++;
            end
            qe = from_q1_14(q_int[2*i]);
            qo = from_q1_14(q_int[2*i+1]);
            re = qe * $cos(angles[i]) - qo * $sin(angles[i]);
            ro = qe * $sin(angles[i]) + qo * $cos(angles[i]);
            if (exp_e != 16'sd32767 && exp_e != -16'sd32768 && abs_real(from_q1_14(out_int[2*i]) - re) > max_err)
                max_err = abs_real(from_q1_14(out_int[2*i]) - re);
            if (exp_o != 16'sd32767 && exp_o != -16'sd32768 && abs_real(from_q1_14(out_int[2*i+1]) - ro) > max_err)
                max_err = abs_real(from_q1_14(out_int[2*i+1]) - ro);
        end
        if (last_at_done !== out_int[HEAD_DIM-1]) begin
            $display("[TB]   FAIL %s: out[%0d] read during done = %0d, later %0d",
                     name, HEAD_DIM-1, last_at_done, out_int[HEAD_DIM-1]);
            bad++;
        end
        if (max_err > 5.0 / 16384.0) begin
            $display("[TB]   FAIL %s: max error vs real rotation %.6f > %.6f", name, max_err, 5.0 / 16384.0);
            bad++;
        end
        if (bad == 0)
            $display("[TB] PASS %s (bit-exact; max error vs real rotation %.6f)", name, max_err);
        errors += bad;
    endtask

    // -----------------------------------------------------------------------
    // Test execution
    // -----------------------------------------------------------------------
    real input_vec   [0:HEAD_DIM-1];
    real angles_pos0 [0:NUM_PAIRS-1];
    real angles_pos1 [0:NUM_PAIRS-1];
    real output_vec  [0:HEAD_DIM-1];
    int  positions [5] = '{1, 7, 100, 1000, 4095};

    initial begin
        $display("=== RoPE Unit Testbench ===");
        $display("HEAD_DIM=%0d, PAIR_ENGINES=%0d, NUM_PAIRS=%0d", HEAD_DIM, PAIR_ENGINES, NUM_PAIRS);
        $display("");

        // Reset
        rst_n          = 1'b0;
        lut_load_valid = 1'b0;
        vec_load_valid = 1'b0;
        start_rotate   = 1'b0;
        out_idx        = '0;
        repeat(4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // -------------------------------------------------------------------
        // TEST 1: Identity rotation (position=0, all angles=0)
        // Expected: output == input exactly
        // -------------------------------------------------------------------
        $display("--- Test 1: Identity rotation (position=0, all angles=0) ---");

        // All angles = 0 => cos=1, sin=0
        for (int i = 0; i < NUM_PAIRS; i++) angles_pos0[i] = 0.0;

        // Input: alternating 1.0 and 0.0 (with one negative and one
        // non-integer element so sign handling is visible)
        for (int i = 0; i < HEAD_DIM; i++)
            input_vec[i] = (i % 2 == 0) ? 1.0 : 0.0;
        input_vec[0] = 0.9999;
        input_vec[4] = -1.0;

        load_sincos(angles_pos0);
        load_vector(input_vec);
        do_rotate();
        read_output(output_vec);
        check_rotation("Test 1 (identity)", angles_pos0);
        for (int i = 0; i < HEAD_DIM; i++)
            if (out_int[i] !== q_int[i]) begin
                $display("[TB]   FAIL Test 1: out[%0d] = %0d, input %0d", i, out_int[i], q_int[i]);
                errors++;
            end

        // -------------------------------------------------------------------
        // TEST 2: Rotation by pi/4 on pair 0, zero angle on other pairs
        // Input: q[0]=1.0, q[1]=0.0, q[2..7]=0.0
        // Expected: q'[0] = cos(pi/4) ≈ 0.7071, q'[1] = sin(pi/4) ≈ 0.7071
        // -------------------------------------------------------------------
        $display("");
        $display("--- Test 2: Pair 0 rotation by pi/4, others zero ---");

        for (int i = 0; i < NUM_PAIRS; i++)
            angles_pos1[i] = (i == 0) ? 3.14159265358979 / 4.0 : 0.0;

        for (int i = 0; i < HEAD_DIM; i++) input_vec[i] = 0.0;
        input_vec[0] = 0.9999; // q[0] ≈ 1.0 in Q1.14

        load_sincos(angles_pos1);
        load_vector(input_vec);
        do_rotate();
        read_output(output_vec);

        $display("[TB] Expected: q'[0] ≈ %.5f, q'[1] ≈ %.5f", $cos(3.14159/4.0), $sin(3.14159/4.0));
        $display("[TB] Got:      q'[0] = %.5f, q'[1] = %.5f", output_vec[0], output_vec[1]);
        check_rotation("Test 2 (pair 0 by pi/4)", angles_pos1);

        // -------------------------------------------------------------------
        // TEST 3: Verify orthogonal pairs are independent
        // Set pair 1 (q[2],q[3]) to (0,1), rotate pair 1 by pi/2
        // Expected: q'[2] = -1.0, q'[3] = 0.0
        // -------------------------------------------------------------------
        $display("");
        $display("--- Test 3: Pair 1 rotation by pi/2, input (0,1) ---");

        for (int i = 0; i < NUM_PAIRS; i++)
            angles_pos1[i] = (i == 1) ? 3.14159265358979 / 2.0 : 0.0;

        for (int i = 0; i < HEAD_DIM; i++) input_vec[i] = 0.0;
        input_vec[3] = 0.9999; // q[3] = 1.0 (odd element of pair 1)
        // pair 1 = (q[2], q[3]) = (0.0, 1.0)
        // rotation by pi/2: q'[2] = 0*cos - 1*sin = -1, q'[3] = 0*sin + 1*cos = 0

        load_sincos(angles_pos1);
        load_vector(input_vec);
        do_rotate();
        read_output(output_vec);

        $display("[TB] Expected: q'[2] ≈ -1.0, q'[3] ≈ 0.0");
        $display("[TB] Got:      q'[2] = %.5f, q'[3] = %.5f", output_vec[2], output_vec[3]);
        check_rotation("Test 3 (pair 1 by pi/2)", angles_pos1);

        // -------------------------------------------------------------------
        // TEST 4: real RoPE angles pos * theta_i, theta_i = 10000^(-2i/HEAD_DIM),
        // random vectors in (-1, 1), several positions: every pair has a
        // different angle, so any pair/bank mis-mapping shows up
        // -------------------------------------------------------------------
        $display("");
        $display("--- Test 4: RoPE angles at positions 1, 7, 100, 1000, 4095 ---");
        foreach (positions[p]) begin
            for (int i = 0; i < NUM_PAIRS; i++)
                angles_pos1[i] = positions[p] * (10000.0 ** (-2.0 * i / HEAD_DIM));
            for (int i = 0; i < HEAD_DIM; i++)
                input_vec[i] = ($urandom_range(20000) - 10000) / 10000.0;
            load_sincos(angles_pos1);
            load_vector(input_vec);
            do_rotate();
            read_output(output_vec);
            check_rotation($sformatf("Test 4 (pos %0d)", positions[p]), angles_pos1);
        end

        // -------------------------------------------------------------------
        // TEST 5: saturation -- pair (1.9, 1.9) rotated by pi/4 has magnitude
        // 2.69 in the odd output, beyond Q1.14's [-2, 2): must clamp to 32767
        // -------------------------------------------------------------------
        $display("");
        $display("--- Test 5: output saturation ---");
        for (int i = 0; i < NUM_PAIRS; i++)
            angles_pos1[i] = (i == 2) ? 3.14159265358979 / 4.0 : 0.0;
        for (int i = 0; i < HEAD_DIM; i++) input_vec[i] = 0.5;
        input_vec[4] = 1.9;
        input_vec[5] = 1.9;
        load_sincos(angles_pos1);
        load_vector(input_vec);
        do_rotate();
        read_output(output_vec);
        check_rotation("Test 5 (saturation)", angles_pos1);
        if (out_int[5] !== 16'sd32767) begin
            $display("[TB]   FAIL Test 5: out[5] = %0d, expected saturation to 32767", out_int[5]);
            errors++;
        end

        $display("");
        if (errors == 0) $display("=== All RoPE tests PASSED ===");
        else             $display("=== RoPE tests FAILED: %0d error(s) ===", errors);
        $finish;
    end

    // Global timeout
    initial begin
        #500000;
        $display("[TB] Global simulation timeout");
        $finish;
    end

endmodule
`endif // SIMULATION
