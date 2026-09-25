// =============================================================================
// Challenge 01: Parameterizable N×N Output-Stationary Systolic Array
// =============================================================================
//
// PROBLEM STATEMENT
// -----------------
// Design a parameterizable N×N systolic array for matrix multiplication using
// output-stationary dataflow:
//
//   C = A × B   where A is (N×K), B is (K×N), C is (N×N)
//
// In output-stationary dataflow:
//   - Each PE holds its partial sum (accumulator) locally.
//   - Row i of A is fed into the left edge of row i, skewed by i cycles.
//   - Column j of B is fed into the top edge of column j, skewed by j cycles.
//   - After K cycles of accumulation, C[i][j] is held in PE[i][j].
//
// INTERFACE SPECIFICATION
// -----------------------
//   Parameters:
//     N         - Array dimension (N×N PEs). Must be power-of-two or arbitrary.
//     DATA_W    - Input operand width in bits (default 8, INT8).
//     ACC_W     - Accumulator width in bits (default 32, INT32).
//
//   Inputs:
//     clk       - Clock.
//     rst_n     - Active-low synchronous reset.
//     a_in[N-1:0][DATA_W-1:0]  - Row of A fed into left edge of each row.
//     b_in[N-1:0][DATA_W-1:0]  - Column of B fed into top edge of each column.
//     load_c    - When high, PE accumulators are reset to zero (start of new tile).
//     valid_in  - Input data is valid (active high).
//
//   Outputs:
//     c_out[N-1:0][N-1:0][ACC_W-1:0] - Accumulated result matrix C.
//     valid_out - Output data is valid (asserted after K accumulation cycles).
//
// SKEWING REQUIREMENT
// -------------------
// The caller must pre-skew input data:
//   - a_in[i] should present A[i][k] at cycle (start + i + k).
//   - b_in[j] should present B[k][j] at cycle (start + j + k).
// Alternatively, the design below implements internal skewing using shift
// registers on both input paths, so the caller presents data without skew:
//   - a_in[i] presents row i of A sequentially (A[i][0], A[i][1], ... A[i][K-1])
//     starting at cycle 0 for all rows simultaneously.
//   - b_in[j] presents column j of B sequentially starting at cycle 0.
// The internal shift registers introduce the required i/j cycle skew.
//
// EXPECTED BEHAVIOUR
// ------------------
// For N=4, K=4, multiplying identity-like matrices:
//   A = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}  (identity)
//   B = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}
//   C = A × B = B (identity times B equals B)
//
// The first valid C output appears at cycle 2*(N-1) + K + 1 after load_c is
// deasserted.
//
// =============================================================================

`timescale 1ns/1ps

// =============================================================================
// Processing Element (PE)
// =============================================================================
// Each PE:
//   1. Receives a_in from the left (or internal shift register).
//   2. Receives b_in from the top.
//   3. Multiplies them and accumulates into a local ACC_W-bit register.
//   4. Passes a_in to the right (horizontal propagation).
//   5. Passes b_in downward (vertical propagation).
//
// The PE uses signed arithmetic throughout.
// =============================================================================

module pe #(
    parameter int DATA_W = 8,   // Input operand width
    parameter int ACC_W  = 32   // Accumulator width (must be >= 2*DATA_W + log2(K))
) (
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   clear,          // Clear accumulator (start of new tile)
    input  logic                   en,             // a_in/b_in valid: when high, perform MAC
    input  logic signed [DATA_W-1:0] a_in,         // Activation from left
    input  logic signed [DATA_W-1:0] b_in,         // Weight from top
    output logic                   en_out,         // Valid travelling with a_out (to right)
    output logic signed [DATA_W-1:0] a_out,        // Activation to right
    output logic signed [DATA_W-1:0] b_out,        // Weight downward
    output logic signed [ACC_W-1:0]  acc_out        // Local accumulator value
);

    logic signed [ACC_W-1:0] acc_reg;
    logic signed [2*DATA_W-1:0] product;

    // Product is 2*DATA_W bits to hold full precision without overflow
    assign product = a_in * b_in;

    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            acc_reg <= '0;
            en_out  <= 1'b0;
            a_out   <= '0;
            b_out   <= '0;
        end else begin
            // Accumulate only when the operands are valid: sign-extend the
            // product to ACC_W before adding
            if (en)
                acc_reg <= acc_reg + ACC_W'(signed'(product));
            // Propagate data (and its valid) to neighbours every cycle, so the
            // one-cycle-per-hop pipeline timing never depends on en
            en_out  <= en;
            a_out   <= a_in;
            b_out   <= b_in;
        end
    end

    assign acc_out = acc_reg;

endmodule : pe


// =============================================================================
// Systolic Array Top-Level
// =============================================================================

module systolic_array #(
    parameter int N      = 4,   // Array dimension (N×N)
    parameter int DATA_W = 8,   // Input operand width
    parameter int ACC_W  = 32   // Accumulator width
) (
    input  logic                            clk,
    input  logic                            rst_n,
    // a_in[i] = data for row i of A, all presented at the same time (no pre-skew needed)
    input  logic signed [N-1:0][DATA_W-1:0] a_in,
    // b_in[j] = data for column j of B, all presented at the same time (no pre-skew needed)
    input  logic signed [N-1:0][DATA_W-1:0] b_in,
    input  logic                            load_c,   // Reset accumulators
    input  logic                            valid_in, // Input data is valid this cycle
    output logic signed [N-1:0][N-1:0][ACC_W-1:0] c_out,
    output logic                            valid_out
);

    // -------------------------------------------------------------------------
    // Internal skewing shift registers
    // Row i of A is delayed by i cycles using a shift register of depth i.
    // Column j of B is delayed by j cycles using a shift register of depth j.
    // -------------------------------------------------------------------------

    // a_skewed[i][j] = a_in[i] delayed by i cycles, then fed to column j within row i
    // Concretely: we need a_skewed[i] to be a_in[i] delayed by i cycles.
    // The shift register for row i has depth i (0 for row 0, 1 for row 1, etc.)

    // b_skewed[j][i] = b_in[j] delayed by j cycles, then fed to row i within col j

    // We implement skewing with 2D arrays of registers.
    // a_shift[i][d] = d-th stage of shift register for row i (d = 0..N-2)
    // a_shift[i][0] = a_in[i] (registered once), a_shift[i][i-1] = output for row i

    // For row 0: no delay, use a_in[0] directly (zero skew).
    // For row i (i>0): chain i flip-flop stages.

    logic signed [N-1:0][N-1:0][DATA_W-1:0] a_shift; // a_shift[row][stage]
    logic signed [N-1:0][N-1:0][DATA_W-1:0] b_shift; // b_shift[col][stage]

    // Skewed outputs fed into the PE array left edge and top edge
    logic signed [N-1:0][DATA_W-1:0] a_skewed; // a_skewed[i] = a_in[i] delayed i cycles
    logic signed [N-1:0][DATA_W-1:0] b_skewed; // b_skewed[j] = b_in[j] delayed j cycles

    // valid_in skewing: valid_in is skewed with row i of A, then travels right
    // through the PEs alongside the a data (see pe.en_out). A reaches PE(i,j)
    // after i (skew) + j (hops) cycles and B after j (skew) + i (hops) cycles,
    // so the valid carried with A marks exactly the cycles in which both
    // operands at PE(i,j) belong to the same k -- no separate B valid needed.
    logic [N-1:0] a_valid_shift [N]; // a_valid_shift[row][stage]
    logic [N-1:0] a_valid_skewed;
    logic [N-1:0] pe_en [N]; // pe_en[i][j] = enable for PE(i,j)

    // Build skewing shift registers
    genvar gi, gd;
    generate
        for (gi = 0; gi < N; gi++) begin : gen_skew_row
            if (gi == 0) begin : no_delay_row
                // Row 0: no delay
                assign a_skewed[0]      = a_in[0];
                assign a_valid_skewed[0] = valid_in;
            end else begin : delay_row
                // Row gi: delay by gi cycles
                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        a_shift[gi][0]       <= '0;
                        a_valid_shift[gi][0] <= 1'b0;
                    end else begin
                        a_shift[gi][0]       <= a_in[gi];
                        a_valid_shift[gi][0] <= valid_in;
                    end
                end
                for (gd = 1; gd < gi; gd++) begin : gen_a_stages
                    always_ff @(posedge clk) begin
                        if (!rst_n) begin
                            a_shift[gi][gd]       <= '0;
                            a_valid_shift[gi][gd] <= 1'b0;
                        end else begin
                            a_shift[gi][gd]       <= a_shift[gi][gd-1];
                            a_valid_shift[gi][gd] <= a_valid_shift[gi][gd-1];
                        end
                    end
                end
                assign a_skewed[gi]      = a_shift[gi][gi-1];
                assign a_valid_skewed[gi] = a_valid_shift[gi][gi-1];
            end
        end

        for (gi = 0; gi < N; gi++) begin : gen_skew_col
            if (gi == 0) begin : no_delay_col
                assign b_skewed[0]      = b_in[0];
            end else begin : delay_col
                always_ff @(posedge clk) begin
                    if (!rst_n)
                        b_shift[gi][0] <= '0;
                    else
                        b_shift[gi][0] <= b_in[gi];
                end
                for (gd = 1; gd < gi; gd++) begin : gen_b_stages
                    always_ff @(posedge clk) begin
                        if (!rst_n)
                            b_shift[gi][gd] <= '0;
                        else
                            b_shift[gi][gd] <= b_shift[gi][gd-1];
                    end
                end
                assign b_skewed[gi]      = b_shift[gi][gi-1];
            end
        end
    endgenerate


    // -------------------------------------------------------------------------
    // PE array instantiation and interconnect
    // Horizontal: a data flows left to right through PEs in each row.
    //             a_wire[i][j] = output of PE(i,j-1), fed into PE(i,j).
    //             a_wire[i][0] = a_skewed[i] (from left edge).
    //             pe_en[i][j] travels with a: pe_en[i][0] = a_valid_skewed[i],
    //             pe_en[i][j+1] = en_out of PE(i,j).
    // Vertical:   b data flows top to bottom through PEs in each column.
    //             b_wire[i][j] = output of PE(i-1,j), fed into PE(i,j).
    //             b_wire[0][j] = b_skewed[j] (from top edge).
    // -------------------------------------------------------------------------

    logic signed [N-1:0][N:0][DATA_W-1:0]   a_wire; // a_wire[row][col], col 0 = left input
    logic signed [N:0][N-1:0][DATA_W-1:0]   b_wire; // b_wire[row][col], row 0 = top input
    logic        [N-1:0][N:0]               v_wire; // v_wire[row][col], valid alongside a_wire

    // Connect left-edge and top-edge inputs
    genvar gi2;
    generate
        for (gi2 = 0; gi2 < N; gi2++) begin : gen_edge_connect
            assign a_wire[gi2][0] = a_skewed[gi2];   // Left edge of each row
            assign v_wire[gi2][0] = a_valid_skewed[gi2];
            assign b_wire[0][gi2] = b_skewed[gi2];   // Top edge of each column
        end
    endgenerate

    // Instantiate N×N PEs
    genvar row, col;
    generate
        for (row = 0; row < N; row++) begin : gen_row
            for (col = 0; col < N; col++) begin : gen_col
                pe #(
                    .DATA_W(DATA_W),
                    .ACC_W (ACC_W)
                ) u_pe (
                    .clk    (clk),
                    .rst_n  (rst_n),
                    .clear  (load_c),
                    .en     (pe_en[row][col]),
                    .a_in   (a_wire[row][col]),      // From left neighbour
                    .b_in   (b_wire[row][col]),      // From top neighbour
                    .en_out (v_wire[row][col+1]),    // Valid to right neighbour
                    .a_out  (a_wire[row][col+1]),    // To right neighbour
                    .b_out  (b_wire[row+1][col]),    // To bottom neighbour
                    .acc_out(c_out[row][col])         // Output accumulator
                );
                assign pe_en[row][col] = v_wire[row][col];
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // valid_out generation
    // The last PE to produce a valid output is PE[N-1][N-1].
    // It starts accumulating at cycle (N-1) + (N-1) = 2*(N-1) after the first
    // valid_in (due to skewing). It finishes after K accumulation cycles.
    // We track the PE[N-1][N-1] enable signal delayed by K cycles to assert
    // valid_out. For simplicity, valid_out is asserted when PE[N-1][N-1]
    // has received its last valid input (i.e., the pe_en for that PE goes low
    // after K cycles of being high).
    //
    // Simpler implementation: pulse valid_out one cycle after pe_en[N-1][N-1]
    // falls (trailing edge of enable to PE[N-1][N-1]).
    // -------------------------------------------------------------------------

    logic pe_en_last_d;

    always_ff @(posedge clk) begin
        if (!rst_n)
            pe_en_last_d <= 1'b0;
        else
            pe_en_last_d <= pe_en[N-1][N-1];
    end

    // valid_out pulses for one cycle when pe_en[N-1][N-1] falls (trailing edge)
    assign valid_out = pe_en_last_d & ~pe_en[N-1][N-1];

endmodule : systolic_array


// =============================================================================
// TESTBENCH
// =============================================================================
// Tests a 4×4 systolic array with the following matrices:
//
//   A = {{1, 2, 3, 4},
//        {5, 6, 7, 8},
//        {9,10,11,12},
//        {13,14,15,16}}
//
//   B = {{1, 0, 0, 0},
//        {0, 1, 0, 0},
//        {0, 0, 1, 0},
//        {0, 0, 0, 1}}   (identity matrix)
//
//   Expected C = A × B = A
//
// Further tests: A × A and random signed INT8 A × B (expected values from the
// ref_matmul reference model), and A × 0. Each test also checks the valid_out
// latency and pulse width; the final verdict depends on the error count.
// =============================================================================

module tb_systolic_array;

    // -------------------------
    // Parameters
    // -------------------------
    localparam int N      = 4;
    localparam int DATA_W = 8;
    localparam int ACC_W  = 32;
    localparam int K      = 4;     // Inner dimension (same as N for square matrices)

    // -------------------------
    // DUT signals
    // -------------------------
    logic                                   clk;
    logic                                   rst_n;
    logic signed [N-1:0][DATA_W-1:0]        a_in;
    logic signed [N-1:0][DATA_W-1:0]        b_in;
    logic                                   load_c;
    logic                                   valid_in;
    logic signed [N-1:0][N-1:0][ACC_W-1:0]  c_out;
    logic                                   valid_out;

    // -------------------------
    // DUT instantiation
    // -------------------------
    systolic_array #(
        .N     (N),
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .a_in     (a_in),
        .b_in     (b_in),
        .load_c   (load_c),
        .valid_in (valid_in),
        .c_out    (c_out),
        .valid_out(valid_out)
    );

    // -------------------------
    // Clock generation: 10 ns period
    // -------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------
    // Test data
    // -------------------------
    // Matrix A (row-major, INT8)
    logic signed [DATA_W-1:0] mat_a [N][K];
    // Matrix B stored as columns (col-major) since b_in[j] is column j
    logic signed [DATA_W-1:0] mat_b_col [N][K]; // mat_b_col[col][row]
    // Expected result
    logic signed [ACC_W-1:0] expected_c [N][N];
    // Mismatch / protocol error count; the final verdict depends on it
    int errors = 0;

    // -------------------------
    // Task: drive one GEMM computation
    // -------------------------
    task automatic run_gemm(
        input logic signed [DATA_W-1:0] a [N][K],
        input logic signed [DATA_W-1:0] b_col [N][K], // b_col[col][k]
        input logic signed [ACC_W-1:0]  exp_c [N][N]
    );
        integer k_step, i, j;
        int     lat;
        // Assert load_c for one cycle to clear accumulators
        // While valid_in is low the operands are don't-care: drive junk so a
        // PE that accumulated without its enable would be caught
        @(negedge clk);
        load_c   = 1'b1;
        valid_in = 1'b0;
        a_in     = $urandom;
        b_in     = $urandom;
        @(negedge clk);
        load_c   = 1'b0;

        // Drive K cycles of input data with valid_in=1
        for (k_step = 0; k_step < K; k_step++) begin
            @(negedge clk);
            valid_in = 1'b1;
            for (i = 0; i < N; i++) begin
                a_in[i] = a[i][k_step];      // Row i, element k
            end
            for (j = 0; j < N; j++) begin
                b_in[j] = b_col[j][k_step];  // Column j, element k
            end
        end

        // Deassert valid_in after K cycles
        @(negedge clk);
        valid_in = 1'b0;
        a_in     = $urandom;
        b_in     = $urandom;

        // Wait for valid_out. Counting the posedge that samples the first
        // valid_in as posedge 1 (K posedges have been seen so far),
        // PE[N-1][N-1] gets that operand pair 2*(N-1) posedges later and does
        // its last MAC on posedge 2*(N-1) + K; valid_out rises just after it.
        lat = K;
        fork
            begin : wait_valid
                repeat (2*N + K + 5) @(posedge clk);
                $display("FAIL: TIMEOUT -- valid_out never asserted");
                $finish;
            end
            begin : check_valid
                while (valid_out !== 1'b1) begin
                    @(posedge clk);
                    #1 lat++;
                end
                disable wait_valid;
            end
        join
        if (lat != 2*(N-1) + K) begin
            $display("FAIL: valid_out latency = %0d cycles, expected %0d", lat, 2*(N-1) + K);
            errors++;
        end

        // Check results while valid_out is high (accumulators hold afterwards
        // too, since no PE is enabled once the last operands have passed)
        $display("--- GEMM Result ---");
        for (i = 0; i < N; i++) begin
            for (j = 0; j < N; j++) begin
                if (c_out[i][j] !== exp_c[i][j]) begin
                    $display("FAIL: C[%0d][%0d] = %0d, expected %0d",
                              i, j, c_out[i][j], exp_c[i][j]);
                    errors++;
                end else begin
                    $display("PASS: C[%0d][%0d] = %0d", i, j, c_out[i][j]);
                end
            end
        end
        @(posedge clk); #1;
        if (valid_out !== 1'b0) begin
            $display("FAIL: valid_out is not a one-cycle pulse");
            errors++;
        end
    endtask

    // -------------------------
    // Reference model (combinational)
    // -------------------------
    function automatic void ref_matmul(
        input  logic signed [DATA_W-1:0] a [N][K],
        input  logic signed [DATA_W-1:0] b [K][N],   // b[k][col], row-major B
        output logic signed [ACC_W-1:0]  c [N][N]
    );
        integer i, j, k;
        logic signed [ACC_W-1:0] prod;
        for (i = 0; i < N; i++) begin
            for (j = 0; j < N; j++) begin
                c[i][j] = '0;
                for (k = 0; k < K; k++) begin
                    // Multiply at ACC_W bits (a cast would make the multiply
                    // self-determined at DATA_W bits and overflow)
                    prod    = a[i][k];
                    prod    = prod * b[k][j];
                    c[i][j] = c[i][j] + prod;
                end
            end
        end
    endfunction

    // -------------------------
    // Main test sequence
    // -------------------------
    integer i, j, k;
    logic signed [DATA_W-1:0] mat_b [K][N]; // b in row-major for ref model

    initial begin
        // Initialise signals
        rst_n    = 1'b0;
        load_c   = 1'b0;
        valid_in = 1'b0;
        a_in     = '0;
        b_in     = '0;

        // Apply reset for 3 cycles
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        // -----------------------------------------------
        // Test 1: A × I (identity) = A
        // -----------------------------------------------
        $display("\n=== Test 1: A × I ===");

        // Fill mat_a with {1..16}
        for (i = 0; i < N; i++)
            for (j = 0; j < K; j++)
                mat_a[i][j] = 8'(i * K + j + 1);

        // Fill mat_b = identity
        for (i = 0; i < K; i++)
            for (j = 0; j < N; j++)
                mat_b[i][j] = (i == j) ? 8'd1 : 8'd0;

        // Build b_col (column-major view of B for the array input)
        // b_col[col][k] = mat_b[k][col]  (column col, row k of B)
        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                mat_b_col[j][k] = mat_b[k][j];

        // Expected: C = A × I = A
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                expected_c[i][j] = ACC_W'(signed'(mat_a[i][j]));

        run_gemm(mat_a, mat_b_col, expected_c);

        // -----------------------------------------------
        // Test 2: A × A (general square multiply)
        // -----------------------------------------------
        $display("\n=== Test 2: A × A ===");

        // Reuse mat_a (same matrix) for both operands
        // mat_b = mat_a for this test
        for (i = 0; i < K; i++)
            for (j = 0; j < N; j++)
                mat_b[i][j] = mat_a[i][j];

        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                mat_b_col[j][k] = mat_b[k][j];

        // Compute expected C = A × A via reference model
        ref_matmul(mat_a, mat_b, expected_c);

        $display("Expected C = A × A (reference model):");
        for (i = 0; i < N; i++) begin
            for (j = 0; j < N; j++)
                $write("%8d ", expected_c[i][j]);
            $display("");
        end

        run_gemm(mat_a, mat_b_col, expected_c);

        // -----------------------------------------------
        // Test 3: Zero matrix
        // -----------------------------------------------
        $display("\n=== Test 3: A × 0 = 0 ===");

        for (i = 0; i < K; i++)
            for (j = 0; j < N; j++)
                mat_b[i][j] = 8'd0;

        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                mat_b_col[j][k] = 8'd0;

        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                expected_c[i][j] = 32'd0;

        run_gemm(mat_a, mat_b_col, expected_c);

        // -----------------------------------------------
        // Test 4: random signed INT8 A × B, with row 0 of A and column 0 of
        // B forced to -128 so C[0][0] = K * 16384 exercises the full signed
        // product range and accumulation beyond 16 bits
        // -----------------------------------------------
        $display("\n=== Test 4: random signed A × B ===");

        for (i = 0; i < N; i++)
            for (j = 0; j < K; j++)
                mat_a[i][j] = (i == 0) ? -8'sd128 : DATA_W'($urandom);
        for (i = 0; i < K; i++)
            for (j = 0; j < N; j++)
                mat_b[i][j] = (j == 0) ? -8'sd128 : DATA_W'($urandom);
        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                mat_b_col[j][k] = mat_b[k][j];

        ref_matmul(mat_a, mat_b, expected_c);
        run_gemm(mat_a, mat_b_col, expected_c);

        $display("\n=== All tests complete: %0d error(s) ===", errors);
        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("TESTS FAILED");
        $finish;
    end

    // -------------------------
    // Timeout watchdog
    // -------------------------
    initial begin
        #50000;
        $display("GLOBAL TIMEOUT");
        $finish;
    end

    // -------------------------
    // Optional waveform dump
    // -------------------------
    initial begin
        $dumpfile("systolic_array.vcd");
        $dumpvars(0, tb_systolic_array);
    end

endmodule : tb_systolic_array

// =============================================================================
// INTERVIEW DISCUSSION POINTS
// =============================================================================
//
// 1. WHY OUTPUT-STATIONARY?
//    The partial sum never leaves the PE until computation is complete.
//    This eliminates partial-sum bandwidth (reads and writes of INT32 partial
//    sums from SRAM) which would otherwise dominate power for long K dimensions.
//
// 2. SKEWING OVERHEAD
//    The input skewing shift registers use N*(N-1)/2 registers per side.
//    For N=256, that is 256*255/2 = 32,640 registers per edge = ~8 bits x 32K
//    = 256 Kb of shift register state just for skewing. At 8b x 1 GHz this is
//    non-trivial. Production designs often pre-skew in the DMA engine rather
//    than in the array.
//
// 3. CLEAR SYNCHRONISATION
//    The load_c (clear) signal must reach all PEs simultaneously. For large
//    arrays, this is a high-fanout signal requiring careful buffering.
//    Alternatively, a "clear done" acknowledgement can be used.
//
// 4. EXTENDING TO RECTANGULAR ARRAYS
//    Change N to separate NM (rows) and NN (cols) parameters. The skewing
//    depth changes accordingly: A is skewed by row index (0 to NM-1),
//    B is skewed by column index (0 to NN-1).
//
// 5. VALID_OUT TIMING
//    The valid_out scheme shown (trailing edge of last PE enable) is suitable
//    for tile-at-a-time operation. For continuous streaming across K-tiles,
//    valid_out should be replaced with a counter that fires every K input cycles
//    starting at cycle 2*(N-1) + K.
//
// =============================================================================
