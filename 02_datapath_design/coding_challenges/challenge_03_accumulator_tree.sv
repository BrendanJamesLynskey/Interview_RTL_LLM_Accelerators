// =============================================================================
// Challenge 03: Pipelined Binary Adder Tree (Accumulator / Reduction Tree)
// =============================================================================
//
// PROBLEM STATEMENT
// -----------------
// Design a pipelined binary adder tree that sums N input values into a single
// output. The design must be:
//
//   1. Parameterizable: N (number of inputs) and WIDTH (input bit width) are
//      compile-time parameters. N must be a power of two for simplicity, but
//      the design handles non-power-of-two N by zero-padding to the next
//      power of two (shown in the wrapper).
//
//   2. Pipelined: Pipeline registers are inserted between each level of the
//      adder tree. A depth-log2(N) tree has log2(N) adder stages, plus an
//      input register stage.
//      Throughput: one result per clock cycle once the pipeline is full.
//      Latency: log2(N) + 1 clock cycles.
//
//   3. Width-growing: Each adder level adds one bit to prevent overflow.
//      After log2(N) levels, the output width is WIDTH + log2(N) bits.
//
//   4. Correct for signed inputs (two's complement sign extension).
//
// USE CASE IN LLM ACCELERATORS
// -----------------------------
// Adder trees appear in:
//   - Column reduction in systolic arrays (summing N partial products)
//   - Softmax denominator computation (summing N exp values)
//   - Layer normalisation (summing N values for mean)
//   - Dot-product engines (final reduction of parallel multiply lanes)
//
// INTERFACE
// ---------
//   Parameters:
//     N       - Number of input operands (must be power of 2 >= 2)
//     WIDTH   - Input operand width in bits (signed)
//
//   Ports:
//     clk          - Clock
//     rst_n        - Active-low synchronous reset
//     valid_in     - Input data is valid (all N inputs sampled together)
//     data_in[N-1:0][WIDTH-1:0] - N parallel input values
//     sum_out[WIDTH+$clog2(N)-1:0] - Final sum (width-extended to prevent overflow)
//     valid_out    - Output sum is valid (valid_in delayed by log2(N) cycles)
//
// EXAMPLE
// -------
//   N=8, WIDTH=8:
//   - Tree depth: log2(8) = 3 levels
//   - Output width: 8 + 3 = 11 bits
//   - Latency: 3 adder levels + 1 input register = 4 clock cycles
//   - Inputs (8x 8-bit): {10, 20, 30, 40, 50, 60, 70, 80}
//   - Expected sum: 360 (fits in 11 bits: max 8 * 127 = 1016 < 2^10 = 1024)
//
// =============================================================================

`timescale 1ns/1ps

// =============================================================================
// Accumulator tree — recursive generate-based implementation
//
// Architecture:
//   Level 0 (inputs): N values of WIDTH bits
//   Level 1:          N/2 values of WIDTH+1 bits (pair-wise add at level 0)
//   Level 2:          N/4 values of WIDTH+2 bits
//   ...
//   Level log2(N):    1 value  of WIDTH+log2(N) bits (final sum)
//
// Each level is separated by a pipeline register stage.
// The valid signal propagates as a shift register of depth log2(N).
// =============================================================================

module accumulator_tree #(
    parameter int N     = 8,   // Number of inputs (must be a power of 2, >= 2)
    parameter int WIDTH = 8    // Input operand width (signed)
) (
    input  logic                                    clk,
    input  logic                                    rst_n,
    input  logic                                    valid_in,
    input  logic signed [N-1:0][WIDTH-1:0]          data_in,
    output logic signed [WIDTH+$clog2(N)-1:0]       sum_out,
    output logic                                     valid_out
);

    // Compute tree depth
    localparam int DEPTH = $clog2(N);
    localparam int OUT_W = WIDTH + DEPTH;

    // -------------------------------------------------------------------------
    // Generate the multi-level pipelined adder tree using a 2D array of wires.
    //
    // tree[level][node] holds the value at that level and node index.
    //   - level 0: N nodes, each WIDTH bits (the inputs)
    //   - level d: N/(2^d) nodes, each (WIDTH+d) bits
    //
    // We store each level in a packed array. To handle variable width per level,
    // we use the maximum output width (OUT_W) for all levels and sign-extend.
    // This is less area-efficient than tight packing but synthesises correctly.
    // -------------------------------------------------------------------------

    // Pipeline registers between levels: tree_reg[level][node][OUT_W-1:0]
    // tree_reg[0] = registered inputs
    // tree_reg[d] = result of adding tree_reg[d-1] pairs
    //
    // We declare a generate-time array of logic arrays.
    // Workaround for parameterized 3D arrays: use a macro-level generate.

    // Maximum nodes at any level = N (level 0). At level d: N >> d nodes.
    // We over-provision the node dimension to N and only use N>>d entries.

    logic signed [OUT_W-1:0] tree_reg [DEPTH+1][N]; // tree_reg[level][node]
    logic                    valid_pipe [DEPTH+1];    // valid delay line

    // Level 0: register the inputs (sign-extend to OUT_W)
    genvar node;
    generate
        for (node = 0; node < N; node++) begin : gen_input_reg
            always_ff @(posedge clk) begin
                if (!rst_n)
                    tree_reg[0][node] <= '0;
                else
                    tree_reg[0][node] <= OUT_W'(signed'(data_in[node]));
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) valid_pipe[0] <= 1'b0;
        else        valid_pipe[0] <= valid_in;
    end

    // Levels 1 through DEPTH: pair-wise addition with pipeline registers
    genvar level, n;
    generate
        for (level = 1; level <= DEPTH; level++) begin : gen_level
            // At this level, there are N >> level nodes.
            // Node i = tree_reg[level-1][2*i] + tree_reg[level-1][2*i+1]
            localparam int NODES_THIS_LEVEL = N >> level;
            for (n = 0; n < NODES_THIS_LEVEL; n++) begin : gen_node
                always_ff @(posedge clk) begin
                    if (!rst_n)
                        tree_reg[level][n] <= '0;
                    else
                        // Add the two children from the previous level.
                        // Both are already sign-extended to OUT_W bits, so
                        // the addition cannot overflow OUT_W bits.
                        tree_reg[level][n] <=
                            tree_reg[level-1][2*n] + tree_reg[level-1][2*n+1];
                end
            end

            // Propagate valid signal
            always_ff @(posedge clk) begin
                if (!rst_n) valid_pipe[level] <= 1'b0;
                else        valid_pipe[level] <= valid_pipe[level-1];
            end
        end
    endgenerate

    // Final output: the single remaining node at level DEPTH
    assign sum_out   = tree_reg[DEPTH][0];
    assign valid_out = valid_pipe[DEPTH];

endmodule : accumulator_tree


// =============================================================================
// Non-power-of-two wrapper
// Pads inputs to the next power of two before feeding the core tree.
// The padding uses zero (for unsigned) or sign extension (for signed sums
// where padding with zero is equivalent to padding with 0-valued inputs).
// =============================================================================

module accumulator_tree_wrap #(
    parameter int N_ACTUAL = 6,  // Actual number of inputs (any positive integer)
    parameter int WIDTH    = 8
) (
    input  logic                                          clk,
    input  logic                                          rst_n,
    input  logic                                          valid_in,
    input  logic signed [N_ACTUAL-1:0][WIDTH-1:0]         data_in,
    output logic signed [WIDTH+$clog2(N_ACTUAL)-1:0]      sum_out,
    output logic                                           valid_out
);

    // Next power of two >= N_ACTUAL
    localparam int N_POW2 = 2 ** $clog2(N_ACTUAL);
    // Output width of the padded tree
    localparam int TREE_OUT_W = WIDTH + $clog2(N_POW2);
    // Output width we advertise (based on actual N)
    localparam int ADV_OUT_W  = WIDTH + $clog2(N_ACTUAL);

    // Padded input array
    logic signed [N_POW2-1:0][WIDTH-1:0] data_padded;

    // Zero-pad unused inputs
    always_comb begin
        data_padded = '0;
        for (int i = 0; i < N_ACTUAL; i++) begin
            data_padded[i] = data_in[i];
        end
    end

    // Internal tree output (wider than we advertise)
    logic signed [TREE_OUT_W-1:0] sum_internal;

    accumulator_tree #(
        .N    (N_POW2),
        .WIDTH(WIDTH)
    ) u_tree (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in),
        .data_in  (data_padded),
        .sum_out  (sum_internal),
        .valid_out(valid_out)
    );

    // Truncate to advertised width (safe because padded zeros don't affect sum magnitude)
    assign sum_out = sum_internal[ADV_OUT_W-1:0];

endmodule : accumulator_tree_wrap


// =============================================================================
// TESTBENCH
// =============================================================================
// Tests:
//   Test 1 (N=8, WIDTH=8): Sum of {10,20,30,40,50,60,70,80} = 360
//   Test 2 (N=8, WIDTH=8): Sum of {127,127,127,127,-128,-128,-128,-128} = -4
//   Test 3 (N=8, WIDTH=8): All zeros
//   Test 4 (N=4, WIDTH=16): Larger width test
//   Test 5 (Non-pow2 wrapper, N=6, WIDTH=8): Sum of {1,2,3,4,5,6} = 21
//   Test 6 (N=8, WIDTH=8): Throughput -- four back-to-back inputs must give
//           four results on consecutive cycles (one result per clock)
//
// Latency: one input register plus log2(N) adder levels = log2(N) + 1 cycles.
// Every check waits for valid_out and verifies both the sum and the latency.
// =============================================================================

module tb_accumulator_tree;

    // ========================
    // Test case 1: N=8, W=8
    // ========================
    localparam int N8   = 8;
    localparam int W8   = 8;
    localparam int D8   = $clog2(N8);    // 3 adder levels (latency D8 + 1)
    localparam int OW8  = W8 + D8;       // 11-bit output

    logic                      clk;
    logic                      rst_n;
    logic                      valid_in_8;
    logic signed [N8-1:0][W8-1:0] data_in_8;
    logic signed [OW8-1:0]     sum_out_8;
    logic                      valid_out_8;

    accumulator_tree #(.N(N8), .WIDTH(W8)) dut_8 (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in_8),
        .data_in  (data_in_8),
        .sum_out  (sum_out_8),
        .valid_out(valid_out_8)
    );

    // ========================
    // Test case 2: N=4, W=16
    // ========================
    localparam int N4   = 4;
    localparam int W16  = 16;
    localparam int D4   = $clog2(N4);    // 2 adder levels (latency D4 + 1)
    localparam int OW16 = W16 + D4;      // 18-bit output

    logic                       valid_in_4;
    logic signed [N4-1:0][W16-1:0] data_in_4;
    logic signed [OW16-1:0]     sum_out_4;
    logic                       valid_out_4;

    accumulator_tree #(.N(N4), .WIDTH(W16)) dut_4 (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in_4),
        .data_in  (data_in_4),
        .sum_out  (sum_out_4),
        .valid_out(valid_out_4)
    );

    // ========================
    // Test case 3: N=6 wrapper
    // ========================
    localparam int N6  = 6;
    localparam int W6  = 8;
    localparam int OW6 = W6 + $clog2(N6); // 8+3=11 bits

    logic                      valid_in_6;
    logic signed [N6-1:0][W6-1:0] data_in_6;
    logic signed [OW6-1:0]     sum_out_6;
    logic                      valid_out_6;

    accumulator_tree_wrap #(.N_ACTUAL(N6), .WIDTH(W6)) dut_6 (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in_6),
        .data_in  (data_in_6),
        .sum_out  (sum_out_6),
        .valid_out(valid_out_6)
    );

    // Clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Result checking
    // -----------------------------------------------------------------------
    int fail_count = 0;

    // Wait for valid_out (sampled just after each rising edge) and check the
    // sum and the latency. Called after the edge that captured the input, so
    // `latency` is the number of further edges (tree depth); the total
    // latency reported is one more.
    task automatic expect_result(input string name,
                                 ref   logic  valid_out_s,
                                 ref   logic signed [31:0] sum_s,
                                 input longint expected,
                                 input int     latency);
        int cycles;
        cycles = 0;
        do begin
            @(posedge clk); #1;
            cycles++;
        end while (!valid_out_s && cycles < 20);

        if (!valid_out_s) begin
            $display("FAIL (%s): valid_out never asserted", name);
            fail_count++;
        end else if (sum_s !== 32'(expected) || cycles != latency) begin
            $display("FAIL (%s): sum = %0d, latency %0d cycles (expected %0d, latency %0d)",
                     name, sum_s, cycles + 1, expected, latency + 1);
            fail_count++;
        end else begin
            $display("PASS (%s): sum = %0d, latency %0d cycles", name, sum_s, cycles + 1);
        end
    endtask

    // Sign-extended views of the three DUT outputs, for the shared checker
    logic signed [31:0] sum_8_x, sum_4_x, sum_6_x;
    assign sum_8_x = 32'(sum_out_8);
    assign sum_4_x = 32'(sum_out_4);
    assign sum_6_x = 32'(sum_out_6);

    task automatic check_sum_8(
        input logic signed [W8-1:0] vals [N8],
        input logic signed [OW8-1:0] expected
    );
        @(negedge clk);
        valid_in_8 = 1'b1;
        for (int i = 0; i < N8; i++) data_in_8[i] = vals[i];
        @(negedge clk);
        valid_in_8 = 1'b0;
        // One input edge has passed; D8 more edges bring the sum out
        expect_result("N=8", valid_out_8, sum_8_x, expected, D8);
    endtask

    task automatic check_sum_4(
        input logic signed [W16-1:0] vals [N4],
        input logic signed [OW16-1:0] expected
    );
        @(negedge clk);
        valid_in_4 = 1'b1;
        for (int i = 0; i < N4; i++) data_in_4[i] = vals[i];
        @(negedge clk);
        valid_in_4 = 1'b0;
        expect_result("N=4,W16", valid_out_4, sum_4_x, expected, D4);
    endtask

    task automatic check_sum_6(
        input logic signed [W6-1:0] vals [N6],
        input logic signed [OW6-1:0] expected
    );
        @(negedge clk);
        valid_in_6 = 1'b1;
        for (int i = 0; i < N6; i++) data_in_6[i] = vals[i];
        @(negedge clk);
        valid_in_6 = 1'b0;
        // Padded to 8 inputs internally, so the tree depth is clog2(8) = 3
        expect_result("N=6 wrap", valid_out_6, sum_6_x, expected, $clog2(N6));
    endtask

    // Throughput monitor: record every valid N=8 output
    int throughput_sums [$];
    always @(posedge clk) begin
        #1;
        if (valid_out_8) throughput_sums.push_back(int'(sum_out_8));
    end

    // -----------------------------------------------------------------------
    // Main test sequence
    // -----------------------------------------------------------------------
    logic signed [W8-1:0]  test_vals_8[N8];
    logic signed [W16-1:0] test_vals_4[N4];
    logic signed [W6-1:0]  test_vals_6[N6];

    initial begin
        rst_n      = 1'b0;
        valid_in_8 = 1'b0;
        valid_in_4 = 1'b0;
        valid_in_6 = 1'b0;
        data_in_8  = '0;
        data_in_4  = '0;
        data_in_6  = '0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        // -----------------------------------------------
        // Test 1: {10,20,30,40,50,60,70,80} = 360
        // -----------------------------------------------
        $display("\n=== Test 1: N=8, positive values ===");
        for (int i = 0; i < N8; i++) test_vals_8[i] = 8'(10 * (i + 1));
        check_sum_8(test_vals_8, 11'd360);

        // -----------------------------------------------
        // Test 2: Mixed positive and negative
        // {127,127,127,127,-128,-128,-128,-128}
        // = 4*127 + 4*(-128) = 508 - 512 = -4
        // -----------------------------------------------
        $display("\n=== Test 2: N=8, mixed signs ===");
        test_vals_8[0] = 8'sd127; test_vals_8[1] = 8'sd127;
        test_vals_8[2] = 8'sd127; test_vals_8[3] = 8'sd127;
        test_vals_8[4] = -8'sd128; test_vals_8[5] = -8'sd128;
        test_vals_8[6] = -8'sd128; test_vals_8[7] = -8'sd128;
        check_sum_8(test_vals_8, 11'(-4));

        // -----------------------------------------------
        // Test 3: All zeros
        // -----------------------------------------------
        $display("\n=== Test 3: N=8, all zeros ===");
        for (int i = 0; i < N8; i++) test_vals_8[i] = 8'd0;
        check_sum_8(test_vals_8, 11'd0);

        // -----------------------------------------------
        // Test 4: N=4, WIDTH=16, large values
        // {1000, -2000, 3000, -4000} = -2000
        // -----------------------------------------------
        $display("\n=== Test 4: N=4, W=16, large values ===");
        test_vals_4[0] =  16'sd1000;
        test_vals_4[1] = -16'sd2000;
        test_vals_4[2] =  16'sd3000;
        test_vals_4[3] = -16'sd4000;
        check_sum_4(test_vals_4, 18'(-2000));

        // -----------------------------------------------
        // Test 5: Non-power-of-two wrapper, N=6
        // {1,2,3,4,5,6} = 21
        // -----------------------------------------------
        $display("\n=== Test 5: N=6 (non-pow2 wrapper) ===");
        for (int i = 0; i < N6; i++) test_vals_6[i] = 8'(i + 1);
        check_sum_6(test_vals_6, 11'd21);

        // -----------------------------------------------
        // Test 6: Throughput test — consecutive inputs
        // Drive 4 consecutive valid cycles for N=8, check 4 consecutive outputs
        // Inputs:  cycle 0: {1,1,1,1,1,1,1,1} = 8
        //          cycle 1: {2,2,2,2,2,2,2,2} = 16
        //          cycle 2: {3,3,3,3,3,3,3,3} = 24
        //          cycle 3: {4,4,4,4,4,4,4,4} = 32
        // Outputs appear at cycles D8=3 later: cycle 3,4,5,6
        // -----------------------------------------------
        $display("\n=== Test 6: Throughput — consecutive inputs ===");
        repeat (D8 + 2) @(negedge clk);   // let earlier results drain
        throughput_sums.delete();
        valid_in_8 = 1'b1;

        // Drive 4 cycles of consecutive inputs
        for (int cycle = 1; cycle <= 4; cycle++) begin
            for (int i = 0; i < N8; i++) data_in_8[i] = 8'(cycle);
            @(negedge clk);
        end
        valid_in_8 = 1'b0;

        // Latency is D8 + 1 cycles, so all four results are out well within
        // D8 + 4 further edges
        repeat (D8 + 4) @(posedge clk);
        #1;
        if (throughput_sums.size() == 4 &&
            throughput_sums[0] == 8  && throughput_sums[1] == 16 &&
            throughput_sums[2] == 24 && throughput_sums[3] == 32)
            $display("PASS (throughput): results 8, 16, 24, 32 on consecutive cycles");
        else begin
            $display("FAIL (throughput): got %p (expected '{8, 16, 24, 32})", throughput_sums);
            fail_count++;
        end

        if (fail_count == 0)
            $display("\n=== ALL ACCUMULATOR TREE TESTS PASSED ===");
        else
            $display("\n=== %0d ACCUMULATOR TREE TESTS FAILED ===", fail_count);
        $finish;
    end

    initial begin
        #20000;
        $display("TIMEOUT");
        $finish;
    end

    initial begin
        $dumpfile("accumulator_tree.vcd");
        $dumpvars(0, tb_accumulator_tree);
    end

    // Monitor for debugging
    // Uncomment to trace each cycle:
    // always @(posedge clk) begin
    //     if (valid_out_8)
    //         $display("t=%0t valid_out_8=1 sum_out_8=%0d", $time, sum_out_8);
    // end

endmodule : tb_accumulator_tree

// =============================================================================
// INTERVIEW DISCUSSION POINTS
// =============================================================================
//
// 1. WHY LOG2(N) STAGES AND NOT MORE OR FEWER?
//    A binary tree reduces N inputs to 1 output in exactly log2(N) levels.
//    Each level halves the number of nodes. Fewer stages would require multi-
//    input adders (3:2 compressors or carry-save adders) to further reduce
//    critical path, at the cost of more complex hardware per node.
//    More stages would mean sub-optimal use of pipeline registers.
//
// 2. OVERFLOW PREVENTION: WIDTH GROWTH PER LEVEL
//    At each level, the sum of two N-bit values requires N+1 bits. After
//    log2(N) levels, the output is WIDTH + log2(N) bits. Failure to grow
//    the width causes silent overflow (wraparound), which is catastrophic
//    for correctness. The over-provisioned OUT_W approach here ensures no
//    overflow at any level.
//
// 3. NON-POWER-OF-TWO N
//    The wrapper pads to the next power of two with zeros. For signed
//    summation, padding with zero is correct (0 is the additive identity).
//    For unsigned summation, same. For max/min reductions, padding depends
//    on the identity element (+Inf for min, -Inf for max).
//
// 4. CRITICAL PATH IN PRACTICE
//    The critical path through each adder stage depends on the adder
//    implementation. For WIDTH=8, a simple ripple-carry adder per level is
//    fine at 1 GHz (< 2 ns). For WIDTH=32 or larger, a carry-lookahead or
//    carry-select adder is needed to meet timing at high frequencies.
//    Synthesis tools handle this automatically when given timing constraints.
//
// 5. ALTERNATIVE: SERIAL ACCUMULATOR
//    A single adder with a counter can accumulate N values in N cycles with
//    minimal area (1 adder + 1 accumulator register). The adder tree is used
//    when N values arrive in parallel and throughput (not area) is the goal.
//    In a systolic array column reduction, all N partial products are
//    available simultaneously, making the parallel adder tree necessary.
//
// 6. COMPARISON WITH DADDA/WALLACE TREES
//    A Wallace tree uses carry-save adders (CSAs) to reduce N inputs in
//    O(log1.5(N)) stages (fewer than the binary tree's log2(N) stages).
//    Each CSA stage reduces 3 inputs to 2 (sum + carry) without propagating
//    carry, so it is a single gate delay. The Wallace tree trades more
//    complex interconnect and a wider final adder for fewer pipeline stages.
//    For synthesis, the binary tree here is preferable for clarity; a tools-
//    optimised implementation may produce a Wallace-tree-like structure
//    automatically when optimising for speed.
//
// =============================================================================
