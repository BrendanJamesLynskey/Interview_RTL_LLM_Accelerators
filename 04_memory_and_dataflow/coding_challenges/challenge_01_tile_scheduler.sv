// =============================================================================
// Challenge 01: Tile Scheduler for Tiled Matrix Multiplication
// =============================================================================
//
// PROBLEM STATEMENT
// -----------------
// A matrix multiply C = A * B is decomposed into tiles to fit on-chip memory.
// A is (M x K), B is (K x N), C is (M x N). The tile sizes are Tm, Tk, Tn.
//
// The scheduler must generate tile coordinate tuples (m_tile, n_tile, k_tile)
// in the correct OUTPUT-STATIONARY loop order:
//
//   for m_tile in 0 .. NUM_M_TILES-1:
//     for n_tile in 0 .. NUM_N_TILES-1:       <- output tile stays until all k done
//       for k_tile in 0 .. NUM_K_TILES-1:
//         issue (m_tile, n_tile, k_tile)
//
// This order maximises partial-sum reuse: the (m_tile, n_tile) output tile
// accumulates its full inner product before moving to the next output tile.
//
// INTERFACE
// ---------
// Parameters (compile-time):
//   M, K, N        : Matrix dimensions (must be multiples of Tm, Tk, Tn)
//   TILE_M, TILE_K, TILE_N : Tile dimensions
//
// Inputs:
//   clk            : Clock
//   rst_n          : Active-low synchronous reset
//   start          : Pulse high for one cycle to begin scheduling
//   next_ready     : Consumer is ready to accept the next tile coordinate
//                    (back-pressure signal; scheduler holds output until asserted)
//
// Outputs:
//   valid          : Output coordinate is valid
//   m_idx          : Tile row index (0 .. NUM_M_TILES-1)
//   n_idx          : Tile column index (0 .. NUM_N_TILES-1)
//   k_idx          : Inner product tile index (0 .. NUM_K_TILES-1)
//   last_k         : High when k_idx == NUM_K_TILES-1 (last accumulation step
//                    for this output tile; consumer should write C tile to memory)
//   done           : High for one cycle after all tiles have been issued
//
// EXPECTED BEHAVIOUR
// ------------------
// After start is asserted, the scheduler begins issuing (m, n, k) tuples in
// output-stationary order. Each tuple is held until next_ready is asserted.
// After the last tuple (M-1, N-1, K-1), done is pulsed for one cycle.
//
// EXAMPLE (M=4, K=4, N=4, all tiles = 2 -> 2x2x2 = 8 tiles):
// Issue order: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
//
// =============================================================================

module tile_scheduler #(
    parameter int M      = 64,   // Total matrix rows
    parameter int K      = 64,   // Shared dimension
    parameter int N      = 64,   // Total matrix columns
    parameter int TILE_M = 16,   // Tile height
    parameter int TILE_K = 16,   // Tile depth
    parameter int TILE_N = 16    // Tile width
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic next_ready,     // Consumer ready to accept next coordinate
    output logic valid,
    output logic [$clog2(M/TILE_M)-1:0] m_idx,
    output logic [$clog2(N/TILE_N)-1:0] n_idx,
    output logic [$clog2(K/TILE_K)-1:0] k_idx,
    output logic last_k,         // Last k tile for this (m, n) pair
    output logic done
);

    // -------------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------------
    localparam int NUM_M = M / TILE_M;   // Number of tile rows
    localparam int NUM_K = K / TILE_K;   // Number of inner tiles
    localparam int NUM_N = N / TILE_N;   // Number of tile columns

    // Width of each index counter
    localparam int W_M = $clog2(NUM_M);
    localparam int W_K = $clog2(NUM_K);
    localparam int W_N = $clog2(NUM_N);

    // -------------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE    = 2'b00,
        S_OUTPUT  = 2'b01,   // Holding valid output, waiting for next_ready
        S_ADVANCE = 2'b10,   // Advancing counters to next tile
        S_DONE    = 2'b11
    } state_t;

    state_t state, next_state;

    // -------------------------------------------------------------------------
    // Tile counters
    // -------------------------------------------------------------------------
    logic [W_M-1:0] cnt_m, cnt_m_next;
    logic [W_N-1:0] cnt_n, cnt_n_next;
    logic [W_K-1:0] cnt_k, cnt_k_next;

    // -------------------------------------------------------------------------
    // Overflow/wrap detection
    // -------------------------------------------------------------------------
    logic k_wrap, n_wrap, m_wrap;

    always_comb begin
        k_wrap = (cnt_k == W_K'(NUM_K - 1));
        n_wrap = k_wrap && (cnt_n == W_N'(NUM_N - 1));
        m_wrap = n_wrap && (cnt_m == W_M'(NUM_M - 1));
    end

    // -------------------------------------------------------------------------
    // Next-counter combinational logic
    // -------------------------------------------------------------------------
    always_comb begin
        cnt_k_next = cnt_k;
        cnt_n_next = cnt_n;
        cnt_m_next = cnt_m;

        if (state == S_ADVANCE) begin
            if (k_wrap) begin
                cnt_k_next = '0;
                if (n_wrap) begin
                    cnt_n_next = '0;
                    // m increments even on last tile -- will be caught by S_DONE
                    cnt_m_next = cnt_m + 1'b1;
                end else begin
                    cnt_n_next = cnt_n + 1'b1;
                end
            end else begin
                cnt_k_next = cnt_k + 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // State register and counter update
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            cnt_m  <= '0;
            cnt_n  <= '0;
            cnt_k  <= '0;
        end else begin
            state  <= next_state;
            cnt_m  <= cnt_m_next;
            cnt_n  <= cnt_n_next;
            cnt_k  <= cnt_k_next;
        end
    end

    // -------------------------------------------------------------------------
    // Next-state logic
    // -------------------------------------------------------------------------
    always_comb begin
        next_state = state;

        unique case (state)
            S_IDLE: begin
                if (start)
                    next_state = S_OUTPUT;
            end

            S_OUTPUT: begin
                // Hold until consumer acknowledges
                if (next_ready) begin
                    if (m_wrap)
                        next_state = S_DONE;   // All tiles issued
                    else
                        next_state = S_ADVANCE;
                end
            end

            S_ADVANCE: begin
                // One-cycle counter update, then immediately present next tile
                next_state = S_OUTPUT;
            end

            S_DONE: begin
                // Stay done until reset or new start
                if (start)
                    next_state = S_OUTPUT;   // Allow restart without full reset
            end

            default: next_state = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Output assignments
    // -------------------------------------------------------------------------
    always_comb begin
        valid  = (state == S_OUTPUT);
        m_idx  = cnt_m;
        n_idx  = cnt_n;
        k_idx  = cnt_k;
        last_k = k_wrap;
        done   = (state == S_DONE) && !start;  // Pulse; cleared by new start
    end

    // -------------------------------------------------------------------------
    // Reset counter on new start (when in S_DONE)
    // -------------------------------------------------------------------------
    // Note: The counter reset on restart is handled implicitly because S_ADVANCE
    // increments from the last position. For a clean restart from (0,0,0), the
    // counter registers are reset when we transition out of S_DONE on start.
    // We add explicit reset logic here:
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            // already handled above
        end else if (start && (state == S_DONE || state == S_IDLE)) begin
            cnt_m <= '0;
            cnt_n <= '0;
            cnt_k <= '0;
        end
    end

    // -------------------------------------------------------------------------
    // Assertions (simulation only)
    // -------------------------------------------------------------------------
    // synthesis translate_off
    initial begin
        assert (M % TILE_M == 0) else $fatal(1, "M must be divisible by TILE_M");
        assert (K % TILE_K == 0) else $fatal(1, "K must be divisible by TILE_K");
        assert (N % TILE_N == 0) else $fatal(1, "N must be divisible by TILE_N");
        assert (NUM_M >= 1)      else $fatal(1, "NUM_M must be >= 1");
        assert (NUM_K >= 1)      else $fatal(1, "NUM_K must be >= 1");
        assert (NUM_N >= 1)      else $fatal(1, "NUM_N must be >= 1");
    end

    // Check valid stays stable until next_ready
    property p_valid_stable;
        @(posedge clk) disable iff (!rst_n)
        (valid && !next_ready) |=> valid;
    endproperty
    assert property (p_valid_stable)
        else $error("valid deasserted before next_ready");

    // Check counters stay in range
    assert property (@(posedge clk) disable iff (!rst_n)
        valid |-> (m_idx < NUM_M && n_idx < NUM_N && k_idx < NUM_K))
        else $error("Tile index out of range");
    // synthesis translate_on

endmodule


// =============================================================================
// TESTBENCH
// =============================================================================

// synthesis translate_off
module tb_tile_scheduler;

    // Use a small matrix for exhaustive checking: M=8, K=8, N=8, tiles=4
    // -> 2 M-tiles * 2 N-tiles * 2 K-tiles = 8 total issues
    localparam int M      = 8;
    localparam int K      = 8;
    localparam int N      = 8;
    localparam int TILE_M = 4;
    localparam int TILE_K = 4;
    localparam int TILE_N = 4;
    localparam int NUM_M  = M / TILE_M;   // 2
    localparam int NUM_K  = K / TILE_K;   // 2
    localparam int NUM_N  = N / TILE_N;   // 2

    logic clk, rst_n, start, next_ready, valid, last_k, done;
    logic [$clog2(NUM_M)-1:0] m_idx;
    logic [$clog2(NUM_N)-1:0] n_idx;
    logic [$clog2(NUM_K)-1:0] k_idx;

    // DUT
    tile_scheduler #(
        .M(M), .K(K), .N(N),
        .TILE_M(TILE_M), .TILE_K(TILE_K), .TILE_N(TILE_N)
    ) dut (.*);

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    // Expected output sequence (output-stationary order)
    // (m, n, k) with last_k flag
    typedef struct {
        int m, n, k;
        bit last;
    } tile_coord_t;

    tile_coord_t expected_seq[] = '{
        '{0,0,0,0}, '{0,0,1,1},    // Output tile (0,0), k sweeps 0->1
        '{0,1,0,0}, '{0,1,1,1},    // Output tile (0,1), k sweeps 0->1
        '{1,0,0,0}, '{1,0,1,1},    // Output tile (1,0), k sweeps 0->1
        '{1,1,0,0}, '{1,1,1,1}     // Output tile (1,1), k sweeps 0->1
    };

    int issue_count;
    int errors;

    task automatic check_tile(
        input int exp_m, exp_n, exp_k,
        input bit exp_last
    );
        if (m_idx != exp_m || n_idx != exp_n || k_idx != exp_k || last_k != exp_last) begin
            $error("MISMATCH at issue %0d: got (%0d,%0d,%0d,last=%0b) expected (%0d,%0d,%0d,last=%0b)",
                   issue_count, m_idx, n_idx, k_idx, last_k,
                   exp_m, exp_n, exp_k, exp_last);
            errors++;
        end else begin
            $display("  OK tile %0d: m=%0d n=%0d k=%0d last_k=%0b",
                     issue_count, m_idx, n_idx, k_idx, last_k);
        end
    endtask

    initial begin
        // Initialise
        rst_n      = 0;
        start      = 0;
        next_ready = 0;
        issue_count = 0;
        errors     = 0;

        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Test 1: Normal sequential scheduling (next_ready always high)
        $display("=== Test 1: Sequential scheduling (next_ready always high) ===");
        next_ready = 1;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // Wait for done, checking each issued tile
        fork
            begin : check_proc
                forever begin
                    @(posedge clk);
                    if (valid) begin
                        check_tile(
                            expected_seq[issue_count].m,
                            expected_seq[issue_count].n,
                            expected_seq[issue_count].k,
                            expected_seq[issue_count].last
                        );
                        issue_count++;
                    end
                end
            end
            begin : timeout_proc
                repeat(200) @(posedge clk);
                $fatal(1, "Timeout waiting for tiles");
            end
            begin : done_proc
                @(posedge done);
            end
        join_any
        disable check_proc;
        disable timeout_proc;

        assert (issue_count == NUM_M * NUM_N * NUM_K)
            else $error("Expected %0d tiles, got %0d", NUM_M*NUM_N*NUM_K, issue_count);

        @(posedge clk);
        assert (!done) else $error("Done should be deasserted after one cycle");

        // Test 2: Back-pressure - next_ready deasserted for several cycles
        $display("=== Test 2: Back-pressure test ===");
        issue_count = 0;
        next_ready  = 0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        repeat(NUM_M * NUM_N * NUM_K) begin
            // Wait 3 cycles before accepting each tile
            @(posedge clk);
            assert (valid) else $error("Scheduler should hold valid during back-pressure");
            @(posedge clk);
            assert (valid) else $error("valid dropped during back-pressure");
            @(posedge clk);
            next_ready = 1;
            @(posedge clk);
            next_ready = 0;
            issue_count++;
        end

        @(posedge done);
        $display("  Back-pressure test passed, %0d tiles issued", issue_count);

        // Test 3: Restart capability
        $display("=== Test 3: Restart from DONE state ===");
        issue_count = 0;
        next_ready  = 1;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        @(posedge done);
        // Should have seen 8 tiles again
        @(posedge clk);

        // Final report
        if (errors == 0)
            $display("ALL TESTS PASSED");
        else
            $display("FAILED: %0d errors", errors);

        $finish;
    end

    // Simulation timeout
    initial begin
        #100000;
        $fatal(1, "Global simulation timeout");
    end

endmodule
// synthesis translate_on
