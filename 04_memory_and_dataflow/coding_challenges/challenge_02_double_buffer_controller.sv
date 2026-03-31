// =============================================================================
// Challenge 02: Double-Buffer Controller for DMA/Compute Overlap
// =============================================================================
//
// PROBLEM STATEMENT
// -----------------
// An LLM accelerator must overlap DMA transfers (loading the next weight tile
// from HBM into SRAM) with the PE array computing the current weight tile.
// This is achieved with two SRAM banks (Bank 0 and Bank 1) used in a
// ping-pong (double-buffer) scheme.
//
// At any moment:
//   ACTIVE bank  -> being read by the compute unit
//   INACTIVE bank -> being written by the DMA engine
//
// After each tile compute completes AND each DMA transfer completes, the
// banks swap roles.
//
// SYSTEM ARCHITECTURE
//
//   ┌─────────────────────────────────────────────────────────────┐
//   │                   Double-Buffer Controller                   │
//   │                                                              │
//   │  ┌──────────┐   compute_bank   ┌──────────────────────────┐ │
//   │  │  Bank 0  │◄─────────────────│                          │ │
//   │  │  (SRAM)  │                  │     Compute Unit         │ │
//   │  └──────────┘   dma_bank       │  (PE Array / Systolic)   │ │
//   │  ┌──────────┐◄─────────────────│                          │ │
//   │  │  Bank 1  │                  └──────────────────────────┘ │
//   │  │  (SRAM)  │                                                │
//   │  └──────────┘                                                │
//   └─────────────────────────────────────────────────────────────┘
//
// INTERFACE
// ---------
// Inputs:
//   clk             : Clock
//   rst_n           : Active-low synchronous reset
//   start           : Begin double-buffering; first DMA is issued immediately
//   num_tiles       : Total number of tiles to process (must be >= 1)
//
//   -- DMA engine interface --
//   dma_done        : DMA engine signals transfer into inactive bank is complete
//   dma_last        : DMA engine signals this is the last transfer (no more tiles)
//
//   -- Compute unit interface --
//   compute_done    : Compute unit signals it finished processing the active bank
//
// Outputs:
//   -- DMA engine interface --
//   dma_start       : Pulse: tells DMA engine to start loading next tile
//   dma_bank_sel    : Which bank (0 or 1) DMA should write into
//
//   -- Compute unit interface --
//   compute_start   : Pulse: tells compute unit to start processing active bank
//   compute_bank_sel: Which bank (0 or 1) compute should read from
//
//   -- Status --
//   busy            : High while the double-buffer pipeline is active
//   all_done        : Pulse when the last compute tile has finished
//   stall_cycles    : Count of cycles where compute was ready but DMA not done
//                     (performance counter, wraps at max value)
//
// PROTOCOL
// --------
// 1. Host asserts start for one cycle. Controller immediately issues dma_start
//    to load tile 0 into bank 0.
// 2. When dma_done arrives (tile 0 ready in bank 0), controller:
//    a. Issues compute_start with compute_bank_sel = 0
//    b. Issues dma_start for tile 1 into bank 1 (if more tiles exist)
// 3. When BOTH compute_done and dma_done arrive (in any order):
//    a. Swap bank roles
//    b. Issue compute_start with new active bank
//    c. Issue dma_start for next tile into new inactive bank
// 4. On the last tile (no more DMA), when compute_done arrives: assert all_done.
//
// TIMING DIAGRAM (3 tiles, no back-pressure)
//
// Cycle:    0   1   2   3   4   5   6   7   8   9   10  11  12
// start:    _/‾\_____________________________________
// dma_done:                 ____                ____            ____
//                  (tile 0 done)  (tile 1 done)       (tile 2 done)
// compute:           ┌────────────┐   ┌────────────┐  ┌────────────┐
//                    │  tile 0    │   │  tile 1    │  │  tile 2    │
// bank:              │  bank 0    │   │  bank 1    │  │  bank 0    │
// DMA:       ┌───┐  ┌────────────┐   ┌────────────┐  (no more)
//            │ 0 │  │  tile 1    │   │  tile 2    │
//            │ B0│  │  bank 1    │   │  bank 0    │
//
// =============================================================================

module double_buffer_controller #(
    parameter int STALL_CTR_WIDTH = 16   // Width of stall cycle counter
) (
    input  logic clk,
    input  logic rst_n,

    // Host control
    input  logic start,
    input  logic [15:0] num_tiles,       // Total tiles to process (>= 1)

    // DMA engine interface
    input  logic dma_done,               // Transfer into inactive bank complete
    output logic dma_start,              // Initiate next DMA transfer
    output logic dma_bank_sel,           // Bank DMA should write into (0 or 1)

    // Compute unit interface
    input  logic compute_done,           // Compute on active bank complete
    output logic compute_start,          // Begin compute on active bank
    output logic compute_bank_sel,       // Bank compute should read from (0 or 1)

    // Status
    output logic busy,
    output logic all_done,
    output logic [STALL_CTR_WIDTH-1:0] stall_cycles
);

    // -------------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------------
    // The pipeline has two phases per slot:
    //   PHASE_FILL    : Waiting for DMA to fill the inactive bank (startup)
    //   PHASE_OVERLAP : Both compute and DMA running concurrently
    //   PHASE_DRAIN   : Last tile computing, no DMA pending
    //
    typedef enum logic [2:0] {
        S_IDLE        = 3'd0,
        S_FIRST_FILL  = 3'd1,   // Waiting for first DMA to complete (no compute yet)
        S_OVERLAP     = 3'd2,   // Compute and DMA both in flight
        S_WAIT_DMA    = 3'd3,   // Compute done, waiting for DMA to finish
        S_WAIT_COMP   = 3'd4,   // DMA done, waiting for compute to finish
        S_DRAIN       = 3'd5,   // Last tile in compute, no more DMA
        S_DONE        = 3'd6
    } state_t;

    state_t state, next_state;

    // -------------------------------------------------------------------------
    // Internal registers
    // -------------------------------------------------------------------------
    logic       active_bank;          // Which bank is currently being computed
    logic [15:0] tiles_issued;        // Number of DMA requests issued (not computed)
    logic [15:0] tiles_computed;      // Number of tiles compute has finished

    logic       compute_pending;      // Compute has started but not yet done
    logic       dma_pending;          // DMA has started but not yet done

    // Derived signals
    logic       more_tiles_to_dma;    // There are still tiles to fetch
    logic       last_compute;         // Current compute tile is the last one

    always_comb begin
        more_tiles_to_dma = (tiles_issued < num_tiles);
        last_compute      = (tiles_computed == num_tiles - 1);
    end

    // -------------------------------------------------------------------------
    // Sequential state and register updates
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            active_bank    <= 1'b0;
            tiles_issued   <= '0;
            tiles_computed <= '0;
            compute_pending<= 1'b0;
            dma_pending    <= 1'b0;
        end else begin
            state <= next_state;

            // Track DMA issues and completions
            if (dma_start)
                tiles_issued <= tiles_issued + 1'b1;

            // Track compute completions
            if (compute_done)
                tiles_computed <= tiles_computed + 1'b1;

            // Bank swap on overlap resolution
            if ((state == S_OVERLAP  && compute_done && dma_done) ||
                (state == S_WAIT_DMA && dma_done)                 ||
                (state == S_WAIT_COMP && compute_done))
                active_bank <= ~active_bank;

            // Reset on start
            if (start && state == S_IDLE) begin
                tiles_issued   <= '0;
                tiles_computed <= '0;
                active_bank    <= 1'b0;
            end
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
                    next_state = S_FIRST_FILL;  // Issue DMA for tile 0
            end

            S_FIRST_FILL: begin
                // Waiting for first DMA; no compute yet
                if (dma_done) begin
                    if (num_tiles == 1)
                        next_state = S_DRAIN;   // Only one tile: no overlap possible
                    else
                        next_state = S_OVERLAP; // Start compute + prefetch tile 1
                end
            end

            S_OVERLAP: begin
                // Both compute and DMA in flight
                if (compute_done && dma_done) begin
                    // Both finished simultaneously
                    if (!more_tiles_to_dma || tiles_computed + 1 == num_tiles - 1)
                        // After swap, next compute is the last tile
                        next_state = (more_tiles_to_dma) ? S_DRAIN : S_DONE;
                    else
                        next_state = S_OVERLAP; // Both restart immediately
                end else if (compute_done) begin
                    next_state = S_WAIT_DMA;    // Stall: waiting for DMA
                end else if (dma_done) begin
                    next_state = S_WAIT_COMP;   // DMA early: wait for compute
                end
            end

            S_WAIT_DMA: begin
                // Compute done, DMA still running -- stalling
                if (dma_done) begin
                    if (tiles_computed + 1 >= num_tiles)
                        next_state = S_DONE;    // No more tiles to compute after swap
                    else if (!more_tiles_to_dma)
                        next_state = S_DRAIN;
                    else
                        next_state = S_OVERLAP;
                end
            end

            S_WAIT_COMP: begin
                // DMA done early, compute still running
                if (compute_done) begin
                    if (!more_tiles_to_dma)
                        next_state = S_DRAIN;
                    else
                        next_state = S_OVERLAP;
                end
            end

            S_DRAIN: begin
                // Last tile in compute, no DMA pending
                if (compute_done)
                    next_state = S_DONE;
            end

            S_DONE: begin
                next_state = S_IDLE;            // Auto-return to idle after 1 cycle
            end

            default: next_state = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Output logic
    // -------------------------------------------------------------------------

    // dma_start: pulse when transitioning from IDLE->FIRST_FILL (tile 0),
    //            and when a bank swap occurs and more tiles remain.
    always_comb begin
        dma_start = 1'b0;

        case (state)
            S_IDLE: dma_start = start && (num_tiles >= 1);

            // After first DMA lands, immediately start DMA for tile 1
            S_FIRST_FILL: dma_start = dma_done && more_tiles_to_dma && (num_tiles > 1);

            // After a successful overlap resolution, issue next DMA
            S_OVERLAP: begin
                dma_start = dma_done && more_tiles_to_dma;
            end

            S_WAIT_DMA: dma_start = dma_done && more_tiles_to_dma;
            S_WAIT_COMP: dma_start = 1'b0;  // DMA already done, compute catching up

            default: dma_start = 1'b0;
        endcase
    end

    // DMA bank: always the INACTIVE bank (opposite of active)
    assign dma_bank_sel = ~active_bank;

    // compute_start: pulse when a new tile is ready to be computed
    always_comb begin
        compute_start = 1'b0;

        case (state)
            // First compute starts when first DMA completes
            S_FIRST_FILL: compute_start = dma_done;

            // After overlap, if both just finished, start next compute immediately
            S_OVERLAP: compute_start = dma_done && compute_done;

            // DMA just finished while compute was stalling us
            S_WAIT_DMA: compute_start = dma_done;

            // Compute finished, DMA was already done -- start compute on next bank
            S_WAIT_COMP: compute_start = compute_done;

            default: compute_start = 1'b0;
        endcase
    end

    // Compute bank: always the ACTIVE bank (but after swap, active_bank already flipped)
    // compute_start is asserted on the same cycle as the swap, so we use the post-swap value.
    // Since active_bank updates in the FF on the same edge, compute_start should use the
    // next-cycle value. We hold compute_bank_sel as the registered active bank, which will
    // be correct by the cycle after compute_start is issued.
    //
    // Simpler model: compute_start is a "go" signal; compute_bank_sel tells compute which
    // bank to use. We present the CURRENT active_bank (before swap) alongside compute_start.
    // The bank swap happens on the same clock edge as compute_start, so the compute unit
    // should register compute_bank_sel when compute_start is asserted.
    always_comb begin
        if (state == S_FIRST_FILL && dma_done)
            compute_bank_sel = active_bank;  // First tile: active bank hasn't swapped yet
        else
            compute_bank_sel = ~active_bank; // After swap, the new active bank is ~old active
    end

    // Status
    assign busy       = (state != S_IDLE && state != S_DONE);
    assign all_done   = (state == S_DONE);

    // -------------------------------------------------------------------------
    // Stall cycle counter
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            stall_cycles <= '0;
        end else begin
            if (start && state == S_IDLE)
                stall_cycles <= '0;                    // Reset on new job
            else if (state == S_WAIT_DMA)
                stall_cycles <= stall_cycles + 1'b1;   // Saturating not needed; wraps OK
        end
    end

    // -------------------------------------------------------------------------
    // Assertions (simulation only)
    // -------------------------------------------------------------------------
    // synthesis translate_off

    // DMA and compute should never both start on the same cycle for the same bank
    assert property (@(posedge clk) disable iff (!rst_n)
        !(dma_start && compute_start && (dma_bank_sel == compute_bank_sel)))
        else $error("DMA and compute targeting same bank simultaneously");

    // Bank select must be valid binary
    assert property (@(posedge clk) disable iff (!rst_n)
        busy |-> (dma_bank_sel !== 1'bx && compute_bank_sel !== 1'bx))
        else $error("Bank select is X during busy");

    // dma_start must not be asserted in DRAIN or DONE (no tiles to prefetch)
    assert property (@(posedge clk) disable iff (!rst_n)
        (state == S_DRAIN || state == S_DONE) |-> !dma_start)
        else $error("DMA started in DRAIN/DONE state");

    // synthesis translate_on

endmodule


// =============================================================================
// TESTBENCH
// =============================================================================

// synthesis translate_off
module tb_double_buffer_controller;

    logic clk, rst_n, start;
    logic [15:0] num_tiles;
    logic dma_done, dma_start, dma_bank_sel;
    logic compute_done, compute_start, compute_bank_sel;
    logic busy, all_done;
    logic [15:0] stall_cycles;

    double_buffer_controller dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    // Simple DMA and compute models
    // dma_delay and compute_delay can be varied to test different overlap scenarios

    task automatic run_test(
        input int n_tiles,
        input int dma_delay_cycles,   // cycles DMA takes per tile after dma_start
        input int comp_delay_cycles,  // cycles compute takes per tile after compute_start
        input string test_name
    );
        int tiles_checked;
        int start_cycle;
        int expected_stalls;

        $display("=== %s: %0d tiles, DMA=%0d cycles, Compute=%0d cycles ===",
                 test_name, n_tiles, dma_delay_cycles, comp_delay_cycles);

        @(posedge clk);
        num_tiles = n_tiles;
        start = 1;
        @(posedge clk);
        start = 0;
        start_cycle = $time;

        // Drive DMA done responses
        fork
            // DMA model: respond to dma_start after dma_delay_cycles
            begin : dma_model
                int issued = 0;
                while (issued < n_tiles) begin
                    @(posedge clk iff dma_start);
                    issued++;
                    repeat(dma_delay_cycles - 1) @(posedge clk);
                    dma_done = 1;
                    @(posedge clk);
                    dma_done = 0;
                end
            end

            // Compute model: respond to compute_start after comp_delay_cycles
            begin : compute_model
                int computed = 0;
                while (computed < n_tiles) begin
                    @(posedge clk iff compute_start);
                    computed++;
                    repeat(comp_delay_cycles - 1) @(posedge clk);
                    compute_done = 1;
                    @(posedge clk);
                    compute_done = 0;
                end
            end

            // Timeout watchdog
            begin : watchdog
                repeat(10000) @(posedge clk);
                $fatal(1, "Test '%s' timed out", test_name);
            end
        join_any
        disable dma_model;
        disable compute_model;
        disable watchdog;

        @(posedge all_done);
        @(posedge clk);  // Observe done for one cycle

        $display("  Completed. Stall cycles = %0d", stall_cycles);
        if (dma_delay_cycles > comp_delay_cycles)
            $display("  Expected stalls (DMA-bound): %0d stalls per tile",
                     dma_delay_cycles - comp_delay_cycles);
        else
            $display("  No stalls expected (compute-bound or balanced)");

        // Verify not still busy
        assert (!busy) else $error("Controller still busy after all_done");
        @(posedge clk);

    endtask

    initial begin
        rst_n       = 0;
        start       = 0;
        num_tiles   = 0;
        dma_done    = 0;
        compute_done= 0;

        repeat(4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Test 1: Single tile (no overlap possible)
        run_test(1, 4, 4, "Single tile");

        // Test 2: Balanced (DMA == compute) - no stalls expected
        run_test(8, 10, 10, "Balanced DMA==Compute");

        // Test 3: Compute-bound (compute slower than DMA) - no stalls expected
        run_test(6, 5, 15, "Compute-bound (DMA fast)");

        // Test 4: DMA-bound (DMA slower than compute) - stalls expected
        run_test(6, 15, 5, "DMA-bound (stalls expected)");

        // Test 5: Large tile count
        run_test(32, 8, 8, "32 tiles balanced");

        $display("ALL TESTS PASSED");
        $finish;
    end

    // Monitor: display key transitions
    always @(posedge clk) begin
        if (dma_start)
            $display("  t=%0t DMA_START -> bank %0b", $time, dma_bank_sel);
        if (dma_done)
            $display("  t=%0t DMA_DONE", $time);
        if (compute_start)
            $display("  t=%0t COMPUTE_START <- bank %0b", $time, compute_bank_sel);
        if (compute_done)
            $display("  t=%0t COMPUTE_DONE", $time);
        if (all_done)
            $display("  t=%0t ALL_DONE (stalls=%0d)", $time, stall_cycles);
    end

    initial #500000 $fatal(1, "Global simulation timeout");

endmodule
// synthesis translate_on
