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
    // The protocol is tracked per bank rather than with one state per overlap
    // case: the FSM only says whether a job is running, and the start
    // decisions below follow from which bank is full / being filled / being
    // computed. With two banks this gives exactly the PROTOCOL above:
    //   - DMA fills banks 0,1,0,1,... ; compute reads banks 0,1,0,1,...
    //   - the next DMA starts when the DMA engine is free AND its bank is empty
    //     (its previous tile has been computed) -> "both done" in step 3
    //   - the next compute starts when the compute unit is free AND its bank
    //     has been filled.
    // Starts are issued in the same cycle as the done pulse that enables them.
    //
    typedef enum logic [1:0] {
        S_IDLE        = 2'd0,
        S_RUN         = 2'd1,   // DMA and/or compute in flight
        S_DONE        = 2'd2    // One cycle: all_done
    } state_t;

    state_t state;

    // -------------------------------------------------------------------------
    // Internal registers
    // -------------------------------------------------------------------------
    logic [15:0] n_tiles;             // num_tiles latched at start
    logic [15:0] tiles_issued;        // Number of DMA requests issued
    logic [15:0] comp_issued;         // Number of compute requests issued
    logic [15:0] tiles_computed;      // Number of tiles compute has finished

    logic       fill_bank;            // Bank the next DMA writes
    logic       comp_bank;            // Bank the next compute reads
    logic       dma_busy, dma_cur;    // DMA in flight, and the bank it is filling
    logic       comp_busy, comp_cur;  // Compute in flight, and the bank it is reading
    logic [1:0] bank_full;            // Bank holds a loaded tile not yet computed

    // "Effective" view for this cycle: registered state updated with this
    // cycle's done pulses, so a start can be issued in the same cycle as the
    // done that allows it
    logic       dma_busy_eff, comp_busy_eff;
    logic [1:0] full_eff;
    logic [15:0] computed_eff;

    always_comb begin
        dma_busy_eff  = dma_busy  && !dma_done;
        comp_busy_eff = comp_busy && !compute_done;
        for (int b = 0; b < 2; b++)
            full_eff[b] = (bank_full[b] || (dma_busy && dma_done && dma_cur == b))
                          && !(comp_busy && compute_done && comp_cur == b);
        computed_eff  = tiles_computed + ((comp_busy && compute_done) ? 16'd1 : 16'd0);
    end

    // -------------------------------------------------------------------------
    // Start decisions
    // -------------------------------------------------------------------------
    always_comb begin
        dma_start     = 1'b0;
        compute_start = 1'b0;
        if (state == S_IDLE) begin
            // Tile 0 into bank 0 as soon as the job starts
            dma_start = start && (num_tiles >= 1);
        end else if (state == S_RUN) begin
            // (With in-order ping-pong either bank condition alone is enough --
            //  a full fill bank is always the one being computed -- but both
            //  are kept so the safety rule reads directly off the code.)
            dma_start     = !dma_busy_eff && (tiles_issued < n_tiles) &&
                            !full_eff[fill_bank] && !(comp_busy_eff && comp_cur == fill_bank);
            compute_start = !comp_busy_eff && (comp_issued < n_tiles) && full_eff[comp_bank];
        end
    end

    // Bank selects, valid in the cycle of the corresponding start pulse (the
    // DMA engine / compute unit registers them together with the start)
    assign dma_bank_sel     = (state == S_IDLE) ? 1'b0 : fill_bank;
    assign compute_bank_sel = comp_bank;

    // -------------------------------------------------------------------------
    // Sequential state and register updates
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            n_tiles        <= '0;
            tiles_issued   <= '0;
            comp_issued    <= '0;
            tiles_computed <= '0;
            fill_bank      <= 1'b0;
            comp_bank      <= 1'b0;
            dma_busy       <= 1'b0;
            dma_cur        <= 1'b0;
            comp_busy      <= 1'b0;
            comp_cur       <= 1'b0;
            bank_full      <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        // New job: tile 0 is issued to bank 0 this cycle
                        n_tiles        <= num_tiles;
                        tiles_issued   <= (num_tiles >= 1) ? 16'd1 : 16'd0;
                        comp_issued    <= '0;
                        tiles_computed <= '0;
                        fill_bank      <= 1'b1;
                        comp_bank      <= 1'b0;
                        dma_busy       <= (num_tiles >= 1);
                        dma_cur        <= 1'b0;
                        comp_busy      <= 1'b0;
                        bank_full      <= '0;
                        state          <= (num_tiles >= 1) ? S_RUN : S_DONE;
                    end
                end

                S_RUN: begin
                    bank_full      <= full_eff;
                    dma_busy       <= dma_busy_eff;
                    comp_busy      <= comp_busy_eff;
                    tiles_computed <= computed_eff;
                    if (dma_start) begin
                        dma_busy     <= 1'b1;
                        dma_cur      <= fill_bank;
                        fill_bank    <= ~fill_bank;
                        tiles_issued <= tiles_issued + 1'b1;
                    end
                    if (compute_start) begin
                        comp_busy   <= 1'b1;
                        comp_cur    <= comp_bank;
                        comp_bank   <= ~comp_bank;
                        comp_issued <= comp_issued + 1'b1;
                    end
                    if (computed_eff == n_tiles)
                        state <= S_DONE;            // Last compute tile finished
                end

                S_DONE: begin
                    state <= S_IDLE;                // Auto-return to idle after 1 cycle
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Status
    assign busy       = (state == S_RUN);
    assign all_done   = (state == S_DONE);

    // -------------------------------------------------------------------------
    // Stall cycle counter
    // A stall cycle: compute unit idle (after its first tile) with tiles left
    // to compute, but the next bank is not yet filled -- waiting on the DMA.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            stall_cycles <= '0;
        end else begin
            if (start && state == S_IDLE)
                stall_cycles <= '0;                    // Reset on new job
            else if (state == S_RUN && !comp_busy_eff && computed_eff != 0 &&
                     comp_issued < n_tiles && !compute_start)
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

    // No more than num_tiles DMA transfers per job
    assert property (@(posedge clk) disable iff (!rst_n)
        (state == S_RUN && dma_start) |-> (tiles_issued < n_tiles))
        else $error("DMA started with no tiles left to fetch");

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

    // -------------------------------------------------------------------------
    // DMA and compute models (cycle based, sampled on posedge, driven with
    // NBAs so the DUT sees them race-free). A request seen at edge E completes
    // with a one-cycle done pulse sampled at edge E + delay.
    // The models move real tile ids through the two banks:
    //   DMA writes tile t into bank_mem[bank] when it completes;
    //   compute checks it reads tile 0,1,2,... in order from the bank it was
    //   given, and that no DMA ever targets a bank still holding an
    //   uncomputed tile or being computed.
    // -------------------------------------------------------------------------
    int  dma_delay, comp_delay;        // fixed delays (cycles, >= 1)
    bit  rand_delay;                   // per-tile random delays 1..12 instead
    int  bank_mem [2];                 // tile id held by each bank (-1 = empty)
    int  dma_rem,  comp_rem;
    bit  dma_act,  comp_act;
    int  dma_b,    comp_b;
    int  dma_tile, comp_tile;          // next tile id for DMA / expected by compute
    int  computed, n_dma, n_comp, n_done, stalls_seen, errors;
    bit  running;
    longint t_start, t_last;           // edge counts
    longint edge_no;

    always @(posedge clk) begin
        edge_no++;
        dma_done     <= 1'b0;
        compute_done <= 1'b0;

        // ---- stall observation (compute ready but next tile not loaded) ----
        if (running && !comp_act && computed > 0 && n_comp < num_tiles && !compute_start)
            stalls_seen++;

        // ---- DMA model ----
        if (dma_start) begin
            n_dma++;
            if (dma_act) begin
                $display("  ERROR: dma_start while a DMA is in flight"); errors++;
            end
            if (bank_mem[dma_bank_sel] != -1) begin
                $display("  ERROR: DMA into bank %0b which still holds uncomputed tile %0d",
                         dma_bank_sel, bank_mem[dma_bank_sel]); errors++;
            end
            if (comp_act && comp_b == dma_bank_sel) begin
                $display("  ERROR: DMA into bank %0b while it is being computed", dma_bank_sel); errors++;
            end
            dma_act = 1; dma_b = dma_bank_sel;
            dma_rem = (rand_delay ? $urandom_range(12, 1) : dma_delay) - 1;
        end else if (dma_act)
            dma_rem--;
        if (dma_act && dma_rem == 0) begin
            dma_done <= 1'b1;
            dma_act   = 0;
            bank_mem[dma_b] = dma_tile++;
        end

        // ---- compute model ----
        if (compute_start) begin
            n_comp++;
            if (comp_act) begin
                $display("  ERROR: compute_start while compute is busy"); errors++;
            end
            if (bank_mem[compute_bank_sel] != comp_tile) begin
                $display("  ERROR: compute %0d reads bank %0b holding tile %0d (expected tile %0d)",
                         n_comp - 1, compute_bank_sel, bank_mem[compute_bank_sel], comp_tile); errors++;
            end
            comp_act = 1; comp_b = compute_bank_sel; comp_tile++;
            comp_rem = (rand_delay ? $urandom_range(12, 1) : comp_delay) - 1;
        end else if (comp_act)
            comp_rem--;
        if (comp_act && comp_rem == 0) begin
            compute_done <= 1'b1;
            comp_act = 0;
            bank_mem[comp_b] = -1;          // tile consumed, bank free
            computed++;
            t_last = edge_no + 1;           // done is sampled at the next edge
        end

        if (all_done) n_done++;
    end

    task automatic run_test(
        input int n_tiles,
        input int dma_delay_cycles,   // cycles DMA takes per tile after dma_start
        input int comp_delay_cycles,  // cycles compute takes per tile after compute_start
        input bit random,             // random 1..12 delays per tile instead
        input string test_name
    );
        longint t_exp, stall_exp;
        int     slow;

        $display("=== %s: %0d tiles, DMA=%0d cycles, Compute=%0d cycles%s ===",
                 test_name, n_tiles, dma_delay_cycles, comp_delay_cycles,
                 random ? " (random 1..12)" : "");

        dma_delay  = dma_delay_cycles;
        comp_delay = comp_delay_cycles;
        rand_delay = random;
        bank_mem   = '{-1, -1};
        dma_tile = 0; comp_tile = 0; computed = 0;
        n_dma = 0; n_comp = 0; n_done = 0; stalls_seen = 0;

        @(negedge clk);
        num_tiles = n_tiles;
        start = 1;
        running = 1;
        t_start = edge_no + 1;              // start is sampled at the next edge
        @(negedge clk);
        start = 0;

        fork
            begin : wait_done
                wait (n_done > 0);
            end
            begin : watchdog
                repeat(10000) @(posedge clk);
                $display("  ERROR: Test '%s' timed out", test_name);
                errors++;
            end
        join_any
        disable wait_done;
        disable watchdog;
        running = 0;
        repeat (3) @(negedge clk);          // all_done must not repeat

        $display("  Completed in %0d cycles. Stall cycles = %0d (observed %0d)",
                 t_last - t_start, stall_cycles, stalls_seen);
        if (n_dma != n_tiles || n_comp != n_tiles || computed != n_tiles || n_done != 1) begin
            $display("  ERROR: %0d DMAs, %0d computes started, %0d computed, %0d all_done cycles (expected %0d, %0d, %0d, 1)",
                     n_dma, n_comp, computed, n_done, n_tiles, n_tiles, n_tiles);
            errors++;
        end
        if (stall_cycles != stalls_seen) begin
            $display("  ERROR: stall_cycles = %0d, observed %0d", stall_cycles, stalls_seen);
            errors++;
        end
        if (!random) begin
            // Ideal double-buffered schedule: tile 0 load, then n-1 steps of the
            // slower engine, then the last compute. Each compute after the first
            // waits max(0, D-C) cycles for its data.
            slow      = (dma_delay_cycles > comp_delay_cycles) ? dma_delay_cycles : comp_delay_cycles;
            t_exp     = dma_delay_cycles + (n_tiles - 1) * slow + comp_delay_cycles;
            stall_exp = (dma_delay_cycles > comp_delay_cycles) ?
                        (n_tiles - 1) * (dma_delay_cycles - comp_delay_cycles) : 0;
            if (t_last - t_start != t_exp || stall_cycles != stall_exp) begin
                $display("  ERROR: %0d cycles / %0d stalls, expected %0d / %0d (full overlap)",
                         t_last - t_start, stall_cycles, t_exp, stall_exp);
                errors++;
            end
        end
        if (busy) begin
            $display("  ERROR: Controller still busy after all_done");
            errors++;
        end
    endtask

    initial begin
        rst_n       = 0;
        start       = 0;
        num_tiles   = 0;
        dma_done    = 0;
        compute_done= 0;
        errors      = 0;
        running     = 0;
        edge_no     = 0;
        dma_act     = 0;
        comp_act    = 0;

        repeat(4) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // Test 1: Single tile (no overlap possible)
        run_test(1, 4, 4, 0, "Single tile");

        // Test 2: Two tiles
        run_test(2, 3, 6, 0, "Two tiles");

        // Test 3: Balanced (DMA == compute) - no stalls expected
        run_test(8, 10, 10, 0, "Balanced DMA==Compute");

        // Test 4: Compute-bound (compute slower than DMA) - no stalls expected
        run_test(6, 5, 15, 0, "Compute-bound (DMA fast)");

        // Test 5: DMA-bound (DMA slower than compute) - stalls expected
        run_test(6, 15, 5, 0, "DMA-bound (stalls expected)");

        // Test 6: Large tile count, single-cycle engines
        run_test(32, 1, 1, 0, "32 tiles, 1-cycle DMA and compute");

        // Test 7: Random per-tile delays
        run_test(40, 0, 0, 1, "Random delays");

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("TESTS FAILED: %0d error(s)", errors);
        $finish;
    end

    initial #500000 $fatal(1, "Global simulation timeout");

endmodule
// synthesis translate_on
