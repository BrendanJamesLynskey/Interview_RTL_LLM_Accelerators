// =============================================================================
// Challenge 03: Simplified AXI4 DMA Engine
// =============================================================================
//
// PROBLEM STATEMENT
// -----------------
// Implement a DMA engine that moves data between system memory (accessed via
// an AXI4 master interface) and a local on-chip scratchpad SRAM. The engine
// supports:
//   - Configurable burst length (1-256 AXI beats per transaction)
//   - Read-from-system-memory -> write-to-scratchpad (LOAD direction)
//   - Read-from-scratchpad -> write-to-system-memory (STORE direction)
//   - 64-bit AXI data bus (AXDATA_W = 64)
//   - 32-bit AXI address space
//
// AXI4 PROTOCOL NOTES
// -------------------
// Read channel:
//   AR channel: address + burst info issued by master
//   R  channel: data returned by slave, one beat per cycle, RLAST on last beat
//
// Write channel:
//   AW channel: address + burst info issued by master
//   W  channel: data issued by master, WLAST on last beat
//   B  channel: write response from slave
//
// Burst type INCR (AXBURST=2'b01) means addresses increment by (data_width/8)
// per beat. Maximum burst length: 256 beats (AXLEN = 8-bit field, value = len-1).
//
// TRANSFER PROTOCOL
// -----------------
// 1. Host writes: src_addr, dst_addr, byte_len, burst_len, direction to config registers.
// 2. Host asserts go for one cycle.
// 3. Engine decomposes the total transfer into AXI bursts of at most burst_len beats.
// 4. Engine asserts done for one cycle when all data has been transferred.
//
// For LOAD (system -> scratchpad):
//   AXI read bursts from src_addr -> scratchpad write port at dst_addr (local).
//
// For STORE (scratchpad -> system):
//   Scratchpad read port at src_addr (local) -> AXI write bursts to dst_addr.
//
// INTERFACE SIGNALS
// -----------------
// See port list below. Scratchpad is a simple single-port SRAM interface
// (one address, one data, write enable).
//
// SIMPLIFICATIONS (relative to full AXI4)
// ----------------------------------------
// - AXID = 0 (single ID, no out-of-order support)
// - AXPROT = 0 (normal, non-secure, non-privileged)
// - AXCACHE = 4'b0011 (normal non-cacheable bufferable)
// - AXLOCK = 0, AXQOS = 0, AXREGION = 0
// - AXSIZE = 3'b011 (8 bytes / 64-bit per beat, always)
// - No address crossing check (host must align transfers to 4 KB boundaries)
//
// EXPECTED TIMING (LOAD, 4-beat burst, RREADY always high):
//
// Cycle:  0  1  2  3  4  5  6  7  8  9
// ARVALID:_/‾‾\__________________________
// ARREADY:      ‾\______ (slave accepts at cycle 2)
// RVALID: __________/‾‾‾‾‾‾‾‾‾‾\________
// RDATA:           [D0][D1][D2][D3]
// RLAST:                       /‾\_______
// SP_WEN: __________/‾‾‾‾‾‾‾‾‾‾\________  (write to scratchpad each beat)
//
// =============================================================================

module axi_dma_engine #(
    parameter int AXI_ADDR_W  = 32,    // AXI address width
    parameter int AXI_DATA_W  = 64,    // AXI data bus width (bits)
    parameter int SP_ADDR_W   = 20,    // Scratchpad address width (byte-addressed)
    parameter int MAX_BURST   = 256    // Maximum AXI burst length in beats
) (
    input  logic clk,
    input  logic rst_n,

    // -------------------------------------------------------------------------
    // Configuration interface (memory-mapped registers)
    // -------------------------------------------------------------------------
    input  logic [AXI_ADDR_W-1:0] cfg_sys_addr,    // System memory address
    input  logic [SP_ADDR_W-1:0]  cfg_sp_addr,     // Scratchpad base address
    input  logic [23:0]            cfg_byte_len,    // Total transfer size in bytes
    input  logic [7:0]             cfg_burst_len,   // Beats per AXI burst (1-256)
    input  logic                   cfg_direction,   // 0=LOAD (sys->sp), 1=STORE (sp->sys)
    input  logic                   go,              // Start transfer (1-cycle pulse)
    output logic                   done,            // Transfer complete (1-cycle pulse)
    output logic                   busy,            // Transfer in progress

    // -------------------------------------------------------------------------
    // AXI4 Master Read Address Channel (AR)
    // -------------------------------------------------------------------------
    output logic [AXI_ADDR_W-1:0]     m_axi_araddr,
    output logic [7:0]                 m_axi_arlen,   // Burst length - 1
    output logic [2:0]                 m_axi_arsize,  // 3'b011 = 8 bytes
    output logic [1:0]                 m_axi_arburst, // 2'b01 = INCR
    output logic                       m_axi_arvalid,
    input  logic                       m_axi_arready,

    // -------------------------------------------------------------------------
    // AXI4 Master Read Data Channel (R)
    // -------------------------------------------------------------------------
    input  logic [AXI_DATA_W-1:0]     m_axi_rdata,
    input  logic                       m_axi_rlast,
    input  logic                       m_axi_rvalid,
    output logic                       m_axi_rready,

    // -------------------------------------------------------------------------
    // AXI4 Master Write Address Channel (AW)
    // -------------------------------------------------------------------------
    output logic [AXI_ADDR_W-1:0]     m_axi_awaddr,
    output logic [7:0]                 m_axi_awlen,
    output logic [2:0]                 m_axi_awsize,
    output logic [1:0]                 m_axi_awburst,
    output logic                       m_axi_awvalid,
    input  logic                       m_axi_awready,

    // -------------------------------------------------------------------------
    // AXI4 Master Write Data Channel (W)
    // -------------------------------------------------------------------------
    output logic [AXI_DATA_W-1:0]     m_axi_wdata,
    output logic                       m_axi_wlast,
    output logic                       m_axi_wvalid,
    input  logic                       m_axi_wready,

    // -------------------------------------------------------------------------
    // AXI4 Master Write Response Channel (B)
    // -------------------------------------------------------------------------
    input  logic [1:0]                 m_axi_bresp,   // 0=OKAY
    input  logic                       m_axi_bvalid,
    output logic                       m_axi_bready,

    // -------------------------------------------------------------------------
    // Scratchpad SRAM interface
    // -------------------------------------------------------------------------
    output logic [SP_ADDR_W-1:0]      sp_addr,
    output logic [AXI_DATA_W-1:0]     sp_wdata,
    output logic                       sp_wen,         // Write enable (LOAD direction)
    input  logic [AXI_DATA_W-1:0]     sp_rdata,       // Read data (STORE direction)
    output logic                       sp_ren          // Read enable (STORE direction)
);

    // -------------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------------
    localparam int BYTES_PER_BEAT = AXI_DATA_W / 8;   // 8 bytes

    // -------------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE          = 4'd0,
        S_LOAD_AR       = 4'd1,   // Issue AR channel for read burst
        S_LOAD_R        = 4'd2,   // Receive R channel data, write to scratchpad
        S_STORE_AR_READ = 4'd3,   // Read from scratchpad for store
        S_STORE_AW      = 4'd4,   // Issue AW channel for write burst
        S_STORE_W       = 4'd5,   // Issue W channel data
        S_STORE_B       = 4'd6,   // Wait for write response
        S_DONE          = 4'd7
    } state_t;

    state_t state, next_state;

    // -------------------------------------------------------------------------
    // Transfer tracking registers
    // -------------------------------------------------------------------------
    logic [AXI_ADDR_W-1:0] sys_addr_reg;   // Current system address pointer
    logic [SP_ADDR_W-1:0]  sp_addr_reg;    // Current scratchpad address pointer
    logic [23:0]            bytes_remaining;
    logic [7:0]             burst_len_reg;  // Latched burst length

    // Current burst beat count (0-indexed)
    logic [7:0]             beat_cnt;
    logic [7:0]             cur_burst_beats; // Beats for the next burst (from bytes_remaining)
    logic [7:0]             burst_beats_reg; // Beats in the burst in flight (latched at start;
                                             // cur_burst_beats changes as bytes_remaining drops)


    // Scratchpad read buffer for STORE direction. The scratchpad has a
    // one-cycle synchronous read: data requested with sp_ren appears on
    // sp_rdata on the following cycle, flagged by rd_pending.
    logic [AXI_DATA_W-1:0] sp_rdata_buf;
    logic                   sp_rdata_valid;
    logic                   rd_pending;
    logic                   w_beat;          // W handshake this cycle

    // -------------------------------------------------------------------------
    // Burst beat calculation
    // -------------------------------------------------------------------------
    // Clamp transfer to burst_len beats or remaining bytes, whichever is smaller
    always_comb begin
        automatic int beats_needed = (bytes_remaining + BYTES_PER_BEAT - 1) / BYTES_PER_BEAT;
        automatic int max_beats    = {1'b0, burst_len_reg};  // burst_len_reg is 1-256 (value, not len-1)
        cur_burst_beats = (beats_needed < max_beats) ? beats_needed[7:0] : burst_len_reg;
    end

    // -------------------------------------------------------------------------
    // Sequential state
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            sys_addr_reg     <= '0;
            sp_addr_reg      <= '0;
            bytes_remaining  <= '0;
            burst_len_reg    <= '0;
            beat_cnt         <= '0;
            burst_beats_reg  <= '0;
            sp_rdata_buf     <= '0;
            sp_rdata_valid   <= 1'b0;
            rd_pending       <= 1'b0;
        end else begin
            state <= next_state;

            // Scratchpad read pipeline (STORE): capture data one cycle after
            // the read was issued; a W beat consumes the buffered word.
            rd_pending <= sp_ren;
            if (rd_pending) begin
                sp_rdata_buf   <= sp_rdata;
                sp_rdata_valid <= 1'b1;
            end else if (w_beat) begin
                sp_rdata_valid <= 1'b0;
            end


            case (state)
                S_IDLE: begin
                    if (go) begin
                        sys_addr_reg    <= cfg_sys_addr;
                        sp_addr_reg     <= cfg_sp_addr;
                        bytes_remaining <= cfg_byte_len;
                        burst_len_reg   <= (cfg_burst_len == '0) ? 8'd1 : cfg_burst_len;
                    end
                end

                // LOAD: Issue read address
                S_LOAD_AR: begin
                    if (m_axi_arvalid && m_axi_arready) begin
                        // AR handshake done; update address for next burst
                        // (actual update happens when beat_cnt wraps)
                    end
                end

                // LOAD: Receive read data
                S_LOAD_R: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        beat_cnt       <= beat_cnt + 1'b1;
                        sp_addr_reg    <= sp_addr_reg + SP_ADDR_W'(BYTES_PER_BEAT);
                        bytes_remaining <= bytes_remaining - 24'(BYTES_PER_BEAT);
                        sys_addr_reg   <= sys_addr_reg + AXI_ADDR_W'(BYTES_PER_BEAT);

                        if (m_axi_rlast) begin
                            beat_cnt <= '0;
                        end
                    end
                end

                // STORE: start of a burst -- the first scratchpad read is
                // issued this cycle (see sp_ren); latch the burst length
                S_STORE_AR_READ: begin
                    burst_beats_reg <= cur_burst_beats;
                    beat_cnt        <= '0;
                end

                S_STORE_AW: ;  // first word arrives in the buffer meanwhile

                S_STORE_W: begin
                    if (w_beat) begin
                        beat_cnt        <= beat_cnt + 1'b1;
                        sp_addr_reg     <= sp_addr_reg + SP_ADDR_W'(BYTES_PER_BEAT);
                        bytes_remaining <= bytes_remaining - 24'(BYTES_PER_BEAT);
                        sys_addr_reg    <= sys_addr_reg + AXI_ADDR_W'(BYTES_PER_BEAT);
                        if (m_axi_wlast)
                            beat_cnt <= '0;
                    end
                end

                default: ;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Next-state logic
    // -------------------------------------------------------------------------
    always_comb begin
        next_state = state;

        unique case (state)
            S_IDLE: begin
                if (go)
                    next_state = cfg_direction ? S_STORE_AR_READ : S_LOAD_AR;
            end

            // --- LOAD path ---
            S_LOAD_AR: begin
                if (m_axi_arvalid && m_axi_arready)
                    next_state = S_LOAD_R;
            end

            S_LOAD_R: begin
                if (m_axi_rvalid && m_axi_rready && m_axi_rlast) begin
                    if (bytes_remaining <= 24'(BYTES_PER_BEAT))  // Last beat of last burst
                        next_state = S_DONE;
                    else
                        next_state = S_LOAD_AR;  // Issue next burst
                end
            end

            // --- STORE path ---
            S_STORE_AR_READ: begin
                // One cycle to initiate scratchpad read, then move to AW
                next_state = S_STORE_AW;
            end

            S_STORE_AW: begin
                if (m_axi_awvalid && m_axi_awready)
                    next_state = S_STORE_W;
            end

            S_STORE_W: begin
                if (m_axi_wvalid && m_axi_wready && m_axi_wlast)
                    next_state = S_STORE_B;
            end

            S_STORE_B: begin
                if (m_axi_bvalid && m_axi_bready) begin
                    if (bytes_remaining == '0)
                        next_state = S_DONE;
                    else
                        next_state = S_STORE_AR_READ;  // Next burst
                end
            end

            S_DONE: next_state = S_IDLE;

            default: next_state = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // AXI output signals -- LOAD path (AR and R channels)
    // -------------------------------------------------------------------------

    // AR channel
    assign m_axi_araddr  = sys_addr_reg;
    assign m_axi_arlen   = cur_burst_beats - 8'd1;  // AXI len = beats - 1
    assign m_axi_arsize  = 3'b011;                  // 8 bytes per beat
    assign m_axi_arburst = 2'b01;                   // INCR
    assign m_axi_arvalid = (state == S_LOAD_AR);

    // R channel: always ready (backpressure not modelled for simplicity)
    assign m_axi_rready  = (state == S_LOAD_R);

    // -------------------------------------------------------------------------
    // AXI output signals -- STORE path (AW, W, B channels)
    // -------------------------------------------------------------------------

    // AW channel
    assign m_axi_awaddr  = sys_addr_reg;
    assign m_axi_awlen   = cur_burst_beats - 8'd1;
    assign m_axi_awsize  = 3'b011;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = (state == S_STORE_AW);

    // W channel. WLAST uses the burst length latched at burst start: the
    // live cur_burst_beats shrinks as bytes_remaining counts down, so
    // comparing against it would never match on the final burst.
    // Each beat needs a fresh scratchpad read, so WVALID drops for one cycle
    // between beats (half throughput). A deeper prefetch FIFO removes the
    // bubble -- a good follow-up interview question.
    assign m_axi_wdata   = sp_rdata_buf;
    assign m_axi_wlast   = (beat_cnt == burst_beats_reg - 8'd1) && (state == S_STORE_W);
    assign m_axi_wvalid  = (state == S_STORE_W) && sp_rdata_valid;
    assign w_beat        = m_axi_wvalid && m_axi_wready;

    // B channel: always ready to accept response
    assign m_axi_bready  = (state == S_STORE_B);

    // -------------------------------------------------------------------------
    // Scratchpad interface
    // -------------------------------------------------------------------------

    // Write to scratchpad on each received AXI read beat (LOAD direction)
    // On a W beat the next word is requested, so point at the next address
    assign sp_addr  = w_beat ? sp_addr_reg + SP_ADDR_W'(BYTES_PER_BEAT) : sp_addr_reg;
    assign sp_wdata = m_axi_rdata;
    assign sp_wen   = (state == S_LOAD_R) && m_axi_rvalid && m_axi_rready;

    // Read from scratchpad to feed AXI write bursts (STORE direction):
    // the first word of each burst at burst start, then the next word on
    // every W beat except the last
    assign sp_ren   = (state == S_STORE_AR_READ) || (w_beat && !m_axi_wlast);

    // -------------------------------------------------------------------------
    // Status
    // -------------------------------------------------------------------------
    assign busy = (state != S_IDLE && state != S_DONE);
    assign done = (state == S_DONE);

    // -------------------------------------------------------------------------
    // Assertions (simulation only)
    // -------------------------------------------------------------------------
    // synthesis translate_off

    // ARLEN must be < MAX_BURST
    assert property (@(posedge clk) disable iff (!rst_n)
        m_axi_arvalid |-> (m_axi_arlen < MAX_BURST))
        else $error("ARLEN %0d exceeds MAX_BURST %0d", m_axi_arlen, MAX_BURST);

    // AWLEN must be < MAX_BURST
    assert property (@(posedge clk) disable iff (!rst_n)
        m_axi_awvalid |-> (m_axi_awlen < MAX_BURST))
        else $error("AWLEN %0d exceeds MAX_BURST %0d", m_axi_awlen, MAX_BURST);

    // Once ARVALID is asserted, it must stay high until ARREADY
    property p_arvalid_stable;
        @(posedge clk) disable iff (!rst_n)
        (m_axi_arvalid && !m_axi_arready) |=> m_axi_arvalid;
    endproperty
    assert property (p_arvalid_stable)
        else $error("ARVALID dropped before ARREADY");

    // Once AWVALID is asserted, it must stay high until AWREADY
    property p_awvalid_stable;
        @(posedge clk) disable iff (!rst_n)
        (m_axi_awvalid && !m_axi_awready) |=> m_axi_awvalid;
    endproperty
    assert property (p_awvalid_stable)
        else $error("AWVALID dropped before AWREADY");

    // WLAST must be asserted on the last beat only
    assert property (@(posedge clk) disable iff (!rst_n)
        (m_axi_wvalid && m_axi_wlast) |->
            (beat_cnt == burst_beats_reg - 8'd1))
        else $error("WLAST asserted on non-last beat");

    // synthesis translate_on

endmodule


// =============================================================================
// TESTBENCH
// =============================================================================

// synthesis translate_off
module tb_axi_dma_engine;

    localparam int AXI_ADDR_W  = 32;
    localparam int AXI_DATA_W  = 64;
    localparam int SP_ADDR_W   = 20;
    localparam int BYTES_BEAT  = AXI_DATA_W / 8;  // 8

    logic clk, rst_n;

    // Config
    logic [AXI_ADDR_W-1:0] cfg_sys_addr;
    logic [SP_ADDR_W-1:0]  cfg_sp_addr;
    logic [23:0]            cfg_byte_len;
    logic [7:0]             cfg_burst_len;
    logic                   cfg_direction;
    logic                   go, done, busy;

    // AXI AR
    logic [AXI_ADDR_W-1:0] m_axi_araddr;
    logic [7:0]             m_axi_arlen;
    logic [2:0]             m_axi_arsize;
    logic [1:0]             m_axi_arburst;
    logic                   m_axi_arvalid;
    logic                   m_axi_arready;

    // AXI R
    logic [AXI_DATA_W-1:0] m_axi_rdata;
    logic                   m_axi_rlast;
    logic                   m_axi_rvalid;
    logic                   m_axi_rready;

    // AXI AW
    logic [AXI_ADDR_W-1:0] m_axi_awaddr;
    logic [7:0]             m_axi_awlen;
    logic [2:0]             m_axi_awsize;
    logic [1:0]             m_axi_awburst;
    logic                   m_axi_awvalid;
    logic                   m_axi_awready;

    // AXI W
    logic [AXI_DATA_W-1:0] m_axi_wdata;
    logic                   m_axi_wlast;
    logic                   m_axi_wvalid;
    logic                   m_axi_wready;

    // AXI B
    logic [1:0]             m_axi_bresp;
    logic                   m_axi_bvalid;
    logic                   m_axi_bready;

    // Scratchpad
    logic [SP_ADDR_W-1:0]  sp_addr;
    logic [AXI_DATA_W-1:0] sp_wdata;
    logic                   sp_wen;
    logic [AXI_DATA_W-1:0] sp_rdata;
    logic                   sp_ren;

    // DUT
    axi_dma_engine #(
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W),
        .SP_ADDR_W(SP_ADDR_W)
    ) dut (.*);

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Simple memory model (simulates system memory for AXI slave)
    // -------------------------------------------------------------------------
    logic [7:0] sys_mem [0:16383];  // 16 KB of fake system memory

    initial begin
        for (int i = 0; i < 16384; i++)
            sys_mem[i] = i[7:0];   // Fill with known pattern
    end

    // -------------------------------------------------------------------------
    // Simple scratchpad model
    // -------------------------------------------------------------------------
    logic [AXI_DATA_W-1:0] scratchpad [0:4095];  // 4K * 8 bytes = 32 KB

    always_ff @(posedge clk) begin
        if (sp_wen)
            scratchpad[sp_addr[SP_ADDR_W-1:3]] <= sp_wdata;  // Byte to word index
        if (sp_ren)
            sp_rdata <= scratchpad[sp_addr[SP_ADDR_W-1:3]];
    end

    // -------------------------------------------------------------------------
    // AXI slave model (read path)
    // -------------------------------------------------------------------------
    // Accepts AR, returns R data with 2-cycle latency per burst start
    logic [AXI_ADDR_W-1:0] rd_addr_lat;
    logic [7:0]             rd_beats_rem;
    logic [1:0]             rd_latency;

    typedef enum logic [1:0] {RS_IDLE, RS_LATENCY, RS_DATA} rd_state_t;
    rd_state_t rd_state;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b0;
            m_axi_rvalid  <= 1'b0;
            m_axi_rlast   <= 1'b0;
            m_axi_rdata   <= '0;
            rd_state      <= RS_IDLE;
        end else begin
            case (rd_state)
                RS_IDLE: begin
                    m_axi_arready <= 1'b1;  // Always ready to accept address
                    if (m_axi_arvalid && m_axi_arready) begin
                        rd_addr_lat   <= m_axi_araddr;
                        rd_beats_rem  <= m_axi_arlen;  // arlen = beats - 1
                        rd_latency    <= 2'd2;
                        m_axi_arready <= 1'b0;
                        rd_state      <= RS_LATENCY;
                    end
                end

                RS_LATENCY: begin
                    rd_latency <= rd_latency - 2'd1;
                    if (rd_latency == 2'd1) begin
                        rd_state     <= RS_DATA;
                        m_axi_rvalid <= 1'b1;
                        // Return the byte data packed into 64-bit word
                        m_axi_rdata  <= {sys_mem[rd_addr_lat+7], sys_mem[rd_addr_lat+6],
                                         sys_mem[rd_addr_lat+5], sys_mem[rd_addr_lat+4],
                                         sys_mem[rd_addr_lat+3], sys_mem[rd_addr_lat+2],
                                         sys_mem[rd_addr_lat+1], sys_mem[rd_addr_lat+0]};
                        m_axi_rlast  <= (rd_beats_rem == '0);
                    end
                end

                RS_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        if (m_axi_rlast) begin
                            m_axi_rvalid  <= 1'b0;
                            m_axi_rlast   <= 1'b0;
                            rd_state      <= RS_IDLE;
                            m_axi_arready <= 1'b1;
                        end else begin
                            rd_beats_rem <= rd_beats_rem - 8'd1;
                            rd_addr_lat  <= rd_addr_lat + AXI_ADDR_W'(BYTES_BEAT);
                            m_axi_rdata  <= {sys_mem[rd_addr_lat+BYTES_BEAT+7],
                                             sys_mem[rd_addr_lat+BYTES_BEAT+6],
                                             sys_mem[rd_addr_lat+BYTES_BEAT+5],
                                             sys_mem[rd_addr_lat+BYTES_BEAT+4],
                                             sys_mem[rd_addr_lat+BYTES_BEAT+3],
                                             sys_mem[rd_addr_lat+BYTES_BEAT+2],
                                             sys_mem[rd_addr_lat+BYTES_BEAT+1],
                                             sys_mem[rd_addr_lat+BYTES_BEAT+0]};
                            m_axi_rlast  <= (rd_beats_rem == 8'd1);
                        end
                    end
                end
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // AXI slave model (write path)
    // -------------------------------------------------------------------------
    logic [AXI_ADDR_W-1:0] wr_addr_lat;
    logic [7:0]             wr_beats_rem;

    typedef enum logic [1:0] {WS_IDLE, WS_DATA, WS_RESP} wr_state_t;
    wr_state_t wr_state;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready  <= 1'b0;
            m_axi_bvalid  <= 1'b0;
            m_axi_bresp   <= 2'b00;
            wr_state      <= WS_IDLE;
        end else begin
            case (wr_state)
                WS_IDLE: begin
                    m_axi_awready <= 1'b1;
                    if (m_axi_awvalid && m_axi_awready) begin
                        wr_addr_lat   <= m_axi_awaddr;
                        wr_beats_rem  <= m_axi_awlen;
                        m_axi_awready <= 1'b0;
                        m_axi_wready  <= 1'b1;
                        wr_state      <= WS_DATA;
                    end
                end

                WS_DATA: begin
                    if (m_axi_wvalid && m_axi_wready) begin
                        // Write to sys_mem model
                        {sys_mem[wr_addr_lat+7], sys_mem[wr_addr_lat+6],
                         sys_mem[wr_addr_lat+5], sys_mem[wr_addr_lat+4],
                         sys_mem[wr_addr_lat+3], sys_mem[wr_addr_lat+2],
                         sys_mem[wr_addr_lat+1], sys_mem[wr_addr_lat+0]} <= m_axi_wdata;
                        wr_addr_lat <= wr_addr_lat + AXI_ADDR_W'(BYTES_BEAT);

                        if (m_axi_wlast) begin
                            m_axi_wready <= 1'b0;
                            m_axi_bvalid <= 1'b1;
                            m_axi_bresp  <= 2'b00;  // OKAY
                            wr_state     <= WS_RESP;
                        end else begin
                            wr_beats_rem <= wr_beats_rem - 8'd1;
                        end
                    end
                end

                WS_RESP: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bvalid  <= 1'b0;
                        wr_state      <= WS_IDLE;
                        m_axi_awready <= 1'b1;
                    end
                end
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Test tasks
    // -------------------------------------------------------------------------
    int total_errors = 0;

    task automatic do_load(
        input logic [AXI_ADDR_W-1:0] sys_a,
        input logic [SP_ADDR_W-1:0]  sp_a,
        input int                     nbytes,
        input int                     bl,
        input string                  name
    );
        int errors = 0;

        $display("=== LOAD test: %s (sys=0x%08x, sp=0x%05x, len=%0d, burst=%0d) ===",
                 name, sys_a, sp_a, nbytes, bl);
        @(negedge clk);        // drive away from the sampling edge
        cfg_sys_addr   = sys_a;
        cfg_sp_addr    = sp_a;
        cfg_byte_len   = nbytes;
        cfg_burst_len  = bl;
        cfg_direction  = 0;  // LOAD
        go = 1;
        @(negedge clk);
        go = 0;

        @(posedge done);
        @(posedge clk);
        assert (!busy) else $error("Still busy after done");

        // Verify scratchpad contents match sys_mem
        for (int b = 0; b < nbytes; b += BYTES_BEAT) begin
            int word_idx = (sp_a + b) / BYTES_BEAT;
            for (int byte_i = 0; byte_i < BYTES_BEAT && (b + byte_i) < nbytes; byte_i++) begin
                logic [7:0] expected_byte = sys_mem[sys_a + b + byte_i];
                logic [7:0] actual_byte   = scratchpad[word_idx][(byte_i*8) +: 8];
                if (actual_byte !== expected_byte) begin
                    $error("LOAD data mismatch at byte %0d: expected 0x%02x got 0x%02x",
                           b + byte_i, expected_byte, actual_byte);
                    errors++;
                end
            end
        end
        if (errors == 0) $display("  LOAD verification PASSED");
        else             $display("  LOAD verification FAILED (%0d bytes wrong)", errors);
        total_errors += errors;
    endtask

    task automatic do_store(
        input logic [SP_ADDR_W-1:0]  sp_a,
        input logic [AXI_ADDR_W-1:0] sys_a,
        input int                     nbytes,
        input int                     bl,
        input string                  name
    );
        int errors = 0;

        $display("=== STORE test: %s (sp=0x%05x, sys=0x%08x, len=%0d, burst=%0d) ===",
                 name, sp_a, sys_a, nbytes, bl);

        // Pre-fill scratchpad with a known pattern
        for (int i = 0; i < (nbytes + BYTES_BEAT - 1) / BYTES_BEAT; i++)
            scratchpad[(sp_a / BYTES_BEAT) + i] = 64'hDEAD_0000_0000_0000 | i;

        @(negedge clk);        // drive away from the sampling edge
        cfg_sys_addr   = sys_a;
        cfg_sp_addr    = sp_a;
        cfg_byte_len   = nbytes;
        cfg_burst_len  = bl;
        cfg_direction  = 1;  // STORE
        go = 1;
        @(negedge clk);
        go = 0;

        @(posedge done);
        @(posedge clk);
        assert (!busy) else $error("Still busy after done");

        // Verify sys_mem was updated
        for (int b = 0; b < nbytes; b += BYTES_BEAT) begin
            int word_idx = (sp_a + b) / BYTES_BEAT;
            logic [AXI_DATA_W-1:0] expected_word = 64'hDEAD_0000_0000_0000 | (b / BYTES_BEAT);
            logic [AXI_DATA_W-1:0] actual_word;
            for (int byte_i = 0; byte_i < BYTES_BEAT; byte_i++)
                actual_word[(byte_i*8) +: 8] = sys_mem[sys_a + b + byte_i];
            if (actual_word !== expected_word) begin
                $error("STORE data mismatch at word %0d: expected 0x%016x got 0x%016x",
                       b/BYTES_BEAT, expected_word, actual_word);
                errors++;
            end
        end
        if (errors == 0) $display("  STORE verification PASSED");
        else             $display("  STORE verification FAILED (%0d words wrong)", errors);
        total_errors += errors;
    endtask

    // -------------------------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------------------------
    initial begin
        rst_n = 0;
        go    = 0;
        cfg_sys_addr  = '0;
        cfg_sp_addr   = '0;
        cfg_byte_len  = '0;
        cfg_burst_len = 8'd4;
        cfg_direction = 1'b0;

        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        // Test 1: Single-beat LOAD (burst_len=1, 8 bytes)
        do_load(32'h0000_0000, 20'h0_0000, 8, 8'd1, "Single beat");

        // Test 2: Multi-beat LOAD (burst_len=4, 32 bytes = 4 beats)
        do_load(32'h0000_0010, 20'h0_0100, 32, 8'd4, "4-beat burst");

        // Test 3: Multi-burst LOAD (64 bytes with burst_len=4 = 2 bursts of 4)
        do_load(32'h0000_0040, 20'h0_0200, 64, 8'd4, "2 bursts of 4");

        // Test 4: Large burst (128 bytes, burst_len=16)
        do_load(32'h0000_0100, 20'h0_0400, 128, 8'd16, "128 bytes burst=16");

        // Test 5: STORE - write scratchpad data back to system memory
        do_store(20'h0_0100, 32'h0000_1000, 64, 8'd4, "Store 64 bytes");

        // Test 6: STORE with large burst
        do_store(20'h0_0200, 32'h0000_2000, 128, 8'd8, "Store 128 bytes burst=8");

        if (total_errors == 0) $display("ALL DMA TESTS PASSED");
        else                   $display("DMA TESTS FAILED: %0d mismatches", total_errors);
        $finish;
    end

    // Global timeout
    initial begin
        #500000;
        $fatal(1, "Global simulation timeout");
    end

endmodule
// synthesis translate_on
