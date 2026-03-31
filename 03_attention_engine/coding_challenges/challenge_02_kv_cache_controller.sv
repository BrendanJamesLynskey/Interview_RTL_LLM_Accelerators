// =============================================================================
// Challenge 02: KV Cache Memory Controller
// =============================================================================
//
// PROBLEM STATEMENT
// -----------------
// Implement a KV cache memory controller for an LLM inference accelerator.
// The controller manages a shared SRAM that stores Key and Value vectors for
// all past tokens across all transformer layers.
//
// The controller must support three operations:
//   1. PREFILL WRITE  : Bulk write of K/V vectors for an entire input prompt.
//   2. DECODE APPEND  : Append one new token's K/V vectors per decode step.
//   3. ATTENTION READ : Sequential read of all K (or V) vectors for one layer,
//                       for use by the attention score computation engine.
//
// MEMORY LAYOUT
// -------------
// The SRAM is organised layer-first, then position:
//
//   Layer 0: [K_pos0_h0..hH-1] [V_pos0_h0..hH-1] [K_pos1_h0..hH-1] [V_pos1...] ...
//   Layer 1: ...
//   ...
//
// Address formula for K/V at (layer, position, is_value):
//   addr = layer  * LAYER_STRIDE
//        + pos    * TOKEN_STRIDE
//        + is_val * HEAD_STRIDE     (0 for K, HEAD_STRIDE for V)
//
// Where:
//   TOKEN_STRIDE = 2 * NUM_HEADS * HEAD_DIM  (K + V for one token, all heads)
//   HEAD_STRIDE  = NUM_HEADS * HEAD_DIM       (all heads for K or V)
//   LAYER_STRIDE = MAX_SEQ_LEN * TOKEN_STRIDE
//
// INTERFACE: AXI4-LITE-LIKE COMMAND INTERFACE
// -------------------------------------------
// The controller accepts commands via a command FIFO:
//   cmd_type:  2'b00 = PREFILL, 2'b01 = DECODE_APPEND, 2'b10 = ATTN_READ
//   cmd_layer: which transformer layer
//   cmd_pos:   start position (PREFILL/DECODE) or read start pos (ATTN_READ)
//   cmd_len:   number of tokens (PREFILL: > 1; DECODE: always 1; ATTN_READ: cur_seq_len)
//   cmd_kv:    1=read K only, 0=read V only, X for writes (writes always write both K+V)
//
// Write data is presented on wr_data (DATA_WIDTH bits wide) with wr_valid/wr_ready handshake.
// Read data is presented on rd_data (DATA_WIDTH bits wide) with rd_valid output.
//
// PARAMETERS
// ----------
// NUM_LAYERS   : Number of transformer layers (e.g., 32)
// MAX_SEQ_LEN  : Maximum sequence length (e.g., 2048)
// NUM_HEADS    : Number of KV heads (GQA/MQA: number of unique K/V heads, e.g., 8)
// HEAD_DIM     : Head dimension (e.g., 128)
// DATA_WIDTH   : SRAM data bus width in bits (e.g., 128 = 8 BF16 values)
//
// CONSTRAINTS
// -----------
// - HEAD_DIM must be a multiple of (DATA_WIDTH / 16) so that one head's K or V
//   vector transfers in an integer number of DATA_WIDTH-wide beats.
// - The controller maintains a cur_seq_len register that is incremented by
//   DECODE_APPEND and set by PREFILL. Reset on rst_n.
// - ATTN_READ has lower priority than DECODE_APPEND (write beats read).
// =============================================================================

`default_nettype none
`timescale 1ns / 1ps

// =============================================================================
// Parameterised synchronous dual-port SRAM model
// Write port: synchronous write on rising edge when wr_en is asserted.
// Read port:  synchronous read on rising edge; data available next cycle.
// =============================================================================
module dp_sram #(
    parameter int DEPTH      = 65536,   // Number of DATA_WIDTH-wide words
    parameter int DATA_WIDTH = 128      // Bits per word
) (
    input  wire                    clk,
    // Write port
    input  wire                    wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [DATA_WIDTH-1:0]   wr_data,
    // Read port
    input  wire                    rd_en,
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output reg  [DATA_WIDTH-1:0]   rd_data
);
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    always_ff @(posedge clk) begin
        if (rd_en)
            rd_data <= mem[rd_addr];
    end
endmodule


// =============================================================================
// Command FIFO: Stores pending controller commands.
// =============================================================================
module cmd_fifo #(
    parameter int DEPTH    = 8,
    parameter int CMD_BITS = 32  // Total bits for one command struct
) (
    input  wire                clk,
    input  wire                rst_n,
    // Push side
    input  wire                push,
    input  wire [CMD_BITS-1:0] din,
    output wire                full,
    // Pop side
    input  wire                pop,
    output wire [CMD_BITS-1:0] dout,
    output wire                empty
);
    localparam int PTR_BITS = $clog2(DEPTH);

    reg [CMD_BITS-1:0] fifo_mem [0:DEPTH-1];
    reg [PTR_BITS:0]   wr_ptr, rd_ptr; // Extra bit for full/empty detection

    assign full  = (wr_ptr[PTR_BITS] != rd_ptr[PTR_BITS]) &&
                   (wr_ptr[PTR_BITS-1:0] == rd_ptr[PTR_BITS-1:0]);
    assign empty = (wr_ptr == rd_ptr);
    assign dout  = fifo_mem[rd_ptr[PTR_BITS-1:0]];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
        end else begin
            if (push && !full) begin
                fifo_mem[wr_ptr[PTR_BITS-1:0]] <= din;
                wr_ptr <= wr_ptr + 1'b1;
            end
            if (pop && !empty)
                rd_ptr <= rd_ptr + 1'b1;
        end
    end
endmodule


// =============================================================================
// KV Cache Controller
// =============================================================================
module kv_cache_controller #(
    parameter int NUM_LAYERS   = 32,
    parameter int MAX_SEQ_LEN  = 2048,
    parameter int NUM_HEADS    = 8,     // KV heads (GQA: fewer than query heads)
    parameter int HEAD_DIM     = 128,   // Elements per head
    parameter int DATA_WIDTH   = 128    // SRAM bus width in bits (8 x BF16)
) (
    input  wire        clk,
    input  wire        rst_n,

    // -----------------------------------------------------------------------
    // Command interface
    // -----------------------------------------------------------------------
    input  wire        cmd_valid,           // A new command is presented
    output wire        cmd_ready,           // Controller accepts the command
    input  wire [1:0]  cmd_type,            // 2'b00=PREFILL, 01=DECODE_APPEND, 10=ATTN_READ
    input  wire [$clog2(NUM_LAYERS)-1:0]  cmd_layer, // Layer index
    input  wire [$clog2(MAX_SEQ_LEN)-1:0] cmd_pos,   // Start position
    input  wire [$clog2(MAX_SEQ_LEN)-1:0] cmd_len,   // Number of tokens (PREFILL/READ)
    input  wire        cmd_kv,              // 1=K only, 0=V only (ATTN_READ only)

    // -----------------------------------------------------------------------
    // Write data port (PREFILL and DECODE_APPEND)
    // Data is presented HEAD_DIM/8 beats per token per head (8 BF16 per beat)
    // Order: K[h0][0..d-1], K[h1][0..d-1], ..., V[h0][0..d-1], V[h1][0..d-1], ...
    // for each token position sequentially.
    // -----------------------------------------------------------------------
    input  wire                  wr_valid,
    output wire                  wr_ready,
    input  wire [DATA_WIDTH-1:0] wr_data,

    // -----------------------------------------------------------------------
    // Read data port (ATTN_READ)
    // Streams out K or V vectors for all positions: pos=0..cmd_len-1.
    // Data order: all heads for pos 0, all heads for pos 1, ...
    // DATA_WIDTH bits per beat; HEAD_DIM/8 beats per head per position.
    // -----------------------------------------------------------------------
    output wire                  rd_valid,
    output wire [DATA_WIDTH-1:0] rd_data,
    output wire                  rd_last,    // Last beat of the entire ATTN_READ

    // -----------------------------------------------------------------------
    // Status
    // -----------------------------------------------------------------------
    output wire [$clog2(MAX_SEQ_LEN):0] cur_seq_len,  // Current number of valid tokens
    output wire                          cache_full    // At maximum sequence length
);

    // -----------------------------------------------------------------------
    // Derived constants
    // -----------------------------------------------------------------------
    // Elements (BF16 = 16-bit) per beat
    localparam int ELEMS_PER_BEAT  = DATA_WIDTH / 16;
    // Beats to transfer one head's K or V vector
    localparam int BEATS_PER_HEAD  = HEAD_DIM / ELEMS_PER_BEAT;
    // Beats for all heads, K only (or V only)
    localparam int BEATS_PER_KV    = NUM_HEADS * BEATS_PER_HEAD;
    // Beats for all heads, both K and V, one token
    localparam int BEATS_PER_TOKEN = 2 * BEATS_PER_KV;

    // SRAM word (DATA_WIDTH bits) is the atomic unit; compute total SRAM depth.
    // Layout: each layer holds MAX_SEQ_LEN * BEATS_PER_TOKEN words.
    localparam int LAYER_WORDS     = MAX_SEQ_LEN * BEATS_PER_TOKEN;
    localparam int TOTAL_WORDS     = NUM_LAYERS * LAYER_WORDS;
    localparam int ADDR_BITS       = $clog2(TOTAL_WORDS);

    // -----------------------------------------------------------------------
    // Address computation helper function
    // Returns the SRAM word address for (layer, pos, kv_offset_beats)
    // kv_offset_beats: 0 for start of K, BEATS_PER_KV for start of V
    // -----------------------------------------------------------------------
    function automatic [ADDR_BITS-1:0] sram_addr;
        input [$clog2(NUM_LAYERS)-1:0]  layer;
        input [$clog2(MAX_SEQ_LEN)-1:0] pos;
        input [$clog2(BEATS_PER_TOKEN):0] beat_offset; // within the K+V block
        begin
            sram_addr = layer  * LAYER_WORDS
                      + pos    * BEATS_PER_TOKEN
                      + beat_offset;
        end
    endfunction

    // -----------------------------------------------------------------------
    // SRAM instance
    // -----------------------------------------------------------------------
    wire                  sram_wr_en;
    wire [ADDR_BITS-1:0]  sram_wr_addr;
    wire [DATA_WIDTH-1:0] sram_wr_data;
    wire                  sram_rd_en;
    wire [ADDR_BITS-1:0]  sram_rd_addr;
    wire [DATA_WIDTH-1:0] sram_rd_data;

    dp_sram #(
        .DEPTH     (TOTAL_WORDS),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_sram (
        .clk     (clk),
        .wr_en   (sram_wr_en),
        .wr_addr (sram_wr_addr),
        .wr_data (sram_wr_data),
        .rd_en   (sram_rd_en),
        .rd_addr (sram_rd_addr),
        .rd_data (sram_rd_data)
    );

    // -----------------------------------------------------------------------
    // Sequence length tracking
    // -----------------------------------------------------------------------
    reg [$clog2(MAX_SEQ_LEN):0] seq_len_reg;
    assign cur_seq_len = seq_len_reg;
    assign cache_full  = (seq_len_reg == MAX_SEQ_LEN);

    // -----------------------------------------------------------------------
    // Command FIFO
    // Pack command fields into CMD_BITS-wide word
    // Format: [1:0] type | [6:2] layer | [17:7] pos | [28:18] len | [29] kv
    // -----------------------------------------------------------------------
    localparam int CMD_BITS = 30;
    wire [CMD_BITS-1:0] cmd_packed;
    wire [CMD_BITS-1:0] cmd_fifo_dout;
    wire                cmd_fifo_empty, cmd_fifo_full;
    wire                cmd_fifo_pop;

    assign cmd_packed = {cmd_kv, cmd_len, cmd_pos, cmd_layer, cmd_type};
    assign cmd_ready  = !cmd_fifo_full;

    // Unpack current command from FIFO head
    wire [1:0]  cur_cmd_type  = cmd_fifo_dout[1:0];
    wire [$clog2(NUM_LAYERS)-1:0]  cur_cmd_layer =
        cmd_fifo_dout[1 + $clog2(NUM_LAYERS) : 2];
    wire [$clog2(MAX_SEQ_LEN)-1:0] cur_cmd_pos   =
        cmd_fifo_dout[1 + $clog2(NUM_LAYERS) + $clog2(MAX_SEQ_LEN) : 2 + $clog2(NUM_LAYERS)];
    wire [$clog2(MAX_SEQ_LEN)-1:0] cur_cmd_len   =
        cmd_fifo_dout[1 + $clog2(NUM_LAYERS) + 2*$clog2(MAX_SEQ_LEN) : 2 + $clog2(NUM_LAYERS) + $clog2(MAX_SEQ_LEN)];
    wire                           cur_cmd_kv    = cmd_fifo_dout[CMD_BITS-1];

    cmd_fifo #(
        .DEPTH   (8),
        .CMD_BITS(CMD_BITS)
    ) u_cmd_fifo (
        .clk   (clk),
        .rst_n (rst_n),
        .push  (cmd_valid && cmd_ready),
        .din   (cmd_packed),
        .full  (cmd_fifo_full),
        .pop   (cmd_fifo_pop),
        .dout  (cmd_fifo_dout),
        .empty (cmd_fifo_empty)
    );

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE          = 3'd0,  // No active operation
        S_PREFILL_WR    = 3'd1,  // Writing bulk prefill data
        S_DECODE_WR     = 3'd2,  // Writing one token's K+V
        S_ATTN_RD_REQ   = 3'd3,  // Issuing read address to SRAM
        S_ATTN_RD_DATA  = 3'd4   // Streaming read data out
    } state_t;

    state_t state;

    // -----------------------------------------------------------------------
    // Operation counters
    // -----------------------------------------------------------------------
    // Tracks current position and beat within an active write or read operation
    reg [$clog2(MAX_SEQ_LEN)-1:0]   op_pos;       // Current token position
    reg [$clog2(BEATS_PER_TOKEN):0]  op_beat;      // Beat within current token
    reg [$clog2(MAX_SEQ_LEN)-1:0]   op_remaining; // Tokens remaining
    reg [$clog2(BEATS_PER_TOKEN):0]  kv_offset;   // 0 for K, BEATS_PER_KV for V

    // -----------------------------------------------------------------------
    // Write control signals
    // -----------------------------------------------------------------------
    reg                  wr_en_reg;
    reg [ADDR_BITS-1:0]  wr_addr_reg;

    assign sram_wr_en   = wr_en_reg && wr_valid;
    assign sram_wr_addr = wr_addr_reg;
    assign sram_wr_data = wr_data;
    assign wr_ready     = (state == S_PREFILL_WR || state == S_DECODE_WR) && wr_en_reg;

    // -----------------------------------------------------------------------
    // Read control signals
    // -----------------------------------------------------------------------
    reg                  rd_req_reg;    // Assert SRAM rd_en
    reg [ADDR_BITS-1:0]  rd_addr_reg;  // SRAM read address
    reg                  rd_valid_reg; // Output valid (delayed by 1 cycle for SRAM latency)
    reg                  rd_last_reg;

    assign sram_rd_en   = rd_req_reg;
    assign sram_rd_addr = rd_addr_reg;
    assign rd_valid     = rd_valid_reg;
    assign rd_data      = sram_rd_data;
    assign rd_last      = rd_last_reg;

    // -----------------------------------------------------------------------
    // Command completion: pop the FIFO when the operation finishes
    // -----------------------------------------------------------------------
    reg op_done;
    assign cmd_fifo_pop = op_done;

    // -----------------------------------------------------------------------
    // Main FSM
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            seq_len_reg  <= '0;
            wr_en_reg    <= 1'b0;
            rd_req_reg   <= 1'b0;
            rd_valid_reg <= 1'b0;
            rd_last_reg  <= 1'b0;
            op_done      <= 1'b0;
            op_pos       <= '0;
            op_beat      <= '0;
            op_remaining <= '0;
            kv_offset    <= '0;
        end else begin
            // Default pulse signals
            op_done   <= 1'b0;
            wr_en_reg <= 1'b0;

            // Propagate SRAM read latency (1 cycle) to rd_valid
            rd_valid_reg <= rd_req_reg;
            rd_last_reg  <= 1'b0;

            case (state)
                // -----------------------------------------------------------
                S_IDLE : begin
                    rd_req_reg <= 1'b0;
                    if (!cmd_fifo_empty) begin
                        // Decode and start the next command
                        op_pos       <= cur_cmd_pos;
                        op_beat      <= '0;
                        op_remaining <= cur_cmd_len;
                        kv_offset    <= cur_cmd_kv ?
                                        '0 :              // K: offset=0
                                        BEATS_PER_KV[($clog2(BEATS_PER_TOKEN)):0]; // V: offset=BEATS_PER_KV

                        case (cur_cmd_type)
                            2'b00 : state <= S_PREFILL_WR;   // PREFILL
                            2'b01 : state <= S_DECODE_WR;    // DECODE_APPEND
                            2'b10 : state <= S_ATTN_RD_REQ;  // ATTN_READ
                            default: state <= S_IDLE;
                        endcase
                    end
                end

                // -----------------------------------------------------------
                // PREFILL WRITE
                // Write BEATS_PER_TOKEN beats per token for op_remaining tokens.
                // Accept wr_data each cycle wr_valid is asserted.
                // -----------------------------------------------------------
                S_PREFILL_WR : begin
                    if (wr_valid) begin
                        // Compute write address: layer base + pos*TOKEN_STRIDE + beat
                        wr_addr_reg <= sram_addr(cur_cmd_layer, op_pos, op_beat[$clog2(BEATS_PER_TOKEN):0]);
                        wr_en_reg   <= 1'b1;

                        if (op_beat == BEATS_PER_TOKEN - 1) begin
                            // Finished this token's K and V
                            op_beat <= '0;
                            if (op_remaining == 1) begin
                                // Last token of prefill
                                seq_len_reg  <= cur_cmd_pos + cur_cmd_len;
                                op_done      <= 1'b1;
                                state        <= S_IDLE;
                            end else begin
                                op_pos       <= op_pos + 1'b1;
                                op_remaining <= op_remaining - 1'b1;
                            end
                        end else begin
                            op_beat <= op_beat + 1'b1;
                        end
                    end
                end

                // -----------------------------------------------------------
                // DECODE APPEND
                // Append exactly one token: write BEATS_PER_TOKEN beats at
                // position cur_seq_len.
                // -----------------------------------------------------------
                S_DECODE_WR : begin
                    if (wr_valid) begin
                        // Write at the current end of the sequence
                        wr_addr_reg <= sram_addr(cur_cmd_layer,
                                                  seq_len_reg[$clog2(MAX_SEQ_LEN)-1:0],
                                                  op_beat[$clog2(BEATS_PER_TOKEN):0]);
                        wr_en_reg   <= 1'b1;

                        if (op_beat == BEATS_PER_TOKEN - 1) begin
                            // All beats written; increment sequence length
                            seq_len_reg <= seq_len_reg + 1'b1;
                            op_done     <= 1'b1;
                            state       <= S_IDLE;
                        end else begin
                            op_beat <= op_beat + 1'b1;
                        end
                    end
                end

                // -----------------------------------------------------------
                // ATTENTION READ — Request Phase
                // Issue read addresses one beat ahead of data needed.
                // Read K or V (controlled by cmd_kv) for positions 0..cmd_len-1.
                // -----------------------------------------------------------
                S_ATTN_RD_REQ : begin
                    // Issue SRAM read for current address
                    rd_addr_reg <= sram_addr(cur_cmd_layer, op_pos,
                                             kv_offset + op_beat[$clog2(BEATS_PER_TOKEN):0]);
                    rd_req_reg  <= 1'b1;

                    if (op_beat == BEATS_PER_KV - 1) begin
                        // Last beat for this position
                        op_beat <= '0;
                        if (op_remaining == 1) begin
                            // Last position: set last flag (will appear on rd_valid_reg delay)
                            op_done    <= 1'b1;
                            state      <= S_ATTN_RD_DATA; // One more cycle for last data
                        end else begin
                            op_pos       <= op_pos + 1'b1;
                            op_remaining <= op_remaining - 1'b1;
                        end
                    end else begin
                        op_beat <= op_beat + 1'b1;
                    end
                end

                // -----------------------------------------------------------
                // ATTENTION READ — Final data cycle
                // Allow the last SRAM read to propagate through the 1-cycle latency.
                // -----------------------------------------------------------
                S_ATTN_RD_DATA : begin
                    rd_req_reg  <= 1'b0;
                    rd_last_reg <= rd_valid_reg; // Assert rd_last when last data appears
                    if (rd_valid_reg) begin
                        state <= S_IDLE;
                    end
                end

                default : state <= S_IDLE;
            endcase
        end
    end

endmodule


// =============================================================================
// TESTBENCH
// =============================================================================
// Tests the three operations: PREFILL_WRITE, DECODE_APPEND, and ATTN_READ.
//
// Scenario:
//   1. PREFILL: Write K/V for 2 tokens at layer 0.
//   2. DECODE: Append K/V for 1 more token at layer 0.
//   3. ATTN_READ: Read all 3 tokens' K vectors for layer 0.
//   4. Verify that the read data matches the written data.
// =============================================================================
`ifdef SIMULATION
module tb_kv_cache_controller;
    // Parameters matching the DUT
    localparam int NUM_LAYERS  = 4;    // Small for simulation
    localparam int MAX_SEQ_LEN = 16;
    localparam int NUM_HEADS   = 2;
    localparam int HEAD_DIM    = 8;    // Small: 8 elements per head
    localparam int DATA_WIDTH  = 16;   // 1 BF16 per beat (simplifies checking)
    localparam real CLK_PERIOD = 10.0;

    // Derived constants
    localparam int ELEMS_PER_BEAT  = DATA_WIDTH / 16;  // = 1
    localparam int BEATS_PER_HEAD  = HEAD_DIM / ELEMS_PER_BEAT;  // = 8
    localparam int BEATS_PER_KV    = NUM_HEADS * BEATS_PER_HEAD;  // = 16
    localparam int BEATS_PER_TOKEN = 2 * BEATS_PER_KV;             // = 32

    // DUT signals
    logic        clk, rst_n;
    logic        cmd_valid, cmd_ready;
    logic [1:0]  cmd_type;
    logic [$clog2(NUM_LAYERS)-1:0]  cmd_layer;
    logic [$clog2(MAX_SEQ_LEN)-1:0] cmd_pos, cmd_len;
    logic        cmd_kv;
    logic        wr_valid, wr_ready;
    logic [DATA_WIDTH-1:0] wr_data;
    logic        rd_valid, rd_last;
    logic [DATA_WIDTH-1:0] rd_data;
    logic [$clog2(MAX_SEQ_LEN):0] cur_seq_len;
    logic        cache_full;

    // DUT instantiation
    kv_cache_controller #(
        .NUM_LAYERS (NUM_LAYERS),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .NUM_HEADS  (NUM_HEADS),
        .HEAD_DIM   (HEAD_DIM),
        .DATA_WIDTH (DATA_WIDTH)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .cmd_valid  (cmd_valid),
        .cmd_ready  (cmd_ready),
        .cmd_type   (cmd_type),
        .cmd_layer  (cmd_layer),
        .cmd_pos    (cmd_pos),
        .cmd_len    (cmd_len),
        .cmd_kv     (cmd_kv),
        .wr_valid   (wr_valid),
        .wr_ready   (wr_ready),
        .wr_data    (wr_data),
        .rd_valid   (rd_valid),
        .rd_data    (rd_data),
        .rd_last    (rd_last),
        .cur_seq_len(cur_seq_len),
        .cache_full (cache_full)
    );

    // Clock generation
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Task: send one command
    task automatic send_cmd(
        input [1:0]  t,
        input [$clog2(NUM_LAYERS)-1:0]  l,
        input [$clog2(MAX_SEQ_LEN)-1:0] p,
        input [$clog2(MAX_SEQ_LEN)-1:0] n,
        input                            kv
    );
        @(negedge clk);
        cmd_valid = 1'b1;
        cmd_type  = t;
        cmd_layer = l;
        cmd_pos   = p;
        cmd_len   = n;
        cmd_kv    = kv;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        @(negedge clk);
        cmd_valid = 1'b0;
    endtask

    // Task: write N beats of sequential data starting from base_val
    task automatic write_beats(input int n_beats, input int base_val);
        for (int i = 0; i < n_beats; i++) begin
            @(negedge clk);
            wr_valid = 1'b1;
            wr_data  = (base_val + i) & 16'hFFFF;
            @(posedge clk);
            while (!wr_ready) @(posedge clk);
        end
        @(negedge clk);
        wr_valid = 1'b0;
    endtask

    // Collected read data
    int  read_buf_idx;
    logic [DATA_WIDTH-1:0] read_buf [0:511]; // Generous buffer for collected reads

    // Main test
    initial begin
        $display("=== KV Cache Controller Testbench ===");
        $display("NUM_HEADS=%0d, HEAD_DIM=%0d, BEATS_PER_TOKEN=%0d",
                 NUM_HEADS, HEAD_DIM, BEATS_PER_TOKEN);

        // Reset
        rst_n     = 1'b0;
        cmd_valid = 1'b0;
        wr_valid  = 1'b0;
        read_buf_idx = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // ------------------------------------------------------------------
        // TEST 1: PREFILL — write 2 tokens (K+V) at layer 0, starting pos 0
        // Data: beats 0x0001..0x0040 (BEATS_PER_TOKEN*2 = 64 beats)
        // ------------------------------------------------------------------
        $display("[TB] Test 1: PREFILL write of 2 tokens at layer 0");
        send_cmd(.t(2'b00), .l(0), .p(0), .n(2), .kv(1'b0));
        write_beats(.n_beats(BEATS_PER_TOKEN * 2), .base_val(1));
        // Wait for operation to finish (monitor cur_seq_len)
        wait (cur_seq_len == 2);
        $display("[TB]   cur_seq_len = %0d (expected 2)", cur_seq_len);
        assert (cur_seq_len == 2) else $error("FAIL: expected seq_len=2");

        // ------------------------------------------------------------------
        // TEST 2: DECODE APPEND — append 1 token at layer 0
        // Data: beats 0x0041..0x0060 (BEATS_PER_TOKEN = 32 beats)
        // ------------------------------------------------------------------
        $display("[TB] Test 2: DECODE APPEND of 1 token at layer 0");
        send_cmd(.t(2'b01), .l(0), .p(0), .n(1), .kv(1'b0));
        write_beats(.n_beats(BEATS_PER_TOKEN), .base_val(BEATS_PER_TOKEN*2 + 1));
        wait (cur_seq_len == 3);
        $display("[TB]   cur_seq_len = %0d (expected 3)", cur_seq_len);
        assert (cur_seq_len == 3) else $error("FAIL: expected seq_len=3");

        // ------------------------------------------------------------------
        // TEST 3: ATTN_READ — read K vectors for 3 tokens at layer 0
        // Expected: BEATS_PER_KV * 3 = 48 beats of read data
        // The K data for token 0 is beats 0x0001..0x0010 (BEATS_PER_KV=16 beats)
        // The K data for token 1 is beats 0x0021..0x0030
        // The K data for token 2 (appended) is beats 0x0041..0x0050
        // ------------------------------------------------------------------
        $display("[TB] Test 3: ATTN_READ of K vectors, 3 tokens at layer 0");
        send_cmd(.t(2'b10), .l(0), .p(0), .n(3), .kv(1'b1)); // kv=1: read K

        // Wait for rd_last
        fork
            begin : t3_timeout
                repeat (500) @(posedge clk);
                $display("[TB] TIMEOUT in ATTN_READ");
                disable t3_wait;
            end
            begin : t3_wait
                wait (rd_last);
                disable t3_timeout;
            end
        join

        // Report
        $display("[TB]   Received %0d read beats (expected %0d)",
                 read_buf_idx, BEATS_PER_KV * 3);

        // Spot-check: first K beat of token 0 should be 0x0001
        $display("[TB]   First read beat = 0x%04h (expected 0x0001)", read_buf[0]);
        if (read_buf[0] == 16'h0001)
            $display("[TB] PASS: First K beat correct");
        else
            $display("[TB] FAIL: First K beat = 0x%04h, expected 0x0001", read_buf[0]);

        $display("[TB] All tests complete. cur_seq_len=%0d, cache_full=%0b",
                 cur_seq_len, cache_full);
        $finish;
    end

    // Collect read data
    always_ff @(posedge clk) begin
        if (rd_valid) begin
            read_buf[read_buf_idx] <= rd_data;
            read_buf_idx++;
        end
    end

    // Global timeout
    initial begin
        #200000;
        $display("[TB] Global simulation timeout");
        $finish;
    end

endmodule
`endif // SIMULATION
