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
//   cmd_pos:   start position (PREFILL) or read start pos (ATTN_READ);
//              ignored by DECODE_APPEND, which appends at cur_seq_len
//   cmd_len:   number of tokens (PREFILL: >= 1; DECODE: ignored, always 1;
//              ATTN_READ: typically cur_seq_len). 0 = no-op for PREFILL/ATTN_READ
//   cmd_kv:    1=read K only, 0=read V only, X for writes (writes always write both K+V)
//
// Write data is presented on wr_data (DATA_WIDTH bits wide) with wr_valid/wr_ready handshake
// (one beat is written on every cycle both are high).
// Read data is presented on rd_data (DATA_WIDTH bits wide) with rd_valid output,
// one beat per cycle; rd_last marks the final beat of each ATTN_READ.
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
// - The controller maintains a cur_seq_len register, set to cmd_pos + cmd_len by
//   PREFILL (issued once per layer over the same positions) and incremented by
//   the DECODE_APPEND for the last layer (NUM_LAYERS-1): a decode step appends
//   each layer's K/V at the same position cur_seq_len. Reset on rst_n.
// - Commands execute strictly in FIFO order, one at a time, so an ATTN_READ
//   issued after a DECODE_APPEND sees the appended token.
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
    // Format (LSB first): type[1:0] | layer | pos | len | kv
    // (field widths follow the parameters: LAYER_BITS, POS_BITS)
    // -----------------------------------------------------------------------
    localparam int LAYER_BITS = $clog2(NUM_LAYERS);
    localparam int POS_BITS   = $clog2(MAX_SEQ_LEN);
    localparam int CMD_BITS   = 3 + LAYER_BITS + 2 * POS_BITS;
    wire [CMD_BITS-1:0] cmd_packed;
    wire [CMD_BITS-1:0] cmd_fifo_dout;
    wire                cmd_fifo_empty, cmd_fifo_full;
    wire                cmd_fifo_pop;

    assign cmd_packed = {cmd_kv, cmd_len, cmd_pos, cmd_layer, cmd_type};
    assign cmd_ready  = !cmd_fifo_full;

    // Unpack the command at the FIFO head
    wire [1:0]            head_type  = cmd_fifo_dout[1:0];
    wire [LAYER_BITS-1:0] head_layer = cmd_fifo_dout[2 +: LAYER_BITS];
    wire [POS_BITS-1:0]   head_pos   = cmd_fifo_dout[2 + LAYER_BITS +: POS_BITS];
    wire [POS_BITS-1:0]   head_len   = cmd_fifo_dout[2 + LAYER_BITS + POS_BITS +: POS_BITS];
    wire                  head_kv    = cmd_fifo_dout[CMD_BITS-1];

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
    // Operation registers
    // -----------------------------------------------------------------------
    // The command is popped from the FIFO when it starts and its fields are
    // held here for the whole operation (so the next command can never be
    // started twice or mixed into this one).
    reg [LAYER_BITS-1:0]            op_layer;     // Layer of the active command
    reg [POS_BITS-1:0]              op_pos;       // Current token position
    reg [$clog2(BEATS_PER_TOKEN):0] op_beat;      // Beat within current token
    reg [POS_BITS-1:0]              op_remaining; // Tokens remaining
    reg [$clog2(BEATS_PER_TOKEN):0] kv_offset;    // 0 for K, BEATS_PER_KV for V
    reg [POS_BITS:0]                op_end;       // PREFILL: pos + len

    // Start the head command when idle (pop it in the same cycle)
    wire start_cmd = (state == S_IDLE) && !cmd_fifo_empty;
    assign cmd_fifo_pop = start_cmd;

    // -----------------------------------------------------------------------
    // Write path: one beat per cycle on the wr_valid/wr_ready handshake,
    // written straight into the SRAM at the address of the current beat.
    // PREFILL writes at op_pos; DECODE_APPEND writes at position cur_seq_len.
    // -----------------------------------------------------------------------
    wire wr_beat = wr_valid && wr_ready;

    assign wr_ready     = (state == S_PREFILL_WR || state == S_DECODE_WR);
    assign sram_wr_en   = wr_beat;
    assign sram_wr_addr = sram_addr(op_layer,
                                    (state == S_DECODE_WR) ? seq_len_reg[POS_BITS-1:0] : op_pos,
                                    op_beat);
    assign sram_wr_data = wr_data;

    // -----------------------------------------------------------------------
    // Read control signals
    // -----------------------------------------------------------------------
    reg                  rd_req_reg;    // Assert SRAM rd_en
    reg [ADDR_BITS-1:0]  rd_addr_reg;  // SRAM read address
    reg                  rd_valid_reg; // Output valid (delayed by 1 cycle for SRAM latency)
    reg                  rd_last_req;  // The request in flight is the final beat
    reg                  rd_last_reg;

    assign sram_rd_en   = rd_req_reg;
    assign sram_rd_addr = rd_addr_reg;
    assign rd_valid     = rd_valid_reg;
    assign rd_data      = sram_rd_data;
    assign rd_last      = rd_last_reg;

    // -----------------------------------------------------------------------
    // Main FSM
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            seq_len_reg  <= '0;
            rd_req_reg   <= 1'b0;
            rd_addr_reg  <= '0;
            rd_valid_reg <= 1'b0;
            rd_last_req  <= 1'b0;
            rd_last_reg  <= 1'b0;
            op_layer     <= '0;
            op_pos       <= '0;
            op_beat      <= '0;
            op_remaining <= '0;
            kv_offset    <= '0;
            op_end       <= '0;
        end else begin
            // Propagate SRAM read latency (1 cycle) to rd_valid / rd_last
            rd_valid_reg <= rd_req_reg;
            rd_last_reg  <= rd_req_reg && rd_last_req;
            rd_req_reg   <= 1'b0;
            rd_last_req  <= 1'b0;

            case (state)
                // -----------------------------------------------------------
                S_IDLE : begin
                    if (start_cmd) begin
                        // Decode and start the next command
                        op_layer     <= head_layer;
                        op_pos       <= head_pos;
                        op_beat      <= '0;
                        op_remaining <= head_len;
                        op_end       <= head_pos + head_len;
                        kv_offset    <= head_kv ?
                                        '0 :              // K: offset=0
                                        BEATS_PER_KV[($clog2(BEATS_PER_TOKEN)):0]; // V: offset=BEATS_PER_KV

                        case (head_type)
                            // PREFILL / ATTN_READ of zero tokens complete at once
                            2'b00 : state <= (head_len != 0) ? S_PREFILL_WR  : S_IDLE;
                            2'b01 : state <= S_DECODE_WR;    // DECODE_APPEND
                            2'b10 : state <= (head_len != 0) ? S_ATTN_RD_REQ : S_IDLE;
                            default: state <= S_IDLE;
                        endcase
                    end
                end

                // -----------------------------------------------------------
                // PREFILL WRITE
                // Write BEATS_PER_TOKEN beats per token for op_remaining tokens,
                // one beat per accepted wr_valid/wr_ready handshake.
                // -----------------------------------------------------------
                S_PREFILL_WR : begin
                    if (wr_beat) begin
                        if (op_beat == BEATS_PER_TOKEN - 1) begin
                            // Finished this token's K and V
                            op_beat <= '0;
                            if (op_remaining == 1) begin
                                // Last token of prefill
                                seq_len_reg  <= op_end;
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
                // Append exactly one token for layer op_layer: write
                // BEATS_PER_TOKEN beats at position cur_seq_len. A decode step
                // appends every layer at the same position, so cur_seq_len
                // advances after the append to the last layer.
                // -----------------------------------------------------------
                S_DECODE_WR : begin
                    if (wr_beat) begin
                        if (op_beat == BEATS_PER_TOKEN - 1) begin
                            op_beat <= '0;
                            if (op_layer == LAYER_BITS'(NUM_LAYERS - 1))
                                seq_len_reg <= seq_len_reg + 1'b1;
                            state <= S_IDLE;
                        end else begin
                            op_beat <= op_beat + 1'b1;
                        end
                    end
                end

                // -----------------------------------------------------------
                // ATTENTION READ — Request Phase
                // Issue one read address per cycle: K or V (cmd_kv) for
                // positions cmd_pos .. cmd_pos+cmd_len-1.
                // -----------------------------------------------------------
                S_ATTN_RD_REQ : begin
                    // Issue SRAM read for current address
                    rd_addr_reg <= sram_addr(op_layer, op_pos, kv_offset + op_beat);
                    rd_req_reg  <= 1'b1;

                    if (op_beat == BEATS_PER_KV - 1) begin
                        // Last beat for this position
                        op_beat <= '0;
                        if (op_remaining == 1) begin
                            // Last beat of the command: flag it so rd_last
                            // accompanies its data
                            rd_last_req <= 1'b1;
                            state       <= S_ATTN_RD_DATA;
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
                // The last request is read by the SRAM at this edge; its data
                // (with rd_last) is on rd_data during the next cycle.
                // -----------------------------------------------------------
                S_ATTN_RD_DATA : begin
                    state <= S_IDLE;
                end

                default : state <= S_IDLE;
            endcase
        end
    end

endmodule


// =============================================================================
// TESTBENCH
// =============================================================================
// Tests the three operations: PREFILL_WRITE, DECODE_APPEND, and ATTN_READ,
// against a reference model of the cache contents (ref_mem).
//
// Scenario:
//   1. PREFILL: Write K/V for 2 tokens at every layer.
//   2. DECODE: Append K/V for 1 more token at every layer (cur_seq_len must
//      advance only after the last layer).
//   3. ATTN_READ: Read all 3 tokens' K vectors for layer 0.
//   4. ATTN_READ: Read V vectors for layer 2, and K for layer 3 from pos 1.
//   5. Three ATTN_READs queued back-to-back in the command FIFO.
//   Every read beat is compared with ref_mem; beat counts and rd_last (only on
//   the final beat of each read) are checked; the verdict counts all errors.
// Unique data per word: layer*4096 + pos*64 + beat (beat < 64, pos < 64).
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

    // Reference model of the cache: ref_mem[layer][pos][beat within K+V block]
    logic [DATA_WIDTH-1:0] ref_mem [NUM_LAYERS][MAX_SEQ_LEN][BEATS_PER_TOKEN];
    int errors = 0;

    function automatic logic [DATA_WIDTH-1:0] word_val(int layer, int pos, int beat);
        return DATA_WIDTH'(layer * 4096 + pos * 64 + beat);
    endfunction

    // Task: send one command. Stimulus changes on negedge; cmd_ready is
    // sampled at negedge (mid-cycle, stable), so the push happens on the
    // following posedge.
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
        while (cmd_ready !== 1'b1) @(negedge clk);
        @(negedge clk);
        cmd_valid = 1'b0;
    endtask

    // Task: write the K+V beats for n_tok tokens of one layer (same handshake
    // style as send_cmd), updating the reference model
    task automatic write_tokens(input int layer, input int pos0, input int n_tok);
        for (int t = 0; t < n_tok; t++) begin
            for (int b = 0; b < BEATS_PER_TOKEN; b++) begin
                @(negedge clk);
                wr_valid = 1'b1;
                wr_data  = word_val(layer, pos0 + t, b);
                ref_mem[layer][pos0 + t][b] = wr_data;
                while (wr_ready !== 1'b1) @(negedge clk);
            end
        end
        @(negedge clk);
        wr_valid = 1'b0;
    endtask

    // Read monitor
    logic [DATA_WIDTH-1:0] rd_q[$];
    bit                    last_q[$];
    always @(posedge clk) begin
        if (rd_valid === 1'b1) begin
            rd_q.push_back(rd_data);
            last_q.push_back(rd_last);
        end else if (rd_last === 1'b1) begin
            $display("[TB] FAIL: rd_last without rd_valid");
            errors++;
        end
    end

    // Expected stream for one ATTN_READ
    task automatic expect_read(input string name, input int layer, input int pos0, input int n_tok,
                               input bit kv, inout logic [DATA_WIDTH-1:0] exp_q[$], inout bit exp_last[$]);
        int off;
        off = kv ? 0 : BEATS_PER_KV;
        for (int t = 0; t < n_tok; t++)
            for (int b = 0; b < BEATS_PER_KV; b++) begin
                exp_q.push_back(ref_mem[layer][pos0 + t][off + b]);
                exp_last.push_back((t == n_tok - 1) && (b == BEATS_PER_KV - 1));
            end
    endtask

    // Wait for n beats (or time out) and compare the monitor queue with exp
    task automatic check_reads(input string name, input logic [DATA_WIDTH-1:0] exp_q[$], input bit exp_last[$]);
        int bad;
        fork
            begin : rd_timeout
                repeat (1000) @(posedge clk);
                disable rd_wait;
            end
            begin : rd_wait
                wait (rd_q.size() >= exp_q.size());
                disable rd_timeout;
            end
        join
        repeat (4) @(posedge clk);   // catch any extra beats
        bad = 0;
        if (rd_q.size() != exp_q.size()) begin
            $display("[TB] FAIL %s: %0d read beats, expected %0d", name, rd_q.size(), exp_q.size());
            bad++;
        end
        for (int i = 0; i < exp_q.size() && i < rd_q.size(); i++) begin
            if (rd_q[i] !== exp_q[i] || last_q[i] !== exp_last[i]) begin
                if (bad < 5)
                    $display("[TB] FAIL %s beat %0d: data 0x%04h last %0b, expected 0x%04h last %0b",
                             name, i, rd_q[i], last_q[i], exp_q[i], exp_last[i]);
                bad++;
            end
        end
        if (bad == 0) $display("[TB] PASS %s: %0d beats match", name, exp_q.size());
        errors += bad;
        rd_q.delete();
        last_q.delete();
    endtask

    task automatic check_seq_len(input string when, input int exp);
        if (cur_seq_len !== exp) begin
            $display("[TB] FAIL: cur_seq_len = %0d %s (expected %0d)", cur_seq_len, when, exp);
            errors++;
        end else
            $display("[TB]   cur_seq_len = %0d %s", cur_seq_len, when);
    endtask

    logic [DATA_WIDTH-1:0] exp_q[$];
    bit                    exp_last[$];

    // Main test
    initial begin
        $display("=== KV Cache Controller Testbench ===");
        $display("NUM_HEADS=%0d, HEAD_DIM=%0d, BEATS_PER_TOKEN=%0d",
                 NUM_HEADS, HEAD_DIM, BEATS_PER_TOKEN);

        // Reset
        rst_n     = 1'b0;
        cmd_valid = 1'b0;
        wr_valid  = 1'b0;
        wr_data   = '0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // ------------------------------------------------------------------
        // TEST 1: PREFILL — write 2 tokens (K+V) at every layer, pos 0
        // ------------------------------------------------------------------
        $display("[TB] Test 1: PREFILL write of 2 tokens at each layer");
        for (int l = 0; l < NUM_LAYERS; l++) begin
            send_cmd(.t(2'b00), .l(l), .p(0), .n(2), .kv(1'b0));
            write_tokens(l, 0, 2);
        end
        repeat (3) @(posedge clk);
        check_seq_len("after prefill", 2);

        // ------------------------------------------------------------------
        // TEST 2: DECODE APPEND — append 1 token at each layer (position 2)
        // ------------------------------------------------------------------
        $display("[TB] Test 2: DECODE APPEND of 1 token at each layer");
        for (int l = 0; l < NUM_LAYERS; l++) begin
            send_cmd(.t(2'b01), .l(l), .p(0), .n(1), .kv(1'b0));
            write_tokens(l, 2, 1);
            repeat (3) @(posedge clk);
            check_seq_len($sformatf("after layer %0d append", l), (l == NUM_LAYERS - 1) ? 3 : 2);
        end

        // ------------------------------------------------------------------
        // TEST 3: ATTN_READ — read K vectors for 3 tokens at layer 0
        // ------------------------------------------------------------------
        $display("[TB] Test 3: ATTN_READ of K vectors, 3 tokens at layer 0");
        exp_q.delete(); exp_last.delete();
        expect_read("T3", 0, 0, 3, 1'b1, exp_q, exp_last);
        send_cmd(.t(2'b10), .l(0), .p(0), .n(3), .kv(1'b1)); // kv=1: read K
        check_reads("Test 3 (L0 K x3)", exp_q, exp_last);

        // ------------------------------------------------------------------
        // TEST 4: ATTN_READ — V of layer 2, then K of layer 3 from pos 1
        // ------------------------------------------------------------------
        $display("[TB] Test 4: ATTN_READ of V (layer 2) and K from pos 1 (layer 3)");
        exp_q.delete(); exp_last.delete();
        expect_read("T4a", 2, 0, 3, 1'b0, exp_q, exp_last);
        send_cmd(.t(2'b10), .l(2), .p(0), .n(3), .kv(1'b0)); // kv=0: read V
        check_reads("Test 4a (L2 V x3)", exp_q, exp_last);
        exp_q.delete(); exp_last.delete();
        expect_read("T4b", 3, 1, 2, 1'b1, exp_q, exp_last);
        send_cmd(.t(2'b10), .l(3), .p(1), .n(2), .kv(1'b1));
        check_reads("Test 4b (L3 K pos1 x2)", exp_q, exp_last);

        // ------------------------------------------------------------------
        // TEST 5: three reads queued back-to-back in the command FIFO
        // ------------------------------------------------------------------
        $display("[TB] Test 5: three queued ATTN_READs");
        exp_q.delete(); exp_last.delete();
        expect_read("T5a", 1, 0, 3, 1'b1, exp_q, exp_last);
        expect_read("T5b", 1, 0, 3, 1'b0, exp_q, exp_last);
        expect_read("T5c", 0, 2, 1, 1'b0, exp_q, exp_last);
        send_cmd(.t(2'b10), .l(1), .p(0), .n(3), .kv(1'b1));
        send_cmd(.t(2'b10), .l(1), .p(0), .n(3), .kv(1'b0));
        send_cmd(.t(2'b10), .l(0), .p(2), .n(1), .kv(1'b0));
        check_reads("Test 5 (queued reads)", exp_q, exp_last);

        check_seq_len("at end", 3);
        if (cache_full !== 1'b0) begin
            $display("[TB] FAIL: cache_full set at seq_len 3");
            errors++;
        end

        if (errors == 0) $display("[TB] ALL TESTS PASSED");
        else             $display("[TB] FAIL: %0d error(s)", errors);
        $finish;
    end

    // Global timeout
    initial begin
        #200000;
        $display("[TB] FAIL: Global simulation timeout");
        $finish;
    end

endmodule
`endif // SIMULATION
