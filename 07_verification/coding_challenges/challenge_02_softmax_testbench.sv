// =============================================================================
// Challenge: UVM-Lite Testbench for a Softmax Unit
// =============================================================================
//
// Design Under Test (DUT) interface
// ----------------------------------
// The softmax unit accepts a vector of N_ELEMENTS signed 16-bit integers,
// applies the numerically stable softmax, and emits a vector of N_ELEMENTS
// unsigned 8-bit fixed-point outputs (Q0.8 format, i.e. value/256 ≈ probability).
//
// DUT ports:
//   clk            : input  logic         — clock
//   reset_n        : input  logic         — active-low synchronous reset
//   input_valid    : input  logic         — one-cycle pulse: input data valid
//   input_data     : input  logic signed [15:0] input_data [N_ELEMENTS]  — unpacked array
//   output_valid   : output logic         — one-cycle pulse: output data valid
//   output_data    : output logic [7:0] output_data [N_ELEMENTS]         — unpacked array
//
// The DUT has a pipeline latency of PIPE_LATENCY cycles from input_valid to
// output_valid. The DUT is ready to accept a new input every cycle (no stall).
//
// Testbench Structure (UVM-lite, no full UVM library dependency)
// --------------------------------------------------------------
//   - SoftmaxDriver     : generates and applies random input vectors
//   - SoftmaxMonitor    : captures DUT output transactions
//   - SoftmaxScoreboard : compares DUT output against precomputed expected values
//   - SoftmaxCoverage   : functional coverage model
//   - top_tb            : top-level module wiring everything together
//
// Golden Model (DPI-C)
// --------------------
// The expected outputs are computed by a C function called via DPI-C.
// The function signature (implemented in softmax_golden.c) is:
//
//   void softmax_golden_c(
//     const int16_t *input,      // N_ELEMENTS signed 16-bit inputs
//     uint8_t       *output,     // N_ELEMENTS unsigned 8-bit outputs (Q0.8)
//     int            n_elements  // vector length
//   );
//
// The C function computes:
//   1. Find max(input[i])
//   2. Compute exp(input[i] - max) for each i (using float arithmetic)
//   3. Normalise by sum to get probabilities in [0.0, 1.0]
//   4. Convert to Q0.8: output[i] = round(prob[i] * 256), clamped to [0, 255]
//
// For compilation (the C model is in softmax_golden.c next to this file), use a
// simulator with covergroup support, e.g. Questa:
//   vlog -sv -dpiheader dpi_header.h challenge_02_softmax_testbench.sv softmax_golden.c
//   vsim -c softmax_testbench -do "run -all"
// or the Vivado simulator (xsim, verified with 2025.2):
//   xsc softmax_golden.c
//   xvlog -sv challenge_02_softmax_testbench.sv
//   xelab softmax_testbench -sv_lib dpi -s snap && xsim snap -R
//
// A behavioural softmax_dut stand-in is included at the end of this file so
// the testbench elaborates and runs out of the box. Replace it with your RTL.
//
// =============================================================================

`timescale 1ns/1ps

// =============================================================================
// Package: Shared parameters and transaction type
// =============================================================================

package softmax_pkg;

  // Vector length — must match the DUT parameter
  parameter int N_ELEMENTS    = 16;
  // Pipeline latency of the DUT in clock cycles
  parameter int PIPE_LATENCY  = 4;
  // Number of random test vectors to generate
  parameter int NUM_TESTS     = 500;
  // Tolerance: maximum allowed absolute difference per output element (LSBs)
  // Set to 1 to allow for rounding differences between C golden model and RTL
  parameter int OUTPUT_TOLERANCE = 1;

  // One DUT output vector, as captured by the monitor
  typedef logic [7:0] out_vec_t [0:N_ELEMENTS-1];

  // Transaction: one input vector and its corresponding expected output.
  // Unpacked struct: a packed struct cannot contain unpacked arrays.
  typedef struct {
    logic signed [15:0] input_vec  [N_ELEMENTS];
    logic        [7:0]  golden_vec [N_ELEMENTS];
  } softmax_txn_t;

endpackage : softmax_pkg


// DPI-C import: call the C golden model from within SystemVerilog.
// Fixed-size arrays are passed to C as plain pointers (const short*, char*),
// matching the softmax_golden_c() signature. Open arrays ([]) would instead
// arrive as svOpenArrayHandle.
import "DPI-C" function void softmax_golden_c(
    input  shortint input_data  [softmax_pkg::N_ELEMENTS],   // int16_t array
    output byte     output_data [softmax_pkg::N_ELEMENTS],   // uint8_t array
    input  int      n_elements
);


// =============================================================================
// Interface: DUT signal bundle
// =============================================================================

interface softmax_if #(
    parameter int N = softmax_pkg::N_ELEMENTS
) (
    input logic clk
);
  logic        reset_n;
  logic        input_valid;
  logic signed [15:0] input_data [N];   // unpacked array for easy indexing
  logic        output_valid;
  logic        [7:0]  output_data [N];

  // Clocking block for the driver (drive on negedge, sample on posedge)
  clocking driver_cb @(negedge clk);
    default output #1;
    output reset_n;
    output input_valid;
    output input_data;
  endclocking

  // Clocking block for the monitor (sample on posedge, 1ns setup time)
  clocking monitor_cb @(posedge clk);
    default input #1;
    input output_valid;
    input output_data;
  endclocking

endinterface : softmax_if


// =============================================================================
// Driver: generates random input vectors and applies them to the DUT
// =============================================================================

class SoftmaxDriver;

  // Reference to the virtual interface
  virtual softmax_if vif;

  // Scoreboard FIFO: pass expected outputs to the scoreboard
  // (Using a simple mailbox as a lightweight UVM TLM replacement)
  mailbox #(softmax_pkg::softmax_txn_t) sb_fifo;

  // Random number seed (set before calling run() for reproducibility)
  int unsigned seed = 42;

  function new(virtual softmax_if vif,
               mailbox #(softmax_pkg::softmax_txn_t) sb_fifo);
    this.vif    = vif;
    this.sb_fifo = sb_fifo;
  endfunction

  // Generate a random input vector with a specific data pattern.
  // pattern: 0=uniform random, 1=all same, 2=single hot, 3=sorted ascending,
  //          4=all zero, 5=extreme values (max/min mix)
  task generate_input(
      output logic signed [15:0] vec[softmax_pkg::N_ELEMENTS],
      input  int                 pattern
  );
    int N = softmax_pkg::N_ELEMENTS;
    logic [15:0] base_val;

    case (pattern)
      0: begin  // Uniform random: values in [-1000, 1000]
           for (int i = 0; i < N; i++)
             vec[i] = $signed($urandom_range(0, 2000)) - 1000;
         end
      1: begin  // All same value: softmax output should be uniform
           base_val = $urandom_range(0, 1000);
           for (int i = 0; i < N; i++) vec[i] = base_val;
         end
      2: begin  // Single hot: one element much larger than others
           for (int i = 0; i < N; i++) vec[i] = $signed(-100);
           vec[$urandom_range(0, N-1)] = $signed(32767);  // INT16 max
         end
      3: begin  // Sorted ascending: [0, 1, 2, ..., N-1]
           for (int i = 0; i < N; i++) vec[i] = $signed(i * 10);
         end
      4: begin  // All zeros: uniform output expected
           for (int i = 0; i < N; i++) vec[i] = '0;
         end
      5: begin  // Extreme: mix of INT16 max and min
           for (int i = 0; i < N; i++)
             vec[i] = ($urandom_range(0,1)) ? $signed(32767) : $signed(-32768);
         end
      default: begin
           for (int i = 0; i < N; i++)
             vec[i] = $signed($urandom_range(0, 2000)) - 1000;
         end
    endcase
  endtask

  // Main driver task: runs for NUM_TESTS transactions
  task run();
    int N = softmax_pkg::N_ELEMENTS;
    softmax_pkg::softmax_txn_t txn;
    logic signed [15:0]  input_vec  [softmax_pkg::N_ELEMENTS];
    // DPI-C expects byte arrays; shortint = 16-bit signed
    shortint             dpi_input  [softmax_pkg::N_ELEMENTS];
    byte                 dpi_output [softmax_pkg::N_ELEMENTS];
    int pattern;

    process::self().srandom(seed);  // seed this thread's RNG for reproducibility

    for (int t = 0; t < softmax_pkg::NUM_TESTS; t++) begin
      // Choose data pattern: cycle through 0..5 with mostly random
      pattern = (t % 20 == 0) ? (t / 20) % 6 : 0;

      generate_input(input_vec, pattern);

      // Convert to DPI-C compatible types and call golden model
      for (int i = 0; i < N; i++) dpi_input[i] = shortint'(input_vec[i]);
      softmax_golden_c(dpi_input, dpi_output, N);

      // Build transaction and send expected output to scoreboard
      for (int i = 0; i < N; i++) begin
        txn.input_vec[i]  = input_vec[i];
        txn.golden_vec[i] = 8'(dpi_output[i]);
      end
      sb_fifo.put(txn);

      // Drive input onto the DUT
      @(vif.driver_cb);
      vif.driver_cb.input_valid <= 1'b1;
      for (int i = 0; i < N; i++)
        vif.driver_cb.input_data[i] <= input_vec[i];

      @(vif.driver_cb);
      vif.driver_cb.input_valid <= 1'b0;

      // One idle cycle between transactions (can be set to 0 for back-to-back)
      @(vif.driver_cb);
    end

    // Wait for pipeline to drain after last input
    repeat (softmax_pkg::PIPE_LATENCY + 5) @(vif.driver_cb);
  endtask

endclass : SoftmaxDriver


// =============================================================================
// Monitor: captures DUT output transactions
// =============================================================================

class SoftmaxMonitor;

  virtual softmax_if vif;

  // FIFO to pass captured outputs to the scoreboard
  mailbox #(softmax_pkg::out_vec_t) out_fifo;

  function new(virtual softmax_if vif,
               mailbox #(softmax_pkg::out_vec_t) out_fifo);
    this.vif      = vif;
    this.out_fifo = out_fifo;
  endfunction

  task run();
    logic [7:0] captured [softmax_pkg::N_ELEMENTS];
    forever begin
      @(vif.monitor_cb);
      if (vif.monitor_cb.output_valid === 1'b1) begin
        for (int i = 0; i < softmax_pkg::N_ELEMENTS; i++)
          captured[i] = vif.monitor_cb.output_data[i];
        // Pack into a flat representation for the mailbox
        // (using a simple array copy; a struct would be cleaner)
        begin
          logic [7:0] copy [softmax_pkg::N_ELEMENTS];
          for (int i = 0; i < softmax_pkg::N_ELEMENTS; i++) copy[i] = captured[i];
          out_fifo.put(copy);
        end
      end
    end
  endtask

endclass : SoftmaxMonitor


// =============================================================================
// Scoreboard: compares DUT output against golden model expected output
// =============================================================================

class SoftmaxScoreboard;

  // FIFOs connecting to driver (expected) and monitor (actual)
  mailbox #(softmax_pkg::softmax_txn_t)                  txn_fifo;
  mailbox #(softmax_pkg::out_vec_t)    out_fifo;

  // Statistics
  int total_checks  = 0;
  int total_errors  = 0;
  int total_warnings = 0;  // within tolerance but not exact

  function new(
      mailbox #(softmax_pkg::softmax_txn_t) txn_fifo,
      mailbox #(softmax_pkg::out_vec_t) out_fifo
  );
    this.txn_fifo = txn_fifo;
    this.out_fifo  = out_fifo;
  endfunction

  task run();
    softmax_pkg::softmax_txn_t      txn;
    logic [7:0] dut_output [softmax_pkg::N_ELEMENTS];
    int diff;
    bit  txn_error;
    int N = softmax_pkg::N_ELEMENTS;

    forever begin
      txn_fifo.get(txn);            // wait for an expected-output record
      out_fifo.get(dut_output);     // wait for the corresponding DUT output

      txn_error = 1'b0;
      total_checks++;

      for (int i = 0; i < N; i++) begin
        diff = int'(dut_output[i]) - int'(txn.golden_vec[i]);
        if (diff < 0) diff = -diff;  // absolute value

        if (diff > softmax_pkg::OUTPUT_TOLERANCE) begin
          if (!txn_error) begin
            $error("[Scoreboard] Mismatch on transaction %0d at element %0d: DUT=%0d, expected=%0d, diff=%0d",
                   total_checks, i, dut_output[i], txn.golden_vec[i], diff);
            $display("  Input vector: ");
            for (int j = 0; j < N; j++)
              $write("  [%0d]=%0d", j, $signed(txn.input_vec[j]));
            $display("");
          end
          txn_error = 1'b1;
          total_errors++;
        end else if (diff > 0) begin
          total_warnings++;  // within tolerance, not exact (not an error)
        end
      end

      if (!txn_error && (total_checks % 50 == 0)) begin
        $display("[Scoreboard] %0d transactions checked, %0d errors, %0d warnings",
                 total_checks, total_errors, total_warnings);
      end
    end
  endtask

  // Every driven vector must produce exactly one checked output
  function void report(int expected_txns);
    if (total_checks != expected_txns || txn_fifo.num() != 0 || out_fifo.num() != 0) begin
      $error("[Scoreboard] %0d transactions checked (expected %0d); %0d expected and %0d DUT outputs left unmatched",
             total_checks, expected_txns, txn_fifo.num(), out_fifo.num());
      total_errors++;
    end
    $display("");
    $display("%s", {60{"="}});
    $display("[Scoreboard] Final Report");
    $display("  Total transactions  : %0d", total_checks);
    $display("  Errors (diff > %0d) : %0d", softmax_pkg::OUTPUT_TOLERANCE, total_errors);
    $display("  Warnings (diff = 1) : %0d", total_warnings);
    if (total_errors == 0)
      $display("  RESULT: PASS");
    else
      $display("  RESULT: FAIL");
    $display("%s", {60{"="}});
  endfunction

endclass : SoftmaxScoreboard


// =============================================================================
// Coverage Model: functional coverage for the softmax testbench
// =============================================================================

class SoftmaxCoverage;

  // Summary statistics of the most recent input vector (sampled each time a
  // new input vector is driven)
  // Track the maximum and minimum values in the input vector
  logic signed [15:0] vec_max;
  logic signed [15:0] vec_min;
  int                 vec_range;   // max - min (spread of the distribution)
  int                 n_zeros;     // number of zero elements in the vector
  logic [7:0]         first_out_elem;  // representative output element

  covergroup softmax_input_cg;

    // Coverpoint: range of the input vector (spread matters for exp saturation)
    cp_input_range: coverpoint vec_range {
      bins zero_range   = {0};                    // all elements equal
      bins small_range  = {[1:99]};
      bins medium_range = {[100:9999]};
      bins large_range  = {[10000:65535]};        // near full INT16 range
    }

    // Coverpoint: maximum value (tests exp(0) path and positive saturation)
    cp_max_value: coverpoint vec_max {
      bins very_negative = {[$:-1000]};
      bins near_zero_neg = {[-999:-1]};
      bins zero          = {0};
      bins near_zero_pos = {[1:999]};
      bins large_pos     = {[1000:32767]};
      bins int16_max     = {32767};
    }

    // Coverpoint: number of zero elements (sparse inputs)
    cp_zero_elements: coverpoint n_zeros {
      bins no_zeros    = {0};
      bins some_zeros  = {[1:softmax_pkg::N_ELEMENTS/2 - 1]};
      bins half_zeros  = {softmax_pkg::N_ELEMENTS/2};
      bins mostly_zero = {[softmax_pkg::N_ELEMENTS/2 + 1 : softmax_pkg::N_ELEMENTS - 1]};
      bins all_zeros   = {softmax_pkg::N_ELEMENTS};
    }

    // Coverpoint: vector length (always N_ELEMENTS here, but shows the pattern)
    cp_n_elements: coverpoint softmax_pkg::N_ELEMENTS {
      bins supported = {softmax_pkg::N_ELEMENTS};
    }

    // Cross coverage: large range AND maximum at INT16 max (extreme single-hot)
    cx_extreme_singlehot: cross cp_input_range, cp_max_value {
      // Focus on: large range with INT16 max value (single-hot extreme case)
      bins extreme_singlehot =
        binsof(cp_input_range.large_range) && binsof(cp_max_value.int16_max);
    }

  endgroup : softmax_input_cg

  covergroup softmax_output_cg;

    // Sampled output (track which output bins are hit)
    // In practice, we sample the first element of the output as a representative
    // (first_out_elem is a class member: covergroups cannot declare variables)
    cp_output_value: coverpoint first_out_elem {
      bins zero          = {8'h00};         // probability ≈ 0
      bins low_prob      = {[8'h01:8'h0F]}; // < 6%
      bins medium_prob   = {[8'h10:8'h7F]}; // 6% to 50%
      bins high_prob     = {[8'h80:8'hFE]}; // 50% to 99%
      bins near_certain  = {8'hFF};         // probability ≈ 1 (single hot argmax)
    }

  endgroup : softmax_output_cg

  function new();
    softmax_input_cg  = new();
    softmax_output_cg = new();
  endfunction

  // Sample coverage given a new input vector
  function void sample_input(logic signed [15:0] vec[softmax_pkg::N_ELEMENTS]);
    int N = softmax_pkg::N_ELEMENTS;
    vec_max  = vec[0];
    vec_min  = vec[0];
    n_zeros  = 0;

    for (int i = 0; i < N; i++) begin
      if ($signed(vec[i]) > $signed(vec_max)) vec_max = vec[i];
      if ($signed(vec[i]) < $signed(vec_min)) vec_min = vec[i];
      if (vec[i] == '0) n_zeros++;
    end
    vec_range = int'($signed(vec_max)) - int'($signed(vec_min));

    softmax_input_cg.sample();
  endfunction

  function void sample_output(logic [7:0] out_vec[softmax_pkg::N_ELEMENTS]);
    first_out_elem = out_vec[0];
    softmax_output_cg.sample();
  endfunction

  function void report();
    $display("[Coverage] Input  covergroup: %.1f%%",
             softmax_input_cg.get_coverage());
    $display("[Coverage]   cp_input_range %.1f%%, cp_max_value %.1f%%, cp_zero_elements %.1f%%, cp_n_elements %.1f%%, cx_extreme_singlehot %.1f%%",
             softmax_input_cg.cp_input_range.get_coverage(),
             softmax_input_cg.cp_max_value.get_coverage(),
             softmax_input_cg.cp_zero_elements.get_coverage(),
             softmax_input_cg.cp_n_elements.get_coverage(),
             softmax_input_cg.cx_extreme_singlehot.get_coverage());
    $display("[Coverage] Output covergroup: %.1f%%",
             softmax_output_cg.get_coverage());
  endfunction

endclass : SoftmaxCoverage


// =============================================================================
// Top-level testbench module
// =============================================================================

module softmax_testbench;

  import softmax_pkg::*;

  // -------------------------------------------------------------------------
  // Clock and reset generation
  // -------------------------------------------------------------------------
  logic clk;
  initial clk = 1'b0;
  always #5 clk = ~clk;   // 100 MHz clock

  // -------------------------------------------------------------------------
  // Interface instantiation
  // -------------------------------------------------------------------------
  softmax_if #(.N(N_ELEMENTS)) dut_if (.clk(clk));

  // -------------------------------------------------------------------------
  // DUT instantiation
  // Connect the interface to the DUT ports.
  // NOTE: Replace 'softmax_dut' with your actual module name.
  // -------------------------------------------------------------------------
  softmax_dut #(
    .N_ELEMENTS  (N_ELEMENTS),
    .PIPE_LATENCY(PIPE_LATENCY)
  ) dut (
    .clk          (clk),
    .reset_n      (dut_if.reset_n),
    .input_valid  (dut_if.input_valid),
    .input_data   (dut_if.input_data),
    .output_valid (dut_if.output_valid),
    .output_data  (dut_if.output_data)
  );

  // -------------------------------------------------------------------------
  // Testbench component instantiation
  // -------------------------------------------------------------------------
  // Shared mailboxes (TLM-lite channels)
  mailbox #(softmax_txn_t)  sb_txn_fifo;
  mailbox #(out_vec_t)      sb_out_fifo;

  SoftmaxDriver      driver;
  SoftmaxMonitor     monitor;
  SoftmaxScoreboard  scoreboard;
  SoftmaxCoverage    cov_model;

  // Build the components at time 0, in dependency order. (Constructing them
  // in their declarations would rely on the order of static initialisers,
  // which the LRM leaves undefined: a driver could be handed a null mailbox.)
  initial begin
    sb_txn_fifo = new();
    sb_out_fifo = new();
    driver      = new(dut_if, sb_txn_fifo);
    monitor     = new(dut_if, sb_out_fifo);
    scoreboard  = new(sb_txn_fifo, sb_out_fifo);
    cov_model   = new();
  end

  // -------------------------------------------------------------------------
  // Reset sequence
  // -------------------------------------------------------------------------
  initial begin
    dut_if.reset_n      = 1'b0;
    dut_if.input_valid  = 1'b0;
    for (int i = 0; i < N_ELEMENTS; i++) dut_if.input_data[i] = '0;

    repeat (5) @(posedge clk);
    @(negedge clk);
    dut_if.reset_n = 1'b1;
    $display("[TB] Reset released at time %0t", $time);
  end

  // -------------------------------------------------------------------------
  // Spawn driver, monitor, and scoreboard threads
  // -------------------------------------------------------------------------
  initial begin
    // Wait for reset to be released
    @(posedge dut_if.reset_n);
    repeat (2) @(posedge clk);

    // Fork all concurrent threads
    fork
      driver.run();     // generate and drive stimuli; puts expected outputs in sb_txn_fifo
      monitor.run();    // capture DUT outputs; puts them in sb_out_fifo
      scoreboard.run(); // compare; reads from both FIFOs
    join_any

    // Driver finishes first (after NUM_TESTS); give monitor time to drain
    repeat (PIPE_LATENCY + 10) @(posedge clk);

    // Final reports
    scoreboard.report(NUM_TESTS);
    cov_model.report();

    // Fail simulation if any errors were detected
    if (scoreboard.total_errors > 0) begin
      $fatal(1, "[TB] Simulation FAILED: %0d error(s) detected",
             scoreboard.total_errors);
    end else begin
      $display("[TB] Simulation PASSED: all %0d transactions correct",
               scoreboard.total_checks);
      $finish;
    end
  end

  // -------------------------------------------------------------------------
  // Coverage sampling: triggered by the driver's input_valid pulse
  // -------------------------------------------------------------------------
  always @(posedge clk) begin
    if (dut_if.input_valid) begin
      cov_model.sample_input(dut_if.input_data);
    end
    if (dut_if.output_valid) begin
      cov_model.sample_output(dut_if.output_data);
    end
  end

  // -------------------------------------------------------------------------
  // Timeout watchdog: prevent infinite hang if DUT never asserts output_valid
  // -------------------------------------------------------------------------
  initial begin
    // Allow 2x the expected number of cycles plus generous margin
    #(NUM_TESTS * 4 * 10ns + PIPE_LATENCY * 100 * 10ns);
    $fatal(1, "[TB] TIMEOUT: simulation exceeded watchdog limit");
  end

  // -------------------------------------------------------------------------
  // Waveform dump (for debugging; disable for regression speed)
  // -------------------------------------------------------------------------
  initial begin
    $dumpfile("softmax_tb.vcd");
    $dumpvars(0, softmax_testbench);
  end

endmodule : softmax_testbench


// =============================================================================
// Stand-in DUT: behavioural softmax reference (replace with your RTL)
// =============================================================================
//
// Computes the same numerically stable softmax as the C golden model, in
// double precision, and delays the result by PIPE_LATENCY cycles so the
// testbench sees a DUT with the documented interface and timing. It is not
// synthesisable: it exists so the testbench can be compiled and run on its
// own before a real softmax RTL is connected.
// =============================================================================

module softmax_dut #(
    parameter int N_ELEMENTS   = 16,
    parameter int PIPE_LATENCY = 4
) (
    input  logic               clk,
    input  logic               reset_n,
    input  logic               input_valid,
    input  logic signed [15:0] input_data  [N_ELEMENTS],
    output logic               output_valid,
    output logic        [7:0]  output_data [N_ELEMENTS]
);

  logic       valid_pipe [PIPE_LATENCY];
  logic [7:0] data_pipe  [PIPE_LATENCY][N_ELEMENTS];

  function automatic void softmax_q08(input  logic signed [15:0] x [N_ELEMENTS],
                                      output logic        [7:0]  y [N_ELEMENTS]);
    real e [N_ELEMENTS];
    real sum;
    int  max_val, q;
    max_val = x[0];
    for (int i = 1; i < N_ELEMENTS; i++)
      if (x[i] > max_val) max_val = x[i];
    sum = 0.0;
    for (int i = 0; i < N_ELEMENTS; i++) begin
      e[i] = $exp(real'(int'(x[i]) - max_val));
      sum += e[i];
    end
    for (int i = 0; i < N_ELEMENTS; i++) begin
      q = $rtoi(e[i] / sum * 256.0 + 0.5);
      y[i] = (q > 255) ? 8'd255 : 8'(q);
    end
  endfunction

  always_ff @(posedge clk) begin
    if (!reset_n) begin
      for (int s = 0; s < PIPE_LATENCY; s++) valid_pipe[s] <= 1'b0;
    end else begin
      valid_pipe[0] <= input_valid;
      if (input_valid) begin
        logic [7:0] y [N_ELEMENTS];
        softmax_q08(input_data, y);
        data_pipe[0] <= y;
      end
      for (int s = 1; s < PIPE_LATENCY; s++) begin
        valid_pipe[s] <= valid_pipe[s-1];
        data_pipe[s]  <= data_pipe[s-1];
      end
    end
  end

  assign output_valid = valid_pipe[PIPE_LATENCY-1];
  assign output_data  = data_pipe[PIPE_LATENCY-1];

endmodule : softmax_dut
