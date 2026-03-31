# Constrained Random Verification for LLM Accelerators

## Overview

Constrained random verification (CRV) applies pseudo-random stimulus generation within a
defined constraint space to achieve high coverage efficiently. For LLM accelerators, the
challenge is that the input space is enormous (tensor shapes, data values, quantisation
parameters, configuration registers) but most random points are not interesting. Constraints
guide the generator towards the corners, modes, and edge cases that are most likely to expose
hardware bugs.

---

## Tier 1 — Fundamentals

### Q1. What should you randomise in a constrained random testbench for an LLM accelerator?

**Answer.**

For an LLM accelerator the randomisable stimulus can be organised into four categories:

**1. Tensor dimensions.**
Matrix shapes, sequence lengths, and batch sizes define the operational range of every
datapath. Interesting dimensions are not just random integers; they include:

- Powers of two (likely natural tile sizes): 64, 128, 256, 512, 1024
- Powers of two minus one (stress off-by-one in loop control)
- Hardware tile granularity boundaries ± 1 (if tiles are 32 elements wide, test 31, 32, 33)
- Minimum supported size (often 1 or 4)
- Maximum supported size (bounded by on-chip SRAM capacity)
- Non-power-of-two values to stress padding logic

**2. Data values.**
The distribution of values in the tensors:

- All zeros (should produce all-zero output; tests zero propagation)
- All maximum positive (INT8: 127; tests saturation and overflow)
- All minimum (INT8: -128; tests negative overflow and sign handling)
- Mixed positive and negative (random uniform)
- Sparse (most elements zero, a few non-zero; models ReLU-after-activation distributions)
- Alternating max/min (worst case for accumulator carry chains)

**3. Quantisation parameters.**
Scale factors, zero points, and precision modes:

- Scale = 1.0 (trivial case)
- Scale = 2^-N for various N (power-of-two scales, common in hardware-friendly quantisation)
- Very small scale (near subnormal for FP16/BF16)
- Very large scale (near saturation of output range)
- Non-zero zero points (asymmetric quantisation)

**4. Configuration registers.**
Accelerator operating modes:

- Precision mode (INT8, INT4, BF16, FP16, FP8)
- Tiling configuration (tile height, tile width)
- Activation function enable bits (ReLU, GELU, none)
- Bias add enable
- Transposed input flags
- Accumulator clear vs. accumulate mode

---

### Q2. What is coverage-driven verification and why is it preferred over purely random testing?

**Answer.**

In purely random testing you generate a large number of random stimuli and rely on probability
to eventually hit all interesting cases. The problem is that interesting corner cases (e.g.
output value exactly at INT8 saturation boundary) may have a probability of approximately
1/256 of occurring in any given test. To be 99% confident of hitting it you need ~1177 random
tests just for that one corner. With thousands of independent corners, purely random testing
is statistically infeasible.

**Coverage-driven verification (CDV)** solves this by:

1. **Defining a coverage model** that enumerates all the interesting conditions (coverpoints
   and covergroups) that must be exercised.
2. **Measuring coverage** after each test run.
3. **Directing the test generator** (via constraints) towards uncovered bins when random
   generation alone is not hitting them efficiently.

The process is:

```
Generate random      →    Run simulation   →   Measure coverage
constrained stimulus                            ↓
          ↑                                  Coverage < 100%?
          |                                     ↓ Yes
    Tighten constraints              Find uncovered bins
    to target uncovered bins         Add directed tests for them
```

CDV gives a mathematically grounded answer to "when are we done testing?" — when all
coverage bins are closed. Purely random testing has no such endpoint.

---

### Q3. List five specific corner cases for INT8 GEMM that every constrained random test suite must cover.

**Answer.**

1. **Accumulator overflow boundary.**
   Inner dimension K is large enough that K × 127 × 127 approaches INT32_MAX
   (K × 16129 ≈ 2^31 requires K ≈ 133,169). Test with K near the maximum the hardware
   supports to ensure the INT32 accumulator does not overflow and the hardware handles it
   gracefully (either clamps or wraps as per spec).

2. **Requantisation boundary (INT8 saturation).**
   Input values chosen so that after scaling the result is exactly 127.0, 127.5, 128.0, and
   -128.0, -128.5. These stress the rounding and clamping boundary simultaneously.

3. **Zero-point non-zero (asymmetric quantisation).**
   Set `zp_a = 128` (unsigned INT8 with zero-point offset). This is used in PyTorch's default
   UINT8 quantisation. Many RTL bugs only appear when the zero-point is non-zero because
   the correction term `K * zp_a * sum(W)` involves a potentially large constant that must
   be pre-computed and subtracted from the accumulator.

4. **All-zero weight matrix.**
   Output should be all zero points regardless of activation. This tests the data path when
   all multiply-accumulate units are being driven to their quiescent state, and ensures no
   stale accumulated values leak into the output.

5. **K = 1 (single-element dot product).**
   This exercises the pipeline when there is exactly one cycle of valid input data. It stresses
   pipeline control (is the accumulator cleared correctly?) and ensures there is no dependency
   on multiple cycles being present to drive a completion signal.

---

## Tier 2 — Intermediate

### Q4. How do you model and constrain DMA and AXI transactions in a constrained random testbench?

**Answer.**

LLM accelerators typically receive input tensors and emit output tensors via an AXI master DMA
engine. Randomising DMA transactions means randomising:

- **Burst length** (AXI ARLEN/AWLEN): hardware must handle both single-beat transfers (ARLEN=0)
  and maximum-length bursts (ARLEN=255 for AXI4).
- **Burst type** (INCR, WRAP, FIXED): most DMA traffic is INCR; WRAP tests cache-line
  alignment logic; FIXED tests FIFO-style register interfaces.
- **Address alignment**: transfer address aligned to 64B, 32B, or unaligned (the last stresses
  any split-transaction or byte-enable logic).
- **Outstanding transactions**: how many inflight AXI transactions are permitted simultaneously
  (stresses the read/write ordering queues).
- **Interleaving**: a write transaction that is interrupted by a new read, testing arbitration.

**Transaction-Level Model (TLM) in SystemVerilog:**

```systemverilog
// DMA transaction object
class axi_dma_txn extends uvm_sequence_item;
  `uvm_object_utils(axi_dma_txn)

  // Randomisable fields
  rand logic [63:0] addr;
  rand logic [7:0]  burst_len;    // AXI ARLEN (0 = 1 beat, 255 = 256 beats)
  rand logic [1:0]  burst_type;   // 0=FIXED, 1=INCR, 2=WRAP
  rand logic [2:0]  burst_size;   // AXI ARSIZE (0=1B, 1=2B, ..., 6=64B)
  rand logic [7:0]  data[];       // payload bytes

  // Constraints
  // Align address to transfer size (AXI4 requirement)
  constraint addr_aligned_c {
    addr[5:0] == 6'b0;   // 64-byte aligned for simplicity
  }

  // Bias towards larger bursts (more interesting for throughput testing)
  constraint burst_len_dist_c {
    burst_len dist {
      0        := 5,    //  1 beat  — 5% probability
      [1:7]    := 15,   //  2-8 beats
      [8:15]   := 20,   //  9-16 beats (one cache line)
      [16:63]  := 30,
      [64:255] := 30    // long bursts most common
    };
  }

  // Only INCR bursts for data DMA (WRAP reserved for special cache tests)
  constraint burst_type_c {
    burst_type inside {1, 2};   // INCR or WRAP only
  }

  // Burst size must be power of two and <= bus width (64 bytes = AXI512)
  constraint burst_size_c {
    burst_size inside {[0:6]};
  }

  // Payload size must match burst geometry
  constraint payload_size_c {
    data.size() == (burst_len + 1) * (1 << burst_size);
  }

endclass
```

The sequence generator varies these constraints per test scenario:

```systemverilog
class short_burst_sequence extends uvm_sequence;
  task body();
    axi_dma_txn txn;
    // Override distribution: force short bursts to test pipeline drain
    repeat (100) begin
      txn = axi_dma_txn::type_id::create("txn");
      start_item(txn);
      if (!txn.randomize() with { burst_len inside {[0:3]}; })
        `uvm_fatal("RAND", "Randomisation failed")
      finish_item(txn);
    end
  endtask
endclass
```

---

### Q5. How do you handle floating-point denormals as corner cases in a constrained random test for a BF16 accelerator?

**Answer.**

Denormal (subnormal) numbers have an exponent field of all zeros and a non-zero mantissa. They
represent very small numbers close to zero and are typically handled by "flush to zero" (FTZ)
in hardware accelerators to avoid the significant area and latency penalty of a full subnormal
pipeline.

**Why they matter for verification:**

- If the hardware flushes denormals to zero (FTZ mode), the golden model must also flush to
  zero. If the golden model uses `numpy` or Python `float` (which both support subnormals),
  there will be a systematic mismatch for any input that happens to produce a subnormal.
- If the hardware has a switchable FTZ mode, the verification plan must cover both states.
- Accumulation of many small BF16 values can produce a denormal sum even if individual
  inputs are normal.

**How to generate denormal test inputs:**

```python
import numpy as np
import ml_dtypes
import struct

def make_bf16_denormals(count: int) -> np.ndarray:
    """
    Generate `count` BF16 denormal values.
    BF16 denormals: exponent = 0x00, mantissa != 0.
    Range: ±(1..127) × 2^(-133)
    """
    # BF16 bit pattern: [sign(1)] [exp(8)] [mantissa(7)]
    # Denormal: exp = 0b00000000, mantissa = 0b0000001 to 0b1111111
    mantissas = np.random.randint(1, 128, size=count, dtype=np.uint16)
    signs     = np.random.randint(0, 2,   size=count, dtype=np.uint16) << 15
    bits      = signs | mantissas          # exponent bits are 0 already
    # Reinterpret as bfloat16
    return bits.view(ml_dtypes.bfloat16)

def make_bf16_near_denormal(count: int) -> np.ndarray:
    """
    Generate BF16 values just above the denormal threshold (smallest normal numbers).
    These stress FTZ boundary detection logic.
    Smallest normal BF16: exponent=1, mantissa=0 → 0x0080 → 1.175e-38
    """
    bits = np.full(count, 0x0080, dtype=np.uint16)
    # Add small random perturbation to mantissa
    bits += np.random.randint(0, 4, size=count, dtype=np.uint16)
    return bits.view(ml_dtypes.bfloat16)

def flush_to_zero_bf16(arr: np.ndarray) -> np.ndarray:
    """Apply FTZ: replace denormals with +0."""
    bits  = arr.view(np.uint16)
    exp   = (bits >> 7) & 0xFF
    # Denormal: exponent == 0 and mantissa != 0
    is_denormal = (exp == 0)
    result = bits.copy()
    result[is_denormal] = 0   # replace with +0.0
    return result.view(ml_dtypes.bfloat16)
```

**Directed test strategy:**

1. Create a test with BF16 inputs that are exactly denormal. If the hardware FTZ-es them, the
   output should match the golden model with `flush_to_zero_bf16` applied to all inputs and
   intermediate values.
2. Create a test with inputs chosen so that the sum of K BF16 normal values produces a
   denormal result. Verify the hardware's handling of the underflow case.
3. If the hardware has an FTZ configuration bit, toggle it and verify that behaviour changes
   as specified.

---

### Q6. How do you build a sequence library for constrained random testing of a transformer accelerator's compute core?

**Answer.**

A sequence library organises test scenarios into reusable, composable classes. For a
transformer accelerator, the sequence hierarchy mirrors the operations in a forward pass:

```
BaseAcceleratorSequence
├── GemmSequence
│   ├── SmallGemmSequence        (M,K,N ≤ 64)
│   ├── LargeGemmSequence        (M,K,N = 1024, 2048, 4096)
│   ├── TiledGemmSequence        (dimensions not multiples of tile size)
│   └── BatchedGemmSequence      (batch dimension > 1)
├── AttentionSequence
│   ├── ShortSequenceAttention   (seq_len ≤ 32)
│   ├── LongSequenceAttention    (seq_len = 2048, 4096, 8192)
│   └── CausalMaskAttention      (auto-regressive inference, lower-triangular mask)
├── LayerNormSequence
│   ├── ZeroMeanInputSequence
│   └── SaturatedInputSequence
└── ActivationSequence
    ├── ReLUSequence
    └── GELUSequence
```

**Example in SystemVerilog UVM:**

```systemverilog
// Base class: common randomisable knobs for all GEMM sequences
class gemm_base_seq extends uvm_sequence #(gemm_txn);
  `uvm_object_utils(gemm_base_seq)

  rand int unsigned M;     // output rows
  rand int unsigned K;     // inner dimension
  rand int unsigned N;     // output columns
  rand int unsigned num_transactions;

  // Default constraints: valid range for the DUT
  constraint dimensions_valid_c {
    M inside {[1:4096]};
    K inside {[1:4096]};
    N inside {[1:4096]};
    num_transactions inside {[10:200]};
  }

  // Dimensions must be multiples of the tile size (override in subclass for off-tile tests)
  constraint tile_aligned_c {
    M % 32 == 0;
    K % 32 == 0;
    N % 32 == 0;
  }

  task body();
    gemm_txn txn;
    repeat (num_transactions) begin
      txn = gemm_txn::type_id::create("txn");
      start_item(txn);
      if (!txn.randomize() with {
        txn.M == local::M;
        txn.K == local::K;
        txn.N == local::N;
      }) `uvm_fatal("RAND", "randomize() failed")
      finish_item(txn);
    end
  endtask
endclass

// Subclass: off-tile dimensions to stress boundary handling
class gemm_off_tile_seq extends gemm_base_seq;
  `uvm_object_utils(gemm_off_tile_seq)

  // Override: disable tile alignment constraint
  constraint tile_aligned_c { M % 32 != 0; }  // intentionally misaligned

  constraint off_tile_range_c {
    M inside {[1:31], [33:63], [65:95]};  // just above/below tile boundaries
  }
endclass
```

This structure means a regression suite can instantiate `LargeGemmSequence` for throughput
testing, `GemmOffTileSequence` for boundary testing, and `CausalMaskAttentionSequence` for
auto-regressive mode, all sharing the same base randomisation logic.

---

## Tier 3 — Advanced

### Q7. How do you build a self-checking constrained random test that detects accumulator state leakage between consecutive GEMM operations?

**Answer.**

Accumulator leakage occurs when the hardware fails to reset the accumulator between consecutive
tiled GEMM operations. The accumulator retains a partial sum from the previous computation and
adds it to the first partial product of the new computation. This is a subtle bug that will not
be detected by single-transaction tests.

**Test design principle:**

Structure pairs of consecutive transactions such that the first transaction leaves a known,
non-zero value in the accumulator, and the second transaction has an input that would produce
the same output regardless of whether accumulator leakage occurred — except when the leaked
value is added.

```python
import numpy as np

def generate_leakage_detection_pair(K: int, N: int, rng: np.random.Generator):
    """
    Generate two GEMM transaction pairs that detect accumulator leakage.

    Transaction 1 (T1): produces a known non-zero accumulator state.
    Transaction 2 (T2): designed so its correct output is known analytically,
                        making any leakage from T1 detectable as a deviation.
    """

    # T1: random matrices — leave non-zero state in all accumulator lanes
    A1 = rng.integers(64, 127, size=(1, K), dtype=np.int8)   # all positive, non-trivial
    B1 = rng.integers(64, 127, size=(K, N), dtype=np.int8)

    # T2: A2 is all-zero — correct output is all-zero (or the bias if present)
    # If the accumulator is not cleared, output will contain leakage from T1's result
    A2 = np.zeros((1, K), dtype=np.int8)
    B2 = rng.integers(-127, 127, size=(K, N), dtype=np.int8)

    # Expected output for T2 with zero A2 is exactly 0 (for zero bias, zero zero-point)
    expected_T2 = np.zeros((1, N), dtype=np.int8)

    return (A1, B1), (A2, B2), expected_T2


# In the testbench, run T1 immediately followed by T2 with no idle cycles
# Any non-zero element in the T2 output is evidence of accumulator leakage.
```

**Why this works:**

- The all-zero activation matrix for T2 guarantees that `sum(A2[i,k] * B2[k,j]) = 0` for
  all `i, j`, so the correct output is zero (plus zero-point correction).
- If the hardware does not clear the accumulator before starting T2, the output will contain
  the partial-sum state from T1, which is non-zero by construction.

**Variations to increase coverage of this scenario:**

1. Use maximum back-to-back throughput (no idle cycles between T1 and T2).
2. Vary the tile structure: T1 uses 4 tiles, T2 uses 1 tile (tests the last-tile clear signal).
3. Test with K dimensions that require different numbers of GEMM tiles (3, 4, 5 tiles) to
   stress all possible accumulator clear trigger points.

---

### Q8. Explain how you would use functional coverage feedback to automatically steer a constrained random test towards uncovered bins.

**Answer.**

This technique is called **coverage-directed test generation (CDTG)** and is supported natively
in UVM through coverage callbacks and sequence layer feedback.

**Architecture:**

```
+-------------------+         coverage hit?         +--------------------+
|  Coverage Checker  | -----------------------------> |  Constraint        |
|  (covergroup.iff)  |         bins open             |  Controller        |
|                    | <----------------------------- |  (UVM config DB)   |
+-------------------+                                +--------------------+
                                                              |
                                              tighten constraints
                                                              v
                                                    +------------------+
                                                    |  Sequence Gen.   |
                                                    |  (re-randomises) |
                                                    +------------------+
```

**Implementation outline:**

```systemverilog
class coverage_controller extends uvm_component;
  // Reference to the coverage model
  accel_coverage_model cov_model;

  // Called by the coverage model whenever a new sample is taken
  function void on_sample();
    // Check for open bins and adjust the UVM config DB
    if (!cov_model.cp_precision.BF16.is_covered()) begin
      uvm_config_db #(precision_t)::set(null, "*.seq", "force_precision", BF16);
    end
    if (!cov_model.cx_dim_precision.is_covered()) begin
      // Cross coverage: force large dimensions + BF16 simultaneously
      uvm_config_db #(bit)::set(null, "*.seq", "force_large_dim_bf16", 1);
    end
  endfunction
endclass
```

The sequence reads from the config DB at the start of each `body()` call:

```systemverilog
task body();
  precision_t forced_prec;
  if (uvm_config_db #(precision_t)::get(this, "", "force_precision", forced_prec))
    // Use forced_prec instead of randomising it
    txn.randomize() with { txn.precision == forced_prec; };
  else
    txn.randomize();
endtask
```

**Practical considerations:**

- This approach requires that coverpoints be sampled frequently (every transaction, not just
  at the end of each test).
- The feedback loop should have hysteresis: do not change constraints on every missed bin
  because natural randomness may close it in the next few transactions.
- Document which bins required directed forcing vs. random hit, as this gives signal about
  the difficulty of naturally exercising those conditions.
- Tools like Cadence vManager and Siemens Questa CVT automate this feedback loop, providing
  a GUI that shows which bins are open and generating seeds that target them.
