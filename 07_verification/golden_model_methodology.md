# Golden Model Methodology for LLM Accelerator Verification

## Overview

A golden model (also called a reference model) is a trusted, independent implementation of
the computation that your hardware is supposed to perform. For LLM accelerators, this typically
means a Python/NumPy model that produces the expected output for every test stimulus. The RTL
output is then compared against the golden model output to determine pass or fail.

This document covers the theory and practice of golden model verification for LLM accelerators,
organised from foundational concepts up to the kind of nuanced questions asked at senior-level
interviews.

---

## Tier 1 — Fundamentals

### Q1. What is a golden model and why is it essential for verifying an LLM accelerator?

**Answer.**

A golden model is a software reference implementation that computes the same function as the
hardware under test (HUT). It is the source of truth against which RTL simulation outputs are
compared.

For LLM accelerators the golden model is essential for several reasons:

1. **Algorithmic complexity.** Operations such as matrix multiplication, softmax, layer
   normalisation, and attention are mathematically involved. It is impractical for a testbench
   engineer to hand-calculate expected outputs for non-trivial tensor dimensions.

2. **Regression baseline.** The golden model captures the intended hardware behaviour once,
   permanently. Any future RTL change that changes the output relative to the golden model is
   immediately flagged as a regression.

3. **Coverage of a vast input space.** LLM accelerators operate over matrices with millions
   of elements. Constrained-random test generation feeds thousands of random stimuli into both
   the golden model and the RTL simultaneously; mismatches expose bugs automatically.

4. **Debugging aid.** When a mismatch occurs, the golden model output can be inspected
   instruction-by-instruction (or cycle-by-cycle when co-simulating) to identify where the
   RTL diverges.

**Common mistake:** Confusing a floating-point Python model with a *bit-accurate* model. A
naive NumPy model uses IEEE 754 double precision and gives mathematically correct results, but
will not match INT8 or BF16 hardware because the rounding and overflow behaviour differ. A true
golden model must replicate the exact numerical semantics of the hardware.

---

### Q2. What is meant by "bit-accurate" and why does it matter?

**Answer.**

Bit-accurate means the golden model produces output that is identical, bit-for-bit, to what
the hardware produces for every possible legal input. This is a stronger guarantee than merely
being "numerically close."

It matters because:

- **No tolerance ambiguity.** If you allow a tolerance (e.g. ±1 LSB) you may mask real bugs.
  Accumulated ±1 errors across a 2048-element dot product can amount to a meaningful numeric
  error in the final result.
- **Interoperability.** Downstream layers expect specific quantised values. A bias of even 1
  ULP in an intermediate activation can change the argmax of the final logit distribution.
- **Certification and reproducibility.** Production silicon must match the model used during
  training calibration. Bit-accuracy guarantees that inference on the chip is identical to
  the software model used to generate scale factors.

**When bit-accuracy is difficult to achieve.** Floating-point addition is not associative.
Hardware may reorder partial sums in a tree adder, while Python evaluates left-to-right. In
those cases the acceptable practice is to document the exact accumulation order in the micro-
architecture spec and encode that same order into the golden model, or to accept ULP-bounded
tolerances with clear justification.

---

### Q3. Walk me through the basic structure of a golden model regression flow.

**Answer.**

```
+------------------+        stimulus        +-------------------+
|  Stimulus        |  --------------------> |  RTL Simulation   |
|  Generator       |                        |  (EDA tool)       |
+------------------+         |              +-------------------+
                              |                      |
                              v                      v
                    +------------------+    +------------------+
                    |  Golden Model    |    |  RTL Output      |
                    |  (Python)        |    |  (waveform /     |
                    |                  |    |   log)           |
                    +------------------+    +------------------+
                              |                      |
                              v                      v
                    +------------------------------------------+
                    |            Comparator / Scoreboard       |
                    |  Pass: outputs match within tolerance     |
                    |  Fail: log mismatch + dump debug info     |
                    +------------------------------------------+
```

Key steps:

1. **Generate stimulus.** Random or directed test vectors are created (tensor shapes, data
   values, quantisation parameters).
2. **Run golden model.** The Python reference processes the stimulus and stores expected
   outputs.
3. **Drive RTL.** The same stimulus is applied to the RTL simulation via a testbench driver
   (SystemVerilog, cocotb, or UVM).
4. **Capture RTL output.** The monitor collects output transactions from the DUT.
5. **Compare.** The scoreboard compares RTL output against golden model output, applying the
   agreed tolerance policy.
6. **Report.** Pass/fail, coverage, and any mismatch details are written to a report file.

---

### Q4. What Python libraries are commonly used to build golden models for matrix operations?

**Answer.**

| Library | Role |
|---------|------|
| `numpy` | Dense tensor arithmetic, broadcasting, axis reductions |
| `scipy` | Special functions (softmax, GELU reference) |
| `struct` / `ctypes` | Bit manipulation to match hardware packing/unpacking |
| `ml_dtypes` | Google's library for BF16 and FP8 types that match hardware semantics |
| `torch` (CPU, `dtype` explicit) | Useful for end-to-end model-level reference; use `torch.int8`, `torch.bfloat16` explicitly |
| `cocotb` | Python-based co-simulation framework; the golden model runs inside the simulator process |

For INT8 accelerators, `numpy` with `dtype=np.int8` / `dtype=np.int32` is usually sufficient.
For BF16 or FP8 accelerators, `ml_dtypes` or explicit bit-manipulation via `struct.pack` is
required because standard Python `float` is 64-bit.

---

## Tier 2 — Intermediate

### Q5. How do you handle floating-point non-associativity between the golden model and the RTL?

**Answer.**

Floating-point addition is not associative: `(a + b) + c` may differ from `a + (b + c)` in
the last one or two bits due to intermediate rounding. This is a fundamental property of IEEE
754 and cannot be eliminated.

Hardware accelerators often accumulate partial sums in a different order than sequential Python
code:

- A systolic array may produce partial sums from multiple PEs simultaneously.
- A tree reduction adds pairs of elements in log2(N) stages.
- Pipelined accumulators may use a different carry-save format internally.

**Strategies to handle this:**

1. **Model the exact hardware accumulation order.** Read the micro-architecture spec. If the
   hardware accumulates in a tree of depth 4, write the Python model to do the same grouping.
   This is the gold standard but requires close collaboration with the hardware designer.

   ```python
   def tree_sum(values):
       """Mirrors a binary-tree reduction in hardware."""
       arr = list(values)
       while len(arr) > 1:
           # Pair up and sum, matching the hardware tree structure
           arr = [arr[i] + arr[i+1] if i+1 < len(arr) else arr[i]
                  for i in range(0, len(arr), 2)]
       return arr[0]
   ```

2. **Use ULP-based tolerance.** Accept a result as correct if it differs from the golden
   model by at most K ULPs (units in the last place). K=1 or K=2 is defensible for a single
   FP32 MAC. For accumulated operations over N terms, K scales as O(log2 N).

3. **Use integer arithmetic end-to-end.** For INT8 accelerators the accumulation is exact
   integer arithmetic (no rounding until requantisation). Non-associativity is not an issue
   for integer types; the golden model will be bit-accurate without special handling.

4. **Document the tolerance with justification.** Any tolerance that is not zero must be
   documented in the verification plan, with a mathematical justification and sign-off from
   the algorithm team.

**Common interview mistake:** Saying "just use `np.allclose` with `rtol=1e-5`." This is not
a principled answer. The tolerance must be derived from the hardware's numerical specification,
not chosen to make tests pass.

---

### Q6. Describe a tolerance-based comparison strategy for a BF16 matrix multiplication output.

**Answer.**

BF16 has 8 bits of exponent and 7 bits of mantissa (vs FP32's 23). Each rounding step
introduces an error of at most 0.5 ULP relative to the true real-valued result.

For a dot product of length K in BF16:

- Each multiply introduces ≤ 0.5 ULP relative error.
- Each of K-1 additions introduces ≤ 0.5 ULP relative error.
- In the worst case, errors do not cancel; the total relative error is bounded by
  approximately K × machine_epsilon_bf16 / 2 = K × 2^(-8).

For K = 512 that is 512 × 0.004 = 2.0 relative error in the worst case (very conservative).
In practice, errors are random and partially cancel; a tighter empirical bound is used.

**Practical comparison approach:**

```python
import numpy as np
import ml_dtypes

def compare_bf16_matmul(rtl_output, golden_output, K, fail_threshold_ulps=4):
    """
    Compare two BF16 matmul results.
    K: inner dimension of the matmul (dot product length).
    fail_threshold_ulps: maximum acceptable ULP difference per element.
    """
    # Reinterpret both as uint16 to count ULPs
    rtl_u16    = rtl_output.view(np.uint16)
    golden_u16 = golden_output.view(np.uint16)

    # ULP distance (handle sign bit: two's complement for same-sign numbers)
    ulp_diff = np.abs(rtl_u16.astype(np.int32) - golden_u16.astype(np.int32))

    max_ulp   = int(ulp_diff.max())
    mean_ulp  = float(ulp_diff.mean())

    passed = max_ulp <= fail_threshold_ulps
    return passed, max_ulp, mean_ulp
```

The `fail_threshold_ulps` value must be agreed with the algorithm and architecture teams and
documented in the verification plan. A value of 1 or 2 is appropriate when the accumulation
order is exactly matched; 4–8 is acceptable when accumulation order may differ but the
hardware uses the same precision as the model.

---

### Q7. How do you structure a regression test suite around a golden model?

**Answer.**

A well-structured regression suite has four layers:

**Layer 1 — Directed sanity tests.**
A small set of hand-crafted inputs with analytically known outputs (identity matrices, all-
zeros, all-ones, single-hot vectors). These run in seconds and give immediate confidence that
the basic datapath is alive.

**Layer 2 — Directed corner-case tests.**
Inputs designed to stress specific hardware behaviours: maximum positive value, minimum
negative value, overflow conditions, saturation, NaN propagation, denormal inputs. These
should be in the regression from day one of integration testing.

**Layer 3 — Constrained-random tests.**
Random matrices generated with controlled properties (specific shapes, value ranges,
quantisation parameters). Seeds are fixed and stored so the suite is deterministic and
reproducible. The golden model is evaluated for each random stimulus, and the RTL output
is compared.

**Layer 4 — Real-model excerpts.**
Actual weight matrices and activation tensors extracted from a trained LLM (e.g. a single
transformer layer from LLaMA or BERT). These are the highest-fidelity tests and are
especially valuable for quantisation accuracy.

**File organisation:**

```
tests/
  golden/
    __init__.py
    matmul.py          # golden model for GEMM
    softmax.py         # golden model for softmax
    layernorm.py       # golden model for LayerNorm
  directed/
    test_zeros.py
    test_overflow.py
  random/
    test_random_gemm.py   # seed-controlled random tests
  model_excerpts/
    test_llama_layer0.py
  conftest.py            # pytest fixtures, tolerance config
```

Regression is run on every RTL commit via CI. Failed tests output a human-readable diff
showing the stimulus, expected value, actual value, and ULP error for each failing element.

---

### Q8. What is co-simulation and how does it integrate a Python golden model with an RTL simulator?

**Answer.**

Co-simulation means the Python golden model and the RTL simulator run simultaneously in the
same process, sharing state and synchronising at transaction boundaries.

**cocotb** is the dominant open-source framework for this. It runs inside the RTL simulator
(Verilator, Questa, Xcelium, etc.) and exposes a Python coroutine API to drive and monitor
DUT signals. The golden model is just a Python function called from the same coroutine.

```
+-----------------------------+
| Simulator process           |
|  +------------------------+ |
|  | RTL DUT                | |
|  +------------------------+ |
|       ^         |           |
|  drive|         |monitor    |
|       |         v           |
|  +------------------------+ |
|  | cocotb Python TB       | |
|  |  - Driver coroutine    | |
|  |  - Monitor coroutine   | |
|  |  - Scoreboard          | |
|  |  - Golden model call   | |
|  +------------------------+ |
+-----------------------------+
```

At each transaction boundary:

1. The driver applies input tensor data to the DUT input ports.
2. `await RisingEdge(dut.clk)` advances simulation until the output is valid.
3. The monitor reads the DUT output registers.
4. The scoreboard calls `golden_model(input)` and compares the result.
5. Pass or fail is logged immediately.

This tight integration means every random test automatically generates its expected output from
the golden model without any separate offline step.

---

## Tier 3 — Advanced

### Q9. Your INT8 GEMM golden model passes all directed tests but fails 0.1% of random tests with off-by-one errors in the output. What are the most likely causes and how do you diagnose them?

**Answer.**

An off-by-one failure rate of 0.1% in an INT8 GEMM almost certainly comes from a rounding
mode mismatch. The diagnosis and triage process:

**Step 1 — Isolate the failing cases.**
Log the full stimulus (A matrix, B matrix, scale factors, zero points) for every failing test.
Reduce to the smallest failing input — ideally a single output element that is wrong.

**Step 2 — Check the accumulation overflow path.**
INT8 × INT8 → INT16 is the standard MAD, but hardware often accumulates into INT32 to avoid
intermediate overflow. Confirm the golden model uses `np.int32` for the accumulator and does
not accidentally truncate to `np.int16` mid-accumulation.

```python
# Wrong — can overflow for large K
acc = np.dot(a.astype(np.int16), b.astype(np.int16))

# Correct — accumulate in INT32
acc = np.einsum('ij,jk->ik', a.astype(np.int32), b.astype(np.int32))
```

**Step 3 — Examine the requantisation step.**
The most common source of off-by-one is the rounding mode applied when converting the INT32
accumulator back to INT8. Hardware often uses "round half away from zero" (RHAFZ) or "round
half to even" (banker's rounding). Python's built-in `round()` uses RHAFZ for positive
fractions but Python 3's `round()` actually uses banker's rounding, which can differ. NumPy
uses "round half to even" by default.

```python
def round_half_away_from_zero(x):
    """Matches common RTL rounding: ties go away from zero."""
    import numpy as np
    return np.sign(x) * np.floor(np.abs(x) + 0.5).astype(np.int32)
```

If the RTL uses RHAFZ and the golden model uses banker's rounding, exactly the 0.1% of values
that fall on a 0.5 boundary will differ by 1.

**Step 4 — Check the saturation clamp.**
After requantisation, values outside [-128, 127] must be clamped. Confirm both the RTL and
the golden model apply the same clamp — some implementations saturate before rounding, some
after; the order can produce a difference of 1 LSB.

**Step 5 — Check the zero-point subtraction order.**
Quantised GEMM is `Y = clip(round((X - zp_x) * (W - zp_w) * scale + zp_y))`. If zero-point
subtraction happens after accumulation rather than before, results can differ when intermediate
values overflow.

---

### Q10. How would you build a golden model for attention (QKV + scaled dot-product attention + softmax) in a way that is both numerically accurate and maintainable?

**Answer.**

Attention introduces three additional numerical challenges beyond GEMM: the scaling by
1/sqrt(d_k), the softmax (which involves an exp and a division), and the accumulation of
softmax weights multiplied by V.

**Recommended approach — layered reference functions:**

```python
import numpy as np

def quantised_matmul_int8(A, B, scale_a, zp_a, scale_b, zp_b, scale_out, zp_out):
    """Bit-accurate INT8 GEMM. Returns INT8 result."""
    # De-quantise to float32 for the multiply (matches hardware's effective computation)
    A_dq = (A.astype(np.int32) - zp_a) * scale_a
    B_dq = (B.astype(np.int32) - zp_b) * scale_b
    C_fp = A_dq @ B_dq
    # Requantise
    C_q  = np.round(C_fp / scale_out).astype(np.int32) + zp_out
    return np.clip(C_q, -128, 127).astype(np.int8)

def softmax_fp32(logits):
    """Numerically stable softmax (subtract max before exp)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, W_q, W_k, W_v, quant_params):
    """
    Full attention reference.
    quant_params: dict of scale/zero-point for each GEMM.
    """
    d_k = Q.shape[-1]

    # QK^T in INT8
    logits = quantised_matmul_int8(Q, W_k.T, **quant_params['qk'])
    # Scale by 1/sqrt(d_k) — applied in FP32 after dequantisation
    logits_fp = logits.astype(np.float32) * (1.0 / np.sqrt(d_k))
    # Softmax in FP32
    attn_weights = softmax_fp32(logits_fp)
    # AV in INT8 (requantise attn_weights back to INT8 first)
    attn_q = quantise_to_int8(attn_weights, **quant_params['attn'])
    output  = quantised_matmul_int8(attn_q, W_v, **quant_params['av'])
    return output
```

**Maintainability principles:**

1. Each sub-operation (GEMM, softmax, scaling) is a separate function with its own unit tests.
2. Quantisation parameters are passed explicitly, not embedded as magic numbers.
3. The golden model is versioned alongside the RTL in the same repository — when the hardware
   spec changes, both are updated together and the change is code-reviewed.
4. An end-to-end test compares the full attention output against a PyTorch reference with
   explicit dtypes to catch parameter drift.

---

### Q11. Explain how you would use a golden model to debug a timing closure regression that changes RTL behaviour.

**Answer.**

Timing closure changes (synthesis re-runs, retiming, pipeline stage insertion) should not
change functional behaviour, but in practice they occasionally introduce bugs:

- A retimer may move a register to a position where a combinational expression evaluates
  before its operand is updated, creating an off-by-one cycle read of stale data.
- A multi-cycle path exception may be applied incorrectly, causing a register to capture
  data too early.
- A last-minute gate-level optimisation may change the behaviour of a partial-product tree
  in a multiplier.

**Debug flow using the golden model:**

1. **Reproduce with the original RTL.** Run the failing random test on the pre-timing-closure
   netlist. If it passes there, the regression is confirmed to be a timing-closure artefact.

2. **Binary search on stimulus complexity.** Reduce the input to the smallest matrix size
   that still fails. A 4×4 GEMM is easier to trace than a 256×256 one.

3. **Wave-diff the golden model output against RTL.** For each output element that differs,
   trace back to the input elements that contribute (the row of A and the column of B). Add
   probes to the accumulator tree nodes in the waveform.

4. **Check the pipeline latency.** The golden model assumes a specific pipeline latency (e.g.
   N cycles from input valid to output valid). A retimed design may have a different latency.
   Verify the testbench is reading the output at the correct cycle.

5. **Formal equivalence check.** Run a formal tool (Cadence Conformal, Synopsys Formality) to
   prove or disprove equivalence between the pre- and post-synthesis netlists. A formal
   counterexample directly identifies the logic cone that changed.

The golden model does not directly fix the RTL bug, but it provides the unambiguous expected
value that the failing element should have produced, which is the starting point for any
waveform-based root-cause analysis.

---

### Q12. How would you extend a golden model to support multiple quantisation schemes (INT8, INT4, FP8-E4M3, BF16) without duplicating code?

**Answer.**

The key insight is to separate the *numerical semantics* (what arithmetic the hardware
performs) from the *data representation* (how values are stored and transmitted). Both can be
parameterised.

**Design pattern — Quantised Tensor type:**

```python
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class QuantConfig:
    """Encapsulates all quantisation parameters for one tensor."""
    dtype:        np.dtype        # e.g. np.int8, np.float16
    scale:        float
    zero_point:   int             # 0 for symmetric, non-zero for asymmetric
    clamp_min:    float
    clamp_max:    float
    round_fn:     Callable        # rounding mode, e.g. round_half_away_from_zero

    def quantise(self, x_fp32: np.ndarray) -> np.ndarray:
        """Float32 -> quantised representation."""
        x_scaled = x_fp32 / self.scale + self.zero_point
        x_rounded = self.round_fn(x_scaled)
        x_clamped = np.clip(x_rounded, self.clamp_min, self.clamp_max)
        return x_clamped.astype(self.dtype)

    def dequantise(self, x_q: np.ndarray) -> np.ndarray:
        """Quantised -> float32 (for display and comparison only)."""
        return (x_q.astype(np.float32) - self.zero_point) * self.scale

# Pre-built configs for standard formats
INT8_SYMMETRIC  = QuantConfig(np.int8,    scale=1.0, zero_point=0,
                               clamp_min=-128, clamp_max=127,
                               round_fn=round_half_away_from_zero)
INT4_SYMMETRIC  = QuantConfig(np.int8,    scale=1.0, zero_point=0,
                               clamp_min=-8,   clamp_max=7,
                               round_fn=round_half_away_from_zero)
BF16_CONFIG     = QuantConfig(ml_dtypes.bfloat16, scale=1.0, zero_point=0,
                               clamp_min=-3.38e38, clamp_max=3.38e38,
                               round_fn=lambda x: x.astype(ml_dtypes.bfloat16))

def generic_matmul_golden(A, B, cfg_a: QuantConfig, cfg_b: QuantConfig,
                           cfg_out: QuantConfig) -> np.ndarray:
    """
    Single golden model function that works for any combination of
    quantisation configs. The hardware always accumulates in FP64
    (or INT64 for integer types) to avoid precision loss in this reference.
    """
    A_dq  = cfg_a.dequantise(A).astype(np.float64)
    B_dq  = cfg_b.dequantise(B).astype(np.float64)
    C_fp  = A_dq @ B_dq
    return cfg_out.quantise(C_fp.astype(np.float32))
```

This structure means:
- Adding a new format (e.g. FP8-E5M2) requires creating one new `QuantConfig` object.
- The GEMM logic is written once.
- Each config object has its own unit tests verifying its quantise/dequantise cycle.
- The RTL verification team and the algorithm team share the same config definitions,
  ensuring the hardware spec and the software spec are always synchronised.
