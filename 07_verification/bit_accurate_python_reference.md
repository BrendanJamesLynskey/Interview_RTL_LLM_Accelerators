# Bit-Accurate Python Reference Models

## Overview

Building a Python reference model that is bit-accurate to RTL hardware is one of the most
technically demanding tasks in digital verification. It requires understanding both the
hardware's numerical format and the subtle ways Python/NumPy deviate from hardware arithmetic.
This document covers the full range of interview questions on this topic.

---

## Tier 1 — Fundamentals

### Q1. What is fixed-point arithmetic and how do you implement it correctly in Python?

**Answer.**

Fixed-point arithmetic represents numbers as integers, with a programmer-specified implicit
binary point. For example, a Q4.4 format has 4 integer bits and 4 fractional bits; the value
`0b00011010` represents `1.625` (decimal) because the implicit point is between bit 3 and bit 4.

Python integers have arbitrary precision (no overflow), which is both a benefit and a pitfall.
You must explicitly enforce the word width that the hardware uses.

**Key implementation rules:**

1. Do all arithmetic in Python `int` (arbitrary precision) to avoid NumPy type coercion errors.
2. After each operation, apply a mask to truncate to the hardware word width.
3. If the number is signed, apply sign extension from the MSB position.

```python
def fixed_point_add(a: int, b: int, width: int, signed: bool = True) -> int:
    """
    Add two fixed-point integers and return the result truncated to `width` bits.
    Wraps on overflow (matches default RTL behaviour; use saturate() for saturation).

    Parameters
    ----------
    a, b   : integer values (bit patterns, not scaled floats)
    width  : hardware word width in bits
    signed : if True, interpret the MSB as a sign bit

    Returns
    -------
    int : result truncated to `width` bits, sign-extended if signed=True
    """
    mask = (1 << width) - 1          # e.g. 0xFF for 8-bit
    result = (a + b) & mask           # truncate to word width (wrap on overflow)

    if signed:
        # Sign-extend: if MSB is set, the value is negative
        sign_bit = 1 << (width - 1)
        if result & sign_bit:
            result -= (1 << width)    # convert to Python negative int

    return result


def fixed_point_mul(a: int, b: int, a_frac: int, b_frac: int,
                    out_width: int, out_frac: int, signed: bool = True) -> int:
    """
    Multiply two fixed-point numbers and return a truncated result.

    a_frac, b_frac : number of fractional bits in each operand
    out_frac       : number of fractional bits in the output
    """
    # Full-precision multiply (no bits lost yet)
    full_product = a * b                         # 2*width bits, 2*frac fractional bits
    total_frac   = a_frac + b_frac

    # Shift to align with output fractional position
    shift = total_frac - out_frac
    if shift > 0:
        # Arithmetic right shift — discard LSBs (truncation)
        # To match RTL "round half away from zero", add 0.5 before truncating
        rounding_bit = 1 << (shift - 1)
        full_product += rounding_bit             # round
        full_product >>= shift
    elif shift < 0:
        full_product <<= (-shift)

    # Truncate to output width
    mask = (1 << out_width) - 1
    result = full_product & mask

    if signed:
        sign_bit = 1 << (out_width - 1)
        if result & sign_bit:
            result -= (1 << out_width)

    return result
```

**Common mistake:** Using `numpy.int8 * numpy.int8` and expecting INT16 output. NumPy returns
INT8 (truncated to 8 bits), silently discarding the overflow. Always cast operands to a wider
type before multiplying:

```python
# Dangerous: result silently overflows INT8
bad  = np.int8(100) * np.int8(100)          # → -112 (wraps)

# Correct: widen first, then multiply
good = np.int16(100) * np.int16(100)        # → 10000
```

---

### Q2. How do you represent and manipulate BF16 values in Python for a golden model?

**Answer.**

Python's built-in `float` is IEEE 754 double (64-bit). NumPy's `float16` is IEEE 754 half
(5-bit exponent, 10-bit mantissa). Neither is BF16 (8-bit exponent, 7-bit mantissa).

**Option 1 — `ml_dtypes` library (recommended).**

```python
import numpy as np
import ml_dtypes

# Create a BF16 array
a = np.array([1.5, -2.25, 0.125], dtype=ml_dtypes.bfloat16)
b = np.array([2.0,  1.0,  4.0  ], dtype=ml_dtypes.bfloat16)

# Arithmetic is performed in BF16 precision (matches hardware)
c = a * b   # dtype remains bfloat16

print(c)          # [3.    -2.25   0.5  ]
print(c.dtype)    # bfloat16
```

**Option 2 — Manual bit manipulation via `struct`.**

When `ml_dtypes` is not available, or when you need to inspect individual fields:

```python
import struct

def float32_to_bf16_bits(f32: float) -> int:
    """
    Truncate a float32 to BF16 by dropping the low 16 mantissa bits.
    This matches the hardware's default rounding mode (round-to-nearest-even
    in most implementations; here we use truncation / round-to-zero for clarity).
    """
    bits_32 = struct.unpack('I', struct.pack('f', f32))[0]  # uint32 bit pattern
    # BF16 keeps the high 16 bits of the 32-bit representation
    bf16_bits = (bits_32 >> 16) & 0xFFFF
    return bf16_bits

def bf16_bits_to_float32(bf16: int) -> float:
    """Expand BF16 bit pattern back to float32 (zero-extend mantissa bits)."""
    bits_32 = (bf16 & 0xFFFF) << 16
    return struct.unpack('f', struct.pack('I', bits_32))[0]

# Example: round-trip
original = 3.14159
bf16     = float32_to_bf16_bits(original)
restored = bf16_bits_to_float32(bf16)
print(f"{original:.6f} -> BF16 0x{bf16:04X} -> {restored:.6f}")
# 3.141590 -> BF16 0x4049 -> 3.140625
```

The difference (3.141590 vs 3.140625) is the quantisation error introduced by BF16. A golden
model must work with the quantised value, not the original float.

---

### Q3. What is overflow and saturation in fixed-point hardware, and how do you model both in Python?

**Answer.**

**Overflow** occurs when an arithmetic result exceeds the representable range of the output
type. Hardware can respond in two ways:

- **Wrap-around (modular overflow):** The result wraps modulo 2^N. This is the C default for
  unsigned integers and the hardware default for most adders. An 8-bit result of 130 becomes
  -126 in signed interpretation.
- **Saturation:** The result is clamped to the maximum (or minimum) representable value.
  128 in INT8 saturates to 127. This is used in signal processing and some neural network
  quantisation layers to prevent wild swings.

```python
import numpy as np

INT8_MIN = -128
INT8_MAX =  127

def saturate_int8(value: int) -> int:
    """Clamp to INT8 range — matches hardware saturation arithmetic."""
    return max(INT8_MIN, min(INT8_MAX, value))

def wrap_int8(value: int) -> int:
    """Wrap modulo 256 to INT8 range — matches hardware two's-complement wrap."""
    value = value & 0xFF           # keep only low 8 bits
    if value >= 128:
        value -= 256               # interpret as signed
    return value

# NumPy vectorised versions
def saturate_array_int8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, INT8_MIN, INT8_MAX).astype(np.int8)

def wrap_array_int8(arr: np.ndarray) -> np.ndarray:
    # Casting to int8 in NumPy naturally wraps (two's-complement truncation)
    return arr.astype(np.int8)

# Demonstration
values = np.array([100, 127, 128, 200, -129, -200], dtype=np.int32)
print("Saturated:", saturate_array_int8(values))
# Saturated: [100, 127, 127, 127, -128, -128]
print("Wrapped:  ", wrap_array_int8(values))
# Wrapped:   [100, 127, -128, -56, 127,  56]
```

**Interview tip:** Always ask the hardware designer which mode the accumulator uses. Many LLM
accelerators use wrap-around for the INT32 accumulator (because overflow should never occur
with properly calibrated INT8 weights and activations) and saturation only on the final
requantisation step that converts back to INT8.

---

## Tier 2 — Intermediate

### Q4. How do you match RTL rounding modes in your Python golden model?

**Answer.**

There are five common rounding modes in digital hardware. The golden model must use the same
mode as the RTL for bit-accuracy.

| Mode | Description | Ties resolved by |
|------|-------------|-----------------|
| Round half away from zero (RHAFZ) | Classic "school" rounding | Away from zero |
| Round half to even (banker's rounding) | IEEE 754 default | Towards nearest even integer |
| Round half to odd | Rare, used in some DSPs | Towards nearest odd integer |
| Truncation (round towards negative infinity) | Discard fractional bits | Always down |
| Round towards zero (RTZ) | Discard fractional bits with sign correction | Always towards zero |

**Python implementations:**

```python
import numpy as np
import math

def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """
    RHAFZ: matches most fixed-point DSP hardware.
    0.5 rounds to 1, -0.5 rounds to -1.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def round_half_to_even(x: np.ndarray) -> np.ndarray:
    """
    Banker's rounding: matches IEEE 754 default and Python's built-in round().
    0.5 rounds to 0 (even), 1.5 rounds to 2 (even).
    """
    return np.round(x)   # NumPy uses this mode by default

def truncate(x: np.ndarray) -> np.ndarray:
    """
    Truncation: discard fractional bits.
    Equivalent to floor for positive, ceil for negative.
    RTL shift-right without rounding bit.
    """
    return np.trunc(x)

def round_towards_zero(x: np.ndarray) -> np.ndarray:
    """
    RTZ: same as truncation for positive numbers; same as ceil for negative.
    Common in FP hardware for intermediate products.
    """
    return np.fix(x)   # np.fix is RTZ

# --- Verification: show where modes differ ---
test_values = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
print("Values   :", test_values)
print("RHAFZ    :", round_half_away_from_zero(test_values))
print("Banker's :", round_half_to_even(test_values))
print("Truncate :", truncate(test_values))
print("RTZ      :", round_towards_zero(test_values))
```

**How to determine which mode the RTL uses:**

1. Read the micro-architecture specification. It must state the rounding mode.
2. Write a directed test with an input whose quantised value lands exactly on a 0.5 boundary.
   Apply both RHAFZ and banker's rounding and check which output the RTL produces.
3. Automate this as a directed test in the regression suite.

---

### Q5. How do you build a quantisation-aware golden model for a complete GEMM with per-channel scale factors?

**Answer.**

Per-channel (per-output-channel) quantisation uses a different scale factor for each row of
the weight matrix. This is the standard in transformer inference engines (e.g. INT8 with
per-channel weight scales).

The mathematical formula is:
`Y[i,j] = clip(round(sum_k(A[i,k] * W[k,j]) * inv_scale_y / (scale_a * scale_w[j])) + zp_y)`

where `scale_w[j]` is the per-channel scale for output channel `j`.

```python
import numpy as np

def round_half_away_from_zero(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def gemm_per_channel_int8(
    A:         np.ndarray,   # [M, K] INT8 activations
    W:         np.ndarray,   # [K, N] INT8 weights
    scale_a:   float,        # activation scale (scalar, per-tensor)
    zp_a:      int,          # activation zero point
    scale_w:   np.ndarray,   # [N] weight scales (per output channel)
    zp_w:      int,          # weight zero point (usually 0 for symmetric)
    scale_y:   float,        # output scale
    zp_y:      int,          # output zero point
) -> np.ndarray:             # [M, N] INT8 output
    """
    Bit-accurate INT8 GEMM with per-channel weight quantisation.
    Accumulator is INT32. Requantisation uses RHAFZ rounding.
    """
    # Step 1: accumulate in INT32 (avoids overflow for K up to ~2^16)
    # Subtract zero points before accumulation (standard quantised matmul)
    A_shifted = A.astype(np.int32) - int(zp_a)   # [M, K]
    W_shifted = W.astype(np.int32) - int(zp_w)   # [K, N]

    accumulator = A_shifted @ W_shifted            # [M, N] INT32

    # Step 2: apply per-channel effective scale
    # effective_scale[j] = (scale_a * scale_w[j]) / scale_y
    effective_scale = (scale_a * scale_w) / scale_y   # [N] broadcast over M

    # Step 3: scale the accumulator (result is floating-point)
    Y_fp = accumulator.astype(np.float64) * effective_scale   # [M, N]

    # Step 4: add output zero point, round, and clamp to INT8
    Y_fp += zp_y
    Y_rounded = round_half_away_from_zero(Y_fp).astype(np.int32)
    Y_int8    = np.clip(Y_rounded, -128, 127).astype(np.int8)

    return Y_int8


# --- Smoke test ---
if __name__ == "__main__":
    np.random.seed(42)
    M, K, N = 4, 8, 4

    A = np.random.randint(-10, 10, (M, K), dtype=np.int8)
    W = np.random.randint(-10, 10, (K, N), dtype=np.int8)

    # Realistic quantisation parameters
    scale_a = 0.05
    zp_a    = 0
    scale_w = np.array([0.02, 0.03, 0.015, 0.025])
    zp_w    = 0
    scale_y = 0.1
    zp_y    = 0

    result = gemm_per_channel_int8(A, W, scale_a, zp_a, scale_w, zp_w, scale_y, zp_y)
    print("Output shape:", result.shape)
    print("Output dtype:", result.dtype)
    print("Output:\n", result)
```

**Key correctness points:**

1. Zero-point subtraction must happen before the dot product (not after) to match the
   standard quantised matmul identity.
2. The accumulator must be `int32` to accommodate up to K=4096 without overflow when inputs
   are in range [-128, 127].
3. Per-channel scale is applied after the full accumulation, not per partial product.

---

### Q6. How do you use `ctypes` to call RTL simulation outputs from a Python golden model (or vice versa)?

**Answer.**

`ctypes` is the standard library for calling compiled C/C++ code from Python. In a
verification context it has two primary uses:

1. **Calling a C golden model from Python** — when the reference is implemented in C (e.g.
   ported from a firmware library), wrap it with ctypes so it can be called from a cocotb
   testbench.

2. **Calling Python from SystemVerilog via DPI-C** — the simulator calls a C wrapper, which
   in turn calls Python via the C Python API. This allows a Python golden model to be called
   directly from a UVM scoreboard.

**Example: wrapping a C fixed-point MAC in Python via ctypes**

```c
// golden_mac.c — compiled to golden_mac.so
#include <stdint.h>

// INT8 MAC with INT32 accumulator, matching RTL rounding
int32_t int8_mac(const int8_t *a, const int8_t *b, int length) {
    int32_t acc = 0;
    for (int i = 0; i < length; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
}
```

```python
# golden_mac_wrapper.py
import ctypes
import numpy as np
import subprocess, pathlib

# Compile the C library if not already compiled
lib_path = pathlib.Path("/tmp/golden_mac.so")
if not lib_path.exists():
    subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-o", str(lib_path), "golden_mac.c"],
        check=True
    )

lib = ctypes.CDLL(str(lib_path))

# Declare the function signature (required — ctypes does not read headers)
lib.int8_mac.restype  = ctypes.c_int32
lib.int8_mac.argtypes = [
    ctypes.POINTER(ctypes.c_int8),   # const int8_t *a
    ctypes.POINTER(ctypes.c_int8),   # const int8_t *b
    ctypes.c_int,                     # int length
]

def golden_mac(a: np.ndarray, b: np.ndarray) -> int:
    """Call C golden model for a single dot product."""
    assert a.dtype == np.int8 and b.dtype == np.int8
    assert a.shape == b.shape and a.ndim == 1

    n = len(a)
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    return lib.int8_mac(a_ptr, b_ptr, ctypes.c_int(n))

# Test
a = np.array([1, 2, 3, 4], dtype=np.int8)
b = np.array([5, 6, 7, 8], dtype=np.int8)
print(golden_mac(a, b))   # 1*5 + 2*6 + 3*7 + 4*8 = 70
```

**DPI-C integration (SystemVerilog calling Python):**

The pattern is `SV -> DPI-C function -> C wrapper -> Python C API -> Python function`. This
is more complex and is usually only needed when the golden model logic is inherently easier
to express in Python than in C (e.g. transformer topology routing). Most teams prefer the
simpler approach of running the Python golden model offline and comparing log files.

---

## Tier 3 — Advanced

### Q7. Your BF16 softmax golden model disagrees with the RTL on 2% of test vectors, always in elements near the maximum of the input vector. What is the most likely cause?

**Answer.**

The most likely cause is a numerical stability difference in the max-subtraction step of
softmax.

**Numerically stable softmax:**
`softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))`

The RTL may compute `max(x)` using a pipelined tree comparator that finds the exact maximum.
The Python model (if carelessly written) may compute `max(x)` in a different order — but
since max is exact (no rounding), this should be identical.

**The actual sources of the 2% disagreement:**

1. **Max computed in lower precision.** If the RTL computes max in BF16 and the golden model
   computes it in FP32, the max value may differ by one ULP. Because `x_i - max(x)` is near
   zero for the argmax element, a 1 ULP error in max translates directly to a 1 ULP error in
   the shifted input, which then propagates through exp.

2. **exp approximation mismatch.** Many accelerators implement exp as a piecewise polynomial
   or LUT-based approximation rather than exact IEEE 754 exp. The golden model that uses
   `numpy.exp` (which is accurate to < 1 ULP) will not match an RTL exp that has ±2 ULP error.
   The mismatch is worst at values near 0 (after the max subtraction), which is exactly where
   the argmax elements sit.

3. **Accumulator precision for the denominator.** If the RTL accumulates the exp values in
   BF16 and the golden model uses FP32 for the sum, the denominator will differ, causing all
   output values to be slightly off.

**Diagnosis:**

```python
import numpy as np
import ml_dtypes

def softmax_bf16_exact(x_bf16: np.ndarray) -> np.ndarray:
    """
    Softmax that mirrors typical hardware BF16 implementation:
    - max computed in BF16
    - subtraction in BF16
    - exp in BF16 (approximated)
    - accumulation in BF16
    - division in BF16
    """
    bf16 = ml_dtypes.bfloat16

    x      = x_bf16.astype(bf16)
    x_max  = x.max().astype(bf16)          # BF16 max
    x_sub  = (x - x_max).astype(bf16)      # BF16 subtraction
    exp_v  = np.exp(x_sub.astype(np.float32)).astype(bf16)  # FP32 exp, then round to BF16
    sum_v  = exp_v.sum().astype(bf16)       # BF16 accumulation
    result = (exp_v / sum_v).astype(bf16)  # BF16 division

    return result
```

If the disagreement disappears when you switch the golden model from FP32 to BF16 throughout,
the cause is confirmed as precision mismatch in intermediate values.

---

### Q8. How do you verify your Python golden model is itself correct — i.e., how do you validate the validator?

**Answer.**

This is a critical question that is often overlooked. A golden model with a bug will cause
real RTL bugs to be masked (false passes) or correct RTL to be flagged incorrectly (false
failures).

**Layers of golden model validation:**

**Layer 1 — Unit tests with analytically known results.**

```python
def test_int8_matmul_identity():
    """Y = I * A should return A (for appropriate scale factors)."""
    N = 4
    I = np.eye(N, dtype=np.int8)
    A = np.array([[1, 2, 3, 4]], dtype=np.int8)
    result = gemm_per_channel_int8(
        A, I,
        scale_a=1.0, zp_a=0,
        scale_w=np.ones(N), zp_w=0,
        scale_y=1.0, zp_y=0,
    )
    assert np.array_equal(result, A), f"Identity test failed: {result}"

def test_int8_matmul_known_values():
    """Hand-calculated expected output for a 2x2 case."""
    A = np.array([[2, 3]], dtype=np.int8)    # [1, 2]
    W = np.array([[1, 0], [0, 1]], dtype=np.int8)  # identity
    # Expected: [2, 3] with scale=1 throughout
    result = gemm_per_channel_int8(A, W, 1.0, 0, np.ones(2), 0, 1.0, 0)
    np.testing.assert_array_equal(result, np.array([[2, 3]], dtype=np.int8))
```

**Layer 2 — Cross-validation against a trusted software library.**

Compare the golden model output against PyTorch with explicit dtypes:

```python
import torch

def cross_validate_with_pytorch(A_int8, W_int8, scale_a, scale_w, scale_y):
    """Use torch.ao.nn.quantized for cross-validation."""
    # Convert to torch tensors
    A_t = torch.from_numpy(A_int8)
    W_t = torch.from_numpy(W_int8)

    # PyTorch per-channel quantised linear
    A_q = torch.quantize_per_tensor(A_t.float(), scale=scale_a, zero_point=0,
                                     dtype=torch.qint8)
    W_q = torch.quantize_per_channel(W_t.float().T,
                                      scales=torch.tensor(scale_w),
                                      zero_points=torch.zeros(len(scale_w), dtype=torch.int32),
                                      axis=0, dtype=torch.qint8)
    # ... run torch.ops.quantized.linear and compare
```

**Layer 3 — Mutation testing.**
Deliberately introduce known bugs into the golden model (e.g., change INT32 accumulator to
INT16, change RHAFZ to truncation) and verify that the unit tests detect every mutation. If
a mutation does not cause any test to fail, the test suite has a coverage gap.

**Layer 4 — Forward-pass match with a deployed model.**
Run an actual transformer forward pass using the golden model and compare the final output
logits against a known-good software implementation. If the top-1 token predictions match
across thousands of prompt tokens, the golden model is unlikely to have systematic errors.

---

### Q9. Describe how you would build a cocotb testbench where the Python golden model runs cycle-accurately alongside the RTL pipeline.

**Answer.**

Cycle-accurate co-simulation means the golden model tracks the same pipeline stages as the
RTL, producing its output in the same cycle that the DUT asserts `output_valid`.

**Design pattern:**

```python
import cocotb
from cocotb.clock   import Clock
from cocotb.triggers import RisingEdge, FallingEdge
import numpy as np
from collections import deque
import golden_model   # your reference implementation module

# The pipeline latency of the DUT (read from the micro-arch spec)
PIPE_LATENCY = 8  # cycles from input_valid to output_valid

class GemmDriver:
    """Drives random GEMM transactions onto the DUT input ports."""
    def __init__(self, dut):
        self.dut = dut
        self.pending = deque()   # (A, B, golden_result) FIFO

    async def send(self, A: np.ndarray, B: np.ndarray):
        """Drive one GEMM transaction; compute and enqueue golden result."""
        expected = golden_model.gemm(A, B)
        self.pending.append(expected)

        await FallingEdge(self.dut.clk)   # drive on falling edge
        self.dut.input_valid.value = 1
        # Pack A and B into the DUT's flat bus format
        self.dut.a_data.value = int.from_bytes(A.tobytes(), 'little')
        self.dut.b_data.value = int.from_bytes(B.tobytes(), 'little')
        await RisingEdge(self.dut.clk)
        self.dut.input_valid.value = 0

class GemmScoreboard:
    """Monitors DUT output and compares against golden model results."""
    def __init__(self, dut, driver):
        self.dut    = dut
        self.driver = driver
        self.errors = 0
        self.checks = 0

    async def run(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.dut.output_valid.value == 1:
                rtl_bytes = self.dut.c_data.value.buff
                rtl_result = np.frombuffer(rtl_bytes, dtype=np.int8)

                expected = self.driver.pending.popleft()
                self.checks += 1

                # Bit-exact comparison
                if not np.array_equal(rtl_result, expected):
                    self.errors += 1
                    cocotb.log.error(
                        f"Mismatch on check {self.checks}: "
                        f"expected {expected}, got {rtl_result}"
                    )

@cocotb.test()
async def gemm_random_test(dut):
    """Run 1000 random GEMM transactions and check all outputs."""
    clock = Clock(dut.clk, 4, units="ns")
    cocotb.start_soon(clock.start())

    driver     = GemmDriver(dut)
    scoreboard = GemmScoreboard(dut, driver)
    cocotb.start_soon(scoreboard.run())

    # Reset
    dut.reset_n.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.reset_n.value = 1

    # Send random stimuli
    rng = np.random.default_rng(seed=0)
    for i in range(1000):
        A = rng.integers(-128, 127, size=(8, 16), dtype=np.int8)
        B = rng.integers(-128, 127, size=(16, 8), dtype=np.int8)
        await driver.send(A, B)
        await RisingEdge(dut.clk)   # one idle cycle between transactions

    # Wait for pipeline to drain
    for _ in range(PIPE_LATENCY + 5):
        await RisingEdge(dut.clk)

    assert scoreboard.errors == 0, \
        f"{scoreboard.errors}/{scoreboard.checks} transactions failed"
    cocotb.log.info(f"All {scoreboard.checks} transactions passed.")
```

**Key principles:**

- The driver and scoreboard use a shared FIFO (`pending` deque) that decouples stimulus
  generation from output checking. This works because the DUT processes transactions in order.
- The golden model is called at stimulus time (when inputs are known), so there is no need to
  store inputs until the output arrives.
- The pipeline latency is treated as a specification, not measured dynamically. If the DUT's
  latency is wrong, the FIFO will underflow or transactions will be skipped, which is itself
  a detectable error.
