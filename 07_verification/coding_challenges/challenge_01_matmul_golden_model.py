"""
Challenge: Bit-Accurate INT8 Matrix Multiplication Golden Model
===============================================================

Task
----
Implement a golden model for an INT8 matrix multiplication unit that exactly
matches the numerical behaviour of the target RTL accelerator.

The RTL specification states:
  - Inputs A [M x K] and B [K x N] are INT8 (signed, range [-128, 127]).
  - The accumulator is INT32. Overflow of INT32 is undefined behaviour and
    will not be tested (inputs and K are bounded to prevent it).
  - Quantisation uses per-tensor scale factors for A and B and a separate
    per-tensor scale for the output.
  - Zero-points may be non-zero (asymmetric quantisation).
  - Requantisation rounds using "round half away from zero" (RHAFZ).
  - Output is clamped to INT8 range [-128, 127] after rounding.
  - Bias (if present) is added as a per-output-channel INT32 value BEFORE
    requantisation (i.e., bias is in the accumulator domain).

Interface Specification
-----------------------
  gemm(A, B, scale_a, zp_a, scale_b, zp_b, scale_out, zp_out, bias=None)
    A        : numpy int8 array, shape [M, K]
    B        : numpy int8 array, shape [K, N]
    scale_a  : float, scale factor for A
    zp_a     : int,   zero point for A  (input - zp_a is the real value)
    scale_b  : float, scale factor for B
    zp_b     : int,   zero point for B
    scale_out : float, scale factor for output
    zp_out   : int,   zero point for output
    bias     : numpy int32 array of shape [N], or None
    Returns  : numpy int8 array of shape [M, N]

Mathematical identity
---------------------
  real_a[i,k] = (A[i,k] - zp_a) * scale_a
  real_b[k,j] = (B[k,j] - zp_b) * scale_b
  real_c[i,j] = sum_k(real_a[i,k] * real_b[k,j])

  In integer domain (avoids FP in the inner loop):
  acc[i,j] = sum_k((A[i,k] - zp_a) * (B[k,j] - zp_b))

  Scale back to output domain:
  effective_scale = (scale_a * scale_b) / scale_out
  out_fp[i,j]     = acc[i,j] * effective_scale + bias_scaled[j] + zp_out
  out[i,j]        = clip(RHAFZ(out_fp[i,j]), -128, 127)

  where bias_scaled[j] = bias[j] * scale_bias / scale_out
  (In this spec, bias is already in the accumulator INT32 domain, so:)
  out_fp[i,j] = (acc[i,j] + bias[j]) * effective_scale + zp_out

Constraints
-----------
  - K * 127 * 127 < 2^31 - 1  (no INT32 overflow for valid K)
  - Maximum K tested: 4096
  - M, N: 1 to 4096

Your Tasks
----------
  1. Complete the QuantConfig dataclass.
  2. Complete the round_half_away_from_zero() function.
  3. Complete the GemmGoldenModel.compute() method.
  4. Ensure all provided test cases pass.

Run with:
  python challenge_01_matmul_golden_model.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: Rounding Mode
# ---------------------------------------------------------------------------

def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """
    Round each element of x to the nearest integer.
    Ties (exactly 0.5) round away from zero.

    Examples:
      0.5  -> 1
     -0.5  -> -1
      1.4  -> 1
      1.5  -> 2
     -1.5  -> -2

    This matches the rounding mode used by the RTL requantiser.

    Note: Python's built-in round() uses banker's rounding (round half to even),
    which is NOT the same. NumPy's np.round() also uses banker's rounding.
    Both would cause spurious mismatches.
    """
    # np.sign returns 0 for x=0, so ties at exactly 0 round to 0 (correct)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ---------------------------------------------------------------------------
# Quantisation Configuration
# ---------------------------------------------------------------------------

@dataclass
class QuantConfig:
    """
    Encapsulates quantisation parameters for a single tensor.

    The mapping between quantised integer values and real numbers is:
        real_value = (quantised_value - zero_point) * scale

    Attributes
    ----------
    scale      : positive float, quantisation step size
    zero_point : integer, offset applied to quantised values
                 0 for symmetric quantisation
                 non-zero for asymmetric (unsigned) quantisation
    dtype      : numpy dtype for the quantised representation
    clamp_min  : minimum value after clamping (inclusive)
    clamp_max  : maximum value after clamping (inclusive)
    """
    scale:     float
    zero_point: int
    dtype:     np.dtype
    clamp_min: int
    clamp_max: int

    def dequantise(self, x_q: np.ndarray) -> np.ndarray:
        """Convert quantised INT values to float32 real values."""
        return (x_q.astype(np.float64) - self.zero_point) * self.scale

    def quantise(self, x_real: np.ndarray) -> np.ndarray:
        """Convert float32 real values to quantised INT representation."""
        x_scaled  = x_real / self.scale + self.zero_point
        x_rounded = round_half_away_from_zero(x_scaled).astype(np.int32)
        x_clamped = np.clip(x_rounded, self.clamp_min, self.clamp_max)
        return x_clamped.astype(self.dtype)


# Pre-built standard configs
INT8_SYMMETRIC = QuantConfig(
    scale=1.0, zero_point=0, dtype=np.int8, clamp_min=-128, clamp_max=127
)


# ---------------------------------------------------------------------------
# Golden Model
# ---------------------------------------------------------------------------

class GemmGoldenModel:
    """
    Bit-accurate INT8 GEMM golden model.

    Implements signed INT8 x INT8 -> INT32 accumulation, per-tensor
    quantisation with non-zero zero points, optional INT32 bias, and
    RHAFZ requantisation back to INT8.

    All internal accumulation is performed in INT64 to guarantee that no
    intermediate overflow occurs even for the maximum supported K=4096
    (worst case: 4096 * 127 * 127 = 66,060,288, well within INT32 range,
     but we use INT64 for defensive programming).
    """

    def __init__(self):
        # No instance state required; all parameters passed to compute()
        pass

    def compute(
        self,
        A:         np.ndarray,      # [M, K] INT8
        B:         np.ndarray,      # [K, N] INT8
        cfg_a:     QuantConfig,     # quantisation config for A
        cfg_b:     QuantConfig,     # quantisation config for B
        cfg_out:   QuantConfig,     # quantisation config for output
        bias:      Optional[np.ndarray] = None,  # [N] INT32, in accumulator domain
    ) -> np.ndarray:                # [M, N] INT8
        """
        Compute quantised GEMM: Y = quant(dequant(A) @ dequant(B) + bias_dq)

        The computation proceeds in three phases:
          Phase 1: Integer accumulation in INT64 (zero-point-corrected)
          Phase 2: Scaling the accumulator to the output domain (float64)
          Phase 3: Bias add, output zero-point add, RHAFZ rounding, INT8 clamp

        Parameters
        ----------
        A        : INT8 matrix, shape [M, K]
        B        : INT8 matrix, shape [K, N]
        cfg_a    : QuantConfig for activations A
        cfg_b    : QuantConfig for weights B
        cfg_out  : QuantConfig for output
        bias     : optional INT32 bias vector, shape [N]; in accumulator domain
                   (i.e., in units of scale_a * scale_b, NOT in output scale units)

        Returns
        -------
        numpy int8 array of shape [M, N]
        """
        # Validate inputs
        assert A.dtype == np.int8,  f"A must be int8, got {A.dtype}"
        assert B.dtype == np.int8,  f"B must be int8, got {B.dtype}"
        assert A.ndim == 2,          "A must be 2-dimensional"
        assert B.ndim == 2,          "B must be 2-dimensional"
        assert A.shape[1] == B.shape[0], \
            f"Inner dimensions must match: A={A.shape}, B={B.shape}"
        if bias is not None:
            assert bias.dtype == np.int32, f"Bias must be int32, got {bias.dtype}"
            assert bias.shape == (B.shape[1],), \
                f"Bias shape {bias.shape} must match N={B.shape[1]}"

        M, K = A.shape
        _, N = B.shape

        # ------------------------------------------------------------------
        # Phase 1: Integer accumulation
        # ------------------------------------------------------------------
        # Subtract zero points in INT32 before multiplication.
        # This avoids the need to pre-compute the zero-point correction term
        # and is numerically identical to the correction-term approach.
        #
        # A_shifted[i,k] = A[i,k] - zp_a  (each element shifted to true value)
        # B_shifted[k,j] = B[k,j] - zp_b
        # acc[i,j] = sum_k(A_shifted[i,k] * B_shifted[k,j])
        #
        # Widen to INT64 before multiplication (INT32 * INT32 could overflow INT32)
        A_shifted = A.astype(np.int64) - np.int64(cfg_a.zero_point)  # [M, K]
        B_shifted = B.astype(np.int64) - np.int64(cfg_b.zero_point)  # [K, N]

        # Matrix multiply in INT64
        # np.matmul with int64 inputs gives int64 output
        accumulator = np.matmul(A_shifted, B_shifted)  # [M, N] INT64

        # ------------------------------------------------------------------
        # Phase 2: Add bias (in accumulator domain, before scaling)
        # ------------------------------------------------------------------
        if bias is not None:
            accumulator = accumulator + bias.astype(np.int64)  # broadcast over M

        # ------------------------------------------------------------------
        # Phase 3: Requantisation to output
        # ------------------------------------------------------------------
        # Effective scale converts from (scale_a * scale_b) units to scale_out units
        effective_scale = (cfg_a.scale * cfg_b.scale) / cfg_out.scale  # scalar float64

        # Convert accumulator to floating point and apply scale
        # Result is in output quantised units (before adding output zero point)
        out_fp = accumulator.astype(np.float64) * effective_scale  # [M, N] float64

        # Add output zero point (offset for asymmetric output quantisation)
        out_fp += float(cfg_out.zero_point)

        # Apply RHAFZ rounding and convert to INT32
        out_rounded = round_half_away_from_zero(out_fp).astype(np.int32)  # [M, N]

        # Clamp to output range and cast to INT8
        out_int8 = np.clip(out_rounded, cfg_out.clamp_min, cfg_out.clamp_max
                           ).astype(np.int8)  # [M, N]

        return out_int8


# ---------------------------------------------------------------------------
# Test Utilities
# ---------------------------------------------------------------------------

def make_cfg(scale: float, zp: int = 0) -> QuantConfig:
    """Convenience function to create a symmetric INT8 QuantConfig."""
    return QuantConfig(
        scale=scale, zero_point=zp,
        dtype=np.int8, clamp_min=-128, clamp_max=127
    )


def check(name: str, result: np.ndarray, expected: np.ndarray) -> bool:
    """Assert result equals expected; print pass/fail with details."""
    if np.array_equal(result, expected):
        print(f"  PASS  {name}")
        return True
    else:
        diff = result.astype(np.int32) - expected.astype(np.int32)
        max_err = int(np.abs(diff).max())
        n_wrong = int((diff != 0).sum())
        print(f"  FAIL  {name}")
        print(f"         {n_wrong} element(s) wrong, max error = {max_err} LSB")
        print(f"         result:\n{result}")
        print(f"         expected:\n{expected}")
        return False


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

def run_tests():
    model = GemmGoldenModel()
    all_pass = True

    print("=" * 60)
    print("INT8 GEMM Golden Model Test Suite")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Identity — A @ I = A (with unit scales)
    # ------------------------------------------------------------------
    print("\n[T1] Identity matrix product")
    M, N = 4, 4
    rng = np.random.default_rng(seed=1)
    A = rng.integers(-10, 10, (M, N), dtype=np.int8)
    I = np.eye(N, dtype=np.int8)
    result   = model.compute(A, I, make_cfg(1.0), make_cfg(1.0), make_cfg(1.0))
    expected = A.copy()
    all_pass &= check("A @ I = A", result, expected)

    # ------------------------------------------------------------------
    # Test 2: All-zero activation — output should be zero (sym. quant)
    # ------------------------------------------------------------------
    print("\n[T2] All-zero activation")
    M, K, N = 3, 8, 5
    A_zero = np.zeros((M, K), dtype=np.int8)
    B_rand = rng.integers(-50, 50, (K, N), dtype=np.int8)
    result   = model.compute(A_zero, B_rand, make_cfg(1.0), make_cfg(1.0), make_cfg(1.0))
    expected = np.zeros((M, N), dtype=np.int8)
    all_pass &= check("Zero A → zero output", result, expected)

    # ------------------------------------------------------------------
    # Test 3: Known 2x2 output (hand-calculated)
    # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    # acc[0,0] = 1*5 + 2*7 = 19, acc[0,1] = 1*6 + 2*8 = 22
    # acc[1,0] = 3*5 + 4*7 = 43, acc[1,1] = 3*6 + 4*8 = 50
    # With scale_a=1, scale_b=1, scale_out=1, zp=0 → output = clamp(RHAFZ(acc))
    # All values in [-128,127] so no clamping needed.
    # ------------------------------------------------------------------
    print("\n[T3] Hand-calculated 2x2")
    A2 = np.array([[1, 2], [3, 4]], dtype=np.int8)
    B2 = np.array([[5, 6], [7, 8]], dtype=np.int8)
    expected = np.array([[19, 22], [43, 50]], dtype=np.int8)
    result   = model.compute(A2, B2, make_cfg(1.0), make_cfg(1.0), make_cfg(1.0))
    all_pass &= check("2x2 hand-calculated", result, expected)

    # ------------------------------------------------------------------
    # Test 4: Scale factors applied correctly
    # A = [[4]], B = [[8]], scale_a=0.5, scale_b=0.25, scale_out=1.0
    # acc = 4 * 8 = 32
    # effective_scale = (0.5 * 0.25) / 1.0 = 0.125
    # out_fp = 32 * 0.125 = 4.0 → RHAFZ(4.0) = 4 → clamp = 4
    # ------------------------------------------------------------------
    print("\n[T4] Scale factor application")
    A4 = np.array([[4]], dtype=np.int8)
    B4 = np.array([[8]], dtype=np.int8)
    result   = model.compute(A4, B4, make_cfg(0.5), make_cfg(0.25), make_cfg(1.0))
    expected = np.array([[4]], dtype=np.int8)
    all_pass &= check("Scale factors: 0.5 * 0.25 / 1.0", result, expected)

    # ------------------------------------------------------------------
    # Test 5: Rounding — result falls on 0.5 boundary
    # acc = 3, effective_scale = 1/6 → out_fp = 0.5 → RHAFZ → 1 (not 0)
    # Banker's rounding would give 0 (round to even), which is WRONG for this spec.
    # ------------------------------------------------------------------
    print("\n[T5] RHAFZ rounding at 0.5 boundary (not banker's rounding)")
    A5 = np.array([[3]], dtype=np.int8)
    B5 = np.array([[1]], dtype=np.int8)
    # scale_a * scale_b / scale_out = 1/6  →  3 * (1/6) = 0.5
    result   = model.compute(A5, B5, make_cfg(1.0), make_cfg(1.0), make_cfg(6.0))
    expected = np.array([[1]], dtype=np.int8)   # RHAFZ: 0.5 rounds to 1
    all_pass &= check("RHAFZ: 0.5 -> 1 (not 0)", result, expected)

    # ------------------------------------------------------------------
    # Test 6: Negative rounding at -0.5 boundary
    # acc = -3, effective_scale = 1/6 → out_fp = -0.5 → RHAFZ → -1
    # ------------------------------------------------------------------
    print("\n[T6] RHAFZ rounding at -0.5 boundary")
    A6 = np.array([[-3]], dtype=np.int8)
    B6 = np.array([[1]], dtype=np.int8)
    result   = model.compute(A6, B6, make_cfg(1.0), make_cfg(1.0), make_cfg(6.0))
    expected = np.array([[-1]], dtype=np.int8)  # RHAFZ: -0.5 rounds to -1
    all_pass &= check("RHAFZ: -0.5 -> -1", result, expected)

    # ------------------------------------------------------------------
    # Test 7: Output saturation (positive)
    # acc = 127 * 100 = 12700, effective_scale = 100 → out = 127 (clamped)
    # Without clamping: 12700 * 100 / 1 = 1,270,000 (way out of INT8 range)
    # ------------------------------------------------------------------
    print("\n[T7] Positive saturation to INT8 max (127)")
    A7 = np.array([[127]], dtype=np.int8)
    B7 = np.array([[100]], dtype=np.int8)
    # scale_out large enough that result should be 127 after clamp
    # acc = 127 * 100 = 12700; effective = 1.0 * 1.0 / 100.0 = 0.01
    # out_fp = 12700 * 0.01 = 127.0 → no clamping needed (exactly 127)
    result   = model.compute(A7, B7, make_cfg(1.0), make_cfg(1.0), make_cfg(100.0))
    expected = np.array([[127]], dtype=np.int8)
    all_pass &= check("Saturation positive: exact 127", result, expected)

    # Now overflow: result would be 128, must clamp to 127
    # acc = 127 * 101 = 12827; 12827 * 0.01 = 128.27 → RHAFZ → 128 → clamp → 127
    A7b = np.array([[127]], dtype=np.int8)
    B7b = np.array([[101]], dtype=np.int8)
    result2   = model.compute(A7b, B7b, make_cfg(1.0), make_cfg(1.0), make_cfg(100.0))
    expected2 = np.array([[127]], dtype=np.int8)
    all_pass  &= check("Saturation positive: 128 clamps to 127", result2, expected2)

    # ------------------------------------------------------------------
    # Test 8: Output saturation (negative)
    # ------------------------------------------------------------------
    print("\n[T8] Negative saturation to INT8 min (-128)")
    A8 = np.array([[-128]], dtype=np.int8)
    B8 = np.array([[101]], dtype=np.int8)
    # acc = -128 * 101 = -12928; * 0.01 = -129.28 → RHAFZ → -129 → clamp → -128
    result   = model.compute(A8, B8, make_cfg(1.0), make_cfg(1.0), make_cfg(100.0))
    expected = np.array([[-128]], dtype=np.int8)
    all_pass &= check("Saturation negative: -129 clamps to -128", result, expected)

    # ------------------------------------------------------------------
    # Test 9: Non-zero zero points (asymmetric quantisation)
    # A_shifted = A - zp_a = 5 - 128 = -123
    # B_shifted = B - zp_b = 0      (zp_b = 0)
    # Wait — let's use a concrete example:
    # A=[[130]] zp_a=128 → shifted = 2 (real value = 2 * scale_a = 2 * 1.0 = 2.0)
    # B=[[3]]   zp_b=0   → shifted = 3 (real value = 3 * scale_b = 3 * 1.0 = 3.0)
    # acc = 2 * 3 = 6, zp_out = 128
    # out_fp = 6 * (1.0*1.0/1.0) + 128 = 134 → clamp to 127
    #
    # Use INT8 representation: A=[[127]] maps to zp_a=0 for simplicity;
    # instead use: A[0,0]=2 as raw INT8, zp_a=-126 → shifted = 2 - (-126) = 128?
    # That overflows INT8. Let's keep it clean with small values.
    #
    # Clean test: A=[[5]] INT8, zp_a=3 → A_shifted = 5-3 = 2
    #             B=[[4]] INT8, zp_b=2 → B_shifted = 4-2 = 2
    #             acc = 4; effective_scale = 1.0; zp_out = 0
    #             out_fp = 4.0 → expected = 4
    # ------------------------------------------------------------------
    print("\n[T9] Non-zero zero points (asymmetric quantisation)")
    A9 = np.array([[5]], dtype=np.int8)
    B9 = np.array([[4]], dtype=np.int8)
    cfg_a9   = make_cfg(1.0, zp=3)   # A_shifted = 5 - 3 = 2
    cfg_b9   = make_cfg(1.0, zp=2)   # B_shifted = 4 - 2 = 2
    cfg_out9 = make_cfg(1.0, zp=0)   # no output zero point
    result   = model.compute(A9, B9, cfg_a9, cfg_b9, cfg_out9)
    # acc = 2 * 2 = 4 → out = 4
    expected = np.array([[4]], dtype=np.int8)
    all_pass &= check("Asymmetric zero points: (5-3)*(4-2) = 4", result, expected)

    # ------------------------------------------------------------------
    # Test 10: Bias addition (in accumulator domain)
    # acc = 10, bias = [5] → acc_with_bias = 15
    # effective_scale = 1.0, zp_out = 0
    # out = 15
    # ------------------------------------------------------------------
    print("\n[T10] Bias addition")
    A10 = np.array([[2]], dtype=np.int8)
    B10 = np.array([[5]], dtype=np.int8)  # acc = 10
    bias10 = np.array([5], dtype=np.int32)
    result   = model.compute(A10, B10, make_cfg(1.0), make_cfg(1.0), make_cfg(1.0),
                              bias=bias10)
    expected = np.array([[15]], dtype=np.int8)
    all_pass &= check("Bias: 2*5 + bias[5] = 15", result, expected)

    # ------------------------------------------------------------------
    # Test 11: Large K — verify no overflow in accumulator
    # A = [127, 127, ..., 127] (K=128), B = [[1], [1], ..., [1]]
    # acc = 127 * 128 = 16256 (well within INT32/INT64)
    # scale_out = 128.0 → effective = 1/128 → out_fp = 127.0 → expected 127
    # ------------------------------------------------------------------
    print("\n[T11] Large K (K=128) accumulation")
    K11 = 128
    A11 = np.full((1, K11), 127, dtype=np.int8)
    B11 = np.ones((K11, 1), dtype=np.int8)
    # acc = 127 * 128 = 16256; effective = 1.0 * 1.0 / 128.0 = 0.0078125
    # out_fp = 16256 * 0.0078125 = 127.0 → expected 127
    result   = model.compute(A11, B11, make_cfg(1.0), make_cfg(1.0), make_cfg(128.0))
    expected = np.array([[127]], dtype=np.int8)
    all_pass &= check("Large K=128: 127*128 / 128 = 127", result, expected)

    # ------------------------------------------------------------------
    # Test 12: Batch output — random M x K x N with cross-check vs dequant path
    # ------------------------------------------------------------------
    print("\n[T12] Batch random test (cross-check vs dequant-multiply-requant)")
    M12, K12, N12 = 16, 32, 16
    rng12 = np.random.default_rng(seed=42)
    A12   = rng12.integers(-30, 30, (M12, K12), dtype=np.int8)
    B12   = rng12.integers(-30, 30, (K12, N12), dtype=np.int8)
    sa, sb, so = 0.02, 0.03, 0.05

    result = model.compute(A12, B12, make_cfg(sa), make_cfg(sb), make_cfg(so))

    # Cross-check: dequantise, multiply, requantise in float
    A_dq   = A12.astype(np.float64) * sa
    B_dq   = B12.astype(np.float64) * sb
    C_fp   = A_dq @ B_dq
    C_sc   = C_fp / so
    C_rnd  = round_half_away_from_zero(C_sc).astype(np.int32)
    expected = np.clip(C_rnd, -128, 127).astype(np.int8)

    all_pass &= check("Random 16x32x16 vs dequant cross-check", result, expected)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if all_pass:
        print("All tests PASSED.")
    else:
        print("SOME TESTS FAILED — see details above.")
    print("=" * 60)
    return all_pass


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    success = run_tests()
    raise SystemExit(0 if success else 1)
