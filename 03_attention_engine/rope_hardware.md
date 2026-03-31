# Rotary Position Embedding (RoPE) Hardware Implementation

## Background

Rotary Position Embedding (RoPE) is the positional encoding scheme used by LLaMA, Mistral, Falcon,
and most modern open-source LLMs. Unlike absolute positional encodings (added to the embedding once
before all layers), RoPE is applied to the Q and K vectors inside each transformer layer just before
the attention score computation.

**Mathematical definition**: Given a query vector q of dimension d_head, split it into consecutive
pairs: (q_0, q_1), (q_2, q_3), ..., (q_{d-2}, q_{d-1}). Each pair is rotated by angle
theta_i * position, where theta_i is a frequency:

```
theta_i = 1 / (base^(2i / d_head))    # base = 10000 by default
```

The rotation of pair (q_{2i}, q_{2i+1}) at position p is:

```
q'_{2i}   = q_{2i}   * cos(p * theta_i) - q_{2i+1} * sin(p * theta_i)
q'_{2i+1} = q_{2i}   * sin(p * theta_i) + q_{2i+1} * cos(p * theta_i)
```

This is a 2D rotation matrix applied to each consecutive pair. The same transformation is applied
to the K vector. The V vector is NOT modified by RoPE.

**Key properties**:
- Position information is encoded directly into Q and K, not added as a separate embedding.
- The attention score q_i^T k_j depends only on the relative position (i - j), not absolute positions.
- Can generalise to longer sequences than seen during training (with RoPE scaling variants).
- Applied per-layer: L rotations per decode step (one per transformer layer).

---

## Tier 1 — Fundamentals

### Q1: Describe the RoPE rotation operation geometrically and explain why it encodes relative position information.

**Answer**

**Geometric view**: RoPE treats each consecutive pair of components of a vector as a 2D point in
the complex plane. Applying RoPE at position p multiplies each pair by `e^{j * p * theta_i}`, which
is a rotation by angle `p * theta_i`. A vector at position p is "rotated by p * theta_i" in each
2D subspace i.

**Relative position property**: The dot product between a query at position p and a key at position r:

```
q_p^T k_r = sum_i [ (q_p rotated by p*theta_i) . (k_r rotated by r*theta_i) ]
          = sum_i [ 2D_dot(q_i * R(p*theta_i), k_i * R(r*theta_i)) ]
          = sum_i [ 2D_dot(q_i, k_i * R((r-p)*theta_i)) ]   (rotation is associative)
```

The result depends on `r - p` (the relative offset), not on p and r individually. This is why
RoPE is described as providing relative positional information: the attention score between any
two positions is a function of only their distance, not their absolute indices.

**Hardware consequence**: At decode time, the query for the new token at position t needs to be
rotated by angles `(t * theta_i)` for i = 0..d_head/2-1. The key for each cached token at position
r needs rotation by `(r * theta_i)`. Crucially, cached K vectors can be rotated and stored during
prefill and decode writes — they do not need to be re-rotated during the attention read. This means
RoPE for K is applied once at write time, not at every attention read.

---

### Q2: What are the hardware components needed to implement RoPE? List them and give a rough area estimate for each.

**Answer**

**Component 1: Angle computation unit**
Computes `angle_i = position * theta_i` for i = 0..d_head/2-1.
Since theta_i values are fixed at model-loading time, they can be stored in a ROM.
For d_head=128 there are 64 theta values, each stored as FP32 = 256 bytes.
One FP32 multiply per dimension pair to get `angle_i = pos * theta_i`.
Area: 64-entry * 32-bit ROM (trivial) + 64 FP32 multipliers (moderate).

**Component 2: Sin/Cos lookup table (LUT)**
For each angle_i, compute sin(angle_i) and cos(angle_i).
The angle range is large (pos can reach thousands; angle can reach large values).
Using modular angle arithmetic: `angle mod 2*pi` before LUT lookup.
A 1024-entry LUT with linear interpolation gives ~20 bits of accuracy.
Two LUTs (sin and cos) or one LUT with a pi/2 offset trick.
Area: 2 * 1024 * 16 bits = 4 KB (for FP16 output).

**Component 3: Rotation compute unit**
Implements: `q' = q_even * cos(angle) - q_odd * sin(angle)` (one output)
           `q' = q_even * sin(angle) + q_odd * cos(angle)` (another output)
For d_head=128, there are 64 pairs to rotate.
Each rotation requires: 2 multiplies + 1 subtract (for q') and 2 multiplies + 1 add (for q'').
Total: 4 multiplies + 2 adds per pair = 64 * 6 = 384 FP16 operations.
Area: 4 FP16 multipliers + 2 FP16 adders per pair, times 64 pairs (if fully parallel).
Equivalently: one set of 4 multipliers + 2 adders, reused 64 times sequentially (64x latency, 1x area).

**Component 4: Position register**
Tracks the current token position for Q (current decode step) and for K (position of each token
as it is written to cache). One 16-bit counter per active sequence.
Area: negligible.

**Typical ASIC area for a 64-pair parallel RoPE unit at d_head=128, FP16**: roughly 0.1-0.5 mm^2
in a 7nm process (dominated by the 256 multipliers). For area-constrained designs, a serial unit
(one pair per cycle) costs ~1/64 the area at 64x the latency.

---

### Q3: Why is RoPE applied to Q and K but NOT to V? What would happen if it were applied to V?

**Answer**

**Reason for not applying to V**: The purpose of RoPE is to make the attention score `q_i^T k_j`
sensitive to relative position (i - j). This score determines the attention weights (softmax
output). The weighted sum over values — `sum_j softmax_weight_j * v_j` — aggregates value content
with learned weights. The value vectors carry the "what to output" information, not "where from"
information. Position information is already encoded in the attention weights computed from Q and K;
adding RoPE to V would apply position rotation a second time, distorting the output semantics.

**If RoPE were applied to V**: The output of each attention head would be a rotation-weighted
average of rotated value vectors. The rotation of V at position j would interact with the position
at which the output is consumed (position i), creating a position-dependent mixing of value
components. This would break the simple interpretation of values as position-independent content.
Experimentally, applying RoPE to V degrades model quality; the training dynamics are different
and require re-training. In short: the mathematical motivation for RoPE is specific to the
dot-product score computation, not to the aggregation step.

---

## Tier 2 — Intermediate

### Q4: Describe the "split-half" vs "interleaved" RoPE variants and explain their hardware implications.

**Answer**

**Interleaved pairing** (original RoPE paper):
Pairs are formed from consecutive elements: (q_0, q_1), (q_2, q_3), ..., (q_{d-2}, q_{d-1}).
Each pair (q_{2i}, q_{2i+1}) rotates by theta_i.

**Split-half pairing** (used by LLaMA, Hugging Face default):
The vector is split into two halves: first half q[0:d/2] and second half q[d/2:d].
Pairs are (q_i, q_{i+d/2}) for i = 0..d/2-1.
Each pair rotates by theta_i.

**Mathematical equivalence**: Both produce the same dot-product structure (since the pairing
is a permutation), but with different element ordering in the output.

**Hardware implications**:

**Interleaved**: Reading pairs from memory is natural — q_0 and q_1 are adjacent, load in one
pair. No address calculation overhead for pairing. Output can be written back in the same
interleaved order. Fits naturally with SIMD processing of adjacent pairs.

**Split-half**: Pairs span the first and second halves of the vector. Reading one pair requires
loading element i from address base+i and element i+d/2 from address base+i+d/2. For an SRAM
with word width 128 bits (8 FP16 values), the two halves of a pair are d/2*2 bytes = 128 bytes
apart — different cache lines. This can cause 2x memory accesses per pair if not handled.

**Hardware-friendly split-half**: Store Q in a dual-register file where register A holds the
first half and register B holds the second half. Index i into A and B simultaneously to get
the pair (q_i, q_{i+d/2}) with one clock cycle. The RoPE unit's address generation then has
a fixed offset of d/2 between A and B ports — no complex address computation needed.

**Recommendation**: Interleaved pairing is marginally more hardware-friendly for SRAM access.
However, most real implementations store the sin/cos values pre-computed in the interleaved or
split-half order to match the expected software convention for the specific model being implemented.
The critical thing is to match the convention used during model training.

---

### Q5: How are sin and cos values computed in hardware? Compare a ROM LUT approach versus computing them on the fly for each token position.

**Answer**

**Approach A: Full precomputed ROM (angle-indexed)**

Precompute sin(i * theta_j) and cos(i * theta_j) for all position indices i = 0..S_max-1 and
all frequency indices j = 0..d_head/2-1. Store in ROM.

For S_max=4096, d_head=128:
```
Table size = 2 (sin+cos) * 4096 * 64 * 2 bytes = 1,048,576 bytes = 1 MB
```

At a given decode step with position p, read the row `p` from the table to get all 64 sin and
cos values in one burst. With a 128-byte cache line, the entire row (64 * 2 * 2 = 256 bytes)
is read in 2 cache lines — fast.

Pros: Zero computation at inference; fixed 1-cycle latency; no approximation error.
Cons: 1 MB of SRAM per supported max sequence length (grows linearly). For S_max=32768: 8 MB.

**Approach B: LUT with angle argument**

Reduce the angle `pos * theta_j mod 2*pi` and use a shared sin/cos LUT indexed by the reduced angle.
The LUT has N entries covering [0, 2*pi). N=1024 with linear interpolation gives ~20-bit accuracy.

```
LUT size = 2 (sin+cos) * 1024 * 2 bytes = 4 KB  (FP16)
```

For each (pos, j) pair: one FP32 multiply to get angle, one modulo operation to reduce, one LUT
lookup + interpolation. The modulo is a subtraction (angle - floor(angle / 2*pi) * 2*pi) using
the FP exponent field.

Pros: Compact table (4 KB independent of S_max); can handle arbitrary long contexts.
Cons: 3-5 cycle latency per sin/cos value; requires FP multiply and modulo unit; approximation
error from LUT interpolation (but <1 ULP for 1024-entry LUT, acceptable for FP16 softmax).

**Approach C: CORDIC**

Iterative algorithm for sin/cos. For 16-bit accuracy, 16 iterations, each with one shift and one add.
Pros: Extremely area-efficient (no multipliers, no SRAM for LUT).
Cons: High latency (16 cycles serial, or area-proportional parallel); not justified for this use case
when a 4 KB LUT is available.

**Practical recommendation**: Use Approach A (full precomputed ROM) for accelerators targeting
a fixed maximum sequence length, or Approach B (4 KB LUT + on-the-fly angle computation) for
flexible long-context support. Approach A is preferred because it maximises throughput (one read
per cycle) and the memory cost (1 MB) is modest relative to the hundreds of MB of weight storage.

---

### Q6: During the decode phase, RoPE must be applied to one new Q vector and one new K vector per layer. Describe the hardware scheduling and whether RoPE for Q and K can be parallelised.

**Answer**

**Decode step timeline** for one transformer layer:

1. QKV projection: produces q_new, k_new, v_new (all un-rotated).
2. RoPE on q_new: rotate q_new at position `current_seq_len` -> q_rotated.
3. RoPE on k_new: rotate k_new at position `current_seq_len` -> k_rotated.
4. Append k_rotated and v_new to KV cache.
5. Compute attention scores: q_rotated @ K_cache^T.
6. Softmax + weighted sum.

**Parallelism between Q and K RoPE**:
Both q_new and k_new must be rotated at the same position (current_seq_len). The angles are
identical: `pos * theta_i` for all i. Therefore the sin/cos values are shared.

**Hardware scheduling**:
- The sin/cos LUT (or ROM row) is read once for `position = current_seq_len`. This produces
  64 (sin, cos) pairs.
- The same (sin, cos) pairs are used to rotate both q_new and k_new in parallel.
- If the RoPE unit has 64 rotation cells (one per dimension pair), it can rotate one vector
  per clock cycle. Q and K are rotated in back-to-back cycles using the same sin/cos values.
- If the unit is fully parallel (64 rotation cells), both Q and K can be rotated simultaneously
  (q in cycle 1, k in cycle 1 as well if there are duplicate units, or back-to-back in cycles 1 and 2).

**Cached K vector consideration**:
RoPE for K is applied at write time (when k_new is appended). The cached K vectors are already
rotated and do not need re-rotation during attention reads. This is an important hardware simplification:
the KV cache stores pre-rotated keys, so the attention score unit directly reads and uses them
without any RoPE computation in the read path. Only the current step's new Q vector needs RoPE
applied in the read path (to match the pre-rotated cached keys).

---

## Tier 3 — Advanced

### Q7: Describe how RoPE theta scaling (YaRN, LLaMA-3 RoPE extension) changes the hardware sin/cos computation requirements.

**Answer**

**Standard RoPE limitation**: Models trained with base=10000 and max_position=4096 experience
degraded performance beyond 4096 tokens because the positional angles wrap around in a way
not seen during training. Simply extending context beyond the training length causes
out-of-distribution position angles.

**RoPE scaling methods** change the effective frequency schedule:

**Linear scaling (simple)**: Replace `theta_i = 1/base^(2i/d)` with
`theta_i = 1/(base * scale_factor)^(2i/d)`. This uniformly stretches all frequencies,
effectively making the model see position p as position p/scale_factor.
Hardware impact: theta values change (pre-computed ROM must be reloaded) but the computation
structure is identical. No hardware changes required; only the theta ROM contents change.

**YaRN (Yet another RoPE extensioN)**: Applies different scale factors to different frequency
bands. High-frequency components (small i) are scaled differently from low-frequency (large i).
The formula introduces a per-dimension scale factor alpha_i:
```
theta_i_new = theta_i * alpha_i     (alpha_i depends on i and the target context length)
```
Hardware impact: theta values are no longer a simple geometric series. The 64 theta values must
be stored explicitly (64 * 4 bytes = 256 bytes) rather than computed from a formula. The sin/cos
LUT is indexed by these pre-computed theta values rather than a formulaic series. No structural
hardware change; only the theta ROM contents expand slightly.

**LLaMA-3 RoPE (base=500,000)**: Simply increases the base from 10,000 to 500,000, making
the frequency schedule much slower (lower frequencies across all dimensions). The rotations
accumulate more slowly, extending the useful position range before angular aliasing.
Hardware impact: None. The hardware structure is identical; only the pre-computed theta values
change.

**Implication for ROM sizing**: If the hardware pre-computes sin/cos for all positions up to
S_max and stores them in a ROM, the ROM must cover the new extended context. For S_max=128,000
(LLaMA-3 long context), the 1 MB ROM for S=4096 grows to 32 MB — a significant on-chip SRAM cost.
At this scale, the on-the-fly angle computation approach (Approach B from Q5) becomes preferable:
compute angles dynamically using the stored theta values, using a compact 4 KB sin/cos LUT.
This eliminates the S_max dependence from the SRAM area at the cost of a few cycles of latency.

---

### Q8: Design the microarchitecture of a fixed-point RoPE computation unit for d_head=128 that processes one vector per 4 clock cycles. Justify all fixed-point format choices.

**Answer**

**Target**: d_head=128, 64 pairs per vector, 4-cycle latency, throughput = 1 vector/4 cycles.
This means processing 64/4 = 16 pairs per cycle — 16 parallel rotation cells.

**Fixed-point format selection**:

Input Q/K values: INT8 quantised (common for KV cache quantisation). Range: [-128, 127].
Angle values (sin, cos): These are in [-1, +1]. Use Q1.15 fixed-point: 1 sign bit, 0 integer bits,
15 fractional bits. Range [-1, 1-2^{-15}], resolution 2^{-15} ≈ 3e-5.

**Rotation computation in fixed-point**:
```
q'_even = q_even * cos(angle) - q_odd * sin(angle)
        = Q1.15 * INT8 - Q1.15 * INT8
        = Q9.15 * 2 accumulation terms
        = Q10.15 result (26-bit intermediate)
        = round to INT8 (clip + right-shift 15)
```

The intermediate product of INT8 * Q1.15 is a 24-bit value (1 + 15 + 8 = 24 bits including sign).
Two such products are added; the result is a 25-bit value. After rounding and clipping to INT8,
the output matches the input range.

**One rotation cell (16 cells in parallel)**:

```
Inputs per cell:
  q_even[7:0], q_odd[7:0]    (INT8, registered)
  cos_val[15:0], sin_val[15:0]  (Q1.15, registered from LUT)

Stage 1 (cycle 1): Multiplications
  prod_ee = q_even * cos_val   // 8*16 = 24-bit product, signed
  prod_os = q_odd  * sin_val   // 24-bit product, signed
  prod_es = q_even * sin_val   // 24-bit product, signed
  prod_oe = q_odd  * cos_val   // 24-bit product, signed

Stage 2 (cycle 2): Additions
  sum_even = prod_ee - prod_os  // q' = q_e*cos - q_o*sin
  sum_odd  = prod_es + prod_oe  // q' = q_e*sin + q_o*cos

Stage 3 (cycle 3): Rounding and clipping
  out_even = clip(round(sum_even >> 15), -128, 127)
  out_odd  = clip(round(sum_odd  >> 15), -128, 127)
```

**Throughput computation**:
- 16 cells process 16 pairs per cycle.
- 64 pairs per vector / 16 cells = 4 cycles per vector. Target achieved.

**Sin/Cos delivery**:
The LUT must provide 16 (sin, cos) pairs per cycle. A 64-entry table (one entry per pair index)
with a 4-bit wide read port (16 entries per cycle using a wide data bus or 16 parallel LUTs with
the same address) delivers all required values in one cycle. At 2 bytes per sin and 2 bytes per cos,
the LUT read port width is 16 * 4 = 64 bytes = 512 bits per cycle. A typical SRAM supports this
with a sufficiently wide data bus.

**Area estimate**: 16 cells * (4 multipliers + 2 adders) = 64 multipliers (8x16-bit) + 32 adders.
In 7nm, an 8x16-bit multiplier is approximately 100 gates equivalent; 64 of them ~ 6,400 gate
equivalents. The LUT SRAM (64 * 4 bytes = 256 bytes) is negligible. Total RoPE unit: < 0.01 mm^2.
