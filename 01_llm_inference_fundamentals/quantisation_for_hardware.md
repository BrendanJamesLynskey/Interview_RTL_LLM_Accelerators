# Quantisation for Hardware Design

## Overview

Quantisation is the process of representing model weights and/or activations in lower-precision
number formats. From a hardware perspective, this is not just an accuracy trade-off — it
fundamentally changes the required datapath width, MAC array design, accumulator sizing,
memory organisation, and dequantisation logic. This file covers number formats, mixed-precision
strategies, and the RTL implications of each design choice.

**Key notation**

| Symbol | Meaning |
|---|---|
| $W$ | Weight tensor |
| $X$ | Activation tensor |
| $s$ | Quantisation scale factor |
| $z$ | Zero-point offset |
| $b$ | Bit-width |
| $Q(x)$ | Quantised representation of $x$ |
| $G$ | Group size for grouped quantisation |
| $\text{bpe}$ | Bytes per element |

---

## Tier 1 — Fundamentals

### Q1. Explain the difference between INT8, FP16, BF16, and FP8 number formats. What are the hardware cost implications of each?

**Answer**

**Integer formats**

**INT8** represents integers in the range $[-128, 127]$ (signed) or $[0, 255]$ (unsigned)
using 8 bits. There is no concept of exponent — the mapping to real values requires an
external scale factor $s$ and optionally a zero-point $z$:
$$x_{\text{real}} \approx s \cdot (Q - z)$$

Hardware properties:
- Multiplier: $8 \times 8 \to 16$-bit partial product. Very compact — roughly $4\times$ smaller
  die area than FP32 multiply.
- Accumulator: Requires 32-bit accumulation to avoid overflow when summing many 16-bit products
  (e.g., dot products over $d=4096$ elements need $\log_2(4096) + 16 = 28$ bits of headroom,
  so 32 bits is standard).
- Throughput: Hardware can pack $2\times$ more INT8 MACs per mm$^2$ vs INT16, and $4\times$
  vs FP32.

**Floating-point formats**

All floating-point formats have the structure: sign | exponent | mantissa.

| Format | Sign | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | $\sim 7$ decimal digits |
| FP16 | 1 | 5 | 10 | $\pm 65504$ | $\sim 3$ decimal digits |
| BF16 | 1 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | $\sim 2$ decimal digits |
| FP8 E4M3 | 1 | 4 | 3 | $\pm 448$ | $\sim 1$ decimal digit |
| FP8 E5M2 | 1 | 5 | 2 | $\pm 57344$ | Less precision |

**FP16 vs BF16**: BF16 has the same exponent as FP32, making overflow/underflow behaviour
identical to FP32 — this simplifies training and avoids the FP16 overflow issues that
require loss scaling. BF16 has lower mantissa precision, which is acceptable for most
inference tasks.

**Hardware implications**:
- FP16/BF16 multipliers are $4\times$ smaller than FP32 but require exponent alignment logic
  (adds latency vs integer).
- FP8 halves the bandwidth vs FP16/BF16 and allows $2\times$ more MACs in the same area,
  but requires dequantisation or direct FP8 MAC hardware.
- INT8 is cheaper than FP8 to implement (no exponent logic), but requires offline calibration
  to determine the scale factor.

---

### Q2. What is symmetric vs asymmetric quantisation, and when does the choice matter for hardware?

**Answer**

**Symmetric quantisation** maps zero in floating-point exactly to zero in integer:
$$Q(x) = \text{round}\left(\frac{x}{s}\right), \qquad x_{\text{real}} \approx s \cdot Q$$
The zero-point $z = 0$.

**Asymmetric quantisation** allows a non-zero offset:
$$Q(x) = \text{round}\left(\frac{x}{s}\right) + z, \qquad x_{\text{real}} \approx s \cdot (Q - z)$$

**Why it matters for hardware**:

The dequantisation step in an asymmetric matmul $Y = (s_W (W_q - z_W)) \cdot (s_X X_q)$
expands as:
$$Y = s_W s_X \cdot W_q X_q - s_W s_X z_W \cdot \mathbf{1} X_q$$

The second term is a bias correction that requires computing $\sum_k X_q[k]$ for each
input vector — an extra reduction operation that adds latency and logic. In batch mode,
this correction can be precomputed, but for streaming decode it adds overhead per step.

**Symmetric is preferred for weights** because:
1. Weights are static — scale can be computed offline.
2. The zero-point correction is avoided in the MAC loop.
3. Weight distributions are typically symmetric (approximately centred around zero after training).

**Asymmetric is sometimes needed for activations** because ReLU-type activations are
non-negative (range $[0, a_{\max}]$) — symmetric quantisation wastes half the range.
However, post-ReLU quantisation can use unsigned symmetric INT8 $[0, 255]$ instead.

---

### Q3. What is post-training quantisation (PTQ) vs quantisation-aware training (QAT), and what does each require from the hardware team?

**Answer**

**Post-Training Quantisation (PTQ)**

The model is trained in full precision (FP32/BF16) and then quantised after training.
Scale factors are determined by running a small calibration dataset through the model
and observing activation ranges.

Hardware implications:
- The hardware receives weight tensors pre-quantised and packed (e.g., INT8 values stored
  as bytes).
- The hardware must implement dequantisation inline: each weight tile is dequantised as it
  is loaded from HBM, before entering the MAC array.
- For **weight-only quantisation** (W8A16, W4A16): the MAC array operates at FP16/BF16
  precision on activations; weights are just stored compactly and converted on the fly.
- For **weight-and-activation quantisation** (W8A8): the MAC array operates in INT8;
  accumulators are INT32; a dequantisation/rescaling step follows before the next layer.

**Quantisation-Aware Training (QAT)**

Quantisation is simulated during training using fake-quantise operations (quantise then
dequantise), so gradients flow through the model as if it were quantised. This allows
the model weights to adapt to the quantisation noise.

Hardware implications:
- The hardware format required is the same as PTQ — QAT produces the same quantised
  weight tensors.
- QAT typically achieves better accuracy, especially at INT4, so hardware teams can
  often target lower precision with QAT models than PTQ, reducing bandwidth demands further.
- From the hardware team's perspective: no change in the datapath — the quantisation
  simulation during QAT is purely a software/training concern.

---

## Tier 2 — Intermediate

### Q4. Describe the W4A16 quantisation scheme. What does the hardware datapath look like, and what are the bandwidth and compute trade-offs compared to W8A8?

**Answer**

**W4A16**: Weights stored in INT4, activations kept at FP16. No activation quantisation required.

**Datapath**:

```
HBM
  |
  | INT4 weight tiles (packed: 2 weights per byte)
  v
Weight unpacking buffer
  |
  | INT4 pairs unpacked to INT4
  v
Dequantisation unit: INT4 * FP16_scale + FP16_bias --> FP16 weights
  |
  | FP16 weights
  v
FP16 MAC array <-- FP16 activations
  |
  | FP32 accumulators
  v
FP16 output (after rounding)
```

**Bandwidth comparison vs W8A8**:

| Scheme | Weight BW | Activation precision | MAC width |
|---|---|---|---|
| FP16 | $2N$ bytes | FP16 | FP16 x FP16 |
| W8A8 | $N$ bytes | INT8 | INT8 x INT8 |
| W4A16 | $0.5N$ bytes | FP16 | FP16 x FP16 |

W4A16 reduces weight bandwidth $4\times$ vs FP16, but the MAC array still operates at FP16.
This means:
- **Throughput** (for bandwidth-bound decode): up to $4\times$ higher than FP16.
- **Compute throughput** (FLOP/s): same as FP16 — no benefit for compute-bound prefill.

**Accuracy vs W8A8**:
W4A16 generally suffers more accuracy loss than W8A8. Group quantisation (e.g., per-group-of-128
scale factors) substantially recovers accuracy at the cost of additional scale-factor storage
($N/G \times 2$ bytes extra for scales, typically $\sim 3\%$ of weight storage at $G=128$).

**RTL complications of W4A16**:
1. **Unpacking logic**: Two INT4 values are packed per byte. Unpacking requires bit shifting
   and masking on the weight-fetch path.
2. **Dequantisation throughput**: Must match the MAC array throughput. For a 256-element
   dot product per cycle, the dequantisation unit must convert 256 INT4 values to FP16 per cycle.
3. **Non-uniform scale storage**: Per-group scales are stored alongside weights; the address
   generation logic must interleave scale fetches with weight fetches based on group boundaries.

---

### Q5. What is MXFP4/MXINT4 (MX — Microscaling) format? How does it differ from standard block floating point, and what hardware support does it require?

**Answer**

**Microscaling (MX) formats** are defined by the OCP MX specification (adopted by AMD, Intel,
NVIDIA, Qualcomm, and others). They provide a standardised block floating-point format with
a shared exponent over a small group.

**MXFP4 (E2M1)**:
- Each element: 1 sign bit, 2 exponent bits, 1 mantissa bit = 4 bits total.
- Shared scale: one 8-bit shared exponent (E8M0) per block of 32 elements.
- Effective bits per weight: $4 + 8/32 = 4.25$ bits.

**MXINT4**:
- Each element: 4-bit signed integer.
- Shared scale: same 8-bit E8M0 per block of 32.
- Effective bits per weight: $4.25$ bits, but the integer mantissa gives uniform spacing
  rather than logarithmic spacing.

**Difference from standard per-tensor/per-channel INT4**:

| Property | Standard INT4 | MXINT4 |
|---|---|---|
| Scale granularity | Per tensor or per channel | Per 32 elements (fine-grained) |
| Scale format | FP16/BF16 | E8M0 (8-bit exponent only) |
| Scale storage overhead | Very low | $1/32 \times 1\text{B} = 3.1\%$ extra |
| Accuracy | Poor without group quantisation | Better — fine-grained captures local variance |
| Hardware standardisation | Vendor-specific | OCP standardised |

**Hardware requirements for MXINT4**:

1. **E8M0 scale decode**: The 8-bit shared exponent represents a power of 2. Decoding is
   a left shift operation — no multiplication hardware needed, just a barrel shifter.
   This is much cheaper than an FP16 multiply for dequantisation.

2. **Block-aligned fetch**: The memory controller must fetch 32-element blocks aligned to
   the 32-element group boundary. Unaligned access breaks the scale association.

3. **Scale register file**: One scale register per 32 weight elements being processed
   in parallel. For a 256-wide vector unit, this requires 8 scale registers per cycle.

4. **Dot product with mixed scales**: If all 32 elements in a block share the same scale,
   the dot product can be computed as:
   $$\text{output} = s \cdot \sum_{i=0}^{31} q_i x_i$$
   The integer dot product is computed first (cheap), then multiplied by $s$ (one FP multiply).
   This is the standard MX dot product decomposition.

---

### Q6. Why must accumulators be wider than the input precision in quantised MAC arrays? Derive the minimum accumulator width for INT8 and INT4 inputs.

**Answer**

**Why wider accumulators are necessary**:

A dot product $y = \sum_{k=0}^{K-1} w_k x_k$ accumulates $K$ products. Each product is
at most $2^{b_w - 1} \times 2^{b_x - 1}$ (max signed values). Summing $K$ such products:

$$|y| \leq K \cdot 2^{b_w + b_x - 2}$$

The accumulator must represent this without overflow:
$$b_{\text{acc}} \geq b_w + b_x - 2 + \log_2 K + 1 = b_w + b_x - 1 + \log_2 K$$

**INT8 case** ($b_w = b_x = 8$, typical $K = d = 4096$):
$$b_{\text{acc}} \geq 8 + 8 - 1 + \log_2 4096 = 15 + 12 = 27 \text{ bits}$$

In practice, 32-bit accumulators are universal for INT8 (27 bits rounded up, plus headroom
for partial-sum cascading in pipelined arrays).

**INT4 case** ($b_w = b_x = 4$, $K = 4096$):
$$b_{\text{acc}} \geq 4 + 4 - 1 + 12 = 19 \text{ bits}$$

20 or 24-bit accumulators are commonly used for INT4. Some implementations use INT32
accumulators regardless of input width to simplify the datapath (the extra bits add
modest area).

**Hardware area implication**: In a systolic array, each PE contains one accumulator.
For a $256 \times 256$ systolic array:
- INT8 with 32-bit acc: $256^2 \times 32 = 2\text{M}$ flip-flop bits just for accumulators
- INT4 with 20-bit acc: $256^2 \times 20 = 1.25\text{M}$ bits — 37.5% savings

**Common mistake in interviews**: Claiming that INT4 halves the accumulator width. The
accumulator width is not simply proportional to input width — the $\log_2 K$ term from
the accumulation depth dominates for large $K$.

---

### Q7. Explain the accuracy-hardware trade-off between per-tensor, per-channel, and per-group quantisation. What is the RTL cost of each?

**Answer**

**Quantisation granularity** refers to how many weights share a single scale factor $s$.
Finer granularity (more scale factors) generally improves accuracy because each scale
factor can track the local statistics of its weight group.

**Per-tensor quantisation** (one $s$ per weight matrix $W$):
- Accuracy: Poorest for INT4/INT8. Outlier weights distort the scale, crushing small weights.
- RTL cost: One scale register per matrix. Trivially cheap.
- Memory overhead: 1 FP16 value per matrix — negligible.

**Per-channel (per-row/column) quantisation** (one $s$ per output channel):
- Accuracy: Substantially better — each output neuron's weights are scaled independently.
- RTL cost: One scale per row of $W$ (for row-stationary layouts). Requires loading a
  scale vector alongside each weight row. The scale lookup is indexed by the row counter —
  one register per active row in the systolic array, typically 16–256 registers.
- Memory overhead: $d_{\text{out}} \times 2$ bytes per matrix — e.g., $4096 \times 2 = 8\text{KB}$,
  tiny relative to weight matrix ($\sim 90\text{MB}$ for FFN layer).

**Per-group quantisation** (one $s$ per $G$ consecutive weights, $G$ typically 32–128):
- Accuracy: Excellent, close to full-precision for INT4 with $G=128$.
- RTL cost: Scale changes every $G$ weight elements. Requires:
  1. A scale reload every $G$ elements — once per $G$-element vector unit width cycle.
  2. Address generation that maps element index to its group's scale address.
  3. For pipelined systolic arrays, the scale pipeline must be delayed by the same
     latency as the weight pipeline.
- Memory overhead: $N/G \times 2$ bytes. At $G=128$, this is $N/64$ bytes — about 3% of
  INT8 weight storage, or 6% of INT4 weight storage.

**Design choice summary**:

| Granularity | INT8 accuracy | INT4 accuracy | RTL complexity | Scale memory |
|---|---|---|---|---|
| Per-tensor | Good | Poor | Trivial | Negligible |
| Per-channel | Very good | Moderate | Low | Tiny |
| Per-group ($G=128$) | Excellent | Good | Medium | ~3–6% extra |
| Per-group ($G=32$) | Excellent | Very good | Medium-high | ~12–25% extra |

---

## Tier 3 — Advanced

### Q8. Design the dequantisation logic for a W4A16 matrix multiply engine that processes 256 weights per cycle. What are the critical path and area considerations?

**Answer**

**Specification**:
- 256 INT4 weights processed per cycle (128 bytes, since 2 per byte)
- FP16 activations
- Per-group scale, $G = 128$: 2 scale factors active per 256-element vector
- Output: 256 FP16 weight values fed to the MAC array

**Datapath block diagram**:

```
128 bytes from weight buffer
    |
    v
[Unpack] -- 256 x INT4 values (2 values per byte via bit-select)
    |
    v  (2 scale fetches per 256 elements)
[Scale select] -- scale_0 for elements [0:127], scale_1 for elements [128:255]
    |
    v
[INT4 -> INT8 sign extend] -- 256 x INT8 (sign-extended INT4)
    |
    v
[INT8 * FP16_scale] -- 256 multiplies: INT8 x FP16 -> FP16
    |
    v
256 x FP16 dequantised weights to MAC array
```

**Critical path analysis**:

The critical path runs through:
1. **Unpack** (bit-select): 1 gate delay — a 4:1 mux indexed by byte offset.
2. **Sign extension**: Trivial — replicate bit 3 to bits 4-7.
3. **INT8 to FP16 conversion**: INT8 has implicit exponent; conversion requires
   a leading-zero detector (priority encoder, $\sim 4$ gate levels) + normalisation shift.
4. **FP16 multiply** (scale application): FP16 multiplier critical path is typically
   $\sim 8\text{–}12$ FO4 delays in 7nm.

Total critical path for dequantisation: dominated by FP16 multiply, $\approx 8\text{–}12$ FO4.

**Area considerations**:

256 INT8-to-FP16 converters + 256 FP16 multipliers.
- FP16 multiplier: roughly 2000 gates at 7nm = $\sim 1000\mu m^2$
- 256 multipliers: $256\text{K}\mu m^2 = 0.256\text{mm}^2$

Compare to the MAC array itself: 256 FP16 MACs $\approx 256 \times 5000\text{ gates} = 1.28\text{M gates} \approx 1.28\text{mm}^2$.

The dequantisation unit is $\sim 20\%$ of the MAC array area — significant but not dominant.

**Optimisation**: Use BF16 arithmetic instead of FP16 for dequantisation. BF16 multiplies
are slightly cheaper (narrower mantissa multiplier), and BF16 to BF16 conversion avoids
precision mismatches. If the MAC array operates in BF16, dequantisation can output BF16
directly at lower cost.

---

### Q9. Quantisation introduces "quantisation-induced activation outliers" that are problematic for W8A8 schemes. Explain the hardware strategies (SmoothQuant, per-token activation quantisation) and their RTL implications.

**Answer**

**The outlier problem**:

In LLM activations, a small number of channels (often $< 1\%$) can have magnitudes
$100\text{–}1000\times$ larger than the typical channel. A per-tensor INT8 scale chosen to
accommodate these outliers assigns most of the quantisation range to values near zero,
giving those channels effectively only 1–2 bits of precision. This causes large quantisation
error and accuracy degradation.

**SmoothQuant**: Mathematically equivalent transform that migrates quantisation difficulty
from activations to weights:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

The scaled activation $\hat{X} = X \cdot \text{diag}(s)^{-1}$ and scaled weight
$\hat{W} = \text{diag}(s) \cdot W$ are both easier to quantise because the outlier magnitude
is split between them controlled by $s$.

**RTL implication of SmoothQuant**:
- $\hat{W}$ is computed offline during model preparation — no runtime overhead in the weight path.
- The activation scaling $X \cdot \text{diag}(s)^{-1}$ requires an element-wise multiply
  of the activation vector by the smooth scale vector $s^{-1}$ before quantisation.
- This is a vector-by-vector multiply: 1 FP16 multiply per activation element.
- The smooth scale vector $s^{-1} \in \mathbb{R}^{d}$ must be loaded (once per layer,
  trivially small: $d \times 2 = 8\text{KB}$ for $d=4096$).
- Net RTL addition: one FP16 multiply pipeline stage per activation element, before the
  quantise-to-INT8 unit.

**Per-token dynamic activation quantisation**:

Instead of a fixed static scale per tensor/channel, compute the scale of each token's
activation vector dynamically:
$$s_t = \frac{\max_j |X_{t,j}|}{127}$$

Each token's activations are then scaled independently before INT8 conversion.

**RTL implications**:
- Requires a **max-reduction** over $d$ elements per token: $\log_2 d$ reduction stages,
  e.g., 12 pipeline stages for $d=4096$.
- This reduction must complete before the activation is quantised — adds latency to the
  activation path.
- The per-token scale $s_t$ is a runtime value that must be stored (1 FP16 per token)
  and used during output dequantisation.
- **Throughput impact**: For prefill with $S$ tokens in a batch, the max-reduction runs
  in parallel across tokens (embarrassingly parallel) — no throughput cost.
- **Latency impact**: The reduction pipeline depth ($\sim 12$ cycles) adds to the
  activation quantisation latency; must be absorbed by the pipeline.

**Combined approach** (common in production): SmoothQuant offline + per-token dynamic
quantisation online. This handles both the static distribution mismatch (SmoothQuant) and
dynamic variation across tokens (per-token scaling).

---

### Q10. A hardware team is designing a new accelerator for LLM inference and must choose between FP8 E4M3 and INT8 for the MAC array. Walk through the decision criteria from an RTL perspective.

**Answer**

**Comparison axes**:

**1. Numeric representation**

INT8 uses uniform quantisation spacing — equal distance between all representable values.
FP8 E4M3 uses logarithmic spacing — more values near zero, fewer near the extremes.

For weight distributions in trained LLMs (approximately Gaussian, most values near zero),
FP8 provides better effective resolution for typical values but worse coverage for outliers.
For activations with outliers, this cuts both ways.

**2. MAC hardware complexity**

| Operation | INT8 | FP8 E4M3 |
|---|---|---|
| Multiplier | $8 \times 8$ integer | Floating-point: mantissa multiply + exponent add |
| Alignment | None (uniform grid) | Exponent comparison + mantissa shift |
| Adder (accumulation) | Integer adder | FP accumulation (more complex) |
| Overflow handling | Saturate | NaN / Inf propagation |
| Normalisation | None | Normalise per addition |

INT8 MAC is fundamentally simpler: the multiplier is a plain binary multiplier, and the
accumulator is an integer adder. FP8 requires exponent-alignment logic on every accumulation.

**Common approach**: Use FP8 for inputs, but convert to FP32 or FP16 for accumulation.
The NVIDIA H100 FP8 GEMM operates as: FP8 inputs, FP32 accumulators. This means the
accumulator cost is FP32 regardless — the benefit of FP8 is solely in the input bit-width
(bandwidth and storage), not in cheaper multiply-accumulate.

**3. Quantisation overhead**

INT8 requires explicit scale factors stored alongside weights. For per-tensor INT8:
one FP16 scale per matrix — negligible. For per-channel INT8: one FP16 per row — small.

FP8 tensors can sometimes be used without explicit scale factors (the exponent bits handle
the range), but in practice for model weights, a per-tensor or per-tensor scale is still
used to map the FP8 range to the weight distribution. The OCP MX specification uses FP8
with E8M0 block scales — same overhead as MXINT8.

**4. Toolchain and ecosystem**

INT8 inference is mature (NVIDIA TensorRT, PyTorch quantisation toolkit, ONNX runtime).
FP8 tooling is newer (NVIDIA Transformer Engine for H100, limited PyTorch native support).
An RTL team implementing FP8 must invest more in verification infrastructure.

**Decision framework**:

| Criterion | Favours INT8 | Favours FP8 |
|---|---|---|
| MAC area/power | Yes (simpler) | Slightly worse |
| Accuracy (no QAT) | Worse at outliers | Better for smooth distributions |
| Accuracy (with QAT) | Comparable | Comparable |
| Toolchain maturity | Yes | No |
| Standards compliance | Vendor-specific | OCP MX standard |
| Mixed-precision path | Well understood | Still evolving |
| Long-term support | Ubiquitous | Growing |

**Recommendation**: For a new accelerator targeting general LLM inference, INT8 with
per-group scaling is the lower-risk choice today. FP8 E4M3 with accumulation in FP32
is a strong choice if the target models include tasks with smooth activation distributions
(e.g., encoder-only BERT-like models) or if OCP MX ecosystem compliance is required.
For maximum accuracy at 8-bit, combine SmoothQuant + per-token dynamic INT8 quantisation,
which nearly closes the gap between INT8 and FP8 at lower RTL complexity.

---

## Quick Reference: Precision Formats for LLM Accelerators

| Format | Bits | Range | HW complexity | Typical use |
|---|---|---|---|---|
| FP32 | 32 | $\pm 3.4 \times 10^{38}$ | High | Training reference |
| BF16 | 16 | $\pm 3.4 \times 10^{38}$ | Medium | Prefill, training |
| FP16 | 16 | $\pm 65504$ | Medium | Prefill, activations |
| FP8 E4M3 | 8 | $\pm 448$ | Medium-high | Weights, activations |
| FP8 E5M2 | 8 | $\pm 57344$ | Medium-high | Gradients |
| INT8 | 8 | $[-128, 127]$ | Low | Weights W, activations X |
| INT4 | 4 | $[-8, 7]$ | Low | Weights only (W4A16) |
| MXFP4 | 4 + shared exp | — | Medium | Emerging |
| MXINT4 | 4 + shared exp | — | Low-medium | Emerging |

**Accumulator rules of thumb**:
- INT8 inputs $\to$ INT32 accumulator (26-bit minimum, 32-bit standard)
- INT4 inputs $\to$ INT24 accumulator (19-bit minimum, 24 or 32-bit standard)
- FP8 inputs $\to$ FP32 accumulator (industry standard)
- FP16/BF16 inputs $\to$ FP32 accumulator (standard for GEMM)
