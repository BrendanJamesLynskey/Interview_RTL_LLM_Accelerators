# Mixed-Precision Datapath Design

## Overview

Modern LLM inference never runs at a single precision. Weight quantisation (W4, W8), activation formats (FP16, BF16, INT8), and accumulation requirements (FP32, INT32) must coexist in the same datapath. Designing hardware that handles multiple precisions efficiently — without wasting area on precision conversions or introducing numerical errors — is a core interview topic for LLM accelerator roles.

---

## Tier 1 — Fundamentals

### Q1. What does "W8A8" mean and what hardware does it require?

**Question:** The shorthand W8A8 describes a model inference configuration. Define it precisely. What types of operations does the hardware need to support, and what are the input/output precisions at each stage of a linear layer?

**Answer:**

**W8A8 definition:**
- **W8:** Weights are stored and fetched as INT8 (8-bit signed integers).
- **A8:** Activations are represented as INT8 for the multiply-accumulate operations.
- Both operands to the MAC unit are INT8.

**Full precision flow through a W8A8 linear layer:**

```
Step 1: Weight load
  HBM/SRAM → INT8 weights (1 byte/element)

Step 2: Activation quantisation
  FP16/BF16 input activations → quantise → INT8
  (x_int8 = round(x_fp16 / scale_x), where scale_x is a per-tensor or per-channel float)

Step 3: Matrix multiply (MAC units)
  INT8 x INT8 → INT32 accumulator
  (product range: 16-bit; after K accumulations: up to 32-bit)

Step 4: Dequantisation / rescaling
  INT32 accumulator → multiply by (scale_x * scale_w) → FP32 or FP16 output
  (dequant: y_fp16 = acc_int32 * scale_x * scale_w)

Step 5: Bias add (optional)
  FP16 bias added to FP32 dequantised result

Step 6: Activation function (optional)
  ReLU, GELU, SiLU, etc. applied in FP32 or FP16

Step 7: Output quantisation (for next layer, if it is also W8A8)
  FP32/FP16 → INT8 again (for the next layer's activation input)
```

**Hardware requirements:**

| Operation | Hardware unit |
|---|---|
| INT8 x INT8 accumulate to INT32 | INT8 MAC array |
| FP16/BF16 → INT8 | Activation quantisation unit (scale, round, clamp) |
| INT32 × FP32 (dequant scale) | FP32 multiplier (small, few units needed) |
| INT32 + FP16 bias | Mixed INT32/FP16 adder |
| FP32/FP16 → INT8 (requant) | Requantisation unit |

**Common mistake:** Assuming the MAC outputs FP16 or BF16. In W8A8, MAC outputs are INT32. The conversion back to floating point happens *after* the full dot product is complete, not inside the MAC loop.

---

### Q2. What is the difference between per-tensor, per-channel, and per-group quantisation? How does each affect hardware complexity?

**Question:** Define per-tensor, per-channel, and per-group quantisation scales. For each, describe what storage is needed and where the dequantisation operation occurs in the hardware pipeline.

**Answer:**

All three are methods for associating a floating-point scale factor with quantised integers so that the original value can be approximately reconstructed.

**Per-tensor quantisation:**
- One scale value for the entire weight matrix (or activation tensor).
- Storage: 1 FP32 value per matrix. Negligible overhead.
- Dequantisation: multiply the entire INT32 accumulator output by `scale_w * scale_x` (a single FP32 multiply per output element).
- Hardware: one FP32 multiplier at the output of the reduction tree.
- Accuracy: lowest — a single scale cannot capture inter-channel variation in weight distributions.

**Per-channel (per-output-channel) quantisation:**
- One scale per output neuron (row of the weight matrix for a linear layer).
- For a 4096x4096 weight matrix: 4096 FP32 scale values = 16 KB of scale storage.
- Dequantisation: each row of the output uses a different scale. The dequant unit reads `scale_w[row_idx]` from a small SRAM and multiplies it with `scale_x` and the accumulated INT32 value.
- Hardware: the dequant SRAM must be accessed at the rate outputs are produced. For a 256-wide output array: 256 simultaneous scale reads per cycle.
- Accuracy: significantly better than per-tensor. Standard for W8A8 in production (used in bitsandbytes, GPTQ).

**Per-group quantisation:**
- One scale per group of G weights within a row (e.g., G=128).
- For W4 (4-bit weights): the group size is typically 128 elements, so each group has 1 FP16 scale and optionally 1 FP16 zero-point.
- Storage: (N x K / G) FP16 scale values. For 4096x4096, G=128: (4096*4096/128) * 2 bytes = 256 KB of scale storage.
- Dequantisation: the accumulation must be broken into K/G partial sums, each multiplied by the corresponding group scale before final summation. This means the accumulation cannot run K steps uninterrupted — it must flush and rescale every G steps.
- Hardware: more complex. Requires a partial accumulation buffer, per-group scale lookup, and FP32 multiply-and-accumulate every G MAC cycles.
- Accuracy: highest — essential for 4-bit weight quantisation where intra-channel variation is large.

**Comparison:**

| Scheme | Scale storage | Dequant location | Accuracy | HW complexity |
|---|---|---|---|---|
| Per-tensor | 1 FP32 | Post-accumulation | Low | Minimal |
| Per-channel | N FP32 | Post-accumulation per row | Good | Small SRAM + mux |
| Per-group (G=128) | N*K/G FP16 | Every G MAC cycles | High | Partial accum + flush |

---

### Q3. What is the purpose of a zero-point in quantisation and how does it affect the MAC unit?

**Question:** Asymmetric quantisation introduces a zero-point offset: `x_float = scale * (x_int - zero_point)`. How does this change the computation in the MAC unit? Show the expanded form of the dot-product with zero-points.

**Answer:**

**Asymmetric quantisation model:**

```
x_float = scale_x * (x_int - zp_x)
w_float = scale_w * (w_int - zp_w)
```

**Expanding the dot product:**

```
y = sum_k [ x_float[k] * w_float[k] ]
  = sum_k [ scale_x * (x_int[k] - zp_x) * scale_w * (w_int[k] - zp_w) ]
  = scale_x * scale_w * sum_k [ (x_int[k] - zp_x) * (w_int[k] - zp_w) ]
  = scale_x * scale_w * [ sum_k(x_int[k]*w_int[k])
                         - zp_w * sum_k(x_int[k])
                         - zp_x * sum_k(w_int[k])
                         + K * zp_x * zp_w ]
```

The four terms:
1. `sum_k(x_int[k] * w_int[k])` — standard INT8 x INT8 dot product.
2. `zp_w * sum_k(x_int[k])` — scale_w zero-point times sum of activations.
3. `zp_x * sum_k(w_int[k])` — scale_x zero-point times sum of weights.
4. `K * zp_x * zp_w` — a constant (precomputed offline, absorbed into bias).

**Hardware implications:**

- Term 1 is computed by the standard MAC array.
- Term 4 is a constant per weight row and zero-point pair; it is precomputed and absorbed into the bias vector.
- Term 3 `zp_x * sum_w` is also precomputed offline (weight sum per output channel, multiplied by activation zero-point at inference time). This requires broadcasting `zp_x` and having per-channel `sum_w` stored as bias-like constants.
- Term 2 `zp_w * sum_x` is an online computation: the sum of each activation vector multiplied by the weight zero-point. This cannot be precomputed (activations change each inference step). It requires an additional reduction (sum of x_int) computed in parallel with the main MAC.

**Practical consequence:** Symmetric quantisation (zp_w = 0, zp_x = 0) eliminates terms 2, 3, 4 entirely, making the hardware significantly simpler. Most hardware accelerators (NVIDIA TensorRT, Apple ANE) prefer symmetric weight quantisation for this reason. Activation zero-points are also often set to zero by requiring activations to be zero-centred before quantisation (which can be enforced by design choices in quantisation-aware training).

---

## Tier 2 — Intermediate

### Q4. How do you design a datapath that supports both W8A8 (INT8 x INT8) and W8A16 (INT8 x FP16) modes?

**Question:** An accelerator must support two modes: W8A8 with INT32 accumulation, and W8A16 (weight INT8, activation FP16) with FP32 accumulation. Describe the datapath changes needed to support both modes on the same MAC array, including precision conversion hardware.

**Answer:**

**W8A8 mode:** Both inputs are INT8. The MAC unit is an INT8 x INT8 multiply with INT32 accumulation.

**W8A16 mode:** Weights are INT8, activations are FP16. Options:

*Option A: Dequantise weights to FP16 before multiply (lazy dequant).*
- Convert INT8 weight → FP16 using a per-channel scale: `w_fp16 = w_int8 * scale_w`.
- Then compute FP16 x FP16 with FP32 accumulation.
- Requires: INT8-to-FP16 converter in the weight path (multiply by FP16 scale, output FP16).
- MAC unit becomes: FP16 x FP16 → FP32.
- This approach wastes the INT8 storage benefit — once converted to FP16 for compute, the bandwidth advantage of INT8 is only realised in the memory load path, not in compute.
- Area: Both INT8 MAC array AND FP16 MAC array, or a unified array that is larger.

*Option B: Convert activation to INT8 before multiply (activation quantisation online).*
- Apply an online activation quantiser: FP16 → scale → round → clamp → INT8.
- Requires: per-tensor or per-token dynamic quantisation unit in the activation path.
- The INT8 MAC array handles the computation; the output is dequantised with `scale_x * scale_w` as usual.
- This is essentially W8A8 with online activation quantisation.
- Area: Small quantisation unit added to activation path.

*Option C: Unified mixed datapath with mode select.*

The MAC unit contains:
- An INT8 x INT8 multiplier with INT32 accumulator (for W8A8).
- A conversion path that takes INT8 weight → extend sign to INT16 or INT32 → multiply with FP16 activation in a shared multiply-add with FP32 accumulator.

However, INT8 and FP16 use completely different number representations (two's complement vs. IEEE 754 floating-point). A single unified multiplier that handles both requires a reconfigurable design — effectively two separate multipliers with a mux, which costs as much area as two separate MAC units.

**Practical recommendation for hardware design:**

Use Option B (online INT8 quantisation of activations) for the W8A16 path when the model supports it. When W8A16 is architecturally required (e.g., the model has activations that cannot be quantised to INT8 without unacceptable accuracy loss), implement separate FP16 MAC units for those layers and use an overlay tiling scheme that sends certain layers to the INT8 path and others to the FP16 path.

**Mode select signals needed:**
- `precision_mode[1:0]`: 00=INT8, 01=FP16, 10=BF16, 11=reserved.
- `act_quant_en`: enable online activation quantisation (for W8A16 via Option B).
- `acc_fp32_en`: enable FP32 accumulation (disables INT32 acc path).

---

### Q5. Describe the hardware needed for INT4 weight dequantisation in the weight load path for a W4A16 datapath.

**Question:** In W4A16 inference, weights are stored as INT4 and activations as FP16. The MAC unit operates in FP16. Describe the dequantisation unit that converts INT4 weights to FP16, including where it sits in the pipeline, what scales it needs, and the throughput requirement.

**Answer:**

**W4A16 data flow:**

```
HBM → INT4 weights (packed 2/byte) → dequant unit → FP16 weights → MAC array (FP16 x FP16 → FP32 acc)
```

**Dequant unit inputs:**

- `w_int4[3:0]`: 4-bit signed integer weight (range -8 to +7).
- `scale[15:0]`: FP16 scale factor, one per group of G weights (G=128 typically).
- `zero_point[3:0]` (optional, for asymmetric quantisation): INT4 zero-point.

**Dequant unit operation:**

```
w_float = scale * (w_int4 - zero_point)
```

Hardware steps:
1. **Sign-extend INT4 to INT8:** `w_int8 = {{4{w_int4[3]}}, w_int4}` (1 cycle, combinational).
2. **Subtract zero_point (if asymmetric):** INT8 subtractor. Result: INT8, range -15 to +15.
3. **INT8 to FP16 conversion:** Convert signed INT8 to FP16. This involves:
   - Count leading zeros of the magnitude (find exponent).
   - Normalise the mantissa.
   - Construct FP16 with sign, exponent, and 4-bit mantissa (INT8 has only 7 significant bits, FP16 has 10-bit mantissa, so there is no precision loss).
   - This takes approximately 3-4 gate levels.
4. **FP16 multiply by scale:** `w_fp16 = (FP16)w_int8 * scale_fp16`. Full FP16 multiply: ~2 ns critical path.

**Throughput requirement:**

For a 256-wide FP16 MAC array running at 1 GHz:
- The array consumes 256 weights per cycle (one per column).
- Weights are INT4 = 0.5 bytes each.
- Required dequant throughput: 256 elements/cycle at 1 GHz = 256 billion dequant ops/second.
- 256 parallel dequant units must be instantiated (one per MAC column).
- Each unit processes one INT4 → FP16 conversion per cycle.
- Pipeline depth of dequant unit: typically 2-3 cycles. This creates a latency offset between the weight load and the MAC, which must be compensated by inserting corresponding pipeline delay on the activation path.

**Area estimate:**

Each dequant unit: sign-extend (free) + INT8-sub (4 cells) + INT8-to-FP16 (12-15 cells) + FP16-mul (80 cells) ≈ ~100 standard cells per unit.
For 256 units: ~25,600 cells ≈ 0.05 mm^2 at 7nm.
By comparison, the 256-wide FP16 MAC array: 256 x ~500 cells = ~128,000 cells ≈ 0.25 mm^2.
Dequant overhead: ~20% area overhead on top of the MAC array.

**Scale management:**

Scales must be loaded in synchrony with weights. For G=128 per-group scales, one new scale is needed every 128 weight columns. The scale buffer holds 2 scales per column (current group + next group prefetched): 256 x 2 x 2 bytes = 1 KB. This fits in a small register file next to the dequant units.

---

### Q6. How do you handle accumulation precision when INT8 MACs feed an FP16 output in mixed-precision mode?

**Question:** In a W8A8 layer, accumulation is in INT32. The output must be written back as BF16 for the next layer (which operates in BF16). Describe the conversion pipeline from INT32 accumulator to BF16 output, including the dequantisation step and potential sources of numerical error.

**Answer:**

**Full conversion pipeline:**

```
INT32 acc → [dequant scale multiply] → FP32 → [optional bias add] → [BF16 convert] → output
```

**Step 1: INT32 to FP32 conversion**

`acc_fp32 = (float)acc_int32` — exact for all INT32 values (FP32 has 24-bit mantissa, INT32 has 32 significant bits, so INT32 values with |acc| > 2^24 lose precision here).

Implication: For large K (many accumulations), the INT32 accumulator can reach values of up to K * 128^2 = 4096 * 16384 = ~67M, which is well below 2^24 = 16.7M? No — 67M = 6.7 * 10^7, 2^24 = 16.7M. So for K=4096 with worst-case inputs, INT32 to FP32 conversion is *not* exact (64 bits of INT32 value vs. 24-bit FP32 mantissa precision above 16.7M). Values above 16.7M will be rounded to the nearest FP32 representable value.

**Is this a problem?**

The INT32 accumulator itself only overflows at 2^31 ≈ 2.15 billion. For K=4096, worst case = 67.1M, which fits in INT32 without overflow.

But INT32-to-FP32 cast rounds values above 2^24 to the nearest multiple of 2^(exp-23). For 67M: the ULP (unit in the last place) is 67M/2^23 ≈ 8. So the cast can introduce an error of up to ±4 in the integer accumulator value.

For a quantised model where the output range is typically within ±127 after rescaling, a ±4 error in INT32 translates to a ±4 / K = ±0.001 error in the float output — negligible.

**Step 2: Dequantisation (FP32 scale multiply)**

`y_fp32 = acc_fp32 * scale_x * scale_w`

This is one FP32 multiply. If per-channel scales: the scale is looked up from a small SRAM (one FP32 value per output channel).

**Step 3: Bias addition (FP32)**

`y_fp32 = y_fp32 + bias_fp32`

**Step 4: Activation function (optional)**

GELU, SiLU: computed in FP32 using polynomial approximation or lookup table. Output stays in FP32.

**Step 5: BF16 rounding**

`y_bf16 = (bfloat16)y_fp32`

BF16 has 7 mantissa bits vs. FP32's 23. The conversion truncates or rounds the lower 16 bits of the FP32 mantissa.

```systemverilog
// FP32 to BF16 truncation (round to nearest even):
// FP32 = {sign[1], exp[8], mantissa[23]}
// BF16 = {sign[1], exp[8], mantissa[7]}
// Lower 16 bits of mantissa are rounded away

logic [31:0] fp32_in;
logic [15:0] bf16_out;

// Round-to-nearest-even:
logic [15:0] mantissa_low = fp32_in[15:0];
logic round_bit = fp32_in[15];
logic sticky = |fp32_in[14:0];
logic lsb_of_result = fp32_in[16];
logic round_up = round_bit & (sticky | lsb_of_result);

assign bf16_out = fp32_in[31:16] + {15'b0, round_up};
// Handle exponent overflow edge case (if mantissa overflows to increment exponent)
```

**Sources of error at each step:**

| Step | Error source | Magnitude |
|---|---|---|
| INT32 acc | Overflow (extremely rare) | Catastrophic if occurs |
| INT32 → FP32 | Precision loss for values > 2^24 | < 1 ULP of FP32 at that scale |
| FP32 scale multiply | FP32 rounding | ≈ 1 ULP of FP32 |
| BF16 rounding | 16-bit mantissa truncation | ≈ 1 ULP of BF16 ≈ 4 ULP of FP32 |
| Total | Chain of rounding | Typically < 0.1% relative error |

---

## Tier 3 — Advanced

### Q7. Design a datapath that simultaneously supports W8A8 and W4A16 modes, sharing as much hardware as possible.

**Question:** You are designing an accelerator that must efficiently support both W8A8 (for large batch inference) and W4A16 (for decode). Describe a shared datapath architecture. What is the area overhead of supporting both modes vs. a single-mode design?

**Answer:**

**Workload requirements:**

- W8A8: INT8 x INT8 → INT32. High arithmetic intensity (prefill / large batch). Maximise INT8 MACs/mm^2.
- W4A16: INT4 weight → dequant to FP16 → FP16 x FP16 → FP32. Low arithmetic intensity (decode). Maximise weight bandwidth utilisation.

**Key insight: the fundamental units are different.**

INT8 MACs and FP16 MACs are architecturally incompatible at the multiplier level — you cannot share the core multiplier between them without paying full cost for both. The question is how to share everything *around* the multipliers.

**Proposed shared architecture:**

```
                       ┌─────────────────┐
Weight DMA ────────────► Weight SRAM      ├──────────────────────────────┐
(INT8 or INT4 packed)  └─────────────────┘                              │
                                                                         ▼
Activation DMA ────────► Act SRAM (FP16) ──────►  Mode MUX ──► INT8 Quant  ──► INT8 Act
                                                        │
                                                        └──────────────────► FP16 Act

Weight SRAM ─────────────────────────────► Mode MUX ──► INT8 weights ──► INT8 MAC array ──► INT32 Acc
                                                  │
                                                  └──► INT4 dequant → FP16 ──► FP16 MAC array ──► FP32 Acc

INT32 Acc ─────────────────────────────────► Dequant (scale * acc) ──► BF16/FP16 out
FP32 Acc  ─────────────────────────────────► BF16 round ──────────────► BF16/FP16 out
```

**Shared components (no duplication):**

1. Weight SRAM: shared. In W8A8 mode, stores INT8. In W4A16 mode, stores INT4 (twice the weight tiles fit in the same SRAM).
2. Activation SRAM: shared. In W8A8 mode, activations pass through the INT8 quantiser before the INT8 MAC array. In W4A16 mode, FP16 activations go directly to the FP16 MAC array.
3. Output SRAM: shared. BF16 output is the common interface regardless of compute precision.
4. DMA engines, tiling controllers, address generators: fully shared.

**Non-shared components (precision-specific):**

1. INT8 MAC array (for W8A8).
2. FP16 MAC array (for W4A16, after dequant).
3. INT4 dequant unit (for W4A16 weight path only).
4. INT8 activation quantiser (for W8A8 activation path only).

**Area trade-off:**

If we target a 256x256 array:

| Component | W8A8-only area | W4A16-only area | Shared (both) |
|---|---|---|---|
| INT8 MAC 256x256 | 1.0x | 0 | 1.0x |
| FP16 MAC 256x256 | 0 | 2.5x | 2.5x |
| INT4 dequant (256-wide) | 0 | 0.2x | 0.2x |
| INT8 act quant | 0.05x | 0 | 0.05x |
| Shared overhead | 0.3x | 0.3x | 0.3x |
| **Total** | **1.35x** | **3.0x** | **4.05x** |

A pure W8A8-only design (1.35x) vs. a W4A16-only design (3.0x) vs. both (4.05x).

Supporting both modes costs 4.05x vs. 1.35x (W8A8 only) or 4.05x vs. 3.0x (W4A16 only). The incremental cost of adding W8A8 to a W4A16 chip is 1.05x, which is relatively modest because INT8 MACs are small compared to FP16 MACs.

**Practical conclusion:** Most commercial accelerators targeting both prefill (W8A8) and decode (W4A16) include both arrays. The FP16 array dominates the area budget. The INT8 array is a relatively cheap addition. This is consistent with the NVIDIA H100 architecture, which includes both INT8 Tensor Cores and FP16 Tensor Cores.

---

### Q8. How do scaling factors propagate through a multi-layer transformer when all linear layers are quantised, and what hardware is needed to manage this?

**Question:** Consider a transformer block with self-attention (Q, K, V projections, attention, output projection) and FFN (two linear layers), all W8A8 quantised. Describe the chain of scale factors that must be tracked, where dequantisation happens, and how the hardware manages scale factor storage and application.

**Answer:**

**Scale factor chain for one transformer block:**

Let the input activation to the block have quantisation scale `s_x0` (known from the previous block's output requantisation step).

```
1. Q projection: y = W_Q * x
   - Compute: INT32 acc = W_Q_int8 (scale: s_wq) * x_int8 (scale: s_x0)
   - Dequant scale: s_wq * s_x0
   - Output: FP32 → BF16 (for softmax path) OR requantise to INT8 (for memory bandwidth)

2. K projection: same structure, scale chain s_wk * s_x0

3. V projection: same structure, scale chain s_wv * s_x0

4. Attention scores: Q * K^T
   - If Q,K are in BF16: standard BF16 matmul, no quantisation scale chain.
   - If Q,K are INT8: scale of scores = s_q * s_k, where s_q = (s_wq * s_x0) if requantised.
   - Softmax: must be in FP16 or FP32 to avoid numerical issues. Dequant before softmax.

5. Attention output: scores * V
   - If scores BF16, V INT8: mixed-precision path.
   - Output dequant scale: s_scores * s_wv * s_x0 (if fully INT8 chain).

6. Output projection: W_O * attn_out
   - Scale chain: s_wo * s_attn_out (where s_attn_out was determined at softmax output).

7. Residual add (x + attn_out):
   - If both are BF16: straightforward.
   - If one is INT8 and the other BF16: must dequant INT8 to BF16 first.
   - The residual add destroys the INT8 representation; output is in BF16.

8. LayerNorm:
   - Always in FP32 or BF16. Requantises output to INT8 for next layer.

9. FFN layer 1: similar to Q/K/V projection.
10. FFN activation (GELU/SiLU): must be in FP16/FP32; dequant before activation function.
11. FFN layer 2: requantise after activation, run INT8 MAC.
```

**Scale factor storage requirements per layer:**

| Scale | Size | Location |
|---|---|---|
| s_wq, s_wk, s_wv, s_wo | N FP32 per proj (per-channel) | Loaded with weight tiles |
| s_x0 (input activation) | 1 FP32 (per-tensor) | Config register |
| s_ffn1, s_ffn2 | N FP32 (per-channel) | Loaded with weight tiles |
| Per-group scales (if W4) | N*K/G FP16 per matrix | Packed with weight tile in SRAM |

**Hardware management of scale factors:**

1. **Scale SRAM / scratchpad:** A small dedicated SRAM (typically 64-256 KB) holds all scale factors for the current layer. Scales are DMA'd in ahead of the weight tiles and held resident for the duration of the layer.

2. **Scale broadcast network:** For per-channel scales, each output column of the MAC array needs its own scale value. A 256-wide scale broadcast register file (256 entries of FP32) is updated at the start of each N-tile and held for that tile's accumulation.

3. **Requantisation pipeline:** At the output of each linear layer, a requantisation unit:
   - Reads INT32 accumulator.
   - Multiplies by `scale_w[col] * scale_x` (FP32 multiply).
   - Optionally adds bias.
   - Rounds to INT8 or BF16 based on the next layer's requirements.
   - The requant scale for the *output* (`scale_out`) is computed by the runtime as `1 / (scale_w * scale_x)` where `scale_out` is set to target a dynamic range of [-128, 127] for the expected output distribution.

4. **Runtime scale computation:** The output quantisation scale `scale_out` cannot be known at compile time if activations are dynamically quantised (per-token dynamic quantisation). In this case:
   - After dequantisation, the hardware computes the absolute maximum of the output vector (hardware reduction tree or software on the host).
   - `scale_out = max_abs / 127`.
   - This scale is written back to the scale SRAM for the next layer's use.
   - Overhead: one abs-max reduction per linear layer output, typically 10-20 cycles on a parallel reduction unit.

**Critical design insight:** The scale factor chain must be managed *exactly* to maintain numerical correctness. A common implementation bug is using the wrong scale for a layer (e.g., using the weight scale without the activation scale), which causes systematic numerical errors that are often not caught by functional simulation but show up as accuracy degradation.

---
