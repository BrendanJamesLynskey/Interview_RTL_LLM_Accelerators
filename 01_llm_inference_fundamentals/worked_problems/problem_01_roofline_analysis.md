# Worked Problem 01: Roofline Analysis for an LLM Accelerator

## Problem Statement

You are given the following accelerator specification and must perform a full roofline analysis
for specific LLM inference operations. Show all working.

**Accelerator spec (call it "Apex-1")**:
- Peak BF16 compute: $\Pi = 400\ \text{TFLOP/s}$
- Peak HBM bandwidth: $\beta = 5\ \text{TB/s}$
- On-chip SRAM: $64\ \text{MB}$

**Model: LLaMA-style 13B**:
- Layers: $L = 40$
- Hidden dimension: $d = 5120$
- FFN intermediate: $d_{ff} = 13824$ (SwiGLU; three matrices: gate, up, down)
- Attention heads: $h = 40$, $d_h = 128$
- Number format: BF16 for weights and activations ($2$ bytes per element)

**Tasks**:

1. Calculate the ridge-point intensity $I^*$ of Apex-1.
2. For the FFN down-projection ($W_{\text{down}} \in \mathbb{R}^{d \times d_{ff}}$), compute
   arithmetic intensity for:
   (a) Prefill with sequence length $S = 2048$, batch $B = 1$
   (b) Decode with batch size $B = 1$
   (c) Decode with batch size $B = 80$
3. For each of the three scenarios, determine whether the operation is compute-bound or
   memory-bandwidth-bound, and compute the achievable throughput (TFLOP/s) and the time
   to execute this single matrix operation.
4. Plot the roofline diagram (as an ASCII diagram) and mark the three operating points.
5. Identify what batch size is needed to reach the ridge point during decode for this operation.
6. Discuss how INT4 weight quantisation changes the analysis.

---

## Part 1: Ridge-Point Intensity

The ridge point $I^*$ is the arithmetic intensity at which the compute roof and the memory
bandwidth roof intersect:

$$I^* = \frac{\Pi}{\beta} = \frac{400 \times 10^{12}\ \text{FLOP/s}}{5 \times 10^{12}\ \text{Byte/s}} = 80\ \text{FLOP/Byte}$$

**Interpretation**: Any operation with arithmetic intensity $I > 80\ \text{FLOP/Byte}$ is
compute-bound on Apex-1. Any operation with $I < 80\ \text{FLOP/Byte}$ is memory-bandwidth-bound.

---

## Part 2: Arithmetic Intensity of the FFN Down-Projection

The down-projection is the linear map:

$$Y = X W_{\text{down}}, \quad X \in \mathbb{R}^{(B \cdot S) \times d_{ff}},\ W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$$

### FLOPs

Each output element requires $d_{ff}$ multiply-add operations = $2 d_{ff}$ FLOPs.
There are $B \cdot S \cdot d$ output elements.

$$\text{FLOPs} = 2 \cdot B \cdot S \cdot d_{ff} \cdot d$$

$$= 2 \times B \times S \times 13824 \times 5120$$

$$= 141,557,760 \cdot B \cdot S \approx 1.416 \times 10^8 \cdot B \cdot S \ \text{FLOPs}$$

### Bytes transferred

**Weights** ($W_{\text{down}}$, loaded once per forward pass, BF16):

$$\text{Bytes}_W = d_{ff} \times d \times 2 = 13824 \times 5120 \times 2 = 141,557,760 \approx 141.6\ \text{MB}\ (135.0\ \text{MiB})$$

**Input activations** ($X$, loaded from previous layer output, BF16):

$$\text{Bytes}_X = B \cdot S \cdot d_{ff} \times 2 = B \cdot S \times 13824 \times 2 = 27,648 \cdot B \cdot S\ \text{bytes}$$

**Output activations** ($Y$, written to memory, BF16):

$$\text{Bytes}_Y = B \cdot S \cdot d \times 2 = B \cdot S \times 5120 \times 2 = 10,240 \cdot B \cdot S\ \text{bytes}$$

**Total bytes**:

$$\text{Bytes}_{\text{total}} = 141,557,760 + (27,648 + 10,240) \cdot B \cdot S = 141,557,760 + 37,888 \cdot B \cdot S$$

**Note on weight reuse**: The weight matrix is loaded once from HBM regardless of $B$ and $S$.
This is the key insight — as $B \cdot S$ increases, FLOPs scale but the weight transfer cost
does not.

---

### (a) Prefill: $S = 2048$, $B = 1$

$$\text{FLOPs} = 1.416 \times 10^8 \times 1 \times 2048 = 2.90 \times 10^{11}\ \text{FLOPs}$$

$$\text{Bytes} = 141,557,760 + 37,888 \times 2048 = 141,557,760 + 77,594,624 = 219,152,384 \approx 219.2\ \text{MB}$$

$$I_{\text{prefill}} = \frac{2.90 \times 10^{11}}{2.19 \times 10^8} \approx \mathbf{1323\ \text{FLOP/Byte}}$$

$I_{\text{prefill}} = 1323 \gg I^* = 80$ $\Rightarrow$ **Compute-bound**.

---

### (b) Decode: $B = 1$, $S = 1$ (single token step)

$$\text{FLOPs} = 1.416 \times 10^8 \times 1 \times 1 = 1.416 \times 10^8\ \text{FLOPs}$$

$$\text{Bytes} = 141,557,760 + 37,888 \times 1 = 141,595,648 \approx 141.6\ \text{MB}$$

The activation transfer is negligible ($37,888\ \text{B} = 37\ \text{KB}$) relative to weight transfer.

$$I_{\text{decode, B1}} = \frac{1.416 \times 10^8}{1.416 \times 10^8} \approx \mathbf{1.00\ \text{FLOP/Byte}}$$

$I_{\text{decode}} = 1 \ll I^* = 80$ $\Rightarrow$ **Memory-bandwidth-bound**.

This result generalises: for a GEMV (batch=1 decode), the number of FLOPs equals the number
of weight bytes (in BF16), giving exactly 1 FLOP/Byte.

---

### (c) Decode: $B = 80$, $S = 1$

$$\text{FLOPs} = 1.416 \times 10^8 \times 80 = 1.133 \times 10^{10}\ \text{FLOPs}$$

$$\text{Bytes} = 141,557,760 + 37,888 \times 80 = 141,557,760 + 3,031,040 = 144,588,800 \approx 144.6\ \text{MB}$$

$$I_{\text{decode, B80}} = \frac{1.133 \times 10^{10}}{1.446 \times 10^8} \approx \mathbf{78.4\ \text{FLOP/Byte}}$$

$I \approx 78.4 \approx I^* = 80$ $\Rightarrow$ **Near ridge point** (very slightly memory-bound).

---

## Part 3: Achievable Throughput and Execution Time

The roofline model gives attainable performance:

$$P_{\text{attain}} = \min(\Pi,\ \beta \cdot I)$$

For each scenario, execution time:

$$T = \frac{\text{FLOPs}}{P_{\text{attain}}}$$

### (a) Prefill ($I = 1323$, compute-bound)

$$P_{\text{attain}} = \min(400 \text{ TFLOP/s},\ 5 \text{ TB/s} \times 1323) = \min(400,\ 6615) = 400\ \text{TFLOP/s}$$

$$T_{\text{prefill}} = \frac{2.90 \times 10^{11}}{4 \times 10^{14}} = 7.25 \times 10^{-4}\ \text{s} \approx \mathbf{0.725\ \text{ms}}$$

Hardware utilisation: $P_{\text{attain}} / \Pi = 100\%$ (compute-bound, fully utilised).

### (b) Decode $B=1$ ($I = 1$, memory-bound)

$$P_{\text{attain}} = \min(400 \text{ TFLOP/s},\ 5 \text{ TB/s} \times 1) = \min(400,\ 5) = 5\ \text{TFLOP/s}$$

$$T_{\text{decode,B1}} = \frac{1.416 \times 10^8}{5 \times 10^{12}} = 2.83 \times 10^{-5}\ \text{s} \approx \mathbf{28.3\ \mu\text{s}}$$

Hardware utilisation: $5 / 400 = 1.25\%$ (severe underutilisation).

**Sanity check**: Time to stream the weights at bandwidth $\beta$:
$$T_{\text{stream}} = \frac{141.6 \text{ MB}}{5 \text{ TB/s}} = \frac{1.416 \times 10^8}{5 \times 10^{12}} = 28.3\ \mu\text{s}$$
These match, confirming the memory-bound analysis. (Activation I/O is only $37,888\ \text{B}$, about
$0.03\%$ of the bytes moved. Take care with units here: $141,557,760\ \text{B}$ is $135.0\ \text{MiB}$
but $141.6\ \text{MB}$; dividing the MiB figure by a decimal TB/s would wrongly give $27.0\ \mu\text{s}$.)

### (c) Decode $B=80$ ($I = 78.4$, near ridge point)

$$P_{\text{attain}} = \min(400,\ 5 \times 78.4) = \min(400,\ 392) = 392\ \text{TFLOP/s}$$

$$T_{\text{decode,B80}} = \frac{1.133 \times 10^{10}}{3.92 \times 10^{14}} = 2.89 \times 10^{-5}\ \text{s} \approx \mathbf{28.9\ \mu\text{s}}$$

Hardware utilisation: $392/400 = 98\%$.

**Note**: At $B=80$ we perform $80\times$ more useful work ($80\times$ more tokens generated
per step) in essentially the same time ($28.9\ \mu\text{s}$ vs $28.3\ \mu\text{s}$). Throughput
(tokens/second) is $80\times$ higher while latency per step barely changes.

---

## Part 4: ASCII Roofline Diagram

```
Performance
(TFLOP/s)
  400 |----------------------------#####################  <-- Compute roof (Pi = 400 TFLOP/s)
      |                          /
      |                         /  (A) Prefill [x]
      |                        /       S=2048, I=1323
      |                       /
  200 |                      /
      |                     /
      |                    /
      |                   /        (C) Decode B=80 [+]
  ~392|.................../..........+
      |                 /
      |                /
      |               /  <- Memory bandwidth roof (slope = beta = 5 TB/s)
      |              /       slope = 5 TFLOP/s per FLOP/Byte
      |             /
    5 |............*         (B) Decode B=1 [*]
      |           /              I=1, Perf=5 TFLOP/s
      |          /
      |         /
      +----+----+----+----+----+----+----+----+---> Arithmetic
      0   10   20   40   60   80  100  200  1000+  Intensity
                              ^                    (FLOP/Byte)
                              I* = 80 FLOP/Byte
                              (Ridge Point)
```

**Points summary**:

| Point | Scenario | $I$ [FLOP/B] | $P_{\text{attain}}$ [TFLOP/s] | Bottleneck |
|---|---|---|---|---|
| (A) | Prefill $S=2048, B=1$ | 1323 | 400 | Compute |
| (B) | Decode $B=1$ | 1.0 | 5 | Memory BW |
| (C) | Decode $B=80$ | 78.4 | 392 | Near ridge |

---

## Part 5: Batch Size for Ridge Point

We want $I \geq I^* = 80$.

$$I = \frac{2 \cdot B \cdot d_{ff} \cdot d}{2 \cdot d_{ff} \cdot d + 2 \cdot B \cdot (d_{ff} + d)} \approx B \quad \text{for } B \ll d_{ff}, d$$

More precisely, set $I = 80$:

$$\frac{2B \cdot d_{ff} \cdot d}{2 d_{ff} \cdot d + 2B(d_{ff} + d)} = 80$$

$$2B \cdot d_{ff} \cdot d = 80 \left[ 2 d_{ff} d + 2B(d_{ff}+d) \right]$$

$$B \cdot d_{ff} d - 80 B(d_{ff}+d) = 80 d_{ff} d$$

$$B \left[ d_{ff} d - 80(d_{ff}+d) \right] = 80 d_{ff} d$$

$$B = \frac{80 \times d_{ff} \times d}{d_{ff} d - 80(d_{ff}+d)}$$

Substituting $d_{ff} = 13824$, $d = 5120$:
$$d_{ff} d = 70,778,880$$
$$d_{ff}+d = 18,944$$
$$80(d_{ff}+d) = 1,515,520$$

$$B = \frac{80 \times 70,778,880}{70,778,880 - 1,515,520} = \frac{5,662,310,400}{69,263,360} \approx \mathbf{81.7}$$

So $B = 82$ tokens (or batch of 82 concurrent requests) reaches the ridge point for this layer.

**Simplified approximation**: For large weight matrices where activation I/O is negligible
relative to weight I/O, $I \approx B$ (in FLOP/Byte for BF16), so the ridge-point batch size
is simply:
$$B_{\text{ridge}} \approx I^* = 80$$

---

## Part 6: Effect of INT4 Weight Quantisation

**Changed quantity**: Weight bytes quartered (INT4 = 0.5 bytes/param vs 2 for BF16):

$$\text{Bytes}_W^{\text{INT4}} = 13824 \times 5120 \times 0.5 = 35,389,440 \approx 35.4\ \text{MB}$$

**New ridge point**: The accelerator's $\Pi$ stays the same (compute throughput is unchanged),
but if we assume the INT4 dequantisation can match MAC throughput (reasonable for modern
designs), $\Pi$ is effectively unchanged.

However, for the **roofline of this specific operation**, the effective bandwidth demand
decreases because fewer bytes need to be transferred. The ridge point for the accelerator
itself is unchanged at $I^* = 80$.

**Arithmetic intensity for decode ($B=1$)**:

$$I_{\text{decode, INT4}}^{B=1} = \frac{2 \times 1 \times 13824 \times 5120}{0.5 \times 13824 \times 5120} = \frac{2}{0.5} = 4\ \text{FLOP/Byte}$$

Still memory-bandwidth-bound, but intensity increases from 1 to 4 FLOP/Byte.

**TPOT improvement**:

$$P_{\text{attain}}^{\text{INT4}} = \min(400,\ 5 \times 4) = 20\ \text{TFLOP/s}$$

$$T_{\text{decode, INT4}}^{B=1} = \frac{1.416 \times 10^8}{2 \times 10^{13}} = 7.1\ \mu\text{s}$$

Compare to BF16: $28.3\ \mu\text{s}$. INT4 is $4\times$ faster — exactly the ratio of bytes saved.

**Ridge-point batch size with INT4**:

$$B_{\text{ridge}}^{\text{INT4}} \approx \frac{I^*}{\text{FLOP/Byte per request}} = \frac{80}{4} = 20$$

INT4 quantisation reduces the batch size needed to saturate compute from $\approx 80$ to
$\approx 20$, making it much easier to achieve high hardware utilisation at lower request rates.

**Summary of INT4 impact**:

| Metric | BF16 | INT4 | Change |
|---|---|---|---|
| Weight bytes (this layer) | 141.6 MB | 35.4 MB | $-4\times$ |
| Decode $B=1$ intensity | 1 FLOP/B | 4 FLOP/B | $+4\times$ |
| Decode $B=1$ TFLOP/s | 5 | 20 | $+4\times$ |
| Decode $B=1$ latency | 28.3 $\mu$s | 7.1 $\mu$s | $-4\times$ |
| Ridge-point batch size | ~80 | ~20 | $-4\times$ |
| Prefill performance | 400 TFLOP/s | 400 TFLOP/s | No change |

**Key insight**: Quantisation helps memory-bound decode proportionally to the compression
ratio. It does not help compute-bound prefill at all.

---

## Common Mistakes and Edge Cases

1. **Forgetting activation I/O**: For small batches, activation bytes are negligible
   compared to weight bytes. For large $B \cdot S$, activation I/O can exceed weight I/O.
   Always check whether the approximation $I \approx B$ is valid.

2. **Confusing throughput with time**: A higher attainable TFLOP/s does not always mean
   shorter time — the FLOPs also change with batch size. At $B=80$, there are $80\times$
   more FLOPs to complete, so the total time is roughly the same as $B=1$ (both are
   bottlenecked by loading the same weight matrix).

3. **Ignoring KV cache**: This analysis covers only the FFN down-projection. During decode,
   the attention operation also requires loading the KV cache. At long contexts, KV bandwidth
   can equal or exceed weight bandwidth (see `memory_bandwidth_bottlenecks.md` for the
   crossover analysis).

4. **Assuming peak bandwidth is achievable**: The $\beta = 5\ \text{TB/s}$ is theoretical
   peak. Practical efficiency (accounting for HBM scheduling overhead, row activation
   latency, and controller efficiency) is typically $85\text{–}95\%$ of peak for well-optimised
   sequential streaming. Plot the roofline using peak bandwidth, but note real performance
   will be slightly below.

5. **Not normalising**: The roofline plots attainable FLOP/s vs arithmetic intensity.
   Some references instead plot throughput (tokens/s) vs batch size — a valid alternative
   but requires knowing that throughput $= B / T_{\text{step}}$.
