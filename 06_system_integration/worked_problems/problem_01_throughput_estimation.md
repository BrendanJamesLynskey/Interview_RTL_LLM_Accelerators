# Worked Problem 01: Throughput Estimation

## Problem Statement

You are given the following accelerator specification and model specification. Estimate the maximum
sustained **tokens per second** (tokens/s) for both the **prefill** phase and the **decode**
phase. Identify whether each phase is compute-bound or memory-bound, and state the limiting
resource.

### Accelerator Specification

| Parameter | Value |
|---|---|
| Peak TOPS (BF16) | 300 TOPS |
| On-chip SRAM | 64 MB (unified buffer) |
| Off-chip memory | HBM3, 2 stacks |
| HBM3 bandwidth per stack | 819 GB/s |
| Total HBM bandwidth | 1638 GB/s (~1.6 TB/s) |
| HBM capacity | 96 GB (2 × 48 GB) |
| Core clock | 1.8 GHz |
| PCIe interface | Gen5 x16, 63 GB/s |

### Model Specification: LLaMA-2 7B

| Parameter | Value |
|---|---|
| Number of layers | 32 |
| Hidden dimension $d$ | 4096 |
| FFN intermediate dimension | 11008 (SwiGLU) |
| Number of attention heads | 32 |
| Head dimension $d_k$ | 128 |
| Vocabulary size $V$ | 32000 |
| Total parameters | 6.74B |
| Model size (BF16) | $6.74 \times 10^9 \times 2\ \text{B} \approx 13.5\ \text{GB}$ |

### Workload Parameters

| Parameter | Value |
|---|---|
| Prefill batch size $B$ | 16 sequences |
| Prefill sequence length $S$ | 512 tokens |
| Decode batch size $B_{dec}$ | 32 sequences |
| Decode context length $l_{ctx}$ | 256 tokens average |
| Precision | BF16 throughout |

---

## Solution

### Step 1: Characterise the Roofline

The roofline model gives the maximum achievable performance as a function of arithmetic intensity
$I$ (FLOPs per byte of memory traffic):

$$\text{Performance} = \min\!\left(P_{peak},\ I \times B_{mem}\right)$$

where $P_{peak}$ is peak TOPS and $B_{mem}$ is memory bandwidth.

**Crossover (ridge point) arithmetic intensity:**

$$I^* = \frac{P_{peak}}{B_{mem}} = \frac{300 \times 10^{12}\ \text{FLOP/s}}{1.638 \times 10^{12}\ \text{B/s}} \approx 183\ \text{FLOP/byte}$$

Any operation with arithmetic intensity $I > 183$ FLOP/byte is compute-bound; below is
memory-bound.

---

### Step 2: FLOPs per Token — Model Architecture Accounting

We compute FLOPs for a **single token** through all 32 transformer layers.

#### 2a. Self-Attention FLOPs per Layer per Token

**QKV projection:** $3 \times [1, d] \times [d, d]$ — three matrix-vector products.

$$\text{FLOPs}_{QKV} = 3 \times 2 \times d^2 = 6 d^2 = 6 \times 4096^2 = 100.7\ \text{MFLOPs}$$

**Attention scores:** $[n_h, 1, d_k] \times [n_h, d_k, l_{ctx}]$ — $n_h$ dot products of length $d_k$ against $l_{ctx}$ keys.

$$\text{FLOPs}_{scores} = 2 \times n_h \times d_k \times l_{ctx} = 2 \times 32 \times 128 \times l_{ctx} = 8192 \times l_{ctx}$$

**Attention weighted sum:** $[n_h, 1, l_{ctx}] \times [n_h, l_{ctx}, d_k]$

$$\text{FLOPs}_{attn\_out} = 2 \times n_h \times d_k \times l_{ctx} = 8192 \times l_{ctx}$$

**Output projection:** $[1, d] \times [d, d]$

$$\text{FLOPs}_{O} = 2 d^2 = 33.6\ \text{MFLOPs}$$

**Total attention FLOPs per layer per token** (at $l_{ctx} = 256$):

$$\text{FLOPs}_{attn} = 100.7 + 2 \times 8192 \times 256 / 10^6 + 33.6 = 134.3 + 4.2 = 138.5\ \text{MFLOPs}$$

Note: attention score FLOPs are $4.2$ MFLOPs vs. $134.3$ MFLOPs for projections, so attention
dot products are small relative to the linear layers for short contexts.

#### 2b. FFN FLOPs per Layer per Token (SwiGLU)

LLaMA-2 uses SwiGLU: two up-projections of size $[d, d_{ff}]$ and one down-projection $[d_{ff}, d]$,
where $d_{ff} = 11008$.

$$\text{FLOPs}_{FFN} = 3 \times 2 \times d \times d_{ff} = 6 \times 4096 \times 11008 = 270.5\ \text{MFLOPs}$$

#### 2c. Total FLOPs per Token (Single Layer)

$$\text{FLOPs}_{layer} = \text{FLOPs}_{attn} + \text{FLOPs}_{FFN} = 138.5 + 270.5 = 409\ \text{MFLOPs/layer}$$

#### 2d. Total FLOPs per Token (All 32 Layers)

$$\text{FLOPs}_{token} = 32 \times 409\ \text{MFLOPs} = 13.1\ \text{GFLOPs/token}$$

**Rule of thumb check:** $\approx 2 \times \text{parameters} = 2 \times 6.74\ \text{B} = 13.5$ GFLOPs/token.
Our calculated value of 13.1 GFLOPs/token is consistent (small discrepancy from context-length-
dependent attention term).

---

### Step 3: Memory Traffic per Token

The dominant memory traffic is **loading model weights** from HBM on each forward pass (for decode,
where there is no weight reuse across tokens in a step — each token requires a full pass over all
weights).

**Model weight size (BF16):**

$$W_{model} = 6.74 \times 10^9 \times 2\ \text{B} = 13.5\ \text{GB}$$

For a batch of $B_{dec}$ tokens, the same weights are loaded once and reused for all $B_{dec}$
token's computations (weight streaming from HBM is shared across the batch):

$$\text{Weight bytes per token} = \frac{W_{model}}{B_{dec}} = \frac{13.5\ \text{GB}}{32} = 421.9\ \text{MB/token}$$

**KV cache traffic per token (decode, reading existing cache):**

KV cache per layer per sequence, for $l_{ctx} = 256$ tokens at BF16:

$$\text{KV per layer} = 2 \times n_h \times d_k \times l_{ctx} \times 2\ \text{B} = 2 \times 32 \times 128 \times 256 \times 2 = 4\ \text{MB}$$

Total KV read for all 32 layers, for all 32 sequences:

$$\text{KV traffic total} = 32 \times 4\ \text{MB} \times 32\ \text{sequences} = 4096\ \text{MB} = 4\ \text{GB}$$

Per-token share: $\frac{4\ \text{GB}}{32\ \text{tokens}} = 128\ \text{MB/token}$.

**Total memory traffic per token (decode):**

$$B_{token} = 421.9 + 128 = 549.9\ \text{MB/token} \approx 550\ \text{MB/token}$$

---

### Step 4: Arithmetic Intensity

**Decode arithmetic intensity:**

$$I_{decode} = \frac{\text{FLOPs/token}}{\text{bytes/token}} = \frac{13.1 \times 10^9\ \text{FLOP}}{550 \times 10^6\ \text{B}} \approx 23.8\ \text{FLOP/byte}$$

Since $I_{decode} = 23.8 < I^* = 183$:

**The decode phase is MEMORY-BANDWIDTH-BOUND.**

**Prefill arithmetic intensity** (batch $B = 16$, seq len $S = 512$):

Prefill processes $B \times S = 8192$ tokens per step. All weights are loaded once and reused:

$$I_{prefill} = \frac{B \times S \times \text{FLOPs/token}}{W_{model} + \text{KV write traffic}}$$

KV write traffic during prefill (writing keys and values for all $B \times S$ tokens):

$$\text{KV write} = 32\ \text{layers} \times 2 \times n_h \times d_k \times (B \times S) \times 2\ \text{B} = 32 \times 2 \times 32 \times 128 \times 8192 \times 2 = 4\ \text{GB}$$

$$I_{prefill} = \frac{8192 \times 13.1 \times 10^9}{13.5 \times 10^9 + 4 \times 10^9} = \frac{107.4\ \text{TFLOP}}{17.5\ \text{GB}} = 6137\ \text{FLOP/byte}$$

Since $I_{prefill} = 6137 \gg I^* = 183$:

**The prefill phase is COMPUTE-BOUND.**

---

### Step 5: Throughput Estimation

#### 5a. Decode Throughput

The decode throughput is limited by HBM bandwidth:

$$\text{Throughput}_{decode} = \frac{B_{mem}}{B_{token}} = \frac{1.638 \times 10^{12}\ \text{B/s}}{550 \times 10^6\ \text{B/token}} \approx 2978\ \text{tokens/s}$$

Accounting for ~85% HBM utilisation efficiency (bank conflicts, refresh overhead):

$$\text{Throughput}_{decode,eff} \approx 0.85 \times 2978 \approx 2531\ \text{tokens/s}$$

**Per-sequence decode rate:** $2531 / 32 = 79\ \text{tokens/s per sequence}$.

This is consistent with observed performance on real hardware (A100 with 2 TB/s HBM achieves
~2800 tokens/s for LLaMA-2 7B at batch 32; our accelerator has ~82% of A100's HBM bandwidth).

#### 5b. Prefill Throughput

Prefill is compute-bound. Peak throughput limited by TOPS:

$$\text{FLOP/step} = B \times S \times \text{FLOPs/token} = 8192 \times 13.1\ \text{GFLOPs} \approx 107.4\ \text{TFLOPs}$$

$$t_{prefill} = \frac{107.4\ \text{TFLOPs}}{300\ \text{TOPS}} \approx 358\ \text{ms}$$

Accounting for ~70% MAC utilisation efficiency (pipeline fill, synchronisation overhead):

$$t_{prefill,eff} \approx 358\ \text{ms} / 0.70 \approx 511\ \text{ms}$$

**Tokens per second (prefill):** $\frac{B \times S}{t_{prefill,eff}} = \frac{8192}{0.511} \approx 16,032\ \text{tokens/s}$

Prefill throughput is ~6× higher than decode throughput (in tokens/s), but prefill processes many
tokens per request, while decode processes one token per request per step.

---

### Step 6: On-Chip SRAM Capacity Check

Can the 64 MB on-chip SRAM hold the active working set for decode?

| Data | Size |
|---|---|
| One layer's weights (attention + FFN) | $2 \times (4 \times 4096^2 + 3 \times 4096 \times 11008) \times 2\ \text{B} = 2 \times (134.2 + 270.5)\ \text{MB} = 809\ \text{MB}$ |
| Attention activations per layer | $B_{dec} \times d \times 2\ \text{B} = 32 \times 4096 \times 2 = 256\ \text{KB}$ |
| KV cache for current layer, all sequences | $2 \times n_h \times d_k \times l_{ctx} \times B_{dec} \times 2\ \text{B} = 4\ \text{MB}$ |

The per-layer weight size (809 MB) is far larger than the 64 MB SRAM. **Model weights cannot be
staged in SRAM; they must be streamed from HBM.** The SRAM is used for:
- Activations (256 KB) — fits easily.
- Tile-level double-buffering of weight sub-matrices being loaded from HBM.
- KV cache hot-working-set (~4 MB per layer, fits for one layer at a time).

This confirms the decode phase is HBM-bandwidth-limited, not SRAM-limited.

---

### Step 7: Summary Table

| Phase | Bound by | Throughput | Latency per step |
|---|---|---|---|
| Prefill ($B=16$, $S=512$) | Compute (300 TOPS) | ~16,000 tokens/s | ~511 ms per prefill batch |
| Decode ($B=32$) | HBM bandwidth (1.6 TB/s) | ~2,530 tokens/s | ~12.7 ms per decode step |

**Roofline chart (qualitative):**

```
Perf
(TOPS)
300  ┤────────────────────────────────── peak compute
     │                                 ╱ prefill (I=6137)  ← compute bound
     │                               ╱
     │                             ╱
     │                           ╱
  39 ┤- - - - - - - - - - - - -╱- - - - - -  decode effective (I=23.8)
     │                       ╱
     │                     ╱  decode (I=23.8)
     │                   ╱
  0  └─────────────────────────────────────── I (FLOP/byte)
     0        183       6137
              I*
```

**Key takeaways for an interview:**

1. Prefill is compute-bound; improving TOPS (not memory bandwidth) improves prefill latency.
2. Decode is memory-bandwidth-bound; the only way to improve decode throughput is to increase HBM
   bandwidth or increase batch size (to share weight loading across more sequences).
3. Increasing batch size from 32 to 64 would halve the weight bytes per token, improving decode
   throughput by ~2× — as long as HBM capacity holds all KV caches (64 sequences × 2 GB KV each =
   128 GB, exceeding the 96 GB HBM in this spec; therefore INT8 quantisation would be needed first).
