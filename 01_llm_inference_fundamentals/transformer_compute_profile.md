# Transformer Compute Profile

## Overview

This file covers the computational profile of transformer inference from an RTL and hardware
architecture perspective. Topics include per-layer FLOPs breakdowns, the compute-vs-memory-bound
classification of each operation, and how model hyperparameters drive hardware requirements.
Understanding this material is essential for sizing accelerators, predicting bottlenecks, and
making microarchitecture trade-offs.

**Key notation used throughout this file**

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $S$ | Sequence length (number of tokens) |
| $d$ | Model dimension (hidden size) |
| $d_{ff}$ | Feed-forward intermediate dimension (typically $4d$) |
| $h$ | Number of attention heads |
| $d_h$ | Per-head dimension $= d / h$ |
| $L$ | Number of transformer layers |
| $V$ | Vocabulary size |
| $N$ | Total parameter count |

---

## Tier 1 — Fundamentals

### Q1. What are the four main computational blocks in a single transformer decoder layer, and what operation dominates each?

**Answer**

A decoder layer contains:

1. **Multi-Head Self-Attention (MHA)** — dominated by matrix multiplications to project Q, K, V
   and to compute the output projection. The attention score computation ($QK^T$) and value
   aggregation ($\text{score} \cdot V$) are also matrix operations but are proportional to $S^2$
   rather than $S \cdot d$.

2. **Feed-Forward Network (FFN)** — two linear projections, often called the "up" projection
   ($d \to d_{ff}$) and "down" projection ($d_{ff} \to d$), separated by a non-linearity.
   This is universally the largest FLOPs contributor when $d_{ff} = 4d$.

3. **Layer Normalisation (LayerNorm / RMSNorm)** — element-wise operations (mean, variance,
   scale, shift). FLOPs are $O(B \cdot S \cdot d)$ but with very small constants; negligible
   relative to matrix multiplications.

4. **Residual Additions** — element-wise additions, same asymptotic cost as LayerNorm.
   Completely negligible.

**Why it matters in hardware design**: The FFN and attention projections are amenable to
systolic-array acceleration. The attention score computation is a separate dataflow challenge
because its dimension scales with $S^2$, making it increasingly expensive for long contexts.

---

### Q2. Derive the FLOPs for the Feed-Forward Network (FFN) in one transformer layer for a single forward pass with batch size $B$ and sequence length $S$.

**Answer**

The FFN applies two linear transformations:

$$\text{FFN}(x) = \text{activation}(x W_1 + b_1) W_2 + b_2$$

- $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$
- Input $x \in \mathbb{R}^{(B \cdot S) \times d}$

Each matrix multiplication $AB$ where $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$
costs $2mkn$ FLOPs (the factor of 2 accounts for one multiply and one add per MAC).

**Up projection** ($W_1$):
$$\text{FLOPs}_1 = 2 \cdot (B \cdot S) \cdot d \cdot d_{ff}$$

**Down projection** ($W_2$):
$$\text{FLOPs}_2 = 2 \cdot (B \cdot S) \cdot d_{ff} \cdot d$$

**Total FFN FLOPs** (ignoring bias, activation, which are sub-dominant):
$$\text{FLOPs}_{\text{FFN}} = 4 \cdot B \cdot S \cdot d \cdot d_{ff}$$

With the standard $d_{ff} = 4d$:
$$\boxed{\text{FLOPs}_{\text{FFN}} = 16 \cdot B \cdot S \cdot d^2}$$

**Common mistake**: Forgetting the factor of 2 for MACs, or forgetting that both projections
contribute equally. Some texts report FLOPs as $8BSd^2$ — this is for one of the two projections
only, or uses a different FLOPs counting convention (counting only multiplies, not adds).

---

### Q3. For a 7B-parameter model with $d = 4096$, $d_{ff} = 11008$ (LLaMA-style SwiGLU), $L = 32$ layers, what fraction of total model parameters live in FFN weights vs attention projection weights?

**Answer**

**Attention projection weights per layer** (ignoring biases):
- $W_Q, W_K, W_V$: each $d \times d$, so $3d^2$
- $W_O$: $d \times d$, so $d^2$
- Total per layer: $4d^2 = 4 \times 4096^2 \approx 67.1\text{M}$

**FFN weights per layer** (SwiGLU has three matrices: gate, up, down):
- Gate and up projections: $2 \times d \times d_{ff} = 2 \times 4096 \times 11008 \approx 90.2\text{M}$
- Down projection: $d_{ff} \times d \approx 45.1\text{M}$
- Total per layer: $3 \times 4096 \times 11008 \approx 135.3\text{M}$

**Over $L = 32$ layers**:
- Attention total: $32 \times 67.1\text{M} \approx 2.15\text{B}$
- FFN total: $32 \times 135.3\text{M} \approx 4.33\text{B}$

**Remaining parameters**: embeddings ($V \times d = 32000 \times 4096 \approx 131\text{M}$),
plus tied output projection.

**FFN fraction** $\approx 4.33 / 7 \approx 62\%$.
**Attention fraction** $\approx 2.15 / 7 \approx 31\%$.

**Implication for hardware**: Because FFN weights dominate, memory bandwidth for loading FFN
weights is the primary concern during memory-bound (decode) operation.

---

### Q4. What is "arithmetic intensity" and why is it the key quantity for hardware performance analysis?

**Answer**

**Definition**: Arithmetic intensity ($I$) is the ratio of compute operations to memory traffic:

$$I = \frac{\text{FLOPs}}{\text{Bytes transferred}} \quad \left[\frac{\text{FLOP}}{\text{Byte}}\right]$$

It is a property of a specific workload running on a specific hardware configuration — it depends
on what data is already in on-chip memory vs must be fetched from DRAM.

**Why it matters**: Every processor has two fundamental limits:
1. **Peak compute throughput** $\Pi$ [FLOP/s]
2. **Peak memory bandwidth** $\beta$ [Byte/s]

The **roofline model** says the achievable performance is:

$$\text{Attainable FLOP/s} = \min(\Pi, \; \beta \cdot I)$$

The **ridge point** is the intensity $I^* = \Pi / \beta$ at which the compute roof and memory
bandwidth roof intersect. Workloads with $I < I^*$ are **memory-bandwidth-bound**; workloads
with $I > I^*$ are **compute-bound**.

**Example**: An accelerator with $\Pi = 312\ \text{TFLOP/s}$ (BF16) and $\beta = 3.35\ \text{TB/s}$
(HBM3) has $I^* \approx 93\ \text{FLOP/Byte}$. A decode step loading a weight matrix but
performing few MACs has $I \ll 93$, so it is memory-bound.

---

## Tier 2 — Intermediate

### Q5. Derive the total FLOPs for Multi-Head Attention (MHA) in one layer. Break down the contribution from QKV projections, attention score computation, and output projection.

**Answer**

**Setup**: batch $B$, sequence $S$, model dim $d$, $h$ heads, $d_h = d/h$.

**1. QKV projections** — three separate $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$:
$$\text{FLOPs}_{QKV} = 3 \times 2 \cdot (B \cdot S) \cdot d \cdot d = 6BSd^2$$

**2. Attention score computation** — for each head, compute $Q_h K_h^T$:
- $Q_h \in \mathbb{R}^{(BS) \times d_h}$, $K_h \in \mathbb{R}^{S \times d_h}$ (for one batch element, $Q \in \mathbb{R}^{S \times d_h}$)
- Per head, per batch element: $2 \cdot S \cdot d_h \cdot S = 2S^2 d_h$
- Over $B$ batch elements and $h$ heads: $2BS^2 d_h \cdot h = 2BS^2 d$

$$\text{FLOPs}_{\text{score}} = 2BS^2 d$$

**3. Softmax** — $O(BS^2 h)$, negligible vs matrix multiplies for large $d$.

**4. Value aggregation** — $\text{score} \cdot V_h$, same shape as score computation:
$$\text{FLOPs}_{AV} = 2BS^2 d$$

**5. Output projection** $W_O \in \mathbb{R}^{d \times d}$:
$$\text{FLOPs}_{O} = 2BSd^2$$

**Total MHA FLOPs**:
$$\text{FLOPs}_{\text{MHA}} = 8BSd^2 + 4BS^2 d$$

**Key insight**: The $S^2 d$ terms become dominant when $S \gg 2d$ (roughly $S > 8192$ for
$d = 4096$). For typical inference with $S \leq 4096$, the $d^2$ terms dominate and MHA FLOPs
are comparable to one FFN projection.

---

### Q6. For a complete transformer decoder layer, write the expression for total FLOPs per token and identify which terms dominate in short-context vs long-context regimes.

**Answer**

Per forward pass (batch $B$, sequence $S$), total FLOPs for one layer:

$$\text{FLOPs}_{\text{layer}} = \underbrace{8BSd^2}_{\text{MHA projections}} + \underbrace{4BS^2d}_{\text{attention scores + AV}} + \underbrace{4BSd \cdot d_{ff}}_{\text{FFN (standard)}}$$

Normalising to **FLOPs per token** (dividing by $B \cdot S$):

$$\text{FLOPs/token/layer} = 8d^2 + 4Sd + 4d \cdot d_{ff}$$

With $d_{ff} = 4d$:
$$= 8d^2 + 4Sd + 16d^2 = 24d^2 + 4Sd$$

**Short-context regime** ($S \ll 6d$): The $24d^2$ term dominates. FLOPs per token are
approximately constant regardless of sequence length. For $d=4096$: $24 \times 4096^2 \approx 402\text{M}$ FLOPs/token/layer.

**Long-context regime** ($S \gg 6d$, i.e. $S \gg 24576$ for $d=4096$): The $4Sd$ attention
term dominates. FLOPs scale linearly with sequence length, making very long contexts expensive
even per token.

**Over $L$ layers**, FLOPs per generated token:
$$\text{FLOPs/token} \approx 2N + 4LSd$$

where $N$ is total non-embedding parameter count and the $2N$ approximation follows from the
fact that each weight participates in one MAC per token.

---

### Q7. How does the "2N" rule of thumb for FLOPs-per-token arise, and what does it assume?

**Answer**

**Derivation**: Consider the dominant operations — linear projections. Each weight $w$ participates
in exactly one multiply-accumulate (MAC) per token processed. One MAC = 2 FLOPs
(one multiply + one add). If the model has $N$ weight parameters excluding embeddings, then:

$$\text{FLOPs per token} \approx 2N$$

**What it assumes**:
1. **Sequence length is short** relative to $d$ — the $S^2$ attention terms are ignored.
2. **Batch size = 1** or equivalently we are counting per-token cost.
3. **Single forward pass** — i.e., inference, not training (training requires ~3× for backward pass).
4. **Embeddings excluded** — embedding lookups are memory accesses, not FLOPs.

**Example**: LLaMA-2 7B with $N \approx 6.7\text{B}$ parameters:
$$\text{FLOPs/token} \approx 2 \times 6.7 \times 10^9 = 13.4 \text{ GFLOPs}$$

**Common mistake**: Applying the 2N rule to long-context tasks (e.g., 32K context) and
significantly underestimating the actual compute because the $4LSd$ attention term can be
substantial.

---

### Q8. Explain why a larger batch size improves hardware utilisation during the decode phase, up to a point. What limits the batch size?

**Answer**

**The utilisation problem in decode**: In single-token decode, the compute for one linear layer
is a matrix-vector product (GEMV): $y = Wx$ where $W \in \mathbb{R}^{d_{out} \times d_{in}}$
and $x \in \mathbb{R}^{d_{in}}$. The weight matrix $W$ must be loaded from HBM. The
arithmetic intensity is:

$$I_{\text{GEMV}} = \frac{2 \cdot d_{out} \cdot d_{in}}{2 \cdot d_{out} \cdot d_{in}} = 1 \text{ FLOP/Byte (FP16)}$$

This is extremely low — we perform 1 FLOP per byte loaded, whereas a modern accelerator
ridge point is ~90 FLOP/Byte.

**Batching rescues utilisation**: With batch size $B$, the operation becomes a matrix-matrix
product (GEMM): $Y = WX$ where $X \in \mathbb{R}^{d_{in} \times B}$. The weight matrix is
loaded once but used $B$ times:

$$I_{\text{GEMM}} = \frac{2 \cdot d_{out} \cdot d_{in} \cdot B}{2 \cdot d_{out} \cdot d_{in}} = B \text{ FLOP/Byte (FP16)}$$

Intensity increases linearly with $B$. The operation becomes compute-bound when $B \geq I^*$
(the ridge point), i.e. $B \gtrsim 90$ for the accelerator above.

**What limits batch size**:
1. **KV cache memory**: Each sequence requires a KV cache of size $2 \cdot L \cdot S \cdot d \cdot \text{bytes\_per\_element}$. With $L=32, d=4096, S=2048$ at FP16, that is $\approx 512\text{MB}$ per sequence. An 80 GB GPU can hold $\sim 100$ sequences before other memory is exhausted.
2. **Latency SLAs**: Larger batches increase time-to-first-token (TTFT) because all sequences in the batch share compute resources.
3. **Decode divergence**: Different sequences may have different lengths or finish at different times, leading to wasted compute on padding.

---

### Q9. What is the difference in compute character between a prefill step and a decode step for the same model? Use arithmetic intensity to quantify the difference.

**Answer**

**Prefill** processes $S$ tokens simultaneously. For the FFN down-projection
$W \in \mathbb{R}^{d \times d_{ff}}$, batch size $B=1$:

- FLOPs: $2 \cdot S \cdot d_{ff} \cdot d$
- Weight bytes loaded: $2 \cdot d_{ff} \cdot d$ (FP16)
- Intensity: $\frac{2 \cdot S \cdot d_{ff} \cdot d}{2 \cdot d_{ff} \cdot d} = S$ FLOP/Byte

For $S = 1024$: $I_{\text{prefill}} = 1024 \text{ FLOP/Byte}$ — well above the ridge point, so **compute-bound**.

**Decode** generates one token at a time. Same weight matrix, $B=1$:

- FLOPs: $2 \cdot 1 \cdot d_{ff} \cdot d$
- Weight bytes loaded: $2 \cdot d_{ff} \cdot d$ (FP16)
- Intensity: $I_{\text{decode}} = 1 \text{ FLOP/Byte}$ — far below the ridge point, so **memory-bandwidth-bound**.

**Quantitative summary** for the accelerator above ($I^* \approx 93$):

| Phase | Arithmetic Intensity | Bottleneck | Utilisation |
|---|---|---|---|
| Prefill ($S=1024$) | ~1024 FLOP/B | Compute | High |
| Decode ($B=1$) | ~1 FLOP/B | Memory BW | ~1% |
| Decode ($B=90$) | ~90 FLOP/B | At ridge point | ~50% |

---

## Tier 3 — Advanced

### Q10. Grouped Query Attention (GQA) and Multi-Query Attention (MQA) reduce KV head count. Quantify the FLOPs and memory savings compared to standard MHA, and explain the hardware trade-off.

**Answer**

**Standard MHA**: $h$ Q heads, $h$ K heads, $h$ V heads. KV projection parameter count:
$2 \times d \times d = 2d^2$.

**MQA**: 1 K head, 1 V head, $h$ Q heads. KV projection parameter count:
$2 \times d \times d_h = 2d^2 / h$.

**GQA with $g$ groups**: $g$ K heads, $g$ V heads, $h$ Q heads (each K/V head shared by $h/g$
Q heads). KV projection parameter count: $2 \times d \times (g \cdot d_h) = 2d^2 g/h$.

**FLOPs impact** (for projection weights only):

$$\Delta\text{FLOPs}_{\text{KV proj}} = 2BSd^2 \left(1 - \frac{g}{h}\right) \quad \text{(savings)}$$

For MQA ($g=1$, $h=32$): saves $\approx 97\%$ of KV projection FLOPs, but KV projections
are only $2/8 = 25\%$ of total attention FLOPs, so total layer FLOPs reduction is modest
($\sim 6\%$ with $d_{ff}=4d$).

**Memory savings**: The dominant benefit is **KV cache size reduction**. KV cache per token:

- MHA: $2 \cdot h \cdot d_h \cdot \text{bytes} = 2d \cdot \text{bytes}$ per layer
- MQA: $2 \cdot 1 \cdot d_h \cdot \text{bytes} = 2d/h \cdot \text{bytes}$ per layer
- GQA-$g$: $2 \cdot g \cdot d_h \cdot \text{bytes} = 2dg/h \cdot \text{bytes}$ per layer

For MQA: KV cache reduces by $h\times$, enabling $h\times$ larger batch sizes or $h\times$
longer sequences within the same HBM budget.

**Hardware trade-off**: Fewer KV heads mean reduced data reuse opportunity in the attention
score computation. In MQA, the single K,V pair is broadcast to all $h$ Q heads. This is
favourable for **bandwidth** (load K,V once) but creates a **broadcast dependency** in the
microarchitecture — all $h$ query vectors must be staged before issuing the single K/V fetch.
In RTL this manifests as wider interconnects from the KV buffer to the attention engine, versus
narrower independent paths in MHA.

---

### Q11. Derive the operational intensity for the attention score computation ($QK^T$) during **decode** (one new token, KV cache present) and explain why it differs from the projection intensity.

**Answer**

**Setup for decode**: We generate token $t$. The new query vector is $q \in \mathbb{R}^{d_h}$.
The cached keys are $K \in \mathbb{R}^{S_{ctx} \times d_h}$ where $S_{ctx}$ is the context length.
We must compute $q K^T \in \mathbb{R}^{S_{ctx}}$.

**FLOPs**: $2 \cdot S_{ctx} \cdot d_h$ (dot product of $q$ with each of $S_{ctx}$ rows of $K$).

**Bytes transferred**: We must load the entire $K$ cache for this head:
$S_{ctx} \cdot d_h \cdot 2$ bytes (FP16), plus $q$ itself ($d_h \cdot 2$ bytes, negligible for large $S_{ctx}$).

**Arithmetic intensity**:
$$I_{QK^T} = \frac{2 \cdot S_{ctx} \cdot d_h}{2 \cdot S_{ctx} \cdot d_h} = 1 \text{ FLOP/Byte (FP16)}$$

Surprisingly, this is **identical** to the GEMV intensity for weight projections. The attention
score computation in decode is also memory-bandwidth-bound, loading each KV element once to
perform one MAC.

**Compare to prefill** where $Q \in \mathbb{R}^{S \times d_h}$:
$$I_{QK^T}^{\text{prefill}} = \frac{2 S^2 d_h}{2 S d_h} \cdot \frac{S}{S} = S \text{ FLOP/Byte}$$

Wait — more carefully: FLOPs $= 2S \cdot S \cdot d_h = 2S^2 d_h$. Bytes for $K$: $S \cdot d_h \cdot 2$.
But $Q$ must also be loaded: $S \cdot d_h \cdot 2$. Total bytes $\approx 4S \cdot d_h$.
$$I^{\text{prefill}}_{QK^T} = \frac{2S^2 d_h}{4S d_h} = S/2$$

So prefill is compute-bound for $S/2 > I^*$, i.e. $S > 2 \times 93 \approx 186$ tokens on our
example accelerator — easily satisfied.

**RTL implication**: The KV cache access pattern during decode is a sequential streaming read
with no reuse — it is a pure bandwidth problem. Hardware architects respond with high-bandwidth
on-chip SRAM for KV cache (when it fits), or specialised HBM access scheduling to maximise
burst efficiency.

---

### Q12. A hardware team proposes increasing the on-chip SRAM from 50 MB to 200 MB on their LLM accelerator. For which workloads and model configurations will this investment provide the most benefit? Quantify your answer.

**Answer**

The benefit of large on-chip SRAM is to keep frequently-reused data on-chip, avoiding
expensive HBM accesses. Let us categorise what data is accessed repeatedly:

**Case 1: Weights for decode (small batch)**

A weight tile of size $d_{ff} \times d$ for one FFN layer: $11008 \times 4096 \times 2\text{B} = 90.2\text{MB}$.
With 200 MB SRAM, two FFN weight matrices fit on-chip simultaneously. If the engine processes
multiple decode requests sequentially (persistent batch), weights need not be reloaded between
requests, effectively amortising bandwidth:

$$\text{Effective intensity with SRAM reuse} = \frac{2 \cdot d_{ff} \cdot d \cdot R}{2 \cdot d_{ff} \cdot d} = R$$

where $R$ is the number of decode steps before a weight eviction. This can push intensity
above $I^*$ without increasing batch size, benefiting **latency-sensitive single-request** inference.

**Case 2: KV cache for short sequences**

Per-layer KV cache for one sequence at $S=1024$, FP16:
$2 \times 1024 \times 4096 \times 2\text{B} = 16.8\text{MB}$ per layer.
With 200 MB SRAM and $L=32$ layers, total KV cache across all layers is $32 \times 16.8 = 537\text{MB}$ — this does not fit.
However, for the **currently-active layer**, 16.8 MB fits in 200 MB SRAM, so the entire
KV fetch for one attention step is served from SRAM at $\sim 10\times$ lower latency and
higher bandwidth than HBM.

**Case 3: Prefill**

During prefill the bottleneck is compute, not memory bandwidth. SRAM increase provides
minimal benefit here — the compute engines are already saturated.

**Quantitative guidance**:

| Workload | Bottleneck | Benefit of 200 MB SRAM | Magnitude |
|---|---|---|---|
| Decode, small batch | Weight BW | High — weights resident on-chip | Up to $R\times$ throughput |
| Decode, KV attention | KV cache BW | Medium — per-layer KV fits | ~$10\times$ lower latency |
| Prefill, long seq | Compute | Low — already compute-bound | Negligible |
| Decode, large batch | Compute | Low — GEMM already efficient | Negligible |

**Conclusion**: The 200 MB investment primarily benefits **single-request or small-batch decode**
on models where individual weight matrices fit in SRAM. Teams targeting interactive, low-latency
inference see the most return; teams targeting high-throughput batched inference see minimal gain.

---

## Quick-Reference Summary

| Operation | FLOPs (1 layer, batch $B$, seq $S$) | Dominates when |
|---|---|---|
| QKV projections | $6BSd^2$ | Always significant |
| Attention scores ($QK^T + AV$) | $4BS^2 d$ | $S \gg 2d$ |
| Output projection | $2BSd^2$ | Always |
| FFN (standard, $d_{ff}=4d$) | $16BSd^2$ | Almost always |
| Total (short context) | $\approx 24BSd^2$ | $S \ll 6d$ |

**Rules of thumb**

- FLOPs per generated token $\approx 2N$ (where $N$ = non-embedding parameters)
- Memory-bound threshold: batch size $\lesssim I^* = \Pi / \beta$
- FFN holds $\sim 60\%$ of parameters in standard architectures
- Attention $S^2$ term exceeds FFN at $S \approx 4d$ (for $d_{ff}=4d$)
