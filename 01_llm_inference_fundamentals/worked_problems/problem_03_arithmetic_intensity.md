# Worked Problem 03: Arithmetic Intensity — Prefill vs Decode

## Problem Statement

This problem develops a rigorous, component-by-component arithmetic intensity analysis
for both prefill and decode phases of LLM inference, then shows analytically why the
crossover between memory-bound and compute-bound regimes occurs where it does.

**Model specification (LLaMA-2 7B-style)**:

| Parameter | Value |
|---|---|
| Layers ($L$) | 32 |
| Hidden dimension ($d$) | 4096 |
| FFN intermediate ($d_{ff}$) | 11008 (SwiGLU) |
| Attention heads ($h$) | 32 |
| KV heads ($h_{KV}$) | 32 (MHA, no GQA) |
| Head dimension ($d_h = d/h$) | 128 |
| Number format | FP16 ($2$ bytes per element) |

**Accelerator**:

| Spec | Value |
|---|---|
| Peak BF16/FP16 throughput ($\Pi$) | 312 TFLOP/s |
| Peak HBM bandwidth ($\beta$) | 2 TB/s |
| Ridge point ($I^* = \Pi/\beta$) | 156 FLOP/Byte |

**Tasks**:

1. Calculate arithmetic intensity for each operation in one transformer layer during
   **prefill** (batch $B=1$, sequence length $S=512$).
2. Calculate arithmetic intensity for each operation during **decode** (batch $B=1$,
   one token generated, KV cache of length $S_{\text{ctx}}=512$).
3. Identify the bottleneck for each operation in each phase.
4. Find the critical sequence length $S^*$ at which the attention $QK^T$ operation crosses
   from memory-bound to compute-bound during prefill.
5. Derive the general condition on $B$ and $S_{\text{ctx}}$ for decode to reach the ridge point.
6. Build a full table comparing FLOPs, bytes, intensity, and bottleneck for all major operations.
7. Explain the intuition: why is prefill compute-bound while decode is memory-bound,
   and why does this require different hardware solutions?

---

## Part 1: Prefill Arithmetic Intensity ($B=1$, $S=512$)

**Conventions**:
- Arithmetic intensity $I = \text{FLOPs} / \text{Bytes}$
- "Bytes" = all HBM traffic: weight reads + activation reads + output writes
- Each FP16 element = 2 bytes
- One MAC (multiply-accumulate) = 2 FLOPs

### 1a. QKV Projections

Three separate projections: $Q = X W_Q$, $K = X W_K$, $V = X W_V$, each
$W \in \mathbb{R}^{d \times d}$, $X \in \mathbb{R}^{S \times d}$.

**FLOPs** (per projection):
$$\text{FLOPs}_{W_Q} = 2 \times S \times d \times d = 2 \times 512 \times 4096 \times 4096 = 17.18 \times 10^9$$

**Total (three projections)**:
$$\text{FLOPs}_{QKV} = 3 \times 17.18 \times 10^9 = 51.54 \text{ GFLOPs}$$

**Bytes** per projection (weights loaded once, activation input read, output written):
$$\text{Bytes}_{W_Q} = \underbrace{d^2 \times 2}_{\text{weight}} + \underbrace{S \times d \times 2}_{\text{input}} + \underbrace{S \times d \times 2}_{\text{output}}$$
$$= 4096^2 \times 2 + 512 \times 4096 \times 2 + 512 \times 4096 \times 2$$
$$= 33,554,432 + 4,194,304 + 4,194,304 = 41,943,040 \approx 41.9\text{ MB}$$

**Total (three projections)**:
$$\text{Bytes}_{QKV} = 3 \times 41.9\text{ MB} = 125.8\text{ MB}$$

**Arithmetic intensity**:
$$I_{QKV} = \frac{51.54 \times 10^9}{3 \times 41.943 \times 10^6} = \frac{51.54 \times 10^9}{125.83 \times 10^6} \approx \mathbf{409.6 \text{ FLOP/Byte}}$$

$I_{QKV} = 409.6 > I^* = 156$ $\Rightarrow$ **Compute-bound**.

**Approximation**: For large $d$ relative to $S$, weight bytes dominate:
$I \approx \frac{2Sd^2}{d^2 \times 2} = S = 512$. This is an upper bound: here the activation bytes are a quarter of the weight bytes, which pulls the true value down to 409.6. Either way the operation is compute-bound.

---

### 1b. Attention Scores: $QK^T$ (Prefill)

$Q, K \in \mathbb{R}^{S \times d_h}$ per head; $h = 32$ heads.

**FLOPs** (dot products between all query-key pairs, per head):
$$\text{FLOPs}_{QK^T, \text{head}} = 2 \times S \times S \times d_h = 2 \times 512^2 \times 128 = 67.11 \times 10^6$$

**Total over all heads**:
$$\text{FLOPs}_{QK^T} = h \times 67.11 \times 10^6 = 32 \times 67.11 \times 10^6 = 2.15 \times 10^9$$

**Bytes** (Q and K both loaded, score matrix written):
$$\text{Bytes}_{QK^T} = \underbrace{S \times d \times 2}_{Q} + \underbrace{S \times d \times 2}_{K} + \underbrace{S^2 \times h \times 2}_{\text{scores}}$$
$$= 512 \times 4096 \times 2 + 512 \times 4096 \times 2 + 512^2 \times 32 \times 2$$
$$= 4,194,304 + 4,194,304 + 16,777,216 = 25,165,824 \approx 25.2\text{ MB}$$

**Arithmetic intensity**:
$$I_{QK^T} = \frac{2.147 \times 10^9}{25.17 \times 10^6} \approx \mathbf{85.3 \text{ FLOP/Byte}}$$

$I_{QK^T} = 85.3 < I^* = 156$ $\Rightarrow$ **Memory-bandwidth-bound** at $S=512$.

This is an important result: even in prefill, the $QK^T$ attention score computation is
memory-bound at $S=512$. It becomes compute-bound only at larger $S$ (see Part 4).

---

### 1c. Value Aggregation: $\text{score} \cdot V$ (Prefill)

$\text{score} \in \mathbb{R}^{S \times S}$, $V \in \mathbb{R}^{S \times d_h}$, per head.

**FLOPs** (same shape as $QK^T$):
$$\text{FLOPs}_{AV} = h \times 2 \times S \times S \times d_h = 2.15 \times 10^9$$

**Bytes** (scores read, V loaded, output written):
$$\text{Bytes}_{AV} = \underbrace{S^2 \times h \times 2}_{\text{scores}} + \underbrace{S \times d \times 2}_{V} + \underbrace{S \times d \times 2}_{\text{output}}$$
$$= 16,777,216 + 4,194,304 + 4,194,304 = 25,165,824 \approx 25.2\text{ MB}$$

$$I_{AV} \approx \mathbf{85.3 \text{ FLOP/Byte}} \quad \Rightarrow \text{Memory-bound}$$

Identical to $QK^T$ by symmetry of the problem.

---

### 1d. Output Projection ($W_O$)

$W_O \in \mathbb{R}^{d \times d}$, input $\in \mathbb{R}^{S \times d}$.

By identical analysis to one of the QKV projections:
$$I_{W_O} \approx \mathbf{409.6 \text{ FLOP/Byte}} \quad \Rightarrow \text{Compute-bound}$$

---

### 1e. FFN (SwiGLU: gate, up, down projections)

SwiGLU has three matrices: $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$ and
$W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$.

**Per projection** ($d \times d_{ff}$, input $\in \mathbb{R}^{S \times d}$):
$$\text{FLOPs}_{\text{proj}} = 2 \times S \times d \times d_{ff} = 2 \times 512 \times 4096 \times 11008 = 46.17 \times 10^9$$

**Bytes** (weight + input + output):
$$\text{Bytes}_{\text{proj}} = d \times d_{ff} \times 2 + S \times d \times 2 + S \times d_{ff} \times 2$$
$$= 4096 \times 11008 \times 2 + 512 \times 4096 \times 2 + 512 \times 11008 \times 2$$
$$= 90,177,536 + 4,194,304 + 11,272,192 = 105,644,032 \approx 105.6\text{ MB}$$

$$I_{\text{gate/up}} = \frac{46.17 \times 10^9}{105.64 \times 10^6} \approx 437.0 \text{ FLOP/Byte}$$

**Down projection** ($d_{ff} \times d$, input $\in \mathbb{R}^{S \times d_{ff}}$):
$$\text{FLOPs}_{\text{down}} = 2 \times S \times d_{ff} \times d = 46.17 \times 10^9 \quad \text{(same)}$$

$$\text{Bytes}_{\text{down}} = d_{ff} \times d \times 2 + S \times d_{ff} \times 2 + S \times d \times 2$$
$$= 90,177,536 + 11,272,192 + 4,194,304 = 105,644,032 \approx 105.6\text{ MB}$$

$$I_{\text{down}} \approx 437.0 \text{ FLOP/Byte}$$

All three FFN projections: $I_{\text{FFN}} \approx \mathbf{437\ \text{FLOP/Byte}} \Rightarrow$ **Compute-bound**.

---

## Part 2: Decode Arithmetic Intensity ($B=1$, $S_{\text{ctx}}=512$)

At decode, we process **one new token** against a KV cache of 512 existing tokens.
Each linear layer is now a GEMV: $y = Wx$ where $x \in \mathbb{R}^d$.

### 2a. QKV Projections (Decode)

$W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$, input $x \in \mathbb{R}^d$ (one token).

**FLOPs** per projection:
$$\text{FLOPs}_{W_Q}^{\text{dec}} = 2 \times 1 \times d \times d = 2 \times 4096^2 = 33.55 \times 10^6$$

**Bytes**:
$$\text{Bytes}_{W_Q}^{\text{dec}} = d^2 \times 2 + d \times 2 + d \times 2$$
$$= 33,554,432 + 8,192 + 8,192 = 33,570,816 \approx 33.6\text{ MB}$$

(Activation I/O is negligible: $2 \times d \times 2 = 16,384\text{ B} = 16\text{ KB} \ll 33.6\text{ MB}$ weights.)

$$I_{QKV}^{\text{dec}} = \frac{33.55 \times 10^6}{33.57 \times 10^6} \approx \mathbf{0.999 \approx 1 \text{ FLOP/Byte}} \quad \Rightarrow \text{Severely memory-bound}$$

This is the fundamental result: for a GEMV in FP16, FLOPs $= 2d^2$ and bytes $\approx 2d^2$,
giving $I = 1$ FLOP/Byte, independent of $d$.

---

### 2b. Attention Scores: $qK^T$ (Decode, with KV cache)

New query $q \in \mathbb{R}^{d_h}$ for each head. Cached keys $K \in \mathbb{R}^{S_{\text{ctx}} \times d_h}$.

**FLOPs** (dot product of $q$ with each of $S_{\text{ctx}}$ cached keys, per head):
$$\text{FLOPs}_{qK^T, \text{head}}^{\text{dec}} = 2 \times S_{\text{ctx}} \times d_h = 2 \times 512 \times 128 = 131,072$$

**Total over $h = 32$ heads**:
$$\text{FLOPs}_{qK^T}^{\text{dec}} = 32 \times 131,072 = 4.19 \times 10^6$$

**Bytes** (load KV cache, load new query $q$, write score vector):
$$\text{Bytes}_{qK^T}^{\text{dec}} = \underbrace{S_{\text{ctx}} \times d \times 2}_{K \text{ cache}} + \underbrace{d \times 2}_{q} + \underbrace{S_{\text{ctx}} \times h \times 2}_{\text{scores}}$$
$$= 512 \times 4096 \times 2 + 4096 \times 2 + 512 \times 32 \times 2$$
$$= 4,194,304 + 8,192 + 32,768 = 4,235,264 \approx 4.24\text{ MB}$$

$$I_{qK^T}^{\text{dec}} = \frac{4.19 \times 10^6}{4.24 \times 10^6} \approx \mathbf{0.99 \approx 1 \text{ FLOP/Byte}} \quad \Rightarrow \text{Memory-bound}$$

Again we get $I \approx 1$ FLOP/Byte. The KV cache access is also memory-bound for decode.

---

### 2c. Value Aggregation: $\text{score} \cdot V$ (Decode)

$\text{score} \in \mathbb{R}^{S_{\text{ctx}}}$ (per head), $V \in \mathbb{R}^{S_{\text{ctx}} \times d_h}$ (per head).

**FLOPs** (vector-matrix multiply, per head):
$$\text{FLOPs}_{\text{head}} = 2 \times S_{\text{ctx}} \times d_h = 131,072 \quad \text{(same as } qK^T \text{)}$$

**Bytes** (score loaded, V cache loaded, output vector written):
$$\text{Bytes}_{AV}^{\text{dec}} = \underbrace{S_{\text{ctx}} \times h \times 2}_{\text{scores}} + \underbrace{S_{\text{ctx}} \times d \times 2}_{V \text{ cache}} + \underbrace{d \times 2}_{\text{output}}$$
$$= 32,768 + 4,194,304 + 8,192 = 4,235,264 \approx 4.24\text{ MB}$$

$$I_{AV}^{\text{dec}} \approx \mathbf{1 \text{ FLOP/Byte}} \quad \Rightarrow \text{Memory-bound}$$

---

### 2d. FFN Down Projection (Decode)

$W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$, input $x \in \mathbb{R}^{d_{ff}}$ (one token).

$$\text{FLOPs} = 2 \times d_{ff} \times d = 2 \times 11008 \times 4096 = 90.18 \times 10^6$$

$$\text{Bytes} = d_{ff} \times d \times 2 + d_{ff} \times 2 + d \times 2 = 90,177,536 + 22,016 + 8,192 \approx 90.2\text{ MB}$$

$$I_{\text{FFN, down}}^{\text{dec}} = \frac{90.18 \times 10^6}{90.21 \times 10^6} \approx \mathbf{1 \text{ FLOP/Byte}} \quad \Rightarrow \text{Memory-bound}$$

All FFN projections in decode give $I \approx 1$ FLOP/Byte.

---

## Part 3: Bottleneck Summary (Prefill vs Decode)

| Operation | Prefill $I$ | Decode $I$ | Prefill bottleneck | Decode bottleneck |
|---|---|---|---|---|
| $W_Q$ projection | 409.6 | 1.0 | Compute | Memory BW |
| $W_K$ projection | 409.6 | 1.0 | Compute | Memory BW |
| $W_V$ projection | 409.6 | 1.0 | Compute | Memory BW |
| $QK^T / qK^T$ | 85.3 | 1.0 | Memory BW | Memory BW |
| $AV$ / score $\cdot V$ | 85.3 | 1.0 | Memory BW | Memory BW |
| $W_O$ projection | 409.6 | 1.0 | Compute | Memory BW |
| $W_{\text{gate}}$ | 437.0 | 1.0 | Compute | Memory BW |
| $W_{\text{up}}$ | 437.0 | 1.0 | Compute | Memory BW |
| $W_{\text{down}}$ | 437.0 | 1.0 | Compute | Memory BW |

Ridge point $I^* = 156$ FLOP/Byte (312 TFLOP/s, 2 TB/s).

**Prefill**: Most operations are compute-bound. The attention score computation ($QK^T$)
is an exception at $S=512$ — it is memory-bound for this model and sequence length.

**Decode**: All operations have $I \approx 1$ FLOP/Byte, uniformly memory-bound.

---

## Part 4: Critical Sequence Length for $QK^T$ in Prefill

We want to find $S^*$ such that $I_{QK^T}^{\text{prefill}} = I^* = 156$.

**General formula for $QK^T$ intensity** (prefill, all $h$ heads):

$$I_{QK^T}^{\text{prefill}} = \frac{2S^2 d}{2Sd + 2Sd + 2S^2 h} = \frac{2S^2 d}{4Sd + 2S^2 h}$$

Dividing numerator and denominator by $2S$:

$$I_{QK^T}^{\text{prefill}} = \frac{Sd}{2d + Sh} = \frac{S \cdot d}{2d + S \cdot h}$$

For large $S$ (specifically $S \gg 2d/h = 2 \times 4096/32 = 256$):
$$I_{QK^T}^{\text{prefill}} \approx \frac{Sd}{Sh} = \frac{d}{h} = d_h = 128 \text{ FLOP/Byte}$$

This is the asymptotic intensity — it saturates at $d_h$ for large $S$. Since $d_h = 128 < I^* = 156$,
**the $QK^T$ operation is always memory-bandwidth-bound on this accelerator** for large $S$!

Let us verify: setting $I = I^* = 156$:
$$\frac{Sd}{2d + Sh} = 156$$
$$Sd = 156(2d + Sh)$$
$$Sd = 312d + 156Sh$$
$$S(d - 156h) = 312d$$
$$S^* = \frac{312 \times d}{d - 156h} = \frac{312 \times 4096}{4096 - 156 \times 32} = \frac{1,277,952}{4096 - 4992}$$

The denominator $d - 156h = 4096 - 4992 = -896 < 0$. There is no positive solution.

**Conclusion**: $d_h = 128 < I^* = 156$, so $I_{QK^T}$ never reaches $I^*$. The $QK^T$
computation is **always memory-bandwidth-bound** on this specific accelerator, at any sequence length.

**For a lower-ridge-point accelerator** (e.g., $I^* = 64$, representing a more
bandwidth-efficient chip):
$$d_h = 128 > 64 \Rightarrow I_{QK^T} \text{ exceeds ridge point for large } S$$

Setting $I = 64$:
$$S^* = \frac{64 \times 2d}{d - 64h} = \frac{64 \times 2 \times 4096}{4096 - 64 \times 32} = \frac{524,288}{4096 - 2048} = \frac{524,288}{2048} = 256$$

So for an accelerator with $I^* = 64$, the $QK^T$ attention computation becomes compute-bound
at $S > 256$ tokens during prefill.

**Insight**: Hardware with a lower ridge point (more balanced compute-to-bandwidth ratio,
or specifically higher memory bandwidth relative to compute) can utilise the attention
computation more efficiently.

---

## Part 5: Decode Ridge-Point Condition

For decode, the dominant bandwidth consumer is weight loading. For all $N$ parameters in FP16:

$$I_{\text{decode}} = \frac{2N \cdot B}{2N} = B \quad \text{(FLOP/Byte, FP16)}$$

Here $B$ is the batch size: all $B$ tokens share the same weight loads.

**Ridge-point condition**:
$$I_{\text{decode}} = I^* \Rightarrow B = I^* = 156$$

But we must also account for KV cache bandwidth. At batch size $B$ with context $S_{\text{ctx}}$:

**Total bytes per decode step**:
$$\text{Bytes}_{\text{total}} = \underbrace{2N}_{\text{weights}} + \underbrace{4 L S_{\text{ctx}} d B}_{\text{KV cache (FP16)}}$$

**Total FLOPs**:
$$\text{FLOPs}_{\text{total}} = 2NB + 4LS_{\text{ctx}}dB$$

**Combined intensity**:
$$I_{\text{combined}} = \frac{2NB + 4LS_{\text{ctx}}dB}{2N + 4LS_{\text{ctx}}dB}$$

Setting $I_{\text{combined}} = I^* = 156$:

$$2NB + 4LS_{\text{ctx}}dB = 156(2N + 4LS_{\text{ctx}}dB)$$

$$B(2N + 4LS_{\text{ctx}}d - 624LS_{\text{ctx}}d) = 312N$$

$$\boxed{B = \frac{156\,N}{N - 310\,LS_{\text{ctx}}d}}$$

The FLOPs and bytes do **not** share a common factor: the weight bytes $2N$ are shared by all
$B$ requests, but each request reads its own KV cache, so the KV term in the denominator grows
with $B$. KV-cache reads stay at $\approx 1$ FLOP/Byte however large the batch. As $B \to \infty$:

$$I_{\text{combined}} \to \frac{2N + 4LS_{\text{ctx}}d}{4LS_{\text{ctx}}d} = 1 + \frac{N}{2LS_{\text{ctx}}d}$$

With the layer weights of this model, $N = L(4d^2 + 3d\,d_{ff}) = 6.48 \times 10^9$:
- A solution exists only if $S_{\text{ctx}} < N / (310\,L\,d) \approx 159$ tokens.
- At $S_{\text{ctx}} = 512$ the combined intensity can never exceed $1 + 48.3 = 49.3$ FLOP/Byte,
  so decode never reaches the ridge point, whatever the batch size.

This is why MHA models are so hard to run efficiently at long context, and why GQA/MQA
(fewer KV heads, so fewer KV bytes per token) matter.

**Numerical result for this accelerator** (weights only, KV traffic ignored):
$$B_{\text{ridge}} = I^* = \frac{312 \text{ TFLOP/s}}{2 \text{ TB/s}} = 156$$

For INT8 weights ($\text{bpe} = 1$, so 2 FLOP/Byte per element per request):
$$I_{\text{decode}}^{\text{INT8}} = 2B \Rightarrow B_{\text{ridge}}^{\text{INT8}} = I^*/2 = 78$$

For INT4 weights ($\text{bpe} = 0.5$, so 4 FLOP/Byte per element per request):
$$B_{\text{ridge}}^{\text{INT4}} = I^*/4 = 39$$

---

## Part 6: Complete Intensity Summary Table

**Conditions**: $B=1$, $S=512$, $d=4096$, $d_{ff}=11008$, $d_h=128$, $h=32$, FP16.
Ridge point $I^* = 156$ FLOP/Byte.

| Operation | Phase | FLOPs | Bytes | Intensity | Bottleneck |
|---|---|---|---|---|---|
| $W_Q$ GEMM | Prefill | 17.18 G | 41.9 MB | 409.6 | Compute |
| $W_K$ GEMM | Prefill | 17.18 G | 41.9 MB | 409.6 | Compute |
| $W_V$ GEMM | Prefill | 17.18 G | 41.9 MB | 409.6 | Compute |
| $QK^T$ (scores) | Prefill | 2.15 G | 25.2 MB | 85.3 | Memory BW |
| Softmax | Prefill | ~0.05 G | 16.8 MB | ~3 | Memory BW |
| $AV$ (aggregation) | Prefill | 2.15 G | 25.2 MB | 85.3 | Memory BW |
| $W_O$ GEMM | Prefill | 17.18 G | 41.9 MB | 409.6 | Compute |
| $W_{\text{gate}}$ GEMM | Prefill | 46.17 G | 105.6 MB | 437.0 | Compute |
| $W_{\text{up}}$ GEMM | Prefill | 46.17 G | 105.6 MB | 437.0 | Compute |
| $W_{\text{down}}$ GEMM | Prefill | 46.17 G | 105.6 MB | 437.0 | Compute |
| $W_Q$ GEMV | Decode | 33.6 M | 33.6 MB | 1.00 | Memory BW |
| $W_K$ GEMV | Decode | 33.6 M | 33.6 MB | 1.00 | Memory BW |
| $W_V$ GEMV | Decode | 33.6 M | 33.6 MB | 1.00 | Memory BW |
| $qK^T$ (attention) | Decode | 4.19 M | 4.24 MB | 0.99 | Memory BW |
| $AV$ (attention) | Decode | 4.19 M | 4.24 MB | 0.99 | Memory BW |
| $W_O$ GEMV | Decode | 33.6 M | 33.6 MB | 1.00 | Memory BW |
| $W_{\text{gate}}$ GEMV | Decode | 90.2 M | 90.2 MB | 1.00 | Memory BW |
| $W_{\text{up}}$ GEMV | Decode | 90.2 M | 90.2 MB | 1.00 | Memory BW |
| $W_{\text{down}}$ GEMV | Decode | 90.2 M | 90.2 MB | 1.00 | Memory BW |

---

## Part 7: Intuition — Why the Phases Are Different

### The GEMM vs GEMV Dichotomy

**Prefill**: processes $S$ tokens simultaneously. Every linear layer is a GEMM:

$$Y_{(S \times d)} = X_{(S \times d_{\text{in}})} \cdot W_{(d_{\text{in}} \times d_{\text{out}})}$$

The weight matrix $W$ is loaded once from HBM but contributes $S$ output rows — one per token.
The weight-loading cost is amortised over $S$ uses:

$$I_{\text{GEMM}} \approx \frac{2 S d_{\text{in}} d_{\text{out}}}{2 d_{\text{in}} d_{\text{out}}} = S$$

**Decode**: processes 1 token. Every linear layer is a GEMV:

$$y_{(d_{\text{out}})} = W_{(d_{\text{out}} \times d_{\text{in}})} \cdot x_{(d_{\text{in}})}$$

The weight matrix is loaded once but produces only 1 output vector:

$$I_{\text{GEMV}} = \frac{2 d_{\text{in}} d_{\text{out}}}{2 d_{\text{in}} d_{\text{out}}} = 1$$

The weight-loading cost cannot be amortised because there is only one token to process.

### Why Different Hardware Solutions?

**Prefill is compute-bound** — the hardware is limited by how fast it can execute MACs.
Solutions:
- Maximise FLOP/s: larger systolic arrays, higher clock speed, more compute tiles.
- Tensor parallelism: split the weight matrix across multiple accelerators to increase
  aggregate compute and keep arithmetic intensity high.
- Flash Attention: reduce $QK^T$ bandwidth (the one memory-bound prefill operation)
  by tiling the attention computation.

**Decode is memory-bandwidth-bound** — the hardware is limited by how fast it can stream weights.
Solutions:
- Increase $\beta$: more HBM stacks, higher-generation HBM.
- Reduce bytes per parameter: INT8/INT4 weight quantisation.
- Increase batch size: amortise weight loads across multiple concurrent requests,
  driving $I$ from 1 toward $I^*$.
- Keep weights in SRAM: large on-chip SRAM to avoid HBM access entirely (limited by
  SRAM capacity vs model size).
- Disaggregated decoding: specialise hardware for decode with higher $\beta/\Pi$ ratio.

### The Fundamental Tension

The same model parameter $W$ serves two radically different computational roles:

- In **prefill**: a large tiled GEMM kernel that saturates the compute array.
- In **decode**: a streaming memory access that leaves the compute array idle.

This is why LLM accelerators are difficult to optimise: the two phases demand different
hardware characteristics, and running both on the same chip means accepting sub-optimal
efficiency for at least one phase. The optimal solution — disaggregated prefill/decode
clusters with hardware tailored to each phase — is an active area of datacenter-level
system design in 2025 and beyond.
