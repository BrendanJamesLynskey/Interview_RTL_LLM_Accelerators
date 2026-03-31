# Worked Problem 02: End-to-End Latency Budget

## Problem Statement

Break down the **end-to-end latency for generating a single token** during the decode phase of
LLM inference on the accelerator described below. Identify the **critical path**, the dominant
latency components, and propose optimisations targeting the top-two contributors.

### Accelerator and Model Specification

| Parameter | Value |
|---|---|
| Model | LLaMA-2 7B (32 layers, $d = 4096$, 32 heads) |
| Precision | BF16 |
| Batch size $B$ | 1 (single-sequence, latency-critical path) |
| Context length $l_{ctx}$ | 1024 tokens |
| HBM bandwidth | 1.6 TB/s |
| HBM random access latency | 100 ns |
| On-chip SRAM bandwidth | 20 TB/s (64 MB, 256-bit wide, 1.5 GHz) |
| On-chip SRAM latency | 2 ns (3 cycles at 1.5 GHz) |
| Peak TOPS (BF16) | 300 TOPS |
| MAC array dimensions | 256 × 256 systolic array |
| Core clock | 1.5 GHz |
| All-reduce latency (intra-chip) | N/A (single chip) |
| PCIe Gen5 x16 | 63 GB/s, 1 µs one-way latency |

**Note:** $B = 1$ represents the worst-case (minimum batch) latency scenario. This is the
time-to-first-token (TTFT) decomposition for the very first decode step after prefill.

---

## Solution

The decode step for a single token passes through these serial stages, each of which we will
quantify:

```
Host → PCIe → [DMA command transfer]
                     ↓
              [Embedding lookup]
                     ↓
              [For each of 32 layers:]
              [  Weight load: QKV projection weights → SRAM tile ]
              [  GEMV: Q, K, V computation                       ]
              [  KV cache write (K_t, V_t)                       ]
              [  KV cache read  (K_0..t-1, V_0..t-1)             ]
              [  Attention score GEMV                             ]
              [  Softmax                                          ]
              [  Attention weighted sum GEMV                      ]
              [  Weight load: output projection → SRAM tile       ]
              [  GEMV: output projection                          ]
              [  Residual add + LayerNorm                         ]
              [  Weight load: FFN (gate, up, down)                ]
              [  GEMV: FFN gate + up                              ]
              [  SwiGLU activation                                ]
              [  GEMV: FFN down projection                        ]
              [  Residual add + LayerNorm                         ]
                     ↓
              [Output projection (lm_head): GEMV d→V]
                     ↓
              [Top-k sampling + argmax]
                     ↓
                    PCIe → Host [result DMA transfer]
```

---

### Component 1: Host Command Latency

The host CPU writes a command descriptor (~64 bytes) to the device command queue via an MMIO
doorbell. The PCIe posted-write latency is:

$$t_{cmd} = t_{PCIe,one-way} \approx 1\ \mu\text{s}$$

This is a serial latency — the device cannot begin processing until the command arrives.

---

### Component 2: Embedding Lookup

For a single token, the embedding lookup reads one row of $E \in \mathbb{R}^{32000 \times 4096}$
from HBM:

$$\text{bytes} = 4096 \times 2\ \text{B} = 8\ \text{KB}$$

At HBM streaming bandwidth (one sequential row): limited by HBM access latency for the first cache
line, then streaming rate for the rest.

$$t_{emb} = t_{HBM,latency} + \frac{8192\ \text{B} - 128\ \text{B}}{B_{HBM}} = 100\ \text{ns} + \frac{8064}{1.6 \times 10^{12}} \approx 100\ \text{ns} + 5\ \text{ns} = 105\ \text{ns}$$

The embedding lookup is dominated by HBM random access latency: $t_{emb} \approx 0.1\ \mu\text{s}$.

---

### Component 3: Per-Layer Latency (repeated 32×)

We compute the latency for the most expensive operations in a single transformer layer for $B = 1$.

#### 3a. Weight Load: QKV Projection

QKV projection weight matrix: $W_{QKV} \in \mathbb{R}^{3d \times d}$.

$$\text{bytes}_{QKV\_weights} = 3 \times 4096 \times 4096 \times 2\ \text{B} = 100.7\ \text{MB}$$

These weights are streamed from HBM. With streaming access (no random pattern), HBM achieves
near-peak bandwidth:

$$t_{QKV\_load} = \frac{100.7 \times 10^6\ \text{B}}{1.6 \times 10^{12}\ \text{B/s}} \approx 62.9\ \mu\text{s}$$

#### 3b. GEMV: QKV Computation

For $B = 1$, this is a matrix-vector product: $[3d, d] \times [d, 1]$.

FLOPs: $2 \times 3 \times 4096 \times 4096 = 100.7\ \text{MFLOPs}$.

For a single-vector input, the GEMV throughput of the 256×256 systolic array:
The array computes one output row per cycle if kept fed. For a $(3 \times 4096) \times 1$ GEMV,
the array is severely underutilised — only 1 column of activations, so parallelism is limited
to the 256-wide rows of the weight matrix.

Effective throughput (1 activation vector across 256-wide array): 256 MACs/cycle.

$$t_{GEMV\_QKV} = \frac{100.7 \times 10^6\ \text{FLOPs}}{256 \times 2\ \text{FLOPs/cycle} \times 1.5 \times 10^9\ \text{cycles/s}} \approx \frac{100.7\ \text{MFLOPs}}{768\ \text{GFLOPs/s}} \approx 131\ \mu\text{s}$$

However, this is **memory-bandwidth-bound** (not compute-bound). The weight load (62.9 µs)
and GEMV compute can be **pipelined** — load a tile of weights into SRAM, compute GEMV on that
tile, then load the next tile. The critical path is dominated by the weight load:

$$t_{QKV,critical} = t_{QKV\_load} = 62.9\ \mu\text{s}$$

(Compute on each tile overlaps with loading of the next tile.)

#### 3c. KV Cache Write

Write new $K_t$ and $V_t$ to HBM KV cache:

$$\text{bytes}_{KV\_write} = 2 \times n_h \times d_k \times 2\ \text{B} = 2 \times 32 \times 128 \times 2 = 16\ \text{KB}$$

$$t_{KV\_write} = \frac{16 \times 10^3\ \text{B}}{1.6 \times 10^{12}\ \text{B/s}} \approx 0.01\ \mu\text{s}$$

Negligible — 10 ns. Typically pipelined with other HBM traffic.

#### 3d. KV Cache Read

Read $K_{0..l_{ctx}-1}$ and $V_{0..l_{ctx}-1}$ from HBM:

$$\text{bytes}_{KV\_read} = 2 \times n_h \times d_k \times l_{ctx} \times 2\ \text{B} = 2 \times 32 \times 128 \times 1024 \times 2 = 16\ \text{MB}$$

$$t_{KV\_read} = \frac{16 \times 10^6\ \text{B}}{1.6 \times 10^{12}\ \text{B/s}} \approx 10\ \mu\text{s}$$

This is the cost of reading the full KV history. It scales linearly with $l_{ctx}$.

#### 3e. Attention Score Computation

$$\text{score}_i = Q \cdot K_i^T, \quad i = 0, \ldots, l_{ctx}-1$$

FLOPs: $2 \times n_h \times d_k \times l_{ctx} = 2 \times 32 \times 128 \times 1024 = 8.4\ \text{MFLOPs}$

The KV cache read and attention score computation are pipelined (fused attention): as each tile
of K/V arrives from HBM, scores and weighted sums are computed immediately in SRAM.

$$t_{attention,critical} = t_{KV\_read} = 10\ \mu\text{s}$$

(The 8.4 MFLOPs of compute at 300 TOPS takes only 0.028 µs — completely overlapped.)

#### 3f. Softmax

Over $l_{ctx} = 1024$ scores per head, $n_h = 32$ heads: $32768$ values.

$$t_{softmax} = \frac{32768 \times 2\ \text{B (read)} + 32768 \times 2\ \text{B (write)}}{20\ \text{TB/s (SRAM)}} \approx \frac{131\ \text{KB}}{20\ \text{TB/s}} \approx 6.6\ \text{ns}$$

Softmax is negligible: $\approx 0.007\ \mu\text{s}$.

#### 3g. Output Projection Weight Load + GEMV

Weight matrix $W_O \in \mathbb{R}^{d \times d}$:

$$\text{bytes}_{O} = 4096 \times 4096 \times 2\ \text{B} = 33.6\ \text{MB}$$

$$t_{O,critical} = t_{O\_load} = \frac{33.6 \times 10^6}{1.6 \times 10^{12}} \approx 21.0\ \mu\text{s}$$

#### 3h. Residual Add + LayerNorm

Reads and writes $d = 4096$ BF16 values from/to SRAM:

$$t_{LN} = \frac{3 \times 4096 \times 2\ \text{B}}{20\ \text{TB/s}} \approx 1.2\ \text{ns}$$

Negligible: $\approx 0.001\ \mu\text{s}$.

#### 3i. FFN Weight Load + GEMV (SwiGLU)

FFN has three weight matrices: $W_{gate}$, $W_{up}$, $W_{down}$, each of dimension $[d, d_{ff}]$
or $[d_{ff}, d]$ where $d_{ff} = 11008$.

$$\text{bytes}_{FFN} = (2 \times d \times d_{ff} + d_{ff} \times d) \times 2\ \text{B} = 3 \times 4096 \times 11008 \times 2 = 270.5\ \text{MB}$$

$$t_{FFN,critical} = \frac{270.5 \times 10^6}{1.6 \times 10^{12}} \approx 169.1\ \mu\text{s}$$

This is the **dominant per-layer latency component**.

---

### Step 4: Per-Layer Total and Critical Path

The per-layer operations are **partially serialised** and **partially pipelined**. The critical
path through a single layer:

| Operation | Latency (µs) | Pipelined with |
|---|---|---|
| QKV weight load + GEMV | 62.9 | Pipelined (load tiles while computing previous tile) |
| KV cache read + attention | 10.0 | Pipelined (fused attention) |
| Softmax | 0.007 | — |
| Output projection load + GEMV | 21.0 | Pipelined |
| Residual + LayerNorm | 0.001 | — |
| FFN weight load + GEMV | 169.1 | Pipelined |
| Residual + LayerNorm | 0.001 | — |

**Critical path per layer (serial bottleneck):**

Operations that cannot be pipelined with each other (strict data dependency):
1. QKV load + GEMV must complete before KV cache write and attention can begin.
2. Attention must complete before output projection can begin.
3. Output projection must complete before FFN can begin.

Serial critical path per layer:

$$t_{layer} = t_{QKV} + t_{attn} + t_{O} + t_{FFN}$$
$$= 62.9 + 10.0 + 21.0 + 169.1 = 263.0\ \mu\text{s/layer}$$

---

### Step 5: Full Decode Step Latency

Summing all 32 layers plus bookkeeping:

$$t_{decode} = t_{cmd} + t_{emb} + 32 \times t_{layer} + t_{lm\_head} + t_{sampling} + t_{result}$$

**lm_head (output projection, $d \rightarrow V$):**

$$\text{bytes}_{lm\_head} = d \times V \times 2\ \text{B} = 4096 \times 32000 \times 2 = 256\ \text{MB}$$
$$t_{lm\_head} = \frac{256 \times 10^6}{1.6 \times 10^{12}} = 160\ \mu\text{s}$$

**Top-k sampling (from Problem Set on tokenizer hardware):** $\approx 21.7\ \mu\text{s}$

**PCIe result transfer (1 token ID = 4 bytes):**

$$t_{result} \approx 1\ \mu\text{s}\ \text{(dominated by latency, not bandwidth)}$$

**Total decode latency:**

$$t_{decode} = 1.0 + 0.1 + 32 \times 263.0 + 160.0 + 21.7 + 1.0$$
$$= 1.0 + 0.1 + 8416.0 + 160.0 + 21.7 + 1.0 \approx 8599.8\ \mu\text{s} \approx 8.6\ \text{ms}$$

**Tokens per second:** $1000 / 8.6 \approx 116\ \text{tokens/s}$ at batch 1.

---

### Step 6: Latency Breakdown (Critical Path Pie)

```
Component            Latency (µs)   Fraction
──────────────────────────────────────────────
FFN weight load      32 × 169.1 = 5411   62.9%
QKV weight load      32 ×  62.9 = 2013   23.4%
Attention (KV read)  32 ×  10.0 =  320    3.7%
Output proj. load    32 ×  21.0 =  672    7.8%
lm_head                           160     1.9%
Sampling + ctrl                    25     0.3%
──────────────────────────────────────────────
Total                             8601  100.0%
```

**Critical path is dominated by HBM weight loading (94.1% of total time).**

---

### Step 7: Optimisations Targeting Top-Two Contributors

#### Optimisation A: Weight Quantisation (INT4) to Reduce FFN + QKV Load

With INT4 weights (4-bit, packed 2 per byte), weight memory footprint is 4×  smaller:

$$t_{FFN,int4} = 169.1 / 4 = 42.3\ \mu\text{s/layer}$$
$$t_{QKV,int4} = 62.9 / 4 = 15.7\ \mu\text{s/layer}$$
$$t_{O,int4} = 21.0 / 4 = 5.3\ \mu\text{s/layer}$$

New per-layer critical path: $15.7 + 10.0 + 5.3 + 42.3 = 73.3\ \mu\text{s/layer}$

New total: $32 \times 73.3 + 160/4 + 25 + 2 = 2345.6 + 40 + 27 = 2412.6\ \mu\text{s}$

**Improvement: 8600 µs → 2413 µs, a 3.6× speedup** (approximately 4×, with the reduction
partially offset by the lm_head and other terms).

Dequantisation overhead: INT4 → BF16 conversion per tile before the GEMV adder tree. At 1 FLOP
per element for dequant, cost is $0.25 \times$ the MAC FLOPs — small relative to HBM load
savings.

**Quality cost:** State-of-the-art INT4 methods (GPTQ, AWQ) achieve <1% perplexity degradation
on LLaMA-2 7B — acceptable for most deployment scenarios.

#### Optimisation B: KV Cache Compression to Reduce Attention Latency

At $l_{ctx} = 1024$, attention accounts for $320 / 8600 = 3.7\%$ of latency — already modest.
However, as context grows, the KV read latency grows linearly:

At $l_{ctx} = 8192$: $t_{KV\_read} = 16\ \text{MB} \times 8 / 1.6\ \text{TB/s} = 80\ \mu\text{s/layer}$,
contributing $32 \times 80 = 2560\ \mu\text{s}$ — more significant.

**KV cache quantisation (INT8):** Halves KV read traffic. For $l_{ctx} = 8192$:
$t_{KV\_read,int8} = 40\ \mu\text{s/layer}$, saving $32 \times 40 = 1280\ \mu\text{s}$.

**Multi-Query Attention (MQA):** Shares K and V across all heads (only 1 K, 1 V instead of 32):
$$t_{KV\_read,MQA} = \frac{2 \times 1 \times 128 \times 1024 \times 2\ \text{B}}{1.6\ \text{TB/s}} = 0.31\ \mu\text{s/layer}$$

This is a 32× reduction in KV read latency. MQA is used in LLaMA-2 (specifically, Grouped Query
Attention with 8 KV heads vs. 32 query heads — a 4× reduction), which is already accounted for in
the 3.7% figure above.

---

### Summary Table

| Configuration | Total Latency | Tokens/s (B=1) |
|---|---|---|
| Baseline BF16 | 8,600 µs | 116 |
| INT4 weights only | 2,413 µs | 414 |
| INT4 weights + INT8 KV | 2,310 µs | 433 |
| INT4 weights + MQA INT8 KV | ~2,250 µs | 444 |

**Key interview insight:** For single-sequence (B=1) decode latency, the bottleneck is
**weight loading from HBM** (not compute). Weight quantisation is the single most impactful
optimisation — it addresses both the FFN (63%) and QKV (23%) components directly.
