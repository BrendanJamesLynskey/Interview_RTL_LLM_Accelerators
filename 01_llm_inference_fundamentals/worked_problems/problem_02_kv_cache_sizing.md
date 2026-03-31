# Worked Problem 02: KV Cache Memory Sizing

## Problem Statement

You are sizing the memory system for a production LLM inference deployment. The model is a
LLaMA-2 70B-equivalent architecture.

**Model specification**:

| Parameter | Value |
|---|---|
| Layers ($L$) | 80 |
| Hidden dimension ($d$) | 8192 |
| Attention heads ($h_Q$) | 64 |
| KV heads ($h_{KV}$, GQA) | 8 |
| Head dimension ($d_h$) | 128 |
| FFN intermediate ($d_{ff}$) | 28672 (SwiGLU $\approx 3.5d$) |
| Vocabulary size ($V$) | 32000 |
| Number format (weights) | FP16 ($2$ bytes) |
| Number format (KV cache) | FP16 ($2$ bytes) initially |

**Tasks**:

1. Estimate total model weight memory.
2. Derive the per-token KV cache size (one layer, then all layers).
3. Calculate total KV cache memory for a batch of $B$ sequences each of maximum length $S$,
   for the combinations: $S \in \{512, 2048, 4096, 8192\}$ and $B \in \{1, 8, 32, 128\}$.
4. Given an accelerator with 160 GB HBM, determine the maximum batch size for each sequence
   length, accounting for weight memory.
5. Calculate the per-decode-step KV cache bandwidth requirement and determine when it
   exceeds weight-loading bandwidth.
6. Show how INT8 and FP8 KV cache quantisation change the sizing.
7. Extend to PagedAttention: explain how memory fragmentation affects the effective capacity.

---

## Part 1: Model Weight Memory

Enumerate all weight matrices. Using GQA with $h_{KV} = 8$, so Q projects to $d$ but K,V
project to $h_{KV} \cdot d_h = 8 \times 128 = 1024$ dimensions.

**Per-layer weights**:

| Component | Dimensions | Parameters |
|---|---|---|
| $W_Q$ (query projection) | $d \times d = 8192 \times 8192$ | $67,108,864$ |
| $W_K$ (key projection) | $d \times (h_{KV} d_h) = 8192 \times 1024$ | $8,388,608$ |
| $W_V$ (value projection) | $d \times (h_{KV} d_h) = 8192 \times 1024$ | $8,388,608$ |
| $W_O$ (output projection) | $d \times d = 8192 \times 8192$ | $67,108,864$ |
| $W_{\text{gate}}$ (SwiGLU) | $d \times d_{ff} = 8192 \times 28672$ | $234,881,024$ |
| $W_{\text{up}}$ (SwiGLU) | $d \times d_{ff} = 8192 \times 28672$ | $234,881,024$ |
| $W_{\text{down}}$ (SwiGLU) | $d_{ff} \times d = 28672 \times 8192$ | $234,881,024$ |
| RMSNorm (pre-attn, pre-FFN) | $2 \times d$ | $16,384$ |

**Parameters per layer**:

$$N_{\text{layer}} = 2d^2 + 2d \cdot h_{KV} d_h + 3d \cdot d_{ff} + 2d$$

$$= 2(8192)^2 + 2 \times 8192 \times 1024 + 3 \times 8192 \times 28672 + 2 \times 8192$$

$$= 134,217,728 + 16,777,216 + 704,643,072 + 16,384$$

$$= 855,654,400 \approx 855.7\text{M parameters per layer}$$

**Total layer parameters** ($L = 80$ layers):
$$N_{\text{layers}} = 80 \times 855,654,400 = 68,452,352,000 \approx 68.5\text{B}$$

**Embedding and head** ($V \times d$, typically weight-tied):
$$N_{\text{embed}} = 32000 \times 8192 = 262,144,000 \approx 262\text{M}$$

**Total parameters**:
$$N_{\text{total}} \approx 68.5\text{B} + 0.26\text{B} \approx 68.8\text{B} \approx 70\text{B}$$

(The "70B" label is approximate — actual LLaMA-2 70B is 69.7B non-embedding parameters.)

**Weight memory at FP16**:
$$M_{\text{weights}} = N_{\text{total}} \times 2\text{B} = 68.8 \times 10^9 \times 2 = 137.6\text{GB}$$

---

## Part 2: Per-Token KV Cache Size

During decode, each new token appends one $K$ vector and one $V$ vector to the cache, per layer.

**K or V vector per head**: dimension $d_h = 128$, stored at FP16 ($2$ bytes):
$$\text{bytes per head, one tensor (K or V)} = d_h \times 2 = 128 \times 2 = 256\text{B}$$

**Per KV-head pair ($K$ and $V$ together)**:
$$\text{bytes per KV head} = 2 \times d_h \times 2 = 2 \times 128 \times 2 = 512\text{B}$$

**All KV heads per layer** ($h_{KV} = 8$ GQA heads):
$$\text{bytes per token per layer} = h_{KV} \times 2 \times d_h \times 2 = 8 \times 256 \times 2 = 4096\text{B} = 4\text{KB}$$

**Alternatively expressed**:
$$\text{bytes per token per layer} = 2 \times h_{KV} \times d_h \times 2 = 4 h_{KV} d_h = 4 \times 8 \times 128 = 4096\text{B}$$

**All layers**:
$$\text{bytes per token (all layers)} = L \times 4 h_{KV} d_h = 80 \times 4096 = 327,680\text{B} \approx 320\text{KB per token}$$

**For a sequence of $S$ tokens**:
$$M_{\text{KV}}(S) = S \times 320\text{KB} = S \times 327,680\text{B}$$

**Sanity check with alternative formula**:
$$M_{\text{KV}}(S) = 2 \times L \times S \times h_{KV} \times d_h \times \text{bpe} = 2 \times 80 \times S \times 8 \times 128 \times 2 = 327,680 \times S \text{ bytes}$$

---

## Part 3: KV Cache Size Table

$$M_{\text{KV}}(B, S) = B \times S \times 327,680\text{B}$$

| | $S=512$ | $S=2048$ | $S=4096$ | $S=8192$ |
|---|---|---|---|---|
| $B=1$ | 167.8 MB | 671.1 MB | 1,342.2 MB | 2,684.4 MB |
| $B=8$ | 1.34 GB | 5.37 GB | 10.74 GB | 21.47 GB |
| $B=32$ | 5.37 GB | 21.47 GB | 42.95 GB | 85.90 GB |
| $B=128$ | 21.47 GB | 85.90 GB | 171.80 GB | 343.60 GB |

**Derived calculation for one representative cell** ($B=32$, $S=2048$):
$$M_{\text{KV}} = 32 \times 2048 \times 327,680 = 32 \times 671,088,640 = 21,474,836,480\text{B} \approx 21.47\text{GB}$$

---

## Part 4: Maximum Batch Size Given 160 GB HBM

Total HBM memory must accommodate: weights + KV cache + activations (small, $\ll 1\text{GB}$).

$$M_{\text{HBM}} = M_{\text{weights}} + M_{\text{KV}}(B_{\max}, S) + M_{\text{activations}}$$

Available for KV cache:
$$M_{\text{KV, available}} = 160\text{GB} - 137.6\text{GB} = 22.4\text{GB}$$

$$B_{\max}(S) = \left\lfloor \frac{22.4 \times 10^9}{S \times 327,680} \right\rfloor$$

| $S$ | $B_{\max}$ | Calculation |
|---|---|---|
| 512 | **133** | $22.4\text{GB} / 167.8\text{MB} = 133.5$ |
| 2048 | **33** | $22.4\text{GB} / 671.1\text{MB} = 33.4$ |
| 4096 | **16** | $22.4\text{GB} / 1342.2\text{MB} = 16.7$ |
| 8192 | **8** | $22.4\text{GB} / 2684.4\text{MB} = 8.35$ |

**Key observation**: As sequence length quadruples ($512 \to 2048$), the maximum batch size
decreases proportionally ($133 \to 33 \approx 133/4$). The KV cache memory scales as
$B \times S$, so longer sequences directly reduce the viable batch size.

**Practical headroom**: In practice, reserve $5$–$10\%$ of HBM for:
- Activation tensors during forward pass
- CUDA/driver overhead
- PagedAttention block table metadata
- Safety margin for fragmentation

With 10% reserved: $M_{\text{KV, available}} \approx 22.4 - 16 = 6.4\text{GB}$ in the extreme,
but more realistically activations are $\sim 500\text{MB}$, overhead $\sim 1\text{GB}$, so
effective available $\approx 20.9\text{GB}$.

---

## Part 5: Per-Decode-Step KV Cache Bandwidth

At each decode step, to compute attention for all $B$ sequences with context length $S$,
the hardware must load the entire KV cache:

$$M_{\text{KV, per step}}(B, S) = B \times S \times 327,680\text{B}$$

This is identical to the total KV cache size — the entire cache is read once per decode step.

**KV bandwidth dominates weight bandwidth when**:

$$M_{\text{KV}}(B,S) > M_{\text{weights}}$$
$$B \times S \times 327,680 > 137,600,000,000$$
$$B \times S > \frac{137.6 \times 10^9}{327,680} = 419,979 \approx 420,000$$

Equivalently: $B \times S > \frac{M_{\text{weights}}}{2 L h_{KV} d_h \times \text{bpe}}$

**Crossover table**:

| $B$ | $S$ at crossover | Nearest standard $S$ |
|---|---|---|
| 1 | 420,000 | Not reachable (>max context) |
| 8 | 52,500 | 65,536 (would be KV-dominated) |
| 32 | 13,125 | 16,384 |
| 64 | 6,563 | 8,192 |
| 128 | 3,281 | 4,096 |

**Insight from GQA**: If we had used MHA ($h_{KV} = 64$) instead of GQA-8:
$$M_{\text{KV, MHA, per token}} = 2 \times 80 \times 64 \times 128 \times 2 = 2,621,440\text{B} \approx 2.5\text{MB/token}$$

Crossover with MHA: $B \times S > \frac{137.6\text{GB}}{2.5\text{MB}} = 55,040$, i.e. 8× lower
crossover than GQA-8, meaning MHA becomes KV-bandwidth-dominated at much smaller batch/context sizes.

---

## Part 6: KV Cache Quantisation

### INT8 KV Cache

Halve the bytes per element: $\text{bpe} = 1$ instead of $2$.

$$M_{\text{KV, INT8}}(S) = S \times 163,840\text{B/token} = \frac{S \times 327,680}{2}$$

New capacity (same 22.4 GB available):
$$B_{\max}^{\text{INT8}}(S) = 2 \times B_{\max}^{\text{FP16}}(S)$$

| $S$ | FP16 $B_{\max}$ | INT8 $B_{\max}$ | Gain |
|---|---|---|---|
| 512 | 133 | 267 | $2\times$ |
| 2048 | 33 | 66 | $2\times$ |
| 4096 | 16 | 33 | $\approx 2\times$ |
| 8192 | 8 | 16 | $2\times$ |

Bandwidth per decode step is also halved proportionally.

**Accuracy impact**: INT8 KV cache introduces quantisation error in the attention computation.
Research shows $< 0.5\%$ degradation on most benchmarks when using per-head or per-token
scales. This is widely deployed in production (vLLM, TensorRT-LLM support INT8 KV).

### FP8 E4M3 KV Cache

Same as INT8 in byte count ($1$ byte per element), but uses floating-point representation
which better matches the distribution of attention keys and values.

FP8 KV is now supported natively on H100 (Hopper) and used in NVIDIA Transformer Engine.
The hardware dequantises FP8 KV vectors to FP16 on the fly during the attention computation,
with minimal overhead when done in the memory-fetch pipeline.

### INT4 KV Cache

Aggressive: $0.5$ bytes per element — $4\times$ size reduction from FP16.

$$B_{\max}^{\text{INT4}}(S = 2048) = 4 \times 33 = 132$$

Accuracy degradation becomes significant without careful per-token, per-channel quantisation.
KIVI and similar methods show INT4 KV with group quantisation can achieve acceptable accuracy.

**Summary**:

| KV Precision | Relative size | Max batch ($S=2048$, 160 GB) |
|---|---|---|
| FP16 | $1\times$ | 33 |
| BF16 | $1\times$ | 33 |
| FP8 / INT8 | $0.5\times$ | 66 |
| INT4 | $0.25\times$ | 132 |

---

## Part 7: PagedAttention Memory Fragmentation

### Motivation

The naive approach pre-allocates a contiguous block of size $S_{\max} \times \text{KV size/token}$
for each sequence at request creation. If the sequence finishes early at $S_{\text{actual}} < S_{\max}$,
the unused memory is wasted (internal fragmentation). Requests with unknown final lengths
must be over-allocated.

### PagedAttention Block Model

PagedAttention divides the KV cache into fixed-size **pages** (or blocks) of $P$ tokens each.
A block table maps each sequence's logical token positions to physical page addresses in HBM.
Pages are allocated on demand as the sequence grows.

**Block size trade-off**:

| Block size $P$ | Internal fragmentation | Block table overhead | Typical choice |
|---|---|---|---|
| 1 token | Zero fragmentation | Very large table | Too expensive |
| 16 tokens | $\leq 15$ tokens wasted | Small table | Used by vLLM |
| 128 tokens | $\leq 127$ tokens wasted | Tiny | Low fragmentation with large SRAM |

**Fragmentation analysis** for block size $P = 16$:

In the worst case, the last page of each sequence is only 1 token full (15 tokens wasted).
For a batch of $B=32$ sequences: $32 \times 15 \times 163,840\text{B} \approx 75\text{MB}$ wasted.
This is $75\text{MB} / 22.4\text{GB} = 0.33\%$ overhead — negligible.

For block size $P = 16$: average waste = $P/2 = 8$ tokens per sequence.

$$\text{Fragmentation fraction} = \frac{P/2}{S_{\text{avg}}} = \frac{8}{S_{\text{avg}}}$$

At $S_{\text{avg}} = 1024$: $0.78\%$ fragmentation — acceptable.
At $S_{\text{avg}} = 64$: $12.5\%$ fragmentation — significant.

**Block table memory overhead**:

Each sequence needs at most $\lceil S_{\max}/P \rceil$ entries in the block table.
At $S_{\max} = 8192$, $P = 16$: $512$ entries per sequence.
Each entry is a physical page address ($4$ bytes): $512 \times 4 = 2\text{KB}$ per sequence.
For $B = 32$ sequences: $64\text{KB}$ total — negligible.

**Effective capacity with PagedAttention** vs naive:

Naive allocation wastes memory proportional to the gap between $S_{\max}$ and $S_{\text{actual}}$.
If requests average $S_{\text{actual}} = 0.5 S_{\max}$, naive allocation wastes 50% of KV memory.
PagedAttention recovers this waste (minus the $P/2$ average fragmentation):

$$\text{Effective capacity gain} \approx \frac{S_{\max}/S_{\text{actual}} - P/(2 S_{\text{actual}})}{1} \approx 2\times \text{ for common workloads}$$

This is why systems using PagedAttention can serve $\approx 2\times$ more concurrent requests
than naive implementations, even without changing hardware.

---

## Summary: KV Cache Sizing at a Glance

**Per-token KV cache size** (general formula):

$$\text{Bytes/token} = 2 \times L \times h_{KV} \times d_h \times \text{bpe}$$

For the 70B model studied (FP16):

$$= 2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ B} \approx 320 \text{ KB/token}$$

**Available KV memory** on a 160 GB system:

$$M_{\text{KV, avail}} = 160\text{ GB} - M_{\text{weights}} = 160 - 137.6 = 22.4\text{ GB}$$

**Maximum concurrent tokens** (FP16 KV):

$$T_{\max} = \left\lfloor\frac{22.4\text{ GB}}{320\text{ KB}}\right\rfloor = 71,680 \text{ tokens}$$

This 71,680-token budget is shared across all sequences in the batch. Whether it is
spent on more sequences or longer sequences is a scheduling decision:
- $32 \times 2048 = 65,536$ tokens: fits.
- $16 \times 4096 = 65,536$ tokens: also fits.
- $8 \times 8192 = 65,536$ tokens: also fits.
- All three configurations use approximately the same KV memory — the memory budget
  is a constraint on the **product** $B \times S$, not on either independently.
