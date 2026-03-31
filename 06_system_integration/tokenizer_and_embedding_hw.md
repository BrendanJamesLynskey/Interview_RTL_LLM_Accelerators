# Tokenizer and Embedding Hardware

## Overview

Before the transformer stack receives its first tensor, the input text must be converted to token
IDs (tokenization), then each ID must be mapped to a dense embedding vector (embedding table
lookup). At the output side, the logit vector must be projected back to vocabulary space, sampled,
and decoded. These pre- and post-processing steps are often overlooked in accelerator design but
can be surprisingly expensive in latency and area. This document examines the hardware architecture
and implementation tradeoffs for each stage.

---

## Tier 1: Fundamentals

### Q1. Describe the BPE (Byte-Pair Encoding) tokenization algorithm and explain why it is
difficult to accelerate in hardware.

**Answer.**

BPE tokenization converts a string of Unicode characters into a sequence of integer token IDs.
It was introduced by Sennrich et al. (2016) and is used in GPT-2/GPT-4, LLaMA, Mistral, and
most modern LLMs.

**Algorithm (at inference time — encoding only):**

1. **Pre-tokenize:** Split the input string into words (or subwords) using a regex (e.g.,
   `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|...`).
2. **Character-level initialisation:** Convert each word to its UTF-8 bytes; treat each byte as an
   initial token.
3. **Merge loop:** Repeatedly find the most frequent adjacent pair of tokens and replace them with
   the merged token, following a pre-trained merge table (priority-ordered list of ~50K pairs).
4. **Output:** The resulting token sequence is encoded as integer IDs from a vocabulary table.

**Why it is hard to accelerate in hardware:**

| Challenge | Description |
|---|---|
| Data-dependent control flow | The number of merge iterations depends on the input text; no fixed loop count |
| Sequential dependency | Each merge step depends on the result of the previous one; no easy parallelism |
| Variable-length output | Input bytes → output tokens has a variable compression ratio (~2–5×) |
| Large lookup table | The merge priority table is 50K+ entries; random access pattern |
| String processing | Requires regex matching, UTF-8 decoding — inherently irregular |

As a result, tokenisation is almost universally performed on the **host CPU** rather than on the
accelerator. Hardware tokenisation engines exist (e.g., NVIDIA's cuDF has GPU-accelerated
tokenisers), but the control-flow complexity makes them power-inefficient relative to CPUs.

**Throughput perspective:** At 1 Mtokens/s accelerator throughput, the tokeniser only needs to
process ~1 Mtoken/s of input (batch of 1000 requests × 1000 tokens each per second). A single
CPU core running a well-optimised tokeniser (Hugging Face `tokenizers`, written in Rust) can
sustain ~10 Mtokens/s — easily keeping up without dedicated hardware.

---

### Q2. What is an embedding table lookup and what are its memory access characteristics?

**Answer.**

An embedding table is a 2D matrix $E \in \mathbb{R}^{V \times d}$ where:
- $V$ is the vocabulary size (e.g., 32000 for LLaMA-2, 100277 for GPT-4's cl100k_base)
- $d$ is the embedding dimension (e.g., 4096 for LLaMA-2 7B)

**Lookup operation:**

Given a batch of token IDs $\{t_0, t_1, \ldots, t_{S-1}\}$, the embedding for token $t_i$ is:

$$e_i = E[t_i, :] \qquad \text{(row }t_i\text{ of the embedding matrix)}$$

This is a **gather** operation: for each token ID, fetch one row of $E$ from memory.

**Memory characteristics:**

For LLaMA-2 7B (BF16):
$$\text{embedding table size} = V \times d \times 2\ \text{B} = 32000 \times 4096 \times 2 = 256\ \text{MB}$$

Access pattern: random row accesses with no spatial locality across tokens (consecutive tokens in
the sequence may map to any row of $E$). This is an **irregular (scatter-gather) memory access
pattern**, extremely unfriendly to cache.

**Cache miss analysis:**

Assuming a 32 MB SRAM cache:
- Cache holds $32\ \text{MB} / (4096 \times 2\ \text{B/row}) = 4096$ rows.
- For a vocabulary of 32000 rows, only $4096/32000 \approx 12.8\%$ of the vocabulary fits in cache.
- For a batch of $B$ sequences each of length $S$, with approximately uniform token distribution,
  cache miss rate is high unless the working-set vocabulary is much smaller than $V$.

For this reason, embedding lookup is typically **memory-bandwidth-bound**, not compute-bound: the
load of $d$ values from DRAM per token dominates.

---

### Q3. Should the embedding table reside on the accelerator (HBM) or on the host CPU (DRAM)?
Justify your answer with numbers.

**Answer.**

**Case: Accelerator HBM resident**

Embedding table access latency: ~100 ns (HBM random access).
Bandwidth available: 3.2 TB/s (HBM3 with 8 stacks).
Per-token embedding bandwidth: $4096 \times 2\ \text{B} = 8\ \text{KB}$.
Throughput: $3.2\ \text{TB/s} / 8\ \text{KB/token} = 400\ \text{Mtokens/s}$

This is more than sufficient — the prefill token processing rate is far below 400 Mtokens/s.

**Case: Host DRAM resident (PCIe fetch)**

Embedding table access latency: ~1–3 µs (PCIe read round trip, far higher than the 100 ns HBM).
DMA bandwidth: 63 GB/s (PCIe Gen5 x16).
Per-token throughput: $63\ \text{GB/s} / 8\ \text{KB/token} \approx 7.9\ \text{Mtokens/s}$

Still sufficient in throughput, but latency is 10–30× worse per token.

**Recommendation: HBM resident.**

The 256 MB table is small relative to HBM capacity (64–128 GB HBM3 per accelerator). Keeping it
on HBM avoids PCIe round trips, which become significant at low batch sizes. The embedding lookup
initiates the entire inference pipeline; any latency here adds directly to TTFT.

**Exception:** For models with extremely large vocabularies (e.g., a multilingual model with
$V = 500$K, table size ~4 GB), the table is large enough to fragment the HBM budget. In this case,
a tiered approach is used: frequently-accessed tokens (top-10K by frequency, covering >95% of
real-world text) are cached in a 160 MB SRAM, and the full table is in HBM.

---

## Tier 2: Intermediate

### Q4. Design a hardware embedding lookup unit for a batched input with $B = 32$ sequences of
length $S = 512$ each. What memory organisation and pipelining strategy achieves maximum
throughput?

**Answer.**

**Inputs:** $B \times S = 16384$ token IDs per prefill step.
**Output:** $16384 \times d \times 2\ \text{B} = 16384 \times 8192\ \text{B} \approx 128\ \text{MB}$ of embedding vectors.

**Throughput target:** Overlap embedding lookup with the first transformer layer's GEMM so that
embedding lookup is not on the critical path.

**Memory organisation:**

Partition the embedding table into $P$ banks, assigning rows to banks by:
$$\text{bank}(t) = t \bmod P$$

With $P = 8$ banks, 8 simultaneous row reads are possible if the 16384 token IDs are uniformly
distributed. A small address translation table maps token ID to (bank, row-within-bank).

**Conflict analysis:** With uniform random IDs from a 32K vocabulary:

$$P(\text{conflict in one cycle with }P=8) = 1 - \frac{P!/(P-n)!}{P^n}$$

For $n = 8$ requests to 8 banks: $P(\text{no conflict}) = \frac{8!}{8^8} \approx 0.24$.
Conflicts are common; a **bank conflict queue** holds retried requests. With a 4-deep queue per
bank and FIFO scheduling, throughput is ~75% of peak (6 rows per cycle out of 8 possible).

**Pipeline stages:**

```
Stage 1: Token ID buffer drain → bank address decode
Stage 2: Bank SRAM read (2 cycles for SRAM: address setup + data out)
Stage 3: Data output to downstream embedding FIFO
```

At 1.5 GHz, 8 banks × 0.75 efficiency × 1.5 GHz = 9 Giga-row-accesses/second.
Per-embedding read: $4096 \times 2\ \text{B} = 8\ \text{KB}$, but at 512-bit SRAM bus width:
$8192\ \text{B} / 64\ \text{B} = 128$ beats per row.

Total time for 16384 embeddings: $16384 \times 128 / (8\ \text{banks} \times 1.5\ \text{GHz}) \approx 174\ \mu\text{s}$.

The first transformer layer (GEMM) on this same hardware takes approximately:

$$t_{GEMM} = \frac{2 \times 16384 \times 4096 \times 4096}{300 \times 10^{12}\ \text{TOPS}} \approx 1.8\ \text{ms}$$

Since $174\ \mu\text{s} \ll 1.8\ \text{ms}$, the embedding lookup fits comfortably within the
pipelined prefill time.

---

### Q5. Describe the output projection (de-embedding) step and why it is the most expensive single
operation in a transformer layer count sense. How should it be implemented in hardware?

**Answer.**

The final transformer layer produces a hidden state $h \in \mathbb{R}^{B \times d}$. To produce
per-token vocabulary logits, the model applies a linear projection:

$$L = h \cdot E^T, \qquad L \in \mathbb{R}^{B \times V}$$

where $E^T \in \mathbb{R}^{d \times V}$ is the **transposed** embedding table (often tied to the
input embedding matrix — "weight tying").

**Computational cost:**

$$\text{FLOPs} = 2 \times B \times d \times V = 2 \times B \times 4096 \times 32000 \approx 262\ \text{MFLOPs per token in batch}$$

For $B = 32$: $262 \times 32 = 8.4\ \text{GFLOPs}$.

At 300 TOPS (BF16): $8.4\ \text{GFLOPs} / 300\ \text{TOPS} = 28\ \mu\text{s}$.

This is relatively fast because $V$ is large but $B$ is small during decode (batch of 1 token per
sequence per decode step). The memory bandwidth cost is:

$$\text{Weight bytes} = d \times V \times 2\ \text{B} = 4096 \times 32000 \times 2 = 256\ \text{MB}$$

At HBM bandwidth 3.2 TB/s: $256\ \text{MB} / 3.2\ \text{TB/s} = 80\ \mu\text{s}$.

**This operation is memory-bandwidth-bound during decode** (arithmetic intensity = FLOPs / bytes =
$2BV d / (Vd \cdot 2) = B = 1$ FLOP/byte for $B = 1$, far below the roofline crossover of ~200
FLOPs/byte for 300 TOPS / 3.2 TB/s).

**Hardware implementation:**

1. **During prefill:** All $B \times S$ hidden states are computed simultaneously. Output projection
   is a standard GEMM with dimensions $[BS, d] \times [d, V]$. Arithmetic intensity $\approx BS$,
   which for $BS = 16384$ gives ~16 FLOPs/byte — still bandwidth-bound.

2. **During decode:** Only $B$ new hidden states per step. Implemented as $B$ parallel GEMV
   (matrix-vector multiplies). Each GEMV fetches all $V \times d$ weights from HBM: the critical
   path is HBM bandwidth.

3. **Optimisation — top-k filtering before full projection:**
   Rather than projecting to all $V = 32000$ logits, restrict the projection to the top-$k$
   candidate tokens from the previous step (typically $k = 128$ for beam search or nucleus
   sampling). This reduces the projection to $[B, d] \times [d, k]$, shrinking memory access by
   $V/k = 250\times$. Requires a small "candidate token" SRAM holding $k$ embedding vectors.

---

### Q6. Design a hardware top-k sampling unit for LLM output. What data structures and algorithms
are suitable for hardware implementation?

**Answer.**

**Problem:** Given logit vector $L \in \mathbb{R}^V$ ($V = 32000$), compute:
1. Top-k selection: find the $k$ indices with highest values ($k \leq 50$).
2. Softmax over the top-k logits.
3. Multinomial sampling from the softmax distribution.

**Hardware-friendly top-k algorithm: parallel merge sort network**

A direct sort of $V = 32000$ elements is expensive (comparison network depth $O(\log^2 V)$).
Instead, use a **tournament tree** (heap-based top-k):

- Maintain a min-heap of size $k$ in a register file.
- Stream logits from SRAM, one per cycle.
- On each new logit: compare with heap minimum; if greater, replace minimum and heapify ($O(\log k)$
  comparisons, $\log k = 6$ for $k = 64$).
- After $V = 32000$ cycles: heap contains top-k logits.

Total cycles: $V + k \log k = 32000 + 64 \times 6 = 32384 \approx 32K$ cycles.
At 1.5 GHz: $32384 / 1.5\ \text{GHz} \approx 21.6\ \mu\text{s}$.

**Softmax computation (over top-k logits):**

$$p_i = \frac{e^{L_i / T}}{\sum_{j=1}^{k} e^{L_j / T}}$$

Hardware implementation:
1. Find $L_{max}$ among top-k (trivial from heap root).
2. Compute $e^{(L_i - L_{max}) / T}$ for each of the $k$ elements using a piecewise-polynomial
   approximation to $e^x$ (3rd-order polynomial, $\pm 1$ ULP accuracy over $[-10, 0]$).
3. Sum the $k$ exponentials (adder tree: $\log_2 k = 6$ levels, 6 cycles).
4. Divide each exponential by the sum (reciprocal + multiply, 4 cycles with Newton-Raphson).

Total: ~$k + \log_2 k + 4 \approx 76$ cycles for $k = 64$.

**Multinomial sampling hardware:**

1. Generate a uniform random number $u \in [0, 1)$ from a hardware PRNG (LFSR or xorshift64).
2. Compute the cumulative distribution: $C_i = \sum_{j=0}^{i} p_j$.
3. Select the smallest $i$ such that $C_i > u$ (binary search over $k$ elements: $\log_2 k = 6$
   comparisons with a comparison tree, 6 cycles).

**Total top-k sampling latency:** $\approx 21.6\ \mu\text{s} + 0.05\ \mu\text{s} + 0.004\ \mu\text{s} \approx 21.7\ \mu\text{s}$.

This is dominated by the logit streaming time (21.6 µs) and is feasible on the accelerator.

**Alternative: Top-p (nucleus) sampling**

Accumulate sorted probabilities until sum exceeds $p_{nucleus}$ (e.g., 0.9). The hardware
implementation differs only in the stopping criterion — the heap streaming and softmax remain
the same.

---

## Tier 3: Advanced

### Q7. A team proposes implementing the full tokenizer (BPE encode + embedding lookup) on the
accelerator to reduce latency. Analyse the feasibility and design the tokenizer's merge engine
in hardware.

**Answer.**

**Feasibility analysis:**

The tokenizer must complete before the first prefill GEMM. Typical prefill latency for a 512-token
prompt on a 7B model is ~10 ms. The software tokenizer on the host CPU takes:

$$t_{tokenize,CPU} \approx \frac{512\ \text{tokens}}{10\ \text{Mtokens/s}} \approx 51\ \mu\text{s}$$

This is negligible vs. 10 ms prefill. There is **no latency benefit** to hardware tokenisation.

**When hardware tokenisation makes sense:**
- Very high batch rates (>100K requests/s) where CPU tokenization becomes a bottleneck.
- Edge deployment where no host CPU is present.
- Streaming input from a sensor (e.g., ASR output feeding directly into an LLM).

**Hardware BPE merge engine design:**

**Data structures:**
- **Merge priority table (MPT):** A content-addressable memory (CAM) with 50,000 entries, each
  mapping a pair of adjacent token IDs to a merged token ID and a merge priority (integer rank).
  CAM is wide enough for a 32-bit key (two 16-bit token IDs) and a 24-bit value (merged ID +
  rank).
- **Token buffer:** A register file of 2048 entries (maximum token sequence length), each 16 bits
  wide, representing the current token sequence.
- **Priority queue:** A min-heap of (priority, position) pairs, holding the current merge
  candidates.

**Pipeline:**

```
Cycle  1..N  : Load token sequence into register file from SRAM
Cycle  N+1   : Scan all adjacent pairs; issue parallel CAM lookups
Cycle  N+2   : CAM returns merge priorities for all pairs (pipelined)
Cycle  N+3   : Build priority queue (sort network for N-1 entries, O(log^2 N))
Cycle  N+4   : Extract minimum-priority entry (the merge to perform)
Cycle  N+5   : Update register file: replace pair with merged token, shift tail
Cycle  N+6   : Update affected CAM lookups (only 2 new pairs created per merge)
Repeat until no more pairs exist in CAM.
```

**Parallelism opportunity:**
If no two active merge candidates are adjacent (i.e., they operate on non-overlapping positions),
they can be executed simultaneously. This requires a **conflict detection network**: a $(N-1)
\times (N-1)$ dependency matrix that is $O(N^2)$ in area. For $N = 512$, this is prohibitive.

**Practical design:** Process 4 non-conflicting merges per cycle using a greedy non-overlapping
selection algorithm (linear scan, $O(N)$ per step). This improves throughput ~4× over serial
merging with only a small area overhead.

**Area estimate at 7 nm:**
- CAM (50K × 48 bits): ~1.5 mm²
- Register file (2048 × 16 bits): ~0.05 mm²
- Priority queue (2048 entries): ~0.2 mm²
- Control FSM: ~0.1 mm²
- **Total: ~1.85 mm²**

This is non-trivial area. Given that the tokenizer contributes <1% to total inference latency,
this investment is only justified for the edge/streaming use cases above.

---

### Q8. Describe the de-tokenizer hardware (converting output token IDs back to text). What table
lookups and state machines are required, and how do multi-byte UTF-8 sequences complicate the
design?

**Answer.**

**De-tokenisation (decode)** converts a sequence of token IDs back to a UTF-8 string.

**Data structure:**
A vocabulary lookup table maps each token ID to a variable-length byte string:
- Token 0–255: single ASCII bytes (trivial 1-byte outputs)
- Token 256+: multi-byte strings, e.g., token 1234 → `" the"` (5 bytes including leading space)

Storage: average token length is ~4 bytes. For $V = 32000$ tokens:
$$\text{table size} \approx 32000 \times 4\ \text{B} = 128\ \text{KB}$$

This fits comfortably in on-chip SRAM.

**Hardware implementation:**

```
Token ID stream
     │
     ▼
┌───────────────┐
│ ID-to-offset  │  Lookup table: token_id → (byte_offset, byte_length)
│ SRAM (128KB)  │  32000 × (24-bit offset + 8-bit length) = 128 KB
└──────┬────────┘
       │
       ▼
┌───────────────┐
│ Byte-string   │  Sequential SRAM indexed by byte_offset
│ SRAM (128KB)  │  128 KB of concatenated token byte strings
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  UTF-8 valid. │  Check that output bytes form valid UTF-8 sequences
│  and buffer   │
└──────┬────────┘
       │
       ▼
   Output byte stream
```

**UTF-8 complication: split tokens across multi-byte codepoints**

The BPE tokenizer may split a multi-byte UTF-8 codepoint across two tokens. For example, the
UTF-8 encoding of '€' is `0xE2 0x82 0xAC` (3 bytes). A tokenizer might produce:

- Token A → `\xe2\x82` (first 2 bytes)
- Token B → `\xac` (third byte)

Neither A alone nor B alone is a valid UTF-8 sequence. The de-tokenizer must:

1. Buffer incomplete multi-byte sequences: a 4-byte shift register holds bytes waiting for a
   complete codepoint.
2. Detect the leading byte type (0xxxxxxx = 1-byte, 110xxxxx = 2-byte, 1110xxxx = 3-byte,
   11110xxx = 4-byte).
3. Accumulate continuation bytes (10xxxxxx) until the expected count is met.
4. Emit the complete codepoint (or convert to the output encoding, e.g., UTF-16 for some
   applications).

**State machine for UTF-8 validation:**

```
States: IDLE, EXPECT_1, EXPECT_2, EXPECT_3

Transitions:
  IDLE + leading_1byte → emit 1 byte → IDLE
  IDLE + leading_2byte → EXPECT_1
  IDLE + leading_3byte → EXPECT_2
  IDLE + leading_4byte → EXPECT_3
  EXPECT_n + continuation → EXPECT_(n-1) if n>1, else emit codepoint → IDLE
  any + invalid_byte → emit U+FFFD (replacement character) → IDLE
```

This state machine is synthesised as a simple 2-bit state register with 8-bit input, trivial in
hardware. The de-tokenizer does not need to be on the accelerator — the byte string SRAM fits in
128 KB and the state machine is combinatorial. It can run on the host CPU after the token IDs are
DMA'd back, consuming less than 1% of a CPU core.

**When to accelerate on-chip:** Only if the host CPU would become a bottleneck at very high token
rates (>100 Mtokens/s), which requires an accelerator throughput several orders of magnitude beyond
current hardware. For all foreseeable deployments, de-tokenisation belongs on the host CPU.
