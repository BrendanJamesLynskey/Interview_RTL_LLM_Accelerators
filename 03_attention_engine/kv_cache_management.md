# KV Cache Management in Hardware

## Background

The key-value (KV) cache stores the K and V projection outputs for all previously processed tokens
so they need not be recomputed during autoregressive decoding. It is the dominant memory consumer in
LLM inference and the primary reason that hardware design for LLMs differs fundamentally from
training accelerator design.

**Why the KV cache exists**: During decode step t, the attention computation requires K and V vectors
for all tokens 0..t-1. These were computed during prefill (tokens 0..S_prompt-1) and during earlier
decode steps. Without a cache, they would need to be recomputed from the original token embeddings
through all preceding transformer layers — O(t * L * d_model^2) work per step versus O(d_model^2)
with a cache.

**Scope**: This document focuses on the hardware controller design for KV cache SRAM/HBM,
allocation policies, access patterns, and hardware implications of recent algorithmic advances
(PagedAttention, streaming attention).

---

## Tier 1 — Fundamentals

### Q1: What data is stored in the KV cache, and how is it organised in memory for a transformer with L layers, H heads, and d_head dimensions?

**Answer**

For each token position t and each transformer layer l, the KV cache stores:
- K tensor: shape [H, d_head] in FP16 or BF16
- V tensor: shape [H, d_head] in FP16 or BF16

**Total per token per layer**: `2 * H * d_head * 2 bytes`

For LLaMA-2 7B (L=32, H=32, d_head=128):
```
Per token per layer = 2 * 32 * 128 * 2 = 16,384 bytes = 16 KB
Per token all layers = 32 * 16 KB = 512 KB
```

**Memory layout options**:

Option A — Layer-first, then position:
```
[Layer 0: K[0], V[0], K[1], V[1], ..., K[S-1], V[S-1]]
[Layer 1: K[0], V[0], K[1], V[1], ..., K[S-1], V[S-1]]
...
```
Each layer's cache is contiguous. Accessing all keys for layer 0 is sequential — good for the
attention computation that reads the full K cache for one layer at a time.

Option B — Position-first, then layer:
```
[Token 0: L0_K, L0_V, L1_K, L1_V, ..., L31_K, L31_V]
[Token 1: ...]
...
```
All data for one token is contiguous. Good for appending a new token (one contiguous write)
but poor for attention (scattered reads across layers).

**Recommendation**: Option A (layer-first) is standard because the access pattern during attention
computation reads the full K (or V) cache for one layer sequentially, maximising DRAM burst length
and prefetch effectiveness.

**Head interleaving**: Within a layer, keys for all H heads are typically interleaved per position:
`K[pos][0], K[pos][1], ..., K[pos][H-1]`. This allows all H head keys for a given position to be
loaded in one burst, enabling parallel score computation across heads.

---

### Q2: How does KV cache size scale with batch size and sequence length? Why is this a challenge for hardware?

**Answer**

**Scaling equation**:
```
KV cache size = B * S * L * 2 * H * d_head * bytes_per_element
```

Where B = batch size, S = sequence length, L = layers.

**Numerical examples** (LLaMA-2 7B, BF16):

| B  | S    | KV cache size |
|----|------|---------------|
| 1  | 2048 | 1 GB          |
| 1  | 8192 | 4 GB          |
| 4  | 2048 | 4 GB          |
| 16 | 2048 | 16 GB         |
| 1  | 32768| 16 GB         |

LLaMA-2 7B model weights occupy ~14 GB in FP16. A batch of 16 at 2048 tokens uses as much
memory for KV cache as the weights themselves.

**Hardware challenges**:

1. **Dynamic allocation**: At request start, the maximum sequence length is unknown. The cache
   must grow token by token during decode. Static pre-allocation wastes memory; dynamic
   allocation requires a memory manager.

2. **Memory fragmentation**: Multiple concurrent requests of different lengths create
   fragmentation. A 2048-token slot cannot be reused by two 1024-token requests without
   defragmentation logic.

3. **Bandwidth**: At each decode step, the entire KV cache for all active sequences must be
   read. For B=16, S=2048: reading `16 * 2048 * 512 KB = 16 GB` per step across all layers.
   At 2 TB/s HBM bandwidth, this takes 8 ms — fundamentally limiting decode throughput.

4. **Eviction decisions**: When memory is full, which sequence's KV cache to evict is a
   policy decision that must be fast (hardware or firmware, not general-purpose OS scheduler).

---

### Q3: Describe the write pattern for the KV cache during prefill versus decode phases.

**Answer**

**Prefill write pattern**:
All S_prompt tokens are processed in parallel (one forward pass). The K and V projections produce
K and V tensors of shape [S_prompt, H, d_head]. These are written to the KV cache in a bulk
operation: S_prompt * H * d_head * 2 * 2 bytes per layer. This is a sequential write of a large
contiguous block — maximally efficient for DRAM (full row bursts, no wasted bandwidth).

**Decode write pattern** (one new token per step):
Each decode step appends one new K vector [H, d_head] and one V vector [H, d_head] to each layer's
cache. This is a small write: `H * d_head * 2 * 2 = 64 KB` per layer (LLaMA-2 7B), appended at
position S_current = S_prompt + decode_step.

The append address is: `base_address + layer * layer_stride + current_length * token_stride`.
The memory controller needs to track `current_length` per sequence and compute the write address
each step. For HBM, writing 64 KB per layer means one burst per layer — this is short relative
to the read traffic but still adds up: 32 layers * 64 KB = 2 MB of writes per decode step.

**Write-after-read ordering**: The new K and V vectors must be written to the cache before the
attention computation for the current step reads from the cache — otherwise the current token
would not attend to itself. In practice, the new token always attends to itself (its mask bit
is 1), so the write must complete before the score computation for position current_length begins.
Hardware must enforce this ordering either by sequencing (write first, then read) or via a bypass
register that provides the new K to the score unit without going through the SRAM.

---

## Tier 2 — Intermediate

### Q4: Design a KV cache memory controller. Specify the signals, internal state, and key FSM states for handling prefill writes, decode appends, and attention reads.

**Answer**

**Interface signals**:

```
// Clock and reset
input  clk, rst_n

// Configuration
input  [11:0] seq_len_max        // Maximum supported sequence length
input  [4:0]  num_layers         // L
input  [5:0]  num_heads          // H
input  [6:0]  head_dim           // d_head

// Write port (prefill bulk write or decode append)
input         wr_valid           // Write request
input  [1:0]  wr_type            // 0=prefill, 1=decode_append
input  [11:0] wr_pos             // Token position (decode) or start pos (prefill)
input  [4:0]  wr_layer           // Layer index
input  [127:0] wr_data           // 8 BF16 values per cycle (128 bits)
input         wr_last            // Last beat of this write
output        wr_ready           // Controller accepts write

// Read port (attention computation)
input         rd_valid           // Read request
input  [4:0]  rd_layer           // Layer index
input  [11:0] rd_pos             // Token position to read (sequential during attention)
input         rd_key_not_val     // 1=read K, 0=read V
input  [5:0]  rd_head            // Head index
output [15:0] rd_data            // One BF16 value per cycle
output        rd_valid_out       // rd_data is valid
output        rd_last            // Last element of this head's K/V vector

// Status
output [11:0] cur_seq_len        // Current sequence length (for attention addressing)
output        cache_full         // Cache is at maximum capacity
```

**Internal state**:
```
reg [11:0] seq_length;           // Current number of tokens in cache
reg [31:0] rd_address;           // Current SRAM/HBM read byte address
reg [6:0]  rd_beat_count;        // Beats remaining in current read
```

**FSM states**:

```
IDLE: No operation. Monitor wr_valid and rd_valid.

PREFILL_WRITE: Bulk write of S_prompt token K/V vectors.
  - Compute base address: layer * layer_stride
  - Increment write pointer each beat
  - On wr_last: update seq_length = wr_pos + tokens_written
  - Transition: back to IDLE

DECODE_APPEND: Single-token K/V write.
  - Address = layer_base + seq_length * token_stride
  - Write H * d_head * 2 beats for K, then H * d_head * 2 beats for V
  - On completion: seq_length++
  - Transition: back to IDLE (or ATTN_READ if read is pending)

ATTN_READ: Sequential read for attention score/weighted-sum computation.
  - Address = layer_base + rd_pos * token_stride + rd_head * head_stride + kv_offset
  - Issue read; stream d_head beats to rd_data output
  - On rd_last: assert rd_last, transition to IDLE (or next read)
```

**Key design decisions**:

1. **Priority**: Write (append) must have higher priority than read, so the new token's K vector is
   available before the attention computation begins.
2. **Dual-port SRAM**: Use separate read and write ports to avoid arbitration overhead when append
   and attention read overlap in time.
3. **Address generation**: All addresses are computed combinatorially from layer, position, and head
   indices using pre-computed strides. No division needed — strides are powers of 2 if dimensions
   are chosen accordingly.

---

### Q5: Explain prefetch strategies for KV cache reads during attention computation. What information is available to a hardware prefetcher, and what is the ideal prefetch depth?

**Answer**

**Access pattern analysis**: During attention computation for layer l, query q, the access pattern
for the K cache is perfectly predictable:
```
For pos = 0, 1, 2, ..., current_seq_len - 1:
  For head = 0, 1, ..., H-1:
    Read K[l][pos][head][0..d_head-1]   // d_head * 2 bytes
```

This is a linear sequential scan of a contiguous memory region — the simplest possible pattern
for a hardware prefetcher to handle.

**Prefetch depth calculation**:

HBM access latency: ~200-400 ns (100-200 cycles at 500 MHz).
HBM bandwidth: 2 TB/s => one d_head=128 BF16 vector (256 bytes) delivered in ~0.13 ns.

For a prefetcher to fully hide latency, it must issue requests N beats ahead where:
```
N = latency / bandwidth_per_element = 200 ns / 0.13 ns ≈ 1500 elements
```

In practice, issue 4-8 cache-line prefetches ahead (a cache line is typically 64-256 bytes).
With a 64-byte cache line and 256 bytes per K vector, issue prefetches ~8-32 vectors ahead.
This requires a 8-32 entry prefetch queue.

**Hardware prefetcher design**:

1. **Trigger**: When the attention controller transitions to ATTN_READ state, the prefetcher
   receives the starting address and total length.
2. **Queue**: A 16-entry circular queue of outstanding prefetch requests.
3. **Advance logic**: Each time the read consumer advances by one cache line, enqueue the
   next address at `current_address + 16 * 16_line_count`. Advance runs combinatorially, never
   stalling the producer.
4. **Bandwidth limit**: Rate-limit prefetch issuance to avoid consuming all HBM bandwidth for
   prefetch at the expense of demand reads. Prefetch requests get lower priority than demand reads
   at the HBM arbiter.

**Effectiveness**: With correct prefetch depth, effective memory latency for the sequential K/V
scan is reduced to zero — the consumer always finds data ready in a prefetch buffer. The
operation becomes purely bandwidth-limited, achieving the theoretical memory bandwidth utilisation.

---

### Q6: What is cache eviction and when is it needed in the KV cache context? Describe the trade-offs between FIFO, LRU, and sequence-complete eviction policies.

**Answer**

**When eviction is needed**: A hardware accelerator runs multiple requests concurrently for
throughput (batching). Each active request occupies KV cache space proportional to its current
sequence length. When a new request arrives but KV cache memory is full, an existing entry must
be evicted to make room.

Unlike CPU caches (which evict at the granularity of cache lines), KV cache eviction must be at
the granularity of entire sequences — evicting one token of a sequence while retaining the rest
is useless because the surviving tokens cannot form a coherent context without the evicted positions.

**FIFO (First-In, First-Out)**:
Evict the sequence that entered the cache earliest.
- Pros: Simple hardware — just a pointer to the oldest sequence.
- Cons: Evicts sequences regardless of their current length or proximity to completion.
  A long sequence near completion (step 99 of 100) may be evicted in favour of a new short
  sequence, wasting the work already done.

**LRU (Least Recently Used)**:
Evict the sequence whose most recent decode step was longest ago.
- Pros: Naturally keeps active (recently-touched) sequences in cache. Standard CPU cache wisdom.
- Cons: All active decode sequences are touched every step (since each step extends every
  active sequence). LRU degenerates to FIFO for uniform decode workloads. Only useful when
  some sequences are paused (waiting for user input, etc.).

**Sequence-complete eviction**:
Evict only sequences that have finished generation (produced an EOS token or reached max_new_tokens).
- Pros: Never discards in-progress work; the freed slot has zero reuse value.
- Cons: If no sequences are complete, no eviction is possible. Requires a waiting queue for
  new requests, reducing throughput in heavy-load scenarios.

**Priority score (practical policy)**:
Assign each sequence a score based on multiple factors:
```
priority = w1 * (steps_remaining_estimate) + w2 * (tokens_generated) - w3 * (time_waiting)
```
Evict the sequence with the lowest priority. This is the approach taken by vLLM and similar
serving frameworks. Hardware implementation requires a small priority comparator tree
(O(log B) comparator stages for B active sequences).

---

## Tier 3 — Advanced

### Q7: Describe PagedAttention and its hardware implications. How does a non-contiguous KV cache affect the memory controller and attention computation engine?

**Answer**

**PagedAttention concept** (from vLLM): The KV cache for each sequence is divided into fixed-size
pages (e.g., 16 tokens per page). Pages are allocated from a global free list, not contiguously.
A per-sequence page table maps logical block indices to physical page addresses, similar to virtual
memory paging in operating systems.

**Motivation**: In a traditional contiguous KV cache, the memory for sequence i must be reserved
as a contiguous region large enough for its maximum sequence length. If max_seq_len=2048 is reserved
but the sequence generates only 100 tokens, 95% of the reserved memory is wasted until the sequence
completes. PagedAttention allocates pages on demand, reducing internal fragmentation and allowing
higher memory utilisation.

**Hardware implications for the memory controller**:

1. **Address translation hardware**: The controller can no longer use a simple formula
   `base + pos * stride` to compute read addresses. It must perform a page table lookup:
   ```
   physical_page = page_table[sequence_id][pos / page_size]
   offset_in_page = pos % page_size
   address = physical_page * page_size_bytes + offset_in_page * token_stride + ...
   ```
   This requires a TLB (Translation Lookaside Buffer) or on-chip page table SRAM. A 512-entry
   TLB covers 512 active pages; at 16 tokens/page and 64 max sequences of 512 tokens each,
   this is `64 * 32 = 2048` pages — too large for a simple fully-associative TLB.

2. **Non-sequential DRAM access**: Sequential K/V reads are now scattered across non-contiguous
   physical addresses. DRAM row-buffer efficiency decreases. The prefetcher must be aware of
   page boundaries and fetch the page table entry for the next page before reaching the
   boundary.

3. **Page-boundary handling**: The attention computation engine receives a stream of K/V data.
   With paging, there may be a pipeline stall at each page boundary while the next physical
   address is resolved from the page table. For 16 tokens/page at d_head=128 BF16, each page
   holds 16 * 128 * 2 * 2 = 8 KB of K+V data. At 2 TB/s, this takes 4 ns to transfer —
   shorter than DRAM row activation latency. Prefetching the next page's address one page
   in advance is critical to avoid stalls.

4. **KV sharing across sequences**: PagedAttention enables K/V pages to be shared between
   sequences with identical prefix tokens (e.g., system prompts). The page table for two
   sequences can point to the same physical page for shared prefix tokens, halving the
   memory and bandwidth for the shared prefix. The hardware controller must support
   reference-counted shared pages: a page is only returned to the free list when its reference
   count drops to zero.

**Area cost estimate**: A page table for 64 sequences, each up to 4096 tokens / 16 tokens per page
= 256 pages per sequence, requires `64 * 256 * log2(total_pages)` bits. With 1024 total pages,
that is `64 * 256 * 10 = 163,840 bits = 20 KB` of on-chip SRAM for the page table — negligible.

---

### Q8: Analyse the memory bandwidth requirements for a continuous batching serving system. How does the memory controller design differ from a static-batch design?

**Answer**

**Continuous batching** (also called iteration-level scheduling): New requests are added to the
batch at the start of each decode step, and completed requests are removed. Batch size is not
fixed for the duration of a request — it changes every step.

**Memory bandwidth analysis** for B active sequences at various lengths S_1..S_B:

At decode step t, total KV cache read bandwidth per layer:
```
BW_read = sum_{i=1}^{B} S_i * H * d_head * 2 bytes  (K read)
        + sum_{i=1}^{B} S_i * H * d_head * 2 bytes  (V read)
        = 4 * H * d_head * sum(S_i)
```

If sequences have diverse lengths (some just started, some near end of generation), `sum(S_i)`
is lower than `B * S_max`. Continuous batching exploits this: short sequences consume little
bandwidth, allowing more of them to run concurrently.

**Memory controller differences**:

**Static batch**: All B sequences have the same length at all times (since they all started with
the same prompt length and have been decoded for the same number of steps). Memory access
addresses follow a simple pattern: base_i + t * stride, where t is the shared step count.

**Continuous batch**: Each sequence has a different current length S_i. The controller must
maintain per-sequence read pointers and read lengths. At each decode step:
1. For each sequence i, issue reads for S_i K vectors and S_i V vectors.
2. Different sequences complete at different times; their slots must be freed asynchronously.
3. New sequences start in the middle of a decode step's execution, requiring the controller to
   initiate prefill for new arrivals while simultaneously processing decode for existing ones.

**Controller state** (per sequence, up to B_max sequences):
```
struct seq_state {
  bool       active;
  uint32_t   kv_cache_base[L];   // Base address per layer
  uint16_t   current_length;     // S_i
  uint8_t    phase;              // PREFILL or DECODE
  bool       awaiting_eviction;
};
```

**Scheduling challenge**: The order in which per-sequence reads are issued to HBM affects
efficiency. Interleaving reads from multiple sequences on the same HBM bank causes row-buffer
thrashing. A good controller groups reads by DRAM row (spatial locality first, then temporal).
This requires a small reorder buffer that accumulates pending reads and sorts them by DRAM
address before issuance — similar in concept to a DRAM row-buffer-aware memory scheduler (FR-FCFS).
