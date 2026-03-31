# On-Chip Memory Hierarchy for LLM Accelerators

## Overview

The on-chip memory hierarchy is the single most area-dominant component in a modern LLM accelerator.
Getting the hierarchy design wrong means either starving compute units for data (bandwidth wall) or
wasting area on memory that cannot be accessed fast enough to help (capacity wall). This section
covers the architectural options at each level, the engineering decisions that differentiate them,
and the quantitative frameworks used to size each level correctly.

---

## Tier 1 — Fundamentals

### Q1. What are the three main on-chip memory structures used in accelerators, and how do they differ?

**Question:** Compare register files, SRAM scratchpads, and caches as on-chip memory structures.
For each, describe the access model, typical capacity, typical latency, and who controls what data
resides in them.

**Answer:**

**Register Files**

- Access model: Named registers, directly addressed by the instruction encoding. In a PE, each
  register is wired directly to the multiplier and adder inputs — access requires no address decode.
- Typical capacity: 8-256 registers per PE (16-bit: 256-512 bytes per PE; aggregated across a
  large PE array: tens of KB).
- Typical latency: 0 additional cycles (operand available at the start of the same cycle the
  instruction executes — combinational path).
- Who controls contents: Compiler / instruction stream. The programmer or compiler explicitly
  schedules register allocation and spill/fill.
- Energy: Cheapest per access — a register file read in a 7nm process costs roughly 0.01 pJ
  per bit versus 0.1 pJ for SRAM.

**SRAM Scratchpad (Software-Managed Buffer)**

- Access model: Addressed by a software-generated address (byte or word address). No hardware
  replacement policy. Explicit DMA or load/store instructions move data in and out.
- Typical capacity: 64 KB to 4 MB per cluster (GPUs call this "shared memory"; many DNN
  accelerators call it a "global buffer" or "local buffer").
- Typical latency: 1-10 cycles depending on SRAM depth and clock frequency.
- Who controls contents: Software (DMA engine or explicit load instructions). The programmer
  has full control and full responsibility — no automatic data management.
- Energy: ~0.1-1 pJ per bit depending on capacity; area-efficient for regular access patterns.

**Hardware Cache**

- Access model: Addressed by virtual or physical address. The hardware manages placement and
  eviction transparently using an LRU or pseudo-LRU replacement policy.
- Typical capacity: 32 KB (L1) to 32 MB (L2/L3) per die; varies widely.
- Typical latency: L1: 4-10 cycles; L2: 20-50 cycles; L3: 50-200 cycles.
- Who controls contents: Hardware replacement logic. Software has no direct control over which
  lines are present; hints (prefetch instructions, cache bypass hints) exist but are advisory.
- Energy: Higher than scratchpad for the same capacity due to tag array, comparators, and
  replacement logic overhead (typically 2-3x the area of an equivalent scratchpad).

**Why scratchpads dominate DNN accelerators:**

DNN workloads have highly predictable, compiler-visible access patterns. The compiler knows
exactly when each weight tile and activation tile will be needed, enabling perfectly timed DMA
prefetch. A cache would waste area on tag storage and replacement logic that provides no benefit
for these predictable patterns — and might actively hurt by evicting data that the compiler
would have kept. Scratchpads give the compiler full control at lower area cost.

---

### Q2. What is SRAM banking and why is it essential for high-bandwidth access?

**Question:** Explain why a monolithic SRAM cannot satisfy the bandwidth requirements of a large
PE array, and how banking solves this. What is a bank conflict and when does it occur?

**Answer:**

**The monolithic SRAM bandwidth problem:**

A standard 6T-SRAM macro has one read port and one write port (or two read ports at 2x area cost).
A 256x256 systolic array performing one MAC per PE per cycle requires 256 input values and 256
weight values to be delivered simultaneously — 512 reads per cycle from the weight/activation
buffers. A single SRAM macro physically cannot perform 512 reads in one cycle.

**Banking solution:**

An SRAM is divided into B independent banks, each with its own sense amplifiers and address decoder.
Each bank can service one access per cycle independently. If accesses are distributed across banks,
throughput scales linearly with the number of banks.

```
Effective bandwidth = B * (single_bank_width * frequency)
```

For example: 32 banks of 16-bit width at 1 GHz = 32 * 2 bytes * 1 GHz = 64 GB/s of read bandwidth.

**Interleaving:**

The mapping from addresses to banks is typically by low-order address bits (address modulo B).
For a linear access pattern (sequential reads), successive addresses land in successive banks,
spreading the load evenly.

**Bank conflicts:**

A bank conflict occurs when two or more simultaneous access requests map to the *same bank*.
When this happens, one access must wait — effectively reducing bandwidth.

Common causes in matrix operations:
- Column-stride accesses: Reading column j of a row-major matrix with stride = matrix_width.
  If matrix_width is a multiple of B, every column access maps to the same bank (modulo B = same
  remainder). This causes B-way conflicts on every access.
- Solution: Choose B to be a power of 2 and pad matrix rows so that the stride is not a multiple
  of B. Alternatively, use XOR-based address hashing: bank_id = (addr >> log2(word_size)) XOR
  (addr >> (log2(word_size) + log2(B))), which spreads stride accesses across banks.

**Practical rule of thumb:**

Number of banks should equal the maximum number of simultaneous accesses expected. For a systolic
array of width W where each row reads one activation per cycle, you need at least W banks in the
activation buffer. A safety margin of 2-4x is common because real traffic is bursty.

---

### Q3. What is a memory hierarchy for a typical LLM accelerator, from on-chip registers to DRAM?

**Question:** Sketch the memory hierarchy of a representative LLM accelerator (you may use
a TPU-like or GPU-like example). For each level, give approximate capacity, bandwidth, and latency,
and identify which tensor types (weights, KV-cache, activations) reside at each level during
inference.

**Answer:**

```
Level       Capacity    BW (per chip)    Latency     Contents during LLM inference
------      --------    -------------    -------     ----------------------------------
Registers   ~256 KB     ~50 TB/s est.    0 cycles    Active partial sums, current tile
                        (aggregate)                   of weights + activations in compute

L0 / PE
local buf   ~4 KB/PE    ~10 TB/s est.    1-2 cycles  Weight tile for current GEMM tile
            (1 MB tot.) (aggregate)                   (weight-stationary scratchpad)

L1 / Global
buffer      8-32 MB     ~5-20 TB/s       5-15 cycles Weight tiles queued for next layer,
(on-chip                                             KV-cache for short sequences,
SRAM)                                               activation double-buffer

L2 /HBM    16-192 GB   ~1-4 TB/s        ~100-300 ns Full model weights, long-seq KV-cache,
(High-BW                                             intermediate tensors that don't fit
Memory)                                              in L1

DRAM/NVMe  Terabytes   ~50-100 GB/s     ~1-100 us   Model checkpoints, paged KV-cache
(host)                                              overflow (speculative paging)
```

**Notes on specific tensors:**

*Weights (FFN, attention projection):*
During prefill of a layer, the weight matrix for that layer is tiled and streamed: one tile at a
time lives in the L1 global buffer while being consumed by the PE array. Only the active tile is
in the L0 PE-local buffer. The full weight matrix lives in HBM.

*KV-cache:*
For short sequences (< a few thousand tokens per request), the active-sequence KV-cache can fit
in the L1 global buffer, enabling FlashAttention-style attention without HBM round-trips for K/V.
For long-context models or large batch sizes, KV-cache spills to HBM and must be tiled.

*Activations:*
One layer's output activation (input to the next layer) must be live during the inter-layer
transition. Double-buffering in L1 ensures the DMA loads the next layer's weights while the
current layer's activations are being consumed.

---

### Q4. What is double-buffering and how does it hide memory latency?

**Question:** Explain the double-buffering technique for hiding DMA transfer latency. Draw a
timing diagram (in text) showing the difference between single-buffered and double-buffered
execution for a sequence of four tiles.

**Answer:**

**Problem without double-buffering:**

When compute finishes a tile, the accelerator must fetch the next tile from HBM before compute
can start on it. If the DMA transfer takes T_dma cycles and the tile compute takes T_comp cycles,
the total pipeline stalls for T_dma before each tile:

```
Single-buffered timeline:

Cycle:  0        T_comp  T_comp+T_dma  2*T_comp+T_dma ...
        [Tile 0 ] [DMA1 ] [Tile 1    ] [DMA2          ] [Tile 2] ...
         COMPUTE  STALL   COMPUTE       STALL
```

Utilisation = T_comp / (T_comp + T_dma)

If T_comp = T_dma, utilisation is only 50%.

**Double-buffering solution:**

Allocate two SRAM buffers (A and B) for the incoming data. While tile N is being computed from
buffer A, the DMA engine prefetches tile N+1 into buffer B. After compute finishes tile N, it
immediately starts on tile N+1 from buffer B, while DMA begins fetching tile N+2 into buffer A.
The compute and DMA pipelines run concurrently.

```
Double-buffered timeline:

         T_comp   T_comp  T_comp  T_comp
Buffer:  [Tile 0 ][Tile 1][Tile 2][Tile 3]    <- compute uses alternating buffers
DMA:      [DMA1  ][DMA2  ][DMA3  ]            <- DMA prefetches into idle buffer

Time:   0    T    2T   3T   4T
         COMPUTE  overlaps with DMA
```

Utilisation = 1.0 (as long as T_dma <= T_comp; otherwise DMA becomes the bottleneck).

**Key condition for effective double-buffering:**

```
T_dma <= T_comp
```

This means the tile size must be chosen so that compute time is at least as long as the DMA time.
If tiles are too small, T_comp < T_dma and the double-buffer only partially hides the latency.

**Practical consideration:**

Double-buffering doubles the on-chip SRAM requirement for the buffered tensor. If SRAM is scarce,
a triple-buffer (ping-pong-pang) scheme can be used to tolerate longer DMA latency at the cost of
3x the buffer area. In practice, double-buffering with conservatively large tiles almost always
suffices.

---

## Tier 2 — Intermediate

### Q5. How do you determine the required on-chip SRAM bandwidth for a given compute throughput?

**Question:** A systolic array has 1024 INT8 MAC units running at 1 GHz, delivering 2 TOPS.
Assuming a weight-stationary dataflow where weights are held in an on-chip global buffer for an
entire tile and activations are streamed in each cycle, calculate the minimum SRAM read bandwidth
required from the activation buffer and the weight buffer.

**Answer:**

**Given:**
- 1024 MAC units, 1 GHz clock, INT8 (1 byte per operand).
- Weight-stationary: weights for a tile are loaded once at tile start; activations stream every cycle.

**Activation buffer bandwidth:**

In weight-stationary, each MAC unit needs one new input activation per cycle (the corresponding
weight is already in the PE register). The array is 1024 PEs wide (assume 32 rows x 32 columns
for a square array, though the calculation is the same).

For each column of PEs (32 PEs per column), all PEs in the same column receive the *same*
activation value (broadcast). There are 32 columns, so 32 distinct activation values per cycle.

Actually, rethinking for a standard WS systolic: inputs flow across rows. Each row of 32 PEs
receives the same input each cycle (broadcast within a row). There are 32 rows, so 32 independent
activation reads per cycle.

```
Activation BW = 32 reads/cycle * 1 byte/read * 1 GHz = 32 GB/s
```

**Weight buffer bandwidth:**

Weights are loaded into PE registers at tile start. The load happens over a preload phase of
length equal to tile depth (K dimension). For tile depth K:
```
Weight preload BW = 1024 weights/cycle * 1 byte * 1 GHz (during preload only)
                  = 1024 GB/s peak (during the K-cycle preload phase)
```

However, averaged over the tile (K compute cycles + K preload cycles), if preloaded in parallel:
```
Average weight BW = 1024 * 1 byte * K cycles / (K cycles) = 1024 GB/s
```

This is impractical for a single SRAM — the weight buffer must be banked to 1024 parallel banks
or the preload must be spread over multiple cycles before the compute begins. In practice,
weight preload is pipelined across the diagonal of the array over K cycles, not all at once.

**More realistic weight loading:**

In a systolic diagonal-load scheme, each of the 1024 PEs gets one weight loaded per cycle during
the K-cycle initialisation phase. That is exactly 1024 GB/s peak from the weight buffer during
preload — met by 1024 single-byte-wide SRAM banks. Post-load, weight BW = 0 (weights are in PE
registers until tile is done).

**Key takeaway:**

The weight buffer must be banked enough to deliver 1024 bytes/cycle during tile loading. The
activation buffer needs only 32 bytes/cycle (for a 32-row array) sustained during compute.
Getting the banking ratio wrong (e.g., under-banking the weight buffer) is a common design mistake
that creates a bandwidth wall during tile preload and reduces effective throughput.

---

### Q6. Explain the concept of "memory-compute balance" and how to achieve it in a design.

**Question:** Define the memory-compute balance point for an LLM accelerator. Given a hypothetical
chip with 4 TOPS of INT8 compute and 2 TB/s of on-chip SRAM bandwidth, determine whether the
design is compute-bound or memory-bound for (a) a prefill GEMM and (b) a decode GEMV. What
design changes would rebalance each case?

**Answer:**

**Memory-compute balance point:**

The balance point is the arithmetic intensity (AI, in FLOPs/byte) at which compute throughput and
memory bandwidth are simultaneously at their peak:

```
AI_balance = Peak_FLOPS / Peak_BW = 4e12 FLOPS/s / 2e12 bytes/s = 2 FLOPs/byte
```

**Case (a): Prefill GEMM — A(T x d) * W(d x d)**

For T=2048 (tokens), d=4096:
```
FLOPs = 2 * 2048 * 4096 * 4096 = 68.7 GFLOPs
Bytes = (2048*4096 + 4096*4096 + 2048*4096) * 1 (INT8) = ~58 MB
AI = 68.7e9 / 58e6 = 1185 FLOPs/byte
```

AI = 1185 >> AI_balance = 2, so **strongly compute-bound**.
The SRAM bandwidth is massively over-provisioned relative to compute for this workload.

**Rebalancing for prefill:**
- Increase compute: Add more PE rows/columns to raise TOPS.
- Reduce SRAM bandwidth: Save area/power by reducing bank count — you have huge margin.
- Or: Use the excess bandwidth to implement double-buffering and hide the gap between layers.

**Case (b): Decode GEMV — v(1 x d) * W(d x d)**

```
FLOPs = 2 * 1 * 4096 * 4096 = 33.6 MFLOPs
Bytes = (1*4096 + 4096*4096 + 1*4096) * 1 (INT8) = ~16.8 MB
AI = 33.6e6 / 16.8e6 = 2 FLOPs/byte
```

AI = 2 = AI_balance exactly. The design is **at the balance point** for decode GEMV.

But this is on-chip SRAM bandwidth. If W must be loaded from HBM (16.8 MB per token, HBM at
1 TB/s), then:
```
AI_vs_HBM = 33.6 MFLOPs / 16.8 MB = 2 FLOPs/byte (same shape)
HBM_balance_point = 4e12 / 1e12 = 4 FLOPs/byte
```
Now AI < HBM balance point: the design is **HBM-bandwidth-bound** for decode.

**Rebalancing for decode:**
- Quantise weights to INT4: halves bytes loaded, doubling effective AI to 4 F/B.
- Increase HBM bandwidth: Add more HBM stacks.
- Batch decode requests: Increase effective M from 1 to B, raising AI proportionally.
- Cache heavily-reused weights on-chip: For models with tied weights (embedding = unembedding),
  keep those weights in on-chip SRAM to avoid HBM reads.

---

### Q7. How do you design a multi-bank SRAM for conflict-free matrix column access?

**Question:** A matrix of shape (R x C) is stored row-major in an SRAM. You need to read an
entire column simultaneously (C elements with stride R). Describe an SRAM banking scheme that
avoids bank conflicts for both row and column accesses.

**Answer:**

**Problem setup:**

Row-major layout: element [i][j] is at address i*C + j.

For a column j access: addresses are j, j+C, j+2C, ..., j+(R-1)*C.

With B banks using address mod B mapping:
- Bank of element [i][j] = (i*C + j) mod B
- For column j, successive elements have bank = (i*C + j) mod B for i = 0, 1, 2, ...
- If B divides C, then (i*C + j) mod B = j mod B for all i — *all elements of column j land in
  the same bank*. This is a maximum bank conflict.

**Solution 1: Non-power-of-two B or B that does not divide C**

Choose B such that gcd(B, C) = 1 (B and C are coprime). Then bank indices for a column cycle
through all B banks with period B. This gives zero conflicts for column accesses.

Example: C=16, B=5 (gcd(16,5)=1). Column bank assignments: 0, 0+16 mod 5=1, 1+16 mod 5=2, etc.
Each of the 5 banks is used in rotation. No conflicts.

Downside: non-power-of-two bank counts complicate address decoding.

**Solution 2: XOR-based address scrambling (preferred for power-of-two B)**

Map element [i][j] to bank = i XOR j (using log2(B) bits from each index).

For a row access (fixed i, varying j): bank = i XOR j — all j values give distinct banks (as
long as j varies over 0..B-1). No conflicts.

For a column access (fixed j, varying i): bank = i XOR j — all i values give distinct banks.
No conflicts.

For a diagonal access (i == j): bank = i XOR i = 0 for all — conflict! Diagonals are still a
problem for skew XOR. A more complete solution uses a Latin square addressing scheme.

**Solution 3: Skewed storage (interleaved rows)**

Store element [i][j] at logical address i*C + ((j + i) mod C), i.e., rotate each row by i
positions. Column j of the logical matrix is now stored at addresses i*C + (j+i) mod C — which
are spread across all banks in sequence.

This is hardware-transparent: the address generation logic applies the skew, the SRAM itself is
standard power-of-two banks. Both row and column accesses are conflict-free.

**Practical recommendation:**

For LLM accelerators where both row and column accesses are common (weight matrix transposes,
KV-cache row and column reads), use XOR-based banking with B = power of 2, which eliminates
conflicts for the two most common access patterns (row-major and column-major) at the cost of
slightly more complex address generation hardware.

---

## Tier 3 — Advanced

### Q8. Design the on-chip memory hierarchy for a 100 TOPS LLM inference chip targeting Llama-70B.

**Question:** Llama-70B has 80 transformer layers, d_model=8192, FFN hidden dimension=28672,
GQA with 8 KV heads, d_head=128. Design the on-chip memory hierarchy (SRAM levels, capacities,
bandwidths, banking structure) for a chip targeting:
- Prefill throughput: 10K tokens/second
- Decode latency: < 50ms per token (single request)
- INT8 weights, FP16 activations

Show your sizing calculations.

**Answer:**

**Step 1: Establish compute and memory requirements per layer**

Per transformer layer, the dominant operations are:
- QKV projection: (T x 8192) * (8192 x 3*1024) -- GQA: 8192->8192 for Q, 8192->2*1024 for K,V
  Weight bytes: (8192 * 8192 + 8192 * 2*1024) = 84 MB per layer (INT8) ... wait:
  Q projection: 8192 * 8192 = 67M params = 67 MB INT8
  KV projection: 8192 * 2 * 1024 = 16.8M = 16.8 MB (GQA: only 8 heads)
  O projection: 8192 * 8192 = 67 MB
  FFN gate+up: 2 * 8192 * 28672 = 469 MB
  FFN down: 28672 * 8192 = 235 MB
  Total per layer: ~855 MB INT8
  Total 80 layers: ~68 GB -- does NOT fit on-chip. Must stream from HBM.

**Step 2: HBM requirement**

At 100 TOPS (INT8) and AI_balance:
Need minimum HBM bandwidth to sustain decode (M=1):
```
Decode GEMV AI = 2 FLOPs/byte (as computed earlier for square matrix; slightly different for
rectangular but order-of-magnitude the same)
Required HBM BW = 100e12 / 2 = 50 TB/s  -- clearly impossible
```

Reality check: for M=1 decode at 100 TOPS compute, we are compute-overkill. The bottleneck is
HBM. With 4x HBM3 stacks at 3.2 TB/s each = 12.8 TB/s:
```
Sustainable decode throughput = 12.8e12 bytes/s / 855e6 bytes/layer / 80 layers = 0.19 tokens/s
```

To hit < 50ms/token (= 20 tokens/s) with 12.8 TB/s, we need continuous batching with batch size:
```
Minimum batch = 20 / 0.19 = ~105 concurrent requests
```

This drives the on-chip memory hierarchy design toward maximising KV-cache capacity.

**Step 3: On-chip SRAM sizing**

Target: hold one layer's weights on-chip for reuse across a batch.

One layer weights = 855 MB. At 28nm-class SRAM density ~0.15 MB/mm^2, 855 MB = 5700 mm^2 --
impractical. We cannot hold even one full layer on chip.

Practical on-chip SRAM: 128 MB (reasonable for a large chip ~800 mm^2 die, ~60 MB usable for data):
```
- 32 MB: Activation double-buffer (16 MB per bank, enough for 2K tokens x 8192 x FP16 = 32 MB total)
         16 MB per buffer = 1K tokens x 8192 x 2 bytes = 16 MB. OK.
- 16 MB: Weight tile buffer (holds one FFN tile: e.g., 8192 x 1024 x 1 byte = 8 MB, double-buffered)
-  8 MB: KV-cache fast scratchpad (holds K,V for ~256 tokens per active request for FA tiling)
          256 tokens * 2 * 8192 bytes * 2 bytes (FP16) = 8 MB (one layer's K,V for one request)
-  4 MB: Partial sum accumulator (output FP16, 8192 x 256 = 4 MB)
-  4 MB: Misc (instruction buffer, control, NOC buffers)
```

**Step 4: Bandwidth requirements and banking**

PE array: 100 TOPS INT8 = 100e12 INT8 MACs/s.

For weight-stationary with 128-column tiling and 1 GHz:
- Weight preload BW: 100e12 MACs / (2 MACs/cycle for INT8 paired with accumulate) ... 

Actually:
- Active MACs per cycle = 100e12 / 1e9 = 100K MACs/cycle
- Each MAC needs 1 weight + 1 activation per cycle (weight preloaded, activation streamed)
- Activation BW = number of rows in PE array * 2 bytes * 1 GHz
  For 256-row array: 256 * 2 * 1e9 = 512 GB/s from activation buffer
- Weight preload BW: 100K bytes/cycle = 100 TB/s peak during tile preload
  -- must spread over tile depth K=1024: 100K/1024 = ~100 GB/s average weight BW during preload

**Weight buffer banking:**
- 100 GB/s at 2-byte words at 1 GHz = 50 words/cycle = 50 banks minimum.
  Use 64 banks of 128 bits (16 bytes) each = 1024 bytes/cycle = 1 TB/s. More than enough.

**Activation buffer banking:**
- 512 GB/s at 2 bytes/word at 1 GHz = 256 reads/cycle = 256 banks of 2 bytes each.
  Practical: 64 banks of 16 bytes (128 bits) each = 1024 bytes/cycle = 1 TB/s. Sufficient.

**Step 5: Summary table**

```
Level           Capacity  Banks  BW          Primary contents
-----------     --------  -----  ----------  ----------------------
PE registers    ~256 KB   N/A    ~50 TB/s    Active partial sums, weight registers
Weight buffer   16 MB     64     1 TB/s      Current + next weight tile (DBI)
Activation buf  32 MB     64     1 TB/s      Current + prefetched activation tiles
KV scratchpad   8 MB      32     512 GB/s    FA-tiled K,V for active sequence
Partial sum     4 MB      64     1 TB/s      Output accumulation before writeback
HBM (4 stacks)  96 GB     N/A    12.8 TB/s   Full model weights, KV-cache overflow
```

**Key design decisions justified:**
1. No weight caching on chip: model is too large; weights always stream from HBM.
2. Double-buffering of both weights and activations: hides HBM latency.
3. KV-cache scratchpad enables FlashAttention, eliminating the N^2 attention HBM traffic.
4. 64-bank SRAM design throughout: avoids bank conflicts for both row and column access patterns.
