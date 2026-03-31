# DMA and Tiling Strategies for LLM Accelerators

## Overview

Tiling and DMA management are the bridge between the theoretical efficiency of a chosen dataflow
and its realisation in silicon. Even a perfectly designed PE array will stall if its DMA engine
cannot keep data flowing. This section covers the mathematics of tile size selection, loop ordering
for different dataflows, DMA prefetch scheduling, and the techniques used to overlap data movement
with computation to approach 100% PE utilisation.

---

## Tier 1 — Fundamentals

### Q1. What is tiling and why is it necessary for LLM matrix operations?

**Question:** Define tiling (also called blocking) in the context of matrix multiplication on a
hardware accelerator. Why can't we simply execute the full matrix multiply in one shot?

**Answer:**

**Definition:**

Tiling is the decomposition of a large matrix operation into a sequence of smaller sub-operations
(tiles) that fit within the on-chip memory of the accelerator. Each tile is a contiguous submatrix
(or sub-tensor) that can be loaded into on-chip SRAM, processed by the PE array, and written back
to off-chip memory before the next tile is loaded.

**Why tiling is necessary:**

For a fully-connected layer in Llama-70B: weight matrix W is (8192 x 28672), stored in INT8.
```
Size = 8192 * 28672 * 1 byte = 234 MB
```

A typical on-chip SRAM is 8-32 MB. The full weight matrix is 7-30x larger than available SRAM.

Without tiling, two bad alternatives exist:
1. Build an on-chip SRAM large enough to hold all operands — 234 MB of SRAM is ~1600 mm^2
   at 28nm density, consuming most of a large chip's area budget just for one layer.
2. Access operands directly from HBM for every MAC — HBM latency of ~100 ns would mean
   every multiply-accumulate waits 100 cycles, reducing PE utilisation to < 1%.

**Tiling solution:**

Break W into tiles of size (Tr x Tc) that fit in on-chip SRAM alongside the corresponding input
and output tiles. Process each tile completely before moving to the next. The key insight is that
within a tile, all data accesses are to fast on-chip SRAM, achieving high effective bandwidth.
Data movement between HBM and SRAM happens asynchronously via DMA while the previous tile is
being computed (double-buffering).

**Tiling for matrix multiply C = A * B:**

Partition the (M x K x N) GEMM into tiles (Tm x Tk x Tn):
- Load A tile: (Tm x Tk) from HBM
- Load B tile: (Tk x Tn) from HBM  
- Accumulate partial result into C tile: (Tm x Tn) in on-chip SRAM
- Repeat over all K tiles, then write C tile to HBM
- Move to next (m, n) tile pair

---

### Q2. How do you calculate the maximum tile size given an on-chip memory budget?

**Question:** Given a GEMM C = A * B where A is (M x K), B is (K x N), and C is (M x N), derive
the constraint on tile sizes (Tm, Tk, Tn) given a total on-chip SRAM capacity of S bytes. Assume
FP16 for activations and INT8 for weights, FP16 for partial sums. State which tile dimension to
maximise first and why.

**Answer:**

**On-chip memory usage per tile computation:**

- A tile (input activations, FP16): Tm * Tk * 2 bytes
- B tile (weights, INT8): Tk * Tn * 1 byte
- C tile (output/partial sum, FP16): Tm * Tn * 2 bytes

For double-buffering, we need two copies of A and B (one active, one being DMA'd):
```
Total SRAM = 2*(Tm*Tk*2) + 2*(Tk*Tn*1) + Tm*Tn*2 <= S
           = 4*Tm*Tk + 2*Tk*Tn + 2*Tm*Tn <= S
```

The output tile (C) is not double-buffered in the same way because it accumulates in place.

**Which dimension to maximise first:**

Maximise Tn (output columns) first, then Tm (output rows). The reason is reuse efficiency:

- **Tm * Tn** determines output tile size and also the number of partial sums kept on-chip.
  Large output tiles amortise the weight reads over more output elements.
- **Tk** determines how many steps of the inner product are done before the output tile is
  complete. Larger Tk means fewer passes over the output tile, reducing write-back frequency.

**Practical priority order:**
1. Set Tn = min(N, max_Tn) to maximise weight reuse across output columns.
2. Set Tm to fill remaining SRAM after allocating for B and C tiles.
3. Set Tk to fill the remainder, subject to the double-buffer constraint.

**Example calculation:**

S = 16 MB = 16,777,216 bytes. Target square tiles: Tm = Tn = T, and Tk = K_tile.

```
4*T*K_tile + 2*K_tile*T + 2*T^2 <= 16,777,216
(6*T*K_tile) + 2*T^2 <= 16,777,216
```

If T = 1024 and K_tile = 1024:
```
6*1024*1024 + 2*1024^2 = 6,291,456 + 2,097,152 = 8,388,608 = 8 MB <= 16 MB -- fits
```

With double-buffering of the C tile as well (for output streaming):
```
4*T*K_tile + 2*K_tile*T + 2*2*T^2 = 8 MB + 4 MB = 12 MB <= 16 MB -- still fits
```

So tiles of (1024 x 1024) with K_tile=1024 work for a 16 MB on-chip SRAM with double-buffering.

---

### Q3. What is a DMA engine and what are its key configuration parameters?

**Question:** Describe the role of a DMA engine in an LLM accelerator. What are the key
configuration parameters that determine DMA transfer performance, and what constraints does the
AXI4 protocol impose on burst transfers?

**Answer:**

**Role of the DMA engine:**

A DMA (Direct Memory Access) engine transfers data between off-chip memory (HBM/DRAM) and
on-chip SRAM without involving the main processor or PE array. The DMA engine operates in parallel
with the PE array, enabling the compute and memory subsystems to operate concurrently. Without DMA,
the PE array would stall waiting for data loads and stores.

In an LLM accelerator, the DMA engine performs:
1. Weight tile fetch: Load next weight tile from HBM into the inactive SRAM buffer.
2. Activation tile fetch: Load input activation tile for the current or next operation.
3. Output writeback: Write completed output activation tiles back to HBM.
4. KV-cache management: Load/store K and V tensors for attention computation.

**Key DMA configuration parameters:**

| Parameter | Description | Typical range |
|-----------|-------------|---------------|
| Source address | Base HBM address of the transfer | Full 64-bit address space |
| Destination address | On-chip SRAM address | 20-32 bit local address |
| Transfer length | Total bytes to transfer | 64 B to 256 MB |
| Burst length | AXI beats per burst transaction | 1-256 beats |
| Stride / 2D DMA | Row stride for non-contiguous data | Enables matrix column extraction |
| Outstanding count | Number of in-flight AXI transactions | 4-32 |

**2D DMA (strided transfer):**

Matrix data is often stored row-major but needs to be read column-by-column, or a submatrix of
a larger matrix needs to be transferred. A 2D DMA engine supports:
```
for row in range(num_rows):
    transfer(src_base + row * src_stride, dst_base + row * dst_row_size, row_bytes)
```

This is equivalent to copying a rectangle from a larger 2D array, which is the most common
tiling operation. Hardware support for 2D DMA eliminates the need for a separate gather/scatter
pass before computation.

**AXI4 burst constraints:**

- Maximum burst length: 256 beats (AXI4), 16 beats (AXI3)
- All addresses in a burst must not cross a 4 KB address boundary (AXI4 rule)
- Burst type INCR (incrementing address) is most common; FIXED and WRAP bursts are less used
- For a 64-byte cache line with 32-byte AXI data bus (256-bit), one beat = 32 bytes, so a
  256-beat burst = 8 KB maximum per transaction
- Throughput = (burst_length * bus_width * frequency) / (burst_length + overhead_cycles)
  As burst_length increases, efficiency approaches 100%; short bursts have high per-transaction
  overhead from address and handshake cycles

---

### Q4. What is loop ordering in the context of tiled matrix multiplication?

**Question:** For the GEMM C = A * B, write out the six loop orders for iterating over tiles
(m-tiles, k-tiles, n-tiles). Identify which loop order minimises off-chip data movement for
each of the three stationary dataflows (WS, OS, IS).

**Answer:**

**The six loop orders over tiles:**

For a tiled GEMM with M/Tm m-tiles, K/Tk k-tiles, and N/Tn n-tiles:

```
Order 1 (mnk): for m: for n: for k: -- output-stationary
Order 2 (mkn): for m: for k: for n: -- row-stationary over A
Order 3 (nmk): for n: for m: for k: -- output-stationary (n-major)
Order 4 (nkm): for n: for k: for m: -- column-stationary over B
Order 5 (kmn): for k: for m: for n: -- weight-stationary
Order 6 (knm): for k: for n: for m: -- weight-stationary (n-major)
```

**Matching loop order to dataflow:**

**Weight-Stationary (WS) -> Loop orders 5 or 6 (k outermost):**

```
for k_tile in K/Tk:                   -- outermost: load B tile once
    load B[k_tile, :] from HBM        -- B tile stays in SRAM
    for m_tile in M/Tm:               -- sweep all output rows
        load A[m_tile, k_tile] from HBM
        compute C[m_tile, :] += A[m_tile, k_tile] * B[k_tile, :]
```

B (weights) are held in SRAM across the entire m_tile loop. Each B tile is loaded once for all
M/Tm input tiles. Weight re-reads from HBM = N_k (one read per k-tile). Optimal when B is large
and there are many input tile rows (large M).

**Output-Stationary (OS) -> Loop order 1 (m, n outermost, k innermost):**

```
for m_tile in M/Tm:                   -- outermost: one output tile row
    for n_tile in N/Tn:               -- one output tile column
        C_tile = 0                    -- output tile stays in SRAM
        for k_tile in K/Tk:           -- inner: accumulate full partial sum
            load A[m_tile, k_tile], B[k_tile, n_tile]
            C_tile += A_tile * B_tile -- C never written to HBM until k loop done
        write C_tile to HBM           -- one write per output tile
```

C (output partial sums) accumulate entirely in SRAM across the k-loop. No partial C round-trips
to HBM. A and B are streamed; they are each read once per output tile. Optimal when output tiles
are large and SRAM is limited (minimises partial sum traffic).

**Input-Stationary (IS) -> Loop order 2 (m, k outer, n inner):**

```
for m_tile in M/Tm:
    for k_tile in K/Tk:               -- inner: load A tile once
        load A[m_tile, k_tile] from HBM  -- A tile held stationary
        for n_tile in N/Tn:           -- sweep all weight columns
            load B[k_tile, n_tile]
            C[m_tile, n_tile] += A_tile * B_tile
```

A (input activations) are held across the n_tile loop. Each A tile is read once per k-tile,
reused for all N/Tn weight column tiles. Optimal when A is small (single sequence, large batch
of same input) and N is very large.

**Choosing in practice:**

For LLM prefill (large M, large N, moderate batch): output-stationary (loop order 1) minimises
the expensive partial-sum round-trips and matches FlashAttention's tiling philosophy.

For LLM inference serving with many concurrent requests (effectively large batch = large M):
weight-stationary (loop order 5) amortises weight loads across all batch entries.

---

## Tier 2 — Intermediate

### Q5. How do you schedule DMA prefetch to hide memory latency completely?

**Question:** Given: compute time per tile = Tc cycles, DMA transfer time per tile = Td cycles,
and a double-buffer setup. Derive the conditions under which DMA fully hides memory latency.
Then describe the scheduling algorithm (state machine) that a DMA controller uses to maintain
this overlap.

**Answer:**

**Latency hiding condition:**

With double-buffering (buffers A and B alternating):

```
Timeline:
Cycle 0:     DMA fetches tile 0 into buffer A   (takes Td cycles)
Cycle 0:     -- compute is idle waiting for first tile --
Cycle Td:    Compute starts on buffer A (tile 0)
Cycle Td:    DMA starts fetching tile 1 into buffer B
Cycle Td+Tc: Compute finishes tile 0, switches to buffer B
Cycle Td+Td: DMA finishes tile 1 (if Td <= Tc, this is before Td+Tc)
...
```

For cycle i (i >= 1):
- Compute accesses buffer (i mod 2) starting at Td + (i-1)*Tc
- DMA must finish fetching tile i into buffer ((i+1) mod 2) by Td + (i-1)*Tc

The DMA for tile i starts at Td + (i-1)*Tc - Td = (i-1)*Tc (the moment compute starts tile i-1).
DMA completes at (i-1)*Tc + Td.
Compute starts tile i at Td + (i-1)*Tc.

Condition: DMA must complete before compute needs the buffer:
```
(i-1)*Tc + Td <= Td + (i-1)*Tc  =>  True always if Td <= Tc
```

**Full latency hiding condition: Td <= Tc**

If Td > Tc (DMA is slower than compute), compute finishes the tile before the next tile is ready.
In this case, the stall per tile = Td - Tc cycles. To recover, either increase tile size (which
increases both Tc and Td proportionally, but Tc grows faster for compute-bound workloads) or
issue multiple DMA requests in flight simultaneously (depth-2 or depth-3 prefetch pipeline).

**DMA scheduling state machine:**

```
States: IDLE, FILLING_A, FILLING_B, DONE
Signals: dma_start, dma_done, compute_start, compute_done, buf_ready_A, buf_ready_B

IDLE -> FILLING_A: on start command; issue DMA for tile 0 into buffer A
FILLING_A: waiting for dma_done
  on dma_done: set buf_ready_A, signal compute_start(A)
               issue DMA for tile 1 into buffer B
               -> FILLING_B
FILLING_B: waiting for dma_done and compute_done(A)
  on both done: swap buffers
                if more tiles: issue DMA for tile N+1, signal compute_start(B)
                               -> FILLING_A
                else: -> DONE
```

**Handling Td > Tc (DMA-bound case):**

Extend the state machine with a STALL state:
```
  on compute_done but not dma_done: -> STALL
STALL:
  on dma_done: -> normal flow (issue next DMA, start compute)
```

Record stall cycles in a performance counter register for profiling and compiler feedback.

---

### Q6. How do you tile attention for long sequences given limited on-chip memory?

**Question:** For a multi-head attention layer with sequence length N=32768, d_head=128, 32 heads,
FP16, and on-chip SRAM of 32 MB: show that the naive attention matrix does not fit on chip, then
derive the FlashAttention tile sizes that do fit. Specify the exact DMA pattern for K and V tiles.

**Answer:**

**Naive attention memory requirement:**

For one head:
- Full Q matrix: N * d_head * 2 bytes = 32768 * 128 * 2 = 8 MB
- Full K matrix: 8 MB
- Full V matrix: 8 MB
- Attention score matrix S = Q * K^T: N * N * 2 = 32768^2 * 2 = 2 GB

The score matrix alone is 2 GB — far too large. Even for a single head, storing S on-chip is
impossible with 32 MB SRAM.

**FlashAttention tiling derivation:**

FlashAttention tiles Q into blocks of Br rows, K and V into blocks of Bc rows.
Memory requirement for one tile of the inner loop:
```
Q block:  Br * d_head * 2 = Br * 128 * 2 bytes
K block:  Bc * d_head * 2 = Bc * 128 * 2 bytes
V block:  Bc * d_head * 2 = Bc * 128 * 2 bytes
S block:  Br * Bc * 2 bytes  (partial score matrix, ephemeral)
O block:  Br * d_head * 2 bytes  (output accumulator, stationary for inner loop)
Stats:    Br * 2 * 2 bytes  (m and l running statistics, 2 scalars per row)
```

Total SRAM needed:
```
= Br*256 + Bc*256 + Bc*256 + Br*Bc*2 + Br*256 + Br*4
= Br*(256 + 256 + 4) + Bc*(256 + 256) + Br*Bc*2
= 516*Br + 512*Bc + 2*Br*Bc <= 32 MB / (per_head_allocation)
```

With 32 heads sharing the 32 MB, budget per head = 1 MB = 1,048,576 bytes.

Setting Br = Bc = B (square tiles):
```
516*B + 512*B + 2*B^2 <= 1,048,576
1028*B + 2*B^2 <= 1,048,576
B^2 + 514*B - 524,288 <= 0
B <= (-514 + sqrt(514^2 + 4*524288)) / 2 = (-514 + sqrt(2,360,548)) / 2
   = (-514 + 1536) / 2 = 511
```

Round down to a power of 2: **Br = Bc = 256** rows.

Verification: 516*256 + 512*256 + 2*256^2 = 132,096 + 131,072 + 131,072 = 394,240 bytes ≈ 385 KB

With double-buffering (while computing on one K/V tile, DMA loads the next):
```
Total SRAM per head = 2 * (Bc_K + Bc_V) * 128 * 2 + Br * (128 + 128) * 2 + Br * Bc * 2
                    = 2 * 2 * 256 * 256 + 256 * 512 + 256 * 256
                    = 262,144 + 131,072 + 65,536 = 458,752 bytes ≈ 448 KB
```

Still fits in 1 MB per head. Double-buffering is viable.

**DMA pattern for K and V tiles:**

```
Outer loop (over Q tiles, i = 0 to ceil(N/Br)-1 = 127 tiles):
  Load Q[i*Br : (i+1)*Br, :] into Q_buf via DMA (8192 bytes per head per tile)

  Inner loop (over K/V tiles, j = 0 to ceil(N/Bc)-1 = 127 tiles):
    -- DMA pattern: preload KV tile j+1 while computing on tile j --

    DMA issue: load K[j*Bc : (j+1)*Bc, :] into K_buf[active ^ 1]  -- ping-pong
    DMA issue: load V[j*Bc : (j+1)*Bc, :] into V_buf[active ^ 1]  -- ping-pong
    Compute: partial attention update using K_buf[active], V_buf[active]
    Wait for DMA completion
    active ^= 1
  
  Write O[i*Br : (i+1)*Br, :] back to HBM via DMA
```

Number of DMA transactions (per head per attention layer):
- Q loads: 128 tiles * 8 KB = 1 MB
- K loads: 128 tiles * 8 KB = 1 MB (read once per Q outer-loop pass -- but in FA, K is re-read
  for every Q tile, so total K reads = 128 Q tiles * 128 K tiles = 16,384 tile reads... but each
  K tile (256 * 128 * 2 bytes = 64 KB) is streamed, so total K bytes = N * d * 2 = 8 MB, read
  128 times across all Q tiles = 1 GB per head). Note: this is the expected N^2 d complexity.
- O writes: 1 MB per head.

FlashAttention does not reduce total FLOPs, only HBM round-trips for the score matrix.

---

## Tier 3 — Advanced

### Q7. Design a hierarchical tiling strategy for a 3-level memory hierarchy.

**Question:** An accelerator has three memory levels: L0 (PE register file, 512 bytes/PE),
L1 (cluster SRAM scratchpad, 512 KB per cluster of 64 PEs), and L2 (global on-chip SRAM, 32 MB).
Design a hierarchical tiling strategy for a (4096 x 4096) x (4096 x 4096) GEMM in FP16.
Specify the tile sizes at each level, the loop ordering, and the DMA scheduling between levels.

**Answer:**

**Memory hierarchy summary:**

| Level | Capacity | BW to compute | Managed by |
|-------|----------|---------------|------------|
| L0 register | 512 B / PE | ~50 TB/s (aggregate) | Instruction stream |
| L1 SRAM | 512 KB / cluster | ~5 TB/s (aggregate) | Cluster DMA / explicit load |
| L2 SRAM | 32 MB | ~2 TB/s | Global DMA controller |
| HBM | 96 GB | 3.2 TB/s | HBM DMA controller |

**Tile size derivation:**

**L0 tile (innermost, in PE registers):**

A 64-PE cluster arranged as 8x8 PEs. Each PE holds one weight register and accumulates one
partial sum register. The L0 tile is the sub-problem each PE works on per cycle.

In output-stationary, each PE accumulates one element of C. For a dot product of length K_l0:
- PE register file: 1 weight (2 B) + 1 activation (2 B) + 1 partial sum (2 B) = 6 bytes/PE
- K_l0 = min(K, register_file_rows - 3) = use up to 16 weights per PE = 32 bytes/PE weight storage

L0 tile: **(8 x 8) output elements, K_l0 = 16 inner product steps per clock group**
(8 PE rows x 8 PE columns, each PE doing 16 MACs before its partial sum is flushed to L1)

**L1 tile (cluster scratchpad, 512 KB):**

Holds the sub-problem for one cluster's 64 PEs across many L0 tile iterations.

For output-stationary at L1:
- C tile (output, stays in L1): Tm_l1 * Tn_l1 * 2 bytes
- A tile (activation, streamed from L2): Tm_l1 * Tk_l1 * 2 bytes
- B tile (weight, streamed from L2): Tk_l1 * Tn_l1 * 2 bytes
- Double-buffer A and B: 2x each

```
2*Tm*Tk*2 + 2*Tk*Tn*2 + Tm*Tn*2 <= 512 KB = 524,288 bytes
4*Tm*Tk + 4*Tk*Tn + 2*Tm*Tn <= 524,288
```

Choose Tm_l1 = 128, Tn_l1 = 128, Tk_l1 = 256:
```
4*128*256 + 4*256*128 + 2*128*128
= 131,072 + 131,072 + 32,768 = 294,912 bytes = 288 KB <= 512 KB -- fits
```

L1 tile: **(128 rows x 128 cols) output, Tk_l1 = 256 inner product depth**

**L2 tile (global on-chip SRAM, 32 MB):**

Holds the sub-problem across multiple clusters. 32 MB / 512 KB = 64 clusters can be active.

For the global tiling:
- B tile (weights, stationary at L2 across M sweep): Tk_l2 * N * 2 bytes -- too large for full N
- Use Tn_l2 = 1024: B tile = Tk_l2 * 1024 * 2 bytes

Choose Tm_l2 = 1024, Tn_l2 = 1024, Tk_l2 = 1024:
```
A tile: 1024*1024*2 = 2 MB, double-buffered = 4 MB
B tile: 1024*1024*2 = 2 MB, double-buffered = 4 MB  
C tile: 1024*1024*2 = 2 MB (no double-buffer needed -- accumulates in place)
Total: 10 MB <= 32 MB -- fits with room for other allocations
```

L2 tile: **(1024 x 1024) output, Tk_l2 = 1024 inner product depth**

**3-level tiled loop structure:**

```
// L2-level loop (controlled by global DMA, software-scheduled)
for m2 in range(0, M, Tm_l2):              // 4 iterations (4096/1024)
  for n2 in range(0, N, Tn_l2):            // 4 iterations
    C_l2[m2, n2] = 0                        // C tile in L2, zero-init
    for k2 in range(0, K, Tk_l2):          // 4 iterations, output-stationary
      DMA(HBM -> L2): A[m2:m2+Tm_l2, k2:k2+Tk_l2]  // 2 MB
      DMA(HBM -> L2): B[k2:k2+Tk_l2, n2:n2+Tn_l2]  // 2 MB (weight-stationary at L2)
      
      // L1-level loop (controlled by cluster DMA)
      for m1 in range(0, Tm_l2, Tm_l1):    // 8 iterations per cluster
        for n1 in range(0, Tn_l2, Tn_l1):  // 8 iterations per cluster
          for k1 in range(0, Tk_l2, Tk_l1): // 4 iterations
            DMA(L2 -> L1): A[m2+m1, k2+k1, Tm_l1, Tk_l1]  // 64 KB
            DMA(L2 -> L1): B[k2+k1, n2+n1, Tk_l1, Tn_l1]  // 64 KB
            
            // L0-level loop (PE array, instruction-stream scheduled)
            for m0 in range(0, Tm_l1, 8):  // 16 PE row groups
              for n0 in range(0, Tn_l1, 8): // 16 PE col groups
                for k0 in range(0, Tk_l1, K_l0): // 16 L0 steps
                  // 64 PEs each do 16 MACs: 1024 MACs per step
                  PE_array_compute(A_l1[m0, k0], B_l1[k0, n0], C_l1[m0, n0])
    DMA(L2 -> HBM): C_l2[m2, n2]           // 2 MB writeback
```

**DMA scheduling at each level:**

L2 <-> HBM:
- Issue DMA for A[k2+1] and B[k2+1] as soon as the L1 loop for k2 starts.
- L1 loop takes: 8*8*4 * (L1 compute time) > HBM DMA time for 4 MB if tiles are large enough.
- Achieved via double-buffering: while cluster array processes (m2, n2, k2), HBM DMA loads
  the next k2+1 slice.

L1 <-> L2:
- Issue L1 DMA for (m1, n1, k1+1) while cluster computes (m1, n1, k1).
- L1 DMA transfers 128 KB; cluster compute per L1 tile = 128*128*256*2 = 8 GFLOPs / 100 TOPS
  = 80 us >> L1 DMA time (128 KB / 5 TB/s = 25 ns). L1 DMA is trivially hidden.

**Total DMA traffic:**

HBM reads per complete GEMM:
- A reads: M*K*2 bytes * (N/Tn_l2) = 4096^2 * 2 * 4 = 512 MB (re-read for each n2 tile)
- B reads: K*N*2 bytes * (M/Tm_l2) = 4096^2 * 2 * 4 = 512 MB (re-read for each m2 tile)
- C writes: M*N*2 = 32 MB
Total: ~1056 MB for the full 4096^2 GEMM in FP16.

Naive (no reuse): same order since each element must be loaded at least once.
The tiling ensures we do not exceed 1x re-read by keeping the intermediate C in L2.

### Q8. How should tile size be adapted dynamically for prefill vs. decode phase?

**Question:** An LLM accelerator uses a fixed tile size compiled at design time. Explain why
using the same tile size for prefill (large M) and decode (M=1) is suboptimal, and describe
a hardware mechanism that allows the tile scheduler to adapt tile dimensions at runtime.

**Answer:**

**Why fixed tile size is suboptimal:**

*Prefill (M = 2048, K = 4096, N = 4096):*

Optimal tile: large Tm (e.g., 256) and large Tn (e.g., 128) to maximise PE utilisation. A 256x128
output tile drives all 32x32 = 1024 PEs simultaneously (256/32 = 8 m-groups, 128/32 = 4 n-groups,
32 PEs each). Compute time per tile >> DMA time: double-buffering works perfectly.

*Decode (M = 1, K = 4096, N = 4096):*

With M=1, a 256x128 output tile is mostly empty — only the first row of PEs is active (1/256 = 0.4%
utilisation for m-dimension). The correct tile is (1 x Tn, K_tile) — all available PEs should be
mapped to the N dimension, not the M dimension. Optimal: 1x1024 output tile, with 1024 PE columns
each computing one output element of the single output row.

**Hardware mechanism for dynamic tile adaptation:**

**1. Configurable tile descriptor registers:**

The DMA and tile scheduler are controlled by a set of memory-mapped registers:
```
REG_TILE_M   (Tm parameter)
REG_TILE_N   (Tn parameter)
REG_TILE_K   (Tk parameter)
REG_LOOP_ORDER (which loop dimension is outermost)
REG_MODE     (PREFILL | DECODE)
```

The runtime software (or a dedicated context manager unit) writes these registers at the start
of each phase. The tile scheduler reads them and generates the corresponding address sequences.

**2. Phase-aware compiler codegen:**

The compiler generates two versions of each layer's schedule — one for prefill and one for decode —
and stores both in instruction memory. A mode register selects which schedule is executed. The
branch overhead is zero because both schedules are pre-compiled; only the initial register write
changes.

**3. Tile size quantisation for hardware simplicity:**

Rather than arbitrary tile sizes, implement a small set of supported tile configurations (e.g.,
power-of-two multiples of the PE array dimensions). A 3-bit register selects among 8 presets.
This avoids the need for a general divider in the address generation unit.

**4. Partial tile handling:**

When M is not a multiple of Tm (especially when M=1 and Tm=256), the hardware must either:
- Pad the matrix with zeros to the next Tm boundary (wastes compute but simplifies control).
- Support partial tiles: the tile scheduler checks if the remaining m-rows < Tm and enables only
  the corresponding PE rows via a per-row enable mask. This is the preferred approach for decode.

**Quantitative benefit:**

For decode with M=1, switching from a 256x128 tile (0.4% m-utilisation) to a 1x1024 tile
(100% n-utilisation across all PE columns) improves effective PE utilisation from ~0.4% to
~25% (still limited by the GEMV arithmetic intensity). The remaining inefficiency is inherent to
the memory-bound nature of GEMV and cannot be addressed by tiling alone.
