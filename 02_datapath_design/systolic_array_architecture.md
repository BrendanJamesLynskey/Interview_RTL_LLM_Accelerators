# Systolic Array Architecture

## Overview

Systolic arrays are the dominant compute structure in modern ML accelerators (Google TPU, Apple Neural Engine, AWS Trainium). Understanding their dataflow variants, timing behaviour, and utilisation characteristics is essential for LLM accelerator design roles. This document covers interview questions at three difficulty levels.

---

## Tier 1 — Fundamentals

### Q1. What is a systolic array and why is it used for matrix multiplication?

**Question:** Explain the basic structure of a systolic array. Why does it map well to GEMM workloads?

**Answer:**

A systolic array is a 2D mesh of simple processing elements (PEs) where data flows rhythmically through the array like pulses through a heart (hence "systolic"). Each PE performs a multiply-accumulate (MAC) operation on the data passing through it, then passes that data to its neighbour on the next clock cycle.

The key properties that make it suited to GEMM:

1. **Data reuse without centralised memory.** Each input element is reused as it flows across an entire row or column of PEs, amortising the cost of the memory fetch over many MACs. This directly reduces the bandwidth requirement.
2. **Regular, local interconnect.** Every PE only communicates with its immediate neighbours (left-right, up-down). There are no global wires or crossbars, which means the design scales to large arrays without a wiring bottleneck.
3. **High compute density.** The area is dominated by MACs, not routing or control logic. A 256x256 array has 65,536 MACs operating every cycle.
4. **Simple control.** Data alignment and pipeline fill/drain are the only control concerns. Each PE executes the same operation every cycle.

**Why this matters in interviews:** Candidates frequently confuse systolic arrays with vector processors or SIMD units. The distinction is that in a systolic array the data itself moves through fixed compute elements; in SIMD, data is loaded into registers and a single instruction applies to all lanes simultaneously.

**Common mistake:** Stating that systolic arrays eliminate all memory bandwidth. They reduce bandwidth proportionally to the reuse factor, but tiles must still be loaded from SRAM, and results must be written back.

---

### Q2. What are the three primary dataflow strategies for systolic arrays?

**Question:** Describe output-stationary, weight-stationary, and row-stationary dataflows. For each, state what data remains in the PE, what flows horizontally, and what flows vertically.

**Answer:**

**Output-Stationary (OS)**
- The partial sum accumulates inside the PE. It does not move until the full dot-product is complete.
- Activations flow horizontally (left to right) across a row.
- Weights flow vertically (top to bottom) down a column.
- Each PE receives one weight element and one activation element per cycle, multiplies them, and adds to a local accumulator register.
- Best for: large K dimension (long dot products), minimises read/write traffic for partial sums.

**Weight-Stationary (WS)**
- Weights are pre-loaded into each PE and stay fixed for the duration of a tile computation.
- Activations flow horizontally through the array.
- Partial sums flow vertically, accumulating as they pass through each row.
- Used by: Google TPU v1 (weights pre-loaded from a weight FIFO).
- Best for: workloads where the same weight matrix is used many times (e.g., batch inference with large batch size).

**Row-Stationary (RS)**
- A row of the filter (weight) stays in a PE. A row of the input slides through, and partial sums accumulate diagonally.
- More complex than OS or WS but achieves better energy efficiency by minimising total data movement across DRAM, on-chip SRAM, and register levels simultaneously.
- Used in: MIT's Eyeriss architecture.
- Best for: convolutional workloads with small filter sizes.

**Summary table:**

| Dataflow | Stationary data | Flowing data (H) | Flowing data (V) |
|---|---|---|---|
| Output-Stationary | Partial sum | Activation | Weight |
| Weight-Stationary | Weight | Activation | Partial sum |
| Row-Stationary | Weight row | Input row | Partial sum |

**Common mistake:** Confusing which dimension flows which direction. In the Google TPU (WS), activations flow *left to right* and partial sums flow *top to bottom*. The weight is loaded from a separate FIFO, not from a neighbouring PE.

---

### Q3. Describe the pipeline fill and drain phases of a systolic array.

**Question:** A 4x4 output-stationary systolic array is computing the product of two 4x4 matrices. How many cycles does it take? Why is there latency before any output is produced?

**Answer:**

For an N x N output-stationary systolic array multiplying two N x N matrices:

**Fill phase:** The first useful output from PE[0][0] requires K cycles (one per accumulation step). But PE[N-1][N-1] does not receive its first inputs until cycle 2*(N-1) because data must travel N-1 hops to reach it in both the horizontal and vertical directions.

**Computation:** After all PEs have their first data, each PE accumulates for K cycles (K = the shared inner dimension, which equals N for square matrices).

**Total latency for first output:** 2*(N-1) + K cycles from when the first data enters the array.

**For a 4x4 example (N=4, K=4):**
- Fill latency to reach PE[3][3]: 2*(4-1) = 6 cycles.
- Then K=4 accumulation cycles.
- First output available at cycle ~10 (after drain).
- Total cycles to produce all outputs: 2*(N-1) + K + N - 1 = approximately 3N + K - 3.

This is called the **skew latency** or **pipeline fill latency**. To keep the array busy across multiple tiles, input data for the next tile must be presented to the array before the current tile finishes (software pipelining at the tiling level).

**Why this matters:** In roofline analysis, the effective throughput is `N^2 * K / total_cycles`. Fill/drain overhead reduces utilisation for small matrices. An interviewer will often ask you to compute utilisation for a workload to verify you understand this.

---

### Q4. How does the Google TPU v1 systolic array differ from a generic textbook description?

**Question:** Describe the key architectural decisions in the TPU v1 matrix multiply unit (MXU). What precision does it use? How are weights loaded?

**Answer:**

**TPU v1 MXU specifics:**
- **Size:** 256 x 256 array of 8-bit MAC units, delivering 92 TOPS at 700 MHz (256*256*2*700M ops/s).
- **Dataflow:** Weight-stationary. Weights are pre-loaded from a dedicated 256-lane weight FIFO that feeds the left column of the array.
- **Activation path:** 8-bit activations enter from the top row. Because weights are stationary, partial sums flow downward and are accumulated into 32-bit accumulators at the bottom of each column. This avoids re-reading weights from SRAM for each input vector.
- **Unified Buffer (UB):** Activations are held in a 24 MiB on-chip SRAM called the Unified Buffer. The host CPU loads both the weight FIFO and the UB ahead of time.
- **No on-chip data cache:** Unlike a CPU/GPU, there is no cache hierarchy. All data must be explicitly DMA'd into on-chip memories, which makes the compiler responsible for tiling and staging.

**Key insight for interviews:** The TPU v1 was designed specifically for inference with batch sizes of 1 (single-user requests). The weight-stationary dataflow is optimal when the same weight matrix is multiplied against many different activation vectors — which is exactly what happens when serving queries to a deployed model.

---

## Tier 2 — Intermediate

### Q5. How does the output-stationary dataflow affect on-chip memory requirements compared to weight-stationary?

**Question:** You are tiling a GEMM of shape M=1024, N=1024, K=4096 onto a 64x64 systolic array. Compare the on-chip SRAM pressure for output-stationary versus weight-stationary dataflow, assuming INT8 weights and INT8 activations with INT32 accumulation.

**Answer:**

**Parameters:**
- Array size: 64x64
- Tile shape: 64x64 (one tile fills the array)
- GEMM: (1024 x 4096) * (4096 x 1024)
- Number of row-tiles: 1024/64 = 16; Number of col-tiles: 1024/64 = 16; K-tiles: 4096/64 = 64

**Output-Stationary:**

The partial sum for each output tile is held in the 64x64 PE accumulators. On-chip, you need:
- A tile of weight: 64 x 64 x 1 byte = 4 KB (brought in each K-iteration)
- A tile of activations: 64 x 64 x 1 byte = 4 KB (brought in each K-iteration)
- Accumulators: live in PEs as registers, not SRAM. 64x64 x 4 bytes = 16 KB of register state.
- **Total SRAM working set: ~8 KB** (two input tiles; accumulator state is register-based).

**Weight-Stationary:**

Weights are stationary, so an entire 64x64 weight tile is pre-loaded and held.
- Weight tile: 64 x 64 x 1 byte = 4 KB (held for all M-tiles)
- Activation tile: 64 x 64 x 1 byte = 4 KB (streamed through)
- Partial sums exiting the array bottom must be stored before the next K-tile arrives: 64 x 64 x 4 bytes = 16 KB of intermediate accumulation SRAM.
- **Total SRAM working set: ~24 KB** (weight tile + activation tile + partial sum tile).

**Conclusion:** Output-stationary has lower SRAM pressure for partial sums because they live in PE registers. Weight-stationary requires additional output buffer SRAM to accumulate partial sums that exit the array. However, weight-stationary reduces weight re-fetch bandwidth when the same weight tile is used across many M-tiles — the 4 KB weight tile is loaded once and reused 16 times across M-tiles.

**Interview follow-up:** "Which would you choose for a memory-bandwidth-limited design?" Answer: weight-stationary, because the weight reuse factor reduces DRAM bandwidth. For a register-area-limited design (e.g., very large arrays), output-stationary PE accumulators become expensive.

---

### Q6. How do you handle workloads where M or N is not a multiple of the array dimension?

**Question:** Your systolic array is 128x128. An attention layer has head_dim=64 and you are computing Q*K^T where Q is (seq_len x 64). How do you handle the dimension mismatch, and what is the utilisation cost?

**Answer:**

**The problem:** The array is 128 wide but the K dimension is only 64. If you map it naively, half the PEs are idle every cycle.

**Solutions:**

1. **Zero-padding the tile.** Pad the input matrices to the next multiple of 128 (pad to 64x128 by appending zeros). This is simple but guarantees 50% utilisation on the K dimension.

2. **Dual-tile packing.** Pack two independent small GEMMs side-by-side in the same array. For head_dim=64, you can map two attention heads simultaneously: head 0 occupies columns 0-63, head 1 occupies columns 64-127. This requires the compiler/driver to correctly align and extract the two results. Achieves near-100% utilisation.

3. **Rectangular array mode.** Some arrays support a configurable width, implemented by gating unused PE columns. This saves dynamic power but does not recover area efficiency.

4. **Software batching over heads.** Concatenate Q from multiple heads along the M dimension so the array sees a wide enough M dimension. This is effective for prefill but not for decode (where M=1).

**Utilisation analysis for packing:**
- Single head, no packing: utilisation = 64/128 = 50% on the weight dimension.
- Dual head packing: utilisation = 128/128 = 100%, at the cost of compiler complexity and requiring that both heads complete simultaneously.

**Real-world note:** NVIDIA's Tensor Core units handle this at the instruction level with shapes like WMMA 16x16x16. The programmer or compiler must tile to these shapes. Heads smaller than the tile shape are batched at the software level before dispatch.

---

### Q7. Explain the concept of data skewing and why it is necessary in a systolic array.

**Question:** When feeding data into a systolic array, why must input rows (or columns) be skewed in time? Draw the timing diagram for a 3x3 weight-stationary array multiplying two 3x3 matrices.

**Answer:**

**Why skewing is necessary:**

In an N x N array, PE[i][j] should receive `A[i][k]` and `B[k][j]` at the same cycle for the same value of k. If you present all rows of A simultaneously (cycle 0), then:
- PE[0][0] gets A[0][0] and B[0][0] — correct.
- PE[1][0] gets A[1][0] but B[0][0] has already passed — incorrect, it needs to receive B[0][0] one cycle later.

The solution is to **skew input row i of A by i cycles** and **skew input column j of B by j cycles**. This ensures that A[i][k] and B[k][j] arrive at PE[i][j] at the same time.

**Timing diagram — 3x3 weight-stationary, multiplying A (3x3) * B (3x3):**

```
Cycle:         0    1    2    3    4    5
Left inputs (A rows, skewed by row index):
  Row 0:      A00  A01  A02   -    -    -
  Row 1:       -   A10  A11  A12   -    -
  Row 2:       -    -   A20  A21  A22   -

Top inputs (B cols, skewed by col index):
  Col 0:      B00  B10  B20   -    -    -
  Col 1:       -   B01  B11  B21   -    -
  Col 2:       -    -   B02  B12  B22   -

PE[0][0] accumulates: A00*B00 (c0), A01*B10 (c1), A02*B20 (c2) -> C[0][0] ready at c3
PE[1][1] accumulates: A10*B01 (c2), A11*B11 (c3), A12*B21 (c4) -> C[1][1] ready at c5
PE[2][2] accumulates: A20*B02 (c4), A21*B12 (c5), A22*B22 (c6) -> C[2][2] ready at c7
```

The last output is ready at cycle 2*(N-1) + K = 2*2 + 3 = 7.

**Implementation:** Skewing is implemented by inserting pipeline registers (shift registers or FIFOs of depth i or j) on the input buses before the array boundary. This is hardwired and adds no runtime control overhead.

---

### Q8. How does array utilisation change as batch size increases in LLM decode vs. prefill?

**Question:** For a transformer with hidden_dim=4096, using a 256x256 systolic array, calculate the compute utilisation for (a) decode with batch=1, (b) decode with batch=32, and (c) prefill with seq_len=2048. Assume weight-stationary dataflow and square tiles.

**Answer:**

The projection weight matrix has shape (4096, 4096). The GEMM is (batch or seq_len) x 4096 x 4096, i.e., M = batch_or_seqlen, K=4096, N=4096.

The array size is 256x256. Tiling: N-tiles = 4096/256 = 16, K-tiles = 4096/256 = 16, M-tiles = ceil(M/256).

**Utilisation formula:** For a given M, the M-dimension utilisation = min(M, 256) / 256. K and N dimensions are always fully used since 4096 is a multiple of 256.

**(a) Decode, batch=1:**
M=1. Only 1 row of PEs is active per cycle.
M-utilisation = 1/256 = 0.39%.
Overall utilisation = 0.39% x 100% x 100% = **~0.4%**.
This is severely underutilised — this is the fundamental reason decode is memory-bandwidth-bound, not compute-bound.

**(b) Decode, batch=32:**
M=32. 32 rows of PEs active.
M-utilisation = 32/256 = 12.5%.
Overall utilisation = **12.5%**.
Still memory-bandwidth-bound but noticeably better. NVIDIA H100 hardware achieves meaningful throughput improvements from batching even in the 32-128 range.

**(c) Prefill, seq_len=2048:**
M=2048. M-tiles = 2048/256 = 8 tiles, each fully packed.
M-utilisation = 256/256 = 100%.
Overall utilisation = **100%** (assuming weights and activations stream without stalls).
Prefill is compute-bound on large accelerators with sufficient memory bandwidth.

**Key insight:** This analysis explains why decode and prefill have different hardware bottlenecks and why disaggregated serving (running prefill and decode on separate hardware) is architecturally motivated.

---

## Tier 3 — Advanced

### Q9. How would you design a systolic array that efficiently handles both GEMM (prefill) and GEMV (decode) in the same hardware?

**Question:** Most systolic arrays are optimised for GEMM. Describe two hardware techniques that improve GEMV throughput on a systolic array, with the architectural trade-offs.

**Answer:**

The problem with GEMV on a systolic array is that M=1, so the array is utilised at rate 1/N along the M-dimension. Two strategies:

**Strategy 1: Temporal folding / vector replication**

Replicate the input vector across all M rows of the array. Instead of computing one output at a time, compute N partial dot products in parallel, where each row of the array accumulates a different segment of the weight row.

Implementation: The single activation vector is broadcast to all 256 rows simultaneously. Each row of PEs computes one element of the output vector. The full output vector (length N=256 in one tile) is produced in K/256 cycles rather than K cycles.

This requires a broadcast bus (tree or H-tree structure) rather than a shift-register propagation path. Cost: additional wiring and a mux to switch between shift (for GEMM) and broadcast (for GEMV) modes.

**Strategy 2: Transposed weight layout for column-parallel execution**

Pre-transpose the weight matrix and store it as W^T. For GEMV y = Wx, compute instead y = (W^T)^T x. The array now computes K independent dot products simultaneously (one per column of PEs), where each column accumulates along the M-dimension.

In practice this means: each column of PEs independently accumulates a partial sum for one output element. All K weights for one output neuron are spread across one column. This is already how weight-stationary dataflow works with a column-parallel reduction, so no hardware change is needed — but the compiler must tile the GEMV differently (K-first tiling rather than N-first).

**Strategy 3: Subarray activation (power gating unused PEs)**

For GEMV, gate the clock (and optionally power) to M-1 rows of PEs, using only one row. This does not improve throughput but reduces dynamic power by (N-1)/N. It requires per-row clock enable signals, which add area to the control plane.

**Trade-off summary:**

| Technique | Throughput gain | Area cost | Power | Compiler complexity |
|---|---|---|---|---|
| Broadcast bus | N (linear) | ~15% (bus + mux) | Higher (all PEs active) | Medium |
| Transposed tiling | 1 (no gain; reuses existing WS) | 0 | Same | Low |
| Clock gating | 0 (power only) | ~5% (enable logic) | Proportional reduction | None |

**Real-world example:** Apple's ANE and NVIDIA's H100 Tensor Cores both include broadcast paths for this reason.

---

### Q10. Analyse the impact of PE pipeline depth on systolic array throughput and area.

**Question:** A MAC unit in each PE can be implemented as: (a) single-cycle combinational, (b) 2-stage pipeline, or (c) 4-stage pipeline. For a 256x256 array running at a target of 1 GHz, analyse the throughput, latency, and area implications of each option. Assume the critical path of an INT8 multiplier is 1.8 ns and the adder (accumulator) is 0.4 ns.

**Answer:**

**Critical path analysis:**

At 1 GHz, the clock period is 1.0 ns. The critical path of the MAC (INT8 multiply + 32-bit accumulate) is 1.8 + 0.4 = 2.2 ns. This cannot run at 1 GHz without pipelining.

**(a) Single-cycle combinational MAC:**
- Max clock frequency: 1/2.2 ns = 455 MHz.
- Throughput: 256*256 = 65,536 MACs/cycle x 455 MHz = 29.8 TOPS.
- Latency through array: 2*(N-1) + K cycles. At 455 MHz, this is longer in wall-clock time.
- Area: Smallest (no pipeline registers inside PE).
- Cannot meet 1 GHz target.

**(b) 2-stage pipeline (multiplier in stage 1, adder/accumulator in stage 2):**
- Stage 1: 1.8 ns multiplier. Stage 2: 0.4 ns adder. Critical path = max(1.8, 0.4) = 1.8 ns per stage.
- Max clock frequency: 1/1.8 ns = 556 MHz. Still below 1 GHz.
- Each PE now has one pipeline register storing the intermediate product (INT16 or INT32). 256*256 = 65,536 extra registers.
- Area increase: ~10-15% over option (a).

**(c) 4-stage pipeline (multiplier split into 3 stages, accumulator in stage 4):**
- Multiplier critical path 1.8 ns split across 3 stages: ~0.6 ns per stage.
- Accumulator: 0.4 ns in stage 4.
- Critical path per stage: max(0.6, 0.4) = 0.6 ns.
- Max clock frequency: 1/0.6 ns = 1.67 GHz. Meets 1 GHz with margin.
- At 1 GHz: throughput = 65,536 MACs/cycle x 1 GHz = 65.5 TOPS.
- Each PE adds 3 pipeline register stages: 3 x 65,536 registers.
- Area increase: ~35-40% over option (a) for register storage alone, partially offset by better Vdd scaling at faster frequency.
- Latency through the array increases by (pipeline_stages - 1) cycles per PE hop. The effective skew must account for this in the data alignment logic.

**Key trade-off insight:** Pipeline depth trades area and latency for frequency. The optimal depth minimises the product of (area penalty) x (frequency-limited throughput degradation). For most commercial accelerators, 4-8 stage MAC pipelines are typical at 1 GHz+ in 7 nm/5 nm processes.

**Common interview trap:** Candidates forget that a deeper pipeline increases the fill latency of the systolic array. With a 4-stage PE pipeline, the systolic array fill takes an additional 3 cycles per row/column hop compared to a 1-stage design.

---

### Q11. How do you handle output precision accumulation to avoid overflow in a large systolic array?

**Question:** In a 256x256 INT8 weight-stationary systolic array, partial sums flow down each column accumulating 256 INT8 multiplications before being written to the output buffer. What is the minimum accumulator width required? How does this affect the column reduction tree design?

**Answer:**

**Overflow analysis:**

- INT8 value range: -128 to +127.
- INT8 x INT8 product range: -128 * 127 = -16,256 to 127 * 127 = 16,129. Fits in INT16 (range ±32,767). However, the product of -128 * -128 = 16,384 which also fits in INT16.
- After 256 accumulations of INT16 products: worst case = 256 * 16,384 = 4,194,304. This requires log2(4,194,304) = 22 bits to represent. In a signed context, the range is -2^22 to 2^22, so **24-bit signed** accumulation is the minimum safe width.
- In practice, industry designs use **32-bit accumulators** for INT8 to (a) match INT32 standards and simplify downstream processing, (b) leave headroom for K > 256, and (c) allow the accumulator to be shared with FP32 pipelines.

**Column reduction tree design:**

Rather than a simple chain accumulator (which has a loop-carried dependency that limits frequency), the column reduction is structured as a Wallace tree (or Dadda tree):

```
Depth-0 (256 partial products, each INT16):  PP[0] ... PP[255]
Depth-1 (128 INT17 sums):                    PP[i] + PP[i+128]
Depth-2 (64 INT18 sums):                     ...
...
Depth-8 (1 INT24 sum):                       final output
```

The tree has log2(256) = 8 stages. Each stage adds 1 bit of width to prevent carry overflow.

**Pipeline registers in the tree:**

The 8-stage tree can be broken into pipeline stages. A common partition is 4 pipeline stages with 2 tree levels each. This gives:
- 4 cycles of latency through the reduction tree.
- Throughput of one result per column per cycle once the pipeline is full.

**Design implication:** The column reduction tree is physically tall and located at the bottom edge of the array. In layout, this is the densest routing region. For a 256-column array, there are 256 independent trees, each accepting 256 inputs. The total gate count for the reduction trees alone can be 20-30% of the total array area.

---

### Q12. What are the implications of using a 2D torus interconnect versus a 2D mesh for a large systolic array?

**Question:** Compare a 2D mesh interconnect (edges are dead ends) versus a 2D torus (edges wrap around) for a 256x256 systolic array. Discuss data throughput, latency, routing complexity, and practical design concerns.

**Answer:**

**2D Mesh:**
- Simple topology: PE[i][j] connects only to PE[i+1][j], PE[i-1][j], PE[i][j+1], PE[i][j-1]. Edge PEs have fewer neighbours.
- Data enters from the left edge (or top edge) and exits at the right edge (or bottom edge).
- Input bandwidth is limited to N wires per edge. For a 256-wide array carrying INT8 data at 1 GHz, input bandwidth = 256 bytes/cycle = 256 GB/s per edge.
- **Dead ends** mean data cannot wrap around. If you need to re-circulate output data (e.g., for layer-norm computed in-place), you must write back to SRAM and reload.
- Routing is simple and fully deterministic, so timing closure is straightforward.

**2D Torus:**
- PE[i][j] connects to PE[(i+1) mod N][j] and PE[i][(j+1) mod N]. All PEs are topologically equivalent.
- Data can wrap around, enabling recirculation of partial results without off-array memory accesses. Useful for ring-reduce operations across heads or for in-situ softmax.
- **Practical problem:** Wrap-around wires must physically cross the chip from one edge to the opposite edge. At 256x256 with typical 7nm pitch, these wires are 5-10 mm long, creating severe timing and routing congestion at the boundaries.
- Wrap-around wires require repeaters (buffers), which add latency and area.
- Most commercial designs prefer a **mesh with off-array reduction networks** (separate from the systolic array) rather than a true torus.

**Conclusion for interviews:** A torus improves topological flexibility but is impractical at large scale in 2D silicon without multi-chip or 3D stacking. Google TPUs use a **2D torus interconnect between TPU chips** in a pod, but each chip's internal MXU is a simple mesh. The distinction between intra-chip and inter-chip topology is important.

---
