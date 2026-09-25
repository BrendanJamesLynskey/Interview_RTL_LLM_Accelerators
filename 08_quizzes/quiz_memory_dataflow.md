# Quiz: Memory and Dataflow

## Overview

Self-assessment quiz covering on-chip memory hierarchy design, DMA engines, tiling strategies,
double buffering, weight streaming vs. weight caching, SRAM banking and multi-port access,
and memory bandwidth analysis for LLM accelerators. Questions target RTL and microarchitecture
design decisions.

**Instructions:** Select the single best answer for each question. Answers and full explanations
appear at the bottom.

---

## Section 1 — On-Chip Memory Hierarchy (Questions 1-4)

**Q1.** An LLM accelerator uses a two-level on-chip SRAM hierarchy: a small, fast L1 scratchpad
per compute cluster and a larger L2 shared buffer. What is the primary purpose of the L1
scratchpad compared to L2?

- A) L1 stores model weights permanently for the duration of inference to avoid DRAM accesses
- B) L1 acts as a staging buffer for the current computation tile, providing the lowest-latency
   access for operands that are actively being consumed by the MAC array
- C) L1 provides error correction (ECC) for the MAC unit's intermediate results
- D) L1 caches the KV cache for the most recently processed tokens, replacing the DRAM KV store

---

**Q2.** SRAM access energy is proportional to SRAM size. A designer must choose between one
monolithic 4 MB SRAM and four 1 MB SRAMs. Which configuration is lower energy per access, and why?

- A) The monolithic 4 MB SRAM, because a single large array has lower address decoder overhead
- B) The four 1 MB SRAMs, because each access activates a smaller bit-line and word-line structure,
   consuming less energy per bit read
- C) Both configurations consume equal energy per access because SRAM energy scales with data width,
   not array depth
- D) Energy per access cannot be determined without knowing the access frequency

---

**Q3.** A systolic array PE requires one operand from the weight SRAM and one from the activation
SRAM every cycle. To sustain this without stalls, what minimum number of read ports must each
SRAM provide per PE column?

- A) 1 read port shared across all PEs in the column
- B) 1 read port per PE in the column, fully independent
- C) 1 read port per column is sufficient if weights are broadcast and activations are shifted
   through a shift register
- D) 2 read ports per PE (one for the weight, one for the activation simultaneously)

---

**Q4.** In a hardware design with 8 SRAM banks, each bank has a single read/write port. To avoid
access conflicts when computing a matrix tile, the designer applies a bank interleaving scheme.
Which statement correctly describes the benefit?

- A) Bank interleaving ensures that consecutive matrix elements map to the same bank, increasing
   spatial locality
- B) Bank interleaving distributes consecutive elements across different banks, ensuring that
   a stride-1 access pattern activates only one bank per cycle, avoiding conflicts
- C) Bank interleaving distributes consecutive elements across different banks, so a stride-1
   access pattern can activate all 8 banks in parallel for 8x bandwidth
- D) Bank interleaving allows two different memory sizes to be combined in a single logical address
   space

---

## Section 2 — DMA Engines (Questions 5-7)

**Q5.** A DMA engine is used to transfer weight tiles from HBM to on-chip SRAM. The compute engine
processes one tile in 100 cycles, and the DMA transfer for the next tile also takes 100 cycles.
Without double buffering, what is the compute utilisation?

- A) 100% — DMA and compute can overlap
- B) 50% — compute stalls waiting for DMA to complete before starting the next tile
- C) 75% — the DMA is 25% faster than compute due to burst mode
- D) 0% — DMA transfers must complete before any compute begins

---

**Q6.** A DMA engine supports 2D transfers (also called strided DMA). A weight matrix is stored
in row-major order in DRAM with a row stride of 512 elements (padded). The designer wants to
transfer a contiguous 16x16 sub-matrix tile. Without 2D DMA, how many separate 1D DMA requests
are needed?

- A) 1 request — the DMA automatically detects the stride and handles it
- B) 16 requests — one per row of the tile
- C) 256 requests — one per element
- D) 2 requests — one for rows and one for columns

---

**Q7.** In RTL design of a DMA controller, a descriptor-based DMA is preferred over a
register-programmed DMA. What is the primary advantage of descriptor chains?

- A) Descriptors are stored in on-chip SRAM, avoiding DRAM accesses for transfer metadata
- B) Descriptor chains allow the CPU to pre-program a sequence of transfers; the DMA automatically
   processes the chain without CPU intervention between transfers, enabling overlap of CPU
   activity with data movement
- C) Descriptors compress the transfer parameters, reducing the number of configuration cycles
   before each transfer
- D) Descriptor-based DMA eliminates the need for interrupt handling at transfer completion

---

## Section 3 — Tiling Strategies (Questions 8-10)

**Q8.** When tiling a large matrix multiply C = A * B for an on-chip SRAM of capacity S, the
designer must choose tile dimensions. For a square tile of dimension T x T with FP16 data (2 bytes
per element), what constraint must T satisfy to fit A-tile, B-tile, and C-tile simultaneously?

- A) T^2 * 2 <= S (only the output tile needs to fit)
- B) 3 * T^2 * 2 <= S (all three tiles must fit simultaneously)
- C) T^2 * 6 <= S (three tiles of 2 bytes each, with a factor of 3 safety margin)
- D) T <= sqrt(S / 6) which is equivalent to option B, so both B and D are correct

---

**Q9.** A designer tiles a matrix multiply with tiles that are too small. Which performance
problem results?

- A) DRAM bandwidth utilisation decreases because burst transfers are too short to fill the DRAM
   command queue
- B) Compute utilisation decreases because the tile dimensions are smaller than the systolic
   array size, leaving part of the array idle
- C) The KV cache overflows because small tiles require more attention head computations
- D) Register file pressure increases because more partial sums must be stored simultaneously

---

**Q10.** Loop tiling (also called blocking) for cache/SRAM optimisation transforms nested loops
by introducing additional loop levels. For a triple-nested loop implementing matrix multiply,
tiling introduces up to how many additional loop levels (ignoring cleanup loops)?

- A) 1 additional level
- B) 3 additional levels (one per original loop)
- C) 6 additional levels (tiling each of i, j, k)
- D) 9 additional levels

---

## Section 4 — Double Buffering (Questions 11-13)

**Q11.** Double buffering requires two physical buffers (A and B). While the compute engine
processes data in buffer A, the DMA loads the next tile into buffer B. What is the minimum
condition for double buffering to achieve 100% compute utilisation?

- A) The SRAM read bandwidth must equal the DRAM write bandwidth
- B) The DMA transfer time for one tile must be less than or equal to the compute time for one tile
- C) The two buffers must be on separate SRAM banks to prevent port conflicts
- D) The DMA and compute must be on separate clock domains to allow independent scheduling

---

**Q12.** In RTL, double buffering is commonly implemented using a ping-pong buffer scheme. In
SystemVerilog, the buffer selector is a 1-bit register that alternates each tile. Which RTL
hazard must be explicitly handled?

- A) A read-write conflict when the DMA writes to buffer B while compute reads from buffer B
   simultaneously — a pointer swap must not occur until both the DMA write and the compute read
   are complete
- B) Clock domain crossing metastability between the DMA clock and the compute clock
- C) SRAM timing violations caused by the DMA accessing the buffer at a higher frequency than specified
- D) Overflow of the buffer selector register after 2^32 swaps

---

**Q13.** A triple-buffering scheme adds a third buffer compared to double buffering. Under what
condition does triple buffering improve throughput over double buffering?

- A) When the compute time is exactly twice the DMA time, triple buffering keeps all three buffers
   occupied and maintains 100% DMA utilisation
- B) When the compute time is longer than the DMA time, the DMA would otherwise be idle waiting
   for compute to release a buffer; the third buffer allows the DMA to pre-fetch two tiles ahead
- C) Triple buffering always improves throughput because three buffers provide more memory than two
- D) Triple buffering is needed when the SRAM has only one read port and the DMA and compute cannot
   access the same buffer simultaneously

---

## Section 5 — Weight Streaming vs. Weight Caching (Questions 14-16)

**Q14.** Weight streaming means loading weights from DRAM for each layer on every inference step.
Weight caching means storing weights on-chip permanently. For a 7B parameter model in INT8 with
a 16 MB on-chip SRAM, which strategy is feasible and why?

- A) Weight caching, because 7B x 1 byte = 7 GB, which exceeds 16 MB on-chip — this is a
   contradiction, so weight caching is not feasible
- B) Weight streaming, because 7 GB of weights cannot fit in 16 MB of SRAM; weights must be
   streamed from DRAM in tiles
- C) Weight caching in compressed form using delta coding, reducing 7 GB to under 16 MB
- D) Weight caching is feasible because the INT8 format compresses weights by 4x relative to FP32,
   fitting 7B parameters in 1.75 GB on chip

---

**Q15.** For a decode-phase workload (batch size 1, weight-streaming), what determines the
minimum achievable latency per token?

- A) The compute throughput of the MAC array in FLOP/s
- B) The on-chip SRAM bandwidth between the weight buffer and the MAC array
- C) The DRAM bandwidth, because weights must be streamed from DRAM once per token and the
   workload is memory-bandwidth-bound
- D) The clock frequency of the DMA engine

---

**Q16.** A weight caching strategy caches the weights for one transformer layer at a time in
on-chip SRAM, reusing them for a batch of B tokens before evicting and loading the next layer.
What is the critical batch size below which weight caching provides no benefit over streaming?

- A) B = 1 (any batch size improves utilisation)
- B) The batch size at which the compute time processing B tokens equals the DMA load time for
   the layer weights — below this, compute is faster than DMA and weights are loaded before
   they are fully used
- C) B = the number of attention heads (because each head requires a separate weight load)
- D) B = the tile size of the MAC array (because sub-tile batches cannot be parallelised)

---

## Section 6 — SRAM Banking and Bandwidth Analysis (Questions 17-20)

**Q17.** A MAC array has 256 PEs, each requiring one INT8 weight per cycle. The weight SRAM is
clocked at the same frequency as the MAC array. What is the minimum SRAM read bandwidth required
in bytes/cycle?

- A) 1 byte/cycle (shared weight broadcast)
- B) 16 bytes/cycle (one per row of a 16x16 array)
- C) 256 bytes/cycle (one per PE, independent access)
- D) 512 bytes/cycle (double pumped SRAM for higher bandwidth)

---

**Q18.** An SRAM bank conflict occurs when two accesses target the same bank in the same cycle.
In a 16-bank SRAM, a convolution kernel accesses elements with a stride of 16 elements. If each
bank holds consecutive elements in a round-robin mapping, what fraction of accesses experience a
bank conflict?

- A) 0% — stride-16 accesses across 16 banks always distribute evenly
- B) 100% — all stride-16 accesses land on the same bank (bank 0 for address 0, 16, 32, ...)
- C) 50% — every other access lands on the same bank
- D) 6.25% — one out of 16 accesses conflicts

---

**Q19.** To resolve SRAM bank conflicts caused by power-of-two strides, designers use a technique
called XOR-based address scrambling. How does this work?

- A) Row and column indices are swapped before bank selection, distributing elements diagonally
   across banks
- B) The bank index is computed as (element_address XOR (element_address >> log2(num_banks)),
   breaking the regular stride pattern so that power-of-two strides no longer always map to the
   same bank
- C) A pseudo-random number generator seeds the bank selection for each access, making conflicts
   statistically unlikely
- D) Two separate SRAM arrays are interleaved such that odd addresses go to one array and even
   addresses to the other, halving the conflict rate

---

**Q20.** An accelerator processes a 4096-length attention sequence. The Q, K, V matrices each
have shape [4096, 64] in FP16. Computing the attention score matrix S = Q * K^T requires reading
Q and K from SRAM. What is the total SRAM read bandwidth (bytes) consumed for this operation,
assuming Q and K are each read exactly once?

- A) 512 KB (only K is read; Q remains in registers)
- B) 1 MB (Q and K together)
- C) 4 MB (Q, K, and the intermediate score matrix)
- D) 2 MB (Q, K, and V all read once, divided by the number of heads)

---

## Answer Key

| Q  | Answer |
|----|--------|
| 1  | B      |
| 2  | B      |
| 3  | C      |
| 4  | C      |
| 5  | B      |
| 6  | B      |
| 7  | B      |
| 8  | B      |
| 9  | B      |
| 10 | B      |
| 11 | B      |
| 12 | A      |
| 13 | B      |
| 14 | B      |
| 15 | C      |
| 16 | B      |
| 17 | C      |
| 18 | B      |
| 19 | B      |
| 20 | B      |

---

## Detailed Explanations

### Q1 — Correct: B

L1 scratchpad serves as an operand staging buffer for the immediately active computation tile.
The MAC array reads from L1 every cycle, so L1 latency (typically 1-2 cycles) directly affects
MAC array throughput. L2 provides a larger intermediate buffer between L1 and DRAM.

- A incorrect: Storing all model weights on-chip permanently would require tens of gigabytes for
  even small models (7B params = 7 GB INT8) — orders of magnitude beyond typical on-chip SRAM
  (4-32 MB).
- C incorrect: ECC is a reliability feature and is orthogonal to the L1/L2 hierarchy purpose.
  ECC may or may not be included in L1/L2, but it is not the primary purpose of L1.
- D incorrect: The KV cache can be hundreds of megabytes to gigabytes per request. L1 scratchpad
  is far too small and is not structured as a KV store.

### Q2 — Correct: B

SRAM read energy is dominated by the charging and discharging of bit-lines and word-lines within
the array. In a smaller SRAM, these lines are shorter and have less capacitance. Accessing 1 MB of
SRAM costs roughly 4x less energy than accessing one 4 MB monolithic SRAM for the same data.

- A incorrect: Address decoder overhead scales logarithmically with array depth and is a small
  fraction of total SRAM energy. The dominant cost is bit-line swing, not address decoding.
- C incorrect: SRAM energy does scale with data width, but it also strongly scales with array size
  (bit-line and word-line capacitance both increase with array depth).
- D incorrect: While access frequency affects total energy consumption, energy per access is a
  property of the physical structure (capacitances), not the frequency.

### Q3 — Correct: C

In a systolic array, weights are typically broadcast to a column of PEs, and activations shift
through a delay chain. Each PE receives its weight as a broadcast (one source drives all PEs in
the column) and receives its activation from its neighbour through a register chain. The SRAM
therefore needs: one read port per cycle for the weight of the entire column (broadcast), and one
read port for the activation of the top PE (the rest come from the shift register chain).

- A incorrect: One shared read port is insufficient if weights need to be broadcast independently
  per column and activations also need to be read; concurrent access from multiple PEs would
  require arbitration and stalls.
- B incorrect: Independent read ports for every PE in a column would require N read ports for an
  N-PE column, which is extremely costly in area — this is not how systolic arrays work.
- D incorrect: Two ports per PE (one for weight, one for activation) would be N ports for an
  N-deep column — prohibitively expensive and unnecessary given the broadcast/shift architecture.

### Q4 — Correct: C

In a standard round-robin bank interleaving, consecutive addresses map to consecutive banks
(element 0 to bank 0, element 1 to bank 1, ..., element 7 to bank 7, element 8 to bank 0, ...).
A stride-1 access pattern accessing 8 consecutive elements touches all 8 banks in one cycle,
achieving peak bandwidth (8x a single bank).

- A incorrect: Mapping consecutive elements to the same bank is the opposite of interleaving; this
  would maximise conflicts.
- B incorrect: A stride-1 pattern accessing 8 elements touches 8 banks (all different), not just
  one bank. The description has the directions of the implication backwards.
- D incorrect: Bank interleaving is a memory address organisation scheme, not a way to combine
  different-sized SRAMs. Combining different sizes would be a memory map design decision.

### Q5 — Correct: B

Without double buffering, the sequence is: DMA transfer (100 cycles) → Compute (100 cycles) →
DMA transfer (100 cycles) → Compute (100 cycles). Compute is active 50% of the time and idle
during DMA. Utilisation = 100 / (100 + 100) = 50%.

- A incorrect: 100% would require DMA and compute to overlap, which is exactly what double
  buffering enables — not the case without it.
- C incorrect: The problem states both take 100 cycles; there is no statement that DMA is 25%
  faster. 75% has no basis in the given data.
- D incorrect: Even without double buffering, compute begins after each DMA transfer completes;
  compute is definitely not zero.

### Q6 — Correct: B

The DRAM stores the weight matrix in row-major order with padding (stride = 512 elements per row).
The tile is 16 contiguous elements per row, but the next row starts 512 elements later in memory.
Without 2D DMA, a 1D transfer would include the 496-element gap between rows. To transfer exactly
the tile, 16 separate 1D DMA requests are needed — one per row, each 16 elements long at the
appropriate address.

- A incorrect: Standard 1D DMA does not automatically detect strides; it transfers a flat contiguous
  memory range. The designer must explicitly issue separate requests or use a 2D DMA command.
- C incorrect: 256 individual element requests would be used without any DMA burst capability —
  this is the worst case of a software-managed copy loop, not a DMA transfer.
- D incorrect: There is no meaningful partition into "row requests" and "column requests" in 1D DMA.
  Rows are the natural atomic unit because each row is a contiguous segment.

### Q7 — Correct: B

A descriptor chain is a linked list of transfer descriptors in memory. The DMA reads each
descriptor, executes the transfer, then automatically fetches the next descriptor without CPU
involvement. The CPU can program the entire chain before the first transfer begins and then
continue executing other tasks.

- A incorrect: Descriptor chains are typically stored in main DRAM (or a designated descriptor
  region), not on-chip SRAM. Fetching descriptors from DRAM is a deliberate trade-off — the
  descriptor fetches are small and infrequent relative to the data transfers.
- C incorrect: Descriptor size (typically 16-32 bytes per descriptor) is not significantly smaller
  than the equivalent register writes. Compression is not a motivation.
- D incorrect: Descriptor-based DMA still generates completion interrupts (or can use polling);
  the mechanism for indicating transfer completion is unchanged.

### Q8 — Correct: B

Three T x T tiles (A, B, C) each of size T^2 elements at 2 bytes/element must fit simultaneously:
Total memory = 3 * T^2 * 2 bytes <= S. This gives T <= sqrt(S / 6).

- A incorrect: Only fitting the output tile would allow T^2 * 2 = S, giving a much larger T —
  but A and B tiles would overflow SRAM during computation.
- C incorrect: T^2 * 6 is numerically the same as option B, but the stated reasoning is wrong:
  the factor of 3 is the number of tiles (A, B and C), not a safety margin.
- D incorrect: Option D states it is equivalent to B, which is correct as a mathematical statement.
  However, the question asks for a single best answer characterising the constraint, and option B
  is the direct statement of that constraint without a dependency on another option's correctness.

### Q9 — Correct: B

A systolic array has a fixed physical size (e.g., 128x128 PEs). If the tile is smaller than the
array (e.g., a 32x32 tile on a 128x128 array), a large fraction of PEs have no work to do and
remain idle. Utilisation = tile_size / array_size = (32x32) / (128x128) = 6.25%.

- A incorrect: DRAM burst length is a separate consideration. Small SRAM tiles do not necessarily
  mean short DRAM bursts — DMA transfers are sized independently of tile dimensions.
- C incorrect: KV cache size depends on sequence length, attention head count, and precision —
  not on the compute tile size.
- D incorrect: Register file pressure is associated with software-managed vectorisation loops,
  not with hardware tile sizes in a fixed-function systolic array.

### Q10 — Correct: B

The standard triple-nested loop for matrix multiply has three loop variables (i, j, k). Loop tiling
introduces a tiled (outer) loop and a within-tile (inner) loop for each of the three variables.
This gives 3 outer tile loops + 3 inner element loops = 6 loops total, but compared to the
original 3, **3 additional levels** are introduced.

- A incorrect: Tiling a single loop adds one additional level; for three loops, tiling all three
  adds three levels.
- C incorrect: 6 additional levels (to a total of 9) would imply tiling each loop twice (e.g.,
  L2 and L3 tiling for two memory levels), which is register-plus-cache blocking. The question
  asks for single-level tiling.
- D incorrect: 9 levels total would arise from three levels of tiling per loop (e.g., L1, L2,
  L3), which is excessive for the standard description.

### Q11 — Correct: B

Double buffering achieves 100% compute utilisation when the DMA can finish loading the next tile
before (or exactly when) the compute engine finishes the current tile. If DMA time > compute time,
the compute engine stalls waiting for the DMA, and utilisation drops below 100%.

- A incorrect: SRAM read bandwidth and DRAM write bandwidth are asymmetric bandwidth paths
  (SRAM is read by compute; DRAM is written to the buffer by DMA). While bandwidth balance matters
  for other reasons, it is not the condition for 100% utilisation.
- C incorrect: Placing buffers on separate banks is a good practice to avoid port conflicts, but
  it is not the fundamental condition for 100% utilisation. Even with port conflicts, if DMA is
  fast enough, utilisation can approach 100%.
- D incorrect: Double buffering can work within a single clock domain. Separate clock domains
  are not required; only independent access to separate physical buffers is needed.

### Q12 — Correct: A

The pointer swap (buffer A becomes the compute buffer, buffer B becomes the DMA target) must not
happen while compute is still reading from A or while DMA is still writing to B. RTL must include
handshaking: a "compute_done" signal and a "dma_done" signal that are both asserted before the
selector register is updated.

- B incorrect: A single clock domain system has no clock domain crossing. Even if two different
  clock domains are used for DMA and compute, this is a separate design concern that is managed
  with synchronisers, not by double buffering logic.
- C incorrect: SRAM timing violations are a physical implementation concern managed through
  timing constraints and margin analysis, not an RTL hazard of the ping-pong control logic.
- D incorrect: A 1-bit register alternating between 0 and 1 cannot overflow. The binary counter
  naturally wraps at the modulo-2 boundary.

### Q13 — Correct: B

With double buffering: if DMA time < compute time, the DMA finishes loading the next tile and then
sits idle waiting for compute to release the current buffer. A third buffer allows the DMA to load
tile N+2 while compute processes tile N and the second buffer holds the already-loaded tile N+1.
This keeps the DMA busy and hides even greater data transfer latency.

- A incorrect: If compute time = 2 * DMA time, double buffering already achieves 100% compute
  utilisation (DMA loads tile N+1 in half the compute time). Triple buffering adds no benefit here.
- C incorrect: More memory capacity is not intrinsically beneficial for throughput unless it enables
  additional overlap. Three buffers of the same size as two buffers does not improve throughput
  if the workload does not benefit from deeper pre-fetching.
- D incorrect: Double buffering already solves the single read-port conflict by using separate
  buffers. The number of SRAM ports is not the reason to use triple buffering.

### Q14 — Correct: B

7B parameters x 1 byte = 7 GB. On-chip SRAM = 16 MB. 7 GB >> 16 MB: weight caching of the entire
model is physically impossible. Weights must be streamed from DRAM in tiles.

- A incorrect: Option A correctly states that 7 GB > 16 MB but then draws the wrong conclusion
  ("this is a contradiction, so weight caching is not feasible" — the contradiction IS the reason
  weight streaming is used, not weight caching). The phrasing is confusing but the conclusion
  should be that streaming is necessary, which is option B.
- C incorrect: Delta coding compresses data only when it has high temporal correlation (adjacent
  values are similar). Model weights have some structure, but a lossless compression ratio of 7 GB
  to 16 MB = 437x is not achievable with delta coding. Even aggressive compression achieves at
  best 2-4x for neural network weights.
- D incorrect: INT8 is 4x smaller than FP32, so 7B INT8 parameters = 7 GB (not 1.75 GB). The
  7 GB figure already assumes INT8. Even in FP32, 7B parameters would be 28 GB — neither fits.

### Q15 — Correct: C

In single-token decode, the workload is memory-bandwidth-bound: every weight byte must be read
from DRAM once per token. The minimum time to process one token = total weight bytes / DRAM
bandwidth. This is a hard lower bound on latency regardless of compute capability.

- A incorrect: For a memory-bandwidth-bound workload, the compute array is mostly idle. Peak FLOP/s
  does not determine the bottleneck.
- B incorrect: On-chip SRAM bandwidth is typically much higher than DRAM bandwidth (10-100x) and
  is not the binding constraint. Data must first arrive from DRAM.
- D incorrect: The DMA clock frequency determines how quickly DMA descriptor overhead is processed,
  but the bottleneck is the peak DRAM bandwidth, not DMA control overhead.

### Q16 — Correct: B

Weight caching provides benefit by amortising the weight load time over B tokens. If the compute
time to process B tokens is less than the DMA load time, the compute finishes before all weights
are loaded — you get no reuse benefit. The break-even is when compute_time(B tokens) =
DMA_load_time(one layer). Below this crossover, streaming and caching are equivalent.

- A incorrect: B=1 means no reuse — one token processed per weight load. This provides no benefit
  over streaming (both require loading weights once per token).
- C incorrect: The number of attention heads is unrelated to the batch size threshold for weight
  reuse efficiency.
- D incorrect: Sub-tile batches can still benefit from weight caching if the DMA time is longer
  than compute time. The MAC array tile size determines parallelism, not the caching benefit.

### Q17 — Correct: C

If each of the 256 PEs independently reads one weight byte per cycle, the SRAM must provide
256 bytes per cycle of read bandwidth. If weights are not broadcast (i.e., each PE may have a
different weight), this requires 256 independent byte reads per cycle.

- A incorrect: 1 byte/cycle (shared broadcast) would only work if all 256 PEs use the same weight
  simultaneously — this is the case only for specific dataflows (e.g., broadcasting the same
  weight row to all columns). For a general MAC array, PEs have different weights.
- B incorrect: 16 bytes/cycle would serve a 16x16 array but not a 256-PE array. For a 16x16
  array, if each row broadcasts one weight, 16 bytes per cycle suffices, but the question asks
  about 256 PEs requiring independent access.
- D incorrect: Double-pumped SRAM operates at 2x the logic clock, providing 2x bandwidth — this
  is a technique to achieve higher bandwidth, not a requirement specification. The answer should
  state the bandwidth requirement, not the implementation technique.

### Q18 — Correct: B

With 16 banks using round-robin mapping: element at address A maps to bank A mod 16. Elements
with stride 16 have addresses 0, 16, 32, 48, ... All of these are divisible by 16, so all map to
bank 0. Every access in the stride-16 pattern hits bank 0: 100% conflict rate.

- A incorrect: Stride-16 accesses across 16 banks do not distribute evenly — they all land on
  the same bank (bank A mod 16 = 0 for all multiples of 16).
- C incorrect: 50% conflict would imply only every other access conflicts; with stride-16 all
  accesses to the same bank, the conflict rate is 100%.
- D incorrect: 1/16 = 6.25% would be the conflict rate if accesses were uniformly random across
  banks, but stride-16 is a highly structured (worst-case) pattern, not random.

### Q19 — Correct: B

XOR-based scrambling computes the bank index as: `bank = (addr XOR (addr >> log2(num_banks))) mod num_banks`.
This introduces a folding effect that breaks the regular mapping of power-of-two strides: with 16
banks, address 16k maps to bank (0 XOR k) mod 16 = k mod 16. A stride-16 pattern across 16 banks with XOR scrambling maps
to all 16 banks.

- A incorrect: Swapping row and column indices is a transpose, which is a valid technique for
  matrix transposition but is not called "XOR scrambling" and does not directly address
  power-of-two stride conflicts.
- C incorrect: A PRNG-based address randomisation would require storing the mapping or replaying
  the PRNG sequence for each read, which makes address calculation unpredictable and hard to
  pipeline. It is not a practical hardware technique.
- D incorrect: Interleaving odd/even addresses is 2-way bank interleaving, which only resolves
  stride-2 conflicts. For stride-16 conflicts in a 16-bank system, this is completely ineffective.

### Q20 — Correct: B

Q has shape [4096, 64] in FP16: 4096 * 64 * 2 = 524,288 bytes = 512 KB.
K has the same shape: 512 KB.
Total for Q + K = 1 MB.

The question asks for reads during the score computation S = Q * K^T, which requires exactly one
read of Q and one read of K.

- A incorrect: Both Q and K must be read. Q is not stored in registers across the entire 4096 x
  4096 score computation — the compute is tiled and both operands are re-read from SRAM per tile.
- C incorrect: 4 MB would include reading V and possibly the score matrix S. The question asks
  specifically about the reads for S = Q * K^T, which involves only Q and K.
- D incorrect: 2 MB would correspond to Q + K + V (3 x 512 KB = 1.5 MB, not 2 MB) or some other
  miscalculation. The score computation specifically requires Q and K only.
