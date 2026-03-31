# Dataflow Architectures for LLM Accelerators

## Overview

Dataflow architecture refers to the policy that determines which operand (weights, inputs, or
outputs/partial sums) stays stationary in the register file while the other operands flow through
the compute array. The choice of dataflow fundamentally determines on-chip memory requirements,
data reuse opportunities, and off-chip bandwidth pressure. For LLM workloads — dominated by
large matrix multiplications and attention — the right dataflow choice can mean an order-of-magnitude
difference in energy efficiency.

---

## Tier 1 — Fundamentals

### Q1. What is a dataflow in the context of a DNN accelerator, and why does it matter?

**Question:** Define "dataflow" as used in hardware accelerator design. Why is it not sufficient to
simply maximise compute throughput without considering dataflow?

**Answer:**

In accelerator design, **dataflow** describes the schedule of data movement and reuse across a
spatial array of processing elements (PEs). Specifically, it determines:

1. Which tensor (weights W, activations A, or partial sums/outputs O) is held stationary in each
   PE's local storage across multiple MAC operations.
2. The order and direction in which the other tensors are broadcast or multicast across the array.
3. The mapping of loop iterations (over output rows, output columns, input channels, etc.) to either
   time (sequential) or space (parallel PEs).

**Why compute throughput alone is insufficient:**

A systolic array running at 100% PE utilisation still wastes energy and bandwidth if every operand
must be fetched from off-chip DRAM for every MAC. Because DRAM access consumes roughly 100-200x
more energy than a MAC operation on modern process nodes, the data reuse pattern — not raw TOPS —
determines real-world efficiency. A design with 50% PE utilisation but 10x better data reuse can
easily be more efficient end-to-end.

**Key insight:** The dataflow determines the *reuse distance* for each tensor. Shorter reuse
distance means the data is reused while it is still in fast, low-energy on-chip storage.

---

### Q2. Describe the three classical stationary dataflows.

**Question:** Explain weight-stationary (WS), output-stationary (OS), and input-stationary (IS)
dataflows. For each, state what is kept in PE registers, what flows, and what type of workload it
favours.

**Answer:**

**Weight-Stationary (WS)**

- What is stationary: Filter weights are pre-loaded into each PE's register file and held for the
  entire duration of the convolution or matrix multiply for that tile.
- What flows: Input activations and partial sums move through the array (inputs broadcast across a
  row, partial sums accumulated as they traverse columns, or vice versa).
- Energy optimisation: Eliminates repeated off-chip reads of weights. Weight reads from the global
  buffer happen only once per tile.
- Best suited for: Batch size = 1 inference (single-stream), or any scenario where weight reuse
  across the batch is low but reuse within a single computation (depth of the inner product) is
  high. Also good when the weight matrix is small enough to tile into the PE array.
- Example hardware: Google TPU v1 uses a weight-stationary systolic dataflow.

**Output-Stationary (OS)**

- What is stationary: Each PE accumulates a single output element (partial sum) across all
  contributing input-weight pairs before writing it out.
- What flows: Both weights and inputs are streamed into the PE; the output stays and grows.
- Energy optimisation: Eliminates partial-sum read/write traffic to on-chip buffers. Every addition
  happens in a register, never requiring a round-trip to SRAM.
- Best suited for: Workloads with large output tensors and limited on-chip storage for partial sums.
  Common in convolution where the output feature map is large.
- Example hardware: NVIDIA Tensor Cores can be configured to behave output-stationarily.

**Input-Stationary (IS) — also called Activation-Stationary**

- What is stationary: Input activations (or a slice of them) are held in PE registers.
- What flows: Different weight filters pass through while the same input is reused.
- Energy optimisation: Eliminates repeated reads of the input tensor when many output channels are
  computed from the same input.
- Best suited for: Depthwise convolutions, or fully-connected layers with large batches where the
  same activation vector is multiplied by many weight rows simultaneously.

**Quick comparison table:**

| Dataflow | Stationary | Flows           | Minimises              | Weak at                      |
|----------|-----------|-----------------|------------------------|------------------------------|
| WS       | Weights   | Inputs, outputs | Weight bandwidth       | Large batch, many output ch. |
| OS       | Outputs   | Weights, inputs | Partial sum bandwidth  | Small on-chip memory         |
| IS       | Inputs    | Weights, outputs| Input bandwidth        | Varied weight shapes         |

---

### Q3. What is a systolic array and how does it implement dataflow?

**Question:** Explain how a systolic array achieves data reuse through its physical structure. How
does data movement differ from a vector processor?

**Answer:**

A **systolic array** is a 2D grid of PEs where data flows rhythmically from PE to PE in a pipelined
fashion — like blood pulsing through vessels (hence "systolic"). Each PE:

1. Receives data from its left/top neighbour.
2. Performs a MAC (multiply-accumulate).
3. Passes data to its right/bottom neighbour on the next clock cycle.

**How it implements dataflow:**

In a weight-stationary systolic array (e.g., TPU):
- Weights are preloaded diagonally into PE registers.
- Input activations are streamed left-to-right along rows.
- Partial sums accumulate top-to-bottom along columns.

Each weight is used once per input vector passing through, but since it is resident in a register
it requires no additional memory access. Each input value is reused by every row of PEs it passes
through, amortising the cost of that read.

**Difference from a vector processor:**

A vector processor fetches operand vectors from a central register file, broadcasts them to a
functional unit array, and writes results back — all data must pass through the central register
file for every operation. A systolic array eliminates this central bottleneck: data flows directly
between PEs, so the register file bandwidth requirement scales only with the array boundary, not
with the number of PEs. This is why systolic arrays can achieve very high arithmetic intensity
without creating a bandwidth wall at the register file.

---

### Q4. Define "temporal" vs "spatial" architectures.

**Question:** What is the distinction between temporal and spatial architectures? Give a hardware
example of each and explain which LLM operations benefit from each.

**Answer:**

**Temporal Architecture:**
- A single (or small number of) compute unit(s) execute operations sequentially over time.
- The same hardware resource is reused for each operation step.
- Examples: GPU shader cores, CPU SIMD units, simple systolic arrays with software-controlled
  scheduling.
- The programmer/compiler schedules when each piece of data flows through the compute unit.
- Flexibility: high (any operation order is possible with software).
- PE utilisation challenge: Requires careful pipelining to keep the units busy.

**Spatial Architecture:**
- Different operations are mapped to *physically different* PEs simultaneously.
- A dataflow graph is unrolled across the PE array in space rather than time.
- Examples: FPGAs with streaming pipelines, dataflow chips (Cerebras, Groq), Tenstorrent's mesh
  architecture.
- The compiler maps each node in the compute graph to a dedicated region of the chip.
- On-chip SRAM between stages acts as FIFOs; data flows continuously without going off-chip.
- Flexibility: lower (reconfiguration is required to change the operation).

**LLM relevance:**

- Attention score computation (Q*K^T) involves irregular tensor contractions and sequence-length
  variability. Temporal architectures handle the variable shapes more gracefully.
- Feed-forward (FFN) layers are large, regular GEMMs. Spatial architectures can dedicate fixed PE
  regions to each layer of the FFN and pipeline tokens through the entire FFN without any off-chip
  traffic — achieving very high throughput for long-running generative tasks.
- Most commercial LLM accelerators are hybrid: a spatial systolic array for the GEMM core,
  surrounded by temporally-scheduled reduction and softmax units.

---

## Tier 2 — Intermediate

### Q5. When is weight-stationary suboptimal for LLM inference?

**Question:** Weight-stationary dataflow is often cited as ideal for inference. Describe two
concrete scenarios in LLM inference where WS is a poor choice, and explain why.

**Answer:**

**Scenario 1: Large-batch inference with weight matrices that fit on-chip**

In WS, each weight is read once and then the same activation batch is streamed through. The weight
reuse factor equals the batch size. However, if you have a very large batch (e.g., B=512 for
throughput-optimised serving), the *input activation* tensor becomes enormous — and there is no
mechanism in pure WS to reuse inputs. Each row of activations must be fetched from the global
buffer for every weight tile. An output-stationary or row-stationary dataflow reuses both weights
and inputs simultaneously and would be more efficient.

**Scenario 2: Decode-phase autoregressive generation (batch=1 or small batch)**

During the decode phase, each forward pass processes only the single new token. The GEMM degenerates
to a matrix-vector multiplication (GEMV): the input is a single vector, not a matrix. In this case:

- Weight reuse is exactly 1 (each weight is used exactly once per decode step).
- There is no activation reuse regardless of dataflow choice.
- The operation is memory-bandwidth-bound, not compute-bound.

In this regime the dataflow choice is nearly irrelevant to energy efficiency — the bottleneck is
how fast you can stream weights from HBM/DRAM, not how well you reuse them. Optimisations shift
toward weight quantisation, compression, and memory bandwidth rather than dataflow strategy.

**Takeaway:** WS is most beneficial when the batch dimension provides many activation vectors to
multiply against the same weight matrix. Single-token decode phases eliminate this advantage
entirely.

---

### Q6. Explain row-stationary dataflow and why it was important.

**Question:** The MIT Eyeriss paper introduced "row-stationary" dataflow. Explain the concept and
why it achieves better energy efficiency than any single pure dataflow (WS, OS, or IS) for
convolutions. Does this concept translate to transformer attention?

**Answer:**

**Row-Stationary Dataflow (Eyeriss):**

Row-stationary keeps an entire *row* of a filter sliding in a PE across time, while one row of the
input feature map is also kept stationary. Each PE computes a 1D convolution between one filter
row and one input row. Partial sums from adjacent PEs (which compute adjacent filter rows) are
summed via a local reduction network.

This simultaneously achieves:
- Weight reuse: The same filter row is used for multiple input positions (horizontal sliding).
- Input reuse: The same input row is reused for multiple filter rows (from adjacent PEs).
- Partial sum reuse: Partial results accumulate locally across filter rows without going to SRAM.

**Why it beats pure dataflows for convolutions:**

Pure WS eliminates weight bandwidth but not partial sum bandwidth.
Pure OS eliminates partial sum bandwidth but not weight bandwidth.
Pure IS eliminates input bandwidth but not weight bandwidth.
Row-stationary minimises the *total energy* across all three memory levels simultaneously by
partitioning reuse appropriately. Eyeriss showed ~1.4x reduction in total energy vs. the best
pure dataflow for AlexNet-era convolutions.

**Applicability to transformer attention:**

Attention computes:
```
Attention(Q,K,V) = softmax(Q * K^T / sqrt(d_k)) * V
```

The Q*K^T step is a batched matrix multiply with no sliding-window structure — there is no "row"
that slides over a filter. Row-stationary does not map cleanly because there is no spatial reuse
locality analogous to convolutional filter sliding.

For attention, FlashAttention-style tiling (which is an output-stationary approach over Q tiles,
while streaming K/V) has emerged as the practical answer. The concept of minimising SRAM
round-trips survives, but the specific 1D convolution row structure does not apply.

---

### Q7. How do you compute the arithmetic intensity of a GEMM and use it to choose a dataflow?

**Question:** Given a matrix multiplication C = A * B where A is (M x K) and B is (K x N),
compute the arithmetic intensity. Then explain how this number should inform your dataflow
and memory hierarchy design.

**Answer:**

**Arithmetic Intensity (AI) definition:**

```
AI = Total FLOPs / Total bytes moved from off-chip DRAM
```

For C = A * B:
- FLOPs = 2 * M * K * N  (multiply + add per element)
- Bytes (assuming float16, 2 bytes per element):
  - Read A: 2 * M * K bytes
  - Read B: 2 * K * N bytes
  - Write C: 2 * M * N bytes
  - Total bytes = 2 * (M*K + K*N + M*N)

```
AI = (2 * M * K * N) / (2 * (M*K + K*N + M*N))
   = (M * K * N) / (M*K + K*N + M*N)
```

**For a large square matrix (M = K = N = 4096, typical FFN projection):**
```
AI = (4096^3) / (3 * 4096^2) = 4096 / 3 ≈ 1365 FLOPs/byte
```

This is very high — the operation is strongly compute-bound, so any dataflow that keeps data on
chip will do.

**For decode-phase GEMV (M = 1, K = 4096, N = 4096):**
```
AI = (1 * 4096 * 4096) / (1*4096 + 4096*4096 + 1*4096)
   ≈ (4096^2) / (4096^2) = 1 FLOP/byte
```

This is extremely memory-bound. The operation will be limited by HBM bandwidth no matter what.

**Design implications:**

| AI vs hardware roof | Implication | Dataflow priority |
|---|---|---|
| AI >> HBM_BW / Peak_FLOPS | Compute-bound | Any dataflow works; focus on PE utilisation |
| AI ≈ HBM_BW / Peak_FLOPS | Balanced | Match on-chip capacity to tile B matrix (WS often optimal) |
| AI << HBM_BW / Peak_FLOPS | Memory-bound | Reduce weight precision; streaming + compression > dataflow |

For LLM prefill (large M), optimise dataflow for PE utilisation and partial-sum accumulation.
For LLM decode (M=1), optimise HBM bandwidth and weight compression — dataflow is secondary.

---

### Q8. Compare systolic array dataflow with a mesh-of-PEs dataflow architecture.

**Question:** A systolic array and a 2D mesh architecture both tile a PE array, but they differ in
connectivity, dataflow flexibility, and programming model. Compare them across three dimensions:
data movement, programmability, and suitability for irregular LLM operations (e.g., sparse
attention).

**Answer:**

**Data Movement:**

*Systolic array:* Data flows in a fixed, pipelined direction (e.g., inputs left-to-right, weights
top-to-bottom). Each PE has only nearest-neighbour connections in a predetermined pattern.
Communication is implicit and scheduled by the systolic rhythm — no explicit routing decisions are
made at runtime. This makes the interconnect extremely area- and power-efficient.

*2D Mesh (e.g., Tenstorrent, Cerebras):* Each PE has a network-on-chip (NoC) router and can send
packets to any other PE. Data routing is explicit and configurable. This supports arbitrary dataflow
graphs where different tiles hold different layers or operators.

**Programmability:**

*Systolic array:* Highly inflexible — the data movement pattern is essentially hardwired or
constrained to a small number of modes. The compiler must transform every operation into the fixed
GEMM dataflow or execute it on a separate unit. Adding a new operation (e.g., RMSNorm) requires
separate scalar/vector hardware outside the array.

*2D Mesh:* Flexible — any operator can be placed anywhere. Streaming pipelines can be built that
span many PEs without revisiting DRAM. Supports heterogeneous tile types (attention tiles, FFN
tiles, normalisation tiles) all connected.

**Sparse / Irregular Operations (Sparse Attention):**

*Systolic array:* Sparse attention patterns break the regular data feed required for the systolic
rhythm. Injecting zeros for skipped attention positions wastes cycles. Some designs add a sparse
pre-processing unit that compresses sparse blocks before feeding them in, but this adds complexity.

*2D Mesh:* Can route only non-zero blocks to active PEs. Individual PE tiles can be idle while
others process non-zero blocks. More natural fit for structured sparsity (block-sparse patterns)
since routing is dynamic. However, the NoC power overhead can negate this advantage if sparsity
is low.

**Bottom line:** Systolic arrays win on PPA for regular dense GEMMs. Mesh architectures win on
flexibility and are better positioned for the evolving, irregular dataflow patterns of future
LLM variants (sparse attention, mixture-of-experts routing, speculative decoding).

---

## Tier 3 — Advanced

### Q9. Design a hybrid dataflow for an LLM accelerator handling both prefill and decode phases.

**Question:** Prefill (prompt processing) and decode (token generation) have fundamentally different
arithmetic intensities and memory access patterns. Design a dataflow strategy — including on-chip
memory allocation, PE mapping, and off-chip bandwidth management — that serves both phases well
without duplicating hardware.

**Answer:**

**Phase characterisation:**

| Phase   | GEMM shape (typical) | AI          | Bottleneck     |
|---------|---------------------|-------------|----------------|
| Prefill | (T x d) x (d x d_ff) | ~1000 F/B  | Compute        |
| Decode  | (1 x d) x (d x d_ff) | ~1 F/B     | HBM bandwidth  |

**Proposed hybrid dataflow:**

**1. Unified weight-stationary inner core with dynamic M-dimension batching:**

Maintain a fixed PE array (e.g., 256x256 systolic) always configured weight-stationary. During
prefill, feed full token matrices (large M) — the array runs at high utilisation. During decode,
batch multiple decode requests together (continuous batching) to increase effective M and raise AI
above the memory-bandwidth cliff. Target effective batch size such that M*K*N / (K*N) ≈ M >= 16,
which raises AI above ~16 F/B, approaching the hardware balance point.

**2. On-chip memory allocation:**

```
Total on-chip SRAM: 32 MB (example)
- 16 MB: Weight tile buffer (holds one full FFN weight tile — stationary during a pass)
-  8 MB: KV-cache scratchpad (for sequence lengths up to ~2K tokens in fast SRAM)
-  4 MB: Activation double-buffer (ping-pong: one tile computing, one tile DMAing)
-  4 MB: Partial sum accumulator bank (output-stationary accumulation within a tile)
```

**3. Off-chip bandwidth management:**

- Prefill: Weight tiles stream in once per layer while activations are reused across the systolic
  array. HBM is read sequentially in large bursts (optimise for bandwidth not latency).
- Decode: Weights stream at maximum HBM bandwidth — no reuse is possible for single-token.
  Apply weight quantisation (INT4/FP8) to reduce effective bytes fetched per weight, multiplying
  effective bandwidth by 2-4x. Use fused de-quantisation inside PEs.

**4. Attention handling:**

Attention does not fit the GEMM core's weight-stationary model cleanly (KV cache entries are not
"weights"). Dedicate a separate 64x64 attention PE sub-array operating output-stationarily,
accumulating softmax-normalised partial sums. This runs concurrently with the FFN sub-array
operating on different layers, hiding attention latency behind FFN compute.

**5. Mode switching overhead:**

Keep the dataflow mode switch implicit (no PE reconfiguration): the compiler changes only which
SRAM banks are addressed as "weight" vs "activation" buffers by changing DMA source addresses
and buffer ping-pong assignments. The systolic array hardware never changes mode — only the data
feeding it changes.

---

### Q10. Analyse the dataflow implications of FlashAttention.

**Question:** FlashAttention reorders the attention computation to avoid materialising the full
N x N attention matrix. Describe this reordering from a hardware dataflow perspective: what is
stationary, what streams, and what memory hierarchy levels are required. Compare the on-chip
bandwidth requirements to naive attention.

**Answer:**

**Naive attention dataflow:**

```
S = Q * K^T          -- writes (N x N) to HBM
P = softmax(S)       -- reads+writes (N x N) from/to HBM
O = P * V            -- reads (N x N) from HBM
```

For sequence length N=8192, d_model=128, FP16:
- N x N matrix = 8192^2 * 2 bytes = 128 MB per head per layer written to HBM.
- Total HBM traffic for attention: ~3 * 128 MB * num_heads = enormous.

**FlashAttention dataflow:**

FlashAttention tiles the Q matrix into blocks of size B_r (output-stationary over Q tiles).
For each Q tile, it streams through the entire K and V matrices once, accumulating running softmax
statistics (row-max m, row-sum l) in registers.

```
For each Q tile q_i (B_r x d):
    m_i = -inf, l_i = 0, O_i = 0           -- held in on-chip registers
    For each K/V tile k_j, v_j (B_c x d):  -- streamed from on-chip SRAM
        s_ij = q_i * k_j^T                  -- (B_r x B_c), computed and immediately consumed
        m_new = max(m_i, rowmax(s_ij))
        rescale O_i by exp(m_i - m_new)
        O_i += exp(s_ij - m_new) * v_j
        update l_i
    Write O_i to HBM                        -- one write per Q tile
```

**What is stationary:** Q tile (q_i) and running statistics (m_i, l_i, O_i) remain in registers
(or a small register-file-level buffer) for the entire inner loop.

**What streams:** K and V tiles stream from on-chip SRAM (loaded once from HBM, reused across all
Q tiles processed by this PE group). Within the inner loop, s_ij is ephemeral — computed, used for
updates, and discarded without SRAM writes.

**Memory hierarchy requirement:**

On-chip SRAM must hold:
- One Q tile: B_r * d * 2 bytes
- One K tile + one V tile: 2 * B_c * d * 2 bytes
- Must satisfy: B_r * d + 2 * B_c * d <= SRAM_capacity / 2 (leave room for double-buffering)

For d=128, B_r=B_c=64 (typical): (64 + 128) * 128 * 2 = 49,152 bytes ≈ 48 KB per head.
This easily fits in a typical 256 KB per-PE-cluster SRAM.

**HBM traffic comparison:**

| Approach       | HBM reads       | HBM writes    | Total (N=8192, d=128) |
|----------------|-----------------|---------------|-----------------------|
| Naive          | Q+K+V+S+P = 5*N*d + 2*N^2 | S+P+O | ~600 MB / head |
| FlashAttention | Q+K+V once      | O once        | ~3*N*d = ~6 MB / head |

FlashAttention reduces HBM traffic by ~100x for long sequences — this is purely a consequence of
choosing the right dataflow (output-stationary over Q, streaming K/V) and matching tile sizes to
on-chip SRAM capacity.

**Hardware implication:** FlashAttention requires a scratchpad-style on-chip memory (software-managed
SRAM), not a hardware cache. A cache would not guarantee that the K/V tiles remain on-chip for the
entire inner loop duration — they could be evicted between iterations, destroying the bandwidth
saving. This is one reason why LLM accelerators favour scratchpads over caches for the attention
sub-unit.

---

### Q11. How does dataflow interact with sparsity exploitation in large language models?

**Question:** Activation sparsity (ReLU, top-k in MoE) and weight sparsity are both present in
modern LLMs. Explain how a weight-stationary systolic dataflow must be modified to exploit
activation sparsity without degrading throughput on dense workloads. What hardware mechanisms are
required?

**Answer:**

**The problem with weight-stationary + activation sparsity:**

In standard WS, inputs are streamed through the array at a fixed rhythm. If many inputs are zero
(activation sparsity), the corresponding MACs produce zero contributions — but the PE still
executes the multiply and the result is added to the partial sum. No cycles are saved unless the
hardware can *skip* zero inputs.

A naive WS array has no mechanism to skip: the systolic rhythm requires data to arrive at
predictable cycles for correct accumulation. Inserting a "skip" would desynchronise the pipeline.

**Required hardware modifications:**

**1. Compressed input queue with a zero-skip unit:**

Before feeding activations into the array rows, insert a *compressor* that:
- Scans the input vector for non-zero elements.
- Outputs (value, column_index) pairs, removing zeros.
- Buffers compressed inputs in a FIFO.

The systolic array is then fed only non-zero (value, column_index) pairs. Each PE uses the
column_index to select the corresponding weight from its local register file rather than always
using a sequentially-arriving weight.

This converts the weight-stationary array from fixed-stride weight access to indexed weight
access — requiring either a local multiplexer (for small weight tiles) or a small associative
buffer per PE.

**2. Load-balanced PE assignment:**

Activation sparsity is typically non-uniform. If PE rows receive different numbers of non-zeros,
some finish early while others are still computing — destroying PE utilisation. A load balancer
upstream distributes non-zero work evenly across rows.

**3. Synchronisation and output merging:**

Partial sums from PEs that processed different subsets of non-zeros for the same output neuron must
be merged. This requires a reduction tree at the array output that was not needed for dense WS.

**4. Dense fallback mode:**

If the sparsity ratio drops below a threshold (typically < 50% activation sparsity), the overhead
of compression, indexing, and merging exceeds the cycles saved. Hardware must support a bypass mode
that switches back to standard dense WS operation. The threshold and switching are managed by
runtime profiling or static compiler annotation.

**Quantitative break-even:**

If compression overhead = C cycles per vector and the non-zero fraction = s:
```
Speedup = 1 / (s + C/N_elements)
```
For C/N_elements = 0.05 (5% overhead), break-even occurs at s = 0.95 — you need > 95% sparsity
before even a modest compression overhead pays off. Real ReLU activations in LLMs achieve ~50-70%
sparsity, which yields only modest gains (1.4-2x) and requires very low compression overhead
circuitry to remain beneficial.

**MoE gating (structured sparsity):**

Mixture-of-Experts uses top-k routing to activate only k of E expert FFNs. This is *structured*
sparsity — entire weight matrices are skipped, not individual weights. This maps cleanly to
tile-level skipping: the DMA simply does not load weight tiles for inactive experts, and the
systolic array is reused for the k active experts sequentially. No modification to the inner
systolic dataflow is needed — only the tile scheduling and DMA fetch list change.
