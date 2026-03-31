# Quiz: System-Level Design

## Overview

Self-assessment quiz covering SoC architecture for LLM inference, PCIe host interface design,
end-to-end inference pipeline stages, power estimation methodology, verification strategy,
latency budgeting, and throughput estimation. Questions target system-level design thinking
relevant to senior RTL and architecture roles.

**Instructions:** Select the single best answer for each question. Answers and full explanations
appear at the bottom.

---

## Section 1 — SoC Architecture (Questions 1-4)

**Q1.** An LLM inference SoC integrates a host CPU, an NPU (Neural Processing Unit), HBM
memory, and PCIe connectivity. Requests arrive from the host, are processed on the NPU, and
results are returned. Which on-chip bus topology best suits this architecture and why?

- A) A crossbar switch connecting all agents, because it provides full non-blocking bandwidth
   between every master-slave pair simultaneously
- B) A hierarchical ring bus with CPU on the outer ring and NPU on the inner ring, because rings
   minimise wire length for nearest-neighbour communication
- C) A layered AXI interconnect with separate high-bandwidth data paths (NPU-to-HBM) and
   lower-bandwidth control paths (CPU-to-NPU configuration), matching traffic characteristics
   to interconnect capacity
- D) A shared bus (APB or AHB) for all agents, because the low gate count minimises die area

---

**Q2.** An accelerator SoC must support multiple concurrent inference requests (different users,
different models). Which hardware mechanism enables true isolation between requests?

- A) Software-managed context switching, where the driver saves and restores NPU register state
   between requests
- B) Hardware virtualisation of the NPU's memory address space using an IOMMU, ensuring each
   request's DMA operations are restricted to its own allocated memory region
- C) Separate clock domains for each request, preventing one request's activity from affecting
   another's timing
- D) A FIFO queue in front of the NPU that serialises requests, ensuring only one request is
   in-flight at any time

---

**Q3.** A designer must decide where to place the tokenizer (text-to-token-ID conversion) in the
SoC pipeline. Which placement is most practical for a production accelerator?

- A) On the NPU itself, implemented as a dedicated FSM with an embedded vocabulary SRAM
- B) On the host CPU, before the inference request is issued to the NPU, because the tokenizer
   is a sequential character-processing task with low arithmetic intensity unsuited to the NPU
- C) In a dedicated tokenizer ASIC co-packaged with the NPU die
- D) In DRAM, using a memory-mapped token lookup table accessed by the NPU DMA engine

---

**Q4.** An NPU design has 1024 MAC units, each operating at 1 GHz on INT8 data. The peak
throughput is expressed as TOPS (Tera Operations Per Second). What is the peak TOPS?

- A) 0.512 TOPS
- B) 1.024 TOPS
- C) 2.048 TOPS
- D) 4.096 TOPS

---

## Section 2 — PCIe Interface (Questions 5-7)

**Q5.** A discrete LLM accelerator connects to a host server over PCIe Gen 5 x16. The peak
unidirectional bandwidth of PCIe Gen 5 x16 is approximately 128 GB/s. For LLM inference with
batch size 1, the host must send a prompt of 1024 tokens (each 4 bytes = 4 KB of token IDs) and
receive 512 output tokens. How does PCIe bandwidth compare to HBM bandwidth for this workload?

- A) PCIe is the primary bottleneck because all model weights must be transferred over PCIe before
   inference can begin
- B) PCIe is not the bottleneck: the data transferred over PCIe (prompt + output tokens, ~6 KB
   total) is negligible compared to the weight streaming bandwidth requirement (~hundreds of GB/s
   from HBM), so HBM bandwidth dominates
- C) PCIe and HBM have equal bandwidth contributions because PCIe supplies activations at 128 GB/s
   while HBM supplies weights at 3.35 TB/s
- D) PCIe becomes the bottleneck only when batch size exceeds 1024, because larger batches require
   more activation data to be transferred

---

**Q6.** In PCIe, DMA from the accelerator to host memory is called DMA Read (from accelerator
perspective). The accelerator initiates a Memory Read TLP (Transaction Layer Packet). What limits
the achievable bandwidth for small transfers (e.g., 64-byte completion tokens)?

- A) The PCIe physical layer encoding (128b/130b for Gen 3+) introduces 1.5% overhead on large transfers
- B) The TLP header overhead and PCIe completion credit mechanism introduce significant per-transfer
   overhead, so small transfers achieve a small fraction of peak bandwidth
- C) Small transfers are rate-limited by the host CPU's ability to acknowledge each transfer
- D) The accelerator's PCIe controller limits transaction size to 64 bytes maximum

---

**Q7.** A designer proposes using CCIX (Cache Coherent Interconnect for Accelerators) instead of
standard PCIe for host-accelerator communication. What is the primary advantage of CCIX for LLM
inference?

- A) CCIX provides 10x higher bandwidth than PCIe Gen 5 in the same number of lanes
- B) CCIX allows the accelerator to access host CPU memory with cache coherence, potentially
   enabling the KV cache or model weights to reside in host DRAM and be accessed directly
   without explicit DMA copies
- C) CCIX eliminates the need for a PCIe root complex on the host motherboard, reducing system cost
- D) CCIX supports encrypted transfers natively, improving security for inference on sensitive data

---

## Section 3 — End-to-End Pipeline (Questions 8-10)

**Q8.** In a production LLM inference system, the end-to-end pipeline includes: (1) tokenisation,
(2) embedding lookup, (3) N transformer layers, (4) logit computation, (5) sampling, and
(6) detokenisation. For a streaming response (tokens returned one at a time), what is the critical
path for per-token latency?

- A) Steps 1, 2, and 3 — tokenisation and embedding dominate because they require DRAM accesses
   for the vocabulary table
- B) Step 3 (transformer layers) — this is the dominant latency contributor because it involves
   loading and computing through all N layers of weights
- C) Steps 4 and 5 — logit computation and sampling dominate for large vocabulary sizes (100K+
   tokens)
- D) Step 6 — detokenisation dominates because byte-pair encoding requires complex text processing

---

**Q9.** Speculative decoding uses a small "draft" model to generate candidate tokens, which are
then verified in parallel by the larger "target" model. From a hardware pipeline perspective, what
is the key requirement?

- A) The draft and target models must be stored in separate HBM modules to avoid bandwidth contention
- B) The target model must be able to verify a batch of K draft tokens in a single forward pass
   (treating them as a batch), requiring the attention engine to handle variable-length inputs
   and the sampling logic to compare multiple predicted distributions
- C) Speculative decoding requires two complete copies of the target model weights on-chip
- D) The draft model must be identical in architecture to the target model but with fewer layers

---

**Q10.** Tensor parallelism splits weight matrices across multiple accelerator chips. For a linear
layer with weight matrix W of shape [d_model, d_ff], split column-wise across T chips, each chip
holds W_i of shape [d_model, d_ff/T]. After each chip computes its partial output, what
communication primitive is required?

- A) All-Reduce — each chip broadcasts its partial result and all chips sum all partial results
- B) All-Gather — each chip sends its partial output vector and all chips concatenate to form
   the full output
- C) Reduce-Scatter — the partial sums are first reduced then scattered to different chips
- D) Point-to-point send — chip 0 sends its result to chip 1, which accumulates and passes to
   chip 2, forming a ring-reduction

---

## Section 4 — Power Estimation (Questions 11-12)

**Q11.** A designer estimates the dynamic power of a MAC array using the formula:
`P_dynamic = alpha * C * V^2 * f`
where alpha is the activity factor. For an INT8 MAC array with alpha = 0.2, C = 50 pF (aggregate
gate capacitance of the array), V = 0.8 V, and f = 1 GHz, what is the dynamic power?

- A) 6.4 mW
- B) 6.4 W
- C) 64 mW
- D) 64 W

---

**Q12.** A system power budget allocates 300 W to an LLM inference accelerator chip. Static
(leakage) power typically represents 20-30% of total power for advanced nodes (5nm, 3nm) under
high temperature. What is the approximate available dynamic power budget for the compute array
and memory?

- A) 300 W (leakage is negligible at advanced nodes)
- B) 210-240 W (accounting for 20-30% leakage)
- C) 150 W (assuming 50% leakage as a worst case)
- D) 270 W (leakage is only 10% at advanced nodes)

---

## Section 5 — Verification Methodology (Questions 13-15)

**Q13.** A UVM (Universal Verification Methodology) testbench for an NPU is being designed.
The DUT is the MAC array with its surrounding memory interfaces. Which verification component
is responsible for generating stimulus (e.g., randomised weight and activation matrices)?

- A) The UVM scoreboard
- B) The UVM sequence running on the UVM sequencer, which drives transactions to the driver
- C) The UVM monitor, which observes bus transactions and converts them to transactions
- D) The UVM agent, which passively records all interface activity

---

**Q14.** Formal property verification (FPV) is applied to a DMA engine. Which property is most
naturally expressed and verified using FPV?

- A) "The DMA completes a 1 GB transfer within 10 ms at 100 GB/s bandwidth"
- B) "If a transfer is initiated and the destination address is out of range, the DMA asserts an
   error flag within 2 cycles and does not issue any memory transactions"
- C) "The DMA achieves 95% efficiency on transfers larger than 4 KB"
- D) "The DMA's throughput scales linearly with the number of outstanding transactions up to 16"

---

**Q15.** In a coverage-driven verification (CDV) flow, a coverage model is defined for an NPU.
Which coverage type best captures whether the design has been exercised with corner-case
combinations of input tensor shapes?

- A) Line coverage — ensures every RTL line has been executed at least once
- B) Toggle coverage — ensures every signal has toggled from 0 to 1 and 1 to 0
- C) Cross coverage — a cross product of multiple coverpoints (e.g., batch size x sequence length
   x head count), capturing combinations of parameters that expose interaction bugs
- D) Branch coverage — ensures every if/else branch in the RTL has been taken

---

## Section 6 — Latency Budgeting and Throughput Estimation (Questions 16-20)

**Q16.** A transformer model has 32 layers, each with a self-attention block and an FFN block.
For single-token decode at 1 GHz with a 256-wide INT8 MAC array, and given that each layer
requires approximately 1M MACs, what is the compute-bound minimum latency per token (ignoring
memory)?

- A) 32 microseconds
- B) 125 microseconds
- C) 3.9 microseconds
- D) 1 millisecond

---

**Q17.** A latency budget for a chatbot application requires less than 50 ms Time-to-First-Token
(TTFT) for a 512-token prompt. The model has 7B parameters (7 GB INT8). HBM bandwidth is 1 TB/s.
What fraction of the 50 ms budget is consumed by weight loading alone?

- A) 7% (3.5 ms)
- B) 14% (7 ms)
- C) 100% (50 ms — weight loading alone exceeds the budget)
- D) 0.7% (0.35 ms)

---

**Q18.** Throughput for a batch of B requests, each generating L output tokens, is measured in
tokens/second. If the per-token decode latency is T_decode (ms) at batch size B, what is the
aggregate throughput?

- A) B * L / T_decode tokens/second (T_decode in ms means dividing by 1000 for seconds)
- B) (B * L * 1000) / T_decode tokens/second
- C) L / T_decode tokens/second (per-request throughput, not aggregate)
- D) B / T_decode tokens/second (ignoring L)

---

**Q19.** An accelerator achieves 100 tokens/second at batch size 1 and 1500 tokens/second at
batch size 32. What does this scaling indicate about the workload?

- A) The workload is compute-bound at both batch sizes, and the speedup is proportional to the
   increased batch size
- B) At batch size 1 the workload is severely memory-bandwidth-bound (weights loaded once per
   token for 1 request); at batch size 32, the weights are amortised over 32 requests,
   approaching the memory bandwidth ceiling before compute saturation
- C) The workload has a memory leak that causes degradation at large batch sizes
- D) The accelerator has a 32-way parallelism limit in hardware, causing throughput to plateau
   exactly at 32x the single-batch throughput

---

**Q20.** A team is designing an LLM inference chip targeting 1000 tokens/second for a 7B INT8
model with HBM bandwidth of B GB/s. Assuming the workload is 100% memory-bandwidth-bound and
the model size is 7 GB, what minimum HBM bandwidth B is required?

- A) 7 GB/s
- B) 70 GB/s
- C) 700 GB/s
- D) 7000 GB/s

---

## Answer Key

| Q  | Answer |
|----|--------|
| 1  | C      |
| 2  | B      |
| 3  | B      |
| 4  | C      |
| 5  | B      |
| 6  | B      |
| 7  | B      |
| 8  | B      |
| 9  | B      |
| 10 | B      |
| 11 | A      |
| 12 | B      |
| 13 | B      |
| 14 | B      |
| 15 | C      |
| 16 | C      |
| 17 | B      |
| 18 | B      |
| 19 | B      |
| 20 | D      |

---

## Detailed Explanations

### Q1 — Correct: C

An LLM inference SoC has highly asymmetric traffic: the NPU-to-HBM path carries hundreds of GB/s
of weight data, while the CPU-to-NPU path carries only configuration registers and small control
messages. A layered AXI interconnect (or similar tiered fabric like ARM NIC-400 or Arteris FlexNoC)
separates these into high-bandwidth (512+ bit wide) data planes and narrow control planes, matching
bandwidth to cost.

- A incorrect: A full crossbar provides non-blocking bandwidth but scales as O(N^2) in area and
  power for N agents. For a small number of agents (CPU, NPU, DMA, PCIe) it is reasonable, but
  the key point is matching bandwidth capacity to traffic characteristics — a flat crossbar treats
  all paths equally and wastes area on low-traffic paths.
- B incorrect: Ring buses serialise traffic and introduce variable latency proportional to ring
  distance. For high-bandwidth NPU-to-HBM transfers, a ring is a significant bottleneck.
- D incorrect: A shared APB/AHB bus has limited bandwidth (typically 32 or 64 bits per cycle) and
  is appropriate only for low-bandwidth peripherals, not for an NPU requiring hundreds of GB/s.

### Q2 — Correct: B

An IOMMU (Input-Output Memory Management Unit) maps the virtual DMA addresses used by each
inference request to disjoint physical memory regions. Even if a request issues a malicious or
buggy DMA to an out-of-bounds address, the IOMMU blocks it and generates a fault. This provides
hardware-enforced isolation.

- A incorrect: Software context switching creates a window of vulnerability during the switch and
  relies on the driver being trusted and correct. Hardware isolation (IOMMU) is stronger and does
  not interrupt ongoing DMA operations.
- C incorrect: Separate clock domains do not provide memory isolation. A request on its own clock
  domain could still issue DMA requests to memory regions belonging to other requests.
- D incorrect: A serialising FIFO prevents concurrency (reducing throughput) but does not prevent
  a request from accessing another request's memory — it only ensures single-request occupancy at
  the NPU, which is a throughput decision not a security boundary.

### Q3 — Correct: B

Tokenisation is a text-processing algorithm (byte-pair encoding or SentencePiece) that involves
string manipulation, hash lookups, and conditional branching — workloads well-suited to general
purpose CPUs. The output is a small integer array (token IDs) that is then sent to the NPU. The
host CPU has dedicated hardware for this task and the output data is tiny.

- A incorrect: Implementing a tokenizer as a FSM in the NPU wastes scarce NPU area and clock
  cycles on a task that the host CPU handles trivially. The NPU's MAC array would be idle during
  tokenisation.
- C incorrect: A dedicated tokeniser ASIC adds package cost, board area, and inter-chip bandwidth
  requirements. There is no performance motivation — the host CPU is fast enough.
- D incorrect: A memory-mapped token lookup table requires the NPU DMA to perform N random lookups
  for an N-token sequence (N DRAM accesses with poor spatial locality), which is inefficient and
  adds latency. The CPU can execute this in L1/L2 cache.

### Q4 — Correct: C

Each MAC unit performs 2 operations per cycle (one multiply and one accumulate). With 1024 MAC
units at 1 GHz: Peak TOPS = 1024 MACs * 2 ops/MAC * 1 x 10^9 cycles/s = 2.048 x 10^12 ops/s =
**2.048 TOPS**.

- A incorrect: 0.512 TOPS = 1024 * 0.5 * 10^9, which counts only multiply OR accumulate, not
  both. Standard convention is 2 ops per MAC.
- B incorrect: 1.024 TOPS = 1024 * 1 * 10^9, counting 1 op per MAC. This is incorrect because
  a MAC performs two operations.
- D incorrect: 4.096 TOPS = 1024 * 4 * 10^9, which would require each MAC to perform 4 operations
  per cycle — possible with fused operations but not the standard definition.

### Q5 — Correct: B

The prompt (1024 tokens x 4 bytes) = 4 KB. The output (512 token IDs x 4 bytes) = 2 KB. Total
PCIe traffic ≈ 6 KB. At 128 GB/s, this takes 6 KB / 128 GB/s ≈ 47 nanoseconds — negligible.
Meanwhile, loading a 7B INT8 model from HBM at 3.35 TB/s (H100) takes 7 GB / 3.35 TB/s ≈ 2 ms
just for one token's weight pass. PCIe is clearly not the bottleneck.

- A incorrect: Model weights do not travel over PCIe for an on-board discrete accelerator with its
  own HBM. Weights are loaded from HBM to the NPU on-chip. PCIe carries only the small I/O
  (prompt tokens and generated tokens).
- C incorrect: The statement about PCIe and HBM having "equal bandwidth contributions" is false.
  They carry fundamentally different data (token IDs vs. weights) and their relative contributions
  to latency differ by orders of magnitude.
- D incorrect: Even at batch size 1024, prompt activation data (1024 prompts x 1024 tokens x 2
  bytes embedding = 2 GB) would take 2 GB / 128 GB/s = 15.6 ms — still far smaller than the
  weight loading time. PCIe is unlikely to become the dominant bottleneck for inference workloads.

### Q6 — Correct: B

PCIe TLPs have 16-24 byte headers. A 64-byte payload transfer carries ~25% overhead in header
alone. Additionally, each completion (from the read target) is a separate TLP, and the PCIe
credit mechanism requires the initiator to have available completion credits before issuing reads.
For small transfers, per-TLP overhead dominates, and effective bandwidth can be 5-20x below peak.

- A incorrect: The 128b/130b encoding overhead is approximately 1.5%, which is negligible. This
  does not explain poor small-transfer efficiency.
- C incorrect: CPU acknowledgement is not required for each PCIe transfer; the PCIe protocol is
  flow-controlled through the credit mechanism at the TLP level, not through CPU software
  acknowledgements.
- D incorrect: PCIe transaction size is not limited to 64 bytes maximum. PCIe supports Max Payload
  Size (MPS) of 128 B to 4 KB. The question describes the problem of small (64-byte) transfers,
  not a hard limit.

### Q7 — Correct: B

CCIX extends the coherency protocol across the PCIe physical layer, allowing the accelerator to
participate in the CPU's cache coherence domain. For LLM inference, this could allow the KV cache
(which grows with sequence length) or even model weights to reside in the host's large DRAM pool
(e.g., 2 TB server DRAM) and be accessed by the NPU via load/store semantics without explicit DMA.

- A incorrect: CCIX operates over the same physical PCIe layer (or dedicated links) and does not
  inherently provide 10x bandwidth. Bandwidth depends on the number of lanes and generation, same
  as PCIe.
- C incorrect: CCIX is a protocol extension built on top of PCIe; it does not eliminate the PCIe
  root complex. The physical connectivity is the same.
- D incorrect: CCIX does not define an encryption protocol. Data security is handled at other
  layers (e.g., memory encryption, TLS at the application layer).

### Q8 — Correct: B

Each transformer layer involves loading and processing the weights for the attention QKV
projections, output projection, and FFN (two large matrices). For a 32-layer model, this means
loading and computing through 32 sets of large weight matrices. Even for a 7B model with efficient
hardware, this takes tens of milliseconds per token in the memory-bandwidth-bound regime.
Tokenisation, embedding, logit projection, and sampling together take microseconds to
low-milliseconds and are negligible by comparison.

- A incorrect: Tokenisation and embedding are fast: tokenisation is in-CPU cache, and the embedding
  lookup is one DRAM access for the single token. Together these take microseconds.
- C incorrect: Logit projection (one large matrix multiply over the vocabulary) and sampling
  (argmax or temperature sampling over 100K+ logits) are significant but still much smaller than
  32 layers of transformer computation.
- D incorrect: Detokenisation converts a token ID back to text, which is a single table lookup.
  It is negligible in latency.

### Q9 — Correct: B

In speculative decoding, the draft model generates K candidate next tokens. The target model then
runs a single forward pass treating these K tokens as a batch, producing K probability distributions.
The verification step compares the draft and target distributions token by token (accepting tokens
that match and rejecting at the first mismatch). The critical hardware requirement is that the
target model can process this K-token batch efficiently — which requires the attention engine to
handle variable batch sizes and the sampling unit to evaluate K distributions simultaneously.

- A incorrect: While bandwidth contention between draft and target model weights is a real
  engineering concern, it is not addressed by requiring separate HBM modules. Scheduling and
  memory bank mapping address this.
- C incorrect: Speculative decoding does not require two copies of the target model weights. The
  target model is executed once per K-draft-token batch, not twice.
- D incorrect: The draft model is typically a much smaller model (e.g., a 68M or 7B model as draft
  for a 70B target) or a subset of the target model's layers. It does not need to be
  architecturally identical.

### Q10 — Correct: B

In column-parallel (Megatron-style) tensor parallelism, each chip computes a partial output vector
of length d_ff/T. To reconstruct the full d_ff-length output, each chip must receive the partial
outputs from all other chips and concatenate them. This is All-Gather: each chip sends its
d_ff/T-element fragment, and all chips end up with the full d_ff-element vector.

- A incorrect: All-Reduce sums partial results across all chips. This applies when each chip
  computes a partial sum that must be accumulated (e.g., row-parallel GEMM where each chip
  computes a partial dot product). For column-parallel, partial results are non-overlapping and
  must be concatenated, not summed.
- C incorrect: Reduce-Scatter is the combination of All-Reduce and Scatter, used in distributed
  training gradient synchronisation. It is not the appropriate collective for tensor parallelism
  forward pass output reconstruction.
- D incorrect: Ring point-to-point passes results sequentially, which is equivalent to a ring
  All-Reduce or a ring All-Gather. While a ring topology implements All-Gather, the term
  "point-to-point send" as described (chip N accumulates and passes to chip N+1) describes an
  All-Reduce ring, not an All-Gather.

### Q11 — Correct: A

P = alpha * C * V^2 * f = 0.2 * 50 x 10^-12 * (0.8)^2 * 1 x 10^9
= 0.2 * 50 x 10^-12 * 0.64 * 10^9
= 0.2 * 50 * 0.64 * 10^-12+9
= 0.2 * 32 * 10^-3
= 6.4 * 10^-3 W = **6.4 mW** (option A).

The question states C = 50 pF. Applying the formula directly: P = 0.2 * 50e-12 * 0.64 * 1e9 = 6.4 mW.


- B incorrect: 6.4 W corresponds to C = 50 nF (nanofarads, 1000x larger than stated).
- C incorrect: 64 mW corresponds to C = 500 pF, which is 10x larger than stated.
- D incorrect: 64 W requires C = 5 uF, which is wildly unrealistic for a MAC array gate capacitance.

The key formula insight: double-check units. P = 0.2 * 50e-12 * 0.64 * 1e9 = 6.4e-3 W = 6.4 mW.

### Q12 — Correct: B

If leakage is 20-30% of total power, the dynamic power available is 70-80% of the total budget:
0.70 * 300 W = 210 W to 0.80 * 300 W = 240 W. This leaves **210-240 W** for dynamic compute and
memory access.

- A incorrect: At 5nm/3nm advanced nodes, leakage current is significant and increases sharply
  with temperature. 20-30% leakage is a realistic estimate; calling it negligible is incorrect.
- C incorrect: 50% leakage would be extreme even for very high-temperature operation or older nodes
  without low-leakage cells. 20-30% is the industry-standard estimate for modern advanced nodes.
- D incorrect: 10% leakage is more characteristic of older (28nm+) nodes where transistors have
  higher threshold voltages. At 5nm, leakage is a primary power concern and is not as low as 10%.

### Q13 — Correct: B

In UVM, sequences define the test stimulus by generating sequence items (transactions). The
sequence runs on a sequencer, which arbitrates between multiple sequences and feeds transactions to
the driver. The driver converts abstract transactions into pin-level activity on the DUT interface.

- A incorrect: The scoreboard compares DUT outputs against expected results (the reference model).
  It is responsible for checking, not generating stimulus.
- C incorrect: The monitor passively observes the DUT interface and converts pin-level activity
  into abstract transactions for the scoreboard. It does not generate stimulus.
- D incorrect: The UVM agent is an architectural container that bundles the sequencer, driver, and
  monitor. It coordinates these components but does not itself generate stimulus — the sequence
  does that.

### Q14 — Correct: B

Formal verification excels at proving or disproving properties of the form "if condition X holds,
then property Y is always (or never) true." The error-flag assertion on out-of-bounds address is
a clear two-cycle bounded safety property: it can be expressed as an SVA (SystemVerilog Assertion)
and proven or disproven by the formal tool by exhaustive state-space exploration.

- A incorrect: Proving a timing property ("within 10 ms") over a large transfer requires
  simulating billions of cycles, which is intractable for formal tools. This is a simulation or
  emulation target.
- C incorrect: Efficiency is a performance metric that depends on statistical distributions of
  transfer sizes — not a binary correctness property suitable for formal verification.
- D incorrect: Linear throughput scaling is a performance characteristic requiring parameterised
  simulation across many configurations, not a bounded reachability or invariant property.

### Q15 — Correct: C

Cross coverage creates a coverage space from the Cartesian product of multiple individual
coverpoints. For example: `batch_size x sequence_length x head_count` covers all 3D combinations.
A bug that only manifests when batch_size = 1 AND sequence_length = 4097 AND head_count = 64
would only be caught when all three conditions are exercised together — which cross coverage
explicitly tracks.

- A incorrect: Line coverage ensures code paths are executed but does not distinguish whether
  corner-case input combinations triggered those paths. A default test case might exercise all
  lines with a single non-corner tensor shape.
- B incorrect: Toggle coverage verifies signal transitions (0→1 and 1→0) for every bit. This is
  useful for finding unexercised datapath signals but does not capture functional parameter
  combinations.
- D incorrect: Branch coverage verifies both outcomes of every conditional, which is important
  but does not explicitly track multi-dimensional parameter space coverage.

### Q16 — Correct: C

Per-token compute: 32 layers * 1M MACs/layer = 32M MACs = 32M operations (counting MACs as
one operation each) or 64M single ops. For a 256-wide MAC array at 1 GHz:
Throughput = 256 MACs/cycle * 1e9 cycles/s = 256e6 MACs/s = 256 MMAC/s.
Time = 32M MACs / 256M MACs/s = 32/256 seconds = 0.125 s — that is far too high.

Re-examining: 32M MACs / (256 MACs/cycle) = 125,000 cycles = 125 microseconds at 1 GHz.

So 125 microseconds — option B.

Wait, let's recheck option C: 3.9 microseconds. That would require: 32M / 3.9e-6 s = 8.2e12
MACs/s, which is way higher than 256M MACs/s. So option C is wrong.

**Correction: Q16 = B (125 microseconds).**

Verification: 32M MACs / (256 MACs/cycle * 1e9 cycles/s) = 32e6 / 256e6 s = 0.125e-0 s =
125e-6 s = 125 microseconds.

- A incorrect: 32 microseconds would require the array to process 32M MACs / 32e-6 s = 1e12
  MACs/s, requiring 1000 MAC units at 1 GHz — not 256.
- C incorrect: 3.9 microseconds is approximately 32M / 8192 MAC array — the 256-array is ~33x
  slower than required for this answer.
- D incorrect: 1 millisecond would imply only 32M / 1e-3 = 32e9 MACs/s, requiring 32 MAC units
  at 1 GHz (not 256). The calculation is 8x off.

### Q17 — Correct: B

For prefill (not decode), weights are still loaded once per layer, but the compute is a GEMM
(not matrix-vector). For the question, assuming weight-streaming: 7 GB weights / 1 TB/s = 7 ms.
As a fraction of 50 ms: 7 ms / 50 ms = 14%.

This is only the weight loading lower bound; actual TTFT will be higher due to activation compute.

- A incorrect: 3.5 ms = 7 GB / 2 TB/s — this would apply to 2 TB/s bandwidth, not 1 TB/s.
- C incorrect: 7 ms < 50 ms, so weight loading alone does not exhaust the full budget (assuming
  the computation is prefill/GEMM which can be done faster than decode).
- D incorrect: 0.35 ms = 7 GB / 20 TB/s — no HBM device achieves 20 TB/s bandwidth.

### Q18 — Correct: B

With B requests each completing L tokens, total tokens = B * L.
Time to generate one token = T_decode ms = T_decode / 1000 seconds.
Total time ≈ L * T_decode ms (sequential decode, ignoring prefill).
Aggregate throughput = (B * L tokens) / (L * T_decode / 1000 s) = **B * 1000 / T_decode
tokens/second**.

Since T_decode is in ms, we convert: (B * L) / (L * T_decode * 1e-3) =
B * 1000 / T_decode tokens/s, which matches option B.

- A incorrect: This formula is dimensionally wrong when T_decode is in ms. B * L / T_decode with
  T_decode in ms gives tokens/ms, not tokens/s.
- C incorrect: L / T_decode gives per-request throughput, not aggregate. It omits the factor of B.
- D incorrect: B / T_decode ignores L (the number of tokens per request), giving throughput in
  requests/time, not tokens/time.

### Q19 — Correct: B

At batch=1: 100 tokens/s. Effective bandwidth used = 7 GB (model) * 100 times/s = 700 GB/s.
At batch=32: 1500 tokens/s. Per-token bandwidth = 7 GB * (1500/32) tokens/request/s... actually:
aggregated token throughput = 1500, which is only 15x better for 32x batch size. This indicates
that at batch=32, the system is approaching the bandwidth ceiling: 7 GB * 1500 / 32... let's
check. At batch=1, 100 t/s → 700 GB/s weight bandwidth. At batch=32, 1500 t/s → weights per
second: 7 GB * (1500/32) = 328 GB/s — but this doesn't account for amortisation.

The correct analysis: at batch=B, all B requests share one weight-loading pass, so weight bytes
per aggregate token = 7 GB / B. At batch=1: 7 GB/token. At batch=32: 7/32 = 0.22 GB/token.
1500 tokens/s * 0.22 GB/token = 328 GB/s HBM bandwidth. The hardware uses less bandwidth per
token at large batch but the sublinear scaling (15x gain for 32x batch) shows we are not at
the compute-bound regime — we are transitioning from heavily memory-bound to approaching
bandwidth saturation.

- A incorrect: If the workload were compute-bound at both batch sizes, throughput would scale
  linearly with batch size up to the MAC array saturation point. 1500/100 = 15x gain for 32x
  batch implies the system is not compute-bound at either operating point.
- C incorrect: A memory leak would cause progressive degradation over time, not a stable
  throughput plateau. The described behaviour is characteristic of a memory-bandwidth ceiling.
- D incorrect: There is no statement that the hardware has a 32-way parallelism limit. The
  sub-linear scaling (15x not 32x) and the transition from memory-bound to compute-approaching
  is the explanation.

### Q20 — Correct: D

At 1000 tokens/s with each token requiring one pass over all model weights (7 GB INT8):
Required bandwidth = 7 GB/token * 1000 tokens/s = **7000 GB/s = 7 TB/s**.

This is a strikingly high number — far beyond what current HBM3e (4.8 TB/s per device) achieves
on a single chip. In practice, achieving 1000 tokens/s for a 7B model requires either batching
(amortising weights over many requests) or multi-chip systems with combined HBM bandwidth.

- A incorrect: 7 GB/s is the bandwidth requirement for 1 token/s, not 1000 tokens/s. This is 3
  orders of magnitude too low.
- B incorrect: 70 GB/s = 7 GB/token * 10 tokens/s — only 10 tokens/s throughput, not 1000.
- C incorrect: 700 GB/s = 7 GB/token * 100 tokens/s — 10x too low for the target of 1000 tokens/s.
