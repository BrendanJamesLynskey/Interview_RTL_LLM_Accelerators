# Accelerator SoC Architecture

## Overview

Modern LLM accelerator SoCs are complex multi-die or single-die systems integrating a host control
CPU, one or more matrix-engine cores, high-bandwidth memory (HBM or LPDDR), a network-on-chip (NoC)
or crossbar interconnect, clock and power management, and chip-to-chip links for multi-chip model
parallelism. Understanding how these components fit together — and where the design tradeoffs live —
is a core competency for hardware architects working on AI inference silicon.

---

## Tier 1: Fundamentals

### Q1. Draw and explain the major blocks in a typical LLM accelerator SoC.

**Answer.**

A representative block diagram includes the following subsystems:

```
┌─────────────────────────────────────────────────────────┐
│                     LLM Accelerator SoC                 │
│                                                         │
│  ┌──────────┐    ┌──────────────────────────────────┐   │
│  │ Host CPU │◄──►│         PCIe / CXL Root          │   │
│  │ (ARM/x86)│    └──────────────────────────────────┘   │
│  └──────────┘                   │                       │
│       │                         ▼                       │
│       │              ┌──────────────────┐               │
│       │              │  Control / Sched │               │
│       │              │   Processor      │               │
│       │              └────────┬─────────┘               │
│       │                       │                         │
│       ▼                       ▼                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │               On-Chip Interconnect (NoC)        │    │
│  └──┬────────────────────────────────────────┬─────┘    │
│     │                                        │          │
│  ┌──▼──────────┐   ┌──────────────┐   ┌──────▼──────┐  │
│  │ Accel Core 0│   │ Accel Core 1 │   │  SRAM Pool  │  │
│  │ (MAC Array, │   │ (MAC Array,  │   │  (Unified   │  │
│  │  Attn Eng,  │   │  Attn Eng,   │   │   Buffer)   │  │
│  │  Vec Unit)  │   │  Vec Unit)   │   └─────────────┘  │
│  └─────────────┘   └──────────────┘                    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │          HBM PHY  /  LPDDR5 Controller           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │     Chip-to-Chip Link (NVLink / UCIe / AXI-S)    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Key blocks and their roles:

| Block | Role |
|---|---|
| Control/Scheduling CPU | Parses the execution graph, issues DMA descriptors, sequences layers |
| Accelerator core(s) | Systolic or dataflow array for GEMM/GEMV; on-core SRAM for tiles |
| Unified SRAM pool | Shared scratchpad for KV cache, activations, intermediate tensors |
| NoC / crossbar | Low-latency, high-bandwidth fabric connecting all masters and slaves |
| HBM / LPDDR5 | Weight storage, KV cache overflow, large activation tensors |
| PCIe / CXL | Host command path, DMA to host memory, management plane |
| Chip-to-chip link | Model/tensor parallelism across multiple accelerator dies |

**Why it matters.** Interviewers expect you to reason about data flow: weights flow from HBM to
accelerator cores, activations flow between cores via the NoC, and control tokens flow from the
scheduling CPU to each core's command queue.

---

### Q2. What is a Network-on-Chip (NoC), and why is it preferred over a shared bus for accelerator
SoCs?

**Answer.**

A **shared bus** serialises all transactions; at most one master can drive the bus per cycle. With
tens of accelerator cores each needing simultaneous access to SRAM or HBM, bus bandwidth becomes the
bottleneck.

A **NoC** is a packet-switched on-chip interconnect where:
- Each node has a router with input/output ports and a small buffer (FIFO) per port.
- Packets are routed hop-by-hop using a routing algorithm (XY routing for mesh, round-robin for
  ring).
- Multiple packets travel concurrently on disjoint paths.

Common topologies used in AI SoCs:

| Topology | Bisection BW | Latency | Area | Notes |
|---|---|---|---|---|
| Crossbar | O(N) | O(1) | O(N²) | Best BW, only feasible for small N |
| 2D Mesh | O(N) | O(√N) | O(N) | Used in TPU, Groq; scales well |
| Ring | O(1) | O(N) | O(N) | Simple; good for collective ops |
| Torus | O(N) | O(√N) | O(N) | Mesh with wrap-around edges |

For an LLM accelerator with 8–64 cores, a **2D mesh** or **butterfly** offers the best
compute-per-area tradeoff. Ring topologies appear in multi-chip NVLink rings.

---

### Q3. What is a clock domain crossing (CDC), and what are the standard techniques for handling it?

**Answer.**

A CDC occurs when a signal originating in clock domain A (frequency $f_A$) is sampled in clock
domain B (frequency $f_B$). The hazard is **metastability**: the receiving flip-flop may not settle
to a valid logic level within the setup/hold window, causing unpredictable output.

Standard mitigation techniques:

1. **Two-flop synchroniser** — For single-bit control signals. The first flop absorbs metastability;
   the second flop samples a (nearly always) settled value. Mean time between failure (MTBF) is
   exponential in the number of synchroniser stages.

2. **Async FIFO (Gray-code pointer synchroniser)** — For multi-bit data paths. The read and write
   pointers are converted to Gray code (only 1 bit changes per increment), synchronised across the
   domain boundary with a two-flop synchroniser, then converted back to binary.

3. **Handshake synchroniser (req/ack)** — The sender asserts `req`; the receiver detects it,
   samples the data, then asserts `ack`; the sender deasserts `req` after seeing `ack`. Correct but
   slow; used for rare control signals.

4. **Reset synchroniser** — Asynchronous assert, synchronous de-assert. Ensures all flops in a
   domain come out of reset safely.

In an LLM accelerator SoC there are typically three clock domains: the host PCIe clock (~100 MHz
reference), the NoC/fabric clock (800 MHz–1.2 GHz), and the accelerator core clock (1–2 GHz). Each
boundary uses async FIFOs for data paths.

---

### Q4. What does the on-chip control processor do in an LLM accelerator, and why is a dedicated
processor preferred over pure hardwired control?

**Answer.**

The control (or "firmware") processor — often a small in-order ARM Cortex-M or RISC-V core — is
responsible for:

- **Graph scheduling:** Receiving a compiled operator graph from the host driver, sequencing
  operators in dependency order.
- **DMA programming:** Writing source/destination addresses, strides, and byte counts into DMA
  descriptor registers for each weight or activation transfer.
- **Core dispatch:** Writing tile descriptors into each accelerator core's command FIFO to trigger
  GEMM or attention execution.
- **Synchronisation:** Waiting for "done" interrupts or polling status bits to enforce data
  dependencies between operators.
- **Dynamic decisions:** Handling variable-length sequences, KV cache eviction, and fallback paths
  that require runtime branching.

Hardwired finite-state machines can implement fixed sequences at lower power, but they cannot adapt
to:
- Varying model architectures (different layer counts, head configurations).
- Runtime batching changes (continuous batching adds/removes sequences mid-flight).
- Firmware updates to fix scheduling bugs without a chip respin.

The control processor therefore provides the necessary programmability at a modest area cost (a
Cortex-M33 is ~0.02 mm² at 7 nm).

---

## Tier 2: Intermediate

### Q5. Explain how multiple accelerator cores are interconnected and coordinated for tensor
parallelism within a single SoC.

**Answer.**

In tensor parallelism (TP), a single large matrix is split across $N$ cores along one dimension
(column-parallel or row-parallel). Each core computes a partial result, which must be **reduced**
across all cores before the next operator.

**Column-parallel GEMM (A · W, where W is split column-wise):**

$$Y_i = X \cdot W_i, \quad i \in [0, N)$$

Each core $i$ produces a partial output shard $Y_i$. No reduction needed before a non-linearity
applied independently per shard.

**Row-parallel GEMM (reduction required):**

$$Y = \sum_{i=0}^{N-1} X_i \cdot W_i$$

Cores must **all-reduce** their partial sums. On-chip, this is implemented as:

1. Each core writes its partial sum to a shared SRAM region (DMA into a designated slice).
2. A reduction tree (adder forest in hardware, or a dedicated reduce engine) accumulates all slices.
3. The result is broadcast back to all cores.

Alternatively, a ring-reduce protocol over the NoC: core $i$ sends its partial sum to core
$(i+1)\bmod N$, accumulates what it receives, repeats $N-1$ times. Each core then holds the full
sum. Traffic per core is $O(N \cdot \text{tensor size})$, which caps total NoC utilisation.

**Hardware coordination mechanisms:**

- **Barrier synchronisation:** Each core writes a "arrived" bit to a shared register; the control
  processor (or a dedicated barrier unit) waits for all cores to arrive before issuing the next
  dispatch.
- **Credit-based flow control on the NoC:** Prevents buffer overflow when one core is faster than
  its peers.
- **Hardware semaphores:** SRAM-mapped locations that cores increment atomically (using
  read-modify-write with bus locking or LL/SC primitives).

---

### Q6. Describe the design of an async FIFO for crossing from the 1.8 GHz accelerator core clock
to the 900 MHz NoC clock. What parameters must you size?

**Answer.**

An async FIFO consists of dual-port SRAM (or register array), binary-to-Gray write pointer, binary-
to-Gray read pointer, two 2-flop synchronisers (one per crossing direction), and full/empty logic.

**Pointer synchronisation latency:**

The write pointer synchronised into the read domain incurs $n_{sync}$ cycles of latency
($n_{sync} = 2$ typically). At 900 MHz this is:

$$t_{sync} = \frac{n_{sync}}{f_{read}} = \frac{2}{900 \times 10^6} \approx 2.2\ \text{ns}$$

**Minimum FIFO depth to prevent underflow:**

The producer runs at $f_w = 1800\ \text{MHz}$ and can burst data for $B$ cycles. The consumer clock
is $f_r = 900\ \text{MHz}$. During the synchroniser latency the FIFO must absorb all produced data:

$$\text{depth} \geq \frac{f_w}{f_r} \cdot n_{sync} + \text{burst\_depth}$$

For a 64-bit-wide bus with 16-deep burst: $\text{depth} \geq 2 \times 2 + 16 = 20$ entries.
Round up to a power of two: **32 entries**.

**Key sizing parameters:**

| Parameter | Sizing Consideration |
|---|---|
| Depth | Burst size + 2×synchroniser latency (in slower-clock cycles) |
| Width | Data bus width (usually 256–512 bits for cache-line granularity) |
| Gray code bits | $\lceil \log_2(\text{depth}) \rceil$ |
| MTBF target | Drives synchroniser stage count; usually $\geq 10^{9}$ hours |

**Common mistake:** Using binary (thermometer) pointers directly across the CDC boundary. Binary
pointers can have multiple bits change simultaneously; the receiver may sample a transient mid-
transition value as a valid (but incorrect) pointer.

---

### Q7. How do you handle clock domain crossings between the HBM PHY clock and the accelerator core
clock in an LLM accelerator?

**Answer.**

HBM2e/HBM3 PHYs operate at a fixed reference clock (typically the PHY clock is derived from a
PLL locked to a 100 MHz reference, producing a ~1.2–2 GHz bit clock internally, but the
controller-facing AXI interface runs at 450–600 MHz). The accelerator core may run at 1.0–2.0 GHz.

The crossing from HBM controller clock ($f_{hbm} \approx 500\ \text{MHz}$) to core clock
($f_{core} \approx 1.5\ \text{GHz}$) is handled by:

1. **Async FIFO on the read data path** (HBM → core): HBM controller writes 512-bit cache lines at
   500 MHz; the FIFO presents them to the core at 1.5 GHz. Depth is sized for HBM read latency
   variance (~100–200 ns).

2. **Async FIFO on the write/address path** (core → HBM): The core issues 64-byte write requests
   at 1.5 GHz; the FIFO absorbs bursts and feeds them to the HBM controller at 500 MHz.

3. **Credit return path:** HBM controller sends credit tokens (2-bit Gray-coded count) back to the
   core through a 2-flop synchroniser. The core stalls address issue when credits reach zero.

The **key HBM bandwidth numbers** to keep in mind:

$$\text{HBM3 bandwidth per stack} = 2 \times 1024\ \text{bits} \times f_{clock}$$

At 3.2 Gbps per pin: $1024 \times 3.2 = 3.2\ \text{TB/s}$ per stack — or more precisely,
$1024\ \text{pins} \times 3.2\ \text{Gb/s} / 8 = 409.6\ \text{GB/s}$ per stack.

---

### Q8. Compare crossbar, mesh, and ring interconnects on the criteria most relevant to an LLM
accelerator: all-reduce latency, all-gather bandwidth, and silicon area.

**Answer.**

Let $N$ = number of accelerator cores, $M$ = message size in bytes.

**All-reduce latency (ring algorithm):**
$$t_{ring} = 2(N-1) \cdot \left(\alpha + \frac{M}{N \cdot b}\right)$$
where $\alpha$ is per-hop latency and $b$ is per-link bandwidth.

**All-reduce latency (crossbar / tree):**
$$t_{tree} = \log_2(N) \cdot \left(\alpha + \frac{M}{b}\right)$$

| Metric | Crossbar | 2D Mesh | Ring |
|---|---|---|---|
| Bisection BW | $O(N \cdot b)$ | $O(\sqrt{N} \cdot b)$ | $O(b)$ |
| All-reduce latency | $O(\log N)$ hops | $O(\sqrt{N})$ hops | $O(N)$ hops |
| All-gather BW | Near-optimal | Good | Limited by ring BW |
| Area | $O(N^2)$ wires | $O(N)$ wires | $O(N)$ wires |
| Practical $N$ limit | ~8 | 64+ | 8+ (usually multi-chip) |

**Verdict for on-chip TP:** For $N \leq 8$ cores on a single die, a **partial crossbar** or
**butterfly** gives the best all-reduce latency. For $N > 8$, a **2D mesh** with a tree reduce
engine is standard. Rings appear for multi-chip collective operations (NVLink rings span multiple
SoCs).

---

## Tier 3: Advanced

### Q9. A multi-chip LLM system uses UCIe (Universal Chiplet Interconnect Express) die-to-die links
for tensor parallelism. What are the bandwidth, latency, and protocol overhead considerations?
How would you architect the on-chip control to hide the inter-die all-reduce latency?

**Answer.**

**UCIe physical layer characteristics:**

UCIe Advanced (bumped die-to-die at <2 mm pitch) offers:
- Bandwidth density: ~1 Tb/s/mm of die edge at 16 Gbps per lane.
- Latency: ~2–4 ns PHY latency (much less than PCIe).
- Power: ~0.5–1 pJ/bit.

For a 4-chiplet system with 32 mm of edge per interface:

$$\text{BW per direction} = 32\ \text{mm} \times 1\ \text{Tb/s/mm} = 32\ \text{Tb/s} = 4\ \text{TB/s}$$

This is comparable to on-die NoC bandwidth and sufficient for large all-reduce operations.

**Protocol overhead:**

UCIe's FDI (Flit and Data Interface) adds:
- 8b/10b or 128b/130b encoding overhead (~1.5–2%).
- CRC and retry layer: adds ~8–16 bytes per 64-byte flit, ~12–25% overhead in the worst case.
- Flow control credits: 4–8 cycle round trip for credit return adds effective latency.

Net usable bandwidth after encoding: ~90% of raw.

**Hiding all-reduce latency with double-buffering:**

The all-reduce for a row-parallel GEMM in transformer layer $l$ can be overlapped with the
**compute of layer $l+1$** if the operator graph is structured as:

```
Layer l:  [GEMM_partial → AllReduce_start] ──────────────────────►
Layer l+1:                  [Load weights → GEMM_partial ─► AllReduce_start]
                                  ▲
                          overlap window
```

Requirements:
1. **Double-buffered weight SRAM:** While layer $l$'s all-reduce is in flight, layer $l+1$'s
   weights are loaded from HBM into the second buffer.
2. **Non-blocking all-reduce engine:** The control processor issues a "reduce-scatter" descriptor
   that works asynchronously; the core continues with the next GEMM tile immediately.
3. **Dependency tracking hardware:** A completion bitmap (one bit per outstanding all-reduce) that
   the GEMM engine checks before consuming its accumulator input.

**Arithmetic:** At 1.5 ns all-reduce latency for a 512-element float16 vector and 2 TB/s link:

$$t_{reduce} = \frac{512 \times 2\ \text{B}}{2\ \text{TB/s}} + 2 \times t_{UCIe\_hop} = 0.5\ \text{ns} + 4\ \text{ns} \approx 5\ \text{ns}$$

At a core frequency of 1.5 GHz, that is only ~7 cycles — negligible if pipelined correctly.

---

### Q10. Describe the power domain and clock gating strategy for an LLM accelerator SoC to meet a
250W TDP while maximising utilisation of the MAC array.

**Answer.**

**Power domain partitioning:**

```
┌─────────────────────────────────────────────────────┐
│  Domain 0: Always-On (AON)                          │
│  Control CPU, power management, eFuse, RTC          │
│  Voltage: 0.6V  Power: ~500 mW                      │
├─────────────────────────────────────────────────────┤
│  Domain 1: NoC + SRAM                               │
│  Voltage: 0.75–0.85V  Power: ~20–40 W               │
│  Clock gate when no active transactions             │
├─────────────────────────────────────────────────────┤
│  Domain 2: Accelerator Cores (per-core)             │
│  Voltage: 0.75–0.9V  Power: ~100–150 W total        │
│  Power gate idle cores; DVFS during decode phase    │
├─────────────────────────────────────────────────────┤
│  Domain 3: HBM PHY                                  │
│  Fixed voltage (HBM spec)  Power: ~15–25 W          │
├─────────────────────────────────────────────────────┤
│  Domain 4: PCIe / CXL PHY                           │
│  Voltage: 0.9V  Power: ~5–10 W                      │
└─────────────────────────────────────────────────────┘
```

**Clock gating strategy:**

Clock gating is applied at three granularities:

1. **Coarse-grained (power domain level):** Entire accelerator core power-gated when idle (between
   batches). Power-up sequence: assert isolation cells → ramp VDD → release reset → deassert
   isolation. Latency ~1–5 µs, acceptable since batch inter-arrival is >>1 µs.

2. **Fine-grained (functional unit level):** Within a core, individual MAC rows, vector units,
   and the attention engine have independent clock enables driven by the core's local sequencer.
   A row that is not computing (e.g., during weight load stall) has its clock gated within 1 cycle.

3. **SRAM clock gating:** Each SRAM bank has a clock enable tied to its access valid signal.
   An idle bank (not addressed in the current cycle) consumes only leakage current.

**DVFS during decode vs. prefill:**

| Phase | Compute intensity | Core voltage | Core frequency | Core power |
|---|---|---|---|---|
| Prefill | High (GEMM) | 0.9 V | 2.0 GHz | ~160 W |
| Decode | Low (GEMV) | 0.75 V | 1.2 GHz | ~60 W |

Decode is memory-bandwidth-bound, so reducing core voltage/frequency saves power without degrading
throughput (the bottleneck is HBM bandwidth, not TOPS). This is the fundamental reason decode can
be performed at a lower operating point.

**Total TDP budget allocation:**

| Block | Prefill power | Decode power |
|---|---|---|
| MAC array (all cores) | 150 W | 50 W |
| SRAM (on-core + shared) | 30 W | 20 W |
| HBM PHY | 25 W | 25 W |
| NoC | 15 W | 8 W |
| Control CPU + misc | 8 W | 8 W |
| PCIe/CXL | 7 W | 7 W |
| **Total** | **235 W** | **118 W** |

Prefill stays within 250 W TDP. Decode operates well within thermal limits, allowing time-averaged
power to be significantly below TDP for mixed workloads.

---

### Q11. A design team proposes using a single global synchronous clock domain at 800 MHz for the
entire accelerator SoC to simplify CDC. Critique this decision.

**Answer.**

While eliminating CDC simplifies verification and reduces the risk of metastability bugs, a single
global 800 MHz clock domain has serious engineering costs:

**1. Clock distribution power dominates.**
Clock tree power scales as $P_{clk} = \alpha \cdot C_{wire} \cdot V_{DD}^2 \cdot f$ where
$C_{wire}$ is proportional to total wire length. A global 800 MHz clock tree in a 400 mm² die at
7 nm may account for 15–25% of total dynamic power (20–40 W) — often more than useful compute
power.

**2. Skew across a large die becomes unmanageable.**
At 800 MHz, one clock period is 1.25 ns. Achievable clock skew across a 20 mm die (at 7 nm,
signal speed ~0.6× speed of light in dielectric) is:

$$t_{skew} \approx \frac{20\ \text{mm}}{0.6 \times 3 \times 10^{11}\ \text{mm/s}} \approx 111\ \text{ps}$$

With H-tree balancing, achievable skew is ~50–80 ps, consuming 6–10% of the clock budget for
setup margin alone. Higher frequency compounds this problem.

**3. Incompatibility with off-chip interfaces.**
PCIe, HBM, and UCIe all have their own reference clocks defined by their specifications. The PHY
layers inherently require CDCs regardless of the on-chip decision. A "CDC-free" internal design
still requires CDC at every chip boundary.

**4. Frequency mismatch with optimal operating points.**
The MAC array benefits from a high core clock (1.5–2 GHz) during prefill. The control CPU
benefits from a moderate clock (400–600 MHz) for power efficiency. Forcing both to 800 MHz either
over-clocks the CPU (wasted power) or under-clocks the MAC array (lost throughput).

**Better alternative — synchronous islands with async FIFOs:**
Use 3–5 well-defined clock domains, each at their natural operating frequency, connected via
verified async FIFOs. The CDC verification overhead is bounded and manageable (10–20 CDC
crossings in total). The power and performance gains far outweigh the verification cost.

---

### Q12. How does pipeline parallelism across multiple accelerator SoCs differ architecturally from
tensor parallelism, and what are the chip-to-chip bandwidth requirements for each?

**Answer.**

**Tensor parallelism (TP):**
- A single layer's weight matrix is split across $N$ chips.
- All chips work on the same token(s) simultaneously.
- Requires **all-reduce** after each layer: bandwidth is proportional to activation size.

For a transformer layer with hidden dimension $d = 4096$ and float16:

$$\text{all-reduce data} = 2 \times d \times 2\ \text{B} = 16384\ \text{B} \approx 16\ \text{KB per token per layer}$$

At 1000 tokens/s and 32 layers: $16\ \text{KB} \times 32 \times 1000 = 512\ \text{MB/s}$ — very
modest. But for batch size 64 and continuous batching, the figure scales to ~32 GB/s, which
requires NVLink/UCIe class links.

**Pipeline parallelism (PP):**
- Layer 0–$k$ run on chip 0, layers $k+1\text{–}2k$ run on chip 1, etc.
- Each chip processes a **different micro-batch** in a pipeline fashion.
- Requires point-to-point **activation transfer** between adjacent chips at each pipeline boundary.
- Bandwidth requirement is per-boundary: one activation tensor of size $[B, d] \times 2$ B per
  forward step.

For $B = 16$, $d = 4096$, float16:

$$\text{activation transfer} = 16 \times 4096 \times 2\ \text{B} = 131\ \text{KB per pipeline step}$$

At pipeline frequency 5000 steps/s: $131\ \text{KB} \times 5000 = 655\ \text{MB/s}$ per boundary.
This is a **much lower bandwidth requirement** than TP, making PP feasible over PCIe Gen5 (64 GB/s)
or even InfiniBand (400 Gbps = 50 GB/s).

**Key architectural distinction:**

| Attribute | Tensor Parallelism | Pipeline Parallelism |
|---|---|---|
| Synchronisation | Every layer (all-reduce) | Every pipeline boundary |
| Latency effect | Adds all-reduce latency per layer | Adds pipeline fill latency |
| BW requirement | High (activation size × N) | Low (one activation per boundary) |
| Chip-to-chip topology | All-to-all or ring | Linear chain |
| Optimal for | Memory-limited giant layers | Long sequences of smaller layers |

In practice, production LLM serving systems combine both: TP within a node (NVLink, high BW, low
latency), PP across nodes (InfiniBand or Ethernet, lower BW).
