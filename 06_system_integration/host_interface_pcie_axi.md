# Host Interface Design: PCIe and AXI

## Overview

The host interface is the boundary between a general-purpose CPU (running the OS, user application,
and driver stack) and the LLM accelerator. Getting this boundary right determines command latency,
DMA throughput, and how easily the accelerator can be programmed by a framework like PyTorch or
vLLM. This document covers PCIe Gen4/Gen5 as the dominant host interconnect and AXI as the
on-chip fabric bridging the PCIe endpoint to accelerator resources.

---

## Tier 1: Fundamentals

### Q1. What is PCIe and why is it the standard host interconnect for discrete AI accelerators?
Summarise the bandwidth evolution from Gen3 to Gen5.

**Answer.**

PCIe (Peripheral Component Interconnect Express) is a serialised, point-to-point, full-duplex
interconnect defined by the PCI-SIG. Each **lane** carries one differential pair in each direction.
Physical link widths are x1, x2, x4, x8, x16.

Effective bandwidth per lane scales with generation:

| Generation | Line rate | Encoding | Effective BW/lane | x16 link total |
|---|---|---|---|---|
| PCIe Gen3 | 8.0 GT/s | 128b/130b (~1.5% overhead) | ~985 MB/s | ~15.8 GB/s |
| PCIe Gen4 | 16.0 GT/s | 128b/130b | ~1.97 GB/s | ~31.5 GB/s |
| PCIe Gen5 | 32.0 GT/s | 128b/130b | ~3.94 GB/s | ~63 GB/s |
| PCIe Gen6 | 64.0 GT/s | 242b/256b (~6% overhead) | ~7.56 GB/s | ~121 GB/s |

**Why PCIe?**
- Universal: every x86 and ARM server platform has PCIe root complexes.
- CXL (Compute Express Link) is built on the PCIe physical layer, enabling cache-coherent memory
  expansion in the same slot.
- The PCIe transaction layer (TLP) provides reliable delivery, flow control, and error correction
  without additional software overhead.

**Why not USB or SATA?** Neither offers sufficient bandwidth or latency for the weight/activation
transfers required by LLM inference (weights alone for LLaMA-2 70B are 140 GB at float16).

---

### Q2. What is a Base Address Register (BAR)? How does the driver use BARs to communicate with the
accelerator?

**Answer.**

A BAR is a 32- or 64-bit register in PCIe configuration space (offsets 0x10–0x24) that the BIOS
or OS configures during enumeration to assign a range of host physical address space to a device
resource.

**During enumeration:**
1. The OS writes all-ones to a BAR and reads back the result. The number of zero bits at the LSB
   tells the OS the required alignment and size: if bits [19:0] are zero, the BAR requests 1 MB.
2. The OS assigns a physical address range (e.g., 0xC800_0000) and writes it to the BAR.
3. The PCIe root complex now forwards any CPU load/store to that range as a PCIe TLP to the device.

**Typical accelerator BAR layout:**

| BAR | Size | Contents |
|---|---|---|
| BAR0 (64-bit) | 4 MB | MMIO control registers: command queue doorbell, status, interrupt CSRs |
| BAR2 (64-bit) | 256 MB | Frame buffer / device memory visible to host (optional, for CXL) |
| BAR4 (32-bit) | 64 KB | MSI-X interrupt table and PBA |

**Driver usage:**
```c
/* Map BAR0 into kernel virtual address space */
void __iomem *bar0 = pci_iomap(pdev, 0, 0);

/* Write a command ring doorbell (triggers DMA on the device) */
iowrite32(ring_tail, bar0 + DOORBELL_OFFSET);

/* Read accelerator status */
u32 status = ioread32(bar0 + STATUS_OFFSET);
```

MMIO writes to BAR0 must be preceded by a store fence (`wmb()` in Linux) to prevent the CPU from
reordering writes that set up the descriptor before the doorbell write.

---

### Q3. Explain the difference between MMIO-mapped registers and DMA. When should each be used for
communicating with an LLM accelerator?

**Answer.**

**MMIO (Memory-Mapped I/O):**
The CPU performs load/store instructions to device registers via BAR-mapped addresses. Every access
generates a PCIe TLP (typically a 4-byte or 8-byte non-posted memory write). Latency is dominated
by PCIe round-trip time: ~1–3 µs for a posted write, ~5–10 µs for a non-posted read (requires
completion TLP).

**DMA (Direct Memory Access):**
The device's DMA engine reads from or writes to host DRAM autonomously, without CPU involvement per
transfer. The CPU sets up a descriptor (source address, destination address, byte count) via MMIO,
then triggers the DMA engine via a doorbell. The DMA engine issues PCIe read requests (Memory Read
TLPs) or write requests (Memory Write TLPs) to host DRAM, completing at line-rate bandwidth.

**When to use each:**

| Operation | Mechanism | Rationale |
|---|---|---|
| Write inference command (8–64 bytes) | MMIO write to doorbell | Too small to amortise DMA setup; MMIO latency acceptable |
| Read status / completion flag | MMIO read or polling host-written completion record | Polling device register is slow (~10 µs); prefer device DMA-writing a flag to host memory |
| Transfer weight tensors to device (GBs) | Host→Device DMA | Only DMA can sustain >20 GB/s; CPU-driven MMIO would max out at ~4 GB/s |
| Return inference results (~KB) | Device→Host DMA | Interrupt + DMA write of result tensor; driver reads from host memory |

**Key principle:** MMIO is for control-plane operations (commands, configuration, status). DMA is
for data-plane operations (weight/activation/result tensors).

---

### Q4. What is an AXI interconnect and how does it connect the PCIe endpoint to on-chip resources
in an accelerator SoC?

**Answer.**

AXI (Advanced eXtensible Interface, ARM AMBA specification) is a point-to-point master/slave
protocol with five independent channels:

| Channel | Direction | Purpose |
|---|---|---|
| AR (Address Read) | Master → Slave | Initiates a read transaction with address and burst length |
| R (Read Data) | Slave → Master | Returns read data, one beat per burst element |
| AW (Address Write) | Master → Slave | Initiates a write transaction |
| W (Write Data) | Master → Slave | Write data beats (may come after AW) |
| B (Write Response) | Slave → Master | Acknowledges completion of a write |

**In an LLM accelerator SoC:**

```
PCIe TLP → PCIe Bridge IP → AXI Master
                                │
                    AXI Interconnect (crossbar or NI-400)
                    ┌───────────┬─────────────┬────────────┐
                    ▼           ▼             ▼            ▼
              CSR Slave    SRAM Slave    DMA Slave    Core Slave
              (registers)  (unified buf)  (engine)   (cmd queue)
```

The PCIe bridge translates inbound TLPs to AXI master transactions:
- A PCIe Memory Write TLP becomes an AXI write on AW + W channels.
- A PCIe Memory Read TLP becomes an AXI read on AR, with the response returned as a Completion TLP.

**AXI4 burst parameters for HBM-connected slaves:**
- `ARLEN` = 7 (8 beats per burst, 64 B cache line at 8B width = 16 beats for 128-bit data)
- `ARSIZE` = 3 (8 bytes per beat on a 64-bit bus)
- `ARBURST` = INCR (incrementing address)

This matches the AXI interconnect's optimal burst size for HBM efficiency.

---

## Tier 2: Intermediate

### Q5. Design a command queue for an LLM accelerator. What fields must each entry contain? How do
you handle flow control between the host driver and the accelerator?

**Answer.**

A command queue decouples command submission (host) from command execution (accelerator). It is
implemented as a circular buffer in either host DRAM (mapped via IOMMU/DMA) or on-device SRAM.

**Command entry fields (64 bytes per entry):**

```c
struct accel_cmd {
    uint8_t  opcode;        /* GEMM, GEMV, ATTN, DMA_H2D, DMA_D2H, BARRIER */
    uint8_t  flags;         /* bit0=async, bit1=fence_before, bit2=interrupt_on_done */
    uint16_t seq_id;        /* monotonic sequence number for completion tracking */
    uint32_t dependency_mask; /* bitmask of seq_ids that must complete first */

    /* Tensor descriptors */
    uint64_t src_addr_a;    /* physical or device address of first operand */
    uint64_t src_addr_b;    /* second operand (for GEMM/GEMV) */
    uint64_t dst_addr;      /* output tensor address */

    /* Shape */
    uint32_t M, N, K;       /* matrix dimensions */
    uint16_t dtype;         /* float16, bfloat16, int8 */
    uint16_t tile_config;   /* tile M/N/K encoding for MAC array config */

    uint8_t  reserved[16];  /* pad to 64 bytes */
};
```

**Flow control (producer-consumer with head/tail pointers):**

```
Host DRAM:   [ cmd0 | cmd1 | cmd2 | .... | cmd_N-1 ]
                 ▲                           ▲
              tail (host writes here)     head (accel reads here)
```

- Host writes `tail` to a BAR0 register (doorbell) after enqueuing one or more commands.
- Accelerator advances `head` after accepting each command; writes `head` to a host-memory-mapped
  completion word via DMA.
- Flow control: host stalls submission when `(tail - head) >= queue_depth`. Typical queue depth:
  256–1024 entries.
- The `dependency_mask` allows out-of-order execution for independent operations (e.g., two
  independent DMA loads can proceed in parallel even if interleaved in the queue).

**Why on-device SRAM for the command queue vs. host DRAM?**

| Location | Latency to read | Flow control | Use case |
|---|---|---|---|
| Host DRAM (PCIe fetch) | ~1–3 µs per fetch | Host can fill without polling device | Large, variable-length queues |
| Device SRAM | ~10–50 ns per fetch | Limited by SRAM depth | Ultra-low-latency dispatch |

For an LLM accelerator where layers take 50–500 µs each, PCIe-latency command fetch is acceptable.
For sub-microsecond operator scheduling, on-device SRAM command queues are preferred.

---

### Q6. Compare interrupt-driven and polling-based completion notification. When should each be used
for LLM inference?

**Answer.**

**Interrupt-driven:**
The accelerator writes a completion status word to host memory (via DMA), then asserts an MSI-X
interrupt. The CPU's interrupt handler reads the status word and wakes the waiting thread.

Latency breakdown:
- DMA write completion status: ~0.5 µs
- Interrupt assertion to handler entry: ~1–5 µs (APIC + interrupt handler overhead)
- Context switch to waiting thread: ~5–20 µs if the thread was descheduled

Total interrupt notification latency: ~7–25 µs.

**Polling:**
The driver thread busy-waits on a host-memory completion word that the accelerator DMA-writes on
completion:

```c
/* Accelerator DMA-writes completion word to this host address */
volatile uint32_t *completion = dma_alloc_coherent(dev, ...);

/* Driver polls (spin) */
while (*completion != expected_seq_id)
    cpu_relax(); /* pause instruction to reduce bus traffic */
```

Polling notification latency: ~0.5–2 µs (only the DMA write latency). No interrupt overhead.

**When to use each:**

| Scenario | Preferred method | Reason |
|---|---|---|
| High-throughput batched inference | Interrupt | CPU should not busy-wait for 50+ ms batch |
| Low-latency single-request inference | Polling | Eliminates 5–20 µs interrupt overhead |
| Power-constrained deployment | Interrupt | Polling burns a CPU core at 100% |
| Streaming token generation | Interrupt per token | One token every 5–20 ms; interrupt cost amortised |

**Hybrid approach:** Use polling for the first N microseconds (covers normal fast completions),
then fall back to interrupt if the completion has not arrived. This minimises latency in the common
case while avoiding CPU waste in the long-tail case.

---

### Q7. Explain the IOMMU and why it is essential for DMA security in a multi-tenant accelerator
deployment.

**Answer.**

Without an IOMMU, a DMA-capable device is given a **physical address** by the driver and can
read or write any physical memory location on the machine — including kernel data structures,
other processes' memory, and hypervisor state. A compromised or buggy driver (or a malicious
accelerator) could exfiltrate arbitrary data.

The **IOMMU** (Input/Output Memory Management Unit, Intel VT-d / AMD-Vi) interposes on DMA
transactions:
- The kernel allocates a per-device I/O page table (IOPT) with its own virtual address space
  (IOVA).
- The driver maps only the specific buffers the device needs into the IOPT, and passes IOVAs
  (not physical addresses) to the device.
- The IOMMU translates IOVA → physical address on each DMA access and enforces read/write
  permissions. An out-of-bounds access causes an IOMMU fault, not a silent memory corruption.

**In multi-tenant (SR-IOV) LLM inference:**
- Each tenant's accelerator Virtual Function (VF) gets its own IOPT.
- Tenant A's model weights are mapped into IOPT-A but not IOPT-B.
- A bug in Tenant B's driver cannot DMA-read Tenant A's KV cache.

**Performance consideration:** IOMMU page table walks add latency to DMA address translation.
Modern IOMMUs use hardware TLBs (IOTLB) to cache frequently-used IOVA→PA mappings. For
large contiguous DMA regions (as in weight loading), a single huge-page mapping (2 MB or 1 GB)
results in a single IOTLB entry, making IOMMU overhead negligible.

---

### Q8. How does CXL.mem differ from PCIe DMA for accessing host memory? What are the implications
for LLM weight management?

**Answer.**

**PCIe DMA:** The accelerator's DMA engine initiates Memory Read TLPs to fetch data from host DRAM.
The CPU has no visibility into or cache coherence for these transfers. The driver must explicitly
flush/invalidate CPU caches before DMA and after DMA (or use DMA-coherent memory).

**CXL.mem (CXL 2.0 Type 3):** The accelerator's memory (or host DRAM via an CXL device) appears
in the host's physical address space as a coherent, cacheable memory range. The CPU can load/store
to device memory with normal `mov` instructions; the CXL protocol handles coherence.

**Key differences:**

| Property | PCIe DMA | CXL.mem |
|---|---|---|
| Coherence | Software-managed (explicit flush) | Hardware cache-coherent (snoop protocol) |
| Access initiator | Device DMA engine only | CPU and device (peer-to-peer) |
| Latency to host DRAM | ~1–3 µs (DMA setup + transfer) | ~300–500 ns (cache-coherent load) |
| Bandwidth | Up to 63 GB/s (PCIe Gen5 x16) | Same physical layer, same BW |
| Programming model | Descriptor-based DMA | Standard pointer dereference |

**Implications for LLM weight management:**

With CXL.mem, the accelerator can **directly read model weights from host DRAM** without staging
them through on-device HBM. For a LLaMA-2 70B model (140 GB at float16), if device HBM is only
80 GB, CXL.mem allows the overflow weights to remain in host DRAM, accessed on demand with
~400 ns latency. This is called **weight streaming** and is impractical with PCIe DMA (per-access
DMA overhead is too high for fine-grained streaming).

The tradeoff: CXL.mem accesses to host DRAM are ~5× slower than local HBM accesses. Weight layers
that are frequently reused (the first and last transformer layers, shared embedding tables) should
still be staged in HBM; the tail layers that are accessed once per token can stream from CXL.mem.

---

## Tier 3: Advanced

### Q9. Design the full driver-accelerator protocol for submitting a batched prefill request. Include
descriptor layout, DMA sequencing, memory barrier placement, and completion handling.

**Answer.**

**Assumptions:** Device has a command queue in host DRAM (IOMMU-mapped), an HBM weight store
pre-loaded at init time, and a device SRAM KV cache. Host DRAM holds the input token tensor.

**Step-by-step protocol:**

**1. Allocate and fill an inference descriptor in host DRAM:**

```c
struct prefill_desc {
    uint64_t token_ids_iova;    /* host DRAM IOVA of token IDs [B, S] */
    uint64_t kv_cache_dev_addr; /* device address in HBM for KV output */
    uint64_t output_logits_iova;/* host DRAM IOVA for output logits */
    uint32_t batch_size;        /* B */
    uint32_t seq_len;           /* S */
    uint32_t seq_id;            /* unique ID for this request */
    uint32_t flags;
};
```

**2. DMA input token IDs to device SRAM:**

Before issuing the prefill command, token IDs must reach device SRAM (or HBM). The driver
enqueues a `DMA_H2D` command:

```c
struct accel_cmd dma_cmd = {
    .opcode   = DMA_H2D,
    .src_addr_a = token_ids_iova,    /* host source (IOVA) */
    .dst_addr   = device_sram_addr,  /* device destination */
    .M = batch_size * seq_len * sizeof(uint32_t), /* byte count */
    .seq_id   = ALLOC_SEQ(),
    .flags    = FENCE_AFTER,         /* downstream commands wait for this */
};
```

**3. Memory barrier before doorbell:**

The command entry must be fully written to host DRAM before the doorbell write reaches the device:

```c
/* Write command to ring buffer in host DRAM */
memcpy(ring_base + tail_idx * sizeof(struct accel_cmd), &dma_cmd, sizeof(dma_cmd));

wmb();  /* store-store barrier: ensure command is visible before doorbell */

/* Ring the doorbell (MMIO write to BAR0) */
iowrite32(new_tail, bar0 + CMD_TAIL_DOORBELL);
```

Without `wmb()`, a weakly-ordered CPU (e.g., ARM) may reorder the doorbell write before the
descriptor write, causing the device to fetch an uninitialised descriptor.

**4. Submit the prefill compute command:**

After the DMA command (enforced by `FENCE_AFTER`), enqueue the GEMM/attention sequence:

```c
struct accel_cmd prefill_cmd = {
    .opcode          = PREFILL,
    .src_addr_a      = device_sram_addr,     /* token embeddings after lookup */
    .src_addr_b      = weight_base_dev_addr, /* weights in HBM (pre-loaded) */
    .dst_addr        = kv_cache_dev_addr,    /* KV cache output */
    .M = batch_size, .N = hidden_dim, .K = seq_len,
    .seq_id          = ALLOC_SEQ(),
    .dependency_mask = (1 << dma_cmd.seq_id), /* must wait for DMA */
    .flags           = INTERRUPT_ON_DONE,
};
```

**5. Completion handling:**

The device, on completing the prefill, DMA-writes a completion record to host DRAM and asserts
MSI-X interrupt:

```c
/* Completion record DMA'd by device */
struct completion_record {
    uint32_t seq_id;
    uint32_t status;     /* 0 = success, non-zero = error code */
    uint64_t timestamp;  /* device clock cycles, for profiling */
};
```

The interrupt handler:
```c
irqreturn_t accel_irq_handler(int irq, void *dev_id) {
    struct completion_record *rec = dev->completion_ring + dev->comp_head++;
    rmb();  /* read-acquire barrier: ensure we see the completion data */
    complete_inference_request(rec->seq_id, rec->status);
    return IRQ_HANDLED;
}
```

The `rmb()` prevents the CPU from speculating the `seq_id` read before the DMA write has
propagated through the memory interconnect.

---

### Q10. An LLM accelerator is connected to the host over PCIe Gen5 x16 (63 GB/s). The model
weights are 140 GB (LLaMA-2 70B at float16). Characterise the model loading time and what
architectural optimisations reduce time-to-first-token.

**Answer.**

**Baseline model loading time:**

$$t_{load} = \frac{140\ \text{GB}}{63\ \text{GB/s}} \approx 2.2\ \text{seconds}$$

This assumes the PCIe link is fully saturated. In practice, PCIe DMA efficiency is ~85–90%,
so realistic time is ~2.5–2.6 s. This is acceptable for a cold-start scenario but unacceptable
for serving where multiple model variants must be swapped quickly.

**Optimisations to reduce time-to-first-token:**

**1. Persistent weight caching in HBM:**
On first load, weights are DMA'd to device HBM and pinned there. Subsequent requests have zero
load latency. Requires device HBM ≥ model size (e.g., 80 GB HBM for a 70B float16 model
requires quantisation to int8 for 70 GB, or an 8-chip system for full float16).

**2. Quantised weights (int4/int8):**
With INT4 quantisation (4× compression):

$$t_{load,int4} = \frac{35\ \text{GB}}{63\ \text{GB/s}} \approx 0.55\ \text{s}}$$

The dequantisation cost on-device (INT4 → BF16 for compute) is ~0.1 TOPS at 8 MAC/weight,
negligible compared to the 300 TOPS accelerator compute.

**3. Weight streaming with prefetch:**
Overlap weight loading of layer $l+1$ with compute of layer $l$. The PCIe DMA engine and MAC
array operate independently. Effective time-to-first-token:

$$t_{TTFT} = t_{load,layer1} + \max(t_{compute}, t_{load\_remaining})$$

If compute takes 50 ms and the DMA takes 2.5 s, TTFT is still DMA-bound for cold start. However,
for a cached model (HBM-resident), TTFT is only the prefill compute time.

**4. CXL.mem weight streaming:**
With CXL, weights stay in host DRAM (DDR5 at 400 GB/s total, ~50 GB/s per CXL slot). The
accelerator streams weights directly with ~2.5× lower latency than PCIe DMA per access, but
the total bandwidth cap is the same. The advantage is eliminating explicit DMA programming for
each layer, reducing control overhead per-layer from ~5 µs to ~0.5 µs.

**5. Paged weight management (vLLM-style for adapters):**
For LoRA/adapter variants that share base weights, only the adapter delta (typically 0.1–1% of
model size) needs to be swapped. Base weights remain pinned in HBM. Adapter load time:

$$t_{adapter} = \frac{1.4\ \text{GB}\ (1\%\ of\ 140\ \text{GB})}{63\ \text{GB/s}} \approx 22\ \text{ms}$$

This is feasible for per-request adapter switching.
