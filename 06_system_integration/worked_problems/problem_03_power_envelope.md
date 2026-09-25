# Worked Problem 03: Power Envelope Estimation

## Problem Statement

Estimate the **total power consumption** of the LLM accelerator described below during both
prefill and decode phases. Show how to **fit the design within a 300W TDP** and quantify the
implications: how does the TDP constraint limit peak TOPS, and what is the resulting impact on
prefill throughput?

### Accelerator Specification

| Parameter | Value |
|---|---|
| Process node | 5nm TSMC N5 |
| Die area | 450 mm² |
| Core clock | 1.8 GHz |
| Supply voltage (core) | 0.85 V |
| MAC array | 512 × 512 INT8 / 256 × 256 BF16, systolic |
| On-chip SRAM | 64 MB (512 banks, 256-bit wide) |
| HBM | 2 stacks HBM3, 1638 GB/s total |
| PCIe | Gen5 x16 |
| Technology constants | See derivations below |

### Technology Constants (5nm N5, from published data)

| Parameter | Value | Source |
|---|---|---|
| Dynamic energy per MAC (BF16) | 0.25 pJ/MAC | Extrapolated from published AI chip papers |
| Dynamic energy per SRAM read (256-bit) | 4 pJ/access | TSMC 5nm SRAM characterisation |
| Dynamic energy per SRAM write (256-bit) | 5 pJ/access | TSMC 5nm SRAM characterisation |
| Leakage power density | 0.25 W/mm² | 5nm N5 typical |
| Clock tree switching factor | 0.15 × $C_{clock}$ × $V^2$ × $f$ per mm² | |
| HBM3 PHY power (per stack) | 12 W | HBM3 spec (JEDEC) |
| PCIe Gen5 x16 PHY power | 7 W | Published measurements |

---

## Solution

### Step 1: MAC Array Power

The MAC array is the dominant dynamic power consumer. We model power using the standard formula:

$$P_{MAC} = E_{MAC} \times \text{MACs/s} = E_{MAC} \times \text{TOPS} \times 10^{12}$$

**Peak TOPS calculation:**

The MAC array is $512 \times 512$ for INT8. For BF16, the array is half the density (256 × 256)
since BF16 MACs require wider adders. We focus on the BF16 array (inference precision).

$$\text{TOPS}_{BF16} = 2 \times 256 \times 256 \times 1.8\ \text{GHz} / 10^{12} = 2 \times 65536 \times 1.8 \times 10^9 / 10^{12}$$
$$= \frac{2 \times 65536 \times 1.8}{1000} = 235.9 \approx 236\ \text{TOPS (BF16)}$$

**Note on the factor of 2:** Each MAC computes one multiply and one accumulate per cycle —
2 FLOPs per MAC per cycle.

**MAC array power at 100% utilisation:**

$$P_{MAC,peak} = E_{MAC} \times \text{TOPS} \times 10^{12} = 0.25 \times 10^{-12}\ \text{J} \times 236 \times 10^{12}\ \text{MACs/s}$$

Wait — we must be careful with units. The MAC energy $E_{MAC} = 0.25$ pJ is per multiply-
accumulate operation (i.e., per individual MAC). The array performs:

$$\text{MACs/s} = 256 \times 256 \times 1.8 \times 10^9 = 1.18 \times 10^{14}\ \text{MACs/s}$$

$$P_{MAC,peak} = 0.25 \times 10^{-12}\ \text{J/MAC} \times 1.18 \times 10^{14}\ \text{MACs/s} = 29.5\ \text{W}$$

This seems low — let us cross-check. Published values for Google TPU v4 (7nm) give ~100 W for
the 275 TOPS BF16 array; NVIDIA A100 (7nm SXM) is ~400 W total for 312 TOPS. At 5nm, efficiency
improves ~40% over 7nm:

Revised estimate: $0.25 \times (5\text{nm}/7\text{nm})^2 \approx 0.25 \times 0.51 = 0.13$ pJ/MAC
at 5nm... but this contradicts our stated constant. Let us use the stated value and note that real
published data for 5nm AI chips (Apple M2 Neural Engine, Amazon Inferentia2) suggest MAC array
power of ~0.1–0.3 pJ/MAC inclusive of local register file and interconnect.

Using $E_{MAC} = 0.25$ pJ/MAC (consistent with total chip power divided by utilised MACs for
published 5nm chips):

$$P_{MAC,peak} = 29.5\ \text{W}$$

**Sanity check via energy efficiency:** Top 5nm AI chips achieve ~1–2 TOPS/W. Our array at
$236\ \text{TOPS} / 29.5\ \text{W} \approx 8\ \text{TOPS/W}$ for the MAC array alone (before SRAM,
HBM, and overhead). This aligns with the array being the most efficient sub-block; system-level
efficiency is ~1–2 TOPS/W after including everything.

---

### Step 2: On-Chip SRAM Power

The SRAM is accessed during both weight tiling (loading weight tiles from HBM into SRAM before
GEMV) and activation read/write (reading input activations, writing output activations).

**SRAM access rate:**

During compute, each MAC cycle requires reading one weight element and one activation element from
SRAM, and writing the partial sum back:

- Weight reads: one 256-bit access per cycle per row of the MAC array. The systolic array feeds
  256 weights per column advance cycle.
  Accesses/s = $256\ \text{columns} \times 1.8\ \text{GHz} / (256\ \text{bits} / 256\ \text{bits per access}) = 1.8 \times 10^9\ \text{accesses/s}$

Actually: one weight SRAM bank feeds one row of 256 multipliers. Each cycle:
- 1 read from weight SRAM (1 × 256-bit access) per column time
- 1 read from activation SRAM (1 × 256-bit access) per row time

For the systolic array, one 256-bit SRAM access per cycle for weights (across the 256 columns):

$$\text{SRAM read accesses/s} = 2 \times 1.8 \times 10^9 = 3.6 \times 10^9\ \text{accesses/s}$$
$$\text{SRAM write accesses/s} \approx 0.5 \times 1.8 \times 10^9 = 0.9 \times 10^9\ \text{accesses/s}$$

(Writes are less frequent; partial sums accumulate in the systolic array's local registers before
a final writeback.)

**SRAM dynamic power:**

$$P_{SRAM,dyn} = E_{read} \times \text{reads/s} + E_{write} \times \text{writes/s}$$
$$= 4 \times 10^{-12}\ \text{J} \times 3.6 \times 10^9 + 5 \times 10^{-12}\ \text{J} \times 0.9 \times 10^9$$
$$= 14.4\ \text{W} + 4.5\ \text{W} = 18.9\ \text{W}$$

**SRAM leakage power:**

64 MB SRAM at 5nm. SRAM cell density at 5nm: ~0.021 µm²/bit (FinFET SRAM). Total SRAM area:

$$A_{SRAM} = 64 \times 10^6 \times 8\ \text{bits} \times 0.021 \times 10^{-12}\ \text{m}^2/\text{bit} = 10.75\ \text{mm}^2$$

Wait — this is the bit cell area only. With sense amplifiers, decoders, and margins, total SRAM
macro area is typically $3–4\times$ the bit cell area:

$$A_{SRAM,total} \approx 4 \times 10.75 = 43\ \text{mm}^2$$

Leakage power density at 5nm: 0.25 W/mm²:

$$P_{SRAM,leak} = 0.25\ \text{W/mm}^2 \times 43\ \text{mm}^2 = 10.75\ \text{W}$$

**Total SRAM power:** $P_{SRAM} = 18.9 + 10.75 \approx 29.7\ \text{W}$

---

### Step 3: HBM Power

Each HBM3 stack draws ~12 W (per JEDEC HBM3 specification, including I/O and array power):

$$P_{HBM} = 2\ \text{stacks} \times 12\ \text{W/stack} = 24\ \text{W}$$

This is roughly constant regardless of utilisation (HBM3 uses always-on power for the PHY links),
with slight variation (~±20%) between idle and full-rate operation.

---

### Step 4: Clock Distribution Power

The clock tree distributes a 1.8 GHz clock across the 450 mm² die. Clock tree power depends on
the total wire capacitance driven by clock buffers.

**Model:** $P_{clock} = \alpha_{clock} \times C_{wire,total} \times V_{DD}^2 \times f$

For a 450 mm² die at 5nm, typical clock tree wire capacitance is estimated from published global
clock distribution data:

Wire capacitance density at 5nm: ~15 fF/µm (including buffer gate capacitance).
Total clock tree length estimate for a balanced H-tree on 450 mm²:

$$L_{tree} \approx 4 \sqrt{450}\ \text{mm} \times \log_2(N_{sink})\ \text{levels} \approx 4 \times 21.2 \times 12 \approx 1018\ \text{mm}$$

$$C_{total} = 15 \times 10^{-15}\ \text{F/}\mu\text{m} \times 1018 \times 10^3\ \mu\text{m} = 15.3\ \text{nF}$$

The clock net makes one full charge–discharge cycle every clock period, so its activity factor
is $\alpha_{clock} = 1$:

$$P_{clock} = 1 \times C_{total} \times V_{DD}^2 \times f = 15.3 \times 10^{-9} \times 0.85^2 \times 1.8 \times 10^9$$
$$= 15.3 \times 0.7225 \times 1.8 = 19.9\ \text{W}$$

**Cross-check:** Published AI chips at 7nm typically report 8–15% of total chip power for clock
distribution. Our 19.9 W on ~200 W total ≈ 10% — within that range.

---

### Step 5: I/O Power

**PCIe Gen5 x16:** 7 W (from spec).

**NoC and on-chip interconnect:** Estimated from wire switching activity. For a mesh NoC with
typical 10–15% link utilisation during compute:

$$P_{NoC} \approx 0.1\ \text{pJ/bit} \times 256\ \text{bits/link} \times 8\ \text{links} \times 1.8 \times 10^9 \times 0.15 = 0.055\ \text{W}$$

(With only 8 links this is negligible; a large mesh with many more links would draw proportionally more.)

**Control CPU + misc logic (ARM Cortex-M55 equivalent):** 0.5 W.

**Total I/O and misc:** $P_{IO} = 7 + 0.055 + 0.5 \approx 7.6\ \text{W}$

---

### Step 6: Total Power Budget Assembly

#### Prefill Phase (100% MAC utilisation, full HBM bandwidth)

| Block | Power (W) | Notes |
|---|---|---|
| MAC array (BF16, 100%) | 29.5 | Full utilisation |
| SRAM dynamic | 18.9 | Full access rate |
| SRAM leakage | 10.75 | Always on |
| HBM (2 stacks, full rate) | 24.0 | Near-peak BW |
| Clock distribution | 19.9 | Fixed |
| PCIe + NoC + control | 7.6 | Fixed |
| **Total (prefill)** | **110.7** | |

**Result: 111 W during prefill — well within 300 W TDP.**

This figure is suspiciously low compared to real AI accelerators. The discrepancy comes from the
MAC array energy constant — real systolic arrays have higher power due to datapath registers,
local accumulator storage, and interconnect between PE rows. Applying a **MAC array overhead
multiplier of 4×** (consistent with published data for full PE row including local SRAM and
interconnect, not just the multiplier):

$$P_{MAC,system} = 29.5 \times 4 = 118\ \text{W}$$

**Revised prefill total:**

| Block | Power (W) |
|---|---|
| MAC array + PE overhead | 118.0 |
| SRAM dynamic | 18.9 |
| SRAM leakage | 10.75 |
| HBM | 24.0 |
| Clock | 19.9 |
| I/O + misc | 7.6 |
| **Total (prefill)** | **199.2 W** |

**Margin to 300 W TDP:** $300 - 199.2 = 100.8\ \text{W}$ headroom.

#### Decode Phase (MAC underutilised, HBM bandwidth-bound)

During decode ($B = 32$, arithmetic intensity = 24 FLOP/byte), MAC array utilisation is:

$$\text{Utilisation}_{MAC} = \frac{I_{decode}}{I^*} = \frac{24}{183} \approx 13\%$$

MAC array power scales approximately linearly with utilisation (clock gating reduces switching
activity):

$$P_{MAC,decode} = 118\ \text{W} \times 0.13 \approx 15.3\ \text{W}$$

SRAM dynamic power also reduces proportionally:

$$P_{SRAM,dyn,decode} = 18.9 \times 0.13 \approx 2.5\ \text{W}$$

| Block | Prefill (W) | Decode (W) |
|---|---|---|
| MAC array + PE overhead | 118.0 | 15.3 |
| SRAM dynamic | 18.9 | 2.5 |
| SRAM leakage | 10.75 | 10.75 |
| HBM (full BW decode) | 24.0 | 22.0 |
| Clock | 19.9 | 13.3 (DVFS: 1.2 GHz) |
| I/O + misc | 7.6 | 7.6 |
| **Total** | **199.2 W** | **71.5 W** |

**Decode power is ~36% of prefill power.** Peak power occurs during prefill; decode has large
thermal headroom that can be used for higher batch sizes.

---

### Step 7: TDP Constraint — Maximum TOPS Within 300 W

We now ask: if we were to scale the MAC array until total power reaches exactly 300 W during
prefill, what TOPS do we achieve?

**Power budget for MAC array (after fixed overheads):**

$$P_{MAC,budget} = 300 - P_{SRAM} - P_{HBM} - P_{clock} - P_{IO}$$
$$= 300 - (18.9 + 10.75) - 24 - 19.9 - 7.6 = 300 - 81.15 = 218.85\ \text{W}$$

**Solving for TOPS:**

$$P_{MAC} = E_{MAC,system} \times \text{MACs/s}, \qquad \text{TOPS} = 2 \times \text{MACs/s} / 10^{12}$$

where $E_{MAC,system} = 4 \times E_{MAC} = 4 \times 0.25 = 1\ \text{pJ/MAC}$ (with PE overhead).

$$\text{MACs/s} = \frac{P_{MAC,budget}}{E_{MAC,system}} = \frac{218.85}{1 \times 10^{-12}} = 2.19 \times 10^{14}\ \text{MACs/s} \Rightarrow 437.7\ \text{TOPS}$$

**Maximum BF16 TOPS within 300 W TDP: approximately 438 TOPS.** The specified 256 × 256 array
(236 TOPS, 118 W with PE overhead) is well below this ceiling, so the TDP does not limit it; the
array could be scaled up by about 1.85× before the TDP binds.

**Implications for prefill throughput:**

At 438 TOPS (assuming 70% MAC utilisation during prefill):

$$\text{Effective TOPS} = 0.70 \times 437.7 = 306.4\ \text{TOPS}$$

Prefill latency for LLaMA-2 7B ($B=16$, $S=512$, 107.3 TFLOPs):

$$t_{prefill} = \frac{107.3\ \text{TFLOPs}}{306.4\ \text{TOPS}} \approx 350\ \text{ms}$$

Tokens per second (prefill): $\frac{8192\ \text{tokens}}{350\ \text{ms}} \approx 23,400\ \text{tokens/s}$

---

### Step 8: Power Optimisation Strategies

#### Strategy A: Voltage Scaling for Decode Phase (DVFS)

During decode, reduce $V_{DD}$ from 0.85 V to 0.72 V and frequency from 1.8 GHz to 1.2 GHz.

Power scales as $V^2 \times f$:

$$\frac{P_{decode}}{P_{prefill}} = \left(\frac{0.72}{0.85}\right)^2 \times \frac{1.2}{1.8} = 0.716 \times 0.667 = 0.478$$

This ratio applies only to core dynamic power. The decode table above already runs the clock at
1.2 GHz, and the MAC and SRAM work rate is fixed by HBM bandwidth, so the remaining saving is the
$V^2$ factor ($0.72$) on the MAC, SRAM-dynamic and clock terms; leakage, HBM and I/O do not scale.
Net decode power at DVFS: $(15.3 + 2.5 + 13.3) \times 0.72 + 10.75 + 22 + 7.6 \approx 63\ \text{W}$.

This leaves even more thermal headroom, enabling aggressive batch size increases to maximise
decode throughput.

#### Strategy B: SRAM Power Gating for Idle Banks

During decode, only the KV cache and activation SRAM banks are active. The weight tile double-
buffer occupies ~16 MB; the remaining ~48 MB of SRAM can be power-gated.

$$P_{SRAM,leak,saved} = \frac{48}{64} \times 10.75\ \text{W} = 8.06\ \text{W}$$

Saving ~8 W of leakage during decode by power-gating unused SRAM.

#### Strategy C: INT8 Quantisation for Higher TOPS/W

INT8 MACs are cheaper than BF16 MACs. Scaling the multiplier energy by significand width squared
gives $0.25 \times (8/11)^2 \approx 0.13$ pJ/MAC. (The 11 is FP16's significand width; BF16's
significand is only 8 bits, so this is an optimistic estimate for BF16 → INT8.) Applying the same
4× PE overhead gives $E_{INT8,system} \approx 0.53$ pJ/MAC.

At the same 300 W TDP, INT8 array delivers:

$$\text{TOPS}_{INT8} = \frac{2 \times P_{MAC,budget}}{E_{INT8,system} \times 10^{12}} = \frac{2 \times 218.85}{0.53} \approx 826\ \text{TOPS (INT8)}$$

The INT8 array is $826 / 438 \approx 1.9\times$ more power-efficient per TOPS — the reason all
production AI accelerators support INT8 as a primary inference format.

---

### Summary

| Phase | TDP (W) | MAC Util | Effective TOPS | Bottleneck |
|---|---|---|---|---|
| Prefill (BF16) | 199 W | ~70% | ~165 TOPS | Compute |
| Decode (BF16) | 72 W | ~13% | ~31 TOPS | HBM bandwidth |
| Decode (DVFS) | ~63 W | ~13% | ~31 TOPS | HBM bandwidth |

**Maximum TOPS within 300 W TDP: 438 TOPS (BF16), 826 TOPS (INT8).**

**Key interview takeaways:**

1. The MAC array overhead factor (registers, interconnect, local SRAM) typically multiplies the
   raw multiplier energy by 3–5×; never estimate power from the multiplier alone.
2. HBM PHY draws significant fixed power (~24 W) even when utilisation is low — power gating HBM
   stacks selectively is non-trivial due to PHY link training time on wakeup.
3. Decode phase operates at ~36% of prefill power, creating thermal headroom that can be used
   for larger batch sizes (more sequences in flight) to improve total throughput.
4. DVFS between phases helps, but less than $V^2 f$ suggests: decode can run at ~0.85× voltage and
   0.67× frequency, yet leakage, HBM and I/O power don't scale, so the voltage step saves only
   ~12% of the already-lower decode power.
