# Grouped Query Attention (GQA) Hardware

## Background

Grouped Query Attention (GQA) is a generalisation of multi-head attention (MHA) and multi-query
attention (MQA) that reduces KV cache size and bandwidth requirements while preserving most of
the model quality of MHA.

**The family of attention variants**:

| Variant | Q heads | K heads | V heads | K/V sharing ratio |
|---------|---------|---------|---------|-------------------|
| MHA     | H       | H       | H       | 1:1               |
| GQA     | H       | G       | G       | H/G : 1           |
| MQA     | H       | 1       | 1       | H : 1             |

In GQA with G key/value head groups and H query heads, the H queries are divided into G groups of
H/G queries each. All queries within a group share the same K and V head. G=H recovers MHA; G=1
recovers MQA.

**Example — LLaMA-3 8B**: H=32 query heads, G=8 KV heads. Each of the 8 KV heads is shared by
32/8 = 4 query heads. The KV cache is 4x smaller than MHA with the same H and d_head.

**Mathematical formulation** (for GQA with group size G_size = H/G):
```
For group g = 0..G-1:
  K_g, V_g are shared by queries q_{g*G_size} .. q_{(g+1)*G_size - 1}

  For query i in group g:
    scores_{i,g} = q_i @ K_g^T / sqrt(d_head)   # [1, S]
    weights_{i,g} = softmax(scores_{i,g})         # [1, S]
    out_i = weights_{i,g} @ V_g                   # [1, d_head]
```

---

## Tier 1 — Fundamentals

### Q1: Quantify the KV cache memory reduction from GQA compared to MHA. Show the formula and calculate for LLaMA-3 8B.

**Answer**

**MHA KV cache size** (per layer, per token, at precision P bytes):
```
MHA_KV = 2 * H * d_head * P
```

**GQA KV cache size** (per layer, per token):
```
GQA_KV = 2 * G * d_head * P
```

**Reduction factor**:
```
Reduction = MHA_KV / GQA_KV = H / G
```

**LLaMA-3 8B example**: H=32 query heads, G=8 KV heads, d_head=128, BF16 (P=2 bytes), L=32 layers.

```
MHA would use (if it had H=32 KV heads):
  Per token = 2 * 32 * 128 * 2 = 16,384 bytes = 16 KB per layer
  All layers per token = 32 * 16 KB = 512 KB

GQA actually uses (G=8 KV heads):
  Per token = 2 * 8 * 128 * 2 = 4,096 bytes = 4 KB per layer
  All layers per token = 32 * 4 KB = 128 KB

Reduction factor = 32 / 8 = 4x
```

For a sequence of 8192 tokens:
```
MHA KV cache = 8192 * 512 KB = 4 GB
GQA KV cache = 8192 * 128 KB = 1 GB
```

This 4x reduction directly translates to 4x more sequences fitting in the same memory, or 4x
longer contexts for the same memory budget, or 4x reduction in KV cache bandwidth per decode step.

---

### Q2: In hardware, how does the attention computation change for GQA compared to MHA? Describe which data is shared and which is unique per query head.

**Answer**

**MHA computation** for H heads independently:
- Each head h has its own K_h, V_h vectors (both in cache).
- Each head h has its own Q_h vector.
- Score for head h, position j: `q_h @ k_{h,j}^T` — uses K from head h only.
- Weighted sum for head h: `sum_j(w_{h,j} * v_{h,j})` — uses V from head h only.

**GQA computation** for H query heads with G KV heads:
- Query heads are indexed 0..H-1; KV heads are indexed 0..G-1.
- Query head i belongs to KV group: `g = floor(i / (H/G))`
- Score for query head i, position j: `q_i @ k_{g,j}^T` — uses K from group g.
- Multiple query heads i1, i2, ..., i_{H/G} in the same group all read the same K_g and V_g.

**What is shared**: K_g and V_g are read once and used by H/G query heads. In hardware:
- The KV cache read for group g produces K_g[0..S-1] and V_g[0..S-1] once.
- These are routed (broadcast) to all H/G query heads in the group.
- H/G separate score computation engines run in parallel, each computing one head's scores
  against the shared K_g.
- H/G separate weighted sum engines accumulate, each using the same V_g data stream.

**What is unique per query head**: The Q vector q_i is unique per head (Q has H heads in all
variants). Score vectors and softmax weights are unique per query head (since q_i differs per head).

**Hardware broadcast**: The K and V data read from cache is broadcast to all H/G engines in a group.
This is a fan-out of H/G from the memory read port to the compute engines. For LLaMA-3 8B with
H/G = 4, each K/V read fans out to 4 score engines. The data bus width from cache to compute
must be `(H/G) * d_head * 2 = 4 * 128 * 2 = 1024 bytes` per position per cycle to sustain
full-throughput parallel computation.

---

### Q3: How does MQA (Multi-Query Attention) relate to GQA, and what are the hardware trade-offs of MQA versus GQA?

**Answer**

**MQA** is the special case G=1: there is exactly one K head and one V head, shared by all H
query heads. Introduced by Shazeer (2019) for fast autoregressive inference.

**Hardware trade-offs MQA vs GQA**:

**KV cache size**:
- MQA: `2 * 1 * d_head * P` per token = maximum reduction (H-fold vs MHA)
- GQA: `2 * G * d_head * P` per token, where 1 <= G <= H

**Bandwidth**:
- MQA: KV cache bandwidth is minimised (H-fold reduction vs MHA). For H=32, the bandwidth
  savings are 32x relative to MHA. The single K/V head is broadcast to all 32 query heads.
- GQA with G=8: 4x bandwidth reduction vs MHA. Less aggressive than MQA but substantially better.

**Model quality**:
- MQA has been reported to degrade perplexity by 5-10% relative to MHA on many tasks, especially
  tasks requiring diverse attention patterns across heads.
- GQA with G=8 (group size 4) typically achieves quality within 1-2% of MHA while providing
  significant memory/bandwidth savings.

**Hardware utilisation**:
- MQA: The single KV read must feed all H query engines. The fan-out is H (e.g., 32-way broadcast).
  Each query engine's score unit receives the same K vector. If all H score engines run in parallel,
  the K/V bus must have H-fold bandwidth, which is complex routing-wise.
- GQA: G separate KV reads, each feeding H/G engines. The maximum fan-out per read is H/G (e.g., 4).
  This is easier to route and buffer than H=32-way broadcast.

**Memory controller complexity**:
- MQA: Only 1 KV head to manage. Cache address computation is simpler (no head index in K/V address).
- GQA: G KV heads, each accessed by multiple query engines. The memory controller must map
  query head i to KV group `g = i * G / H` (integer multiply or lookup). This is a compile-time
  constant for a given model, so it can be hardwired or stored in a small configuration ROM.

**Practical recommendation**: GQA with G=H/4 or G=H/8 is the industry sweet spot (LLaMA-3, Mistral,
Gemma all use GQA). MQA is acceptable for very memory-constrained deployments or extremely long
contexts where the bandwidth savings outweigh the quality trade-off.

---

## Tier 2 — Intermediate

### Q4: Design the hardware datapath for GQA score computation. Show how the K/V data is routed from cache to multiple query head engines. Address bus width calculations.

**Answer**

**Parameters**: H=32 query heads, G=8 KV groups, d_head=128, BF16. Group size = H/G = 4.

**Data flow diagram**:

```
KV Cache SRAM
  |
  | (8 parallel reads, one per KV group)
  |
  +--KV_Group_0_read-->[ K[g0,pos], V[g0,pos] ]--+-->[Score Engine Q0]
  |                                                +-->[Score Engine Q1]
  |                                                +-->[Score Engine Q2]
  |                                                +-->[Score Engine Q3]
  |
  +--KV_Group_1_read-->[ K[g1,pos], V[g1,pos] ]--+-->[Score Engine Q4]
  |                                                +-->[Score Engine Q5]
  |                                                +-->[Score Engine Q6]
  |                                                +-->[Score Engine Q7]
  ...
  +--KV_Group_7_read-->[ K[g7,pos], V[g7,pos] ]--+-->[Score Engine Q28]
                                                   +-->[Score Engine Q29]
                                                   +-->[Score Engine Q30]
                                                   +-->[Score Engine Q31]
```

**Bus width calculations**:

KV cache read for one group at one position:
- K data: `d_head * 2 = 128 * 2 = 256 bytes`
- V data: `d_head * 2 = 256 bytes`
- Total: 512 bytes per group per position

G=8 groups read in parallel: `8 * 512 = 4096 bytes per position per cycle`

This is a 32,768-bit (4 KB) wide read bus from KV cache to the score engines. For SRAM operating
at 1 GHz, this requires either:
- A single 4 KB-wide SRAM port (unusual, but feasible for on-chip SRAM)
- 8 separate SRAM banks, one per group, each with a 512-byte (4096-bit) read port
- Reading one group per cycle and buffering in registers (8x slower, requires 8 buffering cycles
  per position before all groups' engines can proceed)

**Practical implementation**: Bank the KV SRAM by group. Each of the G=8 banks holds all positions
for one KV group. Each bank has an independent read port. All 8 banks are read simultaneously
each cycle, delivering K and V for all 8 groups in parallel.

**SRAM bank addressing**:
```
bank_id = kv_group_id                                    // 0..7
bank_address = layer_id * S_max * token_stride           // base
             + position * token_stride                   // sequential scan
             + kv_offset                                 // 0 for K, d_head*2 for V
```

The addressing logic is combinatorial (one adder per bank address component), computed one
position ahead using a pipeline register that holds the next address.

---

### Q5: Calculate the bandwidth savings of GQA at decode time and show how this enables either larger batches or longer contexts on a fixed HBM budget.

**Answer**

**Setup**: Single H100 GPU, 80 GB HBM, 3.35 TB/s bandwidth. Model: LLaMA-3 70B (H=64 query heads,
G=8 KV heads, d_head=128, L=80 layers, BF16 weights).

**Model weight size** (approximate):
```
~140 GB (BF16) — does not fit in 80 GB HBM
```
In practice, LLaMA-3 70B is typically run with INT4/INT8 quantisation or split across GPUs.
For this analysis, let's use LLaMA-3 8B (H=32, G=8, d_head=128, L=32) on a single A100 80 GB.

**A100 specs**: 80 GB HBM, 2 TB/s bandwidth.
Model weights (LLaMA-3 8B, BF16): ~16 GB.
Remaining for KV cache: 80 - 16 = 64 GB.

**MHA KV cache per token** (hypothetical, H=32 KV heads):
```
32 * 128 * 2 * 2 (K+V) * 32 (layers) = 512 KB per token
```
Maximum batch * context with 64 GB: `B * S = 64 GB / 512 KB = 128,000 token-slots`

**GQA KV cache per token** (G=8 KV heads):
```
8 * 128 * 2 * 2 (K+V) * 32 (layers) = 128 KB per token
```
Maximum batch * context with 64 GB: `B * S = 64 GB / 128 KB = 524,288 token-slots`

**Comparison**: GQA provides 4x more total token capacity. This enables:
- Option 1 (same batch size): B=16 -> context from `128,000/16 = 8,000` to `524,288/16 = 32,768` tokens
- Option 2 (same context): S=4096 -> batch from `128,000/4096 = 31` to `524,288/4096 = 128` sequences

**Bandwidth at decode time** (one decode step, S=4096, B=32):

MHA (hypothetical): `32 * 4096 * 512 KB = 65 GB per step across all layers`
At 2 TB/s: 32.5 ms per step.

GQA: `32 * 4096 * 128 KB = 16 GB per step`
At 2 TB/s: 8 ms per step — 4x faster decode.

**Tokens per second**: 32 sequences * 1/8ms = 4,000 tokens/sec (GQA) vs 1,000 tokens/sec (MHA).
This is the direct business impact of GQA on serving throughput.

---

### Q6: How does GQA affect the area of the attention engine compared to MHA? Analyse compute area, KV cache SRAM area, and routing complexity.

**Answer**

**Compute area: Score computation**

MHA: H independent score engines, each computing `q_h @ K_h^T`. Total: H * d_head MACs per position.
GQA: H score engines (still H query heads), each computing `q_i @ K_g^T`. Total: H * d_head MACs
per position — identical to MHA.

The score computation area is the same for GQA and MHA (same number of query heads means same
number of MAC units required for full parallel computation). GQA saves no area in the score
compute engines.

**Compute area: Weighted sum**

MHA: H independent weighted sum engines, each accumulating against V_h. Total: H * d_head MACs.
GQA: H weighted sum engines, but all engines in a group accumulate against shared V_g. The
weights differ per engine, but V data is broadcast. Total MAC count: H * d_head — same as MHA.

Again, no MAC area savings in the weighted sum engines.

**KV cache SRAM area**

MHA: SRAM for H * d_head * 2 * S_max values = proportional to H.
GQA: SRAM for G * d_head * 2 * S_max values = proportional to G.

Area saving: H/G factor. For H=32, G=8: 4x SRAM area reduction. This is the dominant area
saving from GQA in hardware. On-chip KV SRAM is typically one of the larger SRAM structures
in an attention engine.

**Routing complexity**

MHA: Each K/V SRAM bank is connected to exactly one score engine. Routing is 1:1 (no broadcast).
GQA: Each K/V SRAM bank (one per KV group) is connected to H/G score engines (fan-out = H/G).
For H/G=4: each bank's output is fanned out to 4 engines. Wire length and buffer area scale
with fan-out. For modest fan-out (2-8), this is acceptable. For MQA (fan-out = H = 32), routing
becomes non-trivial and may require explicit bus buffers or repeated drivers.

**Summary table**:

| Resource            | MHA  | GQA (G=H/4) | MQA (G=1) |
|---------------------|------|-------------|-----------|
| Score compute area  | 1x   | 1x          | 1x        |
| Weighted sum area   | 1x   | 1x          | 1x        |
| KV SRAM area        | 1x   | 0.25x       | 1/H x     |
| KV cache bandwidth  | 1x   | 0.25x       | 1/H x     |
| K/V routing fan-out | 1    | H/G = 4     | H = 32    |

GQA's primary hardware benefit is the KV SRAM and bandwidth reduction, not compute area.

---

## Tier 3 — Advanced

### Q7: An accelerator is being designed to support both MHA models (e.g., GPT-4-era) and GQA models (e.g., LLaMA-3). How would you design the attention engine to be flexible across both, without duplicating hardware?

**Answer**

**Key observation**: MHA is GQA with G=H. A hardware design parameterised by G (number of KV
heads) that supports G=1..H subsumes all three variants (MHA, GQA, MQA) in one design.

**Flexible attention engine architecture**:

**Configuration registers** (loaded at model startup):
```
reg [5:0] cfg_num_q_heads;    // H (1-64)
reg [5:0] cfg_num_kv_heads;   // G (1-H), must divide H
reg [6:0] cfg_head_dim;       // d_head (32, 64, 128, 256)
reg [5:0] cfg_group_size;     // H/G, computed at startup
```

**KV cache controller**: Parameterised by G. For G=H (MHA), G=8 (GQA), or G=1 (MQA), the same
controller reads from G banks with configurable base addresses. The bank assignment for KV group
g is computed as: `kv_bank = g % cfg_num_kv_heads` — for MHA this means kv_bank = query_head,
for GQA it maps multiple query heads to the same bank.

**Routing mux** between KV cache and score engines: A configurable broadcast network.
For each score engine i, its KV group is: `g_i = i / cfg_group_size` (integer divide).
In hardware this is a MUX: select which of the G KV data streams drives score engine i.
MUX select = `i >> log2(cfg_group_size)` (a right-shift, valid only when group_size is power of 2).

To support arbitrary G (not just powers of 2): use a small 6-bit comparison
`(query_head_id * cfg_num_kv_heads) / cfg_num_q_heads` — this requires a division but can be
pre-computed at configuration time for all H query heads and stored in a 64-entry lookup table.
The table is only 64 * 6 bits = 48 bytes — negligible.

**KV SRAM sizing**: The SRAM must accommodate the maximum G (i.e., MHA with G=H_max=64). For
GQA models with G<H, only G banks are populated and H-G banks are unused — they can be power-gated
to reduce static power. Dynamic bank disabling saves ~(H-G)/H of KV SRAM power.

**QKV projection flexibility**: The W_K and W_V weight matrices are of shape [d_model, G*d_head]
for GQA, versus [d_model, H*d_head] for MHA. The GEMV unit must support variable output dimensions.
This is naturally handled by a GEMV unit with a configurable output tile size: set the tile width
to `cfg_num_kv_heads * cfg_head_dim` for K and V projections, and `cfg_num_q_heads * cfg_head_dim`
for Q projection. The same GEMV hardware handles both.

---

### Q8: Quantify the arithmetic intensity of GQA versus MHA attention during decode, and explain the implications for hardware efficiency.

**Answer**

**Attention arithmetic intensity** (decode, one step, sequence length S, batch size B):

**Per-head MAC count** for one query head computing scores and weighted sum:
```
Scores: S * d_head MACs
Weighted sum: S * d_head MACs
Total: 2 * S * d_head FLOPs per query head
Total for H heads: 2 * H * S * d_head FLOPs
```

**Bytes read from KV cache**:

MHA (H KV heads):
```
K read: S * H * d_head * 2 bytes
V read: S * H * d_head * 2 bytes
Total: 4 * S * H * d_head bytes
```

GQA (G KV heads), with each KV head read by H/G query heads:
```
K read: S * G * d_head * 2 bytes  (read once per KV head, broadcast to H/G engines)
V read: S * G * d_head * 2 bytes
Total: 4 * S * G * d_head bytes
```

**Arithmetic intensity**:

MHA:
```
AI = 2 * H * S * d_head FLOPs / (4 * S * H * d_head bytes)
   = 0.5 FLOPs/byte
```

GQA:
```
AI = 2 * H * S * d_head FLOPs / (4 * S * G * d_head bytes)
   = H / (2 * G) FLOPs/byte
   = (H/G) / 2 FLOPs/byte
```

For LLaMA-3 8B (H=32, G=8): `AI = (32/8) / 2 = 2 FLOPs/byte`

**Interpretation**: GQA increases arithmetic intensity by a factor of H/G relative to MHA.
For H/G=4, GQA has 4x higher arithmetic intensity than MHA during decode attention. This means:

- MHA at 0.5 FLOPs/byte is severely bandwidth-bound on all modern hardware (roofline ~30-300 FLOPs/byte).
- GQA at 2 FLOPs/byte is still bandwidth-bound but less severely so. The hardware's compute units
  are somewhat better utilised.
- MQA at H/2 = 16 FLOPs/byte approaches the roofline for some accelerators, potentially shifting
  the bottleneck partially to compute.

**Hardware efficiency implication**: GQA's increased arithmetic intensity means more useful work
is done per byte fetched from HBM. The compute units (score MACs, weighted sum MACs) are more
heavily utilised because each K/V value read from cache feeds H/G score computations rather than 1.
This directly improves the hardware utilisation metric (actual FLOP/s / peak FLOP/s), which is the
primary efficiency figure of merit for AI accelerators.

For an accelerator rated at 200 TFLOP/s compute and 2 TB/s bandwidth (roofline = 100 FLOPs/byte):
- MHA: compute utilisation = 0.5 / 100 = 0.5% (nearly all time spent waiting for memory)
- GQA (H/G=4): compute utilisation = 2 / 100 = 2% (still low, but 4x better than MHA)
- MQA (H/G=32): compute utilisation = 16 / 100 = 16% (approaching meaningful utilisation)

This analysis underscores that even with GQA, decode attention is fundamentally memory-bound, and
future efficiency gains require either higher memory bandwidth or further attention algorithm changes
(sparse attention, linear attention) that reduce the O(S) memory access pattern entirely.
