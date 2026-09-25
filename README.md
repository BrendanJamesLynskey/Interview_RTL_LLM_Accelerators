# RTL Design for LLM Accelerators — Interview Preparation

![Subject Badge](https://img.shields.io/badge/Subject-RTL%20Design%20for%20LLM%20Accelerators-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This repository contains comprehensive interview preparation materials for RTL design and verification of LLM hardware accelerators. The focus spans the full spectrum of accelerator design: from inference fundamentals and architectural trade-offs, through detailed datapath and memory subsystem design, to complete system integration and verification methodology.

Topics covered include:

- **LLM Inference Fundamentals**: Compute profiles, memory bandwidth analysis, prefill vs. decode pipeline stages, and quantization impact on hardware
- **Datapath Design**: Systolic arrays, MAC unit architectures, mixed-precision computation, and matrix operation strategies
- **Attention Engine**: Hardware implementation of multi-head attention, softmax pipelines, KV cache management, RoPE units, and grouped-query attention
- **Memory and Dataflow**: Tiling strategies, DMA engine design, double-buffering, weight streaming, and on-chip memory hierarchies
- **Nonlinear Operations**: Hardware acceleration of activation functions, normalization layers, LUT-based and CORDIC approximations
- **System Integration**: Full SoC architecture, host interfaces (PCIe/AXI), tokenizer and embedding hardware, end-to-end inference pipelines
- **Verification**: Golden models, bit-accurate Python references, constrained random testing, and coverage strategies

## Table of Contents

- [01 - LLM Inference Fundamentals](#01---llm-inference-fundamentals)
- [02 - Datapath Design](#02---datapath-design)
- [03 - Attention Engine](#03---attention-engine)
- [04 - Memory and Dataflow](#04---memory-and-dataflow)
- [05 - Nonlinear and Normalisation](#05---nonlinear-and-normalisation)
- [06 - System Integration](#06---system-integration)
- [07 - Verification](#07---verification)
- [08 - Quizzes](#08---quizzes)
- [How to Use](#how-to-use)
- [Simulating the SystemVerilog Challenges](#simulating-the-systemverilog-challenges)
- [Related Repositories](#related-repositories)
- [Contributing](#contributing)
- [License](#license)

## 01 - LLM Inference Fundamentals

Foundation for understanding accelerator design constraints and opportunities.

- `transformer_compute_profile.md` — Characterizing FLOPs, memory bandwidth, and arithmetic intensity across prefill and decode phases
- `prefill_vs_decode_stages.md` — Pipeline implications of generation vs. prompt processing
- `memory_bandwidth_bottlenecks.md` — Identifying bandwidth-bound vs. compute-bound operations
- `quantisation_for_hardware.md` — Quantization strategies and their impact on accelerator design
- `worked_problems/` — Roofline analysis, KV cache sizing, arithmetic intensity calculations

## 02 - Datapath Design

Core computational building blocks and their integration.

- `systolic_array_architecture.md` — Systolic array principles, timing, and scalability
- `matrix_vector_vs_matrix_matrix.md` — Comparing matrix operation strategies for inference
- `mac_unit_design.md` — MAC (multiply-accumulate) unit architectures and optimization
- `mixed_precision_datapath.md` — FP16, INT8, and mixed-precision datapath design
- `coding_challenges/` — Implement systolic arrays, MAC units with mixed precision, and accumulator trees in SystemVerilog

## 03 - Attention Engine

Specialized hardware for the dominant computational bottleneck in transformer inference.

- `multi_head_attention_hardware.md` — Partitioning and parallelization strategies for multi-head attention
- `softmax_hardware_implementation.md` — Pipelined softmax computation and numerical stability
- `kv_cache_management.md` — KV cache organization, eviction, and prefetch strategies
- `rope_hardware.md` — Rotary positional embeddings in hardware
- `grouped_query_attention.md` — Hardware support for GQA variants
- `coding_challenges/` — Build softmax pipelines, KV cache controllers, and RoPE units

## 04 - Memory and Dataflow

Storage hierarchy and data movement patterns that dominate power and latency.

- `dataflow_architectures.md` — Systolic, dataflow-centric, and stream-based architectures
- `on_chip_memory_hierarchy.md` — SRAM banks, reuse buffers, and capacity planning
- `dma_and_tiling_strategies.md` — DMA-driven tiling for on-chip memory constraints
- `weight_streaming_vs_caching.md` — Trade-offs between weight reuse and streaming
- `coding_challenges/` — Implement tile schedulers, double-buffer controllers, and AXI DMA engines

## 05 - Nonlinear and Normalisation

Hardware for activations, normalizations, and other nonlinear operations.

- `activation_functions_hardware.md` — GELU, ReLU, SiLU, and other activation hardware implementations
- `rmsnorm_and_layernorm.md` — Normalization layer hardware design
- `lut_and_cordic_methods.md` — LUT-based and CORDIC approaches for transcendental functions
- `coding_challenges/` — Implement GELU approximations, RMSNorm pipelines, and exponential LUT interpolation

## 06 - System Integration

Full accelerator SoC architecture and host communication.

- `accelerator_soc_architecture.md` — Block diagram, control flow, and clock domains
- `host_interface_pcie_axi.md` — PCIe and AXI slave interface design
- `tokenizer_and_embedding_hw.md` — Hardware accelerators for tokenization and embedding lookup
- `end_to_end_inference_pipeline.md` — Orchestrating prefill, decode, and output stages
- `worked_problems/` — Throughput estimation, latency budgets, power envelope analysis

## 07 - Verification

Methodology for validating accelerator correctness and performance.

- `golden_model_methodology.md` — Designing golden models for accelerator verification
- `bit_accurate_python_reference.md` — Bit-accurate Python reference implementations
- `constrained_random_for_accelerators.md` — Constrained-random testing for compute-heavy designs
- `coverage_strategy.md` — Coverage metrics for accelerator verification
- `coding_challenges/` — Build matmul golden models, softmax testbenches, and coverage plans
  - [`softmax_golden.c`](07_verification/coding_challenges/softmax_golden.c) — DPI-C golden model linked with the softmax testbench

## 08 - Quizzes

Self-assessment quizzes covering each major topic area.

- `quiz_inference_fundamentals.md`
- `quiz_datapath.md`
- `quiz_attention_engine.md`
- `quiz_memory_dataflow.md`
- `quiz_system_design.md`

## How to Use

This repository is designed as an interview preparation guide for RTL design engineers working on LLM accelerators:

1. **Start with fundamentals**: Begin with section 01 to build a solid understanding of LLM inference characteristics and why certain hardware architectures emerge as solutions.

2. **Understand constraints**: Section 02-05 dive deep into datapath, memory, and computational subsystems. These sections explain the "why" behind design decisions rather than prescribing a single correct answer.

3. **Design trade-offs**: Each topic includes discussion of competing approaches—systolic vs. dataflow, weight streaming vs. caching, pipelining depth vs. area. Use these to develop intuition for trade-off analysis during interviews.

4. **Worked problems**: Sections 01 and 06 include worked problems with step-by-step solutions for quantitative reasoning about performance, power, and area.

5. **Coding challenges**: Sections 02-07 include SystemVerilog and Python coding challenges. These are simplified but realistic problems that exercise the concepts covered in the preceding material.

6. **Verification methodology**: Section 07 emphasizes that verification is not an afterthought but an integral part of accelerator design. Study this in parallel with datapath and system sections.

7. **Self-assess**: Use the quizzes in section 08 to identify weak areas and revisit relevant sections.

**Study approach**: This is not a walkthrough of an existing codebase. Instead, each section presents design decisions, trade-offs, and implementation considerations. Use this to build mental models during interview preparation, then reason through new problems from first principles.

## Simulating the SystemVerilog Challenges

Every `.sv` coding challenge here is a design plus a self-checking testbench in one file. All fourteen have been simulated and pass (September 2026) and are clean in slang: five in Verilator, nine in the Vivado simulator (xsim), which is the only free simulator that can run them. The Python golden model in `07_verification` needs Python 3 with NumPy.

| Challenge | Testbench top | Passes in | Verilator 5.020 | Icarus 12 |
|-----------|---------------|-----------|-----------------|-----------|
| `02_datapath_design/…/challenge_01_systolic_array.sv` | `tb_systolic_array` | **xsim 2025.2** | ✗ `disable` of a named block from another `fork` branch | ✗ unpacked-array task/function ports |
| `02_datapath_design/…/challenge_02_mac_unit_fp16_int8.sv` | `tb_mac_unit_mixed_precision` | **xsim 2025.2** | ✗ fixed array passed to an open-array `[]` argument (C++ compile error) | ✗ `break` |
| `02_datapath_design/…/challenge_03_accumulator_tree.sv` | `tb_accumulator_tree` | Verilator 5.020 | ✓ | ✗ unpacked-array task ports |
| `03_attention_engine/…/challenge_01_softmax_pipeline.sv` | `tb_softmax_pipeline` (`-d SIMULATION`) | **xsim 2025.2** | ✗ named `disable` across `fork` branches | ✗ `automatic` lifetime override |
| `03_attention_engine/…/challenge_02_kv_cache_controller.sv` | `tb_kv_cache_controller` (`-d SIMULATION`) | **xsim 2025.2** | ✗ named `disable` across `fork` branches | ✗ named task arguments `.t(…)` |
| `03_attention_engine/…/challenge_03_rope_unit.sv` | `tb_rope_unit` (`-d SIMULATION`) | **xsim 2025.2** | ✗ named `disable` across `fork` branches | ✗ unpacked-array task ports |
| `04_memory_and_dataflow/…/challenge_01_tile_scheduler.sv` | `tb_tile_scheduler` | **xsim 2025.2** | ✗ `disable` of named `fork` blocks | ✗ concurrent assertions; unpacked structs |
| `04_memory_and_dataflow/…/challenge_02_double_buffer_controller.sv` | `tb_double_buffer_controller` | **xsim 2025.2** | ✗ `disable` of named `fork` blocks | ✗ concurrent assertions |
| `04_memory_and_dataflow/…/challenge_03_axi_dma_engine.sv` | `tb_axi_dma_engine` | Verilator 5.020 | ✓ | ✗ `automatic` lifetime override |
| `05_nonlinear_and_normalisation/…/challenge_01_gelu_approximation.sv` | `tb_gelu_approximation` | Verilator 5.020 | ✓ | ✗ unpacked-array `localparam` with an initialiser |
| `05_nonlinear_and_normalisation/…/challenge_02_rmsnorm_pipeline.sv` | `rmsnorm_pipeline_tb` | Verilator 5.020 | ✓ | ✗ `automatic` lifetime override |
| `05_nonlinear_and_normalisation/…/challenge_03_exp_lut_interpolation.sv` | `exp_lut_interpolation_tb` | Verilator 5.020 | ✓ | ✗ `break` |
| `07_verification/…/challenge_02_softmax_testbench.sv` | `softmax_testbench` | **xsim 2025.2** + DPI-C (`softmax_golden.c`) | ✗ “Unsupported: covergroup”; virtual-interface / clocking-block failures | ✗ unpacked structs |
| `07_verification/…/challenge_03_coverage_plan.sv` | `coverage_plan_tb` | **xsim 2025.2** | ✗ “Unsupported: covergroup” | ✗ interface as a module port |

### Tools, and why each was needed

| Tool | Version used | Used for | Why it was needed |
|------|--------------|----------|-------------------|
| **slang** (`pip install pyslang`) | pyslang 11.0 | Legality check of every `.sv` file | A complete IEEE 1800-2017 front end: it finds code that is not legal SystemVerilog, with exact line numbers, in seconds. It does **not** simulate, so it cannot find functional bugs — and it passed a few illegal constructs that xsim rejected (an `always_ff` variable with a second driver, use before declaration, an out-of-range constant index) |
| **Verilator** | 5.020 (Ubuntu 24.04 package, `--binary --timing --assert`) | Simulating the synthesisable-style challenges with procedural testbenches | Free, fast, and supports what those testbenches use (delays, `fork`, `\|=>`/`$past` assertions). It cannot run the rest: no covergroups, no `##` in sequences, no `disable` of a named block from another `fork` branch, and failures on parameterised virtual interfaces / clocking blocks |
| **AMD Vivado simulator (xsim)** | Vivado 2025.2 (free ML Standard edition) | Simulating the challenges Verilator and Icarus cannot run | The only free simulator available that supports classes, mailboxes, virtual interfaces with clocking blocks, covergroups, `##` sequences, named `disable`, array arguments and DPI-C together |
| Icarus Verilog | 12.0 | Tried on every file | Compiles none of these fourteen: it rejects unpacked-array task ports, `break`, concurrent assertions, `automatic` lifetime overrides, named task arguments, unpacked structs, interface ports and initialised unpacked-array `localparam`s (see the table) |

### Commands

slang (legality):
```bash
python3 -c "from pyslang.syntax import SyntaxTree; from pyslang.ast import Compilation
c = Compilation(); c.addSyntaxTree(SyntaxTree.fromFile('<challenge>.sv'))
print([str(d.code) for d in c.getAllDiagnostics() if d.isError()])"   # [] = clean
```

Verilator (the five plain-RTL challenges):
```bash
verilator --binary --timing --assert -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH \
          -j 1 --top-module <tb_top> <challenge>.sv && obj_dir/V<tb_top>
```
(`-j 1`: parallel builds of 5.020 crash intermittently.)

xsim (the other nine):
```bash
source /opt/Xilinx/2025.2/Vivado/settings64.sh     # puts xvlog / xelab / xsim / xsc on PATH
mkdir work && cd work                              # one work directory per design
xvlog -sv [-d SIMULATION] <challenge>.sv           # -d SIMULATION where the TB is inside `ifdef SIMULATION
xelab <tb_top> -s snap
xsim snap -R
```

For the UVM-lite softmax testbench, build the DPI-C golden model first and link it:
```bash
xsc softmax_golden.c                               # builds xsim.dir/work/xsc/dpi.so
xvlog -sv challenge_02_softmax_testbench.sv
xelab softmax_testbench -sv_lib dpi -s snap && xsim snap -R
```

xsim 2025.2 quirks these files work around: `$urandom(seed)` as a statement is rejected (the UVM-lite testbench seeds with `process::self().srandom(seed)`); `shortreal` is held at double precision (the MAC reference rounds each step through `$shortrealtobits`); `iff` on `illegal_bins`/`ignore_bins` is ignored (the coverage plan checks its NaN rule with an assertion); `option.cross_auto_bin_max` is rejected (an `ignore_bins` complement is used). Also avoid size casts such as `16'(a - b)` in continuous assignments: xsim does not truncate them.

The full construct-by-construct comparison, with every error message, is in the [SystemVerilog_Simulators](https://github.com/BrendanJamesLynskey/SystemVerilog_Simulators) presentation.

## Related Repositories

- **[LLM_Transformer_Decoder_RTL](https://github.com/BrendanJamesLynskey/LLM_Transformer_Decoder_RTL)** — End-to-end RTL implementation of a transformer decoder. While this repo focuses on interview questions and design trade-offs, the referenced repository contains a complete working implementation that brings these concepts together.

- **[Interview_SystemVerilog](https://github.com/BrendanJamesLynskey/Interview_SystemVerilog)** — SystemVerilog language features, verification constructs, and testbench architecture.

- **[Interview_Verilog](https://github.com/BrendanJamesLynskey/Interview_Verilog)** — Verilog fundamentals, RTL design patterns, and synthesis considerations.

- **[Interview_Digital_Hardware_Design](https://github.com/BrendanJamesLynskey/Interview_Digital_Hardware_Design)** — Foundational digital design, logic optimization, and hardware design methodology.

- **[SystemVerilog_Simulators](https://github.com/BrendanJamesLynskey/SystemVerilog_Simulators)** — Which free simulators (Icarus, Verilator, Vivado xsim, Questa Starter) can run the coding challenges in this repository, what each one rejects, and the status of every challenge.

## Contributing

Contributions are welcome. This repository is a study resource, and improvements that clarify concepts, add depth to explanations, or extend coverage to additional accelerator architectures are valuable.

To contribute:

1. Fork the repository
2. Create a branch for your changes
3. Ensure all explanations are clear and technically accurate
4. Submit a pull request with a description of your additions or corrections

Please maintain consistency with the existing structure and style. Each section should include both conceptual material and practical problems.

## License

This repository is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Copyright 2026 Brendan Lynskey
