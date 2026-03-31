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

## Related Repositories

- **[LLM_Transformer_Decoder_RTL](https://github.com/BrendanJamesLynskey/LLM_Transformer_Decoder_RTL)** — End-to-end RTL implementation of a transformer decoder. While this repo focuses on interview questions and design trade-offs, the referenced repository contains a complete working implementation that brings these concepts together.

- **[Interview_SystemVerilog](https://github.com/BrendanJamesLynskey/Interview_SystemVerilog)** — SystemVerilog language features, verification constructs, and testbench architecture.

- **[Interview_Verilog](https://github.com/BrendanJamesLynskey/Interview_Verilog)** — Verilog fundamentals, RTL design patterns, and synthesis considerations.

- **[Interview_Digital_Hardware_Design](https://github.com/BrendanJamesLynskey/Interview_Digital_Hardware_Design)** — Foundational digital design, logic optimization, and hardware design methodology.

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
