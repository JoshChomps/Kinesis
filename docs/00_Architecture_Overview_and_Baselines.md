# 00. Architecture Overview, Research Mission, and Quantitative Hardware Baselines

## Scientific Statement of Purpose: Hardware Capability Exploration

This document outlines an investigative research project exploring the theoretical limits and maximum capabilities of modern compute silicon (spanning NVIDIA Turing, Ampere, Ada Lovelace, and Blackwell GPU architectures, alongside modern CPU vector and matrix extensions).

> [!IMPORTANT]
> **Research Scope vs. Circumvention:**
> This project is **not** an attempt to "jailbreak", "crack", or bypass security controls or internal architectural blockers. Instead, it is an exploratory systems research investigation into:
> 1. **Silicon Operating Envelopes:** Measuring how architectural instruction issue rates (8-cycle vs. 16-cycle issue) shift physical roofline curves, and evaluating the true computational headroom of the underlying silicon dies.
> 2. **Compiler Heuristics & Workstation Profile Emulation:** Investigating how proprietary mathematical library heuristics (tile shapes, unroll depths, and memory staging) respond to different hardware profile queries, demonstrating that tile selection rather than hardware limits often constrains realized throughput.
> 3. **Sub-2-Bit Mathematical Precision & Manifold Rotations:** Designing branchless register unpackers (`lop3.b32`) and Riemannian Stiefel manifold rotations to compress weights by up to 28.44x while maintaining high output fidelity.
> 4. **Interconnect & Memory Fabric Co-Design:** Quantifying peer-to-peer DMA limits, Resizable BAR aperture utilization, and smart tiered CPU-L3 / GPU-VRAM caching.
>
> The sole objective is scientific: understanding the farthest boundaries of hardware capability and making existing computing resources as efficient and performant as physically possible.

---

## Executive Architectural Synthesis

Machine learning systems engineering on consumer and prosumer GPUs (specifically NVIDIA GA10x Ampere, AD10x Ada Lovelace, GB20x Blackwell, and legacy TU102 Turing silicon) has traditionally treated hardware issue-rate configurations, kernel compilers, PCIe bus drivers, and low-bit quantization formats as separate, isolated layers. Standalone investigations, including full-rate instruction issue profiling (`SS0 = 0x88888888`, `SS1 = 0x8`), open-source kernel driver Peer-to-Peer (P2P) DMA patches (`open-gpu-kernel-modules`), and sub-2-bit weight representations (Bonsai Binary g128 and BitNet b1.58), leave large amounts of end-to-end performance unexploited when each software or firmware layer remains unaware of the altered physical roofline of the others.

When a consumer GA102 (RTX 3090 or RTX 3090 Ti) or AD102 (RTX 4090) GPU is configured for full-rate instruction issue via its SEC2 Falcon speed selector latches (`SS0 = 0x88888888`, `SS1 = 0x8`), the hardware throughput of 16-bit floating-point (FP16 and BF16) matrix multiply-accumulate instructions with 32-bit accumulation (`HMMA.16816.F32`) matches workstation counterparts such as the RTX A6000 and RTX 6000 Ada. Raw PTX microbenchmarks confirm $1.94\times$ to $2.00\times$ throughput scaling across architectures:

- **RTX 3090 Ti (GA102, 450W):** Dense BF16 GEMM throughput rises from $87.5\text{ TFLOPS}$ to $157.4\text{ TFLOPS}$ ($+79.9\%$), while TF32 (`HMMA.1688.F32`) jumps from $42.58\text{ TFLOPS}$ to $82.61\text{ TFLOPS}$ ($+94.0\%$).
- **RTX 3090 (GA102, 350W-420W):** Dense BF16 GEMM rises from $77.5\text{ TFLOPS}$ to $129.6\text{ TFLOPS}$ ($+67.2\%$) and TF32 increases from $40.2\text{ TFLOPS}$ to $69.8\text{ TFLOPS}$.
- **RTX 2080 Ti (TU102, 250W):** Raw FP16 to FP32 MMA throughput doubles from $46.94\text{ TFLOPS}$ ($511.7\text{ FLOP/SM/clock}$) to $93.66\text{ TFLOPS}$ ($1,022.4\text{ FLOP/SM/clock}$), with $8192^3$ cuBLAS GEMM climbing from $48.15\text{ TFLOPS}$ to $85.17\text{ TFLOPS}$ ($+76.9\%$).
- **RTX 4090 (AD102, 450W):** Raw FP16/BF16 to FP32 Tensor Core throughput doubles from $165.6\text{ TFLOPS}$ to $331.6\text{ TFLOPS}$ ($+100.2\%$).

Despite this doubling of raw silicon throughput, running unpatched closed-source `cuBLASLt` yields only a $12\%$ to $34.4\%$ speedup on an unlocked RTX 4090, advancing $4096 \times 4096 \times 4096$ FP16/BF16 GEMM throughput from $153.2\text{ TFLOPS}$ to $206.0\text{ TFLOPS}$ and stranding $125.6\text{ TFLOPS}$ of unlocked hardware capacity. In end-to-end vLLM prefill benchmarks on a 27B W4A16 model at a locked $1,920\text{ MHz}$ core clock and $420\text{ W}$ power cap on RTX 3090:
- **2K context:** $1,344 \rightarrow 2,062\text{ tokens/s}$ ($+53.4\%$)
- **10K context:** $1,306 \rightarrow 2,016\text{ tokens/s}$ ($+54.4\%$)
- **90K context:** $909 \rightarrow 1,383\text{ tokens/s}$ ($+52.1\%$)

Power headroom strongly modulates realized gains: restricting an unlocked RTX 3090 to $250\text{ W}$ compresses the prefill speedup to $+36\%$, and capping it at $225\text{ W}$ reduces dense GEMM gains to $+25\%$.

---

## Ground-Truth Hardware & Workload Baseline Table

| GPU Architecture & Workload Regime | Stock Baseline Throughput | Full-Rate Issue (`SS0=0x88888888`) | Realized Gain (%) | Primary Bottleneck or Limiting Confounder |
| :--- | :--- | :--- | :--- | :--- |
| **RTX 4090 (AD102)** Raw PTX MMA (`FP16/BF16 -> FP32`, 450W) | $165.60\text{ TFLOPS}$ | $331.60\text{ TFLOPS}$ | $+100.2\%$ | Saturates full-rate 8-cycle SMSP Tensor Core issue ceiling. |
| **RTX 4090 (AD102)** Unpatched cuBLAS GEMM ($4096^3$, 450W) | $153.20\text{ TFLOPS}$ | $206.00\text{ TFLOPS}$ | $+34.5\%$ | $125.6\text{ TFLOPS}$ stranded by half-rate SMEM tile heuristics. |
| **RTX 3090 Ti (GA102)** Dense GEMM (`BF16 -> FP32`, $8192^3$, 450W) | $87.50\text{ TFLOPS}$ | $157.40\text{ TFLOPS}$ | $+79.9\%$ | Near-workstation RTX A6000 parity under 450W power envelope. |
| **RTX 3090 Ti (GA102)** Dense GEMM (`TF32 -> FP32`, $8192^3$, 450W) | $42.58\text{ TFLOPS}$ | $82.61\text{ TFLOPS}$ | $+94.0\%$ | Full-rate `HMMA.1688.F32` Tensor Core issue restoration. |
| **RTX 3090 (GA102)** Dense GEMM (`BF16 -> FP32`, 350W-420W) | $77.50\text{ TFLOPS}$ | $129.60\text{ TFLOPS}$ | $+67.2\%$ | Constrained by board power limit and unpatched SASS tiles. |
| **RTX 2080 Ti (TU102)** Raw PTX MMA (`FP16 -> FP32`, 250W) | $46.94\text{ TFLOPS}$ | $93.66\text{ TFLOPS}$ | $+99.5\%$ | Doubles per-SM throughput from $511.7$ to $1,022.4\text{ FLOP/SM/clk}$. |
| **RTX 3090** vLLM 27B (`W4A16 -> BF16`) Prefill (2K / 10K / 90K, 420W) | $1,344 / 1,306 / 909\text{ tok/s}$ | $2,062 / 2,016 / 1,383\text{ tok/s}$ | $+53.4\% / +54.4\% / +52.1\%$ | Dequantization overhead and half-rate `cuBLASLt` tile selection. |

---

## Five-Pillar Co-Design Architecture

1. **Pillar 1 (Compiler Roofline Recovery & Heuristic Analysis):** Resolves the SRAM-to-Tensor-Core ridge point shift ($4.0 \rightarrow 8.0\text{ FLOPs/Byte}$) via workstation heuristic evaluation (`compiler_unlock/libcublas_unlock_shim.c`) and custom 8-warp, 4-stage `sm_86`/`sm_89` Triton GEMM kernels (`compiler_unlock/unlocked_gemm.py`).
2. **Pillar 2 (Fused Sub-2-Bit Register Unpackers):** Replaces shared-memory LUTs and `dp4a` integer paths with 3-instruction branchless PTX unpackers for 1-bit Binary g128 (`shl.b32` + `lop3.b32` `0xAE` + `fma.rn.bf16x2`) and 4-instruction zero-preserving unpackers for 2-bit ternary BitNet b1.58 (`shr.u32` + `lop3.b32` `0xEA` + 2x `fma.rn.bf16x2`) feeding directly into 8-cycle `HMMA.16816.F32` instructions (`fused_w1_hadacore/w1a16_hadacore_kernel.cu`).
3. **Pillar 3 (Zero-Materialization QAT & Layer-Wise 8-Bit GaLore):** Aligns `BLOCK_K = 128` with the g128 quantization group size so forward and backward STE kernels quantize on-the-fly in SRAM with zero global VRAM allocated for $\widehat{W}$ (`qat_galore_engine/fused_qat_ste.py`), paired with per-layer backward hooks and 8-bit GaLore low-rank projection (`qat_galore_engine/galore_async_offload.py`).
4. **Pillar 4 (High-Throughput Interconnect Research & Driver BAR1 P2P):** Combines static BAR1 P2P (24GB/32GB cards), dynamic per-allocation BAR1 GMMU indirection for hardware-modded 48GB RTX 4090s ($\text{BAR1} = 32\text{ GB} < \text{FB} = 48\text{ GB}$), `GMMU_APERTURE_SYS_NONCOH` PTE remapping, and `dev_pm_ops.resume_early` hardware state persistence inside `open-gpu-kernel-modules` (`kernel_driver_patch/`).
5. **Pillar 5 (Structural Filters & Closed-Loop Heterogeneous Runtime):** Integrates offline Cayley-SGD Stiefel rotations (`spinquant_packer.py`), online $16 \times 16$ Tensor Core Walsh-Hadamard transforms (HadaCore), column-major L1 subspace energy sieving (`tiled_cbc_sieve.cu`), LServe sparse + 4-bit Hadamard KV attention (`lserve_sparse_attn.py`), 1GB HugePage zero-copy AVX-512/AMX MoE workers (`hetero_runtime/chorus_amx_moe.c`), and an NVRM clock-floor governor (`hetero_runtime/closed_loop_governor.py`).

---

## Two-Plane Language & Runtime Architecture: C11 / C++20 / CUDA / PTX vs. PyTorch

Every component in this repository is split between a **native compiled execution plane** and a **symbolic control/autograd plane** (documented in detail in [`docs/15_Language_and_Runtime_Architecture_C_CPP_CUDA_vs_PyTorch.md`](15_Language_and_Runtime_Architecture_C_CPP_CUDA_vs_PyTorch.md)):

- **Execution & Systems Plane (C11, Zero-Dependency C++20, CUDA C++, Inline PTX):** All latency-critical and hardware-facing subsystems (driver BAR1/S3 state machines, `LD_PRELOAD` cuBLAS ELF interposition, fused sub-2-bit `lop3.b32` + `HMMA.16816` Tensor Core kernels, warp butterfly sieves, lock-free 1 GB HugePage SPSC PCIe rings, AVX-512/AMX MoE tail workers, page-locked `ThreeTierArenaManager` / `SlabAllocator` memory primitives, GotoBLAS `4x8` AVX2 micro-kernels, and `vpshufb` T-MAC LUT execution) are implemented in C11, C++20, or CUDA/PTX with zero Python in the hot loop.
- **Control, Autograd & Calibration Plane (PyTorch / `libtorch` + Triton JIT):** Following the same design pattern as NVIDIA's own Megatron-LM, TransformerEngine, and TensorRT-LLM stacks, PyTorch is used as the high-level autograd graph compiler (`DeployBasisQATLinear` Straight-Through Estimator backward passes), 8-bit GaLore SVD/QR optimizer hook manager, offline Cayley-SGD Stiefel rotation solver, and Hugging Face SafeTensors checkpoint serializer, bridged into the C++20 engine via the capacity-checked `extern "C"` ABI in [`kinesis_unified/kinesis_c_abi.cpp`](../kinesis_unified/kinesis_c_abi.cpp).
