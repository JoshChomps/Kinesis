# Unified Hardware Acceleration (UHA): Systems Co-Design & Extreme Hardware Capability Research

## Investigative Research Mission & Scientific Scope

This repository is an academic and systems engineering research project dedicated to exploring the theoretical limits and maximum capabilities of modern compute silicon. It investigates how close compute hardware (consumer and workstation GPUs, along with modern CPU vector and matrix extensions) can get to physical hardware rooflines when every layer of the compute stack is co-designed from first principles.

### Explicit Research Statement: Capability Exploration vs. Circumvention

> [!IMPORTANT]
> **Scientific Scope and Intent:**
> This project is **not** an attempt to "jailbreak", "crack", or bypass security controls or internal architectural blockers. Rather, it is a first-principles investigative research effort focused on:
> 1. **Hardware Rooflines & Architectural Headroom:** Quantifying the physical limits of memory bandwidth, SRAM crossbars, cache hierarchies, and functional units across different silicon operating configurations.
> 2. **Compiler Heuristics & Profile Emulation:** Investigating how proprietary mathematical library heuristics (tile geometries, unroll depths, and threadblock schedules) impact performance when evaluating professional versus consumer hardware profiles.
> 3. **Sub-2-Bit Mathematical Precision:** Pushing algorithmic boundaries with 1.125-bit Binary and 2.125-bit Ternary formats, Stiefel manifold rotations, and in-SRAM zero-materialization QAT.
> 4. **Heterogeneous Interconnect Co-Design:** Exploring peer-to-peer PCIe DMA transfer limits, Resizable BAR aperture utilization, and smart CPU L3 / GPU VRAM tiered caching.
>
> The objective is strictly scientific: understanding the farthest boundaries of hardware capability and making existing computing resources as efficient and performant as physically possible.

---

## Repository Structure

### 1. Architecture & Research Documentation (`docs/`)
- [`docs/00_Architecture_Overview_and_Baselines.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/00_Architecture_Overview_and_Baselines.md): Executive architectural synthesis and quantitative hardware baselines across GA102, AD102, TU102, and GB202.
- [`docs/01_Pillar1_Silicon_Unlock_and_SRAM_Roofline.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/01_Pillar1_Silicon_Unlock_and_SRAM_Roofline.md): Full-rate silicon architecture, SRAM crossbar ($128\text{ B/cycle/SM}$) vs. 8-cycle `HMMA.16816.F32` roofline derivations, compiler heuristic profiling, and 8-warp/4-stage Triton GEMM tile geometry.
- [`docs/02_Pillar2_Sub2Bit_PTX_Unpackers_and_HadaCore.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/02_Pillar2_Sub2Bit_PTX_Unpackers_and_HadaCore.md): IEEE-754 `bf162` bit geometry, Interleaved 16-Bit Split layout, 3-instruction branchless `lop3.b32` (`0xAE` and `0xEA`) unpackers, and $14.22\times$ DRAM ridge point shift.
- [`docs/03_Pillar3_Zero_Materialization_QAT_and_GaLore.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/03_Pillar3_Zero_Materialization_QAT_and_GaLore.md): In-SRAM `BLOCK_K = 128` Fused STE forward/backward kernels and per-layer post-accumulation 8-bit GaLore fitting a 32B QAT run inside $23.63\text{ GB/GPU}$ ($22.00\text{ GiB/GPU}$) on 2x 24 GiB GPUs.
- [`docs/04_Pillar4_PCIe_BAR1_P2P_and_Modded_48GB_Driver.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/04_Pillar4_PCIe_BAR1_P2P_and_Modded_48GB_Driver.md): High-throughput interconnect research: GSP registry profiling (`RMDisableFeatureDisablement=1`), static 24GB/32GB BAR1 P2P, dynamic BAR1 GMMU indirection (`kbusMapFbAperture_HAL`) for modded 48GB RTX 4090s, and S3 `resume_early` state recovery.
- [`docs/05_Pillar5_Structural_Filters_and_Chorus_AMX_Runtime.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/05_Pillar5_Structural_Filters_and_Chorus_AMX_Runtime.md): SpinQuant Cayley-SGD Stiefel rotations, HadaCore $16 \times 16$ Tensor Core Walsh-Hadamard prologue, Tiled-CBC subspace energy sieve, LServe sparse + 4-bit Hadamard KV attention, and Chorus + NVRM closed-loop runtime.
- [`docs/06_Reference_Index_and_Bibliography.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/06_Reference_Index_and_Bibliography.md): Indexed bibliography and technical source table.
- [`docs/07_Security_Risks_Prior_Art_and_Literature.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/07_Security_Risks_Prior_Art_and_Literature.md): Security risk evaluation, hardware/electrical operating envelope boundaries, ephemeral open-and-relock MMIO containment, isolated IOMMU BDF whitelisting, stock recovery procedures, and prior-art comparison across GitHub, Hugging Face, and academic literature.
- [`docs/08_Kinesis_Evaluation_Fixes_and_Smart_Cache.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/08_Kinesis_Evaluation_Fixes_and_Smart_Cache.md): Evaluation of the legacy `AI_Library` (Kinesis) C++20 engine, remediation of its 6 architectural/numerical flaws, and design of the unified CPU + Single-GPU Smart Tiered Cache.
- [`docs/09_First_Principles_Firmware_Hardware_Code_Review.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/09_First_Principles_Firmware_Hardware_Code_Review.md): Deep firmware, GPU PTX, CPU ISA, and numerical audit catalog (`P0-1..P0-6`, `P1-1..P1-5`).
- [`docs/10_Related_Projects_Community_Buzz_and_Wave2_Upgrades.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/10_Related_Projects_Community_Buzz_and_Wave2_Upgrades.md): Related community/academic systems comparison and Wave 2 architectural upgrades.
- [`docs/11_Adversarial_Review_Status_and_Known_Limitations.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/11_Adversarial_Review_Status_and_Known_Limitations.md): Independent adversarial review evaluation, Wave 3/4 build status, empirical host CPU Roofline benchmark table, and full-stack audit reconciliation (`F1..F21`, `O1..O17`).
- [`docs/12_Independent_Verification_Caveats_and_Inconsistencies.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/12_Independent_Verification_Caveats_and_Inconsistencies.md): Independent re-derivation of every host-verifiable file, caveats, modelling assumptions, and the adversarial defect ledger (`4.1..4.31`) with fix status and seam verification protocol.
- [`docs/15_Language_and_Runtime_Architecture_C_CPP_CUDA_vs_PyTorch.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/15_Language_and_Runtime_Architecture_C_CPP_CUDA_vs_PyTorch.md): Explicit architectural rationale for the two-plane language split (why C11, zero-dependency C++20, and CUDA/PTX power all hot-path execution, why NVIDIA and this stack use PyTorch/Triton as the autograd and SafeTensors control plane, and when to deploy pure standalone C/C++).
- [`docs/16_Subsystem_Audit_Interleaving_Seams_and_Testing_Architecture.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/docs/16_Subsystem_Audit_Interleaving_Seams_and_Testing_Architecture.md): Visual subsystem interleaving graphs, part-by-part individual audit, 4-seam cross-language verification analysis, and complete chronological record of all 6 verification passes (`F1..F21`, `O1..O17`, `4.1..4.31`).

### 2. Engineering Workstreams
- **Workstream 1 (`sim_harness/`):**
  - [`sim_harness/ptx_bf162_emu.c`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/sim_harness/ptx_bf162_emu.c): Bit-exact PTX instruction emulator (`shl.b32`, `shr.u32`, `lop3.b32` `0xAE`/`0xEA`, `fma.rn.bf16x2`, `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`).
  - [`sim_harness/smsp_roofline_sim.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/sim_harness/smsp_roofline_sim.py): Analytical (not cycle-accurate) SMSP roofline model: 32-bank SRAM crossbar throughput, optional per-warp `ldmatrix` latency term, and modelled scoreboard stall (`smsp__SassWait`).
- **Workstream 2 (`compiler_unlock/` - Compiler Heuristics & Custom GEMM Kernels):**
  - [`compiler_unlock/libcublas_unlock_shim.c`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/compiler_unlock/libcublas_unlock_shim.c): `LD_PRELOAD` research interceptor evaluating `cuBLASLt` tile heuristics under workstation profile queries via `cuDeviceGetAttribute`, `cudaDeviceGetAttribute`, `cuDeviceGetName`, `cuGetProcAddress`, `cuGetProcAddress_v2`, and `/dev/nvidiactl` `ioctl` (`NV_ESC_RM_CONTROL = 0x2A`).
  - [`compiler_unlock/test_cublas_shim.c`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/compiler_unlock/test_cublas_shim.c): Mock driver test harness verifying `cuGetProcAddress_v2` function-pointer table redirection (Suite `A1`); [`compiler_unlock/test_shim_dladdr_linux.c`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/compiler_unlock/test_shim_dladdr_linux.c) + [`verify_shim_dladdr_linux.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/compiler_unlock/verify_shim_dladdr_linux.py) exercise the `dladdr` caller classification on Linux or WSL (Suite `A4`).
  - [`compiler_unlock/unlocked_gemm.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/compiler_unlock/unlocked_gemm.py): Custom 8-warp, 4-to-5 stage Triton GEMM kernels tuned for full-rate `sm_86` and `sm_89` Tensor Cores.
- **Workstream 3 (`fused_w1_hadacore/`):**
  - [`fused_w1_hadacore/spinquant_packer.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/fused_w1_hadacore/spinquant_packer.py): Cayley-SGD Stiefel manifold rotation optimizer and Interleaved 16-Bit Split `uint32` bit-packer for Binary g128 and BitNet b1.58.
  - [`fused_w1_hadacore/w1a16_hadacore_kernel.cu`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/fused_w1_hadacore/w1a16_hadacore_kernel.cu), [`fused_w1_hadacore/w1a16_host_cta_shim.hpp`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/fused_w1_hadacore/w1a16_host_cta_shim.hpp), & [`fused_w1_hadacore/w1a16_host_verify_main.hpp`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/fused_w1_hadacore/w1a16_host_verify_main.hpp): Fused CUDA/PTX kernel combining online $16 \times 16$ HadaCore Tensor Core rotations with 3-instruction `lop3.b32` (`0xAE`) and 4-instruction zero-preserving `lop3.b32` (`0xEA`) unpacking, plus 256-fiber host CTA verification.
- **Workstream 4 (`structural_filters/`):**
  - [`structural_filters/tiled_cbc_sieve.cu`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/structural_filters/tiled_cbc_sieve.cu) & [`structural_filters/tiled_cbc_sieve.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/structural_filters/tiled_cbc_sieve.py): Row-max Holder L1 activation energy sieve using `__shfl_xor_sync` and `__ballot_sync` to skip low-energy `BLOCK_K = 128` tiles.
  - [`structural_filters/lserve_sparse_attn.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/structural_filters/lserve_sparse_attn.py): Unified static sink/local + dynamic centroid-bounded 4-bit Hadamard KV attention.
- **Workstream 5 (`qat_galore_engine/`):**
  - [`qat_galore_engine/fused_qat_ste.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/qat_galore_engine/fused_qat_ste.py): In-SRAM `BLOCK_K = 128` Fused STE forward and backward kernels allocating 0 Bytes of global VRAM for $\widehat{W}$.
  - [`qat_galore_engine/galore_async_offload.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/qat_galore_engine/galore_async_offload.py): Per-layer `register_post_accumulate_grad_hook` 8-bit GaLore low-rank optimizer and 32B memory audit.
- **Workstream 6 (`kernel_driver_patch/` - Interconnect & Hardware State Research):**
  - [`kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c): Unified static (24GB/32GB) + dynamic (48GB modded RTX 4090) BAR1 P2P PTE builder, `nv_pci_resume_early_unlock`, on-demand stock relock, isolated IOMMU peer whitelist, and hardware recovery controller.
  - [`kernel_driver_patch/01-gsp-and-static-bar1-p2p.patch`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kernel_driver_patch/01-gsp-and-static-bar1-p2p.patch), [`kernel_driver_patch/02-dynamic-bar1-48gb-and-noncoh-pte.patch`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kernel_driver_patch/02-dynamic-bar1-48gb-and-noncoh-pte.patch), [`kernel_driver_patch/03-falcon-ss0-unlock-and-s3-resume.patch`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kernel_driver_patch/03-falcon-ss0-unlock-and-s3-resume.patch), and [`kernel_driver_patch/nvidia-p2p.conf`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kernel_driver_patch/nvidia-p2p.conf).
- **Workstream 7 (`hetero_runtime/`):**
  - [`hetero_runtime/chorus_amx_moe.c`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/hetero_runtime/chorus_amx_moe.c): 1GB Pinned HugePage zero-copy SPSC ring, multi-threaded OS worker (`_beginthreadex` / `pthread_create`), and AVX-512 `_mm512_mask_blend_epi16` + Intel AMX `_tile_dpbf16ps` 1-bit MoE worker.
  - [`hetero_runtime/closed_loop_governor.c`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/hetero_runtime/closed_loop_governor.c) & [`hetero_runtime/closed_loop_governor.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/hetero_runtime/closed_loop_governor.py): Chorus queue-pressure scheduler ($\Delta Q \le 1$) and NVRM post-arbitration clock-floor governor.
- **Workstream 8 (Build, E2E Pipeline, Verification & Full-Stack Exporter):**
  - [`run_all_tests.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/run_all_tests.py), [`e2e_transformer_pipeline.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/e2e_transformer_pipeline.py), [`Makefile`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/Makefile), [`verify_bare_metal.sh`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/verify_bare_metal.sh), [`export_full_stack.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/export_full_stack.py), and [`Full_Unlocked_GPU_CoDesign_Stack.md`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/Full_Unlocked_GPU_CoDesign_Stack.md). `python run_all_tests.py` runs 22 functional suites plus the H1 hygiene audit (23 total); all are host-only, and nothing runs on a GPU.
- **Workstream 9 (`kinesis_unified/` - Unified C++20 Memory, AVX2 Sub-2-Bit & Smart CPU/GPU Cache):**
  - [`kinesis_unified/unified_sizing.h`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kinesis_unified/unified_sizing.h) & [`kinesis_unified/memory_primitives.hpp`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kinesis_unified/memory_primitives.hpp): Shared C/C++ ring sizing constants, prefaulted and page-locked `VirtualMemory`, `ThreeTierArenaManager` (`Persistent`, `Step`, `Scratch`), `SlabAllocator`, and `MemorySpace`-tagged 64-bit `TensorView`.
  - [`kinesis_unified/hardware_and_sandbox.hpp`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kinesis_unified/hardware_and_sandbox.hpp): Live `CPUID` + `XGETBV` cache discovery, analytical BLIS blocking (`KC`, `MC`, `NC`), separated CPU/GPU latency cycles, and 1-way conflict-free 32-bank XOR swizzling.
  - [`kinesis_unified/sub2bit_gemm_and_smart_cache.hpp`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kinesis_unified/sub2bit_gemm_and_smart_cache.hpp): True `4x8` register-blocked FP32 GotoBLAS GEMM, `1.125-bit` (`28.44x` compressed) and `2.125-bit` (`15.06x` compressed) GotoBLAS panel packers with branchless AVX2 FMA unpacking, 256-bit 32-lane `vpshufb` T-MAC LUT kernel, lossless CPU GotoBLAS to GPU Interleaved `uint32` layout bridge, and `SmartTieredCache` (`Tiled-CBC` Holder error-budget sieve + CPU-L3 to Single-GPU VRAM hot promotion).
  - [`kinesis_unified/kinesis_c_abi.cpp`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kinesis_unified/kinesis_c_abi.cpp) & [`kinesis_unified/kinesis_bridge.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kinesis_unified/kinesis_bridge.py) (Suite `F2`): Native C++20 `extern "C"` shared-library ABI (`libkinesis_unified.dll` / `.so` via Python `ctypes`) + spec-compliant Hugging Face SafeTensors binary serializer/deserializer and sub-2-bit Transformer layer state-dict converter.
  - [`kinesis_unified/bench_cpu_roofline.cpp`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kinesis_unified/bench_cpu_roofline.cpp) (Suite `F3`): Empirical host CPU GFLOPS, cache-residency, and effective bandwidth benchmark across Decode (`M=4, K=1024, N=1024`) and Batched Prefill (`M=64, K=1024, N=1024`).
  - [`kinesis_unified/test_kinesis_unified.cpp`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kinesis_unified/test_kinesis_unified.cpp) (Suite `F1`): C++20 verification suite (`9` check groups); [`kinesis_unified/verify_f21_build_matrix.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/kinesis_unified/verify_f21_build_matrix.py) (Suite `F4`) builds it at `-O0`, `-O3 -fno-inline` and `-O3` and scans for aligned YMM stack moves.

---

## Hardware Testing Disclaimer & Limitation of Liability

> [!CAUTION]
> **Experimental Research & Hardware Safety Notice:**
> The mathematical algorithms, compiler shims, and kernel emulation in this repository have been derived from first principles and rigorously verified on host-CPU test suites (23/23 passing suites). However, **none of the physical driver modifications, MMIO/GSP registers, VBIOS, or BAR1 P2P patches have been executed on live bare-metal hardware in this repository.**
>
> All kernel driver and firmware components (`kernel_driver_patch/`) operate as **host-verifiable architectural models and mocks**. The authors and contributors assume **NO RESPONSIBILITY OR LIABILITY WHATSOEVER** for any hardware damage, silicon degradation, PCIe bus lockups, system instability, data corruption, bricked GPUs, or voided hardware warranties resulting from attempting to compile, flash, or execute these patches or commands on real physical hardware. Users who attempt to deploy or adapt any driver/firmware code onto physical machines do so entirely at their own risk.

---

## Deployment Scope: Software Drop-Ins vs. Hardware-Specific Requirements

This repository spans pure software algorithms (which can be dropped into existing frameworks today) and low-level microarchitectural experiments (which require specific silicon features):

### 1. Pure Software Drop-Ins (Universal & Framework-Agnostic)
These modules require no special hardware permissions or driver hacks and can be directly integrated into existing inference frameworks (PyTorch, vLLM, TensorRT-LLM, llama.cpp):
* **AVX2 Sub-2-Bit T-MAC LUT Engine (`kinesis_unified/`):**
  - *Drop-in for:* CPU inference engines (e.g. `llama.cpp`, BitNet).
  - *Hardware requirement:* Any x86-64 processor supporting AVX2 + FMA (Intel Haswell / Core 4th Gen [2013] or AMD Zen 1 [2017] and newer).
* **Fused In-SRAM Clipped-STE QAT (`qat_galore_engine/fused_qat_ste.py`):**
  - *Drop-in for:* Any PyTorch low-bit training pipeline. Materializes 0 bytes of dequantized weights in global VRAM.
  - *Hardware requirement:* Any standard CUDA or CPU PyTorch setup.
* **Log8-SR GaLore Low-Rank Optimizer (`qat_galore_engine/galore_async_offload.py`):**
  - *Drop-in for:* PyTorch optimizer wrappers (e.g., `bitsandbytes`, Hugging Face PEFT).
  - *Hardware requirement:* Any host CPU + GPU system.
* **Hölder Row-Max Dynamic Sieve (`structural_filters/tiled_cbc_sieve.py`):**
  - *Drop-in for:* Attention/MLP projection sparsification in Triton or CUDA.
  - *Hardware requirement:* Standard CUDA SM80+ GPUs or CPU vector units.
* **LServe Sparse Attention & Centroid Bounding (`structural_filters/lserve_sparse_attn.py`):**
  - *Drop-in for:* Long-context KV-cache management.
  - *Hardware requirement:* Standard CPU/GPU PyTorch environment.

### 2. Hardware-Specific Kernels & Physical Silicon Boundaries
These modules depend on specific ISA extensions or physical interconnect architectures:
* **HadaCore Fused Kronecker CUDA Kernel (`fused_w1_hadacore/w1a16_hadacore_kernel.cu`):**
  - *Targets:* NVIDIA Ada Lovelace (`sm_89`: RTX 4090, RTX 4080, L4, L40) and Ampere (`sm_80`/`sm_86`: A100, RTX 3090).
  - *Hardware requirement:* Native 16-bit Tensor Cores (`HMMA.16816.F32`), 32-bank shared memory crossbar, and `lop3.b32` bit-manipulation ALUs.
* **Chorus AMX MoE Engine (`hetero_runtime/chorus_amx_moe.c`):**
  - *Targets:* Intel Xeon 4th Gen (Sapphire Rapids), 5th Gen (Emerald Rapids), 6th Gen (Granite Rapids), or Core Ultra (Meteor/Arrow Lake).
  - *Hardware requirement:* Intel Advanced Matrix Extensions (`AMX-BF16` / `AMX-INT8` via `_tile_dpbf16ps`) and AVX-512 VNNI (`_mm512_i32scatter_epi32`).
* **Dynamic BAR1 GMMU Aperture & P2P Engine (`kernel_driver_patch/`):**
  - *Targets:* Dual-GPU PCIe configurations (e.g., 2x RTX 3090, 2x RTX 4090, or modded 48GB AD102).
  - *Hardware requirement:* PCIe root complex with Resizable BAR (ReBAR) support, ACS (Access Control Services) compliant switches, and `open-gpu-kernel-modules` driver build trees. (Operates strictly as a host mock in this repository).

---

## License

This project is licensed under the **Apache License, Version 2.0**. See the [`LICENSE`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/LICENSE) file for the full license text.

