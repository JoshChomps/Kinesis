# First-Principles Firmware, Hardware & Microarchitecture Code Review and Remediation Ledger

## Executive Summary
This document records the complete first-principles audit and full remediation of the repository across PCIe bus ordering, GPU BAR0/BAR1 MMIO registers, IEEE-754 BFloat16 arithmetic, SM86/SM89 Tensor Core fragment layouts, Intel AMX/AVX2/AVX-512 ISA semantics, and OS virtual memory contracts.

All **7 Critical (P0)** hardware/ISA correctness bugs, **6 High (P1)** architectural/numerical hazards, and **3 Medium (P2)** structural fidelity issues identified across Domains 1, 2, and 3 have been resolved and verified across all C11, C++20, and Python test suites.

---

## Domain 1: Firmware, PCIe BAR0/BAR1 MMIO, IOMMU & Userspace Interceptor

> [!CAUTION]
> **[P0-1] Posted PCIe MMIO Write vs. `pcie_flr()` Race & Compiler Dead-Store Elimination (FIXED)**
> **Locations:** [`nv_gpu_ops_and_s3_unlock.c`](../kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c#L95-L108), [`nv_gpu_ops_and_s3_unlock.c`](../kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c#L345-L386)
>
> **Hardware Principle & Remediation:**
> 1. Under PCIe Base Specification ordering rules, Memory Writes (`MWr` to BAR0 `0x00823804`, `0x0082381C`, `0x00823820`) are **posted** transactions routed through the GPU's internal PRI (Privilege Ring Interconnect), whereas `pcie_flr(pdev)` issues a **non-posted** Configuration Write (`CfgWr0` to PCIe Device Control bit 15). We added `nv_bar0_flush_posted_writes(pdev)` to issue non-posted MMIO read-backs (`MRd`) on `0x0082381C`, `0x00823820`, `0x00823818`, and `0x00823804` before `pcie_flr(pdev)` so the Function Level Reset never races in-flight PRI writes.
> 2. Qualified `feat_plm_reg`, `feat_readout1_reg`, `ss0_reg`, and `ss1_reg` as `volatile uint32_t` in `struct pci_dev` so `-O2`/`-O3` Dead-Store Elimination (DSE) cannot strip the `0xFFFFFFFFU` PLM unlock write prior to the `0x88000000U` relock write.

---

> [!CAUTION]
> **[P0-2] Linux `pci_restore_state()` Contract Across `pcie_flr()` (FIXED)**
> **Locations:** [`nv_gpu_ops_and_s3_unlock.c`](../kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c#L322-L398), [`03-falcon-ss0-unlock-and-s3-resume.patch`](../kernel_driver_patch/03-falcon-ss0-unlock-and-s3-resume.patch#L16-L20)
>
> **Hardware & OS Kernel Principle & Remediation:**
> In the Linux PCI core (`drivers/pci/pci.c`), `pci_restore_state(pdev)` checks `if (!pdev->state_saved) return;` and clears `pdev->state_saved = false` upon completion. In `nv_pci_resume_early_unlock`, calling `pci_restore_state(pdev)` before MMIO writes consumed `state_saved`, causing the post-FLR `pci_restore_state(pdev)` call to be a no-op and leaving `PCI_COMMAND` (`MSE = 0`, `BME = 0`) and BAR0 unmapped. Added `pci_save_state(pdev)` immediately before `pcie_flr(pdev)` in both `nv_gpu_ops_and_s3_unlock.c` and `03-falcon-ss0-unlock-and-s3-resume.patch`, and verified `pdev.bus_master_enabled == 1` post-FLR.

---

> [!CAUTION]
> **[P0-3] CUDA PCI ID Interception (`NV2080_CTRL_CMD_BUS_GET_PCI_INFO = 0x20801801U`) & `cuGetProcAddress` Recursion Guard (FIXED)**
> **Locations:** [`libcublas_unlock_shim.c`](../compiler_unlock/libcublas_unlock_shim.c#L32-L46), [`libcublas_unlock_shim.c`](../compiler_unlock/libcublas_unlock_shim.c#L124-L280), [`test_cublas_shim.c`](../compiler_unlock/test_cublas_shim.c#L156-L185)
>
> **Hardware & Driver Principle & Remediation:**
> 1. In `cuda.h`, `CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34` returns the **5-bit PCIe device/slot number on the bus (`0..31`)**, whereas the 16-bit PCI hardware device ID (`0x2204` / `0x2684`) is queried via NVRM `ioctl` `NV2080_CTRL_CMD_BUS_GET_PCI_INFO` (`cmd == 0x20801801U`). Added `NV2080_CTRL_BUS_GET_PCI_INFO_PARAMS` packet interception in `shim_spoof_nvos54_packet()` and guarded `cuDeviceGetAttribute` / `cudaDeviceGetAttribute` with `*pi >= 0x1000` so 5-bit PCIe slot numbers (`0..31`) are never corrupted.
> 2. Replaced the single-frame `is_caller_cublas(__builtin_return_address(0))` gate inside `ioctl()` (where the immediate frame is `libcuda.so.1`) with thread-local cuBLAS call-depth tracking (`shim_enter_cublas_context` / `shim_leave_cublas_context`) plus multi-frame backtrace `dladdr` inspection (`is_callstack_cublas`, `O2`).
> 3. Guarded `cuGetProcAddress` and `cuGetProcAddress_v2` with `if (*pfn != (void*)&cuDeviceGetAttribute)` and `if (*pfn != (void*)&cuDeviceGetName)` (and recursive self-symbol interception `F12`) to prevent self-assignment infinite recursion.

---

> [!WARNING]
> **[P1-1] 64-Bit Unsigned Integer Overflow Guards & Dynamic BAR1 8-Way LRU Tile Cache (FIXED)**
> **Locations:** [`nv_gpu_ops_and_s3_unlock.c`](../kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c#L151-L240), [`nv_gpu_ops_and_s3_unlock.c`](../kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c#L430-L450), [`01-gsp-and-static-bar1-p2p.patch`](../kernel_driver_patch/01-gsp-and-static-bar1-p2p.patch#L6-L12)
>
> **Hardware & Security Principle & Remediation:**
> Replaced all three `offset + size > limit` range checks with overflow-safe subtraction checks (`offset > limit || size > (limit - offset)`), added an 8-way `DynBar1CacheEntry` LRU tile cache in `kbusMapFbAperture_HAL` to reuse/evict BAR1 slots on modded 48 GB cards instead of monotonically exhausting the 32 GB BAR1 aperture, and accounted for `cpuVisibleLength` in `01-gsp-and-static-bar1-p2p.patch`.

---

> [!WARNING]
> **[P1-2] Closed-Loop Governor GPUBoost 15 MHz Step Deadband & Power-Floor Guard (FIXED)**
> **Locations:** [`closed_loop_governor.c`](../hetero_runtime/closed_loop_governor.c#L57-L69), [`closed_loop_governor.py`](../hetero_runtime/closed_loop_governor.py#L88-L106)
>
> **Hardware Principle & Remediation:**
> Added a power-floor check (`board_power_watts >= 0.90f * power_cap_watts`) before raising `tau_sieve` on `sag > 0.02f` so idle P8/P5 downclocking is never misclassified as power-brake throttling, and widened the relaxation threshold to `sag <= 0.01f` (`1.0%` deadband, covering 15 MHz GPUBoost step bins) so `tau_sieve` never locks up at `tau_max`.

---

## Domain 2: GPU PTX/CUDA Kernels, Roofline Simulator, QAT & Sparse Attention

> [!CAUTION]
> **[P0-4] BFloat16 Mantissa Overflow on Ternary Zero (`q = 1`) in `0xEA` Unpacker (FIXED)**
> **Locations:** [`w1a16_hadacore_kernel.cu`](../fused_w1_hadacore/w1a16_hadacore_kernel.cu#L82-L103), [`ptx_bf162_emu.c`](../sim_harness/ptx_bf162_emu.c#L150-L169), [`ptx_bf162_emu.c`](../sim_harness/ptx_bf162_emu.c#L312-L355), [`spinquant_packer.py`](../fused_w1_hadacore/spinquant_packer.py#L190-L219)
>
> **IEEE-754 BFloat16 Principle & Remediation:**
> Because $129 = 2^7 + 2^0$ spans all 8 significand bits of a BFloat16 number, pre-multiplying `bf16(-129.0f * alpha_g)` rounded off the lowest mantissa bits of any non-power-of-2 scale `alpha_g`, causing ternary zero (`q = 1`, encoded as `0x4001` = $2.0 + 1/64$) to unpack to `-0.25f` instead of `0.0f` for `alpha_g = 1.75f`.
> Replaced the pre-multiplied FMA constants with exact unscaled BF162 constants `c_mul = 0x42804280` `(64.0, 64.0)` and `c_sub = 0xC301C301` `(-129.0, -129.0)` in `fma.rn.bf16x2` to yield exact unit trits `{-1.0, +0.0, +1.0}` (`0xBF80`, `0x0000`, `0x3F80`), followed by `fma.rn.bf16x2` with `scale_bf162` and `-0.0` (`0x80008000`, `F1`). Verified across 18,874,368 ternary states in `ptx_bf162_emu.c` against the true mathematical target `{-alpha_g, 0.0f, +alpha_g}` with **0 bit errors**.

---

> [!CAUTION]
> **[P0-5] 2x Energy Underestimation & Half-Tile Omission in `tiled_cbc_sieve.cu` (FIXED)**
> **Locations:** [`tiled_cbc_sieve.cu`](../structural_filters/tiled_cbc_sieve.cu#L31-L110)
>
> **SM86/SM89 Fragment Principle & Remediation:**
> Expanded `a_rot_k128` in `tiled_cbc_sieve_pred` from `[4][4][4]` (`64` `bf162` regs/lane = `K=64`) to the complete `[4][2][4][4]` (`4` steps of `K=32` $\times$ `2` `k16` sub-steps $\times$ `4` `m16` row tiles $\times$ `4` `bf162` regs = `128` `bf162` regs/lane = `8,192` BF16 elements across the 32-lane warp), and added a standalone host verification suite (`test_tiled_cbc.exe`) confirming exact energy normalization (`E_k = 1.5000` on `(-1.5, +1.5)` BF162 inputs).

---

> [!WARNING]
> **[P1-3] Shared-Memory `cp.async` + `ldmatrix` Staging, Hoisted Hadamard Signs & Boundary Guards in `w1a16_hadacore_kernel.cu` (FIXED)**
> **Locations:** [`w1a16_hadacore_kernel.cu`](../fused_w1_hadacore/w1a16_hadacore_kernel.cu#L125-L185), [`w1a16_hadacore_kernel.cu`](../fused_w1_hadacore/w1a16_hadacore_kernel.cu#L270-L338)
>
> **SM86/SM89 Principle & Remediation:**
> Added `precompute_hadacore_h16_b_frag` to hoist the loop-invariant Sylvester Hadamard `H_16` B-fragment signs outside the `k_base` loop, wired up 128-byte XOR-swizzled shared-memory staging (`cp_async_16B` + `swizzle_128b_smem_offset` + `ldmatrix_x4_bf16`) for in-bounds tiles, and fixed the tail boundary guards to check `(k_sub + 2 * t + 1) < K` for `raw_a[0..1]` and `(k_sub + 2 * t + 9) < K` for `raw_a[2..3]`.

---

> [!WARNING]
> **[P1-4] `galore_async_offload.py` Root-Second-Moment ($\sqrt{v_t}$) Quantization & Device-Side INT8 Packing (FIXED)**
> **Locations:** [`galore_async_offload.py`](../qat_galore_engine/galore_async_offload.py#L126-L172)
>
> **Numerical Principle & Remediation:**
> Stored and quantized the **root second moment** $\sqrt{v_t}$ in `uint8` (`exp_avg_sq_q`, `exp_avg_sq_scale`) instead of the raw squared moment $v_t$, matching the linear dynamic range of $m_t$ and preventing coordinates with $|R_{r,j}| < 0.044 \max_k |R_{r,k}|$ from underflowing to `0` and triggering $10^5\times$ Adam step spikes. Also quantized tensors to `int8`/`uint8` prior to `.cpu()` transfer so only 1-byte tensors cross PCIe.

---

> [!WARNING]
> **[P1-5] `lserve_sparse_attn.py` Per-Query Convex Upper Bound & Packed `uint8` INT4 Dequantizer (FIXED)**
> **Locations:** [`lserve_sparse_attn.py`](../structural_filters/lserve_sparse_attn.py#L90-L178)
>
> **Mathematical Principle & Remediation:**
> Replaced `q_mean = q_rot.mean(dim=2)` before `torch.maximum` with per-query convex upper-bound evaluation `torch.maximum(q_exp * k_min_exp, q_exp * k_max_exp).sum(dim=-1).amax(dim=2)` so opposing-sign queries across `Sq` never cancel out, and added `unpack_kv_pages_u8_to_int8` to dequantize active KV pages directly from the 4,096-byte `k_packed_u8` and `v_packed_u8` buffers (achieving `0.99313` cosine similarity against dense FP32 attention).

---

## Domain 3: CPU ISA (AMX/AVX2/AVX-512), OS Virtual Memory & C++20 Smart Cache

> [!CAUTION]
> **[P0-6] Intel AMX `_tile_dpbf16ps` Matrix B (`tmm6`) VNNI `[K/2=16, N=16, 2]` Layout & Ring Cache-Line Isolation (FIXED)**
> **Locations:** [`chorus_amx_moe.c`](../hetero_runtime/chorus_amx_moe.c#L29-L48), [`chorus_amx_moe.c`](../hetero_runtime/chorus_amx_moe.c#L75-L248)
>
> **Intel AMX ISA Principle & Remediation:**
> Updated `amx_gemm_1bit_expert_tile` to pack `w_vnni_buf` in hardware VNNI pair-interleaved order (`[K/2 = 16, N = 16, 2]`) via vectorized `_mm512_i32scatter_epi32` (AVX-512 path) and strict-aliasing-clean 32-bit VNNI pair packing (emulation path with AMX BF16/FP32 denormal flush-to-zero semantics, `O10`), padded `gpu_ready_seq` (offset `0`) and `cpu_done_seq` (offset `64`) onto separate 64-byte cache lines in `HeteroMoERingSlot`, added `VirtualLock`/`mlock` physical page-locking and a dedicated OS worker thread (`_beginthreadex`/`pthread_create`), and verified both uniform/non-uniform tiles and concurrent SPSC ring execution with **0 errors**.

---

> [!CAUTION]
> **[P0-7] `XCR0` Bits 17/18 (`XTILECFG`/`XTILEDATA`) & `AVX512BW` Check, Plus C++20 Pointer Provenance Fix (FIXED)**
> **Locations:** [`hardware_and_sandbox.hpp`](../kinesis_unified/hardware_and_sandbox.hpp#L32-L120), [`memory_primitives.hpp`](../kinesis_unified/memory_primitives.hpp#L244-L250), [`memory_primitives.hpp`](../kinesis_unified/memory_primitives.hpp#L388-L395)
>
> **x86-64 ISA & C++20 Standard Principle & Remediation:**
> Added `os_saves_amx = (xcr0 & 0x60000ULL) == 0x60000ULL` (`XTILECFG` and `XTILEDATA`), `has_amx_tile` (`EDX[24]`), and `has_avx512bw` (`EBX[30]`) checks in `HardwareInfo::query_cpu_features()`, and converted relational pointer checks in `ArenaAllocator::owns` and `SlabAllocator::owns` to `uintptr_t` integer range comparisons after null guards.

---

> [!WARNING]
> **[P1-6] Unified `MemorySandbox` Bank Swizzles & `SmartTieredCache` W2 Execution + LFU/LRU Eviction (FIXED)**
> **Locations:** [`hardware_and_sandbox.hpp`](../kinesis_unified/hardware_and_sandbox.hpp#L292-L355), [`sub2bit_gemm_and_smart_cache.hpp`](../kinesis_unified/sub2bit_gemm_and_smart_cache.hpp#L370-L745), [`test_kinesis_unified.cpp`](../kinesis_unified/test_kinesis_unified.cpp#L1-L260)
>
> **Architectural Principle & Remediation:**
> 1. Exposed both `swizzle_32bank_word_offset` (5-bit Latin-square XOR for 4B scalar loads, `32-way -> 1-way`) and `swizzle_128b_smem_offset` (3-bit XOR for 16B `ldmatrix` vectors, `8-way -> 1-way`) in `MemorySandbox` and verified both in `test_kinesis_unified.cpp`.
> 2. Updated `pack_w2_gotoblas_panels` to compute the conditional L2-optimal scale over active non-zero trits, added LFU+LRU Tier-1 VRAM eviction (`record_panel_access_and_promote`), reordered the inner micro-kernel loop to `p -> ir -> g -> k` so `acc0..3` stay in YMM registers across all K-groups before writing `C`, and added `execute_smart_w2_gemm` (verified with `0.00e+00` error vs reference).

---

## Complete Post-Remediation Verification Matrix

| Test Suite / Binary | Command Executed | Key Post-Fix Hardware & Numerical Metrics | Status |
| :--- | :--- | :--- | :--- |
| **Workstream 1A: PTX & BF162 Emulator** | `./sim_harness/ptx_bf162_emu.exe` | `16,777,216` W1 states & `18,874,368` W2 states (with exact `0.0f` at $q=1$ across all scales) & `128` HMMA lanes: **0 bit errors** | **PASS** |
| **Workstream 1B: Analytical SMSP Roofline** | `python sim_harness/smsp_roofline_sim.py` | 32x32 unlocked `SassWait = 13.08%` (`287.3 TF`) -> 64x64 Fused W1A16 `SassWait = 3.31%` (**`320.4 TF`** on AD102). Re-run 2026-11-08 after the FLOP-accounting fix (docs/12 4.7); the earlier `50.76%` / `162.2 TF` and `6.40%` / `309.9 TF` came from a 2x error in `t_mma`. | **PASS** |
| **Workstream 2A: Userspace cuBLAS / NVRM Shim** | `./compiler_unlock/test_cublas_shim.exe` | `cuGetProcAddress_v2` recursion guard + `NV2080_CTRL_CMD_BUS_GET_PCI_INFO` (`0x20801801`) spoof (`0x2684 -> 0x26B1`) | **PASS** |
| **Workstream 2B: Unlocked Triton Autotuner** | `python compiler_unlock/unlocked_gemm.py` | All 6 configs $\le 101,376\text{ B}$ (`98,304 B = 97.0%`), $I_{\text{warp}} \in [21.3, 32.0]$, **0 bit mismatches** | **PASS** |
| **Workstream 3: SpinQuant Cayley-SGD & Packer** | `python fused_w1_hadacore/spinquant_packer.py` | $\|R^T R - I\|_\infty = 1.192 \times 10^{-6}$, kurtosis `25.00 -> 1.82` (Kronecker `25.00 -> 3.65 -> 2.44`), W1 & W2 bit errors = **0 / 8,192** | **PASS** |
| **Workstream 4A: Tiled-CBC Warp Energy Sieve** | `./structural_filters/test_tiled_cbc.exe` & `.py` | Full `[4][2][4][4]` (`8,192` BF16/warp) normalization `E_k = 1.5000`; row-max Holder bound verified on 32-lane warp | **PASS** |
| **Workstream 4B: LServe INT4 Hadamard KV Attn** | `python structural_filters/lserve_sparse_attn.py` | `4,096 B/page` `uint8` unpacker, per-query convex bound, `62.5%` pages skipped (`8.90x` BW), $\cos = \mathbf{0.99313}$ (`0.99305` at 50% streaming) | **PASS** |
| **Workstream 5A: In-SRAM `BLOCK_K=128` QAT STE** | `python qat_galore_engine/fused_qat_ste.py` | `0 B` global `W_hat` allocation, Forward/dX/dW mismatches = **0**, Clipped STE verified | **PASS** |
| **Workstream 5B: 8-Bit Root-Moment GaLore** | `python qat_galore_engine/galore_async_offload.py` | Root-moment $\sqrt{v_t}$ `uint8` quantization, sign-preserving QR moment reprojection, 32B peak VRAM = **`23.63 GB/GPU` (`22.00 GiB/GPU`)** | **PASS** |
| **Workstream 6: Kernel BAR1 P2P & S3 Unlock** | `./kernel_driver_patch/test_kernel_patch.exe` | `volatile` MMIO + read-back flush, `pci_save_state` before FLR (`Post-FLR BME = 1`), 8-way LRU BAR1, overflow guard | **PASS** |
| **Workstream 7: 1GB HugePage Ring & AMX / Governor** | `./hetero_runtime/test_chorus_amx.exe` & `test_governor_c.exe` & `.py` | Producer@`0B` / Consumer@`64B`, AMX VNNI `[K/2, N, 2]` Uniform & Non-Uniform Err = **0**, 1% GPUBoost hysteresis | **PASS** |
| **Workstream 9: Kinesis C++20 & Smart Cache** | `./kinesis_unified/test_kinesis_unified.exe` | `-DNDEBUG` `UHA_VERIFY`, `XCR0` AMX check, 1-way scalar & `ldmatrix` swizzle, W1 (`28.44x`) & W2 (`15.06x`) AVX2 Err = **`0.00e+00`**, LFU VRAM eviction | **PASS** |
