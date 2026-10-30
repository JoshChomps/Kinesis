# 11. Adversarial Review, Verified Status and Known Limitations

This document is the honest ledger for the project. It records what an adversarial logic and code review found, what was fixed, what is still open, and exactly what has and has not been executed. It supersedes any stronger wording in docs 08 to 10 where the two disagree.

## 0. 2026-10-30 development pass (host CPU only; no GPU, driver, or firmware action)

A second pass fixed several review items and audited cross-file consistency. Everything below was built and run on the host CPU with gcc/g++/clang/python; the project is now under local git (baseline commit is the pre-pass snapshot, each round is its own commit). No GPU kernel was compiled or run, no driver patch applied, no shim loaded, no firmware/MMIO touched, and no system setting changed.

* **O8 fixed** - `SmartTieredCache` now defaults to exact (tau = 0). The lossy relative-energy sieve is opt-in; a default-constructed cache is bit-identical to the plain FP32 GEMM (test [8]).
* **O9 fixed** - new `set_error_budget(eps)`: a K-group is skipped only while a worst-case per-output bound (max column |scale| x max row L1 of the activations) keeps the total skipped mass <= eps. Holds for W1 and W2; non-finite activations are never skipped. Verified error <= eps on both; the old summed-energy heuristic breaks the same budget on a single-loud-row input (negative control, test [8]).
* **O11 fixed** - panel promotion counts once per GEMM call, not once per K-group.
* **O14 fixed** - on a GaLore projector refresh the Adam moments are rotated into the new basis (M exactly via P_new^T P_old; V via the squared rotation).
* **GaLore numerical hardening** - 8-bit sqrt(V) now rounds up on quantization so a small coordinate next to a large row max can no longer become 0 and spike the next step; BF16 master-weight writes use unbiased stochastic rounding so sub-ulp updates (e.g. lr 2e-4 on ~0.02 weights) are no longer silently dropped.
* **Memory table corrected** - `compute_32b_qat_memory_table` now derives each peak as sum(components)/n_gpus instead of a hard-coded figure. The GaLore peak was 22.15 but its own components sum to 23.63 GB/GPU; the honest 23.63 still fits a real 24 GiB (= 25.77 decimal GB) card. Host-staged HugePage weights and streamed optimizer moments are explicitly excluded from VRAM.
* **Test-output honesty** - the kinesis "56.89x bandwidth" line is relabelled as a modelled ratio (compression x 1/(1 - skipped)), not a measurement.
* **Sizing-contract audit** - group size 128, weights-per-uint32 (32 for W1, 16 for W2), the Nr=8/Mr=4 CPU micro-tile, and FP32 accumulation are consistent across `unified_sizing.h`, the C++ engine, the PTX emulator and the Python kernels. The C++ constants already derive from the header; the previously free-standing tile literals in `w1a16_hadacore_kernel.cu` are now pinned to the header with `static_assert`s (verified by compiling the asserts on the host; the kernel body itself is still unbuilt, O1).
* **Sanitizers** - the C++ suite, the PTX emulator, the governor and the AMX worker run clean under AddressSanitizer + UBSan.

Scope note: the device-ID interposer (`compiler_unlock/`) and the firmware/driver unlock code (`kernel_driver_patch/`) were left exactly as the baseline; this pass neither extended nor weakened them. O1-O7, O10, O12-O13, O15-O17 below remain open and still need a GPU / real driver tree.

## 1. What can and cannot be verified in this environment

| Component | How it was exercised | What that proves | What it does NOT prove |
| :--- | :--- | :--- | :--- |
| `kinesis_unified` (C++20, AVX2) | Compiled and run on the host CPU | Numerical correctness of packing, FP32 GEMM, W1/W2 unpack, T-MAC LUT vs FP32 | Performance on any target, AVX-512 or AMX paths |
| `sim_harness/ptx_bf162_emu.c` | Host C emulator of PTX semantics | Unpack constants and bit layouts are self-consistent under the emulated semantics | That real hardware or `ptxas` accepts the PTX (see finding F1) |
| `fused_w1_hadacore/*.cu`, `structural_filters/tiled_cbc_sieve.cu` device code | **Never compiled.** No `nvcc`, no GPU here | Nothing beyond code review | Compilation, correctness, speed, occupancy, register use |
| `hetero_runtime/chorus_amx_moe.c` | Runs the scalar emulation branch only | Layout indexing and the ring-slot cache-line layout | Real `_tile_dpbf16ps` behaviour (rounding, denormals), Linux `mmap`/hugepage path, AMX permission request |
| `kernel_driver_patch/*.c` | Host mock of NVIDIA driver types | Logic of our own mock (LRU lifetime, ACS bit math, PLM sequence) | Anything about the real NVIDIA driver, GSP firmware, or hardware latches |
| `kernel_driver_patch/*.patch` | Not applied to any source tree | Nothing | That they apply or build; treat as design sketches |
| `compiler_unlock/libcublas_unlock_shim.c` | Windows build with a mock driver table | Resolver interception and ID mapping logic | The Linux-only `ioctl` hook and `dladdr` caller detection (compiled out here) |
| `sim_harness/smsp_roofline_sim.py` | Python | Arithmetic of a model with stated assumptions | Any measured speedup |
| Python QAT/GaLore/LServe | CPU PyTorch | Algorithms on small synthetic tensors | Behaviour on real models or GPUs |

Nothing in this repository has been run against a real GPU. Every performance figure is either a model output or taken from a cited external source.

## 2. Findings from the adversarial review

Severity: **S1** wrong results or will not build, **S2** unsafe or silently misleading, **S3** performance or hygiene.

### 2.1 Fixed

| ID | Sev | Finding | Fix |
| :-- | :-- | :--- | :--- |
| F1 | S1 | The kernel used `mul.rn.bf162`, `fma.rn.bf162`, `add.rn.bf162`. These are not PTX mnemonics (the type is `bf16x2`), and `mul`/`add` on `bf16x2` need sm_90, so the kernel could not assemble for sm_86/sm_89. The emulator hid this because it only models semantics. | All device PTX now uses `fma.rn.bf16x2`. A multiply is `fma(a, b, -0.0)` and an add is `fma(x, 1.0, y)`; both are bit-identical to the IEEE operation under round-to-nearest and valid on sm_80+. Constants are moved into registers instead of relying on immediates. |
| F2 | S1 | Kronecker butterfly inline asm declared outputs `"=r"` while later instructions still read inputs. The allocator may alias an output with an input, corrupting `t1` mid-sequence. | Every output is early-clobber `"=&r"`. |
| F3 | S1 | `ldmatrix.sync.aligned` was guarded by a per-lane condition (`r_bot < M` with `r_bot = group_id + 8 + ...`). It is a warp collective; divergent execution is undefined and can hang. | Guard is now warp-uniform: `(block_row + mi * 16 + 15) < M`. |
| F4 | S1 | The kernel rotates activations with `I (x) H16` (or `I (x) H2 (x) H16`) but `spinquant_packer.py` merged a dense Cayley or full-size Hadamard rotation into the weights. `(xR1)(WR2)^T != xW^T` unless `R1 == R2`, so end-to-end outputs would be wrong. | Added `build_kernel_online_rotation(K, mode)` that returns exactly what the kernel applies. Tests prove `(xR)(WR)^T = xW^T` to 2e-5 and include a negative control showing the mixed case is off by about 58. Rule: a learned dense rotation is only valid when absorbed offline into neighbouring layers, never as the online rotation. |
| F5 | S1 | `pack_w1_gotoblas_panels` forced the scale of an all-zero group to 1.0, so zero weights produced `+/- sum(activations)` instead of 0. | Zero groups keep scale 0; regression test added. |
| F6 | S2 | `ioctl` hook matched only `_IOC_NR == 0x2A` and then treated the argument as an `NVOS54_PARAMETERS`. Any unrelated ioctl with NR 0x2A (DRM, V4L2, ...) would be read and possibly written as that struct. | Also requires `_IOC_TYPE == 'F'` (NVRM magic). (Linux only; not compiled in this environment.) |
| F7 | S2 | The dynamic BAR1 LRU could evict or overwrite a slot that a live P2P mapping still used, giving two buffers the same BAR1 window (silent data corruption), orphaned slots in the bump path, and no unmap at all. | Reference counting; reclaim only unreferenced slots of identical size; `kbusUnmapFbAperture_HAL` and `nvGpuOpsDestroyExternalAllocPtes_Unified`; tests for full aperture refusal, release and reuse, double unmap. |
| F8 | S2 | `allocate_pinned_1gb_ring` could overflow `num_slots * sizeof` and memset past the single 1 GB mapping. | Overflow and size checks; falls back to the heap for oversize requests. |
| F9 | S1 | AMX code never requested XTILEDATA permission. On Linux 5.16+ the first tile instruction raises SIGILL without `arch_prctl(ARCH_REQ_XCOMP_PERM, 18)`. | `amx_request_permission()` called from `amx_init_tiles_bf16()`, which now returns a status. Per-thread configuration is documented. (Linux AMX only; not executable here.) |
| F10 | S3 | Sieve accumulated 32 values in bf16, so rounding error compounded. The L1 norm is also not invariant under the Hadamard rotation (only L2 is), which a doc claimed. | bf16 only for 4-term partial sums, FP32 accumulation after; invariance claim corrected in code comments. |
| F11 | S3 | Governor divided by `locked_clock_floor_mhz` with no guard. | Guard added. |
| F12 | S2 | Shim answered `cuGetProcAddress(_v2)` lookups only for two symbols. CUDA 12.x asks the resolver for itself and then uses that raw pointer, bypassing the shim. | Resolver names are intercepted via a shared helper; test confirms the pointer handed back is the hook and still spoofs correctly. |

### 2.2 Open (not fixed, by design or because they need hardware)

| ID | Sev | Finding | Notes |
| :-- | :-- | :--- | :--- |
| O1 | S1 | All device code is unbuilt. Expect additional ptxas and runtime errors (register spills with `acc[4][8][4]`, `__launch_bounds__(256,1)`, cp.async alignment: `K % 8 == 0` and 16 B aligned rows are assumed but not checked, W2 loads read `k_base/16 + 1` words which is out of range when `K % 32 != 0`). | Needs `nvcc` and a GPU. |
| O2 | S1 | The ioctl-level spoof (`shim_spoof_nvos54_packet`) is process-global. It is NOT caller-aware, unlike the `cuDeviceGetAttribute` path, because ioctl runs inside `libcuda`. NCCL and PyTorch also see spoofed IDs through this path. Earlier docs overstated selectivity. | Selective spoofing at this layer needs a different mechanism. |
| O3 | S1 | `.patch` files are unapplied sketches: hunk offsets are illustrative, `bar1[GFID_PF].physAddr` and the `kbusMapFbAperture_HAL` signature are assumptions, and patch 02 calls the map function with no matching unmap or LRU. The reference C file is a mock of NVIDIA types, so its LRU does not exist in the real driver. | Port against a real `open-gpu-kernel-modules` checkout. |
| O4 | S1 | The S3 resume hook assumes SS0/SS1 override latches survive a PCIe FLR. No evidence in this repo; FLR may reset them, which would undo the unlock. | Hardware test required. |
| O5 | S2 | Patch 01 returns "P2P supported" for any GPU whose BAR1 is at least 1 GB, skipping the static-BAR1 check. With IOMMU or ACS misconfigured this can cause faults or corruption instead of a clean refusal. | Needs a topology check (same root complex, ACS, IOMMU mode). |
| O6 | S2 | The "isolated IOMMU peer domain" function is an arithmetic checker. It does not program any IOMMU, ACS or access policy, so it does not make devices "only talk to each other". Likewise the VBIOS "rollback" is a struct update; nothing touches flash. | Real enforcement is IOMMU group policy plus ACS; real rollback is a verified flash of the stock image. Neither is implemented. |
| O7 | S2 | `ACS` redirect clearing exists only as an opt-in helper in the mock, never auto-invoked, because it weakens isolation. | Prefer `pci=config_acs=` kernel parameters. |
| **O8 (FIXED 2026-10-30)** | S2 | Tiled-CBC sieve and the default `energy_sieve_tau = 0.85` in `SmartTieredCache` are LOSSY. A caller using defaults silently gets approximate results. The "effective bandwidth amplification" in test output (for example 56.89x) is a computed ratio (compression x 1/(1 - pruned)), not a measurement. | Make exactness the default, require explicit opt-in to prune. |
| **O9 (FIXED 2026-10-30)** | S2 | The new scale-weighted sub-tile gate uses mean row L1, an average-case estimate, not a worst-case bound. | A sound gate needs the max over rows. |
| O10 | S3 | `chorus_amx_moe.c` unpacks weights with a scalar 16x32 store loop per K step, which will dominate the single `tdpbf16ps` it feeds. There is no consumer thread, no `cudaHostRegister`, no NUMA binding; "pinned" currently means hugepage-backed only. The emulation uses plain float arithmetic and will not match TDPBF16PS rounding or denormal flushing. | Vectorise the unpack, add worker and NUMA logic, validate on AMX hardware. |
| **O11 (FIXED 2026-10-30)** | S3 | `record_panel_access_and_promote` is called once per (panel, group), so promotion counts scale with group count rather than GEMM calls; victim search is O(panels) per call. | Count per GEMM, keep a heap or CLOCK. |
| O12 | S3 | The `ptx_bf162_emu.c` Kronecker test checks the H2 mixing and ternary MMA arithmetic against a reference built from the same arithmetic. It does not exercise the H16 stage. | Chain the emulated H16 MMA into the test. |
| O13 | S3 | The two bf16 rounding steps (after H16 and after the H2 butterfly) compound; no end-to-end error budget for rotated activations has been measured. | Measure on real activation statistics. |
| **O14 (FIXED 2026-10-30)** | S3 | After a Q-GaLore projector refresh the Adam moments stay in the old basis (inherited from GaLore). | Re-project or reset moments on refresh. |
| O15 | S3 | The lazy-schedule test forces doubling with threshold 0.0 to test the mechanism; the 0.90 default is taken from the paper, not tuned here. | Tune on a real run. |
| O16 | S3 | The Stream-K and hetero tables in `smsp_roofline_sim.py` are driven by an assumption (50% of SMs saturate DRAM, DDR5 80 GB/s, PCIe 25 GB/s). Their speedups follow from the assumption. Split-K with FP32 `atomicAdd` is also non-deterministic in summation order. | Measure; consider a deterministic reduction. |
| O17 | S3 | Unverified device IDs: RTX 4090 D `0x2685`, RTX 5090 `0x2B85`, RTX PRO 6000 Blackwell `0x2BB1` come from community reports. | Confirm on hardware. |

## 3. Claims corrected from earlier documents

* Doc 10 and earlier summaries said features such as the T-MAC LUT kernel, offline hot-panel seeding, streaming heads, Q-GaLore options, the 4090 D / 5090 mappings, resolver self-interception and the Stream-K model had been "added". At the time of the review they had not been written. They have now been implemented and tested to the extent shown in section 1.
* The streaming-head claim of cosine similarity above 0.985 was wrong. On the planted-needle test, 50% streaming heads give about 0.70, because streaming heads cannot retrieve the needles.
* The "exact Cauchy-Schwarz bound" wording for the sieve was wrong (see O9).
* "Kronecker-Stiefel optimization in `spinquant_packer.py`" was never implemented. Only the exact kernel-matched fixed rotation is.
* "Caller-aware spoofing" holds for the `libcuda` attribute and name calls only (see O2).

## 4. Recommended next steps, in order

1. Build the CUDA files with `nvcc -arch=sm_86 -arch=sm_89` and fix whatever `ptxas` reports (O1).
2. Add a GPU correctness harness that compares each kernel against the CPU reference for odd `M`, `N`, `K`.
3. Make exact computation the default in `SmartTieredCache` and the sieve (O8, O9).
4. Port the P2P design to a real driver tree on a test machine and run it with the IOMMU enabled before touching production hardware (O3 to O7).
5. Only then measure performance and replace modelled numbers in docs 08 and 10.
