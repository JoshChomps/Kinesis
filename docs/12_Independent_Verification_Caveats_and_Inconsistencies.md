# 12. Independent Verification Caveats, Inconsistencies, and Observations

This document records an independent code-and-math verification pass performed on the
host (no GPU, no driver applied, no firmware touched). It complements the honest ledger
in [`docs/11_Adversarial_Review_Status_and_Known_Limitations.md`](11_Adversarial_Review_Status_and_Known_Limitations.md):
doc 11 records what the project's own review found and fixed; this doc records what a
separate reviewer re-derived from scratch, what held, and what still deserves a caveat.

> [!NOTE]
> Scope: everything below was checked by reading the source, independently re-deriving
> the math rather than trusting comments, and running each file's own test. It proves
> host-level numerical and logical correctness only. It does not prove behavior on real
> silicon, `ptxas` acceptance of the PTX, or any performance figure. Where this doc and
> an older doc disagree on a measured number, trust a fresh run of the code.

## 1. Independently re-derived and confirmed clean

| Area | File | What was re-derived | Result |
| :--- | :--- | :--- | :--- |
| Cayley-SGD Stiefel step | `fused_w1_hadacore/spinquant_packer.py` | Skew-symmetry of `Y = G R^T - R G^T`; orthogonality of the Cayley transform; the descent identity `<G, Y R> = 0.5*||Y||_F^2 >= 0` expanded via `P = G R^T` | Descent confirmed; the F13 fix is correct. Second pass: confirmed, and it is true Riemannian GD with a Cayley retraction (5(a)); test credits Cayley with the Hadamard-init drop (4.16) |
| W1 binary unpacker (`lop3.b32` 0xAE) | `fused_w1_hadacore/spinquant_packer.py` | `(~a & 0x80008000) | 0x3F803F80` maps sign bit to bf16 +1.0 / -1.0, then `fma(+/-1, scale, -0.0)` | Bit-exact; 0 mismatches on run |
| W2 ternary unpacker (`lop3.b32` 0xEA) | `fused_w1_hadacore/spinquant_packer.py` | `bf16(0x4000+q)` then `inj*64 - 129` gives -1 / 0 / +1 with a single exact fma rounding; q=1 yields exactly 0.0 for any non-negative finite scale | Zero-preserving property holds; 0 mismatches. Second pass: negative scales give -0.0 and Inf/NaN give NaN (absmean never yields these) |
| Driver BAR1 lifetime (F7 class) | `kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c` | No two live (refCount>0) mappings can share a BAR1 VA window; VA reclaimed only at refCount 0; double-unmap rejected | **Audit Check 11 (2026-11-09):** found and fixed a cold-slot partial physical overlap bug (4.20) and closed caveat 2.1 with a first-fit coalescing VA gap allocator (`A3 [2b-2/3]`) |
| GaLore moment reprojection (F20/O14) | `qat_galore_engine/galore_async_offload.py` | `M_new = (P_new^T P_old) M_old`; `V_new = (rot*rot) @ V_old` from `E[(rot@g)_i^2]`; QR sign fix `q * sign(diag(R))` | Reprojection algebra correct (see caveat 3.1 on V). **Disputed by second pass:** uint8 round-up `sqrt(V)` storage ratchets (4.6) |
| Tiled-CBC row-max Holder sieve (O9) | `structural_filters/tiled_cbc_sieve.py` | `e_k = max_m mean_k(|X[m,tile]|)` is worst-case over rows; by Holder `|sum X W| <= ||X[m,tile]||_1 * max_k|W|`; single loud row keeps its tile | Bound correct (see observation 3.6). **Second pass:** NaN tiles skipped (4.12); `.cu` tau is per sub-tile, not a budget (4.13) |
| H_2 butterfly constant (F18) | `sim_harness/ptx_bf162_emu.c`, `fused_w1_hadacore/w1a16_hadacore_kernel.cu` | `0x3F353F35` = +0.70703125 ~= +1/sqrt(2); `0.25*(1/sqrt(2)) = 1/sqrt(32)` orthonormal H_32 scale | Consistent across both files (see inconsistency 4.2) |
| Fused STE QAT forward/backward | `qat_galore_engine/fused_qat_ste.py` | `Y = X W_hat^T`, `dX = dY W_hat`, STE `dW = dY^T X`, and the clipped-STE gate `1(|W| <= c*alpha_g)` | STE algebra correct (see caveat 2.4 and observation 3.4). **Second pass:** clip c=1.5 freezes ~23% of weights (4.8), bf16 master stall (4.9), unrotated basis and no ternary path (4.10) |
| SPSC ring memory ordering (O10) | `hetero_runtime/chorus_amx_moe.c` | Per-slot sequence handshake: producer release-stores `gpu_ready_seq`, consumer acquire-loads it then release-stores `cpu_done_seq`; slot reuse waits `cpu_done_seq >= seq - num_slots` | Ordering correct (second pass agrees). Latent defects on untested paths: 4.3 (confirmed), 4.4 (unreachable today), 4.15 (new) |
| LServe sparse + INT4 Hadamard attention (F15) | `structural_filters/lserve_sparse_attn.py` | H_16 cancels in `q_rot k_rot^T = q k^T`; V output un-rotated by `H^T`; per-dim `max(q*k_min, q*k_max)` is the exact box-corner upper bound (admissible page prune); salience = out-window minus in-window | Correct (see observation 3.8). **Second pass:** 0.70 and 0.993 come from different datasets; same-data gap is blind 0.85 to 0.86 vs profiled 0.993 (4.16) |
| SafeTensors serializer + round-trip (F2) | `kinesis_unified/kinesis_bridge.py` | `[8-byte LE u64 len][space-padded JSON header][contiguous data]`, offsets from data start, 8-byte header alignment | Spec-compliant; independently bit-exact across 5 dtypes + empty (see observation 3.9) |
| C ABI ctypes marshalling (F2/F21) | `kinesis_unified/kinesis_c_abi.cpp` + `kinesis_bridge.py` | every entry point is `try/catch(...)`-guarded (no exception crosses `extern "C"`); correct `argtypes`/`restype`; `.contiguous()` inputs; buffers kept alive across the synchronous call | Correct for tested shapes. **Second pass:** size contract diverges for `K % 128 != 0`, reachable heap overflow (4.5) |
| cuBLAS device-ID shim logic (F12/O2/F6) | `compiler_unlock/libcublas_unlock_shim.c` | resolver self-hook returns the hook pointer (self-assignment guarded against recursion); caller gating spoofs only when a blas/dnn DSO is on the call chain; ioctl requires both NR 0x2A and TYPE 'F' | Logic correct by review (see observation 3.10). **Second pass:** the mock exercises only the forced override, never the `dladdr`/backtrace classification (4.16) |
| SMSP roofline model (Pillar 1) | `sim_harness/smsp_roofline_sim.py` | `I_warp = W_M W_N/(W_M+W_N)`, peak `= sm_count * (2048/tau) * 4 * clock`, DRAM ridge `= peak/BW` in units of batch M | It is a model, not a measurement (see observation 3.11). **Disputed by second pass:** 2x FLOP-per-HMMA inconsistency between `peak_tflops` and `t_mma` (4.7) |
| Memory primitives (F20/F21 hardening) | `kinesis_unified/memory_primitives.hpp` | arena bump math uses `capacity - offset` form (no overflow); slab `deallocate` rejects double-free via `is_free_` bitmap; `total_elements`/`from_shape_row_major` saturate; Shape/Strides/TensorView are exactly 56/56/128 bytes | Correct; no defects |
| Hardware discovery + swizzle (L5/§2.2) | `kinesis_unified/hardware_and_sandbox.hpp` | CPUID gated by XGETBV OS-save bits (YMM/ZMM/AMX); leaf-4 cache size `ways*partitions*line*sets`; BLIS KC/MC/NC from live cache; `offset ^ ((offset>>3)&0x70)` yields `chunk ^ (r&7)` | Correct; one trivial imprecision (observation 3.12) |
| 4x8 GotoBLAS FP32 GEMM + W1/W2 packers | `kinesis_unified/sub2bit_gemm_and_smart_cache.hpp` | standard BLIS 5-loop (jc/pc/ic/jr/ir) with 4-accumulator micro-kernel, FP32 accumulation, F21-safe `storeu`; branchless AVX2 sign unpack; zero-group scale 0 (F5/F14); W2 threshold conditional-mean ternary scale; RNE `fp32_to_bf16_bits` with NaN preservation | GEMM output correct. **Second pass:** `-O0` segfault (4.11), CPU W2 convention differs from GPU W2 (4.14) |
| SmartTieredCache error budget (O8/O9/O11) | `kinesis_unified/sub2bit_gemm_and_smart_cache.hpp` | exact by default (O8); O9 worst-case bound `bound[g] = max_col|scale| * max_m row-L1`, greedy skip while `sum bound <= eps` guarantees per-output abs error `<= eps`; non-finite rows never skipped; promotion counted once per GEMM, saturating (O11); LFU+LRU eviction only when strictly hotter | Correct and rigorous; no defects (confirms observation 3.6; second pass agrees, NaN handled correctly) |
| Host CTA shim PTX primitives (F17/F18/F19) | `fused_w1_hadacore/w1a16_host_cta_shim.hpp` | `lop3_b32` fast-paths 0xAE/0xEA and the generic bit-LUT, all agreeing; W1/W2 unpack use 64.0=0x4280, -129.0=0xC301, H_2=0x3F35 matching the kernel; `__shfl_xor_sync`/`__ballot_sync` assert on intra-warp divergence | Correct for the kernel's operands; double-rounding note corrected by the second pass (observation 3.13) |
| Closed-loop governor (F11) | `hetero_runtime/closed_loop_governor.py` (+ `.c`, E2) | F11 divide-by-zero guard on `locked_clock_floor_mhz`; queue-pressure routing by `delta_q`; clock-sag hysteresis (raise +5% above 2% sag near TDP, relax -2% below 1% sag with headroom); tau clamped to [min,max] | Correct policy model; no defects |
| Unified sizing constants | `kinesis_unified/unified_sizing.h` | W1 28.44x/14.22x, W2 15.06x/7.53x, CTA 4-stage SMEM 98304<=101376, KV INT4 page 4096=1 OS page, descriptors 56/56/128 | All self-consistent |
| Unlocked Triton GEMM configs | `compiler_unlock/unlocked_gemm.py` | all 6 autotune configs satisfy `(BM+BN)*BK*2*stages <= 101376`; grouped `GROUP_SIZE_M=8` L2 swizzle; warps give `W_M x W_N >= 64x64` | Config logic correct; unexecuted on host (caveat 2.4) |
| E2E 5-pillar pipeline (G1) | `e2e_transformer_pipeline.py` | composition of the already-verified pillars end to end | Passes via suite G1; composition only, components verified individually above |
| T-MAC 256-bit vpshufb LUT (F14) | `kinesis_unified/sub2bit_gemm_and_smart_cache.hpp` | 16-entry sign-combination LUT per 4-chunk, 4-panel (32-column) nibble interleave feeding one `_mm256_shuffle_epi8`, scale chain `acc * a_scale * w_scale` | Correct for finite inputs; honestly labeled int8-approximate (see observation 3.5). **Second pass:** NaN dropped, NaN-to-int8 cast is UB (4.12) |

The full suite reports 20/20 passing (`python run_all_tests.py`, 0 warnings) on the
verification host. After the 2026-11-08 fix pass it reports 22/22 (two suites added: E4 Chorus fallback, F4 F21 build matrix).

## 2. Caveats (true limitations, not bugs)

### 2.1 Dynamic BAR1 allocator is a bump pointer with no free-list
**CLOSED (2026-11-09, Audit Check 11).** Previously in `kbusMapFbAperture_HAL`, `nextDynBar1Offset` only ever increased:
the exact-size victim path reused a cold slot's VA in place, while the different-size eviction path (`anyVictim`)
abandoned the evicted slot's VA and bump-allocated fresh VA, leaking BAR1 VA space under mixed-size peer-to-peer
churn. `kbusMapFbAperture_HAL` now falls back to a first-fit coalescing BAR1 VA gap search over the active
(`refCount > 0`) slots whenever the bump pointer cannot satisfy a request, invalidates any cold (`refCount == 0`)
slot whose cached BAR1 VA window overlaps the chosen gap `[bestCand, bestCand + reqSize)`, and resets
`nextDynBar1Offset` to the active high-water mark. Verified in `A3 [2b-3]` across 200 mixed-size (`8 MB`, `16 MB`,
`32 MB`) churn steps on a 64 MB aperture with live pinned mappings and pairwise BAR1 VA + physical FB disjointness
assertions at every step.

### 2.2 PTX is modeled, not assembled
The Python and host-C references model the semantics of `shl.b32`, `shr.u32`,
`lop3.b32`, and `fma.rn.bf16x2`. They prove the bit layouts and dequant algebra are
self-consistent. They do not prove `ptxas` accepts the kernel PTX for `sm_86` or
`sm_89`, nor the register count or occupancy. This is the same gap doc 11 tracks as
F1 and O1 and remains the single largest unverified surface.

### 2.3 Driver and shim code are host mocks
`nv_gpu_ops_and_s3_unlock.c`, the `.patch` sketches, and
`compiler_unlock/libcublas_unlock_shim.c` exercise the logic of a mock of the NVIDIA
driver and a mock driver table. They prove internal state-machine consistency only.
They prove nothing about the real driver, GSP firmware, or hardware latches. The
`.patch` files are explicitly labeled design sketches and have never been applied to
or built against an `open-gpu-kernel-modules` tree.

### 2.4 The Triton QAT kernels are never executed in host verification
`fused_qat_ste.py` defines the real kernels `fused_qat_g128_fwd_kernel` and
`fused_qat_g128_bwd_dx_kernel` under `@triton.jit`, but the autograd Function dispatches
to them only when `x.is_cuda and w.is_cuda`. On a CPU host it always falls back to
`sram_tiled_qat_fwd_reference` / `sram_tiled_qat_bwd_dx_reference`. Suite D1 then compares
that reference against a second host formulation of the same math, so it verifies the STE
algebra, not the Triton kernels. Unlike the `.cu` files (which have a 256-fiber host CTA
shim), the Triton kernels have no host execution path at all, so they are the single
numeric-kernel surface in the repo with zero host coverage. Their tiling, masking, and
`tl.dot` accumulation remain unverified until run on a real GPU.

## 3. Modeling assumptions worth keeping visible

### 3.1 GaLore second-moment transport is exact only for uncorrelated coordinates
`_reproject_moments` uses `V_new = (rot*rot) @ V_old`. Expanding
`E[(sum_j rot_ij g_j)^2] = sum_jk rot_ij rot_ik E[g_j g_k]`, this equals
`sum_j rot_ij^2 V_j` only when the cross terms `E[g_j g_k]` for `j != k` vanish. The
code comment states this honestly. It is a principled approximation, not an identity;
it is sound under the GaLore assumption that the low-rank coordinates are approximately
uncorrelated, and should not be presented as exact.

### 3.2 32B memory table is an internally consistent model, not a measurement
`compute_32b_qat_memory_table` derives each peak as `sum(components) / n_gpus`, so the
headline (23.63 GB/GPU) cannot drift from its own breakdown, and the test asserts this.
The underlying numbers are still a model: the GaLore row keeps 40.00 GB of master
weights resident and stages a further 24.00 GB in host HugePages, and lists quantized
weights as 0.00 GB because the fused STE kernel quantizes on the fly in SRAM. Both are
aggressive systems assumptions (host-to-device weight streaming at training speed, and
true zero-materialization of the dequantized weight). They are labeled as such, and the
table is consistent, but no measurement backs the 23.63 figure.

### 3.3 L1 norm is not invariant under a Hadamard rotation
Only the L2 norm is preserved by an orthogonal rotation. Any sieve or error bound that
is stated in L1 and then travels across a Hadamard rotation is suspect. Doc 11 records
this as F10 and the code comments were corrected, but it is the easiest place for a
future change to silently re-introduce an invalid bound, so it is restated here as a
standing review item.

### 3.4 Clipped STE masks the gradient but not the forward scale
The clipped-STE gate `1(|W| <= clip_mult * alpha_g)` is applied only in the backward pass
(it zeroes `dW` for out-of-range weights). The forward pass still quantizes every weight,
and `alpha_g = mean(|W|)` over the g128 group still includes those outliers, so an outlier
keeps inflating the group scale for all 128 weights in the forward direction while its own
gradient is frozen. This is a defensible outlier-freezing regularizer, but it is not the
PACT-style clipped STE it is named after (which clamps the forward value and matches the
backward indicator to the clamp). Worth renaming or documenting the difference.

### 3.5 T-MAC LUT path is int8-activation approximate by design
`execute_tmac_lut_w1_gemm` quantizes activations to int8 per (row, g128 group) so a LUT
entry fits int16. It is therefore not bit-exact with the FP32 or the AVX2-FMA-unpack paths;
the error is bounded by int8 activation rounding (<= 0.5 LSB, LSB = max|a|/127 per group).
The code states this. The suite's "0 mismatch" check is AVX2 LUT vs scalar LUT (both int8),
which is the correct equivalence to assert; it is not a claim of FP32 parity. Keep these two
error sources distinct when quoting accuracy.

### 3.6 The Python sieve tau is not the C++ error budget
`tiled_cbc_sieve.py` gates a tile on activation energy alone (`max_m mean_k|X| > tau`).
It does not fold in the weight-scale factor, so its `tau_sieve` is an activation-energy
threshold, not a guaranteed per-output error bound. The weight-aware worst-case budget
that carries the `<= eps` guarantee is the C++ `SmartTieredCache::set_error_budget(eps)`
(O9), which multiplies by max column `|scale|`. The Python file is a correct demonstrator
of the row-max (worst-case-over-rows) versus mean-L1 distinction, and it measures the L1 of
the already-rotated activations directly (so it does not fall into the L1-invariance trap of
3.3), but its `tau` should not be quoted as an error budget.

### 3.7 SPSC ring dispatch metadata shares the consumer cache line
The slot struct places `gpu_ready_seq` alone on cache line 0 (producer) and `cpu_done_seq`
plus `num_tokens` / `hidden_dim` / `alloc_via_mmap` on cache line 1 (consumer). The test
asserts the two sequence counters sit on separate 64 B lines, which is true. But the
producer writes the metadata fields on line 1 every dispatch, so the clean "producer line /
consumer line" split only holds for the two counters, not for the whole slot header. It is
not harmful false sharing because the handshake serializes all access to a given slot (the
producer and consumer never touch one slot's header concurrently), so this is a framing note,
not a correctness or measurable-contention issue.

### 3.8 LServe page bound is admissible; its bandwidth stat is modelled
Two notes on `lserve_sparse_attn.py`. First, the page selection upper bound (per-dim
`max(q*k_min, q*k_max)`) is computed from the full-precision rotated keys, while the actual
attention scores the INT4-dequantized keys. The bound is therefore exact for the full-precision
box but only approximate for the quantized keys, so page selection is a heuristic; it cannot
corrupt the result because sink and local pages are always retained and the final attention is
computed on whatever pages are selected. Second, `effective_kv_bw_amplification` is a modelled
ratio (`(16/4.25) * num_pages / avg_active_pages`), i.e. compression times page sparsity, not a
measured bandwidth. This is the same class as the O8 relabelling and the stat name should not be
read as a measurement.

### 3.9 SafeTensors serializer is little-endian only and trusts the header on load
`save_safetensors_bytes` / `load_safetensors_bytes` copy native byte order with no endianness
swap, so they are correct only on little-endian hosts (all x86 and ARM-LE targets here). On load,
tensor slices are taken at the offsets the header states, with no check that the offsets are
contiguous, non-overlapping, or within bounds beyond the final-size assertion. Both are fine for
round-tripping the project's own checkpoints (independently verified bit-exact across bf16, int8,
uint8, int32, fp32, and an empty tensor), but a hardened loader for untrusted files would validate
offsets and handle endianness.

### 3.10 cuBLAS shim is host-mock-verified only, and leans on community device IDs
`libcublas_unlock_shim.c` is reviewed as logic against a mock driver table; its real
`LD_PRELOAD` behaviour against closed-source `libcuda.so.1` is unverified (the doc 11
caveat stands). The caller gating is the important safety property and it holds in the mock:
only a blas/dnn caller receives the workstation ID, while PyTorch and NCCL always read the
true device ID and the physical SM count. Two notes: the RTX 4090 D, 5090, and PRO 6000
Blackwell IDs are community-reported and unverified on hardware (doc 11 O17), and the `ioctl`
hook reads its third argument with `va_arg(void*)`, which assumes a pointer-typed argument
(the standard shim limitation; non-pointer ioctls are passed through untouched but the read
itself is technically unspecified for an int-argument ioctl).

### 3.11 SMSP roofline peak is calibrated to the documented TFLOPS, not measured
`smsp_roofline_sim.py` is internally consistent: `I_warp`, the compute-to-SRAM ratios, and the
DRAM ridge points all follow from the stated equations. But `peak_tflops` is constructed so that
`sm_count * (2048/tau) * smsp * clock` reproduces the raw-MMA TFLOPS quoted in doc 00/01 (for
example RTX 4090 locked 165.6, unlocked 331.6), which are themselves unverified external claims.
So the roofline's absolute numbers inherit that unverified-ness. The decode 2D-vs-Stream-K and the
hetero CPU/GPU splits are explicitly banner-labelled analytical models with stated assumptions
(`SM_FRACTION_TO_SATURATE_DRAM = 0.5`, DDR5 80 GB/s, PCIe 25 GB/s). The `simulate_smsp_cycle_accurate`
docstring says "cycle-accurate", which oversells what is an analytical contention model with
heuristic stage-refill latencies; the arithmetic is sound but the label should read "analytical".

### 3.12 CPUID L2/L3 discovery accepts cache-type 2 (instruction)
**FIXED (2026-11-09, docs/14 section 8.6).** L2 and L3 now accept only type 1 (data) or type 3 (unified), as L1D did.

`query_deterministic_caches` classifies an L2 or L3 entry when `cache_type == 2 || cache_type == 3`,
where type 2 is an instruction cache and type 3 is unified. On real x86 parts L2 and L3 are reported
as unified (type 3), so this never mis-sizes in practice (the F3 bench reads L1D=48KB, L2=1280KB,
L3=24576KB correctly on the dev host). But an instruction-only L2 would be accepted as the data-path
L2 budget. Harmless today; strictly the L2/L3 branches should accept only type 3 (unified) or type 1
(data), as the L1D branch already does. No functional impact observed.

### 3.13 Host bf16x2 FMA rounds twice; exact in practice for these operands
`host_ptx_fma_rn_bf16x2` (and `ptx_fma_rn_bf162` in the emulator) converts each bf16 lane to f32, calls
`fmaf`, then rounds the f32 result back to bf16. That is two roundings (the add inside `fmaf` rounds to
f32, then f32 rounds to bf16), whereas real `fma.rn.bf16x2` rounds once directly to bf16.

CORRECTED by the second pass ([docs/14](16_Subsystem_Audit_Interleaving_Seams_and_Testing_Architecture.md)): an earlier
version of this note argued that the `fmaf` step "introduces no rounding" because bf16 x bf16 products are
exact in f32. The product is exact, but the add is not. For general bf16 triples, double rounding does
differ from single rounding: a random search found 28,019 counterexamples in 24.7M triples (for example
`a=0x5983, b=0x23C0, c=0x308F` gives 0x3DC4 from the shim against the correctly rounded 0x3DC5, because
the product lands exactly on a bf16 midpoint and the tiny `c` is lost in f32).

The conclusion still holds for the operand sets the kernel actually produces. The W1/W2 unpack is exact,
and the H_2 butterfly and the sieve's `fma(x, 1, y)` each gave 0 mismatches against single rounding over
50M samples. So the shim matches hardware for every value on the current kernel paths, but it is not a
general `fma.rn.bf16x2` model and must not be reused as one without a single-rounding implementation.

## 4. Inconsistencies and latent defects found in this pass

### 4.1 Lazy-SVD calibration: doc 11 describes an older function and older numbers
Doc 11 section 0.3 (Wave 3C) describes a function `calibrate_lazy_svd_threshold`
benchmarking thresholds `{0.00, 0.85, 0.90, 0.95, 1.00}` and states that threshold 0.90
"skips 80% of SVDs (2/10 calls)" while preserving 0.9811 cosine similarity.

The code in `galore_async_offload.py` actually defines
`calibrate_lazy_subspace_drift_o15` benchmarking `{0.85, 0.90, 0.95}` against a
fixed-interval baseline, and a fresh run reports threshold 0.90 skipping about 69%
(5 of 16 SVDs), cosine similarity 0.9810. The function was renamed and recalibrated;
doc 11 was not updated. The cosine similarity still matches (0.981), but the function
name, the threshold set, and the skip fraction in doc 11 are stale. RESOLVED: doc 11
sections 0.3 and the O15 row were updated to the current name and the 5-of-16 / about 69%
figure in the same commit that introduced this document.

### 4.2 Doc 11 F18 claims smsp_roofline_sim.py carries the orthonormal H_2 constant; it does not
Doc 11 section 0.2 and the F18 row state that both `ptx_bf162_emu.c` and
`smsp_roofline_sim.py` were updated to use `0x3F353F35` / `0xBF35BF35`. A search of the
current `smsp_roofline_sim.py` finds no Hadamard, Kronecker, or H_2 butterfly constant at
all: it is a pure SMSP warp-scheduler and roofline simulator. The orthonormal constant is
present and consistent only in `ptx_bf162_emu.c` and `fused_w1_hadacore/w1a16_hadacore_kernel.cu`.
The fix itself is correct where it lives; the claim that it also touched
`smsp_roofline_sim.py` is stale (the file either never held the constant or was later
refactored). Recommend dropping `smsp_roofline_sim.py` from the F18 and section 0.2 text.

### 4.3 Latent defect: single-threaded fallback deadlocks the bounded ring
**FIXED (2026-11-08, docs/14 section 8).** The single-threaded build now drives the consumer inline whenever the producer would block; suite E4 builds that path on the host and passes 8/8.

**Second pass: CONFIRMED by reproduction** (platform macros renamed so the `#else` path builds: the producer
spins at `seq = 3` and the run hangs, timeout exit 124). The same path also misallocates the ring (4.15).

In `chorus_amx_moe.c`, on `_WIN32` and `__linux__` a worker thread is spawned (line ~509 /
~513) before the producer loop, so producer and consumer run concurrently and suite E1 passes.
On any other platform (for example macOS or BSD) no thread is created, and the inline
`chorus_amx_worker_loop(&ctx)` is called only after the producer loop (line ~556). With
`num_slots < total_dispatches` (the test uses `num_slots = 2`, `total_dispatches = 8`), the
producer blocks at `seq = num_slots + 1` waiting for `cpu_done_seq >= seq - num_slots`, which
never advances because the consumer has not run yet. Result: a hard deadlock on non-Win and
non-Linux hosts. The Win/Linux CI never exercises this path, so it is latent. Severity is low
(the project targets Win/Linux hosts), but the fallback's placement is simply wrong for a
bounded ring: on a single-threaded platform the consumer must be interleaved with the producer,
or `num_slots` must be at least `total_dispatches`.

### 4.4 Latent defect: ring slot count truncated to 16 bits on free
**CLOSED (2026-11-08, docs/14 section 8).** `_Static_assert(1 GB / sizeof(slot) <= 0xFFFF)` plus a runtime reject above 65,535 slots.

**Second pass: DISPUTED as stated, unreachable with current constants.** A slot is 263,296 B, so the 1 GB
guard (`bytes <= 1 GB`, which the paragraph below omits) caps the ring at 4,078 slots, not about 7,700, and
65,535 is never reachable. It becomes reachable only if a slot shrinks below 16 KiB (for example
`hidden_dim <= 128`). A `static_assert(HUGE_PAGE_1GB / sizeof(slot) <= 0xFFFF)` closes it.

`allocate_pinned_1gb_ring` packs the slot count into `alloc_via_mmap` bits [31:16]
(`(num_slots & 0xFFFF) << 16`), and `free_pinned_ring` recovers it with `>> 16`. The alloc
guard rejects only `num_slots > SIZE_MAX / sizeof(slot)`, not `num_slots > 0xFFFF`. A caller
passing more than 65535 slots allocates the full buffer but stores a truncated count, so the
free path computes a smaller byte size and unmaps or unlocks the wrong length (partial unmap
or leak). The intended 1 GB regime holds at most 4,078 slots (corrected from "roughly 7700"), well under the limit, so this is
latent, but the allocator should reject `num_slots > 0xFFFF` explicitly or widen the tag.

### 4.5 Defect (reachable): C ABI size contract is implicit and the copies are unbounded
**FIXED (2026-11-08, docs/14 section 8).** Every packed buffer crosses the ABI with its element capacity; short buffers return -3 untouched, and the GPU convert rejects `K % 128 != 0` in C++, the ABI and Python. Regression: F2 [5].

**Second pass: CONFIRMED and UPGRADED from latent to reachable.** `KinesisNativeBridge.convert_w1_to_gpu_interleaved`
allocates `N * (K // 128)` bf16 scales, while `convert_w1_gotoblas_to_gpu_interleaved_u32` writes
`N * ceil(K / 128)` and the ABI only checks `K % 32 == 0`. With K=160 and N=8, the bridge allocates 8 scales and C++
writes 16, a 16-byte heap overflow (measured with a sentinel-padded buffer). Every K that is 32, 64 or 96
mod 128 triggers it from the in-repo caller. The `pack_*` size formulas agree for every odd shape tested.
The original text below ("no bug today") is superseded. Fix plan: docs/14 F-A.

The `kinesis_c_abi.cpp` entry points take raw output pointers (`out_bits`, `out_scales_bf16`,
`out_gpu_u32`, ...) with no capacity argument, and `memcpy` into them a length the C++ side
computes from the panel set (for example `panels.packed_bits.size()`). The reconstruction
entry points do the mirror image: they `.assign(ptr, ptr + num_panels * K)` from the input
pointer with no length check. Safety rests entirely on the Python caller in `kinesis_bridge.py`
allocating buffers whose size formula (`num_panels * K`, `num_panels * groups_per_col * 8`, ...)
matches the C++ layout exactly. For every tested shape the two formulas agree and suite F2
round-trips with zero bit errors, so there is no bug today. But the contract is unchecked: a
caller under-allocation, or any future shape where the Python and C++ size formulas diverge,
is a silent heap overflow (on output) or out-of-bounds read (on reconstruction) with no guard
and no diagnostic. Recommend passing output and input capacities across the ABI and bounding
every `memcpy` / `.assign`, or a two-call size-query pattern (ask for required length, then fill).
Severity is low for the current in-repo caller, but this is the most dangerous failure mode in
the bridge because it corrupts memory rather than returning an error code.

### 4.6 Defect (HIGH, second pass): GaLore uint8 round-up `sqrt(V)` storage ratchets
**FIXED (2026-11-08, docs/14 section 8).** 8-bit log code with stochastic rounding that is unbiased in V (provably E[V_hat_t] = V_t). Measured median 0.988, E[V_hat/V] 0.999 over 3000 steps. Regression: D2 [V].

`quantize_unsigned_uint8(..., round_up=True)` applies `ceil` every step. Because the stored value is
dequantized, updated and re-quantized each step, the round-up error compounds:
`e_{t+1} <= sqrt(b2) e_t + q`. The steady state is about `q / (1 - sqrt(b2))`, roughly 2000 quanta
before saturating. In a simulation (r=16, n=512, b2=0.999, per-column gradient scale 1e-3 to 1), the
median stored/true `sqrt(V)` was 1.48 at step 10, 3.82 at step 100, 9.21 at step 1000 and 10.49 at step
3000. Small-gradient columns reached about 52x and large-gradient columns about 2.5x. Adam steps
therefore shrink by 2.5x to 50x over training: a hidden learning-rate decay that differs per coordinate.
"Rounding up only shrinks steps (conservative)" is true for one step, but the compounded effect is not
bounded in practice. This answers 5(d). The 6-step suite D2 cannot see it. Fix plan: docs/14 F-B.

### 4.7 Defect (HIGH, second pass): roofline FLOP-per-HMMA inconsistency
**FIXED (2026-11-08, docs/14 section 8).** tau re-based to the m16n8k16 interval (32 locked / 16 unlocked, peaks unchanged), with an assertion that `t_mma` and `peak_tflops` agree. Every row is now compute-bound in both states; docs/01 table corrected.

`peak_tflops` uses 2048 FLOP per HMMA (`2048/tau * 4` per SM per clock), but `t_mma` counts m16n8k16
instructions, and each of those is `16*8*16*2 = 4096` FLOP (the code's own comment says this). The implied
SRAM ridge is therefore 8 (locked) / 16 (unlocked) FLOP/B, not the stated 4 / 8. Recomputing the margins
with `I_warp = W_M W_N / (W_M + W_N)` against the stated ridge gives 32x32 at 3.39x / 1.69x, 64x32 at
4.76x / 2.38x, 64x64 at 8.0x / 4.0x, 128x64 at 10.67x / 5.33x and fused W1 at 14.95x / 7.47x. Every row
is compute-bound both locked and unlocked. The only "stalls when unlocked" row (32x32: ratio 0.85x,
50.76% SassWait) is an artifact of the 2x error. Also: the 0.1 split-K factor contradicts its own
"upper bound" comment, and tau, beta_conflict and the stage-refill latency are stated rather than derived.
Fix plan: docs/14 F-E.

### 4.8 Defect (HIGH for training, second pass): clipped STE at c=1.5 permanently freezes weights
**MITIGATED (2026-11-08, docs/14 section 8).** Severity corrected to LOW: the clip is opt-in, and the in-repo e2e value c=2.5 measured harmless (4.5% gated, loss and sign accuracy equal to unclipped). `clip_mult` in (0, 2.5) now warns with the gated fraction erfc(c/sqrt(pi)); negative or non-finite values are rejected. The self-test uses 2.5.

With a Gaussian init, `P(|w| > 1.5 * alpha_g)` is about 23%. Those weights get zero gradient from step 0,
and since `alpha_g` does not shrink, they never re-enter the range. About half of them have the wrong
sign. In a teacher-student run (binary-g128 teacher, M=256, K=512, N=128, Adam 1e-4, 400 steps), sign
accuracy capped at 0.873 against 0.919 unclipped, and loss ended at 0.058 against 0.038. This adds to
observation 3.4. Note: `clip_mult` defaults to 0 (clipping off) in both `FusedQATG128Function` and
`FusedQATG128Linear`, and `c=1.5` is only used by the file's own self-test, so this is a latent risk
of the opt-in clipped mode, not shipped-default behaviour. Fix plan: docs/14 F-C.

### 4.9 Defect (MED, second pass): bf16 master weights stall under a plain optimizer
**MITIGATED (2026-11-08, docs/14 section 8).** `FusedQATG128Linear(master_dtype=...)` added; bf16 stays the default for the memory budget, with the caveat documented. Regression: D1 [6] (fp32 masters 100% of weights updated).

`FusedQATG128Linear` defaults to bf16 master weights. With a plain torch optimizer, 73.5% (SGD lr 0.5) /
56% (Adam 1e-4) of weights get no update on a given step, because the update is below half a bf16 ulp.
SGD loss ended at 7.5e-4 with bf16 masters and 2.4e-4 with fp32 masters (start 0.262). The GaLore engine's
`copy_to_param_stochastic` avoids this; a plain optimizer does not. With fp32 masters a smaller mismatch
appears instead: 2.4% of groups get a scale 1 bf16 ulp off the packer's, and tiny negatives quantize with
the opposite sign. Fix plan: docs/14 F-D.

### 4.10 Convention gap (MED, second pass): QAT trains the wrong basis and has no ternary path
**FIXED (2026-11-09, docs/14 section 8.5).** `DeployBasisQATLinear(fmt="w1"|"w2", rotation=R)` trains in the deploy basis (forward `(x R) W_hat(W R)^T`, STE on `W R`, master kept unrotated) with the packer's exact W1 and W2 rules. D1 [7]: QAT `W_hat` vs the packer's pack -> unpack has 0 mismatches for W1 and W2 under both kernel rotation modes, before and after 60 training steps, and the trained forward equals the deployed forward bit for bit; W2 has about 36% zeros. Control: the old BF16-first scale rule gives 256/16,384 mismatches against the packer. This is a host autograd path; the fused Triton kernels still implement only the unrotated W1 case.

The packer quantizes `W @ R1` but QAT trains unrotated W, and sign agreement between the two is 0.487
(chance), so QAT results do not transfer to the rotated deploy. QAT also implements binary only: the
packer's W2 has about 31% zeros while QAT `W_hat` has 0%. The "0 bytes for W_hat" claim holds for the
Triton forward and dX kernels, but the clip path allocates about 226 MiB at 4096^2 (about 7x a bf16
`W_hat`), and line 242 prints a hardcoded string rather than a measurement. Fix plan: docs/14 F-G.

### 4.11 Defect (MED, second pass): F21 stack-alignment fix does not cover -O0 / -fno-inline
**FIXED (2026-11-08, docs/14 section 8).** No by-value `__m256` anywhere, `-Wa,-muse-unaligned-vector-move` in every mingw build, and a `#warning` for `-mavx2` without `-mfma`. Suite F4 builds -O0, -O3 -fno-inline and -O3, runs them, and scans for aligned YMM stack moves (0 in each).

A `-O0` mingw build of `test_kinesis_unified.cpp` segfaults on `vmovaps %ymm0,-0x50(%rbp)` at hpp:920
(`execute_smart_w1_gemm`). `-O3 -fno-inline` still emits an unrealigned `vmovaps` ymm stack store for the
by-value `__m256 v_scale` argument of `avx2_unpack_*`, and survives only by stack luck. Builds at `-O1`
and above with inlining, and the shipped DLL, have 0 aligned ymm stack moves. This is GCC PR 54412.
Also: the AVX2 gate requires `__FMA__`, so a build with only `-mavx2` silently compiles the scalar path.
Fix plan: docs/14 F-I.

### 4.12 Defect (MED, second pass): NaN activations silently skipped
**FIXED (2026-11-08, docs/14 section 8).** T-MAC accumulates non-finite groups exactly in FP32 (no int8 cast), and all gates use NaN-sticky max and `!(x <= tau)`. Also fixed: one Inf used to make the T-MAC sieve skip every finite group of its row. Regressions: F1 [9], C1 [5], C2 [5], B3 [6].

* The Python sieve does not execute a NaN tile, giving 64/64 finite rows.
* The `.cu` mask is 0x0 even at tau=0, because `NaN*alpha > tau` is false and `fmaxf` drops NaN.
* The in-kernel `mi_max > group_max` is false for NaN; this is safe only at the default `tau <= 0`,
  which short-circuits.
* T-MAC: `amax = std::max(amax, fabs(v))` ignores NaN.
  * A NaN in an otherwise-zero group leaves that group's scale at 0, so the group is skipped and all
    outputs come out finite.
  * A NaN in a live group reaches `static_cast<int8_t>(NaN)`, which is UB and produced finite garbage.

The C++ SmartTieredCache maps non-finite values to inf and never skips them, which is correct.
Fix plan: docs/14 F-H.

### 4.13 Semantics (MED, second pass): only the C++ sieve carries the `<= eps` guarantee
**CLARIFIED (2026-11-08, docs/14 section 8).** The `.cu` gate is documented as a per-sub-tile threshold, and `tiled_cbc_subtile_tau_for_budget(eps, K) = eps/(K/32)` gives a sound total budget by the union bound (C1 [6]). The Python and in-kernel gates are documented as activation-energy thresholds.

The cumulative sorted-bound budget exists only in `SmartTieredCache::plan_panel_sieve`.
`tiled_cbc_subtile_rowmax_bound_mask` thresholds each 32-wide sub-tile independently, so total skipped
error is up to (#skipped sub-tiles) x tau. `warp_sieve_k32_active` has no weight-scale factor and
accumulates the row L1 in bf16 (it rounds and can under-estimate). This generalises observation 3.6 to
the `.cu` and kernel gates. Fix plan: docs/14 F-J.

### 4.14 Convention gap (LOW-MED, second pass): CPU W2 quantizer differs from GPU W2
**FIXED (2026-11-08, docs/14 section 8).** The CPU packer now uses BitNet b1.58 absmean, the GPU rule. CPU vs GPU dequantized mismatches: 0/20,480 (F2 [6]).

The CPU scale is mean|w| over `|w| >= 0.5*mean|w|` with round-half-away; the GPU/spinquant scale is plain
mean|w| with round-half-even and a 1e-6 clamp. On the same weights, 128/128 groups get a different scale,
1507/16384 trits differ, and the zero fraction is 0.402 vs 0.310. The "L2-optimal over active trits"
comment is also not self-consistent, because the set used for the scale differs from the final support.
Fix plan: docs/14 F-F.

### 4.15 Latent defect (LOW, second pass): generic ring allocator is misaligned
**FIXED (2026-11-08, docs/14 section 8).** Portable over-allocated 64-byte alignment (UCRT has no `aligned_alloc`), with an unlock-then-free path. Exercised by E4.

The `#else` allocator in `chorus_amx_moe.c` uses `calloc` for an `aligned(64)` struct. The alignment
assert fires on that build, and over-aligned `_Atomic` access through it is UB. macOS falls into this
branch, because `__unix__` is not defined on Apple. It is the same platform set as 4.3. Fix plan: docs/14 F-K.

### 4.16 Test-coverage gaps (LOW, second pass)
**FIXED (2026-11-08/09, docs/14 sections 8 and 8.5)** except the PTX MMA emulator item. Cayley: improvement over the initial rotation (2.59 -> 1.82). LServe C3 [7]: blind vs profiled on the same tensor, 0.857 vs 0.993. C1 [4b]: the legacy mean-L1 gate is now run (mask 0x0, dropping a sub-tile whose true contribution is 16 > tau 10); C2 [4] already ran its legacy path. D1: dW is checked against autograd through the STE construction, for both the plain and the clipped path. D2: the memory table is checked against bytes per parameter and hand-worked peaks (227.40 / 131.40 / 23.63). A4 (new, Linux via WSL): `dladdr` classification 5/5, including that attribute 34 with a real-driver slot value is never remapped. Still as stated: the PTX MMA emulator matches its reference by construction, so it is no evidence of HMMA bit-exactness.

Several suites pass for reasons other than the property they claim to check:
* **Cayley (B4)**: `k_after < k_before` passes with 0 steps, because the fixed Hadamard init alone gives
  25.00 -> 2.594.
* **LServe (C3)**: blind and profiled are measured on different datasets.
* **cuBLAS shim (A1)**: every control uses the forced override or the TLS latch, so the `dladdr`/backtrace
  classification never runs. Also, on real drivers attribute 34 is a PCI slot number (0..31), so the
  `>= 0x1000` spoof gate never fires outside the mock.
* **Sieve negative controls (C1/C2)**: they compare against a mean or a hand-computed assertion rather than
  running the legacy heuristic.
* **QAT D1**: the dW check compares `dy^T x` against `dy^T x`.
* **GaLore D2**: the peak table assert is tautological (`sum/2` of itself).
* **PTX MMA emulator**: matches its reference by construction (same k order), so it is no evidence of
  hardware HMMA bit-exactness.

### 4.17 Minor items (second pass)
**FIXED (2026-11-08/09, docs/14 sections 8 and 8.5)** except the V-transport caveat, which stays a documented limitation. AMX: the bullet below was wrong about the SDM. The SDM Operation pseudocode keeps two FP32 lanes (even and odd products) that start at zero, accumulates each with RNE and DAZ/FTZ, then adds the lane sum to dst once per instruction. Neither the old code nor `(c + a0b0) + a1b1` matched; the emulator now follows the pseudocode, and E1 [2] checks a hand-derived case where the SDM order gives 1.0 and both alternatives give 2.0. Not measured on hardware. LServe: outward-rounded centroids (RNE would break containment in 4,121 of 8,192 page maxima) and 4.625 / 4.125 bpw. H_2 gain documented in the kernel. "Exhaustive" relabelled "sampled".

* The AMX emulator computes `c + (a0b0 + a1b1)`; the SDM is `(c + a0b0) + a1b1`. (Corrected in 8.5: the SDM
  is a two-lane order, which neither expression matches.)
* `ptx_max_bf162` returns b when b is NaN; PTX max returns the non-NaN operand.
* `GET_NAME_STRING` calls `strstr` on a buffer that may not be NUL-terminated.
* The mode-2 H_2 gain is `2*0x3F35^2 = 0.99979`, a -1.07e-4 systematic bias against the packer's exact
  1/sqrt2. It is below bf16 noise but undocumented. (Precisely: amplitude gain 1 - 1.068e-4 per stage, energy
  gain 1 - 2.136e-4; about 18x below a BF16 half-ulp.)
* LServe centroids are rounded with RNE rather than outward, and the K payload is 4.625 bpw, not 4.25.
* The "exhaustive 2^16 noise states" in the emulator are 65,536 pseudo-random patterns.
* The V transport (3.1) with correlated coordinates has max relative error 1.44; it is not a small
  correction.

## 5. Open cross-check items (pending a second independent reviewer)

The areas in section 1 were reviewed by a single pass. A second independent reviewer
should attempt to break, not confirm, each conclusion: in particular (a) whether the
Cayley step is Riemannian or only embedded-Euclidean descent, (b) whether any bf16
scale defeats the W2 zero-preserving unpack, (c) whether any map/unmap sequence aliases
two live BAR1 mappings, (d) whether the GaLore V transport bias from uint8 round-up
storage is bounded, and (e) whether the SPSC ring has any reordering the acquire/release
pairs miss under a weak memory model (the current pass reasoned on x86 TSO plus the explicit
C11 orderings). The second reviewer should read both the code and this document and confirm
or dispute each finding here, including the three latent defects in section 4. Audit checks for this
second pass are maintained outside the repository.

**Second-pass answers** ([docs/14](16_Subsystem_Audit_Interleaving_Seams_and_Testing_Architecture.md), 2026-11-08 / 2026-11-09):
(a) Riemannian GD with a Cayley retraction, confirmed. (b) Only negative or non-finite scales break the
zero, and absmean never produces them. (c) **Checked via Audit Check 11 (2026-11-09):** found a cold-slot partial
physical overlap sequence that left two live slots covering overlapping physical FB ranges (4.20, now fixed) and
closed the mixed-size VA leak (2.1) with a first-fit coalescing VA gap allocator. (d) Not bounded in practice; it
ratchets (4.6). (e) No missed reordering found. Latent defects: 4.3 confirmed, 4.4 unreachable today,
4.5 confirmed and reachable.

### 4.20 Defect (found 2026-11-09 by Audit Check 11): cold BAR1 slot revived across a live partial physical overlap
**FIXED (2026-11-09).** In `kbusMapFbAperture_HAL`, Step 1 iterated `i = 0..7` and returned immediately on an exact
match (`slotStart == physTarget && slotSize == reqSize`), while ignoring partial physical overlaps against cold
(`refCount == 0`) slots and leaving those cold slots `valid = NV_TRUE`. Consequently, mapping 16 MB at `33 GB` into
slot 0 and unmapping it (`slot 0` cold, `valid = 1`), then mapping an overlapping 8 MB sub-range at `33 GB` into
slot 1 (`slot 1` live, `refCount = 1`), and finally re-mapping 16 MB at `33 GB` while slot 1 was still live caused
the loop to hit `slot 0` at `i = 0` and return `NV_OK` before inspecting `slot 1` at `i = 1`. That left two live
(`refCount = 1`) slots covering overlapping physical VRAM (`[33 GB, +16 MB)` and `[33 GB, +8 MB)`) at distinct BAR1
VA offsets and made `kbusUnmapFbAperture_HAL` ambiguous because both started at `33 GB`. Step 1 now scans all 8
slots before taking `exactHitSlot`, rejects any overlap with a `refCount > 0` slot, and invalidates (`valid =
NV_FALSE`) any cold (`refCount == 0`) slot overlapping `[physTarget, physTarget + reqSize)` with a different start
or size; `refCount == ~0U` is also guarded against overflow. Regression: `A3 [2b-2]`.

### 4.19 Defect (found 2026-11-09): g128 scale rounding depended on summation order
**FIXED (2026-11-09, docs/14 section 8.6).** The new G1 cross-check (deploy-basis QAT vs every packer) found the
C++ W1 packer and the Python packer disagreeing on one group of 1,024 (128 weights). Both computed
bf16(mean|w|), but the FP32 sum is not order-defined: the C++ loop sums sequentially and torch pairwise, and near a
BF16 rounding boundary the two picked adjacent scales. The rule is now: accumulate in FP64, then round once
directly from FP64 to BF16 (round-half-even, via `frexp` / `ulp_exp = max(e - 8, -133)` / `nearbyint` / `ldexp`, so
there is no FP32 double rounding on either normal or subnormal BF16 values).
It is unified across the Python W1/W2 packers (`round_f64_to_bf16_rne`), the C++ W1/W2 packers
(`f64_to_bf16_bits_rne`) and the deploy-basis QAT quantizer. FP64 sums in
different orders still differ by about 1e-16 relative, so a disagreement now needs the exact mean to sit that
close to a BF16 tie. Regressions: G1 (QAT vs Python W1, Python W2 and C++ W1: 0/0/0), F2 [7] (C++ vs Python
scales over 16,384 groups: 0), F1 [9] (W2 scale vs an independent FP64 reference: 0/128, tightened from "a
few ulp", plus all 127 BF16 subnormal double-rounding traps), B4 [7] (all 127 subnormal double-rounding traps in
Python). The unrotated Triton QAT kernels still compute the scale in FP32 on the GPU and can differ from the
packer by one BF16 ulp in rare groups; the host deploy-basis path is the bit-exact one.

### 4.18 Model scope (third pass): roofline table omits per-warp `ldmatrix` latency
**DOCUMENTED + MODELLED (2026-11-09, docs/14 section 8.5).** A third reviewer pointed out that the table models `ldmatrix` as
crossbar throughput only. The claim that it reported 0.00% SassWait for 32x32 was not reproduced (7.01% / 13.08%),
and the claim that 8 HMMA per step cannot hide a ~24-cycle latency holds only without fragment prefetch. The
simulator now has `fragment_prefetch` and an assumed `LDMATRIX_LATENCY_CYCLES = 24`; table [1b] shows the
no-prefetch case (32x32 unlocked 25.22%) and asserts that with prefetch the term changes nothing. docs/01 states
what the table measures.

## 6. Coverage summary

This pass reviewed the full host-verifiable surface of the repository, file by file:

* **Pillar math and kernels**: Cayley-SGD rotation and the sub-2-bit unpackers (`spinquant_packer.py`),
  the PTX emulator constants and the host CTA shim primitives (`ptx_bf162_emu.c`,
  `w1a16_host_cta_shim.hpp`), the Tiled-CBC Holder sieve (`tiled_cbc_sieve.py`), and LServe sparse
  attention (`lserve_sparse_attn.py`).
* **CPU engine**: the 4x8 GotoBLAS GEMM, W1/W2 packers and unpackers, the T-MAC LUT kernel, and the
  SmartTieredCache error budget (`sub2bit_gemm_and_smart_cache.hpp`); the memory subsystem and hardware
  discovery (`memory_primitives.hpp`, `hardware_and_sandbox.hpp`); the C ABI and SafeTensors bridge
  (`kinesis_c_abi.cpp`, `kinesis_bridge.py`); and the sizing single-source-of-truth (`unified_sizing.h`).
* **QAT / optimizer**: the fused STE (`fused_qat_ste.py`) and 8-bit GaLore with moment reprojection
  (`galore_async_offload.py`).
* **Runtime and firmware mocks**: the SPSC AMX ring (`chorus_amx_moe.c`), the closed-loop governor
  (`closed_loop_governor.py` / `.c`), the driver BAR1 and S3 mock (`nv_gpu_ops_and_s3_unlock.c`), and the
  cuBLAS device-ID shim (`libcublas_unlock_shim.c`).
* **Models and integration**: the SMSP roofline model (`smsp_roofline_sim.py`), the unlocked Triton GEMM
  configs (`unlocked_gemm.py`), the empirical CPU roofline bench (`bench_cpu_roofline.cpp`), and the
  end-to-end 5-pillar pipeline (`e2e_transformer_pipeline.py`).

> [!WARNING]
> The second pass ([docs/14](16_Subsystem_Audit_Interleaving_Seams_and_Testing_Architecture.md)) **disputes** the
> paragraph below. It found correctness defects in a quantization path (4.6), a model's published
> verdicts (4.7), a marshalling path (4.5, reachable heap overflow), training behaviour (4.8 to 4.10),
> and NaN handling (4.12). The paragraph is kept as the first pass's record.
> A fix pass on 2026-11-08 (docs/14 section 8) closed 4.3 to 4.7, 4.11, 4.12, 4.14 and 4.15 in code, with
> regression tests (22/22 suites). A third pass on 2026-11-09 (docs/14 section 8.5) closed 4.10 and the rest of
> 4.16/4.17 except the PTX MMA by-construction note and the V-transport caveat (23/23 suites).

Result of the first independent pass: **no correctness defect** was found in any math, quantization,
serialization, or marshalling path. Every such path was re-derived and, where runnable, executed
(all 20 host suites pass). The findings are: **three low-severity latent defects** on untested paths
(sections 4.3, 4.4, 4.5, none reached by the Win/Linux CI), **two documentation inconsistencies** in
doc 11 (sections 4.1, 4.2, both fixed), and a set of caveats and modelling assumptions (sections 2 and 3)
that are true-by-design limits rather than bugs. The single largest unverified surface remains the GPU
device code: the PTX is modelled, not assembled, and the Triton kernels have no host execution path, so
nothing here proves `ptxas` acceptance, register or occupancy counts, or any on-GPU performance figure.
