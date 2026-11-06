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
| Cayley-SGD Stiefel step | `fused_w1_hadacore/spinquant_packer.py` | Skew-symmetry of `Y = G R^T - R G^T`; orthogonality of the Cayley transform; the descent identity `<G, Y R> = 0.5*||Y||_F^2 >= 0` expanded via `P = G R^T` | Descent confirmed; the F13 fix is correct |
| W1 binary unpacker (`lop3.b32` 0xAE) | `fused_w1_hadacore/spinquant_packer.py` | `(~a & 0x80008000) | 0x3F803F80` maps sign bit to bf16 +1.0 / -1.0, then `fma(+/-1, scale, -0.0)` | Bit-exact; 0 mismatches on run |
| W2 ternary unpacker (`lop3.b32` 0xEA) | `fused_w1_hadacore/spinquant_packer.py` | `bf16(0x4000+q)` then `inj*64 - 129` gives -1 / 0 / +1 with a single exact fma rounding; q=1 yields exactly 0.0 for any scale | Zero-preserving property holds; 0 mismatches |
| Driver BAR1 lifetime (F7 class) | `kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c` | No two live (refCount>0) mappings can share a BAR1 VA window; VA reclaimed only at refCount 0; double-unmap rejected | Invariant holds (see caveat 2.1) |
| GaLore moment reprojection (F20/O14) | `qat_galore_engine/galore_async_offload.py` | `M_new = (P_new^T P_old) M_old`; `V_new = (rot*rot) @ V_old` from `E[(rot@g)_i^2]`; QR sign fix `q * sign(diag(R))` | Correct (see caveat 3.1 on V) |
| Tiled-CBC row-max Holder sieve (O9) | `structural_filters/tiled_cbc_sieve.py` | `e_k = max_m mean_k(|X[m,tile]|)` is worst-case over rows; by Holder `|sum X W| <= ||X[m,tile]||_1 * max_k|W|`; single loud row keeps its tile | Correct (see observation 3.6) |
| H_2 butterfly constant (F18) | `sim_harness/ptx_bf162_emu.c`, `fused_w1_hadacore/w1a16_hadacore_kernel.cu` | `0x3F353F35` = +0.70703125 ~= +1/sqrt(2); `0.25*(1/sqrt(2)) = 1/sqrt(32)` orthonormal H_32 scale | Consistent across both files (see inconsistency 4.2) |
| Fused STE QAT forward/backward | `qat_galore_engine/fused_qat_ste.py` | `Y = X W_hat^T`, `dX = dY W_hat`, STE `dW = dY^T X`, and the clipped-STE gate `1(|W| <= c*alpha_g)` | Math correct (see caveat 2.4 and observation 3.4) |
| T-MAC 256-bit vpshufb LUT (F14) | `kinesis_unified/sub2bit_gemm_and_smart_cache.hpp` | 16-entry sign-combination LUT per 4-chunk, 4-panel (32-column) nibble interleave feeding one `_mm256_shuffle_epi8`, scale chain `acc * a_scale * w_scale` | Correct; honestly labeled int8-approximate (see observation 3.5) |

The full suite reports 20/20 passing (`python run_all_tests.py`, 0 warnings) on the
verification host.

## 2. Caveats (true limitations, not bugs)

### 2.1 Dynamic BAR1 allocator is a bump pointer with no free-list
In `kbusMapFbAperture_HAL`, `nextDynBar1Offset` only ever increases. The exact-size
victim path reuses a cold slot's VA in place, but the different-size eviction path
(`anyVictim`) abandons the evicted slot's VA and bump-allocates fresh VA. Under mixed-size
peer-to-peer churn this leaks BAR1 virtual address space and will eventually return
`NV_ERR_INSUFFICIENT_RESOURCES` even when slots are logically free. It fails safe
(it refuses, it never aliases two live mappings), but a production driver needs a real
VA allocator (interval tree or free-list), not a monotonic bump pointer. The 8-slot
test with uniform 16 MB allocations does not exercise this path.

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

## 4. Inconsistencies found in this pass

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

## 5. Open cross-check items (pending a second independent reviewer)

The areas in section 1 were reviewed by a single pass. A second independent reviewer
should attempt to break, not confirm, each conclusion: in particular (a) whether the
Cayley step is Riemannian or only embedded-Euclidean descent, (b) whether any bf16
scale defeats the W2 zero-preserving unpack, (c) whether any map/unmap sequence aliases
two live BAR1 mappings, and (d) whether the GaLore V transport bias from uint8 round-up
storage is bounded. Prompts for this second pass are maintained outside the repository.
