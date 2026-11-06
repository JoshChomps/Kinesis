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
name, the threshold set, and the skip fraction in doc 11 are stale. Recommend updating
doc 11 section 0.3 to the current name and the 5-of-16 / about 69% figure.

## 5. Open cross-check items (pending a second independent reviewer)

The areas in section 1 were reviewed by a single pass. A second independent reviewer
should attempt to break, not confirm, each conclusion: in particular (a) whether the
Cayley step is Riemannian or only embedded-Euclidean descent, (b) whether any bf16
scale defeats the W2 zero-preserving unpack, (c) whether any map/unmap sequence aliases
two live BAR1 mappings, and (d) whether the GaLore V transport bias from uint8 round-up
storage is bounded. Prompts for this second pass are maintained outside the repository.
