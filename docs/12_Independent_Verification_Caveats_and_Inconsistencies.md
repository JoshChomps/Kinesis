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
| SPSC ring memory ordering (O10) | `hetero_runtime/chorus_amx_moe.c` | Per-slot sequence handshake: producer release-stores `gpu_ready_seq`, consumer acquire-loads it then release-stores `cpu_done_seq`; slot reuse waits `cpu_done_seq >= seq - num_slots` | Ordering correct; two latent defects on untested paths (see 4.3, 4.4) |
| LServe sparse + INT4 Hadamard attention (F15) | `structural_filters/lserve_sparse_attn.py` | H_16 cancels in `q_rot k_rot^T = q k^T`; V output un-rotated by `H^T`; per-dim `max(q*k_min, q*k_max)` is the exact box-corner upper bound (admissible page prune); salience = out-window minus in-window | Correct; verified blind 0.70 vs profiled 0.993 (see observation 3.8) |
| SafeTensors serializer + round-trip (F2) | `kinesis_unified/kinesis_bridge.py` | `[8-byte LE u64 len][space-padded JSON header][contiguous data]`, offsets from data start, 8-byte header alignment | Spec-compliant; independently bit-exact across 5 dtypes + empty (see observation 3.9) |
| C ABI ctypes marshalling (F2/F21) | `kinesis_unified/kinesis_c_abi.cpp` + `kinesis_bridge.py` | every entry point is `try/catch(...)`-guarded (no exception crosses `extern "C"`); correct `argtypes`/`restype`; `.contiguous()` inputs; buffers kept alive across the synchronous call | Correct and verified end-to-end; implicit unbounded size contract (see 4.5) |
| cuBLAS device-ID shim logic (F12/O2/F6) | `compiler_unlock/libcublas_unlock_shim.c` | resolver self-hook returns the hook pointer (self-assignment guarded against recursion); caller gating spoofs only when a blas/dnn DSO is on the call chain; ioctl requires both NR 0x2A and TYPE 'F' | Logic correct and mock-verified: blas caller sees workstation ID, PyTorch/NCCL see true ID and SM count (see observation 3.10) |
| SMSP roofline model (Pillar 1) | `sim_harness/smsp_roofline_sim.py` | `I_warp = W_M W_N/(W_M+W_N)`, peak `= sm_count * (2048/tau) * 4 * clock`, DRAM ridge `= peak/BW` in units of batch M | Arithmetic self-consistent; it is a model, not a measurement (see observation 3.11) |
| Memory primitives (F20/F21 hardening) | `kinesis_unified/memory_primitives.hpp` | arena bump math uses `capacity - offset` form (no overflow); slab `deallocate` rejects double-free via `is_free_` bitmap; `total_elements`/`from_shape_row_major` saturate; Shape/Strides/TensorView are exactly 56/56/128 bytes | Correct; no defects |
| Hardware discovery + swizzle (L5/§2.2) | `kinesis_unified/hardware_and_sandbox.hpp` | CPUID gated by XGETBV OS-save bits (YMM/ZMM/AMX); leaf-4 cache size `ways*partitions*line*sets`; BLIS KC/MC/NC from live cache; `offset ^ ((offset>>3)&0x70)` yields `chunk ^ (r&7)` | Correct; one trivial imprecision (observation 3.12) |
| 4x8 GotoBLAS FP32 GEMM + W1/W2 packers | `kinesis_unified/sub2bit_gemm_and_smart_cache.hpp` | standard BLIS 5-loop (jc/pc/ic/jr/ir) with 4-accumulator micro-kernel, FP32 accumulation, F21-safe `storeu`; branchless AVX2 sign unpack; zero-group scale 0 (F5/F14); W2 threshold conditional-mean ternary scale; RNE `fp32_to_bf16_bits` with NaN preservation | Correct; no defects |
| SmartTieredCache error budget (O8/O9/O11) | `kinesis_unified/sub2bit_gemm_and_smart_cache.hpp` | exact by default (O8); O9 worst-case bound `bound[g] = max_col|scale| * max_m row-L1`, greedy skip while `sum bound <= eps` guarantees per-output abs error `<= eps`; non-finite rows never skipped; promotion counted once per GEMM, saturating (O11); LFU+LRU eviction only when strictly hotter | Correct and rigorous; no defects (confirms observation 3.6) |
| Host CTA shim PTX primitives (F17/F18/F19) | `fused_w1_hadacore/w1a16_host_cta_shim.hpp` | `lop3_b32` fast-paths 0xAE/0xEA and the generic bit-LUT, all agreeing; W1/W2 unpack use 64.0=0x4280, -129.0=0xC301, H_2=0x3F35 matching the kernel; `__shfl_xor_sync`/`__ballot_sync` assert on intra-warp divergence | Correct; one theoretical double-rounding note (observation 3.13) |
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
`query_deterministic_caches` classifies an L2 or L3 entry when `cache_type == 2 || cache_type == 3`,
where type 2 is an instruction cache and type 3 is unified. On real x86 parts L2 and L3 are reported
as unified (type 3), so this never mis-sizes in practice (the F3 bench reads L1D=48KB, L2=1280KB,
L3=24576KB correctly on the dev host). But an instruction-only L2 would be accepted as the data-path
L2 budget. Harmless today; strictly the L2/L3 branches should accept only type 3 (unified) or type 1
(data), as the L1D branch already does. No functional impact observed.

### 3.13 Host bf16x2 FMA rounds twice; exact in practice for these operands
`host_ptx_fma_rn_bf16x2` converts each bf16 lane to f32, calls `fmaf`, then rounds the f32 result
back to bf16. That is two roundings (into f32, then into bf16), whereas real `fma.rn.bf16x2` rounds
once directly to bf16. Double rounding can differ from single rounding in adversarial cases. It does
not here: both operands are bf16 (8-bit significand), so each product is exactly representable in
f32 (24-bit) and the `fmaf` step introduces no rounding, leaving only the final f32-to-bf16 rounding.
So the host shim matches hardware `fma.rn.bf16x2` for every value the unpack path produces. Worth
knowing if the shim is ever reused for general bf16x2 FMAs with wider-significand intermediates.

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
`allocate_pinned_1gb_ring` packs the slot count into `alloc_via_mmap` bits [31:16]
(`(num_slots & 0xFFFF) << 16`), and `free_pinned_ring` recovers it with `>> 16`. The alloc
guard rejects only `num_slots > SIZE_MAX / sizeof(slot)`, not `num_slots > 0xFFFF`. A caller
passing more than 65535 slots allocates the full buffer but stores a truncated count, so the
free path computes a smaller byte size and unmaps or unlocks the wrong length (partial unmap
or leak). The intended 1 GB regime holds roughly 7700 slots, well under the limit, so this is
latent, but the allocator should reject `num_slots > 0xFFFF` explicitly or widen the tag.

### 4.5 Latent defect: C ABI size contract is implicit and the copies are unbounded
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

## 5. Open cross-check items (pending a second independent reviewer)

The areas in section 1 were reviewed by a single pass. A second independent reviewer
should attempt to break, not confirm, each conclusion: in particular (a) whether the
Cayley step is Riemannian or only embedded-Euclidean descent, (b) whether any bf16
scale defeats the W2 zero-preserving unpack, (c) whether any map/unmap sequence aliases
two live BAR1 mappings, (d) whether the GaLore V transport bias from uint8 round-up
storage is bounded, and (e) whether the SPSC ring has any reordering the acquire/release
pairs miss under a weak memory model (the current pass reasoned on x86 TSO plus the explicit
C11 orderings). The second reviewer should read both the code and this document and confirm
or dispute each finding here, including the two latent defects in section 4. Prompts for this
second pass are maintained outside the repository.
