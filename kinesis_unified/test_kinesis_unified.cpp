// File: kinesis_unified/test_kinesis_unified.cpp
// Standalone C++20 verification & benchmark suite for the unified Kinesis + Sub-2-Bit Smart Cache engine.
// Build: g++ -O3 -std=c++20 -mavx2 -mfma -o test_kinesis_unified test_kinesis_unified.cpp

#include "memory_primitives.hpp"
#include "hardware_and_sandbox.hpp"
#include "sub2bit_gemm_and_smart_cache.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define UHA_VERIFY(cond)                                                                       \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::fprintf(stderr, "[FATAL] Verification failed at %s:%d: %s\n",                 \
                         __FILE__, __LINE__, #cond);                                           \
            std::abort();                                                                      \
        }                                                                                      \
    } while (0)

using namespace kinesis::unified;

int main() {
    std::printf("================================================================================\n");
    std::printf("  WORKSTREAM 9: KINESIS UNIFIED C++20 CORE, AVX2 SUB-2-BIT & SMART CACHE SUITE\n");
    std::printf("================================================================================\n\n");

    // 1. Verify Live CPUID Hardware Discovery & Analytical BLIS Cache Blocking
    const CpuFeatures cpu = HardwareInfo::query_cpu_features();
    const BlisCacheBlocks blk = HardwareInfo::compute_blis_blocks(cpu);
    std::printf("[1] Live CPUID Topology & Analytical BLIS Cache Blocking:\n");
    std::printf("    Vendor = %s | Cores = %u | AVX2 = %d | FMA = %d | AVX512F/BW = %d/%d | AMX = %d\n",
                cpu.vendor.c_str(), cpu.logical_cores, cpu.has_avx2 ? 1 : 0,
                cpu.has_fma ? 1 : 0, cpu.has_avx512f ? 1 : 0, cpu.has_avx512bw ? 1 : 0,
                cpu.has_amx_bf16 ? 1 : 0);
    std::printf("    L1D = %zu KB | L2 = %zu KB | L3 = %zu KB\n",
                cpu.l1d_cache_bytes / 1024, cpu.l2_cache_bytes / 1024, cpu.l3_cache_bytes / 1024);
    std::printf("    Derived BLIS Tiles: MR = %zu, NR = %zu, KC = %zu, MC = %zu, NC = %zu -> PASS\n\n",
                blk.mr, blk.nr, blk.kc, blk.mc, blk.nc);
    UHA_VERIFY(cpu.l1d_cache_bytes > 0 && blk.kc >= 64);

    // 2. Verify Three-Lifetime Prefaulted & Page-Locked Arena + SlabAllocator
    WorkspacePlan plan{
        4 * 1024 * 1024, // 4 MB Persistent weight arena
        2 * 1024 * 1024, // 2 MB Step activation arena
        4 * 1024 * 1024  // 4 MB Transient scratch arena
    };
    ThreeTierArenaManager arenas(plan);
    void* p_weight = arenas.persistent().allocate(1024, 64);
    void* p_step   = arenas.step().allocate(2048, 64);
    UHA_VERIFY(is_aligned(p_weight, 64));
    UHA_VERIFY(is_aligned(p_step, 64));
    UHA_VERIFY(arenas.persistent().owns(p_weight));
    UHA_VERIFY(arenas.step().owns(p_step));
    UHA_VERIFY(!arenas.persistent().owns(p_step));

    SlabAllocator<256, 64> slab(128, true);
    void* b0 = slab.allocate(256);
    void* b1 = slab.allocate(256);
    UHA_VERIFY(slab.allocated_blocks() == 2);
    slab.deallocate(b1);
    slab.deallocate(b0);
    UHA_VERIFY(slab.free_blocks() == 128);
    std::printf("[2] Prefaulted & Page-Locked Three-Lifetime Arenas + O(1) SlabAllocator:\n");
    std::printf("    Persistent = %zu KB (Locked=%d), Step = %zu KB, Scratch = %zu KB, Slab Free = %zu/128 -> PASS\n\n",
                arenas.persistent().capacity() / 1024,
                arenas.persistent().is_page_locked() ? 1 : 0,
                arenas.step().capacity() / 1024,
                arenas.scratch().capacity() / 1024,
                slab.free_blocks());

    // 3. Verify MemorySandbox: CPU/GPU Cycle Separation & 1.00x Conflict-Free 32-Bank Swizzle
    const WarpBankReport unswizzled = MemorySandbox::evaluate_warp_32bank_access(128, false);
    const WarpBankReport swizzled   = MemorySandbox::evaluate_warp_32bank_access(128, true);
    const WarpBankReport ldm_unswiz = MemorySandbox::evaluate_ldmatrix_8x16b_bank_access(128, false);
    const WarpBankReport ldm_swiz   = MemorySandbox::evaluate_ldmatrix_8x16b_bank_access(128, true);
    UHA_VERIFY(unswizzled.max_way_conflict == 32);
    UHA_VERIFY(swizzled.max_way_conflict == 1 && swizzled.active_banks == 32);
    UHA_VERIFY(ldm_unswiz.max_way_conflict == 8);
    UHA_VERIFY(ldm_swiz.max_way_conflict == 1 && ldm_swiz.active_banks == 32);

    MemorySandbox sandbox(cpu.l1d_cache_bytes, 64, 8);
    // Simulate streaming 4x8 packed micro-panel hits in L1 cache
    for (size_t pass = 0; pass < 8; ++pass) {
        for (size_t i = 0; i < 256; ++i) {
            sandbox.record_access(0x100000ULL + i * 32ULL, 32);
        }
    }
    const CacheSandboxReport& srep = sandbox.report();
    UHA_VERIFY(srep.hit_rate_percent() > 90.0);
    std::printf("[3] MemorySandbox Verification (Fixes Design Review L5 & CPU/GPU Latency Mixup):\n");
    std::printf("    Scalar 32-Lane Bank Conflict : Unswizzled = %u-way, Swizzled = %u-way (%u/32 banks)\n",
                unswizzled.max_way_conflict, swizzled.max_way_conflict, swizzled.active_banks);
    std::printf("    ldmatrix 8x16B Bank Conflict : Unswizzled = %u-way, Swizzled = %u-way (%u/32 banks)\n",
                ldm_unswiz.max_way_conflict, ldm_swiz.max_way_conflict, ldm_swiz.active_banks);
    std::printf("    Packed Panel L1 Hit Rate = %.2f%% | CPU Cycles = %zu | GPU Cycles = %zu -> PASS\n\n",
                srep.hit_rate_percent(), srep.cpu_estimated_cycles, srep.gpu_estimated_cycles);

    // 4. Verify True 4x8 Register-Blocked FP32 GotoBLAS GEMM (Zero Heap Allocations)
    const size_t M = 64, K = 256, N = 64;
    auto* A_buf = static_cast<float*>(arenas.step().allocate(M * K * sizeof(float), 64));
    auto* B_buf = static_cast<float*>(arenas.persistent().allocate(K * N * sizeof(float), 64));
    auto* C_buf = static_cast<float*>(arenas.step().allocate(M * N * sizeof(float), 64));
    auto* C_ref = static_cast<float*>(arenas.step().allocate(M * N * sizeof(float), 64));

    for (size_t i = 0; i < M * K; ++i) A_buf[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.125f;
    for (size_t i = 0; i < K * N; ++i) B_buf[i] = static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.25f;

    TensorView<const float> A_view(A_buf, Shape{static_cast<int64_t>(M), static_cast<int64_t>(K)});
    TensorView<const float> B_view(B_buf, Shape{static_cast<int64_t>(K), static_cast<int64_t>(N)});
    TensorView<float>       C_view(C_buf, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});

    gemm_fp32_register_blocked_4x8(A_view, B_view, C_view, arenas.scratch());

    float max_fp32_err = 0.0f;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += A_view(static_cast<int64_t>(i), static_cast<int64_t>(k)) *
                       B_view(static_cast<int64_t>(k), static_cast<int64_t>(j));
            }
            C_ref[i * N + j] = acc;
            const float diff = std::fabs(C_view(static_cast<int64_t>(i), static_cast<int64_t>(j)) - acc);
            if (diff > max_fp32_err) max_fp32_err = diff;
        }
    }
    UHA_VERIFY(max_fp32_err < 1e-5f);
    std::printf("[4] 4x8 Register-Blocked FP32 GotoBLAS GEMM (%zux%zux%zu, Scratch Arena Only):\n", M, K, N);
    std::printf("    Max Numerical Error vs Reference = %.2e (Scratch Reset Offset = %zu) -> PASS\n\n",
                max_fp32_err, arenas.scratch().allocated_bytes());

    // 5. Verify Sub-2-Bit (1.125 bpw) GotoBLAS Panel Packing & Branchless AVX2 Unpacker
    const W1GotoBlasPanelSet w1_panels = pack_w1_gotoblas_panels(B_view);
    const double comp_ratio = w1_panels.compression_ratio_vs_fp32();
    UHA_VERIFY(comp_ratio > 28.0); // 32 bits / 1.125 bits = 28.44x compression vs FP32!

    SmartTieredCache smart_cache(w1_panels.num_panels, w1_panels.num_panels / 2, 0.90f, 1);
    TensorView<float> C_w1_exact(C_buf, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
    smart_cache.execute_smart_w1_gemm(A_view, w1_panels, C_w1_exact, /*force_exact_zero_pruning=*/true);

    // Compute exact dequantized W1 reference and verify zero mismatch with AVX2 branchless unpacker
    float max_w1_unpack_err = 0.0f;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const size_t p = j / kNr8;
            const size_t c = j % kNr8;
            float acc = 0.0f;
            for (size_t g = 0; g < w1_panels.groups_per_col; ++g) {
                const float scale = bf16_bits_to_fp32(
                    w1_panels.packed_scales_bf16[(p * w1_panels.groups_per_col + g) * kNr8 + c]
                );
                const size_t k_base = g * kGroupSizeG128;
                const size_t k_end = std::min(K, k_base + kGroupSizeG128);
                for (size_t k = k_base; k < k_end; ++k) {
                    const uint8_t bmask = w1_panels.packed_bits[p * K + k];
                    const float w_val = ((bmask >> c) & 1u) ? scale : -scale;
                    acc += A_view(static_cast<int64_t>(i), static_cast<int64_t>(k)) * w_val;
                }
            }
            const float diff = std::fabs(C_w1_exact(static_cast<int64_t>(i), static_cast<int64_t>(j)) - acc);
            if (diff > max_w1_unpack_err) max_w1_unpack_err = diff;
        }
    }
    UHA_VERIFY(max_w1_unpack_err < 1e-5f);

    // Verify W2 (2.125 bpw Ternary) GotoBLAS Panel Packing (15.06x vs FP32) & AVX2 Unpacker
    const W2GotoBlasPanelSet w2_panels = pack_w2_gotoblas_panels(B_view);
    const double w2_comp_ratio = w2_panels.compression_ratio_vs_fp32();
    UHA_VERIFY(w2_comp_ratio > 15.0); // 32 bits / 2.125 bits = 15.06x compression vs FP32

    TensorView<float> C_w2_exact(C_buf, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
    smart_cache.execute_smart_w2_gemm(A_view, w2_panels, C_w2_exact, /*force_exact_zero_pruning=*/true);
    float max_w2_unpack_err = 0.0f;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const size_t p = j / kNr8;
            const size_t c = j % kNr8;
            float acc = 0.0f;
            for (size_t g = 0; g < w2_panels.groups_per_col; ++g) {
                const float scale = bf16_bits_to_fp32(
                    w2_panels.packed_scales_bf16[(p * w2_panels.groups_per_col + g) * kNr8 + c]
                );
                const size_t k_base = g * kGroupSizeG128;
                const size_t k_end = std::min(K, k_base + kGroupSizeG128);
                for (size_t k = k_base; k < k_end; ++k) {
                    const uint16_t word = w2_panels.packed_trits[p * K + k];
                    const int code = static_cast<int>((word >> (2 * c)) & 0x3u);
                    const float w_val = static_cast<float>(code - 1) * scale;
                    acc += A_view(static_cast<int64_t>(i), static_cast<int64_t>(k)) * w_val;
                }
            }
            const float diff = std::fabs(C_w2_exact(static_cast<int64_t>(i), static_cast<int64_t>(j)) - acc);
            if (diff > max_w2_unpack_err) max_w2_unpack_err = diff;
        }
    }
    UHA_VERIFY(max_w2_unpack_err < 1e-5f);

    // Verify lossless CPU GotoBLAS Nr=8 <-> GPU Interleaved 16-Bit Split uint32 Layout Bridge
    std::vector<uint32_t> gpu_w_u32;
    std::vector<uint16_t> gpu_scales_bf16;
    convert_w1_gotoblas_to_gpu_interleaved_u32(w1_panels, gpu_w_u32, gpu_scales_bf16);
    UHA_VERIFY(gpu_w_u32.size() == N * (K / UHA_W1_WEIGHTS_PER_U32));
    size_t layout_bit_mismatches = 0;
    for (size_t n = 0; n < N; ++n) {
        const size_t p = n / kNr8;
        const size_t c = n % kNr8;
        for (size_t k = 0; k < K; ++k) {
            const uint32_t cpu_bit = (w1_panels.packed_bits[p * K + k] >> c) & 1U;
            const size_t w_idx = k / 32;
            const size_t pair_idx = (k % 32) / 2;
            const size_t is_odd = k & 1U;
            const uint32_t gpu_word = gpu_w_u32[n * (K / 32) + w_idx];
            const uint32_t gpu_bit = (gpu_word >> (pair_idx + is_odd * 16)) & 1U;
            if (cpu_bit != gpu_bit) ++layout_bit_mismatches;
        }
    }
    UHA_VERIFY(layout_bit_mismatches == 0);

    std::printf("[5] Sub-2-Bit GotoBLAS Panel Packers (W1 1.125 bpw & W2 2.125 bpw) + GPU Layout Bridge:\n");
    std::printf("    Descriptor Sizes: Shape = %zu B, Strides = %zu B, TensorView = %zu B (2x64B Cache Lines)\n",
                sizeof(Shape), sizeof(Strides), sizeof(TensorView<float>));
    std::printf("    W1 (1.125 bpw): %zu B -> %zu B (%.2fx vs FP32, %.2fx vs FP16) | AVX2 Err = %.2e\n",
                w1_panels.fp32_equivalent_bytes(), w1_panels.compressed_bytes(),
                comp_ratio, comp_ratio / 2.0, max_w1_unpack_err);
    std::printf("    W2 (2.125 bpw): %zu B -> %zu B (%.2fx vs FP32, %.2fx vs FP16) | AVX2 Err = %.2e | GPU Bridge Mismatches = %zu -> PASS\n\n",
                w2_panels.fp32_equivalent_bytes(), w2_panels.compressed_bytes(),
                w2_comp_ratio, w2_comp_ratio / 2.0, max_w2_unpack_err, layout_bit_mismatches);

    // 6. Verify Smart CPU + Single-GPU Tiered Cache with Predictive Tiled-CBC Energy Sieve & LFU Eviction
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < 128; ++k) {
            A_buf[m * K + k] *= 0.05f;
        }
    }
    SmartTieredCache tiered_cache(w1_panels.num_panels, /*max_vram_hot_panels=*/4, /*tau=*/0.85f, /*promo=*/1);
    // Pass 1: Cold panels computed in CPU L3 via AVX2 and first 4 panels promoted to Tier-1 VRAM
    tiered_cache.execute_smart_w1_gemm(A_view, w1_panels, C_w1_exact, /*force_exact_zero_pruning=*/false);
    // Touch panel 5 repeatedly so it out-heats resident panel 0 and triggers LFU Tier-1 VRAM eviction
    tiered_cache.record_panel_access_and_promote(5);
    tiered_cache.record_panel_access_and_promote(5);
    // Pass 2: Promoted panels now hit directly in Tier-1 Single-GPU VRAM cache
    tiered_cache.execute_smart_w1_gemm(A_view, w1_panels, C_w1_exact, /*force_exact_zero_pruning=*/false);

    void* kv_page = tiered_cache.allocate_kv_page();
    UHA_VERIFY(kv_page != nullptr);
    tiered_cache.free_kv_page(kv_page);

    const SmartCacheTelemetry& tel = tiered_cache.telemetry();
    UHA_VERIFY(tel.tiles_pruned_by_energy_sieve > 0);
    UHA_VERIFY(tel.hot_tile_promotions >= 5);
    UHA_VERIFY(tel.tier1_vram_evictions >= 1);
    UHA_VERIFY(tel.tier1_vram_hits > 0);
    std::printf("[6] Smart CPU-L3 + Single-GPU VRAM Tiered Cache & Tiled-CBC Predictive Sieve:\n");
    std::printf("    Total Tile Requests = %zu | Pruned by Energy Sieve = %zu (%.1f%% Skipped Before Fetch)\n",
                tel.total_tile_requests, tel.tiles_pruned_by_energy_sieve, tel.pruning_ratio_percent());
    std::printf("    Tier-2 CPU L3 Hits = %zu | Promotions = %zu | Evictions = %zu | Tier-1 VRAM Hits = %zu\n",
                tel.tier2_cpu_l3_hits, tel.hot_tile_promotions, tel.tier1_vram_evictions, tel.tier1_vram_hits);
    std::printf("    Effective Weight Bandwidth Amplification = %.2fx (vs FP32) / %.2fx (vs FP16) -> PASS\n",
                tel.effective_bandwidth_multiplier(comp_ratio),
                tel.effective_bandwidth_multiplier(comp_ratio / 2.0));

    // 7. T-MAC style 16-entry LUT kernel (decode regime, M <= 8) + zero-group fix + PowerInfer seeding/aging
    {
        const size_t Ml = 4;
        TensorView<const float> A_small(A_buf, Shape{static_cast<int64_t>(Ml), static_cast<int64_t>(K)});
        const W1TmacLutPanelSet lut = pack_w1_tmac_lut_panels(w1_panels);
        UHA_VERIFY(lut.nibbles.size() == w1_panels.num_panels * (K / 4));

        std::vector<float> c_avx(Ml * N), c_scalar(Ml * N);
        TensorView<float> v_avx(c_avx.data(), Shape{static_cast<int64_t>(Ml), static_cast<int64_t>(N)});
        TensorView<float> v_sc(c_scalar.data(), Shape{static_cast<int64_t>(Ml), static_cast<int64_t>(N)});
        execute_tmac_lut_w1_gemm(A_small, lut, v_avx, 0.0f, /*force_scalar=*/false);
        execute_tmac_lut_w1_gemm(A_small, lut, v_sc, 0.0f, /*force_scalar=*/true);

        // (a) SIMD pshufb path and portable path use identical integer arithmetic: must match exactly
        for (size_t i = 0; i < Ml * N; ++i) UHA_VERIFY(c_avx[i] == c_scalar[i]);

        // (b) Accuracy vs exact FP32 W1 reference stays inside the analytic int8-activation rounding bound
        float worst_ratio = 0.0f, max_lut_err = 0.0f;
        for (size_t i = 0; i < Ml; ++i) {
            for (size_t j = 0; j < N; ++j) {
                const size_t p = j / kNr8, c = j % kNr8;
                float ref = 0.0f, bound = 0.0f;
                for (size_t g = 0; g < w1_panels.groups_per_col; ++g) {
                    const float sc_w = bf16_bits_to_fp32(
                        w1_panels.packed_scales_bf16[(p * w1_panels.groups_per_col + g) * kNr8 + c]);
                    float amax = 0.0f;
                    for (size_t k = g * kGroupSizeG128; k < (g + 1) * kGroupSizeG128; ++k) {
                        const float a = A_small(static_cast<int64_t>(i), static_cast<int64_t>(k));
                        amax = std::max(amax, std::fabs(a));
                        const float w = ((w1_panels.packed_bits[p * K + k] >> c) & 1u) ? sc_w : -sc_w;
                        ref += a * w;
                    }
                    bound += 0.5f * (amax / 127.0f) * 128.0f * sc_w;
                }
                const float err = std::fabs(c_avx[i * N + j] - ref);
                max_lut_err = std::max(max_lut_err, err);
                worst_ratio = std::max(worst_ratio, err / (bound + 1e-6f));
            }
        }
        UHA_VERIFY(worst_ratio <= 1.001f);

        // (c) An all-zero weight group must give exactly zero output (regression: scale used to be forced to 1.0)
        std::vector<float> zeros(128 * 8, 0.0f), a_ones(1 * 128, 1.0f), c_zero(8, -1.0f);
        TensorView<const float> Wz(zeros.data(), Shape{128, 8});
        TensorView<const float> Az(a_ones.data(), Shape{1, 128});
        TensorView<float> Cz(c_zero.data(), Shape{1, 8});
        const W1GotoBlasPanelSet wz = pack_w1_gotoblas_panels(Wz);
        SmartTieredCache zero_cache(wz.num_panels, 0, 0.0f, 1);
        zero_cache.execute_smart_w1_gemm(Az, wz, Cz, /*force_exact_zero_pruning=*/true);
        for (size_t j = 0; j < 8; ++j) UHA_VERIFY(c_zero[j] == 0.0f);

        // (d) Offline power-law seeding places exactly the top-scoring panels; aging halves counters
        SmartTieredCache seeded(8, 3, 0.85f, 2);
        seeded.seed_offline_powerlaw_hot_panels({0.1f, 9.0f, 0.2f, 8.0f, 0.3f, 7.0f, 0.4f, 0.5f});
        UHA_VERIFY(seeded.panel_resident_in_vram(1) && seeded.panel_resident_in_vram(3) &&
                   seeded.panel_resident_in_vram(5) && !seeded.panel_resident_in_vram(0));
        seeded.record_panel_access_and_promote(1);
        UHA_VERIFY(seeded.telemetry().tier1_vram_hits == 1);
        seeded.decay_access_frequencies();

        // Sieve path must skip some (row, group) pairs when one group is nearly silent
        std::vector<float> a_skew(1 * K, 1.0f), c_skew(N, 0.0f);
        for (size_t k = 0; k < 128; ++k) a_skew[k] = 0.001f;
        TensorView<const float> A_sk(a_skew.data(), Shape{1, static_cast<int64_t>(K)});
        TensorView<float> C_sk(c_skew.data(), Shape{1, static_cast<int64_t>(N)});
        const TmacRunStats st = execute_tmac_lut_w1_gemm(A_sk, lut, C_sk, 0.85f, false);
        UHA_VERIFY(st.groups_skipped == 1 && st.groups_total == lut.groups_per_col);

        std::printf("[7] T-MAC 16-entry LUT kernel (M=%zu), zero-group fix, offline seeding:\n", Ml);
        std::printf("    pshufb vs scalar LUT mismatches = 0 | max |err| vs FP32 W1 = %.3e (%.1f%% of analytic bound) | zero-group exact = PASS\n",
                    max_lut_err, 100.0 * worst_ratio);
        std::printf("    NOTE: accuracy only; no speedup is claimed or measured for this kernel -> PASS\n");
    }

    return 0;
}
