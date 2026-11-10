// File: kinesis_unified/test_kinesis_unified.cpp
// Standalone C++20 verification & benchmark suite for the unified Kinesis + Sub-2-Bit Smart Cache engine.
// Build: g++ -O3 -std=c++20 -mavx2 -mfma -o test_kinesis_unified test_kinesis_unified.cpp

#include "memory_primitives.hpp"
#include "hardware_and_sandbox.hpp"
#include "sub2bit_gemm_and_smart_cache.hpp"

#include <algorithm>
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
    smart_cache.execute_smart_w1_gemm(A_view, w1_panels, C_w1_exact, /*force_exact=*/true);

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
    smart_cache.execute_smart_w2_gemm(A_view, w2_panels, C_w2_exact, /*force_exact=*/true);
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
    tiered_cache.execute_smart_w1_gemm(A_view, w1_panels, C_w1_exact, /*force_exact=*/false);
    // Touch panel 5 repeatedly so it out-heats resident panel 0 and triggers LFU Tier-1 VRAM eviction
    tiered_cache.record_panel_access_and_promote(5);
    tiered_cache.record_panel_access_and_promote(5);
    // Pass 2: Promoted panels now hit directly in Tier-1 Single-GPU VRAM cache
    tiered_cache.execute_smart_w1_gemm(A_view, w1_panels, C_w1_exact, /*force_exact=*/false);

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
    std::printf("    Modelled weight-traffic ratio (compression x 1/(1 - skipped), NOT measured) = %.2fx (vs FP32) / %.2fx (vs FP16) -> PASS\n",
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
        zero_cache.execute_smart_w1_gemm(Az, wz, Cz, /*force_exact=*/true);
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

    // 8. Sieve policies: exact by default, guaranteed error budget, and a negative control for the
    //    legacy relative-energy heuristic (docs/11 O8, O9)
    {
        const size_t Ms = 8, Ks = 1024, Ns = 24; // 8 K-groups, 3 panels
        uint32_t lcg = 12345U;
        auto rnd = [&lcg]() {
            lcg = lcg * 1664525U + 1013904223U;
            return static_cast<float>((lcg >> 8) & 0xFFFFU) / 32768.0f - 1.0f; // [-1, 1)
        };
        std::vector<float> w_s(Ks * Ns), a_s(Ms * Ks);
        for (auto& v : w_s) v = rnd();
        for (size_t m = 0; m < Ms; ++m) {
            for (size_t k = 0; k < Ks; ++k) {
                const size_t g = k / kGroupSizeG128;
                // Groups 0..3 are quiet for every row; group 1 has ONE loud row (row 5): the case where the
                // summed-over-rows heuristic drops a group that matters to a single row.
                float amp = (g < 4) ? 0.0005f : 1.0f;
                if (g == 1 && m == 5) amp = 0.5f;
                a_s[m * Ks + k] = amp * rnd();
            }
        }
        TensorView<const float> Ws(w_s.data(), Shape{static_cast<int64_t>(Ks), static_cast<int64_t>(Ns)});
        TensorView<const float> As(a_s.data(), Shape{static_cast<int64_t>(Ms), static_cast<int64_t>(Ks)});
        const W1GotoBlasPanelSet w1s = pack_w1_gotoblas_panels(Ws);
        const W2GotoBlasPanelSet w2s = pack_w2_gotoblas_panels(Ws);

        std::vector<float> c_ref(Ms * Ns), c_def(Ms * Ns), c_bud(Ms * Ns), c_tau(Ms * Ns);
        auto view = [&](std::vector<float>& v) {
            return TensorView<float>(v.data(), Shape{static_cast<int64_t>(Ms), static_cast<int64_t>(Ns)});
        };
        auto max_abs_diff = [&](const std::vector<float>& x, const std::vector<float>& y) {
            float d = 0.0f;
            for (size_t i = 0; i < x.size(); ++i) d = std::max(d, std::fabs(x[i] - y[i]));
            return d;
        };
        float c_max = 0.0f;

        // (a) Default-constructed cache is exact and bit-identical to an explicit exact call
        SmartTieredCache c_exact(w1s.num_panels, 1);
        UHA_VERIFY(c_exact.is_exact());
        c_exact.execute_smart_w1_gemm(As, w1s, view(c_ref), /*force_exact=*/true);
        SmartTieredCache c_default(w1s.num_panels, 1);
        c_default.execute_smart_w1_gemm(As, w1s, view(c_def));
        UHA_VERIFY(c_def == c_ref);
        UHA_VERIFY(c_default.telemetry().tiles_pruned_by_energy_sieve == 0);
        for (float v : c_ref) c_max = std::max(c_max, std::fabs(v));

        // (b) Error budget: every output element stays within eps (plus float reassociation slack)
        const float eps = 0.05f;
        const float slack = 1e-5f * c_max;
        size_t budget_skips_w1 = 0, budget_skips_w2 = 0;
        float err_w1 = 0.0f, err_w2 = 0.0f;
        {
            SmartTieredCache c_b(w1s.num_panels, 1);
            c_b.set_error_budget(eps);
            UHA_VERIFY(!c_b.is_exact());
            c_b.execute_smart_w1_gemm(As, w1s, view(c_bud));
            budget_skips_w1 = c_b.telemetry().tiles_pruned_by_energy_sieve;
            err_w1 = max_abs_diff(c_bud, c_ref);
            UHA_VERIFY(err_w1 <= eps + slack);
        }
        {
            std::vector<float> c_ref2(Ms * Ns), c_bud2(Ms * Ns);
            SmartTieredCache c_e(w2s.num_panels, 1), c_b(w2s.num_panels, 1);
            c_e.execute_smart_w2_gemm(As, w2s, view(c_ref2));
            c_b.set_error_budget(eps);
            c_b.execute_smart_w2_gemm(As, w2s, view(c_bud2));
            budget_skips_w2 = c_b.telemetry().tiles_pruned_by_energy_sieve;
            err_w2 = max_abs_diff(c_bud2, c_ref2);
            UHA_VERIFY(err_w2 <= eps + slack);
        }
        UHA_VERIFY(budget_skips_w1 > 0 && budget_skips_w2 > 0); // the budget actually removes work

        // (c) Negative control: the legacy summed-energy heuristic drops group 1 and breaks the same budget
        SmartTieredCache c_t(w1s.num_panels, 1, /*tau=*/0.85f);
        c_t.execute_smart_w1_gemm(As, w1s, view(c_tau));
        const float err_tau = max_abs_diff(c_tau, c_ref);
        UHA_VERIFY(err_tau > eps);

        // (d) Shape mismatch is rejected instead of reading out of bounds
        bool threw = false;
        try {
            std::vector<float> bad(Ms * (Ns - 8));
            TensorView<float> v_bad(bad.data(), Shape{static_cast<int64_t>(Ms), static_cast<int64_t>(Ns - 8)});
            SmartTieredCache c_x(w1s.num_panels, 1);
            c_x.execute_smart_w1_gemm(As, w1s, v_bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        UHA_VERIFY(threw);

        std::printf("[8] Sieve policies (exact default, guaranteed error budget, legacy-heuristic control):\n");
        std::printf("    default == exact (bitwise), 0 tiles skipped | budget eps = %.3f: W1 skipped %zu tiles, max err %.3e; W2 skipped %zu, max err %.3e\n",
                    eps, budget_skips_w1, err_w1, budget_skips_w2, err_w2);
        std::printf("    legacy tau=0.85 heuristic on the same data: max err %.3e (> eps, loud single row dropped) | shape check -> PASS\n",
                    err_tau);
    }

    // 9. Regressions for docs/12 4.12 (non-finite activations) and 4.14 (W2 convention)
    {
        const W1TmacLutPanelSet lut = pack_w1_tmac_lut_panels(w1_panels);
        const size_t Mn = 3;
        std::vector<float> a_nf(Mn * K);
        for (size_t i = 0; i < Mn * K; ++i) a_nf[i] = A_buf[i];
        for (size_t k = 0; k < 128; ++k) a_nf[0 * K + k] = 0.0f;   // row 0: NaN in an otherwise all-zero group
        a_nf[0 * K + 5] = NAN;
        a_nf[1 * K + 7] = NAN;                                     // row 1: NaN in a live group
        a_nf[2 * K + 9] = INFINITY;                                // row 2: +Inf in a live group
        TensorView<const float> A_nf(a_nf.data(), Shape{static_cast<int64_t>(Mn), static_cast<int64_t>(K)});
        std::vector<float> c_t(Mn * N), c_s(Mn * N), c_x(Mn * N);
        TensorView<float> Ct(c_t.data(), Shape{static_cast<int64_t>(Mn), static_cast<int64_t>(N)});
        TensorView<float> Cs(c_s.data(), Shape{static_cast<int64_t>(Mn), static_cast<int64_t>(N)});
        TensorView<float> Cx(c_x.data(), Shape{static_cast<int64_t>(Mn), static_cast<int64_t>(N)});
        execute_tmac_lut_w1_gemm(A_nf, lut, Ct, 0.0f, false);
        execute_tmac_lut_w1_gemm(A_nf, lut, Cs, 0.0f, true);
        SmartTieredCache exact_cache(w1_panels.num_panels, 0, 0.0f, 1);
        exact_cache.execute_smart_w1_gemm(A_nf, w1_panels, Cx, /*force_exact=*/true);
        size_t class_mismatch = 0, nonfinite_out = 0;
        auto cls = [](float v) { return std::isnan(v) ? 2 : (std::isinf(v) ? (v > 0 ? 3 : 4) : 0); };
        for (size_t i = 0; i < Mn * N; ++i) {
            class_mismatch += (cls(c_t[i]) != cls(c_x[i])) + (cls(c_s[i]) != cls(c_x[i]));
            nonfinite_out += !std::isfinite(c_t[i]);
        }
        UHA_VERIFY(class_mismatch == 0);         // NaN/+Inf/-Inf/finite pattern identical to the exact FP32 path
        UHA_VERIFY(nonfinite_out == Mn * N);     // every output of every poisoned row is non-finite

        // One Inf must not make the sieve skip the row's finite groups (mean energy used to become +Inf)
        std::vector<float> a_inf(1 * K, 1.0f), c_inf(N, 0.0f);
        a_inf[3] = INFINITY;
        TensorView<const float> A_inf(a_inf.data(), Shape{1, static_cast<int64_t>(K)});
        TensorView<float> C_inf(c_inf.data(), Shape{1, static_cast<int64_t>(N)});
        const TmacRunStats st_inf = execute_tmac_lut_w1_gemm(A_inf, lut, C_inf, 0.85f, false);
        UHA_VERIFY(st_inf.groups_skipped == 0);

        // W2: independent BitNet b1.58 absmean reference (gamma = bf16(max(mean|w|, 1e-6)), round-half-even, clip)
        const W2GotoBlasPanelSet w2p = pack_w2_gotoblas_panels(B_view);
        size_t trit_diff = 0, scale_diff = 0;
        for (size_t n = 0; n < N; ++n) {
            const size_t p = n / kNr8, c = n % kNr8;
            for (size_t g = 0; g < w2p.groups_per_col; ++g) {
                const size_t k0 = g * kGroupSizeG128, k1 = std::min(K, k0 + kGroupSizeG128);
                double asum = 0.0;
                for (size_t k = k0; k < k1; ++k) asum += std::fabs(B_view(static_cast<int64_t>(k), static_cast<int64_t>(n)));
                // Independent FP64 -> BF16 rounding: pick the nearer of the two BF16 neighbours (even on a tie).
                const double mean = std::max(asum / double(k1 - k0), 1e-6);
                const float f_near = static_cast<float>(mean);
                uint16_t lo = static_cast<uint16_t>([&] { uint32_t u; std::memcpy(&u, &f_near, 4); return u >> 16; }());
                if (static_cast<double>(bf16_bits_to_fp32(lo)) > mean) --lo;  // positive values: bits are monotone
                const uint16_t hi = static_cast<uint16_t>(lo + 1);
                const double d_lo = mean - bf16_bits_to_fp32(lo), d_hi = bf16_bits_to_fp32(hi) - mean;
                const float gamma = bf16_bits_to_fp32((d_lo < d_hi || (d_lo == d_hi && (lo & 1u) == 0)) ? lo : hi);
                const float got = bf16_bits_to_fp32(w2p.packed_scales_bf16[(p * w2p.groups_per_col + g) * kNr8 + c]);
                scale_diff += (got != gamma);
                for (size_t k = k0; k < k1; ++k) {
                    const float r = std::nearbyint(B_view(static_cast<int64_t>(k), static_cast<int64_t>(n)) / got);
                    const int want = static_cast<int>(std::clamp(r, -1.0f, 1.0f));
                    const int code = static_cast<int>((w2p.packed_trits[p * K + k] >> (2 * c)) & 3u) - 1;
                    trit_diff += (want != code);
                }
            }
        }
        // Both sides now use an FP64 mean rounded once to BF16 (docs/12 4.19), so scales must match exactly.
        UHA_VERIFY(trit_diff == 0 && scale_diff == 0);
        std::printf("[9] Non-finite activations & W2 absmean convention (docs/12 4.12, 4.14):\n");
        std::printf("    T-MAC (AVX2 + scalar) NaN/Inf output pattern vs exact FP32 path: %zu mismatches | Inf row sieve skips = %zu\n",
                    class_mismatch, st_inf.groups_skipped);
        std::printf("    W2 vs independent BitNet b1.58 absmean reference: trit mismatches = %zu, scale ulp diffs = %zu/%zu -> PASS\n",
                    trit_diff, scale_diff, N * w2p.groups_per_col);
    }

    return 0;
}
