/**
 * @file bench_cpu_roofline.cpp
 * @brief Suite F3: Empirical Host CPU GFLOPS, Cache Residency & Bandwidth Benchmark Suite.
 *
 * Measures real wall-clock latency (ms), compute throughput (GFLOPS), physical bytes read bandwidth (GB/s),
 * and FP32-equivalent effective weight bandwidth (GB/s) on the host CPU across:
 *   1. FP32 4x8 Register-Blocked GotoBLAS GEMM (gemm_fp32_register_blocked_4x8)
 *   2. W1 (1.125 bpw) 4x8 AVX2 FMA Unpacker Exact (SmartTieredCache::execute_smart_w1_gemm, tau=0)
 *   3. W1 (1.125 bpw) 4x8 AVX2 FMA + Holder Sieve (SmartTieredCache::execute_smart_w1_gemm, 50% cold groups)
 *   4. W2 (2.125 bpw Ternary) 4x8 AVX2 FMA Unpacker (SmartTieredCache::execute_smart_w2_gemm)
 *   5. W1 T-MAC 256-Bit 32-Lane vpshufb 4-Panel LUT (execute_tmac_lut_w1_gemm)
 *
 * Evaluated across two LLM regimes:
 *   - Regime A (Small-Batch Decode / Speculative Verify): M = 4,  K = 1024, N = 1024
 *   - Regime B (Batched Prefill / QAT Forward):           M = 64, K = 1024, N = 1024
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "hardware_and_sandbox.hpp"
#include "memory_primitives.hpp"
#include "sub2bit_gemm_and_smart_cache.hpp"

using namespace kinesis::unified;

#define UHA_BENCH_VERIFY(cond)                                                            \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

struct BenchResult {
    const char* name;
    size_t weight_bytes;
    double best_ms;
    double gflops;
    double phys_gbps;
    double eff_fp32_gbps;
    double speedup_vs_fp32;
};

template <typename Fn>
static double measure_best_ms(Fn&& fn, int warmup_iters, int timed_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        fn();
    }
    double best_ms = 1e30;
    for (int i = 0; i < timed_iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ms) {
            best_ms = ms;
        }
    }
    return best_ms;
}

static void run_regime_benchmark(
    const char* regime_label,
    size_t M,
    size_t K,
    size_t N,
    int warmup_iters,
    int timed_iters,
    const BlisCacheBlocks& blis
) {
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("  %s (M = %zu, K = %zu, N = %zu | FLOPs = %.3f MFLOPs)\n",
                regime_label, M, K, N, (2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K)) / 1e6);
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("  %-26s | %9s | %8s | %8s | %9s | %10s | %7s\n",
                "Kernel / Precision Path", "W Size KB", "Best ms", "GFLOPS", "Phys GB/s", "EffFP32 GB", "Speedup");
    std::printf("  ---------------------------+-----------+----------+----------+-----------+------------+--------\n");

    const size_t scratch_bytes = std::max<size_t>(
        (blis.mc * blis.kc + blis.kc * ((blis.nc + kNr8 - 1) / kNr8) * kNr8) * sizeof(float) + 65536,
        512 * 1024
    );
    ArenaAllocator arena(
        (M * K * 2 + K * N + M * N * 5) * sizeof(float) + scratch_bytes + 1024 * 1024,
        true,
        MemorySpace::HostPinnedLocked
    );
    ArenaAllocator scratch(scratch_bytes, true, MemorySpace::HostPinnedLocked);

    float* a_ptr = static_cast<float*>(arena.allocate(M * K * sizeof(float)));
    float* a_sparse_ptr = static_cast<float*>(arena.allocate(M * K * sizeof(float)));
    float* b_ptr = static_cast<float*>(arena.allocate(K * N * sizeof(float)));
    float* c_fp32_ptr = static_cast<float*>(arena.allocate(M * N * sizeof(float)));
    float* c_w1_ptr = static_cast<float*>(arena.allocate(M * N * sizeof(float)));
    float* c_w1_sieve_ptr = static_cast<float*>(arena.allocate(M * N * sizeof(float)));
    float* c_w2_ptr = static_cast<float*>(arena.allocate(M * N * sizeof(float)));
    float* c_tmac_ptr = static_cast<float*>(arena.allocate(M * N * sizeof(float)));

    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            const float val = 0.05f * std::sin(static_cast<float>(m * 13 + k * 7));
            a_ptr[m * K + k] = val;
            const size_t g = k / kGroupSizeG128;
            a_sparse_ptr[m * K + k] = (g % 2 == 1) ? (val * 1e-4f) : val;
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < N; ++n) {
            b_ptr[k * N + n] = 0.04f * std::cos(static_cast<float>(k * 11 + n * 5));
        }
    }

    TensorView<const float> A(a_ptr, Shape{static_cast<int64_t>(M), static_cast<int64_t>(K)});
    TensorView<const float> A_sparse(a_sparse_ptr, Shape{static_cast<int64_t>(M), static_cast<int64_t>(K)});
    TensorView<const float> B(b_ptr, Shape{static_cast<int64_t>(K), static_cast<int64_t>(N)});
    TensorView<float> C_fp32(c_fp32_ptr, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
    TensorView<float> C_w1(c_w1_ptr, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
    TensorView<float> C_w1_sieve(c_w1_sieve_ptr, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
    TensorView<float> C_w2(c_w2_ptr, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
    TensorView<float> C_tmac(c_tmac_ptr, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});

    const W1GotoBlasPanelSet w1_panels = pack_w1_gotoblas_panels(B);
    const W2GotoBlasPanelSet w2_panels = pack_w2_gotoblas_panels(B);
    const W1TmacLutPanelSet tmac_panels = pack_w1_tmac_lut_panels(w1_panels);

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double fp32_io_bytes = static_cast<double>((M * K + K * N + M * N) * sizeof(float));
    const double act_out_bytes = static_cast<double>((M * K + M * N) * sizeof(float));

    // 1. FP32 4x8 Register-Blocked GotoBLAS
    const double ms_fp32 = measure_best_ms([&]() {
        gemm_fp32_register_blocked_4x8(A, B, C_fp32, scratch);
    }, warmup_iters, timed_iters);

    // 2. W1 (1.125 bpw) AVX2 FMA Exact
    SmartTieredCache cache_w1_exact(w1_panels.num_panels, w1_panels.num_panels / 2, 0.0f, 2);
    const double ms_w1_exact = measure_best_ms([&]() {
        cache_w1_exact.execute_smart_w1_gemm(A, w1_panels, C_w1, true);
    }, warmup_iters, timed_iters);

    // 3. W1 (1.125 bpw) AVX2 FMA + Guaranteed Holder Sieve (50% cold groups)
    SmartTieredCache cache_w1_sieve(w1_panels.num_panels, w1_panels.num_panels / 2, 0.0f, 2);
    cache_w1_sieve.set_error_budget(0.05f);
    const double ms_w1_sieve = measure_best_ms([&]() {
        cache_w1_sieve.execute_smart_w1_gemm(A_sparse, w1_panels, C_w1_sieve, false);
    }, warmup_iters, timed_iters);

    // 4. W2 (2.125 bpw Ternary) AVX2 FMA Exact
    SmartTieredCache cache_w2_exact(w2_panels.num_panels, w2_panels.num_panels / 2, 0.0f, 2);
    const double ms_w2_exact = measure_best_ms([&]() {
        cache_w2_exact.execute_smart_w2_gemm(A, w2_panels, C_w2, true);
    }, warmup_iters, timed_iters);

    // 5. W1 T-MAC 256-bit 32-lane vpshufb 4-Panel LUT
    const double ms_tmac = measure_best_ms([&]() {
        (void)execute_tmac_lut_w1_gemm(A, tmac_panels, C_tmac, 0.0f, false);
    }, warmup_iters, timed_iters);

    // Verify numerical sanity across all 5 benchmarked outputs
    UHA_BENCH_VERIFY(std::isfinite(C_fp32(0, 0)) && std::isfinite(C_w1(0, 0)));
    UHA_BENCH_VERIFY(std::isfinite(C_w1_sieve(0, 0)) && std::isfinite(C_w2(0, 0)) && std::isfinite(C_tmac(0, 0)));
    UHA_BENCH_VERIFY(cache_w1_sieve.telemetry().pruning_ratio_percent() >= 49.0);

    const BenchResult rows[5] = {
        {
            "1. FP32 4x8 GotoBLAS",
            K * N * sizeof(float),
            ms_fp32,
            (flops / (ms_fp32 * 1e-3)) / 1e9,
            (fp32_io_bytes / (ms_fp32 * 1e-3)) / 1e9,
            (fp32_io_bytes / (ms_fp32 * 1e-3)) / 1e9,
            1.00,
        },
        {
            "2. W1 AVX2 FMA (Exact)",
            w1_panels.compressed_bytes(),
            ms_w1_exact,
            (flops / (ms_w1_exact * 1e-3)) / 1e9,
            ((act_out_bytes + static_cast<double>(w1_panels.compressed_bytes())) / (ms_w1_exact * 1e-3)) / 1e9,
            (fp32_io_bytes / (ms_w1_exact * 1e-3)) / 1e9,
            ms_fp32 / ms_w1_exact,
        },
        {
            "3. W1 AVX2 FMA + Holder",
            w1_panels.compressed_bytes() / 2,
            ms_w1_sieve,
            (flops / (ms_w1_sieve * 1e-3)) / 1e9,
            ((act_out_bytes + 0.5 * static_cast<double>(w1_panels.compressed_bytes())) / (ms_w1_sieve * 1e-3)) / 1e9,
            (fp32_io_bytes / (ms_w1_sieve * 1e-3)) / 1e9,
            ms_fp32 / ms_w1_sieve,
        },
        {
            "4. W2 Ternary AVX2 FMA",
            w2_panels.compressed_bytes(),
            ms_w2_exact,
            (flops / (ms_w2_exact * 1e-3)) / 1e9,
            ((act_out_bytes + static_cast<double>(w2_panels.compressed_bytes())) / (ms_w2_exact * 1e-3)) / 1e9,
            (fp32_io_bytes / (ms_w2_exact * 1e-3)) / 1e9,
            ms_fp32 / ms_w2_exact,
        },
        {
            "5. W1 T-MAC 256b vpshufb",
            tmac_panels.compressed_bytes(),
            ms_tmac,
            (flops / (ms_tmac * 1e-3)) / 1e9,
            ((act_out_bytes + static_cast<double>(tmac_panels.compressed_bytes())) / (ms_tmac * 1e-3)) / 1e9,
            (fp32_io_bytes / (ms_tmac * 1e-3)) / 1e9,
            ms_fp32 / ms_tmac,
        },
    };

    for (const auto& r : rows) {
        std::printf("  %-26s | %7.1f K | %6.3f ms | %6.2f GF | %6.2f GB | %7.2f GB | %6.2fx\n",
                    r.name,
                    static_cast<double>(r.weight_bytes) / 1024.0,
                    r.best_ms,
                    r.gflops,
                    r.phys_gbps,
                    r.eff_fp32_gbps,
                    r.speedup_vs_fp32);
    }
    std::printf("\n");
}

int main() {
    std::printf("================================================================================\n");
    std::printf("  SUITE F3: EMPIRICAL HOST CPU GFLOPS, CACHE RESIDENCY & BANDWIDTH BENCHMARK\n");
    std::printf("================================================================================\n\n");

    const CpuFeatures cpu = HardwareInfo::query_cpu_features();
    const BlisCacheBlocks blis = HardwareInfo::compute_blis_blocks(cpu);
    std::printf("[CPU Topology] Vendor: %s | Logical Cores: %u\n", cpu.vendor.c_str(), cpu.logical_cores);
    std::printf("[CPU Caches]   L1D = %zu KB | L2 = %zu KB | L3 = %zu KB | AVX2=%d FMA=%d AVX512BW=%d AMX=%d\n",
                cpu.l1d_cache_bytes / 1024,
                cpu.l2_cache_bytes / 1024,
                cpu.l3_cache_bytes / 1024,
                cpu.has_avx2 ? 1 : 0,
                cpu.has_fma ? 1 : 0,
                cpu.has_avx512bw ? 1 : 0,
                cpu.has_amx_tile ? 1 : 0);
    std::printf("[BLIS Blocks]  KC = %zu | MC = %zu | NC = %zu (Mr = %zu, Nr = %zu)\n\n",
                blis.kc, blis.mc, blis.nc, blis.mr, blis.nr);

    run_regime_benchmark("Regime A: Small-Batch Decode (M=4)", 4, 1024, 1024, 2, 5, blis);
    run_regime_benchmark("Regime B: Batched Prefill (M=64)", 64, 1024, 1024, 1, 3, blis);

    std::printf("[+] PASS: Empirical Host CPU GFLOPS & Bandwidth Benchmark Suite completed.\n");
    return 0;
}
