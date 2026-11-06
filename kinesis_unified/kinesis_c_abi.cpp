/**
 * @file kinesis_c_abi.cpp
 * @brief C-Linkage Shared Library ABI for Kinesis Unified C++20 Sub-2-Bit AVX2 Kernels & Smart Tiered Cache.
 *
 * Exposes exception-safe extern "C" entry points for Python ctypes / PyTorch integration:
 *   1. uha_kinesis_pack_w1_panels
 *   2. uha_kinesis_pack_w2_panels
 *   3. uha_kinesis_pack_tmac_w1_panels
 *   4. uha_kinesis_convert_w1_to_gpu_interleaved_u32
 *   5. uha_kinesis_gemm_fp32_4x8
 *   6. uha_kinesis_smart_w1_gemm
 *   7. uha_kinesis_smart_w2_gemm
 *   8. uha_kinesis_tmac_lut_w1_gemm
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>

#include "hardware_and_sandbox.hpp"
#include "memory_primitives.hpp"
#include "sub2bit_gemm_and_smart_cache.hpp"
#include "unified_sizing.h"

#if defined(_WIN32)
#define UHA_KINESIS_API extern "C" __declspec(dllexport)
#else
#define UHA_KINESIS_API extern "C" __attribute__((visibility("default")))
#endif

using namespace kinesis::unified;

UHA_KINESIS_API int uha_kinesis_pack_w1_panels(
    const float* w_kn,
    size_t K,
    size_t N,
    uint8_t* out_bits,
    uint16_t* out_scales_bf16
) {
    if (!w_kn || !out_bits || !out_scales_bf16 || K == 0 || N == 0) {
        return -1;
    }
    try {
        TensorView<const float> W(
            w_kn, Shape{static_cast<int64_t>(K), static_cast<int64_t>(N)}
        );
        W1GotoBlasPanelSet panels = pack_w1_gotoblas_panels(W);
        std::memcpy(out_bits, panels.packed_bits.data(), panels.packed_bits.size() * sizeof(uint8_t));
        std::memcpy(
            out_scales_bf16,
            panels.packed_scales_bf16.data(),
            panels.packed_scales_bf16.size() * sizeof(uint16_t)
        );
        return 0;
    } catch (...) {
        return -2;
    }
}

UHA_KINESIS_API int uha_kinesis_pack_w2_panels(
    const float* w_kn,
    size_t K,
    size_t N,
    uint16_t* out_trits,
    uint16_t* out_scales_bf16
) {
    if (!w_kn || !out_trits || !out_scales_bf16 || K == 0 || N == 0) {
        return -1;
    }
    try {
        TensorView<const float> W(
            w_kn, Shape{static_cast<int64_t>(K), static_cast<int64_t>(N)}
        );
        W2GotoBlasPanelSet panels = pack_w2_gotoblas_panels(W);
        std::memcpy(out_trits, panels.packed_trits.data(), panels.packed_trits.size() * sizeof(uint16_t));
        std::memcpy(
            out_scales_bf16,
            panels.packed_scales_bf16.data(),
            panels.packed_scales_bf16.size() * sizeof(uint16_t)
        );
        return 0;
    } catch (...) {
        return -2;
    }
}

UHA_KINESIS_API int uha_kinesis_pack_tmac_w1_panels(
    const float* w_kn,
    size_t K,
    size_t N,
    uint32_t* out_nibbles,
    uint16_t* out_scales_bf16
) {
    if (!w_kn || !out_nibbles || !out_scales_bf16 || K == 0 || N == 0 || (K % kGroupSizeG128) != 0) {
        return -1;
    }
    try {
        TensorView<const float> W(
            w_kn, Shape{static_cast<int64_t>(K), static_cast<int64_t>(N)}
        );
        W1GotoBlasPanelSet w1_panels = pack_w1_gotoblas_panels(W);
        W1TmacLutPanelSet tmac_panels = pack_w1_tmac_lut_panels(w1_panels);
        std::memcpy(out_nibbles, tmac_panels.nibbles.data(), tmac_panels.nibbles.size() * sizeof(uint32_t));
        std::memcpy(
            out_scales_bf16,
            tmac_panels.scales_bf16.data(),
            tmac_panels.scales_bf16.size() * sizeof(uint16_t)
        );
        return 0;
    } catch (...) {
        return -2;
    }
}

UHA_KINESIS_API int uha_kinesis_convert_w1_to_gpu_interleaved_u32(
    const uint8_t* in_bits,
    const uint16_t* in_scales_bf16,
    size_t K,
    size_t N,
    uint32_t* out_gpu_u32,
    uint16_t* out_gpu_scales_bf16
) {
    if (!in_bits || !in_scales_bf16 || !out_gpu_u32 || !out_gpu_scales_bf16 ||
        K == 0 || N == 0 || (K % UHA_W1_WEIGHTS_PER_U32) != 0) {
        return -1;
    }
    try {
        W1GotoBlasPanelSet cpu_panels;
        cpu_panels.K = K;
        cpu_panels.N = N;
        cpu_panels.num_panels = (N + kNr8 - 1) / kNr8;
        cpu_panels.groups_per_col = (K + kGroupSizeG128 - 1) / kGroupSizeG128;
        cpu_panels.packed_bits.assign(in_bits, in_bits + cpu_panels.num_panels * K);
        cpu_panels.packed_scales_bf16.assign(
            in_scales_bf16,
            in_scales_bf16 + cpu_panels.num_panels * cpu_panels.groups_per_col * kNr8
        );

        std::vector<uint32_t> gpu_u32;
        std::vector<uint16_t> gpu_scales;
        convert_w1_gotoblas_to_gpu_interleaved_u32(cpu_panels, gpu_u32, gpu_scales);

        std::memcpy(out_gpu_u32, gpu_u32.data(), gpu_u32.size() * sizeof(uint32_t));
        std::memcpy(out_gpu_scales_bf16, gpu_scales.data(), gpu_scales.size() * sizeof(uint16_t));
        return 0;
    } catch (...) {
        return -2;
    }
}

UHA_KINESIS_API int uha_kinesis_gemm_fp32_4x8(
    const float* a_mk,
    const float* b_kn,
    float* c_mn,
    size_t M,
    size_t N,
    size_t K
) {
    if (!a_mk || !b_kn || !c_mn || M == 0 || N == 0 || K == 0) {
        return -1;
    }
    try {
        const CpuFeatures cpu = HardwareInfo::query_cpu_features();
        const BlisCacheBlocks blis = HardwareInfo::compute_blis_blocks(cpu);
        const size_t scratch_bytes = std::max<size_t>(
            (blis.mc * blis.kc + blis.kc * ((blis.nc + kNr8 - 1) / kNr8) * kNr8) * sizeof(float) + 65536,
            256 * 1024
        );
        ArenaAllocator scratch(scratch_bytes, true, MemorySpace::HostPinnedLocked);

        TensorView<const float> A(a_mk, Shape{static_cast<int64_t>(M), static_cast<int64_t>(K)});
        TensorView<const float> B(b_kn, Shape{static_cast<int64_t>(K), static_cast<int64_t>(N)});
        TensorView<float> C(c_mn, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
        gemm_fp32_register_blocked_4x8(A, B, C, scratch);
        return 0;
    } catch (...) {
        return -2;
    }
}

UHA_KINESIS_API int uha_kinesis_smart_w1_gemm(
    const float* a_mk,
    const uint8_t* packed_bits,
    const uint16_t* packed_scales_bf16,
    float* c_mn,
    size_t M,
    size_t N,
    size_t K,
    float error_budget_eps,
    size_t* out_total_groups,
    size_t* out_pruned_groups
) {
    if (!a_mk || !packed_bits || !packed_scales_bf16 || !c_mn || M == 0 || N == 0 || K == 0) {
        return -1;
    }
    try {
        W1GotoBlasPanelSet panels;
        panels.K = K;
        panels.N = N;
        panels.num_panels = (N + kNr8 - 1) / kNr8;
        panels.groups_per_col = (K + kGroupSizeG128 - 1) / kGroupSizeG128;
        panels.packed_bits.assign(packed_bits, packed_bits + panels.num_panels * K);
        panels.packed_scales_bf16.assign(
            packed_scales_bf16,
            packed_scales_bf16 + panels.num_panels * panels.groups_per_col * kNr8
        );

        SmartTieredCache cache(panels.num_panels, panels.num_panels / 2, 0.0f, 1);
        if (error_budget_eps > 0.0f) {
            cache.set_error_budget(error_budget_eps);
        }

        TensorView<const float> A(a_mk, Shape{static_cast<int64_t>(M), static_cast<int64_t>(K)});
        TensorView<float> C(c_mn, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
        cache.execute_smart_w1_gemm(A, panels, C, error_budget_eps <= 0.0f);

        if (out_total_groups) {
            *out_total_groups = cache.telemetry().total_tile_requests;
        }
        if (out_pruned_groups) {
            *out_pruned_groups = cache.telemetry().tiles_pruned_by_energy_sieve;
        }
        return 0;
    } catch (...) {
        return -2;
    }
}

UHA_KINESIS_API int uha_kinesis_smart_w2_gemm(
    const float* a_mk,
    const uint16_t* packed_trits,
    const uint16_t* packed_scales_bf16,
    float* c_mn,
    size_t M,
    size_t N,
    size_t K,
    float error_budget_eps,
    size_t* out_total_groups,
    size_t* out_pruned_groups
) {
    if (!a_mk || !packed_trits || !packed_scales_bf16 || !c_mn || M == 0 || N == 0 || K == 0) {
        return -1;
    }
    try {
        W2GotoBlasPanelSet panels;
        panels.K = K;
        panels.N = N;
        panels.num_panels = (N + kNr8 - 1) / kNr8;
        panels.groups_per_col = (K + kGroupSizeG128 - 1) / kGroupSizeG128;
        panels.packed_trits.assign(packed_trits, packed_trits + panels.num_panels * K);
        panels.packed_scales_bf16.assign(
            packed_scales_bf16,
            packed_scales_bf16 + panels.num_panels * panels.groups_per_col * kNr8
        );

        SmartTieredCache cache(panels.num_panels, panels.num_panels / 2, 0.0f, 1);
        if (error_budget_eps > 0.0f) {
            cache.set_error_budget(error_budget_eps);
        }

        TensorView<const float> A(a_mk, Shape{static_cast<int64_t>(M), static_cast<int64_t>(K)});
        TensorView<float> C(c_mn, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
        cache.execute_smart_w2_gemm(A, panels, C, error_budget_eps <= 0.0f);

        if (out_total_groups) {
            *out_total_groups = cache.telemetry().total_tile_requests;
        }
        if (out_pruned_groups) {
            *out_pruned_groups = cache.telemetry().tiles_pruned_by_energy_sieve;
        }
        return 0;
    } catch (...) {
        return -2;
    }
}

UHA_KINESIS_API int uha_kinesis_tmac_lut_w1_gemm(
    const float* a_mk,
    const uint32_t* nibbles,
    const uint16_t* scales_bf16,
    float* c_mn,
    size_t M,
    size_t N,
    size_t K,
    float energy_sieve_tau,
    size_t* out_total_groups,
    size_t* out_pruned_groups
) {
    if (!a_mk || !nibbles || !scales_bf16 || !c_mn || M == 0 || N == 0 || K == 0 || (K % kGroupSizeG128) != 0) {
        return -1;
    }
    try {
        W1TmacLutPanelSet tmac;
        tmac.K = K;
        tmac.N = N;
        tmac.num_panels = (N + kNr8 - 1) / kNr8;
        tmac.groups_per_col = K / kGroupSizeG128;
        const size_t chunks = K / kTmacChunk;
        tmac.nibbles.assign(nibbles, nibbles + tmac.num_panels * chunks);
        tmac.scales_bf16.assign(
            scales_bf16,
            scales_bf16 + tmac.num_panels * tmac.groups_per_col * kNr8
        );

        TensorView<const float> A(a_mk, Shape{static_cast<int64_t>(M), static_cast<int64_t>(K)});
        TensorView<float> C(c_mn, Shape{static_cast<int64_t>(M), static_cast<int64_t>(N)});
        const TmacRunStats stats = execute_tmac_lut_w1_gemm(A, tmac, C, energy_sieve_tau, false);

        if (out_total_groups) {
            *out_total_groups = stats.groups_total;
        }
        if (out_pruned_groups) {
            *out_pruned_groups = stats.groups_skipped;
        }
        return 0;
    } catch (...) {
        return -2;
    }
}
