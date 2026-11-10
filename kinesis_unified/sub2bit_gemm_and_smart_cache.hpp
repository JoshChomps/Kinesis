// File: kinesis_unified/sub2bit_gemm_and_smart_cache.hpp
// Unified CPU + Single-GPU Smart Cache & Sub-2-Bit AVX2 GotoBLAS Engine.
// Fuses AI_Library (Kinesis) with .Unified_Hardware_Accel:
//   1. Upgrades Kinesis's 1x8 B-only GEMM into a true 4x8 Register-Blocked GotoBLAS AVX2 FMA
//      micro-kernel with live CPUID cache blocking (KC, MC, NC), cutting L1 B-loads by 4x.
//   2. Adds Sub-2-Bit GotoBLAS Micro-Panel Packing (1.125-bit Binary W1 & 2.125-bit Ternary W2)
//      with a branchless AVX2 register unpacker (_mm256_and_si256 + _mm256_cmpeq_epi32 +
//      _mm256_blendv_ps + _mm256_fmadd_ps), shrinking weight panels by 28.4x vs FP32 (14.2x vs FP16).
//   3. Implements SmartTieredCache: combines SlabAllocator (0.64 ns paged KV blocks),
//      Tiled-CBC predictive activation energy filtering, and hot-tile promotion between
//      CPU L3 / Pinned Host Arena and Single-GPU VRAM.

#pragma once

#include "hardware_and_sandbox.hpp"
#include "memory_primitives.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__) && (defined(_MSC_VER) || defined(__FMA__))
#include <immintrin.h>
#define KINESIS_UNIFIED_HAS_AVX2 1
#elif defined(__AVX2__) && !defined(KINESIS_UNIFIED_ALLOW_SCALAR)
// The AVX2 kernels use _mm256_fmadd_ps, so they need -mfma as well. Without it every kernel silently falls back
// to the scalar path (docs/12 4.11). Define KINESIS_UNIFIED_ALLOW_SCALAR to accept that deliberately.
#warning "sub2bit_gemm_and_smart_cache.hpp: -mavx2 without -mfma builds the SCALAR kernels; add -mfma"
#endif

namespace kinesis::unified {

inline constexpr size_t kGroupSizeG128 = UHA_QUANT_GROUP_SIZE;
inline constexpr size_t kMr4           = UHA_CPU_AVX2_MR;
inline constexpr size_t kNr8           = UHA_CPU_AVX2_NR;

template <typename TC>
inline void check_gemm_shapes(
    const TensorView<const float>& A,
    const TensorView<TC>& C,
    size_t K,
    size_t N
) {
    if (A.shape().rank() != 2 || C.shape().rank() != 2 ||
        static_cast<size_t>(A.size(1)) != K ||
        C.size(0) != A.size(0) ||
        static_cast<size_t>(C.size(1)) != N) {
        throw std::invalid_argument("GEMM shape mismatch: expected A[M, K] and C[M, N] matching weight matrix [K, N]");
    }
}

// ============================================================================
// 1. True 4x8 Register-Blocked FP32 GotoBLAS GEMM (Fixes Design Review §1.6, §2.2)
// ============================================================================

inline void pack_A_mr4_panel(
    const TensorView<const float>& A,
    float* packed_A,
    size_t m_start, size_t m_count,
    size_t k_start, size_t k_count
) noexcept {
    size_t i = 0;
    for (; i + kMr4 <= m_count; i += kMr4) {
        for (size_t k = 0; k < k_count; ++k) {
            packed_A[0] = A(static_cast<int64_t>(m_start + i + 0), static_cast<int64_t>(k_start + k));
            packed_A[1] = A(static_cast<int64_t>(m_start + i + 1), static_cast<int64_t>(k_start + k));
            packed_A[2] = A(static_cast<int64_t>(m_start + i + 2), static_cast<int64_t>(k_start + k));
            packed_A[3] = A(static_cast<int64_t>(m_start + i + 3), static_cast<int64_t>(k_start + k));
            packed_A += kMr4;
        }
    }
    if (i < m_count) {
        const size_t rem = m_count - i;
        for (size_t k = 0; k < k_count; ++k) {
            for (size_t r = 0; r < rem; ++r) {
                packed_A[r] = A(static_cast<int64_t>(m_start + i + r), static_cast<int64_t>(k_start + k));
            }
            for (size_t r = rem; r < kMr4; ++r) {
                packed_A[r] = 0.0f;
            }
            packed_A += kMr4;
        }
    }
}

inline void pack_B_nr8_panel(
    const TensorView<const float>& B,
    float* packed_B,
    size_t k_start, size_t k_count,
    size_t n_start, size_t n_count
) noexcept {
    size_t j = 0;
    for (; j + kNr8 <= n_count; j += kNr8) {
        for (size_t k = 0; k < k_count; ++k) {
            for (size_t c = 0; c < kNr8; ++c) {
                packed_B[c] = B(static_cast<int64_t>(k_start + k), static_cast<int64_t>(n_start + j + c));
            }
            packed_B += kNr8;
        }
    }
    if (j < n_count) {
        const size_t rem = n_count - j;
        for (size_t k = 0; k < k_count; ++k) {
            for (size_t c = 0; c < rem; ++c) {
                packed_B[c] = B(static_cast<int64_t>(k_start + k), static_cast<int64_t>(n_start + j + c));
            }
            for (size_t c = rem; c < kNr8; ++c) {
                packed_B[c] = 0.0f;
            }
            packed_B += kNr8;
        }
    }
}

/**
 * @brief 4x8 Register-Blocked GotoBLAS GEMM using live CPUID cache geometry and explicit
 *        Scratch Arena workspace (zero heap allocations in the hot loop).
 */
inline void gemm_fp32_register_blocked_4x8(
    const TensorView<const float>& A,
    const TensorView<const float>& B,
    const TensorView<float>& C,
    ArenaAllocator& scratch_arena
) {
    if (B.shape().rank() != 2) {
        throw std::invalid_argument("gemm_fp32_register_blocked_4x8: B must have rank 2");
    }
    const size_t M = static_cast<size_t>(A.size(0));
    const size_t K = static_cast<size_t>(B.size(0));
    const size_t N = static_cast<size_t>(B.size(1));
    check_gemm_shapes(A, C, K, N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C(static_cast<int64_t>(i), static_cast<int64_t>(j)) = 0.0f;
        }
    }

    const CpuFeatures cpu = HardwareInfo::query_cpu_features();
    const BlisCacheBlocks blk = HardwareInfo::compute_blis_blocks(cpu);

    auto scope = scratch_arena.scope();
    const size_t eff_mc = std::min(M, blk.mc);
    const size_t eff_kc = std::min(K, blk.kc);
    const size_t eff_nc = std::min(N, blk.nc);
    const size_t max_m_padded = align_up(eff_mc, kMr4);
    const size_t max_n_padded = align_up(eff_nc, kNr8);
    auto* packed_A = static_cast<float*>(
        scratch_arena.allocate(max_m_padded * eff_kc * sizeof(float), kCpuCacheLineAlignment)
    );
    auto* packed_B = static_cast<float*>(
        scratch_arena.allocate(eff_kc * max_n_padded * sizeof(float), kCpuCacheLineAlignment)
    );

    for (size_t jc = 0; jc < N; jc += blk.nc) {
        const size_t nc_cur = std::min(blk.nc, N - jc);
        for (size_t pc = 0; pc < K; pc += blk.kc) {
            const size_t kc_cur = std::min(blk.kc, K - pc);
            pack_B_nr8_panel(B, packed_B, pc, kc_cur, jc, nc_cur);

            for (size_t ic = 0; ic < M; ic += blk.mc) {
                const size_t mc_cur = std::min(blk.mc, M - ic);
                pack_A_mr4_panel(A, packed_A, ic, mc_cur, pc, kc_cur);

                // 4x8 Register-Blocked Inner Micro-Kernel
                for (size_t jr = 0; jr < nc_cur; jr += kNr8) {
                    const float* b_panel = packed_B + (jr / kNr8) * (kc_cur * kNr8);
                    const size_t n_valid = std::min(kNr8, nc_cur - jr);

                    for (size_t ir = 0; ir < mc_cur; ir += kMr4) {
                        const float* a_panel = packed_A + (ir / kMr4) * (kc_cur * kMr4);
                        const size_t m_valid = std::min(kMr4, mc_cur - ir);

#if defined(KINESIS_UNIFIED_HAS_AVX2)
                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();

                        for (size_t k = 0; k < kc_cur; ++k) {
                            // Load 1 vector of 8 B elements from L1 cache and reuse across 4 rows of A
                            const __m256 vb = _mm256_loadu_ps(b_panel + k * kNr8);
                            const float* ap = a_panel + k * kMr4;
                            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), vb, acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), vb, acc1);
                            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), vb, acc2);
                            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), vb, acc3);
                        }

                        // Do not use alignas(32) on local stack arrays: on x86_64-w64-mingw32 (GCC PR 54412),
                        // MAX_STACK_ALIGNMENT is 16 bytes, yet alignas(32) causes GCC -O3 to fold _mm256_storeu_ps
                        // back into vmovaps, faulting when RSP % 32 == 16 (e.g. from Python ctypes).
                        float tmp[4][8];
                        _mm256_storeu_ps(tmp[0], acc0);
                        _mm256_storeu_ps(tmp[1], acc1);
                        _mm256_storeu_ps(tmp[2], acc2);
                        _mm256_storeu_ps(tmp[3], acc3);

                        for (size_t r = 0; r < m_valid; ++r) {
                            for (size_t c = 0; c < n_valid; ++c) {
                                C(static_cast<int64_t>(ic + ir + r),
                                  static_cast<int64_t>(jc + jr + c)) += tmp[r][c];
                            }
                        }
#else
                        for (size_t k = 0; k < kc_cur; ++k) {
                            const float* ap = a_panel + k * kMr4;
                            const float* bp = b_panel + k * kNr8;
                            for (size_t r = 0; r < m_valid; ++r) {
                                for (size_t c = 0; c < n_valid; ++c) {
                                    C(static_cast<int64_t>(ic + ir + r),
                                      static_cast<int64_t>(jc + jr + c)) += ap[r] * bp[c];
                                }
                            }
                        }
#endif
                    }
                }
            }
        }
    }
}

// ============================================================================
// 2. Sub-2-Bit GotoBLAS Micro-Panel Packing & Branchless AVX2 FMA Unpacker
//    Shrinks FP32 weights by 28.44x (1.125 bits/weight) so hot layers stay in CPU L2/L3!
// ============================================================================

[[nodiscard]] inline uint16_t fp32_to_bf16_bits(float f) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(uint32_t));
    if ((u & 0x7F800000U) == 0x7F800000U && (u & 0x007FFFFFU) != 0U) {
        return static_cast<uint16_t>((u >> 16) | 0x0040U);
    }
    const uint32_t lsb = (u >> 16) & 1U;
    u += 0x00007FFFU + lsb;
    return static_cast<uint16_t>(u >> 16);
}

/**
 * @brief g128 absmean scale, rounded once from FP64 directly to BF16 (round-half-even).
 *        Shared rule with spinquant_packer.absmean_scale_bf16 (docs/12 4.19): an FP32 sum is not order-defined,
 *        and the sequential FP32 sum here picked a different BF16 scale than torch's pairwise sum for one group
 *        of 1,024 in G1. mean is an FP64 value; the result is exact in BF16 (no FP32 double rounding).
 */
[[nodiscard]] inline uint16_t f64_to_bf16_bits_rne(double mean) noexcept {
    if (!std::isfinite(mean) || mean == 0.0) return fp32_to_bf16_bits(static_cast<float>(mean));
    int e = 0;
    const double m = std::frexp(mean, &e);                               // mean = m * 2^e, 0.5 <= |m| < 1
    const double q = std::nearbyint(m * 256.0);                          // 8 significant bits, ties to even
    return fp32_to_bf16_bits(static_cast<float>(std::ldexp(q, e - 8)));  // exact in FP32 and in BF16
}

[[nodiscard]] inline float bf16_bits_to_fp32(uint16_t b) noexcept {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

struct W1GotoBlasPanelSet {
    size_t K{0};
    size_t N{0};
    size_t num_panels{0};
    size_t groups_per_col{0};
    // Each panel of width Nr=8 stores K bytes: 1 byte per K-step (1 bit per column in Nr=8)
    std::vector<uint8_t>  packed_bits;
    // Per-group (g=128) BF16 scale factors (16 bits / 128 weights = 0.125 bpw -> 1.125 bpw total)
    std::vector<uint16_t> packed_scales_bf16;

    [[nodiscard]] size_t compressed_bytes() const noexcept {
        return packed_bits.size() * sizeof(uint8_t) + packed_scales_bf16.size() * sizeof(uint16_t);
    }

    [[nodiscard]] size_t fp32_equivalent_bytes() const noexcept {
        return K * N * sizeof(float);
    }

    [[nodiscard]] double compression_ratio_vs_fp32() const noexcept {
        return compressed_bytes() == 0
                   ? 0.0
                   : static_cast<double>(fp32_equivalent_bytes()) /
                         static_cast<double>(compressed_bytes());
    }
};

/**
 * @brief Packs a K x N weight matrix into 1-bit Binary g128 GotoBLAS Nr=8 micro-panels.
 *        Each byte in `packed_bits` encodes 8 signs (bit c = 1 -> +alpha, bit c = 0 -> -alpha).
 */
[[nodiscard]] inline W1GotoBlasPanelSet pack_w1_gotoblas_panels(
    const TensorView<const float>& W
) {
    W1GotoBlasPanelSet out;
    out.K = static_cast<size_t>(W.size(0));
    out.N = static_cast<size_t>(W.size(1));
    out.num_panels = (out.N + kNr8 - 1) / kNr8;
    out.groups_per_col = (out.K + kGroupSizeG128 - 1) / kGroupSizeG128;

    out.packed_bits.resize(out.num_panels * out.K, 0);
    out.packed_scales_bf16.resize(out.num_panels * out.groups_per_col * kNr8, 0);

    for (size_t p = 0; p < out.num_panels; ++p) {
        const size_t n_base = p * kNr8;
        for (size_t g = 0; g < out.groups_per_col; ++g) {
            const size_t k_base = g * kGroupSizeG128;
            const size_t k_end = std::min(out.K, k_base + kGroupSizeG128);
            const size_t k_len = k_end - k_base;

            // Compute absmean BF16 scale per column in this 128-element group
            for (size_t c = 0; c < kNr8; ++c) {
                float sc = 1.0f;
                uint16_t sc_bits = fp32_to_bf16_bits(sc);
                if (n_base + c < out.N) {
                    double asum = 0.0;
                    for (size_t k = k_base; k < k_end; ++k) {
                        asum += std::fabs(static_cast<double>(W(static_cast<int64_t>(k), static_cast<int64_t>(n_base + c))));
                    }
                    // FP64 mean, one rounding to BF16 (docs/12 4.19). An all-zero group keeps scale 0 so it
                    // contributes exactly 0. (Forcing 1.0 here made pruned/zero weight groups output
                    // +/- sum(activations) instead of 0.)
                    sc_bits = f64_to_bf16_bits_rne((k_len > 0) ? (asum / static_cast<double>(k_len)) : 1.0);
                }
                out.packed_scales_bf16[(p * out.groups_per_col + g) * kNr8 + c] = sc_bits;
            }

            // Pack 8 sign bits per K-row into 1 contiguous byte for the Nr=8 micro-panel
            for (size_t k = k_base; k < k_end; ++k) {
                uint8_t byte_mask = 0;
                for (size_t c = 0; c < kNr8; ++c) {
                    if (n_base + c < out.N) {
                        const float v = W(static_cast<int64_t>(k), static_cast<int64_t>(n_base + c));
                        if (v >= 0.0f) {
                            byte_mask |= static_cast<uint8_t>(1u << c);
                        }
                    }
                }
                out.packed_bits[p * out.K + k] = byte_mask;
            }
        }
    }
    return out;
}

/**
 * @brief Unpacks a single 8-bit GotoBLAS micro-panel byte into 8 signed FP32 weights in an AVX2
 *        YMM register with zero branches:
 *          bit c == 1 -> +scale[c]
 *          bit c == 0 -> -scale[c]
 */
#if defined(KINESIS_UNIFIED_HAS_AVX2)
// The scale is taken as a pointer to 8 floats and loaded unaligned inside, never as a by-value __m256: on
// x86_64-w64-mingw32 a by-value __m256 is materialized in a caller stack temporary with an aligned vmovaps that
// the 16-byte Windows x64 stack does not guarantee (GCC PR 54412; docs/12 4.11, seen at -O3 -fno-inline).
[[nodiscard]] inline __m256 avx2_unpack_w1_nr8_byte(uint8_t packed_byte, const float* scale8) noexcept {
    const __m256 v_scale = _mm256_loadu_ps(scale8);
    const __m256i bit_positions = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i bcast = _mm256_set1_epi32(static_cast<int>(packed_byte));
    const __m256i masked = _mm256_and_si256(bcast, bit_positions);
    // Lane is 0xFFFFFFFF where bit == 0 (negative weight), 0x00000000 where bit == 1 (positive weight)
    const __m256i is_zero_bit = _mm256_cmpeq_epi32(masked, _mm256_setzero_si256());
    // Shift MSB into IEEE-754 FP32 sign bit (0x80000000) and XOR with +scale
    const __m256 sign_flip = _mm256_and_ps(_mm256_castsi256_ps(is_zero_bit), _mm256_set1_ps(-0.0f));
    return _mm256_xor_ps(v_scale, sign_flip);
}
#endif

/**
 * @brief W2 (1.58-Bit Ternary {-1, 0, +1}) GotoBLAS Nr=8 Panel Set (2.125 bits/weight).
 *        Each K-step in an Nr=8 panel stores 8 2-bit codes in 1 uint16_t word (code = q + 1 in {0, 1, 2})
 *        plus 1 BF16 scale per g=128 group -> 15.06x compression vs FP32 (7.53x vs FP16).
 */
struct W2GotoBlasPanelSet {
    size_t K{0};
    size_t N{0};
    size_t num_panels{0};
    size_t groups_per_col{0};
    std::vector<uint16_t> packed_trits;       // [num_panels * K]: 8 x 2-bit codes per uint16_t
    std::vector<uint16_t> packed_scales_bf16; // [num_panels * groups_per_col * kNr8]

    [[nodiscard]] size_t compressed_bytes() const noexcept {
        return packed_trits.size() * sizeof(uint16_t) + packed_scales_bf16.size() * sizeof(uint16_t);
    }

    [[nodiscard]] size_t fp32_equivalent_bytes() const noexcept {
        return K * N * sizeof(float);
    }

    [[nodiscard]] double compression_ratio_vs_fp32() const noexcept {
        return compressed_bytes() == 0
                   ? 0.0
                   : static_cast<double>(fp32_equivalent_bytes()) /
                         static_cast<double>(compressed_bytes());
    }
};

[[nodiscard]] inline W2GotoBlasPanelSet pack_w2_gotoblas_panels(
    const TensorView<const float>& W
) {
    W2GotoBlasPanelSet out;
    out.K = static_cast<size_t>(W.size(0));
    out.N = static_cast<size_t>(W.size(1));
    out.num_panels = (out.N + kNr8 - 1) / kNr8;
    out.groups_per_col = (out.K + kGroupSizeG128 - 1) / kGroupSizeG128;

    out.packed_trits.resize(out.num_panels * out.K, 0);
    out.packed_scales_bf16.resize(out.num_panels * out.groups_per_col * kNr8, 0);

    for (size_t p = 0; p < out.num_panels; ++p) {
        const size_t n_base = p * kNr8;
        for (size_t g = 0; g < out.groups_per_col; ++g) {
            const size_t k_base = g * kGroupSizeG128;
            const size_t k_end = std::min(out.K, k_base + kGroupSizeG128);
            const size_t k_len = k_end - k_base;

            float scales_f32[kNr8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (size_t c = 0; c < kNr8; ++c) {
                if (n_base + c < out.N) {
                    double asum = 0.0;
                    for (size_t k = k_base; k < k_end; ++k) {
                        asum += std::fabs(static_cast<double>(W(static_cast<int64_t>(k), static_cast<int64_t>(n_base + c))));
                    }
                    const double mean_abs = (k_len > 0) ? (asum / static_cast<double>(k_len)) : 0.0;
                    // BitNet b1.58 absmean (Ma et al. 2024): gamma = mean|W| over the group, W~ = RoundClip(W/gamma,
                    // -1, 1). This is the exact rule the GPU packer (spinquant_packer.quantize_and_pack_w2_ternary_
                    // interleaved) and kernels use: same 1e-6 clamp, bf16-rounded scale, round-half-even division.
                    // The previous conditional-mean scale differed on every group (docs/12 4.14), so a W2 tile
                    // decoded differently on the CPU tier than on the GPU tier.
                    const uint16_t sc_bf16 = f64_to_bf16_bits_rne(std::max(mean_abs, 1e-6));
                    out.packed_scales_bf16[(p * out.groups_per_col + g) * kNr8 + c] = sc_bf16;
                    scales_f32[c] = bf16_bits_to_fp32(sc_bf16);
                }
            }

            for (size_t k = k_base; k < k_end; ++k) {
                uint16_t word_code = 0;
                for (size_t c = 0; c < kNr8; ++c) {
                    uint16_t q_code = 1U; // 0 -> -1, 1 -> 0, 2 -> +1
                    if (n_base + c < out.N) {
                        const float v = W(static_cast<int64_t>(k), static_cast<int64_t>(n_base + c));
                        // std::nearbyint under the default FE_TONEAREST mode is round-half-even, matching torch.round
                        const float scaled = (scales_f32[c] > 0.0f) ? std::nearbyint(v / scales_f32[c]) : 0.0f;
                        const int tr = std::clamp(static_cast<int>(scaled), -1, 1);
                        q_code = static_cast<uint16_t>(tr + 1);
                    }
                    word_code |= static_cast<uint16_t>(q_code << (2 * c));
                }
                out.packed_trits[p * out.K + k] = word_code;
            }
        }
    }
    return out;
}

#if defined(KINESIS_UNIFIED_HAS_AVX2)
[[nodiscard]] inline __m256 avx2_unpack_w2_nr8_u16(uint16_t packed_trits, const float* scale8) noexcept {
    const __m256 v_scale = _mm256_loadu_ps(scale8);
    const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i bcast = _mm256_set1_epi32(static_cast<int>(packed_trits));
    const __m256i codes = _mm256_and_si256(_mm256_srlv_epi32(bcast, shifts), _mm256_set1_epi32(0x3));
    const __m256i signed_trits = _mm256_sub_epi32(codes, _mm256_set1_epi32(1));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(signed_trits), v_scale);
}
#endif

/**
 * @brief Lossless Zero-Loss Layout Bridge: Converts CPU GotoBLAS Nr=8 W1 panels into the GPU's
 *        [N, K/32] Interleaved 16-Bit Split uint32 layout (even k at bit b, odd k+1 at bit b+16)
 *        used by w1a16_hadacore_kernel.cu and ptx_bf162_emu.c when promoting hot tiles to VRAM.
 */
inline void convert_w1_gotoblas_to_gpu_interleaved_u32(
    const W1GotoBlasPanelSet& cpu_panels,
    std::vector<uint32_t>& gpu_w_packed_u32,   // [N, K / 32]
    std::vector<uint16_t>& gpu_scales_bf16     // [N, K / 128]
) {
    // The GPU layout is [N, K/128] scales with no tail group, so K must be a multiple of 128 (which
    // implies K % 32 == 0). Accepting K % 128 != 0 used to emit N*ceil(K/128) scales, one more per
    // column than the GPU layout and the Python caller allocate (docs/12 4.5).
    if (cpu_panels.K == 0 || (cpu_panels.K % kGroupSizeG128) != 0) {
        throw std::invalid_argument("convert_w1_gotoblas_to_gpu_interleaved_u32 requires K > 0 and K % 128 == 0");
    }
    if (cpu_panels.groups_per_col != cpu_panels.K / kGroupSizeG128 ||
        cpu_panels.packed_bits.size() < cpu_panels.num_panels * cpu_panels.K ||
        cpu_panels.packed_scales_bf16.size() < cpu_panels.num_panels * cpu_panels.groups_per_col * kNr8) {
        throw std::invalid_argument("convert_w1_gotoblas_to_gpu_interleaved_u32: panel set is inconsistent with K/N");
    }
    const size_t K = cpu_panels.K;
    const size_t N = cpu_panels.N;
    const size_t words_per_col = K / UHA_W1_WEIGHTS_PER_U32;
    const size_t groups_per_col = cpu_panels.groups_per_col;

    gpu_w_packed_u32.assign(N * words_per_col, 0U);
    gpu_scales_bf16.assign(N * groups_per_col, 0U);

    for (size_t n = 0; n < N; ++n) {
        const size_t p = n / kNr8;
        const size_t c = n % kNr8;

        for (size_t g = 0; g < groups_per_col; ++g) {
            gpu_scales_bf16[n * groups_per_col + g] =
                cpu_panels.packed_scales_bf16[(p * groups_per_col + g) * kNr8 + c];
        }

        for (size_t w = 0; w < words_per_col; ++w) {
            const size_t k_base = w * UHA_W1_WEIGHTS_PER_U32;
            uint32_t interleaved_u32 = 0U;
            for (size_t pair_idx = 0; pair_idx < 16; ++pair_idx) {
                const uint8_t byte_even = cpu_panels.packed_bits[p * K + k_base + 2 * pair_idx];
                const uint8_t byte_odd  = cpu_panels.packed_bits[p * K + k_base + 2 * pair_idx + 1];
                const uint32_t b0 = (byte_even >> c) & 1U;
                const uint32_t b1 = (byte_odd  >> c) & 1U;
                interleaved_u32 |= (b0 << pair_idx) | (b1 << (pair_idx + 16));
            }
            gpu_w_packed_u32[n * words_per_col + w] = interleaved_u32;
        }
    }
}


// ============================================================================
// 2b. T-MAC / bitnet.cpp-style 16-Entry LUT Kernel for Binary W1 (small batch decode, M <= 8)
//     Prior art: Wei et al., "T-MAC" (EuroSys 2025, arXiv:2407.00088).
//
//     Instead of unpacking every weight bit to FP32 and issuing an FMA, a group of 4 binary weights is
//     used as a 4-bit index into a 16-entry table of signed partial sums of the 4 matching activations:
//         T[idx] = sum_{j<4} (bit_j ? +a_j : -a_j)
//     The table is built once per (row, 4-wide K chunk) and reused by every output column.
//
//     PRECISION CONTRACT: activations are quantized per (row, g128 group) to int8 so a table entry fits
//     int16 (|T| <= 4*127). This is NOT bit-exact with the FP32 path; the error is bounded by the int8
//     activation rounding (<= 0.5 LSB per element, LSB = max|a| / 127 per group).
//
//     SIMD NOTE: the AVX2 build hoists the 16-byte low/high LUT split outside the panel loop p (once per
//     (row, group)) and broadcasts each 16-byte table across both 128-bit lanes of a 256-bit YMM register
//     (_mm256_broadcastsi128_si256) so 4 panels (4 * Nr = 32 output columns) are looked up simultaneously
//     with full 32-lane _mm256_shuffle_epi8 (vpshufb) utilization. Tail panels (num_panels % 4) use the
//     hoisted 128-bit _mm_shuffle_epi8 table. Verified bit-for-bit against the scalar LUT reference.
// ============================================================================

inline constexpr size_t kTmacChunk = 4;      // weights per LUT index
inline constexpr size_t kTmacMaxBatch = 8;   // intended regime: decode / speculative verify

struct W1TmacLutPanelSet {
    size_t K{0};
    size_t N{0};
    size_t num_panels{0};
    size_t groups_per_col{0};
    // [panel][chunk] -> one uint32 holding eight 4-bit LUT indices, nibble c belongs to column c of the panel
    std::vector<uint32_t> nibbles;
    std::vector<uint16_t> scales_bf16; // same layout as W1GotoBlasPanelSet::packed_scales_bf16

    [[nodiscard]] size_t chunks() const noexcept { return K / kTmacChunk; }
    [[nodiscard]] size_t compressed_bytes() const noexcept {
        return nibbles.size() * sizeof(uint32_t) + scales_bf16.size() * sizeof(uint16_t);
    }
};

[[nodiscard]] inline W1TmacLutPanelSet pack_w1_tmac_lut_panels(const W1GotoBlasPanelSet& src) {
    if (src.K == 0 || (src.K % kGroupSizeG128) != 0) {
        throw std::invalid_argument("pack_w1_tmac_lut_panels requires K > 0 and K % 128 == 0");
    }
    W1TmacLutPanelSet out;
    out.K = src.K;
    out.N = src.N;
    out.num_panels = src.num_panels;
    out.groups_per_col = src.groups_per_col;
    out.scales_bf16 = src.packed_scales_bf16;
    const size_t chunks = src.K / kTmacChunk;
    out.nibbles.assign(src.num_panels * chunks, 0U);
    for (size_t p = 0; p < src.num_panels; ++p) {
        for (size_t q = 0; q < chunks; ++q) {
            uint32_t word = 0;
            for (size_t c = 0; c < kNr8; ++c) {
                uint32_t idx = 0;
                for (size_t j = 0; j < kTmacChunk; ++j) {
                    const uint32_t bit = (src.packed_bits[p * src.K + q * kTmacChunk + j] >> c) & 1U;
                    idx |= bit << j;
                }
                word |= idx << (4 * c);
            }
            out.nibbles[p * chunks + q] = word;
        }
    }
    return out;
}

struct TmacRunStats {
    size_t groups_total{0};
    size_t groups_skipped{0};
};

/**
 * @brief Binary W1 GEMM through 16-entry LUTs. C[M,N] = A[M,K] * W[K,N].
 * @param energy_sieve_tau  <= 0 disables the Tiled-CBC sieve; otherwise (row, group) pairs whose L1 activation
 *                          energy is below tau * mean(group energy of that row) are skipped (LOSSY).
 * @param force_scalar      run the portable table-lookup path even when AVX2 is available (for verification).
 */
inline TmacRunStats execute_tmac_lut_w1_gemm(
    const TensorView<const float>& A,
    const W1TmacLutPanelSet& B,
    const TensorView<float>& C,
    float energy_sieve_tau = 0.0f,
    bool force_scalar = false
) {
    static_assert(kGroupSizeG128 / kTmacChunk == 32, "T-MAC LUT expects 32 chunks of 4 per g=128 group");
    const size_t K = B.K;
    const size_t N = B.N;
    check_gemm_shapes(A, C, K, N);
    const size_t M = static_cast<size_t>(A.size(0));
    const size_t groups = B.groups_per_col;
    const size_t chunks_per_group = kGroupSizeG128 / kTmacChunk;
    TmacRunStats stats;

    // Allocate per-row scratch buffers once outside the M row loop
    std::vector<float> g_scale(groups, 0.0f), g_energy(groups, 0.0f);
    std::vector<uint8_t> g_nonfinite(groups, 0);
    std::vector<int8_t> aq(K, 0);

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) C(static_cast<int64_t>(m), static_cast<int64_t>(n)) = 0.0f;

        // Per-group int8 activation quantization and L1 energy
        std::fill(g_scale.begin(), g_scale.end(), 0.0f);
        std::fill(g_energy.begin(), g_energy.end(), 0.0f);
        std::fill(aq.begin(), aq.end(), static_cast<int8_t>(0));
        std::fill(g_nonfinite.begin(), g_nonfinite.end(), static_cast<uint8_t>(0));
        // Non-finite activations (docs/12 4.12): std::max(amax, NaN) silently returns amax, so a NaN used to be
        // dropped (or, in a live group, cast to int8, which is UB). A group with any NaN/Inf now skips int8
        // quantization and is accumulated exactly in FP32 below, so NaN/Inf propagate with IEEE semantics. It is
        // never sieved, and it is excluded from the row's mean energy (one Inf would otherwise make the
        // threshold infinite and skip every finite group of the row).
        float mean_energy = 0.0f;
        size_t finite_groups = 0;
        for (size_t g = 0; g < groups; ++g) {
            float amax = 0.0f, l1 = 0.0f;
            bool nonfinite = false;
            for (size_t k = g * kGroupSizeG128; k < (g + 1) * kGroupSizeG128; ++k) {
                const float v = A(static_cast<int64_t>(m), static_cast<int64_t>(k));
                nonfinite |= !std::isfinite(v);
                amax = std::max(amax, std::fabs(v));
                l1 += std::fabs(v);
            }
            if (nonfinite) {
                g_nonfinite[g] = 1;
                g_energy[g] = std::numeric_limits<float>::infinity();
                continue;
            }
            g_energy[g] = l1;
            mean_energy += l1;
            ++finite_groups;
            if (amax > 0.0f) {
                g_scale[g] = amax / 127.0f;
                for (size_t k = g * kGroupSizeG128; k < (g + 1) * kGroupSizeG128; ++k) {
                    const float q = std::nearbyint(A(static_cast<int64_t>(m), static_cast<int64_t>(k)) / g_scale[g]);
                    aq[k] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
                }
            }
        }
        mean_energy = (finite_groups > 0) ? mean_energy / static_cast<float>(finite_groups) : 0.0f;
        const float thresh = (energy_sieve_tau > 0.0f) ? energy_sieve_tau * mean_energy : -1.0f;

        for (size_t g = 0; g < groups; ++g) {
            ++stats.groups_total;
            if (g_nonfinite[g]) {
                // Exact FP32 accumulation from the nibble signs: in word (p, k/4), nibble c bit (k % 4) is the sign
                // of W[k, n_base + c] (1 -> +scale, 0 -> -scale), the same convention the LUT uses.
                for (size_t p = 0; p < B.num_panels; ++p) {
                    const size_t n_base = p * kNr8;
                    const size_t n_valid = std::min(kNr8, N - n_base);
                    const uint16_t* sc = &B.scales_bf16[(p * groups + g) * kNr8];
                    for (size_t c = 0; c < n_valid; ++c) {
                        const float s = bf16_bits_to_fp32(sc[c]);
                        float sum = 0.0f;
                        for (size_t k = g * kGroupSizeG128; k < (g + 1) * kGroupSizeG128; ++k) {
                            const uint32_t word = B.nibbles[p * B.chunks() + k / kTmacChunk];
                            const bool pos = ((word >> (4 * c + (k % kTmacChunk))) & 1U) != 0U;
                            sum += A(static_cast<int64_t>(m), static_cast<int64_t>(k)) * (pos ? s : -s);
                        }
                        C(static_cast<int64_t>(m), static_cast<int64_t>(n_base + c)) += sum;
                    }
                }
                continue;
            }
            if (g_scale[g] == 0.0f || g_energy[g] < thresh) {
                if (g_scale[g] != 0.0f) ++stats.groups_skipped; // skipped by the sieve, not by true zeros
                continue;
            }
            // Build 16-entry int16 tables for every 4-wide chunk of this group, and store
            // the 16-byte low/high LUT splits in unaligned byte tables (avoiding __m256i[32]
            // stack arrays that trigger vmovdqa under GCC PR 54412 on Windows x64).
            int16_t lut[32][16];
            uint8_t lut_lo_bytes[32][16];
            uint8_t lut_hi_bytes[32][16];
            for (size_t q = 0; q < chunks_per_group; ++q) {
                const int8_t* a4 = &aq[g * kGroupSizeG128 + q * kTmacChunk];
                for (int idx = 0; idx < 16; ++idx) {
                    int sum = 0;
                    for (int j = 0; j < 4; ++j) sum += ((idx >> j) & 1) ? a4[j] : -a4[j];
                    const int16_t s16 = static_cast<int16_t>(sum);
                    lut[q][idx] = s16;
                    lut_lo_bytes[q][idx] = static_cast<uint8_t>(static_cast<uint16_t>(s16) & 0xFFU);
                    lut_hi_bytes[q][idx] = static_cast<uint8_t>(static_cast<uint16_t>(s16) >> 8);
                }
            }

            size_t p = 0;
#if defined(KINESIS_UNIFIED_HAS_AVX2)
            if (!force_scalar) {
                const __m256i mask0f_256 = _mm256_set1_epi8(0x0F);
                // 4-panel (32-column) full 32-lane 256-bit vpshufb loop
                for (; p + 4 <= B.num_panels; p += 4) {
                    __m256i vacc0 = _mm256_setzero_si256();
                    __m256i vacc1 = _mm256_setzero_si256();
                    __m256i vacc2 = _mm256_setzero_si256();
                    __m256i vacc3 = _mm256_setzero_si256();
                    const uint32_t* wp0 = &B.nibbles[(p + 0) * B.chunks() + g * chunks_per_group];
                    const uint32_t* wp1 = &B.nibbles[(p + 1) * B.chunks() + g * chunks_per_group];
                    const uint32_t* wp2 = &B.nibbles[(p + 2) * B.chunks() + g * chunks_per_group];
                    const uint32_t* wp3 = &B.nibbles[(p + 3) * B.chunks() + g * chunks_per_group];
                    for (size_t q = 0; q < chunks_per_group; ++q) {
                        const __m256i lut_lo_q = _mm256_broadcastsi128_si256(
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_lo_bytes[q]))
                        );
                        const __m256i lut_hi_q = _mm256_broadcastsi128_si256(
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_hi_bytes[q]))
                        );
                        const __m256i raw0123 = _mm256_setr_epi32(
                            static_cast<int>(wp0[q]), static_cast<int>(wp1[q]), 0, 0,
                            static_cast<int>(wp2[q]), static_cast<int>(wp3[q]), 0, 0);
                        const __m256i idx0123 = _mm256_unpacklo_epi8(
                            _mm256_and_si256(raw0123, mask0f_256),
                            _mm256_and_si256(_mm256_srli_epi16(raw0123, 4), mask0f_256));
                        const __m256i lo = _mm256_shuffle_epi8(lut_lo_q, idx0123);
                        const __m256i hi = _mm256_shuffle_epi8(lut_hi_q, idx0123);
                        const __m256i s16_02 = _mm256_unpacklo_epi8(lo, hi); // low 128b: p+0, high 128b: p+2
                        const __m256i s16_13 = _mm256_unpackhi_epi8(lo, hi); // low 128b: p+1, high 128b: p+3
                        vacc0 = _mm256_add_epi32(vacc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(s16_02)));
                        vacc1 = _mm256_add_epi32(vacc1, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(s16_13)));
                        vacc2 = _mm256_add_epi32(vacc2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(s16_02, 1)));
                        vacc3 = _mm256_add_epi32(vacc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(s16_13, 1)));
                    }
                    int32_t acc4[4][8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc4[0]), vacc0);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc4[1]), vacc1);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc4[2]), vacc2);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc4[3]), vacc3);
                    for (size_t sub = 0; sub < 4; ++sub) {
                        const size_t n_base = (p + sub) * kNr8;
                        const size_t n_valid = std::min(kNr8, N - n_base);
                        const uint16_t* sc = &B.scales_bf16[((p + sub) * groups + g) * kNr8];
                        for (size_t c = 0; c < n_valid; ++c) {
                            C(static_cast<int64_t>(m), static_cast<int64_t>(n_base + c)) +=
                                static_cast<float>(acc4[sub][c]) * g_scale[g] * bf16_bits_to_fp32(sc[c]);
                        }
                    }
                }
            }
#endif
            for (; p < B.num_panels; ++p) {
                const size_t n_base = p * kNr8;
                const size_t n_valid = std::min(kNr8, N - n_base);
                int32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                const uint32_t* wp = &B.nibbles[p * B.chunks() + g * chunks_per_group];
#if defined(KINESIS_UNIFIED_HAS_AVX2)
                if (!force_scalar) {
                    __m256i vacc = _mm256_setzero_si256();
                    const __m128i mask0f = _mm_set1_epi8(0x0F);
                    for (size_t q = 0; q < chunks_per_group; ++q) {
                        const __m128i lut_lo_128_q =
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_lo_bytes[q]));
                        const __m128i lut_hi_128_q =
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_hi_bytes[q]));
                        // Expand eight nibbles (4 bytes) into eight index bytes: nibble c -> byte c
                        const __m128i raw = _mm_cvtsi32_si128(static_cast<int>(wp[q]));
                        const __m128i idx = _mm_unpacklo_epi8(_mm_and_si128(raw, mask0f),
                                                               _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f));
                        const __m128i lo = _mm_shuffle_epi8(lut_lo_128_q, idx);
                        const __m128i hi = _mm_shuffle_epi8(lut_hi_128_q, idx);
                        const __m128i s16 = _mm_unpacklo_epi8(lo, hi); // eight int16 partial sums
                        vacc = _mm256_add_epi32(vacc, _mm256_cvtepi16_epi32(s16));
                    }
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc), vacc);
                } else
#endif
                {
                    for (size_t q = 0; q < chunks_per_group; ++q) {
                        for (size_t c = 0; c < kNr8; ++c) {
                            acc[c] += lut[q][(wp[q] >> (4 * c)) & 0xFU];
                        }
                    }
                }
                const uint16_t* sc = &B.scales_bf16[(p * groups + g) * kNr8];
                for (size_t c = 0; c < n_valid; ++c) {
                    C(static_cast<int64_t>(m), static_cast<int64_t>(n_base + c)) +=
                        static_cast<float>(acc[c]) * g_scale[g] * bf16_bits_to_fp32(sc[c]);
                }
            }
        }
    }
    return stats;
}

// ============================================================================
// 3. Smart CPU + Single-GPU Tiered Cache with Predictive Tiled-CBC Energy Sieve
// ============================================================================

struct SmartCacheTelemetry {
    size_t total_tile_requests{0};
    size_t tiles_pruned_by_energy_sieve{0}; // Cold tiles skipped before touching L1/L2 or PCIe
    size_t tier1_vram_hits{0};              // Served from Single-GPU VRAM hot cache
    size_t tier2_cpu_l3_hits{0};            // Computed in-place in CPU L1/L2/L3 via AVX2 Sub-2-Bit
    size_t hot_tile_promotions{0};          // Promoted from CPU L3 to GPU VRAM after repeat hits
    size_t tier1_vram_evictions{0};         // Cold Tier-1 VRAM panels evicted by hotter CPU L3 panels
    size_t kv_pages_allocated{0};
    size_t kv_pages_skipped_by_centroid{0};

    [[nodiscard]] double pruning_ratio_percent() const noexcept {
        return total_tile_requests == 0
                   ? 0.0
                   : (100.0 * static_cast<double>(tiles_pruned_by_energy_sieve)) /
                         static_cast<double>(total_tile_requests);
    }

    [[nodiscard]] double effective_bandwidth_multiplier(double compression_ratio) const noexcept {
        const size_t active = total_tile_requests - tiles_pruned_by_energy_sieve;
        if (active == 0) return compression_ratio;
        const double sparsity_gain =
            static_cast<double>(total_tile_requests) / static_cast<double>(active);
        return compression_ratio * sparsity_gain;
    }
};

/**
 * Sieve policies (fixes O8 / O9 of docs/11):
 *   - Exact (default, energy_sieve_tau == 0 and no error budget): every K-group is computed.
 *   - Relative energy (energy_sieve_tau > 0): legacy heuristic. A group is skipped when its L1 activation
 *     energy summed over all rows is below tau * mean. LOSSY with NO error guarantee: a single large row can
 *     be dropped when the other rows are small. Opt-in only.
 *   - Error budget (set_error_budget(eps > 0), overrides tau): per panel p, groups are skipped in order of
 *     increasing bound b(p, g) = max_c |scale(p, g, c)| * max_m L1(A[m, g]) while sum of skipped b <= eps.
 *     Because |w| <= |scale| for both W1 (+/- scale) and W2 ({-1, 0, +1} * scale),
 *         |C_exact[m, n] - C_sieved[m, n]| <= sum_{skipped g} |scale(p, g, c)| * L1(A[m, g]) <= eps
 *     for every output element, in exact arithmetic (floating-point rounding of the retained terms aside).
 *     Non-finite activations give an infinite bound, so their groups are never skipped.
 */
class SmartTieredCache {
public:
    explicit SmartTieredCache(size_t num_panels,
                              size_t max_vram_hot_panels,
                              float energy_sieve_tau = 0.0f,
                              uint32_t promotion_threshold = 2)
        : max_vram_panels_(max_vram_hot_panels),
          tau_(energy_sieve_tau),
          promotion_threshold_(promotion_threshold),
          panel_access_counts_(num_panels, 0),
          panel_last_touch_(num_panels, 0),
          panel_in_vram_(num_panels, false),
          kv_page_slab_(256, true) {}

    /** eps > 0 enables the guaranteed-error sieve (see class comment); eps <= 0 turns it off. */
    void set_error_budget(float abs_error_budget) noexcept { error_budget_ = abs_error_budget; }
    [[nodiscard]] bool is_exact() const noexcept { return !(error_budget_ > 0.0f) && !(tau_ > 0.0f); }

    /**
     * @brief Records a panel access and performs automatic hot-panel promotion into Tier-1 VRAM,
     *        using LFU + LRU tie-breaking eviction when Tier-1 VRAM capacity is full.
     */
    void record_panel_access_and_promote(size_t p) noexcept {
        if (p >= panel_in_vram_.size()) return;
        ++access_clock_;
        panel_last_touch_[p] = access_clock_;
        if (panel_in_vram_[p]) {
            if (panel_access_counts_[p] < UINT32_MAX) ++panel_access_counts_[p];
            ++telemetry_.tier1_vram_hits;
            return;
        }

        ++telemetry_.tier2_cpu_l3_hits;
        if (panel_access_counts_[p] < UINT32_MAX) ++panel_access_counts_[p];
        if (panel_access_counts_[p] < promotion_threshold_ || max_vram_panels_ == 0) {
            return;
        }

        if (current_vram_panels_ < max_vram_panels_) {
            panel_in_vram_[p] = true;
            ++current_vram_panels_;
            ++telemetry_.hot_tile_promotions;
            return;
        }

        // Tier-1 VRAM is full: find coldest resident VRAM panel (LFU with LRU tie-breaker)
        size_t victim = panel_in_vram_.size();
        uint32_t min_cnt = UINT32_MAX;
        uint64_t min_touch = UINT64_MAX;
        for (size_t idx = 0; idx < panel_in_vram_.size(); ++idx) {
            if (panel_in_vram_[idx]) {
                const uint32_t cnt = panel_access_counts_[idx];
                const uint64_t touch = panel_last_touch_[idx];
                if (cnt < min_cnt || (cnt == min_cnt && touch < min_touch)) {
                    min_cnt = cnt;
                    min_touch = touch;
                    victim = idx;
                }
            }
        }
        if (victim < panel_in_vram_.size() && panel_access_counts_[p] > min_cnt) {
            panel_in_vram_[victim] = false;
            panel_in_vram_[p] = true;
            ++telemetry_.tier1_vram_evictions;
            ++telemetry_.hot_tile_promotions;
        }
    }

    /**
     * @brief Executes a 4x8 Register-Blocked Sub-2-Bit W1 (1.125 bpw) GEMM with:
     *        1. Tiled-CBC Predictive Activation Energy Sieve (skips cold K-groups before fetching weights)
     *        2. Smart Tiered CPU-L3 / Single-GPU VRAM Cache tracking & LFU/LRU hot-panel promotion
     *        3. Persistent YMM accumulator loop order (p -> ir -> g -> k) so C is written once per panel
     */
    void execute_smart_w1_gemm(
        const TensorView<const float>& A,
        const W1GotoBlasPanelSet& B_packed,
        const TensorView<float>& C,
        bool force_exact = false
    ) {
        const size_t M = static_cast<size_t>(A.size(0));
        const size_t K = B_packed.K;
        const size_t N = B_packed.N;
        const size_t num_groups = B_packed.groups_per_col;

        check_gemm_shapes(A, C, K, N);
        compute_group_energies(A, K, num_groups);

        // Step 2: Iterate over GotoBLAS Nr=8 panels, keeping YMM accumulators live across all K-groups
        for (size_t p = 0; p < B_packed.num_panels; ++p) {
            const size_t n_base = p * kNr8;
            const size_t n_valid = std::min(kNr8, N - n_base);
            const uint8_t* bits_ptr = &B_packed.packed_bits[p * K];

            plan_panel_sieve(B_packed.packed_scales_bf16, p, num_groups, n_valid, force_exact);
            // Telemetry per (panel, group); the panel counts as one access per GEMM call (fixes O11)
            size_t active_groups = 0;
            for (size_t g = 0; g < num_groups; ++g) {
                ++telemetry_.total_tile_requests;
                if (skip_[g]) {
                    ++telemetry_.tiles_pruned_by_energy_sieve;
                } else {
                    ++active_groups;
                }
            }
            if (active_groups > 0) record_panel_access_and_promote(p);

            // 4x8 Register-Blocked Branchless AVX2 Unpack + FMA Micro-Kernel (p -> ir -> g -> k)
            for (size_t ir = 0; ir < M; ir += kMr4) {
                const size_t m_valid = std::min(kMr4, M - ir);

#if defined(KINESIS_UNIFIED_HAS_AVX2)
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (size_t g = 0; g < num_groups; ++g) {
                    if (skip_[g]) {
                        continue;
                    }
                    const size_t k_base = g * kGroupSizeG128;
                    const size_t k_end = std::min(K, k_base + kGroupSizeG128);
                    const uint16_t* scale_bf16_ptr =
                        &B_packed.packed_scales_bf16[(p * num_groups + g) * kNr8];
                    float scale_f32[kNr8];
                    for (size_t c = 0; c < kNr8; ++c) {
                        scale_f32[c] = bf16_bits_to_fp32(scale_bf16_ptr[c]);
                    }
                    for (size_t k = k_base; k < k_end; ++k) {
                        const __m256 vb = avx2_unpack_w1_nr8_byte(bits_ptr[k], scale_f32);
                        if (m_valid > 0)
                            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(A(static_cast<int64_t>(ir + 0), static_cast<int64_t>(k))), vb, acc0);
                        if (m_valid > 1)
                            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(A(static_cast<int64_t>(ir + 1), static_cast<int64_t>(k))), vb, acc1);
                        if (m_valid > 2)
                            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(A(static_cast<int64_t>(ir + 2), static_cast<int64_t>(k))), vb, acc2);
                        if (m_valid > 3)
                            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(A(static_cast<int64_t>(ir + 3), static_cast<int64_t>(k))), vb, acc3);
                    }
                }

                float tmp[4][8];
                _mm256_storeu_ps(tmp[0], acc0);
                _mm256_storeu_ps(tmp[1], acc1);
                _mm256_storeu_ps(tmp[2], acc2);
                _mm256_storeu_ps(tmp[3], acc3);

                for (size_t r = 0; r < m_valid; ++r) {
                    for (size_t c = 0; c < n_valid; ++c) {
                        C(static_cast<int64_t>(ir + r),
                          static_cast<int64_t>(n_base + c)) = tmp[r][c];
                    }
                }
#else
                float tmp[4][8] = {{0.0f}};
                for (size_t g = 0; g < num_groups; ++g) {
                    if (skip_[g]) {
                        continue;
                    }
                    const size_t k_base = g * kGroupSizeG128;
                    const size_t k_end = std::min(K, k_base + kGroupSizeG128);
                    const uint16_t* scale_bf16_ptr =
                        &B_packed.packed_scales_bf16[(p * num_groups + g) * kNr8];
                    float scale_f32[kNr8];
                    for (size_t c = 0; c < kNr8; ++c) {
                        scale_f32[c] = bf16_bits_to_fp32(scale_bf16_ptr[c]);
                    }
                    for (size_t k = k_base; k < k_end; ++k) {
                        const uint8_t bmask = bits_ptr[k];
                        for (size_t r = 0; r < m_valid; ++r) {
                            const float a_val = A(static_cast<int64_t>(ir + r), static_cast<int64_t>(k));
                            for (size_t c = 0; c < n_valid; ++c) {
                                const float w_val = ((bmask >> c) & 1u) ? scale_f32[c] : -scale_f32[c];
                                tmp[r][c] += a_val * w_val;
                            }
                        }
                    }
                }
                for (size_t r = 0; r < m_valid; ++r) {
                    for (size_t c = 0; c < n_valid; ++c) {
                        C(static_cast<int64_t>(ir + r),
                          static_cast<int64_t>(n_base + c)) = tmp[r][c];
                    }
                }
#endif
            }
        }
    }

    /**
     * @brief Executes a 4x8 Register-Blocked Sub-2-Bit W2 (2.125 bpw Ternary) GEMM with:
     *        1. Tiled-CBC Predictive Activation Energy Sieve
     *        2. Smart Tiered CPU-L3 / Single-GPU VRAM Cache tracking & LFU/LRU eviction
     *        3. Branchless AVX2 avx2_unpack_w2_nr8_u16 + FMA micro-kernel
     */
    void execute_smart_w2_gemm(
        const TensorView<const float>& A,
        const W2GotoBlasPanelSet& B_packed,
        const TensorView<float>& C,
        bool force_exact = false
    ) {
        const size_t M = static_cast<size_t>(A.size(0));
        const size_t K = B_packed.K;
        const size_t N = B_packed.N;
        const size_t num_groups = B_packed.groups_per_col;

        check_gemm_shapes(A, C, K, N);
        compute_group_energies(A, K, num_groups);

        for (size_t p = 0; p < B_packed.num_panels; ++p) {
            const size_t n_base = p * kNr8;
            const size_t n_valid = std::min(kNr8, N - n_base);
            const uint16_t* trits_ptr = &B_packed.packed_trits[p * K];

            plan_panel_sieve(B_packed.packed_scales_bf16, p, num_groups, n_valid, force_exact);
            // Telemetry per (panel, group); the panel counts as one access per GEMM call (fixes O11)
            size_t active_groups = 0;
            for (size_t g = 0; g < num_groups; ++g) {
                ++telemetry_.total_tile_requests;
                if (skip_[g]) {
                    ++telemetry_.tiles_pruned_by_energy_sieve;
                } else {
                    ++active_groups;
                }
            }
            if (active_groups > 0) record_panel_access_and_promote(p);

            for (size_t ir = 0; ir < M; ir += kMr4) {
                const size_t m_valid = std::min(kMr4, M - ir);

#if defined(KINESIS_UNIFIED_HAS_AVX2)
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (size_t g = 0; g < num_groups; ++g) {
                    if (skip_[g]) {
                        continue;
                    }
                    const size_t k_base = g * kGroupSizeG128;
                    const size_t k_end = std::min(K, k_base + kGroupSizeG128);
                    const uint16_t* scale_bf16_ptr =
                        &B_packed.packed_scales_bf16[(p * num_groups + g) * kNr8];
                    float scale_f32[kNr8];
                    for (size_t c = 0; c < kNr8; ++c) {
                        scale_f32[c] = bf16_bits_to_fp32(scale_bf16_ptr[c]);
                    }
                    for (size_t k = k_base; k < k_end; ++k) {
                        const __m256 vb = avx2_unpack_w2_nr8_u16(trits_ptr[k], scale_f32);
                        if (m_valid > 0)
                            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(A(static_cast<int64_t>(ir + 0), static_cast<int64_t>(k))), vb, acc0);
                        if (m_valid > 1)
                            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(A(static_cast<int64_t>(ir + 1), static_cast<int64_t>(k))), vb, acc1);
                        if (m_valid > 2)
                            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(A(static_cast<int64_t>(ir + 2), static_cast<int64_t>(k))), vb, acc2);
                        if (m_valid > 3)
                            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(A(static_cast<int64_t>(ir + 3), static_cast<int64_t>(k))), vb, acc3);
                    }
                }

                float tmp[4][8];
                _mm256_storeu_ps(tmp[0], acc0);
                _mm256_storeu_ps(tmp[1], acc1);
                _mm256_storeu_ps(tmp[2], acc2);
                _mm256_storeu_ps(tmp[3], acc3);

                for (size_t r = 0; r < m_valid; ++r) {
                    for (size_t c = 0; c < n_valid; ++c) {
                        C(static_cast<int64_t>(ir + r),
                          static_cast<int64_t>(n_base + c)) = tmp[r][c];
                    }
                }
#else
                float tmp[4][8] = {{0.0f}};
                for (size_t g = 0; g < num_groups; ++g) {
                    if (skip_[g]) {
                        continue;
                    }
                    const size_t k_base = g * kGroupSizeG128;
                    const size_t k_end = std::min(K, k_base + kGroupSizeG128);
                    const uint16_t* scale_bf16_ptr =
                        &B_packed.packed_scales_bf16[(p * num_groups + g) * kNr8];
                    float scale_f32[kNr8];
                    for (size_t c = 0; c < kNr8; ++c) {
                        scale_f32[c] = bf16_bits_to_fp32(scale_bf16_ptr[c]);
                    }
                    for (size_t k = k_base; k < k_end; ++k) {
                        const uint16_t word = trits_ptr[k];
                        for (size_t r = 0; r < m_valid; ++r) {
                            const float a_val = A(static_cast<int64_t>(ir + r), static_cast<int64_t>(k));
                            for (size_t c = 0; c < n_valid; ++c) {
                                const int code = static_cast<int>((word >> (2 * c)) & 0x3u);
                                const float w_val = static_cast<float>(code - 1) * scale_f32[c];
                                tmp[r][c] += a_val * w_val;
                            }
                        }
                    }
                }
                for (size_t r = 0; r < m_valid; ++r) {
                    for (size_t c = 0; c < n_valid; ++c) {
                        C(static_cast<int64_t>(ir + r),
                          static_cast<int64_t>(n_base + c)) = tmp[r][c];
                    }
                }
#endif
            }
        }
    }

    /**
     * @brief PowerInfer-style offline placement: pre-populates Tier-1 VRAM with the highest-scoring panels
     *        (e.g. calibration activation frequency x weight norm) so the first tokens avoid compulsory
     *        misses. Resets residency first. Ties are broken by lower panel index (deterministic).
     */
    void seed_offline_powerlaw_hot_panels(const std::vector<float>& panel_importance_scores) {
        const size_t n = std::min(panel_importance_scores.size(), panel_in_vram_.size());
        std::fill(panel_in_vram_.begin(), panel_in_vram_.end(), false);
        current_vram_panels_ = 0;
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), size_t{0});
        const size_t take = std::min(max_vram_panels_, n);
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(take), order.end(),
                          [&](size_t a, size_t b) {
                              return panel_importance_scores[a] > panel_importance_scores[b] ||
                                     (panel_importance_scores[a] == panel_importance_scores[b] && a < b);
                          });
        for (size_t i = 0; i < take; ++i) {
            panel_in_vram_[order[i]] = true;
            // Give seeded panels the promotion threshold so a single cold access cannot evict them
            panel_access_counts_[order[i]] = std::max(panel_access_counts_[order[i]], promotion_threshold_);
            ++current_vram_panels_;
        }
    }

    /**
     * @brief PowerInfer-2-style windowed aging: halves every access counter so a change of topic can evict
     *        stale panels instead of letting counters saturate. Call every W tokens.
     */
    void decay_access_frequencies() noexcept {
        for (auto& c : panel_access_counts_) c >>= 1;
    }

    [[nodiscard]] bool panel_resident_in_vram(size_t p) const noexcept {
        return p < panel_in_vram_.size() && panel_in_vram_[p];
    }

    /**
     * @brief Allocates a 64B-aligned KV-cache page from the prefaulted SlabAllocator in O(1) time.
     */
    [[nodiscard]] void* allocate_kv_page() {
        ++telemetry_.kv_pages_allocated;
        return kv_page_slab_.allocate();
    }

    void free_kv_page(void* page_ptr) noexcept {
        kv_page_slab_.deallocate(page_ptr);
    }

    void record_kv_page_centroid_filter(size_t total_pages, size_t skipped_pages) noexcept {
        (void)total_pages;
        telemetry_.kv_pages_skipped_by_centroid += skipped_pages;
    }

    [[nodiscard]] const SmartCacheTelemetry& telemetry() const noexcept { return telemetry_; }

private:
    // Per-group L1 activation energy: summed over rows (legacy tau heuristic) and max over rows (error bound).
    // Non-finite row energy is mapped to +inf so the bound never admits skipping it and sorting stays well-defined.
    void compute_group_energies(const TensorView<const float>& A, size_t K, size_t num_groups) {
        const size_t M = static_cast<size_t>(A.size(0));
        group_energy_sum_.assign(num_groups, 0.0f);
        group_row_max_l1_.assign(num_groups, 0.0f);
        float total = 0.0f;
        for (size_t g = 0; g < num_groups; ++g) {
            const size_t k_base = g * kGroupSizeG128;
            const size_t k_end = std::min(K, k_base + kGroupSizeG128);
            float sum = 0.0f, row_max = 0.0f;
            for (size_t m = 0; m < M; ++m) {
                float row_l1 = 0.0f;
                for (size_t k = k_base; k < k_end; ++k) {
                    row_l1 += std::fabs(A(static_cast<int64_t>(m), static_cast<int64_t>(k)));
                }
                if (!std::isfinite(row_l1)) row_l1 = std::numeric_limits<float>::infinity();
                sum += row_l1;
                row_max = std::max(row_max, row_l1);
            }
            group_energy_sum_[g] = sum;
            group_row_max_l1_[g] = row_max;
            total += sum;
        }
        mean_energy_ = (num_groups > 0) ? total / static_cast<float>(num_groups) : 0.0f;
    }

    // Fills skip_[g] for panel p according to the active sieve policy (see class comment).
    void plan_panel_sieve(const std::vector<uint16_t>& scales_bf16, size_t p, size_t num_groups,
                          size_t n_valid, bool force_exact) {
        skip_.assign(num_groups, 0);
        if (force_exact) return;
        if (error_budget_ > 0.0f) {
            bound_.resize(num_groups);
            order_.resize(num_groups);
            for (size_t g = 0; g < num_groups; ++g) {
                float smax = 0.0f;
                for (size_t c = 0; c < n_valid; ++c) {
                    smax = std::max(smax, std::fabs(bf16_bits_to_fp32(scales_bf16[(p * num_groups + g) * kNr8 + c])));
                }
                const float e = group_row_max_l1_[g];
                bound_[g] = std::isinf(e) ? e : smax * e;
                if (!std::isfinite(bound_[g])) bound_[g] = std::numeric_limits<float>::infinity();
            }
            std::iota(order_.begin(), order_.end(), size_t{0});
            std::stable_sort(order_.begin(), order_.end(), [&](size_t x, size_t y) { return bound_[x] < bound_[y]; });
            double spent = 0.0;
            for (const size_t g : order_) {
                if (spent + static_cast<double>(bound_[g]) > static_cast<double>(error_budget_)) break;
                spent += static_cast<double>(bound_[g]);
                skip_[g] = 1;
            }
        } else if (tau_ > 0.0f) {
            const float threshold = tau_ * mean_energy_;
            for (size_t g = 0; g < num_groups; ++g) skip_[g] = (group_energy_sum_[g] < threshold) ? 1 : 0;
        }
    }

private:
    size_t                max_vram_panels_{0};
    size_t                current_vram_panels_{0};
    uint64_t              access_clock_{0};
    float                 tau_{0.0f};
    float                 error_budget_{0.0f};
    uint32_t              promotion_threshold_{2};
    std::vector<uint32_t> panel_access_counts_;
    std::vector<uint64_t> panel_last_touch_;
    std::vector<bool>     panel_in_vram_;
    SlabAllocator<UHA_KV_PAGE_INT4_BYTES> kv_page_slab_;
    SmartCacheTelemetry   telemetry_{};
    // Scratch reused across GEMM calls (not thread-safe: one cache instance per thread)
    std::vector<float>    group_energy_sum_;
    std::vector<float>    group_row_max_l1_;
    std::vector<float>    bound_;
    std::vector<size_t>   order_;
    std::vector<uint8_t>  skip_;
    float                 mean_energy_{0.0f};
};

} // namespace kinesis::unified
