// File: fused_w1_hadacore/w1a16_host_verify_main.hpp
// Standalone Bit-Exact Host CTA/Warp Verification Suite for w1a16_hadacore_kernel.cu
// Compiles unmodified __global__ CUDA kernels on the host CPU and verifies 0 bit mismatches
// across aligned, odd/unaligned (M, N, K) tail shapes, all 3 HadaCore modes, deterministic
// Split-K decode (O16), per-warp row-max Holder energy-sieve gating (O9), and compounding
// BF16 rounding error bounds vs. FP64 Walsh-Hadamard oracle (O13).

#ifndef W1A16_HOST_VERIFY_MAIN_HPP
#define W1A16_HOST_VERIFY_MAIN_HPP

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// ==============================================================================
// 1. Weight Packing Helpers & Bit-Exact Reference Models
// ==============================================================================

static inline uint32_t pack_w1_interleaved_u32(const int8_t signs[32]) {
    uint32_t word = 0U;
    for (int p = 0; p < 16; ++p) {
        uint32_t bit_even = (signs[2 * p + 0] > 0) ? 1U : 0U;
        uint32_t bit_odd  = (signs[2 * p + 1] > 0) ? 1U : 0U;
        word |= (bit_even << p);
        word |= (bit_odd  << (16 + p));
    }
    return word;
}

static inline uint32_t pack_w2_ternary_interleaved_u32(const int8_t trits[16]) {
    uint32_t word = 0U;
    for (int p = 0; p < 8; ++p) {
        uint32_t q_even = (uint32_t)(trits[2 * p + 0] + 1);
        uint32_t q_odd  = (uint32_t)(trits[2 * p + 1] + 1);
        word |= ((q_even & 3U) << (2 * p));
        word |= ((q_odd  & 3U) << (16 + 2 * p));
    }
    return word;
}

static void ref_rotate_m16k32_tile_bf16(
    uint16_t a_out[16][32],
    const uint16_t a_in[16][32],
    int mode
) {
    if (mode <= 0) {
        memcpy(a_out, a_in, 16 * 32 * sizeof(uint16_t));
        return;
    }

    uint16_t t_sub[2][16][16];
    for (int sub = 0; sub < 2; ++sub) {
        for (int m = 0; m < 16; ++m) {
            for (int n = 0; n < 16; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < 16; ++k) {
                    float a_val = host_bf16_to_f32(a_in[m][sub * 16 + k]);
                    uint16_t h_bf = (__builtin_popcount(k & n) & 1) ? 0xBE80U : 0x3E80U;
                    float h_val = host_bf16_to_f32(h_bf);
                    acc += a_val * h_val;
                }
                t_sub[sub][m][n] = host_f32_to_bf16_rn(acc);
            }
        }
    }

    if (mode == 1) {
        for (int m = 0; m < 16; ++m) {
            for (int k = 0; k < 16; ++k) {
                a_out[m][k]      = t_sub[0][m][k];
                a_out[m][16 + k] = t_sub[1][m][k];
            }
        }
        return;
    }

    for (int m = 0; m < 16; ++m) {
        for (int k = 0; k < 16; k += 2) {
            uint32_t p0 = ((uint32_t)t_sub[0][m][k]) | (((uint32_t)t_sub[0][m][k + 1]) << 16);
            uint32_t p1 = ((uint32_t)t_sub[1][m][k]) | (((uint32_t)t_sub[1][m][k + 1]) << 16);
            uint32_t s0 = host_ptx_fma_rn_bf16x2(p0, 0x3F353F35U, 0x80008000U);
            uint32_t r0 = host_ptx_fma_rn_bf16x2(p1, 0x3F353F35U, s0);
            uint32_t r1 = host_ptx_fma_rn_bf16x2(p1, 0xBF35BF35U, s0);

            a_out[m][k + 0]      = (uint16_t)(r0 & 0xFFFFU);
            a_out[m][k + 1]      = (uint16_t)(r0 >> 16);
            a_out[m][16 + k + 0] = (uint16_t)(r1 & 0xFFFFU);
            a_out[m][16 + k + 1] = (uint16_t)(r1 >> 16);
        }
    }
}

// Bit-exact CPU reference for fused_w1a16_hadacore_gemm_kernel and fused_w2a16_hadacore_gemm_kernel
static void ref_hadacore_sub2bit_gemm(
    const std::vector<uint32_t>& a_bf162,
    const std::vector<int8_t>& w_quant,
    const std::vector<uint16_t>& scales_bf16,
    std::vector<float>& c_ref,
    int M, int N, int K,
    int mode,
    bool is_ternary,
    float tau_sieve = 0.0f
) {
    c_ref.assign((size_t)M * (size_t)N, 0.0f);
    const int num_k_tiles = (K + 31) >> 5;
    const int g128_stride = (K + 127) >> 7;
    const int m_blocks = (M + 63) >> 6;

    for (int wb_m = 0; wb_m < m_blocks; ++wb_m) {
        const int warp_row_base = wb_m * 64;
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            const int k_base = kt * 32;
            uint16_t a_rot_warp[4][16][32];

            for (int mi = 0; mi < 4; ++mi) {
                uint16_t a_raw_m16[16][32];
                memset(a_raw_m16, 0, sizeof(a_raw_m16));
                for (int r = 0; r < 16; ++r) {
                    const int g_r = warp_row_base + mi * 16 + r;
                    if (g_r < M) {
                        for (int k_pair = 0; k_pair < 16; ++k_pair) {
                            const int g_k = k_base + 2 * k_pair;
                            if (g_k + 1 < K) {
                                uint32_t packed = a_bf162[g_r * (K >> 1) + (g_k >> 1)];
                                a_raw_m16[r][2 * k_pair + 0] = (uint16_t)(packed & 0xFFFFU);
                                a_raw_m16[r][2 * k_pair + 1] = (uint16_t)(packed >> 16);
                            }
                        }
                    }
                }
                const bool has_sub1 = ((k_base + 16) < K);
                const int eff_mode = (mode >= 2 && !has_sub1) ? 1 : mode;
                ref_rotate_m16k32_tile_bf16(a_rot_warp[mi], a_raw_m16, eff_mode);
            }

            // Evaluate exact row-maximum Holder L1 gate warp_sieve_k32_active (O9) if tau_sieve > 0
            if (tau_sieve > 0.0f) {
                float warp_max_row_l1 = 0.0f;
                for (int mi = 0; mi < 4; ++mi) {
                    for (int group_id = 0; group_id < 8; ++group_id) {
                        const int r_top = group_id;
                        const int r_bot = group_id + 8;
                        float row_top_sum = 0.0f;
                        float row_bot_sum = 0.0f;
                        for (int t = 0; t < 4; ++t) {
                            uint32_t top0 = (((uint32_t)a_rot_warp[mi][r_top][0  + 2 * t + 0]) |
                                            (((uint32_t)a_rot_warp[mi][r_top][0  + 2 * t + 1]) << 16)) & 0x7FFF7FFFU;
                            uint32_t top1 = (((uint32_t)a_rot_warp[mi][r_top][0  + 2 * t + 8]) |
                                            (((uint32_t)a_rot_warp[mi][r_top][0  + 2 * t + 9]) << 16)) & 0x7FFF7FFFU;
                            uint32_t top2 = (((uint32_t)a_rot_warp[mi][r_top][16 + 2 * t + 0]) |
                                            (((uint32_t)a_rot_warp[mi][r_top][16 + 2 * t + 1]) << 16)) & 0x7FFF7FFFU;
                            uint32_t top3 = (((uint32_t)a_rot_warp[mi][r_top][16 + 2 * t + 8]) |
                                            (((uint32_t)a_rot_warp[mi][r_top][16 + 2 * t + 9]) << 16)) & 0x7FFF7FFFU;
                            uint32_t t01 = host_ptx_fma_rn_bf16x2(top0, 0x3F803F80U, top1);
                            uint32_t t23 = host_ptx_fma_rn_bf16x2(top2, 0x3F803F80U, top3);
                            uint32_t pt  = host_ptx_fma_rn_bf16x2(t01,  0x3F803F80U, t23);
                            row_top_sum += host_bf16_to_f32((uint16_t)(pt & 0xFFFFU)) +
                                           host_bf16_to_f32((uint16_t)(pt >> 16));

                            uint32_t bot0 = (((uint32_t)a_rot_warp[mi][r_bot][0  + 2 * t + 0]) |
                                            (((uint32_t)a_rot_warp[mi][r_bot][0  + 2 * t + 1]) << 16)) & 0x7FFF7FFFU;
                            uint32_t bot1 = (((uint32_t)a_rot_warp[mi][r_bot][0  + 2 * t + 8]) |
                                            (((uint32_t)a_rot_warp[mi][r_bot][0  + 2 * t + 9]) << 16)) & 0x7FFF7FFFU;
                            uint32_t bot2 = (((uint32_t)a_rot_warp[mi][r_bot][16 + 2 * t + 0]) |
                                            (((uint32_t)a_rot_warp[mi][r_bot][16 + 2 * t + 1]) << 16)) & 0x7FFF7FFFU;
                            uint32_t bot3 = (((uint32_t)a_rot_warp[mi][r_bot][16 + 2 * t + 8]) |
                                            (((uint32_t)a_rot_warp[mi][r_bot][16 + 2 * t + 9]) << 16)) & 0x7FFF7FFFU;
                            uint32_t b01 = host_ptx_fma_rn_bf16x2(bot0, 0x3F803F80U, bot1);
                            uint32_t b23 = host_ptx_fma_rn_bf16x2(bot2, 0x3F803F80U, bot3);
                            uint32_t pb  = host_ptx_fma_rn_bf16x2(b01,  0x3F803F80U, b23);
                            row_bot_sum += host_bf16_to_f32((uint16_t)(pb & 0xFFFFU)) +
                                           host_bf16_to_f32((uint16_t)(pb >> 16));
                        }
                        if (row_top_sum > warp_max_row_l1) warp_max_row_l1 = row_top_sum;
                        if (row_bot_sum > warp_max_row_l1) warp_max_row_l1 = row_bot_sum;
                    }
                }
                if ((warp_max_row_l1 * (1.0f / 32.0f)) <= tau_sieve) {
                    continue;
                }
            }

            // Accumulate two K=16 MMA sub-steps in exact k=0..15 order
            for (int sub = 0; sub < 2; ++sub) {
                const int k_sub = k_base + sub * 16;
                for (int mi = 0; mi < 4; ++mi) {
                    for (int r = 0; r < 16; ++r) {
                        const int g_r = warp_row_base + mi * 16 + r;
                        if (g_r >= M) continue;
                        for (int c = 0; c < N; ++c) {
                            uint16_t sc_bf16 = scales_bf16[c * g128_stride + (k_base >> 7)];
                            float acc = c_ref[g_r * N + c];
                            for (int k = 0; k < 16; ++k) {
                                float a_f = host_bf16_to_f32(a_rot_warp[mi][r][sub * 16 + k]);
                                float w_f = 0.0f;
                                if ((k_sub + 16) > K) {
                                    w_f = 0.0f;
                                } else {
                                    int8_t q = (k_sub + k < K) ? w_quant[c * K + k_sub + k] : (is_ternary ? 0 : -1);
                                    uint16_t unit_bf = (q < 0) ? 0xBF80U : ((q > 0) ? 0x3F80U : 0x0000U);
                                    uint32_t w_bf2 = host_ptx_fma_rn_bf16x2(
                                        (uint32_t)unit_bf, (uint32_t)sc_bf16, 0x80008000U);
                                    w_f = host_bf16_to_f32((uint16_t)(w_bf2 & 0xFFFFU));
                                }
                                acc += a_f * w_f;
                            }
                            c_ref[g_r * N + c] = acc;
                        }
                    }
                }
            }
        }
    }
}

// ==============================================================================
// 2. Test Runners for W1A16, W2A16, Stream-K Decode, Row-Max Sieve & O13 Oracle
// ==============================================================================

static void verify_w1a16_case(
    int M, int N, int K, int mode,
    float tau_sieve = 0.0f,
    int cold_warp_m = -1,
    int single_outlier_row = -1
) {
    assert(K % 32 == 0);
    const int k32_stride = (K + 31) >> 5;
    const int g128_stride = (K + 127) >> 7;

    std::vector<uint32_t> a_bf162((size_t)M * (size_t)(K >> 1));
    std::vector<int8_t> w_signs((size_t)N * (size_t)K);
    std::vector<uint32_t> w_packed((size_t)N * (size_t)k32_stride);
    std::vector<uint16_t> scales_bf16((size_t)N * (size_t)g128_stride);
    std::vector<float> c_gpu((size_t)M * (size_t)N, -999.0f);
    std::vector<float> c_ref((size_t)M * (size_t)N, 0.0f);

    uint32_t rng = (uint32_t)(M * 131 + N * 37 + K * 17 + mode * 7 + 0x12345U);
    auto next_u32 = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    for (int m = 0; m < M; ++m) {
        for (int kp = 0; kp < (K >> 1); ++kp) {
            float f0 = ((int)(next_u32() % 2001) - 1000) * 0.002f;
            float f1 = ((int)(next_u32() % 2001) - 1000) * 0.002f;
            if (single_outlier_row >= 0) {
                int kt = (2 * kp) >> 5;
                if (kt == 0) {
                    if (m == single_outlier_row) {
                        f0 = 1.5f;
                        f1 = -1.5f;
                    } else {
                        f0 = 0.0f;
                        f1 = 0.0f;
                    }
                }
            } else if (cold_warp_m >= 0) {
                int wm = m >> 6;
                int kt = (2 * kp) >> 5;
                if (wm == cold_warp_m && kt == 0) {
                    f0 *= 0.0001f;
                    f1 *= 0.0001f;
                } else if (wm != cold_warp_m && kt == 1) {
                    f0 *= 0.0001f;
                    f1 *= 0.0001f;
                } else {
                    f0 = (f0 >= 0.0f ? 1.5f : -1.5f);
                    f1 = (f1 >= 0.0f ? 1.5f : -1.5f);
                }
            }
            a_bf162[m * (K >> 1) + kp] = host_cvt_f32x2_to_bf162(f0, f1);
        }
    }

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < g128_stride; ++g) {
            float sc = 0.05f + ((next_u32() % 500) * 0.001f);
            scales_bf16[n * g128_stride + g] = host_f32_to_bf16_rn(sc);
        }
        for (int kt = 0; kt < k32_stride; ++kt) {
            int8_t block_signs[32];
            for (int i = 0; i < 32; ++i) {
                int8_t s = (next_u32() & 1U) ? 1 : -1;
                block_signs[i] = s;
                w_signs[n * K + kt * 32 + i] = s;
            }
            w_packed[n * k32_stride + kt] = pack_w1_interleaved_u32(block_signs);
        }
    }

    dim3 grid((N + 255) / 256, (M + 127) / 128, 1);
    host_launch_grid_cta(grid, 256, [&]() {
        fused_w1a16_hadacore_gemm_kernel(
            a_bf162.data(),
            w_packed.data(),
            scales_bf16.data(),
            c_gpu.data(),
            M, N, K,
            mode,
            tau_sieve
        );
    });

    ref_hadacore_sub2bit_gemm(a_bf162, w_signs, scales_bf16, c_ref, M, N, K, mode, false, tau_sieve);

    if (single_outlier_row >= 0) {
        // Negative control (O9): prove that the single outlier row was NOT skipped (matches exact tau=0.0f GEMM)
        std::vector<float> c_exact;
        ref_hadacore_sub2bit_gemm(a_bf162, w_signs, scales_bf16, c_exact, M, N, K, mode, false, 0.0f);
        for (int i = 0; i < M * N; ++i) {
            assert(__float_as_uint(c_gpu[i]) == __float_as_uint(c_exact[i]));
        }
    }

    int bit_mismatches = 0;
    float max_abs_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        uint32_t u_gpu = __float_as_uint(c_gpu[i]);
        uint32_t u_ref = __float_as_uint(c_ref[i]);
        if (u_gpu != u_ref) {
            bit_mismatches++;
            float err = fabsf(c_gpu[i] - c_ref[i]);
            if (err > max_abs_err) max_abs_err = err;
        }
    }

    if (bit_mismatches != 0) {
        fprintf(stderr, "FAIL [W1A16] M=%d N=%d K=%d mode=%d: %d bit mismatches (max_err=%e)\n",
                M, N, K, mode, bit_mismatches, max_abs_err);
        exit(1);
    }
}

static void verify_w2a16_case(int M, int N, int K, int mode) {
    assert(K % 16 == 0);
    const int k16_stride = (K + 15) >> 4;
    const int g128_stride = (K + 127) >> 7;

    std::vector<uint32_t> a_bf162((size_t)M * (size_t)(K >> 1));
    std::vector<int8_t> w_trits((size_t)N * (size_t)K);
    std::vector<uint32_t> w2_packed((size_t)N * (size_t)k16_stride);
    std::vector<uint16_t> scales_bf16((size_t)N * (size_t)g128_stride);
    std::vector<float> c_gpu((size_t)M * (size_t)N, -999.0f);
    std::vector<float> c_ref((size_t)M * (size_t)N, 0.0f);

    uint32_t rng = (uint32_t)(M * 257 + N * 73 + K * 29 + mode * 11 + 0x9ABCDEFU);
    auto next_u32 = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    for (size_t i = 0; i < a_bf162.size(); ++i) {
        float f0 = ((int)(next_u32() % 2001) - 1000) * 0.002f;
        float f1 = ((int)(next_u32() % 2001) - 1000) * 0.002f;
        a_bf162[i] = host_cvt_f32x2_to_bf162(f0, f1);
    }

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < g128_stride; ++g) {
            float sc = 0.05f + ((next_u32() % 500) * 0.001f);
            scales_bf16[n * g128_stride + g] = host_f32_to_bf16_rn(sc);
        }
        for (int k16 = 0; k16 < k16_stride; ++k16) {
            int8_t block_trits[16];
            for (int i = 0; i < 16; ++i) {
                int8_t trit = (int8_t)((int)(next_u32() % 3) - 1);
                block_trits[i] = trit;
                w_trits[n * K + k16 * 16 + i] = trit;
            }
            w2_packed[n * k16_stride + k16] = pack_w2_ternary_interleaved_u32(block_trits);
        }
    }

    dim3 grid((N + 255) / 256, (M + 127) / 128, 1);
    host_launch_grid_cta(grid, 256, [&]() {
        fused_w2a16_hadacore_gemm_kernel(
            a_bf162.data(),
            w2_packed.data(),
            scales_bf16.data(),
            c_gpu.data(),
            M, N, K,
            mode,
            0.0f
        );
    });

    ref_hadacore_sub2bit_gemm(a_bf162, w_trits, scales_bf16, c_ref, M, N, K, mode, true, 0.0f);

    int bit_mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        if (__float_as_uint(c_gpu[i]) != __float_as_uint(c_ref[i])) {
            bit_mismatches++;
        }
    }
    if (bit_mismatches != 0) {
        fprintf(stderr, "FAIL [W2A16] M=%d N=%d K=%d mode=%d: %d bit mismatches\n",
                M, N, K, mode, bit_mismatches);
        exit(1);
    }
}

static void verify_streamk_decode_case(int M, int N, int K, int split_k_slices) {
    assert(M <= 16 && K % 32 == 0);
    const int k32_stride = (K + 31) >> 5;
    const int g128_stride = (K + 127) >> 7;

    std::vector<uint32_t> a_bf162((size_t)M * (size_t)(K >> 1));
    std::vector<int8_t> w_signs((size_t)N * (size_t)K);
    std::vector<uint32_t> w_packed((size_t)N * (size_t)k32_stride);
    std::vector<uint16_t> scales_bf16((size_t)N * (size_t)g128_stride);
    std::vector<float> c_gpu_atomic((size_t)M * (size_t)N, 0.0f);
    std::vector<float> c_gpu_det((size_t)M * (size_t)N, 0.0f);
    std::vector<float> splitk_ws((size_t)split_k_slices * (size_t)M * (size_t)N, 0.0f);
    std::vector<float> c_ref((size_t)M * (size_t)N, 0.0f);

    uint32_t rng = (uint32_t)(M * 97 + N * 53 + K * 19 + split_k_slices * 3 + 0x55AA55U);
    auto next_u32 = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    for (size_t i = 0; i < a_bf162.size(); ++i) {
        float f0 = ((int)(next_u32() % 2001) - 1000) * 0.002f;
        float f1 = ((int)(next_u32() % 2001) - 1000) * 0.002f;
        a_bf162[i] = host_cvt_f32x2_to_bf162(f0, f1);
    }

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < g128_stride; ++g) {
            float sc = 0.05f + ((next_u32() % 500) * 0.001f);
            scales_bf16[n * g128_stride + g] = host_f32_to_bf16_rn(sc);
        }
        for (int kt = 0; kt < k32_stride; ++kt) {
            int8_t block_signs[32];
            for (int i = 0; i < 32; ++i) {
                int8_t s = (next_u32() & 1U) ? 1 : -1;
                block_signs[i] = s;
                w_signs[n * K + kt * 32 + i] = s;
            }
            w_packed[n * k32_stride + kt] = pack_w1_interleaved_u32(block_signs);
        }
    }

    dim3 grid((N + 63) / 64, split_k_slices, 1);

    // Path A: L2 atomicAdd reduction
    host_launch_grid_cta(grid, 128, [&]() {
        marlin_streamk_w1a16_decode_kernel(
            a_bf162.data(),
            w_packed.data(),
            scales_bf16.data(),
            c_gpu_atomic.data(),
            M, N, K,
            split_k_slices,
            nullptr
        );
    });

    // Path B (O16): Deterministic workspace accumulation + fixed-order slice reduction
    host_launch_grid_cta(grid, 128, [&]() {
        marlin_streamk_w1a16_decode_kernel(
            a_bf162.data(),
            w_packed.data(),
            scales_bf16.data(),
            c_gpu_det.data(),
            M, N, K,
            split_k_slices,
            splitk_ws.data()
        );
    });
    dim3 red_grid((M * N + 255) / 256, 1, 1);
    host_launch_grid_cta(red_grid, 256, [&]() {
        marlin_streamk_reduce_slices_kernel(
            splitk_ws.data(),
            c_gpu_det.data(),
            M, N,
            split_k_slices
        );
    });

    ref_hadacore_sub2bit_gemm(a_bf162, w_signs, scales_bf16, c_ref, M, N, K, 0, false, 0.0f);

    for (int i = 0; i < M * N; ++i) {
        if (fabsf(c_gpu_atomic[i] - c_ref[i]) > 1e-4f ||
            fabsf(c_gpu_det[i] - c_ref[i]) > 1e-4f ||
            __float_as_uint(c_gpu_det[i]) != __float_as_uint(c_gpu_atomic[i])) {
            fprintf(stderr, "FAIL [Stream-K Decode] M=%d N=%d K=%d split_k=%d\n",
                    M, N, K, split_k_slices);
            exit(1);
        }
    }
}

// O13: End-to-End Compounding BF16 Rounding Error Budget vs. True FP64 Walsh-Hadamard Oracle
static void verify_compounding_bf16_error_budget_o13(void) {
    const int M = 16;
    const int K = 32;
    const double inv_sqrt32 = 1.0 / sqrt(32.0);
    double H32_fp64[32][32];
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            H32_fp64[i][j] = ((__builtin_popcount(i & j) & 1) ? -1.0 : 1.0) * inv_sqrt32;
        }
    }

    uint32_t rng = 0xC0FFEE13U;
    auto next_u32 = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    for (int dist = 0; dist < 2; ++dist) {
        // dist 0: unit-scale Gaussian-like [-1.0, +1.0]
        // dist 1: heavy-tailed outlier activation (channel k=7 has 50x spike)
        uint16_t a_in_bf16[16][32];
        uint16_t a_rot_bf16[16][32];
        double a_in_fp64[16][32];

        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                float v = ((int)(next_u32() % 2001) - 1000) * 0.001f;
                if (dist == 1 && k == 7) {
                    v = (m & 1) ? 50.0f : -50.0f;
                }
                a_in_bf16[m][k] = host_f32_to_bf16_rn(v);
                a_in_fp64[m][k] = (double)host_bf16_to_f32(a_in_bf16[m][k]);
            }
        }

        ref_rotate_m16k32_tile_bf16(a_rot_bf16, a_in_bf16, 2);

        double sum_sq_err = 0.0;
        double sum_sq_ref = 0.0;
        double max_rel_row_err = 0.0;
        for (int m = 0; m < M; ++m) {
            double row_sq_err = 0.0;
            double row_sq_ref = 0.0;
            for (int n = 0; n < K; ++n) {
                double ref_val = 0.0;
                for (int k = 0; k < K; ++k) {
                    ref_val += a_in_fp64[m][k] * H32_fp64[k][n];
                }
                double got_val = (double)host_bf16_to_f32(a_rot_bf16[m][n]);
                double diff = got_val - ref_val;
                row_sq_err += diff * diff;
                row_sq_ref += ref_val * ref_val;
            }
            sum_sq_err += row_sq_err;
            sum_sq_ref += row_sq_ref;
            double rel_row = sqrt(row_sq_err / (row_sq_ref + 1e-30));
            if (rel_row > max_rel_row_err) max_rel_row_err = rel_row;
        }
        double rel_l2 = sqrt(sum_sq_err / sum_sq_ref);
        // 2-stage BF16 (7-bit mantissa, ulp = 2^-7 ~= 0.0078125) has theoretical compounding
        // relative L2 bound <= 2 * 2^-7 ~= 0.015625 (1.56%). Verify empirical error < 0.006 (0.6%).
        assert(rel_l2 < 0.006 && max_rel_row_err < 0.008);
    }
}

int main(void) {
    printf("====================================================================\n");
    printf("Host CTA (256-Fiber) & Warp-Collective Verification Suite\n");
    printf("Target: fused_w1_hadacore/w1a16_hadacore_kernel.cu (__global__)\n");
    printf("====================================================================\n");

    // [1] W1A16 Fused HadaCore GEMM across aligned & odd/unaligned (M, N, K) shapes and modes 0, 1, 2
    const int w1_shapes[][3] = {
        {1,   1,   32},
        {7,   9,   32},
        {15,  63,  64},
        {16,  64,  64},
        {17,  65,  64},
        {63,  127, 128},
        {65,  129, 64},
        {128, 256, 64},
        {131, 259, 64}
    };
    const int num_w1_shapes = (int)(sizeof(w1_shapes) / sizeof(w1_shapes[0]));
    int w1_tests_passed = 0;
    for (int s = 0; s < num_w1_shapes; ++s) {
        for (int mode = 0; mode <= 2; ++mode) {
            verify_w1a16_case(w1_shapes[s][0], w1_shapes[s][1], w1_shapes[s][2], mode);
            w1_tests_passed++;
        }
    }
    printf("[1] fused_w1a16_hadacore_gemm_kernel (%d aligned & odd-(M,N,K) shapes x modes 0..2): PASS (0 bit mismatches)\n",
           w1_tests_passed);

    // [2] W2A16 Ternary (BitNet b1.58) Fused HadaCore GEMM with Swizzle<3,3,3>+ldmatrix & K%32==16 O1 tail guard
    const int w2_shapes[][4] = {
        {16, 64,  32, 0},
        {16, 64,  32, 1},
        {16, 64,  32, 2},
        {17, 67,  48, 0},
        {17, 67,  48, 1},
        {65, 71, 128, 0},
        {65, 71, 128, 1},
        {65, 71, 128, 2}
    };
    const int num_w2_shapes = (int)(sizeof(w2_shapes) / sizeof(w2_shapes[0]));
    for (int s = 0; s < num_w2_shapes; ++s) {
        verify_w2a16_case(w2_shapes[s][0], w2_shapes[s][1], w2_shapes[s][2], w2_shapes[s][3]);
    }
    verify_w2a16_case(17, 67, 48, 2); // Verify Mode 2 -> Mode 1 tail fallback on K % 32 == 16
    printf("[2] fused_w2a16_hadacore_gemm_kernel (%d Swizzle<3,3,3>+ldmatrix & O1 tail cases): PASS (0 bit mismatches)\n",
           num_w2_shapes);

    // [3] Marlin Stream-K Small-Batch (M <= 16) Decode Kernel (Atomic & O16 Deterministic Workspace Reduction)
    verify_streamk_decode_case(1,  64,  128, 1);
    verify_streamk_decode_case(1,  64,  128, 4);
    verify_streamk_decode_case(5,  131, 128, 2);
    verify_streamk_decode_case(16, 192, 128, 4);
    printf("[3] marlin_streamk_w1a16_decode_kernel + O16 deterministic slice reduction: PASS\n");

    // [4] Inter-warp divergent warp_sieve_k32_active Row-Max Holder L1 gating + Single-Outlier-Row Negative Control (O9)
    verify_w1a16_case(128, 64, 64, 1, 0.25f, 0, -1);
    verify_w1a16_case(128, 64, 64, 2, 0.25f, 1, -1);
    verify_w1a16_case(64,  64, 64, 0, 0.25f, -1, 19); // 1 loud row (1.5) + 63 zero rows: mean=0.023 < 0.25, rowmax=1.5 > 0.25
    printf("[4] warp_sieve_k32_active O9 row-max Holder L1 gate + single-outlier-row negative control: PASS\n");

    // [5] O13 End-to-End 2-Stage Kronecker H_32 BF16 Compounding Rounding Error Budget vs. FP64 Oracle
    verify_compounding_bf16_error_budget_o13();
    printf("[5] O13 2-stage Kronecker H_32 compounding BF16 rounding error budget (<0.6%% rel L2 vs. FP64): PASS\n");

    printf("====================================================================\n");
    printf("ALL 44 HOST CTA/WARP KERNEL VERIFICATION CASES PASSED (0 ERRORS)\n");
    printf("====================================================================\n");
    return 0;
}

#endif // W1A16_HOST_VERIFY_MAIN_HPP
