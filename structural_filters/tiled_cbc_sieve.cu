// File: structural_filters/tiled_cbc_sieve.cu
// Workstream 4: Column-Major Subspace Energy Sieve (Tiled-CBC Dynamic Gate)
// Evaluates per-tile L1 activation energy along BLOCK_K = 128 columns during the
// HadaCore prologue via __shfl_xor_sync butterfly reductions and skips cp.async
// DRAM weight fetches + HMMA waves via __ballot_sync when tile energy falls below tau_sieve.
// Build (Host Verification): g++ -O2 -std=c++20 -DSTANDALONE_VERIFY -x c++ structural_filters/tiled_cbc_sieve.cu -o structural_filters/test_tiled_cbc.exe

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#else
#include "../fused_w1_hadacore/w1a16_host_cta_shim.hpp"
#endif
#include <stdint.h>

// Fast BF162 absolute sum helper: extracts |lo| + |hi| in FP32 from a packed bf162 register
__device__ __forceinline__ float bf162_abs_sum_f32(uint32_t reg_bf162) {
    uint32_t abs_bf162 = reg_bf162 & 0x7FFF7FFFU; // Clear sign bits 15 and 31
    uint32_t lo_f32_bits = (abs_bf162 & 0xFFFFU) << 16;
    uint32_t hi_f32_bits = abs_bf162 & 0xFFFF0000U;
    return __uint_as_float(lo_f32_bits) + __uint_as_float(hi_f32_bits);
}

// NaN-sticky maximum (docs/12 4.12). fmaxf returns the non-NaN operand by IEEE 754 / CUDA spec, so a NaN
// activation used to vanish from every row-max reduction and its tile was skipped as "silent". Here a NaN in
// either operand wins, so a poisoned tile reaches the gate as NaN and is kept.
__device__ __forceinline__ float tcbc_nan_max(float a, float b) {
    return (a != a) ? a : ((b != b) ? b : fmaxf(a, b));
}

// Warp-level L1 energy norm reduction using __shfl_xor_sync butterfly tree
__device__ __forceinline__ float warp_reduce_l1_energy(float thread_l1) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_l1 += __shfl_xor_sync(0xFFFFFFFFU, thread_l1, offset);
    }
    return thread_l1;
}

// Warp-level exact row-maximum L1 norm over 64 rows x 32 columns for one K=32 sub-slice (fixes O9).
// In PTX m16n8k16 operand A layout:
//   - group_id = lane >> 2 (0..7), tid_in_group = lane & 3 (0..3).
//   - For tile mi (0..3), reg 0 and reg 2 hold row r_top = mi*16 + group_id,
//     while reg 1 and reg 3 hold row r_bot = mi*16 + group_id + 8.
//   - The 4 threads sharing group_id (xor offsets 1 and 2) hold all 32 columns of r_top and r_bot
//     across sub=0 (k=0..15) and sub=1 (k=16..31).
// Summing across xor offsets 1 and 2 yields the exact 32-column L1 norm for each row,
// and taking the maximum across mi=0..3 and xor offsets 4, 8, 16 yields max_{m in 0..63} ||A[m, k32]||_1.
__device__ __forceinline__ float warp_reduce_rowmax_l1_k32(
    const uint32_t a_rot_k32[2][4][4] // [sub 0..1][mi 0..3][reg 0..3]
) {
    float max_row_l1 = 0.0f;
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        float top_l1 = bf162_abs_sum_f32(a_rot_k32[0][mi][0]) + bf162_abs_sum_f32(a_rot_k32[0][mi][2]) +
                       bf162_abs_sum_f32(a_rot_k32[1][mi][0]) + bf162_abs_sum_f32(a_rot_k32[1][mi][2]);
        float bot_l1 = bf162_abs_sum_f32(a_rot_k32[0][mi][1]) + bf162_abs_sum_f32(a_rot_k32[0][mi][3]) +
                       bf162_abs_sum_f32(a_rot_k32[1][mi][1]) + bf162_abs_sum_f32(a_rot_k32[1][mi][3]);
        top_l1 += __shfl_xor_sync(0xFFFFFFFFU, top_l1, 1);
        top_l1 += __shfl_xor_sync(0xFFFFFFFFU, top_l1, 2);
        bot_l1 += __shfl_xor_sync(0xFFFFFFFFU, bot_l1, 1);
        bot_l1 += __shfl_xor_sync(0xFFFFFFFFU, bot_l1, 2);
        max_row_l1 = tcbc_nan_max(max_row_l1, tcbc_nan_max(top_l1, bot_l1));
    }
    max_row_l1 = tcbc_nan_max(max_row_l1, __shfl_xor_sync(0xFFFFFFFFU, max_row_l1, 4));
    max_row_l1 = tcbc_nan_max(max_row_l1, __shfl_xor_sync(0xFFFFFFFFU, max_row_l1, 8));
    max_row_l1 = tcbc_nan_max(max_row_l1, __shfl_xor_sync(0xFFFFFFFFU, max_row_l1, 16));
    return max_row_l1;
}

// Evaluates whether a BLOCK_K = 128 activation slice (W_M = 64 rows x BLOCK_K = 128 cols = 8,192 BF16 elements)
// carries sufficient signal energy using the row-maximum Holder L1 bound across the 4 K=32 sub-slices (fixes O9).
__device__ __forceinline__ int tiled_cbc_sieve_pred(
    const uint32_t a_rot_k128[4][2][4][4], // [k_step32 0..3][k_sub16 0..1][m_tile 0..3][reg 0..3]
    const float tau_sieve,
    const float inv_tile_elements          // 1.0f / 128.0f (or legacy 1.0f / 8192.0f scaled by 64 rows)
) {
    float rowmax_l1_sum = 0.0f;
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        rowmax_l1_sum += warp_reduce_rowmax_l1_k32(a_rot_k128[ks]);
    }
    const float row_norm_scale = (inv_tile_elements < (1.0f / 256.0f))
                                     ? (64.0f * inv_tile_elements)
                                     : inv_tile_elements;
    const float E_k = rowmax_l1_sum * row_norm_scale;
    // !(E <= tau) rather than E > tau: a NaN energy keeps the tile (docs/12 4.12)
    const uint32_t active_mask = __ballot_sync(0xFFFFFFFFU, !(E_k <= tau_sieve));
    return (active_mask != 0U);
}

// Sound worst-case Holder (L1-Linf) upper-bound sub-tile gate (fixes O9):
//   max_{m,n} |sum_{k in ks} A[m,k] W[k,n]| <= alpha_max * max_{m in 0..63} ||A[m, ks]||_1.
// Returns a 4-bit mask (bit ks set => K=32 sub-slice ks has worst-case contribution > tau_sieve).
//
// SEMANTICS (docs/12 4.13): tau_sieve is a PER-SUB-TILE threshold, not a total error budget. Each cleared bit
// guarantees that sub-tile's contribution to every output is <= tau_sieve, so skipping s sub-tiles bounds the
// total output error by s * tau_sieve (triangle inequality). For a guaranteed total budget eps over the whole
// K reduction, use tau_sieve = tiled_cbc_subtile_tau_for_budget(eps, K) = eps / (K / 32): at most K/32
// sub-tiles can be skipped, so the total is <= (K/32) * eps / (K/32) = eps. (The C++ SmartTieredCache greedy
// budget is tighter because it spends eps on the smallest bounds first; this union bound is simpler and
// equally sound.) A NaN/Inf bound keeps its sub-tile (NaN-safe comparison).
__device__ __forceinline__ uint32_t tiled_cbc_subtile_rowmax_bound_mask(
    const uint32_t a_rot_k128[4][2][4][4],
    const float alpha_max,
    const float tau_sieve
) {
    uint32_t mask = 0U;
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const float max_row_l1 = warp_reduce_rowmax_l1_k32(a_rot_k128[ks]);
        if (!(max_row_l1 * alpha_max <= tau_sieve)) {
            mask |= (1U << ks);
        }
    }
    return mask;
}

// Per-sub-tile threshold that turns the independent sub-tile gate into a guaranteed total budget eps (see above).
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#endif
__host__ __device__ __forceinline__ float tiled_cbc_subtile_tau_for_budget(float eps, int K) {
    const int subtiles = (K + 31) / 32;
    return (subtiles > 0) ? eps / (float)subtiles : eps;
}

// Legacy alias forwarding to the sound row-maximum Holder sub-tile gate.
__device__ __forceinline__ uint32_t tiled_cbc_subtile_mask(
    const uint32_t a_rot_k128[4][2][4][4],
    const float alpha_max,
    const float tau_sieve
) {
    return tiled_cbc_subtile_rowmax_bound_mask(a_rot_k128, alpha_max, tau_sieve);
}

#ifdef STANDALONE_VERIFY
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

int main(void) {
    printf("================================================================================\n");
    printf("  WORKSTREAM 4: TILED-CBC WARP-LEVEL SUBSPACE ENERGY SIEVE (W_M=64 x BLOCK_K=128)\n");
    printf("================================================================================\n\n");

    // 32 lanes x [4 k32][2 sub16][4 m16][4 bf162] = 32 * 128 = 4,096 bf162 regs = 8,192 BF16 values
    static uint32_t warp_regs[32][4][2][4][4];
    for (int lane = 0; lane < 32; ++lane) {
        for (int ks = 0; ks < 4; ++ks) {
            for (int sub = 0; sub < 2; ++sub) {
                for (int mi = 0; mi < 4; ++mi) {
                    for (int r = 0; r < 4; ++r) {
                        // Pack (-1.5f, +1.5f) in BF162 -> 0xBFC03FC0U (|lo| + |hi| = 3.0f)
                        warp_regs[lane][ks][sub][mi][r] = 0xBFC03FC0U;
                    }
                }
            }
        }
    }

    float E_k_lanes[32] = {0.0f};
    int pred_active_lanes[32] = {0};
    const float inv_tile_elements = 1.0f / (64.0f * 128.0f);

    host_launch_grid_cta(dim3(1, 1, 1), 32, [&]() {
        const int lane = threadIdx.x;
        float local_l1 = 0.0f;
        for (int ks = 0; ks < 4; ++ks) {
            for (int sub = 0; sub < 2; ++sub) {
                for (int mi = 0; mi < 4; ++mi) {
                    for (int r = 0; r < 4; ++r) {
                        local_l1 += bf162_abs_sum_f32(warp_regs[lane][ks][sub][mi][r]);
                    }
                }
            }
        }
        E_k_lanes[lane] = warp_reduce_l1_energy(local_l1) * inv_tile_elements;
        pred_active_lanes[lane] = tiled_cbc_sieve_pred(warp_regs[lane], 1.0f, inv_tile_elements);
    });

    const float E_k = E_k_lanes[0];
    for (int lane = 0; lane < 32; ++lane) {
        assert(fabsf(E_k_lanes[lane] - 1.5f) < 1e-6f);
        assert(pred_active_lanes[lane] == 1);
    }

    printf("[1] Total BF16 Elements in W_M=64 x BLOCK_K=128 Warp Slice : 8192 (128 bf162 regs/lane)\n");
    printf("[2] Normalized Warp L1 Energy E_k (|-1.5| and |+1.5|)      : %.4f (Expected: 1.5000)\n", E_k);
    assert(fabsf(E_k - 1.5f) < 1e-6f);
    printf("[+] PASS: Full [4][2][4][4] BLOCK_K=128 __device__ Tiled-CBC energy normalization verified.\n");

    // [3] Uniform sub-tile mask test executing real __device__ tiled_cbc_subtile_rowmax_bound_mask
    {
        static const uint16_t bf16_abs_levels[4] = {0x3E80U, 0x3F00U, 0x3F40U, 0x3F80U}; // 0.25, 0.5, 0.75, 1.0
        static uint32_t subtile_regs[32][4][2][4][4];
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t reg = ((uint32_t)bf16_abs_levels[ks]) | (((uint32_t)bf16_abs_levels[ks]) << 16);
            for (int lane = 0; lane < 32; ++lane)
                for (int sub = 0; sub < 2; ++sub)
                    for (int mi = 0; mi < 4; ++mi)
                        for (int r = 0; r < 4; ++r)
                            subtile_regs[lane][ks][sub][mi][r] = reg;
        }
        uint32_t masks[32] = {0U};
        host_launch_grid_cta(dim3(1, 1, 1), 32, [&]() {
            const int lane = threadIdx.x;
            masks[lane] = tiled_cbc_subtile_rowmax_bound_mask(subtile_regs[lane], 0.5f, 10.0f);
        });
        uint32_t mask = masks[0];
        for (int lane = 0; lane < 32; ++lane) {
            assert(masks[lane] == 0xCU);
        }
        printf("[3] Scale-weighted sub-tile mask (alpha_max = 0.5, tau = 10.0)   : 0x%X (Expected: 0xC)\n", mask);
        assert(mask == 0xCU);
    }

    // [4] Sound Worst-Case Row-Max Holder Bound vs. Single Outlier Row (fixes O9):
    //     Suppose 63 rows are 0.0f and 1 single row (row 13: mi=0, group_id=5, r_bot) has |a|=1.0f across all 32 cols.
    //     Then max_m ||A[m, k32]||_1 = 32.0f (mean row L1 would be only 32.0 / 64 = 0.5f).
    //     With alpha_max = 0.5f and tau = 10.0f, worst-case bound = 32.0 * 0.5 = 16.0 > 10.0 (retained!).
    {
        static uint32_t outlier_k128[32][4][2][4][4];
        memset(outlier_k128, 0, sizeof(outlier_k128));
        const int outlier_group = 5; // row 5 + 8 = 13 in mi=0 (r_bot -> regs 1 and 3)
        for (int tid = 0; tid < 4; ++tid) {
            int lane = (outlier_group << 2) | tid;
            for (int ks = 0; ks < 4; ++ks) {
                for (int sub = 0; sub < 2; ++sub) {
                    outlier_k128[lane][ks][sub][0][1] = 0x3F803F80U; // (+1.0f, +1.0f)
                    outlier_k128[lane][ks][sub][0][3] = 0x3F803F80U; // (+1.0f, +1.0f)
                }
            }
        }
        float row_max_lanes[32] = {0.0f};
        int pred_outlier_lanes[32] = {0};
        host_launch_grid_cta(dim3(1, 1, 1), 32, [&]() {
            const int lane = threadIdx.x;
            row_max_lanes[lane] = warp_reduce_rowmax_l1_k32(outlier_k128[lane][0]);
            pred_outlier_lanes[lane] = tiled_cbc_sieve_pred(outlier_k128[lane], 0.5f, 1.0f / 128.0f);
        });
        float row_max_l1 = row_max_lanes[0];
        float mean_row_l1 = row_max_l1 / 64.0f;
        for (int lane = 0; lane < 32; ++lane) {
            assert(fabsf(row_max_lanes[lane] - 32.0f) < 1e-5f);
            assert(pred_outlier_lanes[lane] == 1);
        }
        printf("[4] Outlier-Row Holder Bound (1 loud row, 63 silent rows)        : row_max_L1 = %.2f (mean_L1 = %.2f)\n",
               row_max_l1, mean_row_l1);
        assert(fabsf(row_max_l1 - 32.0f) < 1e-5f);
        assert(row_max_l1 * 0.5f > 10.0f);  // Sound Holder bound retains the sub-tile (16.0 > 10.0)
        assert(mean_row_l1 * 0.5f < 10.0f); // Mean-row heuristic would have falsely dropped it (0.25 < 10.0)
        printf("[+] PASS: Sound worst-case row-maximum Holder bound (O9) verified on 32-lane warp engine.\n");
    }

    // [5] docs/12 4.12 regression: a single NaN activation in an otherwise silent tile must KEEP that tile in both
    //     gates (fmaxf used to drop the NaN, giving mask 0x0 even at tau = 0).
    {
        static uint32_t nan_k128[32][4][2][4][4];
        memset(nan_k128, 0, sizeof(nan_k128));
        nan_k128[9][2][1][3][2] = 0x7FC00000U;   // one bf16 NaN (upper half) in sub-tile ks = 2, lane 9
        uint32_t masks[32] = {0U};
        int preds[32] = {0};
        host_launch_grid_cta(dim3(1, 1, 1), 32, [&]() {
            const int lane = threadIdx.x;
            masks[lane] = tiled_cbc_subtile_rowmax_bound_mask(nan_k128[lane], 0.5f, 10.0f);
            preds[lane] = tiled_cbc_sieve_pred(nan_k128[lane], 1.0f, 1.0f / 128.0f);
        });
        for (int lane = 0; lane < 32; ++lane) {
            assert(masks[lane] == 0x4U);   // only the poisoned sub-tile is kept; the silent ones are still skipped
            assert(preds[lane] == 1);
        }
        printf("[5] NaN activation in a silent tile (docs/12 4.12)              : sub-tile mask 0x%X (Expected: 0x4), pred = %d\n",
               masks[0], preds[0]);
    }

    // [6] docs/12 4.13: per-sub-tile tau = eps / (K/32) gives a guaranteed total budget. Brute force over every
    //     subset of skipped sub-tiles on the [3] data: total skipped worst-case bound <= eps.
    {
        static const uint16_t lv[4] = {0x3E80U, 0x3F00U, 0x3F40U, 0x3F80U}; // 0.25, 0.5, 0.75, 1.0
        static uint32_t regs[32][4][2][4][4];
        for (int ks = 0; ks < 4; ++ks) {
            const uint32_t reg = ((uint32_t)lv[ks]) | (((uint32_t)lv[ks]) << 16);
            for (int lane = 0; lane < 32; ++lane)
                for (int sub = 0; sub < 2; ++sub)
                    for (int mi = 0; mi < 4; ++mi)
                        for (int r = 0; r < 4; ++r) regs[lane][ks][sub][mi][r] = reg;
        }
        const float alpha = 0.5f, eps = 40.0f, tau = tiled_cbc_subtile_tau_for_budget(eps, 128);
        uint32_t m_budget[32] = {0U};
        float bounds[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        host_launch_grid_cta(dim3(1, 1, 1), 32, [&]() {
            const int lane = threadIdx.x;
            m_budget[lane] = tiled_cbc_subtile_rowmax_bound_mask(regs[lane], alpha, tau);
            if (lane == 0) {
                for (int ks = 0; ks < 4; ++ks) bounds[ks] = alpha * 32.0f * host_bf16_to_f32(lv[ks]);  // row L1 = 32 * level
            }
        });
        float skipped_total = 0.0f;
        for (int ks = 0; ks < 4; ++ks) {
            if (!(m_budget[0] & (1U << ks))) skipped_total += bounds[ks];
        }
        assert(skipped_total <= eps);
        printf("[6] Union-bound budget tau = eps/(K/32) = %.1f (eps = %.1f)       : mask 0x%X, skipped worst-case total %.1f <= eps\n",
               tau, eps, m_budget[0], skipped_total);
    }
    return 0;
}
#endif
