// File: structural_filters/tiled_cbc_sieve.cu
// Workstream 4: Column-Major Subspace Energy Sieve (Tiled-CBC Dynamic Gate)
// Evaluates per-tile L1 activation energy along BLOCK_K = 128 columns during the
// HadaCore prologue via __shfl_xor_sync butterfly reductions and skips cp.async
// DRAM weight fetches + HMMA waves via __ballot_sync when tile energy falls below tau_sieve.
// Build (Host Verification): gcc -O3 -std=c11 -DSTANDALONE_VERIFY -x c structural_filters/tiled_cbc_sieve.cu -o structural_filters/test_tiled_cbc.exe -lm

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#endif
#include <stdint.h>

#ifdef __CUDACC__
// Fast BF162 absolute sum helper: extracts |lo| + |hi| in FP32 from a packed bf162 register
__device__ __forceinline__ float bf162_abs_sum_f32(uint32_t reg_bf162) {
    uint32_t abs_bf162 = reg_bf162 & 0x7FFF7FFFU; // Clear sign bits 15 and 31
    uint32_t lo_f32_bits = (abs_bf162 & 0xFFFFU) << 16;
    uint32_t hi_f32_bits = abs_bf162 & 0xFFFF0000U;
    return __uint_as_float(lo_f32_bits) + __uint_as_float(hi_f32_bits);
}

// Warp-level L1 energy norm reduction using __shfl_xor_sync butterfly tree
__device__ __forceinline__ float warp_reduce_l1_energy(float thread_l1) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_l1 += __shfl_xor_sync(0xFFFFFFFFU, thread_l1, offset);
    }
    return thread_l1;
}

// Evaluates whether a BLOCK_K = 128 activation slice (W_M = 64 rows x BLOCK_K = 128 cols = 8,192 BF16 elements)
// carries sufficient signal energy.
// Geometry per lane (32 lanes/warp):
//   4 steps of K=32 x 2 sub-steps of K=16 x 4 row tiles (m16) x 4 bf162 regs = 128 bf162 regs (256 BF16/lane -> 8,192 BF16/warp).
__device__ __forceinline__ int tiled_cbc_sieve_pred(
    const uint32_t a_rot_k128[4][2][4][4], // [k_step32 0..3][k_sub16 0..1][m_tile 0..3][reg 0..3]
    const float tau_sieve,
    const float inv_tile_elements          // 1.0f / (64.0f * 128.0f) = 1.0f / 8192.0f
) {
    float local_l1 = 0.0f;
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                local_l1 += bf162_abs_sum_f32(a_rot_k128[ks][sub][mi][0]);
                local_l1 += bf162_abs_sum_f32(a_rot_k128[ks][sub][mi][1]);
                local_l1 += bf162_abs_sum_f32(a_rot_k128[ks][sub][mi][2]);
                local_l1 += bf162_abs_sum_f32(a_rot_k128[ks][sub][mi][3]);
            }
        }
    }
    float E_k = warp_reduce_l1_energy(local_l1) * inv_tile_elements;
    uint32_t active_mask = __ballot_sync(0xFFFFFFFFU, E_k > tau_sieve);
    return (active_mask != 0U);
}

// Scale-weighted sub-tile gate (idea: Quest-style upper bounds + PowerInfer hot-tile weighting).
// Returns a 4-bit mask (bit ks set => K=32 sub-slice ks must be computed).
// For a binary/ternary weight tile with group scale alpha, |sum_k a[m,k] w[k,n]| <= alpha * sum_k |a[m,k]|.
// This function uses the MEAN row L1 of the sub-slice (E_k32 * 32) times the tile's largest g128 scale
// alpha_max. That is an average-case estimate, NOT a worst-case bound: a single dominant row can exceed it.
// A sound worst-case gate would take the MAX over rows of the row L1 instead of the mean.
__device__ __forceinline__ uint32_t tiled_cbc_subtile_mask(
    const uint32_t a_rot_k128[4][2][4][4],
    const float alpha_max,
    const float tau_sieve
) {
    uint32_t mask = 0U;
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        float l1 = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                l1 += bf162_abs_sum_f32(a_rot_k128[ks][sub][mi][0]);
                l1 += bf162_abs_sum_f32(a_rot_k128[ks][sub][mi][1]);
                l1 += bf162_abs_sum_f32(a_rot_k128[ks][sub][mi][2]);
                l1 += bf162_abs_sum_f32(a_rot_k128[ks][sub][mi][3]);
            }
        }
        const float e32 = warp_reduce_l1_energy(l1) * (1.0f / 2048.0f); // mean |a| over 64 rows x 32 k
        if (e32 * 32.0f * alpha_max > tau_sieve) {
            mask |= (1U << ks);
        }
    }
    return mask; // warp-uniform: every lane holds the same reduced value
}
#endif

#ifdef STANDALONE_VERIFY
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static inline float host_bf162_abs_sum_f32(uint32_t reg_bf162) {
    uint32_t abs_bf162 = reg_bf162 & 0x7FFF7FFFU;
    uint32_t lo_f32_bits = (abs_bf162 & 0xFFFFU) << 16;
    uint32_t hi_f32_bits = abs_bf162 & 0xFFFF0000U;
    float lo, hi;
    memcpy(&lo, &lo_f32_bits, sizeof(float));
    memcpy(&hi, &hi_f32_bits, sizeof(float));
    return lo + hi;
}

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

    float warp_sum = 0.0f;
    for (int lane = 0; lane < 32; ++lane) {
        float local_l1 = 0.0f;
        for (int ks = 0; ks < 4; ++ks) {
            for (int sub = 0; sub < 2; ++sub) {
                for (int mi = 0; mi < 4; ++mi) {
                    for (int r = 0; r < 4; ++r) {
                        local_l1 += host_bf162_abs_sum_f32(warp_regs[lane][ks][sub][mi][r]);
                    }
                }
            }
        }
        warp_sum += local_l1;
    }
    const float inv_tile_elements = 1.0f / (64.0f * 128.0f);
    const float E_k = warp_sum * inv_tile_elements;

    printf("[1] Total BF16 Elements in W_M=64 x BLOCK_K=128 Warp Slice : 8192 (128 bf162 regs/lane)\n");
    printf("[2] Normalized Warp L1 Energy E_k (|-1.5| and |+1.5|)      : %.4f (Expected: 1.5000)\n", E_k);
    assert(fabsf(E_k - 1.5f) < 1e-6f);
    printf("[+] PASS: Full [4][2][4][4] BLOCK_K=128 Tiled-CBC energy normalization verified.\n");

    // Sub-tile mask: sub-slice ks carries |a| = 0.25 * (ks + 1) on every element, so mean row L1 over K=32
    // is 32 * 0.25 * (ks + 1) = 8 * (ks + 1). With alpha_max = 0.5 and tau = 10, the bound is 4 * (ks + 1):
    // active for ks = 2 (12) and ks = 3 (16); inactive for ks = 0 (4) and ks = 1 (8). Expected mask = 0b1100.
    {
        static const uint16_t bf16_abs_levels[4] = {0x3E80U, 0x3F00U, 0x3F40U, 0x3F80U}; // 0.25, 0.5, 0.75, 1.0
        float sub_energy[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t reg = ((uint32_t)bf16_abs_levels[ks]) | (((uint32_t)bf16_abs_levels[ks]) << 16);
            float per_lane = 0.0f;
            for (int sub = 0; sub < 2; ++sub)
                for (int mi = 0; mi < 4; ++mi)
                    for (int r = 0; r < 4; ++r)
                        per_lane += host_bf162_abs_sum_f32(reg);
            sub_energy[ks] = per_lane * 32.0f * (1.0f / 2048.0f); // all 32 lanes identical
        }
        uint32_t mask = 0U;
        for (int ks = 0; ks < 4; ++ks)
            if (sub_energy[ks] * 32.0f * 0.5f > 10.0f) mask |= (1U << ks);
        printf("[3] Scale-weighted sub-tile mask (alpha_max = 0.5, tau = 10.0)   : 0x%X (Expected: 0xC)\n", mask);
        assert(mask == 0xCU);
        printf("[+] PASS: sub-tile mask logic verified on host (device version needs nvcc, not run here).\n");
    }
    return 0;
}
#endif
