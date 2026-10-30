// File: fused_w1_hadacore/w1a16_hadacore_kernel.cu
// Complete Fused Online HadaCore (H_16 & Kronecker H_32 = H_2 x H_16) + lop3.b32
// W1A16_F32 & W2A16_F32 CUDA/PTX Kernels with Marlin Ping-Pong Staging & Stream-K Decode
// Targets: NVIDIA Ampere GA102 (sm_86) and Ada Lovelace AD102 (sm_89) with Unlocked Tensor Cores

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ==============================================================================
// 1. Asynchronous Global-to-Shared Copy & Shared-to-Register ldmatrix Primitives
// ==============================================================================

// L1-bypassing (.cg) async copy for streaming weights (prevents L1/L2 activation eviction)
__device__ __forceinline__ void cp_async_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

// Cache-all (.ca) async copy for L2-persistent activations A (Marlin L2 persistence strategy)
__device__ __forceinline__ void cp_async_ca_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit(void) {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N_WAIT>
__device__ __forceinline__ void cp_async_wait_group(void) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT));
}

__device__ __forceinline__ void ldmatrix_x4_bf16(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

// Conflict-free 128-Byte Shared Memory XOR Swizzle (Swizzle<3, 3, 3>):
// Permutes the eight 16-byte chunks per 128B line by (row & 7) so all 8 threads
// in each ldmatrix.m8n8 sub-matrix access 8 distinct 16B segments (32 banks, beta_conflict = 1.0).
__device__ __forceinline__ uint32_t swizzle_128b_smem_offset(uint32_t row, uint32_t col_bf16) {
    uint32_t chunk_16b = (col_bf16 >> 3) ^ (row & 7U);
    return (row << 7) + (chunk_16b << 4) + ((col_bf16 & 7U) << 1);
}

__device__ __forceinline__ uint32_t cvt_f32x2_to_bf162(float f0, float f1) {
    uint32_t out;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(out) : "f"(f0), "f"(f1));
    return out;
}

// ==============================================================================
// 2. Branchless Register Bit-Unpackers (1-Bit Binary g128 & 2-Bit Ternary b1.58)
// ==============================================================================

// Branchless 1-Bit (Binary g128) unpack via lop3.b32 (immLut = 0xAE)
// w_interleaved: 32-bit word where even bits k in [0..15] sit at bit k, odd bits at bit k+16.
// shift_p0 = 15 - p0, shift_p1 = 15 - p1
__device__ __forceinline__ void unpack_w1_b_frag(
    uint32_t& rb0,
    uint32_t& rb1,
    const uint32_t w_interleaved,
    const uint32_t shift_p0,
    const uint32_t shift_p1,
    const uint32_t scale_bf162
) {
    uint32_t tmp0, tmp1;
    asm volatile(
        "{\n"
        "  .reg .b32 nz;\n"
        "  mov.b32       nz, 0x80008000;\n"
        "  shl.b32       %0, %4, %5;\n"
        "  lop3.b32      %0, %0, 0x80008000, 0x3F803F80, 0xAE;\n"
        "  fma.rn.bf16x2 %2, %0, %7, nz;\n"
        "  shl.b32       %1, %4, %6;\n"
        "  lop3.b32      %1, %1, 0x80008000, 0x3F803F80, 0xAE;\n"
        "  fma.rn.bf16x2 %3, %1, %7, nz;\n"
        "}\n"
        : "=&r"(tmp0), "=&r"(tmp1), "=r"(rb0), "=r"(rb1)
        : "r"(w_interleaved), "r"(shift_p0), "r"(shift_p1), "r"(scale_bf162)
    );
}

// Branchless 2-Bit Ternary (BitNet b1.58) unpack via lop3.b32 (immLut = 0xEA) + fma.rn.bf162 + mul.rn.bf162
// Uses unscaled exact BF162 constants c_mul = 0x42804280 (64.0, 64.0) and c_sub = 0xC301C301 (-129.0, -129.0)
// to produce exact unit trits {-1.0, +0.0, +1.0} [0xBF80, 0x0000, 0x3F80], followed by mul.rn.bf162 with
// scale_bf162. Prevents 8-bit BF16 significand overflow on (-129.0 * alpha_g) and guarantees q=1 -> +0.0f.
__device__ __forceinline__ void unpack_w2_ternary_pair(
    uint32_t& rb_out,
    const uint32_t w_tern_interleaved,
    const uint32_t shift_2k,
    const uint32_t scale_bf162
) {
    uint32_t tmp;
    asm volatile(
        "{\n"
        "  .reg .b32 cm, cs, nz;\n"
        "  mov.b32       cm, 0x42804280;\n"
        "  mov.b32       cs, 0xC301C301;\n"
        "  mov.b32       nz, 0x80008000;\n"
        "  shr.u32       %0, %2, %3;\n"
        "  lop3.b32      %0, %0, 0x00030003, 0x40004000, 0xEA;\n"
        "  fma.rn.bf16x2 %0, %0, cm, cs;\n"
        "  fma.rn.bf16x2 %1, %0, %4, nz;\n"
        "}\n"
        : "=&r"(tmp), "=r"(rb_out)
        : "r"(w_tern_interleaved), "r"(shift_2k), "r"(scale_bf162)
    );
}

// ==============================================================================
// 3. Unlocked HMMA.16816.F32 Tensor Core Instruction & HadaCore Register Prologue
// ==============================================================================

__device__ __forceinline__ void mma_m16n8k16_bf16_f32(
    float& d0, float& d1, float& d2, float& d3,
    const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// Precomputes loop-invariant Sylvester Hadamard H_16 B-fragment signs once per kernel launch
// so hadacore_rotate_m16k16_in_regs executes zero integer ALU setup inside the K-loop.
__device__ __forceinline__ void precompute_hadacore_h16_b_frag(
    uint32_t& h_b0_lo,
    uint32_t& h_b0_hi,
    uint32_t& h_b1_lo,
    uint32_t& h_b1_hi,
    const int lane_id
) {
    const int col_n0 = (lane_id >> 2);
    const int col_n1 = col_n0 + 8;
    const int t = lane_id & 3;

    const int r0 = 2 * t;
    const int r1 = 2 * t + 1;
    const int r2 = 2 * t + 8;
    const int r3 = 2 * t + 9;

    const uint16_t s00 = (__popc(r0 & col_n0) & 1) ? 0xBE80U : 0x3E80U;
    const uint16_t s01 = (__popc(r1 & col_n0) & 1) ? 0xBE80U : 0x3E80U;
    const uint16_t s02 = (__popc(r2 & col_n0) & 1) ? 0xBE80U : 0x3E80U;
    const uint16_t s03 = (__popc(r3 & col_n0) & 1) ? 0xBE80U : 0x3E80U;
    h_b0_lo = ((uint32_t)s00) | (((uint32_t)s01) << 16);
    h_b0_hi = ((uint32_t)s02) | (((uint32_t)s03) << 16);

    const uint16_t s10 = (__popc(r0 & col_n1) & 1) ? 0xBE80U : 0x3E80U;
    const uint16_t s11 = (__popc(r1 & col_n1) & 1) ? 0xBE80U : 0x3E80U;
    const uint16_t s12 = (__popc(r2 & col_n1) & 1) ? 0xBE80U : 0x3E80U;
    const uint16_t s13 = (__popc(r3 & col_n1) & 1) ? 0xBE80U : 0x3E80U;
    h_b1_lo = ((uint32_t)s10) | (((uint32_t)s11) << 16);
    h_b1_hi = ((uint32_t)s12) | (((uint32_t)s13) << 16);
}

// Fused Online HadaCore 16x16 Tensor Core Walsh-Hadamard Transform Prologue:
// Multiplies an incoming 16x16 BF16 activation fragment X by the normalized 16x16
// Hadamard butterfly matrix H_16 in {-0.25, +0.25}^{16x16} directly in registers
// via two unlocked mma.sync.aligned.m16n8k16 instructions (16 SMSP cycles total).
__device__ __forceinline__ void hadacore_rotate_m16k16_in_regs(
    uint32_t a_rot[4],
    const uint32_t a_in[4],
    const uint32_t h_b0_lo,
    const uint32_t h_b0_hi,
    const uint32_t h_b1_lo,
    const uint32_t h_b1_hi
) {
    float d_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    mma_m16n8k16_bf16_f32(d_lo[0], d_lo[1], d_lo[2], d_lo[3],
                          a_in[0], a_in[1], a_in[2], a_in[3], h_b0_lo, h_b0_hi);
    mma_m16n8k16_bf16_f32(d_hi[0], d_hi[1], d_hi[2], d_hi[3],
                          a_in[0], a_in[1], a_in[2], a_in[3], h_b1_lo, h_b1_hi);

    a_rot[0] = cvt_f32x2_to_bf162(d_lo[0], d_lo[1]);
    a_rot[1] = cvt_f32x2_to_bf162(d_lo[2], d_lo[3]);
    a_rot[2] = cvt_f32x2_to_bf162(d_hi[0], d_hi[1]);
    a_rot[3] = cvt_f32x2_to_bf162(d_hi[2], d_hi[3]);
}

// Fused 2-Level Kronecker-Hadamard Transform (H_32 = H_2 x H_16) in Registers:
// Uses the Kronecker mixed-product identity Mat_{2x16}(x (H_2 x H_16)) = H_2 * Mat_{2x16}(x) * H_16.
// Pass 1: Right-multiplies both K=16 sub-tiles by H_16 via Tensor Core mma.sync.aligned.m16n8k16.
// Pass 2: Left-multiplies across the two sub-tiles by H_2 via in-register bf162 FMA butterflies.
__device__ __forceinline__ void hadacore_kronecker_h32_m16k32_in_regs(
    uint32_t a_rot_sub0[4],
    uint32_t a_rot_sub1[4],
    const uint32_t a_in_sub0[4],
    const uint32_t a_in_sub1[4],
    const uint32_t h_b0_lo,
    const uint32_t h_b0_hi,
    const uint32_t h_b1_lo,
    const uint32_t h_b1_hi
) {
    uint32_t t0[4], t1[4];
    hadacore_rotate_m16k16_in_regs(t0, a_in_sub0, h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi);
    hadacore_rotate_m16k16_in_regs(t1, a_in_sub1, h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi);

    // Normalized H_2 butterfly factor 1/sqrt(2) ~= 0.70703125 in BF16 (0x3F35). Every output is
    // early-clobber ("=&r") so the allocator cannot alias an output with an input that a later
    // instruction in the same asm block still reads.
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        uint32_t s0;
        asm volatile(
            "{\n"
            "  .reg .b32 nz, cp, cn;\n"
            "  mov.b32       nz, 0x80008000;\n"
            "  mov.b32       cp, 0x3F353F35;\n"
            "  mov.b32       cn, 0xBF35BF35;\n"
            "  fma.rn.bf16x2 %0, %3, cp, nz;\n"
            "  fma.rn.bf16x2 %1, %4, cp, %0;\n"
            "  fma.rn.bf16x2 %2, %4, cn, %0;\n"
            "}\n"
            : "=&r"(s0), "=&r"(a_rot_sub0[r]), "=&r"(a_rot_sub1[r])
            : "r"(t0[r]), "r"(t1[r])
        );
    }
}

// In-register Tiled-CBC Warp L1 Energy Gate for a K=32 slice (W_M=64 x K=32 = 2,048 BF16 elements).
// Sign bits are cleared, 4 registers are pairwise summed in bf16x2 (3 roundings, bounded relative error
// ~2^-7), then widened and accumulated in FP32 so rounding error does not compound across 32 terms.
// NOTE: the L1 norm is NOT invariant under the Hadamard rotation (only L2 is), so tau_sieve must be
// calibrated on post-rotation statistics and this gate evaluated on the post-rotation fragments.
// add.bf16x2 needs sm_90, so addition is fma(x, 1.0, y), valid on sm_80+.
__device__ __forceinline__ int warp_sieve_k32_active(
    const uint32_t a_frag[2][4][4],
    const float tau_sieve
) {
    if (tau_sieve <= 0.0f) {
        return 1;
    }
    float thread_l1 = 0.0f;
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            const uint32_t abs0 = a_frag[sub][mi][0] & 0x7FFF7FFFU;
            const uint32_t abs1 = a_frag[sub][mi][1] & 0x7FFF7FFFU;
            const uint32_t abs2 = a_frag[sub][mi][2] & 0x7FFF7FFFU;
            const uint32_t abs3 = a_frag[sub][mi][3] & 0x7FFF7FFFU;
            uint32_t p01, p23, p;
            asm volatile(
                "{\n"
                "  .reg .b32 one;\n"
                "  mov.b32       one, 0x3F803F80;\n"
                "  fma.rn.bf16x2 %0, %3, one, %4;\n"
                "  fma.rn.bf16x2 %1, %5, one, %6;\n"
                "  fma.rn.bf16x2 %2, %0, one, %1;\n"
                "}\n"
                : "=&r"(p01), "=&r"(p23), "=&r"(p)
                : "r"(abs0), "r"(abs1), "r"(abs2), "r"(abs3)
            );
            thread_l1 += __uint_as_float((p & 0xFFFFU) << 16) + __uint_as_float(p & 0xFFFF0000U);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_l1 += __shfl_xor_sync(0xFFFFFFFFU, thread_l1, offset);
    }
    const float e_k32 = thread_l1 * (1.0f / 2048.0f);
    return (e_k32 > tau_sieve) ? 1 : 0; // warp-uniform after the xor-butterfly reduction
}

// ==============================================================================
// 4. Complete Warp-Tile K=32 Inner Loops (W1A16 Binary & W2A16 Ternary)
// ==============================================================================

// Warp tile: WM = 64 (4 m16 slices), WN = 64 (8 n8 slices) -> 32 accumulators (128 floats/thread)
__device__ __forceinline__ void warp_step_k32_w1a16(
    float acc[4][8][4],
    const uint32_t a_frag[2][4][4],     // [k_sub_step 0..1][m_tile 0..3][reg 0..3]
    const uint32_t w_packed_col[8],     // 1 uint32 (32 1-bit weights) per n_tile (0..7)
    const uint32_t scale_col_bf162[8],  // Broadcasted g128 scale (alpha, alpha) per n_tile (0..7)
    const int lane_id
) {
    const int t = lane_id & 3; // 0..3

    // Sub-step 0 (K = 0..15): pair indices p0 = t, p1 = t + 4
    const uint32_t sh_0a = 15 - t;
    const uint32_t sh_0b = 11 - t; // 15 - (t + 4)

    #pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        uint32_t rb0, rb1;
        unpack_w1_b_frag(rb0, rb1, w_packed_col[ni], sh_0a, sh_0b, scale_col_bf162[ni]);
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            mma_m16n8k16_bf16_f32(
                acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                a_frag[0][mi][0], a_frag[0][mi][1], a_frag[0][mi][2], a_frag[0][mi][3],
                rb0, rb1
            );
        }
    }

    // Sub-step 1 (K = 16..31): pair indices p2 = t + 8, p3 = t + 12
    const uint32_t sh_1a = 7 - t; // 15 - (t + 8)
    const uint32_t sh_1b = 3 - t; // 15 - (t + 12)

    #pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        uint32_t rb0, rb1;
        unpack_w1_b_frag(rb0, rb1, w_packed_col[ni], sh_1a, sh_1b, scale_col_bf162[ni]);
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            mma_m16n8k16_bf16_f32(
                acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                a_frag[1][mi][0], a_frag[1][mi][1], a_frag[1][mi][2], a_frag[1][mi][3],
                rb0, rb1
            );
        }
    }
}

// Warp-Tile K=32 Inner Loop for 2-Bit Ternary (BitNet b1.58 W2A16_F32):
// Each K=32 step consumes two uint32 words per N-column (16 ternary weights per uint32).
__device__ __forceinline__ void warp_step_k32_w2a16(
    float acc[4][8][4],
    const uint32_t a_frag[2][4][4],        // [k_sub_step 0..1][m_tile 0..3][reg 0..3]
    const uint32_t w2_packed_col[8][2],    // 2 uint32 words (16 trits each) per n_tile (0..7)
    const uint32_t scale_col_bf162[8],     // Broadcasted g128 scale (alpha, alpha) per n_tile (0..7)
    const int lane_id
) {
    const int t = lane_id & 3;
    const uint32_t sh_p0 = (uint32_t)(2 * t);
    const uint32_t sh_p1 = (uint32_t)(2 * (t + 4));

    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            uint32_t rb0, rb1;
            unpack_w2_ternary_pair(rb0, w2_packed_col[ni][sub], sh_p0, scale_col_bf162[ni]);
            unpack_w2_ternary_pair(rb1, w2_packed_col[ni][sub], sh_p1, scale_col_bf162[ni]);
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                mma_m16n8k16_bf16_f32(
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                    a_frag[sub][mi][0], a_frag[sub][mi][1], a_frag[sub][mi][2], a_frag[sub][mi][3],
                    rb0, rb1
                );
            }
        }
    }
}

// ==============================================================================
// 5. Global Fused HadaCore + W1A16_F32 & W2A16_F32 Kernel Entry Points
//    (Double-Buffered cp.async.ca Ping-Pong Staging + In-Register Sieve)
// ==============================================================================

__global__ void __launch_bounds__(256, 1)
fused_w1a16_hadacore_gemm_kernel(
    const uint32_t* __restrict__ a_bf162,       // [M, K/2] packed bf162 activations
    const uint32_t* __restrict__ w_packed_u32,  // [N, K/32] Interleaved 16-bit split 1-bit weights
    const uint16_t* __restrict__ scales_bf16,   // [N, K/128] g128 BF16 scales
    float* __restrict__ c_fp32,                 // [M, N] output accumulator
    const int M,
    const int N,
    const int K,
    const int enable_online_hadacore,           // 0 = off, 1 = H_16, 2 = Kronecker H_32 = H_2 x H_16
    const float tau_sieve = 0.0f
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warp_m_idx = warp_id >> 2; // 0..1 (BM = 128 = 2 * 64)
    const int warp_n_idx = warp_id & 3;  // 0..3 (BN = 256 = 4 * 64)

    const int block_row = blockIdx.y * 128 + warp_m_idx * 64;
    const int block_col = blockIdx.x * 256 + warp_n_idx * 64;

    float acc[4][8][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            acc[mi][ni][0] = 0.0f;
            acc[mi][ni][1] = 0.0f;
            acc[mi][ni][2] = 0.0f;
            acc[mi][ni][3] = 0.0f;
        }
    }

    const int group_id = lane_id >> 2;
    const int t = lane_id & 3;

    // Hoist loop-invariant Sylvester Hadamard H_16 B-fragment signs outside the K loop
    uint32_t h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi;
    precompute_hadacore_h16_b_frag(h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi, lane_id);

    // Double-buffered (2-stage ping-pong) conflict-free Swizzle<3,3,3> activation tile (2 x 16 KB = 32 KB)
    __shared__ __align__(128) uint8_t smem_a_tile[2][128 * 128];

    const int load_row = threadIdx.x >> 1;            // 0..127 rows (2 threads per 32-BF16 row)
    const int load_col_bf16 = (threadIdx.x & 1) << 3; // 8 BF16 elements (16B) per cp.async
    const int g_row = blockIdx.y * 128 + load_row;
    const int num_k_tiles = (K + 31) >> 5;

    // Prime stage 0 via L2-persistent cp.async.ca
    if (num_k_tiles > 0) {
        const uint32_t smem_stage0 = __cvta_generic_to_shared(smem_a_tile[0]);
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const int g_col_bf16 = sub * 16 + load_col_bf16;
            const uint32_t s_off = swizzle_128b_smem_offset(load_row, sub * 16 + load_col_bf16);
            if (g_row < M && (g_col_bf16 + 7) < K) {
                cp_async_ca_16B(smem_stage0 + s_off, &a_bf162[g_row * (K >> 1) + (g_col_bf16 >> 1)]);
            }
        }
        cp_async_commit();
    }

    for (int tile_idx = 0; tile_idx < num_k_tiles; ++tile_idx) {
        const int k_base = tile_idx << 5;
        const int cur_stage = tile_idx & 1;
        const int nxt_stage = cur_stage ^ 1;

        // Prefetch next K=32 stage into ping-pong buffer while current stage is processed
        if (tile_idx + 1 < num_k_tiles) {
            const int next_k_base = (tile_idx + 1) << 5;
            const uint32_t smem_nxt = __cvta_generic_to_shared(smem_a_tile[nxt_stage]);
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int g_col_bf16 = next_k_base + sub * 16 + load_col_bf16;
                const uint32_t s_off = swizzle_128b_smem_offset(load_row, sub * 16 + load_col_bf16);
                if (g_row < M && (g_col_bf16 + 7) < K) {
                    cp_async_ca_16B(smem_nxt + s_off, &a_bf162[g_row * (K >> 1) + (g_col_bf16 >> 1)]);
                }
            }
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        const uint32_t smem_cur = __cvta_generic_to_shared(smem_a_tile[cur_stage]);
        uint32_t raw_a[2][4][4];
        uint32_t a_frag[2][4][4];

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k_sub = k_base + sub * 16;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r_top = block_row + mi * 16 + group_id;
                int r_bot = r_top + 8;
                raw_a[sub][mi][0] = 0U;
                raw_a[sub][mi][1] = 0U;
                raw_a[sub][mi][2] = 0U;
                raw_a[sub][mi][3] = 0U;
                // Guard MUST be warp-uniform: ldmatrix.sync.aligned is a collective over all 32 lanes.
                const bool tile_in_bounds = (block_row + mi * 16 + 15) < M;
                if (tile_in_bounds && ((k_sub + 15) < K)) {
                    const int s_row = warp_m_idx * 64 + mi * 16 + (lane_id & 15);
                    const int s_col_bf16 = sub * 16 + ((lane_id >> 4) << 3);
                    const uint32_t ldm_off = swizzle_128b_smem_offset(s_row, s_col_bf16);
                    ldmatrix_x4_bf16(
                        raw_a[sub][mi][0], raw_a[sub][mi][1],
                        raw_a[sub][mi][2], raw_a[sub][mi][3],
                        smem_cur + ldm_off
                    );
                } else {
                    if (r_top < M) {
                        if ((k_sub + 2 * t + 1) < K) {
                            raw_a[sub][mi][0] = a_bf162[r_top * (K >> 1) + (k_sub >> 1) + t];
                        }
                        if ((k_sub + 2 * t + 9) < K) {
                            raw_a[sub][mi][2] = a_bf162[r_top * (K >> 1) + (k_sub >> 1) + t + 4];
                        }
                    }
                    if (r_bot < M) {
                        if ((k_sub + 2 * t + 1) < K) {
                            raw_a[sub][mi][1] = a_bf162[r_bot * (K >> 1) + (k_sub >> 1) + t];
                        }
                        if ((k_sub + 2 * t + 9) < K) {
                            raw_a[sub][mi][3] = a_bf162[r_bot * (K >> 1) + (k_sub >> 1) + t + 4];
                        }
                    }
                }
            }
        }

        // Apply online HadaCore rotation (Mode 1: Block H_16, Mode 2: Kronecker H_32 = H_2 x H_16)
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            if (enable_online_hadacore >= 2) {
                hadacore_kronecker_h32_m16k32_in_regs(
                    a_frag[0][mi], a_frag[1][mi],
                    raw_a[0][mi], raw_a[1][mi],
                    h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi
                );
            } else if (enable_online_hadacore == 1) {
                hadacore_rotate_m16k16_in_regs(a_frag[0][mi], raw_a[0][mi], h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi);
                hadacore_rotate_m16k16_in_regs(a_frag[1][mi], raw_a[1][mi], h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi);
            } else {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    a_frag[sub][mi][0] = raw_a[sub][mi][0];
                    a_frag[sub][mi][1] = raw_a[sub][mi][1];
                    a_frag[sub][mi][2] = raw_a[sub][mi][2];
                    a_frag[sub][mi][3] = raw_a[sub][mi][3];
                }
            }
        }
        __syncthreads();

        // Fused Tiled-CBC Energy Sieve: skip global weight loads + 16 HMMAs when K=32 slice is cold
        if (!warp_sieve_k32_active(a_frag, tau_sieve)) {
            continue;
        }

        uint32_t w_packed_col[8];
        uint32_t scale_col_bf162[8];
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            int col = block_col + ni * 8 + group_id;
            if (col < N) {
                w_packed_col[ni] = w_packed_u32[col * (K >> 5) + (k_base >> 5)];
                uint16_t sc = scales_bf16[col * (K >> 7) + (k_base >> 7)];
                scale_col_bf162[ni] = ((uint32_t)sc) | (((uint32_t)sc) << 16);
            } else {
                w_packed_col[ni] = 0U;
                scale_col_bf162[ni] = 0U;
            }
        }

        warp_step_k32_w1a16(acc, a_frag, w_packed_col, scale_col_bf162, lane_id);
    }

    // Write back FP32 accumulators to global output C
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        int r0 = block_row + mi * 16 + group_id;
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            int c0 = block_col + ni * 8 + 2 * t;
            int c1 = c0 + 1;
            if (r0 < M && c0 < N) c_fp32[r0 * N + c0] = acc[mi][ni][0];
            if (r0 < M && c1 < N) c_fp32[r0 * N + c1] = acc[mi][ni][1];
            if (r1 < M && c0 < N) c_fp32[r1 * N + c0] = acc[mi][ni][2];
            if (r1 < M && c1 < N) c_fp32[r1 * N + c1] = acc[mi][ni][3];
        }
    }
}

// Global Fused HadaCore + W2A16_F32 (1.58-Bit Ternary) Kernel Entry Point
__global__ void __launch_bounds__(256, 1)
fused_w2a16_hadacore_gemm_kernel(
    const uint32_t* __restrict__ a_bf162,        // [M, K/2] packed bf162 activations
    const uint32_t* __restrict__ w2_packed_u32,  // [N, K/16] Interleaved 16-bit split 2-bit ternary weights
    const uint16_t* __restrict__ scales_bf16,    // [N, K/128] g128 BF16 scales
    float* __restrict__ c_fp32,                  // [M, N] output accumulator
    const int M,
    const int N,
    const int K,
    const int enable_online_hadacore,
    const float tau_sieve = 0.0f
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warp_m_idx = warp_id >> 2;
    const int warp_n_idx = warp_id & 3;

    const int block_row = blockIdx.y * 128 + warp_m_idx * 64;
    const int block_col = blockIdx.x * 256 + warp_n_idx * 64;

    float acc[4][8][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            acc[mi][ni][0] = 0.0f;
            acc[mi][ni][1] = 0.0f;
            acc[mi][ni][2] = 0.0f;
            acc[mi][ni][3] = 0.0f;
        }
    }

    const int group_id = lane_id >> 2;
    const int t = lane_id & 3;

    uint32_t h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi;
    precompute_hadacore_h16_b_frag(h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi, lane_id);

    for (int k_base = 0; k_base < K; k_base += 32) {
        uint32_t raw_a[2][4][4];
        uint32_t a_frag[2][4][4];
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k_sub = k_base + sub * 16;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r_top = block_row + mi * 16 + group_id;
                int r_bot = r_top + 8;
                raw_a[sub][mi][0] = (r_top < M && (k_sub + 2 * t + 1) < K) ? a_bf162[r_top * (K >> 1) + (k_sub >> 1) + t] : 0U;
                raw_a[sub][mi][1] = (r_bot < M && (k_sub + 2 * t + 1) < K) ? a_bf162[r_bot * (K >> 1) + (k_sub >> 1) + t] : 0U;
                raw_a[sub][mi][2] = (r_top < M && (k_sub + 2 * t + 9) < K) ? a_bf162[r_top * (K >> 1) + (k_sub >> 1) + t + 4] : 0U;
                raw_a[sub][mi][3] = (r_bot < M && (k_sub + 2 * t + 9) < K) ? a_bf162[r_bot * (K >> 1) + (k_sub >> 1) + t + 4] : 0U;
            }
        }

        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            if (enable_online_hadacore >= 2) {
                hadacore_kronecker_h32_m16k32_in_regs(
                    a_frag[0][mi], a_frag[1][mi], raw_a[0][mi], raw_a[1][mi],
                    h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi
                );
            } else if (enable_online_hadacore == 1) {
                hadacore_rotate_m16k16_in_regs(a_frag[0][mi], raw_a[0][mi], h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi);
                hadacore_rotate_m16k16_in_regs(a_frag[1][mi], raw_a[1][mi], h_b0_lo, h_b0_hi, h_b1_lo, h_b1_hi);
            } else {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    a_frag[sub][mi][0] = raw_a[sub][mi][0];
                    a_frag[sub][mi][1] = raw_a[sub][mi][1];
                    a_frag[sub][mi][2] = raw_a[sub][mi][2];
                    a_frag[sub][mi][3] = raw_a[sub][mi][3];
                }
            }
        }

        if (!warp_sieve_k32_active(a_frag, tau_sieve)) {
            continue;
        }

        uint32_t w2_packed_col[8][2];
        uint32_t scale_col_bf162[8];
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            int col = block_col + ni * 8 + group_id;
            if (col < N) {
                w2_packed_col[ni][0] = w2_packed_u32[col * (K >> 4) + (k_base >> 4)];
                w2_packed_col[ni][1] = w2_packed_u32[col * (K >> 4) + (k_base >> 4) + 1];
                uint16_t sc = scales_bf16[col * (K >> 7) + (k_base >> 7)];
                scale_col_bf162[ni] = ((uint32_t)sc) | (((uint32_t)sc) << 16);
            } else {
                w2_packed_col[ni][0] = 0x55555555U; // Ternary zero code q=1 across all 16 trits
                w2_packed_col[ni][1] = 0x55555555U;
                scale_col_bf162[ni] = 0U;
            }
        }

        warp_step_k32_w2a16(acc, a_frag, w2_packed_col, scale_col_bf162, lane_id);
    }

    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        int r0 = block_row + mi * 16 + group_id;
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            int c0 = block_col + ni * 8 + 2 * t;
            int c1 = c0 + 1;
            if (r0 < M && c0 < N) c_fp32[r0 * N + c0] = acc[mi][ni][0];
            if (r0 < M && c1 < N) c_fp32[r0 * N + c1] = acc[mi][ni][1];
            if (r1 < M && c0 < N) c_fp32[r1 * N + c0] = acc[mi][ni][2];
            if (r1 < M && c1 < N) c_fp32[r1 * N + c1] = acc[mi][ni][3];
        }
    }
}

// ==============================================================================
// 6. Marlin-Style Small-Batch (M <= 16, BM = 16) Split-K / Stream-K Decode Kernel
//    Eliminates SM idle wave quantization at M = 1..16 by partitioning (N_tiles x K_splits)
//    across all SMs and accumulating in L2 cache via atomicAdd.
// ==============================================================================

__global__ void __launch_bounds__(128, 2)
marlin_streamk_w1a16_decode_kernel(
    const uint32_t* __restrict__ a_bf162,       // [M, K/2] with M <= 16
    const uint32_t* __restrict__ w_packed_u32,  // [N, K/32]
    const uint16_t* __restrict__ scales_bf16,   // [N, K/128]
    float* __restrict__ c_fp32,                 // [M, N] zero-initialized output in L2
    const int M,
    const int N,
    const int K,
    const int split_k_slices
) {
    const int warp_id = threadIdx.x >> 5; // 0..3 (4 warps per CTA, each covering WN=16 -> BN=64)
    const int lane_id = threadIdx.x & 31;
    const int group_id = lane_id >> 2;
    const int t = lane_id & 3;

    const int n_tile_idx = blockIdx.x;
    const int k_split_idx = blockIdx.y;
    const int col_base = n_tile_idx * 64 + warp_id * 16;

    const int k_tiles_total = K >> 5;
    const int k_tiles_per_split = (k_tiles_total + split_k_slices - 1) / split_k_slices;
    const int k_tile_start = k_split_idx * k_tiles_per_split;
    const int k_tile_end = (k_tile_start + k_tiles_per_split < k_tiles_total)
                               ? (k_tile_start + k_tiles_per_split)
                               : k_tiles_total;

    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    const int r_top = group_id;
    const int r_bot = group_id + 8;

    for (int kt = k_tile_start; kt < k_tile_end; ++kt) {
        const int k_base = kt << 5;
        uint32_t a_sub[2][4];
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k_sub = k_base + sub * 16;
            a_sub[sub][0] = (r_top < M && (k_sub + 2 * t + 1) < K) ? a_bf162[r_top * (K >> 1) + (k_sub >> 1) + t] : 0U;
            a_sub[sub][1] = (r_bot < M && (k_sub + 2 * t + 1) < K) ? a_bf162[r_bot * (K >> 1) + (k_sub >> 1) + t] : 0U;
            a_sub[sub][2] = (r_top < M && (k_sub + 2 * t + 9) < K) ? a_bf162[r_top * (K >> 1) + (k_sub >> 1) + t + 4] : 0U;
            a_sub[sub][3] = (r_bot < M && (k_sub + 2 * t + 9) < K) ? a_bf162[r_bot * (K >> 1) + (k_sub >> 1) + t + 4] : 0U;
        }

        const uint32_t sh_0a = 15 - t;
        const uint32_t sh_0b = 11 - t;
        const uint32_t sh_1a = 7 - t;
        const uint32_t sh_1b = 3 - t;

        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            int col = col_base + ni * 8 + group_id;
            uint32_t w_u32 = (col < N) ? w_packed_u32[col * (K >> 5) + kt] : 0U;
            uint16_t sc = (col < N) ? scales_bf16[col * (K >> 7) + (kt >> 2)] : 0U;
            uint32_t sc_bf162 = ((uint32_t)sc) | (((uint32_t)sc) << 16);

            uint32_t rb0, rb1;
            unpack_w1_b_frag(rb0, rb1, w_u32, sh_0a, sh_0b, sc_bf162);
            mma_m16n8k16_bf16_f32(
                acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                a_sub[0][0], a_sub[0][1], a_sub[0][2], a_sub[0][3], rb0, rb1
            );

            unpack_w1_b_frag(rb0, rb1, w_u32, sh_1a, sh_1b, sc_bf162);
            mma_m16n8k16_bf16_f32(
                acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                a_sub[1][0], a_sub[1][1], a_sub[1][2], a_sub[1][3], rb0, rb1
            );
        }
    }

    // L2-resident atomic partial reduction across Split-K slices
    #pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
        int c0 = col_base + ni * 8 + 2 * t;
        int c1 = c0 + 1;
        if (r_top < M && c0 < N) atomicAdd(&c_fp32[r_top * N + c0], acc[ni][0]);
        if (r_top < M && c1 < N) atomicAdd(&c_fp32[r_top * N + c1], acc[ni][1]);
        if (r_bot < M && c0 < N) atomicAdd(&c_fp32[r_bot * N + c0], acc[ni][2]);
        if (r_bot < M && c1 < N) atomicAdd(&c_fp32[r_bot * N + c1], acc[ni][3]);
    }
}
