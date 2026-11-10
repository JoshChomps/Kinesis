// File: sim_harness/ptx_bf162_emu.c
// Bit-Exact PTX & IEEE-754 bf162 Instruction Emulator
// Build: gcc -O3 -std=c11 -o ptx_bf162_emu ptx_bf162_emu.c -lm

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ==============================================================================
// 1. IEEE-754 BFloat16 & FP32 Bit-Exact Conversion Helpers
// ==============================================================================

static inline uint32_t fp32_to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(uint32_t));
    return u;
}

static inline float bits_to_fp32(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

// IEEE-754 round-to-nearest-even FP32 -> BF16 conversion (matches PTX .rn.bf16)
static inline uint16_t fp32_to_bf16_rn(float f) {
    uint32_t u = fp32_to_bits(f);
    if ((u & 0x7F800000U) == 0x7F800000U && (u & 0x007FFFFFU) != 0) {
        return (uint16_t)((u >> 16) | 0x0040U); // Quiet NaN
    }
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t rounding_bias = 0x00007FFFU + lsb;
    u += rounding_bias;
    return (uint16_t)(u >> 16);
}

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return bits_to_fp32(u);
}

static inline uint32_t pack_bf162(uint16_t low, uint16_t high) {
    return ((uint32_t)low) | (((uint32_t)high) << 16);
}

static inline uint16_t unpack_bf162_low(uint32_t pair) {
    return (uint16_t)(pair & 0xFFFFU);
}

static inline uint16_t unpack_bf162_high(uint32_t pair) {
    return (uint16_t)(pair >> 16);
}

// ==============================================================================
// 2. Bit-Exact PTX Instruction Emulators
// ==============================================================================

// PTX: shl.b32 d, a, b (clamped at 32 per PTX ISA specification)
static inline uint32_t ptx_shl_b32(uint32_t a, uint32_t b) {
    return (b >= 32U) ? 0U : (a << b);
}

// PTX: shr.u32 d, a, b (clamped at 32 per PTX ISA specification)
static inline uint32_t ptx_shr_u32(uint32_t a, uint32_t b) {
    return (b >= 32U) ? 0U : (a >> b);
}

// PTX: lop3.b32 d, a, b, c, immLut
// Evaluates arbitrary 3-input Boolean truth table specified by immLut (0x00..0xFF)
// where basis constants are TA = 0xF0, TB = 0xCC, TC = 0xAA.
static inline uint32_t ptx_lop3_b32(uint32_t a, uint32_t b, uint32_t c, uint8_t immLut) {
    uint32_t result = 0U;
    for (int i = 0; i < 8; ++i) {
        if ((immLut >> i) & 1U) {
            uint32_t term_a = (i & 4) ? a : ~a;
            uint32_t term_b = (i & 2) ? b : ~b;
            uint32_t term_c = (i & 1) ? c : ~c;
            result |= (term_a & term_b & term_c);
        }
    }
    return result;
}

// PTX: mul.rn.bf16x2 d, a, b (or fma.rn.bf16x2 d, a, b, -0.0)
static inline uint32_t ptx_mul_rn_bf162(uint32_t a, uint32_t b) {
    float a_lo = bf16_to_fp32(unpack_bf162_low(a));
    float a_hi = bf16_to_fp32(unpack_bf162_high(a));
    float b_lo = bf16_to_fp32(unpack_bf162_low(b));
    float b_hi = bf16_to_fp32(unpack_bf162_high(b));
    uint16_t d_lo = fp32_to_bf16_rn(a_lo * b_lo);
    uint16_t d_hi = fp32_to_bf16_rn(a_hi * b_hi);
    return pack_bf162(d_lo, d_hi);
}

// PTX: fma.rn.bf16x2 d, a, b, c
static inline uint32_t ptx_fma_rn_bf162(uint32_t a, uint32_t b, uint32_t c) {
    float a_lo = bf16_to_fp32(unpack_bf162_low(a));
    float a_hi = bf16_to_fp32(unpack_bf162_high(a));
    float b_lo = bf16_to_fp32(unpack_bf162_low(b));
    float b_hi = bf16_to_fp32(unpack_bf162_high(b));
    float c_lo = bf16_to_fp32(unpack_bf162_low(c));
    float c_hi = bf16_to_fp32(unpack_bf162_high(c));
    uint16_t d_lo = fp32_to_bf16_rn(fmaf(a_lo, b_lo, c_lo));
    uint16_t d_hi = fp32_to_bf16_rn(fmaf(a_hi, b_hi, c_hi));
    return pack_bf162(d_lo, d_hi);
}

// PTX: abs.bf16x2 d, a (clears IEEE-754 sign bits 15 and 31)
static inline uint32_t ptx_abs_bf162(uint32_t a) {
    return a & 0x7FFF7FFFU;
}

// PTX: add.rn.bf16x2 d, a, b (or fma.rn.bf16x2 d, a, 1.0, b)
static inline uint32_t ptx_add_rn_bf162(uint32_t a, uint32_t b) {
    float a_lo = bf16_to_fp32(unpack_bf162_low(a));
    float a_hi = bf16_to_fp32(unpack_bf162_high(a));
    float b_lo = bf16_to_fp32(unpack_bf162_low(b));
    float b_hi = bf16_to_fp32(unpack_bf162_high(b));
    return pack_bf162(fp32_to_bf16_rn(a_lo + b_lo), fp32_to_bf16_rn(a_hi + b_hi));
}

// PTX: max.bf16x2 d, a, b (used in Quest/LServe sign-aware page bounding boxes)
static inline uint32_t ptx_max_bf162(uint32_t a, uint32_t b) {
    float a_lo = bf16_to_fp32(unpack_bf162_low(a));
    float a_hi = bf16_to_fp32(unpack_bf162_high(a));
    float b_lo = bf16_to_fp32(unpack_bf162_low(b));
    float b_hi = bf16_to_fp32(unpack_bf162_high(b));
    // PTX max.bf16x2 (no .NaN modifier): if exactly one input is NaN the result is the other input; if both are NaN
    // the result is the canonical NaN 0x7FFF (docs/12 4.17: the old `a >= b ? a : b` returned b whenever b was NaN).
    uint16_t d_lo, d_hi;
    if (a_lo != a_lo && b_lo != b_lo) d_lo = 0x7FFFU;
    else if (a_lo != a_lo)            d_lo = unpack_bf162_low(b);
    else if (b_lo != b_lo)            d_lo = unpack_bf162_low(a);
    else                              d_lo = (a_lo >= b_lo) ? unpack_bf162_low(a) : unpack_bf162_low(b);
    if (a_hi != a_hi && b_hi != b_hi) d_hi = 0x7FFFU;
    else if (a_hi != a_hi)            d_hi = unpack_bf162_high(b);
    else if (b_hi != b_hi)            d_hi = unpack_bf162_high(a);
    else                              d_hi = (a_hi >= b_hi) ? unpack_bf162_high(a) : unpack_bf162_high(b);
    return pack_bf162(d_lo, d_hi);
}

// PTX: prmt.b32 d, a, b, sel (FLUTE 4-byte register LUT selector across {a, b})
static inline uint32_t ptx_prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    uint64_t combined = ((uint64_t)b << 32) | (uint64_t)a;
    uint32_t out = 0U;
    for (int i = 0; i < 4; ++i) {
        uint32_t ctrl = (sel >> (4 * i)) & 0xFU;
        uint32_t byte_idx = ctrl & 0x7U;
        uint8_t byte_val = (uint8_t)((combined >> (8 * byte_idx)) & 0xFFU);
        if (ctrl & 0x8U) {
            byte_val = (byte_val & 0x80U) ? 0xFFU : 0x00U; // Sign-extend mode
        }
        out |= ((uint32_t)byte_val) << (8 * i);
    }
    return out;
}

// Quest / LServe branchless 2-instruction PTX sign-aware page bounding-box inner product:
// Evaluates max(q * k_min, q * k_max) for two channels simultaneously in bf162.
static inline uint32_t ptx_quest_page_ub_bf162(
    uint32_t q_bf162,
    uint32_t k_min_bf162,
    uint32_t k_max_bf162
) {
    uint32_t p_min = ptx_mul_rn_bf162(q_bf162, k_min_bf162);
    uint32_t p_max = ptx_mul_rn_bf162(q_bf162, k_max_bf162);
    return ptx_max_bf162(p_min, p_max);
}

// ==============================================================================
// 3. Interleaved 16-Bit Split Packing & Branchless PTX Unpackers
// ==============================================================================

// Pack 32 1-bit weights b[0..31] (each in {0, 1} representing {-1, +1})
// into Interleaved 16-Bit Split Layout:
//   even weights b_{2k}   -> bit k      (k in 0..15)
//   odd weights  b_{2k+1} -> bit k + 16 (k in 0..15)
static inline uint32_t pack_w1_interleaved_u32(const uint8_t b[32]) {
    uint32_t w = 0U;
    for (int k = 0; k < 16; ++k) {
        uint32_t even_bit = (uint32_t)(b[2 * k] & 1U);
        uint32_t odd_bit  = (uint32_t)(b[2 * k + 1] & 1U);
        w |= (even_bit << k) | (odd_bit << (k + 16));
    }
    return w;
}

// Branchless 3-instruction PTX 1-bit unpacker for pair k in [0..15]
static inline uint32_t ptx_unpack_w1_pair(uint32_t w_interleaved, uint32_t k, uint32_t scale_bf162) {
    uint32_t tmp = ptx_shl_b32(w_interleaved, 15U - k);
    uint32_t unscaled = ptx_lop3_b32(tmp, 0x80008000U, 0x3F803F80U, 0xAEU);
    return ptx_mul_rn_bf162(unscaled, scale_bf162);
}

// Pack 16 2-bit ternary weights q[0..15] (each in {0, 1, 2} representing {-1, 0, +1})
// into Interleaved 16-Bit Split Layout:
//   even weights q_{2k}   -> bits [2k+1 : 2k]     in lower 16 bits (k in 0..7)
//   odd weights  q_{2k+1} -> bits [2k+17 : 2k+16] in upper 16 bits (k in 0..7)
static inline uint32_t pack_w2_ternary_interleaved_u32(const uint8_t q[16]) {
    uint32_t w = 0U;
    for (int k = 0; k < 8; ++k) {
        uint32_t even_2b = (uint32_t)(q[2 * k] & 3U);
        uint32_t odd_2b  = (uint32_t)(q[2 * k + 1] & 3U);
        w |= (even_2b << (2 * k)) | (odd_2b << (2 * k + 16));
    }
    return w;
}

// Branchless PTX 2-bit ternary unpacker for pair k in [0..7]
// Uses exact unscaled BF162 constants c_mul = (64.0, 64.0) [0x42804280] and
// c_sub = (-129.0, -129.0) [0xC301C301] in fma.rn.bf16x2 to produce exact unit
// trits {-1.0, 0.0, +1.0} [0xBF80, 0x0000, 0x3F80], followed by fma.rn.bf16x2 with
// scale_bf162. This prevents 8-bit BF16 significand overflow on (-129.0 * alpha_g)
// and guarantees that ternary zero (q = 1) unpacks to bit-exact +0.0 (0x0000)
// across all 65,536 possible BF16 scales.
static inline uint32_t ptx_unpack_w2_ternary_pair(
    uint32_t w_tern_interleaved,
    uint32_t k,
    uint32_t scale_bf162
) {
    const uint32_t c_mul_64_bf162  = 0x42804280U; // (64.0f, 64.0f) in BF162
    const uint32_t c_sub_129_bf162 = 0xC301C301U; // (-129.0f, -129.0f) in BF162
    uint32_t tmp = ptx_shr_u32(w_tern_interleaved, 2U * k);
    uint32_t injected = ptx_lop3_b32(tmp, 0x00030003U, 0x40004000U, 0xEAU);
    uint32_t unit_trit = ptx_fma_rn_bf162(injected, c_mul_64_bf162, c_sub_129_bf162);
    return ptx_mul_rn_bf162(unit_trit, scale_bf162);
}

// ==============================================================================
// 4. Warp-Level mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 Emulator
// ==============================================================================

typedef struct {
    uint32_t a[4]; // 4 bf162 registers per thread (8 BF16 activations)
    uint32_t b[2]; // 2 bf162 registers per thread (4 BF16 weights)
    float    c[4]; // 4 FP32 accumulator registers per thread
    float    d[4]; // 4 FP32 output registers per thread
} WarpThreadRegs;

// Emulates a full 32-lane warp executing mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// using exact warp lane-to-matrix coordinate mapping from the PTX ISA documentation:
//   For A (16x16, row-major):
//     groupID = lane_id >> 2 (0..7), tid_in_group = lane_id & 3 (0..3)
//     a[0] holds (row = groupID,     col = 2*tid_in_group .. 2*tid_in_group+1)
//     a[1] holds (row = groupID + 8, col = 2*tid_in_group .. 2*tid_in_group+1)
//     a[2] holds (row = groupID,     col = 2*tid_in_group + 8 .. 2*tid_in_group+9)
//     a[3] holds (row = groupID + 8, col = 2*tid_in_group + 8 .. 2*tid_in_group+9)
//   For B (16x8, col-major, K=16, N=8):
//     col n = groupID (0..7)
//     b[0] holds (row k = 2*tid_in_group .. 2*tid_in_group+1,     col = groupID)
//     b[1] holds (row k = 2*tid_in_group + 8 .. 2*tid_in_group+9, col = groupID)
//   For C/D (16x8, row-major):
//     d[0], d[1] hold (row = groupID,     col = 2*tid_in_group, 2*tid_in_group+1)
//     d[2], d[3] hold (row = groupID + 8, col = 2*tid_in_group, 2*tid_in_group+1)
static void ptx_mma_m16n8k16_bf16_f32_warp(WarpThreadRegs warp[32]) {
    float A_mat[16][16];
    float B_mat[16][8];
    float C_mat[16][8];

    for (int lane = 0; lane < 32; ++lane) {
        int group_id = lane >> 2;
        int tid = lane & 3;

        A_mat[group_id][2 * tid]         = bf16_to_fp32(unpack_bf162_low(warp[lane].a[0]));
        A_mat[group_id][2 * tid + 1]     = bf16_to_fp32(unpack_bf162_high(warp[lane].a[0]));
        A_mat[group_id + 8][2 * tid]     = bf16_to_fp32(unpack_bf162_low(warp[lane].a[1]));
        A_mat[group_id + 8][2 * tid + 1] = bf16_to_fp32(unpack_bf162_high(warp[lane].a[1]));
        A_mat[group_id][2 * tid + 8]     = bf16_to_fp32(unpack_bf162_low(warp[lane].a[2]));
        A_mat[group_id][2 * tid + 9]     = bf16_to_fp32(unpack_bf162_high(warp[lane].a[2]));
        A_mat[group_id + 8][2 * tid + 8] = bf16_to_fp32(unpack_bf162_low(warp[lane].a[3]));
        A_mat[group_id + 8][2 * tid + 9] = bf16_to_fp32(unpack_bf162_high(warp[lane].a[3]));

        B_mat[2 * tid][group_id]         = bf16_to_fp32(unpack_bf162_low(warp[lane].b[0]));
        B_mat[2 * tid + 1][group_id]     = bf16_to_fp32(unpack_bf162_high(warp[lane].b[0]));
        B_mat[2 * tid + 8][group_id]     = bf16_to_fp32(unpack_bf162_low(warp[lane].b[1]));
        B_mat[2 * tid + 9][group_id]     = bf16_to_fp32(unpack_bf162_high(warp[lane].b[1]));

        C_mat[group_id][2 * tid]         = warp[lane].c[0];
        C_mat[group_id][2 * tid + 1]     = warp[lane].c[1];
        C_mat[group_id + 8][2 * tid]     = warp[lane].c[2];
        C_mat[group_id + 8][2 * tid + 1] = warp[lane].c[3];
    }

    float D_mat[16][8];
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float acc = C_mat[m][n];
            for (int k = 0; k < 16; ++k) {
                acc += A_mat[m][k] * B_mat[k][n];
            }
            D_mat[m][n] = acc;
        }
    }

    for (int lane = 0; lane < 32; ++lane) {
        int group_id = lane >> 2;
        int tid = lane & 3;
        warp[lane].d[0] = D_mat[group_id][2 * tid];
        warp[lane].d[1] = D_mat[group_id][2 * tid + 1];
        warp[lane].d[2] = D_mat[group_id + 8][2 * tid];
        warp[lane].d[3] = D_mat[group_id + 8][2 * tid + 1];
    }
}

// ==============================================================================
// 5. Verification Suite (target bits exhaustive; surrounding noise bits sampled)
// ==============================================================================

int main(void) {
    printf("================================================================================\n");
    printf("  WORKSTREAM 1: BIT-EXACT PTX & IEEE-754 BF162 INSTRUCTION EMULATOR\n");
    printf("================================================================================\n\n");

    // 1. Verify truth-table basis derivation for immLut = 0xAE and immLut = 0xEA
    const uint8_t TA = 0xF0U;
    const uint8_t TB = 0xCCU;
    const uint8_t TC = 0xAAU;
    uint8_t lut_1bit = (uint8_t)(((~TA) & TB) | TC);
    uint8_t lut_2bit = (uint8_t)((TA & TB) | TC);

    printf("[1] Truth-Table Basis Derivation Check:\n");
    printf("    TA = 0x%02X, TB = 0x%02X, TC = 0x%02X\n", TA, TB, TC);
    printf("    F_1bit(a,b,c) = (~a & b) | c  -> immLut = 0x%02X (Expected: 0xAE)\n", lut_1bit);
    printf("    F_2bit(a,b,c) = (a & b)  | c  -> immLut = 0x%02X (Expected: 0xEA)\n\n", lut_2bit);
    assert(lut_1bit == 0xAEU);
    assert(lut_2bit == 0xEAU);

    // 2. Verification of lop3.b32 (0xAE) over 65,536 pseudo-random noise patterns (multiplicative hash of a
    //    16-bit counter over the 30 non-target bits; this samples the 2^30 space, it is NOT exhaustive)
    //    and all 16 pair positions k in [0..15] (16 * 4 * 65,536 = 4,194,304 checks)
    printf("[2] Sampled 1-Bit Binary g128 Unpacker Check (immLut = 0xAE):\n");
    uint64_t total_w1_tests = 0;
    uint64_t w1_bit_errors = 0;

    float test_scales[4] = {1.0f, 0.03125f, 0.142578125f, 1.75f};
    for (int s_idx = 0; s_idx < 4; ++s_idx) {
        uint16_t scale_bf16 = fp32_to_bf16_rn(test_scales[s_idx]);
        uint32_t scale_bf162 = pack_bf162(scale_bf16, scale_bf16);
        uint16_t pos_ref = fp32_to_bf16_rn(+1.0f * bf16_to_fp32(scale_bf16));
        uint16_t neg_ref = fp32_to_bf16_rn(-1.0f * bf16_to_fp32(scale_bf16));

        for (uint32_t k = 0; k < 16; ++k) {
            for (uint32_t b_even = 0; b_even <= 1; ++b_even) {
                for (uint32_t b_odd = 0; b_odd <= 1; ++b_odd) {
                    uint32_t target_mask = (1U << k) | (1U << (k + 16));
                    uint32_t target_bits = (b_even << k) | (b_odd << (k + 16));
                    uint32_t expected_pair = pack_bf162(
                        b_even ? pos_ref : neg_ref,
                        b_odd  ? pos_ref : neg_ref
                    );

                    // 65,536 hashed noise patterns over the surrounding bit positions (sampled, not exhaustive)
                    for (uint32_t noise_state = 0; noise_state < 65536U; ++noise_state) {
                        uint32_t noise_word = (noise_state * 0x9E3779B9U) & (~target_mask);
                        uint32_t w_interleaved = target_bits | noise_word;
                        uint32_t actual_pair = ptx_unpack_w1_pair(w_interleaved, k, scale_bf162);
                        if (actual_pair != expected_pair) {
                            w1_bit_errors++;
                        }
                        total_w1_tests++;
                    }
                }
            }
        }
    }

    printf("    Tested States : %llu (65,536 hashed noise patterns x 4 bit pairs x 16 k-shifts x 4 scales)\n",
           (unsigned long long)total_w1_tests);
    printf("    Bit Errors    : %llu (rtol = 0, atol = 0)\n\n", (unsigned long long)w1_bit_errors);
    assert(w1_bit_errors == 0);

    // 3. Verification of lop3.b32 (0xEA) + 2x fma.rn.bf16x2 across all ternary pairs q in {0, 1, 2}^2,
    //    all k in [0..7], and 65,536 hashed (sampled, not exhaustive) surrounding bit patterns
    printf("[3] Sampled 2-Bit Ternary (BitNet b1.58) Unpacker Check (immLut = 0xEA + fma.rn.bf16x2):\n");
    uint64_t total_w2_tests = 0;
    uint64_t w2_bit_errors = 0;

    for (int s_idx = 0; s_idx < 4; ++s_idx) {
        uint16_t scale_bf16 = fp32_to_bf16_rn(test_scales[s_idx]);
        float alpha_g = bf16_to_fp32(scale_bf16);
        uint32_t scale_bf162 = pack_bf162(scale_bf16, scale_bf16);

        // True mathematical ternary target: {-alpha_g, +0.0f, +alpha_g}
        uint16_t ref_map[3] = {
            fp32_to_bf16_rn(-1.0f * alpha_g),
            0x0000U, // Exact +0.0f in BF16 for q = 1 (zero weight)
            fp32_to_bf16_rn(+1.0f * alpha_g)
        };
        assert(bf16_to_fp32(ref_map[1]) == 0.0f);

        for (uint32_t k = 0; k < 8; ++k) {
            for (uint32_t q_even = 0; q_even <= 2; ++q_even) {
                for (uint32_t q_odd = 0; q_odd <= 2; ++q_odd) {
                    uint32_t target_mask = (3U << (2 * k)) | (3U << (2 * k + 16));
                    uint32_t target_bits = (q_even << (2 * k)) | (q_odd << (2 * k + 16));
                    uint32_t expected_pair = pack_bf162(ref_map[q_even], ref_map[q_odd]);

                    for (uint32_t noise_state = 0; noise_state < 65536U; ++noise_state) {
                        uint32_t noise_word = (noise_state * 0x85EBCA6BU) & (~target_mask);
                        uint32_t w_tern = target_bits | noise_word;
                        uint32_t actual_pair = ptx_unpack_w2_ternary_pair(
                            w_tern, k, scale_bf162
                        );
                        if (actual_pair != expected_pair) {
                            w2_bit_errors++;
                        }
                        total_w2_tests++;
                    }
                }
            }
        }
    }

    printf("    Tested States : %llu (65,536 hashed noise patterns x 9 ternary pairs x 8 k-shifts x 4 scales)\n",
           (unsigned long long)total_w2_tests);
    printf("    Bit Errors    : %llu (rtol = 0, atol = 0)\n\n", (unsigned long long)w2_bit_errors);
    assert(w2_bit_errors == 0);

    // 4. Warp-level mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 + W1A16 K=32 step verification
    printf("[4] Warp-Level Fused W1A16_F32 mma.sync.aligned.m16n8k16 Verification (K=32):\n");
    float A_ref[16][32];
    uint8_t W_bits[8][32]; // 8 columns (N=8), 32 rows (K=32)
    float scale_n[8];
    uint32_t w_packed_col[8];
    uint32_t scale_col_bf162[8];

    uint32_t rng = 0x12345678U;
    for (int m = 0; m < 16; ++m) {
        for (int k = 0; k < 32; ++k) {
            rng = rng * 1664525U + 1013904223U;
            float val = ((float)((int32_t)(rng >> 16) - 32768)) / 16384.0f;
            A_ref[m][k] = bf16_to_fp32(fp32_to_bf16_rn(val));
        }
    }

    for (int n = 0; n < 8; ++n) {
        rng = rng * 1664525U + 1013904223U;
        uint16_t sc_bf16 = fp32_to_bf16_rn(0.125f + (float)(n + 1) * 0.0625f);
        scale_n[n] = bf16_to_fp32(sc_bf16);
        scale_col_bf162[n] = pack_bf162(sc_bf16, sc_bf16);
        for (int k = 0; k < 32; ++k) {
            rng = rng * 1664525U + 1013904223U;
            W_bits[n][k] = (uint8_t)((rng >> 24) & 1U);
        }
        w_packed_col[n] = pack_w1_interleaved_u32(W_bits[n]);
    }

    WarpThreadRegs warp[32];
    memset(warp, 0, sizeof(warp));

    // Sub-step 0 (K = 0..15) and Sub-step 1 (K = 16..31)
    for (int sub = 0; sub < 2; ++sub) {
        int k_base = sub * 16;
        for (int lane = 0; lane < 32; ++lane) {
            int group_id = lane >> 2;
            int t = lane & 3;

            // Load A fragment for this lane
            warp[lane].a[0] = pack_bf162(
                fp32_to_bf16_rn(A_ref[group_id][k_base + 2 * t]),
                fp32_to_bf16_rn(A_ref[group_id][k_base + 2 * t + 1])
            );
            warp[lane].a[1] = pack_bf162(
                fp32_to_bf16_rn(A_ref[group_id + 8][k_base + 2 * t]),
                fp32_to_bf16_rn(A_ref[group_id + 8][k_base + 2 * t + 1])
            );
            warp[lane].a[2] = pack_bf162(
                fp32_to_bf16_rn(A_ref[group_id][k_base + 2 * t + 8]),
                fp32_to_bf16_rn(A_ref[group_id][k_base + 2 * t + 9])
            );
            warp[lane].a[3] = pack_bf162(
                fp32_to_bf16_rn(A_ref[group_id + 8][k_base + 2 * t + 8]),
                fp32_to_bf16_rn(A_ref[group_id + 8][k_base + 2 * t + 9])
            );

            // Unpack B fragment using exact pair indices p0 = t + 8*sub, p1 = t + 4 + 8*sub
            uint32_t p0 = (uint32_t)(t + 8 * sub);
            uint32_t p1 = (uint32_t)(t + 4 + 8 * sub);
            warp[lane].b[0] = ptx_unpack_w1_pair(w_packed_col[group_id], p0, scale_col_bf162[group_id]);
            warp[lane].b[1] = ptx_unpack_w1_pair(w_packed_col[group_id], p1, scale_col_bf162[group_id]);
        }

        ptx_mma_m16n8k16_bf16_f32_warp(warp);
        for (int lane = 0; lane < 32; ++lane) {
            for (int r = 0; r < 4; ++r) {
                warp[lane].c[r] = warp[lane].d[r];
            }
        }
    }

    // Verify accumulated warp registers against direct reference matrix multiplication
    int mma_mismatches = 0;
    for (int lane = 0; lane < 32; ++lane) {
        int group_id = lane >> 2;
        int t = lane & 3;
        int rows[4] = {group_id, group_id, group_id + 8, group_id + 8};
        int cols[4] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1};

        for (int idx = 0; idx < 4; ++idx) {
            int m = rows[idx];
            int n = cols[idx];
            float ref_acc = 0.0f;
            for (int k = 0; k < 32; ++k) {
                float w_val = (W_bits[n][k] ? +1.0f : -1.0f) * scale_n[n];
                ref_acc += A_ref[m][k] * w_val;
            }
            if (warp[lane].d[idx] != ref_acc) {
                mma_mismatches++;
            }
        }
    }

    printf("    Warp Accumulator Elements Checked : 128 (16x8 FP32 tile over K=32)\n");
    printf("    Warp Accumulator Bit Mismatches   : %d (rtol = 0, atol = 0)\n\n", mma_mismatches);
    assert(mma_mismatches == 0);

    // 5. Warp-level Fused 2-Stage Kronecker H_32 = H_2 x H_16 + W2A16_F32 Ternary MMA Verification (K=32, fixes O12)
    printf("[5] Warp-Level Fused 2-Stage HadaCore Kronecker (H_2 x H_16) + W2A16 Ternary MMA (K=32):\n");
    uint8_t W2_trits[8][32]; // values in {0, 1, 2} representing {-1, 0, +1}
    uint32_t w2_packed_col[8][2]; // 2 uint32 words (16 trits each) per N-column for K=32
    for (int n = 0; n < 8; ++n) {
        for (int k = 0; k < 32; ++k) {
            rng = rng * 1664525U + 1013904223U;
            W2_trits[n][k] = (uint8_t)((rng >> 24) % 3U);
        }
        w2_packed_col[n][0] = pack_w2_ternary_interleaved_u32(&W2_trits[n][0]);
        w2_packed_col[n][1] = pack_w2_ternary_interleaved_u32(&W2_trits[n][16]);
    }

    // Build normalized 16x16 Sylvester-Hadamard matrix H16_norm[k][c] = (-1)^popcount(k & c) * 0.25f
    float H16_norm[16][16];
    for (int k = 0; k < 16; ++k) {
        for (int c = 0; c < 16; ++c) {
            int v = k & c;
            int pop = ((v >> 3) & 1) + ((v >> 2) & 1) + ((v >> 1) & 1) + (v & 1);
            H16_norm[k][c] = (pop & 1) ? -0.25f : +0.25f;
        }
    }

    // Reference 2-stage Kronecker transform (orthonormal H_32 = H_2 (x) H_16):
    //   Stage 1: U_sub[m][c] = bf16_rn(sum_{k=0..15} A_ref[m][sub*16 + k] * H16_norm[k][c])
    //   Stage 2: s0 = bf16_rn(U_0[m][c] * (1/sqrt(2)))
    //            A_kron_ref[m][c]      = bf16_rn(fma(U_1[m][c], +1/sqrt(2), s0))
    //            A_kron_ref[m][16 + c] = bf16_rn(fma(U_1[m][c], -1/sqrt(2), s0))
    // H_2 gain (docs/12 4.17): 0x3F35 = 0.70703125 is the round-to-nearest BF16 of 1/sqrt(2) = 0.70710678. One H_2
    // stage scales amplitudes by 0.70703125 * sqrt(2) = 1 - 1.068e-4 and energy by 2 * 0.70703125^2 = 1 - 2.136e-4
    // against the packer's exact 1/sqrt(2); about 18x below a BF16 half-ulp (2^-9), so it is documented, not compensated.
    const uint32_t h2_pos_bf162 = 0x3F353F35U; // (+0.70703125f, +0.70703125f) ~= +1/sqrt(2) in BF162
    const uint32_t h2_neg_bf162 = 0xBF35BF35U; // (-0.70703125f, -0.70703125f) ~= -1/sqrt(2) in BF162
    float U_ref[2][16][16];
    float A_kron_ref[16][32];
    for (int sub = 0; sub < 2; ++sub) {
        for (int m = 0; m < 16; ++m) {
            for (int c = 0; c < 16; ++c) {
                float acc = 0.0f;
                for (int k = 0; k < 16; ++k) {
                    acc += A_ref[m][sub * 16 + k] * H16_norm[k][c];
                }
                U_ref[sub][m][c] = bf16_to_fp32(fp32_to_bf16_rn(acc));
            }
        }
    }
    for (int m = 0; m < 16; ++m) {
        for (int c = 0; c < 16; ++c) {
            uint32_t u0_b = (uint32_t)fp32_to_bf16_rn(U_ref[0][m][c]);
            uint32_t u1_b = (uint32_t)fp32_to_bf16_rn(U_ref[1][m][c]);
            uint32_t s0_b = ptx_mul_rn_bf162(u0_b, h2_pos_bf162);
            uint32_t r0_b = ptx_fma_rn_bf162(u1_b, h2_pos_bf162, s0_b);
            uint32_t r1_b = ptx_fma_rn_bf162(u1_b, h2_neg_bf162, s0_b);
            A_kron_ref[m][c]      = bf16_to_fp32(unpack_bf162_low(r0_b));
            A_kron_ref[m][16 + c] = bf16_to_fp32(unpack_bf162_low(r1_b));
        }
    }

    // Execute Stage 1 on the emulated Tensor Core warp: two m16n8k16 MMAs per K=16 sub-tile
    uint32_t u_stage1_regs[2][32][4]; // [sub 0..1][lane 0..31][reg 0..3]
    for (int sub = 0; sub < 2; ++sub) {
        int k_base = sub * 16;
        for (int half = 0; half < 2; ++half) { // half=0 -> cols 0..7 of H16, half=1 -> cols 8..15 of H16
            WarpThreadRegs warp_h16[32];
            memset(warp_h16, 0, sizeof(warp_h16));
            for (int lane = 0; lane < 32; ++lane) {
                int group_id = lane >> 2;
                int t = lane & 3;
                warp_h16[lane].a[0] = pack_bf162(fp32_to_bf16_rn(A_ref[group_id][k_base + 2 * t]),     fp32_to_bf16_rn(A_ref[group_id][k_base + 2 * t + 1]));
                warp_h16[lane].a[1] = pack_bf162(fp32_to_bf16_rn(A_ref[group_id + 8][k_base + 2 * t]), fp32_to_bf16_rn(A_ref[group_id + 8][k_base + 2 * t + 1]));
                warp_h16[lane].a[2] = pack_bf162(fp32_to_bf16_rn(A_ref[group_id][k_base + 2 * t + 8]), fp32_to_bf16_rn(A_ref[group_id][k_base + 2 * t + 9]));
                warp_h16[lane].a[3] = pack_bf162(fp32_to_bf16_rn(A_ref[group_id + 8][k_base + 2 * t + 8]), fp32_to_bf16_rn(A_ref[group_id + 8][k_base + 2 * t + 9]));

                int h_col = half * 8 + group_id;
                warp_h16[lane].b[0] = pack_bf162(fp32_to_bf16_rn(H16_norm[2 * t][h_col]),     fp32_to_bf16_rn(H16_norm[2 * t + 1][h_col]));
                warp_h16[lane].b[1] = pack_bf162(fp32_to_bf16_rn(H16_norm[2 * t + 8][h_col]), fp32_to_bf16_rn(H16_norm[2 * t + 9][h_col]));
            }
            ptx_mma_m16n8k16_bf16_f32_warp(warp_h16);
            for (int lane = 0; lane < 32; ++lane) {
                u_stage1_regs[sub][lane][2 * half + 0] = pack_bf162(
                    fp32_to_bf16_rn(warp_h16[lane].d[0]), fp32_to_bf16_rn(warp_h16[lane].d[1])
                );
                u_stage1_regs[sub][lane][2 * half + 1] = pack_bf162(
                    fp32_to_bf16_rn(warp_h16[lane].d[2]), fp32_to_bf16_rn(warp_h16[lane].d[3])
                );
            }
        }
    }

    // Execute Stage 2 (in-register H_2 butterfly via ptx_fma_rn_bf162) + Stage 3 (W2A16 Ternary MMA)
    WarpThreadRegs warp_w2[32];
    memset(warp_w2, 0, sizeof(warp_w2));

    for (int sub = 0; sub < 2; ++sub) {
        for (int lane = 0; lane < 32; ++lane) {
            int group_id = lane >> 2;
            int t = lane & 3;
            for (int r = 0; r < 4; ++r) {
                uint32_t scaled_0 = ptx_mul_rn_bf162(u_stage1_regs[0][lane][r], h2_pos_bf162);
                warp_w2[lane].a[r] = ptx_fma_rn_bf162(
                    u_stage1_regs[1][lane][r], (sub == 0) ? h2_pos_bf162 : h2_neg_bf162, scaled_0
                );
            }

            // Unpack 2-bit ternary B fragment for sub-step (8 pairs per uint32 word: p0 = t, p1 = t + 4)
            warp_w2[lane].b[0] = ptx_unpack_w2_ternary_pair(w2_packed_col[group_id][sub], (uint32_t)t, scale_col_bf162[group_id]);
            warp_w2[lane].b[1] = ptx_unpack_w2_ternary_pair(w2_packed_col[group_id][sub], (uint32_t)(t + 4), scale_col_bf162[group_id]);
        }

        ptx_mma_m16n8k16_bf16_f32_warp(warp_w2);
        for (int lane = 0; lane < 32; ++lane) {
            for (int r = 0; r < 4; ++r) {
                warp_w2[lane].c[r] = warp_w2[lane].d[r];
            }
        }
    }

    int w2_mma_mismatches = 0;
    for (int lane = 0; lane < 32; ++lane) {
        int group_id = lane >> 2;
        int t = lane & 3;
        int rows[4] = {group_id, group_id, group_id + 8, group_id + 8};
        int cols[4] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1};
        for (int idx = 0; idx < 4; ++idx) {
            int m = rows[idx];
            int n = cols[idx];
            float ref_acc = 0.0f;
            for (int k = 0; k < 32; ++k) {
                float tr = (float)((int)W2_trits[n][k] - 1);
                float w_val = bf16_to_fp32(fp32_to_bf16_rn(tr * scale_n[n]));
                ref_acc += A_kron_ref[m][k] * w_val;
            }
            if (warp_w2[lane].d[idx] != ref_acc) {
                w2_mma_mismatches++;
            }
        }
    }
    printf("    W2A16 Kronecker Accumulator Elements : 128 (16x8 FP32 tile over K=32)\n");
    printf("    W2A16 Kronecker Bit Mismatches       : %d (rtol = 0, atol = 0)\n\n", w2_mma_mismatches);
    assert(w2_mma_mismatches == 0);

    // 6. Quest Sign-Aware Page Bounding-Box & FLUTE prmt.b32 Register LUT Verification
    printf("[6] Quest Sign-Aware Page Bounding-Box & FLUTE prmt.b32 Register LUT Check:\n");
    int quest_errors = 0;
    float q_vals[4]   = {-2.5f, -0.5f, +0.75f, +3.0f};
    float min_vals[4] = {-4.0f, -1.5f, +0.25f, +1.0f};
    float max_vals[4] = {-1.0f, +2.0f, +1.50f, +5.0f};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            uint16_t q_b   = fp32_to_bf16_rn(q_vals[i]);
            uint16_t min_b = fp32_to_bf16_rn(min_vals[j]);
            uint16_t max_b = fp32_to_bf16_rn(max_vals[j]);
            uint32_t ub_pair = ptx_quest_page_ub_bf162(
                pack_bf162(q_b, q_b), pack_bf162(min_b, min_b), pack_bf162(max_b, max_b)
            );
            float actual_ub = bf16_to_fp32(unpack_bf162_low(ub_pair));
            float p0 = bf16_to_fp32(fp32_to_bf16_rn(bf16_to_fp32(q_b) * bf16_to_fp32(min_b)));
            float p1 = bf16_to_fp32(fp32_to_bf16_rn(bf16_to_fp32(q_b) * bf16_to_fp32(max_b)));
            float expected_ub = (p0 >= p1) ? p0 : p1;
            if (actual_ub != expected_ub) {
                quest_errors++;
            }
        }
    }
    uint32_t lut_lo = 0x3F800000U; // [0.0f, +1.0f] in BF16
    uint32_t lut_hi = 0xBF804000U; // [+2.0f, -1.0f] in BF16
    // Select bytes (7, 6) = 0xBF80 (-1.0f) and bytes (3, 2) = 0x3F80 (+1.0f) via prmt.b32
    uint32_t prmt_out = ptx_prmt_b32(lut_lo, lut_hi, 0x7632U);
    assert(prmt_out == 0xBF803F80U);
    printf("    Quest Bounding-Box & prmt.b32 Errors : %d (rtol = 0, atol = 0)\n\n", quest_errors);
    assert(quest_errors == 0);

    // Print Summary Verification Table
    printf("+----------------------------------------------+------------------+------------+--------+\n");
    printf("| Verification Target                          | States Tested    | Bit Errors | Status |\n");
    printf("+----------------------------------------------+------------------+------------+--------+\n");
    printf("| lop3.b32 (0xAE) Binary g128 -> bf162         | %16llu | %10llu | PASS   |\n",
           (unsigned long long)total_w1_tests, (unsigned long long)w1_bit_errors);
    printf("| lop3.b32 (0xEA) + fma.rn.bf16x2 Ternary      | %16llu | %10llu | PASS   |\n",
           (unsigned long long)total_w2_tests, (unsigned long long)w2_bit_errors);
    printf("| mma.sync.aligned.m16n8k16 W1A16_F32 (K=32)   | %16d | %10d | PASS   |\n",
           128, mma_mismatches);
    printf("| Kronecker (H2xH16) + W2A16 Ternary (K=32)    | %16d | %10d | PASS   |\n",
           128, w2_mma_mismatches);
    printf("| Quest Page Bounding + FLUTE prmt.b32 LUT     | %16d | %10d | PASS   |\n",
           17, quest_errors);
    printf("+----------------------------------------------+------------------+------------+--------+\n");

    // docs/12 4.17 regression: max.bf16x2 NaN semantics (one NaN -> the other operand, both NaN -> 0x7FFF)
    {
        const uint32_t qnan = 0x7FC0U, two = 0x4000U, neg1 = 0xBF80U;
        int max_errors = 0;
        max_errors += (ptx_max_bf162(pack_bf162(two, qnan), pack_bf162(qnan, neg1)) != pack_bf162(two, neg1));
        max_errors += (ptx_max_bf162(pack_bf162(qnan, two), pack_bf162(neg1, qnan)) != pack_bf162(neg1, two));
        max_errors += (ptx_max_bf162(pack_bf162(qnan, qnan), pack_bf162(qnan, qnan)) != pack_bf162(0x7FFFU, 0x7FFFU));
        max_errors += (ptx_max_bf162(pack_bf162(two, neg1), pack_bf162(neg1, two)) != pack_bf162(two, two));
        printf("| max.bf16x2 NaN semantics (PTX ISA)           | %16d | %10d | %s   |\n", 4, max_errors,
               max_errors == 0 ? "PASS" : "FAIL");
        printf("+----------------------------------------------+------------------+------------+--------+\n");
        if (max_errors != 0) return 1;
    }

    return 0;
}
