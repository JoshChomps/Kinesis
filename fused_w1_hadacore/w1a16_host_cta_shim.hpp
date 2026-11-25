// File: fused_w1_hadacore/w1a16_host_cta_shim.hpp
// Deterministic Host CTA (256-Fiber) & Warp-Collective (32-Lane) Execution Engine
// Enables direct host compilation (g++ -std=c++20 -x c++) and bit-exact verification
// of unmodified __global__ CUDA/PTX kernels in w1a16_hadacore_kernel.cu.

#ifndef W1A16_HOST_CTA_SHIM_HPP
#define W1A16_HOST_CTA_SHIM_HPP

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include <ucontext.h>
#define HOST_CTA_USE_UCONTEXT 1
#else
#error "w1a16_host_cta_shim.hpp requires Windows Fibers (_WIN32) or POSIX ucontext"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

// ==============================================================================
// 1. CUDA Qualifier, Attribute, and Coordinate Shims
// ==============================================================================
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))

struct dim3 {
    int x;
    int y;
    int z;
    constexpr dim3(int vx = 1, int vy = 1, int vz = 1) : x(vx), y(vy), z(vz) {}
};

static dim3 threadIdx(0, 0, 0);
static dim3 blockIdx(0, 0, 0);
static dim3 blockDim(256, 1, 1);
static dim3 gridDim(1, 1, 1);

static inline int __popc(uint32_t x) {
    return __builtin_popcount(x);
}

static inline float __uint_as_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

static inline uint32_t __float_as_uint(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(uint32_t));
    return u;
}

static inline float atomicAdd(float* addr, float val) {
    float old = *addr;
    *addr = old + val;
    return old;
}

// ==============================================================================
// 2. Shared Memory Address Translation (__cvta_generic_to_shared & cp.async)
// ==============================================================================
static uint8_t* g_smem_base_ptr = nullptr;

static inline uint32_t __cvta_generic_to_shared(const void* ptr) {
    uint8_t* p = (uint8_t*)const_cast<void*>(ptr);
    if (g_smem_base_ptr == nullptr || p < g_smem_base_ptr ||
        (uintptr_t)(p - g_smem_base_ptr) >= (1ULL << 20)) {
        g_smem_base_ptr = p;
    }
    return (uint32_t)((uintptr_t)(p - g_smem_base_ptr));
}

static inline void host_cp_async_16B(uint32_t smem_addr, const void* gmem_ptr) {
    assert(g_smem_base_ptr != nullptr);
    memcpy(g_smem_base_ptr + smem_addr, gmem_ptr, 16);
}

// ==============================================================================
// 3. Bit-Exact PTX BF162 & LOP3 Arithmetic Emulation
// ==============================================================================
static inline uint16_t host_f32_to_bf16_rn(float val) {
    uint32_t u = __float_as_uint(val);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) {
        return 0x7FC0U;
    }
    uint32_t lsb = (u >> 16) & 1U;
    u += 0x7FFFU + lsb;
    return (uint16_t)(u >> 16);
}

static inline float host_bf16_to_f32(uint16_t bf) {
    uint32_t u = ((uint32_t)bf) << 16;
    return __uint_as_float(u);
}

static inline uint32_t host_cvt_f32x2_to_bf162(float f0, float f1) {
    uint16_t lo = host_f32_to_bf16_rn(f0);
    uint16_t hi = host_f32_to_bf16_rn(f1);
    return ((uint32_t)lo) | (((uint32_t)hi) << 16);
}

// Correctly rounded a * b + c in BF16 with ONE rounding (docs/12 4.24). The previous form, bf16_rn(fmaf(a, b, c)),
// rounds twice (exact -> FP32 -> BF16): when the FP32 result lands exactly on a BF16 tie that the exact value only
// approaches (a tie-valued product plus an addend below the FP32 ulp), the second rounding goes to even instead of
// toward the addend. 562 of 65,536 sampled operand triples differed. Method: the product of two BF16 values has a
// 16-bit significand, so it is exact in double; the sum p + c is formed with round-to-odd (TwoSum error term, then
// force the sticky bit), and a round-to-odd double (53 bits >= 8 + 2) rounds to the same BF16 value as the exact
// sum (Boldo and Melquiond, "Emulation of FMA and correctly rounded sums", 2008). Non-finite operands keep the
// IEEE result of fmaf.
static inline uint16_t host_bf16_fma_once(float a, float b, float c) {
    if (!isfinite(a) || !isfinite(b) || !isfinite(c)) {
        return host_f32_to_bf16_rn(fmaf(a, b, c));
    }
    const double p = (double)a * (double)b;
    const double cd = (double)c;
    double hi = p + cd;
    if (hi == 0.0) {
        return host_f32_to_bf16_rn((float)hi);            // exact zero: the IEEE sign of the sum is the sign fma returns
    }
    const double bb = hi - p;
    const double err = (p - (hi - bb)) + (cd - bb);   // TwoSum: p + cd == hi + err exactly
    if (err != 0.0) {
        uint64_t hb;
        memcpy(&hb, &hi, sizeof(hb));
        if ((hb & 1ULL) == 0ULL) {
            hi = nextafter(hi, (err > 0.0) ? INFINITY : -INFINITY);   // the odd neighbour on the side of the exact sum
        }
    }
    int e = 0;
    (void)frexp(hi, &e);
    const int ulp_exp = (e - 8 < -133) ? -133 : (e - 8);              // BF16 quantum: normal e - 8, subnormal 2^-133
    const double q = nearbyint(ldexp(hi, -ulp_exp));                  // the single round-to-nearest-even
    return host_f32_to_bf16_rn((float)ldexp(q, ulp_exp));                           // exact in FP32 and BF16 (or overflows to Inf)
}

static inline uint32_t host_ptx_fma_rn_bf16x2(uint32_t a, uint32_t b, uint32_t c) {
    float a_lo = host_bf16_to_f32((uint16_t)(a & 0xFFFFU));
    float a_hi = host_bf16_to_f32((uint16_t)(a >> 16));
    float b_lo = host_bf16_to_f32((uint16_t)(b & 0xFFFFU));
    float b_hi = host_bf16_to_f32((uint16_t)(b >> 16));
    float c_lo = host_bf16_to_f32((uint16_t)(c & 0xFFFFU));
    float c_hi = host_bf16_to_f32((uint16_t)(c >> 16));

    uint16_t r_lo = host_bf16_fma_once(a_lo, b_lo, c_lo);
    uint16_t r_hi = host_bf16_fma_once(a_hi, b_hi, c_hi);
    return ((uint32_t)r_lo) | (((uint32_t)r_hi) << 16);
}

static inline uint32_t host_ptx_lop3_b32(uint32_t a, uint32_t b, uint32_t c, uint8_t imm_lut) {
    if (imm_lut == 0xAEU) {
        return (~a & b) | c;
    }
    if (imm_lut == 0xEAU) {
        return (a & b) | c;
    }
    uint32_t out = 0U;
    for (int bit = 0; bit < 32; ++bit) {
        uint32_t idx = (((a >> bit) & 1U) << 2) |
                       (((b >> bit) & 1U) << 1) |
                       ((c >> bit) & 1U);
        if ((imm_lut >> idx) & 1U) {
            out |= (1U << bit);
        }
    }
    return out;
}

static inline void host_unpack_w1_b_frag(
    uint32_t& rb0,
    uint32_t& rb1,
    const uint32_t w_interleaved,
    const uint32_t shift_p0,
    const uint32_t shift_p1,
    const uint32_t scale_bf162
) {
    const uint32_t nz = 0x80008000U;
    uint32_t tmp0 = w_interleaved << shift_p0;
    tmp0 = host_ptx_lop3_b32(tmp0, 0x80008000U, 0x3F803F80U, 0xAEU);
    rb0 = host_ptx_fma_rn_bf16x2(tmp0, scale_bf162, nz);

    uint32_t tmp1 = w_interleaved << shift_p1;
    tmp1 = host_ptx_lop3_b32(tmp1, 0x80008000U, 0x3F803F80U, 0xAEU);
    rb1 = host_ptx_fma_rn_bf16x2(tmp1, scale_bf162, nz);
}

static inline void host_unpack_w2_ternary_pair(
    uint32_t& rb_out,
    const uint32_t w_tern_interleaved,
    const uint32_t shift_2k,
    const uint32_t scale_bf162
) {
    const uint32_t cm = 0x42804280U; // (64.0, 64.0)
    const uint32_t cs = 0xC301C301U; // (-129.0, -129.0)
    const uint32_t nz = 0x80008000U; // (-0.0, -0.0)
    uint32_t tmp = w_tern_interleaved >> shift_2k;
    tmp = host_ptx_lop3_b32(tmp, 0x00030003U, 0x40004000U, 0xEAU);
    tmp = host_ptx_fma_rn_bf16x2(tmp, cm, cs);
    rb_out = host_ptx_fma_rn_bf16x2(tmp, scale_bf162, nz);
}

// ==============================================================================
// 4. Deterministic 2-Level (Warp-First, CTA-Second) Fiber Execution Engine
// ==============================================================================
static constexpr int HOST_MAX_CTA_THREADS = 256;
static constexpr int HOST_WARP_SIZE = 32;
static constexpr int HOST_MAX_WARPS = HOST_MAX_CTA_THREADS / HOST_WARP_SIZE;

enum HostCollectiveKind : int {
    HOST_COLL_NONE = 0,
    HOST_COLL_LDMATRIX = 1,
    HOST_COLL_MMA = 2,
    HOST_COLL_SHFL = 3,
    HOST_COLL_BALLOT = 4
};

struct HostWarpCollectiveSlot {
    int kind;
    // ldmatrix.sync.aligned.m8n8.x4.shared.b16 state
    uint32_t ldm_smem_addr[HOST_WARP_SIZE];
    uint32_t ldm_out[HOST_WARP_SIZE][4];
    // mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 state
    uint32_t mma_a[HOST_WARP_SIZE][4];
    uint32_t mma_b[HOST_WARP_SIZE][2];
    float mma_d[HOST_WARP_SIZE][4];
    // __shfl_xor_sync state
    float shfl_in[HOST_WARP_SIZE];
    float shfl_out[HOST_WARP_SIZE];
    // __ballot_sync state
    int ballot_pred[HOST_WARP_SIZE];
    uint32_t ballot_mask_out;
};

#ifdef _WIN32
static LPVOID g_main_fiber = nullptr;
static LPVOID g_cta_fibers[HOST_MAX_CTA_THREADS] = {nullptr};
static bool g_fibers_initialized = false;
#endif

static int g_active_cta_threads = 256;
static int g_current_tid = 0;
static uint64_t g_lane_epoch[HOST_MAX_CTA_THREADS] = {0};
static uint64_t g_cta_barrier_epoch[HOST_MAX_CTA_THREADS] = {0};
static HostWarpCollectiveSlot g_warp_slots[HOST_MAX_WARPS][2];
static void (*g_active_kernel_body)(void) = nullptr;

#ifdef _WIN32
static inline void host_switch_to_tid(int next_tid) {
    g_current_tid = next_tid;
    threadIdx.x = next_tid;
    threadIdx.y = 0;
    threadIdx.z = 0;
    SwitchToFiber(g_cta_fibers[next_tid]);
}

static VOID WINAPI host_cta_fiber_proc(LPVOID param) {
    const int tid = (int)(intptr_t)param;
    while (true) {
        g_active_kernel_body();
        if (tid + 1 < g_active_cta_threads) {
            host_switch_to_tid(tid + 1);
        } else {
            SwitchToFiber(g_main_fiber);
        }
    }
}

static inline void host_init_cta_fibers(void) {
    if (g_fibers_initialized) {
        return;
    }
    g_main_fiber = ConvertThreadToFiber(nullptr);
    if (g_main_fiber == nullptr && GetLastError() == ERROR_ALREADY_FIBER) {
        g_main_fiber = GetCurrentFiber();
    }
    assert(g_main_fiber != nullptr);
    for (int t = 0; t < HOST_MAX_CTA_THREADS; ++t) {
        g_cta_fibers[t] = CreateFiber(65536, host_cta_fiber_proc, (LPVOID)(intptr_t)t);
        assert(g_cta_fibers[t] != nullptr);
    }
    g_fibers_initialized = true;
}
#else
static ucontext_t g_main_ctx;
static ucontext_t g_cta_ctx[HOST_MAX_CTA_THREADS];
alignas(16) static uint8_t g_cta_stacks[HOST_MAX_CTA_THREADS][65536];
static bool g_fibers_initialized = false;
static bool g_in_fiber = false;

static inline void host_switch_to_tid(int next_tid) {
    ucontext_t* from_ctx = g_in_fiber ? &g_cta_ctx[g_current_tid] : &g_main_ctx;
    g_in_fiber = true;
    g_current_tid = next_tid;
    threadIdx.x = next_tid;
    threadIdx.y = 0;
    threadIdx.z = 0;
    int rc = swapcontext(from_ctx, &g_cta_ctx[next_tid]);
    assert(rc == 0);
    (void)rc;
}

static void host_cta_fiber_proc_posix(void) {
    const int tid = g_current_tid;
    while (true) {
        g_active_kernel_body();
        if (tid + 1 < g_active_cta_threads) {
            host_switch_to_tid(tid + 1);
        } else {
            g_in_fiber = false;
            int rc = swapcontext(&g_cta_ctx[tid], &g_main_ctx);
            assert(rc == 0);
            (void)rc;
        }
    }
}

static inline void host_init_cta_fibers(void) {
    if (g_fibers_initialized) {
        return;
    }
    for (int t = 0; t < HOST_MAX_CTA_THREADS; ++t) {
        int rc = getcontext(&g_cta_ctx[t]);
        assert(rc == 0);
        (void)rc;
        g_cta_ctx[t].uc_stack.ss_sp = g_cta_stacks[t];
        g_cta_ctx[t].uc_stack.ss_size = sizeof(g_cta_stacks[t]);
        g_cta_ctx[t].uc_link = &g_main_ctx;
        makecontext(&g_cta_ctx[t], host_cta_fiber_proc_posix, 0);
    }
    g_fibers_initialized = true;
}
#endif

// CTA-wide barrier: cycles tid 0 -> 1 -> ... -> g_active_cta_threads - 1 -> 0
static inline void __syncthreads(void) {
    const int tid = g_current_tid;
    g_cta_barrier_epoch[tid]++;
    const int next_tid = (tid + 1 < g_active_cta_threads) ? (tid + 1) : 0;
    host_switch_to_tid(next_tid);
}

// Warp-level transition: cycles lane 0 -> 1 -> ... -> 31 -> 0 within warp_id
static inline void host_warp_collective_step(int warp_id, int lane_id) {
    const int next_lane = (lane_id + 1) & 31;
    host_switch_to_tid(warp_id * HOST_WARP_SIZE + next_lane);
}

// Warp-level ldmatrix.sync.aligned.m8n8.x4.shared.b16 emulation
static inline void host_ldmatrix_x4_bf16(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t smem_addr
) {
    const int tid = g_current_tid;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int slot = (int)(g_lane_epoch[tid]++ & 1ULL);
    HostWarpCollectiveSlot& ws = g_warp_slots[warp_id][slot];

    if (lane_id == 0) {
        ws.kind = HOST_COLL_LDMATRIX;
    } else {
        assert(ws.kind == HOST_COLL_LDMATRIX && "Intra-warp collective divergence in ldmatrix");
    }
    ws.ldm_smem_addr[lane_id] = smem_addr;

    if (lane_id == 31) {
        assert(g_smem_base_ptr != nullptr);
        uint32_t words[HOST_WARP_SIZE][4];
        for (int L = 0; L < HOST_WARP_SIZE; ++L) {
            memcpy(words[L], g_smem_base_ptr + ws.ldm_smem_addr[L], 16);
        }
        for (int L = 0; L < HOST_WARP_SIZE; ++L) {
            const int group_id = L >> 2;
            const int t = L & 3;
            ws.ldm_out[L][0] = words[0  + group_id][t];
            ws.ldm_out[L][1] = words[8  + group_id][t];
            ws.ldm_out[L][2] = words[16 + group_id][t];
            ws.ldm_out[L][3] = words[24 + group_id][t];
        }
    }

    host_warp_collective_step(warp_id, lane_id);

    r0 = ws.ldm_out[lane_id][0];
    r1 = ws.ldm_out[lane_id][1];
    r2 = ws.ldm_out[lane_id][2];
    r3 = ws.ldm_out[lane_id][3];
}

// Warp-level mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 emulation
static inline void host_mma_m16n8k16_bf16_f32(
    float& d0, float& d1, float& d2, float& d3,
    const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1
) {
    const int tid = g_current_tid;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int slot = (int)(g_lane_epoch[tid]++ & 1ULL);
    HostWarpCollectiveSlot& ws = g_warp_slots[warp_id][slot];

    if (lane_id == 0) {
        ws.kind = HOST_COLL_MMA;
    } else {
        assert(ws.kind == HOST_COLL_MMA && "Intra-warp collective divergence in mma.m16n8k16");
    }

    ws.mma_a[lane_id][0] = a0;
    ws.mma_a[lane_id][1] = a1;
    ws.mma_a[lane_id][2] = a2;
    ws.mma_a[lane_id][3] = a3;
    ws.mma_b[lane_id][0] = b0;
    ws.mma_b[lane_id][1] = b1;
    ws.mma_d[lane_id][0] = d0;
    ws.mma_d[lane_id][1] = d1;
    ws.mma_d[lane_id][2] = d2;
    ws.mma_d[lane_id][3] = d3;

    if (lane_id == 31) {
        float A_mat[16][16];
        float B_mat[16][8];
        float C_mat[16][8];

        for (int L = 0; L < HOST_WARP_SIZE; ++L) {
            const int group_id = L >> 2;
            const int t = L & 3;
            const int row_top = group_id;
            const int row_bot = group_id + 8;

            A_mat[row_top][2 * t + 0] = host_bf16_to_f32((uint16_t)(ws.mma_a[L][0] & 0xFFFFU));
            A_mat[row_top][2 * t + 1] = host_bf16_to_f32((uint16_t)(ws.mma_a[L][0] >> 16));
            A_mat[row_bot][2 * t + 0] = host_bf16_to_f32((uint16_t)(ws.mma_a[L][1] & 0xFFFFU));
            A_mat[row_bot][2 * t + 1] = host_bf16_to_f32((uint16_t)(ws.mma_a[L][1] >> 16));
            A_mat[row_top][2 * t + 8] = host_bf16_to_f32((uint16_t)(ws.mma_a[L][2] & 0xFFFFU));
            A_mat[row_top][2 * t + 9] = host_bf16_to_f32((uint16_t)(ws.mma_a[L][2] >> 16));
            A_mat[row_bot][2 * t + 8] = host_bf16_to_f32((uint16_t)(ws.mma_a[L][3] & 0xFFFFU));
            A_mat[row_bot][2 * t + 9] = host_bf16_to_f32((uint16_t)(ws.mma_a[L][3] >> 16));

            const int col_n = group_id;
            B_mat[2 * t + 0][col_n] = host_bf16_to_f32((uint16_t)(ws.mma_b[L][0] & 0xFFFFU));
            B_mat[2 * t + 1][col_n] = host_bf16_to_f32((uint16_t)(ws.mma_b[L][0] >> 16));
            B_mat[2 * t + 8][col_n] = host_bf16_to_f32((uint16_t)(ws.mma_b[L][1] & 0xFFFFU));
            B_mat[2 * t + 9][col_n] = host_bf16_to_f32((uint16_t)(ws.mma_b[L][1] >> 16));

            C_mat[row_top][2 * t + 0] = ws.mma_d[L][0];
            C_mat[row_top][2 * t + 1] = ws.mma_d[L][1];
            C_mat[row_bot][2 * t + 0] = ws.mma_d[L][2];
            C_mat[row_bot][2 * t + 1] = ws.mma_d[L][3];
        }

        for (int m = 0; m < 16; ++m) {
            for (int n = 0; n < 8; ++n) {
                float acc = C_mat[m][n];
                for (int k = 0; k < 16; ++k) {
                    acc += A_mat[m][k] * B_mat[k][n];
                }
                C_mat[m][n] = acc;
            }
        }

        for (int L = 0; L < HOST_WARP_SIZE; ++L) {
            const int group_id = L >> 2;
            const int t = L & 3;
            const int row_top = group_id;
            const int row_bot = group_id + 8;
            ws.mma_d[L][0] = C_mat[row_top][2 * t + 0];
            ws.mma_d[L][1] = C_mat[row_top][2 * t + 1];
            ws.mma_d[L][2] = C_mat[row_bot][2 * t + 0];
            ws.mma_d[L][3] = C_mat[row_bot][2 * t + 1];
        }
    }

    host_warp_collective_step(warp_id, lane_id);

    d0 = ws.mma_d[lane_id][0];
    d1 = ws.mma_d[lane_id][1];
    d2 = ws.mma_d[lane_id][2];
    d3 = ws.mma_d[lane_id][3];
}

// Warp-level __shfl_xor_sync emulation
static inline float __shfl_xor_sync(uint32_t mask, float val, int lane_mask) {
    (void)mask;
    const int tid = g_current_tid;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int slot = (int)(g_lane_epoch[tid]++ & 1ULL);
    HostWarpCollectiveSlot& ws = g_warp_slots[warp_id][slot];

    if (lane_id == 0) {
        ws.kind = HOST_COLL_SHFL;
    } else {
        assert(ws.kind == HOST_COLL_SHFL && "Intra-warp collective divergence in __shfl_xor_sync");
    }
    ws.shfl_in[lane_id] = val;

    if (lane_id == 31) {
        for (int L = 0; L < HOST_WARP_SIZE; ++L) {
            ws.shfl_out[L] = ws.shfl_in[L ^ lane_mask];
        }
    }

    host_warp_collective_step(warp_id, lane_id);
    return ws.shfl_out[lane_id];
}

// Warp-level __ballot_sync emulation
static inline uint32_t __ballot_sync(uint32_t mask, int pred) {
    const int tid = g_current_tid;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int slot = (int)(g_lane_epoch[tid]++ & 1ULL);
    HostWarpCollectiveSlot& ws = g_warp_slots[warp_id][slot];

    if (lane_id == 0) {
        ws.kind = HOST_COLL_BALLOT;
    } else {
        assert(ws.kind == HOST_COLL_BALLOT && "Intra-warp collective divergence in __ballot_sync");
    }
    ws.ballot_pred[lane_id] = (pred != 0) ? 1 : 0;

    if (lane_id == 31) {
        uint32_t bmask = 0U;
        for (int L = 0; L < HOST_WARP_SIZE; ++L) {
            if (ws.ballot_pred[L]) {
                bmask |= (1U << L);
            }
        }
        ws.ballot_mask_out = bmask & mask;
    }

    host_warp_collective_step(warp_id, lane_id);
    return ws.ballot_mask_out;
}

// Host Grid + CTA Launcher
template <typename KernelLambda>
static void host_launch_grid_cta(dim3 grid, int cta_threads, KernelLambda kernel_lambda) {
    assert(cta_threads > 0 && cta_threads <= HOST_MAX_CTA_THREADS && (cta_threads % 32 == 0));
    host_init_cta_fibers();

    static KernelLambda* s_lambda_ptr = nullptr;
    s_lambda_ptr = &kernel_lambda;

    g_active_cta_threads = cta_threads;
    blockDim = dim3(cta_threads, 1, 1);
    gridDim = grid;
    g_active_kernel_body = +[]() {
        (*s_lambda_ptr)();
    };

    for (int bz = 0; bz < grid.z; ++bz) {
        for (int by = 0; by < grid.y; ++by) {
            for (int bx = 0; bx < grid.x; ++bx) {
                blockIdx = dim3(bx, by, bz);
                g_smem_base_ptr = nullptr;
                memset(g_lane_epoch, 0, sizeof(g_lane_epoch));
                memset(g_cta_barrier_epoch, 0, sizeof(g_cta_barrier_epoch));

                host_switch_to_tid(0);

                // Verify all threads in the CTA hit the exact same number of __syncthreads() barriers
                const uint64_t expected_barriers = g_cta_barrier_epoch[0];
                for (int t = 1; t < cta_threads; ++t) {
                    assert(g_cta_barrier_epoch[t] == expected_barriers &&
                           "CTA __syncthreads() barrier divergence detected");
                }
            }
        }
    }
}

#endif // W1A16_HOST_CTA_SHIM_HPP
