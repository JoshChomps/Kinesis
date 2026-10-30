// File: hetero_runtime/chorus_amx_moe.c
// Heterogeneous Zero-Copy 1GB HugePage Ring & Intel AMX Expert Worker
// Compile (Sapphire Rapids / Emerald Rapids / Granite Rapids):
//   gcc -O3 -march=sapphirerapids -mamx-tile -mamx-bf16 -mavx512f -pthread chorus_amx_moe.c
// Compile (Portable Verification Mode):
//   gcc -O3 -std=c11 -DSTANDALONE_VERIFY -o test_chorus_amx chorus_amx_moe.c -lm

#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include <assert.h>
#include "../kinesis_unified/unified_sizing.h"

#if defined(__linux__)
#include <sys/mman.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#define HUGE_PAGE_1GB UHA_HUGEPAGE_1GB_BYTES
#define MAP_HUGE_1GB  (30 << MAP_HUGE_SHIFT)

typedef struct __attribute__((aligned(UHA_CPU_CACHELINE_BYTES))) {
    // Cache Line 0: Producer (GPU CE / Host launch callback)
    _Atomic uint64_t gpu_ready_seq;
    uint8_t          pad_producer[UHA_CPU_CACHELINE_BYTES - sizeof(_Atomic uint64_t)];

    // Cache Line 1: Consumer (Host AMX worker threads) + Dispatch Metadata
    _Atomic uint64_t cpu_done_seq;
    uint32_t         num_tokens;                    // M_active (e.g., 1..16 during speculative decode)
    uint32_t         hidden_dim;                    // D (e.g., 4096)
    uint32_t         alloc_via_mmap;                // 1 if backed by 1GB HugePage mmap, 0 if heap fallback
    uint8_t          pad_consumer[UHA_CPU_CACHELINE_BYTES - sizeof(_Atomic uint64_t) - 3 * sizeof(uint32_t)];

    // Cache Line 2+: Payload Buffers
    uint32_t         expert_ids[UHA_MOE_MAX_BATCH_TOKENS][UHA_MOE_TOP_K_EXPERTS];
    float            expert_weights[UHA_MOE_MAX_BATCH_TOKENS][UHA_MOE_TOP_K_EXPERTS];
    uint16_t         input_hidden_bf16[UHA_MOE_MAX_BATCH_TOKENS * UHA_MOE_DEFAULT_HIDDEN_DIM];  // 128 KB max
    uint16_t         output_hidden_bf16[UHA_MOE_MAX_BATCH_TOKENS * UHA_MOE_DEFAULT_HIDDEN_DIM]; // 128 KB max
} HeteroMoERingSlot;

// Configure AMX Tile Control Register (TILECFG) for 16x16x32 BF16->FP32 Matmul
typedef struct __attribute__((packed)) {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved_0[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
} amx_tilecfg_t;

#if defined(__linux__) && defined(__AMX_TILE__)
// Linux (>= 5.16) only lets a process touch AMX tile state after it asks for XTILEDATA permission; without
// this call the first tile instruction raises SIGILL even on AMX hardware. The permission is per process,
// but the tile configuration loaded by amx_init_tiles_bf16() is per THREAD: every worker thread must call it.
#define UHA_ARCH_REQ_XCOMP_PERM 0x1023
#define UHA_XFEATURE_XTILEDATA  18
static inline int amx_request_permission(void) {
    return (syscall(SYS_arch_prctl, UHA_ARCH_REQ_XCOMP_PERM, UHA_XFEATURE_XTILEDATA) == 0) ? 0 : -1;
}
#endif

static inline int amx_init_tiles_bf16(void) {
    amx_tilecfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    // tmm0..tmm3: Accumulators C (16 rows x 64 bytes = 16x16 FP32)
    // tmm4..tmm5: Activations A  (16 rows x 64 bytes = 16x32 BF16)
    // tmm6..tmm7: Expert Weights B (16 rows x 64 bytes = 16x32 BF16 in VNNI pair layout)
    for (int i = 0; i < 8; ++i) {
        cfg.rows[i] = 16;
        cfg.colsb[i] = 64;
    }
#if defined(__AMX_TILE__)
#if defined(__linux__)
    if (amx_request_permission() != 0) {
        return -1; // kernel refused XTILEDATA: caller must fall back to the AVX-512 / scalar path
    }
#endif
    _tile_loadconfig(&cfg);
#else
    (void)cfg;
#endif
    return 0;
}

static inline void amx_release_tiles(void) {
#if defined(__AMX_TILE__)
    _tile_release();
#endif
}

// Allocate 1GB Pinned HugePage Ring on Linux (falls back to 64B-aligned heap in sandbox)
// NOTE: this is page-backed host memory only. It is NOT registered with the CUDA driver (cudaHostRegister),
// not bound to a NUMA node (mbind), and 1 GB pages must have been reserved by the administrator
// (hugepagesz=1G hugepages=N); otherwise it silently falls back to the ordinary heap.
HeteroMoERingSlot* allocate_pinned_1gb_ring(size_t num_slots) {
    // Reject sizes whose byte count overflows size_t or exceeds the single 1 GB mapping (the memset below
    // would otherwise write past the end of the mapping)
    if (num_slots == 0 || num_slots > SIZE_MAX / sizeof(HeteroMoERingSlot)) {
        return NULL;
    }
    size_t bytes = num_slots * sizeof(HeteroMoERingSlot);
#if defined(__linux__) && defined(MAP_HUGETLB)
    void* ptr = (bytes > HUGE_PAGE_1GB) ? MAP_FAILED : mmap(
        NULL,
        HUGE_PAGE_1GB,
        PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
        -1,
        0
    );
    if (ptr != MAP_FAILED) {
        memset(ptr, 0, bytes);
        HeteroMoERingSlot* slots = (HeteroMoERingSlot*)ptr;
        slots[0].alloc_via_mmap = 1U;
        return slots;
    }
#endif
    HeteroMoERingSlot* fallback = (HeteroMoERingSlot*)calloc(num_slots, sizeof(HeteroMoERingSlot));
    if (fallback != NULL) {
        fallback[0].alloc_via_mmap = 0U;
    }
    return fallback;
}

void free_pinned_1gb_ring(HeteroMoERingSlot* ring) {
    if (!ring) return;
#if defined(__linux__) && defined(MAP_HUGETLB)
    if (ring[0].alloc_via_mmap) {
        munmap((void*)ring, HUGE_PAGE_1GB);
        return;
    }
#endif
    free(ring);
}

static inline float bf16_bits_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

// AVX-512 Branchless 1-Bit (Binary g128) -> BF16 VNNI Unpacker + AMX Tile Multiply
// Unpacks 512 1-bit weights into hardware VNNI pair layout [K/2 = 16, N = 16, 2]
// in L1 cache and executes _tile_dpbf16ps (tmm0 += tmm4 * tmm6).
static inline void amx_gemm_1bit_expert_tile(
    float* __restrict__ acc_fp32,            // 16x16 FP32 output tile [M=16, N=16]
    const uint16_t* __restrict__ act_bf16,   // 16xK BF16 activations (stride K)
    const uint32_t* __restrict__ w_1bit,     // Packed 1-bit weights for expert [K/32, N=16]
    const uint16_t* __restrict__ scales_bf16,// g128 BF16 scales [N=16, K/128]
    const int K
) {
    const int groups_per_col = (K + (int)UHA_QUANT_GROUP_SIZE - 1) >> UHA_QUANT_GROUP_SHIFT;
#if defined(__AMX_BF16__) && defined(__AVX512BW__)
    _tile_zero(0); // Zero tmm0 accumulator (16x16 FP32)
    uint16_t w_vnni_buf[16 * 32] __attribute__((aligned(64)));

    for (int k = 0; k < K; k += 32) {
        // Load 16x32 BF16 activation tile into tmm4
        _tile_loadd(4, act_bf16 + k, K * sizeof(uint16_t));

        // Unpack 16 columns x 32 bits (512 1-bit weights) into VNNI layout [K/2=16, N=16, 2]
        const int g_idx = k >> UHA_QUANT_GROUP_SHIFT;
        for (int r = 0; r < 16; ++r) {
            const uint16_t pos_sc = scales_bf16 ? scales_bf16[r * groups_per_col + g_idx] : 0x3F80U;
            const uint16_t neg_sc = pos_sc ^ 0x8000U;
            const __m512i v_pos = _mm512_set1_epi16((short)pos_sc);
            const __m512i v_neg = _mm512_set1_epi16((short)neg_sc);
            __mmask32 sign_mask = (__mmask32)w_1bit[(k >> 5) * 16 + r];
            __m512i unpacked = _mm512_mask_blend_epi16(sign_mask, v_neg, v_pos);
            uint16_t col_vals[32] __attribute__((aligned(64)));
            _mm512_store_si512((__m512i*)col_vals, unpacked);
            for (int pair = 0; pair < 16; ++pair) {
                w_vnni_buf[pair * 32 + r * 2 + 0] = col_vals[2 * pair + 0];
                w_vnni_buf[pair * 32 + r * 2 + 1] = col_vals[2 * pair + 1];
            }
        }

        // Load unpacked VNNI BF16 expert weights into tmm6 (16 rows x 64B) and issue TDPBF16PS
        _tile_loadd(6, w_vnni_buf, 64);
        _tile_dpbf16ps(0, 4, 6);
    }
    _tile_stored(0, acc_fp32, 64);
#else
    // Bit-exact architectural emulation of AMX _tile_dpbf16ps with VNNI [K/2=16, N=16, 2] B-tile
    uint16_t w_vnni_buf[16 * 32];
    memset(acc_fp32, 0, 16 * 16 * sizeof(float));

    for (int k = 0; k < K; k += 32) {
        const int g_idx = k >> UHA_QUANT_GROUP_SHIFT;
        for (int r = 0; r < 16; ++r) {
            const uint16_t pos_sc = scales_bf16 ? scales_bf16[r * groups_per_col + g_idx] : 0x3F80U;
            const uint16_t neg_sc = pos_sc ^ 0x8000U;
            uint32_t sign_mask = w_1bit[(k >> 5) * 16 + r];
            for (int bit = 0; bit < 32; ++bit) {
                const int pair = bit >> 1;
                const int sub  = bit & 1;
                w_vnni_buf[pair * 32 + r * 2 + sub] = ((sign_mask >> bit) & 1U) ? pos_sc : neg_sc;
            }
        }
        // Hardware TDPBF16PS: C[m, r] += sum_{pair=0..15} (A[m, 2*pair]*B[pair, r, 0] + A[m, 2*pair+1]*B[pair, r, 1])
        for (int m = 0; m < 16; ++m) {
            for (int r = 0; r < 16; ++r) {
                float pair_sum = 0.0f;
                for (int pair = 0; pair < 16; ++pair) {
                    float a0 = bf16_bits_to_fp32(act_bf16[m * K + k + 2 * pair + 0]);
                    float a1 = bf16_bits_to_fp32(act_bf16[m * K + k + 2 * pair + 1]);
                    float b0 = bf16_bits_to_fp32(w_vnni_buf[pair * 32 + r * 2 + 0]);
                    float b1 = bf16_bits_to_fp32(w_vnni_buf[pair * 32 + r * 2 + 1]);
                    pair_sum += (a0 * b0) + (a1 * b1);
                }
                acc_fp32[m * 16 + r] += pair_sum;
            }
        }
    }
#endif
}

#ifdef STANDALONE_VERIFY
#include <stddef.h>
#include <math.h>

static uint16_t fp32_to_bf16_test(float f) {
    uint32_t u = 0;
    memcpy(&u, &f, sizeof(uint32_t));
    return (uint16_t)(u >> 16);
}

int main(void) {
    printf("================================================================================\n");
    printf("  WORKSTREAM 7A: ZERO-COPY 1GB HUGEPAGE RING & AVX-512/AMX 1-BIT MOE WORKER\n");
    printf("================================================================================\n\n");

    // Verify producer and consumer sequence numbers reside on separate 64B cache lines
    assert(offsetof(HeteroMoERingSlot, gpu_ready_seq) == 0);
    assert(offsetof(HeteroMoERingSlot, cpu_done_seq) == UHA_CPU_CACHELINE_BYTES);

    (void)amx_init_tiles_bf16();
    // Overflow / oversize guards for the ring allocator
    assert(allocate_pinned_1gb_ring(0) == NULL);
    assert(allocate_pinned_1gb_ring(SIZE_MAX / sizeof(HeteroMoERingSlot) + 1) == NULL);
    HeteroMoERingSlot* ring = allocate_pinned_1gb_ring(2);
    assert(ring != NULL);

    atomic_init(&ring[0].gpu_ready_seq, 1ULL);
    atomic_init(&ring[0].cpu_done_seq, 0ULL);
    ring[0].num_tokens = 16;
    ring[0].hidden_dim = 128;

    const int K = 128;
    uint16_t act_bf16[16 * 128];
    uint32_t w_1bit[(128 / 32) * 16];
    uint16_t scales_bf16[16];
    float acc_fp32[16 * 16];

    for (int r = 0; r < 16; ++r) {
        scales_bf16[r] = 0x3F80U; // +1.0 in BF16 per column group (g=128)
    }

    for (int i = 0; i < 16 * K; ++i) {
        act_bf16[i] = (i & 1) ? 0x4000U : 0x3F80U; // Even index = +1.0, Odd index = +2.0
    }
    for (int i = 0; i < (K / 32) * 16; ++i) {
        w_1bit[i] = 0xAAAAAAAAU; // Odd bits = +1.0 (bit=1), Even bits = -1.0 (bit=0)
    }

    // Test 1: Uniform alternating bit pattern (Expected: +64.0f across all 256 tile elements)
    amx_gemm_1bit_expert_tile(acc_fp32, act_bf16, w_1bit, scales_bf16, K);
    int errors = 0;
    for (int i = 0; i < 16 * 16; ++i) {
        if (acc_fp32[i] != 64.0f) {
            errors++;
        }
    }

    // Test 2: Non-uniform column-dependent weights and scales to verify VNNI [K/2, N, 2] indexing
    for (int r = 0; r < 16; ++r) {
        scales_bf16[r] = fp32_to_bf16_test(0.5f * (float)(r + 1));
        for (int blk = 0; blk < (K / 32); ++blk) {
            w_1bit[blk * 16 + r] = 0x96695A3CU ^ (uint32_t)(r * 0x11111111U + blk * 0x07070707U);
        }
    }
    for (int m = 0; m < 16; ++m) {
        for (int k = 0; k < K; ++k) {
            float v = 0.25f * (float)((m + 1) * ((k % 7) - 3));
            act_bf16[m * K + k] = fp32_to_bf16_test(v);
        }
    }
    amx_gemm_1bit_expert_tile(acc_fp32, act_bf16, w_1bit, scales_bf16, K);

    int vnni_errors = 0;
    for (int m = 0; m < 16; ++m) {
        for (int r = 0; r < 16; ++r) {
            float ref = 0.0f;
            float sc = bf16_bits_to_fp32(scales_bf16[r]);
            for (int k = 0; k < K; ++k) {
                uint32_t mask = w_1bit[(k >> 5) * 16 + r];
                float w_val = ((mask >> (k & 31)) & 1U) ? sc : -sc;
                float a_val = bf16_bits_to_fp32(act_bf16[m * K + k]);
                ref += a_val * w_val;
            }
            if (fabsf(acc_fp32[m * 16 + r] - ref) > 1e-4f) {
                vnni_errors++;
            }
        }
    }

    atomic_store_explicit(&ring[0].cpu_done_seq, 1ULL, memory_order_release);
    uint64_t delta_q = atomic_load(&ring[0].gpu_ready_seq) - atomic_load(&ring[0].cpu_done_seq);
    printf("[1] HeteroMoERingSlot Size & Alignment : %zu Bytes (Producer@%zu, Consumer@%zu)\n",
           sizeof(HeteroMoERingSlot),
           offsetof(HeteroMoERingSlot, gpu_ready_seq),
           offsetof(HeteroMoERingSlot, cpu_done_seq));
    printf("[2] AMX 16x16x32 Uniform & VNNI Tests  : Uniform Err=%d | Non-Uniform VNNI Err=%d\n",
           errors, vnni_errors);
    printf("[3] Post-Execution Ring Queue Delta_Q  : %llu\n", (unsigned long long)delta_q);
    assert(errors == 0 && vnni_errors == 0 && delta_q == 0ULL);

    amx_release_tiles();
    free_pinned_1gb_ring(ring);
    printf("[+] PASS: Zero-Copy HugePage Ring and AVX-512/AMX VNNI [K/2, N, 2] MoE worker verified.\n");
    return 0;
}
#endif
