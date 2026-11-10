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
#include <pthread.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#endif

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#define HUGE_PAGE_1GB UHA_HUGEPAGE_1GB_BYTES
#define MAP_HUGE_1GB  (30 << MAP_HUGE_SHIFT)

#define UHA_RING_ALLOC_MMAP_1GB   (1U << 0)
#define UHA_RING_ALLOC_PAGELOCKED (1U << 1)
#define UHA_RING_ALLOC_MANUAL_ALIGN (1U << 2)   // generic fallback: over-allocated malloc, original pointer stored below
#define UHA_RING_SLOTS_SHIFT      16U
#define UHA_RING_MAX_SLOTS        0xFFFFU       // slot count is stored in bits [31:16] of alloc_via_mmap

// Verification switches (host only): exercise the non-Win/Linux code paths on any host (docs/12 4.3 / 4.15).
//   UHA_CHORUS_FORCE_SINGLE_THREAD : no worker thread; the producer drives the consumer inline.
//   UHA_CHORUS_FORCE_GENERIC_ALLOC : use the portable over-allocated aligned allocator.
#if defined(UHA_CHORUS_FORCE_SINGLE_THREAD) || (!defined(_WIN32) && !defined(__linux__))
#define UHA_CHORUS_SINGLE_THREADED 1
#endif

typedef struct __attribute__((aligned(UHA_CPU_CACHELINE_BYTES))) {
    // Cache Line 0: Producer (GPU CE / Host launch callback)
    _Atomic uint64_t gpu_ready_seq;
    uint8_t          pad_producer[UHA_CPU_CACHELINE_BYTES - sizeof(_Atomic uint64_t)];

    // Cache Line 1: Consumer (Host AMX worker threads) + Dispatch Metadata
    _Atomic uint64_t cpu_done_seq;
    uint32_t         num_tokens;                    // M_active (e.g., 1..16 during speculative decode)
    uint32_t         hidden_dim;                    // D (e.g., 4096)
    uint32_t         alloc_via_mmap;                // Bits [1:0]: mmap/pagelock flags; Bits [31:16]: ring slot count
    uint8_t          pad_consumer[UHA_CPU_CACHELINE_BYTES - sizeof(_Atomic uint64_t) - 3 * sizeof(uint32_t)];

    // Cache Line 2+: Payload Buffers
    uint32_t         expert_ids[UHA_MOE_MAX_BATCH_TOKENS][UHA_MOE_TOP_K_EXPERTS];
    float            expert_weights[UHA_MOE_MAX_BATCH_TOKENS][UHA_MOE_TOP_K_EXPERTS];
    uint16_t         input_hidden_bf16[UHA_MOE_MAX_BATCH_TOKENS * UHA_MOE_DEFAULT_HIDDEN_DIM];  // 128 KB max
    uint16_t         output_hidden_bf16[UHA_MOE_MAX_BATCH_TOKENS * UHA_MOE_DEFAULT_HIDDEN_DIM]; // 128 KB max
} HeteroMoERingSlot;

// docs/12 4.4: the slot count is packed into 16 bits on free. With the 1 GB cap this can never truncate as long as
// 1 GB / sizeof(slot) <= 65535 (today: 1073741824 / 263296 = 4078). If the slot ever shrinks below 16 KiB this
// fails at compile time instead of silently freeing the wrong length.
_Static_assert(HUGE_PAGE_1GB / sizeof(HeteroMoERingSlot) <= UHA_RING_MAX_SLOTS,
               "ring slot count no longer fits the 16-bit tag in alloc_via_mmap; widen the tag");
_Static_assert(sizeof(HeteroMoERingSlot) % UHA_CPU_CACHELINE_BYTES == 0,
               "slot size must be a multiple of the cache line so every slot stays 64-byte aligned");

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
// Attempts OS page-locking (mlock on Linux, VirtualLock on Windows) to prevent swap-out.
// NOTE: CUDA DMA registration (cudaHostRegister) is performed separately by the GPU host runtime.
HeteroMoERingSlot* allocate_pinned_1gb_ring(size_t num_slots) {
    // Reject sizes whose byte count overflows size_t or exceeds the single 1 GB mapping (the memset below
    // would otherwise write past the end of the mapping)
    if (num_slots == 0 || num_slots > SIZE_MAX / sizeof(HeteroMoERingSlot)) {
        return NULL;
    }
    size_t bytes = num_slots * sizeof(HeteroMoERingSlot);
    if (bytes > HUGE_PAGE_1GB) return NULL;
    if (num_slots > UHA_RING_MAX_SLOTS) return NULL;   // runtime twin of the static assert (docs/12 4.4)
    uint32_t slot_count_tag = ((uint32_t)num_slots & 0xFFFFU) << UHA_RING_SLOTS_SHIFT;
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
        uint32_t flags = UHA_RING_ALLOC_MMAP_1GB | slot_count_tag;
        if (mlock(ptr, bytes) == 0) {
            flags |= UHA_RING_ALLOC_PAGELOCKED;
        }
        slots[0].alloc_via_mmap = flags;
        return slots;
    }
#endif
    HeteroMoERingSlot* fallback = NULL;
    uint32_t alloc_kind = 0U;
#if defined(_WIN32) && !defined(UHA_CHORUS_FORCE_GENERIC_ALLOC)
    fallback = (HeteroMoERingSlot*)_aligned_malloc(bytes, UHA_CPU_CACHELINE_BYTES);
    if (fallback != NULL) {
        memset(fallback, 0, bytes);
    }
#elif (defined(__linux__) || defined(__unix__)) && !defined(UHA_CHORUS_FORCE_GENERIC_ALLOC)
    void* aligned_ptr = NULL;
    if (posix_memalign(&aligned_ptr, UHA_CPU_CACHELINE_BYTES, bytes) == 0 && aligned_ptr != NULL) {
        memset(aligned_ptr, 0, bytes);
        fallback = (HeteroMoERingSlot*)aligned_ptr;
    }
#else
    // Portable 64-byte aligned allocation (docs/12 4.15). calloc only guarantees alignof(max_align_t) (16 B), which
    // is less than the struct's aligned(64): the old code returned a misaligned ring, so _Atomic accesses through it
    // were UB and the cache-line split was lost. This path is taken on macOS/BSD-without-__unix__ and any other
    // host. Over-allocate, round up to the cache line, and keep the original pointer in the word just below.
    {
        const size_t a = (size_t)UHA_CPU_CACHELINE_BYTES;
        if (bytes <= SIZE_MAX - a - sizeof(void*)) {
            unsigned char* raw = (unsigned char*)malloc(bytes + a + sizeof(void*));
            if (raw != NULL) {
                uintptr_t base = (uintptr_t)(raw + sizeof(void*));
                uintptr_t aligned = (base + (uintptr_t)(a - 1U)) & ~(uintptr_t)(a - 1U);
                ((void**)aligned)[-1] = raw;
                memset((void*)aligned, 0, bytes);
                fallback = (HeteroMoERingSlot*)aligned;
                alloc_kind = UHA_RING_ALLOC_MANUAL_ALIGN;
            }
        }
    }
#endif
    if (fallback != NULL) {
        assert(((uintptr_t)fallback % UHA_CPU_CACHELINE_BYTES) == 0U);
        uint32_t flags = slot_count_tag | alloc_kind;
#if defined(__linux__)
        if (mlock((void*)fallback, bytes) == 0) {
            flags |= UHA_RING_ALLOC_PAGELOCKED;
        }
#elif defined(_WIN32)
        SIZE_T min_ws = 0, max_ws = 0;
        HANDLE h_proc = GetCurrentProcess();
        if (GetProcessWorkingSetSize(h_proc, &min_ws, &max_ws)) {
            SIZE_T needed_min = min_ws + (SIZE_T)bytes + (1024U * 1024U);
            SIZE_T needed_max = (max_ws > needed_min) ? max_ws : (needed_min + (4U * 1024U * 1024U));
            (void)SetProcessWorkingSetSize(h_proc, needed_min, needed_max);
        }
        if (VirtualLock((LPVOID)fallback, bytes) != 0) {
            flags |= UHA_RING_ALLOC_PAGELOCKED;
        }
#endif
        fallback[0].alloc_via_mmap = flags;
    }
    return fallback;
}

void free_pinned_1gb_ring(HeteroMoERingSlot* ring) {
    if (!ring) return;
    size_t num_slots = (size_t)(ring[0].alloc_via_mmap >> UHA_RING_SLOTS_SHIFT);
    if (num_slots == 0) num_slots = 1;
    size_t bytes = num_slots * sizeof(HeteroMoERingSlot);
#if defined(__linux__) && defined(MAP_HUGETLB)
    if (ring[0].alloc_via_mmap & UHA_RING_ALLOC_MMAP_1GB) {
        if (ring[0].alloc_via_mmap & UHA_RING_ALLOC_PAGELOCKED) {
            (void)munlock((void*)ring, bytes);
        }
        munmap((void*)ring, HUGE_PAGE_1GB);
        return;
    }
#endif
    if (ring[0].alloc_via_mmap & UHA_RING_ALLOC_MANUAL_ALIGN) {
        // The portable allocator can still be page-locked when forced on Linux/Windows: unlock before freeing.
        if (ring[0].alloc_via_mmap & UHA_RING_ALLOC_PAGELOCKED) {
#if defined(__linux__)
            (void)munlock((void*)ring, bytes);
#elif defined(_WIN32)
            (void)VirtualUnlock((LPVOID)ring, bytes);
#endif
        }
        free(((void**)ring)[-1]);
        return;
    }
#if defined(__linux__)
    if (ring[0].alloc_via_mmap & UHA_RING_ALLOC_PAGELOCKED) {
        (void)munlock((void*)ring, bytes);
    }
    free(ring);
#elif defined(_WIN32)
    if (ring[0].alloc_via_mmap & UHA_RING_ALLOC_PAGELOCKED) {
        (void)VirtualUnlock((LPVOID)ring, bytes);
    }
    _aligned_free(ring);
#else
    free(ring);
#endif
}

static inline float bf16_bits_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

// Intel AMX TDPBF16PS make_fp32(bf16): flushes input BF16 denormals (exp == 0, mant != 0) to signed zero.
// Reference: Intel Architecture Instruction Set Extensions Programming Reference, Section "TDPBF16PS".
static inline float bf16_to_fp32_amx_ftz(uint16_t b) {
    if ((b & 0x7F80U) == 0U) {
        b &= 0x8000U; // Preserve sign bit, flush subnormal significand to zero
    }
    return bf16_bits_to_fp32(b);
}

// Intel AMX TDPBF16PS output FP32 denormal flush-to-zero
static inline float fp32_amx_ftz(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(uint32_t));
    if ((u & 0x7F800000U) == 0U) {
        u &= 0x80000000U;
        memcpy(&f, &u, sizeof(float));
    }
    return f;
}

static inline uint16_t fp32_to_bf16_rne(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(uint32_t));
    if ((u & 0x7F800000U) == 0x7F800000U) {
        return (uint16_t)((u >> 16) | ((u & 0x007FFFFFU) ? 0x0040U : 0U));
    }
    uint32_t lsb = (u >> 16) & 1U;
    u += 0x00007FFFU + lsb;
    return (uint16_t)(u >> 16);
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
            // Scatter 16 32-bit VNNI pairs (2xBF16 each) directly to w_vnni_u32[pair * 16 + r] (fixes O10)
            const __m512i v_stride16 = _mm512_setr_epi32(
                0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240
            );
            _mm512_i32scatter_epi32((void*)(((uint32_t*)w_vnni_buf) + r), v_stride16, unpacked, 4);
        }

        // Load unpacked VNNI BF16 expert weights into tmm6 (16 rows x 64B) and issue TDPBF16PS
        _tile_loadd(6, w_vnni_buf, 64);
        _tile_dpbf16ps(0, 4, 6);
    }
    _tile_stored(0, acc_fp32, 64);
#else
    // Hardware-accurate architectural emulation of AMX _tile_dpbf16ps with VNNI [K/2=16, N=16, 2] B-tile
    // and Intel AMX TDPBF16PS BF16 input/FP32 output denormal flush-to-zero semantics (closes O10).
    _Alignas(64) uint16_t w_vnni_buf[16 * 32];
    memset(acc_fp32, 0, 16 * 16 * sizeof(float));

    for (int k = 0; k < K; k += 32) {
        const int g_idx = k >> UHA_QUANT_GROUP_SHIFT;
        for (int r = 0; r < 16; ++r) {
            const uint16_t pos_sc = scales_bf16 ? scales_bf16[r * groups_per_col + g_idx] : 0x3F80U;
            const uint16_t neg_sc = pos_sc ^ 0x8000U;
            uint32_t sign_mask = w_1bit[(k >> 5) * 16 + r];
            for (int pair = 0; pair < 16; ++pair) {
                const uint16_t b0 = ((sign_mask >> (2 * pair + 0)) & 1U) ? pos_sc : neg_sc;
                const uint16_t b1 = ((sign_mask >> (2 * pair + 1)) & 1U) ? pos_sc : neg_sc;
                w_vnni_buf[pair * 32 + r * 2 + 0] = b0;
                w_vnni_buf[pair * 32 + r * 2 + 1] = b1;
            }
        }
        // TDPBF16PS order of operations, per the Operation pseudocode of the Intel SDM / ISE reference:
        //   for each (m, n): temp_even = temp_odd = 0;
        //     for each of the 16 BF16 pairs: temp_even += a[2k]*b[2k]; temp_odd += a[2k+1]*b[2k+1]
        //     (each an FP32 FMA, RNE, DAZ = FTZ = 1);
        //   then dst = dst + (temp_even + temp_odd), both adds RNE with DAZ = FTZ = 1.
        // The two lanes accumulate independently and dst is added once per instruction (per 32-wide K block).
        // The FMA is emulated in double: a BF16 x BF16 product has a 16-bit significand, so it is exact in double,
        // and the double sum of two values of at most 24 significant bits, rounded once more to FP32, is the
        // correctly rounded FP32 result because 53 >= 2*24 + 2 (Figueroa, "When is double rounding innocuous?",
        // SIGNUM 1995). FTZ is applied to the rounded result. Whether silicon flushes before or after rounding is
        // not stated by the pseudocode and is not modelled; this emulator is not measured against hardware.
        for (int m = 0; m < 16; ++m) {
            for (int r = 0; r < 16; ++r) {
                float c_acc = fp32_amx_ftz(acc_fp32[m * 16 + r]);
                float t_even = 0.0f;
                float t_odd = 0.0f;
                for (int pair = 0; pair < 16; ++pair) {
                    const double a0 = (double)bf16_to_fp32_amx_ftz(act_bf16[m * K + k + 2 * pair + 0]);
                    const double a1 = (double)bf16_to_fp32_amx_ftz(act_bf16[m * K + k + 2 * pair + 1]);
                    const double b0 = (double)bf16_to_fp32_amx_ftz(w_vnni_buf[pair * 32 + r * 2 + 0]);
                    const double b1 = (double)bf16_to_fp32_amx_ftz(w_vnni_buf[pair * 32 + r * 2 + 1]);
                    t_even = fp32_amx_ftz((float)((double)t_even + a0 * b0));
                    t_odd = fp32_amx_ftz((float)((double)t_odd + a1 * b1));
                }
                c_acc = fp32_amx_ftz(c_acc + fp32_amx_ftz(t_even + t_odd));
                acc_fp32[m * 16 + r] = c_acc;
            }
        }
    }
#endif
}

// Asynchronous SPSC Ring Consumer Worker Thread Context (O10)
typedef struct {
    HeteroMoERingSlot* ring;
    size_t             num_slots;
    const uint32_t*    w_1bit;
    const uint16_t*    scales_bf16;
    uint64_t           target_dispatches;
    _Atomic uint64_t   completed_dispatches;
    _Atomic int        stop_requested;
    uint64_t           consumer_next_seq;   // single-threaded mode: next sequence the inline consumer will process
} ChorusAmxWorkerContext;

// Processes one published dispatch `seq` (the caller has already observed gpu_ready_seq >= seq with acquire).
static void chorus_amx_process_slot(ChorusAmxWorkerContext* ctx, uint64_t seq, float* tile_acc) {
    const size_t slot_idx = (size_t)((seq - 1ULL) % (uint64_t)ctx->num_slots);
    HeteroMoERingSlot* slot = &ctx->ring[slot_idx];
    {
        const int K = (int)slot->hidden_dim;
        const int M = (slot->num_tokens <= 16U) ? (int)slot->num_tokens : 16;
        if (K > 0 && K <= (int)UHA_MOE_DEFAULT_HIDDEN_DIM && (K & 31) == 0) {
            amx_gemm_1bit_expert_tile(tile_acc, slot->input_hidden_bf16, ctx->w_1bit, ctx->scales_bf16, K);

            // Apply router gate weight expert_weights[m][0] and store BF16 result into slot->output_hidden_bf16
            for (int m = 0; m < M; ++m) {
                const float gate_w = slot->expert_weights[m][0];
                for (int r = 0; r < 16; ++r) {
                    slot->output_hidden_bf16[m * 16 + r] = fp32_to_bf16_rne(tile_acc[m * 16 + r] * gate_w);
                }
            }
        }

        // Release slot back to producer
        atomic_store_explicit(&slot->cpu_done_seq, seq, memory_order_release);
        atomic_fetch_add_explicit(&ctx->completed_dispatches, 1ULL, memory_order_release);
    }
}

#if !defined(UHA_CHORUS_SINGLE_THREADED)
static void chorus_amx_worker_loop(ChorusAmxWorkerContext* ctx) {
    (void)amx_init_tiles_bf16();
    float tile_acc[16 * 16];

    for (uint64_t seq = 1; seq <= ctx->target_dispatches; ++seq) {
        const size_t slot_idx = (size_t)((seq - 1ULL) % (uint64_t)ctx->num_slots);
        HeteroMoERingSlot* slot = &ctx->ring[slot_idx];

        // Spin-wait with acquire barrier until producer publishes sequence number `seq`
        while (atomic_load_explicit(&slot->gpu_ready_seq, memory_order_acquire) < seq) {
            if (atomic_load_explicit(&ctx->stop_requested, memory_order_relaxed)) {
                amx_release_tiles();
                return;
            }
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
        chorus_amx_process_slot(ctx, seq, tile_acc);
    }

    amx_release_tiles();
}
#endif

// Single-threaded consumer step (docs/12 4.3). Processes every dispatch the producer has already published and
// returns without waiting. The producer calls it whenever it would otherwise block on cpu_done_seq.
// Liveness: at sequence s > n (n = num_slots) the producer needs cpu_done_seq[slot] >= s - n. It has published
// 1..s-1, so this call advances the consumer to s - 1 >= s - n and the wait terminates; the drain loop is the
// same argument with s = total + 1. The old code ran the whole consumer only after the producer loop, so the
// producer blocked forever at s = n + 1.
#if defined(UHA_CHORUS_SINGLE_THREADED)
static void chorus_amx_consume_ready(ChorusAmxWorkerContext* ctx) {
    float tile_acc[16 * 16];
    while (ctx->consumer_next_seq <= ctx->target_dispatches) {
        const uint64_t seq = ctx->consumer_next_seq;
        HeteroMoERingSlot* slot = &ctx->ring[(size_t)((seq - 1ULL) % (uint64_t)ctx->num_slots)];
        if (atomic_load_explicit(&slot->gpu_ready_seq, memory_order_acquire) < seq) {
            return;   // not published yet
        }
        chorus_amx_process_slot(ctx, seq, tile_acc);
        ctx->consumer_next_seq = seq + 1ULL;
    }
}
#endif

#if defined(UHA_CHORUS_SINGLE_THREADED)
// no worker thread entry points in single-threaded mode
#elif defined(_WIN32)
static unsigned __stdcall chorus_amx_win32_thread_entry(void* arg) {
    chorus_amx_worker_loop((ChorusAmxWorkerContext*)arg);
    return 0U;
}
#elif defined(__linux__)
static void* chorus_amx_posix_thread_entry(void* arg) {
    chorus_amx_worker_loop((ChorusAmxWorkerContext*)arg);
    return NULL;
}
#endif

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
    const size_t num_slots = 2;
    HeteroMoERingSlot* ring = allocate_pinned_1gb_ring(num_slots);
    assert(ring != NULL);
    assert(((uintptr_t)ring % UHA_CPU_CACHELINE_BYTES) == 0U);

    for (size_t s = 0; s < num_slots; ++s) {
        atomic_init(&ring[s].gpu_ready_seq, 0ULL);
        atomic_init(&ring[s].cpu_done_seq, 0ULL);
    }

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
            float sc = bf16_to_fp32_amx_ftz(scales_bf16[r]);
            for (int k = 0; k < K; ++k) {
                uint32_t mask = w_1bit[(k >> 5) * 16 + r];
                float w_val = ((mask >> (k & 31)) & 1U) ? sc : -sc;
                float a_val = bf16_to_fp32_amx_ftz(act_bf16[m * K + k]);
                ref += a_val * w_val;
            }
            if (fabsf(acc_fp32[m * 16 + r] - ref) > 1e-4f) {
                vnni_errors++;
            }
        }
    }

    // Test 3 (O10): Intel AMX TDPBF16PS BF16 subnormal input flush-to-zero verification
    uint16_t subnormal_act[16 * 128];
    float subnormal_acc[16 * 16];
    for (int i = 0; i < 16 * K; ++i) {
        // 0x0001 and 0x807F are non-zero BF16 subnormals (exponent == 0, mantissa != 0)
        subnormal_act[i] = (i & 1) ? 0x807FU : 0x0001U;
    }
    amx_gemm_1bit_expert_tile(subnormal_acc, subnormal_act, w_1bit, scales_bf16, K);
    int ftz_errors = 0;
    for (int i = 0; i < 16 * 16; ++i) {
        if (subnormal_acc[i] != 0.0f) {
            ftz_errors++;
        }
    }

    // Test 3b (docs/12 4.17): SDM order of operations. One K=32 block, all weights +1.0, activations
    //   pair 0: a0 = +2^24, a1 = -2^24; pair 1: a0 = +1, a1 = +1; other pairs 0.
    // SDM (two lanes, then lane sum): even = RNE(2^24 + 1) = 2^24 (tie to even), odd = -2^24 + 1 exactly,
    //   result = 1.0. Pair-sum-into-dst c += (a0b0 + a1b1) gives 2.0, sequential (c + a0b0) + a1b1 gives 2.0,
    //   and the exact sum is 2.0. With K = 64 (two instructions, same pattern) dst = 1.0 + 1.0 = 2.0.
    // This checks the emulator against the hand-derived SDM result, not against a reference with the same loop.
    int order_errors = 0;
    {
        const int K2 = 64;
        uint16_t ord_act[16 * 64];
        uint32_t ord_w[(64 / 32) * 16];
        uint16_t ord_sc[16];
        float ord_acc[16 * 16];
        for (int r = 0; r < 16; ++r) ord_sc[r] = 0x3F80U;
        for (int i = 0; i < (K2 / 32) * 16; ++i) ord_w[i] = 0xFFFFFFFFU;
        for (int K_test = 32; K_test <= 64; K_test += 32) {
            memset(ord_act, 0, sizeof(ord_act));
            for (int m = 0; m < 16; ++m) {
                for (int blk = 0; blk < K_test / 32; ++blk) {
                    uint16_t* a = ord_act + m * K_test + blk * 32;
                    a[0] = 0x4B80U; // +2^24
                    a[1] = 0xCB80U; // -2^24
                    a[2] = 0x3F80U; // +1
                    a[3] = 0x3F80U; // +1
                }
            }
            amx_gemm_1bit_expert_tile(ord_acc, ord_act, ord_w, ord_sc, K_test);
            const float expect = (K_test == 32) ? 1.0f : 2.0f;
            for (int i = 0; i < 16 * 16; ++i) {
                if (ord_acc[i] != expect) order_errors++;
            }
        }
    }

    // Test 4 (O10): Concurrent OS Worker Thread SPSC Ring Producer-Consumer Verification (8 dispatches over 2 slots)
    const uint64_t total_dispatches = 8ULL;
    ChorusAmxWorkerContext ctx;
    ctx.ring = ring;
    ctx.num_slots = num_slots;
    ctx.w_1bit = w_1bit;
    ctx.scales_bf16 = scales_bf16;
    ctx.target_dispatches = total_dispatches;
    atomic_init(&ctx.completed_dispatches, 0ULL);
    atomic_init(&ctx.stop_requested, 0);
    ctx.consumer_next_seq = 1ULL;

#if defined(UHA_CHORUS_SINGLE_THREADED)
    (void)amx_init_tiles_bf16();
    printf("[i] Single-threaded ring mode: producer drives the consumer inline (docs/12 4.3)\n");
#elif defined(_WIN32)
    uintptr_t th = _beginthreadex(NULL, 0, chorus_amx_win32_thread_entry, &ctx, 0, NULL);
    assert(th != 0);
#elif defined(__linux__)
    pthread_t th;
    int rc = pthread_create(&th, NULL, chorus_amx_posix_thread_entry, &ctx);
    assert(rc == 0);
#endif

    // Pipelined producer loop: keeps up to `num_slots` dispatches in flight concurrently with the OS consumer thread.
    // Before reusing a slot at `seq > num_slots`, waits for `cpu_done_seq >= prev_seq` and verifies `prev_seq`'s output.
    int ring_thread_errors = 0;
    for (uint64_t seq = 1; seq <= total_dispatches; ++seq) {
        const size_t slot_idx = (size_t)((seq - 1ULL) % (uint64_t)num_slots);
        HeteroMoERingSlot* slot = &ring[slot_idx];

        if (seq > (uint64_t)num_slots) {
            const uint64_t prev_seq = seq - (uint64_t)num_slots;
            while (atomic_load_explicit(&slot->cpu_done_seq, memory_order_acquire) < prev_seq) {
#if defined(UHA_CHORUS_SINGLE_THREADED)
                chorus_amx_consume_ready(&ctx);
#elif defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#endif
            }
            const float prev_gate_scale = 0.125f * (float)prev_seq;
            for (int m = 0; m < 16; ++m) {
                for (int r = 0; r < 16; ++r) {
                    uint16_t expected_bf16 = fp32_to_bf16_rne(acc_fp32[m * 16 + r] * prev_gate_scale);
                    if (slot->output_hidden_bf16[m * 16 + r] != expected_bf16) {
                        ring_thread_errors++;
                    }
                }
            }
        }

        slot->num_tokens = 16U;
        slot->hidden_dim = (uint32_t)K;
        const float gate_scale = 0.125f * (float)seq;
        for (int m = 0; m < 16; ++m) {
            slot->expert_ids[m][0] = (uint32_t)(seq & 7ULL);
            slot->expert_weights[m][0] = gate_scale;
        }
        memcpy(slot->input_hidden_bf16, act_bf16, sizeof(act_bf16));

        // Publish slot to consumer worker thread
        atomic_store_explicit(&slot->gpu_ready_seq, seq, memory_order_release);
    }


    // Drain and verify the final `num_slots` in-flight dispatches
    for (uint64_t seq = total_dispatches - (uint64_t)num_slots + 1ULL; seq <= total_dispatches; ++seq) {
        const size_t slot_idx = (size_t)((seq - 1ULL) % (uint64_t)num_slots);
        HeteroMoERingSlot* slot = &ring[slot_idx];
        while (atomic_load_explicit(&slot->cpu_done_seq, memory_order_acquire) < seq) {
#if defined(UHA_CHORUS_SINGLE_THREADED)
            chorus_amx_consume_ready(&ctx);
#elif defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
        const float gate_scale = 0.125f * (float)seq;
        for (int m = 0; m < 16; ++m) {
            for (int r = 0; r < 16; ++r) {
                uint16_t expected_bf16 = fp32_to_bf16_rne(acc_fp32[m * 16 + r] * gate_scale);
                if (slot->output_hidden_bf16[m * 16 + r] != expected_bf16) {
                    ring_thread_errors++;
                }
            }
        }
    }

#if defined(UHA_CHORUS_SINGLE_THREADED)
    amx_release_tiles();
#elif defined(_WIN32)
    WaitForSingleObject((HANDLE)th, INFINITE);
    CloseHandle((HANDLE)th);
#elif defined(__linux__)
    pthread_join(th, NULL);
#endif

    uint64_t final_completed = atomic_load_explicit(&ctx.completed_dispatches, memory_order_acquire);
    uint64_t last_slot_idx = (total_dispatches - 1ULL) % (uint64_t)num_slots;
    uint64_t delta_q = atomic_load(&ring[last_slot_idx].gpu_ready_seq) - atomic_load(&ring[last_slot_idx].cpu_done_seq);

    printf("[1] HeteroMoERingSlot Size & Alignment : %zu Bytes (Producer@%zu, Consumer@%zu, PageLocked=%u)\n",
           sizeof(HeteroMoERingSlot),
           offsetof(HeteroMoERingSlot, gpu_ready_seq),
           offsetof(HeteroMoERingSlot, cpu_done_seq),
           (ring[0].alloc_via_mmap & UHA_RING_ALLOC_PAGELOCKED) ? 1U : 0U);
    printf("    Ring base %% 64 = %u (%s allocator)\n", (unsigned)((uintptr_t)ring % 64U),
           (ring[0].alloc_via_mmap & UHA_RING_ALLOC_MANUAL_ALIGN) ? "portable over-aligned" : "platform");
    assert(((uintptr_t)ring % 64U) == 0U);
    printf("[2] AMX 16x16x32 VNNI & TDPBF16PS FTZ  : Uniform Err=%d | VNNI Err=%d | Denormal FTZ Err=%d | SDM Order Err=%d\n",
           errors, vnni_errors, ftz_errors, order_errors);
#if defined(UHA_CHORUS_SINGLE_THREADED)
    printf("[3] Single-Threaded SPSC Ring (inline) : Completed=%llu/%llu | Payload Err=%d | Final Delta_Q=%llu\n",
#else
    printf("[3] Multi-Threaded SPSC Ring Worker    : Completed=%llu/%llu | Payload Err=%d | Final Delta_Q=%llu\n",
#endif
           (unsigned long long)final_completed,
           (unsigned long long)total_dispatches,
           ring_thread_errors,
           (unsigned long long)delta_q);
    assert(errors == 0 && vnni_errors == 0 && ftz_errors == 0 && order_errors == 0);
    assert(final_completed == total_dispatches && ring_thread_errors == 0 && delta_q == 0ULL);

    amx_release_tiles();
    free_pinned_1gb_ring(ring);
    printf("[+] PASS: Zero-Copy HugePage Ring, TDPBF16PS FTZ, and Async Multi-Threaded AMX MoE worker verified.\n");
    return 0;
}
#endif
