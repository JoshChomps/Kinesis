// File: kinesis_unified/unified_sizing.h
// Single Source of Truth for Hardware, Cache, Tile, Quantization, and Memory Sizing
// Shared across C11 (sim_harness, hetero_runtime, kernel_driver_patch),
// CUDA/PTX (fused_w1_hadacore, structural_filters), and C++20 (kinesis_unified).

#ifndef UHA_UNIFIED_SIZING_H
#define UHA_UNIFIED_SIZING_H

#include <stddef.h>
#include <stdint.h>

// ==============================================================================
// 1. Hardware Alignment & OS Virtual Memory Granularities (Bytes)
// ==============================================================================
#define UHA_GPU_SECTOR_BYTES             32U
#define UHA_CPU_CACHELINE_BYTES          64U
#define UHA_GPU_SRAM_SWIZZLE_BYTES       128U
#define UHA_TENSOR_VIEW_DESC_BYTES       128U        // Exact 2 x 64B CPU cache lines / 1 x 128B GPU line
#define UHA_SHAPE_DESC_BYTES             56U
#define UHA_STRIDES_DESC_BYTES           56U
#define UHA_MAX_TENSOR_DIMS              6U

#define UHA_OS_PAGE_BYTES                4096ULL     // 4 KB standard virtual page
#define UHA_WIN_ALLOC_GRANULARITY_BYTES  65536ULL    // 64 KB Windows VirtualAlloc reservation granularity
#define UHA_HUGEPAGE_2MB_BYTES           (2ULL << 20)
#define UHA_HUGEPAGE_1GB_BYTES           (1ULL << 30)

// ==============================================================================
// 2. GPU SM SRAM, Register File & BAR1 Aperture Limits (GA102 sm_86 / AD102 sm_89)
// ==============================================================================
#define UHA_SM_REG_FILE_BYTES            262144U     // 256 KB (65,536 x 32-bit registers per SM)
#define UHA_SM_MAX_REGS_PER_THREAD       255U
#define UHA_SM_L1_SRAM_TOTAL_BYTES       131072U     // 128 KB physical L1/Shared SRAM per SM
#define UHA_SM86_SM89_MAX_DYN_SMEM_BYTES 101376U     // 99 KB (99 * 1024 B) max opt-in dynamic Shared Memory per CTA
#define UHA_SM_SRAM_BANKS                32U         // 32 banks x 4B = 128B per crossbar cycle
#define UHA_SM_SRAM_BYTES_PER_CYCLE      128U
#define UHA_SMSP_PER_SM                  4U
#define UHA_WARP_SIZE                    32U

#define UHA_BAR1_STATIC_APERTURE_BYTES   (32ULL << 30) // 32 GB static BAR1 PCIe aperture ceiling
#define UHA_P2P_PAGE_2MB_BYTES           (2U << 20)    // 2 MB GMMU peer page table entry granularity

// ==============================================================================
// 3. Sub-2-Bit Quantization & HadaCore Subspace Block Sizing
// ==============================================================================
#define UHA_QUANT_GROUP_SIZE             128U        // g = 128 elements per BF16 scale factor
#define UHA_QUANT_GROUP_SHIFT            7U          // log2(128) = 7
#define UHA_HADACORE_BLOCK_DIM           16U         // 16 x 16 Sylvester Walsh-Hadamard block

// W1 (Binary g128): 1 bit/weight + 16-bit BF16 scale per 128 weights = 1.125 bits/weight
// Compression: 32 / 1.125 = 28.444x vs FP32, 16 / 1.125 = 14.222x vs FP16
#define UHA_W1_WEIGHTS_PER_U32           32U
#define UHA_W1_WEIGHTS_PER_U32_SHIFT     5U          // log2(32) = 5
#define UHA_W1_BITS_PER_WEIGHT_NUM       9U          // 9 / 8 = 1.125 bits/weight
#define UHA_W1_BITS_PER_WEIGHT_DEN       8U

// W2 (Ternary b1.58 g128): 2 bits/weight + 16-bit BF16 scale per 128 weights = 2.125 bits/weight
// Compression: 32 / 2.125 = 15.059x vs FP32, 16 / 2.125 = 7.529x vs FP16
#define UHA_W2_WEIGHTS_PER_U32           16U
#define UHA_W2_WEIGHTS_PER_U32_SHIFT     4U          // log2(16) = 4
#define UHA_W2_BITS_PER_WEIGHT_NUM       17U         // 17 / 8 = 2.125 bits/weight
#define UHA_W2_BITS_PER_WEIGHT_DEN       8U

// ==============================================================================
// 4. GPU Threadblock (CTA), Warp-Tile & MMA Instruction Geometry
// ==============================================================================
#define UHA_CTA_WARPS                    8U
#define UHA_CTA_THREADS                  256U        // 8 warps x 32 threads
#define UHA_CTA_BLOCK_M                  128U
#define UHA_CTA_BLOCK_N                  256U
#define UHA_CTA_BLOCK_K                  32U         // 4 stages x (128+256)*32*2B = 98,304 B <= 101,376 B (99 KB)
#define UHA_CTA_PIPELINE_STAGES          4U
#define UHA_CTA_SMEM_STAGE_BYTES         24576U      // (128 + 256) * 32 * 2 bytes = 24 KB per stage
#define UHA_CTA_SMEM_4STAGE_BYTES        98304U      // 4 * 24,576 B = 96 KB <= 101,376 B (99 KB)

#define UHA_WARP_TILE_M                  64U         // 4 x m16 slices
#define UHA_WARP_TILE_N                  64U         // 8 x n8 slices
#define UHA_WARP_TILE_K                  32U         // 2 x k16 sub-steps per uint32 weight word
#define UHA_MMA_M                        16U
#define UHA_MMA_N                        8U
#define UHA_MMA_K                        16U

// ==============================================================================
// 5. CPU GotoBLAS / BLIS AVX2 & Intel AMX Tile Geometry
// ==============================================================================
#define UHA_CPU_AVX2_MR                  4U          // 4 rows x 8 cols = 32 FP32 accumulators in 4 YMM regs
#define UHA_CPU_AVX2_NR                  8U          // 8 FP32 lanes per 256-bit YMM register
#define UHA_CPU_AMX_TILE_M               16U         // 16 rows per AMX tmm tile register
#define UHA_CPU_AMX_TILE_N               16U         // 16 FP32 columns (64 bytes) per AMX tmm accumulator
#define UHA_CPU_AMX_TILE_K               32U         // 32 BF16 elements (64 bytes) per AMX VNNI row

// Default fallback CPU cache sizes when running on non-CPUID platforms
#define UHA_DEFAULT_CPU_L1D_BYTES        32768ULL    // 32 KB
#define UHA_DEFAULT_CPU_L2_BYTES         524288ULL   // 512 KB
#define UHA_DEFAULT_CPU_L3_BYTES         16777216ULL // 16 MB

// ==============================================================================
// 6. Paged KV-Cache (SlabAllocator) & LServe Sparse Centroid Sizing
// ==============================================================================
#define UHA_KV_PAGE_TOKENS               64U         // 64 tokens per KV page
#define UHA_KV_DEFAULT_HEAD_DIM          128U        // Standard 128 head dimension (8 x H_16 blocks)
// Packed INT4 KV page: 64 tokens * 128 head_dim * 4 bits / 8 bits/byte = 4,096 bytes (1 OS page!)
#define UHA_KV_PAGE_INT4_BYTES           4096U
// Unpacked FP16/BF16 KV page: 64 tokens * 128 head_dim * 2 bytes = 16,384 bytes (16 KB)
#define UHA_KV_PAGE_BF16_BYTES           16384U
// Per-page min/max BF16 key centroids: 2 * 128 * 2 bytes = 512 bytes
#define UHA_KV_PAGE_CENTROID_BYTES       512U

// ==============================================================================
// 7. Heterogeneous MoE Ring Buffer Sizing (Chorus + NVRM Governor)
// ==============================================================================
#define UHA_MOE_MAX_BATCH_TOKENS         16U
#define UHA_MOE_TOP_K_EXPERTS            8U
#define UHA_MOE_DEFAULT_HIDDEN_DIM       4096U
#define UHA_MOE_RING_SLOT_PAYLOAD_BYTES  (UHA_MOE_MAX_BATCH_TOKENS * UHA_MOE_DEFAULT_HIDDEN_DIM * 2U) // 128 KB

#endif // UHA_UNIFIED_SIZING_H
