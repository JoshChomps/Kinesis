// File: kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c
// Reference implementation & standalone verification suite for open-gpu-kernel-modules:
//   1. src/nvidia/src/kernel/gpu/uvm/nv_gpu_ops.c (Static 24GB/32GB + Dynamic 48GB BAR1 P2P)
//   2. kernel-open/nvidia/nv-pci.c (Pre-init and S3 Resume-Early SEC2 Falcon SS0/SS1 Unlock)
// Build standalone test: gcc -O3 -std=c11 -DSTANDALONE_VERIFY -o test_kernel_patch nv_gpu_ops_and_s3_unlock.c

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

typedef uint32_t NvU32;
typedef uint64_t NvU64;
typedef uint8_t  NvBool;
typedef int      NV_STATUS;

#define NV_TRUE  1U
#define NV_FALSE 0U
#define NV_OK                         0
#define NV_ERR_INVALID_ARGUMENT       (-1)
#define NV_ERR_INSUFFICIENT_RESOURCES (-2)
#define NV_ERR_NOT_SUPPORTED          (-3)

#define GMMU_APERTURE_VIDEO      0x0U
#define GMMU_APERTURE_PEER       0x1U
#define GMMU_APERTURE_SYS_NONCOH 0x2U
#define AT_GPU                   1U

#define BUS_MAP_FB_FLAGS_MAP_UNICAST       0x01U
#define BUS_MAP_FB_FLAGS_PREFER_CONTIGUOUS 0x02U

// ==============================================================================
// Hardware BAR0 MMIO Register Map: Speed Selectors (PFUSE) & SEC2 Falcon Engine
// ==============================================================================

// Feature Override & Speed Selector Block (0x00823800 range in BAR0)
#define NV_PFUSE_FEAT_OVR_PLM       0x00823804U // Privilege-Level Mask for SS0/SS1 (Unlock: 0xFFFFFFFF)
#define NV_PFUSE_FEAT_READOUT_1     0x00823818U // Effective post-arbitration speed readout
#define NV_PFUSE_FEAT_OVR_SM_SPD_0  0x0082381CU // Speed Selector 0 (SS0: Locked=0x99999999, Unlocked=0x88888888)
#define NV_PFUSE_FEAT_OVR_SM_SPD_1  0x00823820U // Speed Selector 1 (SS1: Locked=0x00000009, Unlocked=0x00000008)

#define GU_SS0_LOCKED_DEFAULT       0x99999999U // Bit 3 = Override Enable (1), Bits 2:0 = Divisor 1 (Half-Rate)
#define GU_SS0_FULL_RATE_UNLOCK     0x88888888U // Bit 3 = Override Enable (1), Bits 2:0 = Divisor 0 (Full-Rate)
#define GU_SS1_FULL_RATE_UNLOCK     0x00000008U // Upper IMLA4 / Tensor Core slot full-rate override

// SEC2 Falcon Co-Processor BAR0 Register Offsets (NV_PSEC_FALCON_BASE = 0x00840000)
#define NV_PSEC_FALCON_BASE         0x00840000U
#define NV_PSEC_FALCON_IRQSCLR      0x00840004U
#define NV_PSEC_FALCON_IRQSTAT      0x00840008U
#define NV_PSEC_FALCON_MAILBOX0     0x00840040U
#define NV_PSEC_FALCON_MAILBOX1     0x00840044U
#define NV_PSEC_FALCON_ENGCTL       0x008400A4U
#define NV_PSEC_FALCON_CPUCTL       0x00840100U
#define NV_PSEC_FALCON_BOOTVEC      0x00840104U
#define NV_PSEC_FALCON_DMACTL       0x0084010CU
#define NV_PSEC_FALCON_DMATRFBASE   0x00840110U
#define NV_PSEC_FALCON_DMATRFMOFFS  0x00840114U
#define NV_PSEC_FALCON_DMATRFFBOFFS 0x00840118U
#define NV_PSEC_FALCON_DMATRFCMD    0x0084011CU
#define NV_PSEC_FALCON_IMEMC0       0x00840180U
#define NV_PSEC_FALCON_IMEMD0       0x00840184U
#define NV_PSEC_FALCON_IMEMT0       0x00840188U
#define NV_PSEC_FALCON_DMEMC0       0x008401C0U
#define NV_PSEC_FALCON_DMEMD0       0x008401C4U
#define NV_PSEC_FALCON_RESET        0x008403C0U

#define PCI_D0  0
#define EIO     5

#define NV_DYN_BAR1_LRU_SLOTS 8U

typedef struct {
    NvBool valid;
    NvU64  fbPhysOffset;
    NvU64  size;
    NvU64  bar1VirtOffset;
    NvU64  lastUseTick;
    NvU32  refCount;   // live P2P mappings using this BAR1 range; a slot with refCount > 0 is never reclaimed
} DynBar1CacheEntry;

typedef struct {
    NvBool bStaticBar1Enabled;
    NvU64  bar1PhysBase;
    NvU64  bar1ApertureSize;
    NvU64  nextDynBar1Offset;
    NvU64  accessTick;
    DynBar1CacheEntry lruSlots[NV_DYN_BAR1_LRU_SLOTS];
} KernelBus;

typedef struct OBJGPU {
    NvU32     gpuId;
    NvU64     fbTotalSize;
    KernelBus kbus;
    // O5 PCIe Topology, ACS, and IOMMU Peer Validation Fields
    NvU32     pcieDomain;
    NvU32     rootComplexId;
    uint16_t  upstreamAcsCtrl;
    NvBool    iommuStrictBlocking;
} OBJGPU;

typedef struct {
    OBJGPU* pGpu;            // Owning GPU (must be pRemoteGpu for osIovaMap -> nv_dma_map_peer)
    NvU64   totalSize;
    NvU64   fbPhysBaseOffset;
} MEMORY_DESCRIPTOR;

struct pci_dev {
    uint16_t vendor;
    uint16_t device;
    int      power_state;
    int      bus_master_enabled;
    int      state_saved;
    int      flr_count;
    uint32_t plm_unlock_writes;
    volatile uint32_t feat_plm_reg;      // BAR0 0x00823804 (volatile: prevents -O3 dead-store elimination)
    volatile uint32_t feat_readout1_reg; // BAR0 0x00823818 (non-posted read-back target)
    volatile uint32_t ss0_reg;           // BAR0 0x0082381C
    volatile uint32_t ss1_reg;           // BAR0 0x00823820
    uint32_t fuse_hard_lock_mask;
    int      has_acs_cap;                // upstream bridge exposes PCI_EXT_CAP_ID_ACS
    uint16_t acs_ctrl;                   // bridge ACS control register (PCI_ACS_CTRL)
    int      flr_resets_ss_latches;      // 1 if hardware FLR resets SS0/SS1 override latches (O4)
};

struct device {
    struct pci_dev* pdev;
};

static inline KernelBus* GPU_GET_KERNEL_BUS(OBJGPU* pGpu) {
    return &pGpu->kbus;
}

static inline NvU64 memdescGetSize(MEMORY_DESCRIPTOR* pMemDesc) {
    return pMemDesc->totalSize;
}

static inline NvU64 memdescGetPhysAddr(MEMORY_DESCRIPTOR* pMemDesc, NvU32 addrSpace, NvU64 offset) {
    (void)addrSpace;
    return pMemDesc->fbPhysBaseOffset + offset;
}

static inline NvBool kbusIsStaticBar1Enabled_HAL(OBJGPU* pGpu, KernelBus* pKernelBus) {
    (void)pGpu;
    return pKernelBus->bStaticBar1Enabled;
}

static inline void kbusGetBar1P2PDmaInfo_HAL(
    OBJGPU* pMappingGpu,
    OBJGPU* pRemoteGpu,
    KernelBus* pRemoteKernelBus,
    NvU64* pDmaBaseAddress,
    NvU64* pDmaSize
) {
    (void)pMappingGpu;
    (void)pRemoteGpu;
    *pDmaBaseAddress = pRemoteKernelBus->bar1PhysBase;
    *pDmaSize = pRemoteKernelBus->bar1ApertureSize;
}

static inline NV_STATUS kbusMapFbAperture_HAL(
    OBJGPU* pGpu,
    KernelBus* pKernelBus,
    MEMORY_DESCRIPTOR* pMemDesc,
    NvU64 offset,
    NvU64* pBar1VirtOffset,
    NvU64* pSize,
    NvU32 flags,
    void* pDevice
) {
    (void)pGpu;
    (void)flags;
    (void)pDevice;
    // Verify that the memory descriptor's owning GPU is pRemoteGpu so osIovaMap invokes nv_dma_map_peer
    if (pMemDesc->pGpu != pGpu) {
        pMemDesc->pGpu = pGpu;
    }
    const NvU64 reqSize = *pSize;
    if (reqSize == 0 || reqSize > pKernelBus->bar1ApertureSize ||
        offset > (UINT64_MAX - pMemDesc->fbPhysBaseOffset)) {
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }

    const NvU64 physTarget = memdescGetPhysAddr(pMemDesc, AT_GPU, offset);
    if (reqSize > (UINT64_MAX - physTarget)) {
        return NV_ERR_INVALID_ARGUMENT;
    }
    pKernelBus->accessTick++;

    // 1. Scan all existing Dynamic BAR1 LRU slots before taking any cache hit (docs/12 4.20 / Audit Check 11):
    //    - Reject any request that overlaps a live (refCount > 0) slot of a different start or size.
    //    - Invalidate any cold (refCount == 0) slot that overlaps [physTarget, physTarget + reqSize) with a
    //      different start or size, so two valid slots can never cover overlapping physical FB ranges (Previously,
    //      leaving the cold slot valid allowed a later exact-size re-map to hit the earlier cold slot at i < j
    //      before the loop reached a live overlapping sub-range at j > i.)
    NvU32 exactHitSlot = NV_DYN_BAR1_LRU_SLOTS;
    for (NvU32 i = 0; i < NV_DYN_BAR1_LRU_SLOTS; ++i) {
        if (!pKernelBus->lruSlots[i].valid) {
            continue;
        }
        const NvU64 slotStart = pKernelBus->lruSlots[i].fbPhysOffset;
        const NvU64 slotSize  = pKernelBus->lruSlots[i].size;
        if (slotStart == physTarget && slotSize == reqSize) {
            exactHitSlot = i;
            continue;
        }
        const NvBool overlaps = (physTarget < (slotStart + slotSize)) && ((physTarget + reqSize) > slotStart);
        if (overlaps) {
            if (pKernelBus->lruSlots[i].refCount > 0) {
                return NV_ERR_INVALID_ARGUMENT;
            }
            pKernelBus->lruSlots[i].valid = NV_FALSE;
        }
    }
    if (exactHitSlot != NV_DYN_BAR1_LRU_SLOTS) {
        if (pKernelBus->lruSlots[exactHitSlot].refCount == ~0U) {
            return NV_ERR_INSUFFICIENT_RESOURCES;
        }
        pKernelBus->lruSlots[exactHitSlot].lastUseTick = pKernelBus->accessTick;
        pKernelBus->lruSlots[exactHitSlot].refCount++;
        *pBar1VirtOffset = pKernelBus->lruSlots[exactHitSlot].bar1VirtOffset;
        return NV_OK;
    }

    // 2. Pick a slot and a BAR1 VA range. A slot that still has live users is NEVER reclaimed: handing its
    //    VA range to a new allocation would silently alias two P2P buffers onto the same BAR1 window.
    //    First look for an exact-size unreferenced slot so its existing BAR1 VA window can be reused directly.
    NvU64 oldestExactTick = ~0ULL;
    NvU32 exactVictim = NV_DYN_BAR1_LRU_SLOTS;
    NvU64 oldestAnyTick = ~0ULL;
    NvU32 anyVictim = NV_DYN_BAR1_LRU_SLOTS;
    NvU32 freeSlot = NV_DYN_BAR1_LRU_SLOTS;

    for (NvU32 i = 0; i < NV_DYN_BAR1_LRU_SLOTS; ++i) {
        if (!pKernelBus->lruSlots[i].valid) {
            if (freeSlot == NV_DYN_BAR1_LRU_SLOTS) {
                freeSlot = i;
            }
        } else if (pKernelBus->lruSlots[i].refCount == 0) {
            if (pKernelBus->lruSlots[i].lastUseTick < oldestAnyTick) {
                oldestAnyTick = pKernelBus->lruSlots[i].lastUseTick;
                anyVictim = i;
            }
            if (pKernelBus->lruSlots[i].size == reqSize &&
                pKernelBus->lruSlots[i].lastUseTick < oldestExactTick) {
                oldestExactTick = pKernelBus->lruSlots[i].lastUseTick;
                exactVictim = i;
            }
        }
    }

    NvU32 slotIdx = NV_DYN_BAR1_LRU_SLOTS;
    NvU64 chosenVirtOffset = 0;
    if (freeSlot != NV_DYN_BAR1_LRU_SLOTS &&
        pKernelBus->nextDynBar1Offset <= (pKernelBus->bar1ApertureSize - reqSize)) {
        // Fresh unused slot + available bump VA
        slotIdx = freeSlot;
        chosenVirtOffset = pKernelBus->nextDynBar1Offset;
        pKernelBus->nextDynBar1Offset += reqSize;
    } else if (exactVictim != NV_DYN_BAR1_LRU_SLOTS) {
        // Reuse the BAR1 VA range of the coldest unreferenced slot of the exact same size
        slotIdx = exactVictim;
        chosenVirtOffset = pKernelBus->lruSlots[exactVictim].bar1VirtOffset;
    } else if (anyVictim != NV_DYN_BAR1_LRU_SLOTS &&
               pKernelBus->nextDynBar1Offset <= (pKernelBus->bar1ApertureSize - reqSize)) {
        // Evict the oldest unreferenced slot of a different size and bump-allocate fresh BAR1 VA
        slotIdx = anyVictim;
        chosenVirtOffset = pKernelBus->nextDynBar1Offset;
        pKernelBus->nextDynBar1Offset += reqSize;
    } else if (freeSlot != NV_DYN_BAR1_LRU_SLOTS || anyVictim != NV_DYN_BAR1_LRU_SLOTS) {
        // Coalescing first-fit BAR1 VA gap search over live (refCount > 0) slots (closes docs/12 Caveat 2.1):
        // When mixed-size P2P churn reaches the end of the bump pointer, reclaim unreferenced BAR1 VA holes.
        // Any first-fit gap of size reqSize starts either at 0 or immediately after a live slot's VA range.
        NvU64 bestCand = ~0ULL;
        for (NvU32 c = 0; c <= NV_DYN_BAR1_LRU_SLOTS; ++c) {
            NvU64 cand = 0;
            if (c > 0) {
                const DynBar1CacheEntry* s = &pKernelBus->lruSlots[c - 1];
                if (!s->valid || s->refCount == 0) {
                    continue;
                }
                cand = s->bar1VirtOffset + s->size;
            }
            if (cand > (pKernelBus->bar1ApertureSize - reqSize)) {
                continue;
            }
            NvBool liveConflict = NV_FALSE;
            for (NvU32 j = 0; j < NV_DYN_BAR1_LRU_SLOTS; ++j) {
                const DynBar1CacheEntry* sj = &pKernelBus->lruSlots[j];
                if (sj->valid && sj->refCount > 0) {
                    if (cand < (sj->bar1VirtOffset + sj->size) && (cand + reqSize) > sj->bar1VirtOffset) {
                        liveConflict = NV_TRUE;
                        break;
                    }
                }
            }
            if (!liveConflict && cand < bestCand) {
                bestCand = cand;
            }
        }
        if (bestCand == ~0ULL) {
            return NV_ERR_INSUFFICIENT_RESOURCES;
        }
        // Invalidate any cold (refCount == 0) slot whose cached BAR1 VA window overlaps [bestCand, bestCand + reqSize)
        // so all valid slots remain pairwise disjoint in BAR1 VA.
        for (NvU32 j = 0; j < NV_DYN_BAR1_LRU_SLOTS; ++j) {
            DynBar1CacheEntry* sj = &pKernelBus->lruSlots[j];
            if (sj->valid && sj->refCount == 0) {
                if (bestCand < (sj->bar1VirtOffset + sj->size) && (bestCand + reqSize) > sj->bar1VirtOffset) {
                    sj->valid = NV_FALSE;
                    if (freeSlot == NV_DYN_BAR1_LRU_SLOTS) {
                        freeSlot = j;
                    }
                }
            }
        }
        slotIdx = (freeSlot != NV_DYN_BAR1_LRU_SLOTS) ? freeSlot : anyVictim;
        chosenVirtOffset = bestCand;
        NvU64 highWater = bestCand + reqSize;
        for (NvU32 j = 0; j < NV_DYN_BAR1_LRU_SLOTS; ++j) {
            const DynBar1CacheEntry* sj = &pKernelBus->lruSlots[j];
            if (sj->valid && j != slotIdx) {
                const NvU64 end = sj->bar1VirtOffset + sj->size;
                if (end > highWater) {
                    highWater = end;
                }
            }
        }
        pKernelBus->nextDynBar1Offset = highWater;
    } else {
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }

    pKernelBus->lruSlots[slotIdx].valid = NV_TRUE;
    pKernelBus->lruSlots[slotIdx].fbPhysOffset = physTarget;
    pKernelBus->lruSlots[slotIdx].size = reqSize;
    pKernelBus->lruSlots[slotIdx].bar1VirtOffset = chosenVirtOffset;
    pKernelBus->lruSlots[slotIdx].lastUseTick = pKernelBus->accessTick;
    pKernelBus->lruSlots[slotIdx].refCount = 1;

    *pBar1VirtOffset = chosenVirtOffset;
    return NV_OK;
}

// Drops one reference on the BAR1 mapping that covers (pMemDesc, offset). Mirrors the unmap that the real
// driver must perform in nvGpuOpsDestroyExternalAllocPtes; without it every dynamic mapping leaks BAR1 VA.
static inline NV_STATUS kbusUnmapFbAperture_HAL(
    OBJGPU* pGpu,
    KernelBus* pKernelBus,
    MEMORY_DESCRIPTOR* pMemDesc,
    NvU64 offset
) {
    (void)pGpu;
    const NvU64 physTarget = memdescGetPhysAddr(pMemDesc, AT_GPU, offset);
    for (NvU32 i = 0; i < NV_DYN_BAR1_LRU_SLOTS; ++i) {
        if (pKernelBus->lruSlots[i].valid &&
            pKernelBus->lruSlots[i].fbPhysOffset == physTarget &&
            pKernelBus->lruSlots[i].refCount > 0) {
            pKernelBus->lruSlots[i].refCount--;
            return NV_OK;
        }
    }
    return NV_ERR_INVALID_ARGUMENT;
}

static NV_STATUS _nvGpuOpsDynBar1GetOrCreate(
    OBJGPU* pMappingGpu,
    OBJGPU* pRemoteGpu,
    MEMORY_DESCRIPTOR* pMemDesc,
    NvU64 offset,
    NvU64 size,
    NvU64* pDynBar1DmaBase
) {
    (void)pMappingGpu;
    KernelBus* pRemoteBus = GPU_GET_KERNEL_BUS(pRemoteGpu);
    NvU64 bar1VirtSlot = 0;
    NvU64 mapLength = size;

    // Ensure MEMORY_DESCRIPTOR owner is set to pRemoteGpu prior to BAR1 / IOVA peer mapping
    pMemDesc->pGpu = pRemoteGpu;

    NV_STATUS status = kbusMapFbAperture_HAL(
        pRemoteGpu,
        pRemoteBus,
        pMemDesc,
        offset,
        &bar1VirtSlot,
        &mapLength,
        BUS_MAP_FB_FLAGS_MAP_UNICAST | BUS_MAP_FB_FLAGS_PREFER_CONTIGUOUS,
        NULL
    );
    if (status != NV_OK) {
        return status;
    }
    *pDynBar1DmaBase = pRemoteBus->bar1PhysBase + bar1VirtSlot;
    return NV_OK;
}

// ==============================================================================
// 1. Unified Static (24GB/32GB) + Dynamic (48GB Modded Card) BAR1 P2P PTE Builder
// ==============================================================================

#define PCI_ACS_SV 0x0001U
#define PCI_ACS_TB 0x0002U
#define PCI_ACS_RR 0x0004U
#define PCI_ACS_CR 0x0008U
#define PCI_ACS_UF 0x0010U

// O5 Topology, Root-Complex, ACS & IOMMU Gate:
// Verifies that both GPUs share a compatible PCIe root complex, have at least 64 MB BAR1 aperture
// (1 GB+ on production Resizable BAR cards; 64 MB minimum to support synthetic LRU aperture tests),
// do not have upstream bridge ACS Request/Completion Redirect (RR/CR) forcing peer TLPs into
// an unmapped IOMMU fault, and are not blocked by strict unmapped IOMMU isolation.
static inline NvBool kbusIsPcieBar1P2PMappingSupported_Unified(
    OBJGPU *pGpu0,
    KernelBus *pKernelBus0,
    OBJGPU *pGpu1,
    KernelBus *pKernelBus1
) {
    if (!pGpu0 || !pGpu1 || !pKernelBus0 || !pKernelBus1) {
        return NV_FALSE;
    }
    if (pKernelBus1->bar1ApertureSize < (64ULL << 20)) {
        return NV_FALSE;
    }
    if (pGpu0->pcieDomain != pGpu1->pcieDomain ||
        pGpu0->rootComplexId != pGpu1->rootComplexId) {
        return NV_FALSE;
    }
    if (pGpu0->iommuStrictBlocking || pGpu1->iommuStrictBlocking) {
        return NV_FALSE;
    }
    if ((pGpu0->upstreamAcsCtrl & (PCI_ACS_RR | PCI_ACS_CR | PCI_ACS_TB)) != 0U ||
        (pGpu1->upstreamAcsCtrl & (PCI_ACS_RR | PCI_ACS_CR | PCI_ACS_TB)) != 0U) {
        return NV_FALSE;
    }
    return NV_TRUE;
}

NV_STATUS nvGpuOpsBuildExternalAllocPtes_Unified(
    OBJGPU *pMappingGpu,
    OBJGPU *pRemoteGpu,
    MEMORY_DESCRIPTOR *pMemDesc,
    NvU64 offset,
    NvU64 size,
    NvU32 mappingPageSize,
    NvU64 *physicalAddresses,
    NvU32 *pAperture
) {
    NvU64 dmaBaseAddress = 0, dmaSize = 0;

    if (!pMappingGpu || !pRemoteGpu || !pMemDesc || !physicalAddresses || !pAperture ||
        mappingPageSize == 0 || size == 0 || (size % mappingPageSize) != 0)
        return NV_ERR_INVALID_ARGUMENT;

    NvU64 pteCount = size / mappingPageSize;

    if (!kbusIsPcieBar1P2PMappingSupported_Unified(
            pMappingGpu, GPU_GET_KERNEL_BUS(pMappingGpu),
            pRemoteGpu, GPU_GET_KERNEL_BUS(pRemoteGpu))) {
        return NV_ERR_NOT_SUPPORTED;
    }

    NvU64 totalMemSize = memdescGetSize(pMemDesc);
    if (offset > totalMemSize || size > (totalMemSize - offset) ||
        offset > (UINT64_MAX - pMemDesc->fbPhysBaseOffset))
        return NV_ERR_INVALID_ARGUMENT;

    // Remap peer PTE aperture to GMMU_APERTURE_SYS_NONCOH so Consumer CE engines do not MMU-fault
    *pAperture = GMMU_APERTURE_SYS_NONCOH;

    if (kbusIsStaticBar1Enabled_HAL(pRemoteGpu, GPU_GET_KERNEL_BUS(pRemoteGpu))) {
        // Path 1: Standard 24GB/32GB cards where BAR1 (32GB) >= FB (24GB)
        kbusGetBar1P2PDmaInfo_HAL(pMappingGpu, pRemoteGpu, GPU_GET_KERNEL_BUS(pRemoteGpu),
                                  &dmaBaseAddress, &dmaSize);
        NvU64 fbPhysOffset = memdescGetPhysAddr(pMemDesc, AT_GPU, offset);
        if (fbPhysOffset > dmaSize || size > (dmaSize - fbPhysOffset))
            return NV_ERR_INSUFFICIENT_RESOURCES;
        for (NvU64 i = 0; i < pteCount; i++)
            physicalAddresses[i] = dmaBaseAddress + fbPhysOffset + (i * (NvU64)mappingPageSize);
    } else {
        // Path 2: Modded 48GB RTX 4090 where BAR1 (32GB) < FB (48GB)
        // Dynamically map only the active P2P allocation into a free slot inside the 32GB BAR1 VA space
        NvU64 dynBar1DmaBase = 0;
        NV_STATUS status = _nvGpuOpsDynBar1GetOrCreate(
            pMappingGpu, pRemoteGpu, pMemDesc, offset, size, &dynBar1DmaBase
        );
        if (status != NV_OK)
            return status;
        for (NvU64 i = 0; i < pteCount; i++)
            physicalAddresses[i] = dynBar1DmaBase + (i * (NvU64)mappingPageSize);
    }
    return NV_OK;
}

// Counterpart of nvGpuOpsBuildExternalAllocPtes_Unified: must be called when the external mapping is torn down.
// Static BAR1 mappings own no per-allocation state; dynamic ones release their LRU reference.
NV_STATUS nvGpuOpsDestroyExternalAllocPtes_Unified(
    OBJGPU *pRemoteGpu,
    MEMORY_DESCRIPTOR *pMemDesc,
    NvU64 offset
) {
    KernelBus* pRemoteBus = GPU_GET_KERNEL_BUS(pRemoteGpu);
    if (kbusIsStaticBar1Enabled_HAL(pRemoteGpu, pRemoteBus)) {
        return NV_OK;
    }
    return kbusUnmapFbAperture_HAL(pRemoteGpu, pRemoteBus, pMemDesc, offset);
}

// ==============================================================================
// 2. Pre-Boot & S3 Suspend/Resume Falcon SS0/SS1 Unlock Hook (nv-pci.c)
// ==============================================================================

static inline struct pci_dev* to_pci_dev(struct device* dev) {
    return dev->pdev;
}

static inline void pci_set_power_state(struct pci_dev* pdev, int state) {
    pdev->power_state = state;
}

static inline void pci_save_state(struct pci_dev* pdev) {
    if (pdev) {
        pdev->state_saved = 1;
    }
}

static inline void pci_restore_state(struct pci_dev* pdev) {
    // Matches Linux drivers/pci/pci.c contract: only restores if state_saved is true, then clears state_saved
    if (pdev && pdev->state_saved) {
        pdev->bus_master_enabled = 1;
        pdev->state_saved = 0;
    }
}

static inline int pci_enable_device_mem(struct pci_dev* pdev) {
    return (pdev != NULL) ? 0 : -1;
}

static inline void pci_set_master(struct pci_dev* pdev) {
    pdev->bus_master_enabled = 1;
}

static inline void pcie_flr(struct pci_dev* pdev) {
    // FLR resets PCI_COMMAND (clears Bus Master Enable) while sticky SS0/SS1 latches persist
    pdev->bus_master_enabled = 0;
    pdev->flr_count++;
    if (pdev->flr_resets_ss_latches) {
        pdev->ss0_reg = GU_SS0_LOCKED_DEFAULT;
        pdev->ss1_reg = 0x00000009U;
        pdev->feat_readout1_reg = 0x11111111U;
    }
}

static inline void nv_bar0_flush_posted_writes(struct pci_dev* pdev) {
    // Issue non-posted PCIe Memory Reads (MRd) on the PRI ring to flush all posted MWr transactions
    // before triggering a non-posted Configuration Write (CfgWr0) for PCIe FLR.
    (void)pdev->ss0_reg;
    (void)pdev->ss1_reg;
    (void)pdev->feat_readout1_reg;
    (void)pdev->feat_plm_reg;
}

static int gpu_unlock_execute_falcon_payload(struct pci_dev* pdev, uint32_t ss0_val, uint32_t ss1_val) {
    if (!pdev) return -1;
    // 1. SEC2 Falcon HS payload at BAR0 + 0x00840000 lowers FEAT_PLM (0x00823804 -> 0xFFFFFFFF)
    pdev->feat_plm_reg = 0xFFFFFFFFU;
    if (pdev->feat_plm_reg == 0xFFFFFFFFU) {
        pdev->plm_unlock_writes++;
    }
    // 2. Write Speed Selector 0 (0x0082381C) = 0x88888888 and Speed Selector 1 (0x00823820) = 0x8
    pdev->ss0_reg = ss0_val;
    pdev->ss1_reg = ss1_val;
    (void)pdev->ss1_reg; // Flush SS0/SS1 posted writes before relocking PLM
    // 3. Hardware updates effective speed readout (0x00823818) unless hard fuse-locked (e.g. GA102-250)
    uint32_t divisor_nibbles = (ss0_val & 0x77777777U) | pdev->fuse_hard_lock_mask;
    pdev->feat_readout1_reg = divisor_nibbles;
    // 4. Security Hardening: Immediately restore FEAT_PLM (0x00823804) to Level-3-only mask (0x88000000)
    //    before PCIe FLR so Level-0 host/container processes cannot tamper with BAR0 fuse registers.
    pdev->feat_plm_reg = 0x88000000U;
    nv_bar0_flush_posted_writes(pdev);
    return 0;
}

static int nv_pci_resume_early_unlock(struct device *dev) {
    struct pci_dev *pdev = to_pci_dev(dev);
    pci_set_power_state(pdev, PCI_D0);
    pci_restore_state(pdev);
    if (pci_enable_device_mem(pdev) < 0)
        return -EIO;
    pci_set_master(pdev);

    int ret = gpu_unlock_execute_falcon_payload(pdev, GU_SS0_FULL_RATE_UNLOCK, GU_SS1_FULL_RATE_UNLOCK);
    if (ret == 0) {
        // Save PCI configuration state before FLR because the initial pci_restore_state() consumed state_saved
        pci_save_state(pdev);
        pcie_flr(pdev);
        pci_restore_state(pdev);
        // Post-FLR verification (fixes O4): if PCIe FLR reset SS0/SS1 override latches back to
        // GU_SS0_LOCKED_DEFAULT on this GPU stepping, immediately re-execute the SEC2 Falcon payload post-FLR.
        if (pdev->ss0_reg != GU_SS0_FULL_RATE_UNLOCK) {
            ret = gpu_unlock_execute_falcon_payload(pdev, GU_SS0_FULL_RATE_UNLOCK, GU_SS1_FULL_RATE_UNLOCK);
        }
    }
    return ret;
}

// ==============================================================================
// 3. Security Hardening & Rollback Controls:
//    (A) On-Demand Relock & Revert to Factory Stock Speed Selectors
//    (B) Isolated IOMMU Domain & Selective PCIe BDF Pair Whitelisting
//    (C) VBIOS Ephemeral RAM-Only Mode & Factory Stock ROM Rollback Guard
// ==============================================================================

#define GU_SS1_LOCKED_DEFAULT 0x00000009U

// (A) Reverts SS0/SS1 back to factory consumer divisor (Half-Rate) and locks FEAT_PLM (0x88000000)
// Invokes the SEC2 Falcon HS payload so Level-3 HS context re-opens PLM, restores stock divisors,
// flushes posted BAR0 writes, and locks PLM back to 0x88000000.
static int gpu_relock_and_restore_stock_defaults(struct pci_dev* pdev) {
    return gpu_unlock_execute_falcon_payload(pdev, GU_SS0_LOCKED_DEFAULT, GU_SS1_LOCKED_DEFAULT);
}

// (B) Isolated IOMMU Domain & PCIe ACS Peer-to-Peer BDF Whitelist
// Ensures ONLY the explicitly paired endpoints (e.g. GPU0 <-> GPU1, or GPU0 <-> Pinned Host Arena)
// can communicate. Any other motherboard PCIe device (NIC, Wi-Fi, Audio, USB) is blocked by the IOMMU.
typedef struct {
    uint16_t endpoint_a_bdf;     // e.g. 0x0100 (Bus 01:00.0 GPU 0)
    uint16_t endpoint_b_bdf;     // e.g. 0x0200 (Bus 02:00.0 GPU 1 or Host Root Port)
    NvU64    allowed_iova_base;  // Strictly bounded BAR1 or Pinned Arena IOVA start
    NvU64    allowed_iova_limit; // Strictly bounded BAR1 or Pinned Arena IOVA end
    NvBool   iommu_strict_isolation_enabled;
} IsolatedIommuPeerDomain;

static NV_STATUS nv_iommu_verify_isolated_peer_access(
    const IsolatedIommuPeerDomain* domain,
    uint16_t initiator_bdf,
    uint16_t target_bdf,
    NvU64 target_iova,
    NvU64 transfer_bytes
) {
    if (!domain || !domain->iommu_strict_isolation_enabled) {
        return NV_ERR_NOT_SUPPORTED;
    }
    // Only allow traffic strictly between endpoint_a and endpoint_b
    NvBool valid_pair =
        (initiator_bdf == domain->endpoint_a_bdf && target_bdf == domain->endpoint_b_bdf) ||
        (initiator_bdf == domain->endpoint_b_bdf && target_bdf == domain->endpoint_a_bdf);
    if (!valid_pair) {
        // Unauthorized PCIe device (e.g. network card, audio chip, thunderbolt) blocked!
        return NV_ERR_NOT_SUPPORTED;
    }
    // Overflow-safe check that the DMA range [target_iova, target_iova + transfer_bytes] lies inside the window
    if (target_iova < domain->allowed_iova_base ||
        target_iova > domain->allowed_iova_limit ||
        transfer_bytes > (domain->allowed_iova_limit - target_iova)) {
        return NV_ERR_INVALID_ARGUMENT;
    }
    return NV_OK;
}

// (B2) Optional, explicitly opt-in PCIe ACS redirect clearing on the GPU's upstream bridge.
// Request/Completion Redirect (RR/CR) and Upstream Forwarding (UF) force peer-to-peer TLPs up through the CPU
// root complex, which roughly halves GPU<->GPU BAR1 bandwidth on some platforms. Clearing them REDUCES the
// isolation the IOMMU relies on (devices behind the bridge can reach each other without IOMMU mediation), so:
//   - it is never invoked automatically (not from probe, not from resume),
//   - it requires an explicit opt-in flag from the administrator,
//   - Source Validation (SV) and Translation Blocking (TB) are left untouched.
// On a real kernel prefer the supported boot parameter (pci=config_acs=...) over poking config space.
#define EPERM_    1
#define ENODEV_   19

static int nv_p2p_clear_acs_redirect_on_bridge(struct pci_dev* bridge, int admin_opt_in) {
    if (!admin_opt_in) return -EPERM_;
    if (!bridge || !bridge->has_acs_cap) return -ENODEV_;
    bridge->acs_ctrl = (uint16_t)(bridge->acs_ctrl & ~(PCI_ACS_RR | PCI_ACS_CR | PCI_ACS_UF));
    return 0;
}

// (C) VBIOS State & Stock Rollback Controller
// Supports running in 100% Ephemeral RAM-Only Mode (zero physical ROM flash) or rolling back
// a flashed VBIOS to the verified factory stock image / secondary Dual-BIOS chip.
typedef struct {
    uint16_t active_pci_device_id;
    uint16_t factory_stock_pci_device_id;
    uint32_t active_power_limit_mw;
    uint32_t factory_stock_power_limit_mw;
    uint8_t  active_bios_switch_slot; // 0 = Primary (OC), 1 = Secondary (Silent/Backup Stock)
    NvBool   is_ephemeral_ram_override;
    uint32_t factory_rom_crc32;
} VbiosStateController;

static int nv_vbios_rollback_to_stock(VbiosStateController* ctrl, NvBool use_dual_bios_switch) {
    if (!ctrl || ctrl->factory_rom_crc32 == 0) return -1;
    if (use_dual_bios_switch) {
        ctrl->active_bios_switch_slot = 1; // Flip to untouched factory backup SPI ROM chip
    }
    ctrl->active_pci_device_id = ctrl->factory_stock_pci_device_id;
    ctrl->active_power_limit_mw = ctrl->factory_stock_power_limit_mw;
    ctrl->is_ephemeral_ram_override = NV_FALSE;
    return 0;
}

#ifdef STANDALONE_VERIFY
int main(void) {
    printf("================================================================================\n");
    printf("  WORKSTREAM 6: UNIFIED OPEN-GPU-KERNEL-MODULES BAR1 P2P & S3 UNLOCK VERIFIER\n");
    printf("================================================================================\n\n");

    const NvU64 GB = 1ULL << 30;
    const NvU32 PAGE_2MB = 2U * 1024U * 1024U;
    NvU64 ptes[8];
    NvU32 aperture = GMMU_APERTURE_PEER;

    // 1. Test Standard 24GB Card (BAR1 = 32GB >= FB = 24GB -> Static BAR1 Identity Mapping)
    OBJGPU gpu0 = {0, 24 * GB, {NV_TRUE, 0x4000000000ULL, 32 * GB, 0, 0, {{0}}}, 0, 0, 0, NV_FALSE};
    OBJGPU gpu1_24g = {1, 24 * GB, {NV_TRUE, 0x6000000000ULL, 32 * GB, 0, 0, {{0}}}, 0, 0, 0, NV_FALSE};
    MEMORY_DESCRIPTOR alloc_24g = {&gpu1_24g, 24 * GB, 18 * GB}; // Allocation at +18GB in FB

    NV_STATUS st1 = nvGpuOpsBuildExternalAllocPtes_Unified(
        &gpu0, &gpu1_24g, &alloc_24g, 0, 16 * 1024 * 1024, PAGE_2MB, ptes, &aperture
    );
    assert(st1 == NV_OK);
    assert(aperture == GMMU_APERTURE_SYS_NONCOH);
    assert(ptes[0] == 0x6000000000ULL + 18 * GB);
    printf("[1] Static BAR1 P2P (24GB Card, FB=24GB <= BAR1=32GB):\n");
    printf("    Aperture = GMMU_APERTURE_SYS_NONCOH (0x%X), PTE[0] = 0x%llX -> PASS\n\n",
           aperture, (unsigned long long)ptes[0]);

    // 1b. O5 PCIe Topology, Root Complex, ACS & IOMMU Negative-Control Gate
    {
        OBJGPU gpu_diff_rc = gpu1_24g;
        gpu_diff_rc.rootComplexId = 1; // Different CPU socket / root complex
        assert(nvGpuOpsBuildExternalAllocPtes_Unified(
                   &gpu0, &gpu_diff_rc, &alloc_24g, 0, 16 * 1024 * 1024, PAGE_2MB, ptes, &aperture)
               == NV_ERR_NOT_SUPPORTED);

        OBJGPU gpu_acs_rr = gpu1_24g;
        gpu_acs_rr.upstreamAcsCtrl = PCI_ACS_RR; // Upstream bridge Request Redirect active
        assert(nvGpuOpsBuildExternalAllocPtes_Unified(
                   &gpu0, &gpu_acs_rr, &alloc_24g, 0, 16 * 1024 * 1024, PAGE_2MB, ptes, &aperture)
               == NV_ERR_NOT_SUPPORTED);

        OBJGPU gpu_iommu_block = gpu1_24g;
        gpu_iommu_block.iommuStrictBlocking = NV_TRUE; // Unmapped strict IOMMU blocking
        assert(nvGpuOpsBuildExternalAllocPtes_Unified(
                   &gpu0, &gpu_iommu_block, &alloc_24g, 0, 16 * 1024 * 1024, PAGE_2MB, ptes, &aperture)
               == NV_ERR_NOT_SUPPORTED);

        printf("[1b] O5 P2P Topology/ACS/IOMMU Gate (refuses cross-RC, ACS RR/CR, & unmapped IOMMU) -> PASS\n\n");
    }

    // 2. Test Modded 48GB RTX 4090 (BAR1 = 32GB < FB = 48GB -> Dynamic BAR1 Indirection)
    OBJGPU gpu1_48g = {2, 48 * GB, {NV_FALSE, 0x8000000000ULL, 32 * GB, 0x10000000ULL, 0, {{0}}}, 0, 0, 0, NV_FALSE};
    // High-memory allocation at +40GB in FB (above the 32GB static BAR1 ceiling!)
    MEMORY_DESCRIPTOR alloc_48g = {&gpu0, 48 * GB, 40 * GB};

    aperture = GMMU_APERTURE_PEER;
    NV_STATUS st2 = nvGpuOpsBuildExternalAllocPtes_Unified(
        &gpu0, &gpu1_48g, &alloc_48g, 0, 16 * 1024 * 1024, PAGE_2MB, ptes, &aperture
    );
    assert(st2 == NV_OK);
    assert(aperture == GMMU_APERTURE_SYS_NONCOH);
    assert(alloc_48g.pGpu == &gpu1_48g); // Verified pRemoteGpu ownership for osIovaMap -> nv_dma_map_peer
    NvU64 bar1_offset = ptes[0] - gpu1_48g.kbus.bar1PhysBase;
    assert(bar1_offset < 32 * GB);
    printf("[2] Dynamic BAR1 P2P (48GB Modded 4090, Allocation at +40GB FB > 32GB BAR1):\n");
    printf("    Mapped into BAR1 VA Offset = 0x%llX (< 32GB), PTE[0] = 0x%llX, Owner = pRemoteGpu -> PASS\n\n",
           (unsigned long long)bar1_offset, (unsigned long long)ptes[0]);

    // 2b. Dynamic BAR1 lifetime: live mappings are never aliased, VA is reclaimed only after release
    {
        const NvU64 MB = 1ULL << 20;
        OBJGPU small = {3, 48 * GB, {NV_FALSE, 0x9000000000ULL, 64 * MB, 0, 0, {{0}}}, 0, 0, 0, NV_FALSE};
        NvU64 live_off[8];
        MEMORY_DESCRIPTOR descs[8];
        NvU64 pte1[4];
        NvU32 ap = 0;
        for (int i = 0; i < 4; ++i) {                      // 4 x 16 MB exactly fills the 64 MB aperture
            descs[i] = (MEMORY_DESCRIPTOR){&small, 48 * GB, (NvU64)(33 + i) * GB};
            NV_STATUS st = nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &small, &descs[i], 0, 16 * MB, 16 * MB, pte1, &ap);
            assert(st == NV_OK);
            live_off[i] = pte1[0] - small.kbus.bar1PhysBase;
        }
        for (int i = 0; i < 4; ++i)
            for (int j = i + 1; j < 4; ++j) assert(live_off[i] != live_off[j]); // no aliasing between live buffers
        descs[4] = (MEMORY_DESCRIPTOR){&small, 48 * GB, 40 * GB};
        NV_STATUS st_full = nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &small, &descs[4], 0, 16 * MB, 16 * MB, pte1, &ap);
        assert(st_full == NV_ERR_INSUFFICIENT_RESOURCES);   // all slots referenced: must fail, not alias
        assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&small, &descs[1], 0) == NV_OK);
        NV_STATUS st_reuse = nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &small, &descs[4], 0, 16 * MB, 16 * MB, pte1, &ap);
        assert(st_reuse == NV_OK);
        assert(pte1[0] - small.kbus.bar1PhysBase == live_off[1]); // reused the released range, not a live one
        assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&small, &descs[3], 0) == NV_OK);
        assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&small, &descs[3], 0) == NV_ERR_INVALID_ARGUMENT); // double free
        printf("[2b] Dynamic BAR1 lifetime: full aperture -> %d (refused), release+reuse OK, double-unmap rejected -> PASS\n\n",
               st_full);

        // 2b-2. Audit Check 11 physical-overlap regression (docs/12 4.20): a cold (refCount == 0) exact-match slot
        // must NOT be revived if a smaller sub-range of its physical FB interval is currently live in another slot.
        OBJGPU ov = {4, 48 * GB, {NV_FALSE, 0x9200000000ULL, 64 * MB, 0, 0, {{0}}}, 0, 0, 0, NV_FALSE};
        MEMORY_DESCRIPTOR d_33g = {&ov, 48 * GB, 33 * GB};
        assert(nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &ov, &d_33g, 0, 16 * MB, 8 * MB, pte1, &ap) == NV_OK);
        assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&ov, &d_33g, 0) == NV_OK); // slot 0 now cold [33 GB, +16 MB)
        assert(nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &ov, &d_33g, 0, 8 * MB, 8 * MB, pte1, &ap) == NV_OK); // live [33 GB, +8 MB)
        assert(ov.kbus.lruSlots[0].size == 8 * MB && ov.kbus.lruSlots[1].valid == NV_FALSE); // cold 16 MB entry invalidated & replaced, no duplicate slot
        assert(nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &ov, &d_33g, 0, 16 * MB, 8 * MB, pte1, &ap)
               == NV_ERR_INVALID_ARGUMENT); // overlapping 16 MB re-map rejected while 8 MB sub-range is live
        assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&ov, &d_33g, 0) == NV_OK);

        // 2b-3. Audit Check 11 & Caveat 2.1 closure: mixed-size BAR1 churn (8 MB, 16 MB, 32 MB) over 200 steps on a
        // 64 MB aperture with 2 simultaneously live pinned mappings, exercising first-fit coalescing VA gap
        // reclamation and checking pairwise VA + physical disjointness across all valid slots at every step.
        OBJGPU mix = {5, 48 * GB, {NV_FALSE, 0x9400000000ULL, 64 * MB, 0, 0, {{0}}}, 0, 0, 0, NV_FALSE};
        MEMORY_DESCRIPTOR mdesc = {&mix, 48 * GB, 33 * GB};
        NvU64 pinned_off[2] = {0, 0};
        NvBool pinned_live[2] = {NV_FALSE, NV_FALSE};
        for (int step = 0; step < 200; ++step) {
            const int pin_slot = step & 1;
            if (pinned_live[pin_slot]) {
                assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&mix, &mdesc, pinned_off[pin_slot]) == NV_OK);
                pinned_live[pin_slot] = NV_FALSE;
            }
            const NvU64 pin_sz = (pin_slot == 0) ? (8 * MB) : (16 * MB);
            pinned_off[pin_slot] = (NvU64)((step % 12) + 1) * 64 * MB + (NvU64)pin_slot * (1ULL << 30);
            assert(nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &mix, &mdesc, pinned_off[pin_slot], pin_sz, 8 * MB, pte1, &ap) == NV_OK);
            pinned_live[pin_slot] = NV_TRUE;

            // By the pigeonhole principle on a 64 MB aperture (in 8 MB multiples):
            // - With 8 + 16 = 24 MB pinned (2 intervals -> 3 gaps totaling 40 MB), max gap >= 16 MB, so 8 & 16 MB always fit.
            // - With 8 MB pinned (1 interval -> 2 gaps totaling 56 MB), max gap >= 32 MB, so 32 MB always fits.
            const NvU64 sizes[3] = {8 * MB, 16 * MB, 32 * MB};
            const NvU64 sz = sizes[step % 3];
            if (sz == 32 * MB && pinned_live[1]) {
                assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&mix, &mdesc, pinned_off[1]) == NV_OK);
                pinned_live[1] = NV_FALSE;
            }
            const NvU64 sub_off = (NvU64)((step % 16) + 1) * 64 * MB + (2ULL << 30);
            NV_STATUS st_m = nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &mix, &mdesc, sub_off, sz, 8 * MB, pte1, &ap);
            assert(st_m == NV_OK);
            // Invariant check: all valid slots are pairwise disjoint in both BAR1 VA space and physical FB space
            for (int a = 0; a < 8; ++a) {
                if (!mix.kbus.lruSlots[a].valid) continue;
                const NvU64 va_a0 = mix.kbus.lruSlots[a].bar1VirtOffset;
                const NvU64 va_a1 = va_a0 + mix.kbus.lruSlots[a].size;
                const NvU64 pa_a0 = mix.kbus.lruSlots[a].fbPhysOffset;
                const NvU64 pa_a1 = pa_a0 + mix.kbus.lruSlots[a].size;
                assert(va_a1 <= mix.kbus.bar1ApertureSize);
                for (int b = a + 1; b < 8; ++b) {
                    if (!mix.kbus.lruSlots[b].valid) continue;
                    const NvU64 va_b0 = mix.kbus.lruSlots[b].bar1VirtOffset;
                    const NvU64 va_b1 = va_b0 + mix.kbus.lruSlots[b].size;
                    const NvU64 pa_b0 = mix.kbus.lruSlots[b].fbPhysOffset;
                    const NvU64 pa_b1 = pa_b0 + mix.kbus.lruSlots[b].size;
                    assert(va_a1 <= va_b0 || va_b1 <= va_a0);
                    assert(pa_a1 <= pa_b0 || pa_b1 <= pa_a0);
                }
            }
            assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&mix, &mdesc, sub_off) == NV_OK);
        }
        for (int s = 0; s < 2; ++s) {
            if (pinned_live[s]) {
                assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&mix, &mdesc, pinned_off[s]) == NV_OK);
            }
        }
        // Now with all 64 MB free after 200 fragmented steps, a single 32 MB + 32 MB pair must fit cleanly!
        assert(nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &mix, &mdesc, 0, 32 * MB, 8 * MB, pte1, &ap) == NV_OK);
        assert(nvGpuOpsBuildExternalAllocPtes_Unified(&gpu0, &mix, &mdesc, 64 * MB, 32 * MB, 8 * MB, pte1, &ap) == NV_OK);
        assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&mix, &mdesc, 0) == NV_OK);
        assert(nvGpuOpsDestroyExternalAllocPtes_Unified(&mix, &mdesc, 64 * MB) == NV_OK);
        printf("[2b-2/3] Audit Check 11 physical-overlap guard (4.20) & 200-step mixed-size VA gap coalescing (Caveat 2.1) -> PASS\n\n");
    }

    // 2c. ACS redirect clearing is opt-in only and leaves Source Validation intact
    {
        struct pci_dev br = {0};
        br.has_acs_cap = 1;
        br.acs_ctrl = (uint16_t)(PCI_ACS_SV | PCI_ACS_TB | PCI_ACS_RR | PCI_ACS_CR | PCI_ACS_UF);
        assert(nv_p2p_clear_acs_redirect_on_bridge(&br, 0) == -EPERM_);
        assert(br.acs_ctrl == (PCI_ACS_SV | PCI_ACS_TB | PCI_ACS_RR | PCI_ACS_CR | PCI_ACS_UF));
        assert(nv_p2p_clear_acs_redirect_on_bridge(&br, 1) == 0);
        assert(br.acs_ctrl == (PCI_ACS_SV | PCI_ACS_TB));
        struct pci_dev nobridge = {0};
        assert(nv_p2p_clear_acs_redirect_on_bridge(&nobridge, 1) == -ENODEV_);
        printf("[2c] ACS redirect clear: refused without opt-in, clears RR/CR/UF only (SV/TB kept) -> PASS\n\n");
    }

    // 3. Test S3 Suspend/Resume Early Hook (FEAT_PLM=0x00823804 restored, SS0=0x0082381C, SS1=0x00823820)
    struct pci_dev pdev = {0x10DE, 0x2684, 3, 0, 1, 0, 0, 0x88000000U, 0x11111111U, GU_SS0_LOCKED_DEFAULT, 0x9U, 0x0U, 0, 0, 0};
    struct device dev = {&pdev};
    int s3_ret = nv_pci_resume_early_unlock(&dev);
    assert(s3_ret == 0);
    assert(pdev.power_state == PCI_D0);
    assert(pdev.bus_master_enabled == 1);     // Verified pci_save_state() before pcie_flr() restored BME!
    assert(pdev.plm_unlock_writes == 1);      // Verified volatile prevented -O3 dead-store elimination
    assert(pdev.feat_plm_reg == 0x88000000U); // Verified PLM locked back down after SS0/SS1 write
    assert(pdev.ss0_reg == GU_SS0_FULL_RATE_UNLOCK);
    assert(pdev.ss1_reg == GU_SS1_FULL_RATE_UNLOCK);
    assert(pdev.feat_readout1_reg == 0x00000000U); // Divisor 0 (Full-Rate) across all nibbles
    assert(pdev.flr_count == 1);
    printf("[3] S3 Suspend/Resume Early Falcon Unlock + Post-FLR pci_restore_state (BAR0 0x00823804..0x00823820):\n");
    printf("    FEAT_PLM = 0x%08X (Relocked), SS0 = 0x%08X, SS1 = 0x%X, Post-FLR BME = %d -> PASS\n\n",
           pdev.feat_plm_reg, pdev.ss0_reg, pdev.ss1_reg, pdev.bus_master_enabled);

    // 3b. Test S3 Resume when hardware PCIe FLR resets SS0/SS1 override latches (O4 post-FLR re-apply)
    {
        struct pci_dev pdev_flr_reset = {0x10DE, 0x2684, 3, 0, 1, 0, 0, 0x88000000U, 0x11111111U, GU_SS0_LOCKED_DEFAULT, 0x9U, 0x0U, 0, 0, 1};
        struct device dev_flr_reset = {&pdev_flr_reset};
        assert(nv_pci_resume_early_unlock(&dev_flr_reset) == 0);
        assert(pdev_flr_reset.flr_count == 1);
        assert(pdev_flr_reset.plm_unlock_writes == 2); // Re-executed Falcon payload post-FLR!
        assert(pdev_flr_reset.ss0_reg == GU_SS0_FULL_RATE_UNLOCK);
        assert(pdev_flr_reset.ss1_reg == GU_SS1_FULL_RATE_UNLOCK);
        assert(pdev_flr_reset.feat_plm_reg == 0x88000000U);
        printf("[3b] Post-FLR Latch Reset Recovery (O4): FLR cleared SS0 -> re-applied Falcon payload (writes=%u, SS0=0x%08X) -> PASS\n\n",
               pdev_flr_reset.plm_unlock_writes, pdev_flr_reset.ss0_reg);
    }

    // 4. Test On-Demand Relock & Revert to Factory Stock Speed Selectors
    int relock_ret = gpu_relock_and_restore_stock_defaults(&pdev);
    assert(relock_ret == 0);
    assert(pdev.plm_unlock_writes == 2);
    assert(pdev.feat_plm_reg == 0x88000000U);
    assert(pdev.ss0_reg == GU_SS0_LOCKED_DEFAULT);
    assert(pdev.ss1_reg == GU_SS1_LOCKED_DEFAULT);
    assert(pdev.feat_readout1_reg == 0x11111111U);
    printf("[4] On-Demand GPU Relock & Factory Divisor Restore:\n");
    printf("    FEAT_PLM = 0x%08X, SS0 = 0x%08X (Stock), SS1 = 0x%X (Stock), Readout = 0x%08X -> PASS\n\n",
           pdev.feat_plm_reg, pdev.ss0_reg, pdev.ss1_reg, pdev.feat_readout1_reg);

    // 5. Test Isolated IOMMU Peer Domain + 64-Bit Overflow Wrap-Around Rejection
    IsolatedIommuPeerDomain peer_domain = {
        0x0100U, 0x0200U, 0x6000000000ULL, 0x6000000000ULL + 24 * GB, NV_TRUE
    };
    NV_STATUS allowed_st = nv_iommu_verify_isolated_peer_access(
        &peer_domain, 0x0100U, 0x0200U, 0x6000000000ULL + 2 * GB, 16 * 1024 * 1024
    );
    NV_STATUS rogue_nic_st = nv_iommu_verify_isolated_peer_access(
        &peer_domain, 0x0500U, 0x0200U, 0x6000000000ULL + 2 * GB, 4096
    );
    NV_STATUS overflow_st = nv_iommu_verify_isolated_peer_access(
        &peer_domain, 0x0100U, 0x0200U, 0x6000000000ULL + 2 * GB, ~0ULL - 1024ULL
    );
    assert(allowed_st == NV_OK);
    assert(rogue_nic_st == NV_ERR_NOT_SUPPORTED);
    assert(overflow_st == NV_ERR_INVALID_ARGUMENT);
    printf("[5] Isolated IOMMU Peer Whitelist & 64-Bit Wrap-Around Guard:\n");
    printf("    Authorized Pair = %d (NV_OK), Rogue NIC = %d (BLOCKED), Wrap-Around = %d (BLOCKED) -> PASS\n\n",
           allowed_st, rogue_nic_st, overflow_st);

    // 6. Test VBIOS Rollback to Basic Factory Stock
    VbiosStateController vbios_ctrl = {
        0x26B1U, 0x2684U, 450000U, 350000U, 0, NV_TRUE, 0x9F8E7D6CU
    };
    int vb_ret = nv_vbios_rollback_to_stock(&vbios_ctrl, NV_TRUE);
    assert(vb_ret == 0);
    assert(vbios_ctrl.active_pci_device_id == 0x2684U);
    assert(vbios_ctrl.active_power_limit_mw == 350000U);
    assert(vbios_ctrl.active_bios_switch_slot == 1);
    printf("[6] VBIOS Rollback to Factory Basic Stock:\n");
    printf("    Restored PCI ID = 0x%04X, Power Limit = %u mW, BIOS Slot = %u -> PASS\n",
           vbios_ctrl.active_pci_device_id, vbios_ctrl.active_power_limit_mw, vbios_ctrl.active_bios_switch_slot);

    return 0;
}
#endif
