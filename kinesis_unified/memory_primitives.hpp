// File: kinesis_unified/memory_primitives.hpp
// Unified & Hardened Memory Subsystem pulled from AI_Library (Kinesis) into .Unified_Hardware_Accel.
// Fixes applied over original AI_Library:
//   1. Prefaulted + Page-Locked OS VirtualMemory (VirtualLock / mlock + 4KB page-stride touch)
//      to eliminate 4x first-touch page-fault latency spikes in hot loops (Design Review §1.10, T1).
//   2. Explicit MemorySpace tagging (HostHeap, HostPinnedLocked, CpuL3Locked, SingleGpuVram)
//      and ahead-of-time WorkspacePlanner so hot loops never allocate or throw (Design Review T1, T13).
//   3. Three-Lifetime Arena Manager (Persistent weights, Step activations/KV, Scratch workspace)
//      solving the two-arena training/inference lifetime conflict (Design Review T2).

#pragma once

#include "unified_sizing.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <sstream>
#include <type_traits>
#include <utility>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace kinesis::unified {

// Hardware alignment constants anchored to unified_sizing.h
inline constexpr size_t kGpuSectorAlignment    = UHA_GPU_SECTOR_BYTES;
inline constexpr size_t kCpuCacheLineAlignment = UHA_CPU_CACHELINE_BYTES;
inline constexpr size_t kGpuSramSwizzleAlign   = UHA_GPU_SRAM_SWIZZLE_BYTES;
inline constexpr size_t kDefaultAlignment      = UHA_CPU_CACHELINE_BYTES;
inline constexpr size_t kMaxDimensions         = UHA_MAX_TENSOR_DIMS;
inline constexpr size_t kKvPageTokens          = UHA_KV_PAGE_TOKENS;
inline constexpr size_t kKvDefaultHeadDim      = UHA_KV_DEFAULT_HEAD_DIM;
inline constexpr size_t kKvPageInt4Bytes       = UHA_KV_PAGE_INT4_BYTES;
inline constexpr size_t kKvPageBf16Bytes       = UHA_KV_PAGE_BF16_BYTES;
inline constexpr size_t kKvPageCentroidBytes   = UHA_KV_PAGE_CENTROID_BYTES;

// Explicit Memory Spaces (Fixes T13: First-class CPU/GPU memory space tracking)
enum class MemorySpace : uint8_t {
    HostHeap          = 0, // Standard 64B-aligned CPU heap
    HostPinnedLocked  = 1, // Prefaulted + OS page-locked (VirtualLock/mlock) zero-copy host RAM
    CpuL3CacheLocked  = 2, // Sub-2-bit micro-panels sized to stay resident in CPU L3 cache
    SingleGpuVram     = 3  // Hot-tier single-GPU device VRAM (or mapped BAR1 slot)
};

[[nodiscard]] constexpr size_t align_up(size_t address, size_t alignment) noexcept {
    return (address + alignment - 1) & ~(alignment - 1);
}

[[nodiscard]] inline bool is_aligned(const void* ptr, size_t alignment) noexcept {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// ============================================================================
// 1. Upgraded VirtualMemory with Prefaulting & Physical Page Locking
// ============================================================================

class VirtualMemory {
public:
    [[nodiscard]] static size_t page_size() noexcept {
#if defined(_WIN32) || defined(_WIN64)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return static_cast<size_t>(si.dwPageSize);
#else
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    }

    [[nodiscard]] static size_t allocation_granularity() noexcept {
#if defined(_WIN32) || defined(_WIN64)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return static_cast<size_t>(si.dwAllocationGranularity);
#else
        return page_size();
#endif
    }

    [[nodiscard]] static void* allocate(size_t bytes) noexcept {
        if (bytes == 0) return nullptr;
        const size_t aligned_size = align_up(bytes, page_size());
#if defined(_WIN32) || defined(_WIN64)
        return VirtualAlloc(nullptr, aligned_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return (ptr == MAP_FAILED) ? nullptr : ptr;
#endif
    }

    /**
     * @brief Allocates, prefaults every 4KB page into the OS page table, and locks physical
     *        frames in RAM (VirtualLock / mlock) so hot inference loops take zero page faults.
     */
    [[nodiscard]] static void* allocate_locked_prefaulted(size_t bytes, bool* out_locked = nullptr) noexcept {
        if (bytes == 0) return nullptr;
        const size_t pg = page_size();
        const size_t aligned_size = align_up(bytes, pg);
        void* ptr = allocate(aligned_size);
        if (!ptr) {
            if (out_locked) *out_locked = false;
            return nullptr;
        }

        // Deterministic page-stride write to prefault all PTEs before entering the hot loop
        volatile auto* byte_ptr = static_cast<volatile std::byte*>(ptr);
        for (size_t offset = 0; offset < aligned_size; offset += pg) {
            byte_ptr[offset] = std::byte{0};
        }

        bool locked = false;
#if defined(_WIN32) || defined(_WIN64)
        locked = (VirtualLock(ptr, aligned_size) != 0);
        if (!locked) {
            SIZE_T min_ws = 0, max_ws = 0;
            if (GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws, &max_ws)) {
                const SIZE_T needed = min_ws + aligned_size + (16ULL << 20);
                if (SetProcessWorkingSetSize(GetCurrentProcess(), needed, std::max(max_ws, needed * 2))) {
                    locked = (VirtualLock(ptr, aligned_size) != 0);
                }
            }
        }
#else
        locked = (mlock(ptr, aligned_size) == 0);
#endif
        if (out_locked) *out_locked = locked;
        return ptr;
    }

    static void free(void* ptr, size_t bytes, bool was_locked = false) noexcept {
        if (!ptr) return;
        const size_t aligned_size = align_up(bytes, page_size());
#if defined(_WIN32) || defined(_WIN64)
        if (was_locked) {
            VirtualUnlock(ptr, aligned_size);
        }
        VirtualFree(ptr, 0, MEM_RELEASE);
#else
        if (was_locked) {
            munlock(ptr, aligned_size);
        }
        munmap(ptr, aligned_size);
#endif
    }
};

// ============================================================================
// 2. Abstract Allocator & Prefaulted ArenaAllocator
// ============================================================================

class Allocator {
public:
    virtual ~Allocator() = default;
    [[nodiscard]] virtual void* allocate(size_t bytes, size_t alignment = kDefaultAlignment) = 0;
    virtual void deallocate(void* ptr, size_t bytes, size_t alignment = kDefaultAlignment) noexcept = 0;
    [[nodiscard]] virtual bool owns(const void* ptr) const noexcept = 0;
    [[nodiscard]] virtual MemorySpace memory_space() const noexcept { return MemorySpace::HostHeap; }
    virtual void reset() noexcept {}
};

class ArenaAllocator final : public Allocator {
public:
    explicit ArenaAllocator(size_t capacityBytes, bool lockAndPrefault = true,
                            MemorySpace space = MemorySpace::HostPinnedLocked)
        : capacity_(capacityBytes), owns_buffer_(true), space_(space) {
        if (capacityBytes > 0) {
            if (lockAndPrefault) {
                buffer_ = static_cast<std::byte*>(
                    VirtualMemory::allocate_locked_prefaulted(capacityBytes, &is_locked_)
                );
            } else {
                buffer_ = static_cast<std::byte*>(VirtualMemory::allocate(capacityBytes));
            }
            if (!buffer_) {
                throw std::bad_alloc();
            }
        }
    }

    explicit ArenaAllocator(std::span<std::byte> externalBuffer,
                            MemorySpace space = MemorySpace::HostHeap) noexcept
        : buffer_(externalBuffer.data()),
          capacity_(externalBuffer.size()),
          current_offset_(0),
          peak_offset_(0),
          owns_buffer_(false),
          is_locked_(false),
          space_(space) {}

    ~ArenaAllocator() override {
        if (owns_buffer_ && buffer_) {
            VirtualMemory::free(buffer_, capacity_, is_locked_);
            buffer_ = nullptr;
        }
    }

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    [[nodiscard]] void* allocate(size_t bytes, size_t alignment = kDefaultAlignment) override {
        if (bytes == 0) return nullptr;
        if (alignment == 0 || (alignment & (alignment - 1)) != 0 || !buffer_) {
            throw std::bad_alloc();
        }
        const uintptr_t current_addr = reinterpret_cast<uintptr_t>(buffer_) + current_offset_;
        const size_t padding = (alignment - (current_addr & (alignment - 1))) & (alignment - 1);
        if (padding > capacity_ - current_offset_) {
            throw std::bad_alloc();
        }
        const size_t offset_after_padding = current_offset_ + padding;
        if (bytes > capacity_ - offset_after_padding) {
            throw std::bad_alloc();
        }
        current_offset_ = offset_after_padding + bytes;
        if (current_offset_ > peak_offset_) {
            peak_offset_ = current_offset_;
        }
        return reinterpret_cast<void*>(current_addr + padding);
    }

    void deallocate(void*, size_t, size_t) noexcept override {}

    [[nodiscard]] bool owns(const void* ptr) const noexcept override {
        if (!buffer_ || !ptr) return false;
        const auto p_addr = reinterpret_cast<uintptr_t>(ptr);
        const auto base_addr = reinterpret_cast<uintptr_t>(buffer_);
        return (p_addr >= base_addr) && (p_addr < (base_addr + capacity_));
    }

    [[nodiscard]] MemorySpace memory_space() const noexcept override { return space_; }

    void reset() noexcept override { current_offset_ = 0; }

    [[nodiscard]] size_t allocated_bytes() const noexcept { return current_offset_; }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] size_t available_bytes() const noexcept {
        return (current_offset_ < capacity_) ? (capacity_ - current_offset_) : 0;
    }
    [[nodiscard]] size_t peak_bytes() const noexcept { return peak_offset_; }
    [[nodiscard]] bool is_page_locked() const noexcept { return is_locked_; }
    [[nodiscard]] std::byte* data() noexcept { return buffer_; }
    [[nodiscard]] const std::byte* data() const noexcept { return buffer_; }

    class [[nodiscard]] ScopedMarker {
    public:
        explicit ScopedMarker(ArenaAllocator& arena) noexcept
            : arena_(arena), saved_offset_(arena.current_offset_) {}
        ~ScopedMarker() noexcept { arena_.current_offset_ = saved_offset_; }
        ScopedMarker(const ScopedMarker&) = delete;
        ScopedMarker& operator=(const ScopedMarker&) = delete;
    private:
        ArenaAllocator& arena_;
        size_t saved_offset_;
    };

    [[nodiscard]] ScopedMarker scope() noexcept { return ScopedMarker(*this); }

private:
    std::byte*  buffer_{nullptr};
    size_t      capacity_{0};
    size_t      current_offset_{0};
    size_t      peak_offset_{0};
    bool        owns_buffer_{false};
    bool        is_locked_{false};
    MemorySpace space_{MemorySpace::HostPinnedLocked};
};

// ============================================================================
// 3. Three-Lifetime Arena System & Ahead-of-Time Workspace Planner (Fixes T1, T2)
// ============================================================================

struct WorkspacePlan {
    size_t persistent_weight_bytes{0}; // Lifetime 1: Packed sub-2-bit model weights & scales
    size_t step_activation_bytes{0};   // Lifetime 2: Forward activations / saved backward state
    size_t transient_scratch_bytes{0}; // Lifetime 3: Per-kernel GotoBLAS micro-panel scratchpad

    [[nodiscard]] constexpr size_t total_bytes() const noexcept {
        return persistent_weight_bytes + step_activation_bytes + transient_scratch_bytes;
    }
};

class ThreeTierArenaManager {
public:
    explicit ThreeTierArenaManager(const WorkspacePlan& plan)
        : plan_(plan),
          persistent_arena_(plan.persistent_weight_bytes, true, MemorySpace::CpuL3CacheLocked),
          step_arena_(plan.step_activation_bytes, true, MemorySpace::HostPinnedLocked),
          scratch_arena_(plan.transient_scratch_bytes, true, MemorySpace::HostPinnedLocked) {}

    [[nodiscard]] ArenaAllocator& persistent() noexcept { return persistent_arena_; }
    [[nodiscard]] ArenaAllocator& step() noexcept { return step_arena_; }
    [[nodiscard]] ArenaAllocator& scratch() noexcept { return scratch_arena_; }
    [[nodiscard]] const WorkspacePlan& plan() const noexcept { return plan_; }

    void end_step() noexcept {
        scratch_arena_.reset();
        step_arena_.reset();
    }

private:
    WorkspacePlan  plan_;
    ArenaAllocator persistent_arena_;
    ArenaAllocator step_arena_;
    ArenaAllocator scratch_arena_;
};

// ============================================================================
// 4. Prefaulted SlabAllocator for O(1) Paged KV-Cache & Weight Tile Recycling
// ============================================================================

template <size_t BlockSize, size_t Alignment = kDefaultAlignment>
class SlabAllocator final : public Allocator {
    static_assert(BlockSize >= sizeof(void*), "BlockSize must be >= sizeof(void*)");
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

public:
    static constexpr size_t kActualBlockSize = align_up(BlockSize, Alignment);

    explicit SlabAllocator(size_t blockCount, bool lockAndPrefault = true)
        : block_count_(blockCount),
          free_count_(blockCount),
          slab_bytes_(align_up(blockCount * kActualBlockSize, VirtualMemory::page_size())) {
        if (slab_bytes_ > 0) {
            if (lockAndPrefault) {
                slab_memory_ = static_cast<std::byte*>(
                    VirtualMemory::allocate_locked_prefaulted(slab_bytes_, &is_locked_)
                );
            } else {
                slab_memory_ = static_cast<std::byte*>(VirtualMemory::allocate(slab_bytes_));
            }
            if (!slab_memory_) throw std::bad_alloc();
            initialize_free_list();
        }
    }

    ~SlabAllocator() override {
        if (slab_memory_) {
            VirtualMemory::free(slab_memory_, slab_bytes_, is_locked_);
            slab_memory_ = nullptr;
        }
    }

    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;

    [[nodiscard]] void* allocate(size_t bytes = kActualBlockSize, size_t alignment = Alignment) override {
        if (bytes > kActualBlockSize || alignment > Alignment || !free_list_) {
            throw std::bad_alloc();
        }
        Node* node = free_list_;
        free_list_ = node->next;
        --free_count_;
        return static_cast<void*>(node);
    }

    void deallocate(void* ptr, size_t = kActualBlockSize, size_t = Alignment) noexcept override {
        if (!ptr || !owns(ptr)) return;
        const size_t offset = static_cast<size_t>(static_cast<std::byte*>(ptr) - slab_memory_);
        if (offset % kActualBlockSize != 0) return;

        auto* node = static_cast<Node*>(ptr);
        node->next = free_list_;
        free_list_ = node;
        ++free_count_;
    }

    [[nodiscard]] bool owns(const void* ptr) const noexcept override {
        if (!slab_memory_ || !ptr) return false;
        const auto p_addr = reinterpret_cast<uintptr_t>(ptr);
        const auto base_addr = reinterpret_cast<uintptr_t>(slab_memory_);
        const size_t usable_bytes = block_count_ * kActualBlockSize;
        return (p_addr >= base_addr) && (p_addr < (base_addr + usable_bytes));
    }

    [[nodiscard]] MemorySpace memory_space() const noexcept override {
        return MemorySpace::HostPinnedLocked;
    }

    void reset() noexcept override {
        free_count_ = block_count_;
        initialize_free_list();
    }

    [[nodiscard]] size_t block_size() const noexcept { return kActualBlockSize; }
    [[nodiscard]] size_t total_blocks() const noexcept { return block_count_; }
    [[nodiscard]] size_t free_blocks() const noexcept { return free_count_; }
    [[nodiscard]] size_t allocated_blocks() const noexcept { return block_count_ - free_count_; }

private:
    struct Node { Node* next; };

    void initialize_free_list() noexcept {
        if (!slab_memory_ || block_count_ == 0) {
            free_list_ = nullptr;
            return;
        }
        free_list_ = reinterpret_cast<Node*>(slab_memory_);
        Node* current = free_list_;
        for (size_t i = 0; i < block_count_ - 1; ++i) {
            auto* next = reinterpret_cast<Node*>(
                reinterpret_cast<std::byte*>(current) + kActualBlockSize
            );
            current->next = next;
            current = next;
        }
        current->next = nullptr;
    }

    std::byte* slab_memory_{nullptr};
    Node*      free_list_{nullptr};
    size_t     block_count_{0};
    size_t     free_count_{0};
    size_t     slab_bytes_{0};
    bool       is_locked_{false};
};

// ============================================================================
// 5. 64-Bit Safe Shape (56B), Strides (56B), and 128-Byte Cache-Aligned TensorView
//    Fixes Design Review L7 (TensorView size claims), T5 (64-bit extents), and T13
// ============================================================================

class Shape {
public:
    constexpr Shape() noexcept = default;

    constexpr Shape(std::initializer_list<int64_t> dims,
                    MemorySpace space = MemorySpace::HostPinnedLocked)
        : rank_(static_cast<uint32_t>(dims.size())),
          space_tag_(static_cast<uint32_t>(space)) {
        if (dims.size() > kMaxDimensions) {
            throw std::invalid_argument("Shape rank exceeds kMaxDimensions (6)");
        }
        size_t i = 0;
        for (int64_t d : dims) {
            if (d < 0) throw std::invalid_argument("Shape dimension must be non-negative");
            dims_[i++] = static_cast<uint64_t>(d);
        }
    }

    [[nodiscard]] constexpr size_t rank() const noexcept { return static_cast<size_t>(rank_); }
    [[nodiscard]] constexpr MemorySpace memory_space() const noexcept {
        return static_cast<MemorySpace>(space_tag_ & 0xFFU);
    }
    constexpr void set_memory_space(MemorySpace space) noexcept {
        space_tag_ = static_cast<uint32_t>(space);
    }

    [[nodiscard]] constexpr uint64_t operator[](size_t index) const noexcept {
        assert(index < rank_);
        return dims_[index];
    }
    [[nodiscard]] constexpr uint64_t& operator[](size_t index) noexcept {
        assert(index < rank_);
        return dims_[index];
    }

    /**
     * @brief Total element count with overflow saturation at UINT64_MAX.
     */
    [[nodiscard]] constexpr uint64_t total_elements() const noexcept {
        if (rank_ == 0) return 0;
        uint64_t total = 1;
        for (size_t i = 0; i < rank_; ++i) {
            const uint64_t dim = dims_[i];
            if (dim == 0) return 0;
            if (total > std::numeric_limits<uint64_t>::max() / dim) {
                return std::numeric_limits<uint64_t>::max();
            }
            total *= dim;
        }
        return total;
    }

    constexpr bool operator==(const Shape& other) const noexcept {
        return rank_ == other.rank_ && dims_ == other.dims_;
    }

private:
    uint32_t rank_{0};
    uint32_t space_tag_{static_cast<uint32_t>(MemorySpace::HostPinnedLocked)};
    std::array<uint64_t, kMaxDimensions> dims_{};
};

static_assert(sizeof(Shape) == UHA_SHAPE_DESC_BYTES,
              "Shape descriptor must be exactly 56 bytes");

class Strides {
public:
    constexpr Strides() noexcept = default;

    constexpr Strides(std::initializer_list<int64_t> s)
        : rank_(static_cast<uint32_t>(s.size())), reserved_(0) {
        if (s.size() > kMaxDimensions) {
            throw std::invalid_argument("Strides rank exceeds kMaxDimensions (6)");
        }
        size_t i = 0;
        for (int64_t v : s) strides_[i++] = v;
    }

    [[nodiscard]] static constexpr Strides from_shape_row_major(const Shape& shape) noexcept {
        Strides s;
        s.rank_ = static_cast<uint32_t>(shape.rank());
        if (s.rank_ == 0) return s;
        int64_t acc = 1;
        for (size_t i = s.rank_; i > 0; --i) {
            s.strides_[i - 1] = acc;
            acc *= static_cast<int64_t>(shape[i - 1]);
        }
        return s;
    }

    [[nodiscard]] constexpr size_t rank() const noexcept { return static_cast<size_t>(rank_); }
    [[nodiscard]] constexpr int64_t operator[](size_t index) const noexcept {
        assert(index < rank_);
        return strides_[index];
    }
    [[nodiscard]] constexpr int64_t& operator[](size_t index) noexcept {
        assert(index < rank_);
        return strides_[index];
    }

private:
    uint32_t rank_{0};
    uint32_t reserved_{0};
    std::array<int64_t, kMaxDimensions> strides_{};
};

static_assert(sizeof(Strides) == UHA_STRIDES_DESC_BYTES,
              "Strides descriptor must be exactly 56 bytes");

template <typename T>
class TensorView {
public:
    constexpr TensorView() noexcept = default;

    constexpr TensorView(T* data, Shape shape,
                         MemorySpace space = MemorySpace::HostPinnedLocked) noexcept
        : data_(data),
          offset_(0),
          shape_(shape),
          strides_(Strides::from_shape_row_major(shape)) {
        shape_.set_memory_space(space);
    }

    constexpr TensorView(T* data, Shape shape, Strides strides, int64_t offset = 0,
                         MemorySpace space = MemorySpace::HostPinnedLocked) noexcept
        : data_(data), offset_(offset), shape_(shape), strides_(strides) {
        shape_.set_memory_space(space);
    }

    template <typename U>
        requires(std::is_const_v<T> && std::is_same_v<const U, T>)
    constexpr TensorView(const TensorView<U>& other) noexcept
        : data_(other.raw_base()),
          offset_(other.offset()),
          shape_(other.shape()),
          strides_(other.strides()) {}

    [[nodiscard]] constexpr T* raw_base() const noexcept { return data_; }
    [[nodiscard]] constexpr T* data() const noexcept { return data_ + offset_; }
    [[nodiscard]] constexpr const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] constexpr const Strides& strides() const noexcept { return strides_; }
    [[nodiscard]] constexpr int64_t offset() const noexcept { return offset_; }
    [[nodiscard]] constexpr MemorySpace memory_space() const noexcept {
        return shape_.memory_space();
    }
    [[nodiscard]] constexpr uint64_t size(size_t dim) const noexcept { return shape_[dim]; }
    [[nodiscard]] constexpr int64_t stride(size_t dim) const noexcept { return strides_[dim]; }
    [[nodiscard]] constexpr uint64_t total_elements() const noexcept {
        return shape_.total_elements();
    }

    [[nodiscard]] constexpr T& operator()(int64_t i, int64_t j) const noexcept {
        assert(shape_.rank() == 2);
        return data_[offset_ + i * strides_[0] + j * strides_[1]];
    }

    [[nodiscard]] TensorView transpose(size_t d0, size_t d1) const {
        if (d0 >= shape_.rank() || d1 >= shape_.rank()) {
            throw std::out_of_range("TensorView::transpose out of range");
        }
        Shape ns = shape_;
        Strides nst = strides_;
        std::swap(ns[d0], ns[d1]);
        std::swap(nst[d0], nst[d1]);
        return TensorView(data_, ns, nst, offset_, shape_.memory_space());
    }

private:
    T*      data_{nullptr}; // 8 bytes
    int64_t offset_{0};     // 8 bytes
    Shape   shape_{};       // 56 bytes
    Strides strides_{};     // 56 bytes -> Total = 128 bytes (exact 2 x 64B cache lines)
};

static_assert(sizeof(TensorView<float>) == UHA_TENSOR_VIEW_DESC_BYTES,
              "TensorView must be exactly 128 bytes (2 x 64B CPU cache lines / 1 x 128B GPU sector)");

} // namespace kinesis::unified
