// File: compiler_unlock/libcublas_unlock_shim.c
// Build (Linux): gcc -O3 -shared -fPIC -o libcublas_unlock_shim.so libcublas_unlock_shim.c -ldl
// Usage: LD_PRELOAD=./libcublas_unlock_shim.so python3 train_or_infer.py

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>

#if defined(__linux__) || defined(__unix__)
#include <dlfcn.h>
#include <sys/ioctl.h>
#endif

typedef int CUresult;
typedef int CUdevice;
typedef int cudaError_t;

#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16

// PCI Device IDs (Vendor 0x10DE)
#define PCI_ID_RTX_3090_TI   0x2203
#define PCI_ID_RTX_3090      0x2204
#define PCI_ID_RTX_3080_10G  0x2206
#define PCI_ID_RTX_A6000     0x2230
#define PCI_ID_RTX_4090      0x2684
#define PCI_ID_RTX_4090_D    0x2685   // from community reports; unverified on hardware
#define PCI_ID_RTX_6000_ADA  0x26B1
#define PCI_ID_RTX_5090      0x2B85   // from community reports; unverified on hardware
#define PCI_ID_RTX_PRO_6000_BW 0x2BB1 // from community reports; unverified on hardware

// NVRM ioctl structures for NV_ESC_RM_CONTROL (cmd 0x2A)
#define NV_IOCTL_MAGIC                       0x46 /* 'F' */
#define NV_ESC_RM_CONTROL                    0x2A
#define NV2080_CTRL_CMD_GPU_GET_NAME_STRING  0x20800110U
#define NV2080_CTRL_CMD_BUS_GET_PCI_INFO     0x20801801U
#define NV0000_CTRL_CMD_GPU_GET_ID_INFO      0x00000202U

typedef struct {
    uint32_t pciDeviceId;    // Upper 16 bits = PCI Device ID (e.g. 0x2684), Lower 16 bits = Vendor ID (0x10DE)
    uint32_t pciSubSystemId;
    uint32_t pciRevisionId;
    uint32_t pciExtDeviceId;
} NV2080_CTRL_BUS_GET_PCI_INFO_PARAMS;

typedef struct {
    uint32_t hClient;
    uint32_t hObject;
    uint32_t cmd;
    uint32_t flags;
    void*    params;
    uint32_t paramsSize;
    uint32_t status;
} NVOS54_PARAMETERS;

typedef CUresult (*cuDeviceGetAttribute_fn)(int*, int, CUdevice);
typedef CUresult (*cuDeviceGetName_fn)(char*, int, CUdevice);
typedef CUresult (*cuGetProcAddress_fn)(const char*, void**, int, uint64_t);
typedef CUresult (*cuGetProcAddress_v2_fn)(const char*, void**, int, uint64_t, void*);
typedef cudaError_t (*cudaDeviceGetAttribute_fn)(int*, int, int);
typedef int (*ioctl_fn)(int, unsigned long, ...);

static cuDeviceGetAttribute_fn real_cuDeviceGetAttribute = NULL;
static cuDeviceGetName_fn real_cuDeviceGetName = NULL;
static cuGetProcAddress_fn real_cuGetProcAddress = NULL;
static cuGetProcAddress_v2_fn real_cuGetProcAddress_v2 = NULL;
static cudaDeviceGetAttribute_fn real_cudaDeviceGetAttribute = NULL;
static ioctl_fn real_ioctl = NULL;

// Optional test-harness override for sandbox caller simulation
static int g_force_caller_is_cublas = -1;

void shim_set_mock_caller_is_cublas(int is_cublas) {
    g_force_caller_is_cublas = is_cublas;
}

void shim_register_mock_driver_table(
    cuDeviceGetAttribute_fn fn_attr,
    cuDeviceGetName_fn fn_name,
    cudaDeviceGetAttribute_fn fn_rt_attr,
    cuGetProcAddress_fn fn_gpa,
    cuGetProcAddress_v2_fn fn_gpa_v2
) {
    real_cuDeviceGetAttribute = fn_attr;
    real_cuDeviceGetName = fn_name;
    real_cudaDeviceGetAttribute = fn_rt_attr;
    real_cuGetProcAddress = fn_gpa;
    real_cuGetProcAddress_v2 = fn_gpa_v2;
}

static inline int is_caller_cublas(void* return_addr) {
    if (g_force_caller_is_cublas >= 0) {
        return g_force_caller_is_cublas;
    }
#if defined(__linux__) || defined(__unix__)
    Dl_info info;
    if (dladdr(return_addr, &info) && info.dli_fname) {
        if (strstr(info.dli_fname, "libcublas") != NULL ||
            strstr(info.dli_fname, "libhipblas") != NULL ||
            strstr(info.dli_fname, "libcudnn") != NULL) {
            return 1;
        }
    }
#else
    (void)return_addr;
#endif
    return 0;
}

static inline int map_consumer_to_workstation_pci_id(int raw_id) {
    switch (raw_id & 0xFFFF) {
        case PCI_ID_RTX_3090_TI:
        case PCI_ID_RTX_3090:
        case PCI_ID_RTX_3080_10G:
            return PCI_ID_RTX_A6000;
        case PCI_ID_RTX_4090:
        case PCI_ID_RTX_4090_D:
            return PCI_ID_RTX_6000_ADA;
        case PCI_ID_RTX_5090:
            return PCI_ID_RTX_PRO_6000_BW;
        default:
            return raw_id;
    }
}

CUresult cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev) {
#if defined(__linux__) || defined(__unix__)
    if (!real_cuDeviceGetAttribute) {
        real_cuDeviceGetAttribute = (cuDeviceGetAttribute_fn)dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
    }
#endif
    if (!real_cuDeviceGetAttribute) {
        return -1;
    }
    CUresult res = real_cuDeviceGetAttribute(pi, attrib, dev);
    // Note: In CUDA, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID (34) returns the 5-bit PCIe slot number (0..31)
    // unless a synthetic/packed 16-bit hardware ID (>= 0x1000) is queried. Only remap when >= 0x1000
    // so genuine PCIe bus slot numbers (0..31) are never corrupted.
    if (res == CUDA_SUCCESS && pi != NULL && attrib == CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID && *pi >= 0x1000) {
        void* caller = __builtin_return_address(0);
        if (is_caller_cublas(caller)) {
            *pi = map_consumer_to_workstation_pci_id(*pi);
        }
    }
    return res;
}

cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device) {
#if defined(__linux__) || defined(__unix__)
    if (!real_cudaDeviceGetAttribute) {
        real_cudaDeviceGetAttribute = (cudaDeviceGetAttribute_fn)dlsym(RTLD_NEXT, "cudaDeviceGetAttribute");
    }
#endif
    if (!real_cudaDeviceGetAttribute) {
        return -1;
    }
    cudaError_t res = real_cudaDeviceGetAttribute(value, attr, device);
    if (res == 0 && value != NULL && attr == CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID && *value >= 0x1000) {
        void* caller = __builtin_return_address(0);
        if (is_caller_cublas(caller)) {
            *value = map_consumer_to_workstation_pci_id(*value);
        }
    }
    return res;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
#if defined(__linux__) || defined(__unix__)
    if (!real_cuDeviceGetName) {
        real_cuDeviceGetName = (cuDeviceGetName_fn)dlsym(RTLD_NEXT, "cuDeviceGetName");
    }
#endif
    if (!real_cuDeviceGetName) {
        return -1;
    }
    CUresult res = real_cuDeviceGetName(name, len, dev);
    if (res == CUDA_SUCCESS && name != NULL) {
        void* caller = __builtin_return_address(0);
        if (is_caller_cublas(caller)) {
            if (strstr(name, "3090") || strstr(name, "3080")) {
                snprintf(name, (size_t)len, "NVIDIA RTX A6000");
            } else if (strstr(name, "4090")) {
                snprintf(name, (size_t)len, "NVIDIA RTX 6000 Ada Generation");
            } else if (strstr(name, "5090")) {
                snprintf(name, (size_t)len, "NVIDIA RTX PRO 6000 Blackwell");
            }
        }
    }
    return res;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, uint64_t flags);
CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, uint64_t flags, void* symbolStatus);

// Substitutes our hooks into the pointer returned by the real resolver. CUDA 12.x libraries ask the
// resolver for the resolver itself ("cuGetProcAddress" / "cuGetProcAddress_v2") and then use that raw
// pointer for every later lookup, which would bypass the shim if we did not trap those names too.
// Each real_* pointer is only updated when it differs from our own symbol (self-assignment guard).
static void shim_substitute_hook(const char* symbol, void** pfn) {
    if (strcmp(symbol, "cuDeviceGetAttribute") == 0) {
        if (*pfn != (void*)&cuDeviceGetAttribute) real_cuDeviceGetAttribute = (cuDeviceGetAttribute_fn)(*pfn);
        *pfn = (void*)&cuDeviceGetAttribute;
    } else if (strcmp(symbol, "cuDeviceGetName") == 0) {
        if (*pfn != (void*)&cuDeviceGetName) real_cuDeviceGetName = (cuDeviceGetName_fn)(*pfn);
        *pfn = (void*)&cuDeviceGetName;
    } else if (strcmp(symbol, "cuGetProcAddress") == 0) {
        if (*pfn != (void*)&cuGetProcAddress) real_cuGetProcAddress = (cuGetProcAddress_fn)(*pfn);
        *pfn = (void*)&cuGetProcAddress;
    } else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        if (*pfn != (void*)&cuGetProcAddress_v2) real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_fn)(*pfn);
        *pfn = (void*)&cuGetProcAddress_v2;
    }
}

// Intercept CUDA 11.3+ / 12.x / 13.x symbol resolution table (cuGetProcAddress)
CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, uint64_t flags) {
#if defined(__linux__) || defined(__unix__)
    if (!real_cuGetProcAddress) {
        real_cuGetProcAddress = (cuGetProcAddress_fn)dlsym(RTLD_NEXT, "cuGetProcAddress");
    }
#endif
    if (!real_cuGetProcAddress) {
        return -1;
    }
    CUresult res = real_cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    if (res == CUDA_SUCCESS && symbol && pfn && *pfn) {
        shim_substitute_hook(symbol, pfn);
    }
    return res;
}

CUresult cuGetProcAddress_v2(
    const char* symbol,
    void** pfn,
    int cudaVersion,
    uint64_t flags,
    void* symbolStatus
) {
#if defined(__linux__) || defined(__unix__)
    if (!real_cuGetProcAddress_v2) {
        real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_fn)dlsym(RTLD_NEXT, "cuGetProcAddress_v2");
    }
#endif
    if (!real_cuGetProcAddress_v2) {
        return -1;
    }
    CUresult res = real_cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
    // symbolStatus (CUdriverProcAddressQueryResult*) is written by the real resolver and is left untouched:
    // when we substitute a hook, the real resolver already reported SUCCESS for that symbol.
    if (res == CUDA_SUCCESS && symbol && pfn && *pfn) {
        shim_substitute_hook(symbol, pfn);
    }
    return res;
}

// Helper for NVRM NV2080_CTRL_CMD_BUS_GET_PCI_INFO (0x20801801U) packet spoofing
void shim_spoof_nvos54_packet(NVOS54_PARAMETERS* nvos) {
    if (!nvos || !nvos->params) return;
    if (nvos->cmd == NV2080_CTRL_CMD_BUS_GET_PCI_INFO &&
        nvos->paramsSize >= sizeof(NV2080_CTRL_BUS_GET_PCI_INFO_PARAMS)) {
        NV2080_CTRL_BUS_GET_PCI_INFO_PARAMS* pci_info =
            (NV2080_CTRL_BUS_GET_PCI_INFO_PARAMS*)nvos->params;
        uint16_t vendor_id = (uint16_t)(pci_info->pciDeviceId & 0xFFFFU);
        uint16_t raw_dev_id = (uint16_t)(pci_info->pciDeviceId >> 16);
        if (vendor_id == 0x10DEU) {
            uint16_t spoofed_dev_id = (uint16_t)map_consumer_to_workstation_pci_id((int)raw_dev_id);
            pci_info->pciDeviceId = ((uint32_t)spoofed_dev_id << 16) | (uint32_t)vendor_id;
            pci_info->pciExtDeviceId = (uint32_t)spoofed_dev_id;
        }
    } else if (nvos->cmd == NV2080_CTRL_CMD_GPU_GET_NAME_STRING) {
        char* name_str = (char*)nvos->params + 4;
        if (strstr(name_str, "3090") || strstr(name_str, "3080")) {
            snprintf(name_str, 64, "NVIDIA RTX A6000");
        } else if (strstr(name_str, "4090")) {
            snprintf(name_str, 64, "NVIDIA RTX 6000 Ada Generation");
        } else if (strstr(name_str, "5090")) {
            snprintf(name_str, 64, "NVIDIA RTX PRO 6000 Blackwell");
        }
    }
}

#if defined(__linux__) || defined(__unix__)
int ioctl(int fd, unsigned long request, ...) {
    if (!real_ioctl) {
        real_ioctl = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");
    }
    va_list args;
    va_start(args, request);
    void* argp = va_arg(args, void*);
    va_end(args);

    int ret = real_ioctl(fd, request, argp);
    // Note: ioctl() is invoked from inside libcuda.so.1 during context initialization,
    // so __builtin_return_address(0) points into libcuda.so.1 rather than libcublas.so.
    // _IOC_TYPE must be NVRM's 'F': matching only the NR (0x2A) would reinterpret the argument of
    // unrelated ioctls (DRM, V4L2, ...) as an NVOS54 packet and read/write arbitrary memory.
    if (ret == 0 && _IOC_TYPE(request) == NV_IOCTL_MAGIC &&
        _IOC_NR(request) == NV_ESC_RM_CONTROL && argp != NULL) {
        shim_spoof_nvos54_packet((NVOS54_PARAMETERS*)argp);
    }
    return ret;
}
#endif
