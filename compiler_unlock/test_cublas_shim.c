// File: compiler_unlock/test_cublas_shim.c
// Mock libcuda.so test harness verifying cuGetProcAddress_v2 function-pointer table redirection
// Build: gcc -O3 -std=c11 -o test_cublas_shim test_cublas_shim.c libcublas_unlock_shim.c

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

typedef int CUresult;
typedef int CUdevice;
typedef int cudaError_t;

#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16

typedef CUresult (*cuDeviceGetAttribute_fn)(int*, int, CUdevice);
typedef CUresult (*cuDeviceGetName_fn)(char*, int, CUdevice);
typedef CUresult (*cuGetProcAddress_fn)(const char*, void**, int, uint64_t);
typedef CUresult (*cuGetProcAddress_v2_fn)(const char*, void**, int, uint64_t, void*);
typedef cudaError_t (*cudaDeviceGetAttribute_fn)(int*, int, int);

extern void shim_set_mock_caller_is_cublas(int is_cublas);
extern void shim_register_mock_driver_table(
    cuDeviceGetAttribute_fn fn_attr,
    cuDeviceGetName_fn fn_name,
    cudaDeviceGetAttribute_fn fn_rt_attr,
    cuGetProcAddress_fn fn_gpa,
    cuGetProcAddress_v2_fn fn_gpa_v2
);
extern CUresult cuGetProcAddress_v2(
    const char* symbol,
    void** pfn,
    int cudaVersion,
    uint64_t flags,
    void* symbolStatus
);

// Simulated physical GPUs:
//   dev 0: RTX 3090    (PCI ID 0x2204, 82 SMs)
//   dev 1: RTX 3090 Ti (PCI ID 0x2203, 84 SMs)
//   dev 2: RTX 4090    (PCI ID 0x2684, 128 SMs)
static CUresult mock_real_cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev) {
    if (!pi) return -1;
    if (attrib == CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID) {
        if (dev == 0) *pi = 0x2204;
        else if (dev == 1) *pi = 0x2203;
        else if (dev == 3) *pi = 0x2685;
        else if (dev == 4) *pi = 0x2B85;
        else *pi = 0x2684;
        return CUDA_SUCCESS;
    }
    if (attrib == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) {
        if (dev == 0) *pi = 82;
        else if (dev == 1) *pi = 84;
        else if (dev == 4) *pi = 170;
        else *pi = 128;
        return CUDA_SUCCESS;
    }
    return -1;
}

static cudaError_t mock_real_cudaDeviceGetAttribute(int* value, int attr, int device) {
    return (cudaError_t)mock_real_cuDeviceGetAttribute(value, attr, (CUdevice)device);
}

static CUresult mock_real_cuDeviceGetName(char* name, int len, CUdevice dev) {
    if (!name || len <= 0) return -1;
    if (dev == 0) snprintf(name, (size_t)len, "NVIDIA GeForce RTX 3090");
    else if (dev == 1) snprintf(name, (size_t)len, "NVIDIA GeForce RTX 3090 Ti");
    else if (dev == 3) snprintf(name, (size_t)len, "NVIDIA GeForce RTX 4090 D");
    else if (dev == 4) snprintf(name, (size_t)len, "NVIDIA GeForce RTX 5090");
    else snprintf(name, (size_t)len, "NVIDIA GeForce RTX 4090");
    return CUDA_SUCCESS;
}

static CUresult mock_real_cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, uint64_t flags);
static CUresult mock_real_cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, uint64_t flags, void* symbolStatus);

static CUresult mock_real_cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, uint64_t flags) {
    (void)cudaVersion;
    (void)flags;
    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        *pfn = (void*)&mock_real_cuGetProcAddress;
        return CUDA_SUCCESS;
    }
    if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        *pfn = (void*)&mock_real_cuGetProcAddress_v2;
        return CUDA_SUCCESS;
    }
    if (strcmp(symbol, "cuDeviceGetAttribute") == 0) {
        *pfn = (void*)&mock_real_cuDeviceGetAttribute;
        return CUDA_SUCCESS;
    }
    if (strcmp(symbol, "cuDeviceGetName") == 0) {
        *pfn = (void*)&mock_real_cuDeviceGetName;
        return CUDA_SUCCESS;
    }
    return -1;
}

static CUresult mock_real_cuGetProcAddress_v2(
    const char* symbol,
    void** pfn,
    int cudaVersion,
    uint64_t flags,
    void* symbolStatus
) {
    (void)symbolStatus;
    return mock_real_cuGetProcAddress(symbol, pfn, cudaVersion, flags);
}

int main(void) {
    printf("================================================================================\n");
    printf("  WORKSTREAM 2: CUBLAS HEURISTIC SPOOFING SHIM & cuGetProcAddress_v2 VERIFIER\n");
    printf("================================================================================\n\n");

    shim_register_mock_driver_table(
        mock_real_cuDeviceGetAttribute,
        mock_real_cuDeviceGetName,
        mock_real_cudaDeviceGetAttribute,
        mock_real_cuGetProcAddress,
        mock_real_cuGetProcAddress_v2
    );

    // Resolve internal dispatch pointers via cuGetProcAddress_v2 just as CUDA 12/13 libcublasLt.so does
    cuDeviceGetAttribute_fn pfn_get_attr = NULL;
    cuDeviceGetName_fn pfn_get_name = NULL;

    CUresult r1 = cuGetProcAddress_v2("cuDeviceGetAttribute", (void**)&pfn_get_attr, 12040, 0, NULL);
    CUresult r2 = cuGetProcAddress_v2("cuDeviceGetName", (void**)&pfn_get_name, 12040, 0, NULL);
    assert(r1 == CUDA_SUCCESS && pfn_get_attr != NULL);
    assert(r2 == CUDA_SUCCESS && pfn_get_name != NULL);

    printf("+----------+----------------------+--------------------+--------------------+----------+\n");
    printf("| Device   | Caller Context       | Reported PCI ID    | Reported Name      | SM Count |\n");
    printf("+----------+----------------------+--------------------+--------------------+----------+\n");

    const char* dev_labels[3] = {"RTX 3090", "3090 Ti ", "RTX 4090"};
    int expected_raw_pci[3]   = {0x2204, 0x2203, 0x2684};
    int expected_spoof_pci[3] = {0x2230, 0x2230, 0x26B1};
    int expected_sms[3]       = {82, 84, 128};

    for (int dev = 0; dev < 3; ++dev) {
        // Case A: Non-cuBLAS caller (PyTorch / NCCL) -> receives physical PCI ID & true SM count
        shim_set_mock_caller_is_cublas(0);
        int pci_raw = 0, sm_raw = 0;
        char name_raw[64] = {0};
        pfn_get_attr(&pci_raw, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
        pfn_get_attr(&sm_raw, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
        pfn_get_name(name_raw, sizeof(name_raw), dev);

        assert(pci_raw == expected_raw_pci[dev]);
        assert(sm_raw == expected_sms[dev]);
        printf("| %s | PyTorch / NCCL       | 0x%04X (Physical)  | %-18.18s | %8d |\n",
               dev_labels[dev], pci_raw, name_raw, sm_raw);

        // Case B: libcublasLt.so caller -> receives Workstation PCI ID & Name, but true physical SM count!
        shim_set_mock_caller_is_cublas(1);
        int pci_spoof = 0, sm_spoof = 0;
        char name_spoof[64] = {0};
        pfn_get_attr(&pci_spoof, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
        pfn_get_attr(&sm_spoof, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
        pfn_get_name(name_spoof, sizeof(name_spoof), dev);

        assert(pci_spoof == expected_spoof_pci[dev]);
        assert(sm_spoof == expected_sms[dev]);
        printf("| %s | libcublasLt.so       | 0x%04X (Spoofed)   | %-18.18s | %8d |\n",
               dev_labels[dev], pci_spoof, name_spoof, sm_spoof);
    }
    printf("+----------+----------------------+--------------------+--------------------+----------+\n");

    // RTX 4090 D (dev 3) and RTX 5090 (dev 4) must map to their workstation parts for cuBLAS only
    {
        int pci = 0, sm = 0;
        char nm[64] = {0};
        shim_set_mock_caller_is_cublas(1);
        pfn_get_attr(&pci, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, 3);
        assert(pci == 0x26B1);
        pfn_get_name(nm, sizeof(nm), 3);
        assert(strstr(nm, "6000 Ada") != NULL);
        pfn_get_attr(&pci, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, 4);
        pfn_get_attr(&sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 4);
        assert(pci == 0x2BB1 && sm == 170);
        pfn_get_name(nm, sizeof(nm), 4);
        assert(strstr(nm, "Blackwell") != NULL);
        shim_set_mock_caller_is_cublas(0);
        pfn_get_attr(&pci, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, 4);
        assert(pci == 0x2B85);
    }

    // CUDA 12.x asks the resolver for the resolver itself; the pointer it gets back MUST be our hook,
    // otherwise every later lookup through that raw pointer bypasses the shim.
    {
        cuGetProcAddress_v2_fn self_v2 = NULL;
        CUresult rs = cuGetProcAddress_v2("cuGetProcAddress_v2", (void**)&self_v2, 12040, 0, NULL);
        assert(rs == CUDA_SUCCESS && self_v2 == &cuGetProcAddress_v2);
        // Resolve a hooked symbol through the pointer we were handed and confirm it is still hooked
        cuDeviceGetAttribute_fn via_self = NULL;
        rs = self_v2("cuDeviceGetAttribute", (void**)&via_self, 12040, 0, NULL);
        assert(rs == CUDA_SUCCESS && via_self != NULL);
        shim_set_mock_caller_is_cublas(1);
        int pci = 0;
        via_self(&pci, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, 2);
        assert(pci == 0x26B1);
        shim_set_mock_caller_is_cublas(0);
        printf("[+] Resolver self-interception (cuGetProcAddress_v2 -> cuGetProcAddress_v2) verified.\n");
    }

    // Verify self-assignment recursion guard when cuGetProcAddress_v2 is invoked repeatedly
    // and verify NVRM NV2080_CTRL_CMD_BUS_GET_PCI_INFO (0x20801801U) packet spoofing
    typedef struct {
        uint32_t pciDeviceId;
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
    extern void shim_spoof_nvos54_packet(NVOS54_PARAMETERS* nvos);

    NV2080_CTRL_BUS_GET_PCI_INFO_PARAMS pci_params = {(0x2684U << 16) | 0x10DEU, 0, 0xA1U, 0x2684U};
    NVOS54_PARAMETERS nvos = {1, 2, 0x20801801U, 0, &pci_params, sizeof(pci_params), 0};
    shim_spoof_nvos54_packet(&nvos);
    assert((pci_params.pciDeviceId >> 16) == 0x26B1U);
    assert(pci_params.pciExtDeviceId == 0x26B1U);

    printf("[+] PASS: cuGetProcAddress_v2 & NV2080_CTRL_CMD_BUS_GET_PCI_INFO (0x20801801) verified.\n");

    return 0;
}
