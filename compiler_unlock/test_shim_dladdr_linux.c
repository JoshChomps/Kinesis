// File: compiler_unlock/test_shim_dladdr_linux.c
// Linux-only test of the shim's caller classification by dladdr() (docs/12 4.16). The Windows suite A1 can only
// exercise the forced-override and TLS-latch paths, because dladdr does not exist there.
//
// One source, three roles selected by macro:
//   -DROLE_CALLER_LIB : a tiny shared library exporting caller_get_name()/caller_get_pci_attr(). Built twice,
//                       once as libcublasLt_mockcaller.so (name contains "libcublas") and once as
//                       libother_mockcaller.so (it does not).
//   (default)         : the test driver. It registers a mock driver table with the shim, leaves the forced
//                       override unset and the TLS latch at 0, and calls through both libraries.
// Host mock only: no CUDA driver or GPU is involved. Built and run by verify_shim_dladdr_linux.py under WSL.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef int cudaError_t;

#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34

extern CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
extern CUresult cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev);

#if defined(ROLE_CALLER_LIB)
// Built with -fno-optimize-sibling-calls so the call is not turned into a jump: the shim reads
// __builtin_return_address(0), which must land inside this library for dladdr to name it.
int caller_get_name(char* buf, int len, int dev) {
    int r = cuDeviceGetName(buf, len, dev);
    return r;
}
int caller_get_pci_attr(int* v, int dev) {
    int r = cuDeviceGetAttribute(v, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
    return r;
}
#else
#include <assert.h>
#include <dlfcn.h>

typedef CUresult (*cuDeviceGetAttribute_fn)(int*, int, CUdevice);
typedef CUresult (*cuDeviceGetName_fn)(char*, int, CUdevice);
typedef CUresult (*cuGetProcAddress_fn)(const char*, void**, int, uint64_t);
typedef CUresult (*cuGetProcAddress_v2_fn)(const char*, void**, int, uint64_t, void*);
typedef cudaError_t (*cudaDeviceGetAttribute_fn)(int*, int, int);
extern void shim_register_mock_driver_table(cuDeviceGetAttribute_fn, cuDeviceGetName_fn,
                                            cudaDeviceGetAttribute_fn, cuGetProcAddress_fn,
                                            cuGetProcAddress_v2_fn);

// dev 0 reports a synthetic packed product ID (what the mock in A1 uses); dev 1 reports a PCI slot number,
// which is what CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID returns on a real driver (0..31).
static CUresult mock_attr(int* pi, int attrib, CUdevice dev) {
    if (!pi || attrib != CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID) return -1;
    *pi = (dev == 0) ? 0x2684 : 3;
    return CUDA_SUCCESS;
}
static CUresult mock_name(char* name, int len, CUdevice dev) {
    (void)dev;
    snprintf(name, (size_t)len, "NVIDIA GeForce RTX 4090");
    return CUDA_SUCCESS;
}

typedef int (*caller_name_fn)(char*, int, int);
typedef int (*caller_attr_fn)(int*, int);

int main(int argc, char** argv) {
    assert(argc == 3);
    shim_register_mock_driver_table(mock_attr, mock_name, NULL, NULL, NULL);

    void* h_cublas = dlopen(argv[1], RTLD_NOW);
    void* h_other = dlopen(argv[2], RTLD_NOW);
    assert(h_cublas && h_other);
    caller_name_fn name_cublas = (caller_name_fn)dlsym(h_cublas, "caller_get_name");
    caller_name_fn name_other = (caller_name_fn)dlsym(h_other, "caller_get_name");
    caller_attr_fn attr_cublas = (caller_attr_fn)dlsym(h_cublas, "caller_get_pci_attr");
    caller_attr_fn attr_other = (caller_attr_fn)dlsym(h_other, "caller_get_pci_attr");
    assert(name_cublas && name_other && attr_cublas && attr_other);

    char buf[64];
    int fails = 0;

    // (1) Direct call from the executable: not a cuBLAS DSO, so no rewrite.
    cuDeviceGetName(buf, sizeof(buf), 0);
    printf("[1] caller = test executable         : \"%s\"\n", buf);
    fails += strcmp(buf, "NVIDIA GeForce RTX 4090") != 0;

    // (2) From libother_mockcaller.so: dladdr names a non-cuBLAS DSO, so no rewrite.
    name_other(buf, sizeof(buf), 0);
    printf("[2] caller = libother_mockcaller.so  : \"%s\"\n", buf);
    fails += strcmp(buf, "NVIDIA GeForce RTX 4090") != 0;

    // (3) From libcublasLt_mockcaller.so: dladdr names a libcublas* DSO, so the name is rewritten.
    name_cublas(buf, sizeof(buf), 0);
    printf("[3] caller = libcublasLt_mockcaller  : \"%s\"\n", buf);
    fails += strcmp(buf, "NVIDIA RTX 6000 Ada Generation") != 0;

    // (4) Attribute 34 with a synthetic product ID is remapped only for the cuBLAS caller.
    int v_c = 0, v_o = 0;
    attr_cublas(&v_c, 0);
    attr_other(&v_o, 0);
    printf("[4] attr 34, synthetic ID 0x2684     : cublas caller 0x%X | other caller 0x%X\n", v_c, v_o);
    fails += (v_c != 0x26B1) || (v_o != 0x2684);

    // (5) Attribute 34 as a real driver reports it (a PCI slot number, here 3): the >= 0x1000 gate never
    //     fires, so this path is a no-op outside the mock even for a cuBLAS caller (docs/12 4.16).
    int slot = 0;
    attr_cublas(&slot, 1);
    printf("[5] attr 34, real-driver slot value  : cublas caller %d (unchanged)\n", slot);
    fails += slot != 3;

    dlclose(h_cublas);
    dlclose(h_other);
    printf(fails == 0 ? "[+] PASS: dladdr caller classification verified on Linux (5/5).\n"
                      : "[-] FAIL: %d dladdr classification cases failed.\n", fails);
    return fails == 0 ? 0 : 1;
}
#endif
