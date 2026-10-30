# Master Makefile: Unified Unlocked GPU Silicon, Zero-Copy Bus Fabrics, and Fused Sub-2-Bit Kernels

CC       ?= gcc
CXX      ?= g++
NVCC     ?= nvcc
PYTHON   ?= python
CFLAGS   := -O3 -std=c11 -Wall -Wextra
CXXFLAGS := -O3 -std=c++20 -mavx2 -mfma -Wall -Wextra

.PHONY: all sim shim kernel_patch hetero kinesis_unified python_tests verify clean

all: verify

sim:
	$(CC) $(CFLAGS) -o sim_harness/ptx_bf162_emu sim_harness/ptx_bf162_emu.c -lm

shim:
	$(CC) $(CFLAGS) -o compiler_unlock/test_cublas_shim compiler_unlock/test_cublas_shim.c compiler_unlock/libcublas_unlock_shim.c

shim_so:
	$(CC) -O3 -shared -fPIC -o compiler_unlock/libcublas_unlock_shim.so compiler_unlock/libcublas_unlock_shim.c -ldl

kernel_patch:
	$(CC) $(CFLAGS) -DSTANDALONE_VERIFY -o kernel_driver_patch/test_kernel_patch kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c

hetero:
	$(CC) $(CFLAGS) -DSTANDALONE_VERIFY -o hetero_runtime/test_chorus_amx hetero_runtime/chorus_amx_moe.c -lm
	$(CC) $(CFLAGS) -o hetero_runtime/test_governor_c hetero_runtime/closed_loop_governor.c

kinesis_unified:
	$(CXX) $(CXXFLAGS) -o kinesis_unified/test_kinesis_unified kinesis_unified/test_kinesis_unified.cpp

cuda_kernels:
	$(NVCC) -O3 -arch=sm_86 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -c fused_w1_hadacore/w1a16_hadacore_kernel.cu -o fused_w1_hadacore/w1a16_hadacore_kernel.o
	$(NVCC) -O3 -arch=sm_86 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -c structural_filters/tiled_cbc_sieve.cu -o structural_filters/tiled_cbc_sieve.cu.o

verify: sim shim kernel_patch hetero kinesis_unified
	./sim_harness/ptx_bf162_emu
	$(PYTHON) sim_harness/smsp_roofline_sim.py
	./compiler_unlock/test_cublas_shim
	$(PYTHON) compiler_unlock/unlocked_gemm.py
	$(PYTHON) fused_w1_hadacore/spinquant_packer.py
	$(PYTHON) structural_filters/tiled_cbc_sieve.py
	$(PYTHON) structural_filters/lserve_sparse_attn.py
	$(PYTHON) qat_galore_engine/fused_qat_ste.py
	$(PYTHON) qat_galore_engine/galore_async_offload.py
	./kernel_driver_patch/test_kernel_patch
	./hetero_runtime/test_chorus_amx
	./hetero_runtime/test_governor_c
	$(PYTHON) hetero_runtime/closed_loop_governor.py
	./kinesis_unified/test_kinesis_unified

clean:
	rm -f sim_harness/ptx_bf162_emu* compiler_unlock/test_cublas_shim* compiler_unlock/*.so
	rm -f kernel_driver_patch/test_kernel_patch* hetero_runtime/test_chorus_amx* hetero_runtime/test_governor_c*
	rm -f kinesis_unified/test_kinesis_unified* fused_w1_hadacore/*.o structural_filters/*.o
