# Master Makefile: Unified Unlocked GPU Silicon, Zero-Copy Bus Fabrics, and Fused Sub-2-Bit Kernels

CC       ?= gcc
CXX      ?= g++
NVCC     ?= nvcc
PYTHON   ?= python
CFLAGS   := -O3 -std=c11 -Wall -Wextra
CXXFLAGS := -O3 -std=c++20 -mavx2 -mfma -Wall -Wextra
ifeq ($(OS),Windows_NT)
# F21 / GCC PR 54412 (docs/12 4.11): no aligned YMM stack moves on the 16-byte Windows x64 stack
CXXFLAGS += -Wa,-muse-unaligned-vector-move
endif

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
	$(CC) $(CFLAGS) -pthread -DSTANDALONE_VERIFY -o hetero_runtime/test_chorus_amx hetero_runtime/chorus_amx_moe.c -lm
	$(CC) $(CFLAGS) -DSTANDALONE_VERIFY -DUHA_CHORUS_FORCE_SINGLE_THREAD -DUHA_CHORUS_FORCE_GENERIC_ALLOC -o hetero_runtime/test_chorus_fallback hetero_runtime/chorus_amx_moe.c -lm
	$(CC) $(CFLAGS) -o hetero_runtime/test_governor_c hetero_runtime/closed_loop_governor.c

kinesis_unified:
	$(CXX) $(CXXFLAGS) -o kinesis_unified/test_kinesis_unified kinesis_unified/test_kinesis_unified.cpp
	$(CXX) $(CXXFLAGS) -o kinesis_unified/bench_cpu_roofline kinesis_unified/bench_cpu_roofline.cpp

cuda_host_verify:
	$(CXX) -O2 -std=c++20 -Wall -Wextra -DSTANDALONE_VERIFY -x c++ fused_w1_hadacore/w1a16_hadacore_kernel.cu -o fused_w1_hadacore/test_w1a16_kernel
	$(CXX) $(CXXFLAGS) -DSTANDALONE_VERIFY -x c++ structural_filters/tiled_cbc_sieve.cu -o structural_filters/test_tiled_cbc

cuda_kernels:
	$(NVCC) -O3 -arch=sm_86 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -c fused_w1_hadacore/w1a16_hadacore_kernel.cu -o fused_w1_hadacore/w1a16_hadacore_kernel.o
	$(NVCC) -O3 -arch=sm_86 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -c structural_filters/tiled_cbc_sieve.cu -o structural_filters/tiled_cbc_sieve.cu.o

verify: sim shim kernel_patch hetero kinesis_unified cuda_host_verify
	./sim_harness/ptx_bf162_emu
	./fused_w1_hadacore/test_w1a16_kernel
	./structural_filters/test_tiled_cbc
	$(PYTHON) sim_harness/smsp_roofline_sim.py
	./compiler_unlock/test_cublas_shim
	$(PYTHON) compiler_unlock/unlocked_gemm.py
	$(PYTHON) compiler_unlock/verify_shim_dladdr_linux.py
	$(PYTHON) fused_w1_hadacore/spinquant_packer.py
	$(PYTHON) structural_filters/tiled_cbc_sieve.py
	$(PYTHON) structural_filters/lserve_sparse_attn.py
	$(PYTHON) qat_galore_engine/fused_qat_ste.py
	$(PYTHON) qat_galore_engine/galore_async_offload.py
	./kernel_driver_patch/test_kernel_patch
	./hetero_runtime/test_chorus_amx
	./hetero_runtime/test_chorus_fallback
	./hetero_runtime/test_governor_c
	$(PYTHON) hetero_runtime/closed_loop_governor.py
	./kinesis_unified/test_kinesis_unified
	$(PYTHON) kinesis_unified/kinesis_bridge.py
	./kinesis_unified/bench_cpu_roofline
	$(PYTHON) kinesis_unified/verify_f21_build_matrix.py
	$(PYTHON) e2e_transformer_pipeline.py

clean:
	rm -f sim_harness/ptx_bf162_emu* compiler_unlock/test_cublas_shim* compiler_unlock/*.so
	rm -f kernel_driver_patch/test_kernel_patch* hetero_runtime/test_chorus_amx* hetero_runtime/test_chorus_fallback* hetero_runtime/test_governor_c*
	rm -f kinesis_unified/test_kinesis_unified* kinesis_unified/bench_cpu_roofline* kinesis_unified/libkinesis_unified* fused_w1_hadacore/*.o structural_filters/*.o
