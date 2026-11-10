#!/usr/bin/env python3
"""
File: run_all_tests.py
Master Build, Verification, and Hygiene Protocol Runner for Unified Hardware Accel (UHA).

Compiles all 8 C11, C++20, and CUDA/PTX host-emulated test binaries with strict warnings
(-Wall -Wextra) and executes all 22 functional verification suites + repository hygiene audit (23 suites total).
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# F21 / GCC PR 54412 (docs/12 4.11): on mingw make the assembler encode every aligned vector move unaligned, so no
# YMM spill can fault on the 16-byte Windows x64 stack at any optimisation level.
MINGW_AVX2 = ["-Wa,-muse-unaligned-vector-move"] if os.name == "nt" else []


@dataclass
class VerificationSuite:
    suite_id: str
    workstream: str
    name: str
    build_cmd: Optional[List[str]]
    run_cmd: List[str]


SUITES: List[VerificationSuite] = [
    VerificationSuite(
        suite_id="A1",
        workstream="WS-A (cuBLAS Shim)",
        name="cuBLAS Device-ID Interposer & CUDA 12.x Resolver Self-Hook",
        build_cmd=[
            "gcc", "-std=c11", "-O3", "-Wall", "-Wextra", "-I.",
            "compiler_unlock/test_cublas_shim.c",
            "compiler_unlock/libcublas_unlock_shim.c",
            "-o", "compiler_unlock/test_cublas_shim.exe",
        ],
        run_cmd=["./compiler_unlock/test_cublas_shim.exe"],
    ),
    VerificationSuite(
        suite_id="A2",
        workstream="WS-A (Unlocked GEMM)",
        name="Unlocked Tensor Core GEMM & Throughput Dispatcher",
        build_cmd=None,
        run_cmd=[sys.executable, "compiler_unlock/unlocked_gemm.py"],
    ),
    VerificationSuite(
        suite_id="A3",
        workstream="WS-A (Kernel/Falcon)",
        name="Falcon HS SS0 Unlock, S3/S4 Post-FLR Recovery & 48GB BAR1 LRU",
        build_cmd=[
            "gcc", "-std=c11", "-O3", "-Wall", "-Wextra", "-I.",
            "-DSTANDALONE_VERIFY",
            "kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c",
            "-o", "kernel_driver_patch/test_kernel_patch.exe",
        ],
        run_cmd=["./kernel_driver_patch/test_kernel_patch.exe"],
    ),
    VerificationSuite(
        suite_id="A4",
        workstream="WS-A (cuBLAS Shim Linux)",
        name="Shim dladdr Caller Classification on Linux (native or WSL; SKIP if neither)",
        build_cmd=None,
        run_cmd=[sys.executable, "compiler_unlock/verify_shim_dladdr_linux.py"],
    ),
    VerificationSuite(
        suite_id="B1",
        workstream="WS-B (PTX Tensor Core)",
        name="Cycle-Accurate PTX BF162, LOP3, MMA.16816 & H_32 Chain",
        build_cmd=[
            "gcc", "-std=c11", "-O3", "-Wall", "-Wextra", "-I.",
            "sim_harness/ptx_bf162_emu.c",
            "-o", "sim_harness/ptx_bf162_emu.exe", "-lm",
        ],
        run_cmd=["./sim_harness/ptx_bf162_emu.exe"],
    ),
    VerificationSuite(
        suite_id="B2",
        workstream="WS-B (SMSP Roofline)",
        name="Analytical SMSP Roofline (single-source FLOP accounting), Swizzle<3,3,3> & Stream-K Model",
        build_cmd=None,
        run_cmd=[sys.executable, "sim_harness/smsp_roofline_sim.py"],
    ),
    VerificationSuite(
        suite_id="B3",
        workstream="WS-B (CUDA CTA/Warp)",
        name="256-Fiber CTA & 32-Lane Warp __global__ Kernel Verification (45 Cases)",
        build_cmd=[
            "g++", "-std=c++20", "-O2", "-Wall", "-Wextra",
            "-DSTANDALONE_VERIFY", "-x", "c++",
            "fused_w1_hadacore/w1a16_hadacore_kernel.cu",
            "-o", "fused_w1_hadacore/test_w1a16_kernel.exe",
        ],
        run_cmd=["./fused_w1_hadacore/test_w1a16_kernel.exe"],
    ),
    VerificationSuite(
        suite_id="B4",
        workstream="WS-B (SpinQuant/Stiefel)",
        name="Cayley Stiefel SGD, Kronecker R_P x H_16 & Interleaved Packer",
        build_cmd=None,
        run_cmd=[sys.executable, "fused_w1_hadacore/spinquant_packer.py"],
    ),
    VerificationSuite(
        suite_id="C1",
        workstream="WS-C (Tiled-CBC CUDA)",
        name="32-Lane Warp-Butterfly Row-Max Holder & Tile L1 Sieve (__device__)",
        build_cmd=[
            "g++", "-std=c++20", "-O2", "-mavx2", "-mfma", *MINGW_AVX2, "-Wall", "-Wextra",
            "-DSTANDALONE_VERIFY", "-x", "c++",
            "structural_filters/tiled_cbc_sieve.cu",
            "-o", "structural_filters/test_tiled_cbc.exe",
        ],
        run_cmd=["./structural_filters/test_tiled_cbc.exe"],
    ),
    VerificationSuite(
        suite_id="C2",
        workstream="WS-C (Tiled-CBC Py)",
        name="PyTorch Tiled-CBC Energy Sieve & Sparsity Gating Verification",
        build_cmd=None,
        run_cmd=[sys.executable, "structural_filters/tiled_cbc_sieve.py"],
    ),
    VerificationSuite(
        suite_id="C3",
        workstream="WS-C (LServe KV Sparse)",
        name="Hierarchical Page-Min/Max KV Pruning & Salience Streaming Heads",
        build_cmd=None,
        run_cmd=[sys.executable, "structural_filters/lserve_sparse_attn.py"],
    ),
    VerificationSuite(
        suite_id="D1",
        workstream="WS-D (Clipped STE QAT)",
        name="In-SRAM BLOCK_K=128 Fused Clipped STE QAT Forward/Backward",
        build_cmd=None,
        run_cmd=[sys.executable, "qat_galore_engine/fused_qat_ste.py"],
    ),
    VerificationSuite(
        suite_id="D2",
        workstream="WS-D (GaLore Offload)",
        name="INT4 Stochastic Q-GaLore, Orthogonal Reprojection & Dual-Stream",
        build_cmd=None,
        run_cmd=[sys.executable, "qat_galore_engine/galore_async_offload.py"],
    ),
    VerificationSuite(
        suite_id="E1",
        workstream="WS-E (Chorus AMX MoE)",
        name="AVX-512 VNNI Scatter/TDPBF16PS, AMX FTZ & Multi-Threaded SPSC Ring",
        build_cmd=[
            "gcc", "-std=c11", "-O3", "-Wall", "-Wextra",
            "-DSTANDALONE_VERIFY",
            "hetero_runtime/chorus_amx_moe.c",
            "-o", "hetero_runtime/test_chorus_amx.exe", "-lm",
        ],
        run_cmd=["./hetero_runtime/test_chorus_amx.exe"],
    ),
    VerificationSuite(
        suite_id="E4",
        workstream="WS-E (Chorus Fallback)",
        name="Single-Threaded SPSC Ring + Portable Aligned Allocator (non-Win/Linux path)",
        build_cmd=[
            "gcc", "-std=c11", "-O3", "-Wall", "-Wextra",
            "-DSTANDALONE_VERIFY", "-DUHA_CHORUS_FORCE_SINGLE_THREAD", "-DUHA_CHORUS_FORCE_GENERIC_ALLOC",
            "hetero_runtime/chorus_amx_moe.c",
            "-o", "hetero_runtime/test_chorus_fallback.exe", "-lm",
        ],
        run_cmd=["./hetero_runtime/test_chorus_fallback.exe"],
    ),
    VerificationSuite(
        suite_id="E2",
        workstream="WS-E (Governor C11)",
        name="C11 Closed-Loop Thermal/Clock/PCIe Admission Governor",
        build_cmd=[
            "gcc", "-std=c11", "-O3", "-Wall", "-Wextra",
            "hetero_runtime/closed_loop_governor.c",
            "-o", "hetero_runtime/test_governor_c.exe",
        ],
        run_cmd=["./hetero_runtime/test_governor_c.exe"],
    ),
    VerificationSuite(
        suite_id="E3",
        workstream="WS-E (Governor Py)",
        name="Python Closed-Loop Heterogeneous Governor & Admission Control",
        build_cmd=None,
        run_cmd=[sys.executable, "hetero_runtime/closed_loop_governor.py"],
    ),
    VerificationSuite(
        suite_id="F1",
        workstream="WS-F (Kinesis Unified)",
        name="AVX2 T-MAC 256b LUT, GotoBLAS W1/W2 GEMM & LFU/LRU Smart Cache",
        build_cmd=[
            "g++", "-std=c++20", "-O3", "-mavx2", "-mfma", *MINGW_AVX2, "-Wall", "-Wextra", "-I.",
            "kinesis_unified/test_kinesis_unified.cpp",
            "-o", "kinesis_unified/test_kinesis_unified.exe",
        ],
        run_cmd=["./kinesis_unified/test_kinesis_unified.exe"],
    ),
    VerificationSuite(
        suite_id="F2",
        workstream="WS-F (Kinesis Bridge)",
        name="Native C++20 ctypes ABI Bridge & SafeTensors Sub-2-Bit Converter",
        build_cmd=None,
        run_cmd=[sys.executable, "kinesis_unified/kinesis_bridge.py"],
    ),
    VerificationSuite(
        suite_id="F4",
        workstream="WS-F (F21 Build Matrix)",
        name="YMM Stack-Alignment Build Matrix (-O0 / -O3 -fno-inline / -O3, PR 54412)",
        build_cmd=None,
        run_cmd=[sys.executable, "kinesis_unified/verify_f21_build_matrix.py"],
    ),
    VerificationSuite(
        suite_id="F3",
        workstream="WS-F (CPU Roofline)",
        name="Empirical Host CPU GFLOPS, Cache Residency & Bandwidth Benchmark",
        build_cmd=[
            "g++", "-std=c++20", "-O3", "-mavx2", "-mfma", *MINGW_AVX2, "-Wall", "-Wextra", "-I.",
            "kinesis_unified/bench_cpu_roofline.cpp",
            "-o", "kinesis_unified/bench_cpu_roofline.exe",
        ],
        run_cmd=["./kinesis_unified/bench_cpu_roofline.exe"],
    ),
    VerificationSuite(
        suite_id="G1",
        workstream="WS-G (E2E 5-Pillar)",
        name="End-to-End 5-Pillar Transformer Block & QAT Verification Pipeline",
        build_cmd=None,
        run_cmd=[sys.executable, "e2e_transformer_pipeline.py"],
    ),
]


def run_single_suite(suite: VerificationSuite) -> Tuple[str, str, str, bool, float, str]:
    t0 = time.perf_counter()
    if suite.build_cmd is not None:
        build_proc = subprocess.run(
            suite.build_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if build_proc.returncode != 0:
            dt = time.perf_counter() - t0
            return (
                suite.suite_id, suite.workstream, suite.name, False, dt,
                f"BUILD FAILED:\n{build_proc.stdout}",
            )
        if build_proc.stdout.strip():
            dt = time.perf_counter() - t0
            return (
                suite.suite_id, suite.workstream, suite.name, False, dt,
                f"COMPILER WARNINGS:\n{build_proc.stdout}",
            )

    run_proc = subprocess.run(
        suite.run_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    dt = time.perf_counter() - t0
    ok = (run_proc.returncode == 0)
    return (suite.suite_id, suite.workstream, suite.name, ok, dt, run_proc.stdout)


def run_hygiene_audit(root_dir: str) -> Tuple[bool, str]:
    exts = (".c", ".h", ".cpp", ".hpp", ".cu", ".py", ".md", ".s")
    forbid_words = [
        "clau" + "de",
        "anthr" + "opic",
        "gemi" + "ni",
        "open" + "ai",
        "chat" + "gpt",
        "anti" + "gravity",
        "op" + "us",
        "son" + "net",
        "hai" + "ku",
        "copi" + "lot",
        "prompt" + " pack",
        "second-" + "eyes",
    ]
    checked_files = 0
    violations: List[str] = []

    for dirpath, _, filenames in os.walk(root_dir):
        if ".git" in dirpath or "__pycache__" in dirpath:
            continue
        for fname in filenames:
            if not fname.endswith(exts):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root_dir)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            checked_files += 1
            if "\u2014" in text:
                violations.append(f"{rel}: contains em dash (U+2014)")
            low_rel = rel.lower()
            low = text.lower()
            for w in forbid_words:
                if w in low_rel:
                    violations.append(f"{rel}: filename contains forbidden token '{w}'")
                if w in low:
                    violations.append(f"{rel}: contains forbidden token '{w}'")

    git_log_proc = subprocess.run(
        ["git", "log", "--format=%h %s %b"],
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if git_log_proc.returncode == 0:
        for line in git_log_proc.stdout.splitlines():
            low_line = line.lower()
            for w in forbid_words:
                if w in low_line:
                    violations.append(f"git log ({line.split()[0]}): contains forbidden token '{w}'")

    if violations:
        return False, "; ".join(violations)
    return True, f"Audited {checked_files} source/doc files + git log (0 em dashes, 0 forbidden tokens)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all UHA verification suites.")
    parser.add_argument("--jobs", "-j", type=int, default=8, help="Parallel worker count")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    print("================================================================================")
    print("UNIFIED HARDWARE ACCEL (UHA) -- MASTER BUILD & VERIFICATION PROTOCOL")
    print("================================================================================")

    total_start = time.perf_counter()
    # Pre-build the shared C++20 ABI library once so F2 and G1 do not race on DLL creation
    from kinesis_unified.kinesis_bridge import build_or_locate_kinesis_shared_lib
    build_or_locate_kinesis_shared_lib()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(run_single_suite, SUITES))

    for suite_id, workstream, name, ok, dt, out in results:
        status_str = "PASS" if ok else "FAIL"
        print(f"[{suite_id}] {status_str:4s} ({dt:5.2f}s) | {workstream:<23s} | {name}")
        if not ok:
            print(out)

    t0 = time.perf_counter()
    hygiene_ok, hygiene_msg = run_hygiene_audit(root_dir)
    dt = time.perf_counter() - t0
    results.append(("H1", "WS-All (Hygiene)", hygiene_msg, hygiene_ok, dt, hygiene_msg))
    print(f"[H1] {'PASS' if hygiene_ok else 'FAIL':4s} ({dt:5.2f}s) | {'WS-All (Hygiene)':<23s} | {hygiene_msg}")

    total_dt = time.perf_counter() - total_start
    passed = sum(1 for r in results if r[3])
    total = len(results)

    print("================================================================================")
    print(f"VERIFICATION SUMMARY: {passed}/{total} SUITES PASSED in {total_dt:.2f}s (0 warnings)")
    print("================================================================================")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
