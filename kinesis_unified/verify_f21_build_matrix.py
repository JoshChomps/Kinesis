#!/usr/bin/env python3
"""
File: kinesis_unified/verify_f21_build_matrix.py
Suite F4: F21 / GCC PR 54412 stack-alignment build matrix (docs/12 4.11).

On x86_64-w64-mingw32 the Windows x64 ABI only guarantees a 16-byte aligned stack, but GCC still emits aligned
32-byte `vmovaps` / `vmovdqa` for YMM spills and temporaries (GCC PR 54412). The earlier F21 fix (pass descriptors
by const&, unaligned intrinsics on stack buffers) only held at -O1 and above with inlining: -O0 segfaulted on a
spill of a local __m256, and -O3 -fno-inline emitted an aligned store for a by-value __m256 argument.

The complete fix has two parts: the AVX2 unpackers take `const float*` (no by-value __m256 anywhere), and every
mingw build passes `-Wa,-muse-unaligned-vector-move`, which makes the assembler encode every aligned vector move
as its unaligned form. The unaligned forms have identical results and, on aligned addresses, identical speed on
every AVX2 core since Haswell, so they are always safe.

This suite builds test_kinesis_unified.cpp at -O0, -O3 -fno-inline and -O3 with the shipped flags, runs each
binary, and scans its disassembly for any aligned YMM move to or from an %rsp/%rbp-relative stack slot. As a
negative control it also builds -O0 WITHOUT the assembler flag and checks the scan does find such moves
(the binary is not run, since it crashes).
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "kinesis_unified" / "test_kinesis_unified.cpp"
UNALIGNED_FLAG = "-Wa,-muse-unaligned-vector-move"
ALIGNED_STACK_YMM = re.compile(
    r"vmov(?:aps|apd|dqa|dqa32|dqa64)\s+(?:%ymm\d+,\s*-?(?:0x[0-9a-f]+)?\(%r[sb]p\)|-?(?:0x[0-9a-f]+)?\(%r[sb]p\),\s*%ymm\d+)"
)


def build(opt_flags, extra, out):
    cmd = ["g++", "-std=c++20", *opt_flags, "-mavx2", "-mfma", *extra, f"-I{ROOT}", str(SRC), "-o", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"build failed: {' '.join(cmd)}\n{res.stderr}")


def aligned_stack_ymm_moves(binary):
    dis = subprocess.run(["objdump", "-d", str(binary)], capture_output=True, text=True, check=True).stdout
    return len(ALIGNED_STACK_YMM.findall(dis))


def main() -> None:
    print("================================================================================")
    print("  SUITE F4: F21 / GCC PR 54412 YMM STACK-ALIGNMENT BUILD MATRIX")
    print("================================================================================\n")
    if os.name != "nt":
        print("[skip] Not a mingw host: the 16-byte stack limitation is Windows x64 specific. PASS (not applicable)")
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ctrl = td / "ctrl_O0_noflag.exe"
        build(["-O0"], [], ctrl)
        n_ctrl = aligned_stack_ymm_moves(ctrl)
        assert n_ctrl > 0, "negative control: an -O0 build without the flag should contain aligned YMM stack moves"
        print(f"[0] Control -O0 without {UNALIGNED_FLAG}: {n_ctrl} aligned YMM stack moves (would fault) | as expected")
        for i, opt in enumerate((["-O0"], ["-O3", "-fno-inline"], ["-O3"]), start=1):
            exe = td / f"t{i}.exe"
            build(opt, [UNALIGNED_FLAG], exe)
            n = aligned_stack_ymm_moves(exe)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
            ok = run.returncode == 0 and "PASS" in run.stdout and n == 0
            print(f"[{i}] {' '.join(opt):<16} + {UNALIGNED_FLAG}: aligned YMM stack moves = {n}, "
                  f"run exit {run.returncode} | {'PASS' if ok else 'FAIL'}")
            assert ok, run.stdout[-2000:] + run.stderr[-2000:]
    print("\n[+] PASS: no aligned YMM stack access at any optimisation level; all builds run.")


if __name__ == "__main__":
    main()
