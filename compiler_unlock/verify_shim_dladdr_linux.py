#!/usr/bin/env python3
"""
File: compiler_unlock/verify_shim_dladdr_linux.py
Builds and runs compiler_unlock/test_shim_dladdr_linux.c on Linux (natively, or through WSL on Windows) to
exercise the shim's dladdr() caller classification (docs/12 4.16). Host mock only: no CUDA driver is loaded.

Exit 0 on PASS. If no Linux toolchain is reachable the script prints SKIP and exits 0, so the suite stays green
on hosts without WSL; the SKIP line is visible in the suite output.
"""

import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def to_wsl_path(p: str) -> str:
    p = os.path.abspath(p).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def build_script(src_dir: str) -> str:
    shim = f"{src_dir}/libcublas_unlock_shim.c"
    test = f"{src_dir}/test_shim_dladdr_linux.c"
    q = shlex.quote
    return " && ".join([
        "set -e",
        "T=$(mktemp -d)",
        f"gcc -O2 -std=c11 -Wall -Wextra -shared -fPIC -o $T/libcublas_unlock_shim.so {q(shim)} -ldl",
        f"gcc -O2 -std=c11 -Wall -Wextra -shared -fPIC -fno-optimize-sibling-calls -DROLE_CALLER_LIB "
        f"-o $T/libcublasLt_mockcaller.so {q(test)} $T/libcublas_unlock_shim.so",
        f"gcc -O2 -std=c11 -Wall -Wextra -shared -fPIC -fno-optimize-sibling-calls -DROLE_CALLER_LIB "
        f"-o $T/libother_mockcaller.so {q(test)} $T/libcublas_unlock_shim.so",
        f"gcc -O2 -std=c11 -Wall -Wextra -o $T/test_dladdr {q(test)} $T/libcublas_unlock_shim.so -ldl",
        "LD_LIBRARY_PATH=$T $T/test_dladdr $T/libcublasLt_mockcaller.so $T/libother_mockcaller.so",
        "rm -rf $T",
    ])


def main() -> int:
    if sys.platform.startswith("linux"):
        cmd = ["sh", "-c", build_script(HERE)]
    elif sys.platform == "win32" and shutil.which("wsl.exe"):
        probe = subprocess.run(["wsl.exe", "-e", "sh", "-c", "command -v gcc"], capture_output=True, text=True)
        if probe.returncode != 0:
            print("[SKIP] WSL present but no gcc inside it; dladdr test not run.")
            return 0
        cmd = ["wsl.exe", "-e", "sh", "-c", build_script(to_wsl_path(HERE))]
    else:
        print("[SKIP] No Linux toolchain (native or WSL); dladdr test not run.")
        return 0
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
