#!/usr/bin/env python3
"""
File: export_full_stack.py
Complete Zero-Truncation Co-Design Stack Exporter.
Assembles all modular architectural documentation (docs/00..14) and all source files
across all workstreams into:
  1. Full_Unlocked_GPU_CoDesign_Stack.md
  2. Full_Unlocked_GPU_CoDesign_Stack.html (Print-ready HTML for 1-click PDF export)
"""

import os
import html


DOC_FILES = [
    "docs/00_Architecture_Overview_and_Baselines.md",
    "docs/01_Pillar1_Silicon_Unlock_and_SRAM_Roofline.md",
    "docs/02_Pillar2_Sub2Bit_PTX_Unpackers_and_HadaCore.md",
    "docs/03_Pillar3_Zero_Materialization_QAT_and_GaLore.md",
    "docs/04_Pillar4_PCIe_BAR1_P2P_and_Modded_48GB_Driver.md",
    "docs/05_Pillar5_Structural_Filters_and_Chorus_AMX_Runtime.md",
    "docs/06_Reference_Index_and_Bibliography.md",
    "docs/07_Security_Risks_Prior_Art_and_Literature.md",
    "docs/08_Kinesis_Evaluation_Fixes_and_Smart_Cache.md",
    "docs/09_First_Principles_Firmware_Hardware_Code_Review.md",
    "docs/10_Related_Projects_Community_Buzz_and_Wave2_Upgrades.md",
    "docs/11_Adversarial_Review_Status_and_Known_Limitations.md",
    "docs/12_Independent_Verification_Caveats_and_Inconsistencies.md",
    "docs/15_Language_and_Runtime_Architecture_C_CPP_CUDA_vs_PyTorch.md",
    "docs/16_Subsystem_Audit_Interleaving_Seams_and_Testing_Architecture.md",
]

SOURCE_FILES = [
    ("kinesis_unified/unified_sizing.h", "c"),
    ("kinesis_unified/memory_primitives.hpp", "cpp"),
    ("kinesis_unified/hardware_and_sandbox.hpp", "cpp"),
    ("kinesis_unified/sub2bit_gemm_and_smart_cache.hpp", "cpp"),
    ("kinesis_unified/test_kinesis_unified.cpp", "cpp"),
    ("kinesis_unified/kinesis_c_abi.cpp", "cpp"),
    ("kinesis_unified/kinesis_bridge.py", "python"),
    ("kinesis_unified/bench_cpu_roofline.cpp", "cpp"),
    ("kinesis_unified/verify_f21_build_matrix.py", "python"),
    ("sim_harness/ptx_bf162_emu.c", "c"),
    ("sim_harness/smsp_roofline_sim.py", "python"),
    ("compiler_unlock/libcublas_unlock_shim.c", "c"),
    ("compiler_unlock/test_cublas_shim.c", "c"),
    ("compiler_unlock/test_shim_dladdr_linux.c", "c"),
    ("compiler_unlock/verify_shim_dladdr_linux.py", "python"),
    ("compiler_unlock/unlocked_gemm.py", "python"),
    ("fused_w1_hadacore/spinquant_packer.py", "python"),
    ("fused_w1_hadacore/w1a16_hadacore_kernel.cu", "cuda"),
    ("fused_w1_hadacore/w1a16_host_cta_shim.hpp", "cpp"),
    ("fused_w1_hadacore/w1a16_host_verify_main.hpp", "cpp"),
    ("structural_filters/tiled_cbc_sieve.cu", "cuda"),
    ("structural_filters/tiled_cbc_sieve.py", "python"),
    ("structural_filters/lserve_sparse_attn.py", "python"),
    ("qat_galore_engine/fused_qat_ste.py", "python"),
    ("qat_galore_engine/galore_async_offload.py", "python"),
    ("kernel_driver_patch/nv_gpu_ops_and_s3_unlock.c", "c"),
    ("kernel_driver_patch/01-gsp-and-static-bar1-p2p.patch", "diff"),
    ("kernel_driver_patch/02-dynamic-bar1-48gb-and-noncoh-pte.patch", "diff"),
    ("kernel_driver_patch/03-falcon-ss0-unlock-and-s3-resume.patch", "diff"),
    ("kernel_driver_patch/nvidia-p2p.conf", "ini"),
    ("hetero_runtime/chorus_amx_moe.c", "c"),
    ("hetero_runtime/closed_loop_governor.c", "c"),
    ("hetero_runtime/closed_loop_governor.py", "python"),
    ("e2e_transformer_pipeline.py", "python"),
    ("verify_windows_gpu.py", "python"),
    ("run_all_tests.py", "python"),
    ("Makefile", "makefile"),
    ("verify_bare_metal.sh", "bash"),
    ("verify_windows_gpu.ps1", "powershell"),
]


def export_everything() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    parts = [
        "# Unified Hardware Acceleration (UHA): Extreme Hardware Capability & Systems Co-Design Research Framework\n"
    ]

    for doc_rel in DOC_FILES:
        doc_path = os.path.join(root, doc_rel)
        with open(doc_path, "r", encoding="utf-8") as f:
            parts.append(f.read().strip() + "\n\n---\n")

    parts.append("# Complete Source Code & Verification Implementation Tree\n")
    for src_rel, lang in SOURCE_FILES:
        src_path = os.path.join(root, src_rel)
        with open(src_path, "r", encoding="utf-8") as f:
            code = f.read().strip()
        parts.append(f"## File: `{src_rel}`\n```{lang}\n{code}\n```\n")

    full_md = "\n".join(parts)
    md_out = os.path.join(root, "Full_Unlocked_GPU_CoDesign_Stack.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(full_md)
    print(f"[+] Wrote complete unabridged Markdown specification: {md_out}")

    html_out = os.path.join(root, "Full_Unlocked_GPU_CoDesign_Stack.html")
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Unified Hardware Acceleration (UHA): Systems Co-Design Research Framework</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 920px; margin: 24px auto; padding: 0 20px; color: #1f2328; }}
  .bar {{ background: #0969da; color: #fff; padding: 12px 16px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
  .bar button {{ background: #fff; color: #0969da; border: 0; padding: 8px 16px; font-weight: 700; border-radius: 5px; cursor: pointer; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f6f8fa; border: 1px solid #d0d7de; padding: 14px; border-radius: 6px; font-size: 11.5px; line-height: 1.45; }}
  @media print {{ .bar {{ display: none; }} pre {{ font-size: 9.5px; border: none; }} }}
</style></head><body>
<div class="bar"><span><strong>Complete Co-Design Stack Bundle</strong> (All 5 Pillars &amp; 23 Verification Suites Included)</span>
<button onclick="window.print()">Save Complete PDF</button></div>
<pre>{html.escape(full_md)}</pre></body></html>""")
    print(f"[+] Wrote print-ready HTML bundle: {html_out}")


if __name__ == "__main__":
    export_everything()
