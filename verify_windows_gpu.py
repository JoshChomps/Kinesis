#!/usr/bin/env python3
"""
File: verify_windows_gpu.py
Live Windows / Linux CUDA Verification & 8 GB RTX 4060 Adversarial Stress Suite (Suite G2).

Exercises the hardware-agnostic software modules of Kinesis on a physical CUDA GPU:
  1. BF16 Tensor Core GEMM parity, speedup vs FP32-cast fallback, and odd/strided/edge-value checks
  2. Kronecker-Stiefel rotation, W1/W2 packing, and DeployBasisQAT/FusedQAT CUDA bit-exact parity
  3. LServe INT4 Hadamard KV attention (Sq=1 decode, CPU-mask, 0-active-pages, extreme keys) & Tiled-CBC sieve
  4. Native C++20 AVX2 DLL <-> CUDA parity, short-buffer -3 guard, 8-thread determinism, CUDA HF converter
  5. Live nvidia-smi telemetry polling + AD107 45W clock-sag & NaN clock-floor HOLD regressions
  6. Non-square/odd-batch/odd-rank/zero-grad QAT + GaLore recovery & persistent pinned host buffer checks
  7. 100.7M-param (24 x 2048x2048) 8 GB VRAM QAT + 4-bit GaLore vs BF16 AdamW scale benchmark & 4096^3 stability

Skips cleanly (exit 0) if no CUDA GPU is available.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import gc
import math
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parent
for subdir in (
    "compiler_unlock",
    "fused_w1_hadacore",
    "qat_galore_engine",
    "structural_filters",
    "hetero_runtime",
    "kinesis_unified",
):
    if str(ROOT / subdir) not in sys.path:
        sys.path.insert(0, str(ROOT / subdir))

from unlocked_gemm import unlocked_bf16_gemm  # noqa: E402
from spinquant_packer import (  # noqa: E402
    build_kernel_online_rotation,
    fused_w1_g128_gemm,
    fused_w2_ternary_gemm,
    optimize_kronecker_stiefel_rotation,
    quantize_and_pack_w1_g128_interleaved,
    unpack_w1_g128_lop3_reference,
    quantize_and_pack_w2_ternary_interleaved,
    unpack_w2_ternary_lop3_reference,
)
from lserve_sparse_attn import (  # noqa: E402
    quantize_kv_pages_int4_hadamard,
    profile_streaming_head_mask,
    lserve_sparse_hadamard_attention,
)
from tiled_cbc_sieve import tiled_cbc_sieve_gemm  # noqa: E402
from kinesis_bridge import (  # noqa: E402
    KinesisNativeBridge,
    convert_hf_transformer_layer_to_sub2bit,
    load_safetensors_bytes,
    save_safetensors_bytes,
)
from closed_loop_governor import (  # noqa: E402
    ClosedLoopHardwareGovernor,
    HardwareTelemetrySample,
    poll_live_gpu_telemetry,
)
from fused_qat_ste import DeployBasisQATLinear, FusedQATG128Linear  # noqa: E402
from galore_async_offload import LayerWise8BitGaLoreEngine  # noqa: E402


def main() -> int:
    print("================================================================================")
    print("  SUITE G2: LIVE WINDOWS / LINUX CUDA GPU VERIFICATION & 8GB STRESS BENCHMARK")
    print("================================================================================\n")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA is not available on this host; live GPU checks skipped.")
        return 0

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    print(
        f"Device : {props.name} | SM {props.major}.{props.minor} | "
        f"VRAM {props.total_memory / (1024 ** 3):.2f} GiB | SMs {props.multi_processor_count} | "
        f"PyTorch {torch.__version__} (CUDA {torch.version.cuda})\n"
    )

    # -------------------------------------------------------------------------
    # Stage 1: unlocked_bf16_gemm Tensor Core parity, speedup & adversarial shapes
    # -------------------------------------------------------------------------
    a = torch.randn((2048, 2048), dtype=torch.bfloat16, device=dev)
    b = torch.randn((2048, 2048), dtype=torch.bfloat16, device=dev)
    c_tc = unlocked_bf16_gemm(a, b)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.bfloat16)
    rel_err_gemm = float((c_tc.float() - c_ref.float()).norm().item() / (c_ref.float().norm().item() + 1e-12))
    assert c_tc.is_cuda and rel_err_gemm < 1.5e-2, f"CUDA BF16 GEMM mismatch: {rel_err_gemm}"

    # Odd, non-aligned, strided/non-contiguous, zero, subnormal, and large inputs
    for m_dim, n_dim, k_dim in ((1, 17, 13), (129, 257, 513)):
        a_odd = torch.randn((m_dim, k_dim), dtype=torch.bfloat16, device=dev)
        b_odd = torch.randn((k_dim, n_dim), dtype=torch.bfloat16, device=dev)
        y_odd = unlocked_bf16_gemm(a_odd, b_odd)
        y_odd_ref = a_odd.float() @ b_odd.float()
        odd_rel = float((y_odd.float() - y_odd_ref).norm().item() / (y_odd_ref.norm().item() + 1e-8))
        assert y_odd.is_cuda and torch.isfinite(y_odd).all() and odd_rel < 2.0e-2, odd_rel

    a_nc = torch.randn((128, 256), dtype=torch.bfloat16, device=dev)[::2, ::2].t()
    b_nc = torch.randn((256, 128), dtype=torch.bfloat16, device=dev)[::4, ::2]
    assert not a_nc.is_contiguous() and not b_nc.is_contiguous()
    y_nc = unlocked_bf16_gemm(a_nc, b_nc)
    y_nc_ref = a_nc.float() @ b_nc.float()
    assert float((y_nc.float() - y_nc_ref).norm().item() / (y_nc_ref.norm().item() + 1e-8)) < 2.0e-2

    z_mat = torch.zeros((33, 65), dtype=torch.bfloat16, device=dev)
    assert bool((unlocked_bf16_gemm(z_mat, z_mat.t()) == 0).all().item())
    sub_mat = torch.full((32, 32), 1e-38, dtype=torch.bfloat16, device=dev)
    big_mat = torch.full((32, 32), 1e3, dtype=torch.bfloat16, device=dev)
    assert bool(torch.isfinite(unlocked_bf16_gemm(sub_mat, sub_mat)).all().item())
    assert bool(torch.isfinite(unlocked_bf16_gemm(big_mat, big_mat)).all().item())

    for _ in range(5):
        _ = unlocked_bf16_gemm(a, b)
        _ = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.bfloat16)
    torch.cuda.synchronize()

    iters = 40
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = unlocked_bf16_gemm(a, b)
    torch.cuda.synchronize()
    dt_tc = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.bfloat16)
    torch.cuda.synchronize()
    dt_fp32_cast = time.perf_counter() - t0

    flops = iters * 2.0 * (2048 ** 3)
    tflops_tc = flops / dt_tc / 1e12
    tflops_fp32 = flops / dt_fp32_cast / 1e12
    speedup = dt_fp32_cast / max(dt_tc, 1e-9)
    print(
        f"[1] unlocked_bf16_gemm (2048^3)   : {tflops_tc:5.2f} TFLOPS (Tensor Core) vs "
        f"{tflops_fp32:5.2f} TFLOPS (FP32 cast) | {speedup:.2f}x | RelErr={rel_err_gemm:.3e} | odd/strided ok"
    )

    # -------------------------------------------------------------------------
    # Stage 2: SpinQuant W1/W2 + DeployBasisQAT & FusedQAT CUDA bit-exact parity
    # -------------------------------------------------------------------------
    x_act = torch.randn((64, 512), dtype=torch.float32, device=dev)
    x_act[:, 0] *= 15.0
    x_act[:, 129] *= 11.0
    r_kron, _, k_init, k_h16, k_kron = optimize_kronecker_stiefel_rotation(x_act, q_hadamard=16, steps=20, lr=0.25)
    assert r_kron.is_cuda and k_kron < k_h16 < k_init, (k_init, k_h16, k_kron)

    w_src = torch.randn((256, 512), dtype=torch.float32, device=dev) * 0.04
    p1, s1 = quantize_and_pack_w1_g128_interleaved(w_src, R1=r_kron)
    u1 = unpack_w1_g128_lop3_reference(p1, s1)
    p2, s2 = quantize_and_pack_w2_ternary_interleaved(w_src, R1=r_kron)
    u2 = unpack_w2_ternary_lop3_reference(p2, s2)
    assert p1.is_cuda and u1.is_cuda and p2.is_cuda and u2.is_cuda

    x_rot = (x_act @ r_kron).to(torch.bfloat16)
    y_gt = x_act @ w_src.T
    y_w1 = unlocked_bf16_gemm(x_rot, u1.T)
    y_w2 = unlocked_bf16_gemm(x_rot, u2.T)
    cos_w1 = float(torch.nn.functional.cosine_similarity(y_w1.float().flatten(), y_gt.flatten(), dim=0).item())
    cos_w2 = float(torch.nn.functional.cosine_similarity(y_w2.float().flatten(), y_gt.flatten(), dim=0).item())
    assert cos_w1 > 0.75 and cos_w2 > 0.85, (cos_w1, cos_w2)
    y_w1_fus = fused_w1_g128_gemm(x_rot, p1, s1)
    y_w2_fus = fused_w2_ternary_gemm(x_rot, p2, s2)
    fus_err1 = float((y_w1_fus.float() - y_w1.float()).norm().item() / (y_w1.float().norm().item() + 1e-8))
    fus_err2 = float((y_w2_fus.float() - y_w2.float()).norm().item() / (y_w2.float().norm().item() + 1e-8))
    assert fus_err1 < 1.5e-2 and fus_err2 < 1.5e-2, (fus_err1, fus_err2)

    for mode in (1, 2):
        r_online = build_kernel_online_rotation(256, mode=mode, device=dev)
        for fmt in ("w1", "w2"):
            dlayer = DeployBasisQATLinear(256, 128, fmt=fmt, rotation=r_online, master_dtype=torch.float32, device=dev)
            x_3d = torch.randn((2, 17, 256), device=dev, dtype=torch.bfloat16, requires_grad=True)
            y_3d = dlayer(x_3d)
            assert y_3d.shape == (2, 17, 128) and y_3d.is_cuda and bool(torch.isfinite(y_3d).all().item())
            y_3d.float().pow(2).mean().backward()
            assert x_3d.grad is not None and bool(torch.isfinite(x_3d.grad).all().item())
            assert dlayer.weight.grad is not None and bool(torch.isfinite(dlayer.weight.grad).all().item())
            w_dep = dlayer.deploy_weight()
            if fmt == "w1":
                pd, sd = quantize_and_pack_w1_g128_interleaved(dlayer.weight.detach(), R1=r_online)
                ud = unpack_w1_g128_lop3_reference(pd, sd)
            else:
                pd, sd = quantize_and_pack_w2_ternary_interleaved(dlayer.weight.detach(), R1=r_online)
                ud = unpack_w2_ternary_lop3_reference(pd, sd)
            assert int((w_dep != ud).sum().item()) == 0, f"DeployBasisQAT CUDA mismatch mode={mode} fmt={fmt}"

    for md in (torch.bfloat16, torch.float32):
        fl = FusedQATG128Linear(512, 256, clip_mult=2.5, device=dev, master_dtype=md)
        x_3d_nc = torch.randn((6, 38, 512), device=dev, dtype=torch.bfloat16)[::2, ::2].detach().requires_grad_(True)
        assert not x_3d_nc.is_contiguous()
        y_fl = fl(x_3d_nc)
        y_fl.float().pow(2).mean().backward()
        assert bool(torch.isfinite(y_fl).all().item()) and bool(torch.isfinite(x_3d_nc.grad).all().item())
        assert bool(torch.isfinite(fl.weight.grad).all().item())
        pf, sf = quantize_and_pack_w1_g128_interleaved(fl.packer_input_weight())
        uf = unpack_w1_g128_lop3_reference(pf, sf)
        with torch.no_grad():
            y_pack = unlocked_bf16_gemm(x_3d_nc.detach().reshape(-1, 512), uf.T).reshape_as(y_fl)
            fl_rel = float((y_fl.detach().float() - y_pack.float()).norm().item() / (y_pack.float().norm().item() + 1e-8))
        assert fl_rel < 1.0e-2, f"FusedQAT vs packed CUDA mismatch: {fl_rel}"

    # Verify fused W2 Ternary (b1.58) FusedQATG128Linear on CUDA (Triton + fallback)
    fl_w2 = FusedQATG128Linear(512, 256, clip_mult=2.5, device=dev, master_dtype=torch.bfloat16, fmt="w2")
    x_w2 = torch.randn((19, 512), device=dev, dtype=torch.bfloat16, requires_grad=True)
    y_w2_qat = fl_w2(x_w2)
    y_w2_qat.float().pow(2).mean().backward()
    assert bool(torch.isfinite(y_w2_qat).all().item()) and bool(torch.isfinite(x_w2.grad).all().item())
    p_w2q, s_w2q = quantize_and_pack_w2_ternary_interleaved(fl_w2.packer_input_weight())
    u_w2q = unpack_w2_ternary_lop3_reference(p_w2q, s_w2q)
    with torch.no_grad():
        y_w2_ref_q = unlocked_bf16_gemm(x_w2.detach(), u_w2q.T)
        w2_rel = float((y_w2_qat.detach().float() - y_w2_ref_q.float()).norm().item() / (y_w2_ref_q.float().norm().item() + 1e-8))
    assert w2_rel < 1.5e-2, f"FusedQAT W2 CUDA mismatch: {w2_rel}"

    print(
        f"[2] SpinQuant + QAT Parity (CUDA) : Kurtosis {k_init:.2f} -> {k_h16:.2f} -> {k_kron:.2f} | "
        f"W1/W2 Cos={cos_w1:.4f}/{cos_w2:.4f} | DeployBasis 0-mismatch"
    )

    # -------------------------------------------------------------------------
    # Stage 3: LServe sparse attention & Tiled-CBC sieve adversarial CUDA checks
    # -------------------------------------------------------------------------
    B, H, S_kv, Sq, D_head = 1, 4, 512, 64, 128
    q = torch.randn((B, H, Sq, D_head), dtype=torch.bfloat16, device=dev)
    k = torch.randn((B, H, S_kv, D_head), dtype=torch.bfloat16, device=dev)
    v = torch.randn((B, H, S_kv, D_head), dtype=torch.bfloat16, device=dev)
    k[:, [0, 2], 8:40, :] = q[:, [0, 2], :32, :] * 4.0
    k[:, [0, 2], 464:496, :] = q[:, [0, 2], 32:64, :] * 4.0
    k[:, [1, 3], 196:228, :] = q[:, [1, 3], :32, :] * 4.0
    k[:, [1, 3], 324:356, :] = q[:, [1, 3], 32:64, :] * 4.0

    kv = quantize_kv_pages_int4_hadamard(k, v, page_size=64)
    stream_mask = profile_streaming_head_mask(q, kv, streaming_head_ratio=0.5, sink_pages=1, local_pages=1)
    attn_out, attn_stats = lserve_sparse_hadamard_attention(
        q, kv, top_k_pages=2, sink_pages=1, local_pages=1, streaming_head_mask=stream_mask
    )
    attn_out_cpu_mask, _ = lserve_sparse_hadamard_attention(
        q, kv, top_k_pages=2, sink_pages=1, local_pages=1, streaming_head_mask=stream_mask.cpu()
    )
    attn_ref = (torch.softmax((q.float() @ k.float().transpose(-1, -2)) / math.sqrt(D_head), dim=-1) @ v.float()).to(torch.bfloat16)
    attn_cos = float(torch.nn.functional.cosine_similarity(attn_out.float().flatten(), attn_ref.float().flatten(), dim=0).item())
    assert attn_out.is_cuda and attn_out_cpu_mask.is_cuda and attn_cos > 0.99, attn_cos

    # Sq=1 single-token decode, 0-active-pages finite guard, and non-contiguous extreme keys
    q_dec = torch.randn((1, 4, 1, D_head), dtype=torch.bfloat16, device=dev)
    out_dec, _ = lserve_sparse_hadamard_attention(q_dec, kv, top_k_pages=2, sink_pages=1, local_pages=1, streaming_head_ratio=0.5)
    out_zero_pages, _ = lserve_sparse_hadamard_attention(q_dec, kv, top_k_pages=0, sink_pages=0, local_pages=0, streaming_head_ratio=1.0)
    assert out_dec.shape == q_dec.shape and bool(torch.isfinite(out_dec).all().item())
    assert bool(torch.isfinite(out_zero_pages).all().item()) and float(out_zero_pages.abs().max().item()) == 0.0

    k_ext_nc = torch.randn((1, 256, 4, D_head), dtype=torch.bfloat16, device=dev).transpose(1, 2) * 1e3
    v_ext_nc = torch.randn((1, 256, 4, D_head), dtype=torch.bfloat16, device=dev).transpose(1, 2)
    q_ext_nc = torch.randn((1, 19, 4, D_head), dtype=torch.bfloat16, device=dev).transpose(1, 2)
    assert not k_ext_nc.is_contiguous() and not q_ext_nc.is_contiguous()
    kv_ext = quantize_kv_pages_int4_hadamard(k_ext_nc, v_ext_nc, page_size=64)
    out_ext, _ = lserve_sparse_hadamard_attention(q_ext_nc, kv_ext, top_k_pages=2, sink_pages=1, local_pages=1, streaming_head_ratio=0.5)
    assert bool(torch.isfinite(out_ext).all().item())

    # Persistent channel-outlier Keys on CUDA: per_channel UINT4 causal LServe
    k_ch = torch.randn((1, 4, 256, 64), dtype=torch.bfloat16, device=dev) * 0.6
    v_ch = torch.randn((1, 4, 256, 64), dtype=torch.bfloat16, device=dev) * 0.25
    q_ch = torch.randn((1, 4, 128, 64), dtype=torch.bfloat16, device=dev)
    k_ch[:, :, :, 0] = 120.0
    k_ch[:, :, :, 1] = -95.0
    kv_pch = quantize_kv_pages_int4_hadamard(k_ch, v_ch, page_size=64, key_quant_mode="per_channel")
    out_pch, _ = lserve_sparse_hadamard_attention(q_ch, kv_pch, top_k_pages=4, sink_pages=1, local_pages=2, is_causal=True)
    qp_ch = torch.arange(128, device=dev).unsqueeze(1)
    kp_ch = torch.arange(256, device=dev).unsqueeze(0)
    sc_ch_ref = (q_ch.float() @ k_ch.float().transpose(-1, -2)) / math.sqrt(64)
    out_ch_ref = (torch.softmax(sc_ch_ref.masked_fill(kp_ch > qp_ch, float("-inf")), dim=-1) @ v_ch.float()).to(torch.bfloat16)
    cos_pch = float(torch.nn.functional.cosine_similarity(out_pch.float().flatten(), out_ch_ref.float().flatten(), dim=0).item())
    assert cos_pch > 0.98, f"per_channel CUDA causal LServe failed: {cos_pch}"

    x_sieve = x_rot.clone()
    x_sieve[:, 128:256] *= 0.005
    x_sieve[:, 384:512] *= 0.005
    y_sieve, sieve_stats = tiled_cbc_sieve_gemm(x_sieve, u1, tau_sieve=0.05, use_row_max=True)
    y_dense = unlocked_bf16_gemm(x_sieve, u1.T)
    sieve_err = float((y_sieve.float() - y_dense.float()).norm().item() / (y_dense.float().norm().item() + 1e-12))
    assert y_sieve.is_cuda and sieve_stats["pruned_pct"] == 50.0 and sieve_err < 0.02

    x_nan = torch.zeros((64, 256), dtype=torch.bfloat16, device=dev)
    x_nan[:, 128:] = 1.0
    x_nan[21, 7] = float("nan")
    y_nan, st_nan = tiled_cbc_sieve_gemm(x_nan, u1[:, :256], tau_sieve=0.5)
    y_nan_dense = x_nan.float() @ u1[:, :256].float().T
    assert int(st_nan["executed_tiles"]) == 2 and torch.equal(torch.isnan(y_nan.float()), torch.isnan(y_nan_dense))

    x_zero = torch.zeros((17, 256), dtype=torch.bfloat16, device=dev)
    y_z, st_z = tiled_cbc_sieve_gemm(x_zero, u1[:, :256], tau_sieve=0.05)
    assert st_z["pruned_pct"] == 100.0 and bool((y_z == 0).all().item())
    x_inf = torch.zeros((17, 256), dtype=torch.bfloat16, device=dev)
    x_inf[5, 3] = float("inf")
    y_inf, st_inf = tiled_cbc_sieve_gemm(x_inf, u1[:, :256], tau_sieve=0.5)
    assert int(st_inf["executed_tiles"]) >= 1 and bool(torch.isinf(y_inf[5]).any().item())
    _, st_nan_tau = tiled_cbc_sieve_gemm(x_sieve[:, :256], u1[:, :256], tau_sieve=float("nan"))
    assert st_nan_tau["pruned_pct"] == 0.0

    print(
        f"[3] LServe + Tiled-CBC (CUDA)     : LServe Cos={attn_cos:.5f} (Sq=1/0-page/1e3 ok) | "
        f"CBC Pruned={sieve_stats['pruned_pct']:.1f}%, RelErr={sieve_err:.3e}"
    )

    # -------------------------------------------------------------------------
    # Stage 4: C++20 AVX2 DLL <-> CUDA Hybrid Bridge, Bounds, Threading & HF CUDA
    # -------------------------------------------------------------------------
    bridge = KinesisNativeBridge()
    w_rot_kn = (w_src.cpu() @ r_kron.cpu()).T.contiguous()
    w1_bits_cpp, w1_scales_cpp = bridge.pack_w1_panels(w_rot_kn)
    gpu_u32_cpu, gpu_scales_cpu = bridge.convert_w1_to_gpu_interleaved(w1_bits_cpp, w1_scales_cpp, 512, 256)
    y_cpu_w1, _ = bridge.smart_w1_gemm(x_rot.cpu().float(), w1_bits_cpp, w1_scales_cpp, N=256, error_budget_eps=0.0)
    u1_gpu_bridged = unpack_w1_g128_lop3_reference(gpu_u32_cpu.to(dev), gpu_scales_cpu.to(dev))
    y_gpu_bridged = unlocked_bf16_gemm(x_rot, u1_gpu_bridged.T)
    bridge_rel_err = float((y_gpu_bridged.cpu().float() - y_cpu_w1).norm().item() / (y_cpu_w1.norm().item() + 1e-12))
    assert bridge_rel_err < 1.5e-2, f"C++ DLL <-> CUDA bridge error too high: {bridge_rel_err}"

    # Odd (M, N, K) C++20 DLL vs CUDA parity
    for m_o, n_o, k_o in ((1, 8, 128), (17, 37, 256)):
        xo = torch.randn((m_o, k_o), dtype=torch.float32)
        wo = torch.randn((k_o, n_o), dtype=torch.float32)
        bo, so = bridge.pack_w1_panels(wo)
        yo_cpu, _ = bridge.smart_w1_gemm(xo, bo, so, N=n_o, error_budget_eps=0.0)
        u32_o, gsc_o = bridge.convert_w1_to_gpu_interleaved(bo, so, k_o, n_o)
        uo = unpack_w1_g128_lop3_reference(u32_o.to(dev), gsc_o.to(dev))
        yo_gpu = unlocked_bf16_gemm(xo.to(dev, dtype=torch.bfloat16), uo.T).cpu().float()
        rel_o = float((yo_gpu - yo_cpu).norm().item() / (yo_cpu.norm().item() + 1e-8))
        assert bool(torch.isfinite(yo_cpu).all().item()) and rel_o < 2.5e-2, rel_o

    # Direct C ABI short-buffer -3 guard
    w_sb = torch.randn((128, 16), dtype=torch.float32)
    sb_bits = torch.empty(1, dtype=torch.uint8)
    sb_sc = torch.empty(1, dtype=torch.uint16)
    rc_sb = bridge._lib.uha_kinesis_pack_w1_panels(
        ctypes.cast(w_sb.data_ptr(), ctypes.POINTER(ctypes.c_float)), 128, 16,
        ctypes.cast(sb_bits.data_ptr(), ctypes.POINTER(ctypes.c_uint8)), 1,
        ctypes.cast(sb_sc.data_ptr(), ctypes.POINTER(ctypes.c_uint16)), 1,
    )
    assert rc_sb == -3, f"Expected -3 for short buffer, got {rc_sb}"

    # 8-thread concurrent DLL determinism check
    x_mt = torch.randn((4, 128), dtype=torch.float32)
    b_mt, s_mt = bridge.pack_w1_panels(w_sb)
    ref_mt, _ = bridge.smart_w1_gemm(x_mt, b_mt, s_mt, N=16, error_budget_eps=0.0)

    def _mt_worker(_: int) -> None:
        for _ in range(16):
            y_t, _ = bridge.smart_w1_gemm(x_mt, b_mt, s_mt, N=16, error_budget_eps=0.0)
            assert torch.equal(y_t, ref_mt), "Non-determinism under concurrent DLL calls"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_mt_worker, range(8)))

    # CUDA HF Transformer layer converter + SafeTensors round-trip regression
    hf_sd = {
        "self_attn.o_proj.weight": torch.randn((128, 128), device=dev),
        "mlp.gate_proj.weight": torch.randn((256, 128), device=dev),
        "mlp.up_proj.weight": torch.randn((256, 128), device=dev),
    }
    hf_cal = torch.randn((16, 128), device=dev)
    packed_hf, _ = convert_hf_transformer_layer_to_sub2bit(hf_sd, hf_cal, bridge=bridge)
    blob = save_safetensors_bytes(packed_hf, metadata={"source": "cuda_g2"})
    reloaded, meta = load_safetensors_bytes(blob)
    assert meta.get("source") == "cuda_g2"
    for k_name, t_val in packed_hf.items():
        assert torch.equal(reloaded[k_name], t_val.detach().cpu())

    print(
        f"[4] C++20 DLL <-> CUDA Bridge     : RelErr={bridge_rel_err:.3e} | short-buf=-3 | "
        f"8-thread deterministic | CUDA HF->SafeTensors ok"
    )

    # -------------------------------------------------------------------------
    # Stage 5: Live nvidia-smi Telemetry + AD107 45W Sag & NaN Floor Regression
    # -------------------------------------------------------------------------
    assert poll_live_gpu_telemetry(gpu_index=99) is None, "Invalid gpu_index=99 must return None"
    sample = poll_live_gpu_telemetry(layer_idx=16, gpu_ready_seq=10, cpu_done_seq=10, fallback_power_cap_watts=45.0)
    assert sample is not None, "poll_live_gpu_telemetry returned None despite active CUDA GPU"
    gov = ClosedLoopHardwareGovernor(arch="AD107", initial_tau_sieve=0.040)
    dec = gov.route_moe_and_regulate_sieve(sample)
    assert math.isfinite(sample.vram_used_gb) and math.isfinite(sample.effective_sm_clock_mhz)

    gov_adv = ClosedLoopHardwareGovernor(arch="AD107", initial_tau_sieve=0.040)
    dec_sag = gov_adv.route_moe_and_regulate_sieve(
        HardwareTelemetrySample(16, 10, 10, 6.8, 8.0, 1450.0, 1800.0, 45.0, 45.0)
    )
    assert dec_sag["governor_action"] == "RAISE_SIEVE_+5%_SHED_POWER" and float(dec_sag["tau_sieve"]) > 0.040
    tau_hold = gov_adv.tau_sieve
    dec_nf = gov_adv.route_moe_and_regulate_sieve(
        HardwareTelemetrySample(32, 10, 10, 4.0, 8.0, 1800.0, float("nan"), 20.0, 45.0)
    )
    assert dec_nf["governor_action"] == "HOLD" and gov_adv.tau_sieve == tau_hold, dec_nf

    print(
        f"[5] Live AD107 Telemetry Governor : VRAM={sample.vram_used_gb:.2f}/{sample.vram_capacity_gb:.2f} GB | "
        f"SM={sample.effective_sm_clock_mhz:.0f} MHz | Power={sample.board_power_watts:.1f}/{sample.power_cap_watts:.0f} W | "
        f"Action={dec['governor_action']} | sag+NaN-floor ok"
    )

    # -------------------------------------------------------------------------
    # Stage 6: Adversarial QAT + GaLore (non-square, M=17, rank=1, 0-grad, persistent pin)
    # -------------------------------------------------------------------------
    ns_model = torch.nn.Sequential(
        FusedQATG128Linear(512, 1024, clip_mult=2.5, device=dev, master_dtype=torch.bfloat16),
        torch.nn.GELU(),
        FusedQATG128Linear(1024, 256, clip_mult=0.0, device=dev, master_dtype=torch.bfloat16),
        torch.nn.GELU(),
        DeployBasisQATLinear(256, 512, fmt="w2", master_dtype=torch.float32, device=dev),
    )
    ns_eng = LayerWise8BitGaLoreEngine(
        ns_model, rank=64, update_proj_gap=2, lr=8e-3, proj_bits=4,
        lazy_subspace=True, subspace_iters=2, reproject_moments=True, refresh_v_fill=True, pin_memory=True,
    )
    x_ns = torch.randn((17, 512), device=dev, dtype=torch.bfloat16)
    y_ns = torch.randn((17, 512), device=dev, dtype=torch.bfloat16)
    ns_losses = []
    pinned_ptrs: dict[tuple[int, str], int] = {}
    for step in range(4):
        out_ns = ns_model(x_ns)
        loss_ns = torch.nn.functional.mse_loss(out_ns.float(), y_ns.float())
        ns_losses.append(float(loss_ns.item()))
        loss_ns.backward()
        for p in ns_model.parameters():
            assert p.grad is None and bool(torch.isfinite(p).all().item())
        for idx, st in enumerate(ns_eng.states.values()):
            for attr in ("proj_q", "proj_scale", "exp_avg_q", "exp_avg_scale", "exp_avg_sq_q", "exp_avg_sq_scale"):
                t_cpu = getattr(st, attr)
                assert t_cpu.device.type == "cpu" and t_cpu.is_pinned(), f"{attr} not pinned"
                key = (idx, attr)
                if step == 0:
                    pinned_ptrs[key] = int(t_cpu.data_ptr())
                else:
                    assert int(t_cpu.data_ptr()) == pinned_ptrs[key], f"{attr} reallocated instead of reusing pinned buffer"
    assert ns_losses[-1] < ns_losses[0], f"Non-square QAT+GaLore loss did not drop: {ns_losses}"
    for h in ns_eng.hooks:
        h.remove()

    # Odd rank=1 INT4 + all-zero gradient -> nonzero gradient recovery
    z_layer = FusedQATG128Linear(256, 128, clip_mult=2.5, device=dev, master_dtype=torch.bfloat16)
    z_eng = LayerWise8BitGaLoreEngine(
        z_layer, rank=1, update_proj_gap=1, lr=1e-3, proj_bits=4,
        subspace_iters=2, reproject_moments=True, refresh_v_fill=True, pin_memory=True,
    )
    x_z0 = torch.zeros((8, 256), device=dev, dtype=torch.bfloat16)
    for _ in range(2):
        (z_layer(x_z0).float() * 0.0).sum().backward()
        assert bool(torch.isfinite(z_layer.weight).all().item())
    x_z1 = torch.randn((8, 256), device=dev, dtype=torch.bfloat16)
    for _ in range(2):
        z_layer(x_z1).float().pow(2).mean().backward()
        assert bool(torch.isfinite(z_layer.weight).all().item())
    for h in z_eng.hooks:
        h.remove()
    del ns_model, ns_eng, z_layer, z_eng, x_ns, y_ns, x_z0, x_z1
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"[6] QAT + GaLore Adversarial      : non-square M=17 Loss {ns_losses[0]:.4f} -> {ns_losses[-1]:.4f} | "
        f"18 persistent pinned buffers reused | rank=1/0-grad recovery ok"
    )

    # -------------------------------------------------------------------------
    # Stage 7: 100.7M-Param 8 GB RTX 4060 Scale Stress + Sustained 4096^3 BF16
    # -------------------------------------------------------------------------
    n_layers, dim, batch = 24, 2048, 16
    n_params = n_layers * dim * dim

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    base_layers = torch.nn.ModuleList(
        [torch.nn.Linear(dim, dim, bias=False, dtype=torch.bfloat16) for _ in range(n_layers)]
    ).to(dev)
    opt = torch.optim.AdamW(base_layers.parameters(), lr=1e-4)
    x_scale = torch.randn((batch, dim), dtype=torch.bfloat16, device=dev) * 0.1
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        h_s = x_scale
        for layer in base_layers:
            h_s = h_s + 0.05 * torch.tanh(layer(h_s))
        h_s.float().pow(2).mean().backward()
        opt.step()
    torch.cuda.synchronize()
    adamw_peak_mib = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
    del base_layers, opt, h_s
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(dev)
    qat_layers = torch.nn.ModuleList(
        [FusedQATG128Linear(dim, dim, clip_mult=2.5, device=dev, master_dtype=torch.bfloat16) for _ in range(n_layers)]
    )
    galore_scale = LayerWise8BitGaLoreEngine(
        qat_layers, rank=128, update_proj_gap=3, lr=2e-3, proj_bits=4,
        lazy_subspace=True, subspace_iters=2, pin_memory=True,
    )
    scale_losses = []
    step_mem_mib = []
    t0 = time.perf_counter()
    for _ in range(5):
        h_s = x_scale
        for layer in qat_layers:
            h_s = h_s + 0.05 * torch.tanh(layer(h_s))
        loss_s = h_s.float().pow(2).mean()
        scale_losses.append(float(loss_s.item()))
        loss_s.backward()
        for p in qat_layers.parameters():
            assert p.grad is None, "Q-GaLore failed to free parameter gradient!"
        torch.cuda.synchronize()
        step_mem_mib.append(torch.cuda.memory_allocated(dev) / (1024 ** 2))
    dt_scale = time.perf_counter() - t0
    qat_peak_mib = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
    for h_hook in galore_scale.hooks:
        h_hook.remove()
    del qat_layers, galore_scale, x_scale, h_s, loss_s
    gc.collect()
    torch.cuda.empty_cache()

    vram_saved_pct = (1.0 - (qat_peak_mib / adamw_peak_mib)) * 100.0
    leak_mib = step_mem_mib[-1] - step_mem_mib[1]
    assert vram_saved_pct > 45.0, f"Expected >45% VRAM savings at 100.7M params, got {vram_saved_pct:.1f}%"
    assert abs(leak_mib) < 2.0, f"Memory leak across steps: {step_mem_mib}"
    assert scale_losses[-1] < scale_losses[0], f"Scale loss did not decrease: {scale_losses}"

    # Sustained 100x 4096^3 BF16 GEMM stability check
    a4k = torch.randn((4096, 4096), dtype=torch.bfloat16, device=dev)
    b4k = torch.randn((4096, 4096), dtype=torch.bfloat16, device=dev)
    for _ in range(5):
        _ = unlocked_bf16_gemm(a4k, b4k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters_4k = 100
    for _ in range(iters_4k):
        c4k = unlocked_bf16_gemm(a4k, b4k)
    torch.cuda.synchronize()
    dt_4k = time.perf_counter() - t0
    assert bool(torch.isfinite(c4k).all().item())
    tflops_4k = (iters_4k * 2.0 * (4096 ** 3)) / dt_4k / 1e12

    print(
        f"[7] 100.7M-Param 8GB Scale Stress : {n_params / 1e6:.1f}M params | Peak VRAM {qat_peak_mib:.1f} MiB vs "
        f"AdamW {adamw_peak_mib:.1f} MiB (-{vram_saved_pct:.1f}%) | leak={leak_mib:+.2f} MiB | "
        f"Loss {scale_losses[0]:.4e} -> {scale_losses[-1]:.4e} ({dt_scale:.1f}s) | 4096^3={tflops_4k:.2f} TFLOPS"
    )
    print("\n[+] PASS: Live CUDA GPU verification and 8GB red-team stress suite completed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
