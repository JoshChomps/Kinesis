#!/usr/bin/env python3
"""
File: e2e_transformer_pipeline.py
End-to-End 5-Pillar Transformer Block Verification Pipeline (Suite G1).
Chains all five hardware-software co-design pillars in a single executable forward + QAT training pipeline:
  - Pillar 1: compiler_unlock/unlocked_gemm.py (Unlocked Tensor Core BF16/FP32-acc GEMM dispatch)
  - Pillar 2: fused_w1_hadacore/spinquant_packer.py (Kronecker-Stiefel R_P (x) H_16 rotation + lop3.b32 W1/W2 packing)
  - Pillar 3: qat_galore_engine/fused_qat_ste.py + galore_async_offload.py (Clipped STE QAT + INT4 Q-GaLore hooks)
  - Pillar 4 & 5: structural_filters/lserve_sparse_attn.py + tiled_cbc_sieve.py + hetero_runtime/closed_loop_governor.py
"""

from pathlib import Path
import math
import sys
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
    sys.path.insert(0, str(ROOT / subdir))

from unlocked_gemm import unlocked_bf16_gemm  # noqa: E402
from kinesis_bridge import KinesisNativeBridge  # noqa: E402
from spinquant_packer import (  # noqa: E402
    build_kernel_online_rotation,
    optimize_kronecker_stiefel_rotation,
    quantize_and_pack_w1_g128_interleaved,
    unpack_w1_g128_lop3_reference,
    quantize_and_pack_w2_ternary_interleaved,
    unpack_w2_ternary_lop3_reference,
)
from fused_qat_ste import FusedQATG128Linear  # noqa: E402
from galore_async_offload import LayerWise8BitGaLoreEngine  # noqa: E402
from lserve_sparse_attn import (  # noqa: E402
    quantize_kv_pages_int4_hadamard,
    profile_streaming_head_mask,
    lserve_sparse_hadamard_attention,
)
from tiled_cbc_sieve import tiled_cbc_sieve_gemm  # noqa: E402
from closed_loop_governor import (  # noqa: E402
    ClosedLoopHardwareGovernor,
    HardwareTelemetrySample,
)


def run_e2e_transformer_pipeline() -> None:
    print("================================================================================")
    print("  SUITE G1: END-TO-END 5-PILLAR TRANSFORMER BLOCK & QAT VERIFICATION PIPELINE")
    print("================================================================================\n")

    torch.manual_seed(2026)

    # -------------------------------------------------------------------------
    # Stage 1 (Pillar 5A): LServe Salience-Profiled Sparse + 4-Bit Hadamard KV Attention
    # -------------------------------------------------------------------------
    B, H, S_kv, Sq, D_head = 1, 4, 512, 64, 128  # D_model = H * D_head = 512 (divisible by g128)
    D_model = H * D_head
    q = torch.randn((B, H, Sq, D_head), dtype=torch.bfloat16)
    k = torch.randn((B, H, S_kv, D_head), dtype=torch.bfloat16)
    v = torch.randn((B, H, S_kv, D_head), dtype=torch.bfloat16)

    # Heads 0, 2 attend to sink/local pages; Heads 1, 3 retrieve planted middle-of-context needles
    k[:, [0, 2], 8:40, :] = q[:, [0, 2], :32, :] * 4.0
    k[:, [0, 2], 464:496, :] = q[:, [0, 2], 32:64, :] * 4.0
    k[:, [1, 3], 196:228, :] = q[:, [1, 3], :32, :] * 4.0
    k[:, [1, 3], 324:356, :] = q[:, [1, 3], 32:64, :] * 4.0

    kv_cache = quantize_kv_pages_int4_hadamard(k, v, page_size=64)
    stream_mask = profile_streaming_head_mask(
        q, kv_cache, streaming_head_ratio=0.5, sink_pages=1, local_pages=1
    )
    attn_out_bhsd, attn_stats = lserve_sparse_hadamard_attention(
        q, kv_cache, top_k_pages=2, sink_pages=1, local_pages=1, streaming_head_mask=stream_mask
    )

    scores_ref = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(D_head)
    attn_ref_bhsd = (torch.softmax(scores_ref, dim=-1) @ v.float()).to(torch.bfloat16)
    attn_cos = float(
        torch.nn.functional.cosine_similarity(
            attn_out_bhsd.float().flatten(), attn_ref_bhsd.float().flatten(), dim=0
        ).item()
    )
    assert attn_cos > 0.99, f"Stage 1 LServe attention cosine similarity too low: {attn_cos}"

    # Reshape attention output to [M = B * Sq = 64, K = D_model = 512] and inject realistic channel outliers
    x_attn = attn_out_bhsd.transpose(1, 2).reshape(B * Sq, D_model).to(torch.float32)
    x_attn[:, 0] *= 16.0
    x_attn[:, 129] *= 12.0

    # -------------------------------------------------------------------------
    # Stage 2 (Pillar 1 + Pillar 2): Kronecker-Stiefel R_P (x) H_16 Rotation +
    # Offline R_P Absorption into W_O + Online H_16 + Interleaved lop3.b32 W1/W2
    # -------------------------------------------------------------------------
    w_o = torch.randn((D_model, D_model), dtype=torch.float32) * (1.0 / math.sqrt(D_model))
    # Inject channel outliers directly into W_O output rows (channels 3 and 67) so h_pre = x_attn @ W_O^T
    # exhibits realistic heavy-tailed residual outliers while preserving exact linearity in W_O.
    w_o[3, :] *= 15.0
    w_o[67, :] *= 11.0
    h_pre = x_attn @ w_o.T

    r_kron, r_p, k_init, k_h16, k_kron = optimize_kronecker_stiefel_rotation(
        h_pre, q_hadamard=16, steps=20, lr=0.25
    )
    r_prev_absorb = torch.kron(r_p, torch.eye(16, dtype=torch.float32))
    r_online_h16 = build_kernel_online_rotation(D_model, mode=1)

    # True offline absorption into preceding projection: W_O_absorbed = (R_P^T (x) I_16) @ W_O
    # so that (x_attn @ W_O_absorbed^T) @ (I_P (x) H_16) == (x_attn @ W_O^T) @ R_kron with ZERO extra runtime GEMM!
    w_o_absorbed = r_prev_absorb.T @ w_o
    h_absorbed = unlocked_bf16_gemm(x_attn.to(torch.bfloat16), w_o_absorbed.T.to(torch.bfloat16))
    h_online_rot = (h_absorbed.to(torch.float32) @ r_online_h16).to(torch.bfloat16)
    h_direct_rot = (h_pre @ r_kron).to(torch.bfloat16)
    rot_rel_err = float(
        (h_online_rot.float() - h_direct_rot.float()).norm().item()
        / (h_direct_rot.float().norm().item() + 1e-12)
    )
    assert k_kron < k_h16 < k_init, (k_init, k_h16, k_kron)
    assert rot_rel_err < 1.5e-2, f"Kronecker rotation BF16 relative error too high: {rot_rel_err}"

    # Pack MoE hot expert (W1 binary g128) and shared FFN expert (W2 ternary g128) with merged R_kron
    N_ffn = 256
    w_hot_fp32 = torch.randn((N_ffn, D_model), dtype=torch.float32) * 0.04
    w_shared_fp32 = torch.randn((N_ffn, D_model), dtype=torch.float32) * 0.04

    packed_w1, scales_w1 = quantize_and_pack_w1_g128_interleaved(w_hot_fp32, R1=r_kron)
    w1_dequant_bf16 = unpack_w1_g128_lop3_reference(packed_w1, scales_w1)

    packed_w2, scales_w2 = quantize_and_pack_w2_ternary_interleaved(w_shared_fp32, R1=r_kron)
    w2_dequant_bf16 = unpack_w2_ternary_lop3_reference(packed_w2, scales_w2)

    # Verify quantized rotated outputs against unrotated FP32 ground-truth projections (h_pre @ W^T)
    y_hot_fp32_gt = h_pre @ w_hot_fp32.T
    y_shared_fp32_gt = h_pre @ w_shared_fp32.T
    y_hot_rot_full = unlocked_bf16_gemm(h_online_rot, w1_dequant_bf16.T)
    y_shared = unlocked_bf16_gemm(h_online_rot, w2_dequant_bf16.T)
    w1_gt_cos = float(
        torch.nn.functional.cosine_similarity(
            y_hot_rot_full.float().flatten(), y_hot_fp32_gt.flatten(), dim=0
        ).item()
    )
    w2_gt_cos = float(
        torch.nn.functional.cosine_similarity(
            y_shared.float().flatten(), y_shared_fp32_gt.flatten(), dim=0
        ).item()
    )
    assert w1_gt_cos > 0.75, f"1-bit W1A16 vs FP32 ground-truth cosine similarity too low: {w1_gt_cos}"
    assert w2_gt_cos > 0.85, f"1.58-bit W2A16 vs FP32 ground-truth cosine similarity too low: {w2_gt_cos}"

    # Also execute the compiled C++20 AVX2 W1 & 256-bit T-MAC LUT kernels via KinesisNativeBridge
    native_bridge = KinesisNativeBridge()
    w_hot_rot_kn = (w_hot_fp32 @ r_kron).T.contiguous()
    w1_bits_cpp, w1_scales_cpp = native_bridge.pack_w1_panels(w_hot_rot_kn)
    gpu_u32_bridged, gpu_scales_bridged = native_bridge.convert_w1_to_gpu_interleaved(
        w1_bits_cpp, w1_scales_cpp, D_model, N_ffn
    )
    w1_dequant_bridged = unpack_w1_g128_lop3_reference(gpu_u32_bridged, gpu_scales_bridged)
    y_hot_cpp_w1, _ = native_bridge.smart_w1_gemm(
        h_online_rot.float(), w1_bits_cpp, w1_scales_cpp, N=N_ffn, error_budget_eps=0.0
    )
    y_hot_fp32_lop3_ref = h_online_rot.float() @ w1_dequant_bridged.float().T
    cpp_vs_py_w1_err = float((y_hot_cpp_w1 - y_hot_fp32_lop3_ref).abs().max().item())
    assert cpp_vs_py_w1_err < 1e-5, f"Native C++20 W1 vs Python lop3 reference error too high: {cpp_vs_py_w1_err}"

    # -------------------------------------------------------------------------
    # Stage 3 (Pillar 4 + Pillar 5B): Closed-Loop Chorus MoE Routing & Row-Max Tiled-CBC Sieve
    # -------------------------------------------------------------------------
    governor = ClosedLoopHardwareGovernor(arch="AD102", initial_tau_sieve=0.035)
    telemetry_normal = HardwareTelemetrySample(
        layer_idx=16,
        gpu_ready_seq=12,
        cpu_done_seq=11,  # Delta_Q = 1 <= 1 -> offload tail expert to Host AMX 1GB HugePage Ring
        vram_used_gb=21.5,
        vram_capacity_gb=24.0,
        effective_sm_clock_mhz=2440.0,  # 3.17% clock sag -> governor raises tau_sieve by +5%
        locked_clock_floor_mhz=2520.0,
        board_power_watts=446.0,
        power_cap_watts=450.0,
    )
    gov_decision = governor.route_moe_and_regulate_sieve(telemetry_normal)
    assert gov_decision["execution_target"] == "HOST_AMX_1GB_HUGEPAGE_RING"
    assert gov_decision["governor_action"] == "RAISE_SIEVE_+5%_SHED_POWER"

    # Simulate structured post-activation sparsity on 2 of 4 K=128 blocks and run Row-Max Tiled-CBC Sieve
    h_sparse_rot = h_online_rot.clone()
    h_sparse_rot[:, 128:256] *= 0.005
    h_sparse_rot[:, 384:512] *= 0.005
    y_hot_sieve, sieve_stats = tiled_cbc_sieve_gemm(
        h_sparse_rot, w1_dequant_bf16, tau_sieve=float(gov_decision["tau_sieve"]), use_row_max=True
    )
    y_hot_dense = unlocked_bf16_gemm(h_sparse_rot, w1_dequant_bf16.T)
    sieve_rel_err = float(
        (y_hot_sieve.float() - y_hot_dense.float()).norm().item()
        / (y_hot_dense.float().norm().item() + 1e-12)
    )
    assert sieve_stats["pruned_pct"] == 50.0, sieve_stats
    assert sieve_rel_err < 0.02, f"Tiled-CBC sieve relative error too high: {sieve_rel_err}"

    ffn_hidden = torch.nn.functional.silu(y_hot_sieve.float()) + y_shared.float()

    # -------------------------------------------------------------------------
    # Stage 4 (Pillar 3): In-SRAM Clipped STE QAT + INT4 Q-GaLore Post-Accumulate Hooks
    # -------------------------------------------------------------------------
    qat_head = torch.nn.Sequential(
        FusedQATG128Linear(N_ffn, 256, clip_mult=2.5),
        FusedQATG128Linear(256, 128, clip_mult=2.5),
    )
    galore_engine = LayerWise8BitGaLoreEngine(
        qat_head,
        rank=16,
        update_proj_gap=2,
        lr=4e-3,
        proj_bits=4,
        lazy_subspace=True,
        cos_sim_threshold=0.90,
        reproject_moments=True,
    )
    qat_in = ffn_hidden.detach().to(torch.bfloat16)
    qat_target = torch.randn((B * Sq, 128), dtype=torch.bfloat16)

    qat_losses = []
    for _ in range(8):
        pred = qat_head(qat_in)
        loss = torch.nn.functional.mse_loss(pred.float(), qat_target.float())
        qat_losses.append(float(loss.item()))
        loss.backward()
        for p in qat_head.parameters():
            assert p.grad is None, "Q-GaLore post-accumulate hook failed to free parameter gradient!"

    max_single_layer_bytes = 256 * 256 * 2  # BF16 [256, 256]
    assert galore_engine.peak_transient_grad_bytes == max_single_layer_bytes
    assert qat_losses[-1] < qat_losses[0], qat_losses

    print(
        f"[1] Pillar 5A LServe INT4-Hadamard KV Attention : Cosine={attn_cos:.5f} | "
        f"Sparsity={attn_stats['page_sparsity_pct']:.1f}% | StreamMask={stream_mask.tolist()}"
    )
    print(
        f"[2] Pillar 1+2 Kronecker R_P(x)H_16 & lop3.b32  : Kurtosis {k_init:.2f} -> {k_h16:.2f} -> {k_kron:.2f} | "
        f"Absorbed RelErr={rot_rel_err:.4e} | W1/W2 GT Cos={w1_gt_cos:.4f}/{w2_gt_cos:.4f} | CppW1Err={cpp_vs_py_w1_err:.2e}"
    )
    print(
        f"[3] Pillar 4+5B Chorus Governor & RowMax Sieve  : Route={gov_decision['execution_target']} | "
        f"Pruned={sieve_stats['pruned_pct']:.1f}% | RelErr={sieve_rel_err:.4e}"
    )
    print(
        f"[4] Pillar 3 Clipped STE QAT + INT4 Q-GaLore    : Loss {qat_losses[0]:.4f} -> {qat_losses[-1]:.4f} | "
        f"PeakTransientGrad={galore_engine.peak_transient_grad_bytes} B | RetainedGrad=0 B"
    )
    print("[+] PASS: End-to-End 5-Pillar Transformer Block & QAT Pipeline verified.")


if __name__ == "__main__":
    run_e2e_transformer_pipeline()
