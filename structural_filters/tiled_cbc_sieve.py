#!/usr/bin/env python3
"""
File: structural_filters/tiled_cbc_sieve.py
Workstream 4: Column-Major Subspace Energy Sieve (Tiled-CBC Dynamic Gate) Reference & Verifier.
Evaluates per-tile row-maximum Holder L1 activation energy E_k = max_m mean_k(|X_tilde[m, 128k:128(k+1)]|)
along BLOCK_K = 128 columns (or K = 32 sub-slices, O9) and dynamically skips cp.async GDDR6X weight reads
and HMMA waves for low-energy subspace tiles without dropping single-token outlier activations.
"""

from typing import Tuple, Dict
import torch


def tiled_cbc_sieve_gemm(
    x_rot_bf16: torch.Tensor,
    w_dequant_bf16: torch.Tensor,
    tau_sieve: float,
    block_m: int = 64,
    block_k: int = 128,
    use_row_max: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Executes Y = X_rot @ W^T in blocks of (BLOCK_M, BLOCK_K = 128) with dynamic
    Tiled-CBC subspace gating based on per-tile row-maximum Holder L1 norm
    E_k = max_{m in tile} mean_{k in tile} |X[m, k]| > tau_sieve (fixes O9).

    tau_sieve is an ACTIVATION-ENERGY threshold: it has no weight-scale factor, so it is not an output-error
    budget (docs/12 3.6 / 4.13). The guaranteed `<= eps` sieve is the C++ SmartTieredCache error budget.
    Non-finite tiles are always executed: the gate is `not (E_k <= tau)`, so a NaN energy keeps its tile and the
    NaN propagates to the output exactly as in the dense GEMM (docs/12 4.12).
    """
    M, K = x_rot_bf16.shape
    N, K_w = w_dequant_bf16.shape
    assert K == K_w and K % block_k == 0

    num_m_tiles = (M + block_m - 1) // block_m
    num_k_tiles = K // block_k

    y_out = torch.zeros((M, N), dtype=torch.float32, device=x_rot_bf16.device)
    total_tiles = num_m_tiles * num_k_tiles
    executed_tiles = 0

    for mi in range(num_m_tiles):
        m_start = mi * block_m
        m_end = min(M, m_start + block_m)
        x_m = x_rot_bf16[m_start:m_end, :].to(torch.float32)

        for ki in range(num_k_tiles):
            k_start = ki * block_k
            k_end = k_start + block_k
            x_tile = x_m[:, k_start:k_end]
            if use_row_max:
                e_k = float(x_tile.abs().mean(dim=-1).amax().item())
            else:
                e_k = float(x_tile.abs().mean().item())
            if not (e_k <= tau_sieve):   # NaN-safe: a NaN energy keeps the tile
                w_tile = w_dequant_bf16[:, k_start:k_end].to(torch.float32)
                y_out[m_start:m_end, :] += x_tile @ w_tile.T
                executed_tiles += 1

    pruned_tiles = total_tiles - executed_tiles
    pruned_pct = (pruned_tiles / max(1, total_tiles)) * 100.0
    return y_out.to(torch.bfloat16), {
        "total_tiles": float(total_tiles),
        "executed_tiles": float(executed_tiles),
        "pruned_tiles": float(pruned_tiles),
        "pruned_pct": pruned_pct,
    }


def main() -> None:
    print("================================================================================")
    print("  WORKSTREAM 4A: TILED-CBC SUBSPACE ENERGY SIEVE VERIFICATION")
    print("================================================================================\n")

    torch.manual_seed(7)
    M, K, N = 256, 1024, 512
    x = torch.randn((M, K), dtype=torch.float32)
    # Simulate structured subspace sparsity post-MLP / post-Hadamard where 50% of K-blocks are low-energy
    for ki in range(0, K // 128, 2):
        x[:, ki * 128 : (ki + 1) * 128] *= 0.015

    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = (torch.randn((N, K), dtype=torch.float32) * 0.05).to(torch.bfloat16)

    y_dense = (x_bf16.float() @ w_bf16.float().T).to(torch.bfloat16)
    y_sieve, stats = tiled_cbc_sieve_gemm(x_bf16, w_bf16, tau_sieve=0.05)

    rel_frob_err = float((torch.norm(y_sieve.float() - y_dense.float()) / torch.norm(y_dense.float())).item())
    print(f"[1] Total BLOCK_K=128 Subspace Tiles : {int(stats['total_tiles'])}")
    print(f"[2] Dynamically Pruned Weight Tiles  : {int(stats['pruned_tiles'])} ({stats['pruned_pct']:.1f}% DRAM & HMMA savings)")
    print(f"[3] Relative Output Frobenius Error  : {rel_frob_err:.4e}")
    assert 40.0 <= stats["pruned_pct"] <= 65.0
    assert rel_frob_err < 0.025

    # [4] O9 Single-Outlier-Row Holder Bound Verification (1 loud row, 63 silent rows)
    x_outlier = torch.zeros((64, 128), dtype=torch.bfloat16)
    x_outlier[13, :] = 1.0
    y_out_rowmax, stats_rowmax = tiled_cbc_sieve_gemm(x_outlier, w_bf16[:, :128], tau_sieve=0.5, use_row_max=True)
    _, stats_mean = tiled_cbc_sieve_gemm(x_outlier, w_bf16[:, :128], tau_sieve=0.5, use_row_max=False)
    print(f"[4] O9 Single-Outlier-Row Gate       : RowMax Executed={int(stats_rowmax['executed_tiles'])}/1 | Mean-L1 Executed={int(stats_mean['executed_tiles'])}/1")
    assert int(stats_rowmax["executed_tiles"]) == 1 and int(stats_mean["executed_tiles"]) == 0
    assert float(y_out_rowmax[13].abs().max().item()) > 0.0
    # [5] docs/12 4.12 regression: one NaN in an otherwise silent tile must reach the output, exactly like the dense
    #     GEMM (the tile used to be skipped, giving 64/64 finite rows where dense gives NaN in that row).
    x_nan = torch.zeros((64, 256), dtype=torch.bfloat16)
    x_nan[:, 128:] = 1.0
    x_nan[21, 7] = float("nan")
    y_nan, st_nan = tiled_cbc_sieve_gemm(x_nan, w_bf16[:, :256], tau_sieve=0.5)
    y_nan_dense = (x_nan.float() @ w_bf16[:, :256].float().T)
    same_pattern = bool(torch.equal(torch.isnan(y_nan.float()), torch.isnan(y_nan_dense)))
    print(f"[5] NaN in a silent tile (4.12)      : executed {int(st_nan['executed_tiles'])}/2 tiles | "
          f"NaN rows sieve/dense = {int(torch.isnan(y_nan.float()).any(1).sum())}/{int(torch.isnan(y_nan_dense).any(1).sum())}")
    assert int(st_nan["executed_tiles"]) == 2 and same_pattern
    print("[+] PASS: Tiled-CBC dynamic subspace energy sieve and O9 row-max Holder bound verified.")


if __name__ == "__main__":
    main()
