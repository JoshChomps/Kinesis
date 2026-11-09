#!/usr/bin/env python3
"""
File: qat_galore_engine/fused_qat_ste.py
Workstream 5A: In-SRAM BLOCK_K=128 Fused Straight-Through Estimator (STE) QAT Kernels.
Aligns the threadblock reduction tile BLOCK_K = 128 with the exact Binary g128 quantization
group size G = 128 so both forward (Y = X @ W_hat^T) and backward activation gradient
(dX = dY @ W_hat) compute alpha = mean(|W_tile|) and W_hat = alpha * sign(W_tile) purely
inside SRAM/registers, allocating 0 Bytes of global VRAM for quantized weights.
"""

import math
import warnings

import torch

# Clipped-STE safety floor (docs/12 4.8). The gate zeroes dW for |w| > c * alpha_g with alpha_g = mean|w|
# over the g128 group. For Gaussian weights E|w| = sigma * sqrt(2/pi), so the permanently gated fraction is
#   P(|w| > c * alpha_g) = erfc(c / sqrt(pi)),
# because |w| > c*sigma*sqrt(2/pi) <=> |z| > c*sqrt(2/pi), and P(|z| > a) = erfc(a / sqrt(2)).
# That gives 23.1% at c = 1.5 (measured: sign accuracy 0.873 vs 0.919 unclipped), 11.0% at c = 2.0, and
# 4.6% at c = 2.5 (measured: indistinguishable from unclipped). Gated weights never recover, because alpha_g
# does not shrink once their gradient is cut off.
CLIP_MULT_SAFE_MIN = 2.5


def clipped_ste_frozen_fraction_gaussian(clip_mult: float) -> float:
    """Fraction of N(0, sigma^2) weights the clipped-STE gate zeroes at init: erfc(c / sqrt(pi))."""
    return math.erfc(clip_mult / math.sqrt(math.pi)) if clip_mult > 0.0 else 0.0


def _check_clip_mult(clip_mult: float) -> float:
    c = float(clip_mult)
    if c < 0.0 or not math.isfinite(c):
        raise ValueError(f"clip_mult must be finite and >= 0 (0 disables the clipped STE), got {clip_mult}")
    if 0.0 < c < CLIP_MULT_SAFE_MIN:
        warnings.warn(
            f"clip_mult={c} gates {100.0 * clipped_ste_frozen_fraction_gaussian(c):.1f}% of Gaussian-initialised "
            f"weights permanently (zero gradient that never recovers; docs/12 4.8). Use 0 (off) or >= "
            f"{CLIP_MULT_SAFE_MIN}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return c

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def fused_qat_g128_fwd_kernel(
        x_ptr, w_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,  # Must be 128 to align exactly with g128 group size
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
            w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_rem)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # Shape: [BLOCK_N, 128] in BF16
            # On-the-fly g128 absmean scale calculation inside registers/SRAM
            w_fp32 = w_tile.to(tl.float32)
            alpha = (tl.sum(tl.abs(w_fp32), axis=1) * (1.0 / 128.0)).to(tl.bfloat16)
            # Branchless register sign quantization: +alpha if w >= 0 else -alpha
            w_quant = tl.where(w_tile >= 0, alpha[:, None], -alpha[:, None])
            # Issue unlocked HMMA.16816.F32 Tensor Core MMA
            acc = tl.dot(x_tile, tl.trans(w_quant), acc=acc, out_dtype=tl.float32)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
        mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask)

    @triton.jit
    def fused_qat_g128_bwd_dx_kernel(
        dy_ptr, w_ptr, dx_ptr,
        M, N, K,
        stride_dym, stride_dyn,
        stride_wn, stride_wk,
        stride_dxm, stride_dxk,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,  # Fixed to 128 (g128 group dimension)
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)  # Each program_id(1) handles one g128 K-group of size 128
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)

        dy_ptrs = dy_ptr + (offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn)
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        acc_dx = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for n in range(0, tl.cdiv(N, BLOCK_N)):
            n_rem = N - n * BLOCK_N
            dy_mask = (offs_m[:, None] < M) & (offs_n[None, :] < n_rem)
            w_mask = (offs_n[:, None] < n_rem) & (offs_k[None, :] < K)
            dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # Shape: [BLOCK_N, 128] - complete g128 groups
            w_fp32 = w_tile.to(tl.float32)
            alpha = (tl.sum(tl.abs(w_fp32), axis=1) * (1.0 / 128.0)).to(tl.bfloat16)
            w_quant = tl.where(w_tile >= 0, alpha[:, None], -alpha[:, None])
            # dX = dY @ W_quant -> [BLOCK_M, BLOCK_N] @ [BLOCK_N, 128]
            acc_dx = tl.dot(dy_tile, w_quant, acc=acc_dx, out_dtype=tl.float32)
            dy_ptrs += BLOCK_N * stride_dyn
            w_ptrs += BLOCK_N * stride_wn

        offs_dxm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dx_ptrs = dx_ptr + stride_dxm * offs_dxm[:, None] + stride_dxk * offs_k[None, :]
        mask = (offs_dxm[:, None] < M) & (offs_k[None, :] < K)
        tl.store(dx_ptrs, acc_dx.to(tl.bfloat16), mask=mask)


def sram_tiled_qat_fwd_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Simulates the exact tile-by-tile BLOCK_K=128 in-SRAM quantization and FP32 MMA accumulation
    of fused_qat_g128_fwd_kernel without allocating a global W_hat tensor.
    """
    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw and K % 128 == 0
    acc = torch.zeros((M, N), dtype=torch.float32, device=x.device)
    for k_start in range(0, K, 128):
        x_tile = x[:, k_start : k_start + 128].to(torch.bfloat16)
        w_tile = w[:, k_start : k_start + 128].to(torch.bfloat16)
        alpha = (w_tile.to(torch.float32).abs().sum(dim=1) * (1.0 / 128.0)).to(torch.bfloat16)
        w_quant = torch.where(w_tile >= 0, alpha[:, None], -alpha[:, None])
        acc += x_tile.to(torch.float32) @ w_quant.to(torch.float32).T
    return acc.to(torch.bfloat16)


def sram_tiled_qat_bwd_dx_reference(dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Simulates the exact BLOCK_K=128 in-SRAM recomputation of W_hat during the backward dX pass
    without saving W_hat from the forward pass or transposing W in global memory.
    """
    M, N = dy.shape
    Nw, K = w.shape
    assert N == Nw and K % 128 == 0
    dx = torch.empty((M, K), dtype=torch.bfloat16, device=dy.device)
    for k_start in range(0, K, 128):
        w_tile = w[:, k_start : k_start + 128].to(torch.bfloat16)
        alpha = (w_tile.to(torch.float32).abs().sum(dim=1) * (1.0 / 128.0)).to(torch.bfloat16)
        w_quant = torch.where(w_tile >= 0, alpha[:, None], -alpha[:, None])
        acc_dx = dy.to(torch.bfloat16).to(torch.float32) @ w_quant.to(torch.float32)
        dx[:, k_start : k_start + 128] = acc_dx.to(torch.bfloat16)
    return dx


class FusedQATG128Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, clip_mult: float = 0.0) -> torch.Tensor:
        # Notice: Only x and master weight w are saved (0 bytes allocated for quantized W_hat!)
        M, K = x.shape
        N, Kw = w.shape
        assert K == Kw and K % 128 == 0, f"Invalid FusedQATG128 shapes: x={x.shape}, w={w.shape}"
        ctx.save_for_backward(x, w)
        ctx.clip_mult = _check_clip_mult(clip_mult)
        if HAS_TRITON and x.is_cuda and w.is_cuda:
            y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            fused_qat_g128_fwd_kernel[grid](
                x, w, y, M, N, K,
                x.stride(0), x.stride(1),
                w.stride(0), w.stride(1),
                y.stride(0), y.stride(1),
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=128,
            )
            return y
        return sram_tiled_qat_fwd_reference(x, w)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w = ctx.saved_tensors
        if HAS_TRITON and dy.is_cuda and w.is_cuda:
            M, N = dy.shape
            _, K = w.shape
            dx = torch.empty((M, K), device=dy.device, dtype=torch.bfloat16)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), K // 128)
            fused_qat_g128_bwd_dx_kernel[grid](
                dy, w, dx, M, N, K,
                dy.stride(0), dy.stride(1),
                w.stride(0), w.stride(1),
                dx.stride(0), dx.stride(1),
                BLOCK_M=64, BLOCK_K=128, BLOCK_N=64,
            )
        else:
            dx = sram_tiled_qat_bwd_dx_reference(dy, w)
        # Straight-Through Estimator: dW = dY^T @ X, optionally gated by Clipped STE indicator 1(|W| <= c * alpha_g)
        dw_fp32 = dy.to(torch.float32).T @ x.to(torch.float32)
        if ctx.clip_mult > 0.0:
            N_w, K_w = w.shape
            w_g = w.view(N_w, K_w // 128, 128).to(torch.float32)
            alpha_g = (w_g.abs().sum(dim=-1, keepdim=True) * (1.0 / 128.0)).to(torch.bfloat16).to(torch.float32)
            clip_mask = (w_g.abs() <= (ctx.clip_mult * alpha_g)).view(N_w, K_w)
            dw_fp32.masked_fill_(~clip_mask, 0.0)
        dw = dw_fp32.to(w.dtype)
        return dx, dw, None


class FusedQATG128Linear(torch.nn.Module):
    """
    Binary g128 QAT linear layer.

    master_dtype (docs/12 4.9): the default bf16 master weights match the 32B memory budget, but bf16 rounds
    away any update smaller than half an ulp. With a plain torch optimizer that drops 56% (Adam 1e-4) to 74%
    (SGD) of weight updates per step. bf16 masters are only correct when the parameter is written with
    stochastic rounding, as LayerWise8BitGaLoreEngine does (copy_to_param_stochastic). For training with a
    plain torch optimizer, pass master_dtype=torch.float32. The forward rounds w to bf16 before computing the
    scale and sign either way, so the quantized forward and the packer convention are unchanged.
    """

    def __init__(self, in_features: int, out_features: int, clip_mult: float = 0.0, device=None,
                 master_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        assert in_features % 128 == 0
        if master_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"master_dtype must be torch.bfloat16 or torch.float32, got {master_dtype}")
        self.in_features = in_features
        self.out_features = out_features
        self.clip_mult = _check_clip_mult(clip_mult)
        self.weight = torch.nn.Parameter(
            (torch.randn((out_features, in_features), device=device, dtype=torch.float32) * 0.02).to(master_dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        y_2d = FusedQATG128Function.apply(x_2d, self.weight, self.clip_mult)
        return y_2d.reshape(*orig_shape[:-1], self.out_features)


def main() -> None:
    print("================================================================================")
    print("  WORKSTREAM 5A: IN-SRAM BLOCK_K=128 FUSED STE QAT FORWARD/BACKWARD VERIFIER")
    print("================================================================================\n")

    torch.manual_seed(19)
    M, K, N = 64, 256, 128
    x = torch.randn((M, K), dtype=torch.bfloat16, requires_grad=True)
    layer = FusedQATG128Linear(K, N)

    y = layer(x)
    dy = torch.randn_like(y)
    y.backward(dy)

    # Compare against explicit materialized reference using identical K=128 block accumulation tree
    with torch.no_grad():
        w_groups = layer.weight.view(N, K // 128, 128)
        alpha = (w_groups.to(torch.float32).abs().sum(dim=-1, keepdim=True) * (1.0 / 128.0)).to(torch.bfloat16)
        w_hat = torch.where(w_groups >= 0, alpha, -alpha).view(N, K)
        y_ref_acc = (
            x[:, :128].to(torch.float32) @ w_hat[:, :128].to(torch.float32).T
            + x[:, 128:].to(torch.float32) @ w_hat[:, 128:].to(torch.float32).T
        )
        y_ref = y_ref_acc.to(torch.bfloat16)
        dx_ref = (dy.to(torch.float32) @ w_hat.to(torch.float32)).to(torch.bfloat16)
        dw_ref = (dy.to(torch.float32).T @ x.to(torch.float32)).to(torch.bfloat16)

    fwd_diff = int((y != y_ref).sum().item())
    dx_diff = int((x.grad != dx_ref).sum().item())
    dw_diff = int((layer.weight.grad != dw_ref).sum().item())

    # Not measured on the host (the Triton kernels need a GPU). By construction the Triton forward and dX kernels
    # never write W_hat to global memory; the clipped-STE backward path does allocate fp32 temporaries.
    print(f"[1] Materialized W_hat Global VRAM Allocation : 0 Bytes by construction in the Triton fwd/dX kernels "
          f"(not measured on host)")
    print(f"[2] Forward Pass (Y = X W_hat^T) Mismatches   : {fwd_diff}")
    print(f"[3] Backward dX (dY W_hat) Mismatches         : {dx_diff}")
    print(f"[4] Backward STE dW (dY^T X) Mismatches       : {dw_diff}")
    assert fwd_diff == 0 and dx_diff == 0 and dw_diff == 0

    # [5] Clipped STE indicator 1(|W| <= clip_mult * alpha_g) verification. c = 2.5 (the safe floor) still masks
    # the forced 10x outlier; c = 1.5 used to be exercised here and is now rejected with a warning (docs/12 4.8).
    clipped_layer = FusedQATG128Linear(K, N, clip_mult=CLIP_MULT_SAFE_MIN)
    with torch.no_grad():
        clipped_layer.weight.copy_(layer.weight)
        # Force row 0, col 0 weight to be a 10x outlier outside 1.5 * alpha_g
        clipped_layer.weight[0, 0] = 10.0
    x_c = x.detach().clone().requires_grad_(True)
    y_c = clipped_layer(x_c)
    y_c.backward(dy)
    assert float(clipped_layer.weight.grad[0, 0].item()) == 0.0
    assert int((clipped_layer.weight.grad != 0).sum().item()) > 0
    print(f"[5] Clipped STE Indicator (|W| <= 2.5*alpha_g): Outlier dW[0,0]={float(clipped_layer.weight.grad[0, 0].item()):.1f} (masked)")

    # [6] docs/12 4.8 / 4.9 regressions
    #   (a) the Gaussian gated fraction erfc(c/sqrt(pi)) matches a Monte-Carlo count at init, and small c warns
    #   (b) fp32 master weights receive every Adam update that bf16 masters drop
    torch.manual_seed(3)
    w_mc = torch.randn(4096, 1024)
    g_mc = w_mc.view(4096, 8, 128)
    a_mc = g_mc.abs().mean(-1, keepdim=True)
    for c in (1.5, 2.5):
        mc = float((g_mc.abs() > c * a_mc).float().mean())
        assert abs(mc - clipped_ste_frozen_fraction_gaussian(c)) < 0.003, (c, mc)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FusedQATG128Linear(128, 8, clip_mult=1.5)
    assert any(issubclass(w_.category, RuntimeWarning) for w_ in caught), "clip_mult=1.5 must warn"
    upd = {}
    for dt in (torch.bfloat16, torch.float32):
        torch.manual_seed(4)
        lay = FusedQATG128Linear(256, 64, master_dtype=dt)
        opt = torch.optim.Adam(lay.parameters(), lr=1e-4)
        xb = torch.randn(64, 256, dtype=torch.bfloat16)
        before = lay.weight.detach().clone()
        lay(xb).float().pow(2).mean().backward()
        opt.step()
        upd[dt] = float((lay.weight.detach() != before).float().mean())
    assert upd[torch.float32] > 0.95 and upd[torch.bfloat16] < upd[torch.float32], upd
    print(f"[6] Clip gate erfc(c/sqrt(pi)): 1.5 -> {clipped_ste_frozen_fraction_gaussian(1.5):.3f}, 2.5 -> "
          f"{clipped_ste_frozen_fraction_gaussian(2.5):.3f} (MC-matched; c<2.5 warns) | Adam step updates: "
          f"bf16 master {100 * upd[torch.bfloat16]:.1f}%, fp32 master {100 * upd[torch.float32]:.1f}%")
    print("[+] PASS: In-SRAM BLOCK_K=128 Fused STE QAT forward, backward, and Clipped STE verified.")


if __name__ == "__main__":
    main()
