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
        # Tiled by BLOCK_K = 128 (docs/12 4.10 follow-up): the FP32 dW block, the |W| block and the clip mask are
        # each N x 128, so the backward never holds an N x K FP32 temporary (it used to hold several, about 7x a
        # BF16 W_hat at 4096^2). Only the dW output itself (in the master dtype) is N x K.
        N_w, K_w = w.shape
        dy_t = dy.to(torch.float32).T.contiguous()
        dw = torch.empty((N_w, K_w), dtype=w.dtype, device=w.device)
        for k0 in range(0, K_w, 128):
            blk = dy_t @ x[:, k0:k0 + 128].to(torch.float32)
            if ctx.clip_mult > 0.0:
                w_abs = w[:, k0:k0 + 128].to(torch.float32).abs()
                alpha_g = (w_abs.sum(dim=-1, keepdim=True) * (1.0 / 128.0)).to(torch.bfloat16).to(torch.float32)
                blk.masked_fill_(~(w_abs <= ctx.clip_mult * alpha_g), 0.0)
            dw[:, k0:k0 + 128] = blk.to(w.dtype)
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


def quantize_g128_deploy_convention(w_rot: torch.Tensor, fmt: str) -> torch.Tensor:
    """
    The packer's quantization rule (spinquant_packer.py), written independently here so QAT trains exactly the
    weights that get deployed (docs/12 4.10). Input: FP32 weights already in the deploy basis, [N, K], K % 128 == 0.
      fmt "w1" (Binary g128):     s = bf16(mean|w|) per 128-group (FP64 mean, one rounding); W_hat = +-s.
      fmt "w2" (BitNet b1.58):    s = bf16(max(mean|w|, 1e-6)); q = clamp(round_half_even(w / s), -1, 1);
                                  W_hat = q * s (exact in BF16, since q is -1, 0 or +1).
    Returns BF16 W_hat. The scale is taken from the FP32 rotated weight, not from a BF16-rounded copy: that is
    what the packer does (the unrotated Triton path rounds to BF16 first, which is equivalent only for BF16 masters).
    """
    w = w_rot.to(torch.float32)
    N, K = w.shape
    assert K % 128 == 0, "K must be a multiple of 128"
    g = w.reshape(N, K // 128, 128)
    # Scale: FP64 mean, rounded once to BF16 (docs/12 4.19). Rounding is done by splitting the FP64 value into
    # its BF16 neighbours below and above and taking the nearer (the even one on a tie).
    mean64 = g.to(torch.float64).abs().sum(dim=-1, keepdim=True) / 128.0
    if fmt == "w2":
        mean64 = mean64.clamp(min=1e-6)
    lo = mean64.to(torch.float32).to(torch.bfloat16)        # a BF16 neighbour (may be above or below)
    lo = torch.where(lo.to(torch.float64) > mean64, torch.nextafter(lo, torch.zeros_like(lo)), lo)
    hi = torch.nextafter(lo, torch.full_like(lo, float("inf")))
    d_lo = mean64 - lo.to(torch.float64)
    d_hi = hi.to(torch.float64) - mean64
    lo_even = (lo.view(torch.int16) & 1) == 0
    pick_lo = (d_lo < d_hi) | ((d_lo == d_hi) & lo_even)
    s = torch.where(pick_lo, lo, hi).to(torch.float32)
    s = torch.where(mean64 == 0, torch.zeros_like(s), s)
    if fmt == "w1":
        w_hat = torch.where(g >= 0, s, -s)
    elif fmt == "w2":
        w_hat = torch.clamp(torch.round(g / s), -1.0, 1.0) * s
    else:
        raise ValueError(f"fmt must be 'w1' or 'w2', got {fmt!r}")
    return w_hat.reshape(N, K).to(torch.bfloat16)


class DeployBasisQATLinear(torch.nn.Module):
    """
    QAT linear layer trained in the deployment basis (docs/12 4.10).

    The deployed kernel computes y = (x R) W_hat(W R)^T, with R the online rotation (for example
    build_kernel_online_rotation(K, mode) from spinquant_packer.py) and W_hat the packer's W1 or W2 quantizer
    applied to the rotated weight. This layer runs exactly that forward, so the packed weights reproduce the
    trained layer bit for bit. The master weight W stays in the original basis (so R can still be absorbed into
    neighbouring layers), and the straight-through estimator is applied to W R:
        W_ste = W_hat.detach() + (W R - (W R).detach()),
    so autograd gives dW = (dy^T (x R)) R^T. This is a host reference path (plain autograd, it materializes
    W_hat); the fused Triton kernels above implement only the unrotated W1 case.
    """

    def __init__(self, in_features: int, out_features: int, fmt: str = "w1",
                 rotation: torch.Tensor = None, master_dtype: torch.dtype = torch.float32, device=None) -> None:
        super().__init__()
        assert in_features % 128 == 0
        if fmt not in ("w1", "w2"):
            raise ValueError(f"fmt must be 'w1' or 'w2', got {fmt!r}")
        self.fmt = fmt
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            (torch.randn((out_features, in_features), device=device, dtype=torch.float32) * 0.02).to(master_dtype)
        )
        if rotation is not None:
            assert rotation.shape == (in_features, in_features)
            self.register_buffer("rotation", rotation.to(torch.float32))
        else:
            self.rotation = None

    def rotated_weight(self) -> torch.Tensor:
        w = self.weight.to(torch.float32)
        return w @ self.rotation if self.rotation is not None else w

    @torch.no_grad()
    def deploy_weight(self) -> torch.Tensor:
        """The BF16 W_hat the packer would produce for this layer's current master weight."""
        return quantize_g128_deploy_convention(self.rotated_weight(), self.fmt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).to(torch.float32)
        if self.rotation is not None:
            x_2d = x_2d @ self.rotation
        x_2d = x_2d.to(torch.bfloat16).to(torch.float32)   # the kernel consumes BF16 rotated activations
        w_rot = self.rotated_weight()
        w_hat = quantize_g128_deploy_convention(w_rot.detach(), self.fmt).to(torch.float32)
        w_ste = w_hat + (w_rot - w_rot.detach())
        y = (x_2d @ w_ste.T).to(torch.bfloat16)
        return y.reshape(*orig_shape[:-1], self.out_features)


def main() -> None:
    print("================================================================================")
    print("  WORKSTREAM 5A: IN-SRAM BLOCK_K=128 FUSED STE QAT FORWARD/BACKWARD VERIFIER")
    print("================================================================================\n")

    torch.manual_seed(19)
    M, K, N = 64, 256, 128
    x = torch.randn((M, K), dtype=torch.bfloat16, requires_grad=True)
    layer = FusedQATG128Linear(K, N)

    y = layer(x)
    # Measured (docs/12 4.10 follow-up): what autograd keeps between forward and backward is exactly x and the
    # master w; no quantized W_hat tensor is saved.
    node = y.grad_fn
    while node is not None and not type(node).__name__.startswith("FusedQATG128Function"):
        node = node.next_functions[0][0]   # step through the reshape view to the custom Function node
    saved = node.saved_tensors
    saved_bytes = sum(t.numel() * t.element_size() for t in saved)
    expect_saved = x.numel() * x.element_size() + layer.weight.numel() * layer.weight.element_size()
    assert len(saved) == 2 and saved_bytes == expect_saved, (len(saved), saved_bytes, expect_saved)
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
    # dW reference by autograd through the canonical STE construction W_ste = W_hat.detach() + (W - W.detach()),
    # which is an independent implementation (docs/12 4.16: the old check compared dy^T x with dy^T x).
    w_leaf = layer.weight.detach().to(torch.float32).requires_grad_(True)
    w_ste = w_hat.to(torch.float32).detach() + (w_leaf - w_leaf.detach())
    (x.detach().to(torch.float32) @ w_ste.T).backward(dy.to(torch.float32))
    dw_ref = w_leaf.grad.to(torch.bfloat16)

    fwd_diff = int((y != y_ref).sum().item())
    dx_diff = int((x.grad != dx_ref).sum().item())
    dw_diff = int((layer.weight.grad != dw_ref).sum().item())

    # Saved-tensor bytes are measured above. The Triton kernels' VRAM is not measurable on this host; by
    # construction they never write W_hat to global memory, and the backward temporaries are N x 128 per block.
    print(f"[1] Saved for backward (measured)             : {saved_bytes} B = x + master w, 0 B of W_hat "
          f"(Triton VRAM not measured on host)")
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
    # Independent reference for the clipped path: autograd through W_hat.detach() + m * (W - W.detach()), with the
    # indicator m = 1(|w| <= c * alpha_g) built from a separately written group mean.
    with torch.no_grad():
        wc = clipped_layer.weight.to(torch.float32)
        alpha_c = wc.abs().reshape(N, K // 128, 128).mean(-1, keepdim=True).to(torch.bfloat16).to(torch.float32)
        m_c = (wc.abs().reshape(N, K // 128, 128) <= CLIP_MULT_SAFE_MIN * alpha_c).reshape(N, K).to(torch.float32)
    wl = wc.clone().requires_grad_(True)
    w_ste_c = wl.detach() + m_c * (wl - wl.detach())
    (x.detach().to(torch.float32) @ w_ste_c.T).backward(dy.to(torch.float32))
    clip_dw_diff = int((clipped_layer.weight.grad != wl.grad.to(torch.bfloat16)).sum().item())
    assert clip_dw_diff == 0, clip_dw_diff
    print(f"[5] Clipped STE Indicator (|W| <= 2.5*alpha_g): Outlier dW[0,0]={float(clipped_layer.weight.grad[0, 0].item()):.1f} (masked) | dW vs autograd-STE ref mismatches {clip_dw_diff}")

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

    # [7] docs/12 4.10: QAT in the deployment basis, for W1 and W2, checked against the PACKER's pack -> unpack
    #     (bit packing and the lop3/fma unpack emulation), with the kernel's exact online rotation R applied.
    import sys as _sys
    from pathlib import Path as _Path
    _pk = str(_Path(__file__).resolve().parent.parent / "fused_w1_hadacore")
    if _pk not in _sys.path:
        _sys.path.insert(0, _pk)
    from spinquant_packer import (  # noqa: E402
        build_kernel_online_rotation, quantize_and_pack_w1_g128_interleaved, unpack_w1_g128_lop3_reference,
        quantize_and_pack_w2_ternary_interleaved, unpack_w2_ternary_lop3_reference,
    )
    pack = {"w1": (quantize_and_pack_w1_g128_interleaved, unpack_w1_g128_lop3_reference),
            "w2": (quantize_and_pack_w2_ternary_interleaved, unpack_w2_ternary_lop3_reference)}
    Kd, Nd, Md = 256, 64, 128
    torch.manual_seed(23)
    x_t = torch.randn(Md, Kd)
    x_t[:, 7] *= 25.0                        # an outlier channel, the reason the rotation exists
    w_teacher = torch.randn(Nd, Kd) * 0.05
    y_t = x_t @ w_teacher.T
    deploy_report = []
    for mode in (1, 2):
        R = build_kernel_online_rotation(Kd, mode)
        for fmt in ("w1", "w2"):
            torch.manual_seed(29)
            lay = DeployBasisQATLinear(Kd, Nd, fmt=fmt, rotation=R)
            init_mis = int((lay.deploy_weight() != pack[fmt][1](*pack[fmt][0](lay.weight.detach(), R1=R))).sum())
            opt = torch.optim.Adam(lay.parameters(), lr=2e-3)
            l0 = None
            for _ in range(60):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(lay(x_t).float(), y_t)
                l0 = float(loss) if l0 is None else l0
                loss.backward()
                opt.step()
            w_dep = pack[fmt][1](*pack[fmt][0](lay.weight.detach(), R1=R))
            w_mis = int((lay.deploy_weight() != w_dep).sum())
            # Deployed forward: the kernel's x R in BF16, times the unpacked weights, vs the trained layer.
            y_dep = ((x_t @ R).to(torch.bfloat16).float() @ w_dep.float().T).to(torch.bfloat16)
            with torch.no_grad():
                y_mis = int((lay(x_t) != y_dep).sum())
            zero_frac = float((w_dep == 0).float().mean())
            deploy_report.append((mode, fmt, init_mis, w_mis, y_mis, l0, float(loss), zero_frac))
            assert init_mis == 0 and w_mis == 0 and y_mis == 0, (mode, fmt, init_mis, w_mis, y_mis)
            assert float(loss) < 0.8 * l0, (mode, fmt, l0, float(loss))
            if fmt == "w2":
                assert zero_frac > 0.1, zero_frac   # ternary zeros exist (binary W_hat has none)
    # Negative control: the unrotated binary QAT weights do not describe the rotated deploy (sign agreement ~ 0.5).
    w_plain = torch.randn(Nd, Kd)
    R1m = build_kernel_online_rotation(Kd, 1)
    agree = float(((w_plain >= 0) == ((w_plain @ R1m) >= 0)).float().mean())
    assert 0.4 < agree < 0.6, agree
    for mode, fmt, im, wm, ym, l0, l1, zf in deploy_report:
        print(f"[7] Deploy-basis QAT mode {mode} {fmt}: W_hat vs packer unpack {im}/{wm} mismatches (init/trained), "
              f"forward vs deployed {ym} | loss {l0:.4f} -> {l1:.4f} | zeros {100 * zf:.1f}%")
    print(f"    Negative control: unrotated vs rotated sign agreement {agree:.3f} (chance), so the old QAT basis "
          f"did not transfer")
    print("[+] PASS: In-SRAM BLOCK_K=128 Fused STE QAT forward, backward, and Clipped STE verified.")


if __name__ == "__main__":
    main()
