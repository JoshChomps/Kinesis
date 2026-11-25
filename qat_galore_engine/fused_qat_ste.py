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


_TRITON_STAGE_CACHE = {}


def select_triton_num_stages(
    device: torch.device,
    w_dtype: torch.dtype,
    block_m: int,
    block_n: int,
    block_k: int,
    is_bwd_dx: bool = False,
    max_stages: int = 3,
    smem_limit_override: int = None,
) -> int:
    """
    Picks the largest Triton software-pipelining stage count (1..max_stages) whose staged input/weight
    shared memory fits within the active CUDA device's opt-in per-block SRAM ceiling.
    """
    if smem_limit_override is None:
        if not isinstance(device, torch.device) or device.type != "cuda":
            return 2
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        cache_key = (dev_idx, w_dtype, block_m, block_n, block_k, is_bwd_dx, max_stages)
        cached = _TRITON_STAGE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        props = torch.cuda.get_device_properties(dev_idx)
        smem_limit = int(
            getattr(props, "shared_memory_per_block_optin", 0)
            or getattr(props, "shared_memory_per_block", 49152)
        )
    else:
        cache_key = None
        smem_limit = int(smem_limit_override)

    w_bytes = 2 if w_dtype in (torch.bfloat16, torch.float16) else 4
    if is_bwd_dx:
        # dy_tile [BLOCK_M, BLOCK_N] in BF16 + w_tile [BLOCK_N, BLOCK_K] in w.dtype
        stage_bytes = (block_m * block_n * 2) + (block_n * block_k * w_bytes)
    else:
        # x_tile [BLOCK_M, BLOCK_K] in BF16 + w_tile [BLOCK_N, BLOCK_K] in w.dtype
        stage_bytes = (block_m * block_k * 2) + (block_n * block_k * w_bytes)

    chosen = 1
    for stages in range(max_stages, 0, -1):
        if stages * stage_bytes <= smem_limit:
            chosen = stages
            break

    if cache_key is not None:
        _TRITON_STAGE_CACHE[cache_key] = chosen
    return chosen


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
        IS_TERNARY: tl.constexpr = False,
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
            # .to(bfloat16) is a no-op for BF16 inputs. For FP32 masters (master_dtype=float32, docs/12 4.25) it is
            # the same rounding the host reference applies; without it the kernel took alpha and the sign from the
            # unrounded FP32 weight and fed tl.dot mixed dtypes. Both launches query select_triton_num_stages() so
            # pipeline depth adapts to the GPU's per-block SRAM ceiling (e.g. 3 stages for BF16, 2 for FP32 at 99 KB).
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.bfloat16)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.bfloat16)  # Shape: [BLOCK_N, 128] in BF16
            # On-the-fly g128 absmean scale calculation inside registers/SRAM
            w_fp32 = w_tile.to(tl.float32)
            if IS_TERNARY:
                alpha = tl.maximum(tl.sum(tl.abs(w_fp32), axis=1) * (1.0 / 128.0), 1e-6).to(tl.bfloat16)
                ratio = w_fp32 / alpha.to(tl.float32)[:, None]
                trit = tl.where(ratio > 0.5, 1.0, tl.where(ratio < -0.5, -1.0, 0.0))
                w_quant = (trit * alpha.to(tl.float32)[:, None]).to(tl.bfloat16)
            else:
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
        IS_TERNARY: tl.constexpr = False,
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
            dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0).to(tl.bfloat16)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.bfloat16)  # [BLOCK_N, 128] - complete g128 groups
            w_fp32 = w_tile.to(tl.float32)
            if IS_TERNARY:
                alpha = tl.maximum(tl.sum(tl.abs(w_fp32), axis=1) * (1.0 / 128.0), 1e-6).to(tl.bfloat16)
                ratio = w_fp32 / alpha.to(tl.float32)[:, None]
                trit = tl.where(ratio > 0.5, 1.0, tl.where(ratio < -0.5, -1.0, 0.0))
                w_quant = (trit * alpha.to(tl.float32)[:, None]).to(tl.bfloat16)
            else:
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

    @triton.jit
    def fused_qat_g128_bwd_dw_kernel(
        dy_ptr, x_ptr, w_ptr, dw_ptr,
        M, N, K,
        stride_dym, stride_dyn,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_dwn, stride_dwk,
        clip_mult,
        USE_CLIP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)

        dy_ptrs = dy_ptr + (offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for m in range(0, tl.cdiv(M, BLOCK_M)):
            m_rem = M - m * BLOCK_M
            dy_mask = (offs_m[:, None] < m_rem) & (offs_n[None, :] < N)
            x_mask = (offs_m[:, None] < m_rem) & (offs_k[None, :] < K)
            dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0).to(tl.bfloat16)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.bfloat16)
            acc = tl.dot(tl.trans(dy_tile), x_tile, acc=acc, out_dtype=tl.float32)
            dy_ptrs += BLOCK_M * stride_dym
            x_ptrs += BLOCK_M * stride_xm

        if USE_CLIP:
            w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
            w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.bfloat16)
            w_abs = tl.abs(w_tile.to(tl.float32))
            alpha_g = (tl.sum(w_abs, axis=1) * (1.0 / 128.0)).to(tl.bfloat16).to(tl.float32)
            acc = tl.where(w_abs <= clip_mult * alpha_g[:, None], acc, 0.0)

        dw_ptrs = dw_ptr + (offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk)
        dw_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.store(dw_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=dw_mask)


def _quant_tile_bf16(w_tile: torch.Tensor, fmt: str = "w1") -> torch.Tensor:
    w_fp32 = w_tile.to(torch.float32)
    mean_abs = w_fp32.abs().sum(dim=1) * (1.0 / 128.0)
    if fmt == "w2":
        alpha = mean_abs.clamp(min=1e-6).to(torch.bfloat16)
        ratio = w_fp32 / alpha.to(torch.float32)[:, None]
        trit = torch.where(ratio > 0.5, 1.0, torch.where(ratio < -0.5, -1.0, 0.0))
        return (trit * alpha.to(torch.float32)[:, None]).to(torch.bfloat16)
    alpha = mean_abs.to(torch.bfloat16)
    return torch.where(w_tile >= 0, alpha[:, None], -alpha[:, None])


def sram_tiled_qat_fwd_reference(x: torch.Tensor, w: torch.Tensor, fmt: str = "w1") -> torch.Tensor:
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
        w_quant = _quant_tile_bf16(w_tile, fmt=fmt)
        acc += x_tile.to(torch.float32) @ w_quant.to(torch.float32).T
    return acc.to(torch.bfloat16)


def sram_tiled_qat_bwd_dx_reference(dy: torch.Tensor, w: torch.Tensor, fmt: str = "w1") -> torch.Tensor:
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
        w_quant = _quant_tile_bf16(w_tile, fmt=fmt)
        acc_dx = dy.to(torch.bfloat16).to(torch.float32) @ w_quant.to(torch.float32)
        dx[:, k_start : k_start + 128] = acc_dx.to(torch.bfloat16)
    return dx


class FusedQATG128Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, clip_mult: float = 0.0, fmt: str = "w1") -> torch.Tensor:
        # Notice: Only x and master weight w are saved (0 bytes allocated for quantized W_hat!)
        M, K = x.shape
        N, Kw = w.shape
        assert K == Kw and K % 128 == 0, f"Invalid FusedQATG128 shapes: x={x.shape}, w={w.shape}"
        if fmt not in ("w1", "w2"):
            raise ValueError(f"fmt must be 'w1' or 'w2', got {fmt!r}")
        ctx.save_for_backward(x, w)
        ctx.clip_mult = _check_clip_mult(clip_mult)
        ctx.fmt = fmt
        if HAS_TRITON and x.is_cuda and w.is_cuda:
            y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            stages = select_triton_num_stages(x.device, w.dtype, 64, 64, 128, is_bwd_dx=False)
            fused_qat_g128_fwd_kernel[grid](
                x, w, y, M, N, K,
                x.stride(0), x.stride(1),
                w.stride(0), w.stride(1),
                y.stride(0), y.stride(1),
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, IS_TERNARY=(fmt == "w2"), num_stages=stages,
            )
            return y
        return sram_tiled_qat_fwd_reference(x, w, fmt=fmt)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w = ctx.saved_tensors
        fmt = ctx.fmt
        if HAS_TRITON and dy.is_cuda and w.is_cuda:
            M, N = dy.shape
            _, K = w.shape
            dx = torch.empty((M, K), device=dy.device, dtype=torch.bfloat16)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), K // 128)
            stages = select_triton_num_stages(dy.device, w.dtype, 64, 64, 128, is_bwd_dx=True)
            fused_qat_g128_bwd_dx_kernel[grid](
                dy, w, dx, M, N, K,
                dy.stride(0), dy.stride(1),
                w.stride(0), w.stride(1),
                dx.stride(0), dx.stride(1),
                BLOCK_M=64, BLOCK_K=128, BLOCK_N=64, IS_TERNARY=(fmt == "w2"), num_stages=stages,
            )
        else:
            dx = sram_tiled_qat_bwd_dx_reference(dy, w, fmt=fmt)
        # Straight-Through Estimator: dW = dY^T @ X, optionally gated by Clipped STE indicator 1(|W| <= c * alpha_g)
        N_w, K_w = w.shape
        if HAS_TRITON and dy.is_cuda and x.is_cuda and w.is_cuda:
            M_dy, _ = dy.shape
            dw = torch.empty((N_w, K_w), dtype=w.dtype, device=w.device)
            grid_dw = lambda meta: (triton.cdiv(N_w, meta['BLOCK_N']), K_w // 128)
            stages_dw = select_triton_num_stages(dy.device, torch.bfloat16, 64, 64, 128, is_bwd_dx=True)
            fused_qat_g128_bwd_dw_kernel[grid_dw](
                dy, x, w, dw,
                M_dy, N_w, K_w,
                dy.stride(0), dy.stride(1),
                x.stride(0), x.stride(1),
                w.stride(0), w.stride(1),
                dw.stride(0), dw.stride(1),
                float(ctx.clip_mult),
                USE_CLIP=(ctx.clip_mult > 0.0),
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_stages=stages_dw,
            )
            return dx, dw, None, None
        # Tiled by BLOCK_K = 128 fallback when Triton / CUDA is unavailable.
        dy_t = dy.to(torch.float32).T.contiguous()
        dw = torch.empty((N_w, K_w), dtype=w.dtype, device=w.device)
        for k0 in range(0, K_w, 128):
            blk = dy_t @ x[:, k0:k0 + 128].to(torch.float32)
            if ctx.clip_mult > 0.0:
                w_abs = w[:, k0:k0 + 128].to(torch.bfloat16).to(torch.float32).abs()
                alpha_g = (w_abs.sum(dim=-1, keepdim=True) * (1.0 / 128.0)).to(torch.bfloat16).to(torch.float32)
                blk.masked_fill_(~(w_abs <= ctx.clip_mult * alpha_g), 0.0)
            dw[:, k0:k0 + 128] = blk.to(w.dtype)
        return dx, dw, None, None


class FusedQATG128Linear(torch.nn.Module):
    """
    Binary g128 QAT linear layer.

    master_dtype (docs/12 4.9): the default bf16 master weights match the 32B memory budget, but bf16 rounds
    away any update smaller than half an ulp. With a plain torch optimizer that drops 56% (Adam 1e-4) to 74%
    (SGD) of weight updates per step. bf16 masters are only correct when the parameter is written with
    stochastic rounding, as LayerWise8BitGaLoreEngine does (copy_to_param_stochastic). For training with a
    plain torch optimizer, pass master_dtype=torch.float32. The forward rounds w to bf16 before computing the
    scale and sign either way.

    Deployment (docs/12 4.25): the packer must be given packer_input_weight(), the BF16-rounded master. Packing
    the raw FP32 master is NOT equivalent: the packer takes its scale from the FP32 values, and 876 of 32,768
    groups (2.7% of weights) then unpack to a different scale than this layer trained with. With BF16 masters
    the two are the same tensor. The agreement between this layer's FP32 group sum and the packer's FP64 mean
    is measured (0 mismatches over 65,536 groups), not proven for every input.
    """

    def __init__(self, in_features: int, out_features: int, clip_mult: float = 0.0, device=None,
                 master_dtype: torch.dtype = torch.bfloat16, fmt: str = "w1") -> None:
        super().__init__()
        assert in_features % 128 == 0
        if master_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"master_dtype must be torch.bfloat16 or torch.float32, got {master_dtype}")
        if fmt not in ("w1", "w2"):
            raise ValueError(f"fmt must be 'w1' or 'w2', got {fmt!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.fmt = fmt
        self.clip_mult = _check_clip_mult(clip_mult)
        self.weight = torch.nn.Parameter(
            (torch.randn((out_features, in_features), device=device, dtype=torch.float32) * 0.02).to(master_dtype)
        )

    @torch.no_grad()
    def packer_input_weight(self) -> torch.Tensor:
        """FP32 copy of the BF16-rounded master: the tensor to hand to quantize_and_pack_w1/w2_interleaved."""
        return self.weight.detach().to(torch.bfloat16).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        y_2d = FusedQATG128Function.apply(x_2d, self.weight, self.clip_mult, self.fmt)
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
    # Scale: FP64 mean, rounded once to BF16 (docs/12 4.19) using the unified normal+subnormal ULP quantum
    # ulp_exp = max(e - 8, -133), identical to spinquant_packer.round_f64_to_bf16_rne and C++ f64_to_bf16_bits_rne.
    mean64 = g.to(torch.float64).abs().sum(dim=-1, keepdim=True) / 128.0
    if fmt == "w2":
        mean64 = mean64.clamp(min=1e-6)
    _, e = torch.frexp(mean64)
    ulp_exp = torch.clamp(e.to(torch.int32) - 8, min=-133)
    r64 = torch.ldexp(torch.round(torch.ldexp(mean64, -ulp_exp)), ulp_exp)
    r64 = torch.where(torch.isfinite(mean64), r64, mean64)
    s = r64.to(torch.float32).to(torch.bfloat16).to(torch.float32)
    if fmt == "w1":
        w_hat = torch.where(g >= 0, s, -s)
    elif fmt == "w2":
        # Non-finite ratio (docs/12 4.21): Inf / Inf = NaN would otherwise reach an integer cast in the packer.
        ratio = g / s
        trit = torch.where(torch.isfinite(ratio), torch.clamp(torch.round(ratio), -1.0, 1.0),
                           torch.sign(torch.nan_to_num(g, nan=0.0)))
        w_hat = trit * s
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
                l0 = float(loss.detach()) if l0 is None else l0
                loss.backward()
                opt.step()
            w_dep = pack[fmt][1](*pack[fmt][0](lay.weight.detach(), R1=R))
            w_mis = int((lay.deploy_weight() != w_dep).sum())
            # Deployed forward: the kernel's x R in BF16, times the unpacked weights, vs the trained layer.
            y_dep = ((x_t @ R).to(torch.bfloat16).float() @ w_dep.float().T).to(torch.bfloat16)
            with torch.no_grad():
                y_mis = int((lay(x_t) != y_dep).sum())
            zero_frac = float((w_dep == 0).float().mean())
            l1 = float(loss.detach())
            deploy_report.append((mode, fmt, init_mis, w_mis, y_mis, l0, l1, zero_frac))
            assert init_mis == 0 and w_mis == 0 and y_mis == 0, (mode, fmt, init_mis, w_mis, y_mis)
            assert l1 < 0.8 * l0, (mode, fmt, l0, l1)
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
    # [8] docs/12 4.25: an FP32-master layer deploys through packer_input_weight(). W_hat is read back from the
    #     layer itself (forward of the identity gives W_hat^T exactly), then compared with the packer's unpack.
    torch.manual_seed(31)
    lay32 = FusedQATG128Linear(256, 2048, master_dtype=torch.float32)
    with torch.no_grad():
        w_hat_layer = lay32(torch.eye(256, dtype=torch.bfloat16)).T.contiguous()
    w_ok = pack["w1"][1](*pack["w1"][0](lay32.packer_input_weight()))
    w_raw = pack["w1"][1](*pack["w1"][0](lay32.weight.detach()))
    mis_ok, mis_raw = int((w_hat_layer != w_ok).sum()), int((w_hat_layer != w_raw).sum())
    assert mis_ok == 0, mis_ok
    assert mis_raw > 0, "negative control: packing the raw FP32 master should disagree with the trained W_hat"
    # Clip gate and forward now share one alpha: the gate decision recomputed from the BF16-rounded master.
    layc = FusedQATG128Linear(256, 512, clip_mult=CLIP_MULT_SAFE_MIN, master_dtype=torch.float32)
    xg = torch.randn(64, 256, dtype=torch.bfloat16)
    layc(xg).float().sum().backward()
    wb = layc.weight.detach().to(torch.bfloat16).float().view(512, 2, 128)
    a_fwd = (wb.abs().sum(-1, keepdim=True) * (1.0 / 128.0)).to(torch.bfloat16).float()
    gate_ref = (wb.abs() <= CLIP_MULT_SAFE_MIN * a_fwd).view(512, 256)
    gate_got = layc.weight.grad != 0
    gate_mis = int((gate_got & ~gate_ref).sum())      # a nonzero gradient where the forward's alpha gates it off
    assert gate_mis == 0, gate_mis
    print(f"[8] FP32 master deploy (4.25): W_hat vs packer(packer_input_weight) {mis_ok} mismatches | raw FP32 master "
          f"{mis_raw} (control) | clip gate vs forward alpha {gate_mis}")

    # [9] GPU-aware Triton SRAM stage selection across hardware opt-in shared memory limits
    expected_stages = {
        49152:  (1, 2, 1, 1),   # 48 KB (legacy/Pascal)
        65536:  (2, 2, 1, 1),   # 64 KB (Turing sm_75)
        101376: (3, 3, 2, 2),   # 99 KB (Ampere sm_86 / Ada sm_89)
        167936: (3, 3, 3, 3),   # 164 KB (A100 sm_80)
        232448: (3, 3, 3, 3),   # 227 KB (H100 sm_90)
    }
    cpu_dev = torch.device("cpu")
    for smem_lim, (bf_f, bf_b, fp_f, fp_b) in expected_stages.items():
        got = (
            select_triton_num_stages(cpu_dev, torch.bfloat16, 64, 64, 128, is_bwd_dx=False, smem_limit_override=smem_lim),
            select_triton_num_stages(cpu_dev, torch.bfloat16, 64, 64, 128, is_bwd_dx=True, smem_limit_override=smem_lim),
            select_triton_num_stages(cpu_dev, torch.float32, 64, 64, 128, is_bwd_dx=False, smem_limit_override=smem_lim),
            select_triton_num_stages(cpu_dev, torch.float32, 64, 64, 128, is_bwd_dx=True, smem_limit_override=smem_lim),
        )
        assert got == (bf_f, bf_b, fp_f, fp_b), f"smem={smem_lim}: got {got}, expected {(bf_f, bf_b, fp_f, fp_b)}"
    print("[9] GPU SRAM stage selector (48/64/99/164/227 KB): verified adaptive num_stages for BF16 & FP32")

    # [10] Fused W2 Ternary (BitNet b1.58) STE QAT parity + >10% zero-weight verification
    torch.manual_seed(41)
    lay_w2 = FusedQATG128Linear(256, 128, clip_mult=CLIP_MULT_SAFE_MIN, master_dtype=torch.bfloat16, fmt="w2")
    x_w2 = torch.randn((32, 256), dtype=torch.bfloat16, requires_grad=True)
    y_w2 = lay_w2(x_w2)
    dy_w2 = torch.randn_like(y_w2)
    y_w2.backward(dy_w2)
    with torch.no_grad():
        w2_hat = lay_w2(torch.eye(256, dtype=torch.bfloat16)).T.contiguous()
        y_w2_ref = sram_tiled_qat_fwd_reference(x_w2, lay_w2.weight, fmt="w2")
        dx_w2_ref = sram_tiled_qat_bwd_dx_reference(dy_w2, lay_w2.weight, fmt="w2")
        zero_w2 = float((w2_hat == 0).float().mean())
    assert zero_w2 > 0.10, zero_w2
    assert int((y_w2 != y_w2_ref).sum().item()) == 0 and int((x_w2.grad != dx_w2_ref).sum().item()) == 0
    print(f"[10] Fused W2 Ternary STE (b1.58): 0 fwd/dx mismatches | ternary zeros={100.0 * zero_w2:.1f}% | PASS")
    print("[+] PASS: In-SRAM BLOCK_K=128 Fused STE QAT forward, backward, and Clipped STE verified.")


if __name__ == "__main__":
    main()
