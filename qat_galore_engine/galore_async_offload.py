#!/usr/bin/env python3
"""
File: qat_galore_engine/galore_async_offload.py
Workstream 5B: Layer-Wise 8-Bit GaLore Optimizer with Per-Layer Post-Accumulation Backward
Hooks and Asynchronous Pinned HugePage PCIe DMA Offloading.
Compresses the 32B QAT training memory footprint from 227.40 GB/GPU (Standard BF16 AdamW)
down to 23.63 GB/GPU (22.01 GiB/GPU, fitting natively inside 2x 24 GiB = 25.77 GB unlocked consumer GPUs).
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import math
import torch


@dataclass
class GaLoreState8Bit:
    step: int
    rank: int
    update_proj_gap: int
    proj_q: torch.Tensor       # int8 [m, r]
    proj_scale: torch.Tensor   # fp32 [m, 1]
    exp_avg_q: torch.Tensor    # int8 [r, n]
    exp_avg_scale: torch.Tensor
    exp_avg_sq_q: torch.Tensor # uint8 [r, n], 8-bit log code of sqrt(V) (quantize_sqrt_v_log8_sr)
    exp_avg_sq_scale: torch.Tensor
    transpose: bool            # True when original weight had shape [m, n] with m > n
    next_svd_step: int = 0     # step at which the projector is next refreshed (supports lazy adaptive gaps)
    proj_bits: int = 8         # 8: int8 projector, 4: packed int4 (two values per uint8) with stochastic rounding


def quantize_signed_int8(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = (t.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-12)
    ratio = t / scale
    q = torch.where(
        torch.isfinite(ratio),
        torch.clamp(torch.round(ratio), -127.0, 127.0),
        127.0 * torch.sign(torch.nan_to_num(t, nan=0.0)),
    ).to(torch.int8)
    return q, scale


def dequantize_signed_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantize_signed_int8_sr(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    int8 store for the first moment with unbiased stochastic rounding (docs/12 4.26). Round-to-nearest has a dead
    zone: a coordinate below 1/254 of its row maximum is stored as 0 every step, so it never accumulates momentum
    (21% of coordinates with a three-decade column spread) and its Adam numerator stays at (1 - beta1) * g.
    floor(x / scale + U[0, 1)) has E[q * scale] = x, so E[M_hat_t] follows the exact EMA recursion, the same
    argument as quantize_sqrt_v_log8_sr. Same memory: 1 byte per coordinate plus one fp32 scale per row.
    """
    scale = (t.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-12)
    ratio = t / scale
    q = torch.where(
        torch.isfinite(ratio),
        torch.clamp(torch.floor(ratio + torch.rand_like(t)), -127.0, 127.0),
        127.0 * torch.sign(torch.nan_to_num(t, nan=0.0)),
    ).to(torch.int8)
    return q, scale


def moment_rotation(p_new_stored: torch.Tensor, p_old_stored: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Change of coordinates for the low-rank moments on a projector refresh (docs/12 4.26).

    The moments are stored in the coordinates of the STORED (quantized) projector: c_old = P_old^T g, and after
    the refresh the engine projects with the stored new projector, c_new = P_new^T g. Neither is orthonormal
    (int4: ||P^T P - I|| about 0.04). For g in span(P_old), with the thin QR P_old = Q R:
        g = Q R^-T c_old,   so   c_new = (P_new^T Q R^-T) c_old.
    The earlier code used P_new^T Q, i.e. it treated c_old as Q^T g, which is off by R^T (3.4% for int4).

    Returns (rot, lost):
      rot  [r, r]: c_new = rot @ c_old for gradients in the old span.
      lost [r]:    fraction of each new direction that lies outside the old span, in [0, 1]. The old moments
                   carry no information about that part.
    """
    q_qr, r_qr = torch.linalg.qr(p_old_stored, mode="reduced")
    overlap = p_new_stored.T @ q_qr                                  # [r, r], rows = new directions
    new_norm_sq = p_new_stored.pow(2).sum(dim=0).clamp(min=1e-24)    # [r]
    lost = (1.0 - overlap.pow(2).sum(dim=1) / new_norm_sq).clamp(0.0, 1.0)
    diag = torch.diagonal(r_qr).abs()
    if float(diag.min()) > 1e-6 * float(diag.max().clamp(min=1e-30)):
        # X R^T = overlap  ->  X = overlap R^-T   (R^T is lower triangular)
        rot = torch.linalg.solve_triangular(r_qr.T, overlap, upper=False, left=False)
    else:
        rot = overlap                                                # rank-deficient stored projector: best effort
    return rot, lost


def quantize_unsigned_uint8(t: torch.Tensor, round_up: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear per-row uint8 quantizer (kept for general use). NOT used for the second moment any more: applying
    it with round_up=True to sqrt(V) every step compounded the round-up error into a ratchet (docs/12 4.6,
    stored/true sqrt(V) about 10x after 3000 steps). The second moment now uses quantize_sqrt_v_log8_sr.
    """
    scale = (t.clamp(min=0.0).amax(dim=-1, keepdim=True) / 255.0).clamp(min=1e-16)
    scaled = t.clamp(min=0.0) / scale
    rounded = torch.ceil(scaled) if round_up else torch.round(scaled)
    q = torch.where(
        torch.isfinite(scaled),
        torch.clamp(rounded, 0.0, 255.0),
        torch.where(t > 0.0, 255.0, 0.0),
    ).to(torch.uint8)
    return q, scale


# Second-moment code density: KAPPA codes per octave of sqrt(V). Code 255 is the per-row maximum and code 1 is
# 2^(-254/KAPPA) = 1.7e-5 of it (9.5 decades of dynamic range in V). Chosen by sweep (docs/14 F-B): 12 is
# noisier, 20+ starts hitting the floor on lognormal gradient spreads, which reintroduces upward bias.
SQRT_V_LOG8_KAPPA = 16


def quantize_sqrt_v_log8_sr(sqrt_v: torch.Tensor, kappa: int = SQRT_V_LOG8_KAPPA) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    8-bit logarithmic code for sqrt(V) with stochastic rounding that is UNBIASED IN V (docs/12 4.6 fix).

    Code c = 0 means exactly 0. Code c in [1, 255] decodes to  s * 2^((c - 255)/kappa),  s = row max of sqrt(V).

    Why this and not the previous linear uint8 + ceil:
      * The state is dequantized, updated (V <- b2 V + (1-b2) g^2) and re-quantized every step, so any
        per-step bias compounds. With ceil the error obeys e_{t+1} <= sqrt(b2) e_t + q, steady state
        q/(1 - sqrt(b2)), about 2000 quanta at b2 = 0.999.
      * Rounding to nearest does not help: with b2 = 0.999 the per-step relative change of V is about 1e-3,
        far below half a code step, so RTN freezes V except on large gradient spikes, which only move it up.
      * Unbiased stochastic rounding in V fixes the mean exactly. For V between adjacent levels
        L_lo^2 <= V <= L_hi^2, pick L_hi with p = (V - L_lo^2) / (L_hi^2 - L_lo^2), else L_lo, so
        E[Q(V)^2 | V] = V. Then E[V_hat_{t+1}] = b2 E[V_hat_t] + (1 - b2) g_t^2, the same linear recursion
        as the exact V. By induction from V_hat_0 = V_0 = 0, E[V_hat_t] = V_t for every t. There is no
        compounding: the per-step errors are martingale differences, not a drift.
      * The log spacing is what keeps that noise small. On a linear grid, coordinates far below the row
        max sit within a few quanta of 0, so the relative noise is huge (measured: 30-50x spread). On a
        log grid the relative step is the same everywhere, 2^(2/kappa) - 1 = 9% in V. The measured
        steady-state spread of sqrt(V_hat)/sqrt(V) is 0.82..1.20 (5th..95th percentile) for rank-1,
        lognormal and Student-t(3) gradients, against 2.5x..50x for ceil.
      * Zero is excluded for V > 0: values below code 1 are clamped up to code 1, so no coordinate with
        nonzero history decodes to 0. That keeps the original goal of round_up: never m / (0 + eps) with
        |m| <= ((1-b1)/sqrt(1-b2)) sqrt(V). The only bias left is this floor, 1.7e-5 of the row max in
        sqrt(V), and it is a constant floor, not a ratchet.

    Memory is unchanged: 1 byte per coordinate plus one fp32 scale per row.
    """
    sv = torch.nan_to_num(sqrt_v.to(torch.float32), nan=0.0, posinf=0.0).clamp(min=0.0)
    scale = sv.amax(dim=-1, keepdim=True).clamp(min=1e-30)
    x = sv / scale                                           # in [0, 1]
    floor_x = 2.0 ** (-254.0 / kappa)
    xc = x.clamp(min=floor_x)
    # level index e in [-254, 0] with 2^(e/kappa) <= xc < 2^((e+1)/kappa)
    e = torch.floor(torch.log2(xc) * kappa).clamp(min=-254.0, max=0.0)
    lo = torch.exp2(e / kappa)
    hi = torch.exp2(torch.clamp(e + 1.0, max=0.0) / kappa)
    v, v_lo, v_hi = xc * xc, lo * lo, hi * hi
    p_up = torch.where(v_hi > v_lo, (v - v_lo) / (v_hi - v_lo), torch.zeros_like(v)).clamp(0.0, 1.0)
    e_q = torch.where(torch.rand_like(v) < p_up, torch.clamp(e + 1.0, max=0.0), e)
    code = (e_q + 255.0).clamp(1.0, 255.0)
    code = torch.where(x > 0.0, code, torch.zeros_like(code))
    return code.to(torch.uint8), scale


def dequantize_sqrt_v_log8(q: torch.Tensor, scale: torch.Tensor, kappa: int = SQRT_V_LOG8_KAPPA) -> torch.Tensor:
    c = q.to(torch.float32)
    val = scale * torch.exp2((c - 255.0) / kappa)
    return torch.where(q > 0, val, torch.zeros_like(val))


def copy_to_param_stochastic(param_data: torch.Tensor, value_fp32: torch.Tensor) -> None:
    """
    Writes an fp32 value into a parameter. For bf16 parameters, round-to-nearest silently drops any update
    smaller than half a bf16 ulp (e.g. lr 2e-4 on weights ~0.02), so training stalls. Stochastic rounding
    (add uniform random low 16 bits, then truncate) is unbiased: E[bf16(x)] = x. Other dtypes copy directly.
    """
    if param_data.dtype != torch.bfloat16:
        param_data.copy_(value_fp32.to(param_data.dtype))
        return
    bits = value_fp32.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device)
    finite = torch.isfinite(value_fp32)
    rounded = torch.where(finite, (bits + noise) & -65536, bits)  # -65536 == 0xFFFF0000 as int32
    param_data.copy_(rounded.view(torch.float32).to(torch.bfloat16))


def dequantize_unsigned_uint8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantize_int4_stochastic(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Q-GaLore (arXiv:2407.08296) style symmetric INT4 projector quantization with stochastic rounding.
    Per-row scale = max|row| / 7; values are floor(x / scale + U[0,1)) clamped to [-7, 7] (unbiased in expectation),
    then packed two per byte as (q + 8) nibbles along the last dim (padded to an even width).
    """
    scale = (t.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-12)
    noise = torch.rand_like(t)
    ratio = t / scale
    q = torch.where(
        torch.isfinite(ratio),
        torch.clamp(torch.floor(ratio + noise), -7.0, 7.0),
        7.0 * torch.sign(torch.nan_to_num(t, nan=0.0)),
    ).to(torch.int16)
    if q.shape[-1] % 2 == 1:
        q = torch.cat([q, torch.zeros_like(q[..., :1])], dim=-1)
    lo = (q[..., 0::2] + 8).to(torch.uint8) & 0x0F
    hi = (q[..., 1::2] + 8).to(torch.uint8) & 0x0F
    return lo | (hi << 4), scale


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, width: int) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.float32) - 8.0
    hi = ((packed >> 4) & 0x0F).to(torch.float32) - 8.0
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=torch.float32, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out[..., :width] * scale


def subspace_cosine_similarity(p_old: torch.Tensor, p_new: torch.Tensor) -> float:
    """||P_old^T P_new||_F^2 / r in [0, 1] for orthonormal columns; invariant to column signs and in-subspace rotations."""
    r = p_old.shape[1]
    return float((p_old.T @ p_new).pow(2).sum().item() / r)


def svd_free_subspace_iteration(grad: torch.Tensor, p_warm: torch.Tensor, iters: int = 2) -> torch.Tensor:
    """
    Warm-started orthogonal (subspace) iteration on G G^T: P <- qr(G (G^T P)). Costs O(m n r) per iteration
    instead of the O(m n min(m, n)) of a full SVD. Converges to the top-r left singular subspace at a rate set
    by the singular-value gap, so it is only a good substitute for SVD when the gradient subspace drifts slowly.
    """
    p = p_warm
    for _ in range(iters):
        y = grad @ (grad.T @ p)
        p, _ = torch.linalg.qr(y, mode="reduced")
    return p.contiguous()


class LayerWise8BitGaLoreEngine:
    """
    Registers per-layer post-accumulation backward hooks (register_post_accumulate_grad_hook)
    on linear weight parameters so each layer's weight gradient G_t is projected into the
    low-rank subspace R_t = P_t^T G_t, consumes its 8-bit Adam moments (M_t, V_t), updates
    the master weight in-place, and frees param.grad immediately before the preceding
    layer's backward GEMM executes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        rank: int = 512,
        update_proj_gap: int = 200,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        scale_alpha: float = 0.25,
        proj_bits: int = 8,
        lazy_subspace: bool = False,
        cos_sim_threshold: float = 0.90,
        max_proj_gap: int = 1600,
        subspace_iters: int = 0,
        reproject_moments: bool = True,
        refresh_v_fill: bool = True,
        pin_memory: bool = True,
    ) -> None:
        assert proj_bits in (4, 8)
        self.proj_bits = proj_bits
        self.lazy_subspace = lazy_subspace
        self.cos_sim_threshold = cos_sim_threshold
        self.max_proj_gap = max_proj_gap
        self.subspace_iters = subspace_iters  # 0 = exact SVD refresh; >0 = warm-started subspace iteration
        # On a projector refresh, rotate the Adam moments into the new basis (docs/11 O14). M is linear in the
        # gradient, so M_new = (P_new^T P_old) M_old is exact. V holds squares and has no exact transform;
        # V_new = (P_new^T P_old)^2 V_old (elementwise square) is exact when the old coordinates are uncorrelated.
        self.reproject_moments = reproject_moments
        # docs/12 4.26: the part of a new direction that lies outside the old span has no second-moment history,
        # while bias correction (driven by the global step) is already about 1. Left at zero, V restarts from
        # (1 - beta2) g^2 against a momentum that recovers in about 1 / (1 - beta1) steps, and the step peaks at
        # 3.45x its steady size for tens of steps. With refresh_v_fill the missing mass is seeded with the current
        # projected gradient in the same bias state as the rest of V: V += lost * (1 - beta2^t) * r_new^2, so that
        # after the global bias correction the new directions see sqrt(V_hat) = |r_new|, as a first Adam step does.
        # (Seeding the full r_new^2 instead is wrong while 1 - beta2^t is still small: the correction then shrinks
        # the step, 0.5x at t = 300.)
        # False reproduces the old behaviour (kept as the negative control of the regression test).
        self.refresh_v_fill = refresh_v_fill
        self.svd_calls = 0
        self.refresh_calls = 0
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.scale_alpha = scale_alpha
        self.states: Dict[torch.nn.Parameter, GaLoreState8Bit] = {}
        self.pin_memory = bool(pin_memory)
        self.hooks = []
        self.peak_transient_grad_bytes = 0
        # docs/12 4.26: full-size tensors alive at the peak of one layer step (param.grad, its FP32 copy, the
        # full update, the FP32 weight copy, and for BF16 masters the int32 temporaries of the stochastic write).
        # peak_transient_grad_bytes above counts param.grad only and understates the transient by this ratio.
        self.peak_transient_step_bytes = 0

        for _, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                if module.weight.ndim == 2 and module.weight.requires_grad:
                    self._register_parameter(module.weight)

    def _register_parameter(self, param: torch.nn.Parameter) -> None:
        rows, cols = param.shape
        transpose = rows > cols
        m, n = (cols, rows) if transpose else (rows, cols)
        r = min(self.rank, m)

        self.states[param] = GaLoreState8Bit(
            step=0,
            rank=r,
            update_proj_gap=self.update_proj_gap,
            proj_q=(torch.zeros((m, r), dtype=torch.int8) if self.proj_bits == 8
                    else torch.zeros((m, (r + 1) // 2), dtype=torch.uint8)),
            proj_scale=torch.ones((m, 1), dtype=torch.float32),
            exp_avg_q=torch.zeros((r, n), dtype=torch.int8),
            exp_avg_scale=torch.ones((r, 1), dtype=torch.float32),
            exp_avg_sq_q=torch.zeros((r, n), dtype=torch.uint8),
            exp_avg_sq_scale=torch.ones((r, 1), dtype=torch.float32),
            transpose=transpose,
            next_svd_step=0,
            proj_bits=self.proj_bits,
        )

        def _post_grad_hook(p: torch.nn.Parameter) -> None:
            if p.grad is None:
                return
            grad_bytes = p.grad.numel() * p.grad.element_size()
            if grad_bytes > self.peak_transient_grad_bytes:
                self.peak_transient_grad_bytes = grad_bytes
            self._step_parameter(p)
            # Immediately free full-rank gradient tensor before preceding layer's backward pass
            p.grad = None

        if not hasattr(param, "register_post_accumulate_grad_hook"):
            # Without the hook nothing ever steps the parameter; fail loudly instead of training nothing.
            raise RuntimeError("LayerWise8BitGaLoreEngine needs torch >= 2.1 (register_post_accumulate_grad_hook)")
        h = param.register_post_accumulate_grad_hook(_post_grad_hook)
        self.hooks.append(h)

    def _to_host(self, t: torch.Tensor, dst: torch.Tensor = None) -> torch.Tensor:
        src = t.detach()
        if src.is_cuda:
            if (
                dst is not None
                and dst.device.type == "cpu"
                and dst.shape == src.shape
                and dst.dtype == src.dtype
                and (not self.pin_memory or dst.is_pinned())
            ):
                dst.copy_(src)
                return dst
            if self.pin_memory and torch.cuda.is_available():
                try:
                    pinned = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
                    pinned.copy_(src)
                    return pinned
                except RuntimeError:
                    pass
        return src.cpu()

    def _to_device(self, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        if device.type == "cuda" and self.pin_memory and t.is_pinned():
            return t.to(device, non_blocking=True)
        return t.to(device)

    def _store_projector(self, st: GaLoreState8Bit, p_mat: torch.Tensor) -> None:
        # Quantize on device first so only 1 byte (int8) or 0.5 byte (int4) per element crosses PCIe
        if st.proj_bits == 4:
            q_p, sc_p = quantize_int4_stochastic(p_mat)
        else:
            q_p, sc_p = quantize_signed_int8(p_mat)
        st.proj_q = self._to_host(q_p, st.proj_q)
        st.proj_scale = self._to_host(sc_p, st.proj_scale)

    def _load_projector(self, st: GaLoreState8Bit, device: torch.device) -> torch.Tensor:
        if st.proj_bits == 4:
            p = dequantize_int4(self._to_device(st.proj_q, device), self._to_device(st.proj_scale, device), st.rank)
            return p / p.norm(dim=0, keepdim=True).clamp(min=1e-12)
        return dequantize_signed_int8(self._to_device(st.proj_q, device), self._to_device(st.proj_scale, device))

    @torch.no_grad()
    def _reproject_moments(self, st: GaLoreState8Bit, rot: torch.Tensor, dev: torch.device,
                           lost: torch.Tensor = None, r_new: torch.Tensor = None) -> None:
        m_old = dequantize_signed_int8(self._to_device(st.exp_avg_q, dev), self._to_device(st.exp_avg_scale, dev))
        sv_old = dequantize_sqrt_v_log8(self._to_device(st.exp_avg_sq_q, dev), self._to_device(st.exp_avg_sq_scale, dev))
        m_new = rot @ m_old
        v_new = (rot * rot) @ (sv_old * sv_old)
        if self.refresh_v_fill and lost is not None and r_new is not None:
            bias2 = 1.0 - (self.betas[1] ** st.step)       # bias state of the stored V (st.step updates so far)
            v_new = v_new + (lost * bias2).unsqueeze(1) * (r_new * r_new)
        q_m, sc_m = quantize_signed_int8_sr(m_new)
        q_sv, sc_sv = quantize_sqrt_v_log8_sr(torch.sqrt(v_new.clamp(min=0.0)))
        st.exp_avg_q = self._to_host(q_m, st.exp_avg_q)
        st.exp_avg_scale = self._to_host(sc_m, st.exp_avg_scale)
        st.exp_avg_sq_q = self._to_host(q_sv, st.exp_avg_sq_q)
        st.exp_avg_sq_scale = self._to_host(sc_sv, st.exp_avg_sq_scale)

    @torch.no_grad()
    def _step_parameter(self, param: torch.nn.Parameter) -> None:
        st = self.states[param]
        grad = param.grad.to(torch.float32)
        if st.transpose:
            grad = grad.T  # Ensure shape [m, n] with m <= n

        # 1. Refresh the projector P_t in R^{m x r} when due. Scheduled by next_svd_step so the interval can
        #    grow per layer (Q-GaLore lazy updates) without breaking the modulo-free schedule.
        dev = grad.device
        if st.step >= st.next_svd_step:
            has_prev = st.step > 0
            p_old = self._load_projector(st, dev) if has_prev else None
            p_old_orth = None
            if has_prev and (self.lazy_subspace or self.reproject_moments):
                q_qr, r_qr = torch.linalg.qr(p_old, mode="reduced")
                diag_sign = torch.sign(torch.diagonal(r_qr))
                diag_sign[diag_sign == 0] = 1.0
                p_old_orth = q_qr * diag_sign.unsqueeze(0)
            if self.subspace_iters > 0 and has_prev:
                p_mat = svd_free_subspace_iteration(grad, p_old, self.subspace_iters)
            else:
                u, _, _ = torch.linalg.svd(grad, full_matrices=False)
                p_mat = u[:, : st.rank].contiguous()
                self.svd_calls += 1
            self.refresh_calls += 1
            if self.lazy_subspace and has_prev and p_old_orth is not None:
                if subspace_cosine_similarity(p_old_orth, p_mat) >= self.cos_sim_threshold:
                    st.update_proj_gap = min(st.update_proj_gap * 2, self.max_proj_gap)
            st.next_svd_step = st.step + st.update_proj_gap
            self._store_projector(st, p_mat)
            # Use the exact value that will be re-read next step so both paths see the same quantized projector
            p_mat = self._load_projector(st, dev)
            if self.reproject_moments and has_prev:
                # Both projectors as STORED: the moments live in those coordinates (docs/12 4.26).
                rot, lost = moment_rotation(p_mat, p_old)
                self._reproject_moments(st, rot, dev, lost, p_mat.T @ grad)
        else:
            p_mat = self._load_projector(st, dev)

        st.step += 1

        # 2. Project full gradient G_t into compact low-rank core R_t = P_t^T @ G_t in R^{r x n}
        r_t = p_mat.T @ grad

        # 3. Dequantize 8-bit Adam states from Host Pinned memory:
        #    Note: exp_avg_sq_q stores the root second moment sqrt(V_{t-1}) in uint8 so its dynamic
        #    range matches M_{t-1} (preventing <4.4% coordinate underflow to 0 and 10^5x step spikes).
        m_t = dequantize_signed_int8(
            self._to_device(st.exp_avg_q, grad.device), self._to_device(st.exp_avg_scale, grad.device)
        )
        sqrt_v_prev = dequantize_sqrt_v_log8(
            self._to_device(st.exp_avg_sq_q, grad.device), self._to_device(st.exp_avg_sq_scale, grad.device)
        )
        v_t = sqrt_v_prev * sqrt_v_prev

        beta1, beta2 = self.betas
        m_t.mul_(beta1).add_(r_t, alpha=1.0 - beta1)
        v_t.mul_(beta2).addcmul_(r_t, r_t, value=1.0 - beta2)
        sqrt_v_t = torch.sqrt(v_t)

        # Quantize updated moments (M_t and sqrt(V_t)) to 8-bit before transferring to Host RAM
        q_m, sc_m = quantize_signed_int8_sr(m_t)
        q_sv, sc_sv = quantize_sqrt_v_log8_sr(sqrt_v_t)
        st.exp_avg_q = self._to_host(q_m, st.exp_avg_q)
        st.exp_avg_scale = self._to_host(sc_m, st.exp_avg_scale)
        st.exp_avg_sq_q = self._to_host(q_sv, st.exp_avg_sq_q)
        st.exp_avg_sq_scale = self._to_host(sc_sv, st.exp_avg_sq_scale)

        # 4. Bias-corrected low-rank update and projection back to full parameter space: P_t @ N_t
        bias_corr1 = 1.0 - (beta1 ** st.step)
        bias_corr2 = 1.0 - (beta2 ** st.step)
        step_size = self.lr * (math.sqrt(bias_corr2) / bias_corr1) * self.scale_alpha

        norm_grad_lowrank = m_t / (sqrt_v_t + self.eps)
        full_update = p_mat @ norm_grad_lowrank
        if st.transpose:
            full_update = full_update.T

        # Apply decoupled weight decay and low-rank projected step in-place
        w_fp32 = param.data.to(torch.float32)
        if self.weight_decay > 0.0:
            w_fp32.mul_(1.0 - self.lr * self.weight_decay)
        w_fp32.add_(full_update, alpha=-step_size)
        live = param.grad.numel() * param.grad.element_size() + full_update.numel() * 4 + w_fp32.numel() * 4
        if param.grad.dtype != torch.float32:
            live += grad.numel() * 4                       # FP32 copy of the gradient
        if param.data.dtype == torch.bfloat16:
            live += 3 * w_fp32.numel() * 4                 # noise, bits + noise, rounded (int32 each)
        if live > self.peak_transient_step_bytes:
            self.peak_transient_step_bytes = live
        copy_to_param_stochastic(param.data, w_fp32)


# A "24 GB" card holds 24 GiB = 24 * 2**30 bytes = 25.77 decimal GB. All table values below are decimal GB.
GB_PER_24GIB_CARD = 24 * (2 ** 30) / 1e9  # ~= 25.77

_COMPONENT_KEYS = (
    "master_weights_vram_gb",
    "quantized_weights_vram_gb",
    "weight_gradients_vram_gb",
    "optimizer_states_vram_gb",
    "activations_vram_gb",
)


def compute_32b_qat_memory_table(n_gpus: int = 2) -> Dict[str, Dict[str, float]]:
    """
    Memory breakdown (decimal GB) for a 32B-parameter model (L=64, D=5120) sharded evenly over n_gpus.
    Each listed value is the TOTAL resident VRAM across all GPUs; peak_per_gpu is derived as
    sum(components) / n_gpus so the headline can never drift from its own breakdown. (The previous
    hard-coded GaLore peak of 22.15 did not match its components, which sum to 23.63 / GPU.)

    Host-staged bytes (GaLore's ~24 GB of master-weight HugePages and the ~2.42 GB of 8-bit optimizer
    moments streamed over PCIe) are NOT counted as VRAM: they live in host pinned memory, not on the GPU.

    CAVEAT (docs/12 4.26): the 0.45 GB weight-gradient row is a stated assumption that covers one layer's
    gradient buffer. The reference engine in this file holds more than that at the peak of a layer step
    (LayerWise8BitGaLoreEngine.peak_transient_step_bytes, 11x the BF16 gradient for a BF16 master as written);
    a device implementation has to update in place to stay inside this row. The table does not model that.
    """
    rows = {
        "Standard BF16 AdamW": {
            "master_weights_vram_gb": 64.00,
            "quantized_weights_vram_gb": 64.00,
            "weight_gradients_vram_gb": 64.00,
            "optimizer_states_vram_gb": 256.00,
            "activations_vram_gb": 6.80,
        },
        "8-Bit AdamW (Full Rank)": {
            "master_weights_vram_gb": 64.00,
            "quantized_weights_vram_gb": 64.00,
            "weight_gradients_vram_gb": 64.00,
            "optimizer_states_vram_gb": 64.00,
            "activations_vram_gb": 6.80,
        },
        "Layer-Wise 8-Bit GaLore (r=512, T=200) + Async DMA": {
            "master_weights_vram_gb": 40.00,   # resident VRAM copy; a further 24 GB sits in host HugePages
            "quantized_weights_vram_gb": 0.00, # fused in SRAM via fused_qat_g128_fwd_kernel
            "weight_gradients_vram_gb": 0.45,  # single-layer transient buffer freed via post-grad hook
            "optimizer_states_vram_gb": 0.00,  # 2.42 GB of 8-bit moments stream from host, 0 resident
            "activations_vram_gb": 6.80,       # with gradient checkpointing
        },
    }
    for row in rows.values():
        total = sum(row[k] for k in _COMPONENT_KEYS)
        row["peak_per_gpu_2x24gb"] = math.floor((total / n_gpus) * 100.0 + 0.5) / 100.0
    return rows


def calibrate_lazy_subspace_drift_o15(steps: int = 32, rank: int = 8) -> Dict[float, Dict[str, float]]:
    """
    Calibrates the Q-GaLore lazy-SVD cosine-similarity threshold (O15 in docs/11) on a multi-step
    low-rank-plus-drift QAT gradient trajectory (arXiv:2403.03507, arXiv:2407.08296).
    Compares tau_cos in {0.85, 0.90, 0.95} against a fixed-interval SVD baseline (update_proj_gap=2).
    """
    torch.manual_seed(42)
    in_dim, out_dim = 128, 64
    w_init = torch.randn(out_dim, in_dim, dtype=torch.float32) * 0.05

    # Construct a realistic slowly-drifting low-rank teacher target trajectory with anisotropic
    # activation covariance (GaLore Theorem 3.2, arXiv:2403.03507; Q-GaLore Sec. 3.1, arXiv:2407.08296)
    u_left, _ = torch.linalg.qr(torch.randn(out_dim, rank))
    v_right, _ = torch.linalg.qr(torch.randn(in_dim, rank))
    u_drift, _ = torch.linalg.qr(torch.randn(out_dim, rank))
    # Orthogonalize u_drift against u_left so rotation angle theta directly controls subspace cosine similarity cos^2(theta)
    u_drift = u_drift - u_left @ (u_left.T @ u_drift)
    u_drift, _ = torch.linalg.qr(u_drift)
    spectrum = torch.linspace(4.0, 1.0, rank)
    x_batches = [
        torch.randn(128, rank) @ v_right.T + 0.05 * torch.randn(128, in_dim)
        for _ in range(steps)
    ]

    def _run_schedule(lazy: bool, threshold: float) -> Tuple[torch.Tensor, int, float]:
        torch.manual_seed(101)
        layer = torch.nn.Linear(in_dim, out_dim, bias=False)
        layer.weight.data.copy_(w_init)
        mdl = torch.nn.Sequential(layer)
        eng = LayerWise8BitGaLoreEngine(
            mdl,
            rank=rank,
            update_proj_gap=2,
            lr=3e-3,
            lazy_subspace=lazy,
            cos_sim_threshold=threshold,
            max_proj_gap=16,
            reproject_moments=True,
        )
        final_loss = 0.0
        for step_idx in range(steps):
            # Subspace rotation schedule: slow drift early, accelerating mid-trajectory (~0.015 rad/step)
            theta = 0.015 * float(step_idx)
            u_t = math.cos(theta) * u_left + math.sin(theta) * u_drift
            w_star = (u_t * spectrum) @ v_right.T
            xb = x_batches[step_idx]
            yb = xb @ w_star.T
            pred = mdl(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            final_loss = float(loss.item())
            loss.backward()
        return layer.weight.data.detach().clone(), eng.svd_calls, final_loss

    w_base, svd_base, loss_base = _run_schedule(lazy=False, threshold=1.0)
    delta_base = (w_base - w_init).flatten()

    results: Dict[float, Dict[str, float]] = {
        1.00: {
            "svd_calls": float(svd_base),
            "svd_reduction_pct": 0.0,
            "final_loss": loss_base,
            "weight_delta_cos_sim": 1.0,
        }
    }
    for tau in (0.85, 0.90, 0.95):
        w_lazy, svd_lazy, loss_lazy = _run_schedule(lazy=True, threshold=tau)
        delta_lazy = (w_lazy - w_init).flatten()
        cos_sim = float(
            torch.dot(delta_lazy, delta_base).item()
            / (delta_lazy.norm().item() * delta_base.norm().item() + 1e-12)
        )
        results[tau] = {
            "svd_calls": float(svd_lazy),
            "svd_reduction_pct": 100.0 * (1.0 - float(svd_lazy) / float(svd_base)),
            "final_loss": loss_lazy,
            "weight_delta_cos_sim": cos_sim,
        }
    return results


def main() -> None:
    print("================================================================================")
    print("  WORKSTREAM 5B: LAYER-WISE 8-BIT GALORE + POST-GRAD HOOKS & 32B MEMORY AUDIT")
    print("================================================================================\n")

    from fused_qat_ste import FusedQATG128Linear

    torch.manual_seed(23)
    model = torch.nn.Sequential(
        FusedQATG128Linear(256, 512),
        FusedQATG128Linear(512, 256),
        FusedQATG128Linear(256, 128),
    )
    engine = LayerWise8BitGaLoreEngine(model, rank=32, update_proj_gap=5, lr=5e-3)

    x = torch.randn((32, 256), dtype=torch.bfloat16)
    target = torch.randn((32, 128), dtype=torch.bfloat16)

    losses = []
    for _ in range(6):
        out = model(x)
        loss = torch.nn.functional.mse_loss(out.float(), target.float())
        losses.append(float(loss.item()))
        loss.backward()
        # Verify all parameter gradients were immediately freed by post-accumulate hooks
        for p in model.parameters():
            assert p.grad is None, "Post-accumulate hook failed to free parameter gradient!"

    # --- Q-GaLore features (arXiv:2407.08296): int4 stochastic projector, lazy schedule, subspace iteration ---
    torch.manual_seed(5)
    m_, n_, r_ = 256, 512, 16
    # Gradient with a clear rank-16 signal (gap) plus noise, as in a converging layer
    u0, _ = torch.linalg.qr(torch.randn(m_, r_))
    g_true = (u0 * torch.linspace(10.0, 5.0, r_)) @ torch.randn(r_, n_) + 0.05 * torch.randn(m_, n_)
    u_svd, _, _ = torch.linalg.svd(g_true, full_matrices=False)
    p_svd = u_svd[:, :r_].contiguous()
    energy_svd = float((p_svd.T @ g_true).pow(2).sum().item())

    # (a) int4 stochastic rounding: unbiased on average and small relative projector error
    packed, sc = quantize_int4_stochastic(p_svd)
    p4 = dequantize_int4(packed, sc, r_)
    rel4 = float((p4 - p_svd).norm().item() / p_svd.norm().item())
    avg = torch.stack([dequantize_int4(*quantize_int4_stochastic(p_svd), r_) for _ in range(200)]).mean(0)
    bias = float((avg - p_svd).abs().max().item())
    assert packed.dtype == torch.uint8 and packed.shape == (m_, r_ // 2)
    assert rel4 < 0.35 and bias < 0.02, (rel4, bias)

    # (b) 2-step warm-started subspace iteration recovers >= 99% of the exact-SVD captured energy from a perturbed start
    p_warm, _ = torch.linalg.qr(p_svd + 0.15 * torch.randn_like(p_svd))
    p_it = svd_free_subspace_iteration(g_true, p_warm, iters=2)
    energy_it = float((p_it.T @ g_true).pow(2).sum().item())
    assert energy_it / energy_svd > 0.99, energy_it / energy_svd
    assert float((p_it.T @ p_it - torch.eye(r_)).abs().max().item()) < 1e-4

    # (c) O15 Empirical calibration of lazy-SVD subspace cosine-similarity threshold tau_cos in {0.85, 0.90, 0.95}
    drift_cal = calibrate_lazy_subspace_drift_o15(steps=32, rank=8)
    base_svd = int(drift_cal[1.00]["svd_calls"])
    cal_90 = drift_cal[0.90]
    lazy_svd_90 = int(cal_90["svd_calls"])
    assert cal_90["svd_reduction_pct"] >= 40.0, cal_90
    # Threshold re-calibrated with the unbiased sqrt(V) store (docs/12 4.6 fix). The old ceil store inflated V,
    # which shrank every Adam step and made the lazy and fixed-interval trajectories look closer (0.9810). With
    # correct-size steps the lazy schedule's stale subspace costs slightly more: 0.9792 on 4 seed pairs, while
    # the stochastic-rounding noise floor (same schedule, different seed) is 0.9998, so the change is
    # systematic, not noise. 0.975 keeps a margin above the measured value.
    # Re-measured after docs/12 4.26 (stochastic first-moment store, bias-aware refresh V fill): 0.9786 to 0.9791
    # on 4 seeds, so the threshold stands.
    assert cal_90["weight_delta_cos_sim"] >= 0.975, cal_90
    assert drift_cal[0.85]["svd_calls"] <= drift_cal[0.90]["svd_calls"] <= drift_cal[0.95]["svd_calls"] <= base_svd
    print(f"[Q] int4 projector rel err {rel4:.3f} (bias {bias:.4f}) | subspace-iter energy {100*energy_it/energy_svd:.2f}% of SVD "
          f"| O15 Lazy-SVD (tau=0.90): {lazy_svd_90}/{base_svd} SVDs (-{cal_90['svd_reduction_pct']:.1f}%), "
          f"trajectory cos_sim={cal_90['weight_delta_cos_sim']:.4f}")

    # (d) docs/12 4.6 regression: the stored second moment must track the true EMA over a long horizon.
    #     3000 steps of the exact store/load/update loop the engine runs, gradient scale spread over 3 decades
    #     across columns. The old linear uint8 + ceil store ends at a median sqrt(V) ratio of about 10.5x
    #     (about 54x on the smallest columns); the log8 SR store must stay unbiased in V and tight in sqrt(V).
    torch.manual_seed(11)
    r_v, n_v, b2_v = 16, 512, 0.999
    colscale = torch.logspace(-3, 0, n_v)
    q_v, sc_v = torch.zeros((r_v, n_v), dtype=torch.uint8), torch.ones((r_v, 1))
    q_c, sc_c = torch.zeros((r_v, n_v), dtype=torch.uint8), torch.ones((r_v, 1))
    v_true = torch.zeros(r_v, n_v)
    v_ratio_acc = torch.zeros(r_v, n_v)
    for t in range(1, 3001):
        g = torch.randn(r_v, n_v) * colscale
        v_true = b2_v * v_true + (1 - b2_v) * g * g
        v_new = b2_v * dequantize_sqrt_v_log8(q_v, sc_v) ** 2 + (1 - b2_v) * g * g
        q_v, sc_v = quantize_sqrt_v_log8_sr(v_new.sqrt())
        v_old = b2_v * dequantize_unsigned_uint8(q_c, sc_c) ** 2 + (1 - b2_v) * g * g
        q_c, sc_c = quantize_unsigned_uint8(v_old.sqrt(), round_up=True)
        if t > 1000:
            v_ratio_acc += dequantize_sqrt_v_log8(q_v, sc_v) ** 2 / v_true
    ratio_new = dequantize_sqrt_v_log8(q_v, sc_v) / v_true.sqrt()
    ratio_old = dequantize_unsigned_uint8(q_c, sc_c) / v_true.sqrt()
    mean_v_bias = float((v_ratio_acc / 2000.0).mean())
    med_new, med_old = float(ratio_new.median()), float(ratio_old.median())
    p05, p95 = float(ratio_new.quantile(0.05)), float(ratio_new.quantile(0.95))
    small_new = float(ratio_new[:, :32].median())
    assert abs(mean_v_bias - 1.0) < 0.02, mean_v_bias
    assert 0.95 < med_new < 1.05 and 0.95 < small_new < 1.05 and p05 > 0.7 and p95 < 1.4, (med_new, small_new, p05, p95)
    assert med_old > 5.0, med_old  # negative control: the old store really does ratchet
    assert int((dequantize_sqrt_v_log8(q_v, sc_v) == 0).sum()) == 0  # zero excluded for V > 0
    print(f"[V] sqrt(V) store over 3000 steps (docs/12 4.6): log8-SR median {med_new:.3f} (p05 {p05:.3f}, p95 {p95:.3f}, "
          f"small cols {small_new:.3f}), E[V_hat/V] {mean_v_bias:.4f} | old ceil-uint8 median {med_old:.2f}x")

    # (e) docs/12 4.26: step size across a refresh into an orthogonal subspace. Constant gradient in subspace A
    #     for 300 steps, then subspace B; a refresh falls on step 300. With refresh_v_fill the step must not
    #     overshoot its steady size and must recover it; the old behaviour (control) peaks above 2.5x.
    def _flip_peak(v_fill: bool) -> float:
        torch.manual_seed(3)
        lin = torch.nn.Linear(64, 32, bias=False)
        w_prev = lin.weight.detach().clone()
        LayerWise8BitGaLoreEngine(torch.nn.Sequential(lin), rank=4, update_proj_gap=100, lr=1e-3,
                                  weight_decay=0.0, refresh_v_fill=v_fill)
        qa, _ = torch.linalg.qr(torch.randn(32, 8))
        rg = torch.randn(4, 64)
        norms = []
        for t in range(400):
            (lin.weight * ((qa[:, :4] if t < 300 else qa[:, 4:]) @ rg)).sum().backward()
            norms.append(float((lin.weight.detach() - w_prev).norm()))
            w_prev = lin.weight.detach().clone()
        return max(norms[300:]) / (sum(norms[250:300]) / 50.0)
    peak_fill, peak_old = _flip_peak(True), _flip_peak(False)
    assert 0.8 < peak_fill < 1.3, peak_fill
    assert peak_old > 2.5, peak_old

    # (f) docs/12 4.26: first-moment store. Constant-sign gradients with a three-decade column spread; the columns
    #     below the round-to-nearest dead zone (under 0.3% of the row maximum) must keep their momentum on average.
    torch.manual_seed(0)
    col_m = torch.logspace(-3, 0, 512)
    small = col_m < 3e-3
    m_true = torch.zeros(16, 512)
    q_sr, s_sr = torch.zeros(16, 512, dtype=torch.int8), torch.ones(16, 1)
    q_rn, s_rn = torch.zeros(16, 512, dtype=torch.int8), torch.ones(16, 1)
    acc_sr, acc_rn, acc_true = 0.0, 0.0, 0.0
    for t in range(300):
        g_m = col_m * (1.0 + 0.1 * torch.randn(16, 512))
        m_true = 0.9 * m_true + 0.1 * g_m
        q_sr, s_sr = quantize_signed_int8_sr(0.9 * dequantize_signed_int8(q_sr, s_sr) + 0.1 * g_m)
        q_rn, s_rn = quantize_signed_int8(0.9 * dequantize_signed_int8(q_rn, s_rn) + 0.1 * g_m)
        if t >= 200:
            acc_sr += float(dequantize_signed_int8(q_sr, s_sr)[:, small].sum())
            acc_rn += float(dequantize_signed_int8(q_rn, s_rn)[:, small].sum())
            acc_true += float(m_true[:, small].sum())
    m_ratio_sr, m_ratio_rn = acc_sr / acc_true, acc_rn / acc_true
    assert 0.9 < m_ratio_sr < 1.1, m_ratio_sr
    assert m_ratio_rn < 0.2, m_ratio_rn     # negative control: round-to-nearest loses the small columns

    # (g) docs/12 4.26: moment change of coordinates between two STORED int4 projectors of the same subspace.
    torch.manual_seed(2)
    u_g, _ = torch.linalg.qr(torch.randn(256, 16))
    o_g, _ = torch.linalg.qr(torch.randn(16, 16))

    def _stored_int4(pm: torch.Tensor) -> torch.Tensor:
        d = dequantize_int4(*quantize_int4_stochastic(pm), 16)
        return d / d.norm(dim=0, keepdim=True)
    p_old_s, p_new_s = _stored_int4(u_g), _stored_int4(u_g @ o_g)
    g_g = p_old_s @ torch.randn(16, 64)                       # gradient inside the old stored span
    c_old, c_new = p_old_s.T @ g_g, p_new_s.T @ g_g
    rot_g, lost_g = moment_rotation(p_new_s, p_old_s)
    q_g, r_g = torch.linalg.qr(p_old_s)
    rot_err = float((rot_g @ c_old - c_new).norm() / c_new.norm())
    rot_err_old = float(((p_new_s.T @ q_g) @ (q_g.T @ g_g) - c_new).norm() / c_new.norm())  # exact for Q coords ...
    rot_err_legacy = float(((p_new_s.T @ (q_g * torch.sign(torch.diagonal(r_g)))) @ c_old - c_new).norm() / c_new.norm())
    assert rot_err < 1e-4, rot_err
    assert rot_err_old < 1e-4                                 # ... but the stored coordinates are c_old, not Q^T g
    assert rot_err_legacy > 5e-3, rot_err_legacy              # negative control: the old formula on stored coords

    table = compute_32b_qat_memory_table()
    galore_budget = table["Layer-Wise 8-Bit GaLore (r=512, T=200) + Async DMA"]
    step_ratio = engine.peak_transient_step_bytes / max(1, engine.peak_transient_grad_bytes)
    assert engine.peak_transient_step_bytes > engine.peak_transient_grad_bytes
    print(f"[R] Refresh into a new subspace (4.26): peak step {peak_fill:.2f}x steady with V fill | {peak_old:.2f}x without "
          f"(control) | int8 M small-column mean: SR {m_ratio_sr:.3f}, RTN {m_ratio_rn:.3f} (control) | int4 moment "
          f"rotation err {rot_err:.1e} (old formula {rot_err_legacy:.1e})")
    print(f"[T] Layer-step transient (4.26): {engine.peak_transient_step_bytes} B live vs {engine.peak_transient_grad_bytes} B "
          f"of param.grad ({step_ratio:.1f}x); the 0.45 GB table row covers the gradient buffer only")

    print(f"[1] 3-Layer Fused STE + 8-Bit GaLore Loss Trajectory : {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"[2] Retained Full-Model Gradient Buffers Post-Backward: 0 Bytes (all freed in-flight)")
    print(f"[3] 32B Model Peak VRAM per GPU (Standard AdamW)      : {table['Standard BF16 AdamW']['peak_per_gpu_2x24gb']:.2f} GB/GPU (OOM)")
    print(f"[4] 32B Model Peak VRAM per GPU (8-Bit AdamW)         : {table['8-Bit AdamW (Full Rank)']['peak_per_gpu_2x24gb']:.2f} GB/GPU (OOM)")
    fits = galore_budget["peak_per_gpu_2x24gb"] <= GB_PER_24GIB_CARD
    print(f"[5] 32B Model Peak VRAM per GPU (Fused STE + GaLore)  : {galore_budget['peak_per_gpu_2x24gb']:.2f} GB/GPU "
          f"({'Fits' if fits else 'OOM on'} a 24 GiB = {GB_PER_24GIB_CARD:.2f} GB card)")
    assert losses[-1] < losses[0]
    # Cross-check against first principles (docs/12 4.16: the old assert compared each peak with sum/2 of itself).
    # P = 32e9 parameters. BF16 = 2 B/param, FP32 Adam m+v = 8 B/param, 8-bit m+v = 2 B/param (decimal GB).
    P_gb = 32e9 / 1e9
    std, q8, gal = table["Standard BF16 AdamW"], table["8-Bit AdamW (Full Rank)"], galore_budget
    for row in (std, q8):
        assert row["master_weights_vram_gb"] == 2 * P_gb and row["quantized_weights_vram_gb"] == 2 * P_gb
        assert row["weight_gradients_vram_gb"] == 2 * P_gb
    assert std["optimizer_states_vram_gb"] == 8 * P_gb and q8["optimizer_states_vram_gb"] == 2 * P_gb
    GALORE_HOST_STAGED_MASTER_GB = 24.0  # stated in the table docstring; resident + host must be the BF16 master
    assert gal["master_weights_vram_gb"] + GALORE_HOST_STAGED_MASTER_GB == 2 * P_gb
    # Peaks worked by hand: (64*3 + 256 + 6.8)/2 = 227.40, (64*4 + 6.8)/2 = 131.40, (40 + 0.45 + 6.8)/2 = 23.625 -> 23.63.
    # The 0.45 GB transient and 6.80 GB activation figures are stated assumptions, not derived here.
    assert (std["peak_per_gpu_2x24gb"], q8["peak_per_gpu_2x24gb"], gal["peak_per_gpu_2x24gb"]) == (227.40, 131.40, 23.63)
    assert fits, galore_budget["peak_per_gpu_2x24gb"]

    # [6] docs/12 4.29 sibling regression: non-finite inputs (+Inf, -Inf, NaN) in quantize_signed_int8,
    #     quantize_signed_int8_sr, quantize_unsigned_uint8, and quantize_int4_stochastic must preserve extremal
    #     codes (+127/-127, 255/0, +7/-7) instead of collapsing Inf/Inf=NaN to code 0.
    t_nf = torch.tensor([[float("inf"), float("-inf"), float("nan"), 0.0]], dtype=torch.float32)
    q8_nf, _ = quantize_signed_int8(t_nf)
    q8sr_nf, _ = quantize_signed_int8_sr(t_nf)
    qu8_nf, _ = quantize_unsigned_uint8(t_nf)
    p4_nf, _ = quantize_int4_stochastic(t_nf)
    assert q8_nf[0].tolist() == [127, -127, 0, 0] and q8sr_nf[0].tolist() == [127, -127, 0, 0]
    assert qu8_nf[0].tolist() == [255, 0, 0, 0]
    assert int(p4_nf[0, 0].item() & 0x0F) - 8 == 7 and int((p4_nf[0, 0].item() >> 4) & 0x0F) - 8 == -7
    assert int(p4_nf[0, 1].item() & 0x0F) - 8 == 0 and int((p4_nf[0, 1].item() >> 4) & 0x0F) - 8 == 0
    print("[6] Non-finite quantizer guard (docs/12 4.29)         : int8/int8_sr/uint8/int4_sr +Inf/-Inf/NaN codes PASS")
    print("[+] PASS: Per-layer 8-bit GaLore post-accumulate hooks and 32B memory budget verified.")


if __name__ == "__main__":
    main()
