#!/usr/bin/env python3
"""
File: fused_w1_hadacore/spinquant_packer.py
Workstream 3: Offline Stiefel Rotation Merger (Cayley SGD SpinQuant) & Interleaved 16-Bit Split Bit-Packer.
Implements:
  1. Cayley SGD optimization on the Stiefel manifold:
     R' = (I + (eta/2) * Y)^(-1) (I - (eta/2) * Y) R, where Y = G R^T - R G^T is skew-symmetric.
  2. Fast Walsh-Hadamard Transform (FWHT) construction for 16x16 HadaCore Tensor Core tiles.
  3. Offline rotation merge (W_rot = W @ R_1), g128 Binary (1.125 bpw) and BitNet b1.58 (2.125 bpw)
     quantization, and Interleaved 16-Bit Split uint32 packing for single-instruction lop3.b32 unpacking.
"""

from typing import Tuple
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def build_hadamard_matrix(n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Constructs a normalized n x n Sylvester Walsh-Hadamard matrix H_n with H_n @ H_n^T = I."""
    assert n > 0 and (n & (n - 1)) == 0, "Hadamard dimension must be a power of 2"
    h = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / (n ** 0.5)


def build_kernel_online_rotation(k: int, mode: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Returns the EXACT K x K orthogonal matrix applied to activations by fused_w1a16_hadacore_gemm_kernel,
    so offline weight packing can use the same rotation (W_rot = W @ R) and x_rot @ W_rot^T == x @ W^T.
      mode 1: I_{K/16} (x) H_16                 (block-diagonal radix-16 HadaCore, enable_online_hadacore = 1)
      mode 2: I_{K/32} (x) (H_2 (x) H_16)       (Kronecker H_32, enable_online_hadacore = 2)
    NOTE: this is a FIXED Hadamard rotation. A dense Cayley-optimized R_1 (optimize_stiefel_rotation) is
    only valid when it can be absorbed offline into neighbouring layers; it cannot be applied online by the
    kernel, so do not pack weights with a dense R_1 and then run the kernel with online rotation enabled.
    """
    assert mode in (1, 2)
    block = 16 if mode == 1 else 32
    assert k % block == 0, "K must be a multiple of the rotation block"
    h16 = build_hadamard_matrix(16, device=device)
    if mode == 1:
        h_block = h16
    else:
        h2 = build_hadamard_matrix(2, device=device)
        h_block = torch.kron(h2, h16)
    return torch.block_diag(*([h_block] * (k // block)))


def compute_kurtosis(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the basis-dependent L4/L2^2 standardized fourth moment (kurtosis) across the channel
    dimension (dim=-1) for each token activation vector x_i in R^d, plus per-channel sample kurtosis
    across tokens (dim=0). Under an orthogonal rotation x -> x @ R, the L2 norm ||x_i @ R||_2^2 is
    invariant while the fourth moment sum_j (x_i @ R)_j^4 is minimized when outlier energy is spread
    uniformly across all d channels.
    """
    x_f = x.to(torch.float32)
    m2_vec = torch.mean(x_f * x_f, dim=-1, keepdim=True) + eps
    m4_vec = torch.mean((x_f * x_f) ** 2, dim=-1, keepdim=True)
    kurt_vec = torch.mean(m4_vec / (m2_vec * m2_vec))

    m2_ch = torch.mean(x_f * x_f, dim=0, keepdim=True) + eps
    m4_ch = torch.mean((x_f * x_f) ** 2, dim=0, keepdim=True)
    kurt_ch = torch.mean(m4_ch / (m2_ch * m2_ch))
    return 0.5 * (kurt_vec + kurt_ch)


def cayley_sgd_stiefel_step(
    R: torch.Tensor,
    G: torch.Tensor,
    lr: float = 0.15,
) -> torch.Tensor:
    """
    Executes one exact Cayley transform update step on the Stiefel manifold O(d):
      Y  = G @ R^T - R @ G^T   (skew-symmetric: Y^T = -Y)
      R' = (I + (lr/2)*Y)^(-1) @ (I - (lr/2)*Y) @ R
    Preserves R' @ R'^T = I to machine precision.
    """
    d = R.shape[0]
    eye = torch.eye(d, dtype=R.dtype, device=R.device)
    Y = G @ R.T - R @ G.T
    half_lr_Y = 0.5 * lr * Y
    lhs = eye + half_lr_Y
    rhs = (eye - half_lr_Y) @ R
    return torch.linalg.solve(lhs, rhs).contiguous()


def optimize_stiefel_rotation(
    calib_activations: torch.Tensor,
    steps: int = 25,
    lr: float = 0.25,
) -> Tuple[torch.Tensor, float, float]:
    """
    Optimizes an orthogonal rotation matrix R_1 on the Stiefel manifold via Cayley SGD
    to minimize activation outlier kurtosis prior to 1-bit/ternary quantization.
    """
    d = calib_activations.shape[-1]
    if d > 0 and (d & (d - 1)) == 0:
        R = build_hadamard_matrix(d, device=calib_activations.device).detach().clone()
    elif d % 16 == 0:
        h16 = build_hadamard_matrix(16, device=calib_activations.device)
        R = torch.block_diag(*([h16] * (d // 16))).detach().clone()
    else:
        q_orth, _ = torch.linalg.qr(
            torch.randn(d, d, dtype=torch.float32, device=calib_activations.device)
        )
        R = q_orth.detach().clone()
    R = R.to(calib_activations.dtype)
    init_kurt = float(compute_kurtosis(calib_activations).item())

    for _ in range(steps):
        R.requires_grad_(True)
        x_rot = calib_activations @ R
        loss = compute_kurtosis(x_rot)
        (grad_R,) = torch.autograd.grad(loss, R)
        with torch.no_grad():
            # Normalize gradient norm for stable Cayley step
            g_norm = torch.norm(grad_R) + 1e-8
            R = cayley_sgd_stiefel_step(R, grad_R / g_norm, lr=lr)

    final_kurt = float(compute_kurtosis(calib_activations @ R).item())
    return R.detach(), init_kurt, final_kurt


def optimize_kronecker_stiefel_rotation(
    calib_activations: torch.Tensor,
    q_hadamard: int = 16,
    steps: int = 25,
    lr: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    Optimizes a Kronecker-factored orthogonal rotation R_kron = R_P (x) H_16 on the smaller
    Stiefel manifold O(P) (where P = K // 16) via Cayley SGD:
      R_kron = (R_P (x) I_16) @ (I_P (x) H_16)
    Because (R_P (x) I_16) and (I_P (x) H_16) commute:
      - (R_P (x) I_16) is absorbed offline into the preceding linear layer's output weights;
      - (I_P (x) H_16) is the EXACT online block-H_16 rotation applied in registers by the
        GPU kernel's hadacore_rotate_m16k16_in_regs prologue (enable_online_hadacore = 1);
      - R_kron = R_P (x) H_16 is merged offline into the current layer's quantized weights.
    This reduces Cayley SGD matrix inversion complexity by 16^3 = 4,096x while strictly improving
    outlier kurtosis reduction over fixed block-H_16 alone.
    Returns (R_kron, R_P, init_kurt, fixed_h16_kurt, final_kron_kurt).
    """
    d = calib_activations.shape[-1]
    assert d % q_hadamard == 0, "K must be divisible by q_hadamard"
    p_dim = d // q_hadamard
    h_q = build_hadamard_matrix(q_hadamard, device=calib_activations.device)
    r_p = torch.eye(p_dim, dtype=calib_activations.dtype, device=calib_activations.device)

    init_kurt = float(compute_kurtosis(calib_activations).item())
    fixed_h16_rot = torch.kron(torch.eye(p_dim, dtype=calib_activations.dtype, device=calib_activations.device), h_q)
    fixed_h16_kurt = float(compute_kurtosis(calib_activations @ fixed_h16_rot).item())

    for _ in range(steps):
        r_p.requires_grad_(True)
        r_kron = torch.kron(r_p, h_q)
        x_rot = calib_activations @ r_kron
        loss = compute_kurtosis(x_rot)
        (grad_rp,) = torch.autograd.grad(loss, r_p)
        with torch.no_grad():
            g_norm = torch.norm(grad_rp) + 1e-8
            r_p = cayley_sgd_stiefel_step(r_p, grad_rp / g_norm, lr=lr)

    r_kron_final = torch.kron(r_p.detach(), h_q)
    final_kron_kurt = float(compute_kurtosis(calib_activations @ r_kron_final).item())
    return r_kron_final, r_p.detach(), init_kurt, fixed_h16_kurt, final_kron_kurt


def round_f64_to_bf16_rne(y: torch.Tensor) -> torch.Tensor:
    """Round FP64 values directly to BF16 (round-half-even), with no intermediate FP32 rounding (normal + subnormal)."""
    y = y.to(torch.float64)
    _, e = torch.frexp(y)                         # y = m * 2^e, 0.5 <= |m| < 1
    ulp_exp = torch.clamp(e.to(torch.int32) - 8, min=-133)
    q = torch.round(torch.ldexp(y, -ulp_exp))     # 8 significant bits (or 2^-133 subnormal quantum), half-to-even
    r = torch.ldexp(q, ulp_exp)                   # exact in FP64, FP32, and BF16
    r = torch.where(torch.isfinite(y), r, y)
    return r.to(torch.float32).to(torch.bfloat16)  # both conversions exact


def absmean_scale_bf16(w_groups: torch.Tensor, floor: float = None) -> torch.Tensor:
    """
    BF16 g128 scale = mean|w| over the last dim.
    Scale rule shared by every packer and the deploy-basis QAT (docs/12 4.19): mean|w| accumulated in FP64, then
    rounded ONCE, directly from FP64 to BF16 with round-half-even. An FP32 sum is not order-defined (torch sums
    pairwise, the C++ packer sequentially), and near a BF16 rounding boundary the two orders picked different
    scales (one group of 1,024 in G1). The FP64 sums still differ by about 1e-16 relative, so the rule can
    disagree only if the exact mean lies that close to a BF16 tie.
    """
    m = w_groups.to(torch.float64).abs().sum(dim=-1) / float(w_groups.shape[-1])
    if floor is not None:
        m = m.clamp(min=floor)
    return round_f64_to_bf16_rne(m)


def quantize_and_pack_w1_g128_interleaved(
    weight: torch.Tensor,
    R1: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merges optional orthogonal rotation R1 (W_rot = W @ R1), quantizes into Binary g128
    (1 bit per weight + 1 BF16 scale per 128 weights = 1.125 bpw), and packs every 32 weights
    along K into a uint32 word in the Interleaved 16-Bit Split Layout:
      - even weights b_{2k}   -> bit k      (k in 0..15)
      - odd weights  b_{2k+1} -> bit k + 16 (k in 0..15)
    Returns:
      packed_w1: int32 tensor of shape [N, K // 32] (holding unsigned 32-bit patterns)
      scales_bf16: bfloat16 tensor of shape [N, K // 128]
    """
    w = weight.to(torch.float32)
    if R1 is not None:
        w = w @ R1.to(torch.float32)

    N, K = w.shape
    assert K % 128 == 0, "K dimension must be a multiple of 128 for g128 quantization"

    w_groups = w.view(N, K // 128, 128)
    scales_bf16 = absmean_scale_bf16(w_groups)

    # Binary sign bits: 1 for w >= 0 (+1.0), 0 for w < 0 (-1.0)
    sign_bits = (w >= 0).to(torch.int64).view(N, K // 32, 16, 2)
    even_bits = sign_bits[:, :, :, 0]  # [N, K // 32, 16]
    odd_bits = sign_bits[:, :, :, 1]   # [N, K // 32, 16]

    shifts = torch.arange(16, dtype=torch.int64, device=w.device).view(1, 1, 16)
    packed_u64 = ((even_bits << shifts) | (odd_bits << (shifts + 16))).sum(dim=-1)
    packed_w1 = packed_u64.to(torch.uint32 if hasattr(torch, "uint32") else torch.int32)
    return packed_w1, scales_bf16


def unpack_w1_g128_lop3_reference(
    packed_w1: torch.Tensor,
    scales_bf16: torch.Tensor,
) -> torch.Tensor:
    """
    Emulates the exact 3-instruction PTX sequence:
      shl.b32       %tmp, %w, (15 - k);
      lop3.b32      %bf162, %tmp, 0x80008000, 0x3F803F80, 0xAE;
      fma.rn.bf16x2 %bf162, %bf162, %scale, 0x80008000;
    and returns the dequantized BF16 weight matrix of shape [N, K].
    """
    N, k_words = packed_w1.shape
    K = k_words * 32
    w_u64 = packed_w1.to(torch.int64) & 0xFFFFFFFF

    out_bf16 = torch.empty((N, k_words, 16, 2), dtype=torch.bfloat16, device=packed_w1.device)
    for k in range(16):
        shifted = (w_u64 << (15 - k)) & 0xFFFFFFFF
        # lop3.b32 with immLut = 0xAE: (~a & 0x80008000) | 0x3F803F80
        lop3_out = ((~shifted) & 0x80008000) | 0x3F803F80
        lo_u16 = (lop3_out & 0xFFFF).to(torch.int16)
        hi_u16 = ((lop3_out >> 16) & 0xFFFF).to(torch.int16)
        out_bf16[:, :, k, 0] = lo_u16.view(torch.bfloat16)
        out_bf16[:, :, k, 1] = hi_u16.view(torch.bfloat16)

    unscaled = out_bf16.view(N, K // 128, 128)
    scaled = (unscaled.to(torch.float32) * scales_bf16.unsqueeze(-1).to(torch.float32)).to(torch.bfloat16)
    return scaled.view(N, K)


def quantize_and_pack_w2_ternary_interleaved(
    weight: torch.Tensor,
    R1: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes weights into BitNet b1.58 ternary values w_q in {-1, 0, +1}, encodes as
    q = w_q + 1 in {0, 1, 2}, and packs 16 ternary weights per uint32 word in the
    Interleaved 16-Bit Split Layout:
      - even q_{2k}   -> bits [2k+1 : 2k]     in lower 16 bits (k in 0..7)
      - odd  q_{2k+1} -> bits [2k+17 : 2k+16] in upper 16 bits (k in 0..7)
    """
    w = weight.to(torch.float32)
    if R1 is not None:
        w = w @ R1.to(torch.float32)

    N, K = w.shape
    assert K % 128 == 0, "K dimension must be a multiple of 128"

    w_groups = w.view(N, K // 128, 128)
    scales_bf16 = absmean_scale_bf16(w_groups, floor=1e-6)
    scales_f32 = scales_bf16.to(torch.float32).unsqueeze(-1)

    # Non-finite ratio (docs/12 4.21): an Inf weight gives an Inf scale and Inf / Inf = NaN, and NaN -> int64 is
    # undefined. Keep the sign of the weight instead (NaN weight -> 0), the same rule as the C++ packer.
    ratio = w_groups / scales_f32
    w_ternary = torch.where(torch.isfinite(ratio), torch.clamp(torch.round(ratio), -1.0, 1.0),
                            torch.sign(torch.nan_to_num(w_groups, nan=0.0))).to(torch.int64)
    q_codes = (w_ternary + 1).view(N, K // 16, 8, 2)  # values in {0, 1, 2}
    even_q = q_codes[:, :, :, 0]
    odd_q = q_codes[:, :, :, 1]

    shifts = (torch.arange(8, dtype=torch.int64, device=w.device) * 2).view(1, 1, 8)
    packed_u64 = ((even_q << shifts) | (odd_q << (shifts + 16))).sum(dim=-1)
    packed_w2 = packed_u64.to(torch.int32)
    return packed_w2, scales_bf16


def unpack_w2_ternary_lop3_reference(
    packed_w2: torch.Tensor,
    scales_bf16: torch.Tensor,
) -> torch.Tensor:
    """
    Emulates the exact PTX sequence:
      shr.u32       %tmp, %w, (2 * k);
      lop3.b32      %inj, %tmp, 0x00030003, 0x40004000, 0xEA;
      fma.rn.bf16x2 %unit, %inj, 0x42804280 (64.0), 0xC301C301 (-129.0);
      fma.rn.bf16x2 %out, %unit, %scale, 0x80008000;
    Guarantees ternary zero (q=1) unpacks to bit-exact 0.0f across all BF16 scales.
    """
    N, k_words = packed_w2.shape
    K = k_words * 16
    w_u64 = packed_w2.to(torch.int64) & 0xFFFFFFFF

    out_bf16 = torch.empty((N, k_words, 8, 2), dtype=torch.bfloat16, device=packed_w2.device)
    for k in range(8):
        shifted = (w_u64 >> (2 * k)) & 0xFFFFFFFF
        lop3_out = (shifted & 0x00030003) | 0x40004000
        lo_u16 = (lop3_out & 0xFFFF).to(torch.int16).view(torch.bfloat16)
        hi_u16 = ((lop3_out >> 16) & 0xFFFF).to(torch.int16).view(torch.bfloat16)
        unit_lo = (lo_u16.to(torch.float32) * 64.0 - 129.0).to(torch.bfloat16)
        unit_hi = (hi_u16.to(torch.float32) * 64.0 - 129.0).to(torch.bfloat16)
        out_bf16[:, :, k, 0] = unit_lo
        out_bf16[:, :, k, 1] = unit_hi

    unscaled = out_bf16.view(N, K // 128, 128)
    scaled = (unscaled.to(torch.float32) * scales_bf16.unsqueeze(-1).to(torch.float32)).to(torch.bfloat16)
    return scaled.view(N, K)


if HAS_TRITON:
    @triton.jit
    def fused_w1_g128_gemm_kernel(
        x_ptr, pw_ptr, sc_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_sn, stride_sk,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        word_in_grp = offs_k >> 5
        bit_pos = ((offs_k >> 1) & 15) + ((offs_k & 1) * 16)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        num_groups = K // BLOCK_K
        for g in range(0, num_groups):
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (g * BLOCK_K + offs_k[None, :]) * stride_xk
            x_mask = (offs_m[:, None] < M) & ((g * BLOCK_K + offs_k[None, :]) < K)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.bfloat16)

            pw_ptrs = pw_ptr + offs_n[:, None] * stride_wn + (g * 4 + word_in_grp[None, :]) * stride_wk
            pw_mask = (offs_n[:, None] < N) & ((g * 4 + word_in_grp[None, :]) < (K // 32))
            p_tile = tl.load(pw_ptrs, mask=pw_mask, other=0)

            sc_ptrs = sc_ptr + offs_n * stride_sn + g * stride_sk
            s_tile = tl.load(sc_ptrs, mask=(offs_n < N), other=0.0).to(tl.bfloat16)

            sign_bit = (p_tile >> bit_pos[None, :]) & 1
            w_bf16 = tl.where(sign_bit != 0, s_tile[:, None], -s_tile[:, None])
            acc = tl.dot(x_tile, tl.trans(w_bf16), acc=acc, out_dtype=tl.float32)

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=y_mask)

    @triton.jit
    def fused_w2_ternary_gemm_kernel(
        x_ptr, pw_ptr, sc_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_sn, stride_sk,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        word_in_grp = offs_k >> 4
        bit_pos = (((offs_k >> 1) & 7) * 2) + ((offs_k & 1) * 16)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        num_groups = K // BLOCK_K
        for g in range(0, num_groups):
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (g * BLOCK_K + offs_k[None, :]) * stride_xk
            x_mask = (offs_m[:, None] < M) & ((g * BLOCK_K + offs_k[None, :]) < K)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.bfloat16)

            pw_ptrs = pw_ptr + offs_n[:, None] * stride_wn + (g * 8 + word_in_grp[None, :]) * stride_wk
            pw_mask = (offs_n[:, None] < N) & ((g * 8 + word_in_grp[None, :]) < (K // 16))
            p_tile = tl.load(pw_ptrs, mask=pw_mask, other=0)

            sc_ptrs = sc_ptr + offs_n * stride_sn + g * stride_sk
            s_tile = tl.load(sc_ptrs, mask=(offs_n < N), other=0.0).to(tl.float32)

            code = ((p_tile >> bit_pos[None, :]) & 3).to(tl.float32) - 1.0
            w_bf16 = (code * s_tile[:, None]).to(tl.bfloat16)
            acc = tl.dot(x_tile, tl.trans(w_bf16), acc=acc, out_dtype=tl.float32)

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=y_mask)


def _select_packed_gemm_launch(device: torch.device, M: int) -> Tuple[int, int, int]:
    block_m = 16 if M <= 16 else (32 if M <= 32 else 64)
    block_n = 64
    if not torch.cuda.is_available() or device.type != "cuda":
        return block_m, block_n, 1
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev_idx)
    smem_limit = int(
        getattr(props, "shared_memory_per_block_optin", 0)
        or getattr(props, "shared_memory_per_block", 49152)
    )
    # x_tile [BM, 128] bf16 + p_tile [64, 128] int32 (32 KB)
    stage_bytes = (block_m * 128 * 2) + (block_n * 128 * 4)
    num_stages = 2 if (2 * stage_bytes <= smem_limit) else 1
    return block_m, block_n, num_stages


def fused_w1_g128_gemm(
    x_bf16: torch.Tensor,
    packed_w1: torch.Tensor,
    scales_bf16: torch.Tensor,
) -> torch.Tensor:
    """
    Zero-materialization inference GEMM Y = X @ W_hat^T directly from 1.125 bpw W1 packed words.
    """
    assert x_bf16.ndim == 2 and packed_w1.ndim == 2
    M, K = x_bf16.shape
    N, k_words = packed_w1.shape
    assert K == k_words * 32 and K % 128 == 0
    if HAS_TRITON and x_bf16.is_cuda and packed_w1.is_cuda:
        pw_i32 = packed_w1.view(torch.int32)
        sc = scales_bf16.to(torch.bfloat16)
        y = torch.empty((M, N), device=x_bf16.device, dtype=torch.bfloat16)
        bm, bn, stages = _select_packed_gemm_launch(x_bf16.device, M)
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
        fused_w1_g128_gemm_kernel[grid](
            x_bf16, pw_i32, sc, y,
            M, N, K,
            x_bf16.stride(0), x_bf16.stride(1),
            pw_i32.stride(0), pw_i32.stride(1),
            sc.stride(0), sc.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, num_stages=stages,
        )
        return y
    w_bf16 = unpack_w1_g128_lop3_reference(packed_w1, scales_bf16)
    if x_bf16.is_cuda and w_bf16.is_cuda and x_bf16.dtype == torch.bfloat16:
        return torch.matmul(x_bf16, w_bf16.T)
    return (x_bf16.to(torch.float32) @ w_bf16.to(torch.float32).T).to(torch.bfloat16)


def fused_w2_ternary_gemm(
    x_bf16: torch.Tensor,
    packed_w2: torch.Tensor,
    scales_bf16: torch.Tensor,
) -> torch.Tensor:
    """
    Zero-materialization inference GEMM Y = X @ W_hat^T directly from 2.125 bpw W2 ternary packed words.
    """
    assert x_bf16.ndim == 2 and packed_w2.ndim == 2
    M, K = x_bf16.shape
    N, k_words = packed_w2.shape
    assert K == k_words * 16 and K % 128 == 0
    if HAS_TRITON and x_bf16.is_cuda and packed_w2.is_cuda:
        pw_i32 = packed_w2.view(torch.int32)
        sc = scales_bf16.to(torch.bfloat16)
        y = torch.empty((M, N), device=x_bf16.device, dtype=torch.bfloat16)
        bm, bn, stages = _select_packed_gemm_launch(x_bf16.device, M)
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
        fused_w2_ternary_gemm_kernel[grid](
            x_bf16, pw_i32, sc, y,
            M, N, K,
            x_bf16.stride(0), x_bf16.stride(1),
            pw_i32.stride(0), pw_i32.stride(1),
            sc.stride(0), sc.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, num_stages=stages,
        )
        return y
    w_bf16 = unpack_w2_ternary_lop3_reference(packed_w2, scales_bf16)
    if x_bf16.is_cuda and w_bf16.is_cuda and x_bf16.dtype == torch.bfloat16:
        return torch.matmul(x_bf16, w_bf16.T)
    return (x_bf16.to(torch.float32) @ w_bf16.to(torch.float32).T).to(torch.bfloat16)


def main() -> None:
    print("================================================================================")
    print("  WORKSTREAM 3: SPINQUANT CAYLEY-SGD ROTATION & INTERLEAVED BIT-PACKER")
    print("================================================================================\n")

    torch.manual_seed(42)
    M, K, N = 64, 128, 64
    # Create heavy-tailed synthetic activations with channel outliers (kurtosis >> 3)
    x = torch.randn((M, K), dtype=torch.float32)
    x[:, 0] *= 18.0
    x[:, 17] *= 14.0

    R1, k_before, k_after = optimize_stiefel_rotation(x, steps=20, lr=0.25)
    # docs/12 4.16: the initial rotation alone already flattens most of the kurtosis, so `k_after < k_before`
    # would pass even with a broken optimiser. Measure the 0-step kurtosis of the same initial rotation and require
    # the Cayley-SGD steps themselves to improve on it.
    _, _, k_init_rot = optimize_stiefel_rotation(x, steps=0, lr=0.25)
    orth_err = float(torch.max(torch.abs(R1.T @ R1 - torch.eye(K))).item())

    w = torch.randn((N, K), dtype=torch.float32)
    packed_w1, scales_bf16 = quantize_and_pack_w1_g128_interleaved(w, R1=R1)
    unpacked_w1 = unpack_w1_g128_lop3_reference(packed_w1, scales_bf16)

    # Direct reference quantization check
    w_rot = w @ R1
    ref_scales = absmean_scale_bf16(w_rot.view(N, 1, 128))
    ref_signs = torch.where(w_rot >= 0, 1.0, -1.0).view(N, 1, 128)
    ref_w1 = (ref_signs * ref_scales.to(torch.float32).unsqueeze(-1)).to(torch.bfloat16).view(N, K)

    bit_errors = int((unpacked_w1 != ref_w1).sum().item())

    # Also verify W2 ternary packing and exact zero-preserving lop3.b32 (0xEA) unpacking
    packed_w2, w2_scales_bf16 = quantize_and_pack_w2_ternary_interleaved(w, R1=R1)
    unpacked_w2 = unpack_w2_ternary_lop3_reference(packed_w2, w2_scales_bf16)
    w2_trits = torch.clamp(torch.round(w_rot.view(N, 1, 128) / w2_scales_bf16.to(torch.float32).unsqueeze(-1)), -1.0, 1.0)
    ref_w2 = (w2_trits * w2_scales_bf16.to(torch.float32).unsqueeze(-1)).to(torch.bfloat16).view(N, K)
    w2_bit_errors = int((unpacked_w2 != ref_w2).sum().item())

    print(f"[1] Stiefel Manifold Orthogonality Error ||R^T R - I||_inf : {orth_err:.3e}")
    print(f"[2] Activation Kurtosis Reduction (Outlier Flattening)     : {k_before:.2f} -> {k_init_rot:.2f} (initial rotation) "
          f"-> {k_after:.2f} (after 20 Cayley-SGD steps)")
    print(f"[3] Interleaved 16-Bit Split W1 & W2 Bit Mismatches        : W1={bit_errors}, W2={w2_bit_errors} (across {N*K} weights)")
    assert orth_err < 1e-5
    assert k_after < k_before
    assert k_after < k_init_rot - 0.25, (k_init_rot, k_after)   # the optimiser itself must contribute
    assert bit_errors == 0 and w2_bit_errors == 0
    print("[+] PASS: Stiefel Cayley-SGD rotation and Interleaved lop3.b32 bit-packing verified.")

    # Consistency between the rotation the KERNEL applies online and the one the PACKER merges offline
    xs = torch.randn((16, K), dtype=torch.float32)
    y_ref = xs @ w.T
    for mode in (1, 2):
        Rk = build_kernel_online_rotation(K, mode)
        orth = float(torch.max(torch.abs(Rk.T @ Rk - torch.eye(K))).item())
        y_rot = (xs @ Rk) @ (w @ Rk).T
        err = float(torch.max(torch.abs(y_rot - y_ref)).item())
        assert orth < 1e-5 and err < 1e-3, (mode, orth, err)
        print(f"[4] Kernel-matched online rotation mode {mode}: orthogonality {orth:.2e}, (xR)(WR)^T vs xW^T max err {err:.2e}")
    # Negative control: the dense Cayley R_1 used above is NOT what the kernel applies, so it must not match
    Rk1 = build_kernel_online_rotation(K, 1)
    mismatch = float(torch.max(torch.abs((xs @ Rk1) @ (w @ R1).T - y_ref)).item())
    assert mismatch > 1e-2, "dense R_1 unexpectedly equals the kernel rotation"
    print(f"[5] Negative control (kernel H_16 on x, dense Cayley R_1 on W): error {mismatch:.2f} -> mixing them is WRONG")

    # [6] Factorized Kronecker-Stiefel Cayley-SGD rotation: R_kron = (R_P (x) I_16) @ (I_P (x) H_16)
    R_kron, R_P, k_init, k_h16, k_kron = optimize_kronecker_stiefel_rotation(x, q_hadamard=16, steps=25, lr=0.25)
    kron_orth_err = float(torch.max(torch.abs(R_kron.T @ R_kron - torch.eye(K))).item())
    # Verify exact factorization: (xs @ (R_P (x) I_16) @ Rk1) @ (w @ R_kron).T == xs @ w.T
    r_prev_absorb = torch.kron(R_P, torch.eye(16, dtype=torch.float32))
    x_after_prev_and_kernel_h16 = (xs @ r_prev_absorb) @ Rk1
    y_kron = x_after_prev_and_kernel_h16 @ (w @ R_kron).T
    kron_match_err = float(torch.max(torch.abs(y_kron - y_ref)).item())
    print(f"[6] Kronecker-Stiefel (R_P (x) H_16, P={K//16}): kurtosis {k_init:.2f} -> {k_h16:.2f} (fixed H_16) -> {k_kron:.2f} (learned R_P), match err {kron_match_err:.2e}")
    assert kron_orth_err < 1e-5 and k_kron < k_h16 and kron_match_err < 1e-3
    print("[+] PASS: Kronecker-Stiefel Cayley-SGD rotation and online H_16 prologue factorization verified.")

    # [7] Single-step FP64 -> BF16 RNE rounding (docs/12 4.19): verify both normal and subnormal double-rounding traps
    ks = torch.arange(1, 128, dtype=torch.float64)
    sub_traps = (ks + 0.5 - 2.0 ** -10) * (2.0 ** -133)
    sub_got = round_f64_to_bf16_rne(sub_traps).to(torch.float64)
    sub_want = (ks * (2.0 ** -133)).to(torch.bfloat16).to(torch.float64)
    assert torch.equal(sub_got, sub_want), "subnormal FP64 -> BF16 double-rounding regression"
    print(f"[7] Unified round_f64_to_bf16_rne: 0/{sub_traps.numel()} subnormal double-rounding traps missed -> PASS")

    # [8] docs/12 4.21: non-finite weights pack to the sign of the weight (+Inf -> +1, -Inf -> -1, NaN -> 0).
    w_nf = torch.randn(1, 128)
    w_nf[0, 3], w_nf[0, 4], w_nf[0, 5] = float("inf"), float("-inf"), float("nan")
    p_nf, _ = quantize_and_pack_w2_ternary_interleaved(w_nf)
    word0 = int(p_nf[0, 0]) & 0xFFFFFFFF
    codes = [(word0 >> (2 * (k // 2) + 16 * (k % 2))) & 3 for k in (3, 4, 5)]
    assert codes == [2, 0, 1], codes
    print(f"[8] W2 packer non-finite weights (+Inf, -Inf, NaN) -> codes {codes} (want [2, 0, 1]) -> PASS")

    # [9] Fused W1 / W2 packed inference GEMM parity (reads uint32/int32 packed words directly)
    xg_inf = torch.randn((17, 256), dtype=torch.bfloat16)
    wg_inf = torch.randn((128, 256), dtype=torch.float32)
    pw1_g, sc1_g = quantize_and_pack_w1_g128_interleaved(wg_inf)
    pw2_g, sc2_g = quantize_and_pack_w2_ternary_interleaved(wg_inf)
    y1_fus = fused_w1_g128_gemm(xg_inf, pw1_g, sc1_g)
    y1_ref = (xg_inf.float() @ unpack_w1_g128_lop3_reference(pw1_g, sc1_g).float().T).to(torch.bfloat16)
    y2_fus = fused_w2_ternary_gemm(xg_inf, pw2_g, sc2_g)
    y2_ref = (xg_inf.float() @ unpack_w2_ternary_lop3_reference(pw2_g, sc2_g).float().T).to(torch.bfloat16)
    e1_g = float((y1_fus.float() - y1_ref.float()).norm() / (y1_ref.float().norm() + 1e-8))
    e2_g = float((y2_fus.float() - y2_ref.float()).norm() / (y2_ref.float().norm() + 1e-8))
    assert e1_g < 1.5e-2 and e2_g < 1.5e-2, (e1_g, e2_g)
    print(f"[9] Fused W1/W2 packed inference GEMM parity: W1_rel={e1_g:.3e}, W2_rel={e2_g:.3e} -> PASS")


if __name__ == "__main__":
    main()
