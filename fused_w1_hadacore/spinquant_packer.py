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
    scales_bf16 = w_groups.abs().mean(dim=-1).to(torch.bfloat16)

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
    scales_bf16 = w_groups.abs().mean(dim=-1).clamp(min=1e-6).to(torch.bfloat16)
    scales_f32 = scales_bf16.to(torch.float32).unsqueeze(-1)

    w_ternary = torch.clamp(torch.round(w_groups / scales_f32), -1.0, 1.0).to(torch.int64)
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
    ref_scales = w_rot.view(N, 1, 128).abs().mean(dim=-1).to(torch.bfloat16)
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


if __name__ == "__main__":
    main()
