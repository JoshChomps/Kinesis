#!/usr/bin/env python3
"""
File: sim_harness/smsp_roofline_sim.py
Cycle-Accurate SMSP Roofline & Scoreboard Simulator for GA102 (sm_86) and AD102 (sm_89).
Models:
  - 4 SMSP warp schedulers per SM (N_SMSP = 4)
  - 32-bank Shared Memory crossbar (32 banks x 4B = 128 Bytes/cycle/SM)
  - cp.async.cg.shared.global L2/DRAM multi-stage arrival pipeline
  - ldmatrix.sync.aligned.m8n8.x4.shared.b16 (512 Bytes = 4 SM cycles conflict-free)
  - HMMA.16816.F32 Tensor Core issue interval: tau = 16 cycles (Locked) vs tau = 8 cycles (Unlocked)
  - Fused W1A16_F32 (Binary g128, 1.125 bpw, 14.22x compression) vs Dense W16A16 (BF16)
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class GPUArchSpec:
    name: str
    arch: str
    sm_count: int
    clock_mhz: float
    dram_bw_gbps: float
    sram_bytes_per_sm: int = 102400       # 100 KB total SRAM per SM
    max_dyn_sram_bytes: int = 101376      # 99 KB opt-in dynamic Shared Memory
    smsp_per_sm: int = 4
    sram_banks: int = 32
    bytes_per_bank_per_cycle: int = 4

    @property
    def sram_bw_bytes_per_cycle(self) -> int:
        return self.sram_banks * self.bytes_per_bank_per_cycle  # 128 B/cycle/SM

    def peak_tflops(self, tau_cycles: int) -> float:
        # Locked (tau=16): 512 FLOPs/cycle/SM; Unlocked (tau=8): 1,024 FLOPs/cycle/SM
        flops_per_sm_per_cycle = (2048.0 / tau_cycles) * self.smsp_per_sm
        return (self.sm_count * flops_per_sm_per_cycle * (self.clock_mhz * 1e6)) / 1e12


GA102_RTX_3090 = GPUArchSpec(
    name="RTX 3090 (GA102)",
    arch="sm_86",
    sm_count=82,
    clock_mhz=1691.2,
    dram_bw_gbps=936.2,
)

GA102_RTX_3090_TI = GPUArchSpec(
    name="RTX 3090 Ti (GA102)",
    arch="sm_86",
    sm_count=84,
    clock_mhz=1830.0,
    dram_bw_gbps=1008.0,
)

AD102_RTX_4090 = GPUArchSpec(
    name="RTX 4090 (AD102)",
    arch="sm_89",
    sm_count=128,
    clock_mhz=2530.0,
    dram_bw_gbps=1008.0,
)


@dataclass(frozen=True)
class TileConfig:
    label: str
    block_m: int
    block_n: int
    block_k: int
    warp_m: int
    warp_n: int
    num_warps: int
    num_stages: int
    beta_conflict: float = 1.0
    fused_w1a16: bool = False

    @property
    def dynamic_smem_bytes(self) -> int:
        if self.fused_w1a16:
            # Activations A in BF16 (2B), Weights B in 1.125 bpw (9/64 bytes/weight)
            a_bytes = self.block_m * self.block_k * 2
            b_bytes = int((self.block_n * self.block_k * 9) // 64)
            return (a_bytes + b_bytes) * self.num_stages
        return (self.block_m + self.block_n) * self.block_k * 2 * self.num_stages

    @property
    def warp_arithmetic_intensity(self) -> float:
        # FLOPs per K_step=16 is 2 * W_M * W_N * 16 = 32 * W_M * W_N
        flops = 32.0 * self.warp_m * self.warp_n
        if self.fused_w1a16:
            # Over K_step=16: A is W_M * 16 * 2B = 32 * W_M bytes;
            # B is compressed 14.222x: (32 * W_N) / 14.2222 = 2.25 * W_N bytes
            sram_bytes = 32.0 * self.warp_m + 2.25 * self.warp_n
        else:
            sram_bytes = 32.0 * (self.warp_m + self.warp_n)
        return flops / sram_bytes


def simulate_smsp_cycle_accurate(
    gpu: GPUArchSpec,
    cfg: TileConfig,
    tau_cycles: int,
    k_iters: int = 64,
) -> Dict[str, float]:
    """
    Cycle-accurate discrete-event simulation of 1 SM (4 SMSPs, 32-bank SRAM crossbar,
    multi-stage cp.async pipeline, ldmatrix unit, and Tensor Core MMA pipe).
    Simulates k_iters steps of K_step = 16.
    """
    hmma_per_warp_step = (cfg.warp_m // 16) * (cfg.warp_n // 8)
    # Analytical cycle equations per K_step = 16 across N_w warps:
    if cfg.fused_w1a16:
        bytes_ldmatrix = cfg.num_warps * (32.0 * cfg.warp_m + 2.25 * cfg.warp_n)
    else:
        bytes_ldmatrix = cfg.num_warps * 32.0 * (cfg.warp_m + cfg.warp_n)

    t_sram_step = (bytes_ldmatrix / gpu.sram_bw_bytes_per_cycle) * cfg.beta_conflict
    t_mma_step = (cfg.num_warps * hmma_per_warp_step / gpu.smsp_per_sm) * tau_cycles

    # Discrete-event simulation across k_iters steps with cp.async stage ring buffer
    sram_Crossbar_free_cycle = 0.0
    smsp_free_cycle = [0.0 for _ in range(gpu.smsp_per_sm)]
    total_smsp_busy_cycles = 0.0
    total_smsp_stall_cycles = 0.0

    # Pipeline stage latency penalty when num_stages < 4 on unlocked core (tau=8)
    stage_refill_latency = max(0.0, (4 - cfg.num_stages) * 14.0 * (16.0 / tau_cycles))

    for step in range(k_iters):
        # 1. cp.async stage arrival time
        stage_ready_cycle = 0.0
        if step >= cfg.num_stages:
            stage_ready_cycle = sram_Crossbar_free_cycle + stage_refill_latency

        # 2. Warps issue ldmatrix across shared 32-bank SRAM crossbar once both the crossbar
        #    and the earliest warp scheduler are ready for the next K_step=16 wave
        earliest_smsp_ready = min(smsp_free_cycle)
        ldm_start = max(sram_Crossbar_free_cycle, stage_ready_cycle, earliest_smsp_ready)
        ldm_done = ldm_start + t_sram_step
        sram_Crossbar_free_cycle = ldm_done

        # 3. Warps assigned to the 4 SMSPs execute HMMA.16816.F32 instructions
        #    In fused W1A16_F32, ALU unpack (shl.b32 + lop3.b32 + fma.rn.bf16x2) dual-issues
        #    on the INT32/FP16 pipes concurrently with HMMA on the Tensor Core pipe.
        mma_cycles_per_smsp = t_mma_step
        for s in range(gpu.smsp_per_sm):
            Operand_ready = ldm_start + (t_sram_step * ((s + 1) / gpu.smsp_per_sm))
            if Operand_ready > smsp_free_cycle[s]:
                stall = Operand_ready - smsp_free_cycle[s]
                total_smsp_stall_cycles += stall
                smsp_free_cycle[s] = Operand_ready
            smsp_free_cycle[s] += mma_cycles_per_smsp
            total_smsp_busy_cycles += mma_cycles_per_smsp

    total_wall_cycles = max(smsp_free_cycle)
    total_smsp_wall_slots = total_wall_cycles * gpu.smsp_per_sm
    tc_utilization_pct = (total_smsp_busy_cycles / total_smsp_wall_slots) * 100.0
    sass_wait_stall_pct = (total_smsp_stall_cycles / total_smsp_wall_slots) * 100.0
    achieved_tflops = gpu.peak_tflops(tau_cycles) * (tc_utilization_pct / 100.0)

    return {
        "t_mma_cycles": t_mma_step,
        "t_sram_cycles": t_sram_step,
        "compute_to_sram_ratio": t_mma_step / t_sram_step,
        "tc_util_pct": tc_utilization_pct,
        "sass_wait_pct": sass_wait_stall_pct,
        "achieved_tflops": achieved_tflops,
        "smem_kb": cfg.dynamic_smem_bytes / 1024.0,
    }


def evaluate_dram_ridge_points(gpu: GPUArchSpec, tau_cycles: int = 8) -> Dict[str, float]:
    peak_flops = gpu.peak_tflops(tau_cycles) * 1e12
    dram_bw = gpu.dram_bw_gbps * 1e9
    # Standard W16A16: 2 bytes per weight -> ridge point in FLOPs/Byte = Peak_FLOPs / DRAM_BW
    # Since each weight does 2*M FLOPs for 2 bytes read, FLOPs/Byte = M -> M_ridge_bf16 = Peak / DRAM_BW
    ridge_w16a16 = peak_flops / dram_bw
    # Fused Binary g128 (1.125 bits/weight = 14.222x compression):
    compression_ratio = 16.0 / 1.125
    effective_bw_tbps = (gpu.dram_bw_gbps * compression_ratio) / 1000.0
    ridge_w1a16 = ridge_w16a16 / compression_ratio
    return {
        "peak_tflops": gpu.peak_tflops(tau_cycles),
        "compression_ratio": compression_ratio,
        "effective_bw_tbps": effective_bw_tbps,
        "ridge_w16a16": ridge_w16a16,
        "ridge_w1a16": ridge_w1a16,
    }


# ------------------------------------------------------------------------------------------------
# Small-batch decode: wave quantization (2D tiling) vs Marlin-style Stream-K / Split-K
# Prior art: Frantar et al., MARLIN (arXiv:2408.11743). THIS IS AN ANALYTICAL MODEL WITH LABELED ASSUMPTIONS,
# NOT A MEASUREMENT. Its only job is to show which regime the 2D tile grid leaves SMs idle.
# ------------------------------------------------------------------------------------------------

# ASSUMPTION: a decode GEMV is DRAM-bound and one SM can only pull a fraction of DRAM bandwidth; the GPU
# needs roughly this fraction of its SMs active to saturate DRAM. Real values vary by part and access pattern.
SM_FRACTION_TO_SATURATE_DRAM = 0.5


def decode_gemv_time_us(gpu: GPUArchSpec, m: int, n: int, k: int, bytes_per_weight: float,
                        active_sms: int, extra_dram_bytes: float = 0.0) -> float:
    """DRAM-bound time (us) to stream the weights once: bandwidth scales with active SMs up to saturation."""
    weight_bytes = n * k * bytes_per_weight
    saturating_sms = gpu.sm_count * SM_FRACTION_TO_SATURATE_DRAM
    bw_eff = gpu.dram_bw_gbps * 1e9 * min(1.0, active_sms / saturating_sms)
    return (weight_bytes + extra_dram_bytes) / bw_eff * 1e6


def compare_2d_tiling_vs_stream_k(gpu: GPUArchSpec, m: int, n: int, k: int,
                                  bytes_per_weight: float = 9.0 / 64.0,
                                  bm: int = 128, bn: int = 256) -> Dict[str, float]:
    ctas_2d = ((n + bn - 1) // bn) * ((m + bm - 1) // bm)
    active_2d = min(ctas_2d, gpu.sm_count)
    t_2d = decode_gemv_time_us(gpu, m, n, k, bytes_per_weight, active_2d)

    # Stream-K: BM = 16 (no wasted rows for M <= 16) and K is split so that every SM owns a contiguous slice
    n_tiles = (n + 63) // 64                      # BN = 64 per CTA in marlin_streamk_w1a16_decode_kernel
    splits = max(1, (gpu.sm_count * 2 + n_tiles - 1) // n_tiles)  # target 2 CTAs per SM
    ctas_sk = n_tiles * splits
    active_sk = min(ctas_sk, gpu.sm_count)
    # Split-K partials are accumulated by FP32 atomics in L2: model their cost as extra traffic of
    # M * N * 4 B * splits (an upper bound; it mostly stays in L2 rather than DRAM)
    t_sk = decode_gemv_time_us(gpu, m, n, k, bytes_per_weight, active_sk,
                               extra_dram_bytes=0.1 * m * n * 4.0 * splits)
    return {"ctas_2d": float(ctas_2d), "active_sms_2d": float(active_2d), "t_2d_us": t_2d,
            "ctas_sk": float(ctas_sk), "active_sms_sk": float(active_sk), "t_sk_us": t_sk,
            "speedup": t_2d / t_sk, "splits": float(splits)}


def simulate_hetero_cpu_gpu_roofline(
    weight_bytes: float,          # total compressed weight bytes streamed per token
    gpu_bw_gbps: float,           # effective GPU DRAM bandwidth
    cpu_bw_gbps: float,           # CPU DDR bandwidth available to the cold slice
    pcie_bw_gbps: float,          # effective PCIe bandwidth for activation dispatch / partial-sum merge
    vram_hot_bytes: float,        # VRAM budget that can hold hot weights
    m: int = 1, hidden: int = 4096, pcie_latency_us: float = 10.0,
) -> Dict[str, float]:
    """
    KTransformers / PowerInfer style split: a hot fraction f runs on the GPU, the rest on the CPU in parallel.
      t_gpu(f) = f * W / BW_gpu      t_cpu(f) = (1 - f) * W / BW_cpu + t_pcie
    The balanced split solves t_gpu = t_cpu; it is then capped by what fits in VRAM. Both slices are assumed
    purely bandwidth-bound, and the merge cost is one activation round trip (2 * M * D * 2 B + latency).
    """
    t_pcie = (2.0 * m * hidden * 2.0) / (pcie_bw_gbps * 1e9) + pcie_latency_us * 1e-6
    bg, bc = gpu_bw_gbps * 1e9, cpu_bw_gbps * 1e9
    f_bal = (weight_bytes / bc + t_pcie) / (weight_bytes / bg + weight_bytes / bc)
    f_bal = min(1.0, max(0.0, f_bal))
    f_cap = min(1.0, vram_hot_bytes / weight_bytes)
    f = min(f_bal, f_cap)
    t_gpu = f * weight_bytes / bg
    t_cpu = (1.0 - f) * weight_bytes / bc + t_pcie if f < 1.0 else 0.0
    return {"f_balanced": f_bal, "f_vram_cap": f_cap, "f_used": f,
            "t_gpu_ms": t_gpu * 1e3, "t_cpu_ms": t_cpu * 1e3, "t_token_ms": max(t_gpu, t_cpu) * 1e3,
            "t_pcie_us": t_pcie * 1e6, "bound_by": 0.0 if t_gpu >= t_cpu else 1.0}


def main() -> None:
    print("================================================================================")
    print("  WORKSTREAM 1: CYCLE-ACCURATE SMSP ROOFLINE & SCOREBOARD SIMULATOR")
    print("================================================================================\n")

    warp_tiles: List[TileConfig] = [
        TileConfig("Warp Tile 32x32 (Consumer Default)", 128, 64, 32, 32, 32, 4, 2, beta_conflict=1.18),
        TileConfig("Warp Tile 64x32 (Half-Rate cuBLAS)", 128, 128, 32, 64, 32, 4, 3, beta_conflict=1.12),
        TileConfig("Warp Tile 64x64 (Unlocked Optimal)", 128, 256, 32, 64, 64, 8, 4, beta_conflict=1.00),
        TileConfig("Warp Tile 128x64 (Max Reuse)",       256, 128, 32, 128, 64, 8, 4, beta_conflict=1.00),
        TileConfig("Warp Tile 64x64 Fused W1A16_F32",    128, 256, 32, 64, 64, 8, 4, beta_conflict=1.00, fused_w1a16=True),
    ]

    print("[1] Warp-Tile SRAM-to-Tensor-Core Roofline & Scoreboard Stall Analysis (AD102 RTX 4090):")
    print("-" * 114)
    print(f"{'Configuration':<35} | {'I_warp':>8} | {'State':<8} | {'T_MMA':>7} | {'T_SRAM':>7} | {'Ratio':>6} | {'SassWait':>8} | {'Achieved':>10}")
    print("-" * 114)

    for cfg in warp_tiles:
        for state_label, tau in [("Locked", 16), ("Unlocked", 8)]:
            res = simulate_smsp_cycle_accurate(AD102_RTX_4090, cfg, tau_cycles=tau)
            print(
                f"{cfg.label:<35} | {cfg.warp_arithmetic_intensity:>6.1f}   | {state_label:<8} | "
                f"{res['t_mma_cycles']:>7.1f} | {res['t_sram_cycles']:>7.1f} | "
                f"{res['compute_to_sram_ratio']:>5.2f}x | {res['sass_wait_pct']:>7.2f}% | "
                f"{res['achieved_tflops']:>7.1f} TF"
            )
    print("-" * 114)
    print()

    print("[2] Multi-Architecture Unlocked Roofline & Binary g128 (14.22x) DRAM Ridge Shift:")
    print("-" * 106)
    print(f"{'GPU Architecture':<22} | {'Phys BW':>10} | {'Eff W1 BW':>10} | {'Unlocked Peak':>14} | {'W16A16 Ridge':>14} | {'W1A16 Ridge':>14}")
    print("-" * 106)
    for gpu in [GA102_RTX_3090, GA102_RTX_3090_TI, AD102_RTX_4090]:
        r = evaluate_dram_ridge_points(gpu, tau_cycles=8)
        print(
            f"{gpu.name:<22} | {gpu.dram_bw_gbps:>7.1f} GB/s | {r['effective_bw_tbps']:>7.2f} TB/s | "
            f"{r['peak_tflops']:>10.1f} TF | M >= {r['ridge_w16a16']:>6.1f}   | M >= {r['ridge_w1a16']:>6.1f}"
        )
    print("-" * 106)
    print()

    print("[3] Small-batch decode, N = K = 4096, W1 (1.125 bpw): 2D tile grid vs Stream-K (ANALYTICAL MODEL, ASSUMPTION: "
          f"{int(SM_FRACTION_TO_SATURATE_DRAM * 100)}% of SMs saturate DRAM)")
    print("-" * 106)
    print(f"{'GPU':<20} | {'M':>3} | {'CTAs 2D':>7} | {'SMs 2D':>6} | {'t 2D us':>8} | {'CTAs SK':>7} | {'splits':>6} | {'t SK us':>8} | {'speedup':>7}")
    print("-" * 106)
    for gpu in [GA102_RTX_3090, AD102_RTX_4090]:
        for m in (1, 4, 8, 16, 32):
            r = compare_2d_tiling_vs_stream_k(gpu, m, 4096, 4096)
            print(f"{gpu.name:<20} | {m:>3} | {int(r['ctas_2d']):>7} | {int(r['active_sms_2d']):>6} | {r['t_2d_us']:>8.2f} | "
                  f"{int(r['ctas_sk']):>7} | {int(r['splits']):>6} | {r['t_sk_us']:>8.2f} | {r['speedup']:>6.2f}x")
    print("-" * 106)
    print()

    print("[4] Hetero CPU+GPU split (KTransformers / PowerInfer style), W = 6 GB per token, RTX 4090, DDR5 80 GB/s, PCIe 25 GB/s")
    print("-" * 106)
    for vram_gb in (2.0, 4.0, 6.0):
        h = simulate_hetero_cpu_gpu_roofline(6e9, 1008.0 * 0.85, 80.0, 25.0, vram_gb * 1e9)
        who = "GPU" if h["bound_by"] == 0.0 else "CPU"
        print(f"VRAM hot budget {vram_gb:>3.0f} GB: balanced f = {h['f_balanced']:.3f}, used f = {h['f_used']:.3f} "
              f"| t_gpu = {h['t_gpu_ms']:.2f} ms, t_cpu = {h['t_cpu_ms']:.2f} ms, token = {h['t_token_ms']:.2f} ms ({who}-bound)")
    print("-" * 106)


if __name__ == "__main__":
    main()
