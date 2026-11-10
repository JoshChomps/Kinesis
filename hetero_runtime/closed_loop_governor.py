#!/usr/bin/env python3
"""
File: hetero_runtime/closed_loop_governor.py
Workstream 7B: Closed-Loop Hardware Telemetry Governor (Chorus Queue-Pressure Scheduler +
NVRM Post-Arbitration Clock-Floor & Tiled-CBC Sparsity Controller).
Implements:
  1. Queue-Pressure-Guided AMX Offload (Chorus Policy):
     Inspects Delta_Q = gpu_ready_seq - cpu_done_seq in the 1GB Pinned HugePage ring and
     VRAM page pressure, routing secondary MoE experts or long-context decode to Host Intel AMX
     workers strictly when Delta_Q <= 1 (eliminating the 196.3%-215.9% p99 GPU stall).
  2. Active Clock-Floor & Power-Brake Governor:
     Polls effective post-arbitration SM clock frequency against the locked clock floor
     (1,920 MHz on GA102; 2,520 MHz on AD102) every 16 layers. When CTRL_PERF_LIMITS_SET_STATUS_V2
     power-cap arbitration drops effective SM clocks by > 2%, increments tau_sieve by +5% to
     shed marginal subspace tiles and restore locked peak frequency.
"""

from dataclasses import dataclass
import math
from typing import Dict, List


@dataclass
class HardwareTelemetrySample:
    layer_idx: int
    gpu_ready_seq: int
    cpu_done_seq: int
    vram_used_gb: float
    vram_capacity_gb: float
    effective_sm_clock_mhz: float
    locked_clock_floor_mhz: float
    board_power_watts: float
    power_cap_watts: float

    @property
    def delta_q(self) -> int:
        return max(0, self.gpu_ready_seq - self.cpu_done_seq)

    @property
    def vram_pressure(self) -> float:
        if not (math.isfinite(self.vram_used_gb) and math.isfinite(self.vram_capacity_gb)):
            return float("nan")
        return self.vram_used_gb / max(1e-6, self.vram_capacity_gb)

    @property
    def clock_sag_fraction(self) -> float:
        if not (math.isfinite(self.locked_clock_floor_mhz) and math.isfinite(self.effective_sm_clock_mhz)):
            return float("nan")
        if self.locked_clock_floor_mhz <= 0.0:
            return 0.0
        return max(0.0, (self.locked_clock_floor_mhz - self.effective_sm_clock_mhz) / self.locked_clock_floor_mhz)


class ClosedLoopHardwareGovernor:
    def __init__(
        self,
        arch: str = "AD102",
        initial_tau_sieve: float = 0.040,
        tau_min: float = 0.015,
        tau_max: float = 0.150,
        max_delta_q_for_offload: int = 1,
        vram_watermark: float = 0.80,
        clock_poll_interval_layers: int = 16,
        clock_sag_threshold: float = 0.02,
    ) -> None:
        self.arch = arch
        self.locked_clock_floor_mhz = 2520.0 if arch.upper() == "AD102" else 1920.0
        self.tau_sieve = initial_tau_sieve
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.max_delta_q_for_offload = max_delta_q_for_offload
        self.vram_watermark = vram_watermark
        self.clock_poll_interval_layers = clock_poll_interval_layers
        self.clock_sag_threshold = clock_sag_threshold

    def route_moe_and_regulate_sieve(
        self,
        sample: HardwareTelemetrySample,
        is_prefill_or_short_decode: bool = False,
    ) -> Dict[str, object]:
        """
        Executes closed-loop routing and Tiled-CBC threshold regulation for a single transformer layer.
        """
        # 1. Chorus Queue-Pressure-Guided AMX Offload Decision
        if is_prefill_or_short_decode:
            execution_target = "GPU_STREAM_0_TENSOR_CORE"
        elif sample.delta_q <= self.max_delta_q_for_offload and sample.vram_pressure >= self.vram_watermark:
            execution_target = "HOST_AMX_1GB_HUGEPAGE_RING"
        elif sample.delta_q <= self.max_delta_q_for_offload:
            execution_target = "COOPERATIVE_GPU_PLUS_HOST_AMX"
        else:
            # Host AMX ring has backlog (Delta_Q > 1): retain on GPU Stream 0 to prevent 215.9% p99 stall
            execution_target = "GPU_STREAM_0_BACKPRESSURE_FALLBACK"

        # 2. Active Clock-Floor & Power-Brake Governor (polls every 16 layers)
        # docs/12 4.31: Python's max(0.0, float("nan")) returns 0.0 (because nan > 0.0 is False), which used to turn
        # a NaN clock reading into sag_frac = 0.0 <= 0.01 and trigger RELAX_SIEVE_-2%_RESTORE_DENSITY. Guard finite
        # telemetry explicitly so non-finite readings hold tau_sieve, matching C nvrm_poll_and_adjust_sieve.
        governor_action = "HOLD"
        floor_mhz = sample.locked_clock_floor_mhz if sample.locked_clock_floor_mhz > 0.0 else self.locked_clock_floor_mhz
        telemetry_finite = (
            math.isfinite(floor_mhz)
            and math.isfinite(sample.effective_sm_clock_mhz)
            and math.isfinite(sample.board_power_watts)
            and math.isfinite(sample.power_cap_watts)
        )
        if not math.isfinite(floor_mhz) or not math.isfinite(sample.effective_sm_clock_mhz):
            sag_frac = float("nan")
        elif floor_mhz > 0.0:
            sag_frac = max(0.0, (floor_mhz - sample.effective_sm_clock_mhz) / floor_mhz)
        else:
            sag_frac = 0.0
        if (
            self.clock_poll_interval_layers > 0
            and sample.layer_idx % self.clock_poll_interval_layers == 0
            and telemetry_finite
            and floor_mhz > 0.0
            and sample.power_cap_watts > 0.0
        ):
            if (
                sag_frac > self.clock_sag_threshold
                and sample.board_power_watts >= 0.90 * sample.power_cap_watts
            ):
                # Power-cap arbitration drooped SM clock by > 2% near TDP: raise tau_sieve by +5%
                self.tau_sieve = min(self.tau_max, self.tau_sieve * 1.05)
                governor_action = "RAISE_SIEVE_+5%_SHED_POWER"
            elif (
                sag_frac <= 0.01
                and sample.board_power_watts < 0.88 * sample.power_cap_watts
            ):
                # Ample thermal/power headroom and clock within 1% GPUBoost bin: relax tau_sieve by -2%
                self.tau_sieve = max(self.tau_min, self.tau_sieve * 0.98)
                governor_action = "RELAX_SIEVE_-2%_RESTORE_DENSITY"

        return {
            "layer_idx": sample.layer_idx,
            "delta_q": sample.delta_q,
            "clock_sag_pct": sag_frac * 100.0,
            "execution_target": execution_target,
            "governor_action": governor_action,
            "tau_sieve": self.tau_sieve,
        }


def main() -> None:
    print("================================================================================")
    print("  WORKSTREAM 7B: CLOSED-LOOP TELEMETRY GOVERNOR (CHORUS + NVRM CLOCK ARBITRATION)")
    print("================================================================================\n")

    gov = ClosedLoopHardwareGovernor(arch="AD102", initial_tau_sieve=0.040)
    trace: List[HardwareTelemetrySample] = [
        HardwareTelemetrySample(0,  10, 10, 20.5, 24.0, 2520.0, 2520.0, 410.0, 450.0),
        HardwareTelemetrySample(16, 14, 13, 21.0, 24.0, 2445.0, 2520.0, 448.0, 450.0), # 3.0% clock sag -> +5% tau
        HardwareTelemetrySample(32, 19, 16, 21.2, 24.0, 2430.0, 2520.0, 449.0, 450.0), # Delta_Q=3 stall avoidance + +5% tau
        HardwareTelemetrySample(48, 22, 22, 18.0, 24.0, 2520.0, 2520.0, 380.0, 450.0), # Clock recovered -> relax -2%
    ]

    print(f"{'Layer':>5} | {'Delta_Q':>7} | {'Clock (MHz)':>11} | {'Sag %':>6} | {'Target Route':<34} | {'Governor Action':<28} | {'tau_sieve':>9}")
    print("-" * 116)
    results = []
    for s in trace:
        r = gov.route_moe_and_regulate_sieve(s)
        results.append(r)
        print(
            f"{s.layer_idx:>5} | {r['delta_q']:>7} | {s.effective_sm_clock_mhz:>11.1f} | "
            f"{r['clock_sag_pct']:>5.2f}% | {str(r['execution_target']):<34} | "
            f"{str(r['governor_action']):<28} | {r['tau_sieve']:>9.5f}"
        )
    print("-" * 116)

    assert results[0]["execution_target"] == "HOST_AMX_1GB_HUGEPAGE_RING"
    assert results[1]["governor_action"] == "RAISE_SIEVE_+5%_SHED_POWER"
    assert abs(float(results[1]["tau_sieve"]) - 0.0420) < 1e-6
    assert results[2]["execution_target"] == "GPU_STREAM_0_BACKPRESSURE_FALLBACK"
    assert abs(float(results[2]["tau_sieve"]) - 0.0441) < 1e-6
    # docs/12 4.31 regression: NaN clock or power telemetry must HOLD tau_sieve (max(0.0, nan) == 0.0 used to relax it)
    tau_before = gov.tau_sieve
    r_nan = gov.route_moe_and_regulate_sieve(
        HardwareTelemetrySample(64, 24, 24, 18.0, 24.0, float("nan"), 2520.0, 350.0, 450.0)
    )
    assert r_nan["governor_action"] == "HOLD" and gov.tau_sieve == tau_before, r_nan
    print("[+] PASS: Closed-loop Chorus AMX offload routing and NVRM clock-sag governor verified.")


if __name__ == "__main__":
    main()
