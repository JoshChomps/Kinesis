// File: hetero_runtime/closed_loop_governor.c
// C Runtime Implementation of the Closed-Loop Hardware Telemetry Governor
// (Chorus Queue-Pressure Scheduler + NVRM / NVML Clock-Floor Arbitration)
// Build: gcc -O3 -std=c11 -o test_governor_c closed_loop_governor.c

#include <stdio.h>
#include <stdint.h>
#include <stdatomic.h>
#include <assert.h>

typedef enum {
    ROUTE_GPU_STREAM_0_TENSOR_CORE = 0,
    ROUTE_HOST_AMX_1GB_HUGEPAGE_RING = 1,
    ROUTE_COOPERATIVE_GPU_PLUS_HOST_AMX = 2,
    ROUTE_GPU_STREAM_0_BACKPRESSURE_FALLBACK = 3
} MoERouteTarget;

typedef struct {
    uint32_t locked_clock_floor_mhz; // 1920 MHz (GA102) or 2520 MHz (AD102)
    float    tau_sieve;
    float    tau_min;
    float    tau_max;
    uint64_t max_delta_q_for_offload;
    float    vram_watermark;
} ClosedLoopGovernorState;

static inline MoERouteTarget chorus_select_moe_route(
    const ClosedLoopGovernorState* gov,
    uint64_t gpu_ready_seq,
    uint64_t cpu_done_seq,
    float vram_pressure,
    int is_prefill_or_short_decode
) {
    if (is_prefill_or_short_decode) {
        return ROUTE_GPU_STREAM_0_TENSOR_CORE;
    }
    uint64_t delta_q = (gpu_ready_seq >= cpu_done_seq) ? (gpu_ready_seq - cpu_done_seq) : 0ULL;
    if (delta_q <= gov->max_delta_q_for_offload) {
        if (vram_pressure >= gov->vram_watermark) {
            return ROUTE_HOST_AMX_1GB_HUGEPAGE_RING;
        }
        return ROUTE_COOPERATIVE_GPU_PLUS_HOST_AMX;
    }
    return ROUTE_GPU_STREAM_0_BACKPRESSURE_FALLBACK;
}

static inline void nvrm_poll_and_adjust_sieve(
    ClosedLoopGovernorState* gov,
    uint32_t layer_idx,
    uint32_t effective_sm_clock_mhz,
    float board_power_watts,
    float power_cap_watts
) {
    if ((layer_idx & 15U) != 0U) {
        return; // Poll every 16 layers
    }
    float floor_f = (float)gov->locked_clock_floor_mhz;
    if (floor_f <= 0.0f || power_cap_watts <= 0.0f) {
        return; // unconfigured governor: avoid division by zero / NaN sag
    }
    float sag = (floor_f - (float)effective_sm_clock_mhz) / floor_f;
    // Only treat clock sag > 2.0% as a power-brake event when board power is near TDP (>= 90% cap)
    // so low-utilization P8/P5 idle downclocking is never misclassified as power throttling.
    if (sag > 0.02f && board_power_watts >= 0.90f * power_cap_watts) {
        float next_tau = gov->tau_sieve * 1.05f;
        gov->tau_sieve = (next_tau < gov->tau_max) ? next_tau : gov->tau_max;
    } else if (sag <= 0.01f && board_power_watts < 0.88f * power_cap_watts) {
        // Allow 1% hysteresis deadband (~15 MHz GPUBoost step bin) so tau_sieve never ratchets permanently
        float next_tau = gov->tau_sieve * 0.98f;
        gov->tau_sieve = (next_tau > gov->tau_min) ? next_tau : gov->tau_min;
    }
}

int main(void) {
    ClosedLoopGovernorState gov = {2520U, 0.040f, 0.015f, 0.150f, 1ULL, 0.80f};
    MoERouteTarget r0 = chorus_select_moe_route(&gov, 10, 10, 0.85f, 0);
    assert(r0 == ROUTE_HOST_AMX_1GB_HUGEPAGE_RING);

    nvrm_poll_and_adjust_sieve(&gov, 16U, 2445U, 448.0f, 450.0f);
    assert(gov.tau_sieve > 0.0419f && gov.tau_sieve < 0.0421f);

    MoERouteTarget r1 = chorus_select_moe_route(&gov, 19, 16, 0.88f, 0);
    assert(r1 == ROUTE_GPU_STREAM_0_BACKPRESSURE_FALLBACK);

    printf("[+] PASS: C Closed-Loop Governor verified (tau_sieve = %.5f after >2%% clock sag).\n", gov.tau_sieve);
    return 0;
}
