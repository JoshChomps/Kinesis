# 01. Pillar 1: Full-Rate Silicon Architecture, SRAM Rooflines, and Compiler Heuristic Emulation

## Scientific Context & Architectural Scope

This research workstream explores the microarchitectural behavior and roofline characteristics of compute hardware operating at maximum functional unit throughput. By analyzing instruction issue rates, register timing, and compiler heuristic selection, we study the theoretical performance limits of the hardware and evaluate why default software compilers often underutilize available silicon capabilities.

---

## 1. Hardware BAR0 Register Map & SEC2 Falcon Issue-Rate Architecture

On NVIDIA Turing (`TU102`), Ampere (`GA10x`), and Ada Lovelace (`AD10x`), SM instruction issue-rate configurations for mixed-precision Tensor Core operations (`HMMA.16816.F32` / `HMMA.1688.F32`, FP16/BF16/TF32 with FP32 accumulation) are governed by the **Feature Override / Fuse Readout block** in BAR0 MMIO space (`0x00823800` range) and the **SEC2 Falcon co-processor** (`NV_PSEC_FALCON_BASE = 0x00840000`).

### A. Feature Override & Speed Selector Registers (BAR0 Offsets)

| Symbolic Register Name | BAR0 Hex Offset | Default Configured Value (GeForce) | Target Full-Rate Value | Functional Bitfield Breakdown & Hardware Semantics |
| :--- | :--- | :--- | :--- | :--- |
| **`NV_PFUSE_FEAT_OVR_PLM`** (`FEAT_PLM`) | `0x00823804` | `0x88000000` / `0xF0000000` (Level 2/3 write-only) | `0xFFFFFFFF` | **Feature Override Privilege-Level Mask (PLM):** Controls read/write access permissions for `SS0`/`SS1`. Setting PLM to `0xFFFFFFFF` enables Level-0 (Host CPU / BAR0 MMIO) write access during pre-boot initialization before `FWSEC-FRTS` or the driver locks down the bus. |
| **`NV_PFUSE_FEAT_READOUT_1`** (`FEAT_READOUT_1`) | `0x00823818` | `0x11111111` (Divisor 1 = Half-Rate) | `0x00000000` (Divisor 0 = Full-Rate) | **Effective Feature Speed Readout:** Reflects the post-arbitration effective speed selector state (`fuse_hard_lock \| override_divisor`). Used to verify effective silicon issue rate in hardware. |
| **`NV_PFUSE_FEAT_OVR_SM_SPD_0`** (`SS0` / `GU_SS0_UNLOCK`) | `0x0082381C` | `0x99999999` (Half-speed on GeForce) | `0x88888888` | **Speed Selector 0 (`SS0`):** Eight 4-bit nibbles controlling functional unit issue-rate divisors across SM sub-partitions (`SMSP`s): `FFMA`, `FMLA16`, `FMLA32` (`HMMA.16816.F32` FP16/BF16 -> FP32, `HMMA.1688.F32` TF32), and lower `IMLA0..3` slots (note: FP64 DP rate is governed separately by `NV_PFUSE_OPT_FP64_SPEED_SELECT`). In each 4-bit nibble: bit `3` (`0x8`) = Override Enable, bits `2:0` = Speed Divisor (`0` = Full Speed $\tau = 16$; `1` = Half Speed $\tau = 32$ cycles per `HMMA.16816` per SMSP, i.e. 8 / 16 cycles per 2,048-FLOP half-instruction; so `0x9 = 0b1001`). Writing `0x88888888` sets `Override_Enable = 1, Divisor = 0` across all 8 pipeline slots. |
| **`NV_PFUSE_FEAT_OVR_SM_SPD_1`** (`SS1` / `GU_SS1_UNLOCK`) | `0x00823820` | `0x00000009` or `0x00000000` | `0x00000008` | **Speed Selector 1 (`SS1`):** 4-bit nibble (`[3:0]`) governing the upper `IMLA4` / Tensor Core pipeline slot. Writing `0x8` (`0b1000`) enables the override at full speed (`Divisor = 0`). |

### B. SEC2 Falcon Heavy-Secure (HS) Privilege Elevation ("A Canary in the Crypto Mine")

1. **The Privilege Ring Barrier:** NVIDIA GPUs enforce a 4-tier hardware privilege ring (Level 0 = Host CPU / untrusted driver; Level 1 = Light Secure; Level 2 = Medium; Level 3 = Heavy Secure Falcon/RISC-V microcode signed by NVIDIA's hardware root of trust). During cold boot, VBIOS `FWSEC` runs in HS mode (Level 3), latches `SS0 = 0x99999999`, and locks `FEAT_PLM` (`0x00823804`) so that direct BAR0 `writel()` calls from the Linux kernel (Level 0) are dropped by the `PRI_RING` arbiter.
2. **Signed HS Booter Payload (`0x00840000..0x008403C0`):**
   - Reset SEC2 Falcon via `NV_PSEC_FALCON_RESET` (`0x008403C0`) and poll `NV_PSEC_FALCON_DMACTL` (`0x0084010C`).
   - Load the NVIDIA-signed HS `Booter` ucode into Falcon Instruction Memory via `NV_PSEC_FALCON_IMEMC0` (`0x00840180`), `IMEMD0` (`0x00840184`), and `IMEMT0` (`0x00840188`). Because the binary carries a valid RSA-3K / AES-CBC hardware signature, the Falcon Brom transitions into **Heavy Secure (Level 3)** mode.
   - Stage the parameter/ROP payload into Falcon Data Memory via `NV_PSEC_FALCON_DMEMC0` (`0x008401C0`) and `DMEMD0` (`0x008401C4`), set `NV_PSEC_FALCON_BOOTVEC` (`0x00840104`), and start execution via `NV_PSEC_FALCON_CPUCTL` (`0x00840100`).
   - Executing at Level 3, the payload writes `0xFFFFFFFF` to `FEAT_PLM` (`0x00823804`), writes `0x88888888` to `SS0` (`0x0082381C`), and writes `0x00000008` to `SS1` (`0x00823820`).
   - Trigger `pcie_flr(pdev)`. On platforms where the `0x00823800` fuse-override latch domain sits on the sticky reset rail, `SS0` and `SS1` retain `0x88888888` and `0x8` across PCIe Function-Level Reset; if `pcie_flr` resets the speed-selector latches on a given SKU (`SS0 != 0x88888888` on post-FLR verification), `nv_pci_resume_early_unlock` immediately re-applies the SEC2 Falcon unlock payload without a second FLR (`O4`).
3. **Soft-Strapped vs. Hard Fuse-Locked Dies (`GA102-300/301` & `AD102-300` vs. `GA102-250`):**
   - Retail `GA102-300` (RTX 3090), `GA102-301` (RTX 3090 Ti), `AD102-300` (RTX 4090), and `TU102-300A` (RTX 2080 Ti) leave the physical master override-disable e-fuse unblown (`OPT_FUSE_DISABLE_FEAT_OVR = 0`), so `FEAT_READOUT_1` (`0x00823818`) transitions cleanly to divisor `0` (full-rate).
   - Certain salvage/engineering variants such as the `GA102-250` (RTX 3080 Ti 20GB) have a physical e-fuse hard-lock mask blown on the `FMLA32` nibble (`effective_divisor = fuse_hard_lock | ovr_div`), preventing BF16 issue-rate doubling even when `SS0 = 0x88888888` is preserved.

---

## 2. Root-Cause Microarchitectural Analysis: The SRAM-to-Tensor-Core Roofline Shift

Both NVIDIA Ampere GA102 (`sm_86`, 82 SMs on RTX 3090, 84 SMs on RTX 3090 Ti) and Ada Lovelace AD102 (`sm_89`, 128 SMs on RTX 4090) partition each Streaming Multiprocessor (SM) into $N_{\text{SMSP}} = 4$ sub-partitions. All four sub-partitions share a unified L1 Data Cache and Shared Memory (SRAM) crossbar composed of 32 banks, each delivering $4\text{ Bytes/cycle}$:

$$B_{\text{SRAM}} = 32\text{ banks} \times 4\text{ Bytes/bank/cycle} = 128\text{ Bytes/cycle/SM}$$

A single warp executing `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`HMMA.16816.F32.BF16`) multiplies a $16 \times 16$ BF16 matrix $A$ ($512\text{ Bytes}$) by a $16 \times 8$ BF16 matrix $B$ ($256\text{ Bytes}$) and accumulates into a $16 \times 8$ FP32 matrix $C/D$ ($512\text{ Bytes}$), performing:

$$W_{\text{HMMA}} = 2 \times 16 \times 8 \times 16 = 4,096\text{ FLOPs} \quad (2,048\text{ FMA operations})$$

In a threadblock decomposed into $N_w$ warps where each warp computes a register-resident accumulator tile of shape $(W_M, W_N)$ in steps of $K_{\text{step}} = 16$:

$$\text{Bytes}_{\text{ldmatrix}} = N_w \cdot 2 \cdot (16 W_M + 16 W_N) = 32 N_w (W_M + W_N)\text{ Bytes}$$

$$T_{\text{SRAM}} = \frac{\text{Bytes}_{\text{ldmatrix}}}{B_{\text{SRAM}}} \cdot \beta_{\text{conflict}} = \frac{32 N_w (W_M + W_N)}{128} \cdot \beta_{\text{conflict}} = \frac{N_w (W_M + W_N)}{4} \cdot \beta_{\text{conflict}}\text{ cycles}$$

$$T_{\text{MMA}}(\tau) = \frac{N_w \cdot \left(\frac{W_M}{16}\right) \cdot \left(\frac{W_N}{8}\right)}{N_{\text{SMSP}}} \cdot \tau = \frac{N_w W_M W_N}{512} \cdot \tau\text{ cycles}$$

For the Tensor Core pipeline to remain compute-bound ($T_{\text{MMA}}(\tau) \ge T_{\text{SRAM}}$), the **Warp-Tile Arithmetic Intensity** ($\mathcal{I}_{\text{warp}}$) must satisfy:

$$\mathcal{I}_{\text{warp}} = \frac{\text{FLOPs}_{\text{warp}}}{\text{Bytes}_{\text{warp, SRAM}}} = \frac{2 \cdot W_M \cdot W_N \cdot 16}{32(W_M + W_N)} = \frac{W_M W_N}{W_M + W_N}\text{ FLOPs/Byte}$$

> [!NOTE]
> FLOP accounting corrected 2026-11-08 (docs/12 4.7, docs/14 section 8). $\tau$ is the interval between
> `HMMA.16816` issues on one SMSP. One such instruction is $W_{\text{HMMA}} = 4096$ FLOPs, so an SM does
> $4 \cdot 4096 / \tau$ FLOPs/cycle. The published GeForce FP32-accumulate rate (512 FLOPs/cycle/SM, 165 TF on
> AD102) therefore means $\tau = 32$ locked and $\tau = 16$ unlocked. This table previously used $\tau = 16 / 8$
> with 4,096 FLOPs per instruction, which double-counted throughput relative to its own first row and halved every
> $T_{\text{MMA}}$ below. The ridge points (4.0 / 8.0 FLOPs/Byte) were right; the $T_{\text{MMA}}$ entries and the
> margins were not.

| Architectural / Warp-Tile Metric | Locked State ($\tau = 32\text{ cycles}$) | Unlocked State ($\tau = 16\text{ cycles}$) | Physical & Microarchitectural Consequence |
| :--- | :--- | :--- | :--- |
| **SM Tensor Core Throughput (`HMMA.16816.F32`)** | $512\text{ FLOPs/cycle/SM}$ | $1,024\text{ FLOPs/cycle/SM}$ | Hardware issue interval per SMSP halves from 32 to 16 cycles. |
| **SM SRAM Read Bandwidth ($32 \times 4\text{ B}$)** | $128\text{ Bytes/cycle/SM}$ | $128\text{ Bytes/cycle/SM}$ | Physical SRAM crossbar bandwidth remains fixed. |
| **Conflict-Free Ridge Point ($\beta_{\text{conflict}} = 1.0$)** | $4.0\text{ FLOPs/Byte}$ | $8.0\text{ FLOPs/Byte}$ | Minimum operational intensity doubles on unlocked silicon. |
| **Warp Tile $32 \times 32$ ($\mathcal{I}_{\text{warp}} = 16.0\text{ FLOP/B}$)** | $T_{\text{MMA}} = 64 N_w$, $T_{\text{SRAM}} = 16 N_w \beta$ | $T_{\text{MMA}} = 32 N_w$, $T_{\text{SRAM}} = 16 N_w \beta$ | Compute-bound in both states: $4:1$ locked, $2:1$ unlocked. Even the modelled $\beta = 1.18$ bank-conflict factor leaves $1.69\times$ unlocked (previously reported as a $0.85\times$ stall, an artifact of the FLOP error). |
| **Warp Tile $64 \times 32$ ($\mathcal{I}_{\text{warp}} = 21.3\text{ FLOP/B}$)** | $T_{\text{MMA}} = 128 N_w$, $T_{\text{SRAM}} = 24 N_w \beta$ | $T_{\text{MMA}} = 64 N_w$, $T_{\text{SRAM}} = 24 N_w \beta$ | $5.33:1$ locked, $2.67:1$ unlocked ($2.38\times$ at the modelled $\beta = 1.12$). |
| **Warp Tile $64 \times 64$ ($\mathcal{I}_{\text{warp}} = 32.0\text{ FLOP/B}$)** | $T_{\text{MMA}} = 256 N_w$, $T_{\text{SRAM}} = 32 N_w \beta$ | $T_{\text{MMA}} = 128 N_w$, $T_{\text{SRAM}} = 32 N_w \beta$ | $8:1$ locked, $4:1$ unlocked. |
| **Warp Tile $128 \times 64$ ($\mathcal{I}_{\text{warp}} = 42.7\text{ FLOP/B}$)** | $T_{\text{MMA}} = 512 N_w$, $T_{\text{SRAM}} = 48 N_w \beta$ | $T_{\text{MMA}} = 256 N_w$, $T_{\text{SRAM}} = 48 N_w \beta$ | $10.67:1$ locked, $5.33:1$ unlocked, requiring $\sim 218$ registers/thread. |

The margin in every row equals $\mathcal{I}_{\text{warp}} / (\text{ridge} \cdot \beta)$ with ridge $= 4$ (locked) or $8$ (unlocked)
FLOPs/Byte, which is a quick consistency check. `smsp_roofline_sim.py` now asserts that the FLOP rate implied by
$T_{\text{MMA}}$ equals `peak_tflops` for every row, so the two cannot drift apart again.

**What the table measures (docs/12 4.18).** The margins are crossbar-throughput margins: `ldmatrix` is modelled
as bytes over a 128 B/cycle crossbar. Per-warp register-scoreboard latency is a separate term,
`LDMATRIX_LATENCY_CYCLES = 24` (an assumption, not measured). With fragment double-buffering (the usual CUTLASS
mainloop) it is covered in every row, because one warp's MMA work per K=16 step (8 HMMA x tau = 256 / 128 cycles
for 32x32) exceeds it. Without prefetch, the 1-warp-per-SMSP tiles expose it: section [1b] of the simulator
shows 32x32 SassWait rising from 7.01% to 14.45% locked and from 13.08% to 25.22% unlocked (64x32: 5.09 -> 9.13%,
9.68 -> 16.72%). The 2-warp-per-SMSP tiles are unchanged, since the second warp covers the wait.

### Conflict-Free `ldmatrix.x4` 128-Byte XOR Swizzling (`Swizzle<3, 3, 3>` in 16-bit elements / CuTe `Swizzle<3, 4, 3>` in bytes)

In an unswizzled row-major Shared Memory tile of stride $K_{\text{tile}} = 64$ BF16 elements ($128\text{ Bytes/row} = 32\text{ banks} \times 4\text{ Bytes}$), row $r$ and row $r+1$ start at the same bank. When 8 threads in `ldmatrix.sync.aligned.m8n8.x4.shared.b16` read 16-byte segments across rows $r = 0\dots 7$, all 8 threads hit banks `0..3` (an 8-way bank conflict). Permuting the eight 16-byte chunks $c_{\text{16B}} = \lfloor k / 8 \rfloor \in \{0, \dots, 7\}$ within each 128-byte line via bitwise XOR with $(r \land 7)$ eliminates bank conflicts completely ($\beta_{\text{conflict}} = 1.0$):

$$\text{ByteOffset}_{\text{swizzled}}(r, k) = 128 r + 16 \cdot \left( \left\lfloor \frac{k}{8} \right\rfloor \oplus (r \land 7) \right) + 2(k \bmod 8)$$

---

## 3. Path A (`libcublas_unlock_shim.c`) & Path B (`unlocked_gemm.py`)

- **Path A ([`compiler_unlock/libcublas_unlock_shim.c`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/compiler_unlock/libcublas_unlock_shim.c)):** Hooks `cuDeviceGetAttribute`, `cudaDeviceGetAttribute`, `cuDeviceGetName`, `cuGetProcAddress`, `cuGetProcAddress_v2`, and `/dev/nvidiactl` `ioctl` (`NV_ESC_RM_CONTROL = 0x2A`). Translates `0x2203`/`0x2204`/`0x2206` $\rightarrow$ `0x2230` (RTX A6000) and `0x2684` $\rightarrow$ `0x26B1` (RTX 6000 Ada) exclusively when `dladdr` (or the thread-local / multi-frame backtrace caller guard `O2`) matches `libcublas`, `libhipblas`, or `libcudnn`, while returning true physical SM counts (`82`, `84`, `128`) to PyTorch and NCCL.
- **Path B ([`compiler_unlock/unlocked_gemm.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/compiler_unlock/unlocked_gemm.py)):** Enforces $W_M \times W_N \ge 64 \times 64$ per warp and 4-to-5 `cp.async` stages within the $101,376\text{ Byte}$ ($99\text{ KiB}$, `99 * 1024`) dynamic Shared Memory ceiling of `sm_86` and `sm_89`.
