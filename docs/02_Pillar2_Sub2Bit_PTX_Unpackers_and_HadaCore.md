# 02. Pillar 2: Fused Sub-2-Bit (Binary g128 & Ternary) to Unlocked BF16 Tensor Core Kernels

## 1. End-to-End Memory-to-Register-to-Tensor-Core Pipeline

```
+-------------------------------------------------------------------+
|                     GLOBAL VRAM (1.125 BPW)                       |
|    [ Packed 1-Bit Weights ]          [ Group Scales (g128) ]      |
+-------------------------------------------------------------------+
                                 |
                                 v  cp.async.cg.shared.global (Bypass L1)
+-------------------------------------------------------------------+
|                      SHARED MEMORY (SRAM)                         |
|            [ 4-Stage Swizzled cp.async Pipeline ]                 |
+-------------------------------------------------------------------+
                                 |
                                 v  ldmatrix.sync.aligned.m8n8.x4 / LDS.128
+-------------------------------------------------------------------+
|                     WARP REGISTER UNPACKER                        |
|   [ Interleaved Shift & lop3.b32 ] --> [ Packed BF162 Scale Mul ] |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
|                     UNLOCKED TENSOR CORES                         |
|    [ Full-Rate HMMA.16816.F32 ]    --> [ FP32 Accumulator Tile ]  |
+-------------------------------------------------------------------+
```

---

## 2. Register-Level Bitwise Dequantization Math (`uint32 -> bf162`)

### A. Branchless 1-Bit (Binary g128, $1.125\text{ bpw}$) Unpacking via `lop3.b32 (0xAE)`

In IEEE-754 BFloat16 (`bf16`: 1 sign bit `[15]`, 8 exponent bits `[14:7]`, 7 mantissa bits `[6:0]`), $+1.0_{\text{bf16}} = \texttt{0x3F80}$ and $-1.0_{\text{bf16}} = \texttt{0xBF80}$ share identical lower 15 bits (`0x3F80`) and differ solely in bit 15. Dequantizing a pair of 1-bit weights $(b_{2k}, b_{2k+1}) \in \{0, 1\}^2$ (where $0 \mapsto -1.0$ and $1 \mapsto +1.0$) into a packed `bf162` register requires setting bit 15 to $\neg b_{2k}$, bit 31 to $\neg b_{2k+1}$, and bits $[14:0]$ and $[30:16]$ to `0x3F80`:

$$\text{bf162\_pair}_k = ((\neg b_{2k} \ll 15) \mid (\neg b_{2k+1} \ll 31)) \mid \texttt{0x3F803F80}$$

By pre-packing 32 weights along $K$ into a `uint32` word using the **Interleaved 16-Bit Split Layout**, where even weights $b_{2k}$ sit at bit $k \in \{0, \dots, 15\}$ and odd weights $b_{2k+1}$ sit at bit $k + 16 \in \{16, \dots, 31\}$:

$$W_{\text{packed}} = \sum_{k=0}^{15} \left( (b_{2k} \ll k) \mid (b_{2k+1} \ll (k + 16)) \right)$$

a single logical left shift by $(15 - k)$ aligns both sign bits simultaneously (`shl.b32`). Passing $a = W_{\text{packed}} \ll (15 - k)$, $b = \texttt{0x80008000}$, and $c = \texttt{0x3F803F80}$ into `lop3.b32` evaluates $F(a, b, c) = (\neg a \land b) \lor c$ with truth-table constant:

$$\text{immLut} = (\neg \texttt{0xF0} \land \texttt{0xCC}) \lor \texttt{0xAA} = \texttt{0x0C} \lor \texttt{0xAA} = \texttt{0xAE}$$

```ptx
shl.b32        %r_tmp, %r_w, (15 - k);
lop3.b32       %r_bf162_k, %r_tmp, 0x80008000, 0x3F803F80, 0xAE;
fma.rn.bf16x2  %r_bf162_k, %r_bf162_k, %r_scale, %r_neg_zero;
```

### B. Branchless 2-Bit Ternary (BitNet b1.58, $2.125\text{ bpw}$) Unpacking via `lop3.b32 (0xEA)` + 2x `fma.rn.bf16x2`

Each ternary weight $w \in \{-1, 0, +1\}$ is encoded offline as $q = w + 1 \in \{0, 1, 2\}$, with even codes $q_{2k}$ at bits $[2k+1 : 2k]$ and odd codes $q_{2k+1}$ at bits $[2k+17 : 2k+16]$ ($k \in \{0, \dots, 7\}$). Shifting right by $2k$ (`shr.u32`) and executing `lop3.b32` with `immLut = 0xEA` ($(\text{TA} \land \text{TB}) \lor \text{TC}$) against mask `0x00030003` and base `0x40004000` (which encodes $(2.0, 2.0)$ in `bf162`) deposits $q \in \{0, 1, 2\}$ into the bottom mantissa bits of $2.0$:

$$V(q) = 2.0 + \frac{q}{64} \in \{2.000000, 2.015625, 2.031250\}$$

Because `-129.0 * alpha_g` is not always exactly representable in 7-bit BF16 mantissa when `alpha_g` is arbitrary (which would otherwise leave a non-zero residual on $q=1$), we first extract exact unit trits `{-1.0, +0.0, +1.0}` using exact BF16 constants $c_{\text{mul}} = 64.0_{\text{bf162}} = \texttt{0x42804280}$ and $c_{\text{sub}} = -129.0_{\text{bf162}} = \texttt{0xC301C301}$, and then scale by $\alpha_g$ (`scale_bf162`) with $-0.0_{\text{bf162}} = \texttt{0x80008000}$ (`P0-4`):

$$\text{trit}(q) = \texttt{fma.rn.bf16x2}(V(q), 64.0, -129.0) \in \{-1.0, +0.0, +1.0\}$$

$$\widehat{w}(q) = \texttt{fma.rn.bf16x2}(\text{trit}(q), \alpha_g, -0.0) \in \{-\alpha_g, +0.0, +\alpha_g\}$$

---

## 3. Exact `HMMA.16816.F32` Warp Lane Mapping & Online HadaCore Prologue

For a 32-thread warp ($L \in \{0, \dots, 31\}$), let $\text{groupID} = L \gg 2 \in \{0, \dots, 7\}$ and $t = L \land 3 \in \{0, \dots, 3\}$:
- **Operand $A$ ($16 \times 16$ BF16, row-major, 4 `bf162` registers `a[0..3]`):**
  - `a[0]`: row $\text{groupID}$, cols $\{2t, 2t+1\}$; `a[1]`: row $\text{groupID} + 8$, cols $\{2t, 2t+1\}$
  - `a[2]`: row $\text{groupID}$, cols $\{2t+8, 2t+9\}$; `a[3]`: row $\text{groupID} + 8$, cols $\{2t+8, 2t+9\}$
- **Operand $B$ ($16 \times 8$ BF16, col-major, $K=16, N=8$, 2 `bf162` registers `b[0..1]`):**
  - Column index $n = \text{groupID}$. `b[0]` holds rows $k \in \{2t, 2t+1\}$ (pair index $p_0 = t + 8\cdot\text{sub}$), and `b[1]` holds rows $k \in \{2t+8, 2t+9\}$ (pair index $p_1 = t + 4 + 8\cdot\text{sub}$). One 32-bit `w_packed_col` register feeds two consecutive $K=16$ Tensor Core steps ($K \in [0, 31]$).
- **Online HadaCore Radix-16 & Kronecker Transform (`hadacore_rotate_m16k16_in_regs`):**
  - The normalized $16 \times 16$ Sylvester Walsh-Hadamard matrix $H_{16}$ has closed-form entries $(H_{16})_{r,c} = 0.25 \cdot (-1)^{\text{popc}(r \land c)} \in \{\texttt{0x3E80}, \texttt{0xBE80}\}$.
  - Two `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instructions rotate a $16 \times 16$ activation tile directly in registers in 16 SMSP cycles. For larger dimensions $N = (N/16) \times 16$, Sylvester's Kronecker identity $H_N = H_{N/16} \otimes H_{16}$ evaluates $\text{Mat}_{N/16 \times 16}(x H_N) = H_{N/16}^\top \cdot \text{Mat}_{N/16 \times 16}(x) \cdot H_{16}$ via back-to-back radix-16 Tensor Core and butterfly passes.

---

## 4. Analytical Bandwidth Amplification and SMSP Roofline Projections (`smsp_roofline_sim.py`)

| Hardware & Kernel Metric | Unlocked RTX 3090 (GA102) | Unlocked RTX 3090 Ti (GA102) | Unlocked RTX 4090 (AD102) |
| :--- | :--- | :--- | :--- |
| **Physical GDDR6X Memory Bandwidth** | $936.2\text{ GB/s}$ | $1,008.0\text{ GB/s}$ | $1,008.0\text{ GB/s}$ |
| **Binary g128 Compression Ratio ($16 / 1.125$)** | $14.22\times$ | $14.22\times$ | $14.22\times$ |
| **Effective Weight Bandwidth Amplification** | $13.31\text{ TB/s}$ | $14.33\text{ TB/s}$ | $14.33\text{ TB/s}$ |
| **Unlocked `BF16 -> FP32` Tensor Core Peak** | $142.0\text{ TFLOPS}$ | $157.4\text{ TFLOPS}$ | $331.6\text{ TFLOPS}$ |
| **DRAM Ridge Point (Standard W16A16 GEMM)** | $151.7\text{ FLOPs/Byte}$ ($M \ge 152$) | $156.2\text{ FLOPs/Byte}$ ($M \ge 157$) | $329.0\text{ FLOPs/Byte}$ ($M \ge 329$) |
| **DRAM Ridge Point (Fused W1A16_F32 GEMM)** | $10.7\text{ FLOPs/Byte}$ ($M \ge 11$) | $11.0\text{ FLOPs/Byte}$ ($M \ge 11$) | $23.1\text{ FLOPs/Byte}$ ($M \ge 24$) |
| **Projected W1A16_F32 Compute ($M = 32$, SMSP Roofline Model)** | $118.4\text{ TFLOPS}$ ($83.4\%$ peak) | $131.6\text{ TFLOPS}$ ($83.6\%$ peak) | $268.5\text{ TFLOPS}$ ($81.0\%$ peak) |
| **Projected W1A16_F32 Compute ($M = 2048$, SMSP Roofline Model)** | $132.8\text{ TFLOPS}$ ($93.5\%$ peak) | $147.2\text{ TFLOPS}$ ($93.5\%$ peak) | $308.4\text{ TFLOPS}$ ($93.0\%$ peak) |
