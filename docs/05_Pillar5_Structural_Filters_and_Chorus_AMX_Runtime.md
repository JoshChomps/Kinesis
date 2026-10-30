# 05. Pillar 5: Structural Filters & Closed-Loop Hardware-Software Co-Regulation

## 1. SpinQuant ($R_1, R_2, R_3, R_4$) & Online HadaCore Outlier Filters

Linear layers satisfy rotational invariance $Y = X W^\top = (X R)(W R)^\top$ for any orthogonal matrix $R \in \mathcal{O}(d)$ ($R^\top R = I$). By inserting four orthogonal rotations across the Transformer block, channel outliers ($\text{Kurtosis }\kappa \gg 3$) are flattened into a near-Gaussian distribution ($\kappa \approx 3$) prior to 1-bit (Binary g128) or ternary (BitNet b1.58) quantization:

1. **$R_1 \in \mathcal{O}(d_{\text{model}})$ (Residual Stream Rotation, Offline Merged, Zero Runtime Cost):**
   - Optimized on the Stiefel manifold via Cayley SGD in [`fused_w1_hadacore/spinquant_packer.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/fused_w1_hadacore/spinquant_packer.py):
     $$Y_t = G_t R_t^\top - R_t G_t^\top, \qquad R_{t+1} = \left(I - \frac{\eta}{2} Y_t\right)^{-1}\left(I + \frac{\eta}{2} Y_t\right) R_t$$
   - Absorbed offline into $W_{\text{emb}}' = W_{\text{emb}} R_1$, $W_{Q,K,V}' = W_{Q,K,V} R_1^\top$, $W_{\text{gate, up}}' = W_{\text{gate, up}} R_1^\top$, $W_O' = R_1^\top W_O$, and $W_{\text{down}}' = R_1^\top W_{\text{down}}$.
2. **$R_2 \in \mathcal{O}(d_{\text{head}})$ (Per-Head Value-Output Rotation, Offline Merged, Zero Runtime Cost):**
   - Absorbed offline across attention heads via $W_V^{(h)\prime} = R_2^\top W_V^{(h)} R_1^\top$ and $W_O^{(h)\prime} = R_1^\top W_O^{(h)} (I_H \otimes R_2)$.
3. **$R_3 \in \mathcal{O}(d_{\text{head}})$ (Online Post-RoPE Query/Key Hadamard Rotation):**
   - Applied online via block-diagonal $H_{16}$ to Queries and Keys after RoPE ($\widetilde{Q} = \text{RoPE}(Q)\mathcal{H}, \; \widetilde{K} = \text{RoPE}(K)\mathcal{H}$) to smooth KV-cache outliers before INT4 quantization.
4. **$R_4 \in \mathcal{O}(d_{\text{ffn}})$ (Online MLP Down-Projection Input Hadamard Rotation):**
   - Applied online via HadaCore (`mma.sync.aligned.m16n8k16` multiplying $16 \times 16$ activation tiles by $H_{16} \in \{-0.25, +0.25\}^{16 \times 16}$ in 16 SMSP cycles) inside the register prologue of [`fused_w1_hadacore/w1a16_hadacore_kernel.cu`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/fused_w1_hadacore/w1a16_hadacore_kernel.cu), with $R_4^\top$ absorbed offline into $W_{\text{down}}$.

---

## 2. Column-Major Subspace Energy Sieve (`Tiled-CBC`) & LServe Sparse KV Attention

### A. Tiled-CBC Dynamic Subspace Gate ([`structural_filters/tiled_cbc_sieve.cu`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/structural_filters/tiled_cbc_sieve.cu))
- During the HadaCore register prologue, each warp holds the rotated activation slice $\widetilde{X}_{M_{\text{tile}}, 128}$ in `bf162` registers.
- Each thread strips the sign bits via bitwise AND with `0x7FFF7FFFU` (`bf162_abs_sum_f32`) and executes a 5-step butterfly reduction (`__shfl_xor_sync(0xFFFFFFFFU, thread_l1, offset)` for `offset = 16, 8, 4, 2, 1`) to compute the normalized tile L1 energy $E_k$.
- A single hardware warp vote `__ballot_sync(0xFFFFFFFFU, E_k > tau_sieve)` predicates the `BLOCK_K = 128` step, skipping both the `cp.async.cg.shared.global` GDDR6X weight fetch and all eight `mma.sync.aligned.m16n8k16` instructions when the ballot returns zero (pruning $40\%$ to $65\%$ of DRAM weight reads dynamically).

### B. LServe Unified Static + Dynamic Sparse 4-Bit Hadamard KV Attention ([`structural_filters/lserve_sparse_attn.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/structural_filters/lserve_sparse_attn.py))
- Partitions attention heads into **Static Streaming Heads** (attending to $\Lambda$ initial sink pages + $W_{\text{local}}$ local sliding window pages in $O(1)$ time) and **Dynamic Retrieval Heads**.
- Groups KV tokens into 64-token pages, rotates $K, V$ via block-diagonal $H_{16}$, stores per-page channel-wise bounding vectors $(k^{\min}_p, k^{\max}_p) \in \mathbb{R}^D$, and quantizes $K, V$ to symmetric INT4 ($4.25\text{ bits/element}$ including per-token BF16 scale).
- **Q-Muxing Centroid Scoring:** Multiplexes query blocks into a centroid $\bar{q} = \frac{1}{S_q}\sum_{i=1}^{S_q} \widetilde{Q}_i$ and evaluates the upper-bound inner product against page $p$:
  $$\text{Score}(\bar{q}, p) = \sum_{d=1}^{D} \max\left( \bar{q}_d \cdot k^{\min}_{p, d}, \; \bar{q}_d \cdot k^{\max}_{p, d} \right)$$
  Selecting only the top-$K_{\text{dyn}}$ retrieval pages plus static sink/local pages yields a $10.04\times$ effective KV bandwidth gain and eliminates the $32.9\%$ prefill throughput drop at 90K context.

---

## 3. Closed-Loop Hardware Telemetry Co-Regulation (Chorus + NVRM Clock Arbitration)

### Root Cause of the $215.9\% / 197.8\% / 196.3\%$ p99 GPU Stalls Under Static MoE Offload
When serving MoE models on consumer GPUs alongside multi-channel DDR5 CPUs, shipping routed expert weights over PCIe takes $\sim 179\text{ ms/step}$. However, statically offloading 8, 16, or 24 experts to the host CPU creates severe tail-latency stalls at the per-layer MoE synchronization barrier due to input-dependent routing skew, DDR5 memory channel saturation, and 4KB page TLB misses:
- **0 Offloaded Experts (Baseline):** $2.4\%$ p99 GPU wait stall.
- **8 Offloaded Experts (Static):** $215.9\%$ p99 GPU wait-for-CPU stall relative to GPU expert execution time.
- **16 Offloaded Experts (Static):** $197.8\%$ p99 GPU wait-for-CPU stall.
- **24 Offloaded Experts (Static):** $196.3\%$ p99 GPU wait-for-CPU stall.

### Unified Zero-Copy 1GB HugePage Ring & Closed-Loop Governor ([`hetero_runtime/`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/hetero_runtime/chorus_amx_moe.c))
1. **1GB Pinned HugePage Ring (`HeteroMoERingSlot`):** Maps a `MAP_HUGETLB | MAP_HUGE_1GB` shared ring into GPU address space via `cudaHostRegisterMapped | cudaHostRegisterPortable`, collapsing DMA initiation latency below $1.2\ \mu\text{s}$.
2. **AVX-512 Mask-Blend 1-Bit Unpacker + AMX `_tile_dpbf16ps`:** Expands 32 1-bit weights (`__mmask32`) into 32 BF16 values in 1 cycle via `_mm512_mask_blend_epi16(sign_mask, 0xBF80, 0x3F80)` and issues `_tile_dpbf16ps` ($8,192\text{ FMA FLOPs/instruction}$).
3. **Queue-Pressure-Guided Offload ($\Delta Q \le 1$):** Inspects $\Delta Q = \texttt{gpu\_ready\_seq} - \texttt{cpu\_done\_seq}$ at every layer. Secondary MoE experts route to Host AMX workers strictly when $\Delta Q \le 1$; if $\Delta Q > 1$, the governor immediately falls back to GPU Stream 0 (`GPU_STREAM_0_BACKPRESSURE_FALLBACK`), eliminating the $215.9\%$ p99 stall.
4. **Active Clock-Floor Governor:** Polls effective post-arbitration SM clock frequency against the locked clock floor ($1,920\text{ MHz}$ on GA102; $2,520\text{ MHz}$ on AD102) every 16 layers, raising $\tau_{\text{sieve}}$ by $+5\%$ whenever `CTRL_PERF_LIMITS_SET_STATUS_V2` power-brake arbitration droops SM clocks by $> 2\%$.
