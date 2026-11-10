# 05. Pillar 5: Structural Filters & Closed-Loop Hardware-Software Co-Regulation

## 1. SpinQuant ($R_1, R_2, R_3, R_4$) & Online HadaCore Outlier Filters

Linear layers satisfy rotational invariance $Y = X W^\top = (X R)(W R)^\top$ for any orthogonal matrix $R \in \mathcal{O}(d)$ ($R^\top R = I$). By inserting four orthogonal rotations across the Transformer block, channel outliers ($\text{Kurtosis }\kappa \gg 3$) are flattened into a near-Gaussian distribution ($\kappa \approx 3$) prior to 1-bit (Binary g128) or ternary (BitNet b1.58) quantization:

1. **$R_1 \in \mathcal{O}(d_{\text{model}})$ (Residual Stream Rotation, Offline Merged, Zero Runtime Cost):**
   - Optimized on the Stiefel manifold via Riemannian Cayley SGD (`F13`) in [`fused_w1_hadacore/spinquant_packer.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/fused_w1_hadacore/spinquant_packer.py):
     $$Y_t = G_t R_t^\top - R_t G_t^\top, \qquad R_{t+1} = \left(I + \frac{\eta}{2} Y_t\right)^{-1}\left(I - \frac{\eta}{2} Y_t\right) R_t$$
   - Or factorized as a Kronecker-Stiefel rotation $R_{\text{kron}} = R_P \otimes H_{16} = (R_P \otimes I_{16})(I_P \otimes H_{16})$, where $(R_P^\top \otimes I_{16})$ is absorbed offline into the preceding projection ($W_O' = (R_P^\top \otimes I_{16}) W_O$) and $(I_P \otimes H_{16})$ executes online in the kernel's `mode=1` HadaCore register prologue.
   - Absorbed offline into $W_{\text{emb}}' = W_{\text{emb}} R_1$, $W_{Q,K,V}' = W_{Q,K,V} R_1$, $W_{\text{gate, up}}' = W_{\text{gate, up}} R_1$, $W_O' = R_1^\top W_O$, and $W_{\text{down}}' = R_1^\top W_{\text{down}}$.
2. **$R_2 \in \mathcal{O}(d_{\text{head}})$ (Per-Head Value-Output Rotation, Offline Merged, Zero Runtime Cost):**
   - Absorbed offline across attention heads via $W_V^{(h)\prime} = R_2^\top W_V^{(h)} R_1$ and $W_O^{(h)\prime} = R_1^\top W_O^{(h)} (I_H \otimes R_2)$.
3. **$R_3 \in \mathcal{O}(d_{\text{head}})$ (Online Post-RoPE Query/Key Hadamard Rotation):**
   - Applied online via block-diagonal $H_{16}$ to Queries and Keys after RoPE ($\widetilde{Q} = \text{RoPE}(Q)\mathcal{H}, \; \widetilde{K} = \text{RoPE}(K)\mathcal{H}$) to smooth KV-cache outliers before INT4 quantization.
4. **$R_4 \in \mathcal{O}(d_{\text{ffn}})$ (Online MLP Down-Projection Input Hadamard Rotation):**
   - Applied online via HadaCore (`mma.sync.aligned.m16n8k16` multiplying $16 \times 16$ activation tiles by $H_{16} \in \{-0.25, +0.25\}^{16 \times 16}$ in 16 SMSP cycles) inside the register prologue of [`fused_w1_hadacore/w1a16_hadacore_kernel.cu`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/fused_w1_hadacore/w1a16_hadacore_kernel.cu), with $R_4^\top$ absorbed offline into $W_{\text{down}}$.

---

## 2. Row-Max Holder Subspace Energy Sieve (`Tiled-CBC`) & LServe Sparse KV Attention

### A. Tiled-CBC Dynamic Subspace Gate ([`structural_filters/tiled_cbc_sieve.cu`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/structural_filters/tiled_cbc_sieve.cu))
- During the HadaCore register prologue, each warp holds the rotated activation slice $\widetilde{X}_{M_{\text{tile}}, 128}$ in `bf162` registers.
- Each thread strips the sign bits via bitwise AND with `0x7FFF7FFFU` (`bf162_abs_sum_f32`) and executes a 2-phase butterfly reduction (`warp_reduce_rowmax_l1_k32`: `__shfl_xor_sync` sums across the 4 lanes of each row `offset = 1, 2` followed by `__shfl_xor_sync` maxima across the 8 row groups `offset = 4, 8, 16`) to compute the exact worst-case row-maximum L1 energy (`O9`).
- A single hardware warp vote `__ballot_sync(0xFFFFFFFFU, E_k > tau_sieve)` (or per-$K=32$ sub-tile Holder bound $\max_m \|A_{m, k32}\|_1 \cdot \alpha_{\max} > \tau_{\text{sub}}$) predicates the `BLOCK_K = 128` step, skipping both the `cp.async.cg.shared.global` GDDR6X weight fetch and the `mma.sync.aligned.m16n8k16` instructions when the ballot returns zero (defaulting to exact $\tau = 0$, `O8`).

### B. LServe Unified Static + Dynamic Sparse 4-Bit Hadamard KV Attention ([`structural_filters/lserve_sparse_attn.py`](file:///d:/Coding_Projects/.Unified_Hardware_Accel/structural_filters/lserve_sparse_attn.py))
- Partitions attention heads into **Static Streaming Heads** (attending to $\Lambda$ initial sink pages + $W_{\text{local}}$ local sliding window pages in $O(1)$ time, selected by offline out-of-window vs. in-window salience profiling `profile_streaming_head_mask`, `F15`) and **Dynamic Retrieval Heads**.
- Groups KV tokens into 64-token pages, rotates $K, V$ via block-diagonal $H_{16}$, stores per-page channel-wise bounding vectors $(k^{\min}_p, k^{\max}_p) \in \mathbb{R}^D$, and quantizes $K, V$ to symmetric INT4 ($4.125\text{ bits/element}$ including the per-token BF16 scale; $K$ is $4.625$ once the per-page BF16 min/max centroids are counted). Centroids are rounded outward (min toward $-\infty$, max toward $+\infty$) so the page box contains every key and the score is a true upper bound.
- **Per-Query Convex Upper-Bound Page Scoring (`P1-5`):** Evaluates the worst-case convex upper-bound inner product across all queries $s \in \{1, \dots, S_q\}$ in the query block against page $p$ so opposing-sign queries never cancel out:
  $$\text{Score}(Q, p) = \max_{1 \le s \le S_q} \sum_{d=1}^{D} \max\left( \widetilde{Q}_{s, d} \cdot k^{\min}_{p, d}, \; \widetilde{Q}_{s, d} \cdot k^{\max}_{p, d} \right)$$
  Selecting only the top-$K_{\text{dyn}}$ retrieval pages plus static sink/local pages yields an $8.90\times$ modelled KV bandwidth reduction (bits actually read: selected pages' K and V payload plus the unselected pages' centroids; the earlier $10.04\times$ used 4.25 bpw and ignored centroid reads) while preserving `0.99313` (`0.99305` at 50% salience-profiled streaming heads) cosine similarity against dense FP32 attention.

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
