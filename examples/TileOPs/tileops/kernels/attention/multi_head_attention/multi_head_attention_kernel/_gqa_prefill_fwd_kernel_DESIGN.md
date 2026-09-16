# _gqa_prefill_fwd_kernel 算子设计文档（MultiHeadAttentionFwdOp · GPU→NPU 迁移 · Expert 模式 · manifest causal 多头域重设计 v3）

> **本轮任务定位**：同一源算子（`_gqa_prefill_fwd_kernel`，源码未变）的**全新 Stage 1 设计**（用户指令「从 Stage 1 算子生成重新开始」）。前序谱系：developer 迁移（2026-09-04，23/23 精度）→ expert E1-E7 单遍 per-block 跨引擎流水迁移（2026-09-07，design_v2，29/29 精度）→ 三轮 TileOPs 侧 case_fa 域（non-causal、H=1、大 S）调优（2026-09-08~09，均已归档）。用户判定上一版 kernel 是「基于 case_fa 特定 workload 生成的特化实现」——**causal manifest 域（本任务第一目标域）从未被结构化优化**：当前 causal full 路径跑在 wrapper 默认 (64,64,1) 的 E1-E7 单遍结构上，msprof 实测 reg8bshort 3764.05µs（≈2.25 TFLOPS）、reg8blong 13265.27µs（≈5.17 TFLOPS），而同机同口径的 fa4096 两相位结构已达 87.6 TFLOPS——headroom >10×。**本轮从算法调研层重新选型，不预设 E1-E7 结构为答案**；前序产物仅作证据输入。
>
> **上一版两处设计流程缺陷（perf_feedback.md 修正附录记录，本轮必须修复）**：① design_v2 §1.6/§0.6 算法调研未纳入仓库内两遍式 S/P 物化先例（`examples/flash_attention/flash_attn_npuir.py`，同 API 可达结构），单遍 per-block 结构被错误选定为唯一结构；② 对两遍式的否决理由（「S 需全任务驻留 → flag 预算超 15 上限」）系推理错误——全局 ws + n-block 下标 flag（nk≤16）即可成立，第三轮 v11nt 为直接反例。**本轮调研四问已完整覆盖 examples/ 全目录先例检索（§1.6.0 调研范围清单）**。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 references/migration-analysis.md）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。源码未变，§0.1–§0.4 与 design_v2 同源同结论（已经 29/29 测试验证），本轮独立重读源码复核后转录，并新增 §0.5/§0.6 的**结构级重选型**（对照 §1.6.0 调研结论）。

### 0.1 源算子语义（做什么）

**源算子路径（source of truth，三问解读对象）**：`/home/tilelang/l00970450/TileOPs/tileops/kernels/attention/gqa_fwd.py` L788–L1033（`_gqa_prefill_fwd_kernel` 工厂 + `_gqa_prefill_fwd_wrapped_kernel` + `GQAPrefillFwdKernel` 类）；辅助 macro 工厂：`/home/tilelang/l00970450/TileOPs/tileops/kernels/online_softmax.py`（`make_online_softmax_with_mask_guard` / `make_apply_softcap` / `make_rescale` / `LOG2E`）。本仓提取件（逐字节）：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/_multi_head_attention_fwd_kernels.py`。
**NPU wrapper 调用契约（Stage 3 产物必须匹配，不可协商）**：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention.py`——`_gqa_prefill_fwd_kernel(batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, is_causal, sm_scale, softcap, dtype)(block_m, block_n, num_stages)(q, k, v)`（**无 `threads`**，K3/K9），返回 `(output, lse)`；`default_config = {"block_m": 64, "block_n": 64(dim≤128 else 32), "num_stages": 1}`；custom_op `npub::gqa_prefill_fwd_wrapped_kernel` 以位置参数调用工厂并以 `(block_m, block_n, num_stages)(q, k, v)` 形态取结果（该文件 L96–L102 亲自核对）。
**源输出 shape（从源码 + wrapper 契约核实）**：`output (batch, seq_len_q, heads, dim)`（同 q，BSHD，非转置布局）；`lse (batch, heads, seq_len_q)` fp32。依据：源码 L832–L833（`output: T.Tensor(o_shape, dtype)`、`lse: T.Tensor([batch, heads, seq_len_q], accum_dtype)`，accum_dtype="float"）与 wrapper `register_fake`（`fake_o = torch.empty_like(inputs[0])`、`fake_lse = fake_o.new_empty([batch, heads, seq_len_q])`）。
**迁移范围**：仅 `_gqa_prefill_fwd_kernel`（GQA/MHA prefill 前向 dense 路径）；GPU 侧 warp-specialized persistent 变体与 H200 square fast path 是 GPU arch 特化，NPU 侧 collapse 为本 kernel（wrapper Part B 已按 K5 收敛到 `GQAPrefillFwdKernel`，manifest `examples/TileOPs/tileops/manifest/attention.yaml` L59–L65 同口径）。

**数学语义**（flash-attention 风格 online-softmax 前向，BSHD 布局，MHA 为 `heads_kv == heads` 特例、GQA 为 `heads % heads_kv == 0` 分组共享）：

$$
S_{b,h,i,j} = \langle q_{b,i,h,:},\, k_{b,j,h_{\text{kv}},:} \rangle,\qquad h_{\text{kv}} = h \,//\, \text{groups},\quad \text{groups} = \text{heads}\,//\,\text{heads\_kv}
$$

$$
\tilde{S}_{b,h,i,j} = \begin{cases} \text{softcap}\cdot\tanh\!\big(\text{scale}_{\text{sc}}\cdot S_{b,h,i,j}\,/\,\text{softcap}\big) & \text{softcap} > 0 \\ S_{b,h,i,j} & \text{else} \end{cases},\qquad \text{scale}_{\text{sc}} = \text{dim}^{-0.5}（sm\_scale 可覆盖）
$$

$$
M_{b,h,i,j} = \begin{cases} -\infty & masked（causal 越界 / seq 越界） \\ 0 & \text{valid} \end{cases},\qquad
o_{b,h,i,:} = \frac{\sum_j e^{\tilde{S}_{b,h,i,j} + M_{b,h,i,j}}\, v_{b,j,h_{\text{kv}},:}}{\sum_j e^{\tilde{S}_{b,h,i,j} + M_{b,h,i,j}}}
$$

$$
\text{lse}_{b,h,i} = \log_2\Big( \sum_j e^{\tilde{S}_{b,h,i,j} + M_{b,h,i,j}} \Big)\quad（log2 域 log-sum-exp）
$$

causal 右对齐偏移 `causal_offset = seq_len_kv − seq_len_q`，mask 条件 `valid = (q_pos < seq_len_q) & (kv_pos < seq_len_kv) & (q_pos + causal_offset >= kv_pos)`（causal）/ `valid = (q_pos < seq_len_q) & (kv_pos < seq_len_kv)`（非 causal），源码 L888–L900 逐字核对。ref_api：`torch.nn.functional.scaled_dot_product_attention`（is_causal；L==S 时与右对齐 causal 等价）。

- **规约语义（累加顺序）**：softmax 分母与 PV 分子均按 **KV 块顺序增量累加**（online softmax）：`logsum ← logsum · e^{m_prev − m_cur} + Σ_j e^(块内分数 − m_cur)`；PV 累加器 `acc_o ← acc_o · e^{m_prev − m_cur} + P_tile @ V_tile`。KV 块大小与块内归约并行分组迁移后会变（源 warp 归约树 → NPU L0C 累加 + Vector 行归约 + 本设计的两相位部分和延迟回放），仅产生 fp32 ulp 级舍入差异（论证见 §0.6 E2/E4；机器验证见 §1.6.1 M14）。
- **dtype 语义**：q/k/v 输入 fp16/bf16；**中间计算全程 fp32**（`accum_dtype = "float"`：acc_s/acc_o/scores_max/scores_max_prev/scores_scale/scores_sum/logsum 全 fp32，源码 L808）；`acc_s → acc_s_cast` 在第二次 GEMM 前 **cast 回输入 dtype**（源 L911 `T.copy(acc_s, acc_s_cast)`，round-to-nearest）；output 同输入 dtype；lse fp32。
- **lse 域语义（易错点，从源码逐项推导）**：源 `scale = score_scale * LOG2E`（无 softcap）或 `scale = LOG2E`（有 softcap，源 L799–L801），online softmax 用 `exp2(x·scale − m·scale)`；最终 `lse = log2(logsum) + scores_max * scale`（源 L922/L933）。代入展开：无 softcap 时 `lse = log2(Σ e^{scale_sc·qk})`；有 softcap 时 `lse = log2(Σ e^{softcap·tanh(·)})`——**两分支 lse 均为 log2 域 log-sum-exp**。golden 按 `torch.logsumexp(scores, dim=-1) / math.log(2.0)` 对齐（谱系 D7 校准教训：不能写成 `torch.log2(torch.logsumexp(...))` 双重对数）。
- **边界语义**：
  - **q 尾块**（`seq_len_q % block_m ≠ 0`）：源 OOB 行填 0（L855–L860），对应行输出走 predicated store（只写 `q_pos < seq_len_q` 行，L926–L933）；
  - **kv 尾块**（`seq_len_kv % block_n ≠ 0`）：k/v OOB 行填 0 且 mask 把 OOB 列置 −inf（softmax 不可见，L880–L900）；
  - **causal 对角块**：块内逐位置 mask（`q_pos + causal_offset >= kv_pos`）；
  - **NaN guard（防御性）**：单调 max（`scores_max = max(scores_max, scores_max_prev)`）+ clamp ≥ −1e38（online_softmax.py L128–L131），防全 −inf 块的 `exp2(−inf − −inf)` 产生 NaN。causal 下每有效行至少一个有效位置（`q_pos + causal_offset ≥ 0 = kv_pos 0` 恒成立），正常路径不触发；q 尾块垃圾行可能全 −inf 但输出被行截断丢弃——NaN guard 是源防御逻辑，**原样保留**（§1.6.1 M6）；
  - **参数校验**：`heads % heads_kv != 0` 与（causal 且 `seq_len_q > seq_len_kv`）在工厂层 raise ValueError（host 语义，保留）；
  - NaN/Inf：逐元素传播（sm_scale=0 × masked −inf = NaN 的病态路径，源与迁移后行为一致，§0.6 E3/E4 论证）。
- **host 侧语义**：`functools.lru_cache(32)` 编译缓存；`sm_scale=None → dim**-0.5`、`softcap=0.0` 禁用、`use_softcap → scale = LOG2E`——工厂层常量折叠，属算子契约。无 im2col 类预处理（源 host 无 tensor 改写）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（优先移植源仓测试 `examples/TileOPs/tests/ops/test_multi_head_attention.py` 的 SDPA 模式 + lse 补充），**不复刻 §0.6 的 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `q` | 输入 | `(batch, seq_len_q, heads, dim)`（BSHD） | float16 / bfloat16 | query；全静态 shape（工厂参数确定，无运行时动态轴） |
| `k` | 输入 | `(batch, seq_len_kv, heads_kv, dim)`（BSHD） | same_as(q) | key（GQA 分组共享，`by // groups` 映射 kv 头） |
| `v` | 输入 | `(batch, seq_len_kv, heads_kv, dim)`（BSHD） | same_as(q) | value |
| `output` | 输出 | `(batch, seq_len_q, heads, dim)` | same_as(q) | **输出 shape 同 q（非转置布局）** |
| `lse` | 输出 | `(batch, heads, seq_len_q)` | float32 | 每行 log2 域 log-sum-exp |

prim_func 计算参数名与顺序 `(q, k, v, output, lse)` 迁移前后语义保持（本设计新增 workspace 参数与 lse 视图增维的插入位置见 §0.6 E6：q/k/v 仍居前、output/lse 仍居末，`out_idx` 输出居末语义等价保持）。

### 0.3 实现算法解读（怎么算）

**工厂与 host 侧逻辑**（源 gqa_fwd.py L788–L827）：

1. `_gqa_prefill_fwd_kernel(batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, is_causal, sm_scale=None, softcap=0.0, dtype='float16')`（`@functools.lru_cache(maxsize=32)`）→ 返回内层工厂 `_gqa_prefill_fwd_func`。
2. 工厂层常量折叠：`score_scale = dim**-0.5 if sm_scale is None else sm_scale`；`use_softcap = softcap > 0.0`；`scale = LOG2E if use_softcap else score_scale * LOG2E`（LOG2E = 1.44269504）；`groups = heads // heads_kv`；`causal_offset = seq_len_kv − seq_len_q`；`accum_dtype = "float"`；参数校验。
3. `@tilelang.jit(out_idx=[3, 4], pass_configs={TL_ENABLE_FAST_MATH: True}, compile_flags=["-O3", "-DENABLE_BF16"])` 装饰 `_gqa_prefill_fwd_func(block_m, block_n, num_stages, threads)` → 返回 `@T.prim_func _gqa_prefill_fwd_main`。
4. 三个 `@T.macro` 工厂在 JIT 闭包内实例化（`make_online_softmax_with_mask_guard` / `make_apply_softcap`（条件）/ `make_rescale`）。

**kernel 计算步骤分解**（`T.Kernel(ceildiv(seq_len_q, block_m), heads, batch, threads=threads) as (bx, by, bz)` **三维 grid**；每 block 处理一个 (q-block, query-head, batch)；源码每个计算语句均归入下表，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| S1 q 装载 | 对齐块 `T.copy(q[bz, bx·bm:(bx+1)·bm, by, :], q_shared, disable_tma=True)`；尾块 `T.Parallel(bm, dim)` 谓词装载（OOB 行填 0） | q (GM 4D) | q_shared `[bm, dim]` (SMEM) | q 的 O 块驻留 |
| S2 初始化 | `T.clear(acc_o)`；`T.clear(logsum)`；`T.fill(scores_max, −inf)` | — | acc_o/logsum/scores_max (fragment fp32) | 累加器清零 / max 初值 |
| S3 KV 流水循环 | `for k_idx in T.Pipelined(loop_range, num_stages)`，`loop_range = ceildiv((bx+1)·bm + causal_offset, block_n)`（causal，**依赖 bx 的运行时边界**）或 `ceildiv(seq_len_kv, block_n)` | — | — | KV 维分块（causal 裁剪） |
| S4 k/v 装载 | 对齐块两条 `T.copy(k/v[...], k/v_shared, disable_tma=True)`；尾块谓词装载（OOB 行填 0） | k/v (GM 4D) | k_shared/v_shared `[bn, dim]` (SMEM) | K/V 流式块 |
| S5 mask 预填 | `T.Parallel(bm, bn)`：`acc_s[i,j] = if_then_else(valid, 0, −inf)` | 坐标谓词 | acc_s `[bm, bn]` (fragment fp32) | mask M（causal + OOB） |
| S6 QK GEMM | `T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=FullRow)`（**累加模式**：acc_s = mask + QK^T） | q_shared/k_shared | acc_s (fragment fp32) | 点积（fp32 累加） |
| S7 softcap（条件） | `capped = softcap·tanh(acc_s·score_scale/softcap)`，`acc_s = if_then_else(acc_s == −inf, −inf, capped)` | acc_s | acc_s（原地） | softcap 分支（−inf 保持） |
| S8 online softmax | macro：① `copy(scores_max→prev)`；② `fill(−inf)`；③ `reduce_max(clear=False)`；④ 单调 max；⑤ clamp ≥ −1e38；⑥ `scores_scale = exp2(prev·scale − cur·scale)`；⑦ `acc_s = exp2(acc_s·scale − max·scale)`；⑧ `reduce_sum`；⑨ `logsum = logsum·scores_scale + scores_sum` | acc_s | max/scale/sum/logsum (fragment) | online softmax（exp2·LOG2E 域） |
| S9 P cast | `T.copy(acc_s, acc_s_cast)`（fp32 → 输入 dtype） | acc_s | acc_s_cast `[bm, bn]` | P_tile 降精度（GEMM2 输入） |
| S10 rescale | `rescale(acc_o, scores_scale)`：`T.Parallel(bm, dim)` `acc_o[i,j] *= scores_scale[i]` | acc_o | acc_o | 历史累加器缩放 |
| S11 PV GEMM | `T.gemm(acc_s_cast, v_shared, acc_o, policy=FullRow)`（**累加模式**） | acc_s_cast/v_shared | acc_o (fragment fp32) | PV 累加 |
| S12 epilogue（对齐块） | `acc_o /= logsum`；`T.copy(acc_o → output)`；`logsum = log2(logsum) + scores_max·scale`；`T.copy(logsum → lse)` | acc_o/logsum/max | output (GM)、lse (GM) | 归一化输出 + log2 域 lse |
| S13 epilogue（尾块） | 谓词版：仅 `q_pos < seq_len_q` 行写 output/lse（标量 GM 写） | 同上 | 同上 | 行截断写出 |

**数据流（源硬件视角）**：

```
GM[q] ──T.copy/谓词装载──> SMEM[q_shared (bm, dim)]（一次装载，全部 KV 块复用）
GM[k]/GM[v] ──T.copy/谓词装载──> SMEM[k_shared/v_shared (bn, dim)]（Pipelined 流式）
SMEM[q,k] ──T.gemm(transpose_B, FullRow, 累加)──> REG[acc_s (bm, bn) fp32]（含 mask 预填）
REG: acc_s ← softcap·tanh（条件）→ online_softmax（exp2 域 max/sum/logsum 增量）
REG: acc_s ──T.copy(cast)──> REG[acc_s_cast (bm, bn) dtype]
REG: acc_o *= scores_scale（行广播）；SMEM[v] ──T.gemm(累加)──> REG[acc_o (bm, dim) fp32]
REG: acc_o /= logsum；logsum = log2(logsum) + max·scale
REG[acc_o/logsum] ──T.copy / 谓词标量写──> GM[output (4D)] / GM[lse (3D)]
```

**循环与并行结构**：grid = `(ceil(seq_len_q/block_m), heads, batch)` 三维 + `threads`（CUDA 线程块）；核内 1 个 `T.Pipelined(loop_range, num_stages)` KV 流水（loop_range 依赖 bx）；块内 `T.Parallel` 元素级循环（mask 预填 / exp2 / rescale / epilogue）+ `T.reduce_max/sum(dim=1)` warp 归约。

**host 侧逻辑清单**：`lru_cache(32)`；参数校验与常量折叠；`pass_configs={TL_ENABLE_FAST_MATH: True}` + `compile_flags=["-O3", "-DENABLE_BF16"]`（CUDA 编译配置）；无 tensor 预处理/后处理。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | online softmax（分块增量 max/sum/logsum + rescale） | 消 O(S_kv×S_q) score 矩形 GM 往返，单遍 KV | 分块 e 域稳定化 + 增量累加 | **无（纯算法层）** | **可移植** |
| 2 | SMEM tiling（q 驻留复用 + k/v 流式分块） | q 一次装载全 KV 复用；k/v 分块限 SMEM | `alloc_shared` + 分块 `T.copy` | CUDA shared memory | 硬件强相关 |
| 3 | `T.Pipelined(loop_range, num_stages)` 软件流水 | KV 装载与计算重叠（cp.async） | 多 stage SMEM 乒乓 | Ampere+ 异步拷贝引擎 | 硬件强相关 |
| 4 | Tensor Core GEMM（QK^T + PV） | 矩阵算力 | `T.gemm` + `GemmWarpPolicy.FullRow` | NVIDIA tensor core / warp | 硬件强相关（NPU 有 Cube 等价） |
| 5 | exp2 + LOG2E 折叠（`exp2(x·scale − m·scale)`） | 乘法融进 ffma，消独立 mul | 换底恒等式 e^u = 2^(u·log2e) | CUDA ffma 融合 | 硬件强相关（实现层技巧） |
| 6 | register/fragment 累加（acc_s/acc_o 全程驻留，epilogue 融合） | 消中间 GM/SMEM 往返 | fragment 寄存器复用 | CUDA 大寄存器堆 | 硬件强相关 |
| 7 | `T.copy(acc_s, acc_s_cast)` fp32→fp16 降精度后进 GEMM2 | GEMM2 用 fp16 tensor core | dtype cast copy | tensor core 输入精度 | 模式可移植（NPU Cube 同款约束） |
| 8 | causal 循环裁剪（`loop_range = ceildiv((bx+1)·bm + offset, bn)`） | 跳过因果窗外 KV 块（约省一半 GEMM/装载） | 运行时标量循环边界 | GPU 运行时循环边界 | 硬件强相关（NPU 边界须静态/符号化） |
| 9 | mask 预填 + 累加 GEMM（S5+S6 一条 GEMM） | mask 融合进 GEMM 累加，免独立 mask pass | gemm 累加语义（C += AB） | tensor core C 累加 | 模式可移植（两相位下由 Vector vselect 承接，见 §0.5） |
| 10 | 对齐/非对齐双路径（整块 T.copy vs 谓词逐元素） | 对齐块向量化、非对齐正确性兜底 | CUDA 访存向量化 + 标量谓词 | warp 标量访存 | 硬件强相关 |
| 11 | `disable_tma=True` | 规避 TMA 对 4D 切片限制 | copy 选项 | NVIDIA TMA | 硬件强相关（NPU 无 TMA） |
| 12 | `TL_ENABLE_FAST_MATH` + `-O3 -DENABLE_BF16` | 放宽浮点变换 / 使能 bf16 | nvcc / TVM pass | CUDA 编译链 | 硬件强相关 |
| 13 | host `lru_cache(32)` 编译缓存 | 消重复 JIT | functools | 无 | 可移植 |
| 14 | 三维 grid (bx, by, bz) + threads 并行组织 | (q-block × head × batch) 一次铺满 SM | CUDA grid/block 两级并行 | CUDA 线程模型 | 硬件强相关 |

> 识别不出来 ≠ 不存在。未列出的优化会在迁移中被静默丢弃，导致性能回退无法追溯。

### 0.5 硬件耦合性分析与 NPU 适配决策（Expert 模式口径）

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？**本轮判定输入含 §1.6.0 调研结论**：源算法（单遍 per-block online-softmax 链）与调研候选（两相位 S/P 物化等）同台比较——调研选定两相位结构后，§0.4 各优化手段的处置对照该结论逐项重判（承接规则见 §1.6.0 调研结论段）。与前序 E1-E7 设计的关键差异：**承载结构从「per-block 跨引擎串行链（C1→FLAG_S→V1→FLAG_P→C2→FLAG_O→V2）」换为「两相位独立大循环（Cube pass-1 全 S → pass-2 全 PV；Vector pass-1 softmax 链 + rescale 因子延迟回放 → pass-2 累加）」**，依据为第三轮 msprof 实测（fa4096: 226.72 → 98.05µs，PL-1.9-twophase）与本轮目标域画像（§1.6.0 R3/R4）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（公式/右对齐 causal/lse log2 域/dtype/P 量化点/边界/输出 shape） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1） |
| online softmax 算法结构（增量 max/sum/logsum + rescale + 最终 ÷logsum） | 算法 | 无 | — | **保留** | 纯算法层（§5.4 规则 2）；两相位下 online 递推在 Vector pass-1 逐 n-block 保持，rescale 因子存 UB `scales[]` 延迟到 pass-2 回放（flash_attn_npuir.py L170–L172/L253–L261 同构）；换底方式等价替换（E4） |
| q 驻留复用 + k/v 流式分块 + causal 块级上界裁剪 | 算法 | 无 | — | **保留** | 分块与裁剪逻辑是算法层；承载走 E1（persistent 任务解码）/E2（两相位 pass 循环）/E7（size-form 尾钳位） |
| softcap 的 −inf 保持逻辑 | 算法 | 无 | — | **保留** | `T.vtanh`（docs/Tilelang.language/数学操作/T.vtanh.md，fp32 操作数——debug_log D5）+ `T.vselect`（docs/Tilelang.language/条件操作/T.vselect.md）NaN 无关选择，mask 在 softcap 之后恢复 −inf（E3 顺序论证） |
| `acc_s → acc_s_cast` fp32→fp16 后进 GEMM2 | 算法 | tensor core 输入精度 | 有（Cube fp16/bf16 输入 fp32 累加，T.gemm.md §2.2.1；bf16×bf16→fp32 §2.3 实测可用；PL-1.8 Expert 端到端实测） | **等价替换** | Vector 侧 `T.vcast(..., round_mode="rint")` 到**输入 dtype** 后经 ws_p 进 Cube（量化点与源逐位同 dtype，M10） |
| host `lru_cache(32)` + 参数校验 + 常量折叠 | host | 无 | — | **保留** | host 层与接口兼容性 |
| SMEM tiling | 优化 | CUDA SMEM | 有（L1 512KB / UB 192KB） | **等价替换** | Cube 侧 `T.alloc_L1`（l1_a/l1_b 生命周期复用，§4.3）；migration-analysis.md §5.3「shared memory → L1/UB」 |
| `T.Pipelined` 软件流水 | 优化 | cp.async | **意图由两相位结构承接**（见下行） | **重新设计** | → E2：装载/计算重叠意图不再依赖 per-block 乒乓——Cube pass-1 无等待直跑（装载与 gemm 经 MTE2/MTE1/C 通道天然重叠）、pass-2 逐块等 P；`T.Pipelined` 不再使用（v11nt 同构，PL-1.9-twophase） |
| Tensor Core GEMM | 优化 | NVIDIA tensor core | 有（Cube `T.gemm` + L0A/L0B/L0C） | **等价替换** | `T.gemm(l1_a, l1_b, l0_c, initC=True, b_transpose=True, size=[tm, dim, tn])` / `T.gemm(l1_a, l1_b, l0_c, initC=True, size=[tm, tn, dim])`（Expert 签名，T.gemm.md §2.1；flash_attn_npuir.py L102–109/L145–151 同构） |
| fragment fp32 累加 + 寄存器 epilogue | 优化 | CUDA 寄存器堆 | **无直接等价**（L0C 不能被 v 算子直接操作；L0C 出数目标仅 GM——docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1 L17–L18「src 必须来自 L0C / dst 必须来自 GM」） | **重新设计** | → E2：S 与每块 PV 部分和经 GM workspace 跨引擎往返（API 强制）；acc_o 驻留 Vector UB，pass-2 延迟 rescale 累加（v11nt 同构） |
| warp 归约（`reduce_max/sum(dim=1)`）+ `T.Parallel(bm)` 单调 max | 优化 | warp shuffle 归约 | 有（Vector 硬件行归约 + vmax） | **等价替换** | `T.reduce(ub, dst, dims=[1], reduce_mode="max"/"sum", clear=True)`（docs/Tilelang.language/规约操作/T.reduce.md §2.2.2「src/dst 仅一维不同且为 1」；**clear 必须显式 True**——§2.3「clear=False 对未初始化 buffer 不报错但静默数值错误」，v11 的 ell 修正即此）+ `T.vmax`（数学操作/T.vmax.md）；累加分组变化论证 E2/E4 |
| 行级 rescale / logsum 更新 | 优化 | warp 标量并行 | 有（v 算子行广播） | **等价替换** | `T.vmul(ub, ub_rowvec, ub)`（[M,N]×[M,1] 行广播，数学操作/T.vmul.md §2.2.2）+ per-block 因子存取 `T.copy(scales[i·half:…])`（v11 L217–L222/L253–L259 同构） |
| **mask 预填 + 累加 GEMM（S5+S6 融合）** | 算法 | tensor core C 累加 | 有（initC 累加），**但两相位 gemm1 恒 initC=True 逐块覆写（无 mask 预填位）**；且 developer 实测 T.Parallel 谓词预填是 aiv_scalar 88–89% 主瓶颈（opt_log §1） | **重新设计** | → E3：mask 移至 **Vector pass-1、S 到达后 softmax 前**，全向量构造（`T.arange`/`T.vsub`/`T.vcmp`/`T.vand`/`T.vselect`）+ 两段式 K_A 分界使全有效块零 mask 开销（E1-E7 E3 全套论证与 floordiv 修正直接承接——两相位下 mask 链位置从「V1 块内」平移到「pass-1 块内」，机制不变） |
| causal 运行时循环边界 | 优化 | GPU 运行时边界 | 部分（kernel 索引派生 PrimExpr 边界有先例） | **保留（符号化）** | `NK = T.ceildiv(T.min((bx+1)·bm + causal_offset, seq_len_kv), bn)`（E1-E7 D3 实证可 lower；开发指南 §3.3 模板同类的 kernel_id 派生边界；v11 亦用 `T.min` 尾钳位同族形态） |
| 对齐/非对齐双路径（谓词装载/谓词回写） | 优化 | CUDA 标量谓词访存 | 部分免（size-form gemm + 尾钳位切片 copy 官方模式） | **重新设计** | → E7：`tail = T.min(...)` + size-form `T.gemm(size=[tm, k, tn])` + 同形切片 `T.copy`（flash_attn_npuir.py L88–118 全套先例）+ 分形下限钳位 + trace-time 尾带 OOB 掩码变体 + 行截断写出（§6.4）；**per-scalar GM 写与逐元素谓词装载全部消除** |
| **bf16 输入路径** | 优化（dtype 承载） | GPU bf16 tensor core | **有**（T.gemm.md §2.2.1 Ascend 行 bf16 = √；§2.3「bf16 x bf16 场景中，dst/累加类型建议设置为 fp32（实测可用）」；PL-1.8 Expert 端到端实测无延迟税） | **等价替换** | Cube bf16 直连（q/k/v slice-form `T.copy` bf16 进 L1、`T.gemm` bf16×bf16→fp32）；**两相位 bf16 端到端组合无先例（v11 为 fp16 专用）→ R-2 探针（L0-2 前置首验）+ 回退（ws_s/ws_o 升 f32，结构不变）** |
| per-scalar GM 写（尾块 predicated store） | 优化（结构） | GPU 标量 store | **无**（触发 MTE 异常，E1-E7 谱系结论） | **重新设计**（并入 E7） | 行级切片 `T.copy` 写出（v11 L1443–L1446 `output[0, bx:bx+real_m, 0, 0:dim]` 4D 切片先例） |
| `GemmWarpPolicy.FullRow` | 优化 | CUDA warp | 无（NPU 无 warp 概念） | **舍弃** | Cube 矩阵并行由分形（M/N 切分进 L0A/L0B/L0C）承担，`b_transpose=True` 即完整语义。warp 划分是 CUDA 调度细节，NPU 无对应物且无需对应 |
| `threads=128`（内层工厂参数） | 优化（结构） | CUDA 线程模型 | 无 | **舍弃** | K3/K9：wrapper 已按 `(block_m, block_n, num_stages)` 无 threads 签名调用（multi_head_attention.py L96–L102 注释亲自核对） |
| `disable_tma=True` | 优化 | NVIDIA TMA | 无 | **舍弃** | NPU 侧搬运走 slice-form `T.copy`（D1 约束），无此参数 |
| `TL_ENABLE_FAST_MATH` + `compile_flags` | 优化 | CUDA 编译链 | 无 | **舍弃** | npuir 分支无此 pass（迁移硬性规则）；改设 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False` + `NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`（v11nt / highperf 先例：防编译器重排手动两相位流） |
| exp2 + LOG2E 折叠 | 优化 | CUDA ffma 融合 | 有等价（`T.vexp` e 底） | **等价替换（换底方式变更）** | → E4：e 域 `T.vexp` 直接实现（数学恒等、数值路径更短，v11 L1392–L1395 同构）；lse 保持 log2 域（`T.vlog2(src, dst, tmp)` 三参签名，docs/Tilelang.language/数学操作/T.vlog2.md §1） |
| 三维 `T.Kernel(grid_m, heads, batch, threads=)` | 优化（结构） | CUDA grid/block 两级并行 | **无**（T.Kernel 一维约束） | **重新设计** | → E1：一维 persistent `T.Kernel(24, is_npu=True) as (kernel_id, subid)` + 分核策略（§5.5） |

**统计**（v2 口径纪律：每行恰计一次，合计 = 表体 23 行）：保留 6 项（语义 1〔计算语义〕+ 算法 4〔online softmax 递推 / q-kv 分块与 causal 裁剪 / softcap −inf / causal 符号化边界〕+ host 1〔lru_cache 与校验〕）/ 等价替换 7 项（SMEM tiling、Tensor Core GEMM、warp 归约、行级 rescale、exp2 换底、P cast 量化点、bf16 直连）/ 重新设计 6 项（E1 三维 grid 一行、E2 两相位数据流两行〔T.Pipelined 与 fragment 累加〕、E3 mask 一行、E7 尾块两行〔对齐双路径与谓词标量 GM 写〕）/ 舍弃 4 项（FullRow、threads、disable_tma、FAST_MATH+compile_flags）。E6 接口适配为 §0.6 契约适配项（无 §0.5 源优化对应行，不参与计数）。6+7+6+4 = 23 ✓。

### 0.6 NPU 算法重设计（Expert 模式）

> 总体结构（§1.6.0 调研选定，结构原型 = `examples/flash_attention/flash_attn_npuir.py` + TileOPs 侧 v11nt tuned 变体，PL-1.9-twophase）：**一维 persistent Kernel（24 核）× 双 Scope（Cube/Vector）× 两相位独立大循环**。每个逻辑任务 = (q-block, query-head, batch)；任务内：**Cube pass-1** 串行计算全部 S 块（QK^T，Q-hoist 一次装载）物化到 per-core GM workspace；**Vector pass-1** 逐块消费 S（vcast → softcap → mask → prescale → online max/ell 递推 → P 量化）回写 workspace 并把 per-block rescale 因子存 UB `scales[]`；**Cube pass-2** 逐块消费 P 计算 PV 部分和（initC=True 逐块覆写）物化；**Vector pass-2** 逐块延迟 rescale 累加 acc_o；epilogue 归一化 + transpose-free lse 写出。跨引擎协同 = per-n-block 下标 flag（S/P/O 三次握手复用同一 id，严格 per-id 交替）+ 任务边界 FLAG_TASKDONE。**与 E1-E7 单遍结构的关键差异**：Cube pass-1 全程无等待（装载/gemm 重叠不被 flag 自旋阻塞——实测该自旋是单遍结构的 aic_scalar 38–47% 地板，PL-1.9-hardlimits）；PV 与 acc_o 的 α 耦合被 scales 延迟回放解耦（Cube 无需知道 α）。

**重设计项 E1: 三维 grid + threads → 一维 persistent 双 Scope Kernel + 分核策略**

- **源方案**：`T.Kernel(ceildiv(seq_len_q, block_m), heads, batch, threads=threads)` 三维 grid；每个 (q-block, head, batch) 一个 CTA。
- **NPU 新算法**：一维 `T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid)`，**NUM_KERNELS = 24 = `NPUUtils.get().get_aicore_num()` 实查值**（查询代码与记录见 §5.5；本机 910B2C 于 2026-09-15 实查返回 24，与 2026-09-07 双源记录一致）。逻辑任务数 `num_logical = ceildiv(seq_len_q, bm) × heads × batch`，任务解码 `cid = task_id·24 + kernel_id → (bx = cid // HB, by = (cid // batch) % heads, bz = cid % batch)`，`HB = heads·batch`（E1-E7 同款，D3 实证）。核内 persistent 串行：`num_local_tasks = T.ceildiv(num_logical − kernel_id, 24)`（官方模板形态，`cid < num_logical` 恒成立无需 guard）。**同一 kernel 内两个 Scope 顺序书写**：`with T.Scope("Cube")` [persistent 任务循环 + pass-1/pass-2 两个 `T.serial(NK)` 大循环] 与 `with T.Scope("Vector")` [persistent 任务循环 + pass-1/pass-2 + epilogue]——两段在硬件上并行执行于同一 AI Core 的 Cube 单元与 Vector 子单元（×2，`subid` 区分）。Vector 侧按 v11 自适应行半分：`real_m0 = (tail_m + 1) // 2`，AIV0 行域 `[s_lo, s_lo + real_m0)`、AIV1 行域 `[s_lo + real_m0, s_lo + tail_m)`（flash_attn_npuir.py L188–L191 同构，奇数尾自然处理）。
- **语义保持论证**：(q-block, head, batch) 逻辑任务之间完全独立——读写不相交的 `output[bz, s_lo:s_lo+tail_m, by, :]` / `lse[bz, by, s_lo:s_lo+tail_m]` 切片，q/k/v 只读共享（GQA 下多 query head 读同一 kv head 切片，只读无竞争）。任务到核映射与核内串行顺序不改变任何数值——**数学等价**。subid 行半分只切分行维：所有 Vector 操作（mask/exp/归约/rescale/累加）均为行内或行广播操作，行间无数据流动，两半并集 = 整块——**逐位等价**。causal 负载均衡：q-block 越大 KV 循环越长（NK 随 bx 线性递增），轮转分配（cid 步长 24）使每核任务的 bx 均匀覆盖 `[0, num_q_blocks)`（对账见 §5.5）。**NK 的 `T.min(·, seq_len_kv)` 截断等价性**（E1-E7 v1 补论证，直接承接）：源 loop_range 无 seq_len_kv 截断，q 尾块场景可迭代到全 OOB 尾块；本设计跳过这些块。等价性分两域：① **有效行**——被跳过的全 OOB 块若不跳过则为恒等更新（单调 max 不下降 → α = e^0 = 1；P = e^{−inf − m} 对有限 m 下溢精确 +0.0 → Δℓ = 0、Δo = 0）；② **垃圾行**（q 尾 OOB 行）——两序数值不同但输出被行截断丢弃（E7）。结论：截断仅省略对任何写出行为无贡献的块，安全且必须保留。

**重设计项 E2: fragment 流水 + per-block 跨引擎链 → 两相位独立大循环 + GM workspace 物化 + per-n-block flag ⭐（本轮核心）**

- **源方案**：`T.Pipelined` KV 流水 + fragment 上 v 链（acc_s/acc_o 全程驻留寄存器，epilogue 融合）。E1-E7 迁移后形态：每 KV 块 Cube/Vector 四段握手链 C1(i)→FLAG_S→V1(i)→FLAG_P→C2(i)→FLAG_O→V2(i)。性能意图：装载与计算重叠、消中间往返。**实测推翻（fa 域，PL-1.9-hardlimits）**：`sync_block_wait` 阻塞整条发射流——Cube 的 C1(i+1) 排在 C2(i) 的 P-wait 之后，每次等待自旋毒化后续全部发射（busy 核 aic_scalar 38–47%）；ns=1≈ns=2、wide≈serial、深流水 v10_dp +12% 回退三重实测。causal 域同结构 (64,64,1) 实测 3764/13265µs（≈2.25/5.17 TFLOPS）。
- **NPU 新算法（两相位，v11nt 同构 + causal 适配）**：数据通路拆为**每核独立的单槽 GM workspace 三族**（尺寸 §4.3；单槽论证见下）：
  - `ws_s [24, bm, pad_kv]`（fp16 路径 f16 / bf16 路径 f32）：Cube→Vector 的 QK 分数块（`T.copy(l0_c[0:tmc, 0:tnc] → ws_s[kid, 0:tmc, n_lo:n_lo+tnc])`，L0C f32 → f16 隐式转换——PL-1.9「store_fixpipe f32→f16 为正确数值转换（probe 实测 max_rel 4.7e-4 = 1 f16 ulp）」；v11p2 实测 `T.copy` 与 `T.store_fixpipe` 同效，取 v11nt 终版 `T.copy` 形态）；
  - `ws_p [24, bm, pad_kv]` 输入 dtype：Vector→Cube 的 P 块；
  - `ws_o [24, bm, dim·NK_max]`（fp16 f16 / bf16 f32）：Cube→Vector 的每块 PV 部分和（v11 D4 探针：f16 传输 tier-1 0 flips、lse 8.7e-5——采纳）。

  **Cube 流**（`T.Scope("Cube")` 内，persistent 任务循环 → 任务内两相位；`kid`=kernel_id）：

```
任务头: [task_id > 0: 等 FLAG_TASKDONE]                                    # 单槽跨任务 WAR 保护
        Q-hoist: T.copy(q[bz, s_lo:s_lo+tmc, by, 0:dim], l1_a[0:tmc, 0:dim])  # 一次装载（v11nt D5：免参考实现每 n-block 重读）
pass-1（全 S，无等待直跑）: for i in T.serial(NK):
    n_lo = i·bn; tn_real = min(bn, S_kv − n_lo); tnc = max(32, min(bn, ceil16(tn_real)))   # 分形下限钳位（§6.4）
    T.copy(k[bz, n_lo:n_lo+tnc, kv_head, 0:dim], l1_b[0:tnc, 0:dim])                        # slice-form（D1 约束）
    T.gemm(l1_a, l1_b, l0_c, initC=True, b_transpose=True, size=[tmc, dim, tnc])
    with T.rs("PIPE_FIX"): T.copy(l0_c[0:tmc, 0:tnc], ws_s[kid, 0:tmc, n_lo:n_lo+tnc]); T.sync_block_set(i)
pass-2（全 PV）: for i in T.serial(NK):
    with T.rs("PIPE_MTE2"): T.sync_block_wait(i); T.copy(ws_p[kid, 0:tmc, n_lo:n_lo+tnc], l1_a[0:tmc, 0:tnc])
    T.copy(v[bz, n_lo:n_lo+tnc, kv_head, 0:dim], l1_b[0:tnc, 0:dim])
    T.gemm(l1_a, l1_b, l0_c, initC=True, size=[tmc, tnc, dim])                              # 每块独立部分和（覆写）
    with T.rs("PIPE_FIX"): T.copy(l0_c[0:tmc, 0:dim], ws_o[kid, 0:tmc, i·dim:(i+1)·dim]); T.sync_block_set(i)
```

  **Vector 流**（`T.Scope("Vector")` 内，persistent 任务循环 → 两相位 + epilogue，per-AIV 行半分；`rm`/`bx_r` 为本 AIV 的 real_m 与全局行基）：

```
任务头: vbrc 初始化（0 → logsum/acc_o/ub_alpha/scales；−inf → scores_max；let-bound 标量，D4）
pass-1（softmax 链）: for i in T.serial(NK):
    T.copy(ub_m, ub_mprev)
    with T.rs("PIPE_MTE2"): T.sync_block_wait(i); T.copy(ws_s[kid, bx_r:bx_r+rm, n_lo:n_lo+tn_v], ub_f16_N[0:rm, 0:tn_v])
                            [bf16 trace: ws_s 为 f32，直拷 ub_f32_N，跳过 vcast]
    [fp16 trace: T.vcast(ub_f16_N, ub_f32_N, "rint")]
    [softcap: vtanh 链（fp32 scratch，D5）]
    [causal 且 i ≥ K_A: mask 链（E3——vsub/vcmp/vselect；尾带变体 + vand）]
    [prescale ≠ 1: T.vmul(ub_f32_N, prescale, ub_f32_N)]
    T.reduce(ub_f32_N, ub_mcur, dims=[1], "max", clear=True)
    if i != 0: T.vmax(ub_mprev, ub_mcur, ub_m); T.vmax(ub_m, NEG_CLAMP, ub_m)     # 单调 max + 源 NaN guard（M6）
               T.vsub(ub_mprev, ub_m, ub_t); T.vexp(ub_t, ub_alpha)               # α_i = e^{m_prev − m}
               T.copy(ub_alpha, scales[i·half : i·half+half, 0:1])
    else:      T.copy(ub_mcur, ub_m, ub_m)                                        # 首块直接采纳（α_0 值无关：logsum/acc_o 为 0）
    T.vsub(ub_f32_N, ub_m, ub_f32_N); T.vexp(ub_f32_N, ub_f32_N)                  # P = e^{S'' − m}（行广播）
    T.reduce(ub_f32_N, ub_ellcur, dims=[1], "sum", clear=True)                    # clear=True 强制（T.reduce.md §2.3）
    [fp16 trace: T.vcast(ub_f32_N, ub_f16_N, "rint")]                             # P 量化到输入 dtype（M10）
    with T.rs("PIPE_MTE3"): T.copy(ub_f16_N[0:rm, 0:tn_v], ws_p[kid, bx_r:bx_r+rm, n_lo:n_lo+tn_v]); T.sync_block_set(i)
    T.vmul(ub_ell, ub_alpha, ub_ell); T.vadd(ub_ell, ub_ellcur, ub_ell)
pass-2（延迟 rescale 累加）: for i in T.serial(NK):
    with T.rs("PIPE_MTE2"): T.sync_block_wait(i); T.copy(ws_o[kid, bx_r:bx_r+rm, i·dim:(i+1)·dim], ub_f16_D[0:rm, 0:dim])
                            [fp16: T.vcast(ub_f16_D, ub_f32_D, "rint")]
    if i != 0: T.copy(scales[i·half : i·half+half, 0:1], ub_alpha)
    T.vmul(acc_o, ub_alpha, acc_o); T.vadd(acc_o, ub_f32_D, acc_o)
epilogue: T.vdiv(acc_o, ub_ell, acc_o); T.vcast(acc_o, ub_f16_D, "rint")
    T.copy(ub_f16_D[0:rm, 0:dim], output[bz, bx_r:bx_r+rm, by, 0:dim])            # 行截断写出（E7）
    T.vlog2(ub_ell, ub_lse, ub_lse_tmp); T.vmul(ub_m, LOG2E, ub_t); T.vadd(ub_lse, ub_t, ub_lse)
    T.copy(ub_lse[0:rm, 0:1], lse[bz, by, bx_r:bx_r+rm, 0])                       # [B,H,S,1] 视图（E6，零 transpose）
    T.sync_block_set(FLAG_TASKDONE)
```

  **flag 协议**（id 分配 §7.2）：n-block 下标 id `i ∈ [0, NK_max)`——每任务内每 id 严格三次握手（S-ready Cube→Vec → P-ready Vec→Cube → O-ready Cube→Vec），**per-id 严格交替**（每个 id 任一时刻至多一个未决事件，对任意 flag 语义安全——v11 同款）；`FLAG_TASKDONE = NK_max`——Vector（双 AIV 各 set 同一 id，聚合语义依 E1-E7 + v11nt 双生产先例，R-1 风险项带探针/回退）在 epilogue 后 set，Cube 在下一任务 Q-hoist 前 wait（单槽跨任务 WAR：防止 pass-1 覆写未消费的上一任务 ws）。总 id 数 NK_max + 1 ≤ 16（**bn 下限公式强制**：`bn_eff ≥ ceil16(ceildiv(S_kv, 15))`，§5.2；T.set_flag.md §2.1 event_id ∈ 0–15）。
  **稳态重叠**：Cube pass-1 无任何 wait（MTE2 装载 k / MTE1 喂 L0 / C gemm / FIX 写 ws_s 四通道天然重叠，跑在 Vector 前面）；pass-2 的 P-wait 到达时其依赖（Vector pass-1 的 P(i)）在稳态已就绪或即将就绪，等待不阻塞有用发射（pass-1 已排空）。Vector pass-1 连续消费 S 块（无 O 依赖插入），pass-2 连续消费 O 块。**这正是单遍结构「wait 阻塞发射流」问题的结构解**——实测依据：v11nt fa4096 每核 cube 管道 busy 171.1→65.7µs（2.60× 削减）、L1 r+w 流量 18.70→11.17MB（-40%）（PL-1.9-twophase）。
- **语义保持论证**：① 数据流等价——S 块内容 = 源 S6 的 QK^T（fp32 L0C 累加；f16 物化在容差内，M12 论证）；P 块 = 源 S9 的 acc_s_cast（fp32→输入 dtype rint 量化点一致，M10）；O 部分和 = 源 S11 的单块 PV（initC=True 覆写 = 每块独立部分和；f16 物化容差论证 M11）；Vector pass-2 的 `o = o·α_i + O_partial(i)` 与源 `acc_o = acc_o·α + P@V` **代数同构**（α 乘法从 gemm 间移到 Vector，两侧均 fp32 乘加；仅 L0C 内积累加分组与源 fragment 累加分组不同——fp32 ulp 级，被 §8.2 容差覆盖）；② 归约语义——块间顺序累加结构保留（两流均按块序推进，flag 强保序）；ℓ 与 o 的更新次序与源 macro ⑨/S10/S11 一致——pass-1 逐块更新 ℓ 与 m（α_i 当场作用于 logsum、存档供 pass-2 回放），pass-2 以同一 α_i 序列作用于 acc_o，两序列恒等（同一 scales[] 源）；③ 边界语义——mask/尾块见 E3/E7；NaN guard 见 E4；④ 槽位安全——单槽下 Cube 下一任务 pass-1 与 Vector 上一任务 pass-2 的 WAR 由 FLAG_TASKDONE 握手消除（Cube wait 在 Q-hoist 前，先于任何 ws 写）；n-block id 的跨任务复用安全：Cube 到达任务 t+1 的 set_S(i) 前，程序序上必先经过 pass-2(t) 的 wait_P(i)（同 id 前一生命周期的消费点），且 TASKDONE(t) wait 已保证 Vector 对任务 t 的全部 wait 完成——所有 id 事件在跨任务边界前排空。

**重设计项 E3: mask 预填 + initC 累加 GEMM → Vector pass-1 向量化 mask + 两段式免 mask（E1-E7 E3 承接 + 位置平移）**

- **源方案**：`T.Parallel(bm, bn)` 逐元素谓词写 `acc_s[i,j] = if_then_else(valid, 0, −inf)` 后 `T.gemm(initC=False)` 累加。E1-E7 迁移实测该形态为 aiv_scalar 88–89% 第一瓶颈（opt_log §1）。
- **NPU 新算法**：mask 移至 **Vector pass-1、S vcast 到 f32 之后 softmax 之前**，全向量构造（NaN 无关），E1-E7 E3 全套机制承接、位置从单遍 V1 平移到两相位 pass-1：
  1. **每核一次**的索引矩阵（任务循环外构建）：`T.arange(ub_colmat, strides=[0,1], offset=0)`（[half, bn_eff] 列索引 j）与 `T.arange(ub_rowmat, strides=[1,0], offset=0)`（AIV 本地行索引 i）（docs/Tilelang.language/创建操作/T.arange.md：value = Σ idx·stride + offset）；`T.vbrc(−inf, ub_neg)`（f32 −inf 哨兵阵，docs/Tilelang.language/shape操作/T.vbrc.md；let-bound 标量，D4）。
  2. **每 masked 块**：`ub_diff = T.vsub(ub_colmat, ub_rowmat)`（j − i，int16——**vcmp 标量操作数拒绝 tir.Cast**（debug_log D4），索引矩阵 int16 + 整数 PrimExpr 阈值，highperf ub_arrange_mask 同款）；causal 条件 `j − i ≤ K_blk`：`T.vcmp(ub_diff, K_blk, ub_cond, "le")`（docs/Tilelang.language/比较操作/T.vcmp.md）；**尾带变体**（trace-time 门控，§6.4）另加 kv OOB 条件 `j < tn_real`：`T.vcmp(ub_colmat, J_lim, ub_cond2, "lt")` + `T.vand(ub_cond, ub_cond2, ub_cond)`（docs/Tilelang.language/逻辑操作/T.vand.md）；应用 `T.vselect(ub_cond, ub_f32_N, ub_neg, ub_f32_N)`（docs/Tilelang.language/条件操作/T.vselect.md）。**阈值推导（per-AIV，双 lane 端点验证纪律——RETROSPECTIVE P 类教训承接）**：源 causal 条件 `q_pos + causal_offset ≥ kv_pos`，代入 `q_pos = s_lo + row0 + i`（row0 = subid·real_m0，v11 自适应半分）与 `kv_pos = n_lo + j` 得 `j − i ≤ (s_lo + causal_offset − n_lo) + row0 ≡ K_blk`。数值验证例（smoke (1,512,8,64) causal，bx=4 → s_lo=256，对角块 i=4 → n_lo=256，tail_m=64 → real_m0=32）：subid=1 端点（row0=32）：K_blk = 256 + 0 − 256 + 32 = 32 → `j ≤ 32 + i`；本地行 i=0（全局行 288）允许 kv_pos ≤ 288 即 j ≤ 32 ✓。subid=0 端点（row0=0）：K_blk = 0 → `j ≤ i`；本地行 i=0（全局行 256）允许 kv_pos ≤ 256 即 j ≤ 0 ✓。
  3. **两段式分界**：`K_A = min(floordiv(s_lo + causal_offset + 1, bn_eff), seq_len_kv // bn_eff)`（causal；**floordiv 取整方向——E1-E7 v2 修正公式逐字承接**：块 n 全有效要求 `s_lo + causal_offset ≥ (n+1)·bn − 1` ⟺ `(n+1)·bn ≤ s_lo + causal_offset + 1` ⟺ `n + 1 ≤ floordiv(s_lo + causal_offset + 1, bn)`；`a = s_lo + causal_offset + 1 ≥ 1` 恒正无负数除法歧义；`T.floordiv` 或 `//` 同为 floordiv）——块 `i < K_A` 全有效**零 mask 开销**；仅 `i ≥ K_A`（对角块，S_q=S_kv causal 下每任务 NK − K_A = 1 块）走 mask 链。运行时 `if i >= K_A:`（Expert 模式合法：highperf 运行时 if 先例 + E1-E7 生产验证）。对 subid=1（行域更高、因果窗更宽），按 subid=0 最劣行推导的 K_A 是保守下界（`K_A′ = floordiv(s_lo + real_m0 + causal_offset + 1, bn) ≥ K_A` 恒立——分子更大、floordiv 单调不减）——subid=1 在 `[K_A, K_A′)` 内的块实际全有效但被计入 mask 链：mask 条件全真、vselect 保留原值，输出不变——**两 AIV 共用 K_A 仅损失性能（≤1 块的 5 个向量 op），不损正确性**。非 causal：无 mask 链（trace 剔除 mask 缓冲——VP-P2 工厂级变体分派）。
  4. Cube 侧 gemm1 **恒 initC=True**（无 mask 预填，S 每块覆写）。
  5. 顺序：vcast → **softcap（若有）→ mask → prescale**——softcap 先作用于 raw S、mask 的 vselect 再把 masked 位覆写为 −inf（源 S7 在 mask 之后作用于已含 −inf 的 acc_s 并保持 −inf；两序 valid 位同为 `softcap·tanh(·)`、masked 位两序同为 softmax 不可见〔源 −inf → e^{−inf} = 0；本设计 −inf → e^{−inf − m}，m 经 M6 clamp 恒 ≥ −1e38 有限 → 精确 0〕，**两区域分别一致**）；prescale 在 mask 后（源语义序：mask 先、scale 后；sm_scale=0 病态路径 `−inf·0 = NaN` 与源逐位一致）。
- **语义保持论证**：源把 mask 加在 QK 之后、softmax 之前（S5 预填 + S6 累加 = mask + QK）；本设计把 mask 选择在 S 到达 Vector 后、softmax 前应用——**作用点相同**。条件等价（per-AIV 阈值 + 双 lane 端点数值例，见第 2 步）；vselect 按 cond 选择**与 S 值无关**——垃圾 q 行/k 尾带产生的 NaN/Inf 分数被无条件覆写为 −inf（比源 `if_then_else(acc_s == −inf, ...)` 的等值判定更稳健，覆盖 NaN 输入）。全有效块跳过 mask ≡ 源 mask 条件在全块恒真的情形（K_A 定义即「块尾 kv 位置 ≤ 块首行 causal 边界 且 块完整」的全称量化——对 subid=1 为保守下界），输出逐位一致。

**重设计项 E4: exp2+LOG2E 折叠与 NaN guard → e 域 vexp 链（换底方式等价替换）**

- **源方案**：`scale = score_scale·LOG2E`（无 softcap）或 `LOG2E`（有 softcap），`exp2(x·scale − m·scale)`——GPU 编译器把乘法融进 ffma。
- **NPU 新算法**：e 域直接实现（`T.vexp` e 底，docs/Tilelang.language/数学操作/T.vexp.md，v11nt L1392–L1395 同构）：① prescale 预乘 `T.vmul(ub_f32_N, prescale, ub_f32_N)`（标量广播；softcap 分支 prescale=1.0 时 trace-time 跳过，M8）；② 行 max `T.reduce(..., "max", clear=True)` → 单调 `T.vmax(ub_mprev, ub_mcur, ub_m)` → clamp `T.vmax(ub_m, −1e38, ub_m)`（源 NaN guard 保留：m ≥ −1e38 恒成立使 masked 位的 `e^{−inf − m}` 精确 0、无 −inf−(−inf) 的 NaN 路径）；③ α = `vexp(vsub(m_prev, m_new))`；④ P = `vexp(vsub(S, m_new))`；⑤ `T.reduce(..., "sum", clear=True)`；⑥ ℓ/o 更新（E2）；⑦ lse = `T.vlog2(ub_ell, ub_lse, ub_lse_tmp) + ub_m·LOG2E`（三参签名，docs/Tilelang.language/数学操作/T.vlog2.md §1）。
- **语义保持论证**：**数学恒等**——源 `exp2(u·log2e) = e^u` 逐点相等；lse：源 `log2(ℓ) + m·s·LOG2E = log2(ℓ·e^{s·m}) = log2(Σe^{s·x})`，本设计 `log2(ℓ) + LOG2E·m′`（m′ = 预乘域 max）为同一表达式（谱系 23/23 + v11nt fa-tuned 实证）。数值路径更短（少两次 [half,bn] 乘法舍入），差异 fp32 ulp 级 « §8.2 容差。

**重设计项 E5: bf16 输入路径 → Cube bf16 直连 + f32 S/O 物化（dtype 承载）**

- **源方案**：q/k/v 以 bf16 直接进 bf16 tensor core（fp32 累加），`-DENABLE_BF16` 使能。
- **NPU 新算法**：bf16 trace 下 q/k/v **直接** slice-form `T.copy` bf16 进 L1；`T.gemm` bf16×bf16→fp32（L0C f32，T.gemm.md §2.3）；P 量化 `T.vcast(f32 → bf16, "rint")`（量化点与源逐位同 dtype）后经 ws_p（bf16）回传，GEMM2 bf16×bf16→fp32；输出 `T.vcast(f32 → bf16, "rint")`。**ws 承载与 fp16 路径不同**：`ws_s = f32`（S 原始值域 |S| ≲ 60（randn 输入 √dim 尺度），bf16 尾数 8 位 → S 相对误差 ~2^-8·|S|，经 exp 传播 ΔP/P ~2% 威胁 §8.2 冻结容差；f16 路径 2^-11·|S|·scale ≈ 0.26% 已实证通过——bf16 升 f32 为保守承载，Vector 侧跳过 vcast 直拷 f32）、`ws_p = bf16`（P 量化到输入 dtype 是源语义）、`ws_o = f32`（部分和保守承载）。fp16 与 bf16 路径**结构逐语句同构，仅 dtype 字符串与 ws dtype 随 trace 切换**。
- **语义保持论证**：① 输入精度与源一致（bf16 直进 Cube fp32 累加，无中间量化）；② P 量化点与源**逐位同 dtype**（源 `acc_s_cast` cast 到输入 dtype；本设计 vcast f32→bf16 rint）；③ 输出 cast 同为 f32→bf16 rint；④ NaN 传播一致（vcast 保 NaN，slice copy 逐位搬运）；⑤ GEMM 累加分组差异 fp32 ulp 级（E2 同款论证）。
- **风险与回退（R-2）**：两相位 bf16 端到端组合（slice copy bf16 → L1 + 两相位 gemm + bf16 ws_p 往返）**无先例**（v11 为 fp16 专用；PL-1.8 的 bf16 直连证据来自 E1-E7 单遍结构）——标注未实证假设；Stage 3 以 L0-2（bf16 smoke）**前置首验**；失败回退 = ws 族全 f32 + bf16 输入在装载点 vcast 链（E1-E7 载体思路的装载点局部化，结构不变）。

**重设计项 E6: 接口契约适配（workspace 参数 + 闭包 wrapper + lse 视图增维 + config 语义）**

- **源方案**：`@tilelang.jit(out_idx=[3, 4], pass_configs={FAST_MATH}, compile_flags)`（CUDA target）内层工厂签名 `(block_m, block_m, num_stages, threads)`。
- **NPU 新方案**：
  - 工厂外层签名**完全保留**（含 `@functools.lru_cache(maxsize=32)`、参数校验、常量折叠）；
  - 内层 callable 签名 `(block_m, block_n, num_stages)`（threads 移除，K3/K9）——返回值为 **Python 闭包**：闭包内按 workspace shape 以 `torch.empty(..., device=q.device)` 分配 ws 族并调用编译好的 kernel，出口 `lse_t.reshape(batch, heads, seq_len_q)` 还原契约形状（零拷贝视图）。先例：v11nt `wrapped()`（TileOPs 侧 perf_opt/_gqa_prefill_fwd_kernel.py L1457–L1471）与 highperf `sparse_attn()` 接口函数。wrapper 的 `(block_m, block_n, num_stages)(q, k, v) → (output, lse)` 调用形态**逐字保持**（custom_op 与 Kernel class 无需任何改动）。
  - prim_func 参数序：`(q, k, v, ws_s, ws_p, ws_o, output, lse)`（8 参），`@tilelang.jit(out_idx=[-2, -1], target="npuir", pass_configs={TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False, NPUIR_ENABLE_AUTO_MULTI_BUFFER: False})`——输出居末两位（与源 out_idx=[3,4] 在 5 参中居末两位的约定同构；`testing/npuir/compiler_hint_ops/test_out_idx.py` L75–107 `out_idx=[-2,-1]` 先例）。
  - **lse 以 `[B, H, S, 1]` 四维视图声明**（同一块连续内存），epilogue 用自然 2D 区域拷贝 `T.copy(ub_lse[0:rm, 0:1] → lse[bz, by, bx_r:bx_r+rm, 0])`——**零 transpose**。依据：**活跃数据源的 `T.transpose([N,1]→[1,N])` epilogue 毒化整 kernel 2.6×**（VP-D6，pattern-library §2：14 点 morph 实测矩阵 + v9 -20% 双证；v11nt 修复即本形态，-61% vs v11 初版）；闭包出口 reshape 还原 [B,H,S]（零拷贝）。
  - **config 语义（有效 config 计算，trace-time）**：`bn_eff = max(bn_caller, ceil16(ceildiv(S_kv, 15)))`（**flag 预算硬守卫**：n-block id 数 NK_max = ceildiv(S_kv, bn_eff) ≤ 15，加 FLAG_TASKDONE 共 ≤ 16 = T.set_flag.md §2.1 event_id 上限）；当 caller 传 **wrapper 默认** `(64, 64, 1)`（或 dim>128 时的 `(64, 32, 1)`）时替换为**设计默认 `(bm=64, bn=256, slots=1)`** 后再套用上述守卫（S4-5 工厂内分派模式，PL-1.9「caller 传任何显式非默认 config 一律尊重原值」纪律）；`num_stages` 在 Stage 3 基线映射为**保留旋钮**（两相位单槽结构下不改变行为；双槽 id-offset 变体 = Stage 4 实验候选，§9.2 R-7）。目标域核验：S_kv=2048 → ceildiv(2048,15)=137 → ceil16=144 ≤ 256 ✓ NK_max=8，ids 0..8 共 9 个；S_kv=512 → ceildiv(512,15)=35 → ceil16=48 ≤ 256 ✓ NK_max=2。
- **语义保持论证**：接口层适配不触碰数值语义；workspace 为 kernel 私有暂存（每调用新分配，内容全由生产者写满后消费者才读——flag 协议保证），不改变 I/O 契约；lse 视图增维 + host reshape 为同一内存的形状视图（[B,H,S,1] 与 [B,H,S] 逐元素一一对应），数值与布局零变化；闭包返回 `(output, lse)` 的 shape/dtype 与 `register_fake` 声明一致。

**重设计项 E7: 对齐/非对齐双路径 + 谓词标量 GM 写 → size-form 尾钳位统一路径 + 分形下限钳位 + trace-time 尾带掩码 + 行截断写出**

- **源方案**：对齐块整块 `T.copy`，非对齐块谓词逐元素装载/回写（OOB 填 0 / 只写真实行列）。
- **NPU 新算法**：统一单路径（flash_attn_npuir.py L88–118 全套先例 + 分形下限扩展）：
  - **尺寸定义（每块，PrimExpr）**：`tail_m_real = min(bm, S_q − s_lo)`（真实行数）；`tmc = max(16, min(bm, ceil16(tail_m_real)))`（**gemm M/N 维下限钳位 16**——migration-analysis.md §5.3 分形限制 M≥16/N≥16；v11 的 fa 域从未触发 tail<16（S%96 ∈ {0,32,64}），本设计显式覆盖）；`tn_real = min(bn_eff, S_kv − n_lo)`（真实列数）；`tnc = max(32, min(bn_eff, ceil16(tn_real)))`（**gemm K 维下限钳位 32**——分形限制 K≥32；16 与 32 两档分别对应 M/N 与 K 的最小值）。Vector 侧的 S 读宽 `tn_v`：无带 trace 恒 `= tn_real`（只读真实列），has_band trace `= tnc`（含尾带，见下条）；`has_band = (tnc > tn_real)`（**尾带存在性**：尾块非 16 倍数或 < 32 时成立；对最后一块是工厂期常量 → trace-time 门控）。
  - **q 装载（任务内一次，Q-hoist）**：`T.copy(q[bz, s_lo:s_lo+tmc, by, 0:dim], l1_a[0:tmc, 0:dim])`——尾行 [tail_m_real, tmc) 为残留旧值 → 垃圾分数行 → **行截断写出隔离**（Vector 自适应半分只覆盖 tail_m_real 真实行，垃圾行所有下游数据不被读/写）；
  - **k/v 装载（每块）**：`T.copy(k[bz, n_lo:n_lo+tnc, kv_head, 0:dim], l1_b[0:tnc, 0:dim])`——尾行 [tn_real, tnc) 残留 → 仅进入 gemm 的 K 带（见下）；
  - **gemm1**：`size=[tmc, dim, tnc]`（M=tmc ≥ 16 ✓，K=dim ≥ 32 ✓（dim∈{64,128}），N=tnc ≥ 32 ≥ 16 ✓）；S 物化列宽 tnc——**尾带列 [tn_real, tnc) 为 ws_s 残留（上一任务同位置数据）**；
  - **尾带清零（has_band trace 的 Vector pass-1）**：Vector 处理列宽 **tnc**（`tn_v = tnc`，读 `ws_s[..., n_lo:n_lo+tnc]`——含尾带；无带 trace 恒 `tn_v = tn_real` 只读真实列），对 `j ≥ tn_real` 的列以 OOB 掩码条件强制 P=0：`ub_cond2 = T.vcmp(ub_colmat, tn_real, "lt")` + `T.vand(ub_cond, ub_cond2, ub_cond)`（vselect −inf → P = e^{−inf−m} = 0）——**残留 S 值无论为何（含 NaN/Inf），vselect 值无关选择保证 P 带 = 0**；gemm2 的 K 带中 `P带(=0) × V残留(任意) = 0`，污染精确为零。causal 与非 causal 的 has_band trace 同款（causal 变体为 vand(causal_cond, oob_cond)，非 causal 变体仅 oob_cond）。
  - **输出写出**：`T.copy(ub_f16_D[0:rm, 0:dim], output[bz, bx_r:bx_r+rm, by, 0:dim])`（行截断，`rm` 按真实行自适应半分；D3 教训：src 切片 `[0:rm, 0:dim]` 与 dst 同形）；lse：`T.copy(ub_lse[0:rm, 0:1], lse[bz, by, bx_r:bx_r+rm, 0])`（[B,H,S,1] 视图，E6）。
  - **ws_o 的尾块列**：`T.copy(l0_c[0:tmc, 0:dim], ws_o[kid, 0:tmc, i·dim:(i+1)·dim])`——O 部分和列宽恒 dim（无尾带）；P 带 = 0 使 O 部分和不受残留 V 行影响 ✓。
- **语义保持论证**：① size-form gemm 与尾钳位切片 copy 的元素映射在真实行列上与源谓词装载一一对应（src/dst 同形无重排）；② OOB 行为等价：源 OOB 填 0 → 分数 0 → mask −inf → exp 0；本设计 OOB 残留 → 任意值 → OOB 掩码 −inf → exp 精确 0（**尾带掩码使 OOB 分数值无关紧要**）——q 尾垃圾行不进入 Vector 处理域（自适应半分只覆盖真实行）；③ 输出域：行截断写出只写 `q_pos < seq_len_q` 行 ≡ 源 predicated store 行集；④ 无 per-scalar GM 写（规避 MTE aicore 异常，E1-E7 谱系结论）；⑤ 全 OOB kv 块（`n_lo ≥ S_kv`）不在 NK 循环内（E1 截断论证）。**与 E1-E7 E7 的差异**：单遍结构的 v-guard（nk_total ≤ slots 的零填充守卫）在两相位下**结构性消除**——V 经由尾钳位切片 copy 装载（不再有 load_nd2nz 的 L1 残留问题），P 带零贡献由掩码保证（不再依赖 ws_v 零填充缓冲）。

**接口契约适配清单**（`_gqa_prefill_fwd_kernel` 是对外接口，逐项说明）：

| 接口项 | 迁移前（GPU） | 迁移后（NPU Expert 两相位） | 理由 |
|--------|--------------|---------------------|------|
| 工厂签名 `_gqa_prefill_fwd_kernel(batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, is_causal, sm_scale=None, softcap=0.0, dtype='float16')` | 同左 | **完全保留**（含 `@functools.lru_cache(maxsize=32)`） | 对外接口不变（迁移硬性规则）；wrapper 以位置参数调用 |
| 内层 callable 签名 | `(block_m, block_n, num_stages, threads)` | **`(block_m, block_n, num_stages)`**（移除 threads） | K3/K9；wrapper 已按此签名调用（L100–L102 亲自核对） |
| 内层 callable 返回值 | JITKernel（`f(q,k,v)→(output,lse)`） | **Python 闭包**（内持 JITKernel + workspace 分配 + lse reshape；`f(q,k,v)→(output,lse)` 调用形态不变） | workspace 显式参数需要；v11nt `wrapped()` 先例；wrapper 零改动 |
| prim_func 参数 | `(q, k, v, output, lse)` | `(q, k, v, ws_s, ws_p, ws_o, output, lse)`——q/k/v 前三、output/lse 末二不变 | E2/E6 |
| `@tilelang.jit` 装饰 | `out_idx=[3,4]` + FAST_MATH + compile_flags | `out_idx=[-2, -1], target="npuir"` + 两个 Expert pass_configs | 输出居末约定同构；CUDA pass 不迁移；test_out_idx.py 先例 |
| `T.Kernel(ceildiv(S,bm), heads, batch, threads=)` | 三维 + threads | `T.Kernel(24, is_npu=True) as (kernel_id, subid)` 一维 persistent | E1；T.Kernel 一维约束 |
| lse 形状声明 | `[batch, heads, seq_len_q]` 3D | `[batch, heads, seq_len_q, 1]` 4D 视图（闭包出口 reshape 还原 3D） | E6 transpose-free（VP-D6） |
| gemm 调用 | `transpose_B=True, policy=FullRow` | `T.gemm(l1_a, l1_b, l0_c, initC=True, b_transpose=True, size=[M,K,N])`（Scope("Cube") 内） | T.gemm.md Expert 签名；FullRow 舍弃（§0.5） |
| macro 工厂（3 个 `@T.macro`） | `@T.macro` | **语义体内联为两相位流水段**（pass-1/pass-2 前缀链；可保留 `@T.macro` 组织代码——highperf 即用宏定义流水段） | 计算体 = Expert 两相位结构 |
| `T.Tensor` 4D 声明 | 关键字 `dtype=` | **位置参数写法** `T.Tensor(shape, dtype)` | design-template §3.3 规则（false-alarm 规避） |
| 环境变量 | 无 | `TILELANG_ASCEND_MODE=Expert`（setdefault） | docs/developer/EnvironmentVariables.md L31（默认即 Expert，显式防残留） |

### 0.7 标杆实现

- **源算子路径（golden 语义来源）**：`/home/tilelang/l00970450/TileOPs/tileops/kernels/attention/gqa_fwd.py`（L788–L1033）+ `online_softmax.py`；本仓提取件 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/_multi_head_attention_fwd_kernels.py`。
- **源仓参考实现 / 测试基准**：`examples/TileOPs/tests/ops/test_multi_head_attention.py` 的 `MhaFwdTest.ref_program`（SDPA is_causal + BSHD↔BHSD 转置，`atol=5e-3, rtol=1e-5`，不检查 lse）——§8.1 golden 优先移植此实现并补 lse（D7 校准）。
- **NPU 侧同类标杆**：
  - `examples/flash_attention/flash_attn_npuir.py`——**两相位 S/P 物化结构原型（本轮选定结构的直接来源）**：双 Scope 两相位大循环、per-n-block flag、单 L0C 顺序复用、尾钳位 size-form、scales 延迟回放、自适应行半分；同机实测 12.91/18.20/32.41/99.09µs @ S=512/1024/2048/4096；
  - TileOPs 侧 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py`（v11nt 终版，SHA aacc606f）——两相位 + 本算子契约（BSHD/GQA/lse/softcap）+ Q-hoist + ell clear=True 修正 + transpose-free lse 的 tuned 实现（fa 域 12.94/18.11/30.88/98.05µs）；**其 `_builder_2phase` 为本设计 §3.3 伪代码的直接母本（causal 适配为本轮新增工作）**；
  - `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py`——Expert persistent 双 Scope、per-slot flag、vid 半分、`T.arange`+`T.vcmp` 向量 mask、运行时 if 先例（E1/E3 依据）；
  - `examples/deepseek_v32/sparse_mla_fwd_exp.py`——online softmax + lse 的 Expert 语义先例；
  - `examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v2.md`——E1-E7 单遍设计与其检视修正（K_blk/K_A 公式、E7 守卫判据、bf16 直连裁决三步法——正确性论证直接承接）；
  - `examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md`——工具链实测事实 D1–D6（slice-form copy / copy 区域语义 / parser 限制 / int16 阈值 / vtanh fp32 / tier-3 分类）；
  - `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/`（opt_log.md 三轮 + perf_records.jsonl + perf_feedback.md + profiles/）——fa 域全部实测证据与本轮目标域基线（causal 回归 3764.05/13265.27µs）。

---

## 1. 概述

### 1.1 算子名称

`_gqa_prefill_fwd_kernel`（GQA/MHA prefill 前向，MultiHeadAttentionFwdOp 的 TileLang kernel；NPU 侧 Expert 模式两相位重实现，目标域 = manifest causal 多头 workload）

### 1.2 功能描述

BSHD 布局 q/k/v 上的 flash-attention 风格 online-softmax 前向：输出 `o = softmax(scale·q@k^T + causal/oob mask) @ v` 与每行 log2 域 log-sum-exp `lse`；支持 GQA 分组（`heads % heads_kv == 0`）、右对齐 causal、softcap、S_q ≠ S_kv、fp16/bf16。

### 1.3 数学公式

同 §0.1 四式（S / softcap / mask+o / lse）。

### 1.4 算法描述（迁移决策后的 NPU 侧 Expert 算法）

**选定算法族**（§1.6.0 调研结论）：**两相位 S/P 物化 + per-n-block online softmax**（flash_attn_npuir.py 结构族；源算法的单遍 per-block 链形态在本轮调研中被实测证据淘汰为目标域的承载结构，但 online-softmax 数学递推完整保留）。**NPU Expert 形态**（§0.6 E1–E7 落地）：

1. **任务划分**：一维 persistent 24 核；逻辑任务 = (q-block × head × batch)；核内 `T.serial` 串行处理 `ceildiv(num_logical − kernel_id, 24)` 个任务（E1）。
2. **双引擎两相位分工**：Cube 段承担全部 GEMM——pass-1 串行算完全部 S 块（QK^T，Q-hoist，无等待直跑），pass-2 串行算完全部 PV 部分和（initC=True 逐块覆写）；Vector 段 pass-1 逐块 softmax 链（vcast → softcap → mask → prescale → online max/α/ℓ 递推 → P 量化）+ rescale 因子存 UB `scales[]`，pass-2 延迟 rescale 累加 acc_o，epilogue 归一化 + transpose-free lse（E2，双 AIV 自适应行半分）。
3. **跨引擎数据环**：`l0c_s →(T.copy FIX)→ ws_s →(MTE2)→ UB →(v 链)→ ws_p →(MTE2)→ l1_p →(gemm2)→ l0c_o →(FIX)→ ws_o →(MTE2)→ UB 累加`；每核单槽 ws 三族 + per-n-block 下标 flag（严格 per-id 交替三次握手）+ FLAG_TASKDONE 任务边界（E2）。
4. **mask**：Vector pass-1 `arange/vsub/vcmp/vand/vselect` 向量构造（f32 −inf 哨兵 + 值无关选择），两段式 K_A（floordiv）分界全有效块零开销（E3；per-AIV 阈值 `K_blk = s_lo + causal_offset − n_lo + row0`，双 lane 端点验证）。
5. **尾块**：size-form gemm + 尾钳位切片 copy + 分形下限钳位（M 16 / K 32）+ trace-time 尾带 OOB 掩码 + 行截断写出（E7）。
6. **dtype 路径**：fp16/bf16 **同构直连**——q/k/v slice-form copy 进 L1，`T.gemm` fp16×fp16 / bf16×bf16 → fp32 累加；P 量化 vcast 到输入 dtype（M10）；ws 承载 fp16: f16/f16/f16（v11nt 实证）、bf16: f32/bf16/f32（保守，R-2 探针 + 回退）。

与源算法的结构差异（来源标注）：① 承载结构从单遍 per-block 链换为两相位独立大循环（§1.6.0 调研结论 + §0.6 E2）；② acc_o 从 fragment 驻留改为 Vector UB 驻留 + 每块 PV 部分和 GM 物化 + scales 延迟回放（E2）；③ mask 从 Cube gemm 累加预填移至 Vector 向量选择（E3）；④ exp2 域改 e 域（E4，数学恒等）；⑤ 三维 grid 改一维 persistent（E1）；⑥ 尾块双路径改 size-form 统一路径（E7）。算法层（online softmax 递推、分块结构、causal 裁剪、P 量化点、lse 域、NaN guard）**逐项保留**。

### 1.5 数据流图

```
GM[q] ──T.copy(slice, Q-hoist)──> L1[l1_a [bm, block_share]] ──────────────────────────┐
GM[k] ──T.copy(slice)──> L1[l1_b [bn, dim]]（pass-1 K / pass-2 V 生命周期复用）          │
   L1[l1_a]×L1[l1_b] ──T.gemm(b_transpose, initC=True)──> L0C[l0_c] ──T.copy(FIX, f32→f16)──> GM[ws_s]
                                                                        │ set flag i
GM[ws_s] ──T.copy(MTE2)──> UB[ub_f16_N] ──vcast──> UB[ub_f32_N] ──[softcap]─[mask vselect]─[prescale]─
   ──reduce(max)─vmax×2─vsub/vexp(α→scales[])─vsub/vexp(P)─reduce(sum)─vcast──> UB[ub_f16_N]
UB[ub_f16_N] ──T.copy(MTE3)──> GM[ws_p] ──T.copy(MTE2)──> L1[l1_a] ─┐ set flag i
   L1[l1_a]×L1[l1_b(V)] ──T.gemm(initC=True)──> L0C[l0_c] ──T.copy(FIX, f32→f16)──> GM[ws_o]
                                                        │ set flag i
GM[ws_o] ──T.copy(MTE2)──> UB[ub_f16_D] ──vcast──> UB[ub_f32_D] ──vmul(α)+vadd──> UB[acc_o]（pass-2 逐块累加）
UB[acc_o]/[ub_ell]/[ub_m] ──vdiv/vlog2/vcast──> UB ──T.copy(行截断切片)──> GM[output (4D)] / GM[lse ([B,H,S,1] 视图)]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**（§1.6.0，调研对象为同一数学语义的算法族，源算法只是基线候选之一——本轮为修复上一版「漏检仓库两遍式先例」缺陷，调研范围显式覆盖 examples/ 全目录）；再数学等价地优化公式（§1.6.1）；再把全部计算点向量化（§1.6.2）；最后决定向量化轴与核内布局（§1.6.3）。本节结论是 §3.1 与 §6 的输入。分析对象 = §1.6.0 选定、经 §0.6 E1–E7 重设计落地的 NPU Expert 两相位算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

**调研对象**：`o = softmax(scale·q@k^T + mask) @ v`（+ lse）的 dense prefill 前向算法族，**目标域画像 = manifest causal 多头（H×B ∈ {8, 64, 128}，任务数 64–2048，S ∈ {512, 2048}，D ∈ {64, 128}）**。**调研深度**：融合类（GEMM + softmax + GEMM）→ 完整调研（R1 候选表 + R3 四口径表 + R4 逐候选评估）。**信息源**（本轮显式全量列举，修复上一版盲区）：① skill 参考表 §5「attention」命中行（algorithm-candidates.md）；② pattern-library 全主题文件（attention.md PL-1.7/1.8/1.9/1.9-hardlimits/1.9-twophase、traps-compiler.md VP-D6、constants.md）；③ **本仓 examples/ 全目录先例检索**（grep "softmax|attention|Attention" 全目录命中清单：flash_attention/{flash_attn_npuir.py, flash_attn_npuir_dev.py}、deepseek_v4/{example_sparse_attn_kernel(_highperf/_bwd).py, example_sparse_mla_fwd_kernel.py, example_hc_split_sinkhorn_kernel.py}、deepseek_v32/{sparse_mla_fwd(_exp/_dynamic_shape/_bwd).py}、fp8_lightning_indexer/、mixcv/、torch_tl_ops/{src,compile}/ops/flash_attention.py、Wechat-YATT/deepseek_v4/、TileOPs 侧本算子归档）——**两遍式 S/P 物化先例（flash_attn_npuir.py）本轮纳入为第一候选**；④ 源算法本身（§0.3/§0.4）；⑤ 外部已知算法（PyTorch/flash-attention 生态模型知识——仅取算法思路）。**网络检索**：未执行——本地信息源已充分覆盖本算子族的全部主流结构（flash 单遍 / 两相位物化 / flash-decoding / developer 形态均有本地先例或实测档案）；按 algorithm-research.md §6 第 6 层纪律如实记录（如 Stage 2 复核认为存在覆盖缺口可补检索）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1 完成）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源 flash online softmax**（单遍 per-block，exp2·LOG2E 域） | 分块 running max/sum + rescale，P 量化后进 GEMM2，fragment 驻留 | — | — | ✅（基线；其 NPU Expert 承载 = E1-E7 形态，见 #1） |
| 1 | **E1-E7 单遍 per-block 跨引擎链**（基线的上一版 NPU 承载） | 每 KV 块 C1→FLAG_S→V1→FLAG_P→C2→FLAG_O→V2 握手链 | 数学等价（同族算法不同承载） | —（**承载结构已被实测淘汰为目标域选择**：causal 域 3764/13265µs、fa 域调优终值仍 2.3× 慢于两相位） | ✅（证据对照） |
| 2 | **两相位 S/P 物化**（flash_attn_npuir.py 结构族） | Cube pass-1 全 S 物化 → Vector pass-1 softmax + scales[] 延迟回放 → Cube pass-2 全 PV 部分和物化 → Vector pass-2 累加；per-n-block flag | 数学等价（online 递推保留，α 回放序列与源逐块更新恒等——E2 论证） | Cube pass-1 零等待直跑；α/PV 解耦；实测 fa 域 2.3× 加速 | ✅ **（选定）** |
| 3 | e 域 vexp 换底（`e^{x−m}` 直接计算） | 换底恒等 e^u = 2^{u·log2e} | 数学恒等 | 少 2 次 [half,bn] 乘法舍入；数值路径更短 | ✅（§1.6.1 M1 采纳，并入 #2） |
| 4 | **KV-split / flash-decoding**（KV 维切分多核并行 + 跨核合并） | 每 split 独立 online 状态，合并 `ℓ = Σℓ_k·e^{m_k−m}`、`o = Σo_k·e^{m_k−m}` | 数学等价（log-sum-exp 合并律标准） | 小 Sq 欠载时并行度↑ | ✅（R3/R4 评估后 defer） |
| 5 | 批量重基准 G（G 块共享批末 max，O 回传降频） | batch 内延迟 m 收敛，O 每 G 块回传一次 | 数学等价（m 为真 running max，P ≤ 1 安全） | O 跨引擎流量 ÷G | ✅（R3/R4 评估后 defer——两相位 scales[] 回放已解除 α/PV 耦合，收益主体被 #2 吸收） |
| 6 | S/P/O 跨引擎 f16 传输（fixpipe/copy 隐式量化） | S、P、O_partial 以 f16 物化 | **容差内等价**（S 域 2^-11 相对误差 → ΔP/P ~0.26%；O 部分和 f16 实测 tier-1 0 flips） | 三族 ws 字节减半 | ✅（§1.6.1 M11/M12 采纳，并入 #2） |
| 7 | scale 折入 q（任务头一次 [bm,dim] 缩放替代每块 prescale） | (s·q)·k ≡ s·(q·k) | **容差内等价但精度降级**（源在 fp32 域乘 scale；f16 q 折入引入 2^-11 级 S 误差） | 每 task 省 NK−1 次 [half,bn] vmul | ✅（精度论证后否决，§1.6.1 M4——谱系结论承接） |
| 8 | lse 域公式 `log2(ℓ) + m·LOG2E` | log2(ℓ·e^m) 恒等 | 数学恒等 | 免大数指数化 | ✅（§1.6.1 M2 采纳） |
| 9 | f16 softmax 链（[half,bn] 大 pass 全 f16 算术） | vexp/vsub/vmul/reduce f16 + f16 精确 running max | 容差内等价（v9 实测四 case -3.5%~-14.1%；f16 ell 求和间距是残余 lse 误差主导项 ~1e-3） | 大 pass 字节减半 + 消 2 个 vcast pass | ✅（**defer 至 Stage 4**：v11nt 终版用 f32 链 + f16 传输已达标；f16 链为本域增量候选，§9.2 R-10） |
| 10 | NK==1 退化特化（online → direct softmax，删 α 携带链） | α_0 = e^{−inf−m} 无关，单块直接归一化 | **恒等精确**（bit-exact，v9 M7 先例） | 删 ~9 op/任务 | ✅（**defer**：causal 域 NK 随 bx 变化（bx=0 任务 NK=1），trace-time 全任务特化仅非 causal S_kv ≤ bn 域成立；per-task 运行时特化收益微小，§1.6.1 M7） |
| 11 | 除法转乘倒数（o/ℓ → o·(1/ℓ)） | 行广播除一次 vs 倒数+乘 | 数学恒等 | 无收益（每任务 1 次 vdiv，非热点） | ❌（op 数相同） |
| 12 | Winograd / 变换域 attention | 不适用（非卷积结构；attention 的 GEMM K=dim 小，无变换域收益先例；参考表无此族行） | — | — | ❌（结构不匹配） |

**R2 在线算法**：**有**——直接证据：源算法本身就是 online softmax（§0.4 #1：分块增量 max/sum/logsum + rescale，单遍 KV）；**选定结构 #2 同样保留 online 递推**（Vector pass-1 逐块 running max/ℓ，per-block α 因子存档回放——「在线」性质在 softmax 层完整保留；物化的是跨引擎传输的 S/P/O tile，非全量 score 矩形）。收益口径：KV 扫描 1 遍（vs 两遍 materialize-softmax 的 2 遍）；中间缓冲 O(bm × S_kv)（per-core 单槽 ws，8blong 基准 640KB/核）vs O(S_q × S_kv)（全量物化 512MB+）；S/P/O tile 均 L2 常驻可行（§4.3 总量 15.7MB « L2）。结构判据复核：softmax 分母与 PV 分子均可分解为 running 统计量 → 在线变体存在且源已采用 ✓。无需进一步检索。

**R3 复杂度对比**（四口径；以 manifest `llama-3.1-8b-long` (2,2048,32,128) causal fp16 为基准，有效 config 统一 bm=64/bn=256（#1 取其 causal 实测 config (64,64,1) 时另注）；FLOPs 口径 = 乘加计 2；访存分列「GM 必流量」（HBM 域）与「跨引擎 fabric 流量」（L2 命中域，含 ws 往返与 KV 重读）；扫描遍数 = q/k/v 各自 GM 读取特征；中间缓冲 = 峰值驻留）：

| 算法候选 | FLOPs | 访存量 | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|--------|---------|-------------|---------------------|
| #0/#1 基线 flash online（单遍 per-block 链，E1-E7 承载） | 68.7 GFLOP（causal 减半后；2 GEMM 主导） | HBM 必流量 ≈ 134MB（q/k/v 读 + o/lse 写各一次）；fabric：KV 重读 ≈ 1.13GB（bm=64）+ ws 往返 ≈ 1.65GB（S+P f16 各 [64,256]·2 + O f32 [64,128]·2 per 块 × 8602 块）≈ **2.78GB** | q 1 遍；k/v 每任务流式 1 遍（H×B=64 组 × 32 q-block 重读，L2 命中） | per 块 S 32KB + P 32KB + O 32KB；ws 多槽 2.6MB（L2） | 任务级 24 核并行（2048 任务）；跨核零同步；**每块 3 次 flag 握手 × 8602 块 = 2.6 万次跨引擎往返——发射流阻塞实测为结构地板**（fa 域 busy 核 aic_scalar 38–47%）；causal 域实测 13265µs |
| #2 两相位 S/P 物化（本设计） | 同 68.7 GFLOP（GEMM 工作量与 #1 逐块相同——同一批 S/PV 块，仅调度序不同） | HBM 必流量同 ≈ 134MB；fabric：KV 重读 ≈ 1.13GB + ws 往返 ≈ 1.38GB（S/P f16 同 #1，O **f16** [64,128]·2 减半）≈ **2.51GB**（比 #1 少 ~10%，非决定项） | 同 #1 | per 任务 S/P/O 窗口 ≈ 340KB（avg NK≈4.2：S/P 各 [64, ~1075] f16 + O [64, ~537] f16）；per-core 单槽 ws 640KB（最大窗口）；总量 15.7MB（L2 ✓） | 任务级 24 核并行（2048 任务）；跨核零同步；flag 次数同量级 **但 wait 与真依赖对齐**（pass-1 零等待直跑）——fa 域同口径实测 **98.05µs @ fa4096 vs #1 调优终值 226.72µs（2.31×）**；causal 域预估见 §11 |
| #4 KV-split（split=2） | 同 + 合并开销 O(S·D·split) 可忽略 | HBM 同；fabric + 每 split 状态回传 | 同 | + 每 split 状态 (m, ℓ, o) × 核数 | 任务级 × split 并行度↑（仅 Sq 很小时有意义）；**需跨核合并**：本仓同步原语均为核内语义（T.set_flag.md §1「在同一个Cube/Vector核内的不同通道之间」、T.sync_block_set.md §1「同一 block 中的其他执行单元」）——**无核间原语**，跨核合并须两阶段 kernel（host 串行两次 launch）或原子重构（o/m/ℓ 三态合并非单原子可表达） |
| #5 批量重基准（G=2/4） | 同 | O 回传 ÷G（两相位下 O 已 f16，收益再 ÷2） | 同 | UB + G×S 块缓冲 | 两相位下 α/PV 已解耦，G 的主要收益（消 α 耦合）被吸收；残余仅 O 流量 |
| #6 f16 传输（并入 #2 后） | 同 | O 列已计入 #2 | 同 | 同 #2 | 同 #2 |

**口径说明**：FLOPs 各候选相同（数学等价族，GEMM 块集合一致）——**本算子族的决定项是 fabric 流量与发射流行为**，非 FLOPs；#1 与 #2 的 fabric 流量差 ~8%，但实测性能差 2.31×——**结构差异（wait 与依赖的对齐方式）才是主导**，这正是 R4 的评估重心，也是上一版设计漏检 #2 后误选 #1 的教训。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 跨核结构 | 结论 |
|---------|-------------|---------|------------|---------|------------|---------|------|
| #2 两相位（本设计） | GEMM→Cube（分形 M≥16/K≥32 由 E7 钳位保证、K=dim 64/128 ✓）、softmax 链→Vector ✓ | L1 96KB/512KB（l1_a [64,256]+l1_b [256,128] f16，生命周期复用）✓；L0C [64,256] f32 = 64KB/128KB 单缓冲顺序复用 ✓；UB 177KB/192KB per AIV（§4.5 逐 buffer 预算）✓ | 尾轴 bn_eff/dim 为 16 倍数 ✓（bn_eff 公式保证）；[half,1] 行向量非搬运主路径（v11 `[NK·half,1]` scales 先例） | NK/K_A/tmc/tnc 为 kernel_id/task_id 派生 PrimExpr（E1-E7 D3 实证 + 开发指南 §3.3 模板同类） | Cube pass-1 零等待（MTE2/MTE1/C/FIX 四通道天然重叠）；pass-2 P-wait 与真依赖对齐；per-n-block flag 严格交替（v11 同款） | 任务独立零跨核同步 | ✅ **选定**（目标域任务数 64–2048 ≥ 24 核，饱和域无欠载；唯一例外 smoke 64 任务轻度欠载，§5.5 判定） |
| #0/#1 单遍 per-block 链 | 同上 | 同上（E1-E7 预算） | ✓ | ✓ | **实测否决**：`sync_block_wait` 阻塞整条发射流（PL-1.9-hardlimits：ns=1≈ns=2、wide≈serial、深流水 +12% 三重证据）；causal 域 (64,64,1) 3764/13265µs、fa 域调优终值仍 2.31× 慢于 #2 | 同上 | ❌（发射流结构地板，非 tiling 可解——fa 域 11 轮调优封闭验证） |
| #4 KV-split | 合并段 Vector ✓ | ✓ | ✓ | ✓ | 需**跨核**数据交换：T.set_flag.md §1 / T.sync_block_set.md §1 均为核内语义——**无核间原语**（负向断言依据：两文档 §1 适用范围条款亲自核对）；两阶段 kernel 的 host launch 开销 ~130µs 事件口径（opt_log §0）淹没小 shape 收益 | 目标域 H×B ∈ {64, 128}（smoke 除外）→ 任务数 1024–2048 » 24 核，**无欠载场景**——split 增加的并行度无需求方 | ⏸ **defer**：收益域（Sq 很小的 decode/小 batch）非本算子 workload；列入 §9.2 后续方向 |
| #5 批量重基准 | ✓ | UB +G 块缓冲 | ✓ | ✓ | 两相位下 α/PV 已解耦（scales[] 回放），G 的结构收益被吸收；残余仅 O 流量 ÷G（O 已 f16，绝对量小） | ✓ | ⏸ **defer**：并入 #2 后边际收益 <5%（O 占 ws 流量 ~20%），复杂度（Vector 缓冲 G 块 + 延迟处理）不成比例；Stage 4 候选 |
| #6 f16 传输（fp16 路径） | ✓ | ✓ | ✓ | ✓ | ✓（PL-1.9：fixpipe f32→f16 正确数值转换 probe 实测；v11nt D4 O-f16 tier-1 0 flips） | ✓ | ✅ **采纳并入 #2**（bf16 路径升 f32 承载，E5 论证） |
| #9 f16 softmax 链 | ✓ | ✓（大 pass 减半） | ✓ | ✓ | ✓（v9 实测有效） | ✓ | ⏸ **defer Stage 4**：v11nt f32 链 + f16 传输已达标，f16 链为本域增量候选（lse 求和间距风险已有 v9 量化先例 1e-3 « 有效容差 1.5e-2） |
| #7 scale 折入 q | ✓ | ✓ | ✓ | ✓ | 引入任务头跨引擎串行依赖 | ✓ | ❌ **否决**：精度语义降级（§1.6.1 M4 详证，谱系结论承接） |

**调研结论**：**选定算法族 = 两相位 S/P 物化 + per-n-block online softmax（候选 #2，含 #3/#6/#8 子项采纳）**。关键依据：① R4 实测证据——同机同口径 fa 域两相位 98.05µs vs 单遍链调优终值 226.72µs（2.31×），且单遍链的发射流阻塞（aic_scalar 38–47% flag 自旋）经 11 轮调优被证实为结构地板；causal 域当前跑在单遍链 (64,64,1) 上（3764/13265µs ≈ 2.25/5.17 TFLOPS，峰值的 0.6–1.4%），目标域 headroom >10×；② R3——两相位不增加 GEMM 工作量与 HBM 必流量，fabric 增量反而略优（O f16）；③ R4 全项亲和通过（L1/L0C/UB 预算 §4.5、PrimExpr 边界 D3 实证、per-n-block flag v11 同款、任务独立零跨核）。**与基线（源算法）的结构差异一句话**：online softmax 数学递推逐项保留，承载结构从「单遍 per-block 跨引擎握手链」换为「Cube/Vector 两相位独立大循环 + S/P/O GM 物化 + scales 延迟回放」。**无更优替代的调研范围**：skill 参考表 attention 命中行、pattern-library 全主题（attention/traps/constants/cases）、**examples/ 全目录 attention 族检索（12+ 文件，清单见本节信息源段）**、源码 §0.3/§0.4、TileOPs 侧本算子三轮调优全档案；候选 #4/#5/#9/#10 以量化理由 defer（非「未考虑」），#1/#7 以实测/精度理由否决。**源算法优化手段的意图承接**（§0.4 → 本设计，逐项）：#1 online softmax→保留（递推同构，pass-1 逐块 + scales 回放）；#2 SMEM tiling→L1 l1_a/l1_b 生命周期复用 + Q-hoist（E2）；#3 Pipelined→两相位 pass 结构（E2：重叠意图由 pass-1 零等待直跑承接）；#4 tensor core→Cube（E2）；#5 exp2 折叠→e 域 vexp（E4，意图「消独立乘法」由更短数值路径承接）；#6 fragment 驻留→UB 驻留 + ws 物化（E2——意图「消 GM 往返」让渡给 L2 命中的 ws 流量，为解锁发射流对齐，代价已计入 R3 且实测净赚 2.31×）；#7 P 量化→vcast 到输入 dtype（M10）；#8 causal 裁剪→NK PrimExpr（E1）；#9 mask 融合→Vector 向量 mask + 两段式 K_A（E3，意图「免独立 mask pass」由「全有效块零 mask + 向量化构造」承接）；#10 双路径→size-form 统一路径（E7）；#11–#12 CUDA 专属→舍弃；#13 lru_cache→保留；#14 三维 grid→一维 persistent（E1）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的两相位算法。R1 中初判采纳的候选在此完成四要素论证（原式 → 优化后公式 → 等价性论证 → 收益量化）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| M1 | e 域换底 | `exp2(x·scale − m·scale)`，scale = s·LOG2E | `vexp(x' − m')`，x' = prescale·x（fp32 域 vmul，softcap 分支 prescale=1 跳过） | 数学恒等：e^u = 2^{u·log2e}；源 NPU 等价路径 vexp2 底层 exp(a·ln2) 与 LOG2E·ln2=1 相消亦等价；本设计少 2 次 [half,bn] 乘法舍入 | 每 [rm,bn] 块省 2 个 v-op 的数值路径长度；数值更稳（谱系 23/23 + v11nt fa-tuned 实证） |
| M2 | lse 尾式 | `log2(Σ e^{x−m} · e^m)`（需恢复大数指数） | `lse = vlog2(ℓ) + m·LOG2E` | log2(ℓ·e^m) = log2(ℓ) + m·log2(e) 恒等 | 免 [rm,1] 大数乘法与潜在溢出；与源 S12 同式 |
| M3 | softmax 稳定化（减 max） | `e^{x_i}/Σe^{x_j}` | `e^{x_i−m}/Σe^{x_j−m}`，m = 行 max | 指数平移恒等（分子分母同乘 e^{−m}） | 消溢出（必选，源同构） |
| M4 | **否决**：scale 折入 q | 每块 `vmul(S_f32, s)`（fp32 域） | 任务头 `vmul(q_f16, s)` 后进 GEMM | **容差内等价但精度语义降级，不采纳**：源乘 scale 在 fp32 域（相对误差 2^-24）；f16 q 折入使每 q 元素先舍入（2^-11）再进 L0C fp32 累加——S 相对误差升至 ~2^-11，经 exp 传播 ΔP/P ~ 2^-11·\|S\|·scale，叠加既有误差后威胁 §8.2 冻结容差（谱系 v2 M4 结论承接） | （否决记录）理论省 NK−1 次 [half,bn] vmul/任务，但精度代价不成比例 |
| M5 | mask 选择式（NaN 无关） | `if_then_else(valid, S, −inf)`（源；等值判定对 NaN 失效） | `vselect(cond, S, −inf)`，f32 −inf 哨兵 | 按位选择与 S 值无关；masked 位 `e^{−inf−m}`（m 经 M6 clamp 恒 ≥ −1e38 有限）= 精确 +0.0 ≡ 源 `e^{−inf}=0`；杜绝 −inf−(−inf) NaN 路径（clamp 保证 m 有限） | 消 4096+ 次/块谓词标量写（developer 实测 aiv_scalar 88–89% 主导项）→ 5 个向量 op/块且仅对角块触发（K_A 分界：S_q=S_kv causal 每任务 1 块） |
| M6 | 单调 max + clamp | `max(max_prev, block_max)` + clamp ≥ −1e38 | 同式（vmax ×2，[rm,1] 行向量） | 源 NaN guard 逐字保留（online_softmax.py L128–L131）：m ≥ −1e38 恒成立 → masked 位 `e^{−inf−m}` 精确 0、无 NaN 路径 | 防御性（必选，源同构）；2 个行向量 op/块，开销可忽略 |
| M7 | NK==1 退化特化 | online 递推（α_0 值无关：logsum/acc_o 为 0） | **defer**：causal 域 NK 随 bx 变化（bx=0 任务 NK=1、末任务 NK=NK_max），trace-time 全任务特化仅当 `非 causal 且 ceildiv(S_kv,bn_eff)==1` 成立；该域（S_kv ≤ bn_eff 非 causal）属契约测试小 shape 非目标域；per-task 运行时特化（`if NK == 1` 分支）省 ~9 op 仅对 bx=0 任务，收益微小 | 恒等精确（v9 M7 六 fixture 实证先例） | （defer 记录）非 causal 小 KV 域留 Stage 4 按需启用 |
| M8 | prescale=1.0 跳过 | softcap 分支 `vmul(S, 1.0)` | trace-time `if prescale != 1.0` 跳过 | 乘 1 恒等（bit-exact：x·1.0 = x） | softcap 路径省 1 个 [rm,bn] op/块（微小，卫生项） |
| M9 | **否决**：O 部分和 bf16 回传（bf16 路径） | O_partial f32（bf16 输入时） | **维持 f32**（bf16 路径）；fp16 路径 f16（M11） | O_partial = Σ_(j∈块) P·V；bf16 尾数 7 位 → 部分和相对误差 ~2^-8，NK 块累加且 \|O\| 可达 ~25 → 绝对误差 ~0.1 » atol 5e-3（**超容差 ~20×**）；f16 路径 2^-11 且 v11nt 实测 tier-1 0 flips 通过——fp16 采纳 f16、bf16 保持 f32 为分 dtype 裁决 | （否决记录）bf16 路径 O 保持 f32（4.5MB/核 @8blong）；fp16 路径 f16 减半 |
| M10 | P 量化点保持 | 源：acc_s(fp32) → cast 输入 dtype → GEMM2 | `vcast(P_f32 → 输入 dtype, rint)` 后经 ws_p 进 Cube | 量化点与源一致（softmax 后、GEMM2 前、最近舍入）；**与源逐位同 dtype**——fp16 输入时 f16（10 位尾数）、bf16 输入时 bf16（7 位） | 语义对齐（防漂移项，无独立收益） |
| M11 | O 部分和 f16 物化（fp16 路径） | O_partial f32 跨引擎 | `T.copy(l0_c f32 → ws_o f16)`（隐式 rint 量化） | **容差内等价，实测采纳**：v11nt D4 探针 + fa-tuned 4 case tier-1 0 flips、lse 8.7e-5——L0C f32→GM f16 为正确数值转换（PL-1.9 probe 实测 max_rel 4.7e-4 = 1 f16 ulp）；部分和幅度 \|P·V\|（P ≤ 1、行和 ≤ ℓ_block）下 f16 舍入在累加中部分抵消，实测远低于 M9 的 bf16 界 | O 跨引擎字节减半（32→16KB/块 [64,128]）；8blong 全量 ws_o 流量 578→289MB |
| M12 | S f16 物化（fp16 路径） | S f32 跨引擎 | `T.copy(l0_c f32 → ws_s f16)` | **容差内等价**：S 原始值（prescale 前）\|S\| ≲ 60（randn 输入 √dim 尺度）f16 相对误差 2^-11 → Δ(S·scale) ≈ 60·0.088·2^-11 ≈ 2.6e-3 → ΔP/P ~2.6e-3，叠加 P 自身量化（2^-11）与 fp32 累加后仍在 §8.2 tier-1/tier-2 预算内（v11nt fa-tuned 实证 + S-f16 probe 4.7e-4） | S 跨引擎字节减半（[64,256] f32 256KB→128KB/块）；Vector 侧 MTE2 读减半 |
| M13 | Q-hoist | 参考实现 pass-1 每 n-block 重读 Q（flash_attn_npuir.py L88–94 在 i 循环内） | Q 在任务头一次装载 l1_a，pass-1 全程复用 | 数据不变（同一 Q 块），装载位置前移 | 免 NK−1 次 [tmc,dim] GM→L1 重读（8blong 每 (b,h) 32 任务 × ~3.2 次冗余 × 16KB ≈ 33MB 冗余消除）；v11nt vs 参考实现的优势项之一（PL-1.9-twophase） |
| M14 | ell 递推 clear=True | `T.reduce(..., "sum")` 无 clear（flash_attn_npuir.py 参考形态） | `T.reduce(..., "sum", clear=True)` | 参考的无 clear 形态为**文档化静默错误形态**（docs/Tilelang.language/规约操作/T.reduce.md §2.3：「clear=False 对未初始化 buffer 不会报错，但会产生静默数值错误」——dst 残留累加）；clear=True 后 `ℓ ← ℓ·α + ΣP` 与源 macro ⑨ 恒等 | 正确性修复项（v11 OP6：修正后 lse 精度 8.7e-5 优于参考实现） |

**优化结论**：采纳 M1/M2/M3/M5/M6/M8/M10/M11/M12/M13/M14 共 11 项；否决 M4/M9（精度论证）；defer M7（域外）。**全部采纳项经 `verify_equiv.py` 机器验证 13/13 EQUIV_PASS（11 项逐项 + E2E fp16/bf16 双 trace 组合，结果表见本节末）**。**优化后公式**（供 §3.1 使用）：

$$
\text{prescale} = \begin{cases} 1.0 & \text{softcap}>0 \\ \text{scale}_{\text{sc}} & \text{else} \end{cases},\quad
S'' = \text{prescale}\cdot\text{vselect}\big(\text{cond},\ \text{softcap}(S)\ 或\ S,\ -\infty\big)
$$
$$
m_{\text{cur}} = \max_j S''_{i,j},\quad m = \max(m_{\text{prev}}, m_{\text{cur}}, -10^{38}),\quad
\alpha_i = e^{m_{\text{prev}} - m},\quad P = e^{S'' - m}
$$
$$
\ell \leftarrow \ell\cdot\alpha_i + \textstyle\sum_j P_{i,j},\qquad
o \leftarrow o\cdot\alpha_i + \big(\textstyle\sum_j P_{i,j}\, v_{j,:}\big)
$$
$$
\text{out} = o/\ell,\quad \text{lse} = \log_2 \ell + m\cdot\text{LOG2E}
$$

（α_i 序列在 pass-1 计算并作用于 ℓ，同时存档 `scales[]`；pass-2 以同一序列作用于 o——两侧恒等，E2 论证。上式第二条更新中的 PV 部分和 Σ_j P_{i,j}·v_{j,:} 由 Cube pass-2 计算、经 ws_o 以 f16 物化回传（M11；bf16 路径 f32——E5）。）

**等价性机器验证（S1-EQUIV-EXEC 门禁项；`verify_equiv.py`，torch 2.7.1+cpu，2026-09-15 实跑，退出码 0，双次运行输出一致）**

对照口径：**基线式** = 上表「原式」列（GPU 源算子公式忠实模拟：exp2 域 + 源截断常数 LOG2E=1.44269504 + 源运算序，gqa_fwd.py L788–L1033 + online_softmax.py）；**优化式** = 上表「优化后公式」列（e 域 + 全精度常数 1.4426950408889634，v11nt 谱系同值）；**参照** = fp64（torch CPU double）同输入真值，M11/M12/E2E 的容差门另以 §8.1 golden 语义（fp32 物化 softmax + P 量化到输入 dtype）为参照并套 §8.2 冻结门（tier-1 out atol 5e-3/rtol 1e-5、lse atol/rtol 1e-3；bf16 按 §8.2 双门 + tier-3 分类口径）；**max ULP** = IEEE 单调键法（优化式 vs 基线式）；**违反率** = |a−b| > atol + rtol·|ref| 的元素占比，分「vs 参照」与「vs 基线」两口径。工作负载 = 5 组 causal 右对齐随机域（L∈{128,256}、S∈{256,512}、D∈{64,128}、含 softcap=30 一组）fp16 + 4 组 bf16 镜像。判定阈值：恒等链类 ≤128 ulp（第一性原理最坏界：softcap 域 exp 入参 |t| ≤ 60·LOG2E ≈ 87，两式 RN 链差 |δt| ≤ 173·2^-24 → P 域 ≤ 173·ln2 ≈ 120，取整 128；实测 15/1/8）；递推链 ≤16（实测 3）；逐位类 =0；容差类 tier-1 违反率 =0。M4/M9（已否决）与 M7（defer）不在采纳验证范围。

| 项 | 比较域 (dtype) | max ULP (opt vs base) | 违反率 (opt vs 参照) | 违反率 (opt vs base) | 结论 |
|----|---------------|----------------------|--------------------|--------------------|------|
| M1 | P (fp32) | 15 | 0 | 0 | EQUIV_PASS |
| M2 | lse (fp32) | 1 | 0 | 0 | EQUIV_PASS |
| M3 | softmax p (fp32) | 8 | 0 | 0 | EQUIV_PASS |
| M5 | P masked/valid (fp32) | 0 | 0 | 0 | EQUIV_PASS |
| M6 | m/P (fp32) | 0 | 0 | 0 | EQUIV_PASS |
| M8 | x·1.0 (fp32/f16/bf16) | 0 | 0 | 0 | EQUIV_PASS |
| M10 | P 量化 (f16/bf16) | 0 | 0（另：>0.5 ulp 舍入误差率 0） | 0 | EQUIV_PASS |
| M11 | out/lse (f16 trace) | 33213※ | 0 | 0 | EQUIV_PASS |
| M12 | out/lse (f16 trace) | 34463※ | 0 | 0 | EQUIV_PASS |
| M13 | Q/S (f16/bf16/fp32) | 0 | 0 | 0 | EQUIV_PASS |
| M14 | ℓ (fp32) | 3 | 0 | 0 | EQUIV_PASS |
| E2E-fp16 | out/lse vs golden (f16) | 34407※ | 0（tier-1 双门） | — | EQUIV_PASS |
| E2E-bf16 | out/lse vs golden (bf16) | 33685※ | 0（tier-1/tier-2/lse 三门） | — | EQUIV_PASS |

※ 容差类行的 ULP 为**诊断值**：小幅度 out 元素（|out| ~ 1e-4）在 f16 网格上 ulp ≈ 2^-24，atol=5e-3 主导——判定以违反率为准。max abs 佐证：M11 vs base(O-f32) 2.4e-4 / vs golden 6.1e-4（lse 不受扰，ulp=0）；M12 vs base(S-f32) 7.3e-4 / vs golden 6.1e-4；E2E-fp16 vs golden 6.1e-4、vs fp64 理想 5.9e-4；E2E-bf16 tier-1 率 0、tier-2 率 0、lse 门 0 违反。

关键断言明细（M5/M6 精确零类）：masked 位 P 逐位 +0.0（bit-exact，非 −0.0）、输出无 NaN、NaN/±Inf 垃圾注入下 vselect 形态干净；有效行 clamp 恒不绑定（与无 clamp 逐位一致）；全掩码行 clamp 后 m=−1e38 有限、P 全 +0.0 无 NaN。逐位类明细：M8 对 ±0/±Inf/亚正规/有限值三 dtype 逐位相等；M10 rint 量化与 f64 双舍入参照逐位一致且误差 ≤0.5 ulp；M13 拷贝位保持 + hoist/重读两形态同算逐位一致。

探针结论（info，不参与判定，佐证采纳理由）：M2 溢出探针 m=120 基线式 inf / 优化式有限；M3 大分数探针 x∈[80,96) 基线式 NaN / 优化式 err 2e-8；M5 NaN 垃圾下源预填+累加形态（−inf+NaN=NaN）与 softcap 等值判定形态（NaN≠−inf 存活）均 NaN 污染、vselect 形态干净；M6 无 clamp 探针 −inf−(−inf)=NaN；M14 no-clear 残留累加使 ℓ 膨胀 24.74×（T.reduce.md §2.3 静默错误形态——clear=True 为正确性修复项）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 分析对象 = 两相位实现方案的全部循环与标量计算点。替代方案所用 API 均有 docs/Tilelang.language/ 或 examples/ 佐证（docs 目录存在性已逐一核验：数学操作/、规约操作/、创建操作/、比较操作/、条件操作/、逻辑操作/、shape操作/、数据类型转换操作/）。

| # | 计算点 | 原实现形态（源 / E1-E7） | 向量替代方案（本设计） | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|---------------------------|------------------------|---------|------------------------------|
| 1 | mask 构造 | `T.Parallel(bm, bn)` 嵌套 if/else 谓词写（4096+ 次/块） | `T.arange`(strides=[0,1]/[1,0]) + `T.vsub` + `T.vcmp`("le") + [`T.vcmp`("lt") + `T.vand` 尾带变体] + `T.vselect`（E3；API：创建操作/T.arange.md、数学操作/T.vsub.md、比较操作/T.vcmp.md、逻辑操作/T.vand.md、条件操作/T.vselect.md；highperf L345–354 + v11 E3 同构） | ✅ | — |
| 2 | softcap 变换 | macro 内 `T.Parallel(bm,bn)` tanh + if_then_else | `T.vtanh`（fp32 操作数，数学操作/T.vtanh.md；D5 教训：专用 fp32 scratch）+ `T.vmul` 标量广播 + `T.vselect`（−inf 保持由 mask 链统一承担——softcap 后 mask 恢复 −inf，E3 顺序） | ✅ | — |
| 3 | prescale 乘 | `T.Parallel` / 每元素 | `T.vmul(ub_f32_N, prescale, ub_f32_N)`（标量广播，数学操作/T.vmul.md §2.2.2） | ✅ | — |
| 4 | 行 max | warp 归约树 / reduce_max | `T.reduce(ub_f32_N, ub_mcur, dims=[1], reduce_mode="max", clear=True)`（规约操作/T.reduce.md §2.2.2「src/dst 仅一维不同且为 1」） | ✅ | — |
| 5 | 单调 max / clamp | `T.Parallel(bm)` 标量 max | `T.vmax` ×2（行向量 [rm,1]，数学操作/T.vmax.md） | ✅ | — |
| 6 | α = e^{m_prev−m} | `T.Parallel(bm)` exp2 | `T.vsub` + `T.vexp`（行向量，数学操作/T.vexp.md） | ✅ | — |
| 7 | P = e^{S−m} | `T.Parallel(bm,bn)` exp2 | `T.vsub`（[M,N]−[M,1] 行广播）+ `T.vexp` | ✅ | — |
| 8 | 行 sum | warp 归约 | `T.reduce(..., "sum", clear=True)`（**clear 显式 True**——T.reduce.md §2.3 静默错误条款） | ✅ | — |
| 9 | ℓ 更新 / o rescale 累加 | `T.Parallel` 标量 | `T.vmul`/`T.vadd`（行广播 / [rm,dim]，数学操作/T.vadd.md） | ✅ | — |
| 10 | P 量化 cast | `T.copy(acc_s, acc_s_cast)` | `T.vcast(ub_f32_N, ub_f16_N, round_mode="rint")`（数据类型转换操作/T.vcast.md） | ✅ | — |
| 11 | epilogue 归一化 | `T.Parallel` 除法 | `T.vdiv(acc_o, ub_ell, acc_o)`（[M,N]÷[M,1] 行广播，数学操作/T.vdiv.md） | ✅ | — |
| 12 | lse 尾式 | `T.Parallel(bm)` log2 + 乘加 | `T.vlog2(ub_ell, ub_lse, ub_lse_tmp)`（三参，数学操作/T.vlog2.md §1）+ `T.vmul` + `T.vadd` | ✅ | — |
| 13 | S/O 到达后的 vcast（f16→f32） | —（两相位新增搬运点） | `T.vcast(ub_f16_N, ub_f32_N, "rint")`（v11 L1375–1377 同构） | ✅ | — |
| 14 | 输出 / lse 写出 | 谓词标量 GM 写（源 S13） | 行截断切片 `T.copy`（E7；v11 L1443–L1446 4D 切片先例；D3 教训：src 切片 [0:rm, 0:dim]） | ✅ | — |
| 15 | 状态初始化 | `T.clear`/`T.fill`（源 S2） | `T.vbrc(0/−inf, ub)`（shape操作/T.vbrc.md；标量 let-bound，D4） | ✅ | — |
| 16 | 任务解码 / NK / K_A / K_blk / tail 钳位标量 | — | 无向量等价（每任务 O(1) 索引算术） | ❌ | **block 级索引 / 任务映射计算**：每任务 ~10 次整数标量运算（cid → bx/by/bz、NK/K_A/tmc/tnc），非逐元素热点（对比每任务数万向量元素）；GPU 源码同款结构（loop_range 亦为标量） |
| 17 | `if i != 0` / `if i >= K_A` / `if task_id > 0` 控制分支 | — | 无向量等价 | ❌ | **tile 级顺序依赖**：首块 α 语义（logsum/acc_o 为 0，α_0 值无关——跳过 vmax/vsub/vexp 是恒等消除）、mask 链的两段式分界（K_A）、任务边界握手——均为块序控制流非数据并行；E3/E1-E7 生产先例 |
| 18 | 工厂层 config/校验/常量折叠 | — | 无向量等价 | ❌ | **host 侧元数据计算**（不在 kernel 内）：trace-time Python（VP-P2 变体分派先例） |

**向量化结论**：逐元素计算已全部向量化（替代方案 API 均有 docs 佐证）；保留 3 类标量/控制流——block 索引（#16）、tile 级顺序依赖控制流（#17）、host 元数据（#18），逐项理由见表。**与 §6 一致性**：§6 循环结构中不出现任何逐元素标量循环。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout（BSHD 4D 进 / BSHD 4D + [B,H,S] lse 出）是契约；核内布局与 lane 映射是设计变量。本算子为 Cube/Vector 混合类（MixCV）：Cube 侧分形布局由 `T.gemm`/`b_transpose` 内部承载（§1.6.3 记录决策与依据），Vector 侧的 softmax 链与累加链须枚举轴候选。

**Vector 侧候选矩阵**（fp16 向量宽度 ×8 = 128bit；行向量 = [rm,1] 广播语义）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **[rm, bn] 行主序 tile（S/P 链）+ [rm, dim] 行主序 tile（O 链），lane = 列连续轴** | 列轴（n 维 / d 维，stride=1 连续） | **无需 repack**——ws_s/ws_p 的 [bm, pad_kv] 行主序与 UB tile 同构，切片 copy 直达（v11 L1369–1374 同构） | bn=256 fp16 → 每行 32 个 8-lane 向量组，尾 lane 浪费 0%；行广播（vmul [M,N]×[M,1]）沿行维展开为常数广播——文档支持（T.vmul.md §2.2.2）；行归约 `T.reduce(dims=[1])` 沿连续轴归约（T.reduce.md §2.2.2 支持形态） | ✅ **主选** |
| 2 | [bn, rm] 列主序（lane = 行轴） | 行轴 | 需核内转置（T.transpose 二轴交换，创建操作/T.transpose.md——UB 级执行但 **VP-D6 实测活跃源 transpose 毒化整 kernel 2.6×**，pattern-library §2 陷阱条目绑定同工具链戳） | 行广播退化为跨步广播（stride=bn）；归约沿跨步轴——**且 transpose 毒化税 2.6× 直接否决** | ❌（毒化实测 + 广播退化双重否决；依据：pattern-library traps-compiler.md VP-D6 + v9 -20% 双证） |
| 3 | 单 AIV 全行 [bm, bn]（不半分） | 列轴 | 无 | UB 预算翻倍（§4.5：177KB → ~330KB » 192KB ✗ 编译失败）；且放弃 2×AIV 并行 | ❌（UB 容量硬上限否决；UB 192KB per AIV，PL-1.9-hardlimits 实测口径） |
| 4 | GPU 源码轴映射（warp 32 线程 × GemmWarpPolicy.FullRow） | — | — | — | ❌（GPU thread/warp 轴与 NPU Vector 轴不对应——migration-analysis.md §5.3「block×thread 两级并行 → 重新设计」；本设计 Vector 轴独立评估，见 #1） |

**Cube 侧决策**（按 mixcv/cube 候选清单记录）：分形 NZ 布局由 `T.gemm(l1_a, l1_b, l0_c, b_transpose=True, size=[tmc, dim, tnc])` 内部承载（T.gemm.md §2.1 Expert 签名；`b_transpose` 表达 K^T 语义——flash_attn_npuir.py L102–109 同构）；**不采用** `T.load_nd2nz`（D1 实测：对 BSHD 跨头切片静默平铺误读——`T.load_nd2nz(q[bz, s_lo, by, 0], l1, [m, dim])` 把跨维 (S,D) tile 当连续内存读，probe 复现 max diff 49；**BSHD 定头切片必须 slice-form `T.copy`**，CG-2026-0004）；L1 缓冲 l1_a/l1_b 生命周期复用（pass-1 Q/K → pass-2 P/V，v11 block_share=max(bn,dim) 形态）；epilogue 归属：归一化/lse/cast 全在 Vector 侧（L0C 不能被 v 算子直接操作——store_fixpipe.md §2.1 dst 仅 GM，L0C 出数必经 GM ws 中转，E2）。

**布局决策结论**：选定方案 #1——Vector 链核内布局 = 行主序 [rm, bn]/[rm, dim] tile、向量化轴 = 列连续轴、repack = 无（ws 与 UB 同构切片直达）；Cube 侧 = T.gemm 内部分形 + slice-form T.copy 装载 + L1 生命周期复用。该结论同步落入 §3.3 伪代码与 §6 循环结构（累加循环的内层向量维 = 列轴）；弃选方案量化理由见候选矩阵（#2 毒化税 2.6×、#3 UB 超限、#4 轴系不对应）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **Expert**（用户显式指定延续 expert 谱系；结构与证据双确认，见 2.2）

### 2.2 选型理由

1. **结构表达力（决定性）**：选定算法（§1.6.0 #2 两相位）要求——`T.Scope("Cube")`/`T.Scope("Vector")` 显式双引擎分工、两相位独立大循环的手动调度、`T.alloc_L1`/`T.alloc_L0C`/`T.alloc_ub` 显式分层、per-n-block `T.sync_block_set/wait` 手动握手、`T.rs("PIPE_*")` 通道标注。这些均为 Expert 形态 API（T.load_nd2nz.md §2.3「适用 Expert 模式，在 Developer 模式中使用 copy 接口」同族口径；TILELANG_ASCEND_MODE 默认即 Expert——docs/developer/EnvironmentVariables.md L31）。**结构原型 flash_attn_npuir.py 与 tuned 母本 v11nt 均为 Expert 形态**（双 Scope + alloc_L1/L0C/ub + sync_block_set/wait）。
2. **Developer 模式对照（用户要求的论证）**：Developer 形态（fragment + 自动 cv_split + T.Pipelined）的并行度由编译器自动切分，**无法表达「Cube pass-1 全 S 无等待直跑 + pass-2 逐块等 P」的两相位调度与 per-n-block 手动握手**——其自动流水产出的是 per-block 串行链形态（E1-E7 迁移前的 developer 基线即此形态：causal 域 2872/20883µs，PL-1.7——短 workload 比 E1-E7 快 1.31× 但 long 慢 1.57–1.63×，两者较两相位预期均差 >10×）。**结论：目标域不存在 developer 模式更优的结构**——developer 可表达的算法子集（单遍自动流水）已被实测排除，两相位结构超出 developer 表达力。
3. **谱系延续**：2026-09-07 起四轮任务均 expert；本轮改道仅前移重启点，未变更模式诉求。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | 显式 `T.alloc_L1`（l1_a/l1_b）、`T.alloc_L0C`（l0_c 单缓冲）、`T.alloc_ub`（Vector 链全量缓冲，§4.5 预算） |
| 计算方式 | Cube：`T.gemm`（Scope("Cube") 内，initC/b_transpose/size）；Vector：v-prefix 链（vcast/vtanh/vsub/vcmp/vand/vselect/vmul/vexp/vmax/reduce/vadd/vdiv/vlog2/vbrc） |
| 同步 | 手动 `T.sync_block_set/wait`（per-n-block id + FLAG_TASKDONE，§7.2）；`T.rs("PIPE_MTE2"/"PIPE_MTE3"/"PIPE_FIX"/"PIPE_V")` 通道标注 |
| 流水 | 手动两相位（无 T.Pipelined）；pass_configs 双关闭防编译器重排（§7.3） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式；每步与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `S = q_tile @ k_tile^T`（size [tmc, dim, tnc]） | QK^T（Cube pass-1，Q-hoist） |
| 2 | `S_f16 ← f16(S_f32)`；`Ŝ ← f32(S_f16)` | S 物化 + 到达（M12；bf16 路径 f32 直传） |
| 3 | `S̃ = softcap·tanh(prescale_sc·Ŝ/softcap)`（softcap>0） | softcap（Vector pass-1，M8 分支） |
| 4 | `S'' = prescale · vselect(cond, S̃ or Ŝ, −inf)` | mask + prescale（M5；cond = causal ∧ [尾带 OOB]） |
| 5 | `m_cur = max_j S''`；`m = max(m_prev, m_cur, −1e38)` | 行 max + 单调 + clamp（M3/M6） |
| 6 | `α_i = e^{m_prev − m}`；`P = e^{S'' − m}` | α 与 P（M1 e 域） |
| 7 | `ℓ ← ℓ·α_i + Σ_j P`；`scales[i] ← α_i` | ℓ 递推 + α 存档（clear=True，M14） |
| 8 | `P_dtype ← dtype(P)`；`O_partial(i) = P_dtype @ v_tile`（size [tmc, tnc, dim]） | P 量化（M10）+ PV 部分和（Cube pass-2） |
| 9 | `o ← o·α_i + f32(O_partial_f16)` | 延迟 rescale 累加（Vector pass-2，M11） |
| 10 | `out = o/ℓ`；`lse = log2(ℓ) + m·LOG2E` | epilogue（M2；transpose-free 写出） |

### 3.2 TileLang API 映射（Expert 模式，逐段）

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | Q/K/V 装载 | `T.copy`（slice 形态） | `T.copy(q[bz, s_lo:s_lo+tmc, by, 0:dim], l1_a[0:tmc, 0:dim])`；k/v 同款（D1 约束：BSHD 定头切片必须 slice 形态） | Expert |
| 1 | QK^T | `T.gemm` | `T.gemm(l1_a, l1_b, l0_c, initC=True, b_transpose=True, size=[tmc, dim, tnc])`（T.gemm.md §2.1） | Expert |
| 2 | S 物化 | `T.copy` | `T.copy(l0_c[0:tmc, 0:tnc], ws_s[kid, 0:tmc, n_lo:n_lo+tnc])`（f32→f16 隐式转换，PL-1.9 probe） | Expert |
| 3 | softcap | `T.vtanh` + `T.vmul` + `T.vselect` | fp32 操作数（D5）；vselect −inf 保持并入 mask 链 | Expert |
| 4 | mask | `T.arange` + `T.vsub` + `T.vcmp` + `T.vand` + `T.vselect` | int16 索引矩阵 + 整数 PrimExpr 阈值（D4）；−inf f32 哨兵（M5） | Expert |
| 4 | prescale | `T.vmul` | 标量广播（trace-time prescale≠1 门控，M8） | Expert |
| 5 | 行 max / 单调 / clamp | `T.reduce` + `T.vmax` ×2 | `dims=[1], reduce_mode="max", clear=True`（T.reduce.md §2.2.2） | Expert |
| 6 | α / P | `T.vsub` + `T.vexp` | 行向量 / 行广播 | Expert |
| 7 | 行 sum + ℓ 更新 + α 存档 | `T.reduce` + `T.vmul`/`T.vadd` + `T.copy` | `reduce_mode="sum", clear=True`（§2.3 强制）；`scales[i·half:…, 0:1]` 切片存取（v11 同构） | Expert |
| 8 | P 量化 + 回传 | `T.vcast` + `T.copy` | `round_mode="rint"` 到输入 dtype（M10）；PIPE_MTE3 段内 copy + set flag | Expert |
| 8 | PV 部分和 | `T.gemm` | `T.gemm(l1_a, l1_b, l0_c, initC=True, size=[tmc, tnc, dim])` | Expert |
| 9 | O 累加 | `T.vcast` + `T.vmul` + `T.vadd` | `vcast(ub_f16_D, ub_f32_D, "rint")`；行广播 rescale | Expert |
| 10 | epilogue | `T.vdiv` + `T.vcast` + `T.vlog2` + `T.copy` | `vlog2(ub_ell, ub_lse, ub_lse_tmp)` 三参；行截断切片写出（[B,H,S,1] 视图） | Expert |
| — | 跨引擎握手 | `T.sync_block_set` / `T.sync_block_wait` | flag id = n-block 下标 + FLAG_TASKDONE（§7.2 表）；通道标注 `T.rs("PIPE_MTE2"/"PIPE_MTE3"/"PIPE_FIX"/"PIPE_V")` | Expert |
| — | 状态初始化 | `T.vbrc` | let-bound 标量（D4） | Expert |

### 3.3 计算伪代码

> T.Tensor 的 dtype 按模板 §3.3 规则以位置参数传递（false-alarm 规避）。完整结构 = §0.6 E2 的 Cube/Vector 流伪代码（权威形态），此处补任务解码外壳与 trace 家族说明：

```python
# 工厂层（trace-time）
# bn_eff = max(bn_caller_or_default, ceil16(ceildiv(S_kv, 15)))   # flag 预算守卫（E6）
# NK_max = ceildiv(S_kv, bn_eff); pad_kv = NK_max * bn_eff
# trace 家族（VP-P2 工厂级多 @T.prim_func 分派，Python if 选 builder）:
#   causal-nosoftcap（主 trace，manifest 域）/ causal-softcap / noncausal（无 mask 缓冲）
#   × fp16 / bf16（ws dtype: f16/f16/f16 vs f32/bf16/f32）
#   × has_band（尾带 OOB 掩码变体，S_kv % bn_eff ∉ {0} ∪ [32,∞)∩16 倍数 时）

@T.prim_func
def _gqa_prefill_fwd_main(
    q: T.Tensor((batch, seq_len_q, heads, dim), dtype),
    k: T.Tensor((batch, seq_len_kv, heads_kv, dim), dtype),
    v: T.Tensor((batch, seq_len_kv, heads_kv, dim), dtype),
    ws_s: T.Tensor((NUM_KERNELS, bm, pad_kv), ws_s_dtype),
    ws_p: T.Tensor((NUM_KERNELS, bm, pad_kv), dtype),
    ws_o: T.Tensor((NUM_KERNELS, bm, dim * NK_max), ws_o_dtype),
    output: T.Tensor((batch, seq_len_q, heads, dim), dtype),
    lse: T.Tensor((batch, heads, seq_len_q, 1), "float32"),
):
    with T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid):
        HB = heads * batch
        num_local = T.ceildiv(num_logical - kernel_id, NUM_KERNELS)
        # ---- Cube Scope: persistent 任务循环 → [TASKDONE wait] → Q-hoist → pass-1 → pass-2（§0.6 E2 Cube 流）
        with T.Scope("Cube"):
            l1_a = T.alloc_L1([bm, block_share], dtype)      # block_share = max(bn_eff, dim)
            l1_b = T.alloc_L1([bn_eff, dim], dtype)
            l0_c = T.alloc_L0C([bm, block_share], "float32")  # 单缓冲，gemm1/gemm2 顺序复用
            for task_id in T.serial(num_local):
                cid = task_id * NUM_KERNELS + kernel_id
                bz = cid % batch; by = (cid // batch) % heads; bx = cid // HB
                s_lo = bx * bm; tail_m_real = T.min(bm, seq_len_q - s_lo)
                tmc = T.max(16, T.min(bm, ceil16(tail_m_real)))
                kv_head = by // groups
                NK = T.ceildiv(T.min((bx + 1) * bm + causal_offset, seq_len_kv), bn_eff)  # causal；非 causal = ceildiv(S_kv, bn_eff)
                # ...（§0.6 E2 Cube 流：任务头 / pass-1 / pass-2）
        # ---- Vector Scope: persistent 任务循环 → init → pass-1 → pass-2 → epilogue（§0.6 E2 Vector 流）
        with T.Scope("Vector"):
            # ub 缓冲族（§4.5 预算表）；ub_colmat/ub_rowmat/ub_neg 任务循环外一次构建
            for task_id in T.serial(num_local):
                # ...（§0.6 E2 Vector 流；real_m 自适应半分；K_A/K_blk per-AIV 阈值）
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.gemm`（Expert 签名 initC/b_transpose/size） | docs/Tilelang.language/线性代数操作/T.gemm.md §2.1/§2.2.1/§2.3；flash_attn_npuir.py L102–109/L145–151 | ✅ 先例实测（v11nt 98.05µs 达标） |
| `T.copy`（slice 形态 GM↔L1/UB/L0C→GM，跨 dtype 隐式转换） | debug_log D1（slice 形态对 strided BSHD 逐位正确）；PL-1.9（f32→f16 转换 probe）；v11 全套 | ✅ 实测 |
| `T.vcast`（rint） | docs/Tilelang.language/数据类型转换操作/T.vcast.md；v11 L1375/L1396/L1423 | ✅ 实测 |
| `T.reduce`（dims=[1]，clear=True） | docs/Tilelang.language/规约操作/T.reduce.md §2.2.2/§2.3；v11 L1380/L1409 | ✅ 实测 |
| `T.arange`/`T.vcmp`/`T.vand`/`T.vselect`/`T.vbrc`/`T.vexp`/`T.vtanh`/`T.vlog2`/`T.vmax`/`T.vsub`/`T.vmul`/`T.vadd`/`T.vdiv` | docs/Tilelang.language/ 对应子目录（数学操作/、比较操作/、条件操作/、逻辑操作/、创建操作/、shape操作/）逐一存在性核验；E1-E7 + v11 生产使用 | ✅ 实测 |
| `T.sync_block_set/wait`（n-block id 复用 + TASKDONE） | docs/Tilelang.language/同步管道操作/T.sync_block_set.md §1（同一 block 内）；v11 L1292/L1298 同款 id 复用 | ✅ 实测（v11nt 生产） |
| `T.Kernel(24, is_npu=True) as (kernel_id, subid)` | E1-E7 生产（persistent 24）；T.sync_block_set.md §2.4 示例同款 (cid, subid) 形态 | ✅ 实测 |
| `T.rs("PIPE_*")` 通道标注 | mixcv/deepseek 先例；v11 | ✅ 实测 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测 5 项）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源为三维 grid） | E1：一维 persistent 24 核 + 任务解码（bx/by/bz 派生） |
| L0C 容量 128KB | **Yes**（l0_c [bm, block_share] f32） | 单缓冲 [64, 256] f32 = 64KB ≤ 128KB ✓（实测上限：PL-1.9-hardlimits「cc overflow」BishengIR 报错反推 128KB；bm 上限公式 bm·max(bn,dim)·4 ≤ 128KB → bm ≤ 128@bn=256——设计默认 64 带宽裕） |
| L1 容量 512KB / UB 192KB | **Yes** | L1 = l1_a 32KB + l1_b 64KB = 96KB ✓；UB = 177.2KB per AIV（§4.5 逐项）✓（两者实测上限：PL-1.9-hardlimits / E1-E7 谱系） |
| GEMM 非整除 / 小尺寸分形 | **Yes**（尾块 tail < 16/32、S_kv % bn ≠ 0） | E7：tmc/tnc 分形下限钳位（16/32）+ 尾带 OOB 掩码（trace-time 门控）+ 行截断写出 |
| store_fixpipe / L0C 出数 dst 仅 GM | **Yes**（S/O 跨引擎） | E2：ws 三族 GM 物化（store_fixpipe.md §2.1 L17–L18 强制条款；L0C→UB 直连不存在） |

#### 3.5.2 参考实现差异说明（影响 API 选型的关键差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | 三维 T.Kernel(m, heads, batch) + threads | 一维 persistent 24 + 任务解码 | E1 |
| GEMM API | T.gemm(transpose_B, policy=FullRow) fragment | T.gemm(L1, L1, L0C, initC, b_transpose, size) | §3.2；FullRow 舍弃 |
| 内存分配 | alloc_shared/fragment 自动 | alloc_L1/L0C/ub 显式分层 + GM ws | Expert 模式 |
| 流水 | T.Pipelined（cp.async） | 两相位手动调度 + per-n-block flag | E2 |
| 尾块处理 | 谓词逐元素装载/回写 | size-form + 尾钳位切片 + 掩码 | E7 |
| lse 写出 | T.copy 3D | [B,H,S,1] 视图 + 自然 2D 区域写 + host reshape | E6（VP-D6） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/flash_attention/flash_attn_npuir.py` | **极高（结构原型）** | 两相位大循环、per-n-block flag、单 L0C 复用、尾钳位 size-form、scales 回放、自适应行半分、ws 三族布局 |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py` | **极高（tuned 母本）** | 同结构 + 本算子契约（BSHD/GQA/lse/softcap）+ Q-hoist + ell clear=True + transpose-free lse + TUNED 分派表模式 |
| `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py` | 高 | Expert persistent 双 Scope、运行时 if、向量 mask、pass_configs 双关闭 |
| `examples/deepseek_v32/sparse_mla_fwd_exp.py` | 中 | online softmax + lse Expert 语义、行向量 reduce 形态 |
| `examples/mixcv/mixcv_mixkernel.py` | 中 | `T.rs` + sync_block_set/wait 最小形态 |
| `examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md` | 证据 | D1–D6 工具链实测事实（本设计全项承接） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| q | (batch, seq_len_q, heads, dim) | float16 / bfloat16 | BSHD；工厂期全静态 |
| k | (batch, seq_len_kv, heads_kv, dim) | same_as(q) | GQA：`heads % heads_kv == 0` |
| v | (batch, seq_len_kv, heads_kv, dim) | same_as(q) | |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| output | (batch, seq_len_q, heads, dim) | same_as(q) | kernel 内以 [B,H,S,1] lse 视图 + host reshape 输出契约形状 |
| lse | (batch, heads, seq_len_q) | float32 | prim_func 声明 [B,H,S,1] 视图（E6） |

### 4.3 中间缓冲区（Expert 显式分层；设计默认 bm=64, bn_eff=256, dim=128, NK_max=8 基准）

**GM workspace（每核单槽，闭包内 `torch.empty` 分配；fp16 路径）**：

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| ws_s | [24, 64, 2048]（pad_kv = NK_max·bn_eff） | float16 | GM/L2 | Cube→Vector 的 S 块（f32 L0C → f16 隐式转换；bf16 路径 f32） |
| ws_p | [24, 64, 2048] | float16/bfloat16（输入 dtype） | GM/L2 | Vector→Cube 的 P 块（已量化到输入 dtype，M10） |
| ws_o | [24, 64, 1024]（dim·NK_max = 128·8） | float16 | GM/L2 | Cube→Vector 的每块 PV 部分和（bf16 路径 f32，M9） |

fp16 路径总量 = 24 × 64 × (2048+2048+1024) × 2B = **15.7MB**（S_kv=2048 基准）；bf16 路径 = 24 × 64 × (2048×4 + 2048×2 + 1024×4) = **25.2MB**。L2 驻留可行（L2 192MB 口径，KV 工作集 33.5–67MB + ws ≤ 25.2MB « 上限；fabric 聚合 ≥1.73TB/s 实测未触顶，PL-1.9-hardlimits）。

**L1（Cube 侧，生命周期复用——v11 block_share 形态）**：

| Buffer 名 | Shape | dtype | 大小 | 用途 |
|-----------|-------|-------|------|------|
| l1_a | [64, 256]（block_share = max(bn_eff, dim)） | dtype | 32KB | pass-1: Q（Q-hoist，[tmc, dim] 区域）；pass-2: P（[tmc, tnc] 区域） |
| l1_b | [256, 128] | dtype | 64KB | pass-1: K；pass-2: V（[tnc, dim] 区域） |
| **L1 合计** | | | **96KB ≤ 512KB** ✓ | |

**L0C（单缓冲顺序复用）**：

| Buffer 名 | Shape | dtype | 大小 | 用途 |
|-----------|-------|-------|------|------|
| l0_c | [64, 256] | float32 | **64KB ≤ 128KB** ✓ | gemm1 出数（[tmc, tnc] 区域）→ FIX 写出后 gemm2 覆写（[tmc, dim] 区域）——两相位程序序下生命周期不重叠（pass-1 全部完成才进 pass-2），单缓冲合法（D2：v11 同款） |

**UB（Vector 侧，per AIV；dim=128 基准，dim=64 时 f16_D/f32_D/acc_o 减半）**：见 §4.5 预算表。

### 4.4 内存搬运路径

```
GM[q] ──T.copy(slice, Q-hoist)──> L1[l1_a] ─┐
GM[k] ──T.copy(slice)──> L1[l1_b] ──────────┤ gemm1（PIPE_C）
                                    L0C[l0_c] ──T.copy(PIPE_FIX, f32→f16)──> GM[ws_s]
GM[ws_s] ──T.copy(PIPE_MTE2)──> UB[ub_f16_N] ──vcast──> UB[ub_f32_N] ──softcap→mask→prescale→
   reduce(max)→vmax×2→vsub/vexp(α→scales[])→vsub/vexp(P)→reduce(sum)→vcast──> UB[ub_f16_N]
UB[ub_f16_N] ──T.copy(PIPE_MTE3)──> GM[ws_p] ──T.copy(PIPE_MTE2)──> L1[l1_a] ─┐
GM[v] ──T.copy(slice)──> L1[l1_b] ────────────────────────────────────────────┤ gemm2（PIPE_C）
                                    L0C[l0_c] ──T.copy(PIPE_FIX, f32→f16)──> GM[ws_o]
GM[ws_o] ──T.copy(PIPE_MTE2)──> UB[ub_f16_D] ──vcast──> UB[ub_f32_D] ──vmul(α)+vadd──> UB[acc_o]
UB[acc_o] ──vdiv/vcast──> UB[ub_f16_D] ──T.copy(行截断切片)──> GM[output]
UB[ub_ell] ──vlog2/vmul/vadd──> UB[ub_lse] ──T.copy(自然 2D 区域)──> GM[lse [B,H,S,1] 视图]
```

### 4.5 UB 内存预算（设计默认 bm=64/bn_eff=256/dim=128/half=32/NK_max=8；causal-nosoftcap 主 trace；per AIV）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| ub_f16_N（S/P f16 staging，vcast 双向复用） | [32, 256] | fp16 | 16,384 |
| ub_f32_N（S f32 softmax 链） | [32, 256] | fp32 | 32,768 |
| ub_f16_D（O partial f16 staging / 输出 cast staging 复用） | [32, 128] | fp16 | 8,192 |
| ub_f32_D（O partial f32） | [32, 128] | fp32 | 16,384 |
| acc_o | [32, 128] | fp32 | 16,384 |
| ub_m / ub_mprev / ub_mcur / ub_alpha / ub_ell / ub_ellcur / ub_t / ub_zero / ub_lse / ub_lse_tmp | [32, 1] ×10 | fp32 | 1,280 |
| scales | [NK_max·32, 1] = [256, 1] | fp32 | 1,024 |
| ub_colmat / ub_rowmat / ub_diff | [32, 256] ×3 | int16 | 49,152 |
| ub_neg（−inf 哨兵阵） | [32, 256] | fp32 | 32,768 |
| ub_cond（+ 尾带变体复用为 vand 输出） | [32, 256] | bool | 8,192 |
| **总计** | | | **182,528 ≈ 178.2KB / 192KB（196,608）** ✓（余量 ~18KB） |

**口径说明**：① mask 链缓冲（colmat/rowmat/diff/neg/cond，合计 137KB）仅 causal trace 分配——non-causal trace 剔除（VP-P2 变体分派，§1.6.0 R4 引用的 clean 特化先例），non-causal 总量 ≈ 92KB；② causal-softcap trace 增 fp32 scratch ub_sc [32,256] 32KB → 210KB **超限**——softcap trace 采用 **ub_neg 复用为 scratch**（softcap 链先于 mask：`ub_sc = vmul(S, c1) → vtanh(ub_sc, ub_sc) → S = vmul(ub_sc, softcap)`，随后 `vbrc(−inf → ub_neg)` 重建哨兵再 vselect——每块 +1 个 vbrc，正确性无损），总量维持 178.2KB ✓；③ bf16 trace：ub_f16_N/ub_f16_D 变 bf16 同宽，ws 侧 dtype 差异不进 UB——总量不变；④ dim=64（smoke）：f16_D/f32_D/acc_o 合计 40,960 → 20,480，总量 ≈ 161.7KB ✓；⑤ E1-E7 先例 ~105–121KB、highperf 单 AIV ~161KB 分配先例佐证容量口径（design_v2 §4.5 引证）——本预算 178KB 在先例范围内。

### 4.6 动态轴定义

**无运行时动态轴**（全静态 shape，工厂参数确定；NK/K_A/tmc/tnc 为 kernel 内 kernel_id/task_id 派生 PrimExpr——E1-E7 D3 实证可 lower）。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[-2, -1],
    target="npuir",
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
        tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    },
)
```

（输出居末两位；两个 Expert pass_configs 防
编译器重排手动两相位流与显式 workspace——v11nt / highperf L13–19 先例；`testing/npuir/compiler_hint_ops/test_out_idx.py` L75–107 `out_idx=[-2,-1]` 先例。）

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Cube GEMM × 2 + Vector softmax/归约/累加链）——MixCV（同一 kernel 含 Cube 侧 T.gemm 与 Vector 侧多个 v-prefix 算子，AGENTS.md Developer-mode MixCV 判据同款结构，Expert 形态）

**判定依据**: QK^T 与 PV 为 MAC 密集段（Cube），softmax 链/掩码/累加/epilogue 为逐元素与规约段（Vector）；两相位结构以 per-n-block flag 协同双引擎。

### 5.2 Block 划分

```python
bm = 64        # 设计默认。依据：① UB 预算（§4.5：bm=64 时 causal trace 178KB ✓；bm=96 需 mask 缓冲精简形态，列 Stage 4 候选）；
               # ② 与 wrapper 默认 block_m 一致（分派语义最简）；③ L0C [bm,256] f32 = 64KB 余量大（上限 bm=128）
bn_eff = 256   # 设计默认（caller 传 wrapper 默认时替换）。依据：① 向量发射开销定律（PL-1.9-hardlimits：每 op ~0.5µs 固定成本，
               # 宽块 op 恒优于窄块 op 翻倍）；② v11nt tuned 锚点 (96,256)；③ flag 预算 NK_max=8 ≤ 15；
               # 强制守卫：bn_eff = max(bn_caller, ceil16(ceildiv(S_kv, 15)))——保证 NK_max ≤ 15、总 flag id ≤ 16（E6）
NK_max = ceildiv(seq_len_kv, bn_eff)   # 8（S_kv=2048）/ 2（S_kv=512）
num_q_blocks = ceildiv(seq_len_q, bm)  # 8（S=512）/ 32（S=2048）
# GEMM1 tile：M=tmc(≤bm, ≥16)、N=tnc(≤bn_eff, ≥32)、K=dim(64/128 ≥32)——QK^T（b_transpose=True）
# GEMM2 tile：M=tmc、N=dim、K=tnc(≥32)——PV（initC=True 每块覆写）
```

**目标域 config 核验表**（设计默认经 E6 公式后的有效值）：

| workload | S_q=S_kv | bn_eff | NK_max | flag ids | 尾块 | 尾带 |
|---|---|---|---|---|---|---|
| smoke (1,512,8,64) | 512 | 256 | 2 | 3 | 无（512%64=0） | 无（512%256=0） |
| 8b/70b-short (·,512,·,128) | 512 | 256 | 2 | 3 | 无 | 无 |
| 8b/70b-long (·,2048,·,128) | 2048 | 256 | 8 | 9 | 无 | 无 |

（manifest 5 workload 全部整除、零尾块零尾带——尾块机制仅服务契约测试域。）

### 5.3 约束分析

- **对齐约束**: bn_eff 与 dim 为 16 倍数（bn_eff 公式 ceil16 保证；dim ∈ {64,128}）；fp16/bf16 尾轴 256·2B = 512B ≥ 32B ✓；分形 M/N ≥ 16、K ≥ 32 由 E7 钳位保证。
- **L1 容量**: l1_a + l1_b = 96KB ≤ 512KB ✓（实测上限，PL-1.9-hardlimits）。
- **L0C 容量**: l0_c [64, 256] f32 = 64KB ≤ 128KB ✓（实测上限，PL-1.9-hardlimits）。
- **UB 容量**: 178.2KB ≤ 192KB per AIV ✓（§4.5）。
- **flag 预算**: NK_max + 1 ≤ 16 ✓（T.set_flag.md §2.1 event_id 0–15；bn_eff 守卫公式强制）。

### 5.4 注意事项

- **非整除 / 小尾块**（契约域）：tmc/tnc 分形下限钳位（M/N≥16、K≥32）+ has_band trace-time 门控（尾带 OOB 掩码）+ 行截断写出（E7 全套，§6.4）；manifest 域零触发。
- **causal 对角块**：每任务约 1 块走 mask 链（K_A 两段式，S_q=S_kv 时 NK − K_A = 1）——mask 链 5 个向量 op/块的对角开销为结构性常数。
- **causal 负载均衡**：NK 随 bx 线性递增——轮转任务分配（cid 步长 24）均衡（§5.5）。
- **bm 上界**：L0C 公式 bm·max(bn_eff,dim)·4 ≤ 128KB → bm ≤ 128@bn=256；UB 公式（causal trace）约 (0.7·bm + 137)·1024 ≤ 192KB → **bm ≤ 76**——UB 先绑定（bm=64 取安全值；bm=96 需 §1.6.3 讨论的 mask 缓冲精简形态，Stage 4 候选）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。

- **物理核数（要素②，实查）**: **24**。查询代码与返回值记录：

```python
from tilelang.utils import NPUUtils
print(NPUUtils.get().get_aicore_num())   # 2026-09-15 03:06 实查输出：24（Ascend910B2C）
```

（混合算子直接使用返回值；与 2026-09-07 E1-E7 谱系双源记录〔NPUUtils 实查 + `torch.npu.get_device_properties(0).cube_core_num == 24`〕一致。）

- **逻辑核数（要素①）**: `num_logical = ceildiv(S_q, bm) × heads × batch`（bm=64 设计默认）：

| workload | q_shape [B,S,H,D] | num_q_blocks | num_logical | 与 24 核关系 |
|---|---|---|---|---|
| smoke | [1,512,8,64] | 8 | 8×8×1 = **64** | 2.67×（轻度欠载边缘） |
| 8b-short | [4,512,32,128] | 8 | 8×32×4 = **1024** | 42.7×（饱和） |
| 8b-long | [2,2048,32,128] | 32 | 32×32×2 = **2048** | 85.3×（饱和） |
| 70b-short | [2,512,64,128] | 8 | 8×64×2 = **1024** | 42.7×（饱和） |
| 70b-long | [1,2048,64,128] | 32 | 32×64×1 = **2048** | 85.3×（饱和） |

- **规模判定（要素③）**: **极大规模**（4/5 workload 逻辑任务数 1024–2048 » 24，无法通过调整分块缩减到核数量级——bm 上调受 UB/L0C 双上限约束〔§5.4〕且上调减少任务数与均衡性冲突）；smoke 64 任务为轻度欠载（64/24 = 2.67 任务/核，核心全忙）。
- **分核方案**: **固定启动内核数 = 物理核数 24，核内 `T.serial` 串行处理多个逻辑任务**：`T.Kernel(24, is_npu=True)`，每核 `num_local_tasks = T.ceildiv(num_logical − kernel_id, 24)` 个任务、任务解码 `cid = task_id·24 + kernel_id`（轮转——causal NK 随 bx 递增时每核 bx 均匀覆盖；E1-E7 生产同款，官方模板形态）。循环边界为 kernel_id 派生的 PrimExpr 表达式（进入循环前一次计算、循环内不变——E1-E7 D3 实证可 lower；开发指南 §3.3 模板同形的 kernel_id 派生边界）。**smoke 欠载处置**：64 任务下 24 核全忙（16 核 3 任务 / 8 核 2 任务，不均衡比 1.5×）——Stage 4 可按 shape 分派 bm=32（128 任务，5.33/核）或依赖两相位单任务流水性吸收，不在 Stage 1 特化。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务维（per 核） | persistent 串行 | `for task_id in T.serial(num_local_tasks)` | 极大规模分核（§5.5）；runtime PrimExpr 边界（kernel_id 派生，D3 实证） |
| KV pass-1 维（Cube） | 串行大循环 | `for i in T.serial(NK)` | 两相位全 S（无等待直跑）；NK 为任务派生 PrimExpr |
| KV pass-2 维（Cube） | 串行大循环 | `for i in T.serial(NK)` | 两相位全 PV（逐块等 P） |
| KV pass-1/pass-2 维（Vector） | 串行大循环 ×2 | `for i in T.serial(NK)` | softmax 链 / 延迟累加 |
| 元素级 | **向量化（无循环）** | v-prefix 链 + `T.reduce` | §1.6.2 全覆盖（15 项 ✅）；无逐元素标量循环 |

### 6.2 循环伪代码

```python
with T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid):
    # 每核一次：colmat/rowmat/neg 构建（causal trace）；vbrc 状态初值准备
    with T.Scope("Cube"):
        for task_id in T.serial(num_local_tasks):
            # cid → (bx, by, bz)；NK/K_A/tmc/tnc 计算（E1/E3/E7 公式）
            if task_id > 0:
                with T.rs("PIPE_MTE2"): T.sync_block_wait(FLAG_TASKDONE)   # 单槽跨任务 WAR
            T.copy(q[...], l1_a[...])                                       # Q-hoist
            for i in T.serial(NK):   # pass-1（§0.6 E2 Cube 流）
                ...
            for i in T.serial(NK):   # pass-2（§0.6 E2 Cube 流）
                ...
    with T.Scope("Vector"):
        for task_id in T.serial(num_local_tasks):
            # 同款任务解码；real_m 自适应半分
            for i in T.serial(NK):   # pass-1（§0.6 E2 Vector 流）
                ...
            for i in T.serial(NK):   # pass-2
                ...
            # epilogue + set FLAG_TASKDONE（§0.6 E2 Vector 流）
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**（Expert 手动两相位，§0.5）。流水机制 = 两相位结构自带的跨引擎重叠：① Cube pass-1 的 MTE2（K 装载）/MTE1（喂 L0）/C（gemm）/FIX（写 ws_s）四通道硬件排队自然重叠（无软件等待插入）；② Cube pass-2(i) 与 Vector pass-1(j>i) 经 per-n-block flag 重叠推进；③ Vector pass-1 的 MTE2（读 ws_s）/V（softmax 链）/MTE3（写 ws_p）三通道重叠。**槽位安全性论证**（对照 E1-E7 §6.3 的教训——本轮结构更简）：单槽 + n-block id 严格交替 + TASKDONE 边界排空（E2 论证④）——每个 flag id 任一时刻至多一个未决事件，无 WAR 窗口；任务内 ws 生命段：pass-1 写 ws_s[i] → Vector 读 → Vector 写 ws_p[i] → Cube 读 → Cube 写 ws_o[i] → Vector 读——全部由同一 id 的三次握手定序，槽位复用安全由程序序传递保证。

### 6.4 尾块处理

**manifest 域零触发**（S%bm=0、S_kv%bn_eff=0）；契约域全套机制（E7）：

1. **q 尾块**：`tail_m_real = min(bm, S_q−s_lo)`；`tmc = max(16, min(bm, ceil16(tail_m_real)))`——gemm 计算宽 tmc（垃圾行 [tail_m_real, tmc) 为 l1_a 残留），Vector 自适应半分只覆盖真实行（`real_m0 = (tail_m_real+1)//2`），输出行截断写出——垃圾行全链路不可达。数值例：S_q=520/bm=64 → 尾块 tail_m_real=8 → tmc=16 ✓（≥16 分形下限）。
2. **kv 尾块无带域**（`tn_real ≥ 32 且 16 | tn_real`）：`tnc = tn_real`，gemm 直接用真实宽，ws 读写同宽——无残留。
3. **kv 尾带域**（`tn_real < 32 或 16 ∤ tn_real`，has_band trace-time 门控）：`tnc = max(32, min(bn_eff, ceil16(tn_real)))`；Vector pass-1 处理宽 tnc，对 `j ≥ tn_real` 列以 OOB 掩码（`vcmp(colmat, tn_real, "lt")` + `vand`）强制 P=0——gemm2 K 带中 `P带(0) × V残留(任意) = 0`，污染精确为零（残留值含 NaN 亦被值无关 vselect 隔离）。数值例：S_kv=520/bn_eff=64 → 尾块 tn_real=8 → tnc=32，带 [8,32)；S_kv=100/bn_eff=64 → tn_real=36 → tnc=48，带 [36,48)。
4. **全 OOB kv 块**：NK 的 `T.min(·, S_kv)` 截断排除（E1 等价性论证）。
5. **GQA / S_q≠S_kv**：kv_head = by//groups 切片寻址；causal_offset 进 NK/K_A/K_blk 公式（E1/E3）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 手动同步（Expert）——per-n-block `T.sync_block_set/wait` + `T.rs("PIPE_*")` 通道标注；无 T.Pipelined / 自动同步依赖。

### 7.2 同步点说明（flag id 分配表 + 逐点理由）

**flag id 分配**（设计默认 NK_max ≤ 15；**总 id 数 = NK_max + 1 ≤ 16 = 事件预算 0–15**——T.set_flag.md §2.1 event_id 表；bn_eff 守卫公式强制 §5.2）：

| flag 族 | id | 方向 | 设置点 | 等待点 | 语义 |
|---------|---|------|--------|--------|------|
| n-block 三次握手 | i ∈ [0, NK_max) | Cube↔Vector（三次转向） | ① Cube pass-1 FIX 段 set(i)〔S 就绪〕② Vector pass-1 MTE3 段 set(i)〔P 就绪〕③ Cube pass-2 FIX 段 set(i)〔O 就绪〕 | ① Vector pass-1 MTE2 段 wait(i) ② Cube pass-2 MTE2 段 wait(i) ③ Vector pass-2 MTE2 段 wait(i) | 每 id 严格三次握手、per-id 交替（任一时刻 ≤1 未决事件）——v11 L1292/L1298/L1339/L1405 同款复用 |
| FLAG_TASKDONE | NK_max | Vector→Cube | Vector epilogue 写出后 set（双 AIV 各 set 同一 id） | Cube 下一任务 Q-hoist 前 wait（`if task_id > 0`） | 单槽跨任务 WAR 保护（Vector 全部消费完成才允许覆写） |

**逐点设计依据**：

| 位置 | 同步 API | 理由 |
|------|----------|------|
| Cube pass-1 FIX 后 `set(i)` | `T.sync_block_set`（PIPE_FIX 段内） | 通知 Vector S(i) 可读（L0C→GM copy 完成） |
| Vector pass-1 MTE2 前 `wait(i)` | `T.sync_block_wait`（PIPE_MTE2 段内） | 等 S 数据落 GM |
| Vector pass-1 MTE3 后 `set(i)` | `T.sync_block_set`（PIPE_MTE3 段内） | 通知 Cube P(i) 可读（含「S 已消费」传递语义——Vector 程序序读 S 先于写 P） |
| Cube pass-2 MTE2 前 `wait(i)` | `T.sync_block_wait`（PIPE_MTE2 段内） | 等 P 数据落 GM |
| Cube pass-2 FIX 后 `set(i)` | `T.sync_block_set`（PIPE_FIX 段内） | 通知 Vector O(i) 可读（含「P 已消费」传递语义） |
| Vector pass-2 MTE2 前 `wait(i)` | `T.sync_block_wait`（PIPE_MTE2 段内） | 等 O 部分和落 GM |
| Vector epilogue 后 `set(FLAG_TASKDONE)` | `T.sync_block_set` | 任务排空握手 |
| Cube 下一任务 Q-hoist 前 `wait(FLAG_TASKDONE)` | `T.sync_block_wait`（`if task_id > 0`） | 防新任务 pass-1 覆写未消费 ws |

**2-AIV 聚合语义（R-1 风险项）**：Vector 段代码在每核两个 AIV（subid 0/1）上各执行一份，二者写各自半区的 P 后 set 同一 flag id——Cube 单次 wait 的聚合语义（「两 AIV 均完成」才放行）依 **E1-E7（29/29 测试 + 11 轮调优生产）与 v11nt（fa 域达标终版）双先例**成立（两先例的 FLAG_P/FLAG_S 均为双 AIV 同 id set + Cube 单 wait 形态）。若 Stage 3 探针证伪（L0 首验覆盖跨任务槽位复现场景），回退 = FLAG_TASKDONE 按 subid 拆双 id（TASKDONE_0/1）+ Cube 双 wait 串联（id 总数 NK_max+2 ≤ 16 仍合规）。

**跨核同步**：无（任务独立、q/k/v 只读共享、output/lse 切片不相交——E1 论证）；本仓同步原语均为核内语义（T.set_flag.md §1「同一个Cube/Vector核内的不同通道之间」/ T.sync_block_set.md §1「同一 block 中的其他执行单元」——适用范围条款亲自核对）。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}
```

（防编译器重排手动两相位流水与显式 workspace——v11nt / highperf L13–19 先例；CUDA 的 FAST_MATH/compile_flags 不迁移，§0.5。）

### 7.4 CV 融合设计（MixCV）说明

同一 kernel 内 `T.Scope("Cube")`（两相位 GEMM 大循环）与 `T.Scope("Vector")`（softmax/累加/epilogue）顺序书写、硬件上并行执行于同一 AI Core 的双引擎，经 per-n-block flag 协同——MixCV 结构（AGENTS.md 判据同款）。tilelang-mixcv-skill 的核心关注点（sync_block_set/wait、Scope 分工、PIPE_FIX/MTE2/MTE3 通道）在本设计 §0.6 E2/§7.2 全量落地；Stage 3 实现时可加载 mixcv skill 复核同步细节。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现/测试基准），**不得复刻 §0.6 的两相位 NPU 算法**——保证验证独立性。以下为谱系验证过的 golden 草案（23/23 + 29/29 两轮实证，直接承接）：

```python
def golden_gqa_prefill_fwd(q, k, v, is_causal, sm_scale=None, softcap=0.0):
    """GQA/MHA prefill forward reference (batched materialize softmax).

    q:    [batch, seq_len_q, heads, dim]        fp16/bf16 (BSHD)
    k/v:  [batch, seq_len_kv, heads_kv, dim]    fp16/bf16 (BSHD)
    Returns (output: q shape/dtype, lse: [batch, heads, seq_len_q] fp32).
    """
    import math
    batch, seq_len_q, heads, dim = q.shape
    seq_len_kv, heads_kv = k.shape[1], k.shape[2]
    scale = dim ** -0.5 if sm_scale is None else sm_scale

    q_ = q.transpose(1, 2).float()  # [B, H, L, D] fp32
    k_ = k.transpose(1, 2).float()
    v_ = v.transpose(1, 2).float()

    if is_causal:  # right-aligned causal (source semantics, 0.1)
        kv_pos = torch.arange(seq_len_kv, device=q.device)[None, :]
        q_pos = torch.arange(seq_len_q, device=q.device)[:, None]
        valid = q_pos + (seq_len_kv - seq_len_q) >= kv_pos

    output = torch.empty(batch, seq_len_q, heads, dim, dtype=q.dtype)
    lse = torch.empty(batch, heads, seq_len_q, dtype=torch.float32)

    for b in range(batch):
        for h in range(heads):
            h_kv = h // (heads // heads_kv)
            scores = q_[b, h] @ k_[b, h_kv].transpose(-2, -1)  # [L, S] fp32
            scores = scores * scale
            if softcap > 0.0:
                scores = softcap * torch.tanh(scores / softcap)
            if is_causal:
                scores = scores.masked_fill(~valid, float("-inf"))

            lse[b, h] = torch.logsumexp(scores, dim=-1) / math.log(2.0)  # log2 域（D7 校准）
            p = torch.softmax(scores, dim=-1)
            o = (p.to(q.dtype) @ v_[b, h_kv].to(q.dtype)).to(q.dtype)     # P 量化到输入 dtype（源语义）
            output[b, :, h, :] = o

    return output, lse
```

（SDPA 交叉验证：`F.scaled_dot_product_attention`（is_causal）在 L==S 时与本 golden 等价——源仓测试 `MhaFwdTest.ref_program` 模式，L0 中作为第二对照。）

### 8.2 精度标准与 L0 门槛测试计划

**精度门（D6/D8 校准的双门 + tier-3 分类，谱系冻结值直接承接）**：

| dtype | 输出 atol/rtol（tier-1） | tier-2（2-ulp 回退） | lse atol/rtol |
|-------|--------------------------|---------------------|----------------|
| float16 | 5e-3 / 1e-5 | +2·2^-10 相对余量 | 1e-3 / 1e-3 |
| bfloat16 | 同上 | +2·2^-7 相对余量 | 同上 |

tier-3 噪声分类（P 量化放大域：率 ≤ 0.01%、包络 2·flip_rel·v_absmax、lse 门不受扰）按谱系口径保留（debug_log D6：真实缺陷〔丢失 mask 79.8% mismatch〕仍响亮失败）。

**分层测试策略**（回归入口 = `python _gqa_prefill_fwd_kernel.py --level all`；L0/L1 阻塞、L2/Boundary 告警不阻塞——与现驱动同构，用例面继承并标注新增覆盖点）：

| 层 | 阻塞 | 用例（继承现驱动 + 本设计新增标注） | 覆盖目标 |
|---|---|---|---|
| L0 | ✅ | L0-1 smoke causal fp16（seed=0）；L0-2 smoke causal bf16（**R-2 两相位 bf16 探针前置首验**）；L0-3/L0-4 契约校验（接口 shape/dtype/lse 域 + SDPA 交叉） | 门禁最小集 |
| L1 | ✅ | manifest 全 sweep：8b/70b short/long × fp16/bf16 × causal = 8 例（标签 L1-8b-short 等） | 目标域全量 |
| L2 | ⚠️ | noncausal-s1024（B=2,H=16,S=1024——**bn_eff 守卫触发域：S_kv=1024 → bn_eff=80**）；gqa-h8-hkv2；lne-s causal/noncausal（S_q=128≠S_kv=256）；tiny-s16（**tmc/tnc 钳位域 + 尾带 [16,32)**）；kv32（NK=1 单块域）；gap100 四例（**尾带 [36,48) 域 × causal/noncausal × fp16/bf16**） | 契约覆盖域 + E7 尾带机制全部触发点 |
| Boundary | ⚠️ | tail520 causal/noncausal（**tmc 钳位 [8,16) + 尾带 [8,32) 组合**）；softcap30 fp16/bf16（**softcap trace：ub_neg 复用 scratch 形态**）；zeros（bit-exact + lse==log2(512)=9.0 精确）；gqa-tail520；sm_scale=0.3 | 边界 + 病态 |

**L0 门槛计划**（Stage 3 首验顺序）：① L0-1（fp16 smoke）→ 验证两相位主 trace；② L0-2（bf16 smoke）→ R-2 探针（bf16 直连 + f32 ws）；③ 契约校验（shape/dtype/lse 域/参数校验 raise 行为）。任一失败按 §9.2 风险表的回退路径处置后重验。

---

## 9. 风险点与注意事项

### 9.1 已知约束（技术约束检测结论汇总）

1. **一维 Kernel**（三维 grid 禁止）→ E1 persistent 任务解码。
2. **L0C(cc) = 128KB / L1(cbuf) = 512KB 单端口 / UB = 192KB per AIV**（三项实测上限，PL-1.9-hardlimits）→ §4.3/§4.5 预算内设计。
3. **L0C 出数 dst 仅 GM**（store_fixpipe.md §2.1）→ S/O 跨引擎必经 GM ws（E2）。
4. **store_fixpipe f32→f16 为正确数值转换**（probe 4.7e-4 = 1 ulp）→ M11/M12 采纳依据。
5. **活跃源 T.transpose epilogue 毒化 2.6×**（VP-D6）→ transpose-free lse（E6）。
6. **load_nd2nz 对 BSHD 跨头切片静默平铺误读**（D1，CG-2026-0004）→ 全部 GM→L1 用 slice-form T.copy。
7. **向量算子发射开销 ~0.5µs/op**（PL-1.9-hardlimits）→ bn=256 宽块主 trace；f16 链等窄化优化 defer Stage 4。
8. **flag event_id ∈ 0–15**（T.set_flag.md §2.1）→ bn_eff 守卫公式（E6）+ n-block id 严格交替（E2）。
9. **parser 不折叠 if / 条件 alloc 非法 / vbrc 标量须 let-bound / vcmp 标量拒 tir.Cast（int16 阈值）/ vtanh 须 fp32 操作数 / T.reduce 无 clear 为静默错误**（D4/D5/T.reduce.md §2.3）→ §3.2/§3.3 全项落实。
10. **`sync_block_wait` 阻塞整条发射流**（PL-1.9-hardlimits）→ 两相位结构本身就是该约束的规避（pass-1 零等待）。

### 9.2 风险清单与常见错误

| # | 风险 | 等级 | 触发场景 | 缓解 / 回退 |
|---|------|------|----------|------------|
| R-1 | 2-AIV flag 聚合语义（双 AIV set 同 id、Cube 单 wait） | 中 | 跨任务槽位复用时 TASKDONE 单 wait 可能提前放行 | E1-E7 + v11nt 双生产先例；L0-1 首验含多任务复现；回退 = TASKDONE 按 subid 拆双 id + Cube 双 wait（id 预算 NK_max+2 ≤ 16 ✓） |
| R-2 | 两相位 bf16 端到端（bf16 L1 gemm + f32 ws_s + bf16 ws_p）无先例 | 中 | L0-2 bf16 smoke | E5 保守承载（f32 ws_s/ws_o）；回退 = ws 全 f32 + 装载点 vcast（结构不变） |
| R-3 | causal PrimExpr 边界（NK/K_A/K_blk/tmc/tnc）下 lower 失败 | 低 | 首编译 | E1-E7 D3/E3 生产实证同族形态；K_blk 双 lane 端点数值例已入 E3；失败时以标量三元式（D4 合法形态）改写 |
| R-4 | UB 预算 178KB 贴近 192KB（bn_eff 上调 / dim>128 时溢出） | 中 | bn_eff > 256（S_kv > 3840 的守卫触发）或 dim=256 扩展 | §4.5 口径公式前置校验（工厂期）；溢出时 mask 缓冲精简形态（rowmat→阈值矩阵原位 + neg f16 预 vcast——§1.6.3 弃选方案的 Stage 4 降级路径）或 bn_eff 降 128（S_kv ≤ 1792 域 flag 合规） |
| R-5 | causal 对角掩码正确性（K_blk per-AIV row0 / K_A floordiv） | 高→已缓 | 任何 causal 用例 | E3 双 lane 端点数值例 + K_A floordiv v2 修正公式逐字承接 + L1 全 sweep + B-tail520-causal 守护；谱系同类错误（v0/v1 ceildiv）教训已固化为「代数重推 + 双端点代入」纪律 |
| R-6 | S/O f16 物化精度（causal 域新承载） | 低 | L1 fp16 全 sweep | v11nt fa-tuned 实证（tier-1 0 flips）+ §8.2 双门 + tier-3 分类兜底；超限时 ws_s/ws_o 升 f32（带宽代价 ~+9%，正确性优先） |
| R-7 | 双槽变体（num_stages≥2）的 flag 预算 | 低 | Stage 4 | Stage 3 基线恒单槽（num_stages 为保留旋钮）；双槽需 id-offset 方案（2·NK_max+2 ≤ 16 才合规——S_kv ≤ 1792 域）或窗口化 flag（设计已论证，未入基线） |
| R-8 | lse [B,H,S,1] 视图 + host reshape 契约 | 低 | 契约校验 | v11nt 生产同款（wrapped reshape 零拷贝）；register_fake 形状一致 |
| R-9 | 尾带机制（has_band trace）遗漏域 | 中 | L2/Boundary 尾带用例 | §6.4 数值例全覆盖（tiny-s16 [16,32) / gap100 [36,48) / tail520 [8,32)）；套件已含全部触发点 |
| R-10 | 目标域性能预期的不确定常数（fabric 聚合带宽在多头 KV 重读画像下的有效值未实测） | 中 | Stage 4 首采 | §11 预期区间已按保守/乐观双口径给出；Stage 4 Phase 1 首采 5 case 后校准；结构性下界（两相位 vs 单遍 2.31× fa 域实测）不受该常数影响 |
| R-11 | ws GM 足迹 15.7MB（fp16）/ 25.2MB（bf16）@ S_kv=2048 | 低 | 大 S_kv 扩展 | L2 192MB 口径下与 KV 工作集（≤67MB）共存无压力；S_kv > 8192 扩展时按 pad_kv 公式复核 |
| R-12 | smoke 欠载（64 任务 / 24 核，1.5× 不均衡） | 低 | smoke case | 核心全忙、两相位单任务流水吸收；Stage 4 按 shape 分派 bm=32 候选 |

### 9.3 特殊场景处理

- **非整除分块**：§6.4 尾块全套（tmc/tnc 钳位 + 尾带掩码 + 行截断）。
- **极小 shape**（S=16/32）：NK=1 单块域——两相位退化为 pass-1/pass-2 各一块（正确性同构，性能非目标）。
- **混合精度**：fp16/bf16 双 trace 同构（E5）；ws dtype 分 dtype 承载。
- **GQA**：kv_head = by//groups；heads_kv < heads 时 k/v 切片跨头共享只读。
- **sm_scale 覆盖 / softcap**：prescale/softcap 链（E3/E4 顺序）；sm_scale=0 病态路径与源 NaN 语义逐位一致（E3 论证）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/multi_head_attention/_gqa_prefill_fwd_kernel/
├── _gqa_prefill_fwd_kernel.py   # 算子实现 + 分层测试驱动（Stage 3 从本设计重生成，覆盖现 E1-E7 版）
├── DESIGN.md                     # 本设计文档
├── verify_equiv.py               # §1.6.1 采纳项等价性机器验证（torch CPU，S1-EQUIV-EXEC 门禁配套）
├── RETROSPECTIVE.md              # 任务复盘（追加）
├── debug_log.md                  # 工具链实测事实（继承，新发现追加）
└── history_version/              # 历史版本归档（design_v0/v1/v2 + preregen 快照）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（本文件，v3） | 设计文档 |
| `verify_equiv.py` | ✅ 已完成（§1.6.1 结果表联动） | 数学等价优化采纳项机器验证（torch CPU、fp64 参照、13/13 EQUIV_PASS，`python verify_equiv.py` 直接运行） |
| `_gqa_prefill_fwd_kernel.py` | ⬜ 待实现（Stage 3 重生成） | 两相位 kernel + golden + L0/L1/L2/Boundary 驱动（`--level all`） |
| TileOPs 侧 wrapper | 不改动 | `multi_head_attention.py` 契约零变化（E6 闭包兼容） |

### 10.3 命名规范

- 项目目录: `multi_head_attention`；算子目录: `_gqa_prefill_fwd_kernel`；实现文件: `_gqa_prefill_fwd_kernel.py`——均与现目录一致。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md v3）
2. ⬜ Golden 函数 + 精度门（§8.1/§8.2，继承谱系实现）
3. ⬜ 工厂层（trace 家族分派 + bn_eff 守卫 + ws 闭包）+ `_builder_2phase` causal-nosoftcap 主 trace（fp16）
4. ⬜ **L0-1 首验**（fp16 smoke）→ 通过后 L0-2（bf16，R-2 探针）→ 契约校验
5. ⬜ causal-softcap / noncausal / has_band 变体 trace + L2/Boundary 全套
6. ⬜ `--level all` 全绿 → Stage 2 检视入口

---

## 11. 性能预期与度量计划（manifest causal 多头域 · Stage 4 度量锚点）

> 本节为 Stage 4 调优提供预期锚点与度量计划（best_effort，无硬性数值目标——与 §11.4 第四轮框架一致；本轮为 Stage 1 结构重设计，预期基于结构证据外推，全部待实测校准）。

**结构证据锚点**（同机同口径 msprof op Task Duration）：
- fa 域两相位 vs 单遍链（同 workload 族、调优终值对比）：**2.31×**（fa4096: 98.05 vs 226.72µs；fa2048: 30.88 vs 69.13；fa1024: 18.11 vs 27.24；fa512: 12.94 vs 17.55）——结构性下界，不依赖目标域常数。
- 当前 causal 基线（单遍链 (64,64,1) full 路径）：reg8bshort **3764.05µs**（≈2.25 TFLOPS）/ reg8blong **13265.27µs**（≈5.17 TFLOPS）。
- fa4096 两相位有效吞吐 **87.6 TFLOPS**（饱和域，22 任务块以上）。

**预期区间**（保守 = 仅套用 2.31× 结构下界；乐观 = 按饱和域吞吐外推——fabric 常数未实测，R-10）：

| workload | 因果 FLOPs（4·B·H·S²·D/2） | 当前基线 | 保守预期（×2.31） | 乐观预期（吞吐外推） |
|---|---|---|---|---|
| 8b-short | 8.59 GFLOP | 3764µs | ~1630µs | ~120–400µs（短 S 摊销不足，取下限） |
| 8b-long | 68.7 GFLOP | 13265µs | ~5740µs | ~790–1700µs（fabric 1.7–2.6TB/s 敏感） |
| 70b-short | 8.59 GFLOP | ~3768µs（同 trace） | ~1630µs | 同 8b-short |
| 70b-long | 68.7 GFLOP | ~13368µs（同 trace） | ~5790µs | 同 8b-long |
| smoke | 0.27 GFLOP | ~282µs（PL-1.7 口径） | ~122µs | ~30–80µs（轻度欠载） |

**Stage 4 度量计划**：① Phase 1 首采 5 case fp16（msprof op Task Duration，launch-count=20/warm-up=5/median，captured op = 新 kernel 的 mix_aic 名）；② 对照上表校准 fabric 有效常数（多头 KV 重读画像）；③ 调优候选优先序：bm ∈ {64, 80} × bn_eff ∈ {128, 256} 扫描（UB/L0C 公式约束内）→ f16 softmax 链（#9 defer 项）→ 双槽 id-offset 变体（R-7，S_kv ≤ 1792 域）→ M7 非 causal 小 KV 特化；④ 回归纪律：`--level all` 全绿为每轮前置，causal 域互不回退 ≤3%。

**与上一版设计的关键差异总结**（本文件 §0.5/§1.6.0 的浓缩，供检视导航）：
1. **承载结构**：单遍 per-block 跨引擎链（E1-E7）→ 两相位独立大循环 + S/P/O GM 物化 + scales 延迟回放（调研四问完整覆盖 examples/ 全目录后选定，修复上版漏检盲区）；
2. **grid**：persistent 24 核任务循环保留（E1 承接），但任务内调度换两相位（v11 母本 + causal 适配）；
3. **mask/尾块**：E3 向量 mask + K_A 两段式 + E7 尾带机制全套承接，位置平移到 pass-1，v-guard 结构性消除；
4. **lse**：transpose-free（VP-D6 绕法）——上一版 E1-E7 的 epilogue 仍带 transpose 税（§11.4 项 1 的已知负债，本轮清偿）；
5. **dtype 承载**：fp16 f16 传输（v11nt 实证）+ bf16 保守 f32（R-2 探针）；
6. **目标域适配逻辑**：5 个 manifest causal workload 全部整除（零尾块零尾带）+ 任务数 64–2048（饱和域）+ causal 对角块每任务 1 块（K_A 分界后 mask 开销为结构性常数）——结构选择以该域画像为准，case_fa 域（H=1 欠载）不再是设计驱动项。
