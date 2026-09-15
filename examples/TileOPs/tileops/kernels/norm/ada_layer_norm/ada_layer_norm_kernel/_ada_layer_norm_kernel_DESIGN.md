# _ada_layer_norm_kernel 算子设计文档

> 迁移任务（harness 模式）：AdaLayerNormFwdOp（has_gate=False 路径）。
> 源算子（GPU 原始源码）：`/home/tilelang/zuochuanuong/TileOPs-fork/tileops/kernels/norm/ada_layer_norm.py`；
> Stage 0 提取件（本函数迁移的精确源）：`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/_ada_layer_norm_fwd_kernels.py`。
> 工具链版本戳：tilelang 0.1.2+a83118285a（2026-09-10 build）+ Ascend910B2C。

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

**数学语义**（AdaLayerNorm 前向，无 gate 变体）：对 `(M, N)` 输入 `x` 的每一行（最后一维 N）做 layer_norm 后施加逐元素仿射调制：

$$
\text{mean}_i = \frac{1}{N}\sum_{j<N} x_{ij}, \qquad
\text{var}_i = \frac{1}{N}\sum_{j<N}(x_{ij}-\text{mean}_i)^2, \qquad
\text{rstd}_i = \frac{1}{\sqrt{\text{var}_i + \varepsilon}}
$$
$$
y_{ij} = \text{scale}_{ij} \cdot (x_{ij}-\text{mean}_i)\cdot \text{rstd}_i + \text{shift}_{ij}
$$

`scale`/`shift` 是与 `x` 同 shape 同 dtype 的逐 token 调制张量（由调用方从条件信号预计算），**不是** layer_norm 的 weight/bias 向量。方差为有偏方差（ddof=0）；`eps` 加在方差上（默认 1e-5）。

**规约语义**：规约轴 = 最后一维（dim=-1），行间完全独立、无跨行依赖；源码中累加在 **fp32** 域进行（`x_f32` fragment 上 `T.reduce_sum`），先求 S1 得 mean，再对 `(x−mean)²` 求 S2 得方差（中心化两遍式）。

**dtype 语义**：中间计算全程 fp32——x/scale/shift 逐元素 cast 到 fp32 后参与运算，输出单个舍入回输入 dtype（`y[...] = value` 或经 `x_local`（dtype）赋值隐式截断）。即「fp32 opmath + 单次舍回」，与 golden `F.layer_norm(x.float(), ...) → scale.float()*...+shift.float() → .to(x.dtype)` 一致。

**边界语义**：
- `M % block_m ≠ 0`：行尾用谓词 `pid_m*block_m + i < M` 掩码，越界行不读不写；
- N 非 256 对齐：`N_padded = align_up(N, 256)`，越界列零填充参与规约，方差中显式减去 pad 贡献（`(S2 − pad_count·mean²)/N`），输出只写真实列；
- NaN/Inf：自然传播（无特殊处理）——全 inf 行 → mean=inf → 输出 NaN；
- eps=1e-5 默认（manifest params）；M≥1、N≥1。

**I/O 契约**：输入 `x/scale/shift` 均 `(M, N)`、dtype ∈ {float32, float16, bfloat16}、同 dtype；输出 `y` 与 `x` 同 shape 同 dtype（manifest shape_rules: `output.shape == x.shape`），非 in-place。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `x` | 输入 | `(M, N)` | float16 / bfloat16 / float32 | M=非规约维乘积（Op 层把任意前导维 reshape 成 2D），N=hidden dim |
| `scale` | 输入 | `(M, N)` | same_as(x) | 逐 token 缩放 |
| `shift` | 输入 | `(M, N)` | same_as(x) | 逐 token 平移 |
| `_dummy` | 输入 | `(1,)` | same_as(x) | 占位张量：使 gated/非 gated 变体的输出都落在 index 4（`out_idx=[4]` 一致） |
| `y` | 输出 | `(M, N)` | same_as(x) | 输出（`out_idx=[4]` 自动分配） |

输出 shape 从源码确认：`main` 的 `y: T.Tensor[(M, N), dtype]` + `register_fake` 返回 `torch.empty((M, N), ...)`——**无转置、无布局变化**（回填标注依据：源码 L204-213、L258-260）。

### 0.3 实现算法解读（怎么算）

**源码结构**（GPU，工厂模式）：`_ada_layer_norm_kernel(M, N, eps, dtype, has_gate, use_cp_async)` → `N_padded = align_up(N, 256)`，`needs_pad = N_padded != N` → `@tilelang.jit(out_idx=[4])` 内 `_func(block_m, threads)` → `@T.prim_func main(x, scale, shift, _dummy, y)` → `kernel_body`（`@T.macro` 组织）。

**计算步骤分解**（源码每个计算语句均已归入；以 has_gate=False 为准）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| L1 | 装载 x：padded 路径 `load_x_padded`（T.Parallel + if_then_else + T.And 谓词掩码，越界列填 0）；aligned 路径 `load_x_aligned`（T.copy GM→SMEM→fragment） | GM x | shared_buf (bm, N_padded)、x_local（aligned 路径） | 输入搬运 |
| L2 | fp32 上cast：`x_f32[i,j] = T.cast(shared_buf/x_local[i,j], "float32")` | dtype 副本 | x_f32 | fp32 opmath |
| S1 | 行和：`T.reduce_sum(x_f32, acc, dim=1)`；均值 `mean_val[i] = acc[i] / float(N)`（除以真实 N；pad 列为 0 不贡献） | x_f32 | acc、mean_val | mean |
| S2 | 中心化平方（in-place 覆写）：`x_f32[i,j] = (x_f32[i,j]−mean)²`（pad 列变为 mean²）；行和 `T.reduce_sum(x_f32, acc, dim=1)`；`rstd[i] = T.rsqrt((acc[i] − pad_count·mean²)/N + eps)` | x_f32、mean_val | acc、rstd | var（pad 校正中心化两遍式） |
| E1 | modulation 预取（仅 needs_pad 且 use_cp_async）：`prefetch_modulation` 用 `T.ptx_cp_async` 谓词预取 scale/shift 到 shared（4 字节事务对齐），`T.ptx_wait_group(0) + T.sync_threads()` 后消费 | GM scale/shift | scale_shared/shift_shared | 装载重叠 |
| E2 | 仿射 epilogue：`value = scale32·(x32−mean)·rstd + shift32`（padded 路径逐元素从 GM/shared 读 scale/shift + 谓词限定写出；aligned 路径 staged 副本 + `T.copy` 写回） | x 副本、mean、rstd、scale、shift | y | y 公式 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[x] --T.copy/谓词load--> SMEM[shared_buf (bm,N_padded)] --T.copy--> FRAG[x_local](aligned)
  --T.cast--> FRAG[x_f32 (bm,N_padded) fp32]
  --T.reduce_sum--> FRAG[acc] --> FRAG[mean_val]
  --(in-place 平方)--> x_f32 --T.reduce_sum--> acc --> FRAG[rstd](T.rsqrt)
GM[scale] --cp.async 谓词预取 或 T.copy--> SMEM --> (epilogue 逐元素读)
GM[shift] --同上-->
FRAG[x_local/shared_buf] --epilogue 标量复合式+隐式截断--> SMEM/FRAG --> T.copy --> GM[y]
```

**循环与并行结构**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)` 一维 grid（每 block 处理 block_m 行）；block 内 `T.Parallel(bm, N_padded)` 元素级并行（线程映射到行内元素，行内归约经线程级归约塌缩）；GPU 默认配置 `block_m=1`（`select_row_config`：单行/CTA 保证行归约均匀性，跨线程 AllReduce 塌缩结构上不可能）+ `threads=128`（实测 CUDA 最优）。

**host 侧逻辑**：`AdaLayerNormKernel.__init__` 计算 `N_padded`、`use_cp_async = _should_use_cp_async(N, dtype, has_gate)`（48KB SMEM 预算 + 4 字节行宽判定）、构建 kernel（`lru_cache` 工厂按 (M,N,eps,dtype,flags) 特化）；`forward` 经 `torch.library.custom_op("top::ada_layer_norm_fwd")` wrapper 调 `_ada_layer_norm_kernel(...)(block_m, threads)(x, scale, shift, dummy)`；`register_fake` 返回 `torch.empty((M, N))`。无数据内容预处理（纯 shape 元数据）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 行块 SMEM/fragment 驻留 | 消 GM 重读（x 跨三遍复用） | (bm, N_padded) 块驻留 shared + fragment | shared memory / 大寄存器堆 | 硬件强相关 |
| 2 | 256 元素对齐 padding（N_padded） | T.copy 向量化整块搬运 | align_up(N,256) 零填充 | CUDA SMEM 拷贝对齐约束 | 硬件强相关 |
| 3 | 中心化两遍方差 + pad 校正 | 数值稳定（vs 矩式 E[x²]−E[x]²） | (S2 − pad·mean²)/N 恒等式 | 无（纯算法层） | 可移植 |
| 4 | fp32 fragment 全程计算 | 精度（fp32 opmath + 单次舍回） | cast 一次、算完截断 | fragment 寄存器 | 硬件强相关 |
| 5 | cp.async 谓词预取 modulation | 统计计算与 scale/shift 装载重叠 | T.ptx_cp_async + commit/wait_group | cp.async 异步拷贝引擎（CUDA 专属） | 硬件强相关 |
| 6 | 融合 epilogue | 消中间写回 | 单 T.Parallel 复合表达式 + cast 折叠 | 无（模式可移植） | 模式可移植 |
| 7 | 谓词掩码边界 load/store | 越界安全（M 尾行 / N 尾列） | if_then_else + T.And | 无（模式可移植，NPU 有更优 slice 形态） | 模式可移植 |
| 8 | block_m=1 + threads=128 配置 | 行内归约均匀性 + 实测最优线程数 | 单行/CTA 结构性规避 AllReduce 塌缩 | CTA/warp 组织 | 硬件强相关 |
| 9 | lru_cache 工厂 + per-shape JIT 特化 | 摊编译开销 | functools.lru_cache(maxsize=32) | 无（host 层） | 可移植 |
| 10 | T.copy GM↔SMEM↔fragment 分级搬运 | coalesced 带宽 | 整块拷贝 | memory coalescing | 硬件强相关 |
| 11 | out_idx=[4] + _dummy 占位 | gated/非 gated 签名对齐 | 第 4 参数恒为输出 | 无 | 可移植（接口契约） |

> 识别不出的优化会被静默丢弃。Stage 0 观察项（threads 移除、use_cp_async 签名保留、lru_cache 丢失由 Kernel 类实例缓存承接、Kernel 类与 Zero 变体共享）已并入上表与 §0.5 处置。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据：migration-analysis.md §5 三层模型 + §5.3 映射表；`examples/` 佐证；`docs/` 条目。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（公式/规约轴/ddof/eps 位置） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| 中心化两遍方差算法 | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis §5.3「online 算法/分块累积 → 保留」同类）；§1.6.0 调研确认稳定性优于矩式（D-1 机器证据） |
| 行块驻留（SMEM/fragment → UB） | 优化 | shared memory | 有（UB 192KB/AIV） | **等价替换** | `T.alloc_shared`（Developer 映射 UB，T.alloc_shared.md）；预算按 192KB/1.7 膨胀裕度重算（§4.5，TRAP-UB-multibuffer-inflation） |
| fp32 全程计算 | 优化 | fragment 寄存器 | 有（fp32 UB + vcast） | **等价替换** | `T.vcast(rint)` 升 fp32 → fp32 v-prefix 链 → `T.vcast(rint)` 单次舍回（TRAP-fp16-opmath-golden 对齐通解；examples/lerp_tensor 同款） |
| 融合 epilogue | 优化 | 无 | 有 | **等价替换** | v-op 链（vmul×2 + vadd，行广播），意图（消中间写回）保留 |
| 分级搬运 T.copy | 优化 | coalescing | 有 | **等价替换** | GM↔UB 双显式 slice `T.copy`（T.copy.md §2.4；TRAP-T-copy-region-semantics 推荐形态） |
| 256 对齐 padding + pad 方差校正 | 优化 | SMEM 拷贝对齐 | 无此约束 | **舍弃** | NPU UB 对齐由编译器管理（32B，T.alloc_shared.md）；slice 精确 N 直接消零填充（logsumexp 设计同款结论「NPU 无 256 对齐要求」）；pad 校正恒等式并入 §1.6.1 O4（机器验证）。优化意图（向量化整块搬运）由 UB 自然对齐 + slice copy 承接 |
| cp.async 谓词预取 | 优化 | cp.async 引擎 | 无（CUDA 专属 API） | **舍弃** | `T.ptx_cp_async` 在 npuir 不存在；重叠意图由编译器 auto multi-buffer 承接——PL-1.6 实测：≥16M 元素档全部向量 pass 隐藏于 MTE2 窗口内（CONST-copy-floor-method）；Stage 4 可再评估显式双缓冲 |
| threads 参数 / block×thread 两级并行 | 优化 | CUDA 线程 | 无 | **舍弃 + 重新设计** | TRAP-threads-kwarg-noop（`T.Kernel(threads=)` 在 npuir 无效果，src/ir.cc NPU 分支无 threadIdx 绑定）；并行结构 → §0.6 重设计项 R1 |
| block_m=1 结构性默认 | 优化 | CTA 行均匀性 | 担忧不存在 | **重新设计** | NPU reduce 按行硬件归约（T.reduce_sum dim=1），无跨线程 AllReduce 塌缩问题；bm 改由 UB 预算表驱动（§0.6 R2） |
| 谓词掩码边界（if_then_else + T.And） | 优化 | 无 | 有更优形态 | **重新设计** | `T.min` + src/dst 双显式 slice + 垃圾行丢弃（TRAP-T-copy-region-semantics；lerp/logsumexp 尾块模式）→ §0.6 R3 |
| lru_cache 工厂 + JIT 特化 | host | 无 | — | **保留** | host 层硬件无关；Stage 0 已确认提取件由 Kernel 类实例缓存 `self.kernel` 承接 |
| out_idx=[4] + _dummy | 接口 | 无 | — | **保留** | 接口契约（§0.2）；NPU wrapper 已按此移植（`ada_layer_norm.py` L107-114） |
| use_cp_async 形状策略（_should_use_cp_async） | host | CUDA SMEM 预算 | 无意义 | **舍弃**（参数保留） | 工厂签名保留 `use_cp_async=False`（接口不变），NPU 实现忽略该标志（统一路径覆盖对齐/非对齐 N）；wrapper 继续计算该值无副作用 |

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（grid 超发 → 1D persistent 分核）**

- **源方案**：`T.Kernel(ceildiv(M, block_m), threads=threads)`——逻辑 block 数随 M 超发（如 M=2048, bm=1 → 2048 个 block），依赖 CUDA 运行时调度；threads 折叠 CUDA 线程并行。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True)`，`num_kernels = min(ceil(M/block_m), 48)`（48 = 24 AICore × 2，纯 Vector 算子翻倍，§5.5 实查）；核内 `for i in T.serial(num_local_tasks)` grid-stride 任务映射 `block_id = i*num_kernels + cid`，`if block_id < num_logical` 守卫尾任务；所有量为 host 层 Python 编译期常量（PL-1.5 折叠先例）。意图承接：消除超发 block 的串行调度开销（core-split-strategy.md §1 极大规模分支）+ grid-stride 聚集窗口访问对 MTE2 更优（PL-1.6 实测 +3.2%）。
- **语义保持论证**：行间独立（规约轴=行内 N），任务映射是行块集合的双射，每行公式与累加顺序不变；尾任务守卫保证不越界。参照 `examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py`（同款 3 输入 1 输出流量形态的已调优 persistent 形态）。

**重设计项 R2: block_m 语义（结构性 1 → UB 预算表驱动）**

- **源方案**：`block_m=1`（每 CTA 单行，规避跨线程 AllReduce 塌缩）+ threads=128。
- **NPU 新算法**：`_func(block_m=默认表值)`，默认 `bm = min(8, max(1, UB_SAFE_BYTES // (bytes_per_elem_per_row · N)))`（fp16/bf16 10 B/元素/行、fp32 8 B/元素/行；UB_SAFE_BYTES = 192KB/1.7 ≈ 115KB，TRAP-UB-multibuffer-inflation 裕度），工厂层加 UB 预算 guard（lerp E1 模式）。行均匀性担忧在 NPU 不存在（硬件按行归约）。
- **语义保持论证**：bm 只改变每任务行数，不改变行内计算；M % bm ≠ 0 的尾块由 R3 处理。NPU wrapper 默认 `block_m=1` 仍有效（功能等价，性能由 Stage 4 扫描定优，§5.4）。

**重设计项 R3: 边界处理（谓词掩码 + 零填充 → T.min + 双显式 slice + 垃圾行丢弃）**

- **源方案**：padded 路径 `T.if_then_else(T.And(row<M, j<N), x[...], 0)` 掩码装载 + pad 方差校正 + 谓词限定写出；aligned 路径整块 T.copy。
- **NPU 新算法**：统一单路径（无 needs_pad 分支）：`real_m = T.min(block_m, M - off_m)`；装载 `T.copy(x[off:off+real_m, 0:N], stage[0:real_m, 0:N])`（src/dst 双显式 slice，dst 零起点——TRAP-UB-dst-align）；v-op 全 buffer 操作（含未初始化垃圾行，规约/广播按行隔离）；写出 `T.copy(stage[0:real_m, 0:N], y[off:off+real_m, 0:N])` 只写有效行。
- **语义保持论证**：垃圾行（real_m..bm）的统计量是垃圾但**按行隔离**（reduce dim=1 逐行、v-op 逐元素/行广播，无跨行混合），输出切片只取有效行（lerp「v-ops on full block_size buffer, only [0:tail] copied out」同款、logsumexp 设计 §5.4 同款）；N 不做任何 padding，方差直接对真实 N 归约（pad 校正恒等式见 §1.6.1 O4，机器验证 0 违反）。

**重设计项 R4: dtype 路径（fragment 隐式 cast 链 → 显式 vcast(rint) fp32 中转链）**

- **源方案**：`T.Parallel` 内 `T.cast(..., "float32")` 上cast + epilogue 赋值隐式截断回 dtype。
- **NPU 新算法**：fp16/bf16 输入：`T.vcast(stage, x_f32, round_mode="rint")` 上cast（f16→f32 / bf16→f32 rint，无损）→ fp32 v-prefix 链 → `T.vcast(x_f32, stage, round_mode="rint")` 单次舍回（f32→f16 / f32→bf16 rint = RNE，与 torch `.to()` 一致）；fp32 输入：无 cast（同 dtype 直拷直算）。
- **语义保持论证**：fp16/bf16→fp32 精确无损（fp16/bf16 是 fp32 子集，rint 无操作用）；fp32→目标 dtype 单次 RNE 舍回 = golden 的 `y.to(x.dtype)` 舍入路径；TRAP-fp16-opmath-golden 实证该「fp32 中转链」与 torch CPU golden（fp32 opmath + 单次舍回）差 ≤1 ulp fp16；bf16 无 Vector 算术（vadd/vmul/vsub/reduce_sum 的 bf16 列均为 ×，见 T.vmul.md/T.vadd.md/T.vsub.md/T.reduce_sum.md dtype 表）——fp32 中转是**必需**而非可选。跨 dtype `T.copy` 的隐藏 VCast 不用作 cast 融合（TRAP-C12：不减 vector pass 且舍入不可控）。

**舍弃项的意图承接汇总**（不得静默丢弃，migration-analysis §6.4）：
- padding（§0.4 #2）意图=向量化整块搬运 → UB 自然 32B 管理 + slice copy 承接；
- cp.async（#5）意图=装载与计算重叠 → 编译器 auto multi-buffer 承接（PL-1.6 实测向量链全隐藏于 MTE2 窗口），Stage 4 复评显式双缓冲；
- threads（#8 部分）意图=块内并行度 → Vector lane 向量化（§1.6.2/§1.6.3）承接。

### 0.7 标杆实现

- 源算子：`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/_ada_layer_norm_fwd_kernels.py`（提取件，迁移精确源）；GPU 原始：`/home/tilelang/zuochuanuong/TileOPs-fork/tileops/kernels/norm/ada_layer_norm.py`。
- 参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_ada_layer_norm.py` `AdaLayerNormTest.ref_program`（`F.layer_norm(x.float(), (n,), weight=None, bias=None, eps)` → `scale.float()*normed + shift.float()` → `.to(x.dtype)`）——golden 以 §0.1 语义为依据移植此实现，不复刻 NPU 算法。
- manifest：`examples/TileOPs/tileops/manifest/norm.yaml` `AdaLayerNormFwdOp` 条目（workloads + roofline 公式 flops=5MN / bytes=4MN·elem_bytes）。
- 性能基准：`examples/TileOPs/benchmarks/ops/bench_ada_layer_norm.py`（8 组参数，manifest 驱动 + torch 复合基线对照）。

---

## 1. 概述

### 1.1 算子名称

`_ada_layer_norm_kernel`（AdaLayerNormFwdOp，has_gate=False）

### 1.2 功能描述

对 `(M, N)` 张量按最后一维做 layer_norm，再施加逐 token 的 scale/shift 仿射调制（NPU 重设计后算法，与源算法的差异见 §0.6/§1.6.0：中心化两遍方差保留、pad 机制舍弃、persistent 分核、fp32 中转链）。

### 1.3 数学公式

$$
y_{ij} = \text{scale}_{ij}\cdot\frac{x_{ij}-\mu_i}{\sqrt{\tfrac{1}{N}\sum_{j}(x_{ij}-\mu_i)^2+\varepsilon}} + \text{shift}_{ij},\qquad \mu_i=\frac{1}{N}\sum_j x_{ij}
$$

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

与 §1.6.0 调研选定算法一致（中心化两遍驻留式），每任务处理 block_m 行、整行驻留 UB：

1. **装载+上cast**：x 的 (real_m, N) 块 GM→UB（dtype）→ `vcast(rint)` → `x_f32`（fp32 输入直拷）；
2. **均值**：`reduce_sum(dim=1)` → S1；`mean = S1 · fl(1/N)`（O1 倒数乘）；
3. **中心化（保留）**：`x_f32 := x_f32 − mean`（行广播，得 d，epilogue 复用——O3）；
4. **方差**：`sq := d·d`；`reduce_sum` → S2；`var = S2·fl(1/N)`；`var += eps`；`rstd = vrsqrt(var)`；
5. **归一化**：`x_f32 := d · rstd`（行广播）；
6. **仿射**：scale 块 GM→UB→`vcast`→fp32 → `x_f32 ·= scale`；shift 同款 → `x_f32 += shift`；
7. **舍回+写出**：`vcast(rint)` 回 dtype → GM y（双显式 slice，只写有效行）。

与源算法差异及来源：无 pad 机制（§0.6 R3 / §1.6.1 O4）；persistent 分核（§0.6 R1）；倒数乘与中心化复用（§1.6.1 O1/O3）；fp32 中转链（§0.6 R4）。

### 1.5 数据流图

```
GM[x] --T.copy(slice)--> UB[stage(dtype)] --vcast(rint)--> UB[x_f32]
  --reduce_sum(dim=1)--> UB[s1(bm,1)] --vmul(N_inv)--> UB[mean]
  --vsub(行广播)--> UB[x_f32 = d] --vmul--> UB[sq_f32]
  --reduce_sum(dim=1)--> UB[s2] --vmul(N_inv)--> UB[var] --vadd(eps)--> --vrsqrt--> UB[rstd]
  --vmul(行广播)--> UB[x_f32 = d·rstd]
GM[scale] --copy(slice)--> UB[stage] --vcast--> UB[sq_f32] --vmul--> UB[x_f32]
GM[shift] --copy(slice)--> UB[stage] --vcast--> UB[sq_f32] --vadd--> UB[x_f32]
  --vcast(rint)--> UB[stage(dtype)] --T.copy(slice)--> GM[y]
（fp32 dtype：全部 vcast 消失，GM↔UB 同 dtype 直拷；buffer 复用：stage 4 角色、sq_f32 3 角色）
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（per-row LN + 逐 token 仿射）的算法族。源算法（中心化两遍 + SMEM 驻留 + pad 校正）只是基线候选之一。调研深度：**完整调研**（规约/统计类）。信息源：algorithm-candidates.md `ALG-layernorm`/`ALG-rmsnorm`/`ALG-reduction` 命中行 + 本仓 examples（`examples/norm/layer_norm.py`、`examples/norm/ada_layer_norm_and_zero.py`、logsumexp 案例）+ pattern-library（PL-1.5/PL-1.6、TRAP-fp16-opmath-golden/TRAP-C12）+ 源码 §0.3/§0.4 + 结构判据。无互联网检索（本地源全覆盖四问所需候选，无未命中缺口）。

**R1 等价化简公式候选**：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3？ |
|---|------|------------|-----------|---------|----------|
| 0 | 基线（GPU 源算法） | 两遍中心化 + pad 校正 + SMEM 驻留 + epilogue 从原始副本重推 (x−m) | — | — | ✅（基线） |
| 1 | 精确 N 两遍（去 pad） | 同基线数学，(bm, N) 精确缓冲 | 数学恒等（pad 校正恒等式，§1.6.1 O4 论证） | 去 pad 记账；小 N 大均值时比基线 pad 校正更稳（verify_equiv N=2 案例：基线 fp64err 1.6e-4 vs 候选 1.9e-7） | ✅（并入主选） |
| 2 | 单遍矩式 | var = E[x²] − (E[x])²（`examples/norm/ada_layer_norm_and_zero.py` 现用） | **否**——大均值小方差行 fp32 catastrophic cancellation（D-1 机器证据：fp64err 高达 1.6e3，单案例违反率至 99.6%） | 驻留域内仅省统计相约 2N 向量操作（GM 流量不变） | ❌（数值稳定性否决） |
| 3 | Welford 在线 | 逐 tile 增量 mean/M2 合并 | 数学等价（容差内） | 驻留域内无 GM 收益；合并链逐 tile 标量序列依赖 | ❌（R2 详述） |
| 4 | N-tile 流式两遍 | block_n<N 分块 + clear=False 累加 + 输出遍重读 x | 等价 | 任意 N 可扩展（UB 无关） | ❌（域内 x 重读 1 遍，+25% GM 流量） |
| 5 | rsqrt 替代 1/sqrt | 源已用 `T.rsqrt` → NPU `T.vrsqrt` | 恒等 | 已具备（源算法既有） | n/a（保留） |
| 6 | 除法转乘倒数（O1） | S/N → S·fl(1/N)（N 编译期常量） | 容差内（fp32 ≤1-2 ulp；fp16/bf16 舍回后逐位一致——D-1） | (bm,1) 上 vdiv→vmul（op 型替换；PL-1.5「乘编译期常数倒数」先例） | ✅（§1.6.1 采纳） |
| 7 | 中心化值复用（O3，CSE） | epilogue 复用 d=x−mean，免重cast重减 | 恒等（同值复用） | 每任务 −1 vcast −1 vsub（(bm,N) 级） | ✅（§1.6.1 采纳） |

**R2 在线算法**：**有**在线变体——Welford 增量统计与分块 running 统计（algorithm-candidates.md `ALG-layernorm` 行：「Welford 增量 mean/var；分块 running 统计」；本仓 `examples/norm/ada_layer_norm_and_zero.py` 的 per-tile `clear=False` 累加即分块 running 形态的现存证据）。**但不采纳**，结构依据：在线变体的收益口径是「扫描遍数（两遍→单遍）/ 中间缓冲 O(N)→O(tile)」，而本算子工作负载域（manifest N ∈ {1152, 4096}，测试 N ∈ {514, 1152, 3000, 4096}）内整行可驻留 UB（fp16 全工作集 10B·bm·N：N=1152@bm=8 → 90KB、N=4096@bm=2 → 80KB，均 ≤ 113KB 预算）——驻留式 x 的 GM 扫描已是 1 遍、中间缓冲已是 O(行)，在线化无任何 GM 收益；Welford 的逐 tile 合并链（每 tile 一次 reduce + 标量合并序列）反引入 Vector 不亲和的标量依赖链。仅当 N 超出驻留上限（fp16 约 N>11520，§4.5）时流式结构才有意义（超出本任务域，§9.3 备注）。

**R3 复杂度对比**（每行 N 元素、eb=元素字节；「驻留」= 整行驻留 UB；FLOPs 口径=统计相+epilogue 的向量操作量纲，无超越函数；GM Bytes 口径=输入读+输出写+中间 GM 往返（本算子无中间往返项）；扫描遍数=同一张量的 GM 重复读取次数）：

| 算法候选 | FLOPs（每行） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线两遍驻留（=主选） | ≈5N（统计 3N + epilogue 2N；与 manifest roofline flops=5MN 一致） | **4MN·eb**（读 x/scale/shift + 写 y） | **1** | bm×N×10B（fp16，UB） | 行独立；无跨核同步 |
| 矩式单遍驻留 | ≈3N 统计 + 2N epilogue | 4MN·eb | 1 | 同上 | 同上 |
| Welford 在线驻留 | ≈4N + 逐 tile 合并链 | 4MN·eb | 1 | 同上 + running 统计量 | 同上 |
| N-tile 流式两遍 | ≈5N + 每 tile 归约开销 | **5MN·eb**（x 读 2 遍：统计遍 + 输出遍） | **2** | 8B·bm·bn（更小，N 无上限） | 同上 |
| N-tile 流式矩式（现有 example 形态） | ≈3N | 5MN·eb | 2 | 最小 | 同上 |

prefill 2048×4096 fp16 参照：驻留式 67.1MB vs 流式 83.9MB——**+25% 流量**（带宽受限算子上是主导项；PL-1.6：该流量规模下计算链已被 MTE2 完全隐藏，流量差即时间差）。

**R4 硬件亲和性评估**（检查清单逐项；负向淘汰证据：docs 路径+条款 / pattern-library 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 两遍驻留（主选） | Vector：reduce_sum + v-prefix 逐元素（T.reduce_sum.md / T.vmul.md），无 MAC 段 ✓ | bm·N×10B ≤ 113KB（192KB/1.7 裕度，CONST-capacity-910B2C + TRAP-UB-multibuffer-inflation）✓ | N 尾轴连续 stride=1；fp16 向量宽 ×8：1152/4096/3000 整除 ✓，514 尾 2 元素浪费 1.17% | M/N/bm 编译期常量；serial 边界静态（PL-1.5 折叠）✓ | 顺序依赖链（reduce→center→reduce→epilogue）；装载与计算 overlap 交由 auto multi-buffer（PL-1.6 实测隐藏）✓ | ✅ 主选 |
| 矩式驻留 | 同上 ✓ | 同上 | 同上 | 同上 | 同上 | ❌ 数值否决（R1#2，D-1 证据） |
| Welford 在线 | 合并链逐 tile 标量序列依赖，Vector 失分 | 同上 | 同上 | 同上 | 依赖链长，融合性差 | ❌ |
| N-tile 流式 | Vector ✓ | 更松（N 无上限） | 同上 | block_n 切片 ✓ | ✓ | ❌ 域内流量 +25%（R3）；仅 N>驻留上限时备用 |

**调研结论**：选定**「中心化两遍方差 + 整行 UB 驻留 + 精确 N」算法族**（候选 #0+#1 融合，即源算法的数学内核 + NPU 去 pad 实现）。关键依据：R3 表中驻留式 GM 流量 4MN·eb 为全候选最小（流式 +25%）；R4 全项亲和；矩式/Welford 在驻留域内无流量收益且各有数值/向量化否决项。与基线的结构差异一句话：数学内核不变，去掉 GPU 对齐 padding 机制、并行组织从 grid 超发改为 persistent 分核、dtype 路径改为显式 fp32 中转链。源算法优化手段意图承接见 §0.6 末尾汇总。「无更优替代」不适用（有采纳项 O1/O3/O4）；调研范围：algorithm-candidates.md ALG-layernorm/ALG-rmsnorm/ALG-reduction 命中行、examples/norm/{layer_norm,ada_layer_norm_and_zero}.py、logsumexp 案例、pattern-library PL-1.5/PL-1.6/TRAP-fp16-opmath-golden/TRAP-C12、源码 §0.3/§0.4、结构判据（R2）。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；代表 workload = llama-3.1-8b-prefill 2048×4096 fp16）**：

- **估算下界: ≈53–67 µs**（流量项 67.1MB ÷ 1.0–1.26 TB/s——3:1 R/W 混合地板 1.26TB/s〔CONST-mte2-degradation〕与 64M 档退化曲线 20.9GB/s/core×48 并列给出区间；发射项：~15 op/task × 深流水被 MTE2 隐藏〔CONST-copy-floor-method：≥16M 元素档向量 pass 全隐藏，同款 3 输入 1 输出 lerp 实测 delta ≤1.1µs〕；容量项：0——纯 Vector 无跨引擎往返，无 CONST-store-fixpipe-gm-only 类约束）。
- 分 workload 估算（流量项 / 发射项取大者）：smoke-dit 64×1152（0.59MB）≈8–15µs（发射主导，8 task 单波）；dit-xl-2 1024×1152（9.4MB）≈10–20µs（流量 6.5µs@~1.45TB/s + 浅流水部分暴露的发射，PL-1.6 的 1M 档 mte2_ratio 0.70-0.84 区间）；decode 1×4096 bf16（32.8KB）≈10µs（单 task，15 op × ~0.5µs〔CONST-vector-launch-overhead〕+ 搬运延迟）。
- 常数引用：CONST-mte2-degradation（tilelang 0.1.2+ed787bb，2026-09-07）+ CONST-vector-launch-overhead（0.1.2+3a214cde，2026-09-09）+ CONST-copy-floor-method（0.1.2+ed787bb）+ CONST-capacity-910B2C（3a214cde / 2026-08-24）。**⚠️ stale 标注**：上述条目均被 kb_stale_check 标记 stale（当前工具链 a831182，2026-09-10 > 各条目版本戳）——数值仅用于候选排序（各候选流量同为 4MN·eb，排序不依赖常数精确值）与 Stage 4 baseline 对账基线（hardware-cost-model.md §3 要求 Stage 4 回填「设计估算 vs 实测」偏差行）；结构性结论（流式 +25% 流量、驻留可行）不依赖其精确值。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定算法（中心化两遍驻留式）。逐项四要素：

| # | 优化项 | 原式（源/GPU 语义） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 除法转乘倒数（PL-1.5「乘编译期常数倒数」先例） | `mean = S1 / N`；`var = S2 / N`（真除法） | `mean = S1 · fl(1/N)`；`var = S2 · fl(1/N)`（N_inv 工厂期 Python 常量） | N 为编译期常量；fl(1/N) 相对误差 ≤2⁻²⁴（N=2^k 时精确）；D-1 机器验证 45 案例全过：fp16/bf16 舍回后**逐位一致**（max ulp=0），fp32 ≤128 ulp（容差 1e-5 内） | (bm,1) buffer 上 2 次 vdiv → 2 次 vmul（op 型替换；buffer 极小收益有限，但零风险且与 PL-1.5 模式一致） |
| O3 | 中心化值复用（公共子表达式消除） | epilogue 从原始 x 副本重推：`value = scale32·(cast(x_local)−mean)·rstd + shift32`（重 cast + 重减） | 统计相中心化后**保留** d：`x_f32 := x_f32−mean`（驻留），epilogue `y = ((d·rstd)·scale32) + shift32` | 恒等——d 是同一子表达式的值复用；乘法结合序由 (scale·d)·rstd 变为 (d·rstd)·scale，fp32 结合序差异 ≤1 ulp（D-1 覆盖：45 案例含 extreme-mod 消catastrophic角点，候选 fp64err 不劣于基线） | 每任务 −1 次 (bm,N) vcast −1 次 (bm,N) vsub；浅流水档（smoke/dit/decode）约 −1µs/task；深流水档被 MTE2 隐藏（PL-1.6） |
| O4 | 精确 N 方差（去 pad 校正） | `var = (Σ_{j<Np}(x̃_j−m)² − pad·m²)/N`（x̃=零填充副本） | `var = (Σ_{j<N}(x_j−m)²)·fl(1/N)` | **数学恒等**：pad 列贡献 (0−m)²=m²，故 Σ_{j<Np}(x̃−m)² = Σ_{j<N}(x−m)² + pad·m²，减去 pad·m² 后实数域严格相等；fp32 实现域中基线的 pad 校正在 pad·m²≫Σ(x−m)² 时自身有 catastrophic cancellation——D-1 N=2 案例：基线 fp64err 1.58e-4 vs 候选 1.9e-7（候选**更**接近精确数学 830 倍） | 去 pad 记账（−1 标量乘 −1 标量减/行）+ 免 N_padded 缓冲加宽（(bm,N_padded)→(bm,N)，N=514 时省 49% UB）+ 小 N 大均值行数值更稳 |
| — | rsqrt | 源已用（`T.rsqrt`）→ NPU `T.vrsqrt` | 恒等（T.rsqrt.md：rsqrt = 1/x^0.5） | 已具备（源算法既有，非本次新增） | — |

**D-1 等价性机器验证结果表**（脚本 `examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py` 已执行，2026-09-10，torch CPU、fp64 参照 + 角点值；判定双准则：(a) 候选式 vs 基线式在 dtype 容差内，或 (b) 候选式对 fp64 精确数学的误差 ≤ 基线式的误差（候选不得更差））：

| 验证项 | 案例数 | fp16/bf16 舍回后 max ulp | fp32 表现 | 违反率 | 结论 |
|--------|--------|---------|---------|--------|------|
| O1+O3+O4 联合（候选式 vs 基线式） | 45（3 dtype × 15 形态） | **0（逐位一致）** | randn 类 max_abs ≤9.5e-7（≤128 ulp，容差 1e-5 内）；2 角点（randn-N2、extreme-mod）经准则 (b)：候选 fp64err ≤ 基线（1.9e-7 vs 1.6e-4；3.9e-2 vs 3.9e-2）——基线自身在这些角点退化（pad 校正消灾难 / 大中间量结合序），候选持平或更优 | **0**（准则 a 违反 0 + 准则 b 违反 0） | **EQUIV_PASS** |
| C2 矩式（拒绝证据，非采纳项） | 9 对抗案例（均值 3e3/3e4 扫描 + fp16 粗栅格 6e4） | — | — | 7/9 案例违反（fp32 大均值：fp64err 1.4e3 vs 基线 6.3e-4，违反 4096/16384；fp16 栅格 6e4：违反 13556/16384） | **验证失败（FAIL）→ 拒绝采纳**（机器证据支撑 R1#2 否决；2 例「舍入巧合通过」已在脚本输出如实标注） |

形态覆盖：randn-N{1,2,514,1152,3000,4096}、大均值小方差（fp32 域 3e4±0.5 / fp16 顶格 65472↔65504）、常量行（var=0 → rstd=rsqrt(eps)）、全零行、次正规（×1e-40）、±inf 混合行、NaN 点位、极值 scale/shift（±65504 消catastrophic角点）、M=1。

**优化结论**：采纳 O1/O3/O4 三项。优化后公式（§3.1 唯一输入）：

$$
\mu = S_1\cdot r_N,\quad d = x-\mu,\quad \sigma^{-1} = \text{rsqrt}\!\big(\textstyle\sum d^2\cdot r_N + \varepsilon\big),\quad y = ((d\cdot\sigma^{-1})\cdot\text{scale}) + \text{shift},\qquad r_N = \underline{\text{fl}(1/N)}
$$

**否决项及依据**（负向断言举证，negative-claim-evidence.md §1）：
- 矩式 var=E[x²]−E[x]²：D-1 机器证据（上表），非纸面论证；
- fp16 原生域计算链（不经 fp32 中转）：TRAP-fp16-opmath-golden（torch CPU golden fp32 opmath + 单次舍回；fp16 原生三步链差 2–3 ulp，N=2^24 违反率 ~0.125%）——fp32 中转为精度必需；
- 跨 dtype `T.copy` 作 cast 融合（如 GM fp16 → UB fp32 一步到位）：TRAP-C12（`docs/Tilelang.language/内存操作/T.copy.md` §3——Developer GM→UB lowering 为隐藏 hivm::VCastOp，「跨 dtype copy 不减少 vector pass 且舍入不可控」，lerp 2026-09-07 证伪）——上cast 数值无损故安全，但按陷阱条目纪律统一走显式 `vcast(rint)`。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态（GPU 源） | 向量替代方案（API 佐证） | 是否替代 | 不可替代理由 |
|---|--------|---------------------|--------------------------|---------|--------------|
| 1 | x 上cast fp32 | `T.Parallel(bm,Np)` 逐元素 `T.cast` | `T.vcast(stage, x_f32, round_mode="rint")`（T.vcast.md：f16→f32 rint ✓ bf16→f32 rint ✓，shape 一致） | ✅ | — |
| 2 | 行和 S1/S2 | `T.reduce_sum(x_f32, acc, dim=1)` | 同 API（T.reduce_sum.md：(M,N)→(M,1)，fp32 ✓；layer_norm.py L45 在用） | ✅ | — |
| 3 | mean/var 标量行计算 | `T.Parallel(bm)` 循环 `mean_val[i]=acc[i]/N` | `T.vmul(s1, N_inv, mean)`（(bm,1) 单个向量 op；T.vmul.md §2.2.2 标量广播 ✓） | ✅ | — |
| 4 | 中心化+平方 | `T.Parallel(bm,Np)` 标量复合式 `(x−m)·(x−m)` | `T.vsub(x_f32, mean, x_f32)`（行广播 [M,N]−[M,1]，T.vsub.md §2.2.2）+ `T.vmul(x_f32, x_f32, sq_f32)` | ✅ | — |
| 5 | var+eps / rsqrt | `T.Parallel(bm)` 标量式 `T.rsqrt(...)` | `T.vadd(var, eps, var)`（标量 ✓ T.vadd.md）+ `T.vrsqrt(var, rstd)`（T.rsqrt.md，layer_norm.py L61 在用） | ✅ | — |
| 6 | epilogue 仿射 | `T.Parallel(bm,Np)` 复合标量表达式（cast 折叠） | `T.vmul(x_f32, rstd, x_f32)`（行广播）+ scale/shift 各 [copy+vcast+]`vmul`/`vadd`（§3.2 全表） | ✅ | — |
| 7 | 下cast + 写出 | 赋值隐式截断 + `T.copy` | `T.vcast(x_f32, stage, rint)`（f32→f16 / f32→bf16 rint ✓）+ `T.copy` 双显式 slice | ✅ | — |
| 8 | 任务循环（persistent） | grid 超发（每 block 一任务，无核内循环） | `for i in T.serial(num_local_tasks)` + `if block_id < num_logical` 守卫 | ❌ 保留 | **block 级任务映射计算**：每 task 数个标量索引（block_id/off_m），非逐元素热点；serial 循环本身是 persistent 分核意图（§0.6 R1）的载体，无逐元素等价 API（lerp 同款结构） |
| 9 | 行尾 real_m | 谓词掩码（逐元素 T.And） | `real_m = T.min(block_m, M - off_m)`（标量，1 次/task） | ❌ 保留 | **动态边界元数据**：M % bm ≠ 0 的尾任务行数；per-task 1 次标量 min + slice 界，深流水档被 MTE2 隐藏（CONST-copy-floor-method：mte2_ratio≥0.95 时标量/分支开销隐藏）；静态化需为尾任务复制 kernel 体（复杂度不值） |
| 10 | N_inv / num_kernels / num_local_tasks | host Python | host Python 编译期常量（PL-1.5 折叠） | ❌（非 kernel 内） | **host 侧元数据计算**，不在 kernel 内 |

**向量化结论**：逐元素计算（cast/加减乘/rsqrt/规约/仿射）**已全部向量化**（替代 API 均有 docs+examples 佐证，上表）；保留 3 类标量/循环——任务映射索引（#8）、动态边界 min（#9）、host 元数据（#10），逐项理由如上，无「实现简单」类理由。§6 循环结构与本表一致（kernel 内无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

规约类算子必选候选：**水平归约（lane→归约轴）vs 垂直扫描（lane→独立行）**；另评估 N-tile 流式与转置布局；两遍 vs online 与 lane 映射的交互已在 §1.6.0 R2 覆盖（驻留域内在线化无收益）。

**轴质量评分**（主选轴 N；昇腾 Vector 128bit：fp16/bf16 ×8、fp32 ×4）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | fp16：N=1152（144×8 ✓）、4096（512×8 ✓）、3000（375×8 ✓）、514（64×8+2，尾 2）；fp32 ×4：514（128×4+2） |
| 尾 lane 浪费率 | N=514：fp16 (65×8−514)/514=**1.17%**；fp32 (129×4−514)/514=0.39%；其余形状 0% |
| 累加链形态 | 归约维 = 尾轴连续 stride=1（v-op 全 buffer 行广播 [M,N]↔[M,1] 为文档形态，T.vsub.md §2.2.2）；无跨步系数 |
| repack 代价 | **无**（I/O 原生行主序 = 核内布局） |
| UB 容量影响 | 全 N 驻留 → bm 受 10B·bm·N ≤ 113KB 约束（§4.5 表）；对价 = x 单遍 GM 扫描 |

**候选矩阵**：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳？ |
|---|---------|---------|------------|--------------|--------|
| 1 | I/O 原生 (M,N) 行主序 + **全 N 驻留** | N（尾轴连续；`reduce_sum dim=1` 水平归约 + 行广播 v-op） | 无 | GM 流量 4MN·eb（全候选最小）；x 扫描 1 遍；UB 预算驱动 bm（§4.5） | ✅ **主选** |
| 2 | N-tile 流式（block_n<N，`T.serial` 分块 + `clear=False` 累加；`examples/norm/ada_layer_norm_and_zero.py` 现用形态） | N（分 tile） | 无 | UB 更松（8B·bm·bn）但 **x GM 重读 1 遍 → 5MN·eb（+25% 流量，prefill +16.8MB ≈ +13µs）**；仅 N>驻留上限（fp16 约 N>11520）时必要 | ❌ 域内 strictly worse（R3 表）；留作 N 上界 fallback（§9.3） |
| 3 | 垂直扫描（lane→独立行，串行步进 N） | 行 | 无 | 归约维访问 stride=M（跨步）→ 无向量宽度；且跨步 gather 向量指令缺失（TRAP-C10，traps-compiler.md，截至 2026-08-28 build） | ❌ |
| 4 | 转置布局 (N,M)（host permute 或核内转置链） | M | host permute ~百 µs（PL-1.3 净亏）/ 核内 T.transpose 二轴交换链 ~µs（PL-1.1） | 归约轴变跨步——repack 纯开销零收益 | ❌ |

**GPU 源码隐式轴选择的独立复核**（迁移任务要求）：GPU 将 CTA 内 threads 映射到行内元素（block_m=1 + threads=128，线程级跨 N 归约）——即 GPU 也选了 N 为归约/向量轴，但那是 CUDA warp 组织的结果；本节依据 NPU Vector 宽度/广播形态/UB 容量独立评估后**结论一致、理由不同**（NPU：尾轴连续 stride=1 + [M,N]↔[M,1] 广播 + 全 N 驻留单遍扫描）。

**实验裁决模式评估**：主选 #1 vs #2 的判定裕度 = 25% GM 流量差，远超任何未实证常数的不确定区间 → 无需三件套；唯一落入未实证区间的量是 **auto-multi-buffer 膨胀系数（~1.7x，TRAP-UB-multibuffer-inflation）对本 kernel 3-buffer 形态的适用性**（决定 N=4096 fp16 时 bm=2 还是 3）——这是**参数级**（block_m 取值）而非结构级决策：主选结构对 bm∈{1,2,3} 均成立，设计期取保守值 bm=2（80KB×1.7=136KB ≤192KB 有界），bm∈{3,4} 留 Stage 4 实测扫描（pattern-library INDEX.md §3「本库模式是起点不是终点：每算子的最优 tiling 仍须实测扫描」；编译期 UB 溢出报错即天然 guard）。**不启动设计期探针（D-3）**：结构决策不依赖该常数，探针预算留给 Stage 4。

**布局决策结论**：选定 #1：核内布局 = I/O 原生 (M,N) 行主序、向量化轴 = N（尾轴）、repack = 无；该结论同步落入 §3.3 伪代码与 §6 循环结构（归约/广播的内层向量维 = N；buffer 形状 = (block_m, N)）；弃选方案的量化理由见候选矩阵。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（reduce_sum + v-prefix 逐元素），无 matmul | 无 Cube/L0/L1 参与 |
| 归约 | T.reduce_sum dim=1 | Developer/Expert 均可用（decision-tree.md §2「T.reduce_sum/max/min 在两种模式下都可使用」），无需手动控制 |
| 内存层级 | 仅 GM↔UB | `T.alloc_shared`（Developer 映射 UB，T.alloc_shared.md） |
| 同步 | 单 kernel 内顺序依赖链、无核间协作 | Developer 自动同步 |
| 先例 | logsumexp（同族 row-reduction，Developer）已交付；lerp（同款 3 输入 1 输出流量，Developer persistent）已调优；layer_norm.py（Developer）在库 | Developer 已验证可行 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB）——v-op 操作数必须在 UB（T.vmul.md §2.3.1），alloc_shared 即 UB（T.vsub.md 示例 2 用 alloc_shared 作 v-op 操作数） |
| 计算方式 | v-prefix API（vcast/vmul/vsub/vadd/vrsqrt）+ `T.reduce_sum` + `T.min` 标量边界 |
| 同步 | Developer 自动同步（无手动 sync_block_set/wait） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维） |

---

## 3. API 映射设计

### 3.1 公式拆解

输入公式 = §1.6.1 优化后公式（fp32 计算域；r_N = fl(1/N)）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_f32 = fp32(x)` | 上cast（fp32 输入为直拷） |
| 2 | `S1 = Σ_j x_f32[i,j]` | 行和 |
| 3 | `mean = S1 · r_N` | 均值（O1） |
| 4 | `d = x_f32 − mean` | 中心化（保留，O3） |
| 5 | `S2 = Σ_j d[i,j]²` | 中心化平方和 |
| 6 | `rstd = rsqrt(S2 · r_N + eps)` | （O1 + O4） |
| 7 | `t = d · rstd` | 归一化 |
| 8 | `t = t · fp32(scale)`；`t = t + fp32(shift)` | 仿射 |
| 9 | `y = dtype_round(t)` | 单次舍回（rint） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 依据 |
|------|----------|-------------|------|------|------|
| 任务映射 | `block_id = i·K + cid` | `T.serial(num_local_tasks)` + `if block_id < num_logical` | 静态边界（host 折叠） | Developer | lerp L193-197 同款 |
| 行尾 | `real_m = min(bm, M−off)` | `T.min(block_m, M - off_m)` | 标量 | Developer | flash_attn_npuir_dev.py L86 / lerp |
| 载入 x | GM→UB | `T.copy(x[off:off+real_m, 0:N], stage[0:real_m, 0:N])` | src/dst 双显式 slice，dst 零起点 | Developer | T.copy.md §2.4 示例 1 + TRAP-T-copy-region-semantics（推荐形态）+ TRAP-UB-dst-align（0 起点绕开） |
| 上cast | fp16/bf16→fp32 | `T.vcast(stage, x_f32, round_mode="rint")` | (bm,N)→(bm,N) 同 shape | Developer | T.vcast.md（f16→f32 rint ✓ bf16→f32 rint ✓）；lerp L209 |
| 行和 | S1, S2 | `T.reduce_sum(x_f32, s1, dim=1)` | (bm,N)→(bm,1)，clear=True（默认） | Developer | T.reduce_sum.md；layer_norm.py L45 |
| 均值 | `S1·r_N` | `T.vmul(s1, N_inv, mean)` | (bm,1)×标量 | Developer | T.vmul.md §2.2.2 标量广播；ada_layer_norm_and_zero.py L78 |
| 中心化 | `d = x−mean` | `T.vsub(x_f32, mean, x_f32)` | [bm,N]−[bm,1] 行广播，in-place | Developer | T.vsub.md §2.2.2 行广播 ✓ §2.3 in-place ✓ |
| 平方 | `sq = d²` | `T.vmul(x_f32, x_f32, sq_f32)` | out-of-place（保 d） | Developer | T.vmul.md |
| 方差 | `S2·r_N + eps` | `T.vmul(s2, N_inv, var)`；`T.vadd(var, eps, var)` | 标量广播 | Developer | T.vmul.md / T.vadd.md；ada_layer_norm_and_zero.py L79/L83 |
| rstd | `1/√var` | `T.vrsqrt(var, rstd)` | (bm,1) fp32 ✓ | Developer | T.rsqrt.md（T.vrsqrt）；layer_norm.py L61 |
| 归一化 | `d·rstd` | `T.vmul(x_f32, rstd, x_f32)` | 行广播 in-place | Developer | T.vmul.md §2.2.2 |
| scale 装载+乘 | `t·scale` | `T.copy(scale[...], stage[...])` + `T.vcast(stage, sq_f32, rint)` + `T.vmul(x_f32, sq_f32, x_f32)` | buffer 复用（sq_f32 第 2 角色） | Developer | 同上 |
| shift 装载+加 | `t+shift` | `T.copy(shift[...], stage[...])` + `T.vcast` + `T.vadd(x_f32, sq_f32, x_f32)` | sq_f32 第 3 角色 | Developer | 同上 |
| 舍回 | fp32→dtype | `T.vcast(x_f32, stage, round_mode="rint")` | f32→f16 / f32→bf16 rint ✓；stage 第 4 角色 | Developer | T.vcast.md；TRAP-fp16-opmath-golden 通解 |
| 写出 | UB→GM | `T.copy(stage[0:real_m, 0:N], y[off:off+real_m, 0:N])` | 双显式 slice | Developer | T.copy.md；TRAP-T-copy-region-semantics |
| fp32 路径 | — | 同上但**无 vcast**：`T.copy(x[...], x_f32[0:real_m, 0:N])` 直拷、scale/shift 拷入 `sq_f32[0:real_m, 0:N]`、`T.copy(x_f32[0:real_m, 0:N], y[...])` 直写 | 同 dtype | Developer | T.copy.md（同 dtype copy） |

### 3.3 计算伪代码

> 按模板 §3.3 注记：`T.Tensor` 的 dtype 以第二个位置参数（下标形式）传入，不用 `dtype=` 关键字。工厂接口与源一致（`use_cp_async` 保留但 NPU 忽略；`has_gate=True` 超出本任务范围，显式拒绝）；`_func(block_m)` 移除 threads（TRAP-threads-kwarg-noop；NPU wrapper 已按 `(block_m)` 调用——`ada_layer_norm.py` L107-114）。

```python
def _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False, use_cp_async=False):
    # use_cp_async: CUDA cp.async 专属 —— NPU 忽略（统一 slice 路径覆盖对齐/非对齐 N）。
    # has_gate=True (AdaLN-Zero) 不在本任务迁移范围。
    if has_gate:
        raise NotImplementedError("has_gate=True (AdaLN-Zero) is out of scope for this migration")
    if M < 1 or N < 1:
        raise ValueError(f"M and N must be positive, got M={M}, N={N}")

    N_inv = 1.0 / float(N)                      # O1: 编译期常数（标量操作数，Kernel 内 Scope 外）
    use_fp32_transit = dtype in ("float16", "bfloat16")
    # fp16/bf16: stage(2B)+x_f32(4B)+sq_f32(4B)=10B/elem/row; fp32: x_f32+sq_f32=8B
    bytes_per_elem = 10 if use_fp32_transit else 8
    block_m_default = min(8, max(1, _UB_SAFE_BYTES // (bytes_per_elem * N)))
    vector_cores = NPUUtils.get().get_aicore_num() * 2   # 实查：24 -> 48（§5.5）

    @tilelang.jit(out_idx=[4], target="npuir")
    def _func(block_m=block_m_default):
        if block_m * bytes_per_elem * N > _UB_SAFE_BYTES:
            raise ValueError(f"block_m={block_m} exceeds UB budget for N={N}, dtype={dtype}")
        # host 层编译期常量（PL-1.5：Python int 运算 trace 期折叠为 IntImm）
        num_logical = (M + block_m - 1) // block_m
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        if use_fp32_transit:
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                scale: T.Tensor[(M, N), dtype],
                shift: T.Tensor[(M, N), dtype],
                _dummy: T.Tensor[(1,), dtype],   # 保持 y 在 index 4（源契约）
                y: T.Tensor[(M, N), dtype],
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    stage = T.alloc_shared((block_m, N), dtype)       # x/scale/shift/y 中转（4 角色）
                    x_f32 = T.alloc_shared((block_m, N), "float32")   # x -> d -> 结果
                    sq_f32 = T.alloc_shared((block_m, N), "float32")  # d² -> scale32 -> shift32
                    s1    = T.alloc_shared((block_m, 1), "float32")
                    mean  = T.alloc_shared((block_m, 1), "float32")
                    s2    = T.alloc_shared((block_m, 1), "float32")
                    var   = T.alloc_shared((block_m, 1), "float32")
                    rstd  = T.alloc_shared((block_m, 1), "float32")

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)

                            # --- 1. 装载 + 上cast ---
                            T.copy(x[off_m : off_m + real_m, 0:N], stage[0:real_m, 0:N])
                            T.vcast(stage, x_f32, round_mode="rint")

                            # --- 2. 均值（O1） ---
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, N_inv, mean)

                            # --- 3. 中心化两遍方差（d 保留，O3/O4） ---
                            T.vsub(x_f32, mean, x_f32)        # d = x - mean
                            T.vmul(x_f32, x_f32, sq_f32)      # sq = d^2
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, N_inv, var)
                            T.vadd(var, eps, var)
                            T.vrsqrt(var, rstd)

                            # --- 4. epilogue: y = ((d*rstd)*scale) + shift ---
                            T.vmul(x_f32, rstd, x_f32)        # d * rstd
                            T.copy(scale[off_m : off_m + real_m, 0:N], stage[0:real_m, 0:N])
                            T.vcast(stage, sq_f32, round_mode="rint")
                            T.vmul(x_f32, sq_f32, x_f32)      # * scale
                            T.copy(shift[off_m : off_m + real_m, 0:N], stage[0:real_m, 0:N])
                            T.vcast(stage, sq_f32, round_mode="rint")
                            T.vadd(x_f32, sq_f32, x_f32)      # + shift
                            T.vcast(x_f32, stage, round_mode="rint")   # fp32 -> dtype（单次舍回）

                            # --- 5. 写出（只写有效行） ---
                            T.copy(stage[0:real_m, 0:N], y[off_m : off_m + real_m, 0:N])
        else:
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                scale: T.Tensor[(M, N), dtype],
                shift: T.Tensor[(M, N), dtype],
                _dummy: T.Tensor[(1,), dtype],
                y: T.Tensor[(M, N), dtype],
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_f32 = T.alloc_shared((block_m, N), "float32")
                    sq_f32 = T.alloc_shared((block_m, N), "float32")  # d² -> scale -> shift
                    s1 = T.alloc_shared((block_m, 1), "float32")
                    mean = T.alloc_shared((block_m, 1), "float32")
                    s2 = T.alloc_shared((block_m, 1), "float32")
                    var = T.alloc_shared((block_m, 1), "float32")
                    rstd = T.alloc_shared((block_m, 1), "float32")

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)

                            T.copy(x[off_m : off_m + real_m, 0:N], x_f32[0:real_m, 0:N])  # 同 dtype 直拷
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, N_inv, mean)
                            T.vsub(x_f32, mean, x_f32)
                            T.vmul(x_f32, x_f32, sq_f32)
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, N_inv, var)
                            T.vadd(var, eps, var)
                            T.vrsqrt(var, rstd)
                            T.vmul(x_f32, rstd, x_f32)
                            T.copy(scale[off_m : off_m + real_m, 0:N], sq_f32[0:real_m, 0:N])
                            T.vmul(x_f32, sq_f32, x_f32)
                            T.copy(shift[off_m : off_m + real_m, 0:N], sq_f32[0:real_m, 0:N])
                            T.vadd(x_f32, sq_f32, x_f32)
                            T.copy(x_f32[0:real_m, 0:N], y[off_m : off_m + real_m, 0:N])   # 直写
        return main

    return _func
```

> dtype 分支置于工厂层（trace 体之外），规避 TVM script parser if 块变量作用域限制（lerp 同款，C11 规避注记）。
> 尾块说明：v-op 作用于完整 (block_m, N) buffer，尾任务 [real_m:] 行为未初始化垃圾——按行隔离（reduce dim=1 / 行广播），输出 slice 只取有效行（lerp 官方尾块模式 + logsumexp 设计 §5.4）。
> 接口保持：`_ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False, use_cp_async=False)` 签名不变；`_func(block_m)`（threads 移除=后端适配）；`main(x, scale, shift, _dummy, y)` 参数名/顺序不变、out_idx=[4]。

### 3.4 API 可行性确认

| API | 文档路径 / 佐证 | 验证状态 | 备注 |
|-----|----------------|----------|------|
| `T.Kernel(一维, is_npu=True)` | docs/Tilelang.language/编译器提示操作/T.Kernel.md | ✅ 已核对 | NPU ≤2D block；本设计 1D；`(cid, _)` 解包 |
| `T.alloc_shared` | docs/Tilelang.language/内存操作/T.alloc_shared.md | ✅ 已核对 | Developer 映射 UB；1D~2D ✓；v-op 操作数可用（T.vsub.md 示例 2） |
| `T.copy`（slice 形态） | docs/Tilelang.language/内存操作/T.copy.md §2.1/§2.4 | ✅ 已核对 | GM↔UB ✓；尾块借 slice extents（§2.2.2 规则 2/4）；同 dtype copy |
| `T.vcast` | docs/Tilelang.language/数据类型转换操作/T.vcast.md | ✅ 已核对 | f16→f32 rint ✓；f32→f16 rint ✓；f32→bf16 rint ✓；bf16→f32 rint ✓；shape 须一致；标量 src 触发断言（本设计均为 buffer） |
| `T.reduce_sum` | docs/Tilelang.language/规约操作/T.reduce_sum.md | ✅ 已核对 | fp32 ✓ bf16 ×；(M,N)→(M,1)；clear 默认 True |
| `T.vmul` / `T.vsub` / `T.vadd` | docs/Tilelang.language/数学操作/T.vmul.md / T.vsub.md / T.vadd.md | ✅ 已核对 | fp16/fp32 ✓ bf16 ×；行广播 [M,N]↔[M,1] ✓；标量 src1（vmul/vadd/vsub）✓；in-place ✓；UB 操作数 |
| `T.vrsqrt` | docs/Tilelang.language/数学操作/T.rsqrt.md | ✅ 已核对 | 文档文件名 T.rsqrt.md、API 名 T.vrsqrt（layer_norm.py L61 实调，与 T.vln 同类 doc 命名惯例） |
| `T.min` | examples/flash_attention/flash_attn_npuir_dev.py L86 / lerp / logsumexp 设计 §3.4 | ✅ 佐证 | 尾块行数（无独立文档，三处 examples 佐证） |
| `T.serial` + if 守卫 | examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py L193-197 | ✅ 佐证 | persistent grid-stride 同款（if 体内只赋值预分配 buffer，规避 C11 作用域） |
| `@tilelang.jit(out_idx=[4], target="npuir")` | lerp（out_idx=[3]）/ logsumexp（out_idx=[1]）同款 | ✅ 佐证 | y 在 index 4 自动分配返回；源同款 out_idx=[4] |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 是否涉及 | 处理方案 |
|------|---------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(num_kernels, is_npu=True)` |
| GPU 专用 API | Yes | `T.ptx_cp_async`/`T.ptx_commit_group`/`T.ptx_wait_group`/`T.tvm_access_ptr`/`T.sync_threads`/`@T.macro`/`threads=` 全部不迁移（§0.5 舍弃/替换清单） |
| GEMM 分形/非整除 | No | 无 GEMM；M 非整除经 T.min+slice；N 无 256 对齐要求 |
| L0C 容量 | No | 纯 Vector，无 L0 |
| 物理核数适配 | Yes | §5.5 三要素（实查 24→48） |
| bf16 无 Vector 算术 | Yes | §0.6 R4 fp32 中转链（必需） |

#### 3.5.2 参考实现差异说明（GPU → NPU，汇总影响 API 选型项；完整差异见 §0.5/§0.6）

| 差异项 | GPU 源 | 本项目（Ascend） | 转换方案 |
|--------|--------|-----------------|----------|
| threads 参数 | `T.Kernel(grid, threads=)` | npuir 无效 | 移除；`_func(block_m)`（NPU wrapper 已适配） |
| `T.alloc_shared` 语义 | CUDA SMEM | UB（192KB，1.7 裕度预算） | bm 预算表（§4.5） |
| `T.alloc_fragment` 计算域 | CUDA 寄存器 | v-op 要求 UB 操作数 | 本设计全部 `alloc_shared` |
| `T.Parallel` 标量循环 | 元素级并行 | v-prefix 向量 op | §1.6.2 全表 |
| `T.if_then_else`+`T.And` 谓词 | padded masked load | `T.min`+双显式 slice+垃圾行丢弃 | §0.6 R3 |
| `T.rsqrt`（标量式） | fragment 标量 | `T.vrsqrt`（向量） | (bm,1) buffer |
| N_padded=align_up(N,256) | SMEM 拷贝对齐 | 无此约束（编译器 32B 管理） | 精确 N + pad 恒等式（O4） |
| cp.async 预取 | T.ptx_cp_async | 不存在 | 舍弃；auto multi-buffer 承接（PL-1.6） |
| `@T.macro` 嵌套 | 源码组织方式 | 直接内联展开 | §3.3 伪代码已展开 |
| 隐式 dtype 截断赋值 | `x_local[i,j]=value` | 显式 `T.vcast(rint)` + 同 dtype copy | TRAP-C12 / TRAP-fp16-opmath-golden |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/norm/ada_layer_norm_and_zero.py` | 极高（同算子已有 NPU 原型） | 同名工厂 + (M,N) 语义 + v-op 链 + alloc 缓冲；本设计改进其矩式方差（→中心化两遍，§1.6.0）与 N-tile 流式（→全 N 驻留，§1.6.3） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`（含 DESIGN.md 与实现） | 极高（同族 row-reduction 已交付迁移案例） | Developer 模式行归约结构、尾块 T.min+slice+垃圾行丢弃、fp32 计算域、bf16 fp32 中转、(bm,1) reduce 输出形态 |
| `examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 高（同款 3 输入 1 输出流量形态，已调优） | persistent min(logical,48) 分核 + grid-stride serial、vcast(rint) fp32 中转链、UB 预算 guard、同 dtype copy 纪律 |
| `examples/norm/layer_norm.py` | 高（同族算子） | reduce_sum/行广播形态、`T.vrsqrt` 实调（L61）、N_padded pad 校正（本设计舍弃该机制的依据之一） |
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm.py` | 上下文 | NPU wrapper 移植形态（`_func(block_m)` 调用、_dummy 第 4 参数、K5-K9 适配） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 输入；M、N 为工厂期常量（per-build JIT 特化） |
| `scale` | `(M, N)` | same_as(x) | 逐 token 缩放 |
| `shift` | `(M, N)` | same_as(x) | 逐 token 平移 |
| `_dummy` | `(1,)` | same_as(x) | 接口占位（不参与计算；顺带满足「kernel 须有 ≥1 输入」——TRAP-zero-input-crash） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(M, N)` | same_as(x) | `out_idx=[4]` 自动分配 |

### 4.3 中间缓冲区

**fp16/bf16 路径**（fp32 中转）：

| Buffer 名 | Shape | dtype | 存储层级 | 用途（角色复用时序） |
|-----------|-------|-------|----------|---------------------|
| `stage` | `(block_m, N)` | dtype | UB (alloc_shared) | ① x 中转 → ② scale 中转 → ③ shift 中转 → ④ 结果舍回中转（4 角色） |
| `x_f32` | `(block_m, N)` | float32 | UB | x fp32 → d（中心化，保留）→ d·rstd → 最终结果（O3 复用链） |
| `sq_f32` | `(block_m, N)` | float32 | UB | ① d² → ② scale fp32 → ③ shift fp32（3 角色） |
| `s1`/`mean`/`s2`/`var`/`rstd` | `(block_m, 1)` | float32 | UB | 行统计量（各 20·bm B，可忽略） |

**fp32 路径**（无 cast）：`x_f32` + `sq_f32`（d² → scale → shift）+ 5 个 (bm,1)；无 stage。

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L0/L1 参与）：

GM[x]    --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[x_f32]
GM[scale] --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[sq_f32]
GM[shift] --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[sq_f32]
UB[stage] --vcast(rint)--> GM[y]（经 T.copy(双slice)）
全部计算（reduce/v-op）在 UB 内闭环，无中间 GM 往返、无跨引擎传输。
```

关键约束：GM 不可直达 fragment/L0——本设计不经 fragment，全部经 UB；v-op 操作数必须在 UB（T.vmul.md §2.3.1）。

### 4.5 UB 内存预算

预算目标：总驻留 ≤ 192KB ÷ 1.7（auto-multi-buffer 膨胀裕度，TRAP-UB-multibuffer-inflation + CONST-capacity-910B2C）≈ **113KB**（`_UB_SAFE_BYTES = 115_000`）。

**fp16/bf16（10 B/元素/行 = stage 2 + x_f32 4 + sq_f32 4）**：

| N | block_m 默认 | 驻留 (B) | ×1.7 膨胀后 | ≤192KB？ |
|---|-------------|----------|------------|----------|
| 514 | 8 | 41,120（40.2KB） | 68KB | ✓ |
| 1152 | 8 | 92,160（90KB） | 153KB | ✓ |
| 3000 | 3 | 90,000（87.9KB） | 149KB | ✓ |
| 4096 | 2 | 81,920（80KB） | 136KB | ✓ |
| 8192 | 1 | 81,920（80KB） | 136KB | ✓ |
| 11520 | 1 | 112.5KB | 191KB | 临界 |
| >11520 | — | — | — | ✗ 超预算（需 N-tile 流式变体，§9.3） |

**fp32（8 B/元素/行 = x_f32 4 + sq_f32 4）**：

| N | block_m 默认 | 驻留 (B) | ×1.7 | ≤192KB？ |
|---|-------------|----------|------|----------|
| 514 | 8 | 32,896（32.1KB） | 55KB | ✓ |
| 1152 | 8 | 73,728（72KB） | 122KB | ✓ |
| 3000 | 4 | 96,000（93.8KB） | 159KB | ✓ |
| 4096 | 3 | 98,304（96KB） | 163KB | ✓ |
| 8192 | 1 | 65,536（64KB） | 109KB | ✓ |

默认公式：`block_m = min(8, max(1, _UB_SAFE_BYTES // (bytes_per_elem · N)))`；工厂层 guard 拒绝超预算 block_m（lerp E1 模式）。Stage 4 扫描空间：N=4096 fp16 bm∈{1,2,3}、N=1152 bm∈{4,8,11}（11 需实测膨胀系数，编译期 UB 溢出报错为天然 guard）。

### 4.6 动态轴定义

无运行时动态轴。M、N 均为工厂期常量（per-build JIT 特化，源 lru_cache 工厂同款语义）；尾块 real_m 是 kernel 内 T.min 标量。Op 层（AdaLayerNormFwdOp）把任意前导维 reshape 成 2D——host 元数据视图操作（允许，ascend-constraints.md §4）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[4], target="npuir")   # y 在 index 4（源契约），NPU 编译目标
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector

**判定依据**：仅 reduce_sum + v-prefix 逐元素运算，无 matmul（决策树「含归约 → Developer → alloc_shared → UB」分支）。

### 5.2 Block 划分

```python
block_m = min(8, max(1, _UB_SAFE_BYTES // (bytes_per_elem * N)))   # §4.5 表；仅沿 M 分块
block_n = N                                                         # 全 N 驻留（§1.6.3 主选，不分块）
num_logical = ceil(M / block_m)                                     # 逻辑任务数（行块数）
num_kernels = min(num_logical, 48)                                  # 实际启动内核数（≤ 物理核）
num_local_tasks = ceil(num_logical / num_kernels)                   # 核内串行任务数（静态）
```

**选择理由**：N 维不分块——全 N 驻留使 x 单遍 GM 扫描（R3：流量 4MN·eb 全候选最小）；M 维分块——行间独立；block_m 由 UB 预算决定（§4.5），非 GPU 的结构性 block_m=1（§0.6 R2）。

### 5.3 约束分析

- **对齐约束**：UB 32B 对齐由编译器管理（T.alloc_shared.md）；N 无 256 对齐要求（GPU 约束已消除，logsumexp 同款结论）；尾轴向量宽整除性见 §1.6.3（最差 N=514 浪费 1.17%）。
- **UB 容量**：§4.5 表，全负载域 ≤ 96KB（×1.7 ≤ 163KB）✓。
- **L0/L1/分形**：不适用（纯 Vector）。

### 5.4 注意事项

- **M % block_m ≠ 0（行尾）**：`real_m = T.min(block_m, M − off_m)`，src/dst 双显式 slice 装载/写出；v-op 全 buffer 操作，垃圾行按行隔离、输出截断丢弃（§0.6 R3）。
- **N 非对齐**：无 padded 路径——精确 N 缓冲 + slice（GPU 的 needs_pad 双路径在 NPU 统一为单路径）。
- **wrapper 默认 block_m=1**：NPU wrapper `_select_row_config` 返回 `{"block_m": 1}`——功能 valid（本设计对任意 bm ≥ 1 成立），性能非最优；集成期/Stage 4 以 §4.5 表值或扫描值覆盖。
- **block_m 上限 guard**：工厂层 ValueError（§3.3）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **物理核数**：**24**（AI Core，Ascend910B2C）。实查记录（2026-09-10，本设计执行）：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  n = NPUUtils.get().get_aicore_num()   # -> 24
  vector_cores = n * 2                  # 纯 Vector 算子翻倍 -> 48
  ```
  与 CONST-aicore-910B2C（pattern-library constants.md，24 AICore）一致；lerp 同款查询（L155-158）。
- **逻辑核数**：`num_logical = ceil(M/block_m)`（仅 M 维分块，block_N=N 不分块）：
  | workload | shape | block_m | num_logical |
  |----------|-------|---------|-------------|
  | smoke-dit | 64×1152 | 8 | 8 |
  | dit-xl-2 | 1024×1152 | 8 | 128 |
  | llama-prefill | 2048×4096 | 2 | 1024 |
  | llama-decode | 1×4096 | 2 | 1 |
- **规模判定与分核方案**：persistent 统一式 `num_kernels = min(num_logical, 48)`，核内 `T.serial(num_local_tasks)` grid-stride（`block_id = i·num_kernels + cid`），`if block_id < num_logical` 守卫；**循环边界静态**（num_local_tasks 为 host 层 Python 编译期常量，PL-1.5 折叠先例）：
  - smoke（8 ≤ 48）与 decode（1 ≤ 48）：**无需适配**——min() 自然退化为 num_kernels=num_logical、每核 1 task，无串行轮次；
  - dit-xl-2（128 > 48）：**极大规模分支**——固定启动 48 内核、核内 serial 3 task（128 = 48×2+32：32 核 3 task、16 核 2 task，不平衡 ≤1 task）；
  - prefill（1024 > 48）：同上——48 内核 × 22 task（1024 = 48×21+16：16 核 22、32 核 21，不平衡 ≤1 task ≈ 4.7%）。
  该形态消除逻辑核超发的串行调度开销（core-split-strategy §1），并获 grid-stride 聚集窗口的 MTE2 访问效益（PL-1.6 实测 +3.2%）；lerp（同流量形态）为已调优先例。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表仅 block/task 级调度循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务维（M 行块） | 核级并行 + 核内串行 | `T.Kernel(num_kernels)` + `T.serial(num_local_tasks)` | persistent 分核（§5.5）；行块间独立 |
| 元素维（N） | 向量化（无循环） | v-op / reduce_sum 全 buffer 操作 | §1.6.2 #1-#7 |
| 尾块 | 标量 min + slice | `T.min` + 双显式 slice | §1.6.2 #9 |

### 6.2 循环伪代码

见 §3.3（结构：`with T.Kernel(num_kernels, is_npu=True) as (cid, _):` → buffer 分配（serial 外，跨 task 复用）→ `for i in T.serial(num_local_tasks):` → `if block_id < num_logical:` → 5 段计算体）。

### 6.3 流水线优化

**不使用显式 `T.Pipelined`**。理由：单任务内为顺序依赖链（reduce → center → reduce → epilogue），无 K 维迭代可重叠；任务间的装载/计算 overlap 交由编译器 auto multi-buffer——PL-1.6 实测（同款 3 输入 1 输出流量）：≥16M 元素档全部向量 pass 隐藏于 MTE2 窗口内（delta ≤1.1µs）。Stage 4 选项：显式双缓冲（scale/shift 预取下一 task）与 `T.Pipelined` 包裹 serial 循环——若 baseline 偏离 §1.6.0 估算下界 >2x 再评估。

### 6.4 尾块处理

M % block_m ≠ 0：最后一个行块的 real_m < block_m——装载/写出双显式 slice 截断；v-op 全 buffer（垃圾行按行隔离、输出丢弃）。num_logical % num_kernels ≠ 0：if 守卫跳过越界任务。N 无尾块（精确 N）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步指令，无手动 `T.sync_block_set/wait`：

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| T.copy(GM→UB) 后 | 自动 | 编译器在 MTE2→V 拓扑自动插 pipe_barrier |
| v-op / reduce 之间 | 自动（顺序执行） | 同引擎顺序依赖 |
| vcast 后 | 自动 | 同上 |
| T.copy(UB→GM) 前 | 自动 | 编译器在 V→MTE3 拓扑自动插 pipe_barrier |
| serial 任务间 | 自动 | 同核串行，无并发 |

无核间协作（行独立，无跨核归并）——无需 sync_block_set/wait、无需 workspace。

### 7.3 pass_configs 配置

无特殊 pass_configs。Developer 模式默认编译流水线（logsumexp/lerp 同款）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（移植源仓测试基准 `AdaLayerNormTest.ref_program`），**不复刻 §0.6 的 NPU 算法**（无 vcast / fp32 中转 / persistent 结构）——保证验证独立性。

```python
import torch
import torch.nn.functional as F

def golden_ada_layer_norm(x, scale, shift, eps=1e-5):
    """PyTorch 参考实现（源仓 tests/ops/test_ada_layer_norm.py ref_program 移植）。

    输入: x/scale/shift 同 shape (M, N)，dtype ∈ {fp16, bf16, fp32}
    输出: y 同 shape 同 dtype —— y = scale * LayerNorm(x) + shift
    语义（§0.1）: fp32 opmath（F.layer_norm 上 x.float()）+ 单次舍回 .to(x.dtype)
    """
    N = x.shape[-1]
    normed = F.layer_norm(x.float(), (N,), weight=None, bias=None, eps=eps)
    y = scale.float() * normed + shift.float()
    return y.to(x.dtype)
```

### 8.2 精度标准

来自 `examples/TileOPs/tests/ops/test_ada_layer_norm.py` `_get_tolerances()`（与 logsumexp 案例同源标准）：

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

fp32 中转链对齐 golden opmath（TRAP-fp16-opmath-golden：≤1 ulp fp16；D-1 verify_equiv 佐证：fp16/bf16 逐位一致）。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确」（完整 L1/L2/Boundary 分层套件交由 tilelang-op-develop）：

| # | Shape (M, N) | dtype | block_m | 验证目标 | 容忍度 |
|---|-------------|-------|---------|---------|--------|
| L0-1 | (1024, 4096) | fp32 | 3 | 对齐 shape fp32 直通路径（无 vcast） | 1e-5 |
| L0-2 | (1024, 4096) | fp16 | 2 | 对齐 shape fp32 中转链 | 1e-3 |
| L0-3 | (1024, 4096) | bf16 | 2 | bf16 → fp32 → bf16 路径 | 1.6e-2 |
| L0-4 | (1024, 3000) | fp16 | 3 | **N 非 2 的幂**（GPU 需 pad；NPU 精确 N 单路径） | 1e-3 |
| L0-5 | (1025, 4096) | fp16 | 2 | **M 尾块**（1025 % 2 = 1，垃圾行丢弃） | 1e-3 |
| L0-6 | (1025, 4096) | bf16 | 2 | M 尾块 + bf16 | 1.6e-2 |
| L0-7 | (16, 1152) | fp16 | 8 | 自然非对齐 N（repo smoke 用例） | 1e-3 |
| L0-8 | (17, 514) | fp16 | 4 | **行尾 + 微型 N**（repo async 回归用例形态） | 1e-3 |
| L0-9 | (64, 1152) | fp32/fp16/bf16 | 8 | manifest smoke-dit（3 dtype） | 各自 |
| L0-10 | (1024, 1152) | fp16 | 8 | manifest dit-xl-2 | 1e-3 |
| L0-11 | (2048, 4096) | fp16 | 2 | manifest llama-prefill | 1e-3 |
| L0-12 | (1, 4096) | bf16 | 2 | manifest llama-decode（单行；persistent 退化单核） | 1.6e-2 |
| L0-13 | (2, 512, 4096) 3D | fp16 | 2 | Op 层 reshape 路径（host 元数据视图） | 1e-3 |
| L0-14 | (5, 13) | fp16 | 4 | 微型奇数 shape（防御性） | 1e-3 |

**通过条件**：全部项通过 atol/rtol；对照 = §8.1 golden（CPU）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 |
|------|------|---------|
| **UB auto-multi-buffer 膨胀 ~1.7x**（TRAP-UB-multibuffer-inflation） | bm 表按 1.7 裕度预算；若实际膨胀 >1.7 则编译失败（`ub overflow` 报错文本） | 工厂 guard + 失败时降 bm（§4.5 表内取小一档）；Stage 4 实测膨胀系数回填 |
| **roofline 常数 stale** | §1.6.0 估算引用的 CONST-* 均被 kb_stale_check 标记（当前工具链 a831182 > 条目版本戳） | 估算仅作排序与对账基线；Stage 4 Phase 1 按硬件成本模型 §3 回填「设计估算 vs 实测」偏差行 |
| **bf16 无 Vector 算术**（vadd/vmul/vsub/reduce bf16 ×） | bf16 直算编译失败 | fp32 中转链（§0.6 R4，必需） |
| **未初始化垃圾行流经 v-op/reduce** | 尾任务垃圾行统计量无意义 | 按行隔离 + 输出截断（lerp/logsumexp 已验模式）；若编译器对 garbage 做 DSE 优化改变行为，L0-5/L0-8 用例可捕获 |
| **N > 驻留上限**（fp16 约 N>11520） | UB 超预算 | 超出本任务域（manifest max N=4096）；需 N-tile 流式变体（§1.6.3 候选 #2 结构），工厂 guard 报错拦截 |
| **fp32 求和序 vs golden** | torch F.layer_norm 内部（Welford/向量序）与 NPU reduce 树序不同 | 容差判定（fp32 1e-5）；verify_equiv 显示两式 fp64err ~1e-6 同量级 |
| **inf/NaN 行为** | 全 inf 行 → NaN 输出（公式自然传播） | 与公式语义一致；测试用例均为有限随机输入（源仓同款）；verify_equiv 已对照两式传播一致性 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| UB 溢出 | block_m 超表 / 膨胀 >1.7 | 编译失败 `ub overflow` | 降 bm；§4.5 表 |
| bf16 直传 v-op | 未做 vcast fp32 | 编译错误（dtype 不支持） | fp32 中转链 |
| UB dst 非零起点切片 | 手写列偏移 slice | VEC 对齐错误（TRAP-UB-dst-align） | 全部 0 起点双显式 slice（§3.3 已固化） |
| T.copy base+size 跨步形态 | 图省事用索引+size | 区域语义静默误读（TRAP-T-copy-region-semantics） | 双显式 slice 形态 |
| 跨 dtype T.copy 当 cast | GM fp16 → UB fp32 一步 | 隐藏 VCast 舍入不可控（TRAP-C12） | 显式 vcast(rint) + 同 dtype copy |
| `T.vrsqrt` 误写 `T.rsqrt`（调用量） | 文档文件名 T.rsqrt.md 误导 | AttributeError | 代码导出名 `T.vrsqrt`（layer_norm.py L61 实调） |
| kernel 无输入 tensor | 只留 out_idx | 运行期 MTE DDR 崩（TRAP-zero-input-crash） | 本设计恒有 4 输入（含 _dummy） |
| if 体内新定义变量跨块引用 | TVM parser 作用域（C11） | Undefined variable | if 体内只写预分配 buffer（lerp 同款） |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|---------|
| N > 11520（fp16）/ > 14336（fp32） | 超出驻留域：工厂 guard ValueError 拦截；后续如需支持走 N-tile 流式变体（§1.6.3 #2：T.serial 分块 + clear=False 累加 + 输出遍重读，流量 5MN·eb）——不在本任务范围（manifest max N=4096） |
| M = 1（decode） | num_logical=1 → 单内核单任务；延迟 bound ~10µs（§1.6.0 估算） |
| 3D 输入 (B, S, H) | Op 层 reshape (B·S, H) 后调 kernel（host 元数据视图，允许）；L0-13 覆盖 |
| 常量行（var=0） | rstd = rsqrt(eps)（数学定义）；verify_equiv 角点已对照 |
| wrapper 默认 block_m=1 | 功能 valid；性能优化留给集成期 config 覆盖 / Stage 4 |
| has_gate=True | NotImplementedError（AdaLN-Zero 为后续独立迁移任务，工厂签名已预留） |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/ada_layer_norm/_ada_layer_norm_kernel/
├── DESIGN.md                  # 本设计文档
├── verify_equiv.py            # D-1 等价性机器验证脚本（已执行，EQUIV_PASS）
├── _ada_layer_norm_kernel.py  # 算子实现（Stage 3 产出）
└── RETROSPECTIVE.md           # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | O1+O3+O4 EQUIV_PASS（45 案例）；C2 拒绝证据（7/9 对抗案例 FAIL） |
| `_ada_layer_norm_kernel.py` | ⬜ 待实现（Stage 3） | 工厂 + kernel + golden + L0 测试 |

### 10.3 命名规范

- 项目目录名：`ada_layer_norm`；算子目录名 / 实现文件：`_ada_layer_norm_kernel`（snake_case，不裁剪不变换）
- Golden 函数：`golden_ada_layer_norm(x, scale, shift, eps)`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）——本文档
2. ✅ 等价性机器验证（verify_equiv.py，D-1）
3. ⬜ Golden 函数（§8.1，验证基准）
4. ⬜ 算子实现 `_ada_layer_norm_kernel.py`（§3.3 伪代码 → 代码）+ L0 门槛测试（§8.3 全 14 项）
5. ⬜ 精度比对（vs golden，通过全部 L0）

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 接口参数不变 | ✅ | 工厂签名 `(M, N, eps, dtype, has_gate=False, use_cp_async=False)` 原样；`use_cp_async` 保留但忽略（后端相关适配，§0.5） |
| `@T.prim_func` 参数名/顺序不变 | ✅ | `main(x, scale, shift, _dummy, y)` + out_idx=[4]（源契约） |
| threads 移除（K9） | ✅ | `_func(block_m)`；TRAP-threads-kwarg-noop；NPU wrapper 已适配 |
| host 侧不改输入 | ✅ | 全部计算在 kernel 内；无 host padding/permute |
| 从源码推断规格（不问用户） | ✅ | §0.1/§0.2 从源码 + manifest 解读；输出 shape (M, N) 已回填标注依据（§0.2） |
| has_gate=False 范围 | ✅ | 仅迁移非 gate 路径；gate 路径显式 NotImplementedError |

## 性能目标（optimize 场景追加，2026-09-10）

| 字段 | 值 |
|------|-----|
| 性能目标类型 | `best_effort`（用户诉求"尽力调优"） |
| 目标数值 | —（best_effort：迭代上限内持续压低 Task Duration / 提升 AICore 利用率） |
| Baseline | Stage 5 集成 bench（msprof op，Ascend910B2C，CANN 8.5.0，`profile_run_msprof_20260910_144841.log`）：smoke-dit 64×1152 fp32/fp16/bf16 = 5.18/5.22/5.24 µs；dit-xl-2 1024×1152 fp16/bf16 = 34.06/34.88 µs；llama-prefill 2048×4096 fp16/bf16 = 90.80/91.42 µs；llama-decode 1×4096 bf16 = 3.26 µs。Roofline Ratio 峰值 30.8%（llama-prefill fp16） |
| 测试 shape | manifest workloads 8 组（同 bench：64×1152 ×3 dtype、1024×1152 ×2、2048×4096 ×2、1×4096 ×1） |
| 噪声阈值 | 3%（默认） |
| 最大迭代数 | 10（默认） |

**已知第一杠杆（Stage 5 记录）**：bench 全组跑 wrapper 结构默认 `block_m=1`（非 kernel UB 预算表默认）；调优须优先评估 block_m 提档收益。
