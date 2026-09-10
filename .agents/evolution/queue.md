# 进化提案队列（Evolution Queue）

> TileLang-Op-Conductor 自进化机制的提案队列。由 `tilelang-skill-evolver` 维护（唯一写入者）。
>
> - **用途**：暂存未达合入阈值的 P 类（模式方法）提案与待人工审批的 R 类（流程规则）diff 提案；D/C 类（实测数据/案例索引）不进本队列，由 evolver 直接合入 pattern-library。
> - **字段规范**：见 `.agents/skills/tilelang-skill-evolution/references/queue-schema.md`（本文件头部仅列速查）。
> - **状态**：`pending`（待确认/待审批）→ `verified`（Tier 1 证据达阈值待合入）→ `merged`；或 `rejected` / `expired`（90 天未决）/ `conflict`（与现有条目矛盾且无法自动裁决，留人工）。
> - **审阅方式**：Tier 2（R 类）提案经用户批准后由 conductor 调度 evolver（`mode=apply`）执行写入；Tier 1 达阈值后由 evolver 在下次蒸馏时合入。

## 字段速查

```yaml
proposal_id: VP-{YYYY}-{NNNN}     # 唯一 ID，由 evolver 递增分配
type: D / P / R / C               # D 实测数据 / P 模式方法 / R 流程规则 / C 案例索引
title: <一句话标题>
evidence:                         # 证据链（Tier 1/2 必填；D 类直接合入时也须具备）
  - <工件路径#定位>
repro: <复现命令或 none>
toolchain_stamp: <tilelang build/commit + 设备 + CANN 版本，或 none>
target_doc: <目标文件路径>
delta: <五种合法动作之一：add / update / consolidate / negate / deprecate + 条目正文>
status: pending / verified / merged / rejected / expired / conflict
confirmations: {n}/{2}            # Tier 1 合入阈值：2 次独立证据（须来自不同任务）
created_by: <task_id + 日期>
decided_by: evolver / human / -   # 裁决者
```

---

## Pending

### Tier 1（P 类，2 次独立证据后合入）

## VP-2026-0002
- type: P
- title: sub-fp32 逐元素算子 fp32 中转模式：vcast(rint) 升 fp32 → fp32 域 v-prefix 链 → vcast(rint) 单次舍回（bf16 dtype 支持 + fp16 golden 对齐双触发）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md（Stage 1 bf16 触发 + Stage 3 fp16 修复模式提案）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py（模块「Implementation Notes (attempt-2 precision fix)」等价性论证 + attempt-2 全量 62 PASS）
  - docs/Tilelang.language/数学操作/T.vadd.md §2.2.1（v-prefix 算术 dtype 矩阵不含 bf16）
  - docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 仅 rint；f32→f16/bf16 含 rint；bf16↔f32 仅 rint）
- repro: python examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py --level all
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24；golden torch 2.9.0+cpu
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「sub-fp32 逐元素 fp32 中转模式」：
    双触发条件：① 契约 dtype 含 bf16——v-prefix 算术（vadd/vsub/vmul）dtype 矩阵不含 bf16，vcast 升 fp32 是唯一计算路径；② fp16 需对齐 torch CPU golden——golden 经 fp32 opmath + 单次舍回，原生域逐步舍入差 2–3 ulp（见 pattern-library §2 opmath 行）。
    链结构：GM→UB(同 dtype) → vcast(rint)→fp32 → fp32 域 v-prefix 原地链 → vcast(rint)→原 dtype → UB→GM。
    实测：max_diff ≤1 ulp fp16（抽样 binade 4.883e-04～1.953e-03）、全量 0 violations（fp16 的 1e-3 rtol 恒覆盖 1 ulp）；NaN/Inf 角点 IEEE 传播与舍入路径无关（中转后 inf-corner 逐位不变）。
    UB 预算：中转路径按 Σ(elem_bytes×buffer_count) 复算——bf16 24B/elem（上界 196608/24=8192）、fp16 20B/elem（上界 9830，保守 guard 8192）。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0003
- type: P
- title: 纯 Vector 一维逐元素 persistent 模板：num_kernels=min(ceil(N/block), aicore×2) + T.serial 静态边界 grid-stride + 动态尾块零起点切片
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/DESIGN.md#5.5（分核三要素）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md（Stage 1 提案 + Transferable Lessons）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md（persistent nc=48 全档贴 copy 地板；grid-stride 实测最优访问模式）
  - examples/elementwise/vec_add_1d.py L31–L38（尾块处理官方先例）
- repro: python examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py --level all
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24（纯 Vector 翻倍 48 vector cores）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「纯 Vector 一维逐元素 persistent 模板」（核数实查/纯 Vector 翻倍规则已有 ascend-constraints.md「物理核数限制」行，仅交叉引用不重复）：
    模板：num_kernels = min(ceil(N/block_size), aicore×2)；核内 T.serial(ceil(num_logical/num_kernels)) 静态边界（N 工厂特化保证编译期常量，可直接用 T.ceildiv+min 求值，见 §1.5 host 层常量折叠行）+ block_id = i×num_kernels + cid + if block_id < num_logical 屏蔽越界。
    尾块：tail = T.min(block_size, N−t0) + UB 零起点切片 T.copy(x[t0:t0+tail], x_ub[0:tail])——v-prefix 整 buffer 操作对 UB 陈旧段一并计算、只拷出 [0:tail]，正确性无碍。
    提取期注意：TileOPs wrapper 的 E3/E4 适配注释（如 threads= 取代）是提取期笼统预估，核数/grid 取值以 DESIGN.md 分核三要素为准，冲突时设计文档优先。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0004
- type: P
- title: BP_grid_stride_vs_contig_chunk：带宽饱和区连续分块映射回退 +3.2%（256M），grid-stride 聚集窗口对 HBM 混合流更优
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-3（v3_op2：256M fp16 1765.01 vs 1710.07，+3.2%；1M/16M tie）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/profiles/round3/v3op2_256m_fp16_bs4096
- repro: msprof op --kernel-name=main --launch-count=15 ... python bench.py --impl ./_make_lerp_tensor_kernel_opt_v3_op2.py --dtype float16 --N 268435456 --block-size 4096（对照 grid-stride 基线）
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_grid_stride_vs_contig_chunk：
    触发信号：带宽饱和区（mte2_ratio ≥0.95）考虑把 grid-stride 任务映射改为每核连续分块以改善「DRAM 局部性」。
    反证/判定：lerp_tensor 256M 实测连续分块 +3.2% 回退、小 N tie——grid-stride 的 N 核聚集移动窗口（fewer open pages / 行缓冲命中）对 HBM 混合读写流更优，「连续流局部性」直觉在混合流下反向；数据侧见 pattern-library §1.6 访问模式条。
    动作：饱和区维持 grid-stride；连续分块仅在纯读/写单一流向或 L2 友好 workload 复测后考虑。
    验证：Task Duration 非回退 + mte2_ratio 不恶化。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0005
- type: P
- title: 标量/分支削减优化在 MTE2 饱和区无效的判定式：mte2_ratio ≥0.95 时标量已被搬运隐藏，先查饱和再立项
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-2（v2_op2 静态尾块全档 tie/-0.6%；guard-free 机制证据 defer <0.3%）
  - 同文件 Baseline 表（≥16M mte2_ratio 0.95–0.99；1M 档 0.70–0.84 流水浅区）
- repro: msprof op ... python bench.py --impl ./_make_lerp_tensor_kernel_opt_v2_op2.py --dtype float16 --N 1048576 --block-size 4096
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    update BP_ub_traffic_floor 触发信号，追加：「per-tile 标量/分支开销（动态尾块 T.min、if-guard、地址计算）立项削减前，先查 mte2_ratio——≥0.95（搬运饱和）时标量已被隐藏，实测静态尾块/去 guard 全档 tie/-0.6%（lerp_tensor 2026-09-07）；仅 mte2_ratio <0.9 的流水浅区（小 N 档 0.70–0.84）可能有收益」。数据侧见 pattern-library §1.6 标量削减判定式条。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0006
- type: P
- title: 精度失败定征顺序：dtype 隔离 → 违反率与 shape/block/核数无关性 → golden opmath 域小实验（避免盲目排查分核/内存层级）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 3 Transferable Lessons（定征顺序实证有效）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py（定征输入：fp16 ~0.125% 违反、bf16 逐位一致、fp32 全过、违反率与 shape 无关）
- repro: 对 attempt-1 基线分 dtype 跑 --level L0，观察违反的 dtype 隔离性与违反率 shape 无关性
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24；golden torch 2.9.0+cpu
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新文件首条目（合入时创建 references/precision-patterns.md）「精度失败定征顺序」：
    ① dtype 隔离：哪个 dtype 挂 / 哪些全过（全 dtype 挂 → 结构/同步缺陷；单 dtype 挂 → 舍入路径统计性质）。
    ② 无关性检查：违反率与 shape/block_size/核数无关 → 纯舍入路径性质，排除 tiling/分核/同步排查方向。
    ③ golden opmath 域小实验：数行脚本核实 golden（torch CPU op）在目标 dtype 的实际计算域（fp16 常见 fp32 opmath + 单次舍回），再决定 kernel 原生域 vs fp32 中转（模式见 pattern-library §2 opmath 行 + VP-2026-0002）。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0007
- type: D
- title: T.Kernel(..., threads=) 在 npuir 无效果（KernelLaunch NPU 分支仅消费 grid_size）——设计侧约束表路由
- evidence:
  - src/ir.cc L175–203（NPU 分支无 block_size→threadIdx 消费路径）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/REVIEW.md 附#28
  - tilelang/language/kernel.py L218–223（Python 签名仍接受并透传）
- repro: grep -n -A 30 "is_npu_kernel_frame" src/ir.cc
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07，源码通读）
- target_doc: .agents/skills/tilelang-op-design/references/ascend-constraints.md
- delta: |
    add §1「本项目已知限制」表新行：
    | GPU `threads=` kwarg 无效果 | `T.Kernel(..., threads=)` 的 block_size 在 npuir KernelLaunch 不生成 threadIdx 绑定（Python 签名仍透传） | GPU 源码迁移残留 `threads=` 无声无效，threads/npt 语义需显式落到 block_size | 迁移时直接移除 `threads=`，折叠为单一 block_size 由 wrapper 参数传入（实证见 pattern-library §2 threads= 行；lerp_tensor 2026-09-07） |
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: 三件套齐备；因 target_doc 为 op-design references（Tier 1 写域）而入队——设计期（GPU→NPU 迁移）是该事实的主要消费场景，事实本体已 Tier 0 合入 pattern-library §2。

## VP-2026-0013
- type: P
- title: attention 族 Developer 性能阻塞（aiv_scalar>50% / CG-2026-0001 崩溃类）时先评估 Expert 双 Scope 形态再定编程模式——结构级绕法含硬边界清单
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0（动因：developer 基线 ~3.5 TOps/s = msprof 算力上限 359 TOps/s 的 ~1%，落后 torch-SDPA 10–25×）+ #§0.6 E2/E3
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#§0/§4/Round-7（developer 谱系阻塞项来源——**谱系注明**：tilelang 67db6f3 + CANN 26.0.rc1，2026-09-07 晨 developer 轮 Stage 4 产物，非本 expert 任务产物）
  - examples/deepseek_v4/example_sparse_attn_kernel_highperf.py（Expert 结构全同构先例）
  - pattern-library §1.7（expert vs developer 同门 bench：长 KV 1.57–1.63× / 短 KV ~1.31× 回退——Expert 形态的收益分界数据）
- repro: 触发条件——任一 attention 族 Developer kernel 出现 aiv_scalar >50%（谓词 mask 预填标量化）或 CG-2026-0001 崩溃类触发时，对照 highperf 结构评估 Expert 重写（结构形态见 delta）
- toolchain_stamp: expert 侧 tilelang dev build 21586b5（2026-09-07）+ CANN 8.5.0 + Ascend910B2C；developer 阻塞证据侧 tilelang 67db6f3 + CANN 26.0.rc1（跨代对照已注明）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「Expert 双 Scope 流水形态（Developer 阻塞的结构级绕法）」：
    判据：attention 族 Developer kernel 出现 aiv_scalar >50% 或 CG-2026-0001 崩溃类（Pipelined 体内条件构造 / 跨块计算重叠 / 标量谓词写）时，先评估 Expert 形态再定模式，而非在 Developer 内回退。
    结构形态：双 Scope（Cube：T.alloc_L1/L0C + load_nd2nz/T.copy + T.gemm + T.store_fixpipe；Vector：T.alloc_ub + v 前缀链）；跨引擎数据经 GM workspace 多槽 + per-slot flag（T.sync_block_set/wait）；staggered stream；运行时 if 在 Expert T.serial 流内合法（Developer CG-2026-0001 崩溃面不适用——Developer 的禁忌不带入 Expert 设计）。
    Expert 硬边界：kernel 内无 fragment 抽象（T.alloc_fragment/T.Pipelined/T.Parallel 不可用）；pass_configs 关闭 TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION 与 NPUIR_ENABLE_AUTO_MULTI_BUFFER（highperf 先例）；wrapper 契约兼容（工厂内层闭包 + workspace 显式参数，highperf sparse_attn() 形态——workspace 需求不构成改 wrapper 的理由）。
    实测收益/代价：见 §1.7（长 KV 1.57–1.63×、短 KV ~1.31× 回退——短 workload 为主的算子慎选）；vcmp 标量形态先例与低效警告见 §2 v 算子操作数行。
    未文档化假设：T.sync_block_set/wait 的 id 预算无文档（仅 T.set_flag.md §2.1 标 event_id 0–15）——flag 族 id 数量只能以先例推断（本轮 7 族在安全域）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0014
- type: P
- title: 块内 mask 的向量构造链模式：arrange strides → vsub → vcmp(int16 索引) → vand → vselect；−1e38 有限哨兵 + mask 后置 softcap + 两段式分界零 mask 开销块
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E3 / #§1.6.2-#1
  - examples/deepseek_v4/example_sparse_attn_kernel_highperf.py#L345-354（vcmp 标量 PrimExpr + vand 链）/ #L611（PIPE_V 内 T.arange strides）
  - developer 谱系对照：同函数 T.Parallel 谓词预填形态 aiv_scalar 88–89% 实测（opt_log §1，**谱系注明** tilelang 67db6f3 + CANN 26.0.rc1）
- repro: 任一含块内 mask 的 attention/causal/sliding-window 算子，以向量链替代谓词预填后对照 aiv_scalar 占比与 Task Duration
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C（本轮 Expert 实现验证）；developer 对照侧 67db6f3 + CANN 26.0.rc1
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「向量 mask 构造链（替代谓词预填）」：
    链结构：T.arange strides=[0,1]/[1,0] 生成行列索引 → T.vsub 相对位置 → T.vcmp(整型 PrimExpr 阈值, int16 索引矩阵)（标量操作数拒 tir.Cast，见 §2 v 算子行）→ T.vand 合取 → T.vselect 应用；NaN 无关（vselect 按位选择不看 S 值）。
    三个关键设计点：① masked 哨兵用有限 −1e38 替 −inf（e^{−1e38−m} 对有限 m 下溢为精确 0，与 e^{−inf} 同效，从根上规避 −inf−(−inf) 的 NaN 路径）；② mask 应用放 softcap 之后（vselect 与值无关，可同时吃掉 NaN/Inf 垃圾分数；反例：mask→softcap 顺序下 softcap(−1e38)≈−softcap 非精确 0，破坏 P(OOB)=0 不变式）；③ 两段式 K_A 分界（全有效块零 mask 开销、对角/尾块走 mask 链）——分界公式取整方向推导模板见 VP-2026-0018 与 §4 expert attention 案例行（K_A ceildiv→floordiv 反例）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0015
- type: P
- title: Expert 跨引擎 workspace 的跨任务槽位复用握手：per-task 槽位重启 + FLAG_TASKDONE 边界（避免 causal 变长 NK 的全局块前缀索引）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§6.3（flag 协议安全性传递性论证）+ #§7.2（四族 flag + FLAG_V 写侧 staging 补全）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（维度 5 附核对记录：三族 + 单槽 + 跨任务五条论证独立推演全部成立）
- repro: 任一 per-task 变长内循环（causal NK 变长）的 Expert staggered 流设计
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 附注（Expert 流水模式条目〔VP-2026-0013〕的握手子模式，合入时紧随其后）：
    模式：per-task 槽位重启（slot = i % slots 任务内重计数、跨任务从 0 重启）+ FLAG_TASKDONE 跨任务握手（Vector epilogue 完成后 set、Cube 下一任务首个 S 存储前 wait）——跨任务槽位复用安全，避免 causal 变长 NK 下的全局块前缀索引（数据依赖、不可静态化）。
    配套：flag 协议安全性按「写 i+slots 前读 i 已完成」逐族传递性推演（set 的程序序前驱 → wait 的后继），跨任务握手族单列；写侧 staging（FLAG_V 族，highperf V0S→C1L 同构）补全读侧无序窗口。
    实测：L2-gap100 四例 + kv32-guard（M7 单块退化共存位形）全过（DESIGN §8.2 / debug_log.md Final results）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0016
- type: P
- title: slots≥2 分块装载的尾块守卫从不变式推导（nk_total ≤ slots）而非 shape 枚举——枚举法漏浅块数域；判据同源反推测试域
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E7 + #§8.2（缺口域 4 例：S_kv=100 ∈ (bn,2bn)、ns=2，条件与测试同源）+ #§7.2（Vector 零填充预备段）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 3：v0 守卫仅覆盖 nk_total==1，slots=2/nk_total=2 时部分块为槽位首次使用——L2 套件 kv32/tail520 均不在缺口域，缺口靠检视推导而非测试暴露）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md（Final results：L2-gap100 四例 causal/non-causal × fp16/bf16 全过）
- repro: python examples/multi_head_attention/_gqa_prefill_fwd_kernel/_gqa_prefill_fwd_kernel.py --level all（L2 段 gap 四例即判据同源测试域）
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「多槽分块装载的尾块守卫不变式」：
    判据推导：部分块 p=nk_total−1 的残留安全性 ⟺「该槽位 p%slots 此前已被整块装载」⟺ nk_total > slots；唯一破坏点 = 核启动后槽位首用即部分块（残留为未初始化 L1/GM〔torch.empty〕，0×NaN=NaN 污染有效行）——正确守卫条件 = nk_total ≤ slots（含跨任务归纳：槽位尾行状态恒 ∈ {零, 有限真实数据}）。
    枚举法反例：从 shape 枚举（seq_len_kv < bn ⟺ nk_total==1）出发漏浅块数域（nk_total=2、slots=2）且测试域与条件不同源；判据同源反推测试域 = 按判据取边界值 S_kv ∈ (bn,2bn) + 必须触发 slots=2。
    配套：Vector 零填充预备段 + FLAG_V 握手（无它则 staging 写与 Cube 读无序）；守卫面随路线结构性简化（删载体族后 GM 未初始化域缺口自动消失——路线裁决的「正确性面」第四维）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0017
- type: P
- title: bf16 输出精度门禁设计须评估 golden 自身 P 量化噪声的放大位形——tier-2 绝对下限 max(5e-3, 2·flip_rel·vmax) + tier-3 窄域分类机制
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md#D6（L1-70b-long-bf16：1/16777216 元素超 tier-2 1.1%；fp64 逐元素验证 |kernel−true|=4.3e-4 vs |golden−true|=5.8e-3——golden 是偏离方；P 1-ulp 翻转经 causal 早行 p≈0.33×|v|≈2.06 放大 → 5.4e-3 > tier-2）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md（Stage 3 Precision-gate 行）
  - 交叉引用：queue VP-2026-0006（精度失败定征顺序——同文件合入时互链）、pattern-library §2 torch golden opmath 行（golden 侧域差异同类现象）
- repro: bf16 randn 域 attention：causal 早行（有效位少 × P 量化翻转 × 大 |v|）位形对照 tier-2 门限；元素级 fp64 对照定位偏离侧
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C；golden = torch SDPA NPU
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新条目「bf16 输出门禁的 golden 噪声放大位形与 tier-3 分类」（合入时与 VP-2026-0006 定征顺序条目同文件、互相交叉引用）：
    机理：golden 自身 P 量化到输入 dtype 的 1-ulp 翻转，在 causal 早行（有效位置少、p 大、|v| 大）放大为输出偏差 5.4e-3 > tier-2（5e-3）——kernel 比 golden 更接近 fp64 真值（4.3e-4 vs 5.8e-3）仍可能超差；该机理是源算法 P-quantization-to-input-dtype 的固有性质，非 kernel 缺陷。
    门禁设计：tier-2 绝对下限应取 max(5e-3, 2·flip_rel·vmax)（bf16 randn 域 ≈ 4e-2）；未重校门禁前的过渡机制 = tier-3 窄域分类：发生率 ≤0.01% + 放大包络（2·flip_rel·|v|max）+ lse 门不受影响 + 逐例上报——真实缺陷（如丢 mask：79.8% mismatch）仍为响亮失败。
    定位手法：元素级 fp64 对照判定偏离侧（golden vs kernel 谁离真值远）再定机制，避免把 golden 噪声当 kernel 缺陷排查（亦见 pattern-library §2 零输入探针行的 fp64 对照法）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0018
- type: P
- title: AIV/lane 半分类设计的参数化标量阈值按最劣 lane 全局坐标独立重推 + 双 lane 端点数值例（单 AIV 思维自检必然逃逸）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 1：v0 L220/L536 K_blk = base − row0，正确为 base + row0；smoke bx=4/块 4/vid=1 行 288 数值例复现——正确 j≤32 vs 错误 j≤−32）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E3（v1 修复：行映射式 q_pos = s_lo + row0 + i 先行 + 阈值 PrimExpr 标量）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md → DESIGN.md 修订链
- repro: 任一含 vid/row0 半分 + 块级 mask/阈值构造的 Expert 设计：对每个 per-lane 标量公式代入 vid=0 / vid=1 两个端点数值例
- toolchain_stamp: 设计/检视期证据（无运行时依赖）；审查环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 附注（并入 VP-2026-0014 向量 mask 构造链小节的「分界/阈值推导模板」子条，合入时紧随其后）：
    修复模板：① 先写行映射式（lane 的全局坐标，如 q_pos = s_lo + row0 + i）再代源条件推导阈值（K_blk = base + row0）；② 阈值走 PrimExpr 标量而非动态 offset（T.arange.md offset 参数文档类型为 int，运行时 PrimExpr 支持无文档）；③ 共用分界（K_A 类）对非推导 lane 标注保守性（仅性能不损正确性——取整方向反例见 §4 expert attention 案例行：ceildiv 版对 vid=0 是激进上界、floordiv 版精确）。
    逃逸机理：以 vid=0（row0=0）代入的阈值在 vid=1 恰好相差 2·row0 且符号方向隐蔽（该 +row0 写成 −row0 时 vid=0 完全正确、vid=1 全错）——单 AIV 思维的自检必然逃逸，双端点代入是最低成本拦截。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0034
- type: P
- title: BP 跨引擎 flag 串行链：dual-Scope persistent 结构下块宽摊减优先于流水深度（bn↑ 同削 flag 往返/向量 op/gemm 碎片，ns/slots 在宽块上失效）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Iteration-1（busy 核分解：aic_scalar 41%/aiv_scalar 33% = flag 等待自旋，aic_cube 仅 9% 近峰值——ns=1→2 在 bn=64 上零变化）+ #Iteration-3（bn=512 单槽 -15~-16% vs G2@bn=256；bn=512 双槽 L1 溢出 invalid_compile）+ #Final-Summary（跨轮 busy 核画像主线：flag 等待 25–47% 恒存，块宽摊减是唯一有效方向）
  - pattern-library §1.9（数据侧：块宽摊减律 + cbuf=512KB 上限公式 + wide 单槽模式 + H=1 平衡公式；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-10（2026-09-09 同谱系第二轮续调补充〔task 20260909T033622Z，按 VP-2026-0035 同 op 谱系先例不计独立确认〕：v10_dp bn=256 深流水全 case 回退 +11~12%——向量发射开销定律〔每 op ~0.5µs 固定成本〕给出块宽摊减律的机制解释；另实测第三硬上限 L0C=128KB）
- repro: 同结构 kernel（Expert dual-Scope persistent + per-slot flag）扫 bn 64→256→512 × ns∈{1,2}，fa2048/fa4096 non-causal fp16 对照 msprof op Task Duration
- toolchain_stamp: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_cross_engine_serial_chain（目录行置于 BP_pipeline_overlap 之后，正文紧随 BP_pipeline_overlap 节，交叉引用之）：
    触发信号：Expert dual-Scope persistent kernel（Cube/Vector + GM workspace 多槽 + per-slot flag 跨引擎流水）busy 核 aic_scalar 或 aiv_scalar 占比 30%+ 且归因为 flag 等待自旋；cube/mte 已近峰值而总时延远超计算分量；加深流水（slots/ns↑）零变化或被抵消。
    常见反证/不确定点：非跨引擎结构的 MTE/compute 串行归 BP_pipeline_overlap；bn 增大触发 L1(cbuf)=512KB / UB=192KB 上限、bm 增大触发 L0C(cc)=128KB（bm≤51@bn=512，2026-09-09 第二轮实测；上限公式见 pattern-library §1.9）——上限不否定方向，转 wide 单槽（L1 单槽 + GM ws/flag 双槽）继续换块宽。
    推荐动作：先块宽摊减（bn↑ 同时削减向量 op 数/块、flag 往返数/列、gemm 碎片化）再评估流水深度——宽块上 ns=1≈ns=2（双槽收益被单槽 L1 v-load↔gemm2 WAR 串行抵消）；配 bm 任务平衡（wall-rows = ceil(ceildiv(S,bm)/24)·bm）。机制基础（2026-09-09 第二轮实测）：向量算子按发射开销计费——每 op ~0.5µs 固定成本、f16≈f32 逐元素速率，同元素总量下窄块 op 翻倍使 vec-active 翻倍（v10_dp bn=256 实测 +12% 回退）。
    验证指标：Task Duration 下降 + busy 核 scalar（flag 自旋）占比下降 + causal/full 等非目标 dispatch 非回退（full 变体逐字节保留）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: -

## VP-2026-0035
- type: P
- title: Stage 4 交付的工厂内 tuned 分派表模式：TUNED_DEFAULT_CONFIGS 仅替换 wrapper 默认 config（显式 config 一律尊重）+ tuned 目标 case 纳入内嵌测试套件作 blocking 精度层
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py（L100–110 TUNED_DEFAULT_CONFIGS 表 + L187–195 分派逻辑「仅 caller 传 wrapper 默认 (64,64,1) 且 shape 命中时替换」+ L1447–1490 run_fa_tuned blocking 门〔4 目标 case dispatch 路径 tier-1 全 0 flips〕）
  - 同目录 opt_log.md#Iteration-7（最终组装：工厂内 tuned 分派表〔S4-5 模式〕；caller 传 wrapper 默认时生效、显式 config 一律尊重——保 wrapper 契约）
  - 谱系先例（同 op 前任务、非独立计数证据）：examples/multi_head_attention/_prev_task_20260907_developer_optimize/perf_opt/opt_log.md（S4-5 工厂内 shape 分派调优 config）
- repro: python examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py --level all（fa-tuned 4 例）+ 同文件 --use-default-config 与显式 config 传入对照（验证分派仅替换默认）
- toolchain_stamp: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1.9 附注 bullet（合入时紧随「工厂级多 @T.prim_func 变体分派」bullet 之后）：
    「工厂内 tuned 分派表（config 级，S4-5 模式）」：wrapper 契约保持的分派语义——caller 传 wrapper 默认 config 且 shape 命中 TUNED_DEFAULT_CONFIGS 时替换为调优 config；caller 传任何显式非默认 config 一律尊重原值（不静默覆盖用户意图）。tuned 分派覆盖的目标 case 须纳入内嵌测试套件作 blocking 精度层（fa-tuned 层：dispatch 路径 tier-1 翻转数对照基线），防止分派表与全量套件漂移。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: -

## VP-2026-0038
- type: P
- title: 跨引擎/在线递推 kernel 精度失败的误差结构定征：全块均匀且可复现 → 排除竞态、指向跨迭代状态生命周期（rescale 因子覆盖）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-10-OP2（v10_dp 首版精度 FAIL：全块均匀误差 → alpha/ellcur 跨 defer-2 生命周期被 V1(j) 覆盖；3 路 ping-pong〔平铺 buffer + 运行时三分支〕修复后 L0 全过）
  - 同文件#Skill-Retrospective-第二轮（「先看误差结构再定位」纪律：无它会把递推因子覆盖误判为竞态而放弃深流水方向）
  - pattern-library §1.9 sync_block_wait 行（修复形态交叉引用：v-op codegen 拒绝多维 UB 切片操作数，平铺多 buffer + 运行时三分支为合法形态）
- repro: 任一含 online 递推状态（running max/sum + rescale 因子）跨流水 deferral 生命周期的 Expert kernel：首版精度 FAIL 时先记录误差的空间分布（全块均匀 vs 部分核/非确定）再定排查方向
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新条目「误差结构定征（均匀 vs 部分核）」（合入时与 VP-2026-0006 定征顺序、VP-2026-0017 golden 噪声放大位形同文件并互相交叉引用）：
    判据：精度 FAIL 的误差空间分布是竞态/同步缺陷与系统性状态缺陷的判别器——全块均匀且可复现（各块/各核一致）→ 排除竞态假设，指向跨迭代状态覆盖（online 递推的 rescale 因子〔running max/sum 的 α、ell〕在流水 deferral 生命周期内被后继迭代写入）；部分核/非确定分布 → 竞态/同步方向（flag 协议、握手窗口）。
    修复形态：递推状态跨 deferral 需按流水深度 ping-pong 多路；v-op codegen 拒绝多维 UB 切片操作数——平铺多 buffer + 运行时索引三分支是合法形态（pattern-library §1.9）。
    定位顺序与 §2 零输入探针行的元素级 fp64 对照法互补：先判分布（均匀/部分）定缺陷类别，再下钻元素级对照定偏离侧。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

### Tier 2（R 类，结构化 diff 提案，待人工批准后 mode=apply 执行）

## VP-2026-0008
- type: R
- title: op-design Phase M0 第一问增加 golden 计算域核实检查项（fp16 golden 常见 fp32 opmath，勿假设与源算子舍入路径一致）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 3 Skill Flow Issues（设计/检视期隐含假设未被拦截，L0 实测才暴露）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/.task_timeline.jsonl（Stage 3 attempt-1 precision fail，2485s）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py
- repro: torch.lerp CPU fp16（fp32 opmath+单次舍回）vs 逐步 fp16 舍入三步链：N=2^24 差 2–3 ulp、违反率 ~0.125%（pattern-library §2 opmath 行）
- toolchain_stamp: torch 2.9.0+cpu；tilelang-mlir-dev dev root build（2026-09-07）
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（扩展一行）
    定位锚: "2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。**golden 计算域核实**：golden 若为 `torch.*` CPU 实现，须用数行最小实验核实其在目标 dtype 下的实际 opmath 域——fp16 常见 fp32 opmath + 单次舍回（torch.lerp 实证：NPU 原生域逐步舍入差 2–3 ulp、容差违反率 ~0.125%，pattern-library §2）；「忠实源算子原生 dtype 舍入」与「过 golden 容差」冲突时以 golden 为准（源仓测试 ref 即 golden），kernel 侧用 fp32 中转对齐。
    动机: lerp_tensor Stage 3 attempt-1 precision fail（2485s）源于设计期「golden 舍入 = 源算子原生舍入」隐含假设；REVIEW 维度 8 预警了 v-prefix 侧舍入语义缺口但未覆盖 golden 侧 opmath 差异——设计期一道 3 行实验即可消除。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0009
- type: R
- title: design-review 维度 2 UB 预算检查项补多 dtype 混合字节复算口径（dtype 中转路径禁沿用单一 dtype N-buffer 口径）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/REVIEW.md 维度 2 勘误（DESIGN.md §4.5 注与表自相矛盾：bf16 误写 fp16 同款 24576，正确 196608/24=8192）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 2 Value Point Proposals
- repro: 含 vcast 升精度中转路径的 block_size 上界 = UB容量 / Σ(bytes_i × count_i)（bf16 中转 24B/elem→8192；fp16 中转 20B/elem→9830 保守 8192）
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；A2/A3 UB 192KB
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（改一行）
    定位锚: "| UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB） |"
    old 文本: （即上述定位锚原文）
    new 文本: |
      | UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB）；含 dtype 中转（vcast 升精度）路径须按 Σ(elem_bytes × buffer_count) 混合字节宽复算，不得沿用单一 dtype 的 N-buffer 口径（lerp_tensor 实证：bf16 中转 24B/elem 上界 8192，误沿 fp16 口径 24576 需 480KB UB） |
    动机: lerp_tensor DESIGN.md §4.5 出现口径混用且注表自相矛盾，检视维度 2 原口径只查总和未查混合字节宽——勘误后才发现；若未拦截，Stage 3 assert 与 Stage 4 block_size 扫描范围都会错。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0010
- type: R
- title: optimize iteration-diagnosis Step 6 补小 kernel 交错 A/B 多 run 测量协议前置规则（<5% 差异即启用，勿等采纳后复核）
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-2（v2_op2 首测 +4.4% 复测翻转为 tie，耗一轮）
  - 同文件 #iteration-4（fp32 采纳项单轮 -5.2% 系双态膨胀，A/B/A/B 4 run 合并中位 -3.1%）
  - examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/opt_log.md（首证：run 间 ±2–3.5% 环境双态层）
- repro: 16m fp32 final(bs8192) vs baseline(bs2048) 交替 A/B/A/B 各 2 run：B {158.75, 158.59} vs A {163.65, 162.27}，配对次序稳定
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（评估规则列表新增一条）
    定位锚: "- 只有通过必测 dispatch 非回退检查的候选 winner，才能更新为全局 current best。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 只有通过必测 dispatch 非回退检查的候选 winner，才能更新为全局 current best。
  - 小 kernel（Task Duration <20us）或逼近带宽/搬运地板的 workload，<5% 量级的候选差异须先经**交错 A/B/A/B 多 run 协议**裁决：≥3 次独立 `msprof op` run、候选与基线交替次序、合并中位数比较——同 kernel 跨 run 存在 ±3–5% 快/慢双态（BP_run_state_bimodality），单 run median-of-15 可能把双态膨胀误判为增益或回退（lerp_tensor 实证：单轮 -5.2% 复测收缩为 -3.1%）。协议在首次出现 <5% 差异时即前置使用，而非采纳后复核。
    动机: lerp_tensor 两次踩坑（v2_op2 与 fp32 采纳项）各消耗一轮复测才校正归因；协议前置可省两轮实验。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0011
- type: R
- title: conductor optimize 场景预检 baseline 字段补测量口径标注：Stage 5 events（host 侧 wall）数据禁作 headroom 依据
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/integration_log.md（Stage 5 bench events 口径 1M fp16 182.83us「慢于 torch」vs torch msprof 15.22us）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/.stage_state.json user_requirement（任务框定被 events 基线误导：「中小 N 慢于 torch，派发开销主导」）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md Baseline P1（msprof kernel-only 1M 7.30us 快于 torch 2.0–2.3x，全档快于 torch 1.65–2.3x）
- repro: 同 workload 双口径对照：events（host wall，含派发）vs msprof op Task Duration（kernel-only）
- toolchain_stamp: torch 2.9.0+cpu / torch_npu 2.9.0.post2；msprof cann-8.5.0；tilelang 0.1.2+ed787bb（2026-09-07 build）
- target_doc: .opencode/agents/conductor-scenarios/optimize.md
- delta: |
    动作: update（扩展一行）
    定位锚: "- 性能目标类型 / 数值 / baseline（字段与默认值同 `conductor-scenarios/new-op.md` §5 调优必要信息收集表，缺省 `best_effort`）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 性能目标类型 / 数值 / baseline（字段与默认值同 `conductor-scenarios/new-op.md` §5 调优必要信息收集表，缺省 `best_effort`）。**baseline 口径标注**：来自 Stage 5 bench（`prof_mode=events`，host 侧 wall、含端到端派发开销）的数值仅趋势参考、禁止作为 headroom 依据——kernel 时延唯一口径为 `msprof op Task Duration`（pattern-library §2 已弃用 event 口径；两口径方向可相反：lerp_tensor 2026-09-07 events 判 1M 慢于 torch、msprof kernel-only 判快 2.0–2.3x）；透传给 optimizer 的 events 基线须附此标注；
    动机: lerp_tensor optimize 任务的 user_requirement 即被 Stage 5 events 基线误导框定（「中小 N 慢于 torch」实为端到端派发开销失真）；靠 optimize skill 强制 msprof 唯一口径才免于方向错误——调度层透传时即标注可根除。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0012
- type: R
- title: integrator 第一步后补 lint/format 自检步骤（TileOPs rule set B 对集成包生效，B023 类闭包绑定违反须在 pytest 前修复并留档）
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/integration_log.md History mod 1（B023 "Function definition does not bind loop variable" + format.sh ruff-format auto-join；幂等重跑 integrate_kernel.py 会覆盖修复）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_s5_attempt1.py（pre-fix 备份）
- repro: cd examples/TileOPs && bash format.sh --files tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py（pre-fix 报 B023）
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；TileOPs pyproject rule set B；torch 2.9.0+cpu / torch_npu 2.9.0.post2
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（第一步 bullet 列表末尾追加一条）
    定位锚: "- 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。
  - 脚本完成后、第二步 pytest 前，对集成的 kernel 文件执行 `bash format.sh --files <integrated files>`（TileOPs rule set B 对集成包生效，conductor 产物目录不受其约束）并修复 lint 违反（如 B023 循环变量闭包绑定用 default-arg 绑定：`def _to_npu(t, _dt=torch_dtype): ...`）；format/lint 修改记入 `integration_log.md` History（mod 条目）并在 `history_version/` 留 pre-fix 备份——幂等重跑 integrate_kernel.py 会整文件覆盖这些修复，重跑前须先备份；若修改触及 pytest 不覆盖的内嵌 L0 测试代码，须单独复跑内嵌 L0。
    动机: lerp_tensor 集成后 format.sh 才暴露 1 处 B023（内嵌测试闭包），且 format 自动改动使集成副本与 conductor 产物出现双向分歧；pytest 不覆盖内嵌 L0 代码，须单独复跑。前置自检把该类修改收敛为标准流程步骤。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0019
- type: R
- title: gate 1 S1-PLACEHOLDER 模板变量正则对中文数学集合记法误判——含分隔符的 {零, 有限真实数据} 类花括号集合被判「疑似模板变量未替换」造成 Stage 1 无效重试
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 1 两次 runtime fail：2026-09-07T12:44:47Z〔2960s〕与 13:47:06Z〔1518s〕，stage_retry_count 1→2）
  - .agents/tools/gate_lint.py#L36（TEMPLATE_BRACE_RE 定义）+ #L208-215（S1-PLACEHOLDER 判定）
  - 正则行为机械复现（2026-09-07 蒸馏会话）：`TEMPLATE_BRACE_RE.finditer("槽位尾行状态恒 ∈ {零, 有限真实数据}")` 命中 `{零, 有限真实数据}` → fail；误判文本出现在被门禁拒绝的中间版本上（归档 design_v0/v1/DESIGN.md 已无中文花括号——修订后通过）；发生账户来自 conductor 终态钩子输入（复盘工件未覆盖该两次 fail 的 gate 侧原因，同时构成复盘缺口）
- repro: python3 -c "import re; print([m.group(0) for m in re.finditer(r'\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}', '恒 ∈ {零, 有限真实数据}')])"
- toolchain_stamp: gate_lint.py 现行版本（statectl gate 1 机械核对层）；误判发生于 2026-09-07 expert 迁移任务
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（替换一行正则定义）
    定位锚: 'TEMPLATE_BRACE_RE = re.compile(r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")'
    old 文本: |
      TEMPLATE_BRACE_RE = re.compile(r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")
    new 文本: |
      # CJK-brace groups that look like unfilled template slots. Set/math
      # notation with CJK members and list separators (e.g. "{零, 有限真实数据}")
      # is legitimate prose, not a template variable: skip such groups.
      TEMPLATE_BRACE_RE = re.compile(r"\{(?![^{}\n]*[,、;；∈≤≥])[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")
    动机: 负向前瞻排除含列表分隔符/数学关系符的花括号组——已验证 `{零, 有限真实数据}`/`{保留、舍弃}` 不再命中，`{算子名称}`/`{如: xxx}` 类真模板变量仍命中；本任务 Stage 1 两次 runtime 重试中 gate 误判贡献了无效重试（设计文档的集合记法是合法内容）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0020
- type: R
- title: gate 1 S1-REF-PATHS 路径存在性核对未剥离 `#`/`::` 锚后缀与省略号——`DESIGN.md::修订记录`/`.py#L345` 类合法引用被判「引用的仓库路径不存在」
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（同 VP-2026-0019 的两次 Stage 1 runtime fail）
  - .agents/tools/gate_lint.py#L64-84（_ref_path_failures：REF_PATH_RE 的排除字符类不含 ASCII `:`/`#`，token 连同锚后缀一起做 os.path.exists 判定）+ #L42-44（REF_PATH_RE 定义）
  - 正则行为机械复现（2026-09-07 蒸馏会话）：`examples/.../DESIGN.md::修订记录` 与 `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py#L345` 两个 token 均含锚后缀导致 exists 判 false；发生账户同 VP-2026-0019（conductor 终态钩子输入 + 复盘缺口）
- repro: python3 -c "import re; print([m.group(0) for m in re.finditer(r'(?:docs|examples|testing|\.agents)/[^\s\`\x27\"<>()\[\]（）【】：，。；、！？]+', '见 examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md::修订记录')])"
- toolchain_stamp: gate_lint.py 现行版本；误判发生于 2026-09-07 expert 迁移任务
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（替换 token 清理两行）
    定位锚: |
      (于 _ref_path_failures 函数体内)
          token = m.group(0).rstrip(".,;:!?、。】")
    old 文本: |
          token = m.group(0).rstrip(".,;:!?、。】")
    new 文本: |
          # Strip anchor suffixes (path#heading / path::section) before the
          # existence check; elided paths (examples/foo/...) are not concrete.
          token = re.split(r"#|::", m.group(0), maxsplit=1)[0].rstrip(".,;:!?、。】")
          if token.endswith("...") or "/..." in token:
              continue
    动机: 设计文档以 `path#heading`/`path::section` 引用仓库工件是合法引证形态（本任务 DESIGN/REVIEW 多处使用）；现行正则把锚后缀并入 token 判不存在——已验证修复后 `DESIGN.md::修订记录` → `DESIGN.md`、`.py#L345` → `.py` 正确判定，尾部省略号路径 rstrip 后落到父目录存在性判定。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0021
- type: R
- title: Stage 3 大工件会话超限空返回的防再犯——DESIGN.md 分段读取 + kernel 分段落盘纪律写入 developer agent 定义（两次空返回各耗 1579s/930s）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 3 attempt 1 fail 1579s / attempt 2 fail 930s，verdict=runtime；attempt 3 complete 7701s）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（标题行：「前两次会话超限空返回，本次完成」）
  - 第三次成功的「分段读取 + 增量写入」纪律账户来自 conductor 终态钩子输入（复盘工件未展开该纪律细节——provenance 如实标注）
- repro: 复现条件——一次性整读 175KB/1249 行 DESIGN.md + 单次巨型 Write 949 行 kernel 文件的子 Agent 会话（2026-09-07 本任务前两次 attempt）
- toolchain_stamp: opencode Subagent 会话限制（2026-09-07 两次实证）；tilelang 工具链无关
- target_doc: .opencode/agents/tilelang-op-developer.md
- delta: |
    动作: update（first_impl 模式首行 bullet 扩展）
    定位锚: "- Read `DESIGN.md` + `REVIEW.md`。"
    old 文本: |
  - Read `DESIGN.md` + `REVIEW.md`。
    new 文本: |
  - Read `DESIGN.md` + `REVIEW.md`。**大工件分段读写纪律**（会话超限空返回防再犯，2026-09-07 attention expert 任务两次空返回实证〔1579s/930s 白耗〕）：DESIGN.md 超过 ~1200 行 / 150KB 时按章节分段 Read（先目录 + §0.5/§0.6 决策 + §3 伪代码，再按需下钻），不一次性整读；`{op}.py` 生成用分段落盘（逐段写 /tmp 后 cat 合并，或先写骨架再增量追加），不用单次巨型 Write——超限空返回表现为无输出直接失败（runtime verdict），与代码错误同型、无法从 stderr 区分。
    动机: 本任务 Stage 3 前两次 attempt 以完全相同的输入空返回（会话超限），第三次仅改变读写纪律即成功（7701s 完成 949 行 kernel + 29 用例全过）——每次空返回消耗一次 attempt 预算与 ~15–26 分钟墙钟。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0022
- type: R
- title: 超长 DESIGN.md（>60KB）单次 Write 因 JSON 体积截断——tilelang-op-design SKILL.md Phase 4 补分段落盘规则
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1（Skill Flow Issues 首行：~1200 行 DESIGN.md 单次 Write 截断失败，5 段 /tmp 拼接后 cat 合并才成功；DESIGN.md 1235/1249 行为证）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 1 attempt 1 fail 2960s——含该 Write 截断的恢复成本）
- repro: 复现条件——单次 Write 输出 >60KB 的 DESIGN.md（2026-09-07 本任务首设计 attempt）
- toolchain_stamp: opencode Write 工具 JSON 体积限制（2026-09-07 实证）；tilelang 工具链无关
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 首行后追加一段）
    定位锚: "基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节："
    old 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：
    new 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：

      > **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。
  动机: 截断失败发生在长任务收尾（写盘即交付前），恢复成本一次完整 attempt；attention/mixed 类算子的 DESIGN 普遍超此规模。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0023
- type: R
- title: DESIGN 伪代码单一权威源规则——伪代码与实现注/设计步骤矛盾时必须先消歧再冻结（单边权威声明不够）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev1（VP 行 4：v0 §3.3 伪代码 α 行 vsub(ub_m, ub_mcur) 与实现注①「快照式 m_prev」矛盾——修复 = 伪代码改写为注的形态而非加「以注为准」标记）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（Skill Flow Issues Design-internal 行：§3.3 伪代码 mask→softcap 与 E3/§3.1 softcap→mask 矛盾——照抄伪代码会破坏 P(OOB)=0 不变式，实现被迫自行消歧）
- repro: 复现条件——含伪代码 + 实现注双源且顺序/操作数不一致的设计文档（本任务 v0 两处实证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 章节列表后、Phase 5 前插入附注）
    定位锚: |
      11. 交付清单

  ```
  ### Phase 5：质量自检
  ```
  old 文本: |
      11. 交付清单

  ```
### Phase 5：质量自检
```
  new 文本: |
      11. 交付清单

      > **伪代码单一权威源**：§3 伪代码即权威实现形态，实现注只解释不纠偏——两者矛盾时修正伪代码本身（改写为注的形态），而非加「以注为准」标记；伪代码与其他章节（设计步骤/论证）多源矛盾时必须先消歧再冻结——单边权威声明不够（2026-09-07 attention expert 任务实证：mask/softcap 顺序矛盾若照抄伪代码会破坏 P(OOB)=0 不变式，softcap(−1e38)≈−softcap 非精确 0）。

```
### Phase 5：质量自检
```
  动机: 同一设计文档两处伪代码-注/步骤矛盾（α 行操作数顺序、mask/softcap 顺序），一处靠检视拦截、一处漏到 Stage 3 由实现者自行消歧——冻结前的消歧义务是缺口。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0024
- type: R
- title: tilelang-op-design Phase 6 修订纪律三条——多路线裁决三步法+正确性面第四维 / 公式修复「公式-论证-自述」三同步 / 概念删除与计数预算标注 grep 清零判据
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md → design_v1.md → DESIGN.md（v0→v1→v2 修订链）+ DESIGN.md 修订记录（v1/v2 两节「为何不会再犯」）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev1（路线裁决三步法 + 第四维 + grep 清零 + 三同步各 VP 行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev2（三同步模板 + 计数/预算标注同步义务：「8 参」「6 输入」需带边界 pattern 防子串误命中）
- repro: 复现条件——任一 Stage 2 不通过后的针对性修订（本任务 2 轮修订、牵连面 10+ 章节实证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境同上
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 6 小节扩展）
    定位锚: |
      ### Phase 6：针对性修订

  ```
  仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
  ```
  old 文本: |
      ### Phase 6：针对性修订

  ```
仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
  ```
  new 文本: |
      ### Phase 6：针对性修订

  ```
  仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

  检视不通过后的针对性修订（revision mode）附加纪律（2026-09-07 attention expert 任务 v0→v1→v2 修订链实证）：

1. **多路线修复必须写明路线裁决**：检视给出多条修复路线时，修订版写明裁决依据与回退设计，不默认选改动面小的路线。裁决步骤：① 证据分级（文档规格 > 前序设计断言 > 形态缺口——缺口降级为「未实证假设」而非「不支持」）；② 结构收益量化（如直连使 dtype 同构、删 flag/ws 族）；③ 回退完整性（被否决路线降级为 R-x 回退而非删除）；④ 正确性面作为独立第四维（flag 族数、未初始化域、握手竞态数随路线的削减）。
2. **公式修复的「公式-论证-自述」三同步**：修正分界/阈值/取整公式（ceildiv↔floordiv 类）时同步恢复依赖该公式的整条下游论证链（保守性论证 / 收益前提 / 量化自述占比）——只替换公式 token 而不同步论证会保留旧矛盾或制造新矛盾；修复前执行代数重推 + 整除/不整除双数值例。
3. **概念删除与计数/预算标注的完成判据 = token grep 清零**：删除某概念（API/buffer/flag/参数）或切换预算基准后，以旧 token（含「N 参」「N 输入」「NKB」类标注）全文 grep 清零为完成判据（子串误命中用带边界符号的精确 pattern；合法语境白名单 = 回退方案/修订记录/历史版引用）。
```
  动机: 本任务两轮修订中：路线裁决无模板（自行三步法完成）、v1 删参未同步计数（第 2 轮问题 2/3）、K_A 修复若无三同步会留下 §5.4 占比与 M5 前提的内部矛盾——三条纪律均已被 3 轮检视逐项验证有效。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0025
- type: R
- title: 弃选论证证据规则补第 6 条「前序实证的证据边界」——归档工件「N/M 实证通过」只证明实测路径可行，不得外延为替代路径不可行
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2（问题 2：上一轮 23/23 载体链实证被用作「直连 bf16 不可行」的证据；负向断言「T.gemm bf16 ×」两轮原样继承且引用同一文档——两轮均未打开文档核对表格）
  - docs/Tilelang.language/线性代数操作/T.gemm.md#§2.2.1/§2.3（实际 bf16 √ + 实测可用——断言被推翻的对照证据）
  - examples/multi_head_attention/_prev_task_20260904_developer/_gqa_prefill_fwd_kernel/debug_log.md（前序实证的原始出处——provenance 交叉核对）
- repro: pytest testing/npuir/bf16_support_ops/test_gemm_bf16.py（被误读为「bf16 不可行」的反例证据链复核）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07 两轮设计 + 本轮复核）
- target_doc: .agents/skills/_shared/standards/negative-claim-evidence.md
- delta: |
    动作: update（§2 规则要点追加第 6 条）
    定位锚: "，与实测矛盾即 fail。"
    old 文本: |
      ，与实测矛盾即 fail。
    new 文本: |
      ，与实测矛盾即 fail。
      6. **前序实证的证据边界**：归档工件/前序任务的「N/M 实证通过」只证明其实测过的实现路径可行，不得外延为替代路径不可行——引用前序结论做负向断言时降级为「该路径已验证」并为替代路径另行举证（文档表格/测试/先例）；从前序任务继承的负向断言同样必须亲自打开所引文档复核到具体表格行/限制条款（「引用了正确路径、转述了相反内容」= 两轮设计同款失效形态，2026-09-07 attention expert 任务 bf16 断言实证：T.gemm.md §2.2.1 实为 bf16 √，前序断言驱动了整条载体链重设计后被推翻）。
    动机: bf16 负向断言跨两轮任务存活并驱动 E5 全链设计（载体链 +32KB UB、2 个 flag 族、Vector 载体负载），本轮文档+测试亲核推翻后直连形态 bf16/fp16 延迟差 ≤0.2%（pattern-library §1.8）——继承断言不亲核的代价是整条设计路线。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0026
- type: R
- title: design-review 维度 8 补「公式与参数化标量验证」强制块——⟺ 分界声明代数重推+双数值例 / 量化自述对账 / per-lane 双端点代入 / 继承负向断言亲核
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 2 轮问题 1：K_A ceildiv 假等价在 v0/v1 两轮存活——`块 n 全有效 ⟺ n < ceildiv(a, bn)` 的 +1/−1 补偿直觉使公式「看起来对」；8 个 bx 数值枚举 design−precise=1、免 mask 域违例 2016 位置/任务）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2（VP 行：自述即测试向量——§5.4「每任务 1–2 块走 mask」与公式联立 NK−K_A∈[1,2] 直接暴露公式错误）+ #Stage-2-r1（per-lane 双端点行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md#L221（同款公式两轮存活证据）
- repro: 复现条件——任一含「块级分界 + 条件跳过」的 DESIGN 检视（causal K_A / sliding window / 守卫域判定）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07，3 轮检视）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 引言 blockquote 追加一段）
    定位锚: "不得只复算数字而继承设计的前提。"
    old 文本: |
      **弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**——打开其引用的 API 文档核对代价机制（如 repack 是核内 UB 级还是 GM 级），不得只复算数字而继承设计的前提。
    new 文本: |
      **弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**——打开其引用的 API 文档核对代价机制（如 repack 是核内 UB 级还是 GM 级），不得只复算数字而继承设计的前提。**公式与参数化标量验证（强制）**：① 一切「⟺」分界/阈值/取整类声明（causal 分界、sliding window 边界、守卫域判定）执行代数重推 + 整除/不整除双数值例（a=65/bn=64/n=1 类反例）——ceildiv↔floordiv 差 1 的补偿直觉使公式「看起来对」，文字论证无法发现（K_A 反例两轮存活）；② 设计文档内的量化自述（块数/占比/频次）是与公式对账的不变式，自述-公式联立矛盾即线索；③ AIV/lane 半分类的参数化标量阈值代入两个 lane 端点数值例（vid=0/vid=1——单 AIV 思维推导在另一 lane 符号方向隐蔽出错）；④ 从前序任务/归档工件继承的负向断言必须亲自打开所引文档复核到表格单元格——前序实证只对其实测过的路径有效。
    动机: 本任务 3 轮检视中 K_A 假等价公式在第 1 轮维度 0/8 下 pass（文字论证自洽）、per-lane 符号错误靠 smoke 数值例才发现——四项核对均为「文字论证全对、公式错」形态的最低成本拦截器。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0027
- type: R
- title: 检视报告修改建议中的替代公式/参数写入前必须先经独立数值验证——检视建议是 Stage 1 revision 的逐字输入，建议错误的修复成本高于漏检
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 1 修改建议③把 ceildiv 版 K_A 误背书为「保守下界」；第 2 轮问题 1 代数重推 + 反例 c=64/bn=64/n=1 + 8 个 bx 数值枚举确证为假等价——ceildiv 对 vid=0 是激进上界）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#v2-仲裁说明（v1 曾把建议③逐字固化进「K_A 保守性说明（防 Stage 3 误改）」——错误由「待核公式」升级为「已论证事实」）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2（Skill Flow Issues 首行 + Transferable「检视者对自己前轮输出的建议负有验证义务」）
- repro: 复现条件——检视报告给出具体替代公式/取整方向且被修订版逐字采纳（本任务 v1 实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 4 追加一段）
    定位锚: "按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。"
    old 文本: |
      按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。
    new 文本: |
      按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。

  ```
  修改建议中给出的具体替代公式/参数/取整方向，写入前先经独立数值验证（代数重推或数值例）——检视建议是 Stage 1 revision 的逐字输入，建议自身携带错误会被修订版固化升级为「已论证事实」（2026-09-07 attention expert 任务实证：第 1 轮建议③把 ceildiv 版 K_A 误背书为「保守下界」，v1 逐字固化，第 2 轮代数重推 + 8 个 bx 数值枚举才确证为假等价）；检视者对自己前轮输出的建议负有与设计同等的验证义务——重审时必须重推建议本身的正确性，「建议被执行了」与「建议的目标达成了」是两回事。
```
  动机: 建议③误背书使错误公式获得检视权威加持、跨 v0/v1 两版存活且被「防误改」说明保护——该形态比漏检更难被发现，写入前验证是唯一低成本拦截位。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0028
- type: R
- title: design-review 修订版重审工作流——复用白名单/公式重推黑名单 + diff 锁定深查范围 + 新引入 API 亲验 + 建议原文逐句验收清账 + 三态判定
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 3 轮：v1→v2 diff 仅 8 处与修订记录声明一致；「复用 R2」标注 + K_A 全链重推策略；问题 4 三态判定「核心诉求成立 + 残留建议级」实践）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2/r3（复用白名单/重推黑名单行、diff 全量比对法、不变量对账清单、API 亲验行、原建议文本即 requirements 行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v1.md（diff 比对基准）
- repro: 复现条件——Stage 2 第 N 轮（N≥2）修订版重审（本任务 R2/R3 两轮实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 2 追加重审规则）
    定位锚: "按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。"
    old 文本: |
      按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。
    new 文本: |
      按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。

  ```
  修订版重审（第 N 轮，N ≥ 2）附加规则（2026-09-07 attention expert 任务 3 轮检视实证）：
- **复用白名单 / 重推黑名单**：前轮已验证的事实核对类结论（文档原文/源码行号/路径存在性/数字复算）可复用（标注「复用 R{N-1}」）；一切分界/阈值/取整公式与保守/激进的定性判断无论修订是否触碰均须重推——修订未触碰的段落不等于正确的段落（K_A 公式 v0/v1 两轮存活，唯一新阻塞恰在「复用区」）。
- **diff 锁定深查范围**：对「其余内容逐字保持上一版」类声明，以 `diff design_v{N-1}.md DESIGN.md` 全量比对验证——既验证声明属实（无未声明改动的隐性风险面），又自动锁定本轮深查范围（diff 命中处 = 必查）。
- **修订版新引入的 API 一律亲验存在性**（伪代码/映射表中首次出现者）——设计自述的「实查/亲验」记录不构成佐证，重跑同款 3 行验证才是检视证据。
- **修复验证对照建议原文逐句清账**：原建议文本是修复验收的 requirements 文档（每个验收子句独立判定达成/残留）；多目标建议部分达成时判「核心诉求成立 + 残留建议级」三态（记维度 warn + 修改方向），不整体放行也不整体阻塞。
```
  动机: 本任务 R2 漏检根因 = 复用策略未区分结论类型（事实核对 vs 推演判断）；R3 以 diff 锁定 + 公式全量重推 + 三态判定完成收口——规则均经实战验证。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0029
- type: R
- title: design-review 维度 5 新增「flag 协议安全性」检查行——per-slot flag 逐族传递性推演法（写 i+slots 前读 i 已完成）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§6.3（三族 + 单槽 + 跨任务五条传递性论证）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（维度 5 附核对记录：本轮对五条论证独立推演全部成立）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2（VP 行 5：推演法可直接复用）
- repro: 任一 per-slot flag + 多槽 workspace 的 Expert 设计检视
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 5 表追加一行）
    定位锚: "| 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |"
    old 文本: |
      | 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |
    new 文本: |
      | 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |
      | flag 协议安全性 | per-slot flag + 多槽 workspace 设计按「写 i+slots 前读 i 已完成」逐族传递性推演（set 的程序序前驱 → wait 的后继），跨任务握手（TASKDONE 类）单列核对（2026-09-07 attention expert 任务 §6.3 五条论证推演法） |
    动机: Expert 跨引擎流水的正确性核心在 flag 协议而非循环结构——维度 5 原三行（循环/同步模式/尾块）不覆盖 flag 族传递性，本轮靠检视者自发推演完成，方法已验证可复用。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0030
- type: R
- title: design-review 维度 7 新增「统计行对账」检查行——四态/多态处置表的统计行与表体处置列做分项-标签逐行比对（不只对总数）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 3 轮维度 7 warn：统计行分项归类与表体处置列标签 2 处错位——表体字面分布 6/7/6/4 vs 统计 5/8/5/5，跨 v0/v1/v2 三版存活）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r3（Skill Flow Issues 首行：前三轮检视只数总数 23=23，未做分项-标签逐行比对）
- repro: 复现条件——含四态/多态分类统计行的设计文档（本任务 §0.5 处置表三版存活实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 7 表追加一行）
    定位锚: "| 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |"
    old 文本: |
      | 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |
    new 文本: |
      | 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |
      | 统计行对账 | 四态/多态处置表的统计行与表体处置列做分项-标签逐行比对（不只对总数）——总数对齐但分项归类错位（6/7/6/4 vs 5/8/5/5）在 v0/v1/v2 三版存活，仅靠总数对账无法发现（2026-09-07 attention expert 任务实证） |
    动机: 总数对账只能发现双计类错误；分项归类与标签的错位需逐行比对——低成本高收益动作，防止错误统计被后续章节继承。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0031
- type: R
- title: conductor 修订调度 prompt 附加仲裁规则——检视建议跨轮冲突时显式标注「以最新轮为准 + 禁止恢复项」；design_error_summary 位置冲突以 REVIEW.md 为准
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#v2-仲裁说明（「以第 2 轮报告的 floordiv 公式为准，严禁按第一轮检视建议③的字面恢复」——本轮 conductor 调度输入显式指令防止了修订者回退到已推翻建议）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev2（Skill Flow Issues 两行：仲裁输入行 + design_error_summary 位置引用笔误行——「M5 收益前提（§1.6.2 #5）」实位于 §1.6.1）
- repro: 复现条件——Stage 2 多轮检视且后轮推翻前轮建议（本任务 v0→v1→v2 实证）
- toolchain_stamp: 流程规则（conductor 调度层）；证据环境 2026-09-07
- target_doc: .opencode/agents/tilelang-op-conductor.md
- delta: |
    动作: update（设计修订循环步骤 3 追加仲裁规则）
    定位锚: "3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。"
    old 文本: |
      3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。
    new 文本: |
      3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。修订调度 prompt 附加仲裁规则：① 前轮检视建议被后续轮次推翻时，显式标注「以最新轮为准 + 禁止恢复项」（如「严禁按第 1 轮建议③字面恢复」）——无此指令时修订者可能因「上一轮建议已被执行过」而复用错误公式（2026-09-07 attention expert 任务实证：误背书的 ceildiv 建议曾逐字固化进 v1）；② `design_error_summary` 的位置引用与 REVIEW.md 位置列表冲突时以 REVIEW.md 为准（或摘要只引问题编号不重复位置）——摘要位置笔误会误导修订定位。
    动机: 检视建议跨轮冲突时修订者面对两个「检视权威」；显式仲裁输入是被实证有效的防回退机制（本任务 v2 修订记录明示该指令起了作用）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0032
- type: R
- title: integrate_kernel.py 补 stale perf_opt lineage 告警——换代重集成不清理上一任务 perf_opt/，占位符会静默解析到旧谱系调优 kernel（工厂签名兼容、测试不拦截）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md（§Bench observations 末条 + Files 清单：perf_opt/ 保留的 developer 谱系产物〔mtime 09-07 04:51，43KB〕早于 expert 基线 kernel〔09-07 17:22，52KB〕）
  - examples/multi_head_attention/RETROSPECTIVE.md#Stage-5（Skill Flow Issues 行 + Transferable 首条）
- repro: 换代重集成后 `ls -la tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/` 对比基线源 mtime——本任务实测旧谱系早 12.5 小时
- toolchain_stamp: tilelang dev build 21586b5（2026-09-07）+ CANN 8.5.0；TileOPs pyproject 环境
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（main 复制循环内追加谱系检查）
    定位锚: |
      (main() 的 sources 复制循环内)
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
    old 文本: |
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
    new 文本: |
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
          perf_dst = target_dir / "perf_opt" / src.name
          if perf_dst.exists() and perf_dst.stat().st_mtime < src.stat().st_mtime:
              print(
                  f"[warn] stale perf_opt lineage: {perf_dst} predates the "
                  f"baseline source {src}; it likely derives from a previous "
                  f"task lineage -- verify lineage (mtime / origin task) "
                  f"before activating the perf_opt import"
              )
    动机: 换代重集成（developer→expert）整文件覆盖基线但保留旧 perf_opt/；wrapper 的 perf_opt import 占位符切换后会解析到旧谱系 kernel——工厂签名兼容、smoke/full/bench 均不拦截，只有谱系核对能发现（本任务实测两谱系相差 12.5 小时与 9KB/640↔949 行）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0033
- type: R
- title: op_template.py 的 _run_case 加长签名传参规范注释——位序敏感 case 参数（≥4 个或多个同类整型）用 kwargs 或 case 元组，防 heads↔seq 写反
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（Test-harness 行：6 参位序 (batch, heads, heads_kv, Sq, Skv, dim) 在 13 处手写调用中写反 2 类〔heads↔seq〕，1 次 SDPA GQA 崩溃 + 1 次 ValueError；全 MHA 位形静默通过、GQA 位形才炸）
  - /tmp/opencode/full_run2.log#L48（RuntimeError: The size of tensor a (1024) must match the size of tensor b (16)——位序写反的崩溃实录，session-local）
- repro: 复现条件——长位序 case 签名手写调用 ≥10 处（本任务 13 处写反 2 类实证）
- toolchain_stamp: 测试工程规则（无运行时依赖）；环境 torch_npu 2.x + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-develop/templates/op_template.py
- delta: |
    动作: update（_run_case 定义行前追加规范注释）
    定位锚: "def _run_case(M, N, dtype, tag):"
    old 文本: |
      def _run_case(M, N, dtype, tag):
    new 文本: |
      # Case-signature rule: when a kernel's case parameters grow beyond ~4
      # positional args, or contain several same-type integers (shape / dims /
      # head counts), pass them as kwargs (or wrap a case tuple) at every call
      # site -- transposed positional args (heads<->seq) pass silently on
      # all-MHA shapes and only crash on GQA/dispatch variants (2026-09-07
      # attention expert task: 2 of 13 hand-written calls transposed).
      def _run_case(M, N, dtype, tag):
    动机: 位序写反在 GQA 位形（heads≠heads_kv）才暴露，全 MHA 测试全绿掩盖错误——模板级规范使新算子测试套件从第一处调用就免于此类浪费。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0036
- type: R
- title: optimize SKILL.md Phase 2 步骤 7 补 blocked/invalid 分支归宿——perf_records.jsonl 仅收录完成验证与测量的分支（duration_us 恒为实测数、不得 null），结构性不可行证据归 opt_log invalid 行
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Iteration-3-注记（gate 4 窄修复记录：4 个 blocked-compile 分支〔bn=512 双槽 L1 溢出〕曾以 duration_us: null 写入 perf_records.jsonl，按契约移除、证据归 opt_log invalid_compile 行）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/.task_timeline.jsonl（Stage 4 attempt 1 fail 7546s〔gate S4-PERF-RECORDS-SCHEMA〕→ attempt 2 窄修复 complete 314s）
  - .agents/skills/_shared/standards/signal-registry.md#§5（字段契约：duration_us 类型 number；「每轮每个实验分支验证后追加一行」）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#终局验证-注记（2026-09-09 第二轮同主题实证〔task 20260909T033622Z，并入不新开 id〕：final causal 首行 3779.0 为手抄转录笔误，append-only 纪律下以更正行 3773.39 + opt_log 注记收尾——手抄数字是 null 之外第二类结构化记录失真源）
- repro: 复现条件——调优轮出现 blocked-compile/invalid 分支时的记录行为（本任务 Round 3：bn=512 双槽 4 分支全部编译失败无测量值）
- toolchain_stamp: gate_lint.py 现行 S4-PERF-RECORDS-SCHEMA 规则；发生环境 tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（步骤 7 扩展）
    定位锚: "7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。"
    old 文本: |
      7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。
    new 文本: |
      7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。**blocked-compile / invalid 分支不写入 perf_records.jsonl**——`duration_us` 恒为实测数值、不得为 null（signal-registry §5 契约，gate 4 `S4-PERF-RECORDS-SCHEMA` 强校验）；其结构性不可行证据（报错文本、L1/UB 溢出公式）记入 opt_log 当轮实验分支表的 invalid/blocked 行，证据不丢失。**行内数值一律从 msprof 提取输出直接复制（或管道）生成，禁止手抄**（2026-09-09 第二轮实证：final causal 首行 3779.0 转录笔误，靠 append-only 更正行 + opt_log 注记收尾）。
    动机: 契约的「验证后追加」语义未在写入指令处显式表达负面清单，optimizer 把 blocked 分支也「忠实记录」进结构化文件——2026-09-08 expert Stage 4 实证：4 个 bn=512 双槽 L1 溢出分支误记 null 致 gate 失败一次 attempt（7546s），窄修复又耗 314s；显式归宿规则可省整次重试。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: -

## VP-2026-0037
- type: R
- title: profile-collection.md msprof 输出目录权限规则从叶子目录扩展到父目录——group-writable 父目录下仅 chmod 700 输出目录采集仍被拒（8/8 baseline 首采失败实证）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Skill-Retrospective（BP-R1 proposal：本轮 8 个 baseline 采集全部因 group-writable 父目录失败一次后才修正，前置检查可省一轮）
  - .agents/skills/tilelang-op-optimize/references/profile-collection.md#L96（现行规则仅 chmod 输出目录自身，未覆盖父目录）
- repro: 复现条件——在 group-writable 父目录（如默认 umask 002 的共享 checkout）下新建 msprof 输出目录、仅对叶子目录 chmod 700 后执行 msprof op 采集
- toolchain_stamp: CANN 8.5.0 msprof / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（权限规则行扩展）
    定位锚: "新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。"
    old 文本: |
      新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。
    new 文本: |
      新建 `msprof op` 输出目录后，先确保该目录**及其父目录**（至少直接父目录）group/other 不可写，例如 `chmod 700 {output_dir}` 并同步检查父目录；msprof 的权限校验覆盖父目录——仅叶子目录 700 而父目录 group-writable 时采集仍被拒（2026-09-08 expert Stage 4 实证：8/8 baseline 首采失败，父目录收紧后一次通过）。
    动机: 现行规则只覆盖输出目录自身；本任务 8 个 baseline 采集全部因 group-writable **父目录**失败一次后才修正——预检规则覆盖父目录可直接省一轮重采。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: BP-R1 整合去重：optimizer 自提的「profile-collection.md 补权限预检」经查现行文件已有叶子目录规则（67db6f3 期引入），本任务证据表明其不充分——按 update 形态提案而非 add。

## VP-2026-0039
- type: R
- title: profile-collection.md 补「探针 kernel 测硬件上限」标准形态——复刻目标数据流多模式对照 + 编译报错反推容量 + 真实 kernel 反推双证
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-8（probe_kvbw 四模式〔single/shared/distinct/pp〕同为 ~110.8µs kernel-only → L1 端口 r+w 148–154GB/s/核；bm=64@bn=512 `cc overflow` 报错反推 L0C=128KB；fabric ≥1.73TB/s 聚合）
  - 同文件#Skill-Retrospective-第二轮 有效手法①（硬上限测绘先行——一轮内把设计空间不确定性收敛为可算术的复合地板，直接决定 DESIGN_LIMIT 判定的证据质量）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md（三硬上限构成 [DESIGN_LIMIT] 2.09x/2.27x 缺口归因的实测基础）
  - pattern-library §1.9（数据侧落点：L0C/L1 端口/发射定律/fabric 条目，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z）
- repro: 复刻目标 kernel 数据流（同结构装载/gemm 模式）的最小探针 + 编译期容量压限（`overflow, requires N bits while M bits available` 报错文本）+ 真实 kernel msprof 计数器反推（流量÷时间）三方对照
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（文件末尾追加一节）
    定位锚: |
      返回：

  ```
  ```text
  PERF_DATA_COLLECTED
  ```
  ```
  old 文本: |
      返回：

  ```
  ```text
  PERF_DATA_COLLECTED
```
  ```
    new 文本: |
      返回：

  ```
```text
  PERF_DATA_COLLECTED
```

---

## 6. 探针 kernel 测硬件上限（DESIGN_LIMIT 证据 / 设计空间测绘）

  适用：调优目标涉及硬天花板判定（设计层归因、[DESIGN_LIMIT] 反馈、结构方向投入前的可行性测绘）时，在迭代候选之外用探针 kernel 把硬件/编译器上限测成可算术的数字。标准形态（2026-09-09 attention expert 第二轮实证：一轮测绘直接支撑 DESIGN_LIMIT 判定）：

  1. **复刻目标数据流的最小探针**：与目标 kernel 同结构的装载/gemm/访问模式，多模式对照（single / shared / distinct / ping-pong 双槽）区分 fabric、端口、容量因子——四模式同耗时 ⇒ 端口读写串行（双槽不能解除），非 fabric 争用。
  2. **编译失败报错反推容量上限**：刻意压限触发 BishengIR `overflow, requires N bits while M bits available` 报错（L0C cc / L1 cbuf / UB），从报错文本反推硬容量与 bm/bn 联合约束式。
  3. **真实 kernel 计数器反推双证**：探针值须与真实 kernel 的 msprof 计数器反推值互证（如 L1 r+w 流量 ÷ (mte2+mte1) 时间 = 端口带宽）；单源探针结论降级为待证。
  4. **探针纪律**：跨引擎探针必须带 Cube→Vector flag 同步（缺同步的数值崩坏是探针自身竞态而非 API 缺陷，pattern-library §1.9）；探针 kernel 保留 ≥1 个输入 tensor（§2 零输入行）。
  5. **归宿**：探针结论回写 pattern-library §1（数据侧）；触及编译器能力缺口时登记 `.agents/evolution/capability-gaps.md`；探针 kernel 脚本与日志入 `perf_opt/logs/`（jsonl 不收录非本算子分支的探针行，证据存日志与 opt_log 当轮表）。
```
  动机: DESIGN_LIMIT 判定与结构方向裁决的证据质量取决于硬上限数字的可信度；本轮 round 8 以该形态一轮完成 L0C/L1 端口/fabric 三项测绘 + 复合地板算术，直接封死无效调参方向（bm 上调/深流水/两遍 softmax）并支撑 2.09x/2.27x 缺口归因——无此形态需多轮试错反推。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0040
- type: R
- title: iteration-diagnosis.md「已知迭代结论」补续调任务前置基线复核——上轮交付驱动完整性（手改漂移检测）+ 基线重采对账（偏差 ≤ 噪声阈值）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#第二轮-前置修复（上轮终版测试驱动被用户手改：run_fa_tuned 仅余 fa4096、main 绕过 --level 分派；恢复后 `--level all` 全过〔logs/round8/driver_restore_level_all.log〕——手改漂移重新验证 ✓）
  - 同文件#Round-8-Diagnosis-1（基线复核：4 case 重采 18.19/29.69/77.62/263.93µs，与上轮终值记录偏差 ≤0.4%——谱系/环境一致性验证后才开调）
- repro: 复现条件——任一以「上轮终值为基线」的续调 optimize 任务（用户在手改过的 perf_opt 交付物上发起第二轮）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「已知迭代结论」小节扩展）
    定位锚: |
      ### 已知迭代结论

  ```
每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
  哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
  本轮 profile 相比上一轮 profile 哪些指标发生变化
  ```
  ```
  old 文本: |
      ### 已知迭代结论

  ```
  每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
  哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
  本轮 profile 相比上一轮 profile 哪些指标发生变化
```
  ```
  new 文本: |
      ### 已知迭代结论

  ```
  每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
本轮 profile 相比上一轮 profile 哪些指标发生变化
```

**续调任务（基线 = 上轮终值）的额外前置**（2026-09-09 attention expert 第二轮实证）：开调前完成两项复核——① 上轮交付测试驱动完整性：确认 run/case 分派逻辑与上轮终版一致（用户可能手改交付物——实证：run_fa_tuned 仅余 1 case、main 绕过 --level 分派；恢复后 `--level all` 全过再开调）；② 基线重采对账：重采上轮终值 workload 与上轮记录对账，偏差 ≤ 噪声阈值（实证 4 case ≤0.4%）才算谱系/环境一致，超阈值先查谱系与环境再开调。
```
  动机: 续调任务若直接信任被手改的测试驱动或未复核的基线，后续所有测量与归因都建立在失真地基上；本轮前置修复 + 重采对账各耗数分钟，拦截的是整轮方向的偏移风险。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0041
- type: R
- title: iteration-diagnosis.md 补「参考实现对照（morph ladder）」标准定位法——结构移植后性能不达预期时在参考实现上逐级叠加契约增量二分定位（op 级无罪 ≠ 组合无罪）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Skill-Retrospective-第三轮（有效手法① morph 阶梯决定性 + skill 流程问题①：形态学排查消耗 4 个 no_gain 分支后才启动，应更早切换；流程问题②：第二轮 [DESIGN_LIMIT] 地板归因未做同 contract 最小增量对照——有则 -20% transpose 税第二轮即被发现）
  - 同文件#Round-11-Diagnosis（morph 14 点矩阵：MORPH=0 参考原版 95.8 → MORPH=1 4D BSHD 输入 96.1 无罪 → MORPH=2c +lse 254.5 爆炸 → write-only 97.2 无罪 → 单 op 全无罪（vlog2/vmul+vadd/死源 transpose/4 个平凡 vadd）→ 任意含「活跃源 transpose」组合全部 253–255——唯一毒化点定位）
  - pattern-library §2 活跃源 transpose 毒化行（组合阈值效应数据侧，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md 修正附录（负结论推翻闭环的同款方法论缺口）
- repro: 复现条件——任一「结构移植/重写后性能不达预期（>1.3x）且仓库内存在同族参考实现」的调优轮（本轮实证：v11 初版 250.68 vs 参考 95.8–99µs，逐字转录参考 body 仍 255.53——上下文级差异形态学排查完全无效）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「### 代码结构观察」小节末条之后、「### 已知迭代结论」标题之前插入新小节）
    定位锚: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

  ```
### 已知迭代结论
  ```
    old 文本: （即上述定位锚原文）
  new 文本: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

  ```
### 参考实现对照（morph ladder）

  仓库内存在同族参考实现（`examples/` 同 API 结构先例）且本方 kernel 结构移植/重写后性能显著不达预期（>1.3x）时，在逐项排查 op 形态 / pass_configs / 装载写法（含逐字转录参考 body）均无差异后，**优先切换 morph 阶梯定位，不在形态学排查上继续消耗实验分支**（2026-09-09 attention expert 第三轮实证：4 个形态学分支 no_gain 后 morph 阶梯一轮定位毒化点；且 [DESIGN_LIMIT] 地板归因若做此对照，-20% transpose 毒化税在第二轮即被发现）：

1. 在参考实现上逐级叠加本方契约增量（输入布局 → 输出契约 → 逐 op），每级一个实测点，二分定位差异源——**单 op / 单开关的无罪结论不可外推为组合无罪**（编译级阈值效应：活跃源 transpose 单 op 无罪、死源无罪、4 个平凡 vadd 无罪，任意含活跃源 transpose 的组合全部 2.6x 劣化，pattern-library §2）。
2. 对照点分级：契约级（输入/输出布局）先于 op 级；write-only 探针（只写不算）区分「计算成本」与「数据流形态触发」。
3. 纪律：perf-only 探针（故意破坏输出只测性能）在记录中显式标注；morph/探针脚本与实测点矩阵记入 opt_log 当轮（perf_records.jsonl 不收录非本算子 dispatch 的探针行）；scratch 探针结论须镜像回 opt_log 才可作为证据引用。

### 已知迭代结论
```
  动机: 「逐字转录参考仍慢 2.6x」类上下文级差异只有 morph 阶梯能定位（形态学排查对该类完全无效）；组合阈值效应使 op 级排除法结构性漏检——两条均指向该方法应为标准动作而非末位手段。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0042
- type: R
- title: optimize SKILL.md [DESIGN_LIMIT] 判定补「结构不可达」证据门槛——仓库内同族先例检索 + 同口径实测 + morph 对照排除局部地板误判 + 被推翻时修正附录机制（append-only）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md 修正附录（第二轮「对 H=1 大 S 硬目标不可达、fa4096 结构性地板 ≈125–131µs > 100µs」被第三轮同 API 两相位结构推翻：四目标全达成 12.94/18.11/30.88/98.05µs；两处根因 = ① Stage 1 算法调研漏掉仓库内两遍式 S/P 物化先例 examples/flash_attention/flash_attn_npuir.py（同 API 可达结构）② round-10 否决两遍式的「S 需全任务驻留 → flag 预算超 15 上限」系推理错误——全局 [S,S] ws + n-block 下标 flag（nk≤16）即可成立，参考与本轮终版为直接反例）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§11.3（[DESIGN_LIMIT] 结论修正段：局部地板 vs 绝对地板界定——L1 端口 ~150GB/s / L0C 128KB / store_fixpipe 仅 GM 三硬上限存活，被推翻的仅是「上限内不存在达标数据流」的推断）
  - 同目录 perf_opt/opt_log.md#Final-Summary-第三轮（[DESIGN_LIMIT] 修正回填记录）
  - 交叉引用：queue VP-2026-0025（弃选举证边界——设计期同族规则）、VP-2026-0041（morph ladder——最小增量对照的定位方法）
- repro: 复现条件——任一触发 [DESIGN_LIMIT] 双门槛（设计层归因 + >2x 结构性估计）且拟断言「当前工具链/硬件上限内不可达」的调优终局（实证：第二轮断言后第三轮被同 API 结构推翻）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 3 第 4 条末尾追加证据门槛段）
    定位锚: "4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema 产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema 产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。**「当前工具链/硬件上限内不可达」类断言的证据门槛**（2026-09-09 attention expert 三轮闭环实证：第二轮 [DESIGN_LIMIT] 断言 fa4096 结构性地板 ≈125–131µs > 100µs 目标，第三轮两相位重构同 API 达标 98.05µs 推翻）：① 断言前强制检索仓库内同族先例（`examples/`/`testing/` 是否已有同 API 可达结构），找到先例时同口径实测 + morph 式最小增量对照（iteration-diagnosis.md「参考实现对照」）排除「冻结设计族局部地板」误判——实测外推的地板只对实测过的结构族有效（前序实证证据边界，见 negative-claim-evidence.md 第 6 条 / queue VP-2026-0025）；② 否决备选结构的理由须为实测或文档条款，纯推理否决（如「S 需全任务驻留 → flag 预算超限」）标注为未实证假设；③ 已产出的 [DESIGN_LIMIT] 被后续实测推翻时不删改原文（append-only），以「修正附录」回填 perf_feedback.md（结论限定于冻结设计族 + 修正证据表 + 存活硬上限清单 + 根因记录——本轮已验证形态）。
    动机: 结构性负结论驱动 conductor「设计层重做/等待编译器补齐」逆向路由，误判代价是整轮设计方向（第二轮误判使达标结构延后一轮才被实施，参考实现就在仓库内而两轮算法调研均未纳入）；先例检索 + 同口径实测是最低成本拦截位。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z 2026-09-09
- decided_by: -
- decided_note: -

---

## Decided（merged / rejected / expired / conflict 归档）

## VP-2026-0001
- type: P
- title: run 级双态（同 kernel 跨 run ±3–5%）需交错 A/B 多 run 协议裁决小差异——BP_run_state_bimodality
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/opt_log.md（首证 2026-08：run 间 ±2–3.5% 环境双态层、per-process 快/慢双态、<3% 结论须多 run）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-4（第二证 2026-09-07：1M fp16 跨 run 7.30–7.66；fp32 采纳项单轮 -5.2% 双态膨胀 → A/B/A/B 4 run 合并中位 -3.1%）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/profiles/final/ab_*
- repro: 16m fp32：final(bs8192) vs baseline(bs2048) 交替各 2 run，msprof op --launch-count=15 取 median 后合并比较
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0（lerp）；mish 见其 opt_log 版本戳
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_run_state_bimodality（目录行 + 正文节，追加于 BP_parameter_uncertain 之后）：触发信号（同配置跨 run ±3–5%、<20us 或贴地板 workload 差异落 3% 门槛附近、复测翻转）/ 反证（launch-count 消失→BP_measurement_resolution_limited；大 kernel 单 run 已稳）/ 动作（首次 <5% 差异即前置交错 A/B/A/B：≥3 独立 run、交替次序、合并中位；3% 采纳判断必须合并中位 + 配对次序稳定）/ 验证（run 次序与中位记录、配对方向跨 run 一致）。
- status: merged
- confirmations: 2/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- confirmed_by: task mish optimize 2026-08（证据经 mish opt_log 档案追溯，创建时一并附上）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（mish 2026-08 / lerp_tensor 2026-09-07），创建即达 Tier 1 阈值 2/2，本蒸馏周期合入 bottleneck-patterns.md（目录行 + BP_parameter_uncertain 节后追加正文）。
