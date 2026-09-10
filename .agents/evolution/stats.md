# 进化统计（Evolution Stats）

> 由 `tilelang-skill-evolver` 在每次蒸馏（distill 模式）后更新。用于度量自进化机制的有效性：条目是否被读到、同类错误是否复发。

## 1. 任务蒸馏记录

> 每次蒸馏任务追加一行。`vp_count` 按 D/P/R/C 分列计数（候选数，非合入数）。

| date | task_id | scenario | final_phase | vp_count (D/P/R/C) | merged | enqueued | verdict |
|------|---------|----------|-------------|--------------------|--------|----------|---------|
| 2026-09-07 | lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z | migration-harness | DONE | 3/3/5/1 | pattern-library §1.5×1（host IntImm 折叠）、§2×2（threads= 无效果、torch golden fp16 opmath 分歧）、§4×1（lerp 迁移精度案例） | VP-2026-0002/0003/0006/0007（Tier 1）、VP-2026-0008/0009/0012（Tier 2） | EVOLVE_COMPLETED |
| 2026-09-07 | lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z | optimize | DONE | 3/3/2/1 | bottleneck-patterns BP_run_state_bimodality（VP-2026-0001，mish+lerp 2/2） | VP-2026-0004/0005（Tier 1）、VP-2026-0010/0011（Tier 2）；查重命中 4（§1.6 copy-floor/MTE2 曲线、§2 multi-buffer、§4 perf_opt 案例行——optimizer 已任务内回写） | EVOLVE_COMPLETED |
| 2026-09-07 | multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z | migration-harness | DONE | 7/6/15/2 | pattern-library §1.7×1（Expert persistent 收益分界）、§1.8×1（bf16 直连无延迟税）、§2×5（load_nd2nz 跨步误读、T.copy 三规则、解析器四规则、v 算子操作数、零输入探针）、§4×2（expert attention 档案、highperf Expert 先例入口） | VP-2026-0013..0018（Tier 1）、VP-2026-0019..0033（Tier 2，15 条）；查重丢弃 3（aicore=24 第三源确认、M7 用例即证据、vcmp 低效警告并入 §2 行）；capability-gaps 维护（CG-2026-0004 task_id 规范化、CG-2026-0001 补 Expert 绕法注） | EVOLVE_COMPLETED |
| 2026-09-08 | multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z | optimize | DONE | 5/3/3/2 | pattern-library §1.9×1（update：终局数据/跨谱系对照/fa512 单块链地板/M7-clean/vcmp 闭环）、§4×1（developer 谱系对照档案行） | VP-2026-0034/0035（Tier 1：块宽摊减 BP、tuned 分派表模式）、VP-2026-0036/0037（Tier 2：perf_records 写入纪律、msprof 父目录权限）；查重命中 6（§1.9 五条 + §4 expert 行——optimizer 已任务内回写，防双写跳过；BP-R1 经查重转 update 提案）；capability-gaps 核对无新增无遗漏 | EVOLVE_COMPLETED |
| 2026-09-09 | multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z | optimize | DONE | 6/2/4/1 | pattern-library §1.9×1（update：fabric ≥1.73TB/s 数据条 + 第二轮 origin_task/完整版本戳/复现命令补全）、§4×1（update：expert Stage 4 档案行扩硬上限测绘与 [DESIGN_LIMIT] 五段档案） | VP-2026-0038（Tier 1：误差结构定征）、VP-2026-0039/0040（Tier 2：探针测硬件上限标准形态、续调基线复核协议）；VP-2026-0036 并入转录笔误证据（同主题不新开 id）、VP-2026-0034 补同谱系机制数据（按 VP-2026-0035 先例不计独立确认）；查重命中 4（VP-D3/D4/D5/VP-P3——optimizer 已任务内回写 §1.9，逐项对账相符，防双写跳过）；capability-gaps 维护（CG-0005/0006/0007 task_id 归位 20260909T033622Z + 四条 open 条目自 Resolved 区块移回 Open，纯结构无内容变更） | EVOLVE_COMPLETED |
| 2026-09-09 | multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z | optimize | DONE | 2/1/2/1 | pattern-library §1.9×1（update：溯源行补第三轮 origin_task 20260909T071018Z + raw 对账 round11/final3 + 三轮同 commit）、§2×1（update：活跃源 transpose 毒化行 origin_task 归位——原误标第一轮 task_id，同第二轮 CG 条目误标形态——+ 复现命令补全三件套齐备）、§4×1（update：expert Stage 4 档案行扩第三轮两相位达标 + [DESIGN_LIMIT] 推翻闭环）；capability-gaps 维护（CG-0005/0006/0007 第三轮注记补 task_id 归属） | VP-2026-0041/0042（Tier 2：morph ladder 参考实现对照定位法→iteration-diagnosis.md、[DESIGN_LIMIT]「结构不可达」断言证据门槛+修正附录机制→optimize SKILL.md）；查重命中 3（VP-D6/D7/VP-P4——optimizer 已任务内回写 §1.9/§2，逐项对账相符〔终值/流量/morph 矩阵 vs perf_records round 11 + Final Summary 逐项核对〕，防双写跳过） | EVOLVE_COMPLETED |

## 2. 条目命中统计

> 统计 pattern-library / bottleneck-patterns 条目被后续任务引用的情况。
>
> **计数方式**（由 evolver 在蒸馏时增量更新，不要求全量重扫）：
> - **主动命中**：任务工件（DESIGN.md / opt_log.md / REVIEW.md / RETROSPECTIVE.md）中显式引用了 `pattern-library §x.x` / `BP-xxx` 条目——每个任务每条目计 1 次。
> - **被动命中**：条目位于某 skill 的强制读取路径（design 强制步骤 0.5 / optimize Phase 0 / develop Phase 1 / conductor 带记忆重试注入）且该任务执行过对应阶段——不计入（只统计主动引用）。
>
> **治理规则**：某条目累计主动命中为 0 且超过 2 个蒸馏周期 → consolidate 候选；同类错误在已有对应条目的情况下复发（复盘章节可识别）→ 检索注入点缺失，优先补「失败触发读取」而非新增条目。

| entry | hits | last_hit_task | note |
|-------|------|---------------|------|
| pattern-library §4 mish perf_opt 案例行 | 1 | lerp_tensor-…T025419Z | lerp optimizer 经 §4 路由引用 mish 先例（run 双态协议 / Expert 流水 / multi-buffer 教训） |
| pattern-library §2 event 口径弃用行 | 1 | lerp_tensor-…T025419Z | opt_log Skill Flow Issue 引用「§2 已弃用 event 口径」 |
| pattern-library §2 auto-multi-buffer UB 膨胀行 | 1 | lerp_tensor-…T025419Z | Iteration 1 blocked 分析复现 mish BP_multi_buffer_ub_budget（fp32 bs12288 同报错） |
| bottleneck-patterns BP_run_state_bimodality（2026-09-07 新增） | 2 | multi_head_attention-…T005751Z | lerp 任务应用（A/B/A/B 协议）后，expert Stage 4 于 fa512 f32/sf16 平区按其「结构证据决胜」纪律裁决（opt_log Iteration 6 + Skill Retrospective） |
| bottleneck-patterns BP_pipeline_overlap | 1 | multi_head_attention-…T005751Z | expert Stage 4 Iteration 1 busy 核分解匹配「BP_pipeline_overlap 变体」（跨引擎 flag 串行链）——变体形态已提案 VP-2026-0034 |
| bottleneck-patterns BP_task_concurrency_sweet_spot | 1 | multi_head_attention-…T005751Z | H=1 欠载 bm 扫描（wall-rows 平衡分析）匹配该模式（opt_log Skill Retrospective 自述） |
| pattern-library §2 expert 迁移陷阱条目组（load_nd2nz/T.copy/解析器/v 算子，origin T115424Z） | 1 | multi_head_attention-…T005751Z | opt_log 头部「§2 陷阱条目（同源 expert 迁移任务）按此戳引用」——任务级引用（版本戳核对行为），非单行命中 |
| pattern-library §1.6 copy-floor/MTE2 曲线（2026-09-07 optimizer 任务内回写） | 0 | — | 新条目初始化（非检索命中） |
| pattern-library §2 threads= 无效果行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §2 torch golden fp16 opmath 行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §1.5 host IntImm 折叠行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §4 lerp 迁移精度案例行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §1.7 Expert persistent 收益分界（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §1.8 bf16 Cube 直连链（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §2 load_nd2nz 跨步误读行（2026-09-07 新增） | 0 | — | 新条目初始化（与 CG-2026-0004 互链） |
| pattern-library §2 T.copy 区域语义行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §2 解析器四规则行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §2 v 算子操作数行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §2 零输入探针行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §4 expert attention 档案行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §4 highperf Expert 先例入口行（2026-09-07 新增） | 0 | — | 新条目初始化（补本任务 Stage 1 检索缺口——先例分散 4 目录无索引） |
| pattern-library §1.9 块宽摊减调优序与硬上限（2026-09-08 optimizer 任务内回写 + evolver update 补终局数据） | 0 | — | 新条目初始化（非检索命中）；2026-09-09 第二轮：optimizer 续写「第二轮硬目标续调实测补充」块（f16 链/L0C/L1 端口/发射定律/sync_block_wait/DESIGN_LIMIT 档案）经 evolver 逐项对账合规保留 + update 补 fabric 数据条与溯源归位（origin_task 20260909T033622Z）；2026-09-09 第三轮：optimizer 续写「第三轮：两相位重构达标终值」块（终值 12.94/18.11/30.88/98.05µs/流量 -40%/morph 阶梯/bm 死点）经 evolver 逐项对账合规保留 + update 补溯源归属（origin_task 20260909T071018Z——原缺第三轮 task_id） |
| pattern-library §4 expert Stage 4 perf_opt 档案行（2026-09-08 optimizer 任务内回写） | 0 | — | 新条目初始化；2026-09-09 evolver update 扩第二轮硬上限测绘 + [DESIGN_LIMIT] 五段档案触发条件（origin_task 20260909T033622Z）；2026-09-09 第三轮 update 扩两相位达标 + [DESIGN_LIMIT] 推翻闭环（origin_task 20260909T071018Z） |
| pattern-library §2 活跃源 transpose 毒化行（2026-09-09 第三轮 optimizer 任务内回写） | 0 | — | 新条目初始化（非检索命中）；evolver update：origin_task 归位（原误标第一轮 task_id 20260908T005751Z → 第三轮实际任务 20260909T071018Z）+ 复现命令补全 |
| pattern-library §4 developer 谱系对照档案行（2026-09-08 evolver 新增） | 0 | — | 新条目初始化 |

## 3. 系统指标

> 北极星指标。原始数据来自各任务 `.stage_state.json` 与 RETROSPECTIVE.md；由 evolver 记录原始数据，趋势分析可人工或后续工具化。

| 指标 | 定义 | 当前值 |
|------|------|--------|
| first_pass_rate | Stage 3 `attempt=1` 即 `[PRECISION_PASS]` 的任务占比（长期应随模式库增长而上升） | 0/2（n=2；本任务 Stage 3 attempt-1/2 为**会话超限空返回**〔新失败形态，非 kernel 缺陷——VP-2026-0021 旨在消除〕，attempt-3 [PRECISION_PASS] 且 `--level all` 29 用例全过） |
| 同类错误复发率 | 已有 D/P 条目对应的失败模式在后续任务中复发占比（复发=检索注入失效） | 暂无数据（本任务无既有条目复发；Stage 3 空返回与 gate 误判均为无既有条目的新失败形态——VP-2026-0019..0021 旨在消除） |
| design_revision_avg | 任务平均设计修订次数（`retry_count` 均值） | 1（n=2；lerp 0 + 本任务 2——3 轮检视：R1 符号错误/bf16 断言/守卫缺口，R2 K_A 取整方向〔含 R1 建议自我纠正〕，R3 通过） |

## 4. 变更日志

| date | by | change |
|------|-----|--------|
| 2026-09-01 | 初始化（conductor-self-evolution-design 落地） | 创建骨架 |
| 2026-09-07 | tilelang-skill-evolver | 双任务蒸馏（lerp_tensor 迁移 + optimize）：Tier 0 合入 pattern-library §1.5×1/§2×2/§4×1；Tier 1 达 2/2 合入 bottleneck-patterns BP_run_state_bimodality（mish+lerp 双证据，VP-2026-0001）；入队 6×Tier 1（VP-2026-0002..0007）+ 5×Tier 2（VP-2026-0008..0012）。**快照披露**：pattern-library.md 蒸馏前已有未提交改动（本任务族 optimizer 回写 §1.6/§2 multi-buffer/§4 lerp 行 + mish 档案行 + D2 溯源头注），与进化增量同任务 lineage，按单一进化快照一并提交（未采用 stash 拆分——拆分将使 commit 缺失回写查重基线且 stash pop 必然冲突）；如需回滚整体 revert 本次 commit。 |
| 2026-09-07 | tilelang-skill-evolver | attention expert 迁移蒸馏（multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z）：Tier 0 合入 pattern-library §1.7/§1.8 + §2×5 + §4×2；入队 6×Tier 1（VP-2026-0013..0018）+ 15×Tier 2（VP-2026-0019..0033：gate_lint 误报修复×2、大工件读写纪律×2、设计/检视修订纪律×3、弃选举证边界×1、检视公式验证与重审工作流×3、conductor 修订仲裁×1、integrate 谱系告警×1、测试模板 kwargs×1）；capability-gaps 维护（CG-2026-0004 task_id 规范化、CG-2026-0001 补 Expert 结构级绕法注——occurrences 核对：均 1、无 recurring 升级）。**快照披露**：capability-gaps.md 蒸馏前已有未提交改动（CG-2026-0004〔本任务 Stage 3 登记〕+ CG-2026-0003〔FA 三实现对比任务〕+ CG-2026-0002/CG-2026-0001〔developer 谱系〕——均为各任务 Agent 的登记条目而非用户手改），且本蒸馏对 CG-2026-0004 的规范化编辑依赖该未提交内容（stash 拆分不可行），按前次先例以单一进化快照一并提交；如需回滚整体 revert 本次 commit。 |
| 2026-09-08 | tilelang-skill-evolver | expert Stage 4 调优蒸馏（multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z）：**复核 optimizer 直写 pattern-library §1.9 五条 + §4 expert 行**——三件套齐备（溯源/版本戳/复现条件）、带 origin_task、与既有条目无冲突、写入属 optimize SKILL.md Phase 4 例外授权范围内（pattern-library 头部双写入者契约 + lerp §1.6 先例），全部保留、无降级处置；Tier 0 合入 §1.9 update（终局数据/跨谱系对照/fa512 单块链地板/M7-clean/vcmp 警告闭环）+ §4×1（developer 谱系对照档案行）；入队 2×Tier 1（VP-2026-0034 BP_cross_engine_serial_chain 块宽摊减 / VP-2026-0035 TUNED_DEFAULT_CONFIGS 分派+fa-tuned 测试层）+ 2×Tier 2（VP-2026-0036 perf_records 写入纪律〔gate 4 失败根因〕/ VP-2026-0037 msprof 父目录权限——BP-R1 整合去重：现行规则已覆盖叶子目录，转 update 提案）；capability-gaps 核对（无新增、无遗漏、无 recurring 升级）。**快照披露**：pattern-library.md 蒸馏前已有未提交改动（optimizer 本任务回写 §1.9 + §4 expert 行），与进化增量同任务 lineage，按 2026-09-07 先例以单一进化快照一并提交（stash 拆分将使 commit 缺失回写查重基线且 pop 必然冲突）；如需回滚整体 revert 本次 commit。 |
| 2026-09-09 | tilelang-skill-evolver | expert Stage 4 第二轮续调蒸馏（multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z，终态 DONE + [DESIGN_LIMIT] 附录补记路由）：**复核 optimizer 直写 pattern-library §1.9「第二轮硬目标续调实测补充」**——自报 VP-D3（L0C=128KB）/VP-D4（L1 端口 148–154GB/s/核）/VP-D5（向量发射开销定律 ~0.5µs/op）/VP-P3（f16 softmax 链）四条真实落盘，内容与 opt_log round 8–10 / perf_records.jsonl（final_dispatch_r2 = 17.55/27.24/69.13/226.72µs）/ perf_feedback.md 逐项对账相符，写入属 Phase 4 例外授权范围，全部保留、无降级处置；Tier 0 update：§1.9 补 fabric ≥1.73TB/s 数据条（round 8 实测，此前未收录）+ 第二轮 origin_task/完整 commit 版本戳/复现命令（D2 溯源补全——补充块原缺本轮 task_id 归属）、§4 expert Stage 4 档案行扩第二轮硬上限测绘与 [DESIGN_LIMIT] 五段档案（VP-C2 落地，同目录行 update 优先于 add）；入队 1×Tier 1（VP-2026-0038 误差结构定征：全块均匀→跨迭代状态而非竞态）+ 2×Tier 2（VP-2026-0039 探针测硬件上限标准形态→profile-collection.md / VP-2026-0040 续调任务前置基线复核→iteration-diagnosis.md——含手改驱动漂移检测）；VP-2026-0036 并入同主题转录笔误证据（3779.0 手抄 → append-only 更正行，delta 扩「禁止手抄」条款，不新开 id）、VP-2026-0034 证据链补深流水回退机制数据（同谱系续调，按 VP-2026-0035 同 op 先例**不计独立确认**，维持 1/2）；capability-gaps 维护：CG-2026-0005/0006/0007 登记核验相符 + task_id 归位（误标上轮 20260908T005751Z → 实际识别任务 20260909T033622Z）+ 四条 status:open 条目（CG-0001/0005/0006/0007）自 Resolved 区块移回 Open（纯结构归位，无内容变更；occurrences 均 1、无 recurring 升级）；会话异常终止恢复（Stage 4 attempt 1 runtime fail 1053s → 重调度 attempt 2 complete）经评估为 conductor 既有 stage_retry 标准行为、工件无恢复机制细节证据，不强行立项 R 类（见本次进化报告 issues）。**快照披露**：pattern-library.md 与 capability-gaps.md 蒸馏前已有未提交改动（本任务 optimizer 回写 §1.9 第二轮补充块 + 登记 CG-0005/0006/0007），进化 update 原位修改该未提交内容（stash 拆分不可行——update 锚点即补充块/登记条目自身），按 2026-09-07/08 先例以单一进化快照一并提交；如需回滚整体 revert 本次 commit。 |
| 2026-09-09 | tilelang-skill-evolver | expert Stage 4 第三轮两相位重构蒸馏（multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z，终态 DONE 四硬目标全达成）：**复核 optimizer 直写 pattern-library §1.9「第三轮：两相位重构达标终值」块 + §2 活跃源 transpose 毒化行**——自报 VP-D6（毒化 2.6x + 绕法）/VP-D7（终值 12.94/18.11/30.88/98.05µs 与流量 -40%）/VP-P4（两相位迁移模式）真实落盘，内容与 opt_log round 11 / perf_records.jsonl（final_dispatch_r3 逐行核对：12.94/18.11/30.88/98.05、v11_initial 250.68 回归如实入账、v11nt 98.38、bm 探针 105.16/101.84、causal 3764.05/13265.27）/ perf_feedback.md 修正附录 / DESIGN.md §11.3 逐项对账相符，写入属 Phase 4 例外授权范围，全部保留；**D2 溯源修复 ×2**：① §2 毒化行 origin_task 归位（原误标第一轮 task_id 20260908T005751Z——同第二轮 CG 条目误标形态，实际第三轮任务 20260909T071018Z）+ 复现命令补全（三件套齐备）；② §1.9 溯源行补第三轮 origin_task + raw 对账 round11/final3；Tier 0 update ×3（§1.9 溯源 / §2 归位 / §4 expert Stage 4 档案行扩第三轮达标 + [DESIGN_LIMIT] 推翻闭环——VP-C3 落地）；入队 2×Tier 2（VP-2026-0041 morph ladder 参考实现对照定位法→iteration-diagnosis.md〔含 op 级无罪 ≠ 组合无罪方法论〕/ VP-2026-0042 [DESIGN_LIMIT]「结构不可达」断言证据门槛 + 修正附录机制→optimize SKILL.md〔仓库先例检索 + 同口径实测 + 纯推理否决标注未实证假设；与 VP-2026-0025/0041 交叉引用〕）；capability-gaps 维护（CG-0005/0006/0007 第三轮「不阻塞」注记核验相符 + 补 task_id 归属；无新增、无 recurring 升级）；工具中断经 task_id 恢复会话续跑（timeline 无 fail 事件、未耗 stage_retry、无恢复机制细节证据）不强行立项 R 类（同第二轮评估先例，见进化报告 issues）。**快照披露**：pattern-library.md（optimizer 本轮回写第三轮块 + §2 毒化行）与 capability-gaps.md（第三轮注记）蒸馏前已有未提交改动，进化 update 原位修改该未提交内容（stash 拆分不可行——update 锚点即回写内容自身），按 2026-09-07/08/09 先例以单一进化快照一并提交；如需回滚整体 revert 本次 commit。 |
