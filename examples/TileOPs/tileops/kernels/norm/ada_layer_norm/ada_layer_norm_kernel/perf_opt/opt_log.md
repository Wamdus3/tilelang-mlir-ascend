# Stage 4 调优日志 — `_ada_layer_norm_kernel`（AdaLayerNormFwdOp, has_gate=False）

- 任务：optimize 场景 Stage 4（`mode=full`），best_effort（迭代上限内持续压低 Task Duration）
- 基准：`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py`（Stage 3 [PRECISION_PASS]，只读不改）
- 工具链戳：tilelang 0.1.2+a83118285a（2026-09-10 dev build）+ Ascend910B2C（24 AICore → 48 Vector cores，实查）+ CANN 8.5.0
- 性能口径：**`msprof op` Task Duration(us)**（kernel-only，唯一口径）；`--launch-count=20 --warm-up=5`，median of 20
- 测量 harness：`perf_opt/bench.py`（同输入张量逐 launch，`sets=1`——与 Stage 5 集成 bench harness `tileops/benchmark/msprof.py` 的访问模式一致，保证 round-0 与 Stage 5 记录可直接对账）；每个配置带 `--check` golden 比对（参数分支的 L0 等价门禁）
- 分派结构：2 条真实代码分支——`fp32_direct`（fp32，无 vcast）与 `fp32_transit`（fp16/bf16，fp32 中转链）；必测 workload = manifest 8 组（conductor 指定全组）
- 目标 kernel 名：`main`（captured op name = `main`，op_type=vector，已溯源校验）

---

## Performance Test Data（Round 0 baseline，wrapper 部署态 block_m=1）

| workload | dispatch_path | target_kernel | captured_op | Task Duration(us) | Block Dim | vec_ratio | mte2_bw(GB/s/core) | L0 | raw profile |
|---|---|---|---|---:|---:|---:|---:|---|---|
| smoke-dit/fp32 (64×1152) | fp32_direct | main | main | 5.48 | 48 | 0.148 | 12.7 | pass | profiles/baseline/smoke_fp32_bm1 |
| smoke-dit/fp16 (64×1152) | fp32_transit | main | main | 5.49 | 48 | 0.204 | 7.8 | pass | profiles/baseline/smoke_fp16_bm1 |
| smoke-dit/bf16 (64×1152) | fp32_transit | main | main | 5.43 | 48 | 0.204 | 7.8 | pass | profiles/baseline/smoke_bf16_bm1 |
| dit-xl-2/fp16 (1024×1152) | fp32_transit | main | main | 34.831 | 48 | 0.269 | 7.5 | pass | profiles/baseline/dit_fp16_bm1 |
| dit-xl-2/bf16 (1024×1152) | fp32_transit | main | main | 34.961 | 48 | 0.269 | 7.5 | pass | profiles/baseline/dit_bf16_bm1 |
| llama-prefill/fp16 (2048×4096) | fp32_transit | main | main | 90.514 | 48 | 0.446 | 23.1 | pass | profiles/baseline/prefill_fp16_bm1 |
| llama-prefill/bf16 (2048×4096) | fp32_transit | main | main | 91.224 | 48 | 0.446 | 23.1 | pass | profiles/baseline/prefill_bf16_bm1 |
| llama-decode/bf16 (1×4096) | fp32_transit | main | main | 3.16 | 1 | 0.309 | 18.9 | pass | profiles/baseline/decode_bf16_bm1 |

**与 Stage 5 集成 bench 对账**（profile_run_msprof_20260910_144841.log）：大 kernel 偏差 ≤0.3%（prefill 90.51 vs 90.80 / 91.22 vs 91.42；dit 34.83 vs 34.06 +2.3%）；小 kernel +3.1~5.8%（BP_run_state_bimodality 已知 ±3-5% 双态带内）——测量口径可比性成立。

### 设计估算 vs 实测偏差行（D-2 回填，DESIGN.md §1.6.0）

- DESIGN 估算下界（llama-prefill 2048×4096 fp16）：≈53–67 µs（流量项 67.1MB ÷ 1.0–1.26 TB/s；发射项按 MTE2 隐藏；容量项 0）
- baseline 实测：90.51 µs = 估算带的 **1.35–1.71x**（失准项：**发射/重叠**——向量链与 MTE2 零重叠，见 Round 1 诊断；流量项本身 739 GB/s 也低于假设带宽）
- 调优后 final：40.52 µs，**优于设计估算带下界**（53 µs）——带内假设"MTE2-bound 于 HBM 混合地板"被实测推翻：sets=1 访问模式下 L2 辅助读带宽达 ~2.07 TB/s 聚合（43.1 GB/s/core × 48），重叠后瓶颈转为 Vector/UB 流量（vec_ratio 0.91）

### Baseline 诊断（当前现象，全 workload）

- **P1（主导）**：各 pipe 忙时之和 ≈ wall——MTE2 与 VEC **零重叠**。prefill fp16 bm=1：vec 38.81 + scalar 13.24 + mte2 42.36 + mte3 6.47 ≈ 100.9 µs vs wall 90.5 µs（仅 ~10 µs 并发）。根因：单一 `stage` 缓冲 x→scale→shift 复用链强制 MTE2(x)→VEC(vcast)→MTE2(scale)→VEC→MTE2(shift)→VEC 交替串行。
- **P2**：per-task 标量开销显著——smoke scalar_ratio 0.42-0.43、dit bm=1 scalar 7.7 µs；bm=1 时每核 21-43 个 serial 任务。
- **P3**：bm=1 拷贝粒度小——prefill bm=1 (1,4096)=8KB 拷贝 MTE2 效率 24 GB/s/core，bm=2 (2,4096)=16KB 达 40 GB/s/core。
- 对照参照（morph 前置）：同流量形态 lerp_tensor（3 输入 1 输出，PL-1.6，**装载前置 + 向量链全隐藏**）——结构差异 = 本轮 Round 2 候选的直接来源。

---

## Iteration 1（round 1）：block_m 扫描（已知第一杠杆，参数级）

- base：baseline kernel（`../_ada_layer_norm_kernel.py`）@ bm=1
- 候选：每 workload 扫 bm ∈ {2,4,8}（N=1152）/ {2}（N=4096，20 B/elem UB 律的硬顶；bm=3 编译探针 245,760B 溢出）/ {2}（decode 完整性）
- 结果（18 行记录，全部 L0=pass）：

| workload | bm=1 | bm=2 | bm=4 | bm=8 | winner |
|---|---:|---:|---:|---:|---|
| smoke-dit/fp32 | 5.48 | 3.32 | **3.09** | 3.67 | bm=4（16 任务/16 核，nl=1） |
| smoke-dit/fp16 | 5.49 | 3.38 | **3.16** | 3.58 | bm=4 |
| smoke-dit/bf16 | 5.43 | 3.41 | **3.16** | 3.64 | bm=4 |
| dit-xl-2/fp16 | 34.83 | 19.12 | 14.59 | **10.91** | bm=8（128 任务，nl=3） |
| dit-xl-2/bf16 | 34.96 | 19.66 | 14.84 | **10.76** | bm=8 |
| llama-prefill/fp16 | 90.51 | **58.27** | — | — | bm=2（UB 硬顶） |
| llama-prefill/bf16 | 91.22 | **58.85** | — | — | bm=2 |
| llama-decode/bf16 | 3.16 | 3.44 (+8.9%) | — | — | **bm=1 保持**（M=1，bm=2 垃圾行开销） |

**候选 vs current best 对比表（round 1 主战场，prefill fp16 与 dit fp16）**

| candidate | workload | Task Duration(us) | vs best | vec_ratio(AICore) | mte2_bw(GB/s) | L0 |
|---|---|---:|---:|---:|---:|---|
| baseline bm=1 | prefill/fp16 | 90.51 | — | 0.446 | 23.1 | pass |
| v1_bm2 | prefill/fp16 | **58.27** | **-35.6%** | 0.630 | 38.2 | pass |
| baseline bm=1 | dit/fp16 | 34.83 | — | 0.269 | 7.5 | pass |
| v1_bm8 | dit/fp16 | **10.91** | **-68.7%** | 0.553 | 38.2 | pass |

- winner：上表各 workload 最优 bm；new_current_best = 基准 kernel + 调优 bm 配置（smoke→4 / dit→8 / prefill→2 / decode→1）
- 机制确认：bm 提档同时改善 P2（任务数减半/减四）与 P3（拷贝粒度翻倍）；smoke 呈 U 形甜点（32 核 bm=2 反而慢于 16 核 bm=4）——BP_task_concurrency_sweet_spot
- decode bm=2 回退 +8.9%（>5% 阈值，非噪声）→ 记录 rollback，bm=1 保持

---

## Iteration 2（round 2）：MTE2/VEC 解耦——per-input staging + 前置装载（结构级，v2_op1）

- base：基准 kernel（bm 已由 round 1 定优）
- 结构改动（唯一主要优化点）：fp32_transit 路径 `stage` 单缓冲复用 → `stage_x`（x 入/y 出双角色）+ `stage_s`/`stage_h` 专用 staging，三输入装载全部前置，之后一条不间断 fp32 向量链 + 单次写出；fp32_direct 路径同样加 `scale_f32`/`shift_f32` 专用缓冲前置装载。数学、精度契约（vsqrt+vdiv）、尾块语义零改动。
- UB 膨胀探针（`probe_ub_v2.py`，编译溢出报文精确读数，D 类证据）：
  - transit：**精确 20.01 B/elem**（resident 10B 与 14B 两代结构同为 20B 绝对值——膨胀后 footprint 与 staging 数量无关）→ bm=8×N=1152（184.3KB）与 bm=2×N=4096（163.9KB）均存活，基线 bm 范围完整保留
  - fp32：**精确 26.00 B/elem**（8B 与 16B resident 同为 26B）→ N=1152 bm≤6、N=4096 bm=1
- 结果（16 行记录，全 L0=pass；分支全量 `--level all` 通过）：

**候选 vs current best 对比表（round 2）**

| candidate | workload | bm | Task Duration(us) | vs best | vec_ratio | mte2_bw(GB/s) | L0 |
|---|---|---:|---:|---:|---:|---:|---|
| v1_bm4（best） | smoke/fp32 | 4 | 3.09 | — | 0.50 | 61.7* | pass |
| v2_op1_bm4 | smoke/fp32 | 4 | 3.22 | +4.2% | 0.416 | 62.1 | pass |
| v1_bm4（best） | smoke/fp16 | 4 | 3.16 | — | 0.50 | 33.6 | pass |
| v2_op1_bm4 | smoke/fp16 | 4 | 3.25 | +2.8% | — | — | pass |
| v2_op1_bm8 | smoke/fp16 | 8 | 3.79 | +19.9% | — | — | pass（bm=8 弃） |
| v1_bm8（best） | dit/fp16 | 8 | 10.91 | — | 0.553 | 38.2 | pass |
| v2_op1_bm8 | dit/fp16 | 8 | **9.55** | **-12.5%** | 0.674 | 34.9 | pass |
| v2_op1_bm8 | dit/bf16 | 8 | **9.72** | **-9.7%** | — | — | pass |
| v1_bm2（best） | prefill/fp16 | 2 | 58.27 | — | 0.630 | 38.2 | pass |
| v2_op1_bm2 | prefill/fp16 | 2 | **39.81** | **-31.7%** | **0.914** | 43.1 | pass |
| v2_op1_bm2 | prefill/bf16 | 2 | **40.63** | **-31.0%** | — | — | pass |
| v2_op1_bm1 | prefill/fp16 | 1 | 47.63 | -18.3% | — | — | pass（bm=2 更优） |
| baseline（best） | decode/bf16 | 1 | 3.16 | — | 0.309 | 18.9 | pass |
| v2_op1_bm1 | decode/bf16 | 1 | **2.65** | **-16.1%** | 0.377 | 45.1 | pass |

\* smoke fp32 的 v1_bm4 指标取 final 轮同形态 profile（round 1 未留存 PipeUtilization 摘要，Task Duration 为 round 1 实测）。

- **smoke +1.3~4.2% 差异按规程走 A/B 交错多 run 协议**（<20µs 小 kernel，<5% 差异）：
  - smoke fp32 v1_bm4 vs v2_bm4：merged median 3.18 vs 3.18，b_rel_change -0.0%，sign_test p=1.0，direction_consistent=false → **tie**（round 2 单次的 +4.2% 为双态噪声）
  - smoke fp16 v1_bm4 vs v2_bm4：merged median 3.16 vs 3.16 → **tie**
  - 结论：v2 结构对 smoke 无回退（A/B 裁决，证据 `ab_round2/`）
- winner：v2_op1 结构全面采纳（dit/prefill/decode 显著提升，smoke A/B tie）；new_current_best = v2_op1 + {smoke:bm4, dit:bm8, prefill:bm2, decode:bm1}
- 机制确认：prefill pipe 忙时和 100.9→68.9 µs vs wall 39.8——**~29 µs 跨 pipe 重叠**；kernel 转为 vector-bound（vec_ratio 0.914，vec 33.3 µs ≈ 80 B/elem 链在 ~441 GB/s/core UB 口径速率的 96%）；MTE2 基本隐藏（mte2_ratio 0.62）

---

## Iteration 3（round 3）：静态尾块特化（v3_op2）+ dit bm 精搜 + num_kernels 平衡（v3_op3，blocked）

### v3_op2：`M % block_m == 0` 时 real_m 编译期常量化（静态 slice extents + 去逐任务 T.min）

- factory 层变体选择（TVM parser `if` 不折叠，TRAP-tvm-parser-rules——编译期分支必须在 trace 体之外，同 dtype dispatch 模式）
- 编译探针发现**静态 extents 形态使 transit 膨胀升至 22.00 B/elem**（bm=8×N=1152 溢出 202,784B）→ 加"静态 body 超预算回退 dynamic"条件
- 结果（8 行，全 L0=pass；分支 `--level all` 通过）：

| candidate | workload | bm | Task Duration(us) | vs best | L0 | 结论 |
|---|---|---:|---:|---:|---|---|
| v3_op2_bm2 | prefill/fp16 | 2 | 41.12 | **+3.3%** | pass | 回退，弃 |
| v3_op2_bm2 | prefill/bf16 | 2 | 41.24 | +1.5% | pass | 回退，弃 |
| v3_op2_bm4 | smoke/fp32 | 4 | 3.21 | +3.9% | pass | 平区（A/B 域），弃 |
| v3_op2_bm4 | smoke/fp16 | 4 | 3.23 | +2.2% | pass | 弃 |
| v3_op2_bm1 | decode/bf16 | 1 | 2.66 | +0.4% | pass | tie，弃 |
| v3_op2_bm7 | dit/fp16 | 7 | **9.10** | -4.7% | pass | 见下（混淆隔离） |
| v3_op2_bm7 | dit/bf16 | 7 | **9.26** | -4.7% | pass | 见下 |

**候选 vs current best 对比表（round 3 主战场）**

| candidate | workload | Task Duration(us) | vs best | vec_ratio | scalar_ratio | mte2_bw(GB/s) | L0 |
|---|---:|---:|---:|---:|---:|---:|---|
| v2_op1_bm8（best） | dit/fp16 | 9.55 | — | 0.674 | 0.306 | 34.9 | pass |
| v3_op2_bm7 | dit/fp16 | 9.10 | -4.7% | 0.700 | 0.329 | — | pass |
| v2_op1_bm7（隔离） | dit/fp16 | **8.76** | **-8.3%** | — | — | — | pass |
| v2_op1_bm5 | dit/fp16 | 9.75 | +2.0% | — | — | — | pass |
| v2_op1_bm6 | dit/fp16 | 9.66 | +1.1% | — | — | — | pass |
| v2_op1_bm2（best） | prefill/fp16 | 39.81 | — | 0.914 | 0.235 | 43.1 | pass |
| v3_op2_bm2 | prefill/fp16 | 41.12 | +3.3% | — | — | — | pass |

- **混淆隔离（round 3b）**：v2 dynamic@bm7 = 8.76 µs < v3 static@bm7 = 9.10 µs——dit 的提升来自 **bm 8→7 任务几何**（147 任务/nl=4/UB 82% 对 128 任务/nl=3/UB 94%），**非**静态特化；静态 body 本身在同 bm 下反而 +3.9%
- **bm 精搜（round 3c）**：dit bm 曲线 = {5: 9.75, 6: 9.66, **7: 8.76**, 8: 9.55}——bm=7 局部最优（非单调）
- **dit bf16 bm7 vs bm8 A/B 裁决**（-3.9% <5% 协议区）：merged median 9.72 → 9.11，b_rel_change **-6.28%**，三对方向一致 → **B_better**（证据 `ab_round3/dit_bf16_bm8_vs_bm7`）
- winner：**v2_op1@bm7 为 dit 新 best**（fp16 -8.3% 单轮、bf16 -6.3% A/B 合并）；静态特化全 workload 不采纳（prefill 回退、dit 隔离后无独立贡献、smoke/decode 平区）

### v3_op3：num_kernels cap=32（dit 任务平衡 4 任务/核）——**blocked（编译证据）**

- 动机：dit 128 任务 @48 核 = 32×3+16×2（3:2 不平衡）；cap=32 → 4×32 完美平衡
- 实测：**编译失败**——bm=8×N=1152 在 num_local_tasks=4 时 auto-multi-buffer 需求升至 22 B/elem（202,784B > 192KB，`ub overflow` 报文），与 bm 无关地被 num_local 影响（nl=3 时同配置 20.01 B/elem 可编译）。UB 需求随 serial trip count 变化——新 D 类实证（回写 pattern-library，见 Retrospective）。
- 处置：blocked（有证据）；num_kernels 维持 `min(num_logical, 48)`。L0-10 编译失败复现于分支文件（已删除，证据在 opt_log 与探针记录）。

---

## Iteration 4（round 4）：「UB ~85% 松弛律」泛化检验——**否定（保护性验证）**

- 假设：dit bm=7（UB 82%）胜 bm=8（94%）或为普适"multi-buffer 需余量"律 → 若成立可改进通用 fallback 规则
- 检验点：(4096, 1152) fp16（第三数据点，不在 manifest 内）：bm=7 = 24.86 µs vs bm=8 = **23.90 µs**（bm=8 胜 +4.0%）→ **假设否定**：dit 的 bm=7 是该 shape 任务几何的局部最优，非 UB 松弛律
- 处置：通用 fallback 维持硬顶 UB 规则；manifest shape 走显式调优表（S4-5 分派表模式）——这正是 TUNED_DEFAULT_BLOCK_M 显式逐 shape 记录而非规则化的原因
- 本轮无 workload 提升（防过度泛化的保护性验证轮）

---

## Iteration 6（round 6）：smoke U 曲线闭合（调优表条目鲁棒性验证）

- 动机：smoke 调优表条目 bm=4 的 U 曲线证据链为 {1: 5.49, 2: 3.38, 4: 3.16, 8: 3.58}——bm∈{3,5} 未测；若存在更优点则表条目错误
- 结果（6 行，全 L0=pass，最终 kernel 文件）：

| workload | bm=3 | bm=4（表值） | bm=5 | 结论 |
|---|---:|---:|---:|---|
| smoke-dit/fp32 | 3.12 | **3.09-3.12** | 3.26 | bm=4 确认 |
| smoke-dit/fp16 | 3.32 | **3.16** | 3.28 | bm=4 确认 |
| smoke-dit/bf16 | 3.30 | **3.16** | 3.38 | bm=4 确认 |

- 全部 bm∈{3,5} 相对 bm=4 差 +1.0~7.0%——U 曲线最小值在 bm=4 成立，调优表条目鲁棒；本轮无提升（保护性验证轮）

---

## Final Confirmation（round 5）：最终文件 × 8 workload × 调优表 bm

最终产物 `perf_opt/_ada_layer_norm_kernel.py` = v2_op1 结构（动态 body）+ `TUNED_DEFAULT_BLOCK_M` 分派表 + dtype 预算回退。全量 `--level all` 通过（L0 14 项 + L1 + L2 + Boundary）。

| workload | bm | Task Duration(us) | Block Dim | vec_ratio | mte2_ratio | mte2_bw(GB/s) | L0 |
|---|---:|---:|---:|---:|---:|---:|---|
| smoke-dit/fp32 | 4 | 3.12 | 16 | 0.429 | 0.422 | 61.7 | pass |
| smoke-dit/fp16 | 4 | 3.16 | 16 | 0.506 | 0.204 | 59.0 | pass |
| smoke-dit/bf16 | 4 | 3.26 | 16 | 0.515 | 0.221 | 54.3 | pass |
| dit-xl-2/fp16 | 7 | 8.94 | 48 | 0.700 | 0.592 | 34.8 | pass |
| dit-xl-2/bf16 | 7 | 9.28 | 48 | 0.680 | 0.583 | 33.6 | pass |
| llama-prefill/fp16 | 2 | 40.522 | 48 | 0.908 | 0.704 | 38.1 | pass |
| llama-prefill/bf16 | 2 | 40.822 | 48 | 0.909 | 0.657 | 39.9 | pass |
| llama-decode/bf16 | 1 | 2.66 | 1 | 0.382 | 0.164 | 55.7 | pass |

必测 dispatch 非回退检查：8/8 workload 相对各自历史 best 差异 -0.4% ~ +3.2%，全部在噪声阈值带内（<5%，小 kernel 双态）——无回退。

---

## Final Summary

- **best 版本**：`perf_opt/_ada_layer_norm_kernel.py`（v2 结构 + 调优表；分支文件 `_ada_layer_norm_kernel_opt_v2_op1.py` / `_opt_v3_op2.py` 留档）
- **final_latency（msprof op Task Duration，median of 20，与 perf_records.jsonl round 5 逐条对账）**：
  - smoke-dit/fp32: final_latency: 3.12 us（baseline 5.48，**-43.1%**）
  - smoke-dit/fp16: final_latency: 3.16 us（baseline 5.49，**-42.4%**）
  - smoke-dit/bf16: final_latency: 3.26 us（baseline 5.43，**-40.0%**）
  - dit-xl-2/fp16: final_latency: 8.94 us（baseline 34.831，**-74.3%**）
  - dit-xl-2/bf16: final_latency: 9.28 us（baseline 34.961，**-73.5%**）
  - llama-prefill/fp16: final_latency: 40.522 us（baseline 90.514，**-55.2%**）
  - llama-prefill/bf16: final_latency: 40.822 us（baseline 91.224，**-55.3%**）
  - llama-decode/bf16: final_latency: 2.66 us（baseline 3.16，**-15.8%**）
- **跨 workload 几何平均加速比：2.14x**（几何平均时延下降 53.3%）
- **关键有效优化点**（按贡献排序）：
  1. block_m 提档（wrapper 部署态 bm=1 → 调优表）：dit -69%、prefill -36%、smoke -42~44%（round 1）
  2. per-input staging + 三输入前置装载（MTE2/VEC 解耦）：prefill 再 -32%、dit 再 -12%、decode 再 -16%（round 2）
  3. dit bm=7 任务几何精搜：再 -8.3%（fp16）/ -6.3%（bf16 A/B 合并）（round 3）
- **回退/否决记录**：decode bm=2（+8.9%，垃圾行开销）；v3_op2 静态尾块（prefill +1.5~3.3%、dit 隔离后无独立贡献、smoke/decode 平区）；v3_op3 num_kernels cap=32（**blocked**：nl=4 时 UB 需求升至 22 B/elem 编译失败）；「UB 85% 松弛律」泛化（round 4 否定：4096×1152 上 bm=8 胜 bm=7 +4.0%）
- **收束原因：plateau**——连续 3 轮无超噪声阈值提升（round 4 泛化假设否定、round 5 终测全平区、round 6 smoke U 曲线闭合无增益），且主要结构候选已充分验证（interleaved vs loads-first vs static-extent 三结构 A/B/C 完成；bm 每 shape 全程扫描含 U 曲线闭合；num_kernels 编译受阻；深流水受 20 B/elem UB 律封顶——bm=2×N=4096 已 83.5% 占用，3 实例需 30 B/elem 必然溢出；融合乘加 v-op 不存在〔docs/Tilelang.language/数学操作/ 全目录核对〕；向量链 80 B/elem 中 24 B 为精度契约强制的 4 次 vcast、8 B 为两遍方差强制的 2 次 reduce——剩余结构空间为空）。plateau 前置参照结构 diff 检查（morph 轻量版）：同族最优参照 lerp_tensor（PL-1.6，装载前置 + 链全隐藏）与本 current best 结构同形态，残余差异为算子固有的 reduce 依赖链（数学强制），无未迁移结构。prefill vec_ratio 0.91（96% UB 口径速率）、smoke/decode 处 launch 地板（3.2/2.66 µs）。
- **[DESIGN_LIMIT] 判定：不触发**——① 当前天花板（prefill ≈40 µs ≈ vector-bound）归因于精度契约（fp32 中转 4×vcast + 中心化两遍 2×reduce，DESIGN §0.6 R4/§1.6.0 均为精度决策而非性能假设失误）+ 192KB UB 物理律，且实测**优于** DESIGN §1.6.0 估算带（53–67 µs）下界；② 无 >2x 结构性空间（同精度契约下向量流量下界 ~80 B/elem 已达 96% 利用；矩式单遍被 D-1 精度证据否决）。两项触发条件均不满足，按标准禁止产出 perf_feedback.md。
- 遗留（供 conductor/consumer 决策）：
  - wrapper 切换块引用建议：`from .ada_layer_norm_kernel.perf_opt._ada_layer_norm_kernel import TUNED_DEFAULT_BLOCK_M, select_row_config, _ada_layer_norm_kernel, ALIGNMENT, _align_up`，`_select_row_config` 改为按 (M, N) 查调优表（`select_row_config(M, N)` 已提供同形态返回值）
  - sets=1（Stage 5 口径）下 smoke/dit 为 L2 驻留测量；若部署态为冷数据流，MTE2 压力更大，v2 结构的重叠收益只会更显著（方向一致，无需额外调参）

---

## Skill Retrospective

### 流程复盘

- **多轮诊断驱动有效**：round 1 的"pipe 忙时和 ≈ wall"现象直接定位 stage 复用串行化根因，lerp 先例（PL-1.6 loads-first）提供结构模板——pattern-library 检索（INDEX + 命中条目）在首轮即命中可迁移结构，闭环 2 轮拿到 -56%。
- **A/B 协议两次防误判**：smoke v2 "+4.2% 回退"与 dit bf16 "-3.9% 提升"均为 <5% 小 kernel 差异——前者复测 tie（避免错误否决 v2 结构），后者复测 -6.28% 方向一致（正确采纳 bm=7）。BP_run_state_bimodality 对策有效，继续前置使用。
- **隔离实验抓出混淆变量**：round 3 的 dit 提升初判为"静态特化"，隔离实验（v2@bm7 vs v3@bm7）证明是 bm 任务几何效应——单分支多变量的教训再次验证"一个分支一个优化点"纪律的必要性。
- **保护性泛化验证避免规则化错误**：dit bm=7 若被泛化为"UB 85% 松弛律"写入通用 fallback，会在 (4096,1152) 类 shape 上引入 +4% 回退——round 4 的单点检验以 2 次测量的成本拦下一次可预期的错误泛化。
- **UB 膨胀律的结构依赖性**（新 D 类证据，已回写 pattern-library，见下）：膨胀后 footprint 不随 staging 数量线性增长（10B 与 14B resident 同为 20.01 B/elem；8B 与 16B resident 同为 26 B/elem），且受 num_local_tasks 与 slice extents 静态性影响（nl=4@bm8 → 22 B/elem；静态 extents → 22 B/elem）——**UB 预算 guard 必须按（结构 × dtype × nl）实测标定，不能从 resident 集合线性外推**。

### pattern-library 回写清单（ED-C，任务内已回写）

| 条目 | 类型 | 内容 | 回写位置 |
|---|---|---|---|
| TRAP-UB-multibuffer-inflation 更新 | D（陷阱实证扩展） | 膨胀 footprint 与 staging 数量解耦（20.01/26 B/elem 平台）；num_local_tasks 与静态 extents 影响（22 B/elem 两例） | `references/pattern-library/traps-compiler.md` |
| PL-1.10-loads-first-decoupling | P（优化模式） | 单 staging 复用链致 MTE2/VEC 零重叠 → per-input staging + 前置装载；prefill -31.7%（58.3→39.8µs） | `references/pattern-library/elementwise.md` + `repro/PL-1.10-loads-first-decoupling.py` |

### BP proposal（流程层，不自动改 SKILL）

- **BP_proposal: 参数分支的 L0 等价门禁定义**：本轮参数扫描分支（同 kernel 不同 bm）的精度门禁 = bench `--check`（精确 (M,N,dtype,bm) golden 比对）；结构分支 = 全量 `--level all`。该分层在 SKILL.md 未显式定义（"每个实验分支跑 L0 精度回归"对参数分支语义模糊——内嵌 L0 表并不覆盖被扫的 bm）。建议 SKILL.md Phase 2 第 6 步明确"参数分支可用 bench 内嵌 golden check 等价替代完整 L0 套件，结构分支必须全量"。
- **VP（D 类，建议蒸馏）**：`auto-multi-buffer 的 UB 需求随 num_local_tasks 变化`（nl=3→20.01 vs nl=4→22 B/elem @bm8×N1152）——已作为 TRAP 条目更新的一部分回写；evolver 蒸馏时注意与 TRAP-UB-multibuffer-inflation 合并去重。

### 工具/环境问题记录

- `ab_test.py` 建目录未 chmod 700，msprof 拒写（"writable by any other users or group users"）——需 `umask 077` 前置调用；建议工具内置 `os.chmod(out_dir, 0o700)`（已在复盘提出，不改工具代码）。
- v3_op3 编译失败（blocked）的复现证据保留在本日志与探针记录中；分支文件已删除以保持 perf_opt/ 布局整洁（blocked 结论不依赖文件存续，机制已回写 pattern-library）。
