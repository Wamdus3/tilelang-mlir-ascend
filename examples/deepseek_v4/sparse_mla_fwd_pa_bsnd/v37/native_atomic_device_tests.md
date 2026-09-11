# 显式 atomic 接口设备验证（2026-09-11）

当前 TileLang 原生 `T.set_atomic_add` / `T.set_atomic_none` 版本的54次精度调用全部通过。
编译驱动不再推断或改写 atomic；算子明确设置 atomic 区域及流水屏障。
保持原有手动同步、tile512、170KiB UB 和 multibuffer 布局。

## 精度

B=4、等长 S=T/4，BF16，H32/D512/R64，page128。每个 shape 覆盖 mode0/3、
random/zero/wide 三类输入，各重复3次。包含 workspace 污染后复用、metadata epoch0→1→0，
并检查 Output/Max/Sum 和空行。以下误差为所有模式及输入的最大绝对误差。

| T | topk | 通过调用数 | Output 误差 | Max 误差 | Sum 误差 |
|---:|---:|---:|---:|---:|---:|
| 16 | 512 | 18 | 0.003656745 | 6.6757202e-06 | 0.00067138672 |
| 512 | 2048 | 18 | 0.0045662522 | 1.1444092e-05 | 0.0025024414 |
| 4096 | 2048 | 18 | 0.0047923923 | 1.1444092e-05 | 0.0031738281 |

未修改逐元素容差：Output rtol0.02/atol0.002，Max rtol0.0002/atol0.00002，
Sum rtol0.0002/atol0.001。最大绝对误差本身不作为失败判据。
T512/mode3 首次编译出现一次已知 CANN SIGSEGV，自动重试后通过。
另外两组精度及两组性能采集未出现编译重试，无数值或设备运行失败。

## Kernel 性能

T512/topk2048，910B2C 的13号卡，1800MHz，24 AIC/48 AIV。
使用 msprof op 的 application replay、warm-up5、PipeUtilization，
每个 mode20个有效 `OpBasicInfo` 样本，中位数单位为微秒。
已核对 kernel 名称、频率、设备及 launch 维度；全部样本均计入统计。

| mode | 当前 TileLang | 历史 TileLang | 历史 AscendC | 较旧 TileLang 耗时变化 | 相对 AscendC 性能 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1427.147 | 1410.736 | 1364.495 | +1.16% | 95.61% |
| 3 | 1423.697 | 1412.086 | 1363.665 | +0.82% | 95.78% |

相对性能=AscendC 耗时/TileLang 耗时，两种 mode 均超过80%目标。
旧 TileLang 和 AscendC 均复用冻结记录，没有重跑。
本次耗时比旧记录略高，但不是同批次对照，不能据此判断接口改动造成了性能回退。
此前 T512/topk2048/mode3 的三个目标文件已证明与旧版本逐字节一致。
本次未测其他 shape 的性能，未扩展到非均匀长度或 A5。

详细样本、容差和原始记录路径见 [测试数据](native_atomic_device_tests.json)。
历史完整性能表仍在 [performance.md](performance.md)，原始记录未被覆盖。
