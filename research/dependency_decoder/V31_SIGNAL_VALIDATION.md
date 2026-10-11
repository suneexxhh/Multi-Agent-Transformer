# V3.1 影响监督信号与教师成本验证

本分支基于固定 V3.1 提交 `2caaae2eb4c4279bdf26cac10bfc4f5b0dcc11ae`。只新增 CPU 诊断及可选教师原始结果缓存，不改变排序策略、监督目标、Decoder 或训练预算。运行记录与完整中文结论位于 `/data/sgg/mat-decoder-storage/analysis/v31_influence_signal_validation/research_report.md`。

## 缓存语义

一次 `MATTrainer.train()` 对应一个 rollout 和一个冻结教师快照。原实现每个 PPO epoch、每个 minibatch 都重复计算同一 transition 的原始教师 `(JS, valid, target_entropy, alternative_count)`。启用 `--cache_order_influence_teacher` 后，回放生成器为每个联合 transition 返回该 rollout 内稳定的扁平 ID。教师前向仅对尚未见过的 ID 执行，缓存随下一次 `train()` 清空。

**分层标签仍逐 minibatch 计算**：`alignment_targets` 的同一源/目标解码位置与目标动作熵分层，依赖本次 minibatch 的成员。缓存其结果会改变训练目标。缓存只保存冻结教师原始张量，按当前 minibatch 顺序组装，再调用原 `alignment_targets`。默认缓存关闭；配置写入 schema 3 manifest。旧 schema 1/2 清单可迁移读取，旧模型严格加载不受该选项影响。

## 严格六组消融

| 组 | 固定顺序 | 前缀排序 | 关系注意力 | 影响对齐 | 要回答的问题 |
|---|---:|---:|---:|---:|---|
| A: MAT identity | 是 | 否 | 否 | 否 | 原始 MAT 基线 |
| B: prefix NoBias | 否 | 是 | 否 | 否 | A→B：仅排序的作用 |
| C: prefix fixed Bias | 否 | 是 | 固定 `log(2D)` | 否 | B→C：V3 固定偏置的作用 |
| D: prefix NoBias + alignment | 否 | 是 | 否 | 是 | B→D：仅影响对齐的作用 |
| E: prefix adaptive Bias | 否 | 是 | 可学习有界 `ρ` | 否 | B→E：自适应偏置本身；C→E：相对固定偏置 |
| F: prefix adaptive Bias + alignment | 否 | 是 | 可学习有界 `ρ` | 是 | E→F：注意力开启时对齐的增量；D→F：对齐存在时注意力增量 |

所有组须严格配对地图、随机种子、rollout、PPO、预算和冻结评估；只有 D/F 会调用教师。`CACHE_ORDER_INFLUENCE_TEACHER` 在 D/F 中独立对照，验证纯性能优化。旧五组缺少 E，不能独立区分「自适应注意力本身」与「对齐和注意力的联合效果」。

## 解释边界

离线数据为 V2 机制分析固定采集的 320 个真实 3s5z 联合观测、合法动作及采集策略动作。V2 checkpoint19 的 Bias OFF 信号是**已训练策略的对照**；随机初始化 V3.1 策略在同一真实状态上采样的顺序和历史动作是**初始化时的近似**。两者都不是新的 V3.1 PPO rollout，也没有训练过的 V3.1 `D(target,source)`。成组统计按 8 个 episode 自助抽样，不能把 5836 个 pair 当作独立 episode。教师 JS 仅测冻结策略的条件动作敏感性，不能声称环境因果作用。

## 复现

使用 `new_titans` 环境，将 `CUDA_VISIBLE_DEVICES` 设为空，`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`，并将 `PYTHONPATH` 指向本仓库。将 `TMPDIR/TMP/TEMP` 均设为 `/data/sgg/mat-decoder-storage/analysis/v31_influence_signal_validation/tmp`。从仓库根目录运行：

```bash
python research/dependency_decoder/analyze_v31_signal.py
python research/dependency_decoder/analyze_v31_signal.py --v31-init
python research/dependency_decoder/profile_v31_teacher.py
python research/dependency_decoder/probe_v31_gradients.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

`profile_v31_teacher.py` 是合成 PPO 回放的 CPU 耗时剖析：输入观测和合法动作来自真实 V2 数据，回报和优势是受控合成值，绝不运行 SMAC 环境或正式训练。三次独立进程剖析的原始 JSON、日志、真实状态信号数据和报告均保留在上述分析目录。
