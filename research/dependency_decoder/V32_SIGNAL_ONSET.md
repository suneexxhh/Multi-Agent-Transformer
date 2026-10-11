# V3.2 影响信号启动研究（CPU 离线诊断）

本分支以固定 V3.1 提交 `26cab82978e5ea80bd870619bc2d111be37ba260` 为父提交，只增加诊断脚本和测试。没有改动网络、PPO 目标、`min_effect` 或训练启动逻辑。完整数据与中文报告保存在 `/data/sgg/mat-decoder-storage/analysis/v32_signal_onset_3s5z_seed1`。

## 诊断定义

对真实 3s5z 联合观测、合法动作和固定历史动作，冻结 Bias OFF 的 checkpoint。将已解码源位置 `s` 的合法动作替换成至多 `K` 个其他合法动作，保持排列、观测和其余历史动作不变。只有 `s<t`、目标有至少两个合法动作且源有合法替代动作时，记录目标位置 `t` 的条件动作分布 JS：

`J[t,s] = mean_a JS(pi_t(.|h_s), pi_t(.|h_s←a))`。

在相同 `(t,s, normalized target entropy bin)` 分层内，以该层均值中心化得 `r[t,s]`。原 V3.1 目标为 `y=0.5+0.5*tanh(r/effect_scale)`，置信度还乘以合法替代覆盖率、目标熵、分层支持和 `|r| >= min_effect` 的门槛。不可观测反方向保持无效，不能填零。脚本调用原 `conditional_action_js` 与 `alignment_targets`，没有另造教师定义。

同一 320 个真实状态、8 个 episode、相同合法动作与历史动作供五个 checkpoint 使用。固定 identity 排列用于跨 checkpoint 控制；各 checkpoint 确定性分数排序另作独立对照。checkpoint 0 **已经完成第一次 PPO 更新**。所有模型按训练 manifest 重建并用 `strict=True` 加载。置信区间整组重采样 episode，不能视同轨迹 pair 为独立样本。

## 复现

在此代码工作区运行，使用 `new_titans`、CPU、`PYTHONPATH` 指向工作区，并把 `TMPDIR`、`TMP`、`TEMP` 指向分析目录下 `tmp`：

```bash
CUDA_VISIBLE_DEVICES='' python -m research.dependency_decoder.trace_influence_signal
CUDA_VISIBLE_DEVICES='' python -m research.dependency_decoder.diagnose_initial_conditioning
CUDA_VISIBLE_DEVICES='' python -m research.dependency_decoder.diagnose_label_stability
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_v32_signal_diagnostics -v
```

第一个脚本产生完整时间轨迹和原始冻结教师数组，第三个脚本依赖这些数组。真实状态源是先前已采集的 `real_smac_observations.npz`；脚本只读它及 V2 checkpoint，绝不运行 SMAC/PPO 训练。目标数据目录可用 `V32_ANALYSIS_DIR` 覆盖；仍需保持真实数据与 checkpoint 路径可用。

## 观察与后续实现边界

默认阈值下，V2 checkpoint 0 和 5 的置信覆盖为零，checkpoint 10 首次观测到广泛有效监督。首次形成时间只可定位于 checkpoint 5 与 10 之间；这不是 V3.1 on-policy 信号的实测时间。初始化动作替换引起非零 logit 与 TV 差异，float32/float64 的 JS 基本一致；微弱影响主要来自初始化时历史动作条件路径的衰减，而非 JS 数值计算伪差。

现有 SMAC launcher 每 PPO 更新只有一个 200-transition minibatch，同一 rollout 的 10 个 epoch 仅改变样本顺序，标签近似相同。若改成多个小 minibatch，分层样本与标签会发生实质变化。因此不得在没有重新诊断目标等价性时，直接把现有 `alignment_targets` 改成跨更新 EMA 或其他监督定义。

下一版应先实现可观测、预注册的启动监测和门控，保留固定 warm-up 对照。启动门槛需联合考虑合法有效 pair、episode 分组的 JS 噪声下界、分层残差、非零置信覆盖与权重有效样本量；不能只降低 `min_effect`。此分支没有实现任何训练门控，候选门槛及无法验证之处详见分析目录中的 `activation_strategy_comparison.md` 和 `research_report.md`。
