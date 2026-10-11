# V3.1：策略条件影响对齐与有界自适应关系注意力

**状态：CPU 算法原型。** 本分支从 V3 `research/v3-prefix-dependency-decoder` 的 `519bc237dc6ce254134594ac33950eca8c966b0f` 开始。没有执行 GPU、SMAC 或 PPO 训练。训练、评估与统计均须在新固定源码 SHA 下重新配对；V2 `90e20b8` smoke 和 15 组 Pilot 不可与此分支混用。

## 实证依据与代码审查

V2 在 320 个真实 3s5z 状态上的机制分析见独立报告 `/data/sgg/mat-decoder-storage/analysis/v2_dependency_mechanism_3s5z_seed1/research_report.md`：4k 步 checkpoint 19 的静态 PL 排列熵仍为均匀值的 **99.632%**；内部 `p(source,target)` 与动作条件 JS 的关联随解码上下文改变；Bias ON/OFF 的平均动作 JS 仅 **2.398×10⁻⁶ nats**，虽然注意力明显变化。这些只是短程单种子机制证据，无法断言性能优劣。V2 NoBias 采样与 PPO 不一致已由 PR #4 修复，V3 PR #5 继承了修复。

本次审查确认 V3 `PrefixDependencyOrder` 的 `D[target,source]` 非反对称索引、前缀逐步 softmax 的采样和精确 `log_prob`、按原 agent ID 还原动作与 log probability、PPO 保存旧排列及旧联合排列概率、`shifted_action` 的 source 槽 `s+1`、下三角因果遮罩，以及 NoBias 在采样和回放两侧的统一 gate。V3 的动作因子和排列因子分别裁剪，**并非联合动作与排列的单一 PPO ratio**。原始 MAT Encoder/Decoder 在可选排序网络之前初始化，配对同种子主干权重一致。已发现并修正冻结评估结果中 `dependency_prefix` 的 `effective_order_attention_bias` 报告遗漏。V3 固定偏置在初始化时不为零，所以新增零初始化增益作为独立消融，不改变 V3 固定偏置路径。

## 排序与关系定义

令 (o=(o_1,ldots,o_n))。有向需求 logit 和概率为

\[
d_{i\leftarrow j}=q(h_i)^\top k(h_j)/\sqrt r,\qquad
D_{i\leftarrow j}=\sigma(d_{i\leftarrow j}).
\]

两方向可同时很高；它不是 V2 反对称先后偏好。已选 agent **ID** 集合 (S_t) 下，对候选 (i\notin S_t)：

\[
u_i(S_t)=b_i(o)+\frac{\lambda}{T}
\left[\operatorname{mean}_{j\in S_t}D_{i\leftarrow j}
-\operatorname{mean}_{j\notin S_t,,j\ne i}D_{i\leftarrow j}\right],
\quad P(\sigma\mid o)=\prod_t\operatorname{softmax}_{i\notin S_t}u_i(S_t)[\sigma_t].
\]

空集合均值取 0。完整排列在动作生成之前采样；V3.1 **仍不根据已经解码的动作值改变后续排序**。PPO 回放必须使用 rollout 保存的 `sigma` 和旧 `log P_old(sigma|o)`，不得重新采样或以当前网络概率代替旧概率。

## 冻结 Bias OFF 影响教师

可选系数 `--order_influence_align_coef > 0` 只允许 `dependency_prefix`。每次 PPO 更新前复制当前 Transformer 为快照 \(\bar\theta\)，冻结参数并置 eval，关闭关系偏置。教师只读取该次训练 rollout 的 minibatch：原始观测、合法动作掩码、已采样动作及其已采样排列；不读冻结评估样本，也不调用环境或 optimizer。对已解码源槽 \(s<t\)，将历史动作 \(a_{\sigma_s}\) 换成另一**合法**动作，保持所有观测、目标合法集合、排列与其余历史动作不变。最多取 `K=2` 个合法替代动作，沿动作 ID 范围确定性均匀选取；取不到替代动作、目标仅一个合法动作或反向不可观测时保留无效掩码。

\[
J_{t,s}=\frac{1}{|A'_{s}|}\sum_{a'\in A'_s}
\operatorname{JS}\bigl(\pi^0_{\bar\theta,\sigma_t}(\cdot\mid o,a_{<t}),
\pi^0_{\bar\theta,\sigma_t}(\cdot\mid o,a_{<t}[a_s\gets a'])\bigr).
\]

JS 在合法目标动作上按 double precision log-softmax 求值，取值区间 \([0,\ln2]\)。前向在 `torch.no_grad()` 内；教师参数 `requires_grad=False`。此 \(J\) 是**冻结策略内部**的条件分布敏感性，不是改变环境中某 agent 动作后另一 agent 的因果效果。rollout 动作可来自 Bias ON 学生，而教师始终 Bias OFF；因此此目标带有历史动作分布偏移，需在正式训练中检验。

原始 JS 强烈受源/目标槽位及目标动作熵影响。将同一 minibatch 按 `(source_slot, target_slot, target_entropy_bin)` 分层；每层至少 2 个有效状态。用层均值 \(\mu_g\) 得到残差 \(\delta=J-\mu_g\)，并构造软目标

\[
y_{t,s}=\tfrac12+\tfrac12\tanh(\delta_{t,s}/\tau),\quad \tau=10^{-5}\ \text{nats}.
\]

置信权重为

\[
w_{t,s}=\mathbf1[|\delta|\ge10^{-7}]
\min(1,|\delta|/\tau)
\frac{\#\text{选取替代动作}}{\#\text{全部合法替代动作}}
\frac{H(\pi^0_t)}{\ln |A_t|}
\min(1,|g|/8).
\]

所有 target 与 confidence `detach()`。未观测方向和单例层的权重为 0，不能用 0 JS 假装“无依赖”。监督直接作用于同一排列位置映射的 \(d_{\sigma_t\leftarrow\sigma_s}\)：

`τ` 与有效性门槛分别由 `--order_influence_effect_scale`、`--order_influence_min_effect` 设置并写入 manifest。其默认值参考 V2 真实状态实验中约 `1.5×10⁻⁵` nats 的平均条件动作 JS；这是**量纲标定**而非经训练验证的最优超参数。分层残差可能比原始 JS 小一个数量级，因此门槛设为 `10⁻⁷`，并必须监测实际受监督比例及噪声敏感性。

\[
L_{\mathrm{align}}=\frac{\sum_{b,t,s}w_{b,t,s}\,
\operatorname{BCEWithLogits}(d_{\sigma_t\leftarrow\sigma_s},y_{b,t,s})}
{\sum_{b,t,s}w_{b,t,s}},
\]

权重和为零时损失为可微的 0，并记录有效与真正受监督 pair 数。位置和熵分层只降低这些可见混杂，不控制隐藏历史、动作类别语义、合法动作空间结构或轨迹选择偏差。小 batch 与弱动作敏感性可能使监督长期为零，这必须从日志确认。

## 自适应关系注意力

原 V3 固定偏置保留为独立对照。V3.1 自适应开关只在 Bias ON 时有效：

\[
\beta=\beta_{\max}\tanh(\rho),\quad \rho_0=0,\quad \beta_{\max}=2;
\qquad B_{t,s+1}=\beta\,[\ln 2+\log\sigma(\operatorname{clip}(d_{\sigma_t\leftarrow\sigma_s},-8,8))].
\]

BOS 列为零，未来槽仍被原 MAT 因果遮罩屏蔽。\(\beta_0=0\) 使**相同排列、相同主干权重**下的条件 Decoder 与原 MAT 完全一致；NoBias 消融则完全不计算 bias。增益可以为正或负，绝对值严格小于 2；负号可检验是否学到与当前关系定义相反的融合方向。`attention_scale` 与 `attention_scale_grad` 纳入 PPO 日志；`SelfAttention.capture_attention` 可选地提取 detached 注意力矩阵用于 CPU 诊断，默认关闭，训练时无额外存储。

联合目标仍是现有 MAT 动作 PPO、价值损失、策略熵与排列 PPO；仅在开关开启时增加 \(c_{align}L_{align}\)。教师不经过反向传播；学生的 demand 参数可从排列 PPO、注意力动作 PPO 和对齐 BCE 收梯度。零增益初始化时，注意力动作梯度先到 \(\rho\)，对 demand 的注意力梯度初始为零，随后才打开；demand 仍可由排列及对齐损失更新。

## 严格消融矩阵

| 试验 | `AGENT_ORDER_MODE` | `DISABLE_ORDER_ATTENTION_BIAS` | `ORDER_INFLUENCE_ALIGN_COEF` | `ADAPTIVE_ORDER_ATTENTION_BIAS` |
|---|---|---:|---:|---:|
| 原始 MAT | `identity` | 0 | 0 | 0 |
| V3 前缀排序 | `dependency_prefix` | 1 | 0 | 0 |
| V3 前缀 + 固定关系注意力 | `dependency_prefix` | 0 | 0 | 0 |
| V3 前缀 + 影响对齐 | `dependency_prefix` | 1 | >0 | 0 |
| V3 前缀 + 影响对齐 + 自适应注意力 | `dependency_prefix` | 0 | >0 | 1 |

所有模型从同一源码 SHA、同一配对种子和相同 MAT 训练协议独立从头训练，保存严格 manifest 与 checkpoint。正式冻结评估数据必须与教师训练 minibatch 分离。先比较上表 5 种机制在三个配对种子的原生 SMAC 胜率、样本效率、排序熵、有效教师监督率、位置分层 JS、bias 尺度、注意力 TV、动作 JS、PPO ratio/clip 和梯度范数；NoBias 训练消融不能用同一权重推断期关偏置替代。V2/ V3 的 4k 单种子结果不提供胜率改善证据。

## 计算开销与尚未验证的问题

一次教师前向和至多 \(O(NK)\) 次 batched 反事实 Decoder 前向，另需一份冻结模型快照。当前实现对每个 PPO minibatch、每个 epoch 重算教师 JS；因此即使 `K=2`，训练成本也可能显著高于 V3。它是**正确性优先的原型**，正式长训练前须 CPU/GPU 性能剖析、可重复的标签缓存策略及内存上限评估。没有启动任何 GPU 训练。

严格 CPU 测试覆盖解析 JS、合法动作与单一合法动作跳过、不可观测方向、位置/熵分层、学生/教师梯度隔离、实际 PPO 接入、偏置方向和 shifted 索引、注意力权重及动作概率响应、未来遮罩、零增益 Bias ON/OFF 一致、保存排列回放、原 MAT 配对初始化、manifest 迁移及 strict checkpoint 加载。注意力权重发生变化但输出不一定有实用效果；原始 MAT 初始化下动作 log probability 对尺度的梯度极小，不能仅凭此模块存在声称性能收益。若长训练监督有效率很低，应先改教师采样与置信策略；若关系偏置继续只改变注意力而不改善动作或胜率，应考虑删除偏置、保留更简洁的 Decoder。动作值条件排序需要新的交错采样/回放定义，当前 V3.1 未实现，也不能被本原型的 CPU 测试证明更优。
