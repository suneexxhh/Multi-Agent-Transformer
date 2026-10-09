# MAT Decoder: learnable permutation actor (research prototype)

**Target comparison:** original MAT (identity order) versus MAT with alternate
ordering. **MAT-MSA is not the baseline in this phase**.

## Implementation
In `mat/algorithms/mat/algorithm/learned_agent_order.py`, the Plackett-Luce
permutation policy uses per-agent local observations and a mean-pooled shared
team context to compute scalar agent priorities. Without replacement, its
ordered sample `pi[b,k]` is the *original agent ID* decoded in slot `k`.
A stochastic training rollout saves `pi` and its **joint log probability**
`log p_phi_old(pi|obs)`. Original MAT autoregressive sampling,
teacher-forcing, legal masks, and inverse-order scattering remain unchanged.

During PPO, the original-action-order permutation is taken from the buffer,
NOT recomputed using the updated order model. New likelihood of exactly that
stored permutation is differentiable, and the ordering head receives an
extra clipped factor-wise PPO loss. Team advantage is the active-agent mean
of existing standardized MAT GAE. The usual MAT actor+critic update is
unchanged except for this additional ordering objective.

Plackett-Luce actor starts with equal scores (uniform sampled permutation;
identity for deterministic tied-score evaluation). Untrained priorities do
NOT identify causal dependencies, and rankings are not a learned DAG.

`--algorithm_name mat --agent_order_mode learned --store_agent_orders --order_hidden_dim 64 --order_temperature 1.0 --order_loss_coef 0.1`

**All tests are CPU-only** and must complete before attempting any SMAC
rollouts. The research branch/PR does NOT start any training workflows.

## Accelerated experimental decision rule (user-approved)
For *pilot* algorithm iteration compare matched seeds **1, 2, 3** for
MAT-identity, MAT-obs_norm, MAT-learned. Start with **3s5z** only after
ongoing GPU0 locked workload and automatic chain are finished; also allow
preapproved super-hard maps 3s5z_vs_3s6z, 6h_vs_8z, MMM2, 27m_vs_30m.
Remaining allowed maps are 1c3s5z, 5m_vs_6m, 8m_vs_9m, 10m_vs_11m.
Never initiate new 3m training. For each comparison ensure identical
step budgets, environment count, train/eval code SHA, PPO epochs, clip,
model dimensions, evaluation seeds, frozen checkpoint step and 32 native
evaluation episodes. A 100k single-rollout task is **pilot, not formally
matched multi-million-step MAT training**.

GPU0 maximum simultaneous training count is 6 **subject to available**
VRAM, CPU, RAM and exclusive lock; never touch GPU1/GPU2 or others' jobs.
No GPU submission is authorized merely by creating this branch or PR.
Three-seed evidence supports engineering decisions, not claims of
statistically significant performance superiority.

## Outstanding validation
- CPU integration tests: order normalization; rollout/teacher-forcing
  action likelihood consistency; old and new joint permutation log-probs;
  agent-axis restoration; PPO gradient into ranking head; strict saved
  permutation verification.
- Real native SMAC pilot remains **not run** until existing chained GPU
  pilot is complete, resource checks pass, and no lock is held.
- Factor-wise PPO is an approximation, NOT the exact clipped joint
  action-plus-permutation PPO likelihood ratio. Perform order-loss weight
  ablations and report learning stability.
