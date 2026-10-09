# Dynamic Dependency-Guided MAT Decoder — V0 Research Protocol

Status: **research setup, no algorithm modification or training results yet**.  
Upstream: https://github.com/PKU-MARL/Multi-Agent-Transformer  
Base commit: `be3ff49c8264d454c1fe2c41582aa2bfc98498c8`  
Branch: `research/dependency-decoder-v0`

## Objective

Improve MAT's *action decoder* using state-conditioned, directional agent decision dependencies, without changing the original Encoder/Critic. Distinguish **conditional decision dependency** from **causal effects of interventions**.

## Scope for V0

1. Implement an agent permutation utility for tensors shaped `[batch, num_agents, ...]`: permute by per-sample `order`, restore using inverse permutation.
2. Implement deterministic, fully observed order baselines (identity, fixed randomized permutation with reproducible seed, observation-derived heuristic) before learning a stochastic order policy.
3. Preserve the current Decoder and triangular causal mask; reorder `obs_rep`, `obs`, `available_actions`, and action tokens consistently. `env.step` must always receive actions in the original agent-ID order.
4. Modify sampling in `mat/algorithms/utils/transformer_act.py` and log-probability evaluation using the **same** permutation. Store rollout permutation in `mat/utils/shared_buffer.py` when applicable.
5. Keep existing Encoder, value head, PPO loss and reward setup unchanged for V0. Later learned stochastic ordering requires explicit likelihood/gradient design, not naive reuse of agent-wise PPO ratios.

## Relevant original code

- `mat/algorithms/mat/algorithm/ma_transformer.py` — `MultiAgentTransformer`, `Decoder`, `DecodeBlock`
- `mat/algorithms/utils/transformer_act.py` — discrete/continuous sampling and teacher-forced evaluation
- `mat/algorithms/mat/algorithm/transformer_policy.py` — public policy interface
- `mat/utils/shared_buffer.py` — rollout data and mini-batching
- `mat/algorithms/mat/mat_trainer.py` — PPO probability ratio and loss
- `mat/runner/shared/smac_runner.py` — SMAC policy interaction

## V0 acceptance criteria (before SMAC training)

- `identity` permutation exactly reproduces original MAT sampling distributions and action log-prob evaluation at fixed network weights.
- Sampling and teacher-forced evaluation are consistent for identical inputs, actions, weights, and order.
- Different per-batch permutations invert correctly; available-action masks track original agent identities.
- No causal-mask leakage of ungenerated actions.
- Check behavior on CPU and GPU where available. Validate discrete action first, continuous action next.
- Keep a reproducible baseline with explicit seed, environment steps, map, evaluation episodes and code commit.

## Later versions

- **V1:** directed agent dependency graph estimation, directed edge validation, graph sparsification and cycle handling.
- **V2:** dynamically learned order, careful PPO objective for order likelihood, and sensitivity experiments.
- **V3:** graph-restricted / partial-parallel action decoding *only after* verifying conditional independence assumptions.

## Materials pending from researcher

- A known working SMAC/MAT training command or shell script, with map, seed, GPU, environment versions.
- GPU/CUDA/PyTorch information and approved resource budget if server-side automatic experiments are desired.
- Decide when and how a server-side coding/experiment agent will have access.

## Operating rules

- All algorithm experiments on this research branch or its descendants; do not alter `main`.
- Each experiment requires: hypothesis, diff/commit, correctness checks, training config, metrics, result and keep/revert decision.
- Do not claim a causal relationship from attention scores alone. Do not claim SMAC reproduction without running it.
