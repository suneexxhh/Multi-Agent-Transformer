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


## AR-03 V0 implementation status (2026-10-09)

**Implemented on this research branch**:
- `mat/algorithms/mat/algorithm/agent_ordering.py` defines deterministic `identity`, `random_fixed` and `obs_norm` order modes, plus batch-wise gather/scatter.
- `mat/algorithms/utils/transformer_act.py` wraps the exact original autoregressive sampling and teacher-forcing probability functions for discrete and continuous actions, restoring original agent IDs.
- `ma_transformer.py` computes current-observation-derived ordering in both `get_actions` and `forward`, while retaining the original Encoder, Critic and attention architecture. The identity branch calls the original functions.
- `transformer_policy.py` and `mat/config.py` expose `--agent_order_mode` and `--agent_order_seed`. For V0 nonidentity modes require `--algorithm_name mat`.
- `mat/scripts/train_smac_research.sh` supports `AGENT_ORDER_MODE` and `AGENT_ORDER_SEED`, map/gpu/seed override, and `DRY_RUN=1`.
- `tests/test_agent_ordering.py` validates ordering, inverse permutation, legality masks, action sampling vs log-prob evaluation, full MAT model forward, Decoder gradients, continuous decoding, and causal mask nonleakage.
- `.github/workflows/decoder-v0-cpu-tests.yml` executes the regression tests in GitHub Actions, independently of SMAC.

**Important design decision**: V0 ranking is a deterministic function of the *current observation* (or a constant seeded order). PPO rollout and evaluation are therefore guaranteed to reconstruct identical permutations from the stored observations and the same configuration, even if neural network weights change. **Storing `permutation` in the rollout Buffer is unnecessary in V0** and has been deliberately deferred. V1 stochastic or parameterized order models will need new trajectory metadata and/or carefully defined order-likelihood terms.

**Interpretation**: `obs_norm` is a heuristic sorted by mean absolute raw observation magnitude; it is used to validate the infrastructure, **not** to estimate action dependence or causal relations. Feature scale can affect this order; only empirical comparisons can establish whether the heuristic is useful.

**Confirmed CI result (initial V0 tests)**: run [37879083980](https://github.com/suneexxhh/Multi-Agent-Transformer/actions/runs/37879083980) passed 10 original CPU tests plus launcher syntax/dry-run. Additional regression tests and CI diagnostics were added thereafter; check the latest workflow run for final status.

**No SMAC victory-rate claims**: no remote GPU/SMAC training has been run in these steps. The original MAT reference environment and Torch versions differ from the CPU test runner; server-side reproduction is required before performance claims.

### First training commands

Execute from the repository root on the configured SMAC server:

```bash
git fetch origin
git switch research/dependency-decoder-v0

# Do not launch a training run:
DRY_RUN=1 AGENT_ORDER_MODE=obs_norm bash mat/scripts/train_smac_research.sh 3m 0 1

# Short compatibility experiment after Python/SC2 are verified:
NUM_ENV_STEPS=100000 PPO_EPOCH=1 EXP_TAG=smoke AGENT_ORDER_MODE=identity \
  bash mat/scripts/train_smac_research.sh 3m 0 1

# Comparable long runs: change only the agent-order option
AGENT_ORDER_MODE=identity bash mat/scripts/train_smac_research.sh 3m 0 1
AGENT_ORDER_MODE=random_fixed AGENT_ORDER_SEED=1 bash mat/scripts/train_smac_research.sh 3m 0 1
AGENT_ORDER_MODE=obs_norm bash mat/scripts/train_smac_research.sh 3m 0 1
```

For truly automatic GPU experiment looping, a separately authorized server-side coding/execution agent or runner must be installed and have resource limits, failure conditions and logging configured.
