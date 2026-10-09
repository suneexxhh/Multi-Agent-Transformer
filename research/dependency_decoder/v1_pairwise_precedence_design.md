# AR-04 (V1 scaffold) — Directed Pairwise Agent Precedence

**Status:** V1 pairwise scorer remains *isolated* and untrained. A separate,
opt-in stored-order transport path is now connected across SMAC rollouts,
Replay Buffer, MAT sampling/evaluation, and PPO Trainer. This opt-in path
has been validated with real GPU training; it does not turn on a
learned scorer or claim directed causal dependency inference.

## Verified foundation

- Three V0 modes `identity`, `obs_norm`, `random_fixed` have completed GPU 0 / `new_titans` / SMAC `3m` 2,000-step smoke tests.
- On 2026-10-09 the revised SMAC launcher emitted
  `MAT_obs_norm_seed1_20261009_141548_202424176.log`;
  [private run 37892480786](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37892480786)
  passed its filename, metadata, exit-code, and Artifact upload checks.
- These are functionality checks, **not** win-rate results or convergence claims.

## Precise meaning of the prototype graph

Given local agent observations `o_i`, compute an order-sensitive pair score

`s_ij = f_phi([o_i; o_j; o_i - o_j])`.

The pairwise decoding-precedence probability is

`P_phi(i ≺ j | o) = sigmoid(s_ij - s_ji)`.

Hence `P_ij + P_ji = 1`, and the diagonal is `0.5`. A deterministic
fallback ranking is obtained by descending expected pairwise wins,
`r_i = Σ_j P_ij`, with stable ties by original agent ID. This is a
Borda-style tournament ranking; it remains defined when the soft graph
has cycles.

**These are precedence preferences, not observed causal action
dependencies.** The scorer starts randomly initialized. Predictions
have no scientific interpretation before establishing and evaluating
a reliable training signal.

Implementation: `mat/algorithms/mat/algorithm/dependency_ordering.py`.
Regressions: `tests/test_dependency_ordering_v1.py`.

## Why we must not simply call the scorer in existing MAT PPO

MAT executes agent actions under a causally masked autoregressive decoder.
PPO must evaluate each sampled action under the **same agent permutation**
used when that action was generated. With a learned, mutable scorer,
recomputing `order(o)` after the scorer parameters changed can alter the
autoregressive action factorization; then old/new log-probability ratios
compare different joint action distributions and do not implement the
intended clipped PPO objective.

V0 avoids this by deriving a deterministic order solely from observations
and a constant seed. V1 does **not** have that guarantee.

## Saved-order infrastructure completed on research branch

- `--store_agent_orders` is opt-in and defaults to false; the unchanged
  V0 and other benchmark execution paths still use their legacy tuple
  interfaces. `STORE_AGENT_ORDERS=1` enables the SMAC launcher flag.
- SMAC Runner collects an original-agent permutation per time/environment
  transition; `SharedReplayBuffer.agent_orders` stores `int64[T,E,N]`.
  Validation rejects duplicated, missing, or out-of-range IDs.
- The minibatch generator pairs each order `[B,N]` with its corresponding
  `obs`, `actions` and masks. It does **not** shuffle decoder positions as
  independent experience samples.
- `TransformerPolicy` and `MATTrainer` forward stored orders to MAT's
  teacher-forced action log-probability calculation. No scorer re-evaluation
  occurs in this PPO path. Returned actions and log-probs still use original
  agent IDs; Encoder and Critic remain untouched.
- [CPU CI run 37893867478](https://github.com/suneexxhh/Multi-Agent-Transformer/actions/runs/37893867478):
  26 tests, 23 passed, 3 CUDA-only skipped; includes a trainer spy test
  confirming the permutation is forwarded unchanged. CPU tests do not prove
  that an end-to-end GPU SMAC rollout is correct.
- **Actual GPU validation succeeded:** [private Run #37894004324](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37894004324)
  activated `STORE_AGENT_ORDERS=1` on GPU 0 in `new_titans` and
  executed SMAC `3m` for 2,000 requested steps, 1 rollout env, 1 PPO epoch.
  The run completed with code 0 and executed all 26 unittests with no skips
  (including CUDA tests). It generated and uploaded
  `mat/scripts/logs/MAT_obs_norm_seed1_20261009_143245_446718211.log`
  ([Artifact](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37894004324/artifacts/11599518217)).
  **This validates the transport infrastructure, not the untrained scorer
  or any win-rate improvement.**

## Required next engineering milestones

1. **Still required: trustworthy pairwise supervision.** Define and test
   edge labels on controlled multi-agent toy domains where directed
   influence is specified externally. Decoder action sensitivity alone
   is a proxy for model dependence, **not** environment causation.
2. **Still required: learned scorer PPO integration.** Add a separately
   gated `learned_precedence` mode and a freeze/update policy for the
   scorer. Use the already stored rollout permutation for every PPO
   minibatch, even after scorer weights change.
3. **Still required: weight-change regression.** Generate a rollout,
   deliberately modify the scorer's parameters, and assert the PPO
   log-probabilities and importance ratios use the original stored
   ordering while Decoder weights are fixed.
4. **Still required: empirical validation.** Start with one SMAC env on
   GPU 0, then matched seeds and evaluation budgets. Win rate,
   convergence and stability must be measured separately; a 2,000-step
   smoke test is not a performance result.

## Safety and scope

Do not terminate or manipulate any GPU 1 / GPU 2 workloads. The existing
private controller uses GPU 0 exclusively and `conda activate new_titans`.
Do not launch long multi-environment experiments from this scaffold.
