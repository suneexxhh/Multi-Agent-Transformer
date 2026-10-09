# AR-04 (V1 scaffold) — Directed Pairwise Agent Precedence

**Status: scorer + independent tests only; NOT yet wired to PPO or any live SMAC training.**

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

## Required next engineering milestones

1. Define defensible edge supervision and test it on toy multi-agent
   problems with known directed decision influence. Cross-agent
   *policy* sensitivity under action interventions may serve as a proxy,
   but it does **not** establish environment causal influence.
2. Extend the rollout buffer to retain `agent_order[B,N]` for each
   sampled timestep; transport it through mini-batch generators.
3. Modify `TransformerPolicy.get_actions` and
   `evaluate_actions` to share the stored order for the PPO update,
   with no recomputation from changed scorer weights during epochs.
   Train the scorer at carefully separated update boundaries, or
   freeze it during policy rollout/update cycles.
4. Ensure sampled actions, action masks, log-probs, entropy and values
   are consistently returned by original agent ID. **Do not change the
   current Encoder/Critic**.
5. Add a strict regression where scorer weights are changed after
   rollout; stored-order old/new log-probabilities must remain
   consistent when the decoder is unchanged.
6. Only then expose a gated V1 CLI mode and run a single-environment
   GPU 0 / SMAC `3m` diagnostic. Later evaluate seeds, maps,
   convergence, and significance with fair matched baselines.

## Safety and scope

Do not terminate or manipulate any GPU 1 / GPU 2 workloads. The existing
private controller uses GPU 0 exclusively and `conda activate new_titans`.
Do not launch long multi-environment experiments from this scaffold.
