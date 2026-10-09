# MAT Decoder V1 — Directed dependencies and efficient decoding

**Research status (2026-10-09):** V0 action-permutation and V1 opt-in PPO
stored-order transport passed SMAC GPU 0 functional tests. V1.1 low-rank
precedence scorer is **isolated**, untrained on SMAC, and NOT enabled by
the policy or SMAC launcher. No causal discovery, parallel action generation
or win-rate improvement has been demonstrated.

## Guiding principles (original Decoder research direction)

The contribution must be more than reordering agents: explicitly estimate
and validate directed **decision dependencies**, use a graph to constrain
generation order, and only consider topological-layer parallelism if the
joint action distribution is mathematically correct. Preserve original MAT
Encoder and Critic, action-ID restoration, legal action masks, and PPO
sampling/evaluation consistency. Use minimal new parameters and kernels.

### V1.1: low-rank antisymmetric precedence score (implemented)

Let `h_i in R^D` be per-agent features (initially an observation or,
later, the existing MAT encoder's representation). With rank `r << D`:

```text
q_i = W_q h_i + b_q,    k_i = W_k h_i + b_k       (q_i,k_i in R^r)
s_ij = q_i^T k_j / sqrt(r)
P_ij = sigmoid(s_ij - s_ji)
r_i  = sum_j P_ij
sigma = argsort(r_i, descending, stable=True)
```

Thus `P_ij + P_ji = 1` and `P_ii = 0.5`.
A single batched `Q K^T` and its transpose produce all pair scores,
without allocating the previous `[B,N,N,3D]` concatenated observation
tensor. The scoring work is `O(B*N*D*r + B*N^2*r)`, with auxiliary
activations `O(B*N*r + B*N^2)`, rather than running an MLP on each
observation pair. The parameter count is `2*r*(D+1)`.

This is **directed precedence preference**, NOT policy action influence
and NOT the causal environment-dependency graph. Without valid
supervision, randomly initialized `P_ij` is not meaningful.
Do not enable the scorer as an action scheduler merely because its
source file and synthetic tests work.

### Sparse, acyclic candidate graph (implemented only as a projection)

After deriving `sigma`, keep edges only where:

```text
A_ij = 1[P_ij > tau] * 1[rank_sigma(i) < rank_sigma(j)],   0.5 < tau < 1.
```

Because all edges advance in the same strict ranking, cycles are
impossible. A `B,N,N` boolean adjacency and the agent permutation
can be generated from `precedence_dag`. The projection rejects
confidence-reversing edges to enforce acyclicity; this structural
projection is **not proof that the retained edges are causal**. A graph
containing no edge does not imply the MAT decoder's current policy
already factorizes independently.

### MAT's joint policy and PPO invariant

For a fixed order `sigma`, the decoder models

```text
pi_theta(a | o, sigma)
  = product_{k=1..N} pi_theta(a_sigma[k] | o, a_sigma[:k], sigma)
```

The original MAT decoder is causally masked: the output for position
`k` can depend on the earlier actions only. For each rollout
transition `(t,e)`, the sampled order `sigma[t,e,:]` is persisted in
`SharedReplayBuffer.agent_orders` and replayed in the same PPO
minibatch as observations, actions, legal masks, and old action log-probs.
The learning rate and agent order can otherwise change between collection
and PPO epochs, changing the autoregressive factorization.

The opt-in flag `STORE_AGENT_ORDERS=1` activates the transport path.
Only this existing transport path has been GPU smoke validated;
`identity`, `obs_norm`, and `random_fixed` continue to work by
default. For any stochastic distribution `rho_phi(sigma|o)`, the full
joint policy is `rho_phi(sigma|o) * pi_theta(a|o,sigma)`: a future PPO
ordering update must account for the order probability too. Simply
reusing old orders does NOT train `phi` correctly.

### Mathematical route to meaningful supervised dependencies (offline diagnostic implemented)

An implemented, **diagnostic-only policy-dependence proxy** holds
observations and all other preceding actions fixed, then replaces one
earlier action `a_j` with a different legal action. The KL of a later
conditional action distribution measures this specific perturbation:

```text
C_(j->i) =
  E_[a'_j in legal_j] KL(
      pi_theta(a_i | o, a_<i, sigma)
      ||
      pi_theta(a_i | o, a_<i with a_j <- a'_j, sigma)
  ),  j precedes i under sigma.
```

This measures **within-policy conditional sensitivity**, not a causal
effect on environment dynamics or optimal joint return. The implementation
`research/dependency_decoder/action_influence_probe.py` uses one baseline
decoder pass and one **batched** perturbation decoder pass, returning a
`[B,N,N]` directed KL matrix in the original agent-ID axes. It checks
action availability, requires eval mode, and skips agents without legal
alternatives. The diagnostic has two Python decoder calls but evaluates
up to `N-1` parallel perturbation contexts, so it is **not** constant
compute or a free parallel-policy speedup. The alternative legal action
is deterministic (first available) and the resulting KL is not an
unbiased intervention average or label by itself.

Synthetic tests `tests/test_action_influence_probe.py` isolate a known
toy decoder edge and require only that link to be positive while all
others remain zero. This is a unit test of our calculation, not an
experimental finding on StarCraft II. It only
observes directed pairs present as preceding/succeeding in the
chosen order, so it cannot identify both directions without
additional ordering contexts. It may reflect attention/model
artifacts. Validate candidate labels first on small controlled toy
games whose directed dependencies are specified externally; then
evaluate whether learned precedence predicts held-out sensitivity
or improves return. Do not use arbitrary observation norm as a
label for causal dependence.

### Robust offline sensitivity and identifiability (implemented)

The offline `action_influence_probe.py` now enumerates **all legal
alternative actions** other than the trajectory action for each predecessor,
without adding model parameters or modifying PPO:

```text
A_j^- = A_j^legal \\ {a_j}
C_(j->i)(o,a,sigma) =
    1 / |A_j^-| * sum_(a'_j in A_j^-)
       KL[ pi_theta(. | o,a_<i,sigma) ||
           pi_theta(. | o,a_<i with a_j <- a'_j,sigma) ] .
```

This **uniform average over discrete action alternatives** is not an
expectation under the learned policy, nor a causal influence on game
dynamics. On a given ordering context, the probe can evaluate only
`rank_sigma(j) < rank_sigma(i)` and only when `A_j^-` is nonempty.
The code exposes both `scores[B,N,N]` and `valid[B,N,N]`; an
unknown pair is never silently used as a negative observation.

All valid counterfactuals are batched together. The diagnostic calls
the decoder once for the reference state and once for each bounded
counterfactual chunk (default upper bound: 256 interventions), not
once per pair. This limits peak activation memory but does not make
the overall cost constant: there can be `O(B*N*A)` alternative
contexts, each evaluating an entire `N`-agent decoder.

`combine_order_probes` combines independently evaluated ordering
contexts into averaged directional scores and measurement counts. A
`conservative_precedence_targets` helper only creates optional
**pseudo-labels** when both directions have been separately observed
and the score margin exceeds a specified threshold. Different orders
change the conditional factorization, so even these labels can be
confounded; they should NOT yet be used to train on real SMAC data.
An all-zero score with no valid measurements is unknown, not evidence
of independence.

`tests/test_directional_oracle_supervision_v1.py` provides a
controlled decoder with one known original-agent directed link
0 -> 2. The tests measure it under one ordering, measure the reverse
orientation under another, reject unknown edges, then verify that the
low-rank scorer can learn the known directed label. This establishes
synthetic signal compatibility only, not predictive validity on SMAC.

### Topological-layer parallelism: not yet implemented

The original masked self-attention decoder does **not** automatically
permit parallel action sampling for all vertices in one graph layer.
Such parallelization is valid only after the conditional policy
factorization and attention masks guarantee that actions within a
layer do not condition on each other. Otherwise, producing them
simultaneously changes the modeled joint policy. Implement and
test the necessary structured masks and factorization **before**
claiming a speedup.

## Implementation and tests

- `mat/algorithms/mat/algorithm/dependency_ordering.py`:
  low-rank scorer, stable Borda order, confidence-gated DAG projection
  and masked BCE against external labels.
- `tests/test_dependency_ordering_v1.py`:
  reciprocity, diagonal neutral case, parameter count, stable tie-breaking,
  DAG acyclicity, gradient propagation, and controlled toy supervision.
- `research/dependency_decoder/action_influence_probe.py`:
  chunked all-legal-action KL contrasts, explicit valid-pair masks,
  multi-order coverage and optional high-confidence *proxy* labels.
- `tests/test_action_influence_probe.py` and
  `tests/test_directional_oracle_supervision_v1.py`:
  legal alternatives, unknown versus measured-zero edges, original agent
  ID restoration, synthetic directed-edge recovery and masked supervision.
- `mat/algorithms/mat/algorithm/ma_transformer.py`,
  `mat/algorithms/mat/algorithm/transformer_policy.py`,
  `mat/utils/shared_buffer.py`, `mat/algorithms/mat/mat_trainer.py`,
  `mat/runner/shared/smac_runner.py`:
  opt-in stored order path, **not** scorer training/inference.

### Validated milestones

- Three V0 modes in GPU 0 / `new_titans` / SMAC 3m function tests.
- Timestamped launcher output and artifact verified:
  [run 37892480786](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37892480786).
- Stored agent-order PPO transport verified with actual SMAC rollout:
  [run 37894004324](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37894004324);
  26/26 GPU unit tests and 2,000-step diagnostic successful.
- V1.1 standalone scorer regression: read the latest source branch CI.
  Do not equate toy-test success with SMAC performance.

### Next minimal engineering experiment

1. Keep the low-rank scorer isolated; compare it with the previous
   pair-MLP scoring implementation on synthetic known relationships
   (parameter count, ranking accuracy, inference overhead).
2. The all-legal-action KL diagnostic, explicit observability mask
   and a bidirectional-order synthetic oracle are implemented. Next
   evaluate held-out dependencies and calibrate the KL margin against
   false-positive associations in more diverse controlled toy games;
   do not infer environmental causality from a conditional-policy probe.
3. Define a training/evaluation protocol before integrating the
   learned graph. Add scorer freeze boundaries and an explicit stored
   permutation regression across scorer-weight changes.
4. Only after verification, test the gated mode on GPU 0 with one
   rollout environment. Keep GPU 1/2 untouched and retain
   `conda activate new_titans`. No long training by default.

## Prior-art boundaries

- [PMAT (AAMAS 2025)](https://yingwen.io/en/publications/pmat-aamas-2025/):
  Plackett-Luce agent-generation-order optimization is established
  prior work; generic learned ranking alone is insufficient novelty.
- [Sparse action dependency graphs (UAI 2026)](https://proceedings.mlr.press/v337/ding26a.html):
  sparse coordination structures are also established prior art.
  A new claim must be supported by a specific reliable dependency
  signal, validated structured factorization, efficiency evidence,
  and matched multi-seed experiments.
