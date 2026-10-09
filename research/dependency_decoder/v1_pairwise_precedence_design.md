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

### Mathematical route to meaningful supervised dependencies (future)

A plausible **policy-dependence proxy** is to hold observations and
other preceding actions fixed, perturb a preceding action `a_j`
within its legal set, and measure the effect on a later conditional
action distribution:

```text
C_(j->i) =
  E_[a'_j in legal_j] KL(
      pi_theta(a_i | o, a_<i, sigma)
      ||
      pi_theta(a_i | o, a_<i with a_j <- a'_j, sigma)
  ),  j precedes i under sigma.
```

This measures **within-policy conditional sensitivity**, not a causal
effect on environment dynamics or optimal joint return. It only
observes directed pairs present as preceding/succeeding in the
chosen order, so it cannot identify both directions without
additional ordering contexts. It may reflect attention/model
artifacts. Validate candidate labels first on small controlled toy
games whose directed dependencies are specified externally; then
evaluate whether learned precedence predicts held-out sensitivity
or improves return. Do not use arbitrary observation norm as a
label for causal dependence.

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
2. Implement a *diagnostic-only* legal-action intervention signal;
   quantify whether it agrees with synthetic true dependencies and
   which directed pairs cannot be observed under a single order.
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
