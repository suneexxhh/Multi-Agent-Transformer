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
effect on environment dynamics or optimal joint return. The implemented
probe averages *all different legal action IDs* (uniformly, not under
the learned policy), uses chunked vectorized decoder calls and returns
both the scores and their observability mask. Details and limitations
are specified in the next section.

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
unknown pair is never silently used as a negative observation. A
source must have a different legal action AND its target must have at
least two legal actions: a single-legal-action target has a point-mass
policy whose legal distribution cannot respond, so it is unidentifiable.
Even a measured zero does not rule out an effect via an illegal action.

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

### Held-out synthetic dependency evaluation (offline, CPU)

`research/dependency_decoder/heldout_oracle_benchmark.py` supplies an
explicitly specified **test-only** conditional-action decoder with
edge matrix `G[b,source,target]`. Only a direct edge and the
counterfactually changed action class can alter a target's logits.
The oracle itself gets `G` for testing; **the learned low-rank scorer
receives ONLY a scalar trait per agent, not `G`**.

The synthetic graph generator uses a known scalar threshold rule:

```text
G_ij = 1[trait_i - trait_j > delta], delta = 0.55.
```

These are DAGs from **one known shared priority rule**, not arbitrary
causal graphs or unseen task objectives. The benchmark verifies
KL-probe precision/recall/false-positive rate against known edges
for previously unseen 3-, 4- and 5-agent graphs. Removing the only
action class that triggers a real edge can produce a *measured zero*
even though the edge exists: the support of legal actions matters.

The low-rank scorer trains from candidate preference labels on
72 sampled 4-agent graph contexts. An **independent calibration
split** of 48 distinct 4-agent graph contexts selects the smallest
confidence threshold yielding best recall under a 5% false-positive
budget. A separate 56-context 5-agent holdout is evaluated **once**,
with this frozen threshold. This protocol specifically prevents
tuning on the final test graphs. Test thresholds and confusion
statistics are printed in GitHub CPU CI; see
`tests/test_heldout_oracle_benchmark.py`.

A **rule reversal negative control** intentionally reverses the
structural dependence rule: a scorer respecting the original priority
rule is not expected to generalize. The result should be reported as a
limitation, not silently removed. Synthetic oracle accuracy is not
evidence that PPO learns improved MARL coordination.

### Reproducible held-out quantitative results (synthetic only)

On 2026-10-09, [CPU CI Run #37898081067](https://github.com/suneexxhh/Multi-Agent-Transformer/actions/runs/37898081067)
confirmed the deterministic seed-8302 protocol with 72 **training**
4-agent contexts, 48 independent **calibration** 4-agent contexts,
and 56 independent **test** 5-agent contexts. The scorer receives
scalar traits but not ground-truth graph edges. Its calibrated
threshold was `0.97643` (on calibration FPR `0.04941`), and the
one-time held-out evaluation reported:

| Scenario | Precision | Recall | FPR | TP | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same scalar-rule, unseen 5-agent graphs | 0.89815 | 1.00000 | 0.03981 | 291 | 33 |
| Opposite structural-rule negative control | not used | 0.00000 | 0.52492 | -- | -- |

**The opposite-rule test is a separate intentionally specified
ranker that follows the original direction**, not an unbiased
estimate of an independently retrained model's out-of-distribution
performance. It shows why priority-rule transfer must not be
assumed when dependencies change.

These results are for a deterministic known graph-generation rule,
with synthetic labeled calibration access that **SMAC does not
provide**. They cannot be reported as SMAC graph accuracy,
model causality, or evidence of an improvement in episodic return.
The CPU CI run had 43 total unit tests, 39 passed and 4
CUDA-dependent tests skipped.

### Multi-parent nonlinear dependence and action-history blind spots

A new offline benchmark (`research/dependency_decoder/nonlinear_history_oracle.py`,
`tests/test_nonlinear_history_oracle.py`) explicitly constructs a
three-agent, two-parent conditional action policy. With actions
`a_0,a_1 in {0,1,2}` and target agent 2:

```text
z_AND = 1[a_0=2] * 1[a_1=2]
z_XOR = |1[a_0=2] - 1[a_1=2]|
logit_target[action=1] = lambda * z
```

These are known conditional *policy* mechanisms, not proofs of
environmental cause-effect or multi-agent reward coordination.

For AND at baseline `(a_0,a_1)=(0,0)`, replacing either action by
another legal action in isolation never activates the conjunction,
so BOTH genuine parent dependencies yield KL=0 despite being
measurable. At baseline `(2,0)`, only agent 1's influence is detected;
at `(2,2)`, both are detectable. Aggregating explicitly specified
baseline contexts by *maximum observed* sensitivity (rather than
silently treating unmeasured pairs as zeros) recovers both known
parents in this controlled example. Using a maximum over carefully
selected histories would be a biased diagnostic on real data.

The XOR mechanism additionally shows that unsigned KL cannot tell
whether an action increases or decreases another agent's action
logit: direction `j -> i` refers to who is conditioned on whom, not
the sign of the conditional effect.

An observation-only `PairwisePrecedenceScorer(obs)` necessarily
returns the SAME priorities for two examples with identical
observations but different **current** joint action histories. In
these nonlinear examples, the true *conditional sensitivity* changes.
Thus no observation-only scorer can encode all such same-step
history-contingent dependencies, regardless of its capacity. A
pre-decision ordering can only estimate an expected/aggregated
priority over plausible yet-unknown same-step actions, perhaps
conditioning on a history summary from previous timesteps.

**Implication for simple, efficient MAT Decoder research:**
preserve the light low-rank `O(N^2 r)` precedence scorer as a
pre-decision *expected* ordering candidate; do not add expensive
counterfactual KL queries to every PPO rollout step or claim that
a scalar edge captures multi-parent synergies. Model truly
higher-order same-step effects only if controlled performance tests
justify the extra complexity. No graph-structured parallel decoder
has been activated.

The action-history oracle also verifies dynamically restricted legal
action sets: if the only activating action is illegal, a measured
KL of zero cannot exclude a structural parent, and a singleton
target legal-action set is marked **unknown**. Counterfactual chunks
with no measurable successors are now skipped before any Decoder call.
A held-out, fixed-seed 64-context AND test compares KL detection
to the known context-specific sensitivity truth.

### Offline expected dependency under an explicit action-history distribution

A new **standalone, non-training** module is implemented in
`research/dependency_decoder/expected_action_dependency.py`.
It addresses the nonlinear AND counterexample: the conditional KL
`C_(j->i)(h,o,sigma)` may be exactly zero for common histories even
when a different legal action history reveals a strong dependency.

Given a set of **explicitly sampled or enumerated** histories
`h=1..H`, user-supplied nonnegative relative weights `w_h`,
and a pairwise identifiability mask `m_hji`, compute:

```text
M_ji = (sum_h w_h m_hji) / (sum_h w_h)
Cbar_ji = (sum_h w_h m_hji C_hji) / (sum_h w_h m_hji)
ESS_ji = (sum_h w_h m_hji)^2 / (sum_h w_h^2 m_hji)
```

All three outputs are `[B,N,N]`: conditional measured-history mean,
history-coverage mass, and Kish effective sample size. When
`M_ji=0`, the displayed zero is a **placeholder**, not an estimate
of no dependency. The conditional mean is NOT the unconditional
expectation when histories can be unidentifiable; the missing part
of the history distribution is not imputed. ESS describes weight
concentration, not a guaranteed confidence interval.

The history weights are ONLY a user-specified distribution. These
routines do **not** create on-policy samples, estimate importance
ratios, or assume all histories are equally likely. If a new policy
changes its action distribution, yesterday's weights cannot be
reused as if the history distribution were stationary. If `w_h`
does not represent a known target distribution, `Cbar` is simply
a descriptive weighted summary, not a policy expectation.

The module's `probe_action_histories` evaluates the original
`legal_action_kl_probe` across `[H,B,N,1]` action contexts while
bounding each repeated encoder-feature batch by
`max_context_batch` and counterfactual batch by
`max_counterfactual_batch`. It only runs offline, not once per
PPO rollout step. `reliable_precedence_targets` exposes optional
**pseudo-labels** only when BOTH directed measurements satisfy
`M >= min_coverage`, `ESS >= min_ess`, and exceed the expected
KL margin. Comparing different ordering-conditional policies remains
a confound, so these labels are not automatically valid on SMAC.

The fixed AND synthetic test uses two explicit histories where
`a_1=2` occurs with probabilities 0.1, 0.5 and 0.9. Expected
`0 -> 2` conditional policy sensitivity increases with this
probability; the original observations are identical. Another
test checks that 98% mass on an inactive context makes the
weighted **expected** KL exactly one-fiftieth of the maximum
observed KL. This is why *max-over-selected histories* is not
an unbiased replacement for an expected decision dependency.

**Research decision:** keep the low-rank scorer unchanged until
we have a reproducible sampling distribution (or a defensible,
frozen behavior-policy snapshot) and held-out calibration across
different conditional action-history distributions. Future
`P_phi(i before j | obs)` should approximate a *pre-decision*
expected priority rather than pretend it knows yet-unexecuted
current-step actions. Nothing here implements learned graph PPO
or graph-parallel autoregressive factorization.

### Frozen MAT behavior-policy sampling and change of distribution (offline)

`research/dependency_decoder/frozen_policy_sampling.py` now reuses
**the original MAT ordered discrete autoregressive sampler** and
**the original parallel teacher-forced log-probability evaluator**.
A frozen eval-mode Decoder with fixed observations `o`, per-agent
Encoder representations `h`, legal masks `L`, and agent permutation
`sigma` generates H action vectors:

```text
a^(h) ~ q(a|o,sigma) = product_k
    q(a_sigma[k] | o,a_sigma[<k],sigma)
log q(a^(h)|o,sigma) = sum_k log q(a_sigma[k]^(h)|...)
```

All actions, conditional log-probabilities, and agent IDs are returned
in the **original** agent order. The complete joint `logq` is
recorded per sampled history. Real MAT Decoder sampling and
teacher-forcing likelihoods have been compared and must agree for
identical weights, order, legal masks and Encoder representations.
The offline functions use chunked repeated observations, not
allocations of `[H,B,N,D]` for all histories at once.

If a second **frozen** target policy `p` has a different distribution
but the SAME available action support and fixed `sigma`, score
these same sampled actions under target Decoder and its corresponding
target Encoder representations. For a saved history define:

```text
r_h = exp(log p(a^(h)|o,sigma) - log q(a^(h)|o,sigma))
w_h = r_h / sum_t r_t     (numerically stable softmax)
ESS = 1 / sum_h w_h^2
E_p[C_p] ~ sum_h w_h * C_p(a^(h))
```

This is self-normalized importance sampling (**SNIS**) at a fixed
observation and fixed ordering, NOT a probability for the sampled
dataset and NOT an exact or unbiased finite-H estimator. The helper
accepts an optional `min_ess` to **refuse** heavily degenerate
weights; in that case gather better-support samples rather than
creating overconfident precedence pseudo-labels.
The target history KL `C_p` must be computed with the TARGET
Decoder, not reused from behavior `C_q`, unless the conditional
action-sensitivity functions have been independently shown equal.
**Reweighting cannot repair a change in the dependency mechanism.**
The full proposal/target action support must coincide; dynamic
legal masks, changing order distributions and unrecorded order
probabilities invalidate naive action-only ratios. If the Encoder
changes, recompute target representations with target Encoder.

A real checkpoint must be held fixed for each sampling/probing
pass: `decoder.eval()` disables training-mode randomness but
does not itself prevent parameter updates by an external optimizer.
No checkpoint or SMAC rollout data are loaded by this module.

A fixed-seed toy with an AND conditional action mechanism and
different parent-action logits tests known policy shift without
changing the target interaction rule. With 2,400 histories sampled
from behavior q, the CI reports behavior activation probability,
SNIS-estimated target activation, analytical target probability,
and ESS. A second counterexample explicitly changes the
dependency mechanism and confirms `E_p[C_q]` need not equal
`E_p[C_p]`. A stronger shift shows weight degeneracy and why
low ESS should trigger abstention/new samples rather than confident
candidate precedence labels.

This adds **zero trainable parameters** and changes **no PPO loss**.
Running many counterfactual decodes is an offline diagnostic cost,
not a free improvement in action-generation speed.

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
- `research/dependency_decoder/heldout_oracle_benchmark.py` and
  `tests/test_heldout_oracle_benchmark.py`:
  reproducible held-out directed-graph sensitivity metrics, separate
  confidence-threshold calibration and explicitly failing OOD rule test.
- `research/dependency_decoder/nonlinear_history_oracle.py` and
  `tests/test_nonlinear_history_oracle.py`:
  AND/XOR multi-parent conditional policy, dynamic legal masks,
  history-conditioned identifiability, and fixed-seed action contexts.
- `research/dependency_decoder/expected_action_dependency.py` and
  `tests/test_expected_action_dependency.py`:
  offline bounded history batches, weighted conditional KL mean,
  coverage mass, ESS, conservative reliability gates, rare-event
  maximum-KL negative control and no PPO integration.
- `research/dependency_decoder/frozen_policy_sampling.py` and
  `tests/test_frozen_policy_sampling.py`:
  native frozen MAT autoregressive samples, exact teacher-forced joint
  probabilities, log-space SNIS, overlap constraints, shifted conditional
  mechanisms and ESS failure controls.
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
2. Basic held-out graph calibration, an opposite-rule
   failure control, AND/XOR nonlinear interactions, and distribution-
   weighted history aggregation are implemented offline. Next define
   a realistic sampled/frozen behavior-policy distribution, measure
   uncertainty and coverage under legal-mask shifts and evaluate
   actual distribution transfer. A static pre-decision scorer
   cannot observe still-unknown same-step actions.
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
