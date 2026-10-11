# MAT Decoder V2: directed, reward-trained precedence ordering

**Research status:** A CPU-testable, opt-in extension of the original MAT
autoregressive action decoder. No SMAC win-rate or causal benefit has been
demonstrated by this code change. This is NOT the MAT-MSA algorithm.

## Motivation and comparison against related work

Merely predicting which agent acts next is not novel by itself; AOAD-MAT
already studies learned adaptive decision order. The proposed experimental
distinction of this prototype is an explicitly **anti-symmetric, low-rank
directed precedence model**, with a conservative acyclic projection and
the option to compare a separately measured within-policy intervention
signal. This is a hypothesis to test, not a demonstrated contribution.

Retained ablations: `identity` (original MAT), `obs_norm` (V0 heuristic),
`learned` (V1 scalar priority PL actor), and `dependency` (V2 directed
pairwise priority PL actor). All continue to run within the same MAT
encoder/critic/decoder/PPO code, with per-rollout ordering replay.

## Mechanism

For agent observation `o_i`, let `h_i=MLP(o_i)` and
`c=mean_i h_i`. A shared single-agent priority head outputs
`b_i=f(h_i,c)`. A low-rank asymmetric relation module gives

```
q_i = W_q h_i, k_i = W_k h_i
m_ij = (q_i^T k_j - q_j^T k_i) / sqrt(rank)
p_ij = sigmoid(m_ij), p_ij + p_ji = 1
g_i = mean_{j != i} tanh(m_ij / 2)
z_i = (b_i + softplus(scale) * g_i) / temperature
```

`z_i` parameterizes an exact Plackett-Luce distribution `rho_phi(sigma|o)`,
sampled without replacement. Test-time inference sorts `z` (ties stable).

Joint policy factorization:

```
rho_phi(sigma | o) * product_k pi_theta(
    action_{sigma[k]} | o, earlier actions, sigma
)
```

Rollout stores the actual permutation and its old `log rho_phi`. PPO
evaluation uses the **stored** permutation, NOT one sampled from updated
parameters. The order policy receives its own clipped PPO surrogate using
mean active-agent advantage. This is **factor-wise clipping**, NOT exact
clipping of the joint action-plus-permutation likelihood ratio.

Computational overhead: pairwise matmul `O(B*N^2*r)`, activations
`O(B*N^2)` for `N` agents, low rank `r`. For SMAC maps with many agents,
it is cheaper than constructing `[B,N,N,3*D]` per-pair MLP inputs.

## Decoder attention conditioning (new in V2)

The directed matrix also guides the actual MAT Decoder cross-attention.
For decoded target slot k and action-token key slot l>=1, the token at l
contains the action of the agent at decoded slot l-1 (MAT's one-step
teacher-forcing shift). Add a bounded, differentiable log-prior:

```
bias[k,0] = 0                     # BOS
bias[k,l] = log(2*sigmoid(m[order[l-1], order[k]])), l>=1
```

to the Decoder's cross-attention logits **before** the original causal
lower-triangular mask. This is a *soft relationship bias*, not an
independence test or sparse DAG mask. The actual attention still uses
the original MAT causal mask: no future actions are visible.

The relation bias is computed once per observation and sampled order.
During PPO, it is recomputed differentiably using the **saved rollout
order**, so both action likelihood and ordering likelihood are evaluated
under the correct current policy. The ordinary `identity`, `obs_norm`,
and `learned` modes pass `attention_bias=None` and preserve the old
Decoder computations. CPU tests check teacher-forcing parity, backprop
through graph scores and no future-action leakage.

## Candidate DAG and optional offline alignment

`candidate_dag(obs, threshold=0.7, order=stored_order)` returns a
binary graph `A[i,j] = 1[p_ij>threshold AND rank(i)<rank(j)]`.
This is an **acyclic projection of directed precedence preferences**.
Edges are diagnostics, NOT currently used as HARD sparse attention masks and
NOT proof of environmental or policy causal dependence.

`counterfactual_alignment_loss(obs, influence, valid)` is a CPU/offline
optional hook for independently validated directed conditional-policy-KL
measurements. It refuses to train when either direction is unobserved
and never interprets missing measurements as zero dependencies.
The current SMAC PPO path does not automatically supply such labels;
do not claim that optional supervision has been run.

## Controls and CPU-only smoke test

```bash
python -m unittest tests.test_directed_dependency_order -v
python -m unittest discover -s tests -v

# Dry-run only: emits command; DOES NOT launch GPU / StarCraft II
DRY_RUN=1 STORE_AGENT_ORDERS=1 AGENT_ORDER_MODE=dependency \
  NUM_ENV_STEPS=1000000 PPO_EPOCH=10 CLIP_PARAM=0.05 \
  N_ROLLOUT_THREADS=1 N_TRAINING_THREADS=2 USE_EVAL=0 \
  bash mat/scripts/train_smac_research.sh 3s5z 0 1
```

The launcher enforces the user's nine-map whitelist in learned/dependency
mode and rejects any new 3m training. `--store_agent_orders` is mandatory.
The hard DAG is a *diagnostic* output. The separate **soft** directed
attention bias IS enabled for dependency mode. Neither hard DAG masking nor
parallel sampling is enabled without additional tests and replay logic.

## Strict three-seed exploration protocol

Use seeds **1, 2, 3**, first on a hard or super-hard approved map with
sufficient learning budget. Train `identity`, `obs_norm`, `learned`, and
`dependency` in a matched code revision, map, rollout-env count,
train steps, PPO epochs, clipping, action legal masks, checkpoint schedule,
32 independent native SMAC evaluation games, and evaluation seeds.
The initial 3s5z 100k six-way pilot gave zero wins in all modes tested;
do NOT use it as evidence of equality or convergence.

Before training, restore the independent super-hard controller failure:
`scripts/gpu0_bounded_batch.py` does not currently forward the
map-specific step/PPO/clip protocol; super-hard workflows pin an old
controller SHA. Preserve older successful artifacts; do not repeat them.
No GPU or GitHub Actions self-hosted execution is triggered by this PR.

GPU0 only; exclusive `/tmp/hk-mat-gpu0.lock`; concurrency maximum six,
subject to measured VRAM, RAM and CPU. Never touch GPU1/GPU2 or other
users' workloads. Keep frozen experimental manifests, all checkpoints
and metrics outside Git. Three seeds are for engineering selection only;
they cannot establish statistical significance.

## Research limitations and next steps

1. **Validate graph utility** with observed conditional-policy KL and
   legal-action masks. Record both forward and reverse observability.
2. Distinguish directed *precedence preferences* from genuine
   action-dependence. Negative controls and heldout policies are required.
3. Introduce graph-conditioned decoder attention only after graph
   replay is stored and sample/evaluate likelihoods are proven equal.
4. Explore topological-layer partial parallelism only if conditional
   independence is shown; a DAG projection alone is insufficient.
5. Measure added parameters, wall-clock overhead, paired win-rate
   and agent-order entropy, report all negative results.
