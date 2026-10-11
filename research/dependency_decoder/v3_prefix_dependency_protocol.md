# V3 research prototype: prefix-conditioned directed dependency scheduling

**Status:** algorithm prototype and hosted CPU regression tests only. No SC2/GPU
training has been executed on this branch. Do NOT mix with the 90e20b8
15-run pilot source SHA or its saved run manifests.

## Motivation and prior art

MAT (Wen et al., NeurIPS 2022) performs sequential conditional action
generation in a fixed agent order. AOAD-MAT (Takayama & Fujita, AAAI 2025
student abstract / arXiv:2510.13343) already learns which agent acts next,
including a PPO-style objective; SeqComm (Ding et al., arXiv:2209.12713)
also negotiates priority then communicates actions. ADAPT (Xie et al.,
ECAI 2025, DOI:10.3233/FAIA251227) uses message dependency and
auction priorities. Hence learning a flexible order alone is NOT a
first-in-the-literature novelty claim.

## Two separate directed quantities

1. Antisymmetric *relative precedence* m_ij = -m_ji is inherited from V2.
   It contributes to an observation-conditioned base priority b_i(o).
2. Non-antisymmetric *predecessor demand*
   D_{i<-j} = sigmoid(q_i^T k_j / sqrt(rank)) means that the model currently
   prefers agent j to have been decoded before agent i; D_{i<-j} and
   D_{j<-i} can both be large. D is reward-trained and not an identified
   causal effect or independent edge in the environment.

At selected-prefix set S_k and remaining R_k, score candidate i in R_k as

    u_i(S_k) = b_i(o) + (lambda / temperature) * [
      mean_{j in S_k} D_{i<-j}
      - mean_{j in R_k\{i}} D_{i<-j}
    ].

Empty means are zero. Choose next agent from R_k using masked softmax;
all k logits are recomputed as the prefix grows.

    P(sigma|o) = product_k softmax(u(S_k))[sigma_k]

This is an **exact prefix-conditional permutation policy**, *not*
static-score Plackett-Luce. It uses already selected **agent IDs only**:
the entire permutation is sampled **before** producing actions. The order
does NOT yet depend on actual decoded action values. Upgrading it to an
interleaved action/order process would require a new joint rollout and
teacher-forcing implementation; no such claim is made here.

## MAT Decoder conditioning

For target slot t, shifted key slot (s+1) contains the action of the
agent decoded at slot s. The optional relation attention prior is

    bias[t,s+1] = log(2 * sigmoid(clamp(logit_D[target,source], -8, 8))).

BOS column = 0; the unchanged MAT triangular causal mask excludes future
actions. This is a soft attention preference, not an environment DAG
or a proof that the key slot isolates that source agent after self-attention.

## Training and ablations

The original MAT Encoder and Decoder modules initialize first; the optional
order actor initializes inside a forked CPU RNG context. All modes therefore
have identical MAT trunk weights at equal seeds. Saved rollout permutations
and their joint old order logprobs are replayed during PPO. Existing trainer
clips individual action factors and the permutation factor **separately**;
it is not the exact PPO objective for the joint (action, permutation) policy.

- identity: fixed order original MAT
- learned: static scalar priority with exact PL probabilities
- dependency: V2 antisymmetric preferences, static PL plus soft bias
- dependency-NoBias: V2 preferences, static PL, no bias (now consistent)
- dependency_prefix: V3 prefix-demand ordering with soft demand attention
- dependency_prefix-NoBias: **same V3 ordering** without soft attention

The V3-specific prefix scale is `--order_prefix_scale` (default 1.0).
Use `--agent_order_mode dependency_prefix --store_agent_orders` and
optionally `--disable_order_attention_bias`. Training launcher env vars:
`AGENT_ORDER_MODE=dependency_prefix`, `STORE_AGENT_ORDERS=1`,
`ORDER_PREFIX_SCALE=1.0`, `DISABLE_ORDER_ATTENTION_BIAS=0|1`.

## Tests and claims

CPU tests must verify:
- 3-agent exhaustive probability mass sums to one
- sampled order logprob matches replay on every checked permutation
- changing the selected prefix changes relative candidate probabilities
- non-antisymmetric target/source demand vs antisymmetric precedence
- agent relabeling equivariance (without ties)
- permutation validation and 1/27-agent numerical stability
- MAT rollout vs teacher-forced action/order likelihoods, both bias gates
- shifted source/target indexing and future-action causal mask
- nonzero gradients from prefix ordering and demand attention
- paired MAT trunk initialization and strict incompatible checkpoint handling
- manifest and CLI support, no new map outside existing nine-map whitelist

No performance, causal identification, scalability improvement, or
theoretical monotonic-improvement result follows from these tests alone.
Formal experimental protocols require new run manifests and frozen common
source SHA after CI passes. Existing V1, V2, and 90e20b8 smoke/checkpoint
results are not comparable as if they shared a single training commit.
