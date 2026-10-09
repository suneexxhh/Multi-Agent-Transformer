# MAT Decoder V1 — read-only frozen checkpoint diagnostic protocol

**Status: real SMAC and offline diagnostic verified, 2026-10-09.**
The restricted private [GPU 0 Run #37916803303](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37916803303)
completed successfully on `new_titans` with one SMAC `3m` environment,
2,000 training steps, PPO epoch 1 and no evaluation. The captured snapshot,
the exact matching weights and the offline CPU-only report were uploaded
as [private Artifact #11610860445](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37916803303/artifacts/11610860445).
These are **functionality/provenance diagnostics**, not trained-policy
coordination or causal-discovery results.

## Output provenance

The optional `DECODER_DIAG_CAPTURE=1` requires
`STORE_AGENT_ORDERS=1`. In `SMACRunner`, on episode index **1**
and step **0**, immediately after MAT samples actions but before the
environment step and PPO update, save an observation snapshot and
the **same model's** frozen state dict. The snapshot contains:

- `obs[B,N,D]`: original-agent local observations (not fake SMAC data)
- `available_actions[B,N,A]`: original-agent legal-action mask
- `agent_order[B,N]`: the permutation *actually used* in sampling

The checkpoint is a standard MAT `state_dict`, saved weights-only.
Both files share the name prefix
`MAT_<order>_seed<seed>_YYYYMMDD_HHMMSS_NNNNNNNNN`:

```text
..._snapshot.npz
..._transformer.pt
..._diagnostic.json
```

The experimental log retains its existing
`MAT_<order>_seed<seed>_<timestamp>.log` convention.

The optional capture adds **no parameter**, changes **no PPO loss**,
and does not change default training behavior; any eventual performance
impact of saving files must be considered when benchmarking training
throughput. The smoke test has only 2,000 training steps, so the
checkpoint is **not a converged trained policy**.

## Repeat without StarCraft II (CPU only)

From the checked-out MAT repository root:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m research.dependency_decoder.checkpoint_diagnostic \
  --snapshot /path/to/MAT_obs_norm_seed1_..._snapshot.npz \
  --checkpoint /path/to/MAT_obs_norm_seed1_..._transformer.pt \
  --n-block 1 --n-embd 64 --n-head 1 \
  --num-histories 8 --max-context-batch 2 \
  --max-counterfactual-batch 16 --seed 1 \
  --output /path/to/MAT_obs_norm_seed1_..._diagnostic.json
```

The `--n-block`, `--n-embd`, and `--n-head` values must match
the original training configuration; this is a strict checkpoint
load, so a mismatch fails rather than silently accepting missing
weights. For the default SMAC smoke script these are **1, 64, 1**.
SMAC MAT defaults to `encode_state=False`.

## Measured real SMAC sample (one observation context)

The actual files share this prefix:

```text
mat/scripts/logs/MAT_obs_norm_seed1_20261009_182212_553948306
```

Log: `<prefix>.log`, snapshot: `<prefix>_snapshot.npz`,
weights: `<prefix>_transformer.pt`, JSON: `<prefix>_diagnostic.json`.

The **real** CPU diagnostic reported:

| Item | Observed value |
| --- | ---: |
| SMAC observation contexts | 1 |
| Agents and sampled histories | 3 agents; 8 histories |
| Measurable directed pairs | 3 / 6 |
| Bidirectionally measurable pairs | 0 |
| Max sampled vs teacher-forced log-prob error | 0.0 |
| Mean conditional KL among measured pairs | 2.511641916669305e-08 |
| History-measurement coverage | 1.0 (among the 3 measurable pairs) |
| Minimum effective history sample size | 8.0 |
| CPU sampling / likelihood / intervention kernel times | 0.009379 / 0.003115 / 0.006850 s |

These timings exclude Python startup and model loading. They are single,
short-run routine timings, **not** robust wall-clock benchmarking.
The near-zero conditional KL is consistent with a short, nearly untrained
policy and finite-precision effects; it is **not** evidence that
true SMAC agent dependencies are absent. One ordering observes only
3/6 directed pairs, so it cannot supply bidirectionally verified
pseudo-labels. An ESS of 8 here reflects eight equal-weight measurable
samples; it does not imply low variance or statistical confidence.

A fresh checkpoint was saved at the beginning of episode index 1
(after one 100-step rollout/update), exactly before its paired action
sample. This policy is not converged; multi-seed SMAC win-rate claims
would be inappropriate. Normal SMAC training did finish with exit code
0 and the controlled artifact was uploaded, without modifying any
GPU 1/2 process.

## V1.2: cross-stage and matched-context experiment (opt-in)

The multi-context capture now collects **four original-agent
pre-action contexts** at steps `0,10,20,30` during each of two
training episodes, indexed `1` and `10`. In the bounded one-env
2,000-step smoke these occur after roughly 100 and 1,000 previous
training timesteps. Within each episode, MAT parameters are not
updated until the rollout ends, so all four contexts are paired to
one exact weights-only checkpoint. Filenames receive the suffix
`_ep0001` or `_ep0010` before `_snapshot.npz` and
`_transformer.pt`. Metadata fields `context_step` and
`context_episode` track the provenance of every row. Default
`DECODER_DIAG_CAPTURE=0` preserves ordinary MAT training.

Two different comparisons must **not** be conflated:

1. **Across sampled environments:** the per-context KL scores in
   each stage describe sensitivity at the observed states. Differences
   between early and late snapshots confound *changed policy* and
   *changed observations*. They can assess within-stage variability,
   not isolate learning-induced change.

2. **Matched-state frozen-policy comparison:** use
   `research/dependency_decoder/paired_checkpoint_diagnostic.py`.
   Sample joint actions once from the early decoder on a fixed
   snapshot, then evaluate `C_early(a)` and `C_late(a)` with the
   exact same `o`, legal masks, original-agent order and complete
   action histories. Each checkpoint uses its **own Encoder** to
   produce features. This isolates conditional model sensitivity
   changes from observed-state and chosen-action-history changes;
   however it remains a comparison of two learned conditional
   policies, not environment causality.

The module also teacher-forces both complete action distributions,
uses joint log-likelihood ratios to compute SNIS weights, and
reports importance-weight ESS. If ESS is below the predefined
threshold, it withholds the SNIS-weighted target sensitivity
rather than claiming a meaningful estimate. A large parameter
change does not guarantee a meaningful change in decoded
dependency; both must be measured.

Example CPU command after downloading the matching artifact:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m research.dependency_decoder.paired_checkpoint_diagnostic \
  --checkpoint-early /path/to/MAT_..._ep0001_transformer.pt \
  --checkpoint-late /path/to/MAT_..._ep0010_transformer.pt \
  --snapshot /path/to/MAT_..._ep0001_snapshot.npz \
  --n-block 1 --n-embd 64 --n-head 1 \
  --histories 8 --max-context-batch 2 \
  --max-counterfactual-batch 16 \
  --min-importance-ess 3 \
  --output /path/to/MAT_..._matched_comparison.json
```

Eight histories and one training seed are only a functional
diagnostic; never interpret microscopic near-zero KL changes as
evidence of an improvement in MARL coordination or graph-learning
accuracy. Until real dependency labels are defensible, the low-rank
precedence scorer remains **isolated from PPO**.

## Math and interpretation

1. Under a fixed observation, legal mask and stored permutation,
   sample `H` complete autoregressive joint actions `a ~ pi_theta`
   from frozen MAT using its *existing* sampling function.
2. Record every agent's conditional log probability and sum them to
   get `log pi_theta(a|o,sigma)`, then verify against the original
   MAT teacher-forced likelihood to numerical tolerance.
3. Use the **same frozen** model to compute uniform-legal-alternative
   conditional-action KL for each observed predecessor/successor pair.
4. Report measured pair coverage, conditional mean KL and ESS under
   the sampled history distribution, plus CPU computation time.

In any one saved permutation, only predecessors can affect later
conditionals. Consequently **bidirectionally observable pairs = 0**
for that snapshot, by design. The artifact cannot establish the
causal graph or generate opposite-direction pseudo-labels. A single
short smoke run cannot evaluate task win rate, convergence, or
multi-seed improvements. Measure multiple independently sampled
SMAC episodes and policy checkpoints before any quantitative claims
about real-world dependency stability.

## Safety boundary

The existing private controller checks that GPU 0 is free, runs
one SC2 environment, uses `new_titans`, and does not signal, kill
or inspect GPU 1/2 processes. The diagnostic itself forces CPU only.
No GitHub Actions workflow is created in the public source repo to
start server training automatically.
