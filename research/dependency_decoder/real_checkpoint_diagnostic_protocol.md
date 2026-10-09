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

## Measured cross-stage SMAC results — Run #37919072289

The [restricted private GPU 0 run](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37919072289)
finished successfully in `new_titans`: one `3m` rollout environment,
2,000 steps, 1 PPO epoch, no evaluation. Its
[artifact #11610694060](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37919072289/artifacts/11610694060)
contains 2 frozen state dicts, 2 NPZ snapshots (4 distinct steps
per stage), both stage JSON reports, the fixed-state paired-model
comparison and full execution logs.

All files have prefix
`MAT_obs_norm_seed1_20261009_184144_790312829`.
MAT + CUDA unit tests: **71/71 passed** in the GPU job; CPU source
CI on the same pinned revision: **67 passed, 4 CUDA-only skipped**.

| Statistic | Episode index 1 | Episode index 10 |
| --- | ---: | ---: |
| Real observation contexts | 4 | 4 |
| Original agent count | 3 | 3 |
| H sampled histories per context | 8 | 8 |
| Measured directed pairs / 24 possible | 9/24 | 7/24 |
| Context-level observable pair counts | [3,3,0,3] | [0,3,1,3] |
| Sample vs teacher-forced logp maximum error | 0.0 | 0.0 |
| Mean measurable conditional KL | 3.131848913540125e-08 | 2.5313246609925955e-08 |
| Minimal measured history ESS | 8.0 | 8.0 |

One context in each stage has **zero identifiable directed pairs**
given its legal masks; its `mean_kl_by_context = null`. This is
not evidence of missing coordination. A single ordered
factorization can probe only predecessors, and constraints on
legal predecessor/successor actions can reduce the coverage further.

### Exact same observations, actions and stored order

The paired-model report fixes the **episode-1 snapshot**, then
samples 8 joint actions per context from the early policy and
evaluates both frozen models on the **same** histories and action
masks. Each policy uses its own Encoder features.

| Fixed-state statistic | Observed result |
| --- | ---: |
| Measured original-ID directed pairs | 9 |
| Mean early KL across those pairs | 3.0458149780088206e-08 |
| Mean late KL across those pairs | 2.2167039759324325e-08 |
| Mean absolute per-pair KL change | 1.8340214680279132e-08 |
| Maximum absolute model parameter difference | 0.004470713436603546 |
| Maximum joint action logp difference | 0.23173093795776367 |
| Sample/teacher-forced joint logp discrepancy | 0.0 |
| Importance-sampling ESS per context | [7.947,7.919,7.949,7.897] |
| Contexts passing minimum ESS >= 3 | 4/4 |
| Single paired CPU diagnostic kernel time | 0.113427 s |

**Do not interpret these tiny KL differences as a learned/stable
dependency graph.** Values around `1e-08` in the present
`float32` KL reduction are potentially dominated by numerical
cancellation: the probe computes a difference of action
log-softmax terms and clamps negative roundoff to zero. Compare
against a `float64` reference before using magnitudes or
thresholding purported edges. Two lightly trained policy
snapshots, eight action histories and one SMAC seed are insufficient
for scientific claims about convergence, causal links or
out-of-sample coordination improvements.

The results **do** verify that real MAT model parameters and
joint-action probabilities changed while frozen-context analysis
remained consistent. From here, prioritise numerical robustness,
more realistic checkpoints and multiple trajectories before
even considering learning the low-rank precedence scorer in PPO.

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
