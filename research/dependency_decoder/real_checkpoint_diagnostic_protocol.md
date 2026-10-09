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

## Numerical follow-up: KL below single-precision subtraction scale

The Run #37919072289 values above were obtained from its
**pinned earlier source revision** and used `float32`
`log_softmax` subtraction. The research branch has subsequently
updated only the offline `action_influence_probe.py` KL reduction
to **float64** for reference and counterfactual log-probabilities
and probability-weighted summation; decoder forward calls remain
in the original model dtype, with no new parameters. Invalid
actions are explicitly excluded, avoiding `0*(-inf-(-inf))`.

The standalone analytic test checks binary
`KL((1/2,1/2) || softmax(0, epsilon))`
at `epsilon=1e-4`, whose exact value is
`log(cosh(epsilon/2)) ~= 1.25e-9`. This distinguishes genuine
small positive KL from roundoff-triggered zero/negative values.

**The previous GPU artifact has now been recomputed with the
double-precision reduction.** Its historical float32 values are kept
in the tables above for reproducibility, and should NOT be reported
as accurate dependency magnitudes. See the verified CPU-only replay
below. Neither historical nor corrected short-checkpoint KL values
support a supervised agent-order training label.

## Archived real SMAC replay with float64 KL (completed)

[Private CPU-only replay #37920607495](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37920607495)
**succeeded** on 2026-10-09 without starting StarCraft II or using
any GPU. Its [artifact #11611408127](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37920607495/artifacts/11611408127)
contains six new JSONs/logs plus a self-contained numerical comparison.
The **original frozen weights and snapshots were not altered**.

The CPU workflow checked out MAT source commit
`bc28fa9e7a0c98ba8537941825d48081c705c796` (fixed
decoder logits, stable `float64` KL reduction), downloaded source
run `37919072289`'s exact private Artifact, and replayed **both**
real checkpoint stages and their matched-state comparison.

The H=8 replay verified unchanged observable-pair masks, saved
observation context provenance, equal original MAT joint-action
mean log probabilities, and identical importance-sampling ESS.
Thus the change between historical H=8 and recomputed H=8 KL
reflects the numerical reduction, not a new training run or
deliberately changed behavior-policy histories.

| Real-data diagnostic | Historical float32 H=8 | New float64 H=8 | New float64 H=64 |
| --- | ---: | ---: | ---: |
| Episode 1 mean measurable conditional KL | 3.131848913540125e-08 | 1.8754301497198034e-13 | 1.934263838508643e-13 |
| Episode 10 mean measurable conditional KL | 2.5313246609925955e-08 | 1.0353048713468427e-12 | 9.97297622089277e-13 |
| Fixed-state mean absolute pair KL change | 1.8340214680279132e-08 | 7.232883454492967e-13 | 7.132018500083381e-13 |

For H=64, effective importance sample sizes on the four matched
observations were approximately `[63.55, 63.49, 63.68, 63.41]`.
The H=8 per-context observability counts remained `[3,3,0,3]`
for episode 1 and `[0,3,1,3]` for episode 10. Histories excluded
for action/target legality are never treated as independent edges.

**Interpretation.** In this unusually early MAT checkpoint, the
original float32 KL results were overwhelmingly dominated by
cancellation/rounding from subtracting conditional log-probabilities
near equality. The output logits themselves are still produced by
the original float32 decoder; casting logits before double-precision
`log_softmax` cannot recover information already lost during model
forward. Yet the robust reduction removes a large spurious
`10^-8` background and reveals extremely weak measurable action
sensitivity `~10^-13 to 10^-12`. That is **not** proof of no
structural agent dependency, no cooperation in the environment,
or a converged learned policy. In particular, we have no
bidirectionally identified labels in these one-order snapshots.

**Decision:** do not train the low-rank precedence scorer on these
near-zero pseudo-labels; freeze V0/V1.1 rollout and PPO behavior.
Subsequent substantive experiments require sufficiently trained
checkpoints, multiple independently collected trajectories, a
predeclared numerical noise floor and confidence criteria, and
matched-seed win-rate/compute comparisons if graph ordering is
eventually enabled.

## Archived SMAC total-variation cross-check — 2026-10-09

[Private read-only CPU replay #37922349103](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37922349103)
passed with no SC2 startup or GPU use. Download its
[diagnostic artifact #11612103186](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37922349103/artifacts/11612103186)
for the stage-level JSON and exact-provenance comparison.

The addition to `action_influence_probe.py` derives BOTH KL and
total variation (TV) from the **same decoder logits and legal
action masks**, without another decoder pass:

```text
TV(p,q) = 1/2 * sum_{a in legal} |p(a)-q(a)|
Pinsker: 2 * TV(p,q)^2 <= KL(p||q)
```

The reductions take place after the ordinary float32 Decoder
forward, in float64; the scores are aggregated over the same
legal action alternatives and sampled histories, and are reported
in the original agent-ID axes.

| Archived checkpoint | mean TV, H=8 | mean TV, H=64 | mean KL, H=64 |
| --- | ---: | ---: | ---: |
| Episode index 1 (very early) | 2.4488653593834897e-07 | 2.464694546233659e-07 | 1.934263838508643e-13 |
| Episode index 10 (early) | 5.722741889258032e-07 | 5.590754881268367e-07 | 9.97297622089277e-13 |

The maximum observed Pinsker excess in the aggregated per-pair
reports was `0.0`. The earlier identical-H=8 KL values and joint
sample log-probability checks remained unchanged; extra TV was
computed without extra Decoder calls.

**Interpretation boundary.** TV of order `10^-7` on these two
lightly trained checkpoints is a very small conditional change
in the *policy's* legal action probabilities. The result says
nothing decisive about actual cooperation, optimal behavior,
environment causality, or the capabilities of a fully trained
MAT policy. Satisfying Pinsker checks internal mathematical
consistency but does not establish statistical significance;
the underlying policy logits were still computed in float32.
The sampling runs have only four observation contexts per
checkpoint and one training seed. Do **not** construct synthetic
ground-truth dependency labels from these real-data values.

### Evidence gate before enabling learned Decoder precedence

Keep `learned_precedence` disabled until the following are met:

1. **Sufficiently trained, identified checkpoints.** Record map,
   seed, full architecture/agent-order configuration, training
   timestep, reward/win-rate evaluation and checkpoint hash.
   The accessible public research repository and private smoke
   artifacts currently contain no verified long-trained checkpoint.
   This does not rule out separately stored user checkpoints.
2. **Representative states.** Capture multiple real independent
   episodes and sufficiently diverse legal-action states, not
   just four observations from one short rollout. Report measured,
   unmeasured and singleton-legal-action targets separately.
3. **Numerical reliability.** Evaluate KL with stable float64
   reductions, verify the KL–TV inequality and compare against a
   frozen identical-policy null. Report sensitivity to changes
   in the action-history sample budget.
4. **Bidirectional identification and generalization.** A fixed
   decoder order cannot directly measure the reverse directions;
   alternate order contexts change the conditional joint policy.
   Do not equate reverse-order KL with an environment-causal edge.
   Any precedence supervision must survive held-out
   context/seed checks and a justified, predeclared signal threshold.
5. **PPO validity and speed.** When considering deployment, keep
   rollout orders fixed during PPO replay, explicitly handle
   ordering-policy probabilities if `rho_phi(sigma|o)` is trained,
   and evaluate CPU/GPU time, convergence and win rate against
   matched MAT baselines.

The present gate is **not satisfied**, so the low-rank scorer
remains standalone. Introducing per-step counterfactual probes into
MAT rollout would violate the desired minimal compute overhead.

## Real intermediate-policy SMAC 3m pilot — 100,000 steps (verified)

On 2026-10-09 the bounded private
[GPU 0 experiment #37924371727](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37924371727)
completed successfully, including training, six stage-aligned frozen
checkpoint/snapshot captures, post-run CPU KL–TV diagnostics, and
[artifact #11614615026](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37924371727/artifacts/11614615026).
It used SMAC `3m`, seed 1, `obs_norm` ordering, one rollout
environment, 100,000 environment steps, PPO epoch 5, and evaluation
disabled. None of the runs altered PPO or the MAT Decoder topology.

There are exactly FOUR adjacent pre-action observations per stage;
`N=3` agents permits at most `4*3*(3-1)=24` directed
agent pairs, but only predecessor-to-successor pairs and legal
action contrasts are measurable.

| Approx. environment steps | Mean conditional KL | Mean TV | Measured pairs / 24 |
| ---: | ---: | ---: | ---: |
| 0 | 1.6495076246743345e-13 | 2.2725765802533715e-07 | 9 |
| 100 | 1.5870703189146962e-12 | 6.674547421425814e-07 | 9 |
| 1,000 | 5.11630446453637e-07 | 3.5769265377894044e-04 | 4 |
| 10,000 | 2.4218703620135784e-03 | 8.279197849333286e-03 | 5 |
| 50,000 | 1.036502726492472e-05 | 3.9655115688219666e-04 | 8 |
| 90,000 | 6.027619238011539e-04 | 5.633640103042126e-04 | 7 |

Each stage used 32 independently sampled joint actions *at its
recorded observation*, but the four successive observations are not
four independent trajectories. The positive dependence-sensitivity
change at 10,000 steps establishes that learned conditional MAT action
distributions can respond detectably to predecessor interventions.
It is **not monotonic**, and mean scores across stage snapshots
are confounded by changing observations and measured-pair coverage.

### Matched-state early-vs-late comparison: severe support mismatch

The diagnostic replayed the early and 90k-step frozen checkpoints
on the exact SAME four early observations, legal action masks and
stored agent order, using 32 joint-action histories drawn from the
**initial policy**:

- Mean early-model conditional KL: `1.6727485824437338e-13`.
- Mean later-model conditional KL on **early-drawn actions**:
  `0.006481548305600882`.
- Max absolute joint action log-probability shift: `31.7130699`.
- Importance sampling ESS by context: `[1.009, 1.038, 13.015, 1.046]`
  out of 32. Only one context passed ESS >= 5, and it had no
  measurable pairs; no SNIS-weighted late-model KL is reported.

**Do not misinterpret the matched late KL as E_{a~late}[C_late].**
It is an average over actions drawn from the initial policy.
The enormous importance-ratio shift makes correcting it with
the 32 sampled early actions unreliable. It can show policy
function changes at fixed input and actions, but cannot prove
what dependency magnitudes are typical under the evolved policy.

### Next low-complexity diagnostic: direct frozen target Monte Carlo

The independent `paired_checkpoint_diagnostic.py` now optionally
accepts `--direct-target-histories H`. With original SMAC
observation, legal masks and decoding order held fixed, it samples
joint actions **directly** from the late frozen MAT decoder and
estimates `E_{a~late}[C_late(a)]` (KL and TV), plus
within-context action-history Monte Carlo standard errors for TV.
This does not need importance weights or additional trainable
parameters and runs **only offline on CPU**. It does NOT provide
confidence intervals across independent game episodes, which
would require sampling new environmental trajectories.

Any new results must be reported distinctly from
`E_{a~early}[C_late(a)]` and cannot be converted into
causal ground-truth graph labels. The `stage_quality_review.py`
report explicitly refuses learned-order supervision in this
single-seed setting.

## Directly sampling the late frozen policy after SNIS overlap failure (verified)

A second read-only CPU experiment,
[Run #37926410897](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37926410897),
successfully reused the 100k-step SMAC `3m` run's frozen checkpoints
and original-agent observation snapshots. Its
[artifact #11613623845](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37926410897/artifacts/11613623845)
contains the precise JSON reports and provenance.

**Why change the estimator?** The initial-to-90k importance sampling
weights suffered low overlap: effective sample sizes were near 1
out of 32 for three of the four fixed early observations, and
the one context with adequate overlap had zero measurable pairs.
No valid SNIS-weighted late-policy dependency estimate existed.

The new optional `paired_checkpoint_diagnostic.py
--direct-target-histories H` samples actions from the late
**frozen target decoder itself** at the *same observations, original
agent order and legal action masks*, then applies the same
double-precision KL and TV reductions. There is no importance
weighting or surrogate model in the direct estimate:

```text
a^h ~ pi_late(.|o,L,sigma)
C_late(o) = (1/H) * sum_h KL(pi_late(.|o,a^h_<i) ||
                                 pi_late(.|o,a^h_<i with a_j changed))
TV_late(o) = corresponding legal-alternative total-variation average
```

These are **on-policy history draws at frozen states**, not online
environment rollouts. The separate earlier-draw diagnostic samples
`a~pi_early` and estimates a different quantity. The new routine
reports sample-vs-teacher-forced probability agreement, total KL/TV,
individual-context means and Monte Carlo standard error (across
action histories only) without adding trainable parameters.

| Fixed SMAC observation bank | Late-policy histories | Late-policy mean KL | Late-policy mean TV | Measured pairs |
| --- | ---: | ---: | ---: | ---: |
| Initial snapshot | 64 | 0.003362865187227726 | 0.004623468033969402 | 9 |
| Initial snapshot | 256 | 0.0032817889004945755 | 0.0045886873267591 | 9 |
| 90k snapshot | 64 | 0.0005883383564651012 | 0.0005560711724683642 | 7 |
| 90k snapshot | 256 | 0.0005996432155370712 | 0.0005654250853694975 | 7 |

In both snapshot banks the 64- and 256-history estimates agree
reasonably well. The initial and later snapshot results **cannot
be treated as a cross-training-stage improvement trend**, because
different states, legality and measured pair sets are involved.
The frozen late decoder is exactly the same checkpoint in both
banks. Importantly, the late decoder shows conditional action
sensitivity well above numerical cancellation background; this
does not identify environment-causal influence or demonstrate
better cooperative episodic returns.

**Next stronger offline check:** hold an observation bank fixed,
then run *all six* archived frozen checkpoints with their OWN
action-history samples at identical legal masks and agent order.
This controls state input variation while allowing each frozen
policy to express its own action distribution. The
[dedicated CPU-only workflow](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/workflows/mat-cpu-100k-fixed-states.yml)
performs this six-stage, two-bank diagnostic. None of these
results may train pairwise precedence labels without separate
held-out direction/coverage evidence and independent trajectories.

## Six frozen checkpoints on the SAME SMAC inputs (verified)

A more controlled archived CPU-only comparison,
[Run #37926773351](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37926773351),
evaluated all six archived frozen MAT checkpoints on two separately
held-fixed SMAC observation/mask/order banks, each consisting of
four original-agent contexts. Each checkpoint sampled 128
**own-policy** joint-action histories. The run succeeded; see
[artifact #11614098513](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37926773351/artifacts/11614098513)
for its 12 JSON diagnostics and source verification.

The per-bank set of measured predecessor-successor pairs is
constant across checkpoints: 9/24 for the initial-state bank,
7/24 for the 90k-state bank. This removes one confound present
in comparing each training stage on its own evolving observations.

| Model environment steps | Mean TV at fixed initial states | Mean TV at fixed 90k states |
| ---: | ---: | ---: |
| 0 | 2.2642355190782837e-07 | 2.1464606447807455e-07 |
| 100 | 7.470497962458467e-07 | 6.261464591261756e-07 |
| 1,000 | 2.9725435888394713e-04 | 2.951239584945142e-04 |
| 10,000 | 4.059057682752609e-03 | 3.870528191328049e-02 |
| 50,000 | 1.293928944505751e-04 | 3.8995014620013535e-04 |
| 90,000 | 4.578216467052698e-03 | 5.673421546816826e-04 |

This establishes a **nonmonotonic evolution of the frozen policies'
conditional action sensitivity**, at both fixed context banks and
with consistent legal-pair observability. It does **not** establish
forgetting of useful team dependencies: stronger conditional-action
sensitivity is not inherently better teamwork, nor is a high-KL
policy guaranteed higher win rate. The six policies use the same
architecture but different training steps and sampled action
distributions. Each bank still represents only FOUR temporally
adjacent SMAC observations, not independent episodes.

### Distinguishing low KL/TV from action-policy concentration

A policy with nearly deterministic preferred actions can exhibit
small measured distribution changes even if its action history
representation is complicated. The new optional zero-overhead
entropy summary in `checkpoint_diagnostic.py` uses the SAME
sampled per-agent log probabilities already computed by native
MAT:

```text
a^(h) ~ pi_theta(. | o,L,sigma)
H_hat_joint(o) = - (1/H) * sum_h log pi_theta(a^(h) | o,L,sigma)
H_max(o,L) = sum_{agent j} log |A_j^legal(o)|
H_hat_fraction = H_hat_joint / H_max   (when H_max > 0)
```

`H_hat_joint` is an unbiased Monte Carlo estimator of joint
action entropy *at a fixed state*, and the finite sample
estimate can be noisy. The per-observation Monte Carlo SE
measures action-draw variability, not confidence across
independent game trajectories. The probability capacity is a
theoretical upper bound for the **true** entropy of the legal
joint action distribution, not an ironclad upper bound on a
finite-draw estimate. No new parameter or Decoder call is
required. Inference and PPO stay identical.

The separate CPU-only
[entropy-augmented archived sweep](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37927267006)
**completed successfully**. Its
[artifact #11613869975](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37927267006/artifacts/11613869975)
contains twelve aligned, 128-history checkpoint diagnostics and a
fixed-state comparison table. All joint-policy entropy values were
computed from the *same* sampled action log-probabilities as KL/TV;
there are no extra Decoder passes.

| Frozen-model training steps | Joint entropy at fixed initial states (nats) | Joint entropy at fixed 90k states (nats) |
| ---: | ---: | ---: |
| 0 | 4.141697883605957 | 4.0445170402526855 |
| 100 | 4.128137588500977 | 4.042891502380371 |
| 1,000 | 4.107318878173828 | 3.9901673793792725 |
| 10,000 | 1.1811500787734985 | 1.1531189680099487 |
| 50,000 | 0.43877679109573364 | 0.7203799486160278 |
| 90,000 | 0.3730016350746155 | 0.06606321036815643 |

The low-TV 90k model has **much lower conditional joint-action
entropy** than the initial model at these fixed observations,
consistent with a narrow distribution over legal actions. This
could explain part of the nonmonotonic KL/TV pattern, but neither
proves causal forgetting nor establishes cooperative return.
The native frozen evaluation described below is now completed.

## SMAC 3m deterministic native evaluation on frozen checkpoints (completed)

The read-only [GPU 0 Run #37928312446](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37928312446)
**completed successfully** with [eval logs and JSON artifact #11614684653](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37928312446/artifacts/11614684653).
It restored four archived `obs_norm` MAT checkpoints, verified the
same 3m map and evaluation seed 1, and called the unmodified
`SMACRunner.eval()` / deterministic `TransformerPolicy.act()`.
Every model was evaluated for 32 SMAC episodes. It performed
**zero optimizer updates** and never called `runner.run()` or
PPO training.

| Checkpoint environment steps | SMAC wins / 32 | Win rate |
| ---: | ---: | ---: |
| 0 | 0 / 32 | 0% |
| 10,000 | 25 / 32 | 78.125% |
| 50,000 | 20 / 32 | 62.5% |
| 90,000 | 24 / 32 | 75% |

The native win rate establishes that the low-entropy 90k frozen
policy can nevertheless win 24 out of 32 games under this specific
evaluation. Therefore **small conditional action KL/TV and low
joint action entropy do not, by themselves, establish policy
degeneration**. Conversely, high KL/TV is neither necessary nor
sufficient for higher win rate. The training trajectory was
nonmonotonic in these sparse checkpoints.

This is only ONE training seed and 32 evaluation episodes per
model, with potentially correlated deterministic evaluation
trajectories. No uncertainty across independent training seeds
has been estimated, and winning 25 vs 24 games does not establish
a statistically significant ranking. Evaluation without a
matching original MAT `identity` agent-order baseline cannot
justify an `obs_norm` or learned-dependence sorting improvement.
The [matched identity baseline workflow](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/runs/37928862199)
has been requested using the same seed, code revision, optimizer
budget and GPU0 resource constraints. Its actual conclusion
must be checked separately: a submitted or running experiment
is not itself a successful result.

**Algorithmic implication:** preserve V1 pairwise scorer in
offline-only diagnostic mode, and never derive preference
supervision simply by thresholding KL/TV, entropy or win rate.
The correct benefit/risk tradeoff must be tested in held-out
rollouts and matched identity/learned-order ablations. Still no
Decoder parameters or PPO losses have been changed by these
diagnostics.

## Experiment protocol audit: pilot vs historical MAT-MSA reference

The earlier 100k `3m` identity/obs_norm runs were **engineering pilots**,
not equivalent to the user's map-dependent MAT-MSA benchmark settings.
Their configurations were `num_env_steps=100000`, `ppo_epoch=5`,
`n_rollout_threads=1`, `n_training_threads=2`, and `use_eval=0`
with 32-episode native evaluation applied afterward. Even a paired
five-seed contrast from these runs cannot be reported as a
full-budget benchmark comparison.

The branch launcher
`mat/scripts/train_smac_research.sh` preserves these **provisional**
historical MAT-MSA map-specific *request* defaults, pending
reconciliation with the original map launch scripts:

| Map | Requested environment steps | PPO epochs | PPO clip |
| --- | ---: | ---: | ---: |
| `3m` | 5,000,000 | 15 | 0.20 |
| `3s5z` | 5,000,000 | 10 | 0.05 |
| `5m_vs_6m` | 5,000,000 | 10 | 0.05 |
| `10m_vs_11m` | 5,000,000 | 10 | 0.05 |
| `6h_vs_8z` | 10,000,000 | 15 | 0.05 |
| `MMM2` | 10,000,000 | 5 | 0.05 |
| `3s5z_vs_3s6z` | 20,000,000 | 5 | 0.05 |
| `27m_vs_30m` | 10,000,000 | 5 | 0.20 |

The generic launcher fallback, `10,000,000 / 15 / 0.05`,
must **NOT** be assumed to be the user's verified original
configuration for any unlisted map.

As in the upstream MAT loop, actual environment timesteps
equal `floor(requested/(episode_length*n_rollout_threads)) *
episode_length*n_rollout_threads`. The launcher now reports both
`requested_env_steps` and `effective_env_steps` and emits a
round-down warning. For example, with 32 rollout environments,
5,000,000 requested steps result in 4,998,400 executed steps;
this is expected batching behavior, not a training failure.
Strict divisibility would incorrectly reject authentic runs.

The new metadata field `training_protocol_class` is either
`historical_reference_candidate` or `pilot_or_nonreference`.
Even the *candidate* is explicitly labeled
`historical_reference_candidate_is_unverified=1` until the
user's original launch commands have been checked.
`NUM_ENV_STEPS`, `PPO_EPOCH`, and `CLIP_PARAM` must validate
**before** GPU/SC2 startup. The regression suite
`tests/test_training_protocol.py` dry-runs every explicit map,
the 100k pilot and malformed values without accessing any GPU.
[Source CI #37933180555](https://github.com/suneexxhh/Multi-Agent-Transformer/actions/runs/37933180555)
passed 88 unit tests (84 passed, 4 skipped) and launcher checks.

**Research constraint:** rollout order, update budget, critic,
optimizer, seed, map, evaluation protocol, and effective environment
steps must match between formal identity/obs_norm/learned-order
ablation arms. Sensitivity KL/TV and entropy alone are not
performance-improvement evidence.

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
