# Validated V2: paired 3s5z pilot experiment protocol

This protocol prepares experiments only. No GPU or StarCraft II training starts
when the suite is generated. The main contrast is original MAT identity versus
MAT dependency with attention bias. `dependency-NoBias`, `learned`, and
`obs_norm` are prespecified ablations, not extra primary baselines.

## Budget and common configuration

Use 2,000,000 requested and effective environment steps, 32 rollout threads,
episode length 100, 625 complete updates, PPO epoch 10, clip 0.05, learning
rate 5e-4, one minibatch, 16 CPU training threads, and the same MAT model
sizes, masks, GAE and reward settings. This is a **pilot**, not a final result.
The previous 100k-step run contained about 31 update batches and produced zero
wins across compared methods. Two million steps provides 625 update batches,
about 20 times that exposure and 40% of the map's 5-million-step reference
budget. This gives the policies a more meaningful chance to learn while
remaining cheaper than a full 5-million-step comparison. A zero-win 2M result
would be inconclusive; do not infer ordering equality. Do not choose a method
based on its best checkpoint.

Training checkpoints and periodic native SMAC evaluation both occur every 100
updates; native evaluation uses 32 episodes at each event. The initial update
(index 0) and final update (index 624) are also checkpointed. The final frozen
checkpoint is evaluated separately for 32 deterministic native SMAC episodes
using the same paired evaluation seed (SMAC seed = training seed * 50000).
All five modes receive the same schedule, budget and seeds 1, 2, 3. `learned`
and both dependency modes save sampled orders and old joint order likelihoods.
The NoBias mode sets `--disable_order_attention_bias`; it otherwise retains the
same dependency ordering objective and architecture.

Primary pilot endpoint: paired final frozen win-rate difference between
`dependency` and `identity`, with all three seed-level wins/32 reported.
Secondary: common-step training evaluation curve/AUC, rewards, order drift,
failed runs, elapsed time. The three ablations are descriptive. A subsequent
5-million-step confirmation requires a separate suite and new approval; the
2M pilot must never be presented as a full-budget performance result.

## Directory and recovery rules

Prepare a suite at a new absolute location:

```
/home/sgg/miniconda3/envs/new_titans/bin/python -m research.dependency_decoder.prepare_three_seed_experiments prepare /absolute/new/persistent-suite
```

`suite_manifest.json` lists all 15 run manifests. Each independent run contains
`manifest.json`, `train.sh`, `evaluation_plan.json`, `evaluate_final.sh`,
`results_index.json`, `logs/dry_run.txt`, `logs/`, `evaluation/`, and `analysis/`.
The training command uses a dedicated `training_results/` root. The native
Runner creates `StarCraft2/3s5z/mat/<experiment_name>/run1` below it, with
`mat_decoder_manifest.json`, `models/transformer_<update>.pt`, and
`logs/summary.json` plus TensorBoard events. The outer log directory contains
the launcher's complete timestamped stdout/stderr log. All paths are absolute
in `manifest.json`; run IDs are unique. Source hashes and a compressed source
snapshot live once at the suite root.

The generated train script checks the source SHA, clean tracked code, source
hashes, disk headroom, expected empty output path and GPU0 exclusive lock before
launch. It will not silently create `run2` after an interrupted attempt. The
native checkpoint is a model state dict only: an interrupted run can be
located and evaluated at its last checkpoint, but cannot resume optimizer,
RNG and SC2 state exactly. A replacement run requires a new run ID and must
be reported separately. Rebuild the result index at any time, including after
an interruption:

```
/home/sgg/miniconda3/envs/new_titans/bin/python -m research.dependency_decoder.prepare_three_seed_experiments index /absolute/persistent-suite
```

Frozen evaluation loads the *runtime* `mat_decoder_manifest.json`, recreates
all architecture and bias flags, and loads the state dict with `strict=True`.
`learned`/`dependency` evaluations require that manifest. The suite stores the
expected manifest before training and the evaluator checks the actual one.
The frozen output includes source SHA, checkpoint SHA256, seed, wins and
bias setting. Every mode uses `--manifest` to prevent guessed defaults.

## Resource and authorization gate

Use GPU0 only and run one 32-environment job at a time. CPU and RAM usage are
primarily driven by the 32 SC2 instances, not the small MAT checkpoint.
Observe GPU0 memory, available RAM, free disk and `/tmp/hk-mat-gpu0.lock`
immediately before any launch. The generated train script requires at least
25 GiB free on the suite filesystem. Do not use GPU1/GPU2 or stop existing
processes. The suite preparation and DRY_RUN are safe on CPU and do not acquire
the GPU lock. A prepared `train.sh` is a future command, not authorization to
execute it. The user must explicitly confirm before any GPU training.
