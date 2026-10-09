# Historical MAT-MSA SMAC scripts — reusable training reference

Source: [suneexxhh/MAT-MSA/mat/scripts](https://github.com/suneexxhh/MAT-MSA/tree/main/mat/scripts), inspected 2026-10-09. These are repository scripts, **not verified training results**. No successful logs or access to the past remote server were available.

## Map-specific parameters extracted from MAT-MSA

| Map | num_env_steps | ppo_epoch | clip_param | Source |
|---|---:|---:|---:|---|
| 3m | 5,000,000 | 15 | 0.20 | [train_smac_3m.sh](https://github.com/suneexxhh/MAT-MSA/blob/main/mat/scripts/train_smac_3m.sh) |
| 3s5z | 5,000,000 | 10 | 0.05 | [train_smac_3s5z.sh](https://github.com/suneexxhh/MAT-MSA/blob/main/mat/scripts/train_smac_3s5z.sh) |
| 5m_vs_6m | 5,000,000 | 10 | 0.05 | [train_smac_5m_vs_6m.sh](https://github.com/suneexxhh/MAT-MSA/blob/main/mat/scripts/train_smac_5m_vs_6m.sh) |
| 10m_vs_11m | 5,000,000 | 10 | 0.05 | [train_smac_10m_vs_11m.sh](https://github.com/suneexxhh/MAT-MSA/blob/main/mat/scripts/train_smac_10m_vs_11m.sh) |
| 6h_vs_8z | 10,000,000 | 15 | 0.05 | [train_smac_6h_vs_8z.sh](https://github.com/suneexxhh/MAT-MSA/blob/main/mat/scripts/train_smac_6h_vs_8z.sh) |
| MMM2 | 10,000,000 | 5 | 0.05 | [train_smac_MMM2.sh](https://github.com/suneexxhh/MAT-MSA/blob/main/mat/scripts/train_smac_MMM2.sh) |
| 3s5z_vs_3s6z | 20,000,000 | 5 | 0.05 | [train_smac_3s5z_vs_3s6z.sh](https://github.com/suneexxhh/MAT-MSA/blob/main/mat/scripts/train_smac_3s5z_vs_3s6z.sh) |
| 27m_vs_30m | 10,000,000 | 5 | 0.20 | [train_smac_27m_vs_30m.sh](https://github.com/suneexxhh/MAT-MSA/blob/main/mat/scripts/train_smac_27m_vs_30m.sh) |

Shared parameters across inspected scripts: algorithm=mat, env=StarCraft2, n_training_threads=16, n_rollout_threads=32, num_mini_batch=1, episode_length=100, learning rate 5e-4, checkpoint save interval 100000, value active masks and evaluation enabled.

## Critical incompatibility

MAT-MSA modified `mat/algorithms/mat/algorithm/ma_transformer.py` to substitute the original Encoder self-attention with a multi-scale Continuum Memory System (CMS). Its `mat/scripts/train/train_smac.py` adds --n_cms_levels, --cms_mid_chunk_size, --use_map_adaptive and related flags. The *new MAT decoder research Fork* follows the **official original Encoder/Critic** and must **not** copy these options or CMS code into baseline and V0 training. In the official CLI `parse_known_args` can silently ignore unrecognized parameters, which would be even more dangerous for reproducibility.

The old MAT-MSA `mat/algorithms/utils/transformer_act.py` blob SHA equals original MAT (`9364b696a0a42f3750259c5fc5dfa86c751c22e7`), making the old script settings a useful context while leaving action decoding unchanged.

## Batch experiment infrastructure from MAT-MSA

- `run_smac_main.sh` selects map, seed and GPU, redirects output to timestamped .log and writes a PID.
- `run_all.sh` lists seeds 1..5, dispatches tasks across GPUs, supports STOP file.
- But `run_all.sh` uses **log existence as completion detection**; a failed or incomplete run can have a log and be incorrectly skipped. It also launches jobs using command substitution for PID capture, which may produce child-process/wait bookkeeping issues.
- For rigorous Auto Research, use exit codes and a completion marker + validated metrics/checkpoints, not merely log existence.

## New baseline launcher

The added `mat/scripts/train_smac_research.sh` uses only original MAT-compatible flags, configurable GPU/map/seed and a dry-run print mode, keeping map-specific legacy defaults.

From repository root:

```bash
DRY_RUN=1 bash mat/scripts/train_smac_research.sh 3m 0 1
bash mat/scripts/train_smac_research.sh 3m 0 1

# Low-budget smoke experiment once dependencies are confirmed:
NUM_ENV_STEPS=100000 PPO_EPOCH=1 EXP_TAG=smoke \
  bash mat/scripts/train_smac_research.sh 3m 0 1
```

**No training has been run by creating this launcher.** The actual server's SC2/SMAC installation, path, CUDA/GPU availability, and runtime behavior still require testing.

## Training protocol

The original MAT and modified decoder experiments should reuse the exact same map, seed, env steps, PPO epochs, clip and evaluation settings. First run identity-order parity and action log-prob tests; only then compare learned orders. Distinguish script settings taken from MAT-MSA from observations from a real successful training run.
