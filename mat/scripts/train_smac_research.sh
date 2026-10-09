#!/usr/bin/env bash
# Reproducible SMAC launcher for MAT decoder-order research.
# Usage (from any directory): bash mat/scripts/train_smac_research.sh [map] [gpu_id] [seed]
# Example: DRY_RUN=1 bash mat/scripts/train_smac_research.sh 3m 0 1
# Example: bash mat/scripts/train_smac_research.sh 3m 0 1
# Override: NUM_ENV_STEPS=100000 PPO_EPOCH=1 CLIP_PARAM=0.2 EXP_TAG=smoke bash ...
# Important: training is NOT launched by adding this file. Existing SMAC/SC2 and PyTorch installation required.
set -euo pipefail

MAP="${1:-3m}"
GPU_ID="${2:-0}"
SEED="${3:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
EXP_TAG="${EXP_TAG:-decoder_v0_baseline}"

# Historical MAT-MSA scripts used different map-specific training budgets.
# These are references for matched baselines, not proven-optimal hyperparameters.
case "$MAP" in
  3m)                 default_steps=5000000;  default_epochs=15; default_clip=0.2 ;;
  3s5z)               default_steps=5000000;  default_epochs=10; default_clip=0.05 ;;
  5m_vs_6m)           default_steps=5000000;  default_epochs=10; default_clip=0.05 ;;
  10m_vs_11m)         default_steps=5000000;  default_epochs=10; default_clip=0.05 ;;
  6h_vs_8z)           default_steps=10000000; default_epochs=15; default_clip=0.05 ;;
  MMM2)               default_steps=10000000; default_epochs=5;  default_clip=0.05 ;;
  3s5z_vs_3s6z)      default_steps=20000000; default_epochs=5;  default_clip=0.05 ;;
  27m_vs_30m)         default_steps=10000000; default_epochs=5;  default_clip=0.2 ;;
  *)                  default_steps=10000000; default_epochs=15; default_clip=0.05 ;;
esac

NUM_ENV_STEPS="${NUM_ENV_STEPS:-$default_steps}"
PPO_EPOCH="${PPO_EPOCH:-$default_epochs}"
CLIP_PARAM="${CLIP_PARAM:-$default_clip}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Original MAT entry point expects its working directory to be mat/scripts.
cd "$SCRIPT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_TAG}_seed${SEED}"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${MAP}_${EXP_NAME}_gpu${GPU_ID}_${STAMP}.log"

args=(
  --env_name StarCraft2
  --algorithm_name mat
  --experiment_name "$EXP_NAME"
  --map_name "$MAP"
  --seed "$SEED"
  --n_training_threads 16
  --n_rollout_threads 32
  --num_mini_batch 1
  --episode_length 100
  --num_env_steps "$NUM_ENV_STEPS"
  --lr 5e-4
  --ppo_epoch "$PPO_EPOCH"
  --clip_param "$CLIP_PARAM"
  --save_interval 100000
  --use_value_active_masks
  --use_eval
)

echo "MAT Decoder Auto Research / original Encoder-Critic baseline"
printf 'map=%s gpu=%s seed=%s steps=%s ppo_epoch=%s clip=%s\n' \
  "$MAP" "$GPU_ID" "$SEED" "$NUM_ENV_STEPS" "$PPO_EPOCH" "$CLIP_PARAM"
printf 'repository_root=%s\nlog_file=%s\n' "$REPO_ROOT" "$LOG_FILE"
printf 'command: CUDA_VISIBLE_DEVICES=%q %q -u train/train_smac.py ' "$GPU_ID" "$PYTHON_BIN"
printf '%q ' "${args[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: command printed; no training started."
  exit 0
fi

mkdir -p "$LOG_DIR"
echo "Launching SMAC training; see $LOG_FILE"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train/train_smac.py "${args[@]}" 2>&1 | tee "$LOG_FILE"
# set -o pipefail preserves the Python failure status in the tee pipeline.
