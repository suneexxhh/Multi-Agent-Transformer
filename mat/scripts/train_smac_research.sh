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
AGENT_ORDER_MODE="${AGENT_ORDER_MODE:-identity}"
AGENT_ORDER_SEED="${AGENT_ORDER_SEED:-1}"
# For SC2 startup/resource diagnosis use e.g. N_ROLLOUT_THREADS=2
# and USE_EVAL=0. Full baseline stays at 32 threads + evaluation by default.
N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-32}"
N_TRAINING_THREADS="${N_TRAINING_THREADS:-16}"
N_EVAL_ROLLOUT_THREADS="${N_EVAL_ROLLOUT_THREADS:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
USE_EVAL="${USE_EVAL:-1}"
for positive_int in "$N_ROLLOUT_THREADS" "$N_TRAINING_THREADS" \
                    "$N_EVAL_ROLLOUT_THREADS" "$EVAL_EPISODES"; do
  if ! [[ "$positive_int" =~ ^[1-9][0-9]*$ ]]; then
    echo "Thread counts and eval episodes must be positive integers" >&2
    exit 2
  fi
done
if [[ "$USE_EVAL" != "0" && "$USE_EVAL" != "1" ]]; then
  echo "USE_EVAL must be 0 or 1" >&2
  exit 2
fi
case "$AGENT_ORDER_MODE" in
  identity|random_fixed|obs_norm) ;;
  *) echo "Unknown AGENT_ORDER_MODE: $AGENT_ORDER_MODE" >&2; exit 2 ;;
esac

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

# Every real SMAC run uses the explicitly selected Conda environment.
# Dry runs skip activation so launcher checks also work on CPU CI hosts.
MAT_CONDA_ENV="${MAT_CONDA_ENV:-new_titans}"
if [[ "$DRY_RUN" != "1" ]]; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: Conda initialization script not found: $CONDA_SH" >&2
    exit 2
  fi
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$MAT_CONDA_ENV"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$MAT_CONDA_ENV" ]]; then
    echo "ERROR: expected Conda environment $MAT_CONDA_ENV" >&2
    exit 2
  fi
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python not found in Conda environment: $PYTHON_BIN" >&2
    exit 2
  fi
  printf 'active_conda_env=%s python=%s\n' "$CONDA_DEFAULT_ENV" "$PYTHON_BIN"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_TAG}_${AGENT_ORDER_MODE}_orderseed${AGENT_ORDER_SEED}_seed${SEED}"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${MAP}_${EXP_NAME}_gpu${GPU_ID}_${STAMP}.log"

args=(
  --env_name StarCraft2
  --algorithm_name mat
  --agent_order_mode "$AGENT_ORDER_MODE"
  --agent_order_seed "$AGENT_ORDER_SEED"
  --experiment_name "$EXP_NAME"
  --map_name "$MAP"
  --seed "$SEED"
  --n_training_threads "$N_TRAINING_THREADS"
  --n_rollout_threads "$N_ROLLOUT_THREADS"
  --n_eval_rollout_threads "$N_EVAL_ROLLOUT_THREADS"
  --eval_episodes "$EVAL_EPISODES"
  --num_mini_batch 1
  --episode_length 100
  --num_env_steps "$NUM_ENV_STEPS"
  --lr 5e-4
  --ppo_epoch "$PPO_EPOCH"
  --clip_param "$CLIP_PARAM"
  --save_interval 100000
  --use_value_active_masks
)
if [[ "$USE_EVAL" == "1" ]]; then
  args+=(--use_eval)
fi

echo "MAT Decoder Auto Research / original Encoder-Critic baseline"
printf 'map=%s gpu=%s seed=%s steps=%s ppo_epoch=%s clip=%s\n' \
  "$MAP" "$GPU_ID" "$SEED" "$NUM_ENV_STEPS" "$PPO_EPOCH" "$CLIP_PARAM"
printf 'agent_order_mode=%s agent_order_seed=%s\n' "$AGENT_ORDER_MODE" "$AGENT_ORDER_SEED"
printf 'rollout_threads=%s training_threads=%s eval_threads=%s eval_episodes=%s use_eval=%s\n' \
  "$N_ROLLOUT_THREADS" "$N_TRAINING_THREADS" "$N_EVAL_ROLLOUT_THREADS" "$EVAL_EPISODES" "$USE_EVAL"
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
