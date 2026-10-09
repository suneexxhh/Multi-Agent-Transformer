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
STORE_AGENT_ORDERS="${STORE_AGENT_ORDERS:-0}"
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
if [[ "$STORE_AGENT_ORDERS" != "0" && "$STORE_AGENT_ORDERS" != "1" ]]; then
  echo "STORE_AGENT_ORDERS must be 0 or 1" >&2
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

# The method name is an explicit, filename-safe label. Each run produces
# mat/scripts/logs/<method>_seed<seed>_<timestamp>.log.
LOG_METHOD_NAME="${LOG_METHOD_NAME:-MAT_${AGENT_ORDER_MODE}}"
if [[ ! "$LOG_METHOD_NAME" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]]; then
  echo "LOG_METHOD_NAME must contain only letters, digits, underscores or hyphens, beginning with a letter" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer" >&2
  exit 2
fi
STAMP="$(date +%Y%m%d_%H%M%S_%N)"
EXP_NAME="${EXP_TAG}_${AGENT_ORDER_MODE}_orderseed${AGENT_ORDER_SEED}_seed${SEED}"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${LOG_METHOD_NAME}_seed${SEED}_${STAMP}.log"
MAT_CONDA_ENV="${MAT_CONDA_ENV:-new_titans}"

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
if [[ "$STORE_AGENT_ORDERS" == "1" ]]; then
  args+=(--store_agent_orders)
fi

# Work in a subshell when tee captures output; pipefail ensures that a
# failing Python/SMAC process still causes this launcher to fail.
run_training() {
  if [[ "$DRY_RUN" != "1" ]]; then
    # Explicitly activate the SMAC environment even when called from (base)
    # or a fresh GitHub Actions shell.
    local conda_sh="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    if [[ ! -f "$conda_sh" ]]; then
      echo "ERROR: Conda initialization script not found: $conda_sh" >&2
      return 2
    fi
    # shellcheck source=/dev/null
    source "$conda_sh"
    conda activate "$MAT_CONDA_ENV"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$MAT_CONDA_ENV" ]]; then
      echo "ERROR: expected Conda environment $MAT_CONDA_ENV" >&2
      return 2
    fi
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
      echo "ERROR: Python not found in Conda environment: $PYTHON_BIN" >&2
      return 2
    fi
    printf 'active_conda_env=%s python=%s\n' "$CONDA_DEFAULT_ENV" "$PYTHON_BIN"
  fi

  echo "MAT Decoder Auto Research / original Encoder-Critic baseline"
  printf 'method=%s map=%s gpu=%s seed=%s steps=%s ppo_epoch=%s clip=%s\n' \
    "$LOG_METHOD_NAME" "$MAP" "$GPU_ID" "$SEED" "$NUM_ENV_STEPS" "$PPO_EPOCH" "$CLIP_PARAM"
  printf 'agent_order_mode=%s agent_order_seed=%s\n' "$AGENT_ORDER_MODE" "$AGENT_ORDER_SEED"
  printf 'store_agent_orders=%s\n' "$STORE_AGENT_ORDERS"
  printf 'rollout_threads=%s training_threads=%s eval_threads=%s eval_episodes=%s use_eval=%s\n' \
    "$N_ROLLOUT_THREADS" "$N_TRAINING_THREADS" "$N_EVAL_ROLLOUT_THREADS" "$EVAL_EPISODES" "$USE_EVAL"
  printf 'repository_root=%s\nlog_file=%s\n' "$REPO_ROOT" "$LOG_FILE"
  printf 'command: CUDA_VISIBLE_DEVICES=%q %q -u train/train_smac.py ' "$GPU_ID" "$PYTHON_BIN"
  printf '%q ' "${args[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1: command printed; no training started."
    return 0
  fi

  printf 'training_started_at=%s\n' "$(date -Iseconds)"
  echo "Launching SMAC training; see $LOG_FILE"
  local status=0
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train/train_smac.py "${args[@]}" || status=$?
  printf 'training_finished_at=%s exit_code=%s\n' "$(date -Iseconds)" "$status"
  return "$status"
}

if [[ "$DRY_RUN" == "1" ]]; then
  # CI can exercise argument parsing without creating logs or requiring SC2.
  run_training
else
  mkdir -p "$LOG_DIR"
  # Include launcher metadata, SC2 stdout/stderr, exceptions, and final status.
  # This tee is waited for before returning, so artifacts see complete logs.
  run_training 2>&1 | tee "$LOG_FILE"
fi
