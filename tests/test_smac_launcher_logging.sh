#!/usr/bin/env bash
# Offline regression test: verify the real launcher saves full stdout/stderr
# and reports failures without requiring SMAC, CUDA, or Conda installation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="$repo_root/mat/scripts/train_smac_research.sh"
work="$(mktemp -d)"
log_file=""
cleanup() {
  if [[ -n "$log_file" && -f "$log_file" ]]; then
    rm -f -- "$log_file"
  fi
  rm -rf -- "$work"
}
trap cleanup EXIT

dry="$(DRY_RUN=1 AGENT_ORDER_MODE=obs_norm bash "$launcher" 3m 0 7)"
[[ "$dry" == *"DRY_RUN=1: command printed; no training started."* ]]
grep -Eq 'log_file=.*/mat/scripts/logs/MAT_obs_norm_seed7_[0-9]{8}_[0-9]{6}_[0-9]{9}\.log' <<< "$dry"

# The diagnostic export is explicitly opt-in and must carry stored orders.
dry_diag="$(DRY_RUN=1 STORE_AGENT_ORDERS=1 DECODER_DIAG_CAPTURE=1 \
  AGENT_ORDER_MODE=obs_norm bash "$launcher" 3m 0 7)"
grep -Fq 'decoder_diag_capture=1' <<< "$dry_diag"
grep -Fq ' --store_agent_orders ' <<< "$dry_diag"
grep -Fq ' --decoder_diag_capture_dir ' <<< "$dry_diag"
grep -Fq ' --decoder_diag_capture_id MAT_obs_norm_seed7_' <<< "$dry_diag"
if DRY_RUN=1 DECODER_DIAG_CAPTURE=1 STORE_AGENT_ORDERS=0 \
   bash "$launcher" 3m 0 7 > /dev/null 2>&1; then
  echo "ERROR: must reject capture without saved rollout order" >&2
  exit 1
fi

# Bounded intermediate-policy training may capture 0/1/10/100/500/900.
dry_multi="$(DRY_RUN=1 STORE_AGENT_ORDERS=1 DECODER_DIAG_CAPTURE=1 \
  DECODER_DIAG_CAPTURE_EPISODES=0,1,10,100,500,900 \
  AGENT_ORDER_MODE=obs_norm bash "$launcher" 3m 0 7)"
grep -Fq ' --decoder_diag_capture_episodes 0\\,1\\,10\\,100\\,500\\,900 ' <<< "$dry_multi"
if DRY_RUN=1 STORE_AGENT_ORDERS=1 DECODER_DIAG_CAPTURE=1 \
  DECODER_DIAG_CAPTURE_EPISODES='1,10,abc' bash "$launcher" 3m 0 7 >/dev/null 2>&1; then
  echo "ERROR: invalid diagnostic checkpoint stages must fail before SMAC" >&2
  exit 1
fi

mkdir -p "$work/fake_env/bin"
cat > "$work/conda.sh" <<'FAKE_CONDA'
conda() {
  [[ "$1" == "activate" && "$2" == "new_titans" ]] || return 2
  export CONDA_DEFAULT_ENV="new_titans"
  export CONDA_PREFIX="$FAKE_CONDA_PREFIX"
}
FAKE_CONDA
cat > "$work/fake_env/bin/python" <<'FAKE_PYTHON'
#!/usr/bin/env bash
echo "SMAC_FAKE_STDOUT_MARKER"
echo "SMAC_FAKE_STDERR_MARKER" >&2
exit 19
FAKE_PYTHON
chmod +x "$work/fake_env/bin/python"

# The test checks both log content and the launcher's nonzero exit status.
set +e
output="$(CONDA_SH="$work/conda.sh" FAKE_CONDA_PREFIX="$work/fake_env" \
  AGENT_ORDER_MODE=obs_norm NUM_ENV_STEPS=2000 PPO_EPOCH=1 \
  N_ROLLOUT_THREADS=1 USE_EVAL=0 \
  bash "$launcher" 3m 0 7 2>&1)"
status=$?
set -e
[[ "$status" -eq 19 ]] || { echo "Expected exit 19, got $status" >&2; exit 1; }
log_file="$(sed -n 's/^log_file=//p' <<< "$output" | tail -n 1)"
[[ -f "$log_file" ]] || { echo "Log not created: $log_file" >&2; exit 1; }
[[ "$log_file" =~ /mat/scripts/logs/MAT_obs_norm_seed7_[0-9]{8}_[0-9]{6}_[0-9]{9}\.log$ ]]
grep -Fq 'method=MAT_obs_norm map=3m gpu=0 seed=7 steps=2000' "$log_file"
grep -Fq 'active_conda_env=new_titans' "$log_file"
grep -Fq 'SMAC_FAKE_STDOUT_MARKER' "$log_file"
grep -Fq 'SMAC_FAKE_STDERR_MARKER' "$log_file"
grep -Fq 'training_finished_at=' "$log_file"
grep -Fq 'exit_code=19' "$log_file"
echo "Launcher log naming, output capture, and failure status: PASS"
