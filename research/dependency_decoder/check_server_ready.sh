#!/usr/bin/env bash
# Read-only MAT Decoder server readiness check.
# From repository root: bash research/dependency_decoder/check_server_ready.sh
# Does not print credentials, environment variables or host/IP information.
# Does not install packages, start training, or modify GPU/driver configuration.
set -u

echo "=== MAT Decoder AR-03 | Server readiness ==="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "OS: $(uname -s) $(uname -m)"
echo "Kernel: $(uname -r)"
echo "CPU cores: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
echo "Memory:"
free -h 2>/dev/null | awk 'NR<=2 { print }' || true
echo "Available disk at working directory:"
df -h . 2>/dev/null | tail -n 1 || true

echo
echo "=== Git checkout ==="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'branch: '
  git branch --show-current
  printf 'commit: '
  git rev-parse --short HEAD
  if test -n "$(git status --porcelain 2>/dev/null)"; then
    echo "NOTE: working tree has uncommitted modifications"
  fi
else
  echo "NOTE: run from the MAT repository directory"
fi

echo
echo "=== GPU / NVIDIA ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  # A watchdog protects the shell from hanging indefinitely on an unhealthy driver.
  if command -v timeout >/dev/null 2>&1; then
    timeout 15s nvidia-smi --query-gpu=index,name,memory.total,driver_version \
      --format=csv,noheader 2>&1 || echo "WARNING: nvidia-smi error or timeout"
  else
    echo "timeout is unavailable; nvidia-smi skipped to avoid blocking"
  fi
else
  echo "WARNING: nvidia-smi is not installed or not in PATH"
fi

echo
echo "=== Python / PyTorch / SMAC requirements ==="
if ! command -v python >/dev/null 2>&1; then
  echo "WARNING: python executable was not found"
else
  python - <<'PY'
import importlib.util
import sys
print("Python:", sys.version.split()[0])
for name in ("torch", "numpy", "absl", "pysc2", "s2clientprotocol",
             "tensorboardX", "wandb"):
    try:
        found = importlib.util.find_spec(name) is not None
        print("{}: {}".format(name, "present" if found else "MISSING"))
    except Exception as exc:
        print("{}: detection failed ({})".format(name, type(exc).__name__))
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Visible CUDA devices:", torch.cuda.device_count())
except Exception as exc:
    print("WARNING: torch import/CUDA inspection failed:", type(exc).__name__, str(exc))
PY
fi

echo
echo "=== StarCraft II installation ==="
if [ -n "${SC2PATH:-}" ]; then
  SC2_DIR="$SC2PATH"
else
  SC2_DIR="$HOME/StarCraftII"
fi
if [ -d "$SC2_DIR" ]; then
  echo "SC2 install directory: found"
  if [ -d "$SC2_DIR/Maps" ]; then
    echo "SC2 Maps directory: found"
  else
    echo "WARNING: SC2 Maps directory: missing"
  fi
else
  echo "WARNING: SC2 install directory: missing at SC2PATH or ~/StarCraftII"
fi

echo
echo "=== Summary ==="
echo "This is an environment inventory, not a SMAC rollout or GPU training test."
echo "Review GPU access and Python/SMAC dependencies before enabling automated runs."
