# AR-03 · First GPU SMAC smoke tests and CUDA allocation fix

**Date:** 2026-10-09
**Map:** SMAC `3m`
**Runner:** user's remote Ubuntu GPU server, repository `~/ActionDec/Multi-Agent-Transformer` (user-provided terminal output; remote server not directly accessible to ChatGPT)
**Runtime:** Python 3.9 conda environment `new_titans`; StarCraft II 4.10 B75689
**Parameters:** `--algorithm_name mat`, `--seed 1`, `--n_rollout_threads 32`, `--num_env_steps 100000`, `--ppo_epoch 1`, `--clip_param 0.2`, GPU 0.

## Observations from supplied logs

1. **Identity-order reference smoke test**
   - Reached the last reported update at **99,200 of 100,000 environment steps**, update 30/31, FPS 1479.
   - `eval win rate is 0.09375` was reported early in the run.
   - `eval win rate is 0.59375` was reported at the update around 99,200 steps.
   - Python traceback absent from the supplied run excerpt; normal-looking SC2 shutdown messages.
   - These numbers are **single-seed smoke diagnostics**, not a baseline for statistically meaningful performance claims.

2. **Observation-norm order**
   - Successfully started GPU model and 32 rollout SC2 processes.
   - At initial rollout collection the following exception occurred:
     ```
     RuntimeError: Expected all tensors to be on the same device, but found
     at least two devices, cpu and cuda:0!
     ```
   - Location: `transformer_act.py` ordered action wrapper calling `restore_agents()`; action tensor allocated on CPU by original MAT sampler while agent order index was on CUDA.
   - No meaningful training/evaluation metric can be inferred from this failed run.
   - Subsequent SC2 connection/shutdown warnings are consistent with worker teardown after the Python traceback. There is no evidence here that SC2 installation/driver must be changed.

## Implemented correction

- `mat/algorithms/utils/transformer_act.py`: allocate **discrete sampled action** with `device=shifted_action.device` and **action log-probs** with the same device and float dtype.
- Apply the same principle to **continuous action samples and log-probs**.
- Keep `agent_order` CUDA-resident; no copying tensors to CPU to work around errors.
- No modification to original Encoder, Critic, action semantics, PPO objective or environment.
- `tests/test_agent_ordering.py`: add CUDA-only regression tests for discrete full MAT (`identity`, `random_fixed`, `obs_norm`) and continuous actions; test suite skips CUDA cases only if there is no CUDA runtime.

## Validation / remaining work

- CPU-only GitHub Actions validates import, sampling/evaluation consistency and CPU behavior. **CUDA regression tests must be run on the actual GPU server**, because public GitHub-hosted runner has no CUDA device.
- Pull the research branch and run `python -m unittest discover -s tests -v` from repository root, checking that CUDA tests execute instead of being skipped.
- Repeat the small `obs_norm` smoke test with the same parameters.
- Confirm `total num timesteps` progress, at least one `eval win rate`, process exit code 0, and generated log file before marking GPU validation complete.

## Server follow-up

```bash
cd ~/ActionDec/Multi-Agent-Transformer
git fetch origin
git switch research/dependency-decoder-v0
git pull --ff-only
python -m unittest discover -s tests -v

NUM_ENV_STEPS=100000 PPO_EPOCH=1 \
  AGENT_ORDER_MODE=obs_norm EXP_TAG=smoke \
  bash mat/scripts/train_smac_research.sh 3m 0 1
```
