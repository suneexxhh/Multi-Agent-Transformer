# SMAC map plan grounded in user-supplied difficulty table

**Source of map labels:** the user's uploaded table (2026-10-09).
Map names below are normalized to the **actual MAT SMAC registry IDs**
(`mat/envs/starcraft2/smac_maps.py`). The nine-map list is an
experiment-selection constraint. It is not a claim about a universal
SMAC difficulty taxonomy.

| User difficulty | Approved maps | Agents |
| --- | --- | --- |
| 简单 | `1c3s5z` | 9 |
| 困难 | `3s5z`, `5m_vs_6m`, `8m_vs_9m`, `10m_vs_11m` | 8, 5, 8, 10 |
| 超高难度 | `6h_vs_8z`, `3s5z_vs_3s6z`, `MMM2`, `27m_vs_30m` | 6, 8, 10, 27 |

Do not add maps outside this table to NEW benchmark training without
the user's permission. Previously archived `3m` experiments are
still useful for offline algorithm and numerical verification but are
outside the newly selected benchmark-map set.

## Experiment sequence and expected scope

Three seed-matched `identity` vs `obs_norm` pilots have been
**scheduled conditionally** (not all have started):

1. **Hard `3s5z`**: 8 allied agents; 100k-step functional pilot,
   PPO 10, clip 0.05; six models = 3 seeds × 2 sortings.
   [Adaptive GPU0 workflow](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/workflows/mat-gpu0-sixway-3s5z-pilot.yml)
   is automatically gated on successful five-seed summary.
2. **Super-hard `3s5z_vs_3s6z`**: 8 allied agents and 9 enemies.
   The identical 8-agent ally composition to `3s5z` is useful
   as a contextual comparison across greater opponent difficulty,
   but legal masks/state trajectories still differ. Bounded pilot:
   1 million env steps, PPO 5, clip 0.05, 3 seeds × two ordering
   modes. Its **provisional historical MAT-MSA** reference is
   20 million env steps, thus this pilot is only 5% of that budget.
   [Super-hard workflow](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/workflows/mat-gpu0-sixway-superhard-3s5z-vs-3s6z.yml)
   runs only after successful `3s5z` pilot.
3. **Super-hard `6h_vs_8z`**: 6 allied agents and 8 enemies.
   Pilot: 1 million env steps, PPO 15, clip 0.05, matched 3 seeds.
   Provisional historical MAT-MSA reference is 10 million steps,
   making this pilot 10% of that budget.
   [Super-hard workflow](https://github.com/suneexxhh/MAT-Decoder-Experiments/actions/workflows/mat-gpu0-sixway-superhard-6h-vs-8z.yml)
   runs only after successful prior `3s5z_vs_3s6z` experiment.

All workflows are subject to the GitHub runner being available and
exclusive `/tmp/hk-mat-gpu0.lock`. The controller can launch at
most **six GPU 0 tasks simultaneously** but caps concurrency
adaptively based on free GPU0 VRAM, available RAM, CPU load, and
existing processes. It never inspects, signals, resets or selects
GPU1 or GPU2. A submitted or chained workflow is not evidence
that training has actually started.

More difficult maps are **not guaranteed** to distinguish methods
within a pilot budget: all methods might have zero wins without
adequate training. A pilot's success is validated by training
integrity, exact PPO replay, checkpoint provenance and native
32-episode SMAC win rates, rather than by expecting a particular
winner. The provisional historical map budgets must be reconciled
with the original MAT-MSA commands before a formal benchmark.

## Later maps and metrics

- Extend, after quality review, to super-hard `MMM2` (10 agents,
  heterogeneous units) and `27m_vs_30m` (27 agents).
- Include hard `5m_vs_6m`, `8m_vs_9m`, and
  `10m_vs_11m` for generalization if the pilot shows measurable
  ordering sensitivity. The original map-specific training budget
  for `8m_vs_9m` has not been confirmed.
- Report seed-aligned **win rate, sample efficiency, training
  stability, execution/throughput cost**, and compute overhead.
  Use 5 independent training seeds and complete map-appropriate
  budgets for publishable evidence.
- `identity` vs `obs_norm` compares **fixed heuristic
  decoding orders**, not learned agent dependency ranking. The
  low-rank scorer remains offline until labels and their
  mathematical interpretation are defensible. KL/TV magnitudes,
  entropy, or a single high win rate cannot serve as causal
  edge supervision by themselves.

Current source of truth for the controlled pilot protocol and map
allowlist: private experiment controller
`scripts/selected_smac_maps.py` and
`scripts/gpu0_bounded_batch.py`.
