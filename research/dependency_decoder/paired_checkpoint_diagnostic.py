"""CPU-only paired MAT checkpoint comparison on IDENTICAL SMAC contexts.

Samples joint actions once under the early checkpoint, then evaluates
the exact same observations, masks, stored agent orders and histories
under both fixed checkpoints. This separates observation-context drift
from policy-dependent conditional-action sensitivity. Changes are still
about the modeled policy, never environment causality.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from research.dependency_decoder.checkpoint_diagnostic import (
    _load_snapshot, load_frozen_model,
)
from research.dependency_decoder.frozen_policy_sampling import (
    sample_frozen_decoder_histories, teacher_forced_history_log_probs,
    self_normalized_policy_weights,
)
from research.dependency_decoder.expected_action_dependency import (
    probe_action_histories, expected_dependency,
)


@torch.no_grad()
def compare_frozen_checkpoints(checkpoint_early, checkpoint_late, snapshot,
                               n_block, n_embd, n_head, histories=8,
                               max_context_batch=2, max_counterfactual_batch=16,
                               seed=1, min_importance_ess=3.0,
                               direct_target_histories=0):
    if (histories < 2 or min_importance_ess < 1 or
            not isinstance(direct_target_histories, int) or
            isinstance(direct_target_histories, bool) or
            direct_target_histories < 0 or direct_target_histories == 1):
        raise ValueError("histories >= 2; direct_target_histories=0 or >=2; min_importance_ess >= 1")
    torch.manual_seed(seed)
    obs, legal, order = _load_snapshot(snapshot)
    b, n, obs_dim = obs.shape
    a = legal.shape[-1]
    early = load_frozen_model(
        checkpoint_early, n, obs_dim, a, n_block, n_embd, n_head)
    late = load_frozen_model(
        checkpoint_late, n, obs_dim, a, n_block, n_embd, n_head)
    state = torch.zeros(b, n, 37)
    _, early_rep = early.encoder(state, obs)
    _, late_rep = late.encoder(state, obs)
    first = time.perf_counter()
    actions, _, log_early = sample_frozen_decoder_histories(
        early.decoder, early_rep, obs, legal, order, histories,
        max_context_batch=max_context_batch)
    _, log_early_replay = teacher_forced_history_log_probs(
        early.decoder, early_rep, obs, legal, order, actions,
        max_context_batch=max_context_batch)
    max_replay_error = float((log_early - log_early_replay).abs().max())
    if max_replay_error > 2e-5:
        raise RuntimeError("early sampling and teacher forcing disagree")
    _, log_late = teacher_forced_history_log_probs(
        late.decoder, late_rep, obs, legal, order, actions,
        max_context_batch=max_context_batch)
    weights, importance_ess = self_normalized_policy_weights(
        log_late, log_early)
    old_kl, old_valid, old_tv = probe_action_histories(
        early.decoder, early_rep, obs, actions, legal, order,
        max_context_batch=max_context_batch,
        max_counterfactual_batch=max_counterfactual_batch,
        return_tv=True)
    new_kl, new_valid = probe_action_histories(
        late.decoder, late_rep, obs, actions, legal, order,
        max_context_batch=max_context_batch,
        max_counterfactual_batch=max_counterfactual_batch)
    if not torch.equal(old_valid, new_valid):
        raise RuntimeError("fixed masks and order must produce same identifiability")
    old_mean, coverage, _ = expected_dependency(old_kl, old_valid)
    old_tv_mean, _, _ = expected_dependency(old_tv, old_valid)
    new_mean_on_old_histories, _, _ = expected_dependency(new_kl, new_valid)
    weighted_new_mean, weighted_coverage, _ = expected_dependency(
        new_kl, new_valid, weights)
    total_s = time.perf_counter() - first
    measurable = coverage > 0
    num_measurable = int(measurable.sum())
    abs_difference = (new_mean_on_old_histories - old_mean).abs()
    enough_ess = importance_ess >= min_importance_ess
    safe_mask = measurable & enough_ess[:, None, None]
    num_safe = int(safe_mask.sum())
    # When the behavior/target policies have poor overlap, SNIS must
    # abstain. Sampling directly from the frozen target policy
    # estimates E_{a~p}[C_p(a)] at the same observations/masks/orders,
    # without any importance ratio. These target actions are NOT
    # paired one-to-one with the original early-policy actions.
    direct = dict(
        direct_target_histories=direct_target_histories,
        direct_target_replay_max_error=None,
        direct_target_measurable_pairs=None,
        direct_target_mean_kl=None,
        direct_target_mean_tv=None,
        direct_target_mean_tv_by_context=None,
        direct_target_tv_mc_se_by_context=None,
        direct_target_observable_pair_count_by_context=None,
    )
    if direct_target_histories:
        late_actions, _, late_sample_logp = sample_frozen_decoder_histories(
            late.decoder, late_rep, obs, legal, order,
            direct_target_histories, max_context_batch=max_context_batch)
        _, late_replay_logp = teacher_forced_history_log_probs(
            late.decoder, late_rep, obs, legal, order, late_actions,
            max_context_batch=max_context_batch)
        late_replay_error = float(
            (late_sample_logp - late_replay_logp).abs().max())
        if late_replay_error > 2e-5:
            raise RuntimeError("target sampling and teacher-forcing likelihood mismatch")
        late_kl_direct, late_valid_direct, late_tv_direct = probe_action_histories(
            late.decoder, late_rep, obs, late_actions, legal, order,
            max_context_batch=max_context_batch,
            max_counterfactual_batch=max_counterfactual_batch,
            return_tv=True)
        direct_kl, direct_coverage, _ = expected_dependency(
            late_kl_direct, late_valid_direct)
        direct_tv, _, _ = expected_dependency(
            late_tv_direct, late_valid_direct)
        direct_measured = direct_coverage > 0
        # The same legal-action masks and fixed order make measured
        # original-ID pairs comparable across both policies.
        if not torch.equal(direct_measured, measurable):
            raise RuntimeError("direct target and behavior have different measured-pair support")
        count_per_context = direct_measured.sum(dim=(-1,-2))
        per_hist_context_tv = (
            late_tv_direct.sum(dim=(-1,-2)) /
            count_per_context.clamp_min(1)[None, :])
        # This SE measures sampling variability for the target history
        # distribution CONDITIONAL on fixed contexts; it is NOT a
        # confidence interval across independent SMAC trajectories.
        se = (per_hist_context_tv.std(dim=0, unbiased=True) /
              direct_target_histories ** 0.5)
        direct = dict(
            direct_target_histories=direct_target_histories,
            direct_target_replay_max_error=late_replay_error,
            direct_target_measurable_pairs=int(direct_measured.sum()),
            direct_target_mean_kl=(
                float(direct_kl[direct_measured].mean())
                if direct_measured.any() else None),
            direct_target_mean_tv=(
                float(direct_tv[direct_measured].mean())
                if direct_measured.any() else None),
            direct_target_mean_tv_by_context=[
                float(direct_tv[i][direct_measured[i]].mean())
                if direct_measured[i].any() else None for i in range(b)],
            direct_target_tv_mc_se_by_context=[
                float(se[i]) if direct_measured[i].any() else None
                for i in range(b)],
            direct_target_observable_pair_count_by_context=[
                int(v) for v in count_per_context],
        )
    weight_shift = max(
        float((lhs.detach() - rhs.detach()).abs().max())
        for lhs, rhs in zip(early.parameters(), late.parameters()))
    with torch.no_grad():
        max_joint_logp_change = float((log_late - log_early).abs().max())
    return dict(
        diagnostic_only=True,
        checkpoint_early=str(checkpoint_early),
        checkpoint_late=str(checkpoint_late),
        fixed_observation_snapshot=str(snapshot),
        observations=b, agents=n, histories=histories,
        measurable_pairs=num_measurable,
        sampled_vs_replay_max_error=max_replay_error,
        max_absolute_parameter_shift=weight_shift,
        max_absolute_joint_logp_shift=max_joint_logp_change,
        matched_early_mean_kl=(float(old_mean[measurable].mean())
                               if num_measurable else None),
        matched_early_mean_tv=(float(old_tv_mean[measurable].mean())
                               if num_measurable else None),
        **direct,
        matched_late_mean_kl=(float(new_mean_on_old_histories[measurable].mean())
                              if num_measurable else None),
        mean_absolute_matched_pair_kl_change=(
            float(abs_difference[measurable].mean())
            if num_measurable else None),
        per_context_mean_absolute_kl_change=[
            (float(abs_difference[i][measurable[i]].mean())
             if measurable[i].any() else None) for i in range(b)],
        importance_ess_by_context=[float(value) for value in importance_ess],
        min_ess_threshold=min_importance_ess,
        contexts_with_adequate_ess=int(enough_ess.sum()),
        snis_measurable_pairs_with_adequate_ess=num_safe,
        snis_late_kl_mean=(float(weighted_new_mean[safe_mask].mean())
                           if num_safe else None),
        snis_mean_coverage=(float(weighted_coverage[safe_mask].mean())
                            if num_safe else None),
        cpu_seconds=round(total_s, 6),
        warning=("Same frozen observation/mask/order/action-history contexts "
                 "isolate policy-score differences from observed-state changes, "
                 "but are not a causal test or a win-rate evaluation. "
                 "SNIS values are withheld when importance ESS is too low."),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-early", type=Path, required=True)
    parser.add_argument("--checkpoint-late", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--n-block", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--histories", type=int, default=8)
    parser.add_argument("--max-context-batch", type=int, default=2)
    parser.add_argument("--max-counterfactual-batch", type=int, default=16)
    parser.add_argument("--min-importance-ess", type=float, default=3.0)
    parser.add_argument("--direct-target-histories", type=int, default=0,
                        help="Sample this many histories directly from the frozen target policy (0 disables)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = compare_frozen_checkpoints(
        args.checkpoint_early, args.checkpoint_late, args.snapshot,
        args.n_block, args.n_embd, args.n_head, args.histories,
        args.max_context_batch, args.max_counterfactual_batch,
        args.seed, args.min_importance_ess, args.direct_target_histories)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
