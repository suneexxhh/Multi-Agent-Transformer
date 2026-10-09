"""Read-only, CPU-only MAT checkpoint + observation-snapshot diagnostic.

Use real SMAC snapshots when available. Zero PPO updates, zero SC2 startup,
and no learned dependency inference. The diagnostic reports conditional
policy sensitivities under ONE saved decoder permutation; reverse edges
are unobservable under that same ordering.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.mat.algorithm.agent_ordering import validate_agent_order
from research.dependency_decoder.frozen_policy_sampling import (
    sample_frozen_decoder_histories, teacher_forced_history_log_probs,
)
from research.dependency_decoder.expected_action_dependency import (
    probe_action_histories, expected_dependency,
)


def _load_snapshot(path):
    with np.load(str(path), allow_pickle=False) as data:
        required = ("obs", "available_actions", "agent_order")
        if any(k not in data for k in required):
            raise ValueError("snapshot must contain obs, available_actions, agent_order")
        obs = np.asarray(data["obs"], dtype=np.float32)
        legal = np.asarray(data["available_actions"], dtype=np.float32)
        order = np.asarray(data["agent_order"])
    if (obs.ndim != 3 or legal.ndim != 3 or
            legal.shape[:2] != obs.shape[:2] or order.shape != obs.shape[:2] or
            obs.shape[0] < 1 or obs.shape[1] < 2 or obs.shape[0] > 64 or
            not np.isfinite(obs).all() or not np.isfinite(legal).all() or
            legal.shape[-1] < 2 or (legal < 0).any() or
            not np.isin(legal, [0.0, 1.0]).all()):
        raise ValueError("invalid/nonfinite snapshot, too many contexts or bad masks")
    if order.dtype != np.int64:
        raise ValueError("saved agent_order must be int64")
    validate_agent_order(torch.from_numpy(order), obs.shape[0], obs.shape[1], "cpu")
    if not np.all(legal.any(axis=-1)):
        raise ValueError("snapshot contains an agent without legal actions")
    return (torch.from_numpy(obs), torch.from_numpy(legal),
            torch.from_numpy(order))



def load_frozen_model(checkpoint, n_agent, obs_dim, act_dim,
                      n_block, n_embd, n_head):
    """Strict weights-only CPU restore; shared by single/stage comparisons."""
    if min(n_agent, obs_dim, act_dim, n_block, n_embd, n_head) < 1:
        raise ValueError("model dimensions must be positive")
    model = MultiAgentTransformer(
        state_dim=37, obs_dim=obs_dim, action_dim=act_dim, n_agent=n_agent,
        n_block=n_block, n_embd=n_embd, n_head=n_head,
        device=torch.device("cpu"), action_type="Discrete",
        agent_order_mode="identity", encode_state=False)
    state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

@torch.no_grad()
def run_diagnostic(checkpoint, snapshot, n_block, n_embd, n_head,
                   num_histories=16, max_context_batch=8,
                   max_counterfactual_batch=32, seed=1):
    """Return compact, finite, JSON-serializable diagnostic summary."""
    if (min(n_block, n_embd, n_head, num_histories,
            max_context_batch, max_counterfactual_batch) < 1 or
            n_embd % n_head):
        raise ValueError("positive dimensions needed, with n_embd divisible by n_head")
    torch.manual_seed(seed)
    # Never claim a past checkpoint's original behavior policy without
    # recording the saved architecture, obs snapshot, masks, and order.
    obs, legal, order = _load_snapshot(snapshot)
    b, n, obs_dim = obs.shape
    act_dim = legal.shape[-1]
    model = load_frozen_model(
        checkpoint, n, obs_dim, act_dim, n_block, n_embd, n_head)
    state = torch.zeros((b, n, 37), dtype=torch.float32)
    _, rep = model.encoder(state, obs)
    start = time.perf_counter()
    histories, sampled_agent_lp, sampled_joint_lp = sample_frozen_decoder_histories(
        model.decoder, rep, obs, legal, order, num_histories,
        max_context_batch=max_context_batch)
    sampling_s = time.perf_counter() - start
    start = time.perf_counter()
    replay_lp, replay_joint = teacher_forced_history_log_probs(
        model.decoder, rep, obs, legal, order, histories,
        max_context_batch=max_context_batch)
    likelihood_s = time.perf_counter() - start
    maximum_lp_error = max(
        (sampled_agent_lp - replay_lp).abs().max().item(),
        (sampled_joint_lp - replay_joint).abs().max().item(),
    )
    if maximum_lp_error > 2e-5:
        raise RuntimeError("sampled and teacher-forced log-probabilities disagree")

    # The sampled complete-action negative log-likelihood is an
    # unbiased Monte Carlo estimate of joint policy entropy at each
    # fixed observation/mask/order. Reuse already-recorded log-probs:
    # absolutely NO extra Decoder forward is required.
    action_nll = -sampled_joint_lp
    joint_entropy_by_context = action_nll.mean(dim=0)
    entropy_se = (action_nll.std(dim=0, unbiased=True) /
                  num_histories ** 0.5) if num_histories > 1 else None
    # A policy confined to K_j legal actions for each agent has
    # maximal joint entropy sum_j log(K_j), regardless of agent order.
    legal_capacity_by_context = (
        legal.bool().sum(-1).double().log().sum(-1))

    start = time.perf_counter()
    sensitivity, valid, variation = probe_action_histories(
        model.decoder, rep, obs, histories, legal, order,
        max_context_batch=max_context_batch,
        max_counterfactual_batch=max_counterfactual_batch,
        return_tv=True)
    estimate, coverage, ess = expected_dependency(sensitivity, valid)
    tv_mean, _, _ = expected_dependency(variation, valid)
    sensitivity_s = time.perf_counter() - start
    offdiag = ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
    measured = (coverage > 0) & offdiag
    pairs = int(measured.sum())
    count_by_context = measured.sum(dim=(1, 2))
    mean_by_context = (estimate * measured).sum(dim=(1, 2)) / count_by_context.clamp_min(1)
    tv_by_context = (tv_mean * measured).sum(dim=(1, 2)) / count_by_context.clamp_min(1)
    # Pinsker bound holds after averaging histories/alternative actions
    # by Jensen; small positive excess is diagnostic roundoff only.
    pinsker_excess = (2 * tv_mean.square() - estimate).clamp_min(0)
    with np.load(str(snapshot), allow_pickle=False) as source:
        context_step = (source["context_step"].tolist()
                        if "context_step" in source else None)
        context_episode = (source["context_episode"].tolist()
                           if "context_episode" in source else None)
    if context_step is not None and (len(context_step) != b or
                                     len(context_episode) != b):
        raise ValueError("invalid context provenance lengths")
    bidirectional = measured & measured.transpose(-1, -2)
    return dict(
        diagnostic_only=True, trained_smac_performance_claim=False,
        model_checkpoint=str(checkpoint), observation_snapshot=str(snapshot),
        n_agents=n, obs_dim=obs_dim, action_dim=act_dim,
        architecture=dict(n_block=n_block, n_embd=n_embd, n_head=n_head),
        snapshot_contexts=b, sampled_histories=num_histories,
        context_step=context_step, context_episode=context_episode,
        observable_pairs_by_context=[int(x) for x in count_by_context.tolist()],
        mean_kl_by_context=[float(value) if int(count_by_context[i]) else None
                            for i, value in enumerate(mean_by_context)],
        mean_tv_by_context=[float(value) if int(count_by_context[i]) else None
                            for i, value in enumerate(tv_by_context)],
        sampled_joint_logp_mean=float(sampled_joint_lp.mean()),
        joint_action_entropy_mc_mean_nats=float(joint_entropy_by_context.mean()),
        joint_action_entropy_mc_nats_by_context=[
            float(x) for x in joint_entropy_by_context.tolist()],
        joint_action_entropy_mc_se_nats_by_context=(
            [float(x) for x in entropy_se.tolist()]
            if entropy_se is not None else [None] * b),
        legal_action_entropy_ceiling_nats_by_context=[
            float(x) for x in legal_capacity_by_context.tolist()],
        joint_action_entropy_fraction_by_context=[
            float(joint_entropy_by_context[i] / legal_capacity_by_context[i])
            if legal_capacity_by_context[i] > 0 else None
            for i in range(b)],
        max_sampling_likelihood_error=maximum_lp_error,
        observable_directed_pairs=pairs,
        total_directed_pairs=b*n*(n-1),
        bidirectionally_observable_pairs=int(bidirectional.sum()),
        mean_observable_kl=(float(estimate[measured].mean()) if pairs else None),
        max_observable_kl=(float(estimate[measured].max()) if pairs else None),
        mean_observable_tv=(float(tv_mean[measured].mean()) if pairs else None),
        max_observable_tv=(float(tv_mean[measured].max()) if pairs else None),
        max_pinsker_excess=(float(pinsker_excess[measured].max()) if pairs else None),
        mean_observed_coverage=(float(coverage[measured].mean()) if pairs else None),
        min_observed_ess=(float(ess[measured].min()) if pairs else None),
        cpu_seconds=dict(sampling=round(sampling_s, 6),
                         likelihood=round(likelihood_s, 6),
                         intervention=round(sensitivity_s, 6)),
        warning=("Only predecessor-to-successor conditional dependencies "
                 "can be probed under the stored order. Policy entropy is "
                 "estimated only by Monte Carlo at fixed states; it is not "
                 "a causal graph, win-rate result, or runtime speedup."),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--n-block", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--num-histories", type=int, default=16)
    parser.add_argument("--max-context-batch", type=int, default=8)
    parser.add_argument("--max-counterfactual-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_diagnostic(
        args.checkpoint, args.snapshot, args.n_block, args.n_embd, args.n_head,
        args.num_histories, args.max_context_batch,
        args.max_counterfactual_batch, args.seed)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return report


if __name__ == "__main__":
    main()
