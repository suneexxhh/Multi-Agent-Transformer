"""Offline sampling and density ratios for a frozen MAT discrete Decoder.

Keeps the original MAT sampling/teacher-forcing kernels and fixed agent
permutation. This never runs inside PPO or starts an environment.
Conditional action probabilities are for the specified observations,
encoder representation, legal masks and *same* decoder order.
"""
import torch

from mat.algorithms.mat.algorithm.agent_ordering import validate_agent_order
from mat.algorithms.utils.transformer_act import (
    discrete_autoregreesive_ordered_act, discrete_parallel_ordered_act,
)


def _validate_context(decoder, rep, obs, available, order, batch_limit):
    if decoder.training:
        raise ValueError("Freeze decoder and call eval() before sampling/scoring")
    if not isinstance(rep, torch.Tensor) or not isinstance(obs, torch.Tensor):
        raise ValueError("encoder representations and observations must be tensors")
    if rep.ndim != 3 or obs.ndim != 3 or rep.shape[:2] != obs.shape[:2]:
        raise ValueError("rep and obs must have matching [B,N,...] dimensions")
    b, n, _ = rep.shape
    if b < 1 or n < 1 or rep.device != obs.device or not rep.is_floating_point():
        raise ValueError("require nonempty floating encoder features on obs device")
    if (not isinstance(available, torch.Tensor) or available.ndim != 3 or
            available.shape[:2] != (b, n) or available.device != rep.device or
            available.shape[-1] < 2 or not available.bool().any(-1).all()):
        raise ValueError("each agent needs a nonempty legal discrete-action mask")
    if not isinstance(batch_limit, int) or isinstance(batch_limit, bool) or batch_limit < 1:
        raise ValueError("max_context_batch must be a positive integer")
    return validate_agent_order(order, b, n, rep.device)


@torch.no_grad()
def sample_frozen_decoder_histories(decoder, obs_rep, obs, available_actions,
                                    agent_order, num_histories,
                                    max_context_batch=32):
    """Sample H iid histories from a fixed conditional MAT action policy.

    Returns (actions[H,B,N,1], per_agent_logq[H,B,N,1],
    joint_logq[H,B]). Action IDs and log-probs are in ORIGINAL agent
    order. Joint logq = sum over agent conditional log-probabilities,
    NOT a probability distribution over the sampled set of histories.

    The specified order must remain fixed across these H samples.
    Does not modify decoder parameters or allocate [H,B,N,D] features.
    """
    order = _validate_context(
        decoder, obs_rep, obs, available_actions, agent_order, max_context_batch)
    if (not isinstance(num_histories, int) or isinstance(num_histories, bool) or
            num_histories < 1):
        raise ValueError("num_histories must be a positive integer")
    b, n, _ = obs_rep.shape
    act_dim = available_actions.shape[-1]
    tpdv = dict(dtype=obs_rep.dtype, device=obs_rep.device)
    all_actions, all_logp = [], []
    for start in range(0, b * num_histories, max_context_batch):
        stop = min(start + max_context_batch, b * num_histories)
        idx = torch.arange(start, stop, device=obs_rep.device) % b
        actions, logp = discrete_autoregreesive_ordered_act(
            decoder, obs_rep.index_select(0, idx),
            obs.index_select(0, idx), len(idx), n, act_dim, tpdv,
            order.index_select(0, idx),
            available_actions.index_select(0, idx), deterministic=False)
        all_actions.append(actions)
        all_logp.append(logp)
    actions = torch.cat(all_actions).reshape(num_histories, b, n, 1)
    agent_logp = torch.cat(all_logp).reshape(num_histories, b, n, 1)
    return actions, agent_logp, agent_logp.sum(dim=2).squeeze(-1)


@torch.no_grad()
def teacher_forced_history_log_probs(decoder, obs_rep, obs, available_actions,
                                     agent_order, actions,
                                     max_context_batch=32):
    """Evaluate the EXACT same saved action histories under a fixed decoder.

    Uses MAT's causally shifted, parallel teacher-forced likelihood,
    with identical agent permutation and action masks to sampling.
    An alternate decoder can be passed for policy-shift analysis,
    but its own encoder representations must be supplied separately.
    """
    order = _validate_context(
        decoder, obs_rep, obs, available_actions, agent_order, max_context_batch)
    b, n, _ = obs_rep.shape
    if (not isinstance(actions, torch.Tensor) or actions.ndim != 4 or
            actions.shape[1:] != (b, n, 1) or actions.shape[0] < 1 or
            actions.dtype != torch.long or actions.device != obs_rep.device):
        raise ValueError("actions must be int64 [H,B,N,1] on decoder device")
    legal = available_actions.bool()
    if (torch.any(actions < 0) or torch.any(actions >= available_actions.shape[-1]) or
            not legal.unsqueeze(0).expand(actions.shape[0], -1, -1, -1)
                     .gather(-1, actions).all()):
        raise ValueError("every stored action must be legal in this fixed mask")
    h = actions.shape[0]
    tpdv = dict(dtype=obs_rep.dtype, device=obs_rep.device)
    flat_actions = actions.reshape(h * b, n, 1)
    parts = []
    for start in range(0, h * b, max_context_batch):
        stop = min(start + max_context_batch, h * b)
        idx = torch.arange(start, stop, device=obs_rep.device) % b
        lp, _ = discrete_parallel_ordered_act(
            decoder, obs_rep.index_select(0, idx), obs.index_select(0, idx),
            flat_actions[start:stop], len(idx), n, available_actions.shape[-1],
            tpdv, order.index_select(0, idx),
            available_actions.index_select(0, idx))
        parts.append(lp)
    result = torch.cat(parts).reshape(h, b, n, 1)
    return result, result.sum(2).squeeze(-1)


@torch.no_grad()
def self_normalized_policy_weights(target_joint_logp, behavior_joint_logp,
                                   min_ess=None):
    """SNIS weights for frozen target versus frozen behavior policies.

    Return (weights[H,B], ESS[B]). Both log probabilities must describe
    the SAME saved actions, observations, legal masks and agent orders.
    Requires common action support; the caller must verify it (e.g.
    both decoders use the identical discrete legal-action masks).
    Finite-H self-normalization is biased; reliability deteriorates
    sharply with small ESS. Optional min_ess rejects unusable estimates
    before they reach precedence-label or graph fitting code.
    """
    if (not isinstance(target_joint_logp, torch.Tensor) or
            not isinstance(behavior_joint_logp, torch.Tensor) or
            target_joint_logp.ndim != 2 or
            target_joint_logp.shape != behavior_joint_logp.shape or
            target_joint_logp.shape[0] == 0 or
            not torch.isfinite(target_joint_logp).all() or
            not torch.isfinite(behavior_joint_logp).all()):
        raise ValueError("joint log-probabilities must be finite matching [H,B]")
    log_ratio = target_joint_logp - behavior_joint_logp
    weights = torch.softmax(log_ratio, dim=0)
    ess = weights.sum(0).square() / weights.square().sum(0)
    if min_ess is not None:
        threshold = torch.as_tensor(min_ess, dtype=ess.dtype, device=ess.device)
        if (threshold.numel() != 1 or not torch.isfinite(threshold).all() or
                threshold.item() < 1):
            raise ValueError("min_ess must be finite and >= 1")
        if torch.any(ess < threshold):
            raise ValueError("policy-shift ESS below min_ess; obtain better-support samples")
    return weights, ess
