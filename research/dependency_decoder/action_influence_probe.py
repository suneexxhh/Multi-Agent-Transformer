"""Offline MAT decoder conditional-action-sensitivity diagnostics.

These scores measure changes in a *fixed policy*, not causal influence on
environment dynamics. They are NEVER invoked by PPO or SMAC training.
"""
import torch
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.agent_ordering import (
    reorder_agents, validate_agent_order,
)


@torch.no_grad()
def legal_action_kl_probe(decoder, obs_rep, obs, actions,
                          available_actions=None, agent_order=None,
                          return_valid=False, max_counterfactual_batch=256):
    """Mean intervention KL from each predecessor to each successor.

    Output scores[b,j,i] averages KL across *all* legal alternative
    actions for predecessor j, excluding its observed action. The
    Boolean validity mask disambiguates an unidentifiable pair from an
    observed sensitivity of exactly zero. Only j preceding i under
    the given order can be probed. Scores and masks use ORIGINAL IDs.

    One baseline forward pass and ceil(K/chunk) vectorized counterfactual
    passes, K = number of valid (batch, predecessor, legal alternative)
    interventions. Chunking bounds peak activation memory. The loop
    iterates over chunks, NOT individual agents or pairs.

    This is an average over available alternative *action IDs* conditional
    on a fixed context, not over a policy distribution or action histories.
    """
    if decoder.training:
        raise ValueError("decoder.eval() required for reproducible probe")
    if obs_rep.ndim != 3 or obs.ndim != 3 or actions.ndim != 3:
        raise ValueError("expected [B,N,D], [B,N,O], [B,N,1]")
    b, n, _ = obs_rep.shape
    if obs.shape[:2] != (b, n) or actions.shape != (b, n, 1):
        raise ValueError("batch/agent/action shapes are incompatible")
    if actions.dtype != torch.long or actions.device != obs_rep.device:
        raise ValueError("actions must be torch.long on decoder device")
    if obs.device != obs_rep.device:
        raise ValueError("observation and encoder features must share device")
    if available_actions is None or available_actions.ndim != 3:
        raise ValueError("explicit legal-action masks [B,N,A] are required")
    if available_actions.shape[:2] != (b, n):
        raise ValueError("available_actions batch/agent shapes are incompatible")
    a = available_actions.shape[-1]
    if a < 2 or available_actions.device != obs_rep.device:
        raise ValueError("at least two actions and matching device required")
    if (not isinstance(max_counterfactual_batch, int) or
            max_counterfactual_batch < 1):
        raise ValueError("max_counterfactual_batch must be a positive integer")
    if torch.any(actions < 0) or torch.any(actions >= a):
        raise ValueError("out-of-range observed action")
    legal = available_actions.bool()
    if not torch.all(legal.any(-1)):
        raise ValueError("each agent requires a legal action")
    if not torch.all(legal.gather(-1, actions).squeeze(-1)):
        raise ValueError("observed actions must be legal")

    if agent_order is not None:
        order = validate_agent_order(agent_order, b, n, obs.device)
        obs_rep = reorder_agents(obs_rep, order)
        obs = reorder_agents(obs, order)
        actions = reorder_agents(actions, order)
        legal = reorder_agents(legal, order)

    scores = obs_rep.new_zeros((b, n, n))
    valid = torch.zeros((b, n, n), dtype=torch.bool, device=obs.device)
    if n > 1:
        candidate = legal[:, :-1] & ~F.one_hot(
            actions[:, :-1, 0], a).bool()
        counts = candidate.sum(-1)  # [B,N-1]
        later = (torch.arange(n, device=obs.device)[None, None, :] >
                 torch.arange(n - 1, device=obs.device)[None, :, None])
        # A successor with only one legal action has a point-mass policy:
        # no action intervention can change its legal distribution.
        # Such a pair is UNIDENTIFIABLE rather than measured-zero.
        target_variable = legal.sum(-1) > 1
        valid[:, :-1] = ((counts.unsqueeze(-1) > 0) & later &
                         target_variable.unsqueeze(1))
        # Do not evaluate perturbations that cannot affect any
        # *measurable* successor in the current ordering/context.
        candidate = candidate & valid[:, :-1].any(-1).unsqueeze(-1)
        if torch.any(candidate):
            shift = obs_rep.new_zeros((b, n, a + 1))
            shift[:, 0, 0] = 1
            shift[:, 1:, 1:] = F.one_hot(
                actions[:, :-1, 0], a).to(shift.dtype)
            ref_logits = decoder(shift, obs_rep, obs)
            if ref_logits.shape != (b, n, a):
                raise ValueError("decoder must output [B,N,A] logits")
            # Only the KL reduction uses float64. The decoder itself
            # remains in its original dtype/architecture. This prevents
            # cancellation when true conditional KL is far below the
            # float32 log-softmax precision (e.g. 1e-9).
            reference_log = F.log_softmax(
                ref_logits.double().masked_fill(~legal, float("-inf")), -1)

            # Extract only legal alternatives. A single scatter-add over
            # the original [B,N,N] pair tensor accumulates their KL.
            bi, ji, ai = candidate.nonzero(as_tuple=True)
            k = bi.numel()
            totals = scores.new_zeros((b, n, n))
            all_targets = torch.arange(n, device=obs.device)
            for start in range(0, k, max_counterfactual_batch):
                stop = min(start + max_counterfactual_batch, k)
                batch_idx = bi[start:stop]
                source_idx = ji[start:stop]
                alt_idx = ai[start:stop]
                size = batch_idx.shape[0]
                cf_shift = shift[batch_idx].clone()
                cf_shift[torch.arange(size, device=obs.device),
                         source_idx + 1, 1:] = F.one_hot(
                             alt_idx, a).to(cf_shift.dtype)
                cf_logits = decoder(
                    cf_shift, obs_rep[batch_idx], obs[batch_idx])
                if cf_logits.shape != (size, n, a):
                    raise ValueError("decoder counterfactual output shape mismatch")
                allowed = legal[batch_idx]
                cf_log = F.log_softmax(
                    cf_logits.double().masked_fill(~allowed, float("-inf")), -1)
                ref = reference_log[batch_idx]
                # 0 * (-inf - -inf) is NaN; illegal actions contribute
                # exactly zero and are excluded BEFORE summation.
                delta = torch.where(
                    allowed, ref - cf_log, torch.zeros_like(cf_log))
                kl = (ref.exp() * delta).sum(-1).clamp_min(0).to(scores.dtype)
                successor_ids = all_targets.expand(size, n)
                successor_valid = successor_ids > source_idx.unsqueeze(1)
                batch_out = batch_idx.unsqueeze(1).expand(size, n)
                source_out = source_idx.unsqueeze(1).expand(size, n)
                totals.index_put_(
                    (batch_out[successor_valid],
                     source_out[successor_valid],
                     successor_ids[successor_valid]),
                    kl[successor_valid], accumulate=True)
            scores[:, :-1] = totals[:, :-1] / counts.unsqueeze(-1).clamp_min(1)

    if agent_order is not None:
        inverse = order.argsort(1)
        gather_src = inverse.unsqueeze(-1).expand(b, n, n)
        gather_dst = inverse.unsqueeze(1).expand(b, n, n)
        scores = scores.gather(1, gather_src).gather(2, gather_dst)
        valid = valid.gather(1, gather_src).gather(2, gather_dst)
    return (scores, valid) if return_valid else scores


def combine_order_probes(scores, valid):
    """Combine R different ordering-context probes in original ID space.

    Scores and validity are [R,B,N,N]; only measured pairs contribute.
    Returns (measured_average, counts). Zero counts mean UNKNOWN, not
    absent dependency. Comparability across different order contexts
    is a separate empirical assumption to validate.
    """
    if (scores.ndim != 4 or scores.shape != valid.shape or
            scores.shape[-1] != scores.shape[-2] or
            valid.dtype != torch.bool):
        raise ValueError("scores and Boolean valid must be [R,B,N,N]")
    counts = valid.sum(0)
    average = (scores * valid.to(scores.dtype)).sum(0) / counts.clamp_min(1)
    return average, counts


def conservative_precedence_targets(scores, counts, margin=0.01):
    """Make optional *proxy* preference labels only for bidirectionally
    measured pairs with a sufficient difference between their KL scores.

    Arguments [B,N,N] must be from combine_order_probes. Returns
    (labels, trainable_mask) for PairwisePrecedenceScorer.supervised_loss.
    This is NOT environment-causal ground truth. Since decoder
    factorization differs by ordering, cross-order sensitivity magnitudes
    require independent validation before using on real SMAC data.
    """
    if (scores.ndim != 3 or scores.shape != counts.shape or
            scores.shape[1] != scores.shape[2]):
        raise ValueError("expected scores and counts [B,N,N]")
    if not 0 <= margin < float("inf"):
        raise ValueError("margin must be nonnegative and finite")
    delta = scores - scores.transpose(-1, -2)
    pair_observed = (counts > 0) & (counts.transpose(-1, -2) > 0)
    n = scores.shape[-1]
    diagonal_off = ~torch.eye(n, device=scores.device, dtype=torch.bool)
    trainable = pair_observed & (delta.abs() > margin) & diagonal_off
    return (delta > 0).to(scores.dtype), trainable
