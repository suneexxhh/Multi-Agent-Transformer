"""Offline expectation of fixed-decoder action sensitivity across histories.

This module does NOT run during SMAC rollouts, optimize PPO, or infer
environment causality. The reported expectation is valid for the supplied
history weighting distribution ONLY.
"""
import torch

from research.dependency_decoder.action_influence_probe import (
    legal_action_kl_probe,
)


@torch.no_grad()
def probe_action_histories(decoder, obs_rep, obs, action_histories,
                           available_actions, agent_order=None,
                           max_context_batch=16, max_counterfactual_batch=128,
                           return_tv=False):
    """Apply the existing KL probe to H histories, in bounded batches.

    obs_rep [B,N,D], obs [B,N,O], action_histories [H,B,N,1].
    available_actions [B,N,A] or [H,B,N,A]; agent_order [B,N] or
    [H,B,N]. Return scores and identifiable mask [H,B,N,N].

    The user must obtain histories from a specified distribution,
    e.g. an explicitly known synthetic distribution or frozen policy
    with recorded sampling probabilities. This function does not
    construct or pretend to know that distribution.
    """
    if obs_rep.ndim != 3 or obs.ndim != 3 or obs_rep.shape[:2] != obs.shape[:2]:
        raise ValueError("features/observations require [B,N,D] and [B,N,O]")
    if (action_histories.ndim != 4 or action_histories.shape[1:3] != obs.shape[:2]
            or action_histories.shape[-1] != 1 or action_histories.shape[0] == 0):
        raise ValueError("action_histories must be [H,B,N,1] with H > 0")
    h, b, n, _ = action_histories.shape
    shared_legal = available_actions.ndim == 3
    if shared_legal:
        if available_actions.shape[:2] != (b, n):
            raise ValueError("shared available_actions require [B,N,A]")
    elif available_actions.ndim != 4 or available_actions.shape[:3] != (h, b, n):
        raise ValueError("available_actions must be [H,B,N,A] or [B,N,A]")
    shared_order = agent_order is not None and agent_order.ndim == 2
    if agent_order is not None:
        if shared_order and agent_order.shape != (b, n):
            raise ValueError("shared agent_order requires [B,N]")
        if not shared_order and agent_order.shape != (h, b, n):
            raise ValueError("agent_order must be [B,N] or [H,B,N]")
    if not isinstance(max_context_batch, int) or max_context_batch < 1:
        raise ValueError("max_context_batch must be positive")
    flat = h * b
    action_flat = action_histories.reshape(flat, n, 1)
    # Do not expand and materialize [H,B,N,D] encoder features.
    # Only a bounded slice of repeated contexts enters each probe.
    scores, masks = [], []
    tv_all = [] if return_tv else None
    for start in range(0, flat, max_context_batch):
        stop = min(start + max_context_batch, flat)
        originals = torch.arange(start, stop, device=obs_rep.device) % b
        rep_chunk = obs_rep.index_select(0, originals)
        obs_chunk = obs.index_select(0, originals)
        legal_chunk = (available_actions.index_select(0, originals)
                       if shared_legal else
                       available_actions.reshape(flat, n, -1)[start:stop])
        if agent_order is None:
            order_chunk = None
        elif shared_order:
            order_chunk = agent_order.index_select(0, originals)
        else:
            order_chunk = agent_order.reshape(flat, n)[start:stop]
        result = legal_action_kl_probe(
            decoder, rep_chunk, obs_chunk, action_flat[start:stop], legal_chunk,
            agent_order=order_chunk, return_valid=True,
            max_counterfactual_batch=max_counterfactual_batch,
            return_tv=return_tv,
        )
        s, mask = result[:2]
        scores.append(s)
        masks.append(mask)
        if return_tv:
            tv_all.append(result[2])
    outputs = (torch.cat(scores, dim=0).reshape(h, b, n, n),
               torch.cat(masks, dim=0).reshape(h, b, n, n))
    if return_tv:
        return outputs + (torch.cat(tv_all, dim=0).reshape(h, b, n, n),)
    return outputs


@torch.no_grad()
def expected_dependency(scores, identifiable, history_weights=None):
    """Coverage-aware conditional mean over a supplied history distribution.

    scores and identifiable are [H,B,N,N]. history_weights are
    nonnegative finite [H,B] relative weights, or None for equal weights.

    For pair (j,i), mass=sum_h w_h * valid_hji / sum_h w_h.
    expectation=sum_h w_h * valid_hji * KL_hji / sum_h w_h*valid_hji.
    ESS=(sum valid*w)^2 / sum valid*w^2 (Kish; 0 if unmeasured).

    Returns (conditional_mean, coverage_mass, effective_sample_size).
    A zero mean when mass is zero is a placeholder, NOT measured zero.
    A low coverage mass can make even a high conditional mean unreliable.
    """
    if (scores.ndim != 4 or scores.shape != identifiable.shape or
            scores.shape[-1] != scores.shape[-2] or
            identifiable.dtype != torch.bool or
            not torch.isfinite(scores).all() or (scores < 0).any()):
        raise ValueError("scores must be finite nonnegative [H,B,N,N], with bool mask")
    h, b, _, _ = scores.shape
    if history_weights is None:
        history_weights = scores.new_ones((h, b))
    else:
        if (history_weights.shape != (h, b) or
                not torch.isfinite(history_weights).all() or
                (history_weights < 0).any()):
            raise ValueError("history_weights must be nonnegative finite [H,B]")
        history_weights = history_weights.to(device=scores.device, dtype=scores.dtype)
    normalizer = history_weights.sum(0)
    if (normalizer <= 0).any():
        raise ValueError("every batch item must have positive total history weight")
    weighted = history_weights[:, :, None, None] * identifiable
    measured = weighted.sum(0)
    mean = (weighted * scores).sum(0) / measured.clamp_min(torch.finfo(scores.dtype).tiny)
    coverage = measured / normalizer[:, None, None]
    ess = measured.square() / weighted.square().sum(0).clamp_min(
        torch.finfo(scores.dtype).tiny)
    return mean, coverage, ess


@torch.no_grad()
def reliable_precedence_targets(mean, coverage, ess, margin=0.05,
                                min_coverage=0.5, min_ess=2.0):
    """Conservative offline proxy labels, never environment causal truth.

    Both j->i and i->j must have sufficient evidence from distinct
    ordering contexts. The change of conditional factorization across
    orders remains a confound; do NOT enable training on SMAC from this.
    """
    if (mean.ndim != 3 or mean.shape != coverage.shape or
            mean.shape != ess.shape or mean.shape[-1] != mean.shape[-2]):
        raise ValueError("expected matching [B,N,N] statistics")
    if (not torch.isfinite(mean).all() or not torch.isfinite(coverage).all() or
            not torch.isfinite(ess).all() or (mean < 0).any() or
            (coverage < 0).any() or (coverage > 1).any() or (ess < 0).any()):
        raise ValueError("invalid expectation, coverage or ESS values")
    if not (0 <= margin < float("inf") and 0 < min_coverage <= 1 and
            min_ess >= 1):
        raise ValueError("invalid thresholds")
    n = mean.shape[-1]
    enough = (coverage >= min_coverage) & (ess >= min_ess)
    bidirectional = enough & enough.transpose(-1, -2)
    offdiag = ~torch.eye(n, device=mean.device, dtype=torch.bool)
    delta = mean - mean.transpose(-1, -2)
    mask = bidirectional & (delta.abs() > margin) & offdiag
    return (delta > 0).to(mean.dtype), mask
