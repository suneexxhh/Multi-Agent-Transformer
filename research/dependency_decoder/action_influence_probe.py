"""Eval-only, vectorized decoder-action intervention diagnostic.

This computes a *policy conditional-sensitivity proxy*, not causal
influence in the environment. It never changes MAT's actor, critic, or
PPO loss, and is not invoked in SMAC training.
"""
import torch
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.agent_ordering import (
    reorder_agents, validate_agent_order,
)


@torch.no_grad()
def legal_action_kl_probe(decoder, obs_rep, obs, actions,
                          available_actions=None, agent_order=None):
    """KL sensitivity of later policies to one preceding legal action.

    Returns [B,N,N] where result[b,j,i] is the KL at agent i after
    replacing agent j's teacher-forced action with the *first different*
    legal discrete action (all other past actions held fixed).
    Both agent axes are restored to original IDs when order is passed.

    Exactly TWO decoder forward calls: one baseline and one vectorized
    batch of N-1 action perturbations. Complexity is O(N) forward
    equivalents, not O(N^2) Python decoder calls. Use only offline on
    a fixed, eval-mode decoder; arbitrary alternative-action selection
    does not provide an unbiased influence estimate.
    """
    if decoder.training:
        raise ValueError("decoder.eval() required for reproducible diagnostic")
    if obs_rep.ndim != 3 or obs.ndim != 3 or actions.ndim != 3:
        raise ValueError("expected [B,N,D], [B,N,O], [B,N,1]")
    b, n, _ = obs_rep.shape
    if obs.shape[:2] != (b, n) or actions.shape != (b, n, 1):
        raise ValueError("batch/agent/action shapes are incompatible")
    if actions.dtype != torch.long or actions.device != obs_rep.device:
        raise ValueError("actions must be torch.long on decoder device")
    if obs.device != obs_rep.device:
        raise ValueError("observation and encoder features must share a device")
    if available_actions is None:
        raise ValueError("explicit legal-action masks are required")
    if available_actions.shape[:2] != (b, n) or available_actions.ndim != 3:
        raise ValueError("available_actions shape must be [B,N,A]")
    a = available_actions.shape[-1]
    if a < 2:
        raise ValueError("at least two discrete actions are required")
    if available_actions.device != obs_rep.device:
        raise ValueError("available_actions device must match observations")
    if torch.any(actions < 0) or torch.any(actions >= a):
        raise ValueError("out-of-range action")
    legal = available_actions.bool()
    if not torch.all(legal.any(-1)):
        raise ValueError("each agent must have at least one legal action")
    if not torch.all(legal.gather(-1, actions).squeeze(-1)):
        raise ValueError("observed actions must be legal")

    if agent_order is not None:
        order = validate_agent_order(agent_order, b, n, obs.device)
        obs_rep = reorder_agents(obs_rep, order)
        obs = reorder_agents(obs, order)
        actions = reorder_agents(actions, order)
        legal = reorder_agents(legal, order)

    result = obs_rep.new_zeros((b, n, n))
    if n == 1:
        return result

    shift = obs_rep.new_zeros((b, n, a + 1))
    shift[:, 0, 0] = 1
    shift[:, 1:, 1:] = F.one_hot(actions[:, :-1, 0], a).to(shift.dtype)

    # Only a predecessor's action has any downstream decoder effect.
    candidate = legal & ~F.one_hot(actions[..., 0], a).bool()
    valid_predecessor = candidate.any(-1)[:, :-1]  # [B,N-1]
    if not torch.any(valid_predecessor):
        return result
    alternative = candidate.long().argmax(-1)[:, :-1]  # [B,N-1]
    ref_logits = decoder(shift, obs_rep, obs)  # [B,N,A]

    # Treat all potential j as a batch dimension; exactly ONE decoder
    # call evaluates every one-action intervention.
    cf_shift = shift.unsqueeze(1).expand(b, n - 1, n, a + 1).clone()
    bi = torch.arange(b, device=obs.device)[:, None]
    ji = torch.arange(n - 1, device=obs.device)[None, :]
    cf_shift[bi, ji, ji + 1, 1:] = F.one_hot(
        alternative, a).to(cf_shift.dtype)
    cf_rep = obs_rep.unsqueeze(1).expand(b, n - 1, n, obs_rep.shape[-1])
    cf_obs = obs.unsqueeze(1).expand(b, n - 1, n, obs.shape[-1])
    cf_logits = decoder(
        cf_shift.reshape(b * (n - 1), n, a + 1),
        cf_rep.reshape(b * (n - 1), n, obs_rep.shape[-1]),
        cf_obs.reshape(b * (n - 1), n, obs.shape[-1])
    ).reshape(b, n - 1, n, a)

    ref_log = F.log_softmax(ref_logits.masked_fill(~legal, -1e9), -1)
    cf_log = F.log_softmax(
        cf_logits.masked_fill(~legal.unsqueeze(1), -1e9), -1)
    kl = (ref_log.exp().unsqueeze(1) *
          (ref_log.unsqueeze(1) - cf_log)).sum(-1).clamp_min(0)

    later = (torch.arange(n, device=obs.device)[None, None, :] >
             torch.arange(n - 1, device=obs.device)[None, :, None])
    result[:, :-1] = torch.where(
        later & valid_predecessor.unsqueeze(-1), kl, torch.zeros_like(kl))
    if agent_order is not None:
        # Invert order on BOTH graph axes, not just the source.
        inverse = order.argsort(1)
        result = result.gather(
            1, inverse.unsqueeze(-1).expand(b, n, n)).gather(
            2, inverse.unsqueeze(1).expand(b, n, n))
    return result
