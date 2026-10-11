"""Optional on-policy, frozen-policy conditional-action influence teacher.

This measures a policy's response to legal prior-action substitutions. It is
not an intervention on SMAC dynamics and provides no causal identification.
Inputs come only from PPO rollout minibatches, never evaluation episodes.
"""
import copy
import math

import torch
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.agent_ordering import reorder_agents, validate_agent_order


def frozen_bias_off_teacher(transformer):
    """Snapshot before PPO updates; no student/teacher parameter sharing."""
    if transformer.action_type != "Discrete" or transformer.order_policy is None:
        raise ValueError("influence teacher requires a discrete learned-order MAT")
    teacher = copy.deepcopy(transformer).eval()
    teacher.order_attention_bias = False
    teacher.requires_grad_(False)
    return teacher


def categorical_js(ref_logits, changed_logits, legal):
    """Finite JS in nats, with illegal actions excluded before reduction."""
    if ref_logits.shape != changed_logits.shape or legal.shape != ref_logits.shape:
        raise ValueError("JS logits and legal mask shapes differ")
    legal = legal.bool()
    if not legal.any(-1).all():
        raise ValueError("target has no legal action")
    p_log = F.log_softmax(ref_logits.double().masked_fill(~legal, -torch.inf), -1)
    q_log = F.log_softmax(changed_logits.double().masked_fill(~legal, -torch.inf), -1)
    p, q = p_log.exp(), q_log.exp()
    m_log = torch.log((p + q).clamp_min(torch.finfo(torch.float64).tiny)) - math.log(2.)
    term_p = torch.where(legal, p * (p_log - m_log), torch.zeros_like(p))
    term_q = torch.where(legal, q * (q_log - m_log), torch.zeros_like(q))
    js = (term_p + term_q).sum(-1) * 0.5
    if not torch.isfinite(js).all():
        raise ValueError("non-finite conditional action JS")
    return js.clamp(0., math.log(2.)).to(ref_logits.dtype)


@torch.no_grad()
def conditional_action_js(teacher, obs, actions, available_actions, order,
                          max_alternatives=2):
    """Return JS[target_slot,source_slot], validity, target entropy, alt count.

    All tensors are in decoded-position coordinates. A source has to precede
    the target; unavailable directions remain invalid, never measured-zero.
    Only legal alternatives to the rollout action are used. Other history,
    observations, order and the frozen Bias-OFF network remain fixed.
    """
    if teacher.training or any(p.requires_grad for p in teacher.parameters()) or teacher.order_attention_bias:
        raise ValueError("teacher must be frozen, eval and Bias OFF")
    if not isinstance(max_alternatives, int) or max_alternatives < 1:
        raise ValueError("max_alternatives must be positive")
    obs = torch.as_tensor(obs, dtype=torch.float32, device=next(teacher.parameters()).device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=obs.device)
    legal = torch.as_tensor(available_actions, device=obs.device).bool()
    if obs.ndim != 3 or actions.shape != (*obs.shape[:2], 1) or legal.shape[:2] != obs.shape[:2]:
        raise ValueError("expected observations [B,N,O], actions [B,N,1], legal [B,N,A]")
    b, n, _ = obs.shape
    a = legal.shape[-1]
    if a < 2 or not legal.any(-1).all() or (actions < 0).any() or (actions >= a).any():
        raise ValueError("invalid legal-action inputs")
    if not legal.gather(-1, actions).all():
        raise ValueError("rollout source action is illegal")
    order = validate_agent_order(order, b, n, obs.device)
    obs, actions, legal = (reorder_agents(t, order) for t in (obs, actions, legal))
    state = obs.new_zeros((b, n, 37))
    _, rep = teacher.encoder(state, obs)
    shifted = obs.new_zeros((b, n, a + 1))
    shifted[:, 0, 0] = 1.
    shifted[:, 1:, 1:] = F.one_hot(actions[:, :-1, 0], a).to(obs.dtype)
    base = teacher.decoder(shifted, rep, obs)
    if base.shape != (b, n, a):
        raise ValueError("teacher decoder output shape mismatch")
    base_log = F.log_softmax(base.double().masked_fill(~legal, -torch.inf), -1)
    entropy = -(base_log.exp() * torch.where(legal, base_log, 0.)).sum(-1).to(obs.dtype)
    alternatives = legal & ~F.one_hot(actions[..., 0], a).bool()
    alt_count = alternatives.sum(-1).clamp(max=max_alternatives)
    js_sum = obs.new_zeros((b, n, n))
    counts = obs.new_zeros((b, n, n))
    # Up to N*K vectorized decoder calls. No per-target decoder call.
    for source in range(n - 1):
        candidate_ids = torch.arange(a, device=obs.device).expand(b, a)
        candidate_ids = candidate_ids.masked_fill(~alternatives[:, source], a).sort(-1).values
        number = alternatives[:, source].sum(-1)
        selected_count = number.clamp(max=max_alternatives)
        for rank in range(max_alternatives):
            # Deterministically span the legal alternative IDs instead of
            # systematically taking only the lowest action IDs.
            idx = torch.where(
                selected_count > 1,
                torch.round(rank * (number - 1).float() /
                            (selected_count - 1).clamp_min(1).float()).long(),
                (number - 1).clamp_min(0) // 2).clamp(max=a - 1)
            chosen = candidate_ids.gather(1, idx[:, None]).squeeze(1)
            active = (selected_count > rank) & (chosen < a)
            if not active.any():
                continue
            rows = active.nonzero(as_tuple=True)[0]
            cf_shift = shifted[rows].clone()
            cf_shift[:, source + 1, 1:] = F.one_hot(chosen[rows], a).to(obs.dtype)
            cf_logits = teacher.decoder(cf_shift, rep[rows], obs[rows])
            js = categorical_js(base[rows], cf_logits, legal[rows])
            valid_target = legal[rows, source + 1:].sum(-1) > 1
            js_sum[rows, source + 1:, source] += js[:, source + 1:] * valid_target
            counts[rows, source + 1:, source] += valid_target.to(counts.dtype)
    valid = counts > 0
    return js_sum / counts.clamp_min(1), valid, entropy, alt_count


def alignment_targets(js, valid, entropy, alt_count, legal_counts,
                      max_alternatives=2, entropy_bins=4, effect_scale=0.03,
                      min_effect=1e-4):
    """Match source/target slots and target-entropy bins before labeling.

    Center each observed JS against its stratum mean. Thus the target asks
    whether this pair is more sensitive than other rollout states with the
    SAME ordered positions and similar target entropy. Singleton strata and
    effects below the resolution floor receive zero confidence.
    """
    if js.ndim != 3 or js.shape != valid.shape or js.shape[1] != js.shape[2]:
        raise ValueError("JS/valid must be [B,N,N]")
    b, n, _ = js.shape
    if entropy.shape != (b, n) or alt_count.shape != (b, n) or legal_counts.shape != (b, n):
        raise ValueError("entropy/action-count shapes differ")
    if max_alternatives < 1 or entropy_bins < 1 or effect_scale <= 0 or min_effect < 0:
        raise ValueError("invalid alignment hyperparameters")
    if not torch.isfinite(js[valid]).all() or not torch.isfinite(entropy).all():
        raise ValueError("non-finite teacher statistics")
    normalized_h = (entropy / legal_counts.clamp_min(2).to(entropy.dtype).log()).clamp(0., 1.)
    bins = (normalized_h * entropy_bins).long().clamp(max=entropy_bins - 1)
    targets = torch.full_like(js, 0.5)
    confidence = torch.zeros_like(js)
    for source in range(n - 1):
        for target in range(source + 1, n):
            for bin_id in range(entropy_bins):
                matched = valid[:, target, source] & (bins[:, target] == bin_id)
                if matched.sum() < 2:
                    continue
                effects = js[matched, target, source]
                residual = effects - effects.mean()
                targets[matched, target, source] = 0.5 + 0.5 * torch.tanh(residual / effect_scale)
                strength = torch.where(
                    residual.abs() >= min_effect,
                    (residual.abs() / effect_scale).clamp(max=1.),
                    torch.zeros_like(residual))
                coverage = (alt_count[matched, source].to(js.dtype) /
                            (legal_counts[matched, source] - 1).clamp_min(1).to(js.dtype)).clamp(max=1.)
                support = min(1., matched.sum().item() / 8.)
                confidence[matched, target, source] = (
                    strength * coverage * normalized_h[matched, target] * support)
    return targets.detach(), confidence.detach()


def influence_alignment_loss(student_order_policy, obs, order, targets, confidence):
    """Weighted BCE on non-antisymmetric D[target,source], original IDs."""
    logits = student_order_policy.demand_logits(obs)
    b, n, _ = logits.shape
    order = validate_agent_order(order, b, n, logits.device)
    ordered = logits.gather(1, order.unsqueeze(-1).expand_as(logits))
    ordered = ordered.gather(2, order.unsqueeze(1).expand_as(ordered))
    if targets.shape != ordered.shape or confidence.shape != ordered.shape:
        raise ValueError("alignment target/confidence shapes differ")
    if not torch.isfinite(confidence).all() or (confidence < 0).any():
        raise ValueError("invalid alignment confidence")
    if not (confidence > 0).any():
        return ordered.sum() * 0.
    return (F.binary_cross_entropy_with_logits(ordered, targets, reduction="none") *
            confidence).sum() / confidence.sum()
