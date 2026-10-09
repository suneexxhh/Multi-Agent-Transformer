"""Controlled nonlinear action-history oracle for offline MAT decoder analysis.

Contains no optimization, PPO integration, GPU launch, or SMAC imports.
It exposes a known conjunction / XOR of two distinct predecessor actions.
"""
import torch
from torch import nn


class TwoParentInteractionDecoder(nn.Module):
    """Toy decoder with a known context-dependent (0,1)->2 motif.

    Obs representation is [B,N,N] original-agent one-hot IDs, reordered
    alongside actions. A predecessor's action==2 activates its flag.
    For AND, agent 2's action-1 logit changes ONLY if both 0 and 1 flags
    are active. XOR is a second distinct nonlinear dependency mechanism.

    All conditional logits depend only on previously decoded actions,
    so this implements the same causal-shift convention as MAT.
    """

    def __init__(self, mode="and", strength=5.0):
        super().__init__()
        if mode not in ("and", "xor"):
            raise ValueError("mode must be 'and' or 'xor'")
        self.mode = mode
        self.strength = strength
        self.forward_calls = []

    def forward(self, shifted, obs_rep, obs):
        b, n, a_plus_one = shifted.shape
        if a_plus_one != 4 or n < 3 or obs_rep.shape != (b, n, n):
            raise ValueError("expected 3 actions and original-ID one-hot rep [B,N,N]")
        self.forward_calls.append(b)
        previous_original_id = torch.nn.functional.pad(
            obs_rep[:, :-1], (0, 0, 1, 0))
        action_two = shifted[..., 3:4]
        active_before = (previous_original_id * action_two).cumsum(dim=1)
        flag0, flag1 = active_before[..., 0], active_before[..., 1]
        interaction = (flag0 * flag1 if self.mode == "and"
                       else (flag0 - flag1).abs())
        is_target = obs_rep[..., 2]
        logits = shifted.new_zeros((b, n, 3))
        logits[..., 1] = self.strength * interaction * is_target
        return logits


@torch.no_grad()
def max_sensitivity_across_histories(scores, validity):
    """Maximum measurable KL per directed pair across action histories.

    scores/validity are [H,B,N,N], where each H changes baseline
    *actions* while keeping agent identities, decoder and observation
    representation fixed. Unobservable pairs are not counted as zeros.
    This is an exploratory diagnostic: cherry-picked histories can
    inflate detection and must never become causal ground truth.
    """
    if (scores.ndim != 4 or scores.shape != validity.shape or
            validity.dtype != torch.bool or scores.shape[-1] != scores.shape[-2]):
        raise ValueError("expected scores and bool validity [H,B,N,N]")
    counts = validity.sum(dim=0)
    masked = scores.masked_fill(~validity, float("-inf"))
    maximum = masked.max(dim=0).values
    maximum = maximum.masked_fill(counts == 0, 0.0)
    return maximum, counts
