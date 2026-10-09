"""Compact V1 pairwise *precedence* prototype, not causal discovery.

The module is intentionally outside MAT's trainable actor/critic. Do not use
its untrained scores to control PPO; rollout orders must first be stored and
the objective for learning precedence must be justified.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PairwisePrecedenceScorer(nn.Module):
    """Low-rank, directed precedence preference from per-agent features.

    For h_i in R^D, q_i=W_q h_i+b_q, k_i=W_k h_i+b_k in R^r,
        s_ij = q_i^T k_j / sqrt(r)
        P(i before j) = sigmoid(s_ij - s_ji).

    Computing one BMM + transpose avoids materializing [B,N,N,3D].
    P_ij + P_ji = 1 and P_ii = 0.5. A preference is not an
    interventionally established causal dependency.
    """

    def __init__(self, obs_dim, hidden_dim=16):
        super().__init__()
        if obs_dim <= 0 or hidden_dim <= 0:
            raise ValueError("obs_dim and hidden_dim must be positive")
        self.obs_dim = obs_dim
        self.rank = hidden_dim
        self.query = nn.Linear(obs_dim, hidden_dim)
        self.key = nn.Linear(obs_dim, hidden_dim)

    def forward(self, obs):
        """Return [B,N,N] directed preferences using O(B*N^2*r) scores."""
        if obs.ndim != 3 or obs.shape[-1] != self.obs_dim or obs.shape[1] == 0:
            raise ValueError("expected obs [B,N,obs_dim] with N > 0")
        q = self.query(obs)
        k = self.key(obs)
        scores = torch.bmm(q, k.transpose(1, 2))
        margins = (scores - scores.transpose(1, 2)) / math.sqrt(self.rank)
        return torch.sigmoid(margins)

    def order(self, obs):
        """Stable Borda ranking; equal preferences preserve original ID.

        This is an exact permutation, not a differentiable sampling layer.
        Do not recompute it during PPO optimization with changed weights.
        """
        with torch.no_grad():
            return self._order_from_preferences(self.forward(obs))

    @staticmethod
    def _order_from_preferences(preferences):
        return torch.argsort(preferences.sum(-1), dim=-1,
                             descending=True, stable=True)

    def precedence_dag(self, obs, threshold=0.7):
        """Return (adjacency, order), where adjacency[b,i,j] means i -> j.

        Retain only confident edges consistent with the Borda ordering.
        This guarantees a DAG even with cyclic pairwise preferences,
        and permits later topological scheduling *after* correctness
        of parallel conditional action generation is established.
        """
        if not 0.5 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0.5 and 1")
        with torch.no_grad():
            p = self.forward(obs)
            order = self._order_from_preferences(p)
            n = order.shape[1]
            ranks = torch.empty_like(order)
            ranks.scatter_(1, order, torch.arange(
                n, device=order.device).expand_as(order))
            forward_in_rank = ranks.unsqueeze(2) < ranks.unsqueeze(1)
            return (p > threshold) & forward_in_rank, order

    def supervised_loss(self, obs, preference_labels, valid_pairs=None):
        """Masked BCE for externally justified pairwise precedence labels.

        No labels are derived from attention or observations here. This
        function alone does not provide a supervised learning signal.
        """
        p = self.forward(obs)
        if preference_labels.shape != p.shape:
            raise ValueError("preference_labels must have shape [B,N,N]")
        valid = (~torch.eye(p.shape[1], device=p.device, dtype=torch.bool)
                 .unsqueeze(0).expand_as(p))
        if valid_pairs is not None:
            if valid_pairs.shape != p.shape:
                raise ValueError("valid_pairs must have shape [B,N,N]")
            valid = valid & valid_pairs.to(device=p.device, dtype=torch.bool)
        if not torch.any(valid):
            raise ValueError("no valid off-diagonal agent pairs")
        labels = preference_labels.to(device=p.device, dtype=p.dtype)
        return F.binary_cross_entropy(p[valid], labels[valid])
