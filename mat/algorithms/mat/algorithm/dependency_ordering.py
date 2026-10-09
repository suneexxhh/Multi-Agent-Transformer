"""V1 research scaffold: learned pairwise precedence, NOT causal discovery.

This module is intentionally isolated from MAT policy and PPO. The scorer
is trainable only when a defensible supervision signal is supplied. An
ordering inferred from a changing scorer must be stored with trajectories
before it can be used in PPO likelihood evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwisePrecedenceScorer(nn.Module):
    """Score a directed precedence relation between current agent observations.

    Returns P[b,i,j] = estimated preference that agent i be decoded before j.
    This is a *precedence preference*, not evidence of causal action influence.

    The pair network is order-sensitive; antisymmetrization ensures
    P(i before j) + P(j before i) == 1 up to floating-point precision.
    """

    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        if obs_dim <= 0 or hidden_dim <= 0:
            raise ValueError("obs_dim and hidden_dim must be positive")
        self.obs_dim = obs_dim
        self.pair_net = nn.Sequential(
            nn.Linear(3 * obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        """Return pairwise preference tensor [B,N,N] on obs.device."""
        if obs.ndim != 3 or obs.shape[-1] != self.obs_dim:
            raise ValueError("expected obs shape [batch, n_agent, obs_dim]")
        if obs.shape[1] == 0:
            raise ValueError("at least one agent is required")
        b, n, d = obs.shape
        left = obs.unsqueeze(2).expand(b, n, n, d)
        right = obs.unsqueeze(1).expand(b, n, n, d)
        pair_input = torch.cat((left, right, left - right), dim=-1)
        raw = self.pair_net(pair_input).squeeze(-1)
        preference = torch.sigmoid(raw - raw.transpose(1, 2))
        return preference

    def order(self, obs):
        """Return deterministic [B,N] original IDs sorted by expected wins.

        Stable sorting resolves ties by original ID. Ranking via pairwise
        expected wins (Borda) also handles cyclic pairwise preferences.
        This discrete operation is not differentiable and must not be
        recomputed from updated network weights during PPO epochs.
        """
        with torch.no_grad():
            preference = self.forward(obs)
            expected_wins = preference.sum(dim=-1)
            return torch.argsort(
                expected_wins, dim=-1, descending=True, stable=True
            )

    def supervised_loss(self, obs, preference_labels, valid_pairs=None):
        """BCE loss for *externally provided* directed pairwise labels.

        Target[b,i,j] represents a supervision label of i preceding j.
        The diagonal is always excluded. Caller must justify labels and
        antisymmetry; this method does not derive causal labels.
        """
        prediction = self.forward(obs)
        if preference_labels.shape != prediction.shape:
            raise ValueError("preference labels must match [B,N,N]")
        if valid_pairs is not None and valid_pairs.shape != prediction.shape:
            raise ValueError("valid_pairs must match [B,N,N]")
        n = prediction.shape[-1]
        diag_off = ~torch.eye(n, dtype=torch.bool, device=prediction.device)
        valid = diag_off.unsqueeze(0).expand_as(prediction)
        if valid_pairs is not None:
            valid = valid & valid_pairs.bool()
        if not torch.any(valid):
            raise ValueError("no valid directed pairs for loss")
        return F.binary_cross_entropy(prediction[valid], preference_labels[valid])
