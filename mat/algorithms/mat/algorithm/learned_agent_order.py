"""On-policy Plackett-Luce policy over MAT agent decoding permutations.

Order[b, k] is the ORIGINAL agent ID decoded at position k. This is not
a causal dependency estimator; scores are learned from team reward alone.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mat.algorithms.mat.algorithm.agent_ordering import validate_agent_order


class LearnedAgentOrder(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64, temperature=1.0):
        super().__init__()
        if obs_dim <= 0 or hidden_dim <= 0 or not 0 < temperature < float("inf"):
            raise ValueError("obs_dim, hidden_dim, and finite temperature must be positive")
        self.temperature = float(temperature)
        self.embedding = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(),
                                  nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def scores(self, obs):
        if obs.ndim != 3 or obs.size(1) < 1:
            raise ValueError("obs must be [B,N,D], N > 0")
        x = self.embedding(obs)
        team = x.mean(dim=1, keepdim=True).expand_as(x)
        z = self.head(torch.cat([x, team], -1)).squeeze(-1)
        if not torch.isfinite(z).all():
            raise ValueError("non-finite order scores")
        return z / self.temperature

    @staticmethod
    def _log_prob(scores, order):
        batch, agents = scores.shape
        remaining = torch.ones_like(scores, dtype=torch.bool)
        lp = scores.new_zeros(batch)
        for k in range(agents):
            step_lp = F.log_softmax(scores.masked_fill(~remaining, float("-inf")), -1)
            chosen = order[:, k:k + 1]
            lp = lp + step_lp.gather(1, chosen).squeeze(1)
            remaining.scatter_(1, chosen, False)
        return lp.unsqueeze(-1)

    def sample(self, obs, deterministic=False):
        scores = self.scores(obs)
        if deterministic:
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        else:
            available = torch.ones_like(scores, dtype=torch.bool)
            choices = []
            for _ in range(scores.size(1)):
                picked = torch.distributions.Categorical(
                    logits=scores.masked_fill(~available, float("-inf"))).sample()
                choices.append(picked)
                available.scatter_(1, picked.unsqueeze(-1), False)
            order = torch.stack(choices, dim=1)
        return order, self._log_prob(scores, order)

    def log_prob(self, obs, order):
        scores = self.scores(obs)
        order = validate_agent_order(order, scores.size(0), scores.size(1), scores.device)
        return self._log_prob(scores, order)
