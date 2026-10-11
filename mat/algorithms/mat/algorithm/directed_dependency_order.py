"""Reward-trained directed *precedence* policy for MAT Decoder.

This module does NOT infer causal/environmental dependencies. Directed
preferences represent how the agent-order actor chooses which action is
generated first. Intervention-based alignment is optional and offline only.

For agent representations h_i and low-rank q_i,k_i:
    m_ij = (q_i^T k_j - q_j^T k_i) / sqrt(rank)
    p_ij = sigmoid(m_ij), p_ij + p_ji = 1
    score_i = base_i + softplus(gain) * mean_{j!=i} tanh(m_ij/2)
The inherited exact Plackett-Luce sampler then yields the permutation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.learned_agent_order import LearnedAgentOrder


class DirectedDependencyOrder(LearnedAgentOrder):
    """Permutation-equivariant, low-rank directed preference ordering actor.

    The order policy is optimized by the saved-permutation PPO objective.
    The associated confidence DAG is a *diagnostic projection only*; it is
    NOT yet used to sparsify decoder attention or enable parallel sampling.
    """

    def __init__(self, obs_dim, hidden_dim=64, temperature=1.0,
                 pair_rank=16, graph_scale=0.5):
        super().__init__(obs_dim, hidden_dim, temperature)
        if (not isinstance(pair_rank, int) or pair_rank <= 0 or
                not math.isfinite(graph_scale) or graph_scale <= 0):
            raise ValueError("pair_rank and graph_scale must be positive and finite")
        self.pair_rank = pair_rank
        self.pair_query = nn.Linear(hidden_dim, pair_rank, bias=False)
        self.pair_key = nn.Linear(hidden_dim, pair_rank, bias=False)
        # Positive but trainable gain; initialized exactly to graph_scale.
        self.raw_graph_gain = nn.Parameter(torch.tensor(
            math.log(math.expm1(graph_scale)), dtype=torch.float32))

    def _directional_logits(self, embeddings):
        q = self.pair_query(embeddings)
        k = self.pair_key(embeddings)
        matrix = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.pair_rank)
        return matrix - matrix.transpose(1, 2)

    def pairwise_logits(self, obs):
        if obs.ndim != 3 or obs.shape[1] == 0:
            raise ValueError("obs must be [B,N,D], with N > 0")
        return self._directional_logits(self.embedding(obs))

    def pairwise_preferences(self, obs):
        """p[b,i,j] is the learned preference that agent i precedes j."""
        return torch.sigmoid(self.pairwise_logits(obs))

    def scores(self, obs):
        if obs.ndim != 3 or obs.shape[1] == 0:
            raise ValueError("obs must be [B,N,D], with N > 0")
        h = self.embedding(obs)
        context = h.mean(dim=1, keepdim=True).expand_as(h)
        base = self.head(torch.cat((h, context), dim=-1)).squeeze(-1)
        margins = self._directional_logits(h)
        # tanh(m/2) == 2 sigmoid(m) - 1; zero diagonal.
        prior = torch.tanh(margins * 0.5).sum(dim=-1) / max(h.shape[1] - 1, 1)
        logits = (base + F.softplus(self.raw_graph_gain) * prior) / self.temperature
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite precedence scores")
        return logits

    @torch.no_grad()
    def candidate_dag(self, obs, threshold=0.7, order=None):
        """Return (acyclic candidate edges [B,N,N], exact PL ordering).

        Edges i->j are retained only if i precedes j in the chosen order
        AND p(i before j)>threshold. Missing edges are *not* evidence
        of action independence or environmental non-causality.
        """
        if not 0.5 < threshold < 1.0:
            raise ValueError("graph threshold must be between 0.5 and 1")
        from mat.algorithms.mat.algorithm.agent_ordering import validate_agent_order
        if order is None:
            order, _ = self.sample(obs, deterministic=True)
        else:
            order = validate_agent_order(order, obs.shape[0], obs.shape[1], obs.device)
        ranks = torch.empty_like(order)
        positions = torch.arange(order.shape[1], device=order.device).expand_as(order)
        ranks.scatter_(1, order, positions)
        forward = ranks[:, :, None] < ranks[:, None, :]
        edges = (self.pairwise_preferences(obs) > threshold) & forward
        return edges, order

    def counterfactual_alignment_loss(self, obs, influence, valid,
                                      minimum_effect_gap=1e-4):
        """OPTIONAL offline teacher for matched directed policy-KL probes.

        influence[b,i,j] estimates *within-frozen-decoder conditional policy*
        response of target j when source i's earlier action is perturbed.
        Both i->j and j->i must have been independently observed under
        compatible contexts, otherwise the pair is rejected. Labels are
        not inferred from unavailable directions or zero-filled probes.
        This routine is never called automatically from live SMAC PPO.
        """
        if not math.isfinite(minimum_effect_gap) or minimum_effect_gap < 0:
            raise ValueError("minimum_effect_gap must be finite and >=0")
        logits = self.pairwise_logits(obs)
        if influence.shape != logits.shape or valid.shape != logits.shape:
            raise ValueError("influence and valid must have shape [B,N,N]")
        effects = influence.to(dtype=logits.dtype, device=logits.device)
        known = valid.to(dtype=torch.bool, device=logits.device)
        mutual = known & known.transpose(-1, -2)
        delta = effects - effects.transpose(-1, -2)
        upper = torch.triu(torch.ones_like(mutual, dtype=torch.bool), diagonal=1)
        reliable = mutual & upper & torch.isfinite(delta) & (
            delta.abs() > minimum_effect_gap)
        if not reliable.any():
            raise ValueError("no mutually observed directional preference labels")
        target = (delta[reliable] > 0).to(logits.dtype)
        return F.binary_cross_entropy_with_logits(logits[reliable], target)
