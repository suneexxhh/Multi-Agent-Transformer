"""Prefix-conditioned directed dependency ordering for original MAT Decoder.

Research V3 prototype (not causal/environmental influence identification).
p(order | obs) = prod_t softmax(score(obs, selected_prefix_t))[order_t].
The permutation is sampled *before* any action values; the prefix contains
agent IDs only, not prior actions. Training uses replayed exact logprob for
the chosen permutation and the unchanged MAT autoregressive action model.

Two distinct directed quantities:
- precedence logits m[i,j] = -m[j,i] (inherited DirectedDependencyOrder)
- demand D[target,source] = sigmoid(u_target dot v_source / sqrt(rank)),
  intentionally NOT antisymmetric.  It represents learned preference to have
  source's action available before choosing target's action.
Both are reward-trained preferences, NOT identified causal effects.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.agent_ordering import validate_agent_order
from mat.algorithms.mat.algorithm.directed_dependency_order import DirectedDependencyOrder


class PrefixDependencyOrder(DirectedDependencyOrder):
    """Learned prefix-dependent permutation policy with exact replay logprob."""

    def __init__(self, obs_dim, hidden_dim=64, temperature=1.0,
                 pair_rank=16, graph_scale=0.5, prefix_scale=1.0,
                 adaptive_attention_bias=False, attention_scale_max=2.0):
        super().__init__(obs_dim, hidden_dim, temperature, pair_rank, graph_scale)
        if not math.isfinite(prefix_scale) or prefix_scale <= 0.0:
            raise ValueError("prefix_scale must be finite and strictly positive")
        self.prefix_scale = float(prefix_scale)
        if not math.isfinite(attention_scale_max) or attention_scale_max <= 0:
            raise ValueError("attention_scale_max must be finite and positive")
        self.adaptive_attention_bias = bool(adaptive_attention_bias)
        self.attention_scale_max = float(attention_scale_max)
        if self.adaptive_attention_bias:
            # Zero exactly recovers the original MAT conditional decoder.
            self.raw_attention_scale = nn.Parameter(torch.zeros(()))
        self.demand_query = nn.Linear(hidden_dim, pair_rank, bias=False)
        self.demand_key = nn.Linear(hidden_dim, pair_rank, bias=False)

    def demand_logits(self, obs):
        """[B,target,source]; directions may *both* be high."""
        if obs.ndim != 3 or obs.shape[1] < 1:
            raise ValueError("obs must be [B,N,D], N > 0")
        h = self.embedding(obs)
        q = self.demand_query(h)
        k = self.demand_key(h)
        logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.pair_rank)
        # Self-dependency has no meaning for inter-agent ordering.
        eye = torch.eye(logits.shape[1], device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(eye[None], 0.0)
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite directed demand logits")
        return logits

    def demand(self, obs):
        return torch.sigmoid(self.demand_logits(obs))

    def _step_scores(self, base, demand, selected, step):
        """[B,N] logits for all candidates; selected members masked separately.

        For each target i, reward agents whose prerequisites are already in
        the prefix, penalize demand for prerequisites not yet decoded.
        Dividing by prefix/unselected counts avoids systematic scale changes
        with team size. The diagonal is excluded from unmet dependencies.
        """
        n = base.shape[1]
        if selected.shape != base.shape or demand.shape != (base.shape[0], n, n):
            raise ValueError("invalid prefix conditional score shapes")
        selected_f = selected.to(dtype=base.dtype)
        remaining_f = 1.0 - selected_f
        others = 1.0 - torch.eye(n, device=base.device, dtype=base.dtype)[None]
        fulfilled = (demand * selected_f[:, None, :]).sum(-1) / max(step, 1)
        unmet = (demand * remaining_f[:, None, :] * others).sum(-1) / max(n - step - 1, 1)
        return base + self.prefix_scale * (fulfilled - unmet) / self.temperature

    def prefix_scores(self, obs, selected):
        """Observable next-agent logits (including scores for excluded agents).

        A [B,N] boolean selected set; only the remaining agents are eligible.
        """
        base = super().scores(obs)
        selected = torch.as_tensor(selected, dtype=torch.bool, device=base.device)
        if selected.shape != base.shape:
            raise ValueError("selected prefix mask must be [B,N]")
        steps = selected.sum(-1)
        if not torch.equal(steps, steps[:1].expand_as(steps)):
            raise ValueError("each batch member must have same prefix length")
        step = int(steps[0].item())
        if step >= base.shape[1]:
            raise ValueError("no undecoded agent remains")
        raw = self._step_scores(base, self.demand(obs), selected, step)
        return raw.masked_fill(selected, float("-inf"))

    def sample(self, obs, deterministic=False):
        base = super().scores(obs)
        demand = self.demand(obs)
        batch, agents = base.shape
        selected = torch.zeros_like(base, dtype=torch.bool)
        logprob = base.new_zeros(batch)
        sequence = []
        for step in range(agents):
            scores = self._step_scores(base, demand, selected, step)
            masked = scores.masked_fill(selected, float("-inf"))
            distribution = torch.distributions.Categorical(logits=masked)
            chosen = (torch.argmax(masked, -1) if deterministic
                      else distribution.sample())
            logprob = logprob + distribution.log_prob(chosen)
            sequence.append(chosen)
            selected = selected.scatter(1, chosen.unsqueeze(1), True)
        return torch.stack(sequence, dim=1), logprob[:, None]

    def log_prob(self, obs, order):
        base = super().scores(obs)
        batch, agents = base.shape
        order = validate_agent_order(order, batch, agents, base.device)
        demand = self.demand(obs)
        selected = torch.zeros_like(base, dtype=torch.bool)
        logprob = base.new_zeros(batch)
        for step in range(agents):
            scores = self._step_scores(base, demand, selected, step)
            masked = scores.masked_fill(selected, float("-inf"))
            chosen = order[:, step:step + 1]
            logprob = logprob + F.log_softmax(masked, -1).gather(
                1, chosen).squeeze(1)
            selected = selected.scatter(1, chosen, True)
        return logprob[:, None]

    def conditioning_bias(self, obs, order):
        """A bounded directional *demand* gate for shifted prior action keys.

        Target at decoded position t attends to the prior source at position s.
        bias[t,s+1] = log(2 * sigmoid(D_logit[target,source].clamp(-8,8))).
        BOS bias is zero, future action keys remain blocked by MAT's mask.
        """
        order = validate_agent_order(order, obs.shape[0], obs.shape[1], obs.device)
        logits = self.demand_logits(obs)
        ordered = logits.gather(1, order.unsqueeze(-1).expand_as(logits))
        ordered = ordered.gather(2, order.unsqueeze(1).expand_as(ordered))
        batch, agents, _ = ordered.shape
        prior = math.log(2.) + F.logsigmoid(ordered[:, :, :-1].clamp(-8., 8.))
        bos = ordered.new_zeros(batch, agents, 1)
        bias = torch.cat((bos, prior), dim=-1)
        if self.adaptive_attention_bias:
            bias = self.attention_scale_max * torch.tanh(self.raw_attention_scale) * bias
        return bias

    def attention_scale(self):
        """Effective bounded gain; the legacy V3 fixed-bias path is one."""
        if not self.adaptive_attention_bias:
            return 1.0
        return self.attention_scale_max * torch.tanh(self.raw_attention_scale)
