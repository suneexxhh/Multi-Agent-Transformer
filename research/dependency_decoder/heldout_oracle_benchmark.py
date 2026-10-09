"""Reproducible held-out synthetic decoder-dependence benchmark.

Truth is an EXPLICIT graph constructed by this benchmark. This evaluates
conditional decoder-action sensitivity and directed precedence transfer;
it is not environmental causal discovery, and is never used in PPO.
"""
import torch
from torch import nn

from research.dependency_decoder.action_influence_probe import (
    legal_action_kl_probe, combine_order_probes,
    conservative_precedence_targets,
)


def make_priority_graph(traits, min_gap=0.55):
    """A sparse synthetic DAG: edge i->j iff trait_i-trait_j > min_gap.

    No graph labels are included in scorer inputs: traits[B,N,1] are the
    only scorer features. This toy rule is known before experiments.
    """
    if traits.ndim != 3 or traits.shape[-1] != 1:
        raise ValueError("traits must be [B,N,1]")
    if not min_gap > 0:
        raise ValueError("min_gap must be positive")
    strengths = traits[:, :, 0].unsqueeze(2) - traits[:, :, 0].unsqueeze(1)
    return strengths > min_gap


class GraphOracleDecoder(nn.Module):
    """Causally masked toy decoder with known incoming action edges.

    obs_rep[b,position] = concat(one_hot(original_id), incoming_edges).
    Only a preceding action==2 can affect later agent logits, and ONLY
    if the ground-truth adjacency contains an edge source->destination.
    It deliberately accepts any original-ID permutation.
    """

    def forward(self, shifted, obs_rep, obs):
        b, n, width = shifted.shape
        if width != 4 or obs_rep.shape != (b, n, 2*n):
            raise ValueError("expected 3 discrete actions and [B,N,2N]")
        agent_ids = obs_rep[:, :, :n]
        incoming = obs_rep[:, :, n:]
        # Shifted action at position p is the action taken at position p-1.
        prev_id = torch.nn.functional.pad(
            agent_ids[:, :-1], (0, 0, 1, 0))
        action2 = shifted[..., 3:4]
        predecessor_signal = (prev_id * action2).cumsum(1)
        influence = (incoming * predecessor_signal).sum(-1)
        logits = shifted.new_zeros((b, n, 3))
        logits[:, :, 1] = 4.0 * influence
        return logits


@torch.no_grad()
def probe_known_graphs(graph, chunk=128, legal=None):
    """Probe arbitrary unseen graphs using both ID and reversed-ID orders.

    Scores/coverage are restored to original IDs. The ground-truth graph
    enters ONLY the test oracle decoder; it is never a scorer feature.
    """
    if graph.ndim != 3 or graph.shape[1] != graph.shape[2]:
        raise ValueError("graph must be [B,N,N]")
    b, n, _ = graph.shape
    if n < 2:
        raise ValueError("need at least two agents")
    eye = torch.eye(n, dtype=torch.float32, device=graph.device)
    one_hot_ids = eye.unsqueeze(0).expand(b, -1, -1)
    # The target at original ID j receives the column graph[:, :, j].
    obs_rep = torch.cat([one_hot_ids, graph.transpose(1, 2).float()], -1)
    obs = torch.zeros((b, n, 1), device=graph.device)
    actions = torch.zeros((b, n, 1), dtype=torch.long, device=graph.device)
    legal = (torch.ones((b, n, 3), device=graph.device)
             if legal is None else legal)
    ascending = torch.arange(n, device=graph.device).expand(b, -1)
    descending = ascending.flip(1)
    decoder = GraphOracleDecoder().eval()
    p1, m1 = legal_action_kl_probe(
        decoder, obs_rep, obs, actions, legal, ascending,
        return_valid=True, max_counterfactual_batch=chunk)
    p2, m2 = legal_action_kl_probe(
        decoder, obs_rep, obs, actions, legal, descending,
        return_valid=True, max_counterfactual_batch=chunk)
    return combine_order_probes(
        torch.stack((p1, p2)), torch.stack((m1, m2)))


@torch.no_grad()
def direction_metrics(scores, counts, graph, margin=0.05):
    """Report zero/positive coverage and confusion on held-out graphs.

    All metrics use actually bidirectionally observed OFF-DIAGONAL pairs.
    Unknown pairs do not count as true negatives. Returns integer counts
    plus directed precision/recall/FPR and unordered-pair coverage.
    """
    if graph.shape != scores.shape or counts.shape != graph.shape:
        raise ValueError("all matrices must be [B,N,N]")
    if graph.dtype != torch.bool:
        raise ValueError("graph must have boolean edges")
    labels, mask = conservative_precedence_targets(scores, counts, margin)
    n = graph.shape[-1]
    offdiag = ~torch.eye(n, device=graph.device, dtype=torch.bool)
    measured = (counts > 0) & (counts.transpose(-1, -2) > 0) & offdiag
    prediction = (labels > 0.5) & mask
    tp = int((prediction & graph).sum().item())
    fp = int((prediction & ~graph & measured).sum().item())
    fn = int((~prediction & graph & measured).sum().item())
    tn = int((~prediction & ~graph & measured).sum().item())
    candidate_pairs = int(measured.sum().item())
    total_possible = int(graph.shape[0] * n * (n - 1))
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        observed_directed_pairs=candidate_pairs,
        total_directed_pairs=total_possible,
        coverage=candidate_pairs / max(total_possible, 1),
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        false_positive_rate=fp / max(fp + tn, 1),
        abstained_pairs=int((measured & ~mask).sum().item()),
    )
