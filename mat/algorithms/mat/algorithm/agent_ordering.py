"""Agent-order utilities for the V0 MAT action decoder.

An order[b, k] is the *original* agent ID decoded at position k.
No trainable parameters or causal claims are introduced in V0.
"""
import torch


VALID_ORDER_MODES = ("identity", "random_fixed", "obs_norm")


def compute_agent_order(obs, mode="identity", seed=1):
    """Return [B, N] indices using current observations only.

    identity: original MAT order.
    random_fixed: same seeded permutation for every batch/step (static ablation).
    obs_norm: sort by descending mean absolute observation magnitude.
        This is a deterministic, per-step heuristic; it is *not* causality.
    """
    if obs.ndim != 3:
        raise ValueError("obs must have shape [batch, n_agent, obs_dim]")
    batch_size, n_agent, _ = obs.shape
    if mode == "identity":
        return torch.arange(n_agent, device=obs.device).expand(batch_size, -1)
    if mode == "random_fixed":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(n_agent, generator=generator).to(device=obs.device)
        return perm.expand(batch_size, -1)
    if mode == "obs_norm":
        # An observation-only heuristic keeps PPO's old/new order identical,
        # including when network weights change during policy optimization.
        scores = obs.detach().float().abs().mean(dim=-1)
        # Stable sorting makes tied agents retain their original relative IDs.
        return torch.argsort(scores, dim=1, descending=True, stable=True)
    raise ValueError(f"Unknown agent_order_mode={mode!r}; expected {VALID_ORDER_MODES}")


def reorder_agents(tensor, order):
    """Gather [B, N, ...] by per-batch positions -> original agent IDs."""
    if tensor is None:
        return None
    if order.dtype != torch.long or order.ndim != 2:
        raise ValueError("order must have shape [B,N] with torch.long dtype")
    if tensor.shape[:2] != order.shape:
        raise ValueError("Agent and batch dimensions of tensor/order differ")
    index = order.reshape(*order.shape, *((1,) * (tensor.ndim - 2)))
    return torch.gather(tensor, dim=1, index=index.expand_as(tensor))


def restore_agents(tensor, order):
    """Scatter [B, N, ...] decoded-position data to original agent IDs."""
    if tensor is None:
        return None
    if order.dtype != torch.long or order.ndim != 2:
        raise ValueError("order must have shape [B,N] with torch.long dtype")
    if tensor.shape[:2] != order.shape:
        raise ValueError("Agent and batch dimensions of tensor/order differ")
    index = order.reshape(*order.shape, *((1,) * (tensor.ndim - 2)))
    return torch.zeros_like(tensor).scatter(1, index.expand_as(tensor), tensor)


def validate_agent_order(agent_order, batch_size, n_agent, device):
    """Validate and materialize a stored original-agent permutation [B, N].

    IMPORTANT: never silently repair malformed rollout orders. PPO requires
    exactly the permutation used when collecting the corresponding actions.
    """
    order = torch.as_tensor(agent_order, device=device)
    if order.dtype != torch.long:
        raise ValueError("agent_order must use int64/torch.long indices")
    if order.shape != (batch_size, n_agent):
        raise ValueError("agent_order must have shape [batch, n_agent]")
    expected = torch.arange(n_agent, device=device).expand(batch_size, -1)
    if not torch.equal(torch.sort(order, dim=1).values, expected):
        raise ValueError("each agent_order row must be a permutation of agent IDs")
    return order
