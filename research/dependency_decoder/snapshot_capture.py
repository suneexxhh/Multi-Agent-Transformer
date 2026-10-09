"""Opt-in, single-shot SMAC observation + exactly matching MAT checkpoint.

Invoked only when explicitly enabled in SMAC Runner; never runs on V0
default path. The snapshot contains the ORIGINAL agent IDs and the
exact model weights before the next PPO update.
"""
import os
import re
from pathlib import Path

import numpy as np
import torch

from mat.algorithms.mat.algorithm.agent_ordering import validate_agent_order


@torch.no_grad()
def save_snapshot_and_checkpoint(model, obs, available_actions, agent_order,
                                 directory, identifier):
    """Write matching NPZ and safe weights-only state_dict checkpoint.

    obs, available_actions, agent_order represent the SAME pre-action
    transition. One snapshot is taken immediately after sampling an
    action and BEFORE the environment steps or PPO updates.
    """
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,110}", identifier):
        raise ValueError("snapshot identifier must be a short safe filename")
    if model.training:
        raise ValueError("snapshot model must be in eval mode")
    obs = np.asarray(obs)
    legal = np.asarray(available_actions)
    order = np.asarray(agent_order)
    if (obs.ndim != 3 or legal.ndim != 3 or order.shape != obs.shape[:2] or
            legal.shape[:2] != obs.shape[:2] or order.dtype != np.int64 or
            not np.isfinite(obs).all() or not np.isfinite(legal).all() or
            not np.all(legal.sum(-1) >= 1)):
        raise ValueError("expected finite [B,N,O], [B,N,A], int64 [B,N] arrays")
    validate_agent_order(torch.as_tensor(order), obs.shape[0], obs.shape[1], "cpu")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    snap = root / (identifier + "_snapshot.npz")
    checkpoint = root / (identifier + "_transformer.pt")
    # Fail rather than replacing an earlier pair of experiment artifacts.
    if snap.exists() or checkpoint.exists():
        raise FileExistsError("diagnostic snapshot files already exist")
    tmp_npz = root / (identifier + "_snapshot.tmp.npz")
    tmp_pt = root / (identifier + "_transformer.tmp.pt")
    try:
        np.savez_compressed(
            tmp_npz,
            obs=obs.astype(np.float32, copy=True),
            available_actions=legal.astype(np.float32, copy=True),
            agent_order=order.copy(),
        )
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   tmp_pt)
        os.replace(tmp_npz, snap)
        os.replace(tmp_pt, checkpoint)
    finally:
        tmp_npz.unlink(missing_ok=True)
        tmp_pt.unlink(missing_ok=True)
    return str(snap), str(checkpoint)
