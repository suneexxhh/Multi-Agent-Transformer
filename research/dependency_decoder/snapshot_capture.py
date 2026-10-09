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
                                 directory, identifier, context_steps=None,
                                 episode_index=None):
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
    if context_steps is not None:
        context_steps = np.asarray(context_steps)
        if (context_steps.shape != (obs.shape[0],) or
                context_steps.dtype != np.int64 or
                (context_steps < 0).any() or
                not isinstance(episode_index, int) or episode_index < 0):
            raise ValueError("context_steps must be int64 [B] with a valid episode index")
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
        extra = {}
        if context_steps is not None:
            extra["context_step"] = context_steps.copy()
            extra["context_episode"] = np.full(
                (obs.shape[0],), episode_index, dtype=np.int64)
        np.savez_compressed(
            tmp_npz,
            obs=obs.astype(np.float32, copy=True),
            available_actions=legal.astype(np.float32, copy=True),
            agent_order=order.copy(),
            **extra,
        )
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   tmp_pt)
        os.replace(tmp_npz, snap)
        os.replace(tmp_pt, checkpoint)
    finally:
        tmp_npz.unlink(missing_ok=True)
        tmp_pt.unlink(missing_ok=True)
    return str(snap), str(checkpoint)


class MultiContextCapture:
    """Bounded original-ID snapshots at multiple steps under ONE frozen policy.

    A single SMAC episode contains no PPO update before its end, so all
    sampled contexts correspond to the exact same encoder/decoder weights.
    Never combine contexts from different update episodes with one set
    of checkpoint weights.
    """

    def __init__(self, steps=(0, 10, 20, 30)):
        steps = tuple(steps)
        if (not steps or len(steps) > 16 or
                any(not isinstance(step, int) or step < 0 for step in steps) or
                tuple(sorted(set(steps))) != steps):
            raise ValueError("steps must be sorted unique nonnegative integers (max 16)")
        self.steps = steps
        self._captured = {}

    def observe(self, step, obs, available_actions, order):
        if step not in self.steps:
            return False
        if step in self._captured:
            raise ValueError("same context step captured twice")
        # Explicit deep copies prevent replay buffers reusing storage from
        # corrupting any context while rollout proceeds.
        self._captured[step] = (
            np.array(obs, dtype=np.float32, copy=True),
            np.array(available_actions, dtype=np.float32, copy=True),
            np.array(order, dtype=np.int64, copy=True))
        return True

    @property
    def complete(self):
        return len(self._captured) == len(self.steps)

    def save(self, model, directory, identifier, episode_index):
        if not self.complete:
            raise ValueError("cannot checkpoint a partial context collection")
        shapes = [self._captured[step][0].shape for step in self.steps]
        if len(set(shapes)) != 1 or len(shapes[0]) != 3:
            raise ValueError("all context snapshots must share [E,N,D] shape")
        env_count = shapes[0][0]
        # A bounded number of contexts prevents accidental checkpoint
        # artifacts from exhausting runner RAM or CPU diagnostic time.
        if env_count * len(self.steps) > 64:
            raise ValueError("at most 64 observation contexts per snapshot")
        obs = np.concatenate([self._captured[i][0] for i in self.steps])
        legal = np.concatenate([self._captured[i][1] for i in self.steps])
        orders = np.concatenate([self._captured[i][2] for i in self.steps])
        context_steps = np.repeat(np.asarray(self.steps, dtype=np.int64), env_count)
        return save_snapshot_and_checkpoint(
            model, obs, legal, orders, directory, identifier,
            context_steps=context_steps, episode_index=episode_index)
