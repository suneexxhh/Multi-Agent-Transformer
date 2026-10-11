"""Lightweight, CPU-testable helpers for influence-signal diagnostics."""
import numpy as np
import torch

from mat.algorithms.mat.algorithm.influence_alignment import alignment_targets


def residuals(js, valid, entropy, legal_counts, entropy_bins=4):
    """Observed JS minus position/target-entropy stratum mean; missing stays NaN."""
    b,n,_=js.shape
    if valid.shape!=js.shape or entropy.shape!=(b,n) or legal_counts.shape!=(b,n):
        raise ValueError('incompatible teacher statistic shapes')
    norm=(entropy/legal_counts.clamp_min(2).float().log()).clamp(0.,1.)
    bins=(norm*entropy_bins).long().clamp(max=entropy_bins-1)
    out=torch.full_like(js,float('nan'))
    for source in range(n-1):
        for target in range(source+1,n):
            for bin_id in range(entropy_bins):
                chosen=valid[:,target,source]&(bins[:,target]==bin_id)
                if chosen.sum()>=2:
                    values=js[chosen,target,source]
                    out[chosen,target,source]=values-values.mean()
    return out


def episode_bootstrap(values, episodes, weights=None, repeats=1000, seed=91011):
    """Resample complete episodes, never pair rows or individual states."""
    values=np.asarray(values,dtype=float)
    episodes=np.asarray(episodes)
    if values.ndim!=1 or episodes.shape!=values.shape or not values.size:
        raise ValueError('values and episode IDs must be aligned, nonempty vectors')
    unique=np.unique(episodes)
    groups=[np.flatnonzero(episodes==e) for e in unique]
    rng=np.random.default_rng(seed)
    if weights is None:weights=np.ones(len(values),dtype=float)
    weights=np.asarray(weights,dtype=float)
    if weights.shape!=values.shape or (weights<0).any():
        raise ValueError('invalid bootstrap weights')
    draws=[]
    for _ in range(repeats):
        ids=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        total=weights[ids].sum()
        draws.append(float(np.dot(values[ids],weights[ids])/total) if total else np.nan)
    good=np.asarray(draws)[np.isfinite(draws)]
    return tuple(np.quantile(good,[.025,.975]).tolist()) if good.size else (np.nan,np.nan)


def partition_labels(js, valid, entropy, alternatives, counts, groups, k=2):
    """Original target, recomputed independently for each PPO minibatch."""
    nstate,nagent,_=js.shape
    targets=torch.full_like(js,.5)
    confidence=torch.zeros_like(js)
    singleton_pairs=0
    seen=np.zeros(nstate,dtype=int)
    for group in groups:
        ids=torch.as_tensor(group,dtype=torch.long)
        if ids.numel()==0 or (ids<0).any() or (ids>=nstate).any():
            raise ValueError('invalid minibatch IDs')
        seen[ids.numpy()]+=1
        y,w=alignment_targets(js[ids],valid[ids],entropy[ids],alternatives[ids],
                              counts[ids],max_alternatives=k)
        targets[ids]=y;confidence[ids]=w
        norm_h=(entropy[ids]/counts[ids].clamp_min(2).float().log()).clamp(0.,1.)
        bins=(norm_h*4).long().clamp(max=3)
        for s in range(nagent-1):
            for t in range(s+1,nagent):
                for bin_id in range(4):
                    subset=valid[ids,t,s]&(bins[:,t]==bin_id)
                    if int(subset.sum())==1:singleton_pairs+=1
    if not np.all(seen==1):
        raise ValueError('minibatches must partition every transition exactly once')
    return targets,confidence,singleton_pairs
