"""Offline V3.1 teacher-label diagnostic on the fixed REAL V2 SMAC observations.

This never trains a policy. The V2 checkpoint is a strict-loaded Bias-OFF
teacher; recorded legal random actions are history, not V3.1 rollout actions.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from mat.algorithms.mat.algorithm.agent_ordering import reorder_agents
from mat.algorithms.mat.algorithm.influence_alignment import (
    alignment_targets, conditional_action_js, frozen_bias_off_teacher)
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from research.dependency_decoder.checkpoint_manifest import load_training_manifest


DATA = Path('/data/sgg/mat-decoder-storage/analysis/v2_dependency_mechanism_3s5z_seed1/real_smac_observations.npz')
RUN = Path('/data/sgg/mat-decoder-storage/smoke/3s5z_dependency_seed1_4k_90e20b8/training_results/StarCraft2/3s5z/mat/smoke4k_v2_90e20b8_dependency_orderseed1_seed1/run1')
OUT = Path(os.environ.get('V31_ANALYSIS_DIR', '/data/sgg/mat-decoder-storage/analysis/v31_influence_signal_validation'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_teacher():
    manifest = load_training_manifest(RUN / 'mat_decoder_manifest.json', '3s5z', 'dependency')
    c = manifest['config']
    model = MultiAgentTransformer(
        37, 252, 14, 8, n_block=c['n_block'], n_embd=c['n_embd'],
        n_head=c['n_head'], encode_state=c['encode_state'], device=torch.device('cpu'),
        action_type='Discrete', dec_actor=c['dec_actor'], share_actor=c['share_actor'],
        agent_order_mode=c['agent_order_mode'], agent_order_seed=c['agent_order_seed'],
        order_hidden_dim=c['order_hidden_dim'], order_temperature=c['order_temperature'],
        order_pair_rank=c['order_pair_rank'], order_graph_scale=c['order_graph_scale'],
        order_prefix_scale=c['order_prefix_scale'], order_attention_bias=False)
    ckpt = RUN / 'models/transformer_19.pt'
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    return model, manifest, ckpt


def quantiles(x):
    x = np.asarray(x, dtype=float)
    return dict(n=int(x.size), mean=float(x.mean()) if x.size else None,
                std=float(x.std()) if x.size else None,
                q00=float(x.min()) if x.size else None,
                q05=float(np.quantile(x, .05)) if x.size else None,
                q25=float(np.quantile(x, .25)) if x.size else None,
                q50=float(np.quantile(x, .50)) if x.size else None,
                q75=float(np.quantile(x, .75)) if x.size else None,
                q95=float(np.quantile(x, .95)) if x.size else None,
                q100=float(x.max()) if x.size else None)


def residuals(js, valid, entropy, legal_counts):
    """Same position/target-entropy strata as alignment_targets, for diagnostics."""
    b, n, _ = js.shape
    normalized = (entropy / legal_counts.clamp_min(2).float().log()).clamp(0., 1.)
    bins = (normalized * 4).long().clamp(max=3)
    out = torch.full_like(js, float('nan'))
    for source in range(n - 1):
        for target in range(source + 1, n):
            for bin_id in range(4):
                mask = valid[:, target, source] & (bins[:, target] == bin_id)
                if mask.sum() >= 2:
                    values = js[mask, target, source]
                    out[mask, target, source] = values - values.mean()
    return out


def episode_bootstrap_corr(x, y, episode, repeats=400, seed=99117):
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, episode = x[finite], y[finite], episode[finite]
    if x.size < 10 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return dict(n=int(x.size), spearman=None, episode_bootstrap_95=[None, None])
    point = float(spearmanr(x, y).statistic)
    groups = [np.flatnonzero(episode == e) for e in np.unique(episode)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        ids = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if np.std(x[ids]) >= 1e-12 and np.std(y[ids]) >= 1e-12:
            draws.append(float(spearmanr(x[ids], y[ids]).statistic))
    ci = np.quantile(draws, [.025, .975]).tolist() if draws else [None, None]
    return dict(n=int(x.size), spearman=point, episode_bootstrap_95=ci)


def run(v31_init=False):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    data = np.load(DATA, allow_pickle=False)
    obs = data['obs']
    legal = data['avail_actions']
    actions = data['sampled_legal_actions']
    episode = data['episode']
    if obs.shape != (320, 8, 252) or legal.shape != (320, 8, 14) or actions.shape != (320, 8):
        raise ValueError('real V2 dataset missing required observation/legal/history fields')
    if len(np.unique(episode)) < 8 or not legal[np.arange(320)[:, None], np.arange(8), actions].all():
        raise ValueError('episode coverage or recorded historical action legality failed')
    observation = torch.from_numpy(obs)
    mask = torch.from_numpy(legal).bool()
    if v31_init:
        torch.manual_seed(99117)
        initial = MultiAgentTransformer(37,252,14,8,n_block=1,n_embd=64,n_head=1,
            agent_order_mode='dependency_prefix',order_attention_bias=True,
            adaptive_order_attention_bias=True)
        initial.eval().requires_grad_(False)
        with torch.no_grad():
            rollout = [initial.get_actions(torch.zeros(40,8,37), observation[i:i+40],
                mask[i:i+40].float(), return_agent_order=True,
                return_order_log_prob=True) for i in range(0,320,40)]
        history = torch.cat([r[0] for r in rollout])
        order = torch.cat([r[3] for r in rollout])
        teacher = frozen_bias_off_teacher(initial)
        prediction_model = initial
        source = dict(checkpoint=None, checkpoint_sha256=None,
                      training_source_sha=None,
                      teacher='V3.1 random-init policy snapshot, frozen Bias OFF, CPU',
                      history='random-init V3.1 policy-sampled legal actions on real states; no PPO training',
                      order='random-init V3.1 sampled prefix order; no PPO training')
        suffix='_v31_init'
    else:
        teacher, manifest, ckpt = load_teacher()
        history = torch.from_numpy(actions.astype(np.int64))[:, :, None]
        with torch.no_grad():
            order = torch.cat([torch.argsort(teacher.order_policy.scores(observation[i:i+40]),
                                              dim=-1, descending=True, stable=True)
                               for i in range(0, 320, 40)])
        torch.manual_seed(99117)
        prediction_model = MultiAgentTransformer(37,252,14,8,n_block=1,n_embd=64,n_head=1,
            agent_order_mode='dependency_prefix',order_attention_bias=False)
        prediction_model.eval().requires_grad_(False)
        source = dict(checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),
                      training_source_sha=manifest['training_source_sha'],
                      history='recorded uniform-random legal SMAC collection actions; not V3.1 rollout',
                      order='V2 checkpoint deterministic score order; not V3.1 sampled prefix order',
                      teacher='V2 checkpoint strict=True, Bias OFF, frozen, CPU')
        suffix=''
    if not mask.gather(-1,history.long()).all():
        raise ValueError('chosen history action violates real SMAC legal mask')
    ordered_legal = reorder_agents(mask, order)
    counts = ordered_legal.sum(-1)
    records = []
    summary = dict(source_data=str(DATA), source_data_sha256=sha(DATA),
                   real_states=320, episodes=int(len(np.unique(episode))),
                   **source,
                   trained_v31_demand_available=False, settings={})
    for k in (1, 2, 4):
        raw = [[], [], [], []]
        with torch.no_grad():
            for start in range(0, 320, 40):
                batch = conditional_action_js(teacher, observation[start:start+40],
                                              history[start:start+40], mask[start:start+40],
                                              order[start:start+40], max_alternatives=k)
                for target, component in zip(raw, batch):
                    target.append(component)
        js, valid, entropy, alt_count = [torch.cat(parts) for parts in raw]
        residual = residuals(js, valid, entropy, counts)
        pair_mask = torch.tril(torch.ones(8, 8, dtype=torch.bool), diagonal=-1)[None].expand(320, -1, -1)
        valid_mask = valid & pair_mask
        overall = dict(valid_pairs=int(valid_mask.sum()), possible_ordered_pairs=int(pair_mask.sum()),
                       possible_all_directions=320*8*7,
                       valid_fraction_given_precedence=float(valid_mask.float().sum()/pair_mask.float().sum()),
                       valid_fraction_all_directions=float(valid_mask.sum()/(320*8*7)),
                       js_nats=quantiles(js[valid_mask].numpy()),
                       stratum_residual_nats=quantiles(residual[valid_mask & torch.isfinite(residual)].numpy()))
        for effect in (1e-6, 1e-5, 1e-4):
            for floor in (0., 1e-7, 1e-6):
                target, confidence = alignment_targets(js, valid, entropy, alt_count, counts,
                                                        max_alternatives=k, effect_scale=effect,
                                                        min_effect=floor)
                w = confidence[valid_mask].numpy()
                y = target[valid_mask].numpy()
                ep_grid = np.broadcast_to(episode[:, None, None], (320, 8, 8))[valid_mask.numpy()]
                episode_weight = [float(w[ep_grid == e].sum()) for e in np.unique(episode)]
                rec = dict(k=k, effect_scale=effect, min_effect=floor,
                           supervised_pairs=int((w > 0).sum()),
                           supervised_fraction_valid=float((w > 0).mean()),
                           confidence_sum=float(w.sum()),
                           effective_pair_count=float(w.sum()**2 / np.square(w).sum()) if np.square(w).sum() else 0.,
                           effective_episode_count=float(np.sum(episode_weight)**2 / np.square(episode_weight).sum()) if np.square(episode_weight).sum() else 0.,
                           target=quantiles(y), target_near_half_005=float((np.abs(y-.5)<.05).mean()),
                           target_near_half_001=float((np.abs(y-.5)<.01).mean()))
                records.append(rec)
        summary['settings'][str(k)] = overall
        if k == 2:
            summary['default_label_example'] = next(r for r in records if r['k']==2 and r['effect_scale']==1e-5 and r['min_effect']==1e-7)
            # No trained V3.1 D exists: do not mislabel random-init predictions as learned correlation.
            with torch.no_grad():
                d = prediction_model.order_policy.demand(observation)
                ordered = d.gather(1,order.unsqueeze(-1).expand_as(d))
                ordered = ordered.gather(2,order.unsqueeze(1).expand_as(ordered))
            pred = ordered[valid_mask].numpy()
            ep = np.broadcast_to(episode[:,None,None],(320,8,8))[valid_mask.numpy()]
            summary['random_init_D_control'] = dict(
                warning='Untrained random-init D, not a V3.1 learned predictor',
                vs_raw_js=episode_bootstrap_corr(pred,js[valid_mask].numpy(),ep),
                vs_stratum_residual=episode_bootstrap_corr(pred,residual[valid_mask].numpy(),ep))
            default_targets, _ = alignment_targets(
                js, valid, entropy, alt_count, counts, max_alternatives=2,
                effect_scale=1e-5, min_effect=1e-7)
            for i in range(320):
                for t in range(8):
                    for s in range(t):
                        if bool(valid[i,t,s]):
                            records_for_pair = dict(state=i, episode=int(episode[i]), target_slot=t,
                                source_slot=s, target_agent=int(order[i,t]), source_agent=int(order[i,s]),
                                js_nats=float(js[i,t,s]), residual_nats=float(residual[i,t,s]),
                                random_init_D=float(ordered[i,t,s]))
                            # Pair records are descriptive; state/episode are clustering units.
                            records_for_pair['default_y'] = float(default_targets[i,t,s])
                            records.append(records_for_pair)
    parameter_rows = [r for r in records if 'effect_scale' in r]
    pair_rows = [r for r in records if 'state' in r]
    with (OUT/f'signal_sensitivity{suffix}.csv').open('w',newline='') as f:
        fields = ['k','effect_scale','min_effect','supervised_pairs','supervised_fraction_valid',
                  'confidence_sum','effective_pair_count','effective_episode_count',
                  'target_near_half_005','target_near_half_001']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(parameter_rows)
    with (OUT/f'signal_pairs{suffix}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pair_rows[0]));writer.writeheader();writer.writerows(pair_rows)
    (OUT/f'signal_quality{suffix}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(default=summary['default_label_example'],
                          k2=summary['settings']['2'],
                          random_control=summary['random_init_D_control']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--v31-init',action='store_true')
    run(parser.parse_args().v31_init)
