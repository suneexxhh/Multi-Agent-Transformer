"""CPU-only V2 checkpoint influence timeline on the fixed real 3s5z dataset.

No checkpoint or source data is modified. Checkpoint 0 is AFTER PPO update 0.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mat.algorithms.mat.algorithm.agent_ordering import reorder_agents
from mat.algorithms.mat.algorithm.influence_alignment import alignment_targets, conditional_action_js
from research.dependency_decoder.analyze_v31_signal import DATA, RUN, load_teacher
from research.dependency_decoder.signal_diagnostics_core import episode_bootstrap, residuals


OUT = Path(os.environ.get('V32_ANALYSIS_DIR',
    '/data/sgg/mat-decoder-storage/analysis/v32_signal_onset_3s5z_seed1'))
CHECKPOINTS = (0, 5, 10, 15, 19)
SOURCE_DATA_SHA256 = '941c3574df347680c47183e2e36c6bc07d57f720045eec3df3dfc39add351aec'
V2_TRAINING_SHA = '90e20b828c1047bca1d47b3482c0ce389c4a4183'
FIELDS = ('checkpoint', 'completed_ppo_updates', 'approx_environment_steps',
          'order_condition', 'k', 'valid_pairs', 'possible_precedence_pairs',
          'valid_fraction', 'js_mean', 'js_std', 'js_q05', 'js_q25',
          'js_median', 'js_q75', 'js_q95', 'js_q99', 'js_max',
          'js_near_zero_1e12', 'js_below_min_effect_1e7',
          'js_mean_episode_bootstrap_lo', 'js_mean_episode_bootstrap_hi',
          'residual_mean', 'residual_std', 'residual_abs_median',
          'residual_abs_q95', 'residual_abs_max', 'supervised_pairs',
          'supervised_fraction_valid', 'supervised_fraction_bootstrap_lo',
          'supervised_fraction_bootstrap_hi', 'confidence_sum',
          'confidence_pair_ess', 'confidence_episode_ess',
          'y_mean_all_valid', 'y_std_all_valid', 'y_q05', 'y_median',
          'y_q95', 'y_near_half_005', 'y_mean_supervised',
          'y_std_supervised', 'checkpoint_sha256')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(js, valid, entropy, alternatives, legal_counts, episode, checkpoint,
              order_condition, k, checkpoint_hash):
    target, confidence = alignment_targets(
        js, valid, entropy, alternatives, legal_counts,
        max_alternatives=k, effect_scale=1e-5, min_effect=1e-7)
    res = residuals(js, valid, entropy, legal_counts)
    precedence = torch.tril(torch.ones(8, 8, dtype=torch.bool), diagonal=-1)[None]
    mask = valid & precedence
    j = js[mask].numpy().astype(float)
    r = res[mask].numpy().astype(float)
    y = target[mask].numpy().astype(float)
    w = confidence[mask].numpy().astype(float)
    eps = np.broadcast_to(episode[:, None, None], (len(episode), 8, 8))[mask.numpy()]
    if j.size == 0 or not np.isfinite(j).all() or not np.isfinite(r).all():
        raise ValueError('missing or non-finite observed teacher signal')
    ci_js = episode_bootstrap(j, eps)
    ci_cover = episode_bootstrap((w > 0).astype(float), eps)
    episode_w = np.array([w[eps == e].sum() for e in np.unique(episode)])
    observed = w > 0
    row = dict(checkpoint=checkpoint, completed_ppo_updates=checkpoint+1,
        approx_environment_steps=(checkpoint+1)*200,
        order_condition=order_condition, k=k,
        valid_pairs=int(mask.sum()), possible_precedence_pairs=len(episode)*28,
        valid_fraction=float(mask.sum()/(len(episode)*28)),
        js_mean=float(j.mean()), js_std=float(j.std()),
        js_q05=float(np.quantile(j,.05)), js_q25=float(np.quantile(j,.25)),
        js_median=float(np.median(j)), js_q75=float(np.quantile(j,.75)),
        js_q95=float(np.quantile(j,.95)), js_q99=float(np.quantile(j,.99)),
        js_max=float(j.max()),
        js_near_zero_1e12=float((j<1e-12).mean()),
        js_below_min_effect_1e7=float((j<1e-7).mean()),
        js_mean_episode_bootstrap_lo=ci_js[0], js_mean_episode_bootstrap_hi=ci_js[1],
        residual_mean=float(r.mean()), residual_std=float(r.std()),
        residual_abs_median=float(np.median(np.abs(r))),
        residual_abs_q95=float(np.quantile(np.abs(r),.95)),
        residual_abs_max=float(np.abs(r).max()),
        supervised_pairs=int(observed.sum()),
        supervised_fraction_valid=float(observed.mean()),
        supervised_fraction_bootstrap_lo=ci_cover[0],
        supervised_fraction_bootstrap_hi=ci_cover[1],
        confidence_sum=float(w.sum()),
        confidence_pair_ess=float(w.sum()**2/np.square(w).sum()) if np.square(w).sum() else 0.,
        confidence_episode_ess=float(episode_w.sum()**2/np.square(episode_w).sum())
        if np.square(episode_w).sum() else 0.,
        y_mean_all_valid=float(y.mean()), y_std_all_valid=float(y.std()),
        y_q05=float(np.quantile(y,.05)), y_median=float(np.median(y)),
        y_q95=float(np.quantile(y,.95)),
        y_near_half_005=float((np.abs(y-.5)<.05).mean()),
        y_mean_supervised=float(y[observed].mean()) if observed.any() else np.nan,
        y_std_supervised=float(y[observed].std()) if observed.any() else np.nan,
        checkpoint_sha256=checkpoint_hash)
    return row


def run():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT/'raw_teacher'
    raw_dir.mkdir(exist_ok=True)
    if sha256(DATA) != SOURCE_DATA_SHA256:
        raise ValueError('real SMAC observation checksum differs from the fixed dataset')
    d = np.load(DATA, allow_pickle=False)
    obs, legal, historical_actions, episode = (d[x] for x in
        ('obs','avail_actions','sampled_legal_actions','episode'))
    if obs.shape!=(320,8,252) or legal.shape!=(320,8,14) or historical_actions.shape!=(320,8):
        raise ValueError('real 3s5z source data lacks required fields')
    if len(np.unique(episode)) != 8 or not legal[np.arange(320)[:,None],
                                                 np.arange(8),historical_actions].all():
        raise ValueError('episode or legal historical action validation failed')
    model, manifest, _ = load_teacher()
    if manifest['training_source_sha'] != V2_TRAINING_SHA:
        raise ValueError('V2 training source SHA differs from the fixed experiment')
    observation = torch.from_numpy(obs)
    available = torch.from_numpy(legal).bool()
    actions = torch.from_numpy(historical_actions.astype(np.int64))[:,:,None]
    identity = torch.arange(8).expand(320,8)
    rows=[]
    metadata=dict(source_data=str(DATA),source_data_sha256=sha256(DATA),
                  training_source_sha=manifest['training_source_sha'],
                  states=320, episodes=8, map='3s5z', cpu_only=True,
                  history='identical recorded uniform-legal collection actions for all checkpoints',
                  fixed_control='original agent identity permutation for every state and checkpoint',
                  self_order='each checkpoint deterministic score sort',
                  decoder='frozen Bias OFF; same MAT conditional decoder',
                  checkpoint0='after first PPO update, NOT an untrained initialization',
                  checkpoint_hashes={})
    for checkpoint in CHECKPOINTS:
        path=RUN/'models'/f'transformer_{checkpoint}.pt'
        if not path.is_file():
            raise FileNotFoundError(path)
        ckhash=sha256(path)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        model.eval().requires_grad_(False)
        metadata['checkpoint_hashes'][str(checkpoint)]=ckhash
        with torch.no_grad():
            self_order=torch.cat([torch.argsort(model.order_policy.scores(
                observation[i:i+40]),dim=-1,descending=True,stable=True)
                for i in range(0,320,40)])
        for order_name,order in (('identity_fixed',identity),('own_deterministic',self_order)):
            legal_counts=reorder_agents(available,order).sum(-1)
            for k in (1,2):
                parts=[[],[],[],[]]
                with torch.no_grad():
                    for start in range(0,320,40):
                        raw=conditional_action_js(model,observation[start:start+40],
                            actions[start:start+40],available[start:start+40],
                            order[start:start+40],max_alternatives=k)
                        for stack,value in zip(parts,raw):
                            stack.append(value)
                js,valid,entropy,alt=[torch.cat(item) for item in parts]
                row=summarize(js,valid,entropy,alt,legal_counts,episode,
                              checkpoint,order_name,k,ckhash)
                rows.append(row)
                np.savez_compressed(raw_dir/f'checkpoint_{checkpoint}_{order_name}_k{k}.npz',
                    js=js.numpy(),valid=valid.numpy(),entropy=entropy.numpy(),
                    alt_count=alt.numpy(),legal_counts=legal_counts.numpy(),
                    order=order.numpy(),episode=episode)
                print(f'checkpoint={checkpoint} order={order_name} K={k} '
                      f'JS={row["js_mean"]:.3g} confidence={row["confidence_sum"]:.3g}',flush=True)
    with (OUT/'influence_signal_timeline.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    (OUT/'timeline_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for row_index,k in enumerate((1,2)):
        for label,color in (('identity_fixed','tab:blue'),('own_deterministic','tab:orange')):
            subset=sorted((r for r in rows if r['k']==k and r['order_condition']==label),
                          key=lambda r:r['checkpoint'])
            x=np.array([r['completed_ppo_updates'] for r in subset])
            y=np.array([r['js_mean'] for r in subset])
            lo=np.array([r['js_mean_episode_bootstrap_lo'] for r in subset])
            hi=np.array([r['js_mean_episode_bootstrap_hi'] for r in subset])
            axes[row_index,0].plot(x,y,marker='o',label=label,color=color)
            axes[row_index,0].fill_between(x,lo,hi,alpha=.17,color=color)
            axes[row_index,1].plot(x,[r['supervised_fraction_valid'] for r in subset],
                                   marker='o',label=label,color=color)
        axes[row_index,0].set_ylabel(f'K={k}: mean conditional JS (nats)')
        axes[row_index,1].set_ylabel(f'K={k}: confidence>0 / valid')
        axes[row_index,0].set_yscale('log')
        axes[row_index,1].set_ylim(-.02,1.02)
        axes[row_index,0].legend(fontsize=8)
    for ax in axes[-1]: ax.set_xlabel('completed PPO updates (checkpoint 0 = 1)')
    fig.suptitle('V2 3s5z frozen Bias-OFF conditional influence timeline')
    fig.tight_layout()
    fig.savefig(OUT/'influence_signal_timeline.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    run()
