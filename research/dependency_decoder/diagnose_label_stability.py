"""Fixed-teacher label dependence on PPO minibatch membership, CPU only."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from mat.algorithms.mat.algorithm.agent_ordering import reorder_agents
from mat.algorithms.mat.algorithm.influence_alignment import alignment_targets
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from research.dependency_decoder.analyze_v31_signal import DATA
from research.dependency_decoder.signal_diagnostics_core import partition_labels
from research.dependency_decoder.trace_influence_signal import (
    SOURCE_DATA_SHA256, V2_TRAINING_SHA, sha256)


OUT = Path(os.environ.get('V32_ANALYSIS_DIR',
    '/data/sgg/mat-decoder-storage/analysis/v32_signal_onset_3s5z_seed1'))


def quantiles(values):
    a=np.asarray(values,dtype=float).reshape(-1)
    if not a.size:
        return dict(n=0,mean=None,median=None,q90=None,q95=None,max=None)
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
                q90=float(np.quantile(a,.9)),q95=float(np.quantile(a,.95)),
                max=float(a.max()))


def norm(x):
    return float(x.norm())


def cosine(x,y):
    n=x.norm()*y.norm()
    return float(torch.dot(x,y)/n) if n>0 else None


def run():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    if sha256(DATA) != SOURCE_DATA_SHA256:
        raise ValueError('real SMAC observation checksum differs from the fixed dataset')
    manifest=json.loads((OUT/'timeline_manifest.json').read_text())
    if (manifest['source_data_sha256'] != SOURCE_DATA_SHA256 or
            manifest['training_source_sha'] != V2_TRAINING_SHA):
        raise ValueError('raw teacher manifest uses a different dataset or V2 training source')
    d=np.load(DATA,allow_pickle=False)
    obs=torch.from_numpy(d['obs'])
    if obs.shape!=(320,8,252): raise ValueError('wrong real state source')
    torch.manual_seed(91421)
    student=MultiAgentTransformer(37,252,14,8,n_block=1,n_embd=64,n_head=1,
        agent_order_mode='dependency_prefix',order_attention_bias=False).eval()
    params=[student.order_policy.demand_query.weight,student.order_policy.demand_key.weight]
    reports={}
    cases=[(i,'identity_fixed') for i in (0,5,10,15,19)]+[(19,'own_deterministic')]
    for checkpoint,order_name in cases:
        path=OUT/'raw_teacher'/f'checkpoint_{checkpoint}_{order_name}_k2.npz'
        raw=np.load(path,allow_pickle=False)
        js=torch.from_numpy(raw['js']);valid=torch.from_numpy(raw['valid'])
        entropy=torch.from_numpy(raw['entropy']);alternatives=torch.from_numpy(raw['alt_count'])
        counts=torch.from_numpy(raw['legal_counts']);order=torch.from_numpy(raw['order'])
        if js.shape!=(320,8,8) or not torch.equal(
                torch.from_numpy(raw['episode']),torch.from_numpy(d['episode'])):
            raise ValueError('raw teacher data/state mismatch')
        lower=torch.tril(torch.ones(8,8,dtype=torch.bool),-1)[None]
        pair=valid&lower
        y0,w0=alignment_targets(js,valid,entropy,alternatives,counts,max_alternatives=2)
        supervised=pair&(w0>0)
        logits=student.order_policy.demand_logits(obs)
        ordered=reorder_agents(logits,order)
        ordered=ordered.gather(2,order.unsqueeze(1).expand_as(ordered))
        p=ordered.detach().sigmoid()

        def gradient(y,w):
            denom=w.sum()
            if not denom: return torch.zeros(sum(v.numel() for v in params))
            upstream=((p-y)*w/denom).detach()
            grad=torch.autograd.grad(ordered,params,grad_outputs=upstream,
                                     retain_graph=True)
            return torch.cat([x.detach().flatten() for x in grad])

        g0=gradient(y0,w0)
        size_reports={}
        for size in (8,16,40,80,160,200,320):
            labels=[];weights=[];grads=[];singleton=[]
            for rep in range(12):
                rng=np.random.default_rng(98731+rep)
                indices=rng.permutation(320)
                groups=[indices[i:i+size] for i in range(0,320,size)]
                y,w,single=partition_labels(js,valid,entropy,alternatives,counts,groups)
                labels.append(y.numpy());weights.append(w.numpy())
                grads.append(gradient(y,w));singleton.append(single)
            y=np.stack(labels);w=np.stack(weights)
            current=(w>0)&pair.numpy()[None]
            base=supervised.numpy()
            g=torch.stack(grads)
            cos=[cosine(gi,g0) for gi in g]
            source_y=y[:,base];source_w=w[:,base]
            rows=dict(minibatch_size=size,partitions=12,
                baseline_supervised_pairs=int(supervised.sum()),
                current_supervised_pairs=quantiles(current.sum(axis=(1,2,3))),
                lost_baseline_supervision_fraction=quantiles(
                    ((w[:,base]==0).mean(axis=1)) if base.any() else np.array([])),
                target_abs_change_on_baseline_supervised=quantiles(
                    np.abs(source_y-y0.numpy()[base][None])),
                target_std_across_partitions=quantiles(source_y.std(axis=0)),
                confidence_abs_change_on_baseline_supervised=quantiles(
                    np.abs(source_w-w0.numpy()[base][None])),
                confidence_std_across_partitions=quantiles(source_w.std(axis=0)),
                singleton_valid_pair_count=quantiles(singleton),
                gradient_norm=quantiles([norm(gi) for gi in g]),
                gradient_cosine_to_rollout_labels=quantiles(
                    [c for c in cos if c is not None]),
                gradient_relative_distance_to_rollout=quantiles(
                    [norm(gi-g0)/norm(g0) for gi in g]) if norm(g0)>0 else None)
            if size==320:
                # Changing order alone cannot change the mathematical target.
                rows['full_rollout_permutation_max_target_abs_diff']=float(
                    np.abs(y-y0.numpy()[None]).max())
                rows['full_rollout_permutation_max_confidence_abs_diff']=float(
                    np.abs(w-w0.numpy()[None]).max())
            size_reports[str(size)]=rows
        # The shipped SMAC launcher uses 100 steps * 2 threads and exactly
        # one minibatch, so each epoch only permutes the SAME 200 states.
        subset=np.random.default_rng(91422).choice(320,200,replace=False)
        subset=torch.as_tensor(subset,dtype=torch.long)
        sy,sw,_=partition_labels(js[subset],valid[subset],entropy[subset],
            alternatives[subset],counts[subset],[np.arange(200)])
        order_only_y=[];order_only_w=[]
        for rep in range(12):
            permutation=np.random.default_rng(99100+rep).permutation(200)
            py,pw,_=partition_labels(js[subset],valid[subset],entropy[subset],
                alternatives[subset],counts[subset],[permutation])
            order_only_y.append(float((py-sy).abs().max()))
            order_only_w.append(float((pw-sw).abs().max()))
        reports[f'checkpoint_{checkpoint}_{order_name}']=dict(
            rollout_supervised_pairs=int(supervised.sum()),
            rollout_confidence_sum=float(w0.sum()),
            rollout_gradient_norm=norm(g0),
            actual_launcher_200_states_one_minibatch=dict(
                max_target_abs_change_across_orderings=max(order_only_y),
                max_confidence_abs_change_across_orderings=max(order_only_w),
                permutations=12),
            minibatches=size_reports)
        print(f'checkpoint={checkpoint}, order={order_name}, supervised={int(supervised.sum())}, '
              f'mini8 lost={size_reports["8"]["lost_baseline_supervision_fraction"]["mean"]}',flush=True)
    output=dict(dataset=str(DATA),states=320,episodes=8,
                teacher='fixed raw Bias-OFF V2 signals saved by timeline analysis',
                student='same random-init V3.1 demand network for all partitions; no optimizer step',
                method='12 seeded full-dataset permutations per minibatch size; same raw teacher results',
                reference='one rollout-level alignment_targets pass over all 320 states',
                current_smac_launcher=dict(episode_length=100,rollout_threads=2,
                                           transitions_per_rollout=200,num_mini_batch=1,
                                           ppo_epoch=10),
                result=reports)
    (OUT/'minibatch_label_stability.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    run()
