"""Trace legal source-action perturbations through the real MAT Decoder on CPU."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.agent_ordering import reorder_agents
from mat.algorithms.mat.algorithm.influence_alignment import categorical_js, frozen_bias_off_teacher
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from research.dependency_decoder.analyze_v31_signal import DATA, RUN, load_teacher
from research.dependency_decoder.trace_influence_signal import (
    SOURCE_DATA_SHA256, V2_TRAINING_SHA, sha256)


OUT = Path(os.environ.get('V32_ANALYSIS_DIR',
    '/data/sgg/mat-decoder-storage/analysis/v32_signal_onset_3s5z_seed1'))


def distribution(values):
    x=np.asarray(values,dtype=float).reshape(-1)
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std()),
                median=float(np.median(x)),q95=float(np.quantile(x,.95)),
                max=float(x.max()))


def modules(decoder):
    block=decoder.blocks[0]
    return dict(action_embedding=decoder.action_encoder, action_layer_norm=decoder.ln,
                causal_self_attention=block.attn1, after_self_attention=block.ln1,
                cross_attention=block.attn2, after_cross_attention=block.ln2,
                decoder_mlp=block.mlp, after_decoder_block=block.ln3,
                head_hidden_linear=decoder.head[0], output_logits=decoder.head[3])


def decoder_with_captures(model, shifted, rep, obs):
    captured={}
    hooks=[]
    for name,module in modules(model.decoder).items():
        hooks.append(module.register_forward_hook(
            lambda _module,_inputs,output,name=name: captured.__setitem__(name,output.detach().clone())))
    try:
        logits=model.decoder(shifted,rep,obs,attention_bias=None)
    finally:
        for hook in hooks: hook.remove()
    return logits,captured


def trace(model, obs, legal, actions, order):
    model.eval().requires_grad_(False)
    dtype=next(model.parameters()).dtype
    obs=torch.as_tensor(obs,dtype=dtype)
    legal=torch.as_tensor(legal).bool()
    actions=torch.as_tensor(actions,dtype=torch.long)
    order=torch.as_tensor(order,dtype=torch.long)
    obs,legal,actions=[reorder_agents(v,order) for v in (obs,legal,actions)]
    batch,n,_=obs.shape
    with torch.no_grad():
        _,rep=model.encoder(obs.new_zeros(batch,n,37),obs)
        shifted=obs.new_zeros(batch,n,15)
        shifted[:,0,0]=1.
        shifted[:,1:,1:]=F.one_hot(actions[:,:-1,0],14).to(dtype)
        # Same middle legal alternative as conditional_action_js(K=1).
        alternatives=legal[:,0] & ~F.one_hot(actions[:,0,0],14).bool()
        ids=torch.arange(14).expand(batch,14).masked_fill(~alternatives,14).sort(-1).values
        count=alternatives.sum(-1)
        chosen=ids.gather(1,((count-1).clamp_min(0)//2)[:,None]).squeeze(1)
        source_valid=(count>0)&(chosen<14)
        chosen=torch.where(source_valid,chosen,actions[:,0,0])
        replacement=shifted.clone()
        replacement[:,1,1:]=F.one_hot(chosen,14).to(dtype)
        if not torch.equal(shifted[:,:1],replacement[:,:1]) or not torch.equal(
                shifted[:,2:],replacement[:,2:]):
            raise AssertionError('historical action perturbation leaked to other positions')
        base,base_layers=decoder_with_captures(model,shifted,rep,obs)
        changed,changed_layers=decoder_with_captures(model,replacement,rep,obs)
        variable=(legal.sum(-1)>1)
        target=(torch.arange(n)[None,:]>0)&variable&source_valid[:,None]
        immediate=target&(torch.arange(n)[None,:]==1)
        later=target&(torch.arange(n)[None,:]>=2)
        if not target.any(): raise ValueError('no legal conditional target')
        js=categorical_js(base,changed,legal)[target]
        bp=F.softmax(base.double().masked_fill(~legal,-torch.inf),-1)
        cp=F.softmax(changed.double().masked_fill(~legal,-torch.inf),-1)
        tv=(bp-cp).abs().sum(-1).mul(.5)[target]
        logits=(base-changed).abs().masked_fill(~legal,0.)[target]
        stage={}
        for name in modules(model.decoder):
            b=base_layers[name];c=changed_layers[name]
            delta=(b-c).float()
            if name=='output_logits':
                b=b.masked_fill(~legal,0.);delta=delta.masked_fill(~legal,0.)
            target_rms=delta[target].square().mean(-1).sqrt()
            reference=b.float()[target].square().mean(-1).sqrt()
            key_rms=delta[source_valid,1].square().mean(-1).sqrt()
            stage[name]=dict(target_rms=distribution(target_rms.numpy()),
                             target_relative_rms_mean=float((target_rms/
                                 reference.clamp_min(1e-30)).mean()),
                             immediate_target_rms=distribution(
                                 delta[immediate].square().mean(-1).sqrt().numpy()),
                             later_target_rms=distribution(
                                 delta[later].square().mean(-1).sqrt().numpy()),
                             source_shifted_key_rms=distribution(key_rms.numpy()))
        return dict(valid_target_pairs=int(target.sum()),valid_source_states=int(source_valid.sum()),
            js_nats=distribution(js.numpy()),tv=distribution(tv.numpy()),
            legal_logit_abs=distribution(logits.numpy()),
            legal_logit_max_per_pair=distribution(logits.max(-1).values.numpy()),
            float32_logits_identical_fraction=float((logits.max(-1).values==0).float().mean()),
            stages=stage,
            parameter_norms={name:float(p.detach().norm()) for name,p in model.decoder.named_parameters()
                             if name in ('action_encoder.0.weight','blocks.0.attn1.proj.weight',
                                         'blocks.0.attn2.proj.weight','head.0.weight','head.3.weight')})


def run():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    if sha256(DATA) != SOURCE_DATA_SHA256:
        raise ValueError('real SMAC observation checksum differs from the fixed dataset')
    data=np.load(DATA,allow_pickle=False)
    obs=data['obs'];legal=data['avail_actions']
    history=data['sampled_legal_actions'].astype(np.int64)[:,:,None]
    if obs.shape!=(320,8,252) or legal.shape!=(320,8,14):
        raise ValueError('real SMAC state input shape mismatch')
    if not legal[np.arange(320)[:,None],np.arange(8),history[:,:,0]].all():
        raise ValueError('real collection history has illegal action')
    identity=np.tile(np.arange(8),(320,1))
    cases={}
    torch.manual_seed(99117)
    initial=MultiAgentTransformer(37,252,14,8,n_block=1,n_embd=64,n_head=1,
        agent_order_mode='dependency_prefix',order_attention_bias=True,
        adaptive_order_attention_bias=True).eval()
    with torch.no_grad():
        sampled=[initial.get_actions(torch.zeros(40,8,37),torch.from_numpy(obs[i:i+40]),
            torch.from_numpy(legal[i:i+40]).float(),return_agent_order=True,
            return_order_log_prob=True) for i in range(0,320,40)]
    initial_history=torch.cat([x[0] for x in sampled]).numpy()
    initial_order=torch.cat([x[3] for x in sampled]).numpy()
    if not legal[np.arange(320)[:,None],np.arange(8),initial_history[:,:,0]].all():
        raise ValueError('initial policy sampled illegal history action')
    v31_teacher=frozen_bias_off_teacher(initial)
    cases['v31_initial_own_order_and_policy_actions']=(v31_teacher,initial_history,initial_order)
    cases['v31_initial_identity_and_collection_actions']=(v31_teacher,history,identity)
    v2,manifest,_=load_teacher()
    if manifest['training_source_sha'] != V2_TRAINING_SHA:
        raise ValueError('V2 training source SHA differs from the fixed experiment')
    for index in (0,5,10,19):
        checkpoint=RUN/'models'/f'transformer_{index}.pt'
        v2.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        cases[f'v2_checkpoint_{index}_identity_and_collection_actions']=(copy.deepcopy(v2),history,identity)
    report=dict(source_data=str(DATA),data_states=320,episodes=8,
        diagnostic='source decoded position 0; one legal middle alternative, targets at later positions',
        all_bias_off=True,all_parameters_frozen=True,
        v2_training_source_sha=manifest['training_source_sha'],
        cases={})
    for name,(model,acts,order) in cases.items():
        f32=trace(model,obs,legal,acts,order)
        f64=trace(copy.deepcopy(model).double(),obs,legal,acts,order)
        report['cases'][name]=dict(float32=f32,float64=f64,
             js64_to_js32_mean_ratio=f64['js_nats']['mean']/max(f32['js_nats']['mean'],1e-300),
             float64_large_when_float32_nearzero=(f32['js_nats']['mean']<1e-10
                                                  and f64['js_nats']['mean']>1e-8))
        print(name,'JS32',f32['js_nats']['mean'],'JS64',f64['js_nats']['mean'],
              'logit delta',f32['legal_logit_abs']['mean'],flush=True)
    (OUT/'initialization_conditioning_diagnostics.json').write_text(
        json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
