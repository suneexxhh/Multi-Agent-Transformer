"""Controlled CPU gradient-path probe; synthetic PPO, no optimizer step."""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from mat.algorithms.mat.algorithm.influence_alignment import influence_alignment_loss
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer


OUT=Path(os.environ.get('V31_ANALYSIS_DIR','/data/sgg/mat-decoder-storage/analysis/v31_influence_signal_validation'))


def vector(grads, params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for g,p in zip(grads,params)])


def norm(x):
    return float(x.norm())


def cosine(a,b):
    denom=float(a.norm()*b.norm())
    return float(torch.dot(a,b)/denom) if denom>0 else None


def run():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(91131)
    model=MultiAgentTransformer(37,252,14,8,n_block=1,n_embd=64,n_head=1,
        agent_order_mode='dependency_prefix',order_attention_bias=True,
        adaptive_order_attention_bias=True,order_hidden_dim=64,order_pair_rank=16)
    model.eval()
    b=12
    obs=torch.randn(b,8,252)
    state=torch.zeros(b,8,37)
    legal=torch.ones(b,8,14)
    legal[:,:,9:]=0
    with torch.no_grad():
        action,_,_,order,old_lp=model.get_actions(state,obs,legal,
            return_agent_order=True,return_order_log_prob=True)
    demand_params=[model.order_policy.demand_query.weight,
                   model.order_policy.demand_key.weight]
    rho=model.order_policy.raw_attention_scale
    originals={k:v.detach().clone() for k,v in model.state_dict().items()}
    team_adv=torch.linspace(-1,1,b)[:,None]

    def order_loss():
        current=model.order_policy.log_prob(obs,order)
        ratio=(current-old_lp).exp()
        clipped=ratio.clamp(.95,1.05)
        return -.1*torch.minimum(ratio*team_adv,clipped*team_adv).mean()

    # Controlled nonconstant labels isolate the desired gradient pathway.
    target=torch.full((b,8,8),.5)
    confidence=torch.zeros_like(target)
    for t in range(1,8):
        for s in range(t):
            target[:,t,s]=.2+.6*((torch.arange(b)+t+2*s)%5).float()/4
            confidence[:,t,s]=.3+.7*((torch.arange(b)+s)%3).float()/2

    def align_loss():
        return .2*influence_alignment_loss(model.order_policy,obs,order,target,confidence)

    def action_loss():
        logp,_,_=model(state,obs,action,legal,agent_order=order)
        return -logp.mean()

    def grad_for(loss):
        return torch.autograd.grad(loss,[*demand_params,rho],allow_unused=True)

    with torch.no_grad():
        logp0=model(state,obs,action,legal,agent_order=order)[0]
    g_order=grad_for(order_loss())
    g_align=grad_for(align_loss())
    g_att_zero=grad_for(action_loss())
    with torch.no_grad():
        rho.fill_(math.atanh(.5/2.0))  # effective rho = 0.5 with maximum 2.
        logp_half=model(state,obs,action,legal,agent_order=order)[0]
    g_att_half=grad_for(action_loss())
    demand_order=vector(g_order[:2],demand_params)
    demand_align=vector(g_align[:2],demand_params)
    demand_attention0=vector(g_att_zero[:2],demand_params)
    demand_attention_half=vector(g_att_half[:2],demand_params)
    output=dict(seed=91131,synthetic_ppo_states=b,agents=8,
        intervention='No optimizer.step or environment interactions. Alignment targets/confidence deliberately controlled, not teacher estimates.',
        effective_rho_init=0.,effective_rho_after=.5,
        gradients={
            'order_ppo':dict(query=norm(g_order[0]),key=norm(g_order[1]),
                             demand_total=norm(demand_order),raw_scale=norm(g_order[2]) if g_order[2] is not None else 0.),
            'influence_alignment':dict(query=norm(g_align[0]),key=norm(g_align[1]),
                             demand_total=norm(demand_align),raw_scale=norm(g_align[2]) if g_align[2] is not None else 0.),
            'attention_rho0':dict(query=norm(g_att_zero[0]) if g_att_zero[0] is not None else 0.,
                             key=norm(g_att_zero[1]) if g_att_zero[1] is not None else 0.,
                             demand_total=norm(demand_attention0),
                             raw_scale=norm(g_att_zero[2]) if g_att_zero[2] is not None else 0.),
            'attention_rho05':dict(query=norm(g_att_half[0]) if g_att_half[0] is not None else 0.,
                             key=norm(g_att_half[1]) if g_att_half[1] is not None else 0.,
                             demand_total=norm(demand_attention_half),
                             raw_scale=norm(g_att_half[2]) if g_att_half[2] is not None else 0.)},
        cosines=dict(order_vs_alignment=cosine(demand_order,demand_align),
                     order_vs_attention_rho05=cosine(demand_order,demand_attention_half),
                     alignment_vs_attention_rho05=cosine(demand_align,demand_attention_half)),
        alignment_to_order_gradient_ratio=norm(demand_align)/norm(demand_order),
        logp_abs_change_rho05=dict(mean=float((logp_half-logp0).abs().mean()),
                                   max=float((logp_half-logp0).abs().max())))
    # rho is the only intentionally changed parameter; weights and buffers are otherwise invariant.
    for key,value in model.state_dict().items():
        if key != 'order_policy.raw_attention_scale' and not torch.equal(value,originals[key]):
            raise AssertionError('gradient probe mutated a model tensor: '+key)
    if not all(np.isfinite(v) for d in output['gradients'].values() for v in d.values()):
        raise ValueError('non-finite controlled gradient')
    (OUT/'gradient_probe.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    run()
