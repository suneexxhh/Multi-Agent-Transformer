"""CPU-only 10-epoch PPO teacher cost profile; no SMAC environment or GPU."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.mat.algorithm import influence_alignment as align
from mat.algorithms.mat.mat_trainer import MATTrainer


OUT = Path(os.environ.get('V31_ANALYSIS_DIR', '/data/sgg/mat-decoder-storage/analysis/v31_influence_signal_validation'))
DATA = Path('/data/sgg/mat-decoder-storage/analysis/v2_dependency_mechanism_3s5z_seed1/real_smac_observations.npz')
CASES = ('no_align', 'k1', 'k2', 'k1_cache', 'k2_cache')


def args(case):
    k = 1 if 'k1' in case else 2
    return SimpleNamespace(clip_param=.05, ppo_epoch=10, num_mini_batch=2,
        data_chunk_length=1, value_loss_coef=1., entropy_coef=.01,
        max_grad_norm=10., huber_delta=10., use_recurrent_policy=False,
        use_naive_recurrent_policy=False, use_max_grad_norm=True,
        use_clipped_value_loss=False, use_huber_loss=False,
        use_valuenorm=False, use_value_active_masks=False,
        use_policy_active_masks=False, dec_actor=False,
        agent_order_mode='dependency_prefix', order_loss_coef=.1,
        order_influence_align_coef=0. if case=='no_align' else .2,
        order_influence_max_alternatives=k,
        order_influence_effect_scale=1e-5,
        order_influence_min_effect=1e-7,
        cache_order_influence_teacher='cache' in case)


def make_policy(model):
    policy = TransformerPolicy.__new__(TransformerPolicy)
    policy.algorithm_name, policy.num_agents = 'mat', 8
    policy.share_obs_dim, policy.obs_dim, policy.act_dim, policy.act_num = 37, 252, 14, 1
    policy._use_policy_active_masks = False
    policy.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
    policy.transformer = model
    policy.optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    return policy


class FixedRollout:
    def __init__(self, fields, advantages):
        self.fields = fields
        self.advantages = advantages.reshape(len(advantages), 1, 8, 1)
        self.active_masks = np.ones((len(advantages)+1, 1, 8, 1), np.float32)

    def feed_forward_generator_transformer(self, advantages, num_mini_batch,
                                           include_transition_indices=False):
        count = len(self.fields[0])
        for ids in torch.randperm(count).reshape(num_mini_batch, -1).numpy():
            fields = list(self.fields)
            fields[10] = advantages.reshape(count, 8, 1)
            sample = tuple(field[ids].reshape(-1, *field.shape[2:]) if j < 12
                           else field[ids].copy() for j, field in enumerate(fields))
            if include_transition_indices:
                sample += (ids.copy(),)
            yield sample


def build_rollout(initial, count=16):
    data = np.load(DATA, allow_pickle=False)
    obs = np.asarray(data['obs'][:count], dtype=np.float32)
    legal = np.asarray(data['avail_actions'][:count], dtype=np.float32)
    center = np.zeros((count, 8, 37), np.float32)
    with torch.no_grad():
        action, lp, value, order, order_lp = initial.get_actions(
            center, obs, legal, deterministic=False,
            return_agent_order=True, return_order_log_prob=True)
    value = value.numpy().reshape(count, 8, 1)
    rng = np.random.default_rng(1337)
    advantages = rng.normal(size=(count, 8, 1)).astype(np.float32)
    fields = (center, obs, np.zeros((count,8,1),np.float32),
        np.zeros((count,8,1),np.float32), action.numpy().reshape(count,8,1),
        value, value+advantages, np.ones((count,8,1),np.float32),
        np.ones((count,8,1),np.float32), lp.numpy().reshape(count,8,1),
        advantages, legal, order.numpy(), order_lp.numpy())
    return FixedRollout(fields, advantages)


def profile(case):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(2)
    torch.manual_seed(31371)
    np.random.seed(31371)
    model = MultiAgentTransformer(37,252,14,8,n_block=1,n_embd=64,n_head=1,
        agent_order_mode='dependency_prefix', order_attention_bias=False,
        order_hidden_dim=64, order_pair_rank=16)
    rollout = build_rollout(model)
    policy = make_policy(copy.deepcopy(model))
    trainer = MATTrainer(args(case), policy, 8)
    timings = dict(snapshot=0., js=0., labels=0.)
    counts = dict(snapshot=0, js_calls=0, decoder_forward=0, teacher_states=0)
    original_snapshot, original_js, original_targets = (
        align.frozen_bias_off_teacher, align.conditional_action_js, align.alignment_targets)

    def snapshot(*a, **kw):
        start = time.perf_counter(); obj = original_snapshot(*a, **kw)
        timings['snapshot'] += time.perf_counter()-start;counts['snapshot']+=1
        return obj

    def js_probe(teacher, obs, *a, **kw):
        def count_forward(_model, _inputs, _output):
            counts['decoder_forward'] += 1
        hook = teacher.decoder.register_forward_hook(count_forward)
        start = time.perf_counter()
        try:
            result = original_js(teacher, obs, *a, **kw)
        finally:
            hook.remove();timings['js'] += time.perf_counter()-start
        counts['js_calls'] += 1;counts['teacher_states'] += int(obs.shape[0])
        return result

    def targets(*a, **kw):
        start = time.perf_counter()
        try:
            return original_targets(*a, **kw)
        finally:
            timings['labels'] += time.perf_counter()-start

    start = time.perf_counter()
    with patch.object(align,'frozen_bias_off_teacher',side_effect=snapshot), \
         patch.object(align,'conditional_action_js',side_effect=js_probe), \
         patch.object(align,'alignment_targets',side_effect=targets):
        info = trainer.train(rollout)
    total = time.perf_counter()-start
    teacher_time = sum(timings.values())
    student_time = total-teacher_time
    output = dict(case=case, total_ppo_update_seconds=total,
        per_minibatch_seconds=total/20, teacher_snapshot_seconds=timings['snapshot'],
        teacher_decoder_js_seconds=timings['js'], label_stratification_seconds=timings['labels'],
        student_and_ppo_seconds=student_time, teacher_to_student_ratio=teacher_time/student_time,
        teacher_fraction=teacher_time/total, teacher_calls=counts,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        config=dict(ppo_epochs=10, minibatches_per_epoch=2, rollout_states=16,
                    agents=8, obs_dim=252, action_dim=14, cpu_threads=2,
                    source='real V2 observations/legal masks, synthetic PPO returns/advantages and random-init V3.1'),
        train_metrics={k:float(v) for k,v in info.items()})
    (OUT/f'profile_{case}.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.case:
        profile(args.case)
    else:
        results=[]
        for case in CASES:
            proc=subprocess.run([sys.executable,__file__,'--case',case],
                capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            (OUT/f'profile_{case}.log').write_text(proc.stdout+proc.stderr)
            if proc.returncode:
                raise RuntimeError(f'{case} profile failed: {proc.stderr}')
            results.append(json.loads((OUT/f'profile_{case}.json').read_text()))
        (OUT/'teacher_profile.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps([{k:v for k,v in r.items() if k in (
            'case','total_ppo_update_seconds','teacher_fraction','teacher_to_student_ratio',
            'peak_rss_mib','teacher_calls')} for r in results],indent=2))


if __name__=='__main__':
    main()
