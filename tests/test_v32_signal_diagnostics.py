"""CPU tests for temporal signal and minibatch-label diagnostic logic."""
import unittest

import numpy as np
import torch
from torch import nn

from mat.algorithms.mat.algorithm.influence_alignment import (
    alignment_targets, categorical_js, influence_alignment_loss)
from research.dependency_decoder.signal_diagnostics_core import (
    episode_bootstrap, partition_labels, residuals)


class ToyDemand(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3, 3))

    def demand_logits(self, obs):
        return self.logits[None].expand(obs.shape[0], -1, -1)


class TestV32SignalDiagnostics(unittest.TestCase):
    def setUp(self):
        self.js=torch.zeros(4,3,3)
        for t in range(1,3):
            for s in range(t):
                self.js[:,t,s]=torch.tensor([1.,2.,4.,8.])*1e-5*(t+s)
        self.valid=torch.tril(torch.ones(3,3,dtype=torch.bool),-1)[None].expand(4,-1,-1)
        self.entropy=torch.ones(4,3)
        self.alternatives=torch.full((4,3),2)
        self.counts=torch.full((4,3),3)

    def test_episode_bootstrap_keeps_pairs_together(self):
        data=np.array([0.]*40+[1.]*40)
        episode=np.array([0]*40+[1]*40)
        lo,hi=episode_bootstrap(data,episode,repeats=1000,seed=41)
        self.assertEqual((lo,hi),(0.,1.))
        with self.assertRaisesRegex(ValueError,'weights'):
            episode_bootstrap(data,episode,np.full(80,-1.))

    def test_partition_changes_targets_but_order_only_does_not(self):
        args=(self.js,self.valid,self.entropy,self.alternatives,self.counts)
        global_y,global_w=alignment_targets(*args,max_alternatives=2)
        one_y,one_w,singletons=partition_labels(*args,[np.arange(4)])
        self.assertEqual(singletons,0)
        self.assertTrue(torch.equal(global_y,one_y))
        self.assertTrue(torch.equal(global_w,one_w))
        shuffled_y,shuffled_w,_=partition_labels(*args,[np.array([3,2,1,0])])
        self.assertTrue(torch.allclose(global_y,shuffled_y,atol=1e-7,rtol=0))
        self.assertTrue(torch.allclose(global_w,shuffled_w,atol=1e-7,rtol=0))
        split_y,split_w,_=partition_labels(*args,[np.array([0,1]),np.array([2,3])])
        self.assertGreater(float((split_y-global_y).abs().max()),.05)
        self.assertGreater(float((split_w-global_w).abs().max()),.01)
        self.assertFalse(split_w[~self.valid].any())

    def test_singleton_strata_are_missing_supervision(self):
        args=(self.js,self.valid,self.entropy,self.alternatives,self.counts)
        y,w,singletons=partition_labels(*args,[np.array([i]) for i in range(4)])
        self.assertEqual(singletons,int(self.valid.sum()))
        self.assertFalse((w>0).any())
        self.assertTrue(torch.equal(y,torch.full_like(y,.5)))
        with self.assertRaisesRegex(ValueError,'partition'):
            partition_labels(*args,[np.array([0,1]),np.array([1,2,3])])

    def test_missing_directions_and_residual_centering(self):
        r=residuals(self.js,self.valid,self.entropy,self.counts)
        self.assertTrue(torch.isnan(r[~self.valid]).all())
        for t in range(1,3):
            for s in range(t):
                self.assertAlmostEqual(float(r[:,t,s].sum()),0.,places=9)

    def test_tiny_positive_confidence_is_not_a_small_loss(self):
        # V3.1 normalizes by sum(confidence): a single tiny positive weight
        # can still yield a full auxiliary gradient. Activation must check
        # coverage/support; lowering min_effect alone does not solve this.
        student=ToyDemand()
        obs=torch.zeros(1,3,2)
        order=torch.arange(3)[None]
        y=torch.full((1,3,3),.5);y[0,1,0]=.9
        one=torch.zeros_like(y);one[0,1,0]=1.
        weak=one*1e-12
        a=influence_alignment_loss(student,obs,order,y,one)
        b=influence_alignment_loss(student,obs,order,y,weak)
        self.assertTrue(torch.allclose(a,b,atol=1e-7,rtol=0))
        ga=torch.autograd.grad(a,student.logits,retain_graph=True)[0]
        gb=torch.autograd.grad(b,student.logits)[0]
        self.assertTrue(torch.allclose(ga,gb,atol=1e-7,rtol=0))
        self.assertEqual(float(ga[0,1]),0.)

    def test_js_remains_finite_for_small_logit_effect(self):
        base=torch.tensor([[0.,0.]])
        changed=torch.tensor([[1e-6,0.]])
        js=categorical_js(base,changed,torch.ones_like(base,dtype=torch.bool))
        self.assertTrue(torch.isfinite(js).all())
        self.assertGreater(float(js),0.)
        self.assertLess(float(js),1e-10)


if __name__=='__main__':
    unittest.main()
