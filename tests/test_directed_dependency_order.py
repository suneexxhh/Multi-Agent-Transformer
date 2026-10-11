"""CPU-only checks for reward-trained directed precedence and MAT PPO transport.

These tests are synthetic and do not claim causal discovery or SMAC gains.
"""
import itertools
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from mat.algorithms.mat.algorithm.directed_dependency_order import DirectedDependencyOrder
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.mat.mat_trainer import MATTrainer


class TestDirectedDependencyOrdering(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(418)

    def test_pairwise_antisymmetry_and_relabelling_equivariance(self):
        m = DirectedDependencyOrder(7, hidden_dim=24, pair_rank=8)
        obs = torch.randn(3, 6, 7)
        p = m.pairwise_preferences(obs)
        self.assertTrue(torch.allclose(p+p.transpose(1,2), torch.ones_like(p), atol=1e-6))
        self.assertTrue(torch.allclose(p.diagonal(dim1=1,dim2=2),
                                       torch.full((3,6), 0.5), atol=1e-6))
        ids = torch.tensor([2,4,1,0,5,3])
        relabelled = obs[:,ids]
        self.assertTrue(torch.allclose(m.scores(relabelled), m.scores(obs)[:,ids],
                                       atol=2e-6))
        self.assertTrue(torch.allclose(m.pairwise_preferences(relabelled),
                                       p[:,ids][:,:,ids], atol=2e-6))

    def test_exact_pl_distribution_and_stored_likelihood(self):
        m = DirectedDependencyOrder(5, hidden_dim=16, pair_rank=8)
        obs = torch.randn(3,3,5)
        order, lp = m.sample(obs)
        self.assertEqual(order.shape, (3,3))
        self.assertTrue(torch.equal(order.sort(-1).values, torch.arange(3).expand(3,-1)))
        self.assertTrue(torch.allclose(lp, m.log_prob(obs, order), atol=1e-6))
        mass = sum(m.log_prob(obs[:1],torch.tensor([perm],dtype=torch.long)).exp()
                   for perm in itertools.permutations(range(3)))
        self.assertAlmostEqual(float(mass),1.,places=5)
        with self.assertRaises(ValueError):
            m.log_prob(obs,torch.zeros((3,3),dtype=torch.long))

    def test_projected_graph_is_acyclic_and_order_respecting(self):
        m = DirectedDependencyOrder(4, hidden_dim=16, pair_rank=8)
        obs = torch.randn(2,7,4)
        sampled, _ = m.sample(obs)
        edges, order = m.candidate_dag(obs, threshold=0.5001, order=sampled)
        self.assertTrue(torch.equal(order, sampled))
        self.assertEqual(edges.shape,(2,7,7))
        for b in range(2):
            ranks = {int(agent):pos for pos,agent in enumerate(order[b])}
            for i,j in torch.nonzero(edges[b], as_tuple=False).tolist():
                self.assertLess(ranks[i],ranks[j])
            reach=edges[b].clone()
            for k in range(7):
                reach = reach | (reach[:,k:k+1]&reach[k:k+1,:])
            self.assertFalse(bool(reach.diagonal().any()))
        with self.assertRaises(ValueError):
            m.candidate_dag(obs,threshold=0.4)

    def test_counterfactual_supervision_requires_both_directions(self):
        m=DirectedDependencyOrder(3, hidden_dim=12, pair_rank=6)
        obs=torch.randn(1,3,3)
        effect=torch.zeros(1,3,3)
        known=torch.zeros(1,3,3,dtype=torch.bool)
        effect[0,0,1],effect[0,1,0]=1.0,0.01
        known[0,0,1]=True
        with self.assertRaisesRegex(ValueError,"mutually observed"):
            m.counterfactual_alignment_loss(obs,effect,known)
        known[0,1,0]=True
        optimiser=torch.optim.Adam(m.parameters(),lr=0.015)
        beginning=float(m.counterfactual_alignment_loss(obs,effect,known))
        for _ in range(24):
            loss=m.counterfactual_alignment_loss(obs,effect,known)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        ending=float(m.counterfactual_alignment_loss(obs,effect,known))
        self.assertLess(ending,beginning)
        self.assertTrue(torch.isfinite(m.scores(obs)).all())

    def test_27_agents_memory_safe_and_pair_gradients(self):
        m=DirectedDependencyOrder(11, hidden_dim=20, pair_rank=4)
        obs=torch.randn(3,27,11)
        order,old_lp=m.sample(obs)
        selected_adv=torch.tensor([1.0,-1.0,2.0])
        loss=-(m.log_prob(obs,order).flatten()*selected_adv).mean()
        loss.backward()
        self.assertEqual(old_lp.shape,(3,1))
        self.assertGreater(float(m.pair_query.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.pair_key.weight.grad.abs().sum()),0.)
        self.assertEqual(m.candidate_dag(obs)[0].shape,(3,27,27))

    def test_full_mat_sampling_teacher_forcing_and_ppo(self):
        policy=TransformerPolicy.__new__(TransformerPolicy)
        policy.algorithm_name="mat"
        policy.num_agents=3
        policy.share_obs_dim=9
        policy.obs_dim=7
        policy.act_dim=4
        policy.act_num=1
        policy._use_policy_active_masks=False
        policy.tpdv=dict(dtype=torch.float32,device=torch.device("cpu"))
        policy.transformer=MultiAgentTransformer(
            9,7,4,3,1,16,1,agent_order_mode="dependency",
            order_pair_rank=8,order_graph_scale=0.5)
        policy.optimizer=torch.optim.Adam(policy.transformer.parameters(),lr=.01)
        rng=np.random.RandomState(318)
        obs=rng.randn(2,3,7).astype(np.float32)
        cent=np.zeros((2,3,9),dtype=np.float32)
        available=np.ones((2,3,4),dtype=np.float32)
        zeros=np.zeros((6,1),dtype=np.float32)
        with torch.no_grad():
            v,a,old_action_lp,_,_,old_order,old_order_lp=policy.get_actions(
                cent.reshape(-1,9),obs.reshape(-1,7),zeros,zeros,zeros,
                available.reshape(-1,4),return_agent_order=True,return_order_log_prob=True)
            new_v,new_action_lp,_,new_order_lp=policy.evaluate_actions(
                cent.reshape(-1,9),obs.reshape(-1,7),zeros,zeros,
                a,zeros,available.reshape(-1,4),
                agent_order=old_order,return_order_log_prob=True)
        self.assertTrue(torch.allclose(v,new_v,atol=2e-5))
        self.assertTrue(torch.allclose(old_action_lp,new_action_lp,atol=2e-5))
        self.assertTrue(torch.allclose(old_order_lp,new_order_lp,atol=1e-6))
        args=SimpleNamespace(
            clip_param=.2,ppo_epoch=1,num_mini_batch=1,data_chunk_length=1,
            value_loss_coef=1.,entropy_coef=.01,max_grad_norm=10,huber_delta=10,
            use_recurrent_policy=False,use_naive_recurrent_policy=False,
            use_max_grad_norm=True,use_clipped_value_loss=False,use_huber_loss=False,
            use_valuenorm=False,use_value_active_masks=False,use_policy_active_masks=False,
            dec_actor=False,agent_order_mode="dependency",order_loss_coef=.5)
        trainer=MATTrainer(args,policy,3)
        sample=(cent.reshape(-1,9),obs.reshape(-1,7),zeros,zeros,
                a.detach().numpy(),v.detach().numpy(),
                (v.detach()+.25).numpy(),zeros,np.ones_like(zeros),
                old_action_lp.detach().numpy(),
                np.array([[1.],[1.],[1.],[-1.],[-1.],[-1.]],dtype=np.float32),
                available.reshape(-1,4),old_order.detach().numpy(),
                old_order_lp.detach().numpy())
        trainer.ppo_update(sample)
        self.assertTrue(np.isfinite(trainer.last_order_policy_loss))
        self.assertEqual(trainer.last_order_log_ratio_clamp_fraction,0)
        self.assertGreater(float(policy.transformer.order_policy.pair_query.weight.grad.abs().sum()),0.)


if __name__=="__main__":
    unittest.main()
