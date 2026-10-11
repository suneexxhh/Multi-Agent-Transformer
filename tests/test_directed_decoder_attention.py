"""CPU-only test of learned relation bias in MAT's causal action decoder."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer


class TestDirectedDecoderAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(817)
        self.net = MultiAgentTransformer(
            9, 7, 4, 4, n_block=2, n_embd=16, n_head=1,
            agent_order_mode="dependency", order_pair_rank=8)
        self.obs = torch.randn(2, 4, 7)
        self.state = torch.zeros(2, 4, 37)
        self.order = torch.tensor([[2, 0, 3, 1],[1, 2, 0, 3]])
        self.rep = self.net.encoder(self.state, self.obs)[1]

    def test_bias_shape_and_shifted_action_indexing(self):
        p=self.net.order_policy
        bias=p.conditioning_bias(self.obs, self.order)
        self.assertEqual(bias.shape, (2,4,4))
        self.assertTrue(torch.allclose(bias[:,:,0],torch.zeros(2,4)))
        directed=p.pairwise_logits(self.obs)
        # Decoder target slot 2 receives previous-action key slot 1
        # from the agent decoded at slot 0.
        for b in range(2):
            src=int(self.order[b,0])
            dst=int(self.order[b,2])
            expected=torch.log(torch.tensor(2.))+F.logsigmoid(directed[b,src,dst])
            self.assertTrue(torch.allclose(bias[b,2,1],expected,atol=1e-6))

    def test_bias_changes_logits_and_backpropagates_through_precedence(self):
        shifted=torch.zeros(2,4,5)
        shifted[:,0,0]=1
        shifted[:,1:,1:]=F.one_hot(torch.tensor([[2,1,3],[1,2,3]]),4).float()
        bias=self.net.order_policy.conditioning_bias(self.obs, self.order)
        without=self.net.decoder(shifted,self.rep,self.obs)
        with_rel=self.net.decoder(shifted,self.rep,self.obs,attention_bias=bias)
        self.assertGreater((without-with_rel).abs().max().item(),1e-12)
        self.net.zero_grad()
        with_rel[:,2,:].square().mean().backward()
        self.assertGreater(self.net.order_policy.pair_query.weight.grad.abs().sum().item(),0)

    def test_future_actions_never_enter_past_decoder_logits(self):
        shifted=torch.zeros(2,4,5)
        shifted[:,0,0]=1
        shifted[:,1:,1:]=F.one_hot(torch.tensor([[0,1,2],[1,2,3]]),4).float()
        bias=self.net.order_policy.conditioning_bias(self.obs, self.order)
        logits_a=self.net.decoder(shifted,self.rep,self.obs,attention_bias=bias)
        modified=shifted.clone()
        modified[:,3,1:]=F.one_hot(torch.tensor([3,0]),4).float()
        logits_b=self.net.decoder(modified,self.rep,self.obs,attention_bias=bias)
        self.assertTrue(torch.allclose(logits_a[:,:3], logits_b[:,:3],atol=2e-6))

    def test_full_sampling_matches_parallel_log_prob(self):
        obs=self.obs.numpy().astype(np.float32)
        cent=np.zeros((2,4,9),dtype=np.float32)
        available=np.ones((2,4,4),dtype=np.float32)
        with torch.no_grad():
            a, lp, value, order, order_lp=self.net.get_actions(
                cent,obs,available,return_agent_order=True,
                return_order_log_prob=True)
            replay, value2, entropy, order_lp2=self.net(
                cent,obs,a,available,agent_order=order,
                return_order_log_prob=True)
        self.assertTrue(torch.allclose(lp,replay,atol=2e-5))
        self.assertTrue(torch.allclose(order_lp,order_lp2,atol=2e-6))
        self.assertTrue(torch.allclose(value,value2,atol=2e-5))


if __name__=="__main__":
    unittest.main()
