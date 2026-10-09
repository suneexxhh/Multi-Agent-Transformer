"""CPU/GPU-independent tests for the V1 *isolated* precedence scorer.

No tests in this module assume causal influence or integrate a learned
ordering into PPO rollout/optimization.
"""
import unittest

import torch

from mat.algorithms.mat.algorithm.dependency_ordering import PairwisePrecedenceScorer


class TestPairwisePrecedenceScorer(unittest.TestCase):

    def test_shape_reciprocity_and_neutral_diagonal(self):
        torch.manual_seed(501)
        m = PairwisePrecedenceScorer(obs_dim=4, hidden_dim=8)
        for obs in (torch.randn(2, 3, 4), torch.randn(1, 1, 4)):
            p = m(obs)
            n = obs.shape[1]
            self.assertEqual(p.shape, (obs.shape[0], n, n))
            self.assertEqual(p.device, obs.device)
            self.assertTrue(torch.all((p >= 0) & (p <= 1)))
            self.assertTrue(torch.allclose(p + p.transpose(-1, -2),
                                           torch.ones_like(p), atol=1e-6))
            self.assertTrue(torch.allclose(
                p.diagonal(dim1=1, dim2=2),
                torch.full((obs.shape[0], n), 0.5), atol=1e-7))

    def test_stable_ties_and_independent_batch_orders(self):
        m = PairwisePrecedenceScorer(obs_dim=1, hidden_dim=1)
        # Exactly prefer the agent with larger positive observation.
        with torch.no_grad():
            m.query.weight.fill_(1.0)
            m.query.bias.zero_()
            m.key.weight.zero_()
            m.key.bias.fill_(1.0)
        obs = torch.tensor([[[1.], [3.], [2.]], [[2.], [2.], [2.]]])
        order = m.order(obs)
        self.assertEqual(order.tolist(), [[1, 2, 0], [0, 1, 2]])
        self.assertEqual(order.dtype, torch.long)

    def test_bce_reaches_pairwise_parameters(self):
        torch.manual_seed(502)
        m = PairwisePrecedenceScorer(obs_dim=3, hidden_dim=16)
        obs = torch.randn(2, 4, 3)
        order = m.order(obs)
        target = torch.zeros(2, 4, 4)
        for batch in range(2):
            rank = torch.empty(4, dtype=torch.long)
            rank[order[batch]] = torch.arange(4)
            target[batch] = (rank[:, None] < rank[None, :]).float()
        loss = m.supervised_loss(obs, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(any(g.abs().sum().item() > 0 for g in grads))

    def test_mask_and_input_validation(self):
        m = PairwisePrecedenceScorer(obs_dim=3)
        obs = torch.randn(1, 3, 3)
        p = m(obs)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 3))
        with self.assertRaises(ValueError):
            m(torch.randn(1, 3, 2))
        with self.assertRaises(ValueError):
            m(torch.randn(1, 0, 3))
        with self.assertRaises(ValueError):
            m.supervised_loss(obs, torch.randn(1, 2, 2))
        with self.assertRaises(ValueError):
            m.supervised_loss(obs, p, valid_pairs=torch.zeros_like(p).bool())
        mask = torch.zeros_like(p, dtype=torch.bool)
        mask[:, 0, 1] = True
        loss = m.supervised_loss(obs, p.detach(), valid_pairs=mask)
        self.assertTrue(torch.isfinite(loss))

    def test_low_rank_parameter_budget_and_confident_acyclic_edges(self):
        torch.manual_seed(509)
        m = PairwisePrecedenceScorer(obs_dim=64, hidden_dim=8)
        self.assertEqual(sum(p.numel() for p in m.parameters()),
                         2 * (64 + 1) * 8)
        tiny = PairwisePrecedenceScorer(obs_dim=1, hidden_dim=1)
        with torch.no_grad():
            tiny.query.weight.fill_(1)
            tiny.query.bias.zero_()
            tiny.key.weight.zero_()
            tiny.key.bias.fill_(1)
        obs = torch.tensor([[[10.], [0.], [-10.]]])
        edges, order = tiny.precedence_dag(obs, threshold=0.7)
        self.assertEqual(order.tolist(), [[0, 1, 2]])
        expected = torch.tensor([[[False, True, True],
                                  [False, False, True],
                                  [False, False, False]]])
        self.assertTrue(torch.equal(edges, expected))
        # All graph edges must respect the produced ranking: no cycles.
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(3).expand_as(order))
        self.assertTrue(((rank.unsqueeze(2) < rank.unsqueeze(1)) | ~edges).all())
        self.assertEqual(edges.dtype, torch.bool)

    def test_identical_features_do_not_fabricate_dependencies(self):
        torch.manual_seed(506)
        m = PairwisePrecedenceScorer(obs_dim=4)
        obs = torch.zeros(2, 5, 4)
        graph, order = m.precedence_dag(obs, threshold=0.65)
        self.assertFalse(bool(graph.any()))
        self.assertEqual(order.tolist(), [list(range(5)), list(range(5))])
        self.assertTrue(torch.equal(m(obs), torch.full((2, 5, 5), 0.5)))
        for invalid in (0.5, 1.0, -0.2):
            with self.assertRaises(ValueError):
                m.precedence_dag(obs, threshold=invalid)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_forward_and_order(self):
        m = PairwisePrecedenceScorer(obs_dim=5).cuda()
        obs = torch.randn(2, 4, 5, device="cuda:0")
        self.assertEqual(m(obs).device, obs.device)
        self.assertEqual(m.order(obs).device, obs.device)


if __name__ == "__main__":
    unittest.main()
