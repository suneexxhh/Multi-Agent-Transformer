"""CPU-only learned permutation policy and PPO replay integration tests."""
import itertools
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from mat.algorithms.mat.algorithm.learned_agent_order import LearnedAgentOrder
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.mat.mat_trainer import MATTrainer
from mat.utils.shared_buffer import SharedReplayBuffer


class Box:
    def __init__(self, dim):
        self.shape = (dim,)


class Discrete:
    def __init__(self, dim):
        self.n = dim


class TestLearnedOrder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(171)

    def test_probability_mass_teacher_forcing_and_gradients(self):
        p = LearnedAgentOrder(7, 16)
        obs = torch.randn(5, 3, 7)
        order, logp = p.sample(obs)
        self.assertEqual(logp.shape, (5, 1))
        self.assertTrue(torch.equal(order.sort(1).values, torch.arange(3).expand(5, -1)))
        self.assertTrue(torch.allclose(logp, p.log_prob(obs, order), atol=1e-6))
        total = sum(p.log_prob(obs[:1], torch.tensor([perm])).exp()
                    for perm in itertools.permutations(range(3)))
        self.assertAlmostEqual(total.item(), 1., places=5)
        optim = torch.optim.Adam(p.parameters(), lr=1e-2)
        loss = -(p.log_prob(obs, order).squeeze(-1) *
                 torch.tensor([1., -1., 2., -1., 1.])).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        self.assertGreater(p.head[-1].weight.abs().sum().item(), 0)
        optim.zero_grad()
        p.log_prob(obs, order).mean().backward()
        self.assertGreater(p.embedding[1].weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            p.log_prob(obs, torch.zeros(5, 3, dtype=torch.long))

    def test_model_actions_and_old_order_replay(self):
        m = MultiAgentTransformer(state_dim=9, obs_dim=7, action_dim=4,
                                  n_agent=3, n_block=1, n_embd=16, n_head=1,
                                  agent_order_mode="learned")
        obs = np.random.RandomState(17).randn(2, 3, 7).astype(np.float32)
        state = np.zeros((2, 3, 9), dtype=np.float32)
        legal = np.ones((2, 3, 4), dtype=np.float32)
        with torch.no_grad():
            a, lp, v, order, order_lp = m.get_actions(
                state, obs, legal, return_agent_order=True, return_order_log_prob=True)
            lp2, v2, entropy, order_lp2 = m(
                state, obs, a, legal, agent_order=order, return_order_log_prob=True)
        self.assertTrue(torch.allclose(lp, lp2, atol=2e-5))
        self.assertTrue(torch.allclose(order_lp, order_lp2, atol=1e-6))
        self.assertTrue(torch.allclose(v, v2, atol=2e-5))
        self.assertEqual(entropy.shape, (2, 3, 1))
        with self.assertRaises(ValueError):
            m(state, obs, a, legal, return_order_log_prob=True)

    def test_buffer_stores_exact_joint_order_logp(self):
        args = SimpleNamespace(
            episode_length=2, n_rollout_threads=2, hidden_size=8, recurrent_N=1,
            gamma=.99, gae_lambda=.95, use_gae=True, use_popart=False,
            use_valuenorm=False, use_proper_time_limits=False,
            algorithm_name="mat", store_agent_orders=True,
            agent_order_mode="learned")
        b = SharedReplayBuffer(args, 3, Box(7), Box(9), Discrete(4), "StarCraft2")
        for t in range(2):
            zeros = np.zeros((2, 3, 1), dtype=np.float32)
            rnn = np.zeros((2, 3, 1, 8), dtype=np.float32)
            legal = np.ones((2, 3, 4), dtype=np.float32)
            order = np.array([[2, 0, 1], [0, 2, 1]], dtype=np.int64)
            old_lp = np.array([[-1.5 - t], [-2.0 - t]], dtype=np.float32)
            b.insert(np.zeros((2, 3, 9), dtype=np.float32),
                     np.zeros((2, 3, 7), dtype=np.float32), rnn, rnn,
                     zeros, zeros, zeros, zeros, np.ones_like(zeros),
                     available_actions=legal, agent_orders=order,
                     agent_order_log_probs=old_lp)
        sample = next(b.feed_forward_generator_transformer(
            np.ones_like(b.advantages), num_mini_batch=1))
        self.assertEqual(len(sample), 14)
        self.assertEqual(sample[-2].shape, (4, 3))
        self.assertEqual(sample[-1].shape, (4, 1))
        self.assertEqual(sorted(sample[-1].reshape(-1).tolist()),
                         sorted([-1.5, -2.0, -2.5, -3.0]))

    def test_one_real_trainer_step_updates_order_head(self):
        # Uses actual TransformerPolicy wrapper + MATTrainer; no SC2 or GPU.
        p = TransformerPolicy.__new__(TransformerPolicy)
        p.algorithm_name = "mat"
        p.num_agents, p.share_obs_dim, p.obs_dim, p.act_dim, p.act_num = 3, 9, 7, 4, 1
        p._use_policy_active_masks = False
        p.tpdv = dict(dtype=torch.float32, device=torch.device("cpu"))
        p.transformer = MultiAgentTransformer(
            9, 7, 4, 3, 1, 16, 1, agent_order_mode="learned")
        p.optimizer = torch.optim.Adam(p.transformer.parameters(), lr=0.01)
        rng = np.random.RandomState(65)
        obs = rng.randn(2, 3, 7).astype(np.float32)
        cent = np.zeros((2, 3, 9), dtype=np.float32)
        legal = np.ones((2, 3, 4), dtype=np.float32)
        zeros = np.zeros((6, 1), dtype=np.float32)
        with torch.no_grad():
            v, act, old_lp, _, _, order, old_order_lp = p.get_actions(
                cent.reshape(-1, 9), obs.reshape(-1, 7), zeros, zeros, zeros,
                legal.reshape(-1, 4), return_agent_order=True,
                return_order_log_prob=True)
        args = SimpleNamespace(clip_param=.2, ppo_epoch=1, num_mini_batch=1,
                               data_chunk_length=1, value_loss_coef=1.,
                               entropy_coef=.01, max_grad_norm=10, huber_delta=10,
                               use_recurrent_policy=False, use_naive_recurrent_policy=False,
                               use_max_grad_norm=True, use_clipped_value_loss=False,
                               use_huber_loss=False, use_valuenorm=False,
                               use_value_active_masks=False, use_policy_active_masks=False,
                               dec_actor=False, agent_order_mode="learned",
                               order_loss_coef=0.5)
        trainer = MATTrainer(args, p, 3)
        sample = (cent.reshape(-1, 9), obs.reshape(-1, 7), zeros, zeros,
                  act.detach().numpy(), v.detach().numpy(),
                  (v.detach() + 0.25).numpy(), zeros, np.ones_like(zeros),
                  old_lp.detach().numpy(), np.array([[1], [1], [1], [-1], [-1], [-1]],
                                                    dtype=np.float32),
                  legal.reshape(-1, 4), order.detach().numpy(),
                  old_order_lp.detach().numpy())
        trainer.ppo_update(sample)
        self.assertIsNotNone(trainer.last_order_policy_loss)
        self.assertTrue(np.isfinite(trainer.last_order_policy_loss))
        self.assertGreater(p.transformer.order_policy.head[-1].weight.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
