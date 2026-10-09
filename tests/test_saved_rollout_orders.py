"""V1 PPO consistency scaffolding: sampled permutation survives replay unchanged.

This tests opt-in infrastructure only. Learned precedence scorer is not yet
connected to PPO and cannot be claimed to have learned a causal graph.
"""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from mat.algorithms.mat.algorithm.agent_ordering import validate_agent_order
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.utils.shared_buffer import SharedReplayBuffer


class Box:
    def __init__(self, dim):
        self.shape = (dim,)


class Discrete:
    def __init__(self, num_actions):
        self.n = num_actions


def make_buffer(store=True):
    args = SimpleNamespace(
        episode_length=2, n_rollout_threads=2, hidden_size=8, recurrent_N=1,
        gamma=0.99, gae_lambda=0.95, use_gae=True, use_popart=False,
        use_valuenorm=False, use_proper_time_limits=False,
        algorithm_name="mat", store_agent_orders=store,
    )
    return SharedReplayBuffer(args, 3, Box(4), Box(5), Discrete(4), "StarCraft2")


def insert_fake_rollout(buf, step, order):
    threads, agents = 2, 3
    rnn = np.zeros((threads, agents, 1, 8), dtype=np.float32)
    observations = np.full((threads, agents, 4), step, dtype=np.float32)
    shared_observations = np.full((threads, agents, 5), step, dtype=np.float32)
    acts = np.zeros((threads, agents, 1), dtype=np.float32)
    values = np.zeros_like(acts)
    masks = np.ones_like(acts)
    legal = np.ones((threads, agents, 4), dtype=np.float32)
    buf.insert(shared_observations, observations, rnn, rnn, acts, acts,
               values, values, masks, available_actions=legal, agent_orders=order)


class TestStoredRolloutPermutation(unittest.TestCase):
    def test_validate_valid_and_reject_corrupt_permutation(self):
        dev = torch.device("cpu")
        order = np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int64)
        self.assertTrue(torch.equal(validate_agent_order(order, 2, 3, dev),
                                    torch.as_tensor(order)))
        for invalid in (np.array([[0, 0, 1], [1, 2, 0]], dtype=np.int64),
                        np.array([[0, 1, 2]], dtype=np.int64),
                        np.zeros((2, 3), dtype=np.float32),
                        np.array([[0, 1, 3], [1, 2, 0]], dtype=np.int64)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_agent_order(invalid, 2, 3, dev)

    def test_saved_minibatch_order_is_one_per_transition(self):
        buf = make_buffer(store=True)
        stored = [
            np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int64),
            np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64),
        ]
        for i, order in enumerate(stored):
            insert_fake_rollout(buf, i, order)
        batches = list(buf.feed_forward_generator_transformer(
            np.ones_like(buf.advantages), num_mini_batch=1))
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 13)
        sample = batches[0]
        self.assertEqual(sample[-1].shape, (4, 3))
        self.assertEqual(
            sorted(map(tuple, sample[-1].tolist())),
            sorted(map(tuple, np.concatenate(stored, axis=0).tolist()))
        )
        self.assertEqual(sample[4].shape, (12, 1))
        with self.assertRaises(ValueError):
            insert_fake_rollout(make_buffer(store=True), 0, None)

    def test_opt_in_disabled_preserves_original_minibatch(self):
        buf = make_buffer(store=False)
        for i in range(2):
            insert_fake_rollout(buf, i, None)
        sample = next(buf.feed_forward_generator_transformer(
            np.ones_like(buf.advantages), num_mini_batch=1))
        self.assertEqual(len(sample), 12)
        self.assertIsNone(buf.agent_orders)

    def test_model_sampling_and_explicit_stored_order_eval(self):
        torch.manual_seed(603)
        m = MultiAgentTransformer(
            state_dim=9, obs_dim=7, action_dim=4, n_agent=3,
            n_block=1, n_embd=16, n_head=1, action_type="Discrete",
            agent_order_mode="obs_norm")
        obs = np.array([
            [[1., 0, 0, 0, 0, 0, 0], [3., 0, 0, 0, 0, 0, 0], [2., 0, 0, 0, 0, 0, 0]],
            [[2., 0, 0, 0, 0, 0, 0], [1., 0, 0, 0, 0, 0, 0], [3., 0, 0, 0, 0, 0, 0]],
        ], dtype=np.float32)
        state = np.zeros((2, 3, 9), dtype=np.float32)
        masks = np.ones((2, 3, 4), dtype=np.float32)
        with torch.no_grad():
            acts, logps, values, original_order = m.get_actions(
                state, obs, masks, deterministic=True, return_agent_order=True)
            replay_logps, replay_values, entropies = m(
                state, obs, acts, masks, agent_order=original_order)
        self.assertTrue(torch.allclose(logps, replay_logps, atol=2e-5))
        self.assertTrue(torch.allclose(values, replay_values, atol=2e-5))
        self.assertEqual(original_order.tolist(), [[1, 2, 0], [2, 0, 1]])
        self.assertEqual(entropies.shape, (2, 3, 1))
        # A known different order can be passed explicitly, even when it
        # disagrees with obs_norm. This is essential after scorer updates.
        forced = torch.tensor([[2, 0, 1], [0, 1, 2]], dtype=torch.long)
        with torch.no_grad():
            a2, l2, _, saved2 = m.get_actions(
                state, obs, masks, deterministic=True,
                agent_order=forced, return_agent_order=True)
            l2_eval, _, _ = m(state, obs, a2, masks, agent_order=saved2)
        self.assertTrue(torch.equal(forced, saved2))
        self.assertTrue(torch.allclose(l2, l2_eval, atol=2e-5))
        with self.assertRaises(ValueError):
            m(state, obs, acts, masks, agent_order=torch.tensor([[0, 0, 2], [0, 1, 2]]))

    def test_transformer_policy_transmits_saved_order(self):
        torch.manual_seed(604)
        p = TransformerPolicy.__new__(TransformerPolicy)
        p.algorithm_name = "mat"
        p.num_agents = 3
        p.share_obs_dim = 9
        p.obs_dim = 7
        p.act_dim = 4
        p.act_num = 1
        p._use_policy_active_masks = False
        p.tpdv = dict(dtype=torch.float32, device=torch.device("cpu"))
        p.transformer = MultiAgentTransformer(
            state_dim=9, obs_dim=7, action_dim=4, n_agent=3,
            n_block=1, n_embd=16, n_head=1, action_type="Discrete",
            agent_order_mode="obs_norm")
        rng = np.random.RandomState(606)
        obs = rng.randn(2, 3, 7).astype(np.float32)
        obs[:, 0, 0] += 5.0
        shared = np.zeros((2, 3, 9), dtype=np.float32)
        rnn = np.zeros((6, 1, 8), dtype=np.float32)
        masks = np.ones((6, 1), dtype=np.float32)
        available = np.ones((6, 4), dtype=np.float32)
        with torch.no_grad():
            values, action, old_lp, _, _, order = p.get_actions(
                shared.reshape(-1, 9), obs.reshape(-1, 7), rnn, rnn,
                masks, available, deterministic=True, return_agent_order=True)
            values_eval, new_lp, _ = p.evaluate_actions(
                shared.reshape(-1, 9), obs.reshape(-1, 7), rnn, rnn,
                action, masks, available, agent_order=order)
        self.assertEqual(order.shape, (2, 3))
        self.assertTrue(torch.allclose(old_lp, new_lp, atol=2e-5))
        self.assertTrue(torch.allclose(values, values_eval, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
