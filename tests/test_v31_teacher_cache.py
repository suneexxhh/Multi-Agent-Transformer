"""CPU equivalence tests for frozen-teacher raw-result caching."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.mat.mat_trainer import MATTrainer
from mat.algorithms.mat.algorithm import influence_alignment as alignment
from mat.utils.shared_buffer import SharedReplayBuffer


def make_policy(model):
    policy = TransformerPolicy.__new__(TransformerPolicy)
    policy.algorithm_name, policy.num_agents = "mat", 3
    policy.share_obs_dim, policy.obs_dim, policy.act_dim, policy.act_num = 9, 5, 3, 1
    policy._use_policy_active_masks = False
    policy.tpdv = dict(dtype=torch.float32, device=torch.device("cpu"))
    policy.transformer = model
    policy.optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    return policy


def trainer_args(cache):
    return SimpleNamespace(
        clip_param=.2, ppo_epoch=2, num_mini_batch=2,
        data_chunk_length=1, value_loss_coef=1., entropy_coef=.01,
        max_grad_norm=10., huber_delta=10., use_recurrent_policy=False,
        use_naive_recurrent_policy=False, use_max_grad_norm=True,
        use_clipped_value_loss=False, use_huber_loss=False,
        use_valuenorm=False, use_value_active_masks=False,
        use_policy_active_masks=False, dec_actor=False,
        agent_order_mode="dependency_prefix", order_loss_coef=.1,
        order_influence_align_coef=.2, order_influence_max_alternatives=2,
        cache_order_influence_teacher=cache)


class TestTeacherCache(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(729)
        self.initial = MultiAgentTransformer(
            9, 5, 3, 3, 1, 16, 1, agent_order_mode="dependency_prefix",
            order_attention_bias=False, order_pair_rank=8)
        rng = np.random.RandomState(727)
        obs = rng.randn(4, 3, 5).astype(np.float32)
        cent = np.zeros((4, 3, 9), dtype=np.float32)
        legal = np.ones((4, 3, 3), dtype=np.float32)
        source = make_policy(copy.deepcopy(self.initial))
        zeros = np.zeros((12, 1), dtype=np.float32)
        with torch.no_grad():
            value, action, action_lp, _, _, order, order_lp = source.get_actions(
                cent.reshape(-1, 9), obs.reshape(-1, 5), zeros, zeros,
                zeros, legal.reshape(-1, 3), return_agent_order=True,
                return_order_log_prob=True)
        self.fields = (
            cent, obs, np.zeros((4, 3, 1), dtype=np.float32),
            np.zeros((4, 3, 1), dtype=np.float32),
            action.numpy().reshape(4, 3, 1),
            value.numpy().reshape(4, 3, 1),
            (value + .2).numpy().reshape(4, 3, 1),
            np.zeros((4, 3, 1), dtype=np.float32),
            np.ones((4, 3, 1), dtype=np.float32),
            action_lp.numpy().reshape(4, 3, 1),
            np.array([[[.3]] * 3, [[-.2]] * 3,
                      [[.5]] * 3, [[-.4]] * 3], dtype=np.float32),
            legal, order.numpy(), order_lp.numpy())

    def sample(self, ids, cache):
        fields = tuple(x[np.asarray(ids)].reshape(-1, *x.shape[2:])
                       if i < 12 else x[np.asarray(ids)].copy()
                       for i, x in enumerate(self.fields))
        return fields + ((np.asarray(ids, dtype=np.int64),) if cache else ())

    def run_updates(self, cache, synthetic=False):
        policy = make_policy(copy.deepcopy(self.initial))
        trainer = MATTrainer(trainer_args(cache), policy, 3)
        groups = ((0, 1), (2, 3), (0, 2), (1, 3))
        recorded = []
        original = alignment.alignment_targets
        original_probe = alignment.conditional_action_js
        calls = []

        def record_targets(*args, **kwargs):
            value = original(*args, **kwargs)
            recorded.append(tuple(v.clone() for v in value))
            return value

        def record_probe(*args, **kwargs):
            calls.append(args[1].shape[0])
            if synthetic:
                obs = args[1]
                batch = obs.shape[0]
                js = obs.new_zeros(batch, 3, 3)
                valid = torch.tril(torch.ones(3, 3, dtype=torch.bool), -1)[None].expand(batch, -1, -1)
                for target in range(1, 3):
                    for source in range(target):
                        js[:, target, source] = (obs[:, 0, 0] + 4) * (target + source + 1) * 1e-5
                entropy = obs.new_ones(batch, 3)
                alternatives = torch.full((batch, 3), 2, device=obs.device)
                return js, valid, entropy, alternatives
            return original_probe(*args, **kwargs)

        with patch.object(alignment, "alignment_targets", side_effect=record_targets), \
                patch.object(alignment, "conditional_action_js", side_effect=record_probe):
            for ids in groups:
                trainer.ppo_update(self.sample(ids, cache))
        return trainer, recorded, calls

    def test_cached_and_uncached_labels_gradients_and_ppo_match(self):
        plain, labels_plain, calls_plain = self.run_updates(False)
        cached, labels_cached, calls_cached = self.run_updates(True)
        self.assertEqual(calls_plain, [2, 2, 2, 2])
        self.assertEqual(calls_cached, [2, 2])
        self.assertEqual(len(cached._influence_raw_cache), 4)
        for original, memoized in zip(labels_plain, labels_cached):
            for a, b in zip(original, memoized):
                self.assertTrue(torch.equal(a, b))
        for key, weight in plain.policy.transformer.state_dict().items():
            self.assertTrue(torch.equal(weight, cached.policy.transformer.state_dict()[key]), key)
        for (name, grad), (_, cached_grad) in zip(
                plain.policy.transformer.named_parameters(),
                cached.policy.transformer.named_parameters()):
            if grad.grad is not None:
                self.assertTrue(torch.equal(grad.grad, cached_grad.grad), name)
        self.assertFalse(any(p.requires_grad or p.grad is not None
                             for p in cached._influence_teacher.parameters()))

    def test_cache_preserves_nonzero_minibatch_stratified_targets(self):
        plain, plain_labels, _ = self.run_updates(False, synthetic=True)
        cached, cached_labels, _ = self.run_updates(True, synthetic=True)
        self.assertGreater(sum(float(c.sum()) for _, c in plain_labels), 0.)
        self.assertFalse(torch.equal(plain_labels[0][0][0], plain_labels[2][0][0]))
        for original, memoized in zip(plain_labels, cached_labels):
            for a, b in zip(original, memoized):
                self.assertTrue(torch.equal(a, b))
        for name, weight in plain.policy.transformer.state_dict().items():
            self.assertTrue(torch.equal(weight, cached.policy.transformer.state_dict()[name]), name)

    def test_cache_rejects_missing_or_duplicate_transition_ids(self):
        trainer = MATTrainer(trainer_args(True), make_policy(copy.deepcopy(self.initial)), 3)
        with self.assertRaisesRegex(ValueError, "stable rollout transition"):
            trainer.ppo_update(self.sample((0, 1), False))
        with self.assertRaisesRegex(ValueError, "invalid cache transition"):
            trainer.ppo_update(self.sample((0, 0), True))

    def test_generator_indices_are_opt_in_and_stable_for_rollout(self):
        buffer = SharedReplayBuffer.__new__(SharedReplayBuffer)
        buffer.rewards = np.zeros((2, 2, 3, 1), dtype=np.float32)
        for name, width in (("share_obs", 9), ("obs", 5),
                            ("rnn_states", 1), ("rnn_states_critic", 1),
                            ("value_preds", 1), ("returns", 1),
                            ("masks", 1), ("active_masks", 1),
                            ("available_actions", 3)):
            setattr(buffer, name, np.zeros((3, 2, 3, width), dtype=np.float32))
        buffer.actions = np.zeros((2, 2, 3, 1), dtype=np.float32)
        buffer.action_log_probs = buffer.actions.copy()
        buffer.agent_orders = np.tile(np.arange(3), (2, 2, 1))
        buffer.agent_order_log_probs = np.zeros((2, 2, 1), dtype=np.float32)
        advantage = np.ones((2, 2, 3, 1), dtype=np.float32)
        torch.manual_seed(8)
        batches = list(buffer.feed_forward_generator_transformer(
            advantage, 2, include_transition_indices=True))
        self.assertTrue(all(len(x) == 15 for x in batches))
        self.assertEqual(sorted(np.concatenate([x[-1] for x in batches]).tolist()),
                         list(range(4)))
        torch.manual_seed(8)
        old = list(buffer.feed_forward_generator_transformer(advantage, 2))
        self.assertTrue(all(len(x) == 14 for x in old))

    def test_train_clears_snapshot_and_cache_between_rollouts(self):
        trainer = MATTrainer(trainer_args(True), make_policy(copy.deepcopy(self.initial)), 3)

        class Buffer:
            advantages = np.array([[[[.3], [.3], [.3]]],
                                   [[[-.2], [-.2], [-.2]]],
                                   [[[.5], [.5], [.5]]],
                                   [[[-.4], [-.4], [-.4]]]], dtype=np.float32)
            active_masks = np.ones((5, 1, 3, 1), dtype=np.float32)

            def feed_forward_generator_transformer(_self, advantage, num_mini_batch,
                                                   include_transition_indices=False):
                self.assertTrue(include_transition_indices)
                for ids in ((0, 1), (2, 3)):
                    yield self.sample(ids, True)

        calls = []
        original = alignment.conditional_action_js

        def count(*args, **kwargs):
            calls.append(args[1].shape[0])
            return original(*args, **kwargs)

        with patch.object(alignment, 'conditional_action_js', side_effect=count):
            trainer.train(Buffer())
            self.assertIsNone(trainer._influence_teacher)
            self.assertEqual(trainer._influence_raw_cache, {})
            trainer.train(Buffer())
        self.assertEqual(calls, [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
