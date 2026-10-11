"""CPU-only V3 mathematical, policy-replay, attention and integration tests."""
import itertools
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.directed_dependency_order import DirectedDependencyOrder
from mat.algorithms.mat.algorithm.prefix_dependency_order import PrefixDependencyOrder
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.mat.mat_trainer import MATTrainer
from mat.config import get_config
from research.dependency_decoder.checkpoint_manifest import (
    build_training_manifest, evaluation_model_flags, load_training_manifest)
from pathlib import Path
import json
from tempfile import TemporaryDirectory


class TestPrefixDependencyDistribution(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(639)
        self.p = PrefixDependencyOrder(7, hidden_dim=16, pair_rank=8,
                                       prefix_scale=1.0)
        self.obs = torch.randn(3, 4, 7)

    def test_distribution_normalizes_all_permutations(self):
        obs = self.obs[:1, :3]
        mass = sum(self.p.log_prob(obs, torch.tensor([perm])).exp().item()
                   for perm in itertools.permutations(range(3)))
        self.assertAlmostEqual(mass, 1.0, delta=1e-5)

    def test_sampled_order_has_exact_teacher_forced_likelihood(self):
        for deterministic in (False, True):
            orders, old_lp = self.p.sample(self.obs, deterministic=deterministic)
            self.assertEqual(orders.shape, (3, 4))
            self.assertEqual(old_lp.shape, (3, 1))
            self.assertTrue(torch.equal(orders.sort(dim=-1).values,
                                        torch.arange(4).expand(3, 4)))
            self.assertTrue(torch.isfinite(old_lp).all())
            self.assertTrue(torch.allclose(
                old_lp, self.p.log_prob(self.obs, orders), atol=1e-6))

    def test_prefix_changes_relative_candidate_scores(self):
        # Keep candidates 0 and 1 undecoded. Change which of agents 2/3
        # has already been decoded. Dependencies must alter their odds.
        m = PrefixDependencyOrder(7, hidden_dim=16, pair_rank=8)
        need = torch.full((1, 4, 4), 0.5)
        need[0, 0, 2], need[0, 1, 2] = 0.95, 0.05
        need[0, 0, 3], need[0, 1, 3] = 0.05, 0.95
        with patch.object(DirectedDependencyOrder, "scores",
                          return_value=torch.zeros(1, 4)):
            with patch.object(m, "demand", return_value=need):
                score_after_2 = m.prefix_scores(self.obs[:1], [[False, False, True, False]])
                score_after_3 = m.prefix_scores(self.obs[:1], [[False, False, False, True]])
        self.assertTrue(torch.isinf(score_after_2[0, 2]))
        self.assertTrue(torch.isinf(score_after_3[0, 3]))
        odds_after_2 = score_after_2[0, 0] - score_after_2[0, 1]
        odds_after_3 = score_after_3[0, 0] - score_after_3[0, 1]
        self.assertGreater((odds_after_2 - odds_after_3).abs().item(), .2)

    def test_directed_demand_not_antisymmetric_precedence(self):
        demand = self.p.demand(self.obs)
        self.assertTrue(torch.allclose(demand.diagonal(dim1=1, dim2=2),
                                       torch.full((3, 4), 0.5)))
        self.assertGreater(
            (demand + demand.transpose(1, 2) - 1).abs().max().item(), 1e-4)
        pref = self.p.pairwise_preferences(self.obs)
        self.assertTrue(torch.allclose(pref + pref.transpose(1, 2),
                                       torch.ones_like(pref), atol=1e-6))

    def test_agent_relabeling_equivariance(self):
        perm = torch.tensor([2, 0, 3, 1])
        inv = torch.argsort(perm)
        old = torch.tensor([[0, 3, 1, 2],
                            [2, 1, 3, 0],
                            [3, 0, 2, 1]])
        obs_relabel = self.obs.index_select(1, perm)
        relabeled_order = inv[old]
        self.assertTrue(torch.allclose(
            self.p.log_prob(self.obs, old),
            self.p.log_prob(obs_relabel, relabeled_order),
            atol=2e-6))
        self.assertTrue(torch.allclose(
            self.p.demand(self.obs)[:, perm][:, :, perm],
            self.p.demand(obs_relabel), atol=2e-6))

    def test_reject_invalid_order_and_nonfinite_context(self):
        with self.assertRaises(ValueError):
            self.p.log_prob(self.obs, torch.zeros((3, 4), dtype=torch.long))
        with self.assertRaises(ValueError):
            self.p.prefix_scores(self.obs, torch.ones(3, 4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.p.prefix_scores(self.obs, torch.tensor([
                [True, False, False, False],
                [False, False, False, False],
                [True, False, False, False]]))
        with self.assertRaises(ValueError):
            self.p.demand(torch.full_like(self.obs, float("nan")))

    def test_one_agent_and_large_team_stability(self):
        for n in (1, 27):
            with self.subTest(n=n):
                x = torch.randn(2, n, 7)
                order, lp = self.p.sample(x)
                self.assertTrue(torch.isfinite(lp).all())
                self.assertTrue(torch.allclose(
                    lp, self.p.log_prob(x, order), atol=3e-5))


class TestPrefixMATIntegration(unittest.TestCase):
    def _net(self, enabled=True):
        torch.manual_seed(917)
        return MultiAgentTransformer(
            9, 7, 4, 4, n_block=2, n_embd=16, n_head=1,
            agent_order_mode="dependency_prefix", order_pair_rank=8,
            order_attention_bias=enabled)

    def setUp(self):
        self.obs = np.random.RandomState(41).randn(2, 4, 7).astype("float32")
        self.cent = np.zeros((2, 4, 9), dtype="float32")
        self.avail = np.ones((2, 4, 4), dtype="float32")

    def test_sampler_equals_parallel_replay_for_bias_and_nobias(self):
        for use_bias in (False, True):
            with self.subTest(bias=use_bias):
                net = self._net(use_bias)
                with torch.no_grad():
                    action, rollout_lp, values, order, old_order_lp = net.get_actions(
                        self.cent, self.obs, self.avail,
                        return_agent_order=True, return_order_log_prob=True)
                    replay_lp, replay_values, _, replay_order_lp = net(
                        self.cent, self.obs, action, self.avail,
                        agent_order=order, return_order_log_prob=True)
                self.assertTrue(torch.allclose(
                    rollout_lp, replay_lp, atol=2e-5))
                self.assertTrue(torch.allclose(
                    values, replay_values, atol=2e-5))
                self.assertTrue(torch.allclose(
                    old_order_lp, replay_order_lp, atol=2e-6))

    def test_attention_ablation_skips_bias_computation(self):
        net = self._net(False)
        with patch.object(net.order_policy, "conditioning_bias",
                          side_effect=AssertionError("NoBias called gate")):
            action, _, _, order, _ = net.get_actions(
                self.cent, self.obs, self.avail, return_agent_order=True,
                return_order_log_prob=True)
            net(self.cent, self.obs, action, self.avail,
                agent_order=order, return_order_log_prob=True)

    def test_attention_enabled_uses_directional_target_source(self):
        net = self._net(True)
        obs = torch.tensor(self.obs)
        order = torch.tensor([[2, 0, 3, 1], [0, 1, 3, 2]])
        bias = net.order_policy.conditioning_bias(obs, order)
        self.assertEqual(bias.shape, (2, 4, 4))
        self.assertTrue(torch.equal(bias[:, :, 0], torch.zeros(2, 4)))
        demand_logits = net.order_policy.demand_logits(obs)
        # Target position 2, source position 0; shifted key = 1.
        target = int(order[0, 2])
        source = int(order[0, 0])
        expected = math.log(2.) + F.logsigmoid(
            demand_logits[0, target, source].clamp(-8., 8.))
        self.assertTrue(torch.allclose(bias[0, 2, 1], expected))
        self.assertTrue(torch.isfinite(bias).all())
        self.assertGreaterEqual(bias.min().item(), math.log(2.) - 8.01)
        self.assertLessEqual(bias.max().item(), math.log(2.))

    def test_bias_mask_no_future_action_leak(self):
        net = self._net(True)
        obs = torch.tensor(self.obs)
        rep = net.encoder(torch.zeros(2, 4, 37), obs)[1]
        order = torch.tensor([[2, 0, 3, 1], [0, 1, 3, 2]])
        bias = net.order_policy.conditioning_bias(obs, order)
        shifted = torch.zeros(2, 4, 5)
        shifted[:, 0, 0] = 1
        shifted[:, 1:, 1:] = F.one_hot(torch.tensor(
            [[0, 2, 1], [2, 3, 0]]), 4).float()
        logits1 = net.decoder(shifted, rep, obs, attention_bias=bias)
        changed = shifted.clone()
        changed[:, 3, 1:] = F.one_hot(torch.tensor([3, 2]), 4).float()
        logits2 = net.decoder(changed, rep, obs, attention_bias=bias)
        self.assertTrue(torch.allclose(logits1[:, :3], logits2[:, :3], atol=2e-6))

    def test_order_ppograd_and_attention_grad_are_nonzero(self):
        for attention in (False, True):
            with self.subTest(attention=attention):
                net = self._net(attention)
                obs = torch.tensor(self.obs)
                order, lp = net.order_policy.sample(obs)
                net.zero_grad()
                (-lp.mean()).backward()
                self.assertGreater(
                    net.order_policy.demand_query.weight.grad.abs().sum().item(), 0.)
                self.assertGreater(
                    net.order_policy.pair_query.weight.grad.abs().sum().item(), 0.)
                if attention:
                    net.zero_grad()
                    action, _, _, sampled_order, _ = net.get_actions(
                        self.cent, self.obs, self.avail, return_agent_order=True,
                        return_order_log_prob=True)
                    new_logp, _, _, _ = net(
                        self.cent, self.obs, action, self.avail,
                        agent_order=sampled_order, return_order_log_prob=True)
                    (-new_logp.mean()).backward()
                    self.assertGreater(
                        net.order_policy.demand_query.weight.grad.abs().sum().item(), 0.)

    def test_model_initialization_paired_and_incompatible_checkpoint(self):
        torch.manual_seed(51)
        identity = MultiAgentTransformer(9, 7, 4, 4, 2, 16, 1)
        torch.manual_seed(51)
        new = self._net()
        # _net resets the seed; instantiate identically seeded direct model
        torch.manual_seed(51)
        new = MultiAgentTransformer(
            9, 7, 4, 4, 2, 16, 1, agent_order_mode="dependency_prefix",
            order_pair_rank=8)
        for name, param in identity.state_dict().items():
            self.assertTrue(torch.equal(param, new.state_dict()[name]), name)
        with self.assertRaises(RuntimeError):
            identity.load_state_dict(new.state_dict(), strict=True)

    def test_actual_mat_ppo_update_replays_prefix_policy(self):
        # Synthetic PPO update, including old joint permutation likelihood.
        torch.manual_seed(633)
        policy = TransformerPolicy.__new__(TransformerPolicy)
        policy.algorithm_name = "mat"
        policy.num_agents = 3
        policy.share_obs_dim = 9
        policy.obs_dim = 7
        policy.act_dim = 4
        policy.act_num = 1
        policy._use_policy_active_masks = False
        policy.tpdv = dict(dtype=torch.float32, device=torch.device("cpu"))
        policy.transformer = MultiAgentTransformer(
            9, 7, 4, 3, 1, 16, 1, agent_order_mode="dependency_prefix",
            order_pair_rank=8, order_prefix_scale=1.0)
        policy.optimizer = torch.optim.Adam(policy.transformer.parameters(), lr=.005)
        rng = np.random.RandomState(632)
        obs = rng.randn(2, 3, 7).astype(np.float32)
        cent = np.zeros((2, 3, 9), dtype=np.float32)
        ava = np.ones((2, 3, 4), dtype=np.float32)
        zeros = np.zeros((6, 1), dtype=np.float32)
        with torch.no_grad():
            values, actions, old_act_lp, _, _, old_order, old_perm_lp = policy.get_actions(
                cent.reshape(-1, 9), obs.reshape(-1, 7), zeros, zeros,
                zeros, ava.reshape(-1, 4), return_agent_order=True,
                return_order_log_prob=True)
            v2, act2, _, perm2 = policy.evaluate_actions(
                cent.reshape(-1, 9), obs.reshape(-1, 7), zeros, zeros,
                actions, zeros, ava.reshape(-1, 4),
                agent_order=old_order, return_order_log_prob=True)
        self.assertTrue(torch.allclose(values, v2, atol=2e-5))
        self.assertTrue(torch.allclose(old_act_lp, act2, atol=2e-5))
        self.assertTrue(torch.allclose(old_perm_lp, perm2, atol=1e-6))
        args = SimpleNamespace(
            clip_param=.2, ppo_epoch=1, num_mini_batch=1, data_chunk_length=1,
            value_loss_coef=1., entropy_coef=.01, max_grad_norm=10,
            huber_delta=10, use_recurrent_policy=False,
            use_naive_recurrent_policy=False, use_max_grad_norm=True,
            use_clipped_value_loss=False, use_huber_loss=False,
            use_valuenorm=False, use_value_active_masks=False,
            use_policy_active_masks=False, dec_actor=False,
            agent_order_mode="dependency_prefix", order_loss_coef=.5)
        trainer = MATTrainer(args, policy, 3)
        sample = (cent.reshape(-1, 9), obs.reshape(-1, 7), zeros, zeros,
                  actions.detach().numpy(), values.detach().numpy(),
                  (values.detach() + .25).numpy(), zeros, np.ones_like(zeros),
                  old_act_lp.detach().numpy(),
                  np.array([[1.], [1.], [1.], [-1.], [-1.], [-1.]],
                           dtype=np.float32),
                  ava.reshape(-1, 4), old_order.detach().numpy(),
                  old_perm_lp.detach().numpy())
        trainer.ppo_update(sample)
        self.assertTrue(np.isfinite(trainer.last_order_policy_loss))
        self.assertTrue(np.isfinite(trainer.last_order_approx_kl))
        self.assertEqual(trainer.last_order_log_ratio_clamp_fraction, 0)
        self.assertGreater(
            policy.transformer.order_policy.demand_query.weight.grad.abs().sum().item(), 0)

    def test_manifest_roundtrip_prefix_mode(self):
        args = get_config().parse_args([])
        args.env_name = "StarCraft2"
        args.map_name = "3s5z"
        args.algorithm_name = "mat"
        args.agent_order_mode = "dependency_prefix"
        args.store_agent_orders = True
        args.order_prefix_scale = 0.8
        manifest = build_training_manifest(args, "c" * 40)
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "manifest.json"
            f.write_text(json.dumps(manifest))
            loaded = load_training_manifest(f, "3s5z", "dependency_prefix")
            flags = evaluation_model_flags(loaded)
            self.assertEqual(
                flags[flags.index("--order_prefix_scale") + 1], "0.8")


if __name__ == "__main__":
    unittest.main()
