"""CPU-only V3.1 teacher, attention, replay and manifest regressions."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from mat.algorithms.mat.algorithm.influence_alignment import (
    alignment_targets, categorical_js, conditional_action_js,
    frozen_bias_off_teacher, influence_alignment_loss)
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.mat.mat_trainer import MATTrainer
from mat.config import get_config
from research.dependency_decoder.checkpoint_manifest import (
    build_training_manifest, evaluation_model_flags, load_training_manifest)


class ToyEncoder(nn.Module):
    def forward(self, state, obs):
        return None, obs


class ToyDecoder(nn.Module):
    def forward(self, shifted, rep, obs):
        b, n, width = shifted.shape
        out = shifted.new_zeros((b, n, width - 1))
        # Key s+1 contains source slot s's previous action. Only prior keys
        # can affect a target, so a future-action mutation cannot leak back.
        for target in range(1, n):
            out[:, target, 0] = 3.0 * shifted[:, 1:target + 1, 1].sum(-1)
        return out


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.encoder = ToyEncoder()
        self.decoder = ToyDecoder()
        self.order_attention_bias = False
        self.eval()


class TestV31InfluenceTeacher(unittest.TestCase):
    def setUp(self):
        self.teacher = ToyTeacher()
        self.obs = torch.zeros(3, 3, 5)
        self.actions = torch.tensor([[[0], [0], [1]]] * 3)
        self.legal = torch.ones(3, 3, 2, dtype=torch.bool)
        self.order = torch.tensor([[2, 0, 1]] * 3)

    def test_js_matches_manual_and_excludes_illegal(self):
        p = torch.tensor([[math.log(.8), math.log(.2), -40.]])
        q = torch.tensor([[math.log(.2), math.log(.8), 40.]])
        js = categorical_js(p, q, torch.tensor([[True, True, False]]))
        manual = .8 * math.log(1.6) + .2 * math.log(.4)
        self.assertAlmostEqual(js.item(), manual, places=6)
        self.assertTrue(torch.isfinite(js).all())

    def test_only_legal_prior_action_and_observable_direction(self):
        js, valid, entropy, count = conditional_action_js(
            self.teacher, self.obs, self.actions, self.legal, self.order)
        self.assertEqual(js.shape, (3, 3, 3))
        self.assertTrue(valid[:, 1, 0].all())
        self.assertTrue(valid[:, 2, 0].all())
        self.assertTrue(valid[:, 2, 1].all())
        self.assertFalse(valid[:, 0, 1].any())
        self.assertFalse(valid.diagonal(dim1=1, dim2=2).any())
        self.assertGreater(js[0, 1, 0].item(), .01)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertEqual(count[0, 0].item(), 1)
        illegal = self.actions.clone()
        illegal[:, 0] = 1
        restricted = self.legal.clone()
        restricted[:, 0, 1] = False
        with self.assertRaisesRegex(ValueError, "illegal"):
            conditional_action_js(self.teacher, self.obs, illegal, restricted, self.order)

    def test_single_legal_action_and_missing_reverse_are_skipped(self):
        legal = self.legal.clone()
        legal[:, 2, 0] = False  # source at slot zero has only observed action 1
        js, valid, _, _ = conditional_action_js(
            self.teacher, self.obs, self.actions, legal, self.order)
        self.assertFalse(valid[:, 1, 0].any())
        self.assertFalse(valid[:, 2, 0].any())
        self.assertFalse(valid[:, 0, 2].any())
        self.assertEqual(js[:, 1, 0].abs().sum().item(), 0.)
        legal = self.legal.clone()
        legal[:, 1, 1] = False  # target at last slot has only observed action 0
        _, valid, _, _ = conditional_action_js(
            self.teacher, self.obs, self.actions, legal, self.order)
        self.assertFalse(valid[:, 2].any())

    def test_teacher_cannot_be_trainable_or_bias_on(self):
        self.teacher.anchor.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            conditional_action_js(self.teacher, self.obs, self.actions, self.legal, self.order)
        self.teacher.anchor.requires_grad_(False)
        self.teacher.order_attention_bias = True
        with self.assertRaisesRegex(ValueError, "Bias OFF"):
            conditional_action_js(self.teacher, self.obs, self.actions, self.legal, self.order)

    def test_position_entropy_stratification_and_missing_direction(self):
        js = torch.zeros(3, 3, 3)
        valid = torch.zeros_like(js, dtype=torch.bool)
        valid[:, 2, 0] = True
        js[:, 2, 0] = torch.tensor([.01, .04, .08])
        entropy = torch.full((3, 3), math.log(2.))
        counts = torch.ones(3, 3, dtype=torch.long)
        counts[:, 2] = 2
        alts = torch.ones(3, 3, dtype=torch.long)
        targets, confidence = alignment_targets(
            js, valid, entropy, alts, counts, max_alternatives=1)
        self.assertTrue((confidence[:, 2, 0] > 0).all())
        self.assertTrue((targets[:, 2, 0] != .5).all())
        self.assertEqual(confidence[:, 0, 2].sum().item(), 0.)
        # Different entropy stratum with one observation cannot supervise.
        entropy[2, 2] = 0.
        _, confidence = alignment_targets(js, valid, entropy, alts, counts,
                                          max_alternatives=1)
        self.assertEqual(confidence[2, 2, 0].item(), 0.)

    def test_v2_scale_js_is_not_silently_filtered(self):
        js = torch.zeros(3, 3, 3)
        js[:, 2, 0] = torch.tensor([5e-6, 1.5e-5, 3e-5])
        valid = torch.zeros_like(js, dtype=torch.bool)
        valid[:, 2, 0] = True
        entropy = torch.full((3, 3), math.log(2.))
        legal_counts = torch.full((3, 3), 2, dtype=torch.long)
        alt_count = torch.ones(3, 3, dtype=torch.long)
        targets, confidence = alignment_targets(js, valid, entropy, alt_count,
                                                legal_counts, max_alternatives=1)
        self.assertGreater((confidence > 0).sum().item(), 1)
        self.assertGreater((targets[:, 2, 0] - .5).abs().max().item(), .1)


class TestV31Model(unittest.TestCase):
    def model(self, mode="dependency_prefix", adaptive=False, attention=True):
        torch.manual_seed(1181)
        return MultiAgentTransformer(
            9, 5, 3, 3, 1, 16, 1, agent_order_mode=mode,
            order_attention_bias=attention,
            adaptive_order_attention_bias=adaptive, order_pair_rank=8)

    def setUp(self):
        rng = np.random.RandomState(130)
        self.obs = rng.randn(2, 3, 5).astype(np.float32)
        self.state = np.zeros((2, 3, 9), dtype=np.float32)
        self.legal = np.ones((2, 3, 3), dtype=np.float32)
        self.order = torch.tensor([[2, 0, 1], [1, 2, 0]])

    def test_frozen_teacher_is_copy_and_bias_off(self):
        model = self.model(adaptive=True)
        teacher = frozen_bias_off_teacher(model)
        self.assertFalse(teacher.order_attention_bias)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        old = next(teacher.parameters()).detach().clone()
        with torch.no_grad():
            next(model.parameters()).add_(1.)
        self.assertTrue(torch.equal(next(teacher.parameters()), old))

    def test_real_mat_teacher_uses_saved_order_and_finite_js(self):
        model = self.model(adaptive=True)
        teacher = frozen_bias_off_teacher(model)
        with torch.no_grad():
            actions = model.get_actions(self.state, self.obs, self.legal,
                                        deterministic=True, agent_order=self.order)[0]
        js, valid, entropy, alternatives = conditional_action_js(
            teacher, self.obs, actions, self.legal, self.order)
        self.assertTrue(torch.isfinite(js).all())
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue((js >= 0).all())
        self.assertTrue(valid[:, 2, 0].all())
        self.assertFalse(valid[:, 0, 2].any())
        self.assertTrue((alternatives == 2).all())

    def test_directional_student_gradient_and_teacher_isolation(self):
        model = self.model(attention=False)
        teacher = frozen_bias_off_teacher(model)
        targets = torch.full((2, 3, 3), .5)
        targets[:, 2, 0] = torch.tensor([.9, .2])
        confidence = torch.zeros_like(targets)
        confidence[:, 2, 0] = 1.
        model.zero_grad()
        loss = influence_alignment_loss(model.order_policy, torch.tensor(self.obs),
                                        self.order, targets, confidence)
        loss.backward()
        self.assertGreater(model.order_policy.demand_query.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        zero = influence_alignment_loss(model.order_policy, torch.tensor(self.obs),
                                        self.order, targets, confidence * 0.)
        self.assertEqual(zero.item(), 0.)

    def test_neutral_initial_attention_equals_bias_off_and_replay(self):
        on = self.model(adaptive=True)
        off = self.model(adaptive=True)
        off.load_state_dict(on.state_dict(), strict=True)
        off.order_attention_bias = False
        obs = torch.tensor(self.obs)
        bias = on.order_policy.conditioning_bias(obs, self.order)
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))
        with torch.no_grad():
            actions, act_lp, _, _, old_order_lp = on.get_actions(
                self.state, self.obs, self.legal, deterministic=True,
                agent_order=self.order, return_agent_order=True,
                return_order_log_prob=True)
            on_lp, _, _, on_order_lp = on(self.state, self.obs, actions, self.legal,
                                          self.order, True)
            with patch.object(off.order_policy, "conditioning_bias",
                              side_effect=AssertionError("NoBias must skip")):
                off_lp, _, _, off_order_lp = off(self.state, self.obs, actions,
                                                 self.legal, self.order, True)
        self.assertTrue(torch.allclose(act_lp, on_lp, atol=2e-5))
        self.assertTrue(torch.allclose(on_lp, off_lp, atol=1e-7))
        self.assertTrue(torch.allclose(old_order_lp, on_order_lp, atol=1e-7))
        self.assertTrue(torch.allclose(on_order_lp, off_order_lp, atol=1e-7))

    def test_bounded_scale_shifted_index_and_gradient(self):
        model = self.model(adaptive=True)
        obs = torch.tensor(self.obs)
        fixed = model.order_policy.conditioning_bias(obs, self.order)
        self.assertEqual(fixed.abs().max().item(), 0.)
        with torch.no_grad():
            model.order_policy.raw_attention_scale.fill_(1.)
        bias = model.order_policy.conditioning_bias(obs, self.order)
        self.assertGreater(bias.abs().max().item(), 0.)
        self.assertLessEqual(abs(model.order_policy.attention_scale().item()), 2.)
        self.assertTrue(torch.equal(bias[:, :, 0], torch.zeros_like(bias[:, :, 0])))
        logits = model.order_policy.demand_logits(obs)
        target, source = self.order[0, 2].item(), self.order[0, 0].item()
        expected = model.order_policy.attention_scale() * (
            math.log(2.) + torch.nn.functional.logsigmoid(logits[0, target, source].clamp(-8., 8.)))
        self.assertTrue(torch.allclose(bias[0, 2, 1], expected))
        model.zero_grad()
        bias[0, 2, 1].backward()
        self.assertGreater(abs(model.order_policy.raw_attention_scale.grad.item()), 0.)
        self.assertGreater(model.order_policy.demand_query.weight.grad.abs().sum().item(), 0.)

    def test_attention_weight_and_action_logprob_response(self):
        on = self.model(adaptive=True)
        with torch.no_grad():
            on.order_policy.raw_attention_scale.fill_(2.)
            # The untrained MAT action head has a tiny initialization gain.
            # Amplify it in this pathway test to resolve action logp effects.
            on.decoder.head[-1].weight.mul_(100.)
        off = self.model(adaptive=True)
        off.load_state_dict(on.state_dict(), strict=True)
        off.order_attention_bias = False
        on_att = on.decoder.blocks[0].attn2
        off_att = off.decoder.blocks[0].attn2
        on_att.capture_attention = off_att.capture_attention = True
        actions = torch.tensor([[[0], [1], [2]], [[1], [2], [0]]])
        on_lp = on(self.state, self.obs, actions, self.legal, self.order)[0]
        off_lp = off(self.state, self.obs, actions, self.legal, self.order)[0]
        self.assertGreater((on_att.last_attention - off_att.last_attention).abs().max().item(), .001)
        self.assertGreater((on_lp - off_lp).abs().max().item(), 1e-7)
        self.assertEqual(torch.triu(on_att.last_attention, diagonal=1).abs().sum().item(), 0.)
        on.zero_grad()
        (-on_lp.mean()).backward()
        self.assertGreater(on.order_policy.raw_attention_scale.grad.abs().item(), 0.)

    def test_identity_paired_initialization_and_strict_manifest_checkpoint(self):
        identity = self.model(mode="identity")
        adaptive = self.model(adaptive=True)
        for name, weight in identity.state_dict().items():
            self.assertTrue(torch.equal(weight, adaptive.state_dict()[name]), name)
        args = get_config().parse_args([])
        args.env_name, args.map_name, args.algorithm_name = "StarCraft2", "3s5z", "mat"
        args.agent_order_mode, args.store_agent_orders = "dependency_prefix", True
        args.adaptive_order_attention_bias = True
        args.order_influence_align_coef = .2
        manifest = build_training_manifest(args, "a" * 40)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            ckpt = Path(tmp) / "transformer_0.pt"
            path.write_text(json.dumps(manifest))
            torch.save(adaptive.state_dict(), ckpt)
            loaded = load_training_manifest(path, "3s5z", "dependency_prefix")
            self.assertIn("--adaptive_order_attention_bias", evaluation_model_flags(loaded))
            reconstructed = self.model(adaptive=loaded["config"]["adaptive_order_attention_bias"])
            reconstructed.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
            with self.assertRaises(RuntimeError):
                self.model(adaptive=False).load_state_dict(torch.load(ckpt, weights_only=True), strict=True)
            legacy = dict(manifest)
            legacy["schema_version"] = 1
            legacy["config"] = {k: v for k, v in manifest["config"].items()
                                if k not in ("adaptive_order_attention_bias", "order_attention_scale_max",
                                             "order_influence_align_coef", "order_influence_max_alternatives",
                                             "order_influence_effect_scale", "order_influence_min_effect")}
            path.write_text(json.dumps(legacy))
            self.assertFalse(load_training_manifest(path, "3s5z", "dependency_prefix")
                             ["config"]["adaptive_order_attention_bias"])

    def test_actual_ppo_auxiliary_path_uses_frozen_snapshot(self):
        model = self.model(adaptive=True)
        policy = TransformerPolicy.__new__(TransformerPolicy)
        policy.algorithm_name, policy.num_agents = "mat", 3
        policy.share_obs_dim, policy.obs_dim, policy.act_dim, policy.act_num = 9, 5, 3, 1
        policy._use_policy_active_masks = False
        policy.tpdv = dict(dtype=torch.float32, device=torch.device("cpu"))
        policy.transformer = model
        policy.optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        obs = np.concatenate((self.obs, self.obs[:1] * 1.1), axis=0)
        cent = np.zeros((3, 3, 9), dtype=np.float32)
        legal = np.ones((3, 3, 3), dtype=np.float32)
        zeros = np.zeros((9, 1), dtype=np.float32)
        with torch.no_grad():
            value, action, action_lp, _, _, order, order_lp = policy.get_actions(
                cent.reshape(-1, 9), obs.reshape(-1, 5), zeros, zeros, zeros,
                legal.reshape(-1, 3), return_agent_order=True,
                return_order_log_prob=True)
        args = SimpleNamespace(
            clip_param=.2, ppo_epoch=1, num_mini_batch=1, data_chunk_length=1,
            value_loss_coef=1., entropy_coef=.01, max_grad_norm=10., huber_delta=10.,
            use_recurrent_policy=False, use_naive_recurrent_policy=False,
            use_max_grad_norm=True, use_clipped_value_loss=False,
            use_huber_loss=False, use_valuenorm=False,
            use_value_active_masks=False, use_policy_active_masks=False,
            dec_actor=False, agent_order_mode="dependency_prefix",
            order_loss_coef=.1, order_influence_align_coef=.2,
            order_influence_max_alternatives=2)
        trainer = MATTrainer(args, policy, 3)
        sample = (cent.reshape(-1, 9), obs.reshape(-1, 5), zeros, zeros,
                  action.numpy(), value.numpy(), (value + .2).numpy(),
                  zeros, np.ones_like(zeros), action_lp.numpy(),
                  np.ones_like(zeros), legal.reshape(-1, 3),
                  order.numpy(), order_lp.numpy())
        before = next(model.parameters()).detach().clone()
        js = torch.zeros(3, 3, 3)
        js[:, 2, 0] = torch.tensor([.01, .04, .08])
        valid = torch.zeros_like(js, dtype=torch.bool)
        valid[:, 2, 0] = True
        entropy = torch.full((3, 3), math.log(3.))
        count = torch.full((3, 3), 2, dtype=torch.long)
        with patch("mat.algorithms.mat.algorithm.influence_alignment.conditional_action_js",
                   return_value=(js, valid, entropy, count)):
            with patch.object(policy.optimizer, "step", wraps=policy.optimizer.step) as step:
                trainer.ppo_update(sample)
                self.assertEqual(step.call_count, 1)
        self.assertGreater(trainer.last_order_influence_supervised_pairs, 0)
        self.assertTrue(math.isfinite(trainer.last_order_influence_loss))
        self.assertTrue(torch.equal(next(trainer._influence_teacher.parameters()), before))
        self.assertTrue(all(p.grad is None for p in trainer._influence_teacher.parameters()))
        self.assertTrue(math.isfinite(trainer.last_attention_scale_grad))


if __name__ == "__main__":
    unittest.main()
