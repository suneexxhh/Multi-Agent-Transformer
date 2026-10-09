"""Validate real MAT joint log-probs and off-policy weighted history shift.

Conditional *policy* shift with a fixed encoder representation/order is
not automatically a policy-performance gain, nor a causal discovery test.
"""
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from research.dependency_decoder.frozen_policy_sampling import (
    sample_frozen_decoder_histories, teacher_forced_history_log_probs,
    self_normalized_policy_weights,
)
from research.dependency_decoder.expected_action_dependency import (
    probe_action_histories, expected_dependency,
)


class BiasedANDDecoder(nn.Module):
    """A frozen decoder with parent-action distribution shift.

    The structural target action dependence is the SAME for both
    models. Only logits for agents 0/1's action 2 change via bias.
    This isolates the effect of a changed action-history distribution.
    """

    def __init__(self, parent_bias):
        super().__init__()
        self.parent_bias = parent_bias

    def forward(self, shifted, rep, obs):
        b, n, width = shifted.shape
        if n != 3 or width != 4 or rep.shape != (b, 3, 3):
            raise ValueError("expected [B,3,4] and one-hot agent IDs")
        logits = shifted.new_zeros((b, n, 3))
        active_parent = rep[..., 0] + rep[..., 1]
        logits[..., 2] = self.parent_bias * active_parent
        previous_id = F.pad(rep[:, :-1], (0, 0, 1, 0))
        triggered = (previous_id * shifted[..., 3:4]).cumsum(1)
        logits[..., 1] = 4.0 * triggered[..., 0] * triggered[..., 1] * rep[..., 2]
        return logits


class TestFrozenPolicySampling(unittest.TestCase):
    @staticmethod
    def context(b=2):
        rep = torch.eye(3).unsqueeze(0).expand(b, -1, -1).clone()
        obs = torch.zeros(b, 3, 2)
        legal = torch.ones(b, 3, 3)
        order = torch.arange(3).expand(b, -1).clone()
        return rep, obs, legal, order

    def test_reuse_mat_sampling_and_teacher_forced_joint_logq(self):
        torch.manual_seed(9701)
        model = MultiAgentTransformer(
            state_dim=9, obs_dim=7, action_dim=4, n_agent=3,
            n_block=1, n_embd=16, n_head=1,
            agent_order_mode="obs_norm", action_type="Discrete").eval()
        state = torch.zeros(2, 3, 37)
        obs = torch.randn(2, 3, 7)
        legal = torch.ones(2, 3, 4)
        legal[:, 0, 2] = 0
        order = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.long)
        with torch.no_grad():
            _, rep = model.encoder(state, obs)
        histories, agent_logq, joint_logq = sample_frozen_decoder_histories(
            model.decoder, rep, obs, legal, order, 6, max_context_batch=3)
        replay_lp, replay_joint = teacher_forced_history_log_probs(
            model.decoder, rep, obs, legal, order, histories, max_context_batch=2)
        self.assertEqual(histories.shape, (6, 2, 3, 1))
        self.assertEqual(agent_logq.shape, histories.shape)
        self.assertTrue(torch.allclose(agent_logq, replay_lp, atol=1e-5))
        self.assertTrue(torch.allclose(joint_logq, replay_joint, atol=1e-5))
        self.assertTrue((histories[:, :, 0, 0] != 2).all())
        with torch.no_grad():
            # The FULL MAT's public teacher-forced policy path must
            # agree with the offline decoder-only evaluator.
            for h in (0, 4):
                actual_lp, _, _ = model(
                    state, obs, histories[h], legal, agent_order=order)
                self.assertTrue(torch.allclose(
                    actual_lp, replay_lp[h], atol=1e-5))
        self.assertFalse(agent_logq.requires_grad)
        self.assertFalse(replay_lp.requires_grad)

    def test_policy_shift_weights_recover_exact_target_parent_probability(self):
        torch.manual_seed(9702)
        rep, obs, legal, order = self.context(b=1)
        q, p = BiasedANDDecoder(-1.0).eval(), BiasedANDDecoder(1.0).eval()
        actions, original_lp, logq = sample_frozen_decoder_histories(
            q, rep, obs, legal, order, 2400, max_context_batch=128)
        checked_lp, checked_logq = teacher_forced_history_log_probs(
            q, rep, obs, legal, order, actions, max_context_batch=96)
        _, logp = teacher_forced_history_log_probs(
            p, rep, obs, legal, order, actions, max_context_batch=96)
        self.assertTrue(torch.allclose(logq, checked_logq, atol=1e-6))
        self.assertTrue(torch.allclose(original_lp, checked_lp, atol=1e-6))
        weights, ess = self_normalized_policy_weights(logp, logq)
        p2_q = math.exp(-1)/(2 + math.exp(-1))
        p2_p = math.exp(1)/(2 + math.exp(1))
        indicator = (actions[:, 0, 1, 0] == 2).float()
        observed_q = indicator.mean().item()
        estimated_p = (weights[:, 0] * indicator).sum().item()
        self.assertAlmostEqual(observed_q, p2_q, delta=0.04)
        self.assertAlmostEqual(estimated_p, p2_p, delta=0.05)
        self.assertGreater(estimated_p, observed_q + 0.25)
        self.assertGreater(ess[0].item(), 100)
        self.assertLess(ess[0].item(), 2400)
        self.assertTrue(torch.allclose(weights.sum(0), torch.ones(1)))
        print(
            "FROZEN_MAT_SHIFT "
            f"q_activation={observed_q:.4f} "
            f"target_IS_activation={estimated_p:.4f} "
            f"target_exact_activation={p2_p:.4f} "
            f"ESS={ess[0].item():.1f}", flush=True)

    def test_weighted_dependency_responds_to_frozen_policy_shift(self):
        torch.manual_seed(9703)
        rep, obs, legal, order = self.context(b=1)
        q, p = BiasedANDDecoder(-1.0).eval(), BiasedANDDecoder(1.0).eval()
        histories, _, logq = sample_frozen_decoder_histories(
            q, rep, obs, legal, order, 600, max_context_batch=128)
        _, logp = teacher_forced_history_log_probs(
            p, rep, obs, legal, order, histories, max_context_batch=128)
        weights, _ = self_normalized_policy_weights(logp, logq)
        scores, valid = probe_action_histories(
            q, rep, obs, histories, legal, order,
            max_context_batch=64, max_counterfactual_batch=128)
        unweighted, _, _ = expected_dependency(scores, valid)
        reweighted, _, _ = expected_dependency(scores, valid, weights)
        self.assertGreater(reweighted[0, 0, 2].item(),
                           unweighted[0, 0, 2].item() + 0.07)
        self.assertGreater(reweighted[0, 1, 2].item(),
                           unweighted[0, 1, 2].item() + 0.07)
        self.assertTrue(valid[:, :, 0, 2].all().item())
        self.assertTrue(valid[:, :, 1, 2].all().item())

    def test_failure_modes_and_eval_guards(self):
        rep, obs, legal, order = self.context()
        frozen = BiasedANDDecoder(0).eval()
        with self.assertRaises(ValueError):
            sample_frozen_decoder_histories(frozen, rep, obs, legal, order, 0)
        with self.assertRaises(ValueError):
            sample_frozen_decoder_histories(
                frozen, rep, obs, legal, order, 2, max_context_batch=0)
        frozen.train()
        with self.assertRaises(ValueError):
            sample_frozen_decoder_histories(frozen, rep, obs, legal, order, 2)
        frozen.eval()
        invalid = legal.clone()
        invalid[:, 2] = 0
        with self.assertRaises(ValueError):
            sample_frozen_decoder_histories(frozen, rep, obs, invalid, order, 2)
        with self.assertRaises(ValueError):
            teacher_forced_history_log_probs(
                frozen, rep, obs, legal, order,
                torch.zeros(2, 2, 3, 1, dtype=torch.float))
        histories = torch.zeros(2, 2, 3, 1, dtype=torch.long)
        histories[:, :, 1] = 2
        illegal = legal.clone()
        illegal[:, 1, 2] = 0
        with self.assertRaises(ValueError):
            teacher_forced_history_log_probs(
                frozen, rep, obs, illegal, order, histories)
        with self.assertRaises(ValueError):
            self_normalized_policy_weights(
                torch.zeros(3, 1), torch.zeros(2, 1))
        with self.assertRaises(ValueError):
            self_normalized_policy_weights(
                torch.full((2, 1), float("-inf")), torch.zeros(2, 1))


if __name__ == "__main__":
    unittest.main()
