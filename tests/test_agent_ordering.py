"""CPU unit tests; run with: python -m unittest discover -s tests -v.

The tests do NOT require SMAC/SC2.
"""
import unittest

import numpy as np
import torch

from mat.algorithms.mat.algorithm.agent_ordering import (
    compute_agent_order, reorder_agents, restore_agents,
)
from mat.algorithms.utils.transformer_act import (
    discrete_autoregreesive_act, discrete_parallel_act,
    discrete_autoregreesive_ordered_act, discrete_parallel_ordered_act,
    continuous_autoregreesive_ordered_act, continuous_parallel_ordered_act,
)
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer


class CausalToyDecoder(torch.nn.Module):
    """Simple causally shifted logits with known token dependencies."""

    def forward(self, shifted_action, obs_rep, obs):
        action_dim = obs_rep.shape[-1]
        prev_actions = torch.cumsum(shifted_action[..., 1:], dim=1)
        return obs_rep[..., :action_dim] + 0.4 * prev_actions


class TestAgentOrdering(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.obs = torch.tensor([
            [[1., 0.], [3., 0.], [2., 0.]],
            [[3., 0.], [1., 0.], [2., 0.]],
        ])

    def test_identity_and_dynamic_norm_order(self):
        self.assertEqual(compute_agent_order(self.obs, "identity").tolist(),
                         [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(compute_agent_order(self.obs, "obs_norm").tolist(),
                         [[1, 2, 0], [0, 2, 1]])

    def test_fixed_random_reproducible(self):
        a = compute_agent_order(self.obs, "random_fixed", 42)
        b = compute_agent_order(self.obs, "random_fixed", 42)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[0], a[1]))

    def test_gather_scatter_batch_specific(self):
        order = compute_agent_order(self.obs, "obs_norm")
        for data in [torch.randn(2, 3, 5), torch.randn(2, 3, 2, 4),
                     torch.arange(6).reshape(2, 3)]:
            restored = restore_agents(reorder_agents(data, order), order)
            self.assertTrue(torch.equal(data, restored))

    def test_invalid_modes_and_shapes(self):
        with self.assertRaises(ValueError):
            compute_agent_order(self.obs, "causal")
        with self.assertRaises(ValueError):
            reorder_agents(torch.randn(2, 4, 3), torch.zeros(2, 3, dtype=torch.long))

    def test_tied_observations_keep_agent_id(self):
        zeros = torch.zeros(2, 3, 4)
        self.assertEqual(compute_agent_order(zeros, "obs_norm").tolist(),
                         [[0, 1, 2], [0, 1, 2]])


class TestOrderedActionDecoding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.batch, self.agents, self.actions = 2, 3, 4
        self.tpdv = dict(dtype=torch.float32, device=torch.device("cpu"))
        self.obs_rep = torch.randn(self.batch, self.agents, self.actions)
        self.obs = torch.randn(self.batch, self.agents, 7)
        self.decoder = CausalToyDecoder()
        self.order = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.long)

    def test_identity_order_matches_original_exactly(self):
        identity = compute_agent_order(self.obs, "identity")
        original = discrete_autoregreesive_act(
            self.decoder, self.obs_rep, self.obs, self.batch,
            self.agents, self.actions, self.tpdv, deterministic=True)
        ordered = discrete_autoregreesive_ordered_act(
            self.decoder, self.obs_rep, self.obs, self.batch,
            self.agents, self.actions, self.tpdv, identity, deterministic=True)
        for lhs, rhs in zip(original, ordered):
            self.assertTrue(torch.equal(lhs, rhs))
        eval_old = discrete_parallel_act(
            self.decoder, self.obs_rep, self.obs, original[0], self.batch,
            self.agents, self.actions, self.tpdv)
        eval_new = discrete_parallel_ordered_act(
            self.decoder, self.obs_rep, self.obs, original[0], self.batch,
            self.agents, self.actions, self.tpdv, identity)
        for lhs, rhs in zip(eval_old, eval_new):
            self.assertTrue(torch.equal(lhs, rhs))

    def test_autoregressive_teacher_forcing_consistency(self):
        actions, sample_log = discrete_autoregreesive_ordered_act(
            self.decoder, self.obs_rep, self.obs, self.batch,
            self.agents, self.actions, self.tpdv, self.order)
        train_log, entropy = discrete_parallel_ordered_act(
            self.decoder, self.obs_rep, self.obs, actions, self.batch,
            self.agents, self.actions, self.tpdv, self.order)
        self.assertTrue(torch.allclose(sample_log, train_log, atol=1e-6))
        self.assertEqual(entropy.shape, (self.batch, self.agents, 1))

    def test_available_action_mask_follows_original_agents(self):
        # Exactly one action is legal for each original agent ID.
        legal = torch.tensor([[2, 1, 3], [0, 3, 1]], dtype=torch.long)
        mask = torch.zeros(self.batch, self.agents, self.actions)
        mask.scatter_(-1, legal.unsqueeze(-1), 1)
        action, _ = discrete_autoregreesive_ordered_act(
            self.decoder, self.obs_rep, self.obs, self.batch,
            self.agents, self.actions, self.tpdv, self.order,
            available_actions=mask, deterministic=True)
        self.assertTrue(torch.equal(action.squeeze(-1), legal))
        log_prob, _ = discrete_parallel_ordered_act(
            self.decoder, self.obs_rep, self.obs, action, self.batch,
            self.agents, self.actions, self.tpdv, self.order, mask)
        self.assertTrue(torch.allclose(log_prob, torch.zeros_like(log_prob), atol=1e-6))

    def test_full_mat_forward_and_sampling_identity_random_norm(self):
        # Smoke-test the real transformer, not just a toy decoder.
        for mode in ("identity", "random_fixed", "obs_norm"):
            with self.subTest(mode=mode):
                torch.manual_seed(77)
                mat = MultiAgentTransformer(
                    state_dim=9, obs_dim=7, action_dim=4, n_agent=3,
                    n_block=1, n_embd=16, n_head=1, action_type="Discrete",
                    agent_order_mode=mode, agent_order_seed=19,
                )
                obs = torch.randn(2, 3, 7).numpy()
                state = np.zeros((2, 3, 9), dtype=np.float32)
                masks = np.ones((2, 3, 4), dtype=np.float32)
                mat.eval()
                with torch.no_grad():
                    sampled, lp_sample, values = mat.get_actions(
                        state, obs, masks, deterministic=True)
                    lp_eval, eval_values, entropy = mat(state, obs, sampled, masks)
                self.assertTrue(torch.allclose(lp_sample, lp_eval, atol=2e-5))
                self.assertTrue(torch.allclose(values, eval_values, atol=1e-6))
                self.assertEqual(sampled.shape, (2, 3, 1))
                self.assertEqual(entropy.shape, (2, 3, 1))

    def test_mat_gradient_reaches_decoder(self):
        mat = MultiAgentTransformer(
            state_dim=9, obs_dim=7, action_dim=4, n_agent=3,
            n_block=1, n_embd=16, n_head=1, action_type="Discrete",
            agent_order_mode="obs_norm")
        obs = np.random.RandomState(2).randn(2, 3, 7).astype(np.float32)
        state = np.zeros((2, 3, 9), dtype=np.float32)
        action = np.random.RandomState(3).randint(0, 4, (2, 3, 1))
        action_lp, values, _ = mat(state, obs, action)
        (-action_lp.mean() + values.mean()).backward()
        grads = [p.grad for p in mat.decoder.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum().item() > 0 for g in grads))




class ContinuousCausalToyDecoder(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, shifted_action, obs_rep, obs):
        action_dim = shifted_action.shape[-1]
        return obs_rep[..., :action_dim] + 0.25 * torch.cumsum(shifted_action, dim=1)


class TestContinuousAndCausalMask(unittest.TestCase):
    def test_continuous_sampling_eval_logprob_consistency(self):
        torch.manual_seed(104)
        batch_size, n_agent, action_dim = 2, 3, 2
        order = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.long)
        obs_rep = torch.randn(batch_size, n_agent, 4)
        obs = torch.randn(batch_size, n_agent, 7)
        decoder = ContinuousCausalToyDecoder(action_dim)
        tpdv = dict(dtype=torch.float32, device=torch.device("cpu"))
        actions, lp_sample = continuous_autoregreesive_ordered_act(
            decoder, obs_rep, obs, batch_size, n_agent,
            action_dim, tpdv, order, deterministic=False)
        lp_eval, entropy = continuous_parallel_ordered_act(
            decoder, obs_rep, obs, actions, batch_size,
            n_agent, action_dim, tpdv, order)
        self.assertEqual(actions.shape, (batch_size, n_agent, action_dim))
        self.assertEqual(entropy.shape, (batch_size, n_agent, action_dim))
        self.assertTrue(torch.allclose(lp_sample, lp_eval, atol=1e-6))

    def test_decoder_future_shifted_actions_cannot_change_past_logits(self):
        torch.manual_seed(24)
        model = MultiAgentTransformer(
            state_dim=9, obs_dim=7, action_dim=4, n_agent=3,
            n_block=2, n_embd=16, n_head=1, action_type="Discrete",
            agent_order_mode="obs_norm")
        model.eval()
        rep = torch.randn(2, 3, 16)
        obs = torch.randn(2, 3, 7)
        x0 = torch.zeros(2, 3, 5)
        x0[:, 0, 0] = 1
        x1 = x0.clone()
        # This token is only at the final position; it must not leak into
        # the first two logits, because all Decoder attention is causal.
        x1[:, 2, 1:] = torch.tensor([1., 0., 0., 0.])
        with torch.no_grad():
            logits0 = model.decoder(x0, rep, obs)
            logits1 = model.decoder(x1, rep, obs)
        self.assertTrue(torch.allclose(logits0[:, :2], logits1[:, :2], atol=1e-7))



class TestSamplingDevices(unittest.TestCase):
    """Regression for CPU outputs mixed with CUDA permutation indices.

    GitHub hosted CI tests CPU and skips CUDA; the same unittest suite, run on
    the SMAC GPU server, executes the CUDA checks automatically.
    """

    def _exercise_mat_model(self, device):
        torch.manual_seed(51)
        for mode in ("identity", "random_fixed", "obs_norm"):
            with self.subTest(device=device, mode=mode):
                mat = MultiAgentTransformer(
                    state_dim=9, obs_dim=7, action_dim=4, n_agent=3,
                    n_block=1, n_embd=16, n_head=1,
                    device=torch.device(device), action_type="Discrete",
                    agent_order_mode=mode, agent_order_seed=5)
                mat.eval()
                obs = np.random.RandomState(51).randn(2, 3, 7).astype(np.float32)
                state = np.zeros((2, 3, 9), dtype=np.float32)
                legal = np.ones((2, 3, 4), dtype=np.float32)
                with torch.no_grad():
                    actions, sampled_lp, values = mat.get_actions(
                        state, obs, legal, deterministic=True)
                    evaluated_lp, other_values, _ = mat(
                        state, obs, actions, legal)
                target_device = torch.device(device)
                for tensor in (actions, sampled_lp, values, evaluated_lp, other_values):
                    self.assertEqual(tensor.device, target_device)
                self.assertTrue(torch.allclose(sampled_lp, evaluated_lp, atol=2e-5))

    def test_cpu_sampling_devices(self):
        self._exercise_mat_model("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is not available")
    def test_cuda_sampling_devices(self):
        self._exercise_mat_model("cuda:0")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is not available")
    def test_cuda_continuous_action_devices(self):
        torch.manual_seed(101)
        device = torch.device("cuda:0")
        obs_rep = torch.randn(2, 3, 4, device=device)
        obs = torch.randn(2, 3, 7, device=device)
        order = torch.tensor([[1, 0, 2], [2, 1, 0]], device=device)
        decoder = ContinuousCausalToyDecoder(2).to(device)
        tpdv = dict(dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, sampled_lp = continuous_autoregreesive_ordered_act(
                decoder, obs_rep, obs, 2, 3, 2, tpdv, order)
            evaluated_lp, _ = continuous_parallel_ordered_act(
                decoder, obs_rep, obs, actions, 2, 3, 2, tpdv, order)
        self.assertEqual(actions.device, device)
        self.assertEqual(sampled_lp.device, device)
        self.assertTrue(torch.allclose(sampled_lp, evaluated_lp, atol=1e-6))

if __name__ == "__main__":
    unittest.main()
