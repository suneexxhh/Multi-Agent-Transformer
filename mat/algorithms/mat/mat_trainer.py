import numpy as np
import torch
import torch.nn as nn
from mat.utils.util import get_gard_norm, huber_loss, mse_loss
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check


class MATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor
        self.learned_orders = getattr(args, "agent_order_mode", "identity") in ("learned", "dependency", "dependency_prefix")
        self.order_loss_coef = getattr(args, "order_loss_coef", 0.1)
        self.order_influence_align_coef = float(getattr(args, "order_influence_align_coef", 0.0))
        self.order_influence_max_alternatives = getattr(args, "order_influence_max_alternatives", 2)
        self.order_influence_effect_scale = float(getattr(args, "order_influence_effect_scale", 1e-5))
        self.order_influence_min_effect = float(getattr(args, "order_influence_min_effect", 1e-7))
        self.cache_order_influence_teacher = bool(getattr(args, "cache_order_influence_teacher", False))
        if self.order_influence_align_coef < 0 or not np.isfinite(self.order_influence_align_coef):
            raise ValueError("order_influence_align_coef must be finite and nonnegative")
        if self.order_influence_align_coef and getattr(args, "agent_order_mode", None) != "dependency_prefix":
            raise ValueError("influence alignment requires dependency_prefix")
        if self.cache_order_influence_teacher and not self.order_influence_align_coef:
            raise ValueError("teacher cache requires enabled influence alignment")
        if not isinstance(self.order_influence_max_alternatives, int) or self.order_influence_max_alternatives < 1:
            raise ValueError("order_influence_max_alternatives must be positive")
        if (not np.isfinite(self.order_influence_effect_scale) or self.order_influence_effect_scale <= 0 or
                not np.isfinite(self.order_influence_min_effect) or self.order_influence_min_effect < 0):
            raise ValueError("invalid influence effect scale or floor")
        self._influence_teacher = None
        self._influence_raw_cache = {}
        if self.learned_orders and self.order_loss_coef <= 0:
            raise ValueError("learned-order PPO requires positive order_loss_coef")
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        old_order_logp = None
        transition_indices = None
        if len(sample) == 15 and self.cache_order_influence_teacher:
            *standard_sample, stored_agent_orders, old_order_logp, transition_indices = sample
        elif len(sample) == 14:
            *standard_sample, stored_agent_orders, old_order_logp = sample
        elif len(sample) == 13:
            *standard_sample, stored_agent_orders = sample
        elif len(sample) == 12:
            standard_sample, stored_agent_orders = sample, None
        else:
            raise ValueError("expected 12 standard, 13 saved order or 14 learned-order fields")
        if self.learned_orders != (old_order_logp is not None):
            raise ValueError("learned PPO requires saved permutation and old log probability")
        if self.cache_order_influence_teacher and transition_indices is None:
            raise ValueError("teacher cache requires stable rollout transition indices")
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = standard_sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        evaluation = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, available_actions_batch, active_masks_batch,
            agent_order=stored_agent_orders,
            **({"return_order_log_prob": True} if self.learned_orders else {}))
        if self.learned_orders:
            values, action_log_probs, dist_entropy, new_order_logp = evaluation
        else:
            values, action_log_probs, dist_entropy = evaluation
        # Fail closed on corrupt replay/new policy distributions rather than
        # turning a failed update into poisoned optimizer parameters.
        for name, tensor in (
                ("old action logp", old_action_log_probs_batch),
                ("new action logp", action_log_probs),
                ("advantages", adv_targ),
                ("values", values)):
            if not torch.isfinite(tensor).all():
                raise ValueError("non-finite PPO " + name)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        if not torch.isfinite(imp_weights).all():
            raise ValueError("non-finite PPO action importance ratio")

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef
        self.last_order_policy_loss = None
        self.last_order_influence_loss = 0.0
        self.last_order_influence_valid_pairs = 0.0
        self.last_order_influence_supervised_pairs = 0.0
        if self.learned_orders:
            old_order_logp = check(old_order_logp).to(**self.tpdv)
            if (not torch.isfinite(old_order_logp).all() or
                    not torch.isfinite(new_order_logp).all()):
                raise ValueError("non-finite learned-order PPO log probability")
            B = old_order_logp.shape[0]
            if (new_order_logp.shape != old_order_logp.shape or
                    adv_targ.shape[0] != B * self.num_agents):
                raise ValueError("joint order sample has inconsistent batch shape")
            # One active-agent normalized advantage per joint team permutation.
            adv = adv_targ.reshape(B, self.num_agents, -1).mean(-1)
            alive = active_masks_batch.reshape(B, self.num_agents, -1).mean(-1)
            team_adv = (adv * alive).sum(-1, keepdim=True) / alive.sum(
                -1, keepdim=True).clamp_min(1.0)
            log_ratio = new_order_logp - old_order_logp
            # Guard the permutation likelihood ratio against float32 overflow.
            # A clamp is only hit by pathological PPO drift; its rate is logged.
            self.last_order_log_ratio_clamp_fraction = (
                log_ratio.detach().abs() > 20.).float().mean().item()
            ratio = torch.exp(log_ratio.clamp(min=-20., max=20.))
            # PPO ordering diagnostics: empirical KL estimate (can be
            # negative on small batches), current ratio clip fraction,
            # and saved-order surprisal. None implies causal dependency.
            self.last_order_approx_kl = (
                -log_ratio.detach()).mean().item()
            self.last_order_clip_fraction = (
                (ratio.detach() - 1.0).abs() > self.clip_param
            ).float().mean().item()
            self.last_order_old_surprisal = (
                -old_order_logp.detach()).mean().item()
            self.last_order_mean_log_ratio = log_ratio.detach().mean().item()
            surrogate_1 = ratio * team_adv
            surrogate_2 = torch.clamp(ratio, 1. - self.clip_param,
                                     1. + self.clip_param) * team_adv
            order_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
            # Factor-wise clipping: action factors and permutation factor are
            # clipped separately; not an exact joint-action PPO ratio.
            loss = loss + self.order_loss_coef * order_loss
            self.last_order_policy_loss = order_loss.detach().item()

        if self.order_influence_align_coef:
            from mat.algorithms.mat.algorithm.influence_alignment import (
                frozen_bias_off_teacher, conditional_action_js, alignment_targets,
                influence_alignment_loss)
            if stored_agent_orders is None or available_actions_batch is None:
                raise ValueError("influence alignment needs PPO rollout order and legal actions")
            if self._influence_teacher is None:
                self._influence_teacher = frozen_bias_off_teacher(self.policy.transformer)
            order = check(stored_agent_orders).to(device=self.device, dtype=torch.long)
            obs = check(obs_batch).to(**self.tpdv).reshape(-1, self.num_agents,
                                                          self.policy.transformer.encoder.obs_dim)
            actions = check(actions_batch).to(device=self.device, dtype=torch.long).reshape(
                -1, self.num_agents, 1)
            legal = check(available_actions_batch).to(device=self.device).reshape(
                -1, self.num_agents, self.policy.transformer.action_dim)
            if self.cache_order_influence_teacher:
                ids = np.asarray(transition_indices)
                if (ids.shape != (obs.shape[0],) or not np.issubdtype(ids.dtype, np.integer)
                        or np.unique(ids).size != ids.size or np.any(ids < 0)):
                    raise ValueError("invalid cache transition indices")
                misses = [row for row, idx in enumerate(ids)
                          if int(idx) not in self._influence_raw_cache]
                if misses:
                    miss = torch.as_tensor(misses, dtype=torch.long, device=self.device)
                    fresh = conditional_action_js(
                        self._influence_teacher, obs[miss], actions[miss],
                        legal[miss], order[miss],
                        max_alternatives=self.order_influence_max_alternatives)
                    for position, row in enumerate(misses):
                        self._influence_raw_cache[int(ids[row])] = tuple(
                            component[position].detach().clone() for component in fresh)
                js, valid, entropy, alt_count = (
                    torch.stack([self._influence_raw_cache[int(idx)][field] for idx in ids])
                    for field in range(4))
            else:
                js, valid, entropy, alt_count = conditional_action_js(
                    self._influence_teacher, obs, actions, legal, order,
                    max_alternatives=self.order_influence_max_alternatives)
            from mat.algorithms.mat.algorithm.agent_ordering import reorder_agents
            targets, confidence = alignment_targets(
                js, valid, entropy, alt_count, reorder_agents(legal, order).sum(-1),
                max_alternatives=self.order_influence_max_alternatives,
                effect_scale=self.order_influence_effect_scale,
                min_effect=self.order_influence_min_effect)
            influence_loss = influence_alignment_loss(
                self.policy.transformer.order_policy, obs, order, targets, confidence)
            loss = loss + self.order_influence_align_coef * influence_loss
            self.last_order_influence_loss = influence_loss.detach().item()
            self.last_order_influence_valid_pairs = valid.sum().item()
            self.last_order_influence_supervised_pairs = (confidence > 0).sum().item()

        if not torch.isfinite(loss):
            raise ValueError("non-finite PPO total loss; optimizer not updated")
        self.policy.optimizer.zero_grad()
        loss.backward()
        scale_param = getattr(getattr(self.policy.transformer, "order_policy", None),
                              "raw_attention_scale", None)
        self.last_attention_scale_grad = (scale_param.grad.detach().abs().item()
                                          if scale_param is not None and scale_param.grad is not None else 0.0)

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()
        self.last_attention_scale = (self.policy.transformer.order_policy.attention_scale().detach().item()
                                     if scale_param is not None else 0.0)

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        # Each PPO update sees one pre-update frozen teacher snapshot. Never
        # construct a teacher from evaluation episodes or refresh per epoch.
        self._influence_teacher = None
        self._influence_raw_cache = {}
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        if self.learned_orders:
            train_info['order_policy_loss'] = 0
            train_info['order_log_ratio_clamp_fraction'] = 0
            train_info['order_approx_kl'] = 0
            train_info['order_clip_fraction'] = 0
            train_info['order_old_surprisal'] = 0
            train_info['order_mean_log_ratio'] = 0
        if self.order_influence_align_coef:
            for name in ("order_influence_loss", "order_influence_valid_pairs",
                         "order_influence_supervised_pairs"):
                train_info[name] = 0
        if getattr(getattr(self.policy.transformer, "order_policy", None),
                   "raw_attention_scale", None) is not None:
            train_info['order_attention_scale'] = 0
            train_info['order_attention_scale_grad'] = 0

        for _ in range(self.ppo_epoch):
            if self.cache_order_influence_teacher:
                data_generator = buffer.feed_forward_generator_transformer(
                    advantages, self.num_mini_batch, include_transition_indices=True)
            else:
                data_generator = buffer.feed_forward_generator_transformer(
                    advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                if self.learned_orders:
                    train_info['order_policy_loss'] += self.last_order_policy_loss
                    train_info['order_log_ratio_clamp_fraction'] += (
                        self.last_order_log_ratio_clamp_fraction)
                    train_info['order_approx_kl'] += self.last_order_approx_kl
                    train_info['order_clip_fraction'] += self.last_order_clip_fraction
                    train_info['order_old_surprisal'] += self.last_order_old_surprisal
                    train_info['order_mean_log_ratio'] += self.last_order_mean_log_ratio
                if self.order_influence_align_coef:
                    train_info['order_influence_loss'] += self.last_order_influence_loss
                    train_info['order_influence_valid_pairs'] += self.last_order_influence_valid_pairs
                    train_info['order_influence_supervised_pairs'] += self.last_order_influence_supervised_pairs
                if 'order_attention_scale' in train_info:
                    train_info['order_attention_scale'] += self.last_attention_scale
                    train_info['order_attention_scale_grad'] += self.last_attention_scale_grad

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        self._influence_teacher = None
        self._influence_raw_cache = {}
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
