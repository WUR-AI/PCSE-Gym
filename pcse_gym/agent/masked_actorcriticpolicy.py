import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from typing import Any, Dict, List, Optional, Type, Union, Tuple

from torch.distributions.utils import logits_to_probs


class MaskedRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, max_non_zero_actions: int = 4, mask_duration: int = 3, apply_masking=True, **kwargs):
        super(MaskedRecurrentActorCriticPolicy, self).__init__(*args, **kwargs)
        self.action_dist = MaskedCategoricalDistribution(action_dim=self.action_space.n)
        self.max_non_zero_actions = max_non_zero_actions
        self.non_zero_action_count = 0
        self.apply_masking = apply_masking
        self.mask_duration = mask_duration
        self.consecutive_mask_counter = 0
        self.episode_step = 0
        self.start_actions = 5
        self.end_actions = 30


    def set_masking(self, apply_masking: bool):
        self.apply_masking = apply_masking

    def reset_non_zero_action_count(self):
        self.non_zero_action_count = 0
        self.consecutive_mask_counter = 0
        self.episode_step = 0

    def update_non_zero_action_count(self, actions: th.Tensor):
        self.episode_step += 1
        if th.any(actions != 0):
            self.non_zero_action_count += th.sum(actions != 0).item()
            self.consecutive_mask_counter = self.mask_duration
        elif self.consecutive_mask_counter > 0:
            self.consecutive_mask_counter -= 1

    def forward(self,
                obs: th.Tensor,
                lstm_states: RNNStates,
                episode_starts: th.Tensor,
                deterministic: bool = False
                ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Copy from recurrent SB3_contrib
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alis
        else:
            pi_features, vf_features = features
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts,
                                                               self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        '''Some sanity checks'''

        # get action distribution from policy network
        distribution = self._get_action_dist_from_latent(latent_pi)
        # return action logist from action space
        action_logits = distribution.distribution.logits

        # Apply masking to logits based on the number of non-zero actions taken
        if self.apply_masking and (
                self.non_zero_action_count >= self.max_non_zero_actions or
                self.consecutive_mask_counter > 0 or
                self.episode_step < self.start_actions or
                self.episode_step >= self.end_actions):
            action_mask = th.zeros_like(action_logits)
            # Mask all non-zero actions with -inf
            # ensuring they never get picked after the condition
            action_mask[:, 1:] = th.tensor(-1e+8)
            # modify logits
            action_logits += action_mask

        # Update distribution with masked logits
        # !! Only works with Discrete actions
        distribution.distribution = th.distributions.Categorical(logits=action_logits)

        values = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Update action count if non-zero
        self.update_non_zero_action_count(actions)

        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, max_non_zero_actions: int = 4, mask_duration: int = 3, apply_masking=True, **kwargs):
        super(MaskedActorCriticPolicy, self).__init__(*args, **kwargs)
        # choose action distribution
        # self.action_dist = MaskedCategorical()
        self.max_non_zero_actions = max_non_zero_actions
        self.non_zero_action_count = 0
        self.apply_masking = apply_masking
        self.mask_duration = mask_duration
        self.consecutive_mask_counter = 0
        self.episode_step = 0
        self.start_actions = 5
        self.end_actions = 30

    def set_masking(self, apply_masking: bool):
        self.apply_masking = apply_masking

    def reset_non_zero_action_count(self):
        # print('reset policy counters!')
        self.non_zero_action_count = 0
        self.consecutive_mask_counter = 0
        self.episode_step = 0

    def update_non_zero_action_count(self, actions: th.Tensor):
        self.episode_step += 1
        if th.any(actions != 0):
            self.non_zero_action_count += th.sum(actions != 0).item()
            self.consecutive_mask_counter = self.mask_duration
        elif self.consecutive_mask_counter > 0:
            self.consecutive_mask_counter -= 1

    def forward(self,
                obs: th.Tensor,
                deterministic: bool = False
                ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Get the action logits
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_logits = distribution.distribution.logits

        # Apply masking to logits based on the number of non-zero actions taken and if masking is enabled
        if self.apply_masking and (
                self.non_zero_action_count >= self.max_non_zero_actions or
                self.consecutive_mask_counter > 0 or
                self.episode_step < self.start_actions or
                self.episode_step >= self.end_actions):
            # print(f'masked! at step {self.episode_step}, non zero action {self.non_zero_action_count}, and consecutive mask counter {self.consecutive_mask_counter}')
            action_mask = th.zeros_like(action_logits)
            action_mask[:, 1:] = th.tensor(-1e8)  # Mask all non-zero actions
            action_logits = action_logits + action_mask

        # Update distribution with masked logits
        distribution.distribution = MaskedCategorical(logits=action_logits)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        # Update non-zero action count and consecutive action counter
        # if actions.item() > 0:
        #     print(f'fertilized {actions.item()} in step {self.episode_step}')
        self.update_non_zero_action_count(actions)

        return actions, values, log_prob

    def l2_regularization(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += th.norm(param, 2)

        return l2_reg

    def l1_regularization(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += th.sum(th.abs(param))

        return l2_reg


class MaskedCategorical(Categorical):
    """
    Modified PyTorch Categorical distribution with support for invalid action masking.

    To instantiate, must provide either probs or logits, but not both.

    :param probs: Tensor containing finite non-negative values, which will be renormalized
        to sum to 1 along the last dimension.
    :param logits: Tensor of unnormalized log probabilities.
    :param validate_args: Whether or not to validate that arguments to methods like lob_prob()
        and icdf() match the distribution's shape, support, etc.
    :param masks: An optional boolean ndarray of compatible shape with the distribution.
        If True, the corresponding choice's logit value is preserved. If False, it is set to a
        large negative value, resulting in near 0 probability.
    """

    def __init__(
        self,
        probs: Optional[th.Tensor] = None,
        logits: Optional[th.Tensor] = None,
        validate_args: Optional[bool] = None,
        masks: Optional[np.ndarray] = None,
    ):
        self.masks: Optional[th.Tensor] = None
        super().__init__(probs, logits, validate_args)
        self._original_logits = self.logits
        self.apply_masking(masks)

    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        """
        Eliminate ("mask out") chosen categorical outcomes by setting their probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, it is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """

        if masks is not None:
            device = self.logits.device
            self.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(self.logits.shape)
            HUGE_NEG = th.tensor(-1e8, dtype=self.logits.dtype, device=device)

            logits = th.where(self.masks, self._original_logits, HUGE_NEG)
        else:
            self.masks = None
            logits = self._original_logits

        # Reinitialize with updated logits
        super().__init__(logits=logits)

        # self.probs may already be cached, so we must force an update
        self.probs = logits_to_probs(self.logits)

    def entropy(self) -> th.Tensor:
        if self.masks is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0, device=device))
        return -p_log_p.sum(-1)


class MaskedCategoricalDistribution(MaskedCategorical):
    """
    Categorical distribution for discrete actions. Supports invalid action masking.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.distribution: Optional[MaskedCategorical] = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(
        self, action_logits: th.Tensor
    ):
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, self.action_dim)
        self.distribution = MaskedCategorical(logits=reshaped_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        assert self.distribution is not None, "Must set distribution parameters"
        self.distribution.apply_masking(masks)