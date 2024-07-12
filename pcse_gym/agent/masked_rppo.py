import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.callbacks import BaseCallback
from pcse_gym.utils.eval import EvalCallback
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from typing import Any, Dict, List, Optional, Type, Union, Tuple


class MaskedRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, max_non_zero_actions: int = 4, apply_masking=False, **kwargs):
        super(MaskedRecurrentActorCriticPolicy, self).__init__(*args, **kwargs)
        self.max_non_zero_actions = max_non_zero_actions
        self.non_zero_action_count = 0
        self.apply_masking = apply_masking

    def set_masking(self, apply_masking: bool):
        self.apply_masking = apply_masking

    def reset_non_zero_action_count(self):
        self.non_zero_action_count = 0

    def update_non_zero_action_count(self, actions: torch.Tensor):
        self.non_zero_action_count += torch.sum(actions != 0).item()

    def forward(self,
                obs: torch.Tensor,
                lstm_states: RNNStates,
                episode_starts: torch.Tensor,
                deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
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
        if self.apply_masking and self.non_zero_action_count >= self.max_non_zero_actions:
            action_mask = torch.ones_like(action_logits)
            # Mask all non-zero actions with -inf
            # ensuring they never get picked after the condition
            action_mask[:, 1:] = -float('inf')
            # modify logits
            action_logits += action_mask

        # Update distribution with masked logits
        # !! Only works with Discrete actions
        distribution.distribution = torch.distributions.Categorical(logits=action_logits)

        values = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Update action count if non-zero
        self.update_non_zero_action_count(actions)

        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)


# class MaskedCallback(BaseCallback):
#     def __init__(self, env_eval, test_years,
#                  train_years, train_locations,
#                  test_locations, seed, pcse_model,
#                  comet_log, multiprocess, eval_freq, verbose=0, **kwargs):
#         super(MaskedCallback, self).__init__(verbose)
#         self.eval_callback = EvalCallback(env_eval=env_eval, test_years=test_years,
#                                           train_years=train_years, train_locations=train_locations,
#                                           test_locations=test_locations, seed=seed, pcse_model=pcse_model,
#                                           comet_experiment=comet_log, multiprocess=multiprocess, eval_freq=eval_freq,
#                                           **kwargs)
#
#     def _on_step(self) -> bool:
#         # Call the evaluation callback at the specified frequency
#         self.eval_callback._on_step()
#
#         # Check if the current episode has ended
#
#
#         return True
#
#     # def _on_training_start(self) -> None:
#     #     self.eval_callback._on_training_start()
#     #
#     # def _on_training_end(self) -> None:
#     #     self.eval_callback._on_training_end()
#     #
#     # def _on_rollout_start(self) -> None:
#     #     self.eval_callback._on_rollout_start()
#     #
#     # def _on_rollout_end(self) -> None:
#     #     self.eval_callback._on_rollout_end()
