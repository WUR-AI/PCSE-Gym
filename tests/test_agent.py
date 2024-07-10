import unittest
from pcse_gym.agent.masked_rppo import MaskedRecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
import pcse_gym.initialize_envs as init_env

import torch


class TestMaskedRecurrentActorCriticPolicy(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env(pcse_env=2)
        self.policy = MaskedRecurrentActorCriticPolicy(observation_space=self.env.observation_space,
                                         max_non_zero_actions=4)
        self.dim_hidden_states = 256

    def test_masking_logic(self):

        obs, _ = self.env.reset()
        obs = torch.tensor(obs)
        lstm_states = RNNStates(
            (torch.zeros(1, 1, self.dim_hidden_states), torch.zeros(1, 1, self.dim_hidden_states)),
            (torch.zeros(1, 1, self.dim_hidden_states), torch.zeros(1, 1, self.dim_hidden_states))
        )
        episode_starts = torch.tensor([True])

        # Test without hitting the non-zero action limit
        actions, values, log_prob, lstm_states = self.policy(obs, lstm_states, episode_starts)
        self.assertNotEqual(actions.item(), -float('inf'))

        # Simulate non-zero actions to hit the limit
        self.policy.update_non_zero_action_count(torch.tensor([1]))
        self.policy.update_non_zero_action_count(torch.tensor([1]))
        self.policy.update_non_zero_action_count(torch.tensor([1]))
        self.policy.update_non_zero_action_count(torch.tensor([1]))

        # Test after hitting the non-zero action limit
        actions, values, log_prob, lstm_states = self.policy(obs, lstm_states, episode_starts)
        self.assertEqual(actions.item(), 0)