import unittest
from pcse_gym.agent.masked_actorcriticpolicy import MaskedRecurrentActorCriticPolicy, MaskedActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
import pcse_gym.initialize_envs as init_env

import torch


class TestMaskedRecurrentActorCriticPolicy(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env(pcse_env=2)
        self.policy = MaskedRecurrentActorCriticPolicy(observation_space=self.env.observation_space,
                                                       action_space=self.env.action_space,
                                                       lr_schedule=lambda x: 1e-4,
                                                       max_non_zero_actions=4,
                                                       apply_masking=True)
        self.dim_hidden_states = 256

    def test_masking_logic_freq_limit(self):
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

    def test_masking_logic_consecutive(self):
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
        # self.policy.prev_action = torch.tensor([1])
        # self.policy.prev_prev_action = torch.tensor([0])
        self.policy.update_non_zero_action_count(torch.tensor([1]))
        # self.policy.update_non_zero_action_count(torch.tensor([1]))
        # self.assertEqual(actions.item(), 0)
        #
        # self.policy.prev_action = torch.tensor([1])
        # self.policy.prev_prev_action = torch.tensor([1])

        # Test after hitting the non-zero action limit
        actions, values, log_prob, lstm_states = self.policy(obs, lstm_states, episode_starts)
        self.assertEqual(actions.item(), 0)


class TestCustomActorCriticPolicy(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env(pcse_env=2)

        # Create a simple constant learning rate schedule
        lr_schedule = lambda x: 1e-4

        # Instantiate the custom policy with the learning rate schedule
        self.policy = MaskedActorCriticPolicy(observation_space=self.env.observation_space,
                                              action_space=self.env.action_space,
                                              lr_schedule=lr_schedule,
                                              max_non_zero_actions=4,
                                              mask_duration=2)

    def test_masking_logic_limit(self):
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)

        # Set masking to be applied
        self.policy.set_masking(True)

        # Test without hitting the non-zero action limit
        actions, values, log_prob = self.policy(obs)
        self.assertNotEqual(actions.item(), -float('inf'))

        # Simulate non-zero actions to hit the limit and trigger consecutive masking
        self.policy.update_non_zero_action_count(torch.tensor([1]))  # First non-zero action
        self.policy.update_non_zero_action_count(torch.tensor([1]))  # Second non-zero action

        # Test after hitting the non-zero action limit
        actions, values, log_prob = self.policy(obs)
        self.assertEqual(actions.item(), 0)

        # Simulate consecutive timesteps where non-zero actions should be masked
        actions, values, log_prob = self.policy(obs)
        self.assertEqual(actions.item(), 0)

        # Allow for the duration to pass and non-zero actions to be possible again
        self.policy.update_non_zero_action_count(torch.tensor([0]))  # Timestep 3 (non-zero action masked)
        self.policy.update_non_zero_action_count(torch.tensor([0]))  # Timestep 4 (non-zero action masked)

        # Test after consecutive masking duration has passed
        actions, values, log_prob = self.policy(obs)
        self.assertNotEqual(actions.item(), -float('inf'))

    def test_masking_logic_consecutive(self):
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)

        # Set masking to be applied
        self.policy.set_masking(True)

        # Test without hitting the non-zero action limit
        actions, values, log_prob = self.policy(obs)
        self.assertNotEqual(actions.item(), -float('inf'))

        # Simulate non-zero actions to hit the limit
        # self.policy.prev_action = torch.tensor([1])
        # self.policy.prev_prev_action = torch.tensor([0])
        self.policy.update_non_zero_action_count(torch.tensor([1]))
        self.policy.update_non_zero_action_count(torch.tensor([1]))
        # self.assertEqual(actions.item(), 0)
        #
        # self.policy.prev_action = torch.tensor([1])
        # self.policy.prev_prev_action = torch.tensor([1])

        # Test after hitting the non-zero action limit
        actions, values, log_prob = self.policy(obs)
        print(values)
        self.assertEqual(actions.item(), 0)

    def test_timestep_constraint(self):
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)

        self.policy.set_masking(True)

        actions, values, log_prob = self.policy(obs)
        self.assertNotEqual(actions.item(), -float('inf'))

        for _ in range(1, 5):
            actions, values, log_prob = self.policy(obs)
            self.assertEqual(actions.item(), 0)
            # self.policy.update_non_zero_action_count(actions)

        for _ in range(6, 31):
            actions, values, log_prob = self.policy(obs)
            print(actions)
            # self.policy.update_non_zero_action_count(actions)

        for _ in range(31, 45):
            actions, values, log_prob = self.policy(obs)
            self.assertEqual(actions.item(), 0)
            # self.policy.update_non_zero_action_count(actions)