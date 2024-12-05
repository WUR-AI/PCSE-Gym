from datetime import timedelta

import gymnasium as gym
from gymnasium import ActionWrapper
import numpy as np

import pcse

import pcse_gym.envs.sb3


class ActionConstrainer(ActionWrapper):
    """
    Action Wrapper to limit fertilization actions
    """
    def __init__(self, env, action_limit=0, n_budget=0, temporal=False):
        super(ActionConstrainer, self).__init__(env)
        self.counter = 0
        self.action_limit = action_limit
        self.n_counter = 0
        self.temporal = temporal
        self.n_budget = n_budget

    def action(self, action):
        if self.action_limit > 0:
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.freq_limiter_discrete(action)
            elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
                action = self.freq_limiter_multi_discrete(action)
        if self.n_budget > 0:
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.discrete_n_budget(action)
            elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
                action = self.multi_discrete_n_budget(action)
        if self.temporal is not False:
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.discrete_temporal_constraint(action)
            if isinstance(self.action_space, gym.spaces.MultiDiscrete):
                action = self.multi_discrete_temporal_constraint(action)
        return action

    def discrete_temporal_constraint(self, action):
        if self.sb3_env.dvs < 0.01:
            action = 0
            return action
        if self.sb3_env.dvs > 1:
            action = 0
            return action
        return action

    def multi_discrete_temporal_constraint(self, action):
        if self.sb3_env.dvs < 0.01:
            action[0] = 0
            return action
        if self.sb3_env.dvs > 1:
            action[0] = 0
            return action
        return action

    def discrete_n_budget(self, action):
        if self.n_counter == self.n_budget:
            action = 0
            return action
        self.n_counter += action * 10
        if self.n_counter > self.n_budget:
            action = ((self.n_budget + action * 10) - self.n_counter) / 10
            self.n_counter = self.n_budget
        return action

    def multi_discrete_n_budget(self, action):
        if self.n_counter == self.n_budget:
            action = action.copy()  # Needed for RLlib VecEnvs
            action[0] = 0
            return action
        self.n_counter += action[0] * 10
        if self.n_counter > self.n_budget:
            action = action.copy()
            action[0] = ((self.n_budget + action[0] * 10) - self.n_counter) / 10
            self.n_counter = self.n_budget
        return action

    def freq_limiter_discrete(self, action):
        if action != 0:  # if there's an action, increase the counter
            self.counter += 1
        if self.counter > self.action_limit:  # return 0 if the action exceeds limit
            action = 0
        return action

    def freq_limiter_multi_discrete(self, action):
        if action[0] != 0:
            self.counter += 1
        if self.counter > self.action_limit:
            action = action.copy()
            action[0] = 0
        return action

    def reset(self, **kwargs):
        self.counter = 0
        self.n_counter = 0
        return self.env.reset(**kwargs)


def ratio_rescale(value, old_max=None, old_min=None, new_max=None, new_min=None):
    new_value = (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return new_value


def non_linear_ratio_rescale(value, old_max=None, old_min=None, new_max=None, new_min=None):
    # Normalize the input value to [0, 1]
    normalized_value = (value - old_min) / (old_max - old_min)

    # Determine which segment we are in based on the normalized value
    if normalized_value <= 0.5:
        # Scale sine to [0, 1] and then rescale to [0.3, 0.55]
        return 0.3 + 0.25 * np.sin(np.pi * normalized_value)
    else:
        # Scale sine to [0, 1] and then rescale to [0.55, 0.8]
        return 0.55 + 0.25 * np.sin(np.pi * (normalized_value - 0.5))


class VariableRecoveryRate(pcse_gym.envs.sb3.StableBaselinesWrapper):
    def __init__(self, env):
        super().__init__()
        self.__dict__.update(env.__dict__)

    def _apply_action(self, action) -> None:
        amount = action * self.action_multiplier
        recovery_rate = self.recovery_penalty()
        self._model._send_signal(signal=pcse.signals.apply_n, N_amount=amount * 10, N_recovery=recovery_rate,
                                 amount=amount, recovery=recovery_rate)

    def recovery_penalty(self):
        """
        estimation function due to static recovery rate of WOFOST/LINTUL
        Potentially enforcing the agent not to dump everything at the start
        Not to be used with CERES 'start-dump'
        Adapted, based on the findings of Raun, W.R. and Johnson, G.V. (1999)
        """
        date_now = self.date - self.start_date
        date_end = self.end_date - self.start_date  # TODO end on flowering?
        recovery = ratio_rescale(date_now / timedelta(days=1),
                                 old_max=date_end / timedelta(days=1), old_min=0.0, new_max=0.8, new_min=0.3)
        return recovery


def generate_combinations(elements):
    from itertools import combinations
    all_combinations = []

    for x in range(1, len(elements) + 1):
        all_combinations.extend(combinations(elements, x))

    return all_combinations
