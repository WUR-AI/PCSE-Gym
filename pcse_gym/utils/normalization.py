import numpy as np
from copy import deepcopy

import gymnasium as gym

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd


class VecNormalizePO(VecNormalize):
    """
    A wrapper of the normalizing wrapper for an SB3 vectorized environment
    Overrides the normalization of partially observable crop variables in CropGym
    """
    def __init__(self, venv, *args, **kwargs):
        super(VecNormalizePO, self).__init__(venv, *args, **kwargs)
        self.extend = self.venv.envs[0].unwrapped.mask_binary
        self.obs_len = self.venv.envs[0].obs_len
        self.obs_rms = RunningMeanStdPO(epsilon=self.epsilon, shape=self.observation_space.shape,
                                        extend=self.extend, obs_len=self.obs_len)
        self.mask = None

    def _normalize_obs(self, obs, obs_rms):
        ori_obs = deepcopy(obs)
        if self.extend and len(obs) == 1: # and len(obs) == self.obs_len * 2:
            obs_ = obs[0][:int(len(obs[0]) // 2)]
            self.mask = obs[0][int(len(obs[0]) // 2):]
            obs = np.array([obs_])
        if self.venv.envs[0].unwrapped.sb3_env.index_feature and self.venv.envs[0].unwrapped.sb3_env.step_check:
            index_feature = self.venv.envs[0].unwrapped.measure_features.feature_ind
            actions = self.venv.unwrapped.actions
            norm = super()._normalize_obs(obs, obs_rms)
            if len(norm) == 1:
                norm = norm[0]
            for ind, act in zip(index_feature, actions[1:]):
                if act == 0:
                    norm[ind] = 0.0  # temporarily set normalized value as the mean of the rms
            if self.extend: # and len(obs) == self.obs_len * 2:
                np.append(norm, self.mask)
            return norm
        else:
            return super()._normalize_obs(obs, obs_rms)

    def _unnormalize_obs(self, obs, obs_rms):
        # ori_obs = deepcopy(obs)
        if self.extend and len(obs) == 1: # and len(obs) == self.obs_len * 2:
            obs_ = obs[0][:int(len(obs[0]) // 2)]
            self.mask = obs[0][int(len(obs[0]) // 2):]
            obs = np.array([obs_])
        if self.venv.envs[0].unwrapped.sb3_env.index_feature and self.venv.envs[0].unwrapped.sb3_env.step_check:
            index_feature = self.venv.envs[0].unwrapped.sb3_env.index_feature
            actions = self.venv.unwrapped.actions
            unnorm = super()._unnormalize_obs(obs, obs_rms)
            if len(unnorm) == 1:
                unnorm = unnorm[0]
            for ind, act in zip(index_feature.values(), actions[1:]):
                if act == 0:
                    unnorm[ind] = self.venv.envs[0].unwrapped.placeholder_val
            if self.extend: # and len(obs) == self.obs_len * 2:
                np.append(unnorm, self.mask)
            return unnorm
        else:
            return super()._unnormalize_obs(obs, obs_rms)


class RunningMeanStdPO(RunningMeanStd):
    """
    Running mean standard for the ToMeasureOrNot paradigm.
    Here we assume observations that equal ::param : placeholder_value will
    be equal to the current mean of the observation so to not distort the statistics
    """
    def __init__(self, epsilon=1e-4, shape=(), placeholder_value=-1.11, extend=False, obs_len=6):
        super(RunningMeanStdPO, self).__init__(epsilon=epsilon, shape=shape)
        self.extend = extend
        if self.extend:
            self.mean = np.zeros((int(shape[0]/2)), np.float64)
            self.var = np.ones((int(shape[0]/2)), np.float64)
        else:
            self.mean = np.zeros(shape, np.float64)
            self.var = np.ones(shape, np.float64)
        self.count = epsilon
        self.placeholder = placeholder_value

        self.obs_len = obs_len

    def update(self, arr: np.ndarray) -> None:
        if self.extend:  # and len(arr[0]) == self.obs_len * 2:
            arr_ = arr[0][:int(len(arr[0])//2)]
            arr = np.array([arr_])
        # sanity check
        #
        mask = arr[0] == self.placeholder
        # mask = np.invert(mask)
        for i, m in enumerate(mask):
            if m:
                arr[0][i] = self.mean[i]
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)


class ClipNormalizeObservation(gym.wrappers.NormalizeObservation):
    """
    Wrapper to add capability of clipping the observation running std.
    Mimicking SB3 VecNormalize.
    """
    def __init__(self, venv, clip, *args, **kwargs):
        super(ClipNormalizeObservation, self).__init__(venv, *args, **kwargs)
        self.clip = clip

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip, self.clip)


class ClipNormalizeReward(gym.wrappers.NormalizeReward):
    """
    Wrapper to add capability of clipping the observation running std.
    Mimicking SB3 VecNormalize.
    """

    def __init__(self, venv, clip, *args, **kwargs):
        super(ClipNormalizeReward, self).__init__(venv, *args, **kwargs)
        self.clip = clip

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip, self.clip)


class NormalizeMeasureObservations:
    '''
    A class that normalizes in the step() method of PCSE-Gym

    :param crop_features: the crop features chosen from the environment
    :param index_measure: index of the crop features relative to the observations, taken from the measure or not class
    :param loc: tuple or list of tuples of location coordinate
    :param mask_binary: boolean of masked observations
    :param placeholder: unique placeholder for identifying a masked observation
    :param reward_div: WIP, normalize reward by dividing the max reward

    '''
    def __init__(self,
                 crop_features,
                 index_measure,
                 /,
                 has_random=False,
                 no_weather=False,
                 loc='NL',
                 start_type='sowing',
                 mask_binary=False,
                 placeholder=-1.11,
                 reward_div=600,
                 is_clipped=False,
                 decay_rate=0.95):
        self.mask = mask_binary
        self.no_weather = no_weather
        self.crop_features = crop_features
        self.is_clipped = is_clipped
        self.has_random = has_random
        self.loc = loc
        self.start_type = start_type

        if not self.has_random:
            assert self.crop_features == ["DVS", "TAGP", "LAI", "NuptakeTotal", "NAVAIL", "SM"]
        else:
            assert self.crop_features == ["DVS", "TAGP", "LAI", "NuptakeTotal", "NAVAIL", "SM", "random"]

        self.index_measure = index_measure
        self.placeholder = placeholder

        self.fixed_means = np.array(self.get_fixed_vals('means'))
        self.fixed_std = np.array(self.get_fixed_vals('std'))

        self.reward_div = reward_div

        self.run_rew_min = None
        self.run_rew_max = None
        self.decay_rate = decay_rate

    def normalize_measure_obs(self, obs, measure):
        obs_ = deepcopy(obs)
        if self.mask:
            obs_ = obs[:-len(self.index_measure)]
            mask = obs[-len(self.index_measure):]
            norm = self._normalize_obs(obs_)

            # norm = np.array([0.0 if not m else norm_val for m, norm_val in zip(mask, norm[1:6])])
            # norm_ = []
            for m, i in zip(mask, sorted(self.index_measure)):
                if not m:
                    norm[i] = 0.0
                else:
                    pass

            norm = np.append(norm, mask)
        else:
            norm = self._normalize_obs(obs_)
            if measure is not None:
                for i, m in zip(self.index_measure, measure):
                    if not m:
                        norm[i] = 0.0
        if self.is_clipped:
            norm = np.clip(norm, -1, 1)
        return norm

    def _normalize_obs(self, obs):
        if isinstance(obs, list):
            obs = np.array(obs)
        return (obs - self.fixed_means) / (self.fixed_std - 1e-8)

    def unnormalize_measure_obs(self, obs):
        obs_ = deepcopy(obs)
        if self.mask:
            obs_ = obs_[:int(len(obs_) // 2)]
            mask = obs_[int(len(obs_) // 2):]
            norm = self._unnormalize_obs(obs_)
            norm = np.append(norm, mask)
            return norm
        else:
            return self._unnormalize_obs(obs_)

    def _unnormalize_obs(self, obs):
        return obs * (self.fixed_std + 1e-8) + self.fixed_means

    def normalize_rew(self, reward):
        return reward/self.reward_div

    def unnormalize_rew(self, reward):
        return reward*self.reward_div

    def update_running_rew(self, reward):
        # Initialize running min and max if they are None
        if self.run_rew_min is None or self.run_rew_max is None:
            self.run_rew_min = reward
            self.run_rew_max = reward

        self.run_rew_min = self.decay_rate * self.run_rew_min + (1 - self.decay_rate) * min(reward, self.run_rew_min)
        self.run_rew_max = self.decay_rate * self.run_rew_max + (1 - self.decay_rate) * max(reward, self.run_rew_max)

    def normalize_reward(self, reward):

        if self.run_rew_max == self.run_rew_min:
            return 0.0

        normalized = (reward - self.run_rew_min) / (self.run_rew_max - self.run_rew_min)
        return 2 * normalized - 1

    def unnormalize_reward(self, reward):

        reward = (reward + 1) / 2

        return reward * (self.run_rew_max - self.run_rew_min) + self.run_rew_min

    def get_fixed_vals(self, which):
        if which == 'means':
            vals = self.get_means_vector()
        elif which == 'std':
            vals = self.get_std_vector()
        else:
            raise Exception(f'Error! {which} not a valid choice')

        if self.no_weather is True and self.has_random is True:
            return vals[:len(self.crop_features)]
        elif self.no_weather is True and self.has_random is False:
            return vals[:len(self.crop_features)]
        elif self.no_weather is False and self.has_random is True:
            return vals
        elif self.no_weather is False and self.has_random is False:
            ind = len(self.crop_features)
            del vals[ind]
            return vals

    def get_means_vector(self):
        if self.loc == 'NL' and self.start_type == 'emergence':
            return [0.6246465476467645,
                    4865.136421503844,
                    1.234561809311616,
                    150.7396483914316,
                    161.29929450042695,
                    0.27243507521622345,
                    10.0,
                    1.21883597e+07, 4.95082017e+00, 1.90003949e-01,
                    1.22270535e+07, 4.90367558e+00, 1.95953827e-01,
                    1.23226094e+07, 4.91493013e+00, 2.00468408e-01,
                    1.21937849e+07, 5.04890948e+00, 2.12458688e-01,
                    1.24103797e+07, 5.08396719e+00, 2.08391555e-01,
                    1.25560207e+07, 5.19116343e+00, 2.00330194e-01,
                    1.26053463e+07, 5.34218712e+00, 2.03171324e-01]
        elif self.loc == 'NL' and self.start_type == 'sowing':
            return [0.5515796521949504,
                    6067.981932107783,
                    1.671387128093679,
                    150.24972772934655,
                    345.514943005448,
                    0.2502250899218999,
                    10.0,
                    1.03128280e+07, 5.51675820e+00, 2.13808352e-01,
                    1.04076449e+07, 5.52640351e+00, 1.97988368e-01,
                    1.03945862e+07, 5.48108314e+00, 2.09688024e-01,
                    1.03251793e+07, 5.45084668e+00, 2.13623188e-01,
                    1.02945252e+07, 5.47477307e+00, 2.19228642e-01,
                    1.03684859e+07, 5.44384249e+00, 2.23700801e-01,
                    1.04421720e+07, 5.51555111e+00, 2.20108696e-01]
        elif self.loc == 'LT' and self.start_type == 'sowing':
            return [0.33344140142967654,
                    2675.772872654855,
                    1.092753787112243,
                    57.55634087558257,
                    387.7470016444115,
                    0.30791477941701645,
                    10.0,
                    9.75607378e+06, 4.36845453e-01, 2.12145007e-01,
                    9.83234895e+06, 5.56896332e-01, 1.99204579e-01,
                    9.95673097e+06, 5.62800509e-01, 1.91617766e-01,
                    1.00325419e+07, 5.50421878e-01, 2.10531906e-01,
                    1.00911787e+07, 5.02440110e-01, 1.96787789e-01,
                    9.92729065e+06, 5.40256519e-01, 1.95853933e-01,
                    9.93554166e+06, 5.95882976e-01, 2.01239347e-01]
        elif self.loc == 'LT' and self.start_type == 'emergence':
            raise NotImplementedError
        else:
            raise Exception(f'Error! "{self.loc} and {self.start_type}" not a valid choice!')


    def get_std_vector(self):
        if self.loc == 'NL' and self.start_type == 'emergence':
            return [0.5728143239654657,
                    6177.7736977069735,
                    1.5758559066832276,
                    177.0009703071452,
                    91.93445733139745,
                    0.0341784277318637,
                    10.0,
                    1.21883597e+07, 4.95082017e+00, 1.90003949e-01,
                    1.22270535e+07, 4.90367558e+00, 1.95953827e-01,
                    1.23226094e+07, 4.91493013e+00, 2.00468408e-01,
                    1.21937849e+07, 5.04890948e+00, 2.12458688e-01,
                    1.24103797e+07, 5.08396719e+00, 2.08391555e-01,
                    1.25560207e+07, 5.19116343e+00, 2.00330194e-01,
                    1.26053463e+07, 5.34218712e+00, 2.03171324e-01]
        elif self.loc == 'NL' and self.start_type == 'sowing':
            return [0.7026304587741417,
                    8812.086379020251,
                    2.438678469891358,
                    198.4757260048262,
                    166.9682858183378,
                    0.039759586302687465,
                    10.0,
                    1.03128280e+07, 5.51675820e+00, 2.13808352e-01,
                    1.04076449e+07, 5.52640351e+00, 1.97988368e-01,
                    1.03945862e+07, 5.48108314e+00, 2.09688024e-01,
                    1.03251793e+07, 5.45084668e+00, 2.13623188e-01,
                    1.02945252e+07, 5.47477307e+00, 2.19228642e-01,
                    1.03684859e+07, 5.44384249e+00, 2.23700801e-01,
                    1.04421720e+07, 5.51555111e+00, 2.20108696e-01]
        elif self.loc == 'LT' and self.start_type == 'sowing':
            return [0.5577426661192912,
                    5390.806038545118,
                    1.8315136788380577,
                    109.21366379879503,
                    204.46094641439367,
                    0.021277502711722556,
                    10.0,
                    8.33064223e+06, 8.43375544e+00, 3.59799387e-01,
                    8.35175786e+06, 8.38202910e+00, 3.10559526e-01,
                    8.55516035e+06, 8.23049704e+00, 3.17290068e-01,
                    8.52174349e+06, 8.30084969e+00, 3.56926711e-01,
                    8.51462773e+06, 8.36266632e+00, 3.22802387e-01,
                    8.31931600e+06, 8.54765225e+00, 3.30367139e-01,
                    8.36117416e+06, 8.61383742e+00, 3.40634577e-01]
        elif self.loc == 'LT' and self.start_type == 'emergence':
            raise NotImplementedError
        else:
            raise Exception(f'Error! "{self.loc} and {self.start_type}" not a valid choice!')

class RunningReward:
    '''
    A class that stores the running mean statistics of the reward normalization
    '''
    def __init__(self):
        self.mean = 0
        self.n = 0
        self.var = 1
        self.epsilon = 1e-8

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta_after = x - self.mean
        self.var += delta * delta_after

    def variance(self, sample_population=True):
        if self.n < 2:
            return 1
        if sample_population:
            return self.var / (self.n - 1)
        return self.var / self.n

    def stddev(self, sample=True):
        return np.sqrt(self.variance(sample_population=sample))

    def normalize(self, reward):
        return (reward - self.mean) / (self.stddev() + self.epsilon)

    def unnormalize(self, reward):
        return reward * (self.stddev() + self.mean)


class MinMaxReward:
    def __init__(self):
        self.reward_min = float('inf')
        self.reward_max = float('-inf')

    def update_min_max(self, reward):
        self.reward_min = min(self.reward_min, reward)
        self.reward_max = max(self.reward_max, reward)

    def normalize(self, reward):
        if self.reward_max == self.reward_min:
            return reward
        return (reward - self.reward_min) / (self.reward_max - self.reward_min)

    def unnormalize(self, reward):
        return reward * (self.reward_max - self.reward_min) + self.reward_min


class WOFOSTMinMaxObservations:
    def __init__(self,
                 crop_features,
                 index_measure,
                 /,
                 has_random=False,
                 no_weather=False,
                 loc='NL',
                 start_type='sowing',
                 mask_binary=False,
                 placeholder=-1.11,
                 reward_div=600,
                 is_clipped=False,
                 decay_rate=0.95):
        self.mask = mask_binary
        self.no_weather = no_weather
        self.crop_features = crop_features
        self.is_clipped = is_clipped
        self.has_random = has_random
        self.loc = loc
        self.start_type = start_type

