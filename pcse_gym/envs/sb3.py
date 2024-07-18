import os
from collections import OrderedDict, defaultdict
from datetime import timedelta, date
import gymnasium as gym
import pandas as pd
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import pcse
import numpy as np
import yaml
from pathlib import Path

import pcse_gym.envs.common_env as common_env
import pcse_gym.utils.defaults as defaults
import pcse_gym.utils.process_pcse_output as process_pcse
from .rewards import Rewards
from pcse_gym.utils.nitrogen_helpers import get_aggregated_n_depo_days, m2_to_ha


def to_weather_info(days, weather_data, weather_variables):
    weather_observation = []
    for i, d in enumerate(days):
        def def_value():
            return 0

        w = defaultdict(def_value)
        w['day'] = d
        for var in weather_variables:
            w[var] = getattr(weather_data[i], var)
        weather_observation.append(w)
    weather_info = pd.DataFrame(weather_observation).set_index("day").fillna(value=np.nan)
    return weather_info


def update_info(inf, key, date, value):
    if key not in inf.keys():
        inf[key] = {}
    inf[key][date] = value
    return inf


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Processes input features: average pool timeseries (weather) and concat with scalars (crop features)
    """

    def __init__(self, observation_space: gym.spaces.Box, n_timeseries, n_scalars, n_actions=0, n_timesteps=7, n_po_features=5, mask_binary=False):
        self.n_timeseries = n_timeseries
        self.n_scalars = n_scalars
        self.n_actions = n_actions
        self.n_timesteps = n_timesteps
        self.mask_binary = mask_binary
        self.n_po_features = n_po_features
        if self.mask_binary:
            shape = (n_timeseries + n_scalars + n_po_features,)
            features_dim = n_timeseries + n_scalars + n_po_features
        else:
            shape = (n_timeseries + n_scalars + n_actions,)
            features_dim = n_timeseries + n_scalars + n_actions
        super(CustomFeatureExtractor, self).__init__(gym.spaces.Box(-10, np.inf, shape=shape),
                                                     features_dim=features_dim)

        self.avg_timeseries = nn.Sequential(
            nn.AvgPool1d(kernel_size=self.n_timesteps)
        )

    def forward(self, observations) -> th.Tensor:
        # Returns a torch tensor in a format compatible with Stable Baselines3
        batch_size = observations.shape[0]
        scalars, timeseries = observations[:, 0:self.n_scalars+self.n_actions], \
                              observations[:, self.n_scalars+self.n_actions:]
        mask = None
        if self.mask_binary:
            mask = timeseries[:, -self.n_po_features:]
            timeseries = timeseries[:, :-self.n_po_features]
        reshaped = timeseries.reshape(batch_size, self.n_timesteps, self.n_timeseries).permute(0, 2, 1)
        x1 = self.avg_timeseries(reshaped)
        x1 = th.squeeze(x1, 2)
        if self.mask_binary:
            x = th.cat((scalars, x1, mask), dim=1)
        else:
            x = th.cat((x1, scalars), dim=1)
        return x


def get_policy_kwargs(n_crop_features=len(defaults.get_wofost_default_crop_features(2)),
                      n_weather_features=len(defaults.get_default_weather_features()),
                      n_action_features=len(defaults.get_default_action_features()),
                      n_po_features=len(defaults.get_wofost_default_po_features()),
                      mask_binary=False,
                      n_timesteps=7):
    # Integration with BaseModel from Stable Baselines3
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(n_timeseries=n_weather_features,
                                       n_scalars=n_crop_features,
                                       n_actions=n_action_features,
                                       n_timesteps=n_timesteps,
                                       n_po_features=n_po_features,
                                       mask_binary=mask_binary),
    )
    return policy_kwargs


def get_config_dir():
    from pathlib import Path
    config_dir = os.path.join(Path(os.path.realpath(__file__)).parents[1], 'envs', 'configs')
    return config_dir


def get_wofost_kwargs(config_dir=get_config_dir(), soil_file='arminda_soil.yaml', agro_file='wheat_cropcalendar.yaml',
                      model_file='Wofost81_NWLP_MLWB_SNOMIN.conf', pcse_model=2):
    if pcse_model == 2:
        soil_params = yaml.safe_load(open(os.path.join(config_dir, 'soil', 'arminda_soil.yaml')))
        site_params = yaml.safe_load(open(os.path.join(config_dir, 'site', 'arminda_site.yaml')))
    else:
        soil_params = pcse.input.CABOFileReader(os.path.join(config_dir, 'soil', 'ec3.CAB'))
        site_params = pcse.util.WOFOST80SiteDataProvider(WAV=10, NAVAILI=10, PAVAILI=50, KAVAILI=100)
    wofost_kwargs = dict(
        model_config=os.path.join(config_dir, model_file),
        agro_config=os.path.join(config_dir, 'agro', agro_file),
        crop_parameters=pcse.input.YAMLCropDataProvider(fpath=os.path.join(config_dir, 'crop'), force_reload=True),
        site_parameters=site_params,
        soil_parameters=soil_params,
    )
    return wofost_kwargs


def get_lintul_kwargs(config_dir=get_config_dir()):
    lintul_kwargs = dict(
        model_config=os.path.join(config_dir, 'Lintul3.conf'),
        agro_config=os.path.join(config_dir, 'agro', 'agromanagement_fertilization.yaml'),
        crop_parameters=os.path.join(config_dir, 'crop', 'lintul3_winterwheat.crop'),
        site_parameters=os.path.join(config_dir, 'site', 'lintul3_springwheat.site'),
        soil_parameters=os.path.join(config_dir, 'soil', 'lintul3_springwheat.soil'),
    )
    return lintul_kwargs


def get_model_kwargs(pcse_model, loc=defaults.get_default_location(), start_type='sowing'):
    if not isinstance(loc, list):
        loc = [loc]

    if pcse_model == 0:
        return get_lintul_kwargs()
    elif pcse_model == 1:
        agro_file = 'wheat_cropcalendar_cn.yaml'
        soil_file = 'ec_3.CAB'
        model_file = 'Wofost81_NWLP_CWB_CNB.conf'
        print(f'using agro file {agro_file} and soil file {soil_file} with WOFOST CN')
        return get_wofost_kwargs(soil_file=soil_file, agro_file=agro_file, model_file=model_file, pcse_model=pcse_model)
    elif pcse_model == 2:
        model_file = 'Wofost81_NWLP_MLWB_SNOMIN.conf'
        soil_file = 'arminda_soil.yaml'
        agro_file = 'wheat_cropcalendar.yaml'
        print(f'using agro file {agro_file} and soil file {soil_file} with WOFOST SNOMIN')
        return get_wofost_kwargs(soil_file=soil_file, agro_file=agro_file, model_file=model_file, pcse_model=pcse_model)
    else:
        raise Exception("Choose 0 or 1, or 2 for the environment")


class StableBaselinesWrapper(common_env.PCSEEnv):
    """
    Establishes compatibility with Stable Baselines3

    :param action_multiplier: conversion factor to map output node to g/m2 of nitrogen
        action_space=gym.spaces.Discrete(3), action_multiplier=2.0 gives {0, 2.0, 4.0}
        action_space=gym.spaces.Box(0, np.inf, shape=(1,), action_multiplier=1.0 gives 1.0*x
    """

    def __init__(self, crop_features=defaults.get_wofost_default_crop_features(2),
                 weather_features=defaults.get_default_weather_features(),
                 action_features=defaults.get_default_action_features(), costs_nitrogen=10.0, timestep=7,
                 years=None, location=None, seed=0, action_space=gym.spaces.Box(0, np.inf, shape=(1,)),
                 action_multiplier=1.0, *args, **kwargs):
        self.costs_nitrogen = costs_nitrogen
        self.crop_features = crop_features
        self.weather_features = weather_features
        self.action_features = action_features
        self.step_check = False
        self.no_weather = kwargs.get('no_weather', False)
        self.mask_binary = kwargs.get('mask_binary', False)
        self.po_features = kwargs.get('po_features', [])
        self.random_feature = False
        if 'random' in self.po_features:
            self.random_feature = True
        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)
        if 'Lintul' in kwargs.get('model_config'):
            self.pcse_env = 0
        elif 'CNB' in kwargs.get('model_config'):
            self.pcse_env = 1
        elif 'SNOMIN' in kwargs.get('model_config'):
            self.pcse_env = 2
        super().__init__(timestep=timestep, years=years, location=location, *args, **kwargs)
        self.action_space = action_space
        self.action_multiplier = action_multiplier
        self.args_vrr = kwargs.get('args_vrr', None)
        self.rewards = Rewards(kwargs.get('reward_var'), self.timestep, self.costs_nitrogen)
        self.index_feature = OrderedDict()
        self.cost_measure = kwargs.get('cost_measure', 'real')
        self.start_type = kwargs.get('start_type')

        for i, feature in enumerate(self.crop_features):
            if feature in self.po_features:
                self.index_feature[feature] = i

        cgm_kwargs = kwargs.get('model_config', '')
        if 'lintul' in cgm_kwargs:
            self.multiplier_amount = 1
            print('Using Lintul!')
        elif 'Wofost' in cgm_kwargs:
            self.multiplier_amount = 1
            print('Using Wofost!')
        else:
            self.multiplier_amount = 1

        super().reset(seed=seed)

    def _get_observation_space(self):
        if self.no_weather:
            nvars = len(self.crop_features)
        else:
            nvars = len(self.crop_features) + len(self.action_features) + len(self.weather_features) * self.timestep
        if self.mask_binary:
            nvars = nvars + len(self.po_features)
        return gym.spaces.Box(-np.inf, np.inf, shape=(nvars,))

    def _apply_action(self, action):
        # action = action * self.action_multiplier
        action = action * 10  # kg N / ha
        return action

    def _get_reward(self):
        # Reward gets overwritten in step()
        return 0

    def step(self, action):
        """
        Computes customized reward and populates info
        """
        self.step_check = True
        measure = None
        if isinstance(action, np.ndarray):
            action, measure = action[0], action[1:]

        obs, _, terminated, truncated, _ = super().step(action)

        # populate observation
        observation = self._observation(obs)

        # populate reward
        pcse_output = self.model.get_output()
        amount = action * self.action_multiplier
        reward, growth = self.rewards.growth_storage_organ(pcse_output, amount, self.multiplier_amount)

        # populate info
        crop_info = pd.DataFrame(pcse_output).set_index("day").fillna(value=np.nan)
        days = [day['day'] for day in pcse_output]
        weather_data = [self._weather_data_provider(day) for day in days]
        weather_info = to_weather_info(days, weather_data, self._weather_variables)
        info = {**pd.concat([crop_info, weather_info], axis=1, join="inner").to_dict()}

        start_date = process_pcse.get_start_date(pcse_output, self.timestep)
        # start_date is beginning of the week
        # self.date is the end of the week (if timestep=7)
        info = update_info(info, 'action', start_date, action)
        info = update_info(info, 'fertilizer', start_date, amount*10)
        info = update_info(info, 'reward', self.date, reward)
        if measure is not None:
            info = update_info(info, 'measure', start_date, measure)
        if self.random_feature:
            info = update_info(info, 'random', self.date, observation[len(self.crop_features)-1])

        if self.index_feature:
            if 'indexes' not in info.keys():
                info['indexes'] = OrderedDict()
            info['indexes'] = self.index_feature

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, return_info=False, options=None):
        self.step_check = False
        obs = super().reset(seed=seed, options=options)
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._observation(obs)

    def _observation(self, observation):
        """
        Converts observation into np array to facilitate integration with Stable Baseline3
        """
        obs = np.zeros(self.observation_space.shape)

        if isinstance(observation, tuple):
            observation = observation[0]

        for i, feature in enumerate(self.crop_features):
            if feature == 'random':
                obs[i] = np.clip(self.rng.normal(10, 10), 0.0, None)
            else:
                # In WOFOST SNOMIN some variables are layered.
                # Loop through some checks to grab correct obs
                if feature in ['SM', 'NH4', 'NO3', 'WC']:
                    if feature in ['NH4', 'NO3']:
                        obs[i] = sum(observation['crop_model'][feature][-1]) / m2_to_ha
                    elif feature in ['SM', 'WC'] and self.pcse_env == 1:
                        obs[i] = observation['crop_model'][feature][-1]
                    else:
                        obs[i] = np.mean(observation['crop_model'][feature][-1])
                elif feature in ['RNO3DEPOSTT', 'RNH4DEPOSTT']:
                    obs[i] = observation['crop_model'][feature][-1] / m2_to_ha
                else:
                    obs[i] = observation['crop_model'][feature][-1]

        for i, feature in enumerate(self.action_features):
            j = len(self.crop_features) + i
            obs[j] = sum(observation['action_features'][feature])

        if not self.no_weather:
            for d in range(self.timestep):
                for i, feature in enumerate(self.weather_features):
                    j = d * len(self.weather_features) + len(self.crop_features) + len(self.action_features) + i
                    obs[j] = observation['weather'][feature][d]
        return obs

    def get_harvest_year(self):
        if self.agmt.campaign_date.year < self.agmt.crop_end_date.year:
            if date(self.date.year, 10, 1) < self.date < date(self.date.year, 12, 31):
                return self.date.year + 1
            else:
                return self.date.year
        return self.date.year

    @property
    def model(self):
        return self._model

    @property
    def date(self):
        return self.model.day

    @property
    def loc(self):
        return self._location

    @loc.setter
    def loc(self, location):
        self._location = location

    @property
    def timestep(self):
        return self._timestep

    @property
    def agro_management(self):
        return self._agro_management

    @agro_management.setter
    def agro_management(self, agro):
        self._agro_management = agro

    @property
    def weather_data_provider(self):
        return self._weather_data_provider

    @weather_data_provider.setter
    def weather_data_provider(self, weather):
        self._weather_data_provider = weather


class ZeroNitrogenEnvStorage:
    """
    Container to store results from zero nitrogen policy (for re-use)
    """

    def __init__(self):
        self.results = {}

    def run_episode(self, env):
        env.reset()
        terminated, truncated = False, False
        infos_this_episode = []
        while not terminated or truncated:
            _, _, terminated, truncated, info = env.step(0)
            infos_this_episode.append(info)
        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        return episode_info

    def get_key(self, env):
        ''' We label the year based on the harvest date. e.g. sow in Oct 2002, harvest in Aug 2003, means that
            the labelled year is 2003'''
        # year = env.date.year
        year = env.get_harvest_year()
        location = env.loc
        key = f'{year}-{location}'
        assert 'None' not in key
        return key

    def get_episode_output(self, env):
        key = self.get_key(env)
        if key not in self.results.keys():
            results = self.run_episode(env)
            self.results[key] = results
        assert bool(self.results[key]), "key empty; check PCSE output"
        return self.results[key]

    @property
    def get_result(self):
        return self.results
