import os
import datetime
import pandas as pd
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from scipy.optimize import minimize_scalar
from bisect import bisect_left
from typing import Union
from statistics import mean, median
from tqdm import tqdm
import pickle
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib import RecurrentPPO
import pcse_gym.utils.defaults as defaults
from pcse_gym.utils.process_pcse_output import get_dict_lintul_wofost
from pcse_gym.envs.rewards import calculate_nue
from .plotter import plot_variable, plot_var_vs_freq_scatter, get_ylim_dict


def compute_median(results_dict: dict, filter_list=None):
    if filter_list is None:
        filter_list = list(results_dict.keys())
    filtered_results = [results_dict[f] for f in filter_list if f in results_dict.keys()]
    return np.median(filtered_results)


def get_cumulative_variables():
    return ['fertilizer', 'reward']


def identity_line(ax=None, ls='--', *args, **kwargs):
    # see: https://stackoverflow.com/q/22104256/3986320
    ax = ax or plt.gca()
    identity, = ax.plot([], [], ls=ls, *args, **kwargs)

    def callback(axes):
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        low = min(low_x, low_y)
        high = max(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(ax)
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)
    return ax


def convert_variables(results_storage):
    for lintul, wofost, factor in get_dict_lintul_wofost():
        if lintul in results_storage and wofost not in results_storage:
            results_storage[wofost] = {x: y * factor for x, y in results_storage[lintul].items()}
        if wofost in results_storage and lintul not in results_storage:
            results_storage[lintul] = {x: y / factor for x, y in results_storage[wofost].items()}

    if "RNuptake" in results_storage.keys():
        k = list(results_storage["RNuptake"].keys())
        v = 0.1 * np.cumsum(list(results_storage["RNuptake"].values()))
        results_storage["NUPTT"] = dict(zip(k, v))

    return results_storage


def report_ci(boot_metric, report_p=False):
    ci_lower = np.quantile(boot_metric, 0.025)
    ci_upper = np.quantile(boot_metric, 0.975)
    return_string = f'(95% CI={ci_lower:0.2f} {ci_upper:0.2f})'
    if (report_p):
        boot_metric_sorted = np.sort(boot_metric)
        n_boot = len(boot_metric)
        idx = bisect_left(boot_metric_sorted, 0.0, hi=n_boot - 1)
        return_string = return_string + f' one-sided-p={(idx / n_boot):0.4f}'
    return return_string


def summarize_results(results_dict):
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    all_variables = list(list(results_dict.values())[0][0].keys())
    weather_variables = intersection(['TMIN', 'TMAX', 'IRRAD', 'RAIN'], all_variables)
    variables_average = weather_variables
    variables_cum = intersection(['DVS', 'fertilizer', 'TGROWTHr', 'TRANRF', 'WLL', 'reward'], all_variables)
    variables_end = intersection(['WSO'], all_variables)
    variables_max = intersection(['val'], all_variables)

    save_data = {}
    for k, result in results_dict.items():
        ndays = len(result[0]['IRRAD'].values())
        nfertilizer = sum(map(lambda x: x != 0, list(result[0]['fertilizer'].values())))
        a = [(sum(result[0][variable].values()) / ndays) for variable in variables_average]
        c = [(sum(result[0][variable].values())) for variable in variables_cum]
        d = [(list(result[0][variable].values())[-1]) for variable in variables_end]
        m = [max(list(result[0][variable].values())) for variable in variables_max]
        year, location = k
        location = ';'.join([str(loc) for loc in location])
        save_data[k] = a + c + d + m + [nfertilizer, year, ndays, location]
    df = pd.DataFrame.from_dict(save_data, orient='index',
                                columns=variables_average + variables_cum + variables_end + variables_max +
                                        ['nevents', 'year', 'ndays', 'location'])
    return df


def save_results(results_dict, results_path):
    df = summarize_results(results_dict)
    df.to_csv(results_path, index=False)


def compute_average(results_dict: dict, filter_list=None):
    if filter_list is None:
        filter_list = list(results_dict.keys())
    filtered_results = [results_dict[f] for f in filter_list if f in results_dict.keys()]
    if len(filtered_results) == 0:
        return 0
    return sum(filtered_results) / len(filtered_results)


def get_action_probs(dis: MultiCategoricalDistribution, po_features, crop_features, measure_all):
    if po_features:
        dict = {}
        dict['prob_action'] = dis.distribution[0].probs.detach().numpy()[0]
        if measure_all:
            dict['prob_measure'] = dis.distribution[1].probs.detach().numpy()[0][1]
        else:
            for i, feature in enumerate(crop_features):
                if feature in po_features:
                    feature = "prob_" + feature
                    dict[feature] = dis.distribution[i].probs.detach().numpy()[0][1]
        return dict
    else:
        return None


def evaluate_policy(
        policy,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        amount=1,
):
    """
    Runs policy for ``n_eval_episodes`` episodes.
    This is made to work only with one env.

    :param policy: The (RL) agent you want to evaluate.
        Implemented options:
            (a) RL agent (base_class.BaseAlgorithm)
            (b) Standard Practice ('standard-practice' / 'standard-practise')
            (c) Zero Nitrogen ('no-nitrogen')
            (d) Start Dump ('start-dump')

    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param amount: Multiplier for action
    :return: a list of episode_rewards, and episode_infos
    """
    training = True

    if isinstance(policy, base_class.BaseAlgorithm) and policy.get_env() is not None:
        if isinstance(env, DummyVecEnv) and not env.envs[0].unwrapped.normalize:
            training = policy.get_env().training
            policy.get_env().training = False
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    action_space = env.action_space
    if isinstance(action_space, list):
        action_space = action_space[0]

    if isinstance(policy, base_class.BaseAlgorithm):
        assert (policy.action_space == action_space)

    if isinstance(action_space, gym.spaces.Discrete) and not isinstance(policy, base_class.BaseAlgorithm):
        print('Warning!')

    episode_rewards, episode_infos = [], []
    for i in range(n_eval_episodes):
        if isinstance(policy, base_class.BaseAlgorithm):
            if isinstance(env, DummyVecEnv) and not env.envs[0].unwrapped.normalize:
                sync_envs_normalization(policy.get_env(), env)
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        terminated, truncated, state = False, False, None
        episode_reward = 0.0
        episode_length = 0
        year = env.get_attr("date")[0].year
        fert_dates = [datetime.date(year, 2, 24), datetime.date(year, 3, 26), datetime.date(year, 4, 29)]
        action = [amount * 0]
        infos_this_episode = []
        prob, val = None, None

        lstm_state = None
        episode_starts = np.ones((1,), dtype=bool)
        action_probs = None

        while not terminated or truncated:
            if policy == 'start-dump' and (episode_length == 0):
                action = [amount * 1]
            if isinstance(policy, base_class.BaseAlgorithm):
                device = policy.device.type
                if isinstance(policy, PPO):
                    action, state = policy.predict(obs, state=state, deterministic=deterministic)
                    if 'cuda' in device:
                        sb_actions, sb_values, sb_log_probs = policy.policy(torch.from_numpy(obs).to(device),
                                                                            deterministic=deterministic)
                        sb_prob = np.exp(sb_log_probs.detach().cpu().numpy()).item()
                        sb_val = sb_values.detach().cpu().item()
                    else:
                        sb_actions, sb_values, sb_log_probs = policy.policy(torch.from_numpy(obs),
                                                                            deterministic=deterministic)
                        sb_prob = np.exp(sb_log_probs.detach().numpy()).item()
                        sb_val = sb_values.detach().item()
                    prob = sb_prob
                    val = sb_val
                if isinstance(policy, A2C):
                    action, state = policy.predict(obs, state=state, episode_start=episode_starts,
                                                   deterministic=deterministic)

                if isinstance(policy, RecurrentPPO):
                    action, lstm_state = policy.predict(obs, state=lstm_state, episode_start=episode_starts,
                                                        deterministic=deterministic)

                    if 'cuda' in device:
                        lstm_torch = (torch.from_numpy(lstm_state[0]).to(device),
                                      torch.from_numpy(lstm_state[1]).to(device))

                        dis, _ = policy.policy.get_distribution(torch.from_numpy(obs).to(device),
                                                                lstm_states=lstm_torch,
                                                                episode_starts=torch.from_numpy(episode_starts).to(device))
                        val = policy.policy.predict_values(torch.from_numpy(obs).to(device),
                                                           lstm_states=lstm_torch,
                                                           episode_starts=torch.from_numpy(episode_starts).to(device))
                        val = val.detach().cpu().numpy()[0][0]
                    else:
                        lstm_torch = (torch.from_numpy(lstm_state[0]),
                                      torch.from_numpy(lstm_state[1]))

                        dis, _ = policy.policy.get_distribution(torch.from_numpy(obs),
                                                                lstm_states=lstm_torch,
                                                                episode_starts=torch.from_numpy(episode_starts))
                        val = policy.policy.predict_values(torch.from_numpy(obs),
                                                           lstm_states=lstm_torch,
                                                           episode_starts=torch.from_numpy(episode_starts))
                        val = val.detach().numpy()[0][0]

                    action_probs = get_action_probs(dis, env.envs[0].unwrapped.po_features,
                                                    env.envs[0].unwrapped.crop_features,
                                                    env.envs[0].unwrapped.measure_all)


                if isinstance(policy, DQN):
                    action = policy.predict(obs, deterministic=deterministic)

            # SB3 VecEnvs don't follow the gymnasium step API, this is a quick fix.
            # see: https://github.com/DLR-RM/stable-baselines3/blob/master/docs/guide/vec_envs.rst
            # TODO: add check on function signature
            obs, rew, terminated, info = env.step(action)
            episode_starts = terminated
            truncated = info[0].pop("TimeLimit.truncated")

            if isinstance(env, VecNormalize):
                reward = env.get_original_reward()
            if isinstance(env, DummyVecEnv) and not env.envs[0].unwrapped.normalize:
                reward = env.get_original_reward()
            elif env.envs[0].unwrapped.normalize:
                # reward = env.envs[0].unwrapped.norm_rew.unnormalize(rew)
                # reward = rew * np.sqrt(env.envs[0].unwrapped.norm_rew.var + 1e-8)
                # reward = env.envs[0].unwrapped.norm.unnormalize_rew(rew)
                reward = env.envs[0].unwrapped.norm.unnormalize_reward(rew)

            if prob:
                action_date = list(info[0]['action'].keys())[0]
                info[0]['prob'] = {action_date: prob}
                info[0]['dvs'] = {action_date: info[0]['DVS'][action_date]}
            if val:
                val_date = list(info[0]['action'].keys())[0]
                info[0]['val'] = {val_date: val}
            if action_probs:
                action_date = list(info[0]['action'].keys())[0]
                for key, val in action_probs.items():
                    info[0][key] = {action_date: val}

            action = [amount * 0]
            if policy in ['standard-practice', 'standard-practise']:
                date = env.get_attr("date")[0]
                for fert_date in fert_dates:
                    if date > fert_date and date <= fert_date + datetime.timedelta(7):
                        action = [amount * 3]
            if policy == 'no-nitrogen':
                action = [0]
            episode_reward += reward
            episode_length += 1
            infos_this_episode.append(info[0])
        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        episode_rewards.append(episode_reward)
        episode_infos.append(episode_info)
    if isinstance(policy, base_class.BaseAlgorithm) and policy.get_env() is not None:
        policy.get_env().training = training
    return episode_rewards, episode_infos


class FindOptimum():
    """
    Run optimizer to find action that maximizes return value
    Implemented example: Find optimal amount of nitrogen to "dump" at the start of the season
    Maximizes the sum of rewards over the (train) year(s)
    """

    def __init__(self, env, train_years=None):
        self.train_years = train_years
        self.env = env
        if train_years is None:
            self.train_years = [env.get_attr("date")[0].year]

    def start_dump(self, x):
        def def_value():
            return 0

        self.current_rewards = defaultdict(def_value)
        for train_year in self.train_years:
            self.env.env_method('overwrite_year', train_year)
            self.env.reset()
            terminated = False
            infos_this_episode = []
            total_reward = 0.0
            while not terminated:
                action = [0.0]
                if len(infos_this_episode) == 0:
                    action = [x * 1.0]
                info_this_episode, rew, terminated, _ = self.env.step(action)
                reward = self.env.get_original_reward()
                total_reward = total_reward + reward
                infos_this_episode.append(info_this_episode)
            self.current_rewards[self.env.get_attr("date")[0].year] = total_reward
        returnvalue = 0
        # We use minimize_scalar(); invert reward
        for year, reward in self.current_rewards.items():
            returnvalue = returnvalue - reward
        return returnvalue

    def optimize_start_dump(self, bounds=(0, 100.0)):
        res = minimize_scalar(self.start_dump, bounds=bounds, method='bounded')
        print(f'optimum found for {self.train_years} at {res.x} {-1.0 * res.fun}')
        for year, reward in self.current_rewards.items():
            print(f'- {year} {reward}')
        return res.x


def get_measure_graphs(episode_infos):
    measure_graph = {}
    feature_order = episode_infos[0]['indexes'].keys()
    for date, measurement in episode_infos[0]['measure'].items():
        for feature, measure in zip(feature_order, measurement):
            feature = 'measure_' + feature
            if feature not in measure_graph.keys():
                measure_graph[feature] = {}
            if date not in measure_graph[feature].keys():
                measure_graph[feature][date] = measure
    episode_infos[0] = episode_infos[0] | measure_graph  # Python 3.9.0
    return episode_infos


def get_measure_graph(episode_infos):
    measure_graph = {}
    for date, measurement in episode_infos[0]['measure'].items():
        measurement = measurement[0]
        measure_graph[date] = measurement
    episode_infos[0]['measure'] = measure_graph
    return episode_infos


class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent. Writes the following to tensorboard:
        - Scalars:
            - 'cumulative action', 'WSO', 'cumulative reward': per year, and summarized over years (average and median)
        - Figures:
            - Progress of crop/weather/reward etc. during season
            - Histogram of years and locations used during training
    Currently reporting is quite detailed and therefore time-consuming

    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """

    def __init__(self, env_eval=None, train_years=defaults.get_default_train_years(),
                 test_years=defaults.get_default_test_years(),
                 train_locations=defaults.get_default_location(), test_locations=defaults.get_default_location(),
                 n_eval_episodes=1, eval_freq=20_000, pcse_model=1, seed=0, comet_experiment=None, multiprocess=False,
                 **kwargs):
        super(EvalCallback, self).__init__()
        self.test_years = test_years
        self.train_years = train_years
        self.train_locations = [train_locations] if isinstance(train_locations, tuple) else train_locations
        self.test_locations = [test_locations] if isinstance(test_locations, tuple) else test_locations
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.pcse_model = pcse_model
        self.seed = seed
        self.env_eval = env_eval
        self.comet_experiment = comet_experiment
        self.po_features = kwargs.get('po_features')
        self.random_weather = kwargs.get('random_weather', False)
        self.multiprocess = multiprocess
        self.n_envs = kwargs.get('n_envs')

        def def_value(): return 0

        self.histogram_training_years = defaultdict(def_value)

        def def_value(): return 0

        self.histogram_training_locations = defaultdict(def_value)

    @staticmethod
    def check_year_combination(year, loc):
        if year < 1985 and loc in [(52.5, 5.5), (52.0, 5.5), (51.5, 5.5)]:
            return False
        return True

    def get_locations(self, log_training=False):
        if log_training:
            locations = list(set(self.test_locations + self.train_locations))
        else:
            locations = list(set(self.test_locations))
        return locations

    def get_years(self, log_training=False):
        if log_training and not self.random_weather:
            years = list(set(self.test_years + self.train_years))
        elif log_training and self.random_weather:
            years = list(set(self.test_years + list(np.random.choice(self.train_years, 32))))
        else:
            years = list(set(self.test_years))
        return years

    def get_do_log_training(self):
        log_training = False
        if self.n_calls % (5 * self.eval_freq) == 0 or self.n_calls == 1:
            log_training = True
        return log_training

    def replace_measure_variable(self, variables, cumulative=None):
        variables.remove('measure')
        for variable in self.env_eval.po_features:
            variable = 'measure_' + variable
            variables += [variable]
            if cumulative:
                cumulative += [variable]
        return (variables, cumulative) if cumulative else variables

    def get_nue(self, episode_infos):
        n_so = list(episode_infos[0]['NamountSO'].values())[-1]
        n_in = np.cumsum(list(episode_infos[0]['fertilizer'].values()))[-1]
        n_year = list(episode_infos[0]['NamountSO'].keys())[-1].year
        return calculate_nue(n_input=n_in, n_so=n_so, year=n_year)

    def _on_step(self):
        train_year = self.model.get_env().get_attr("date")[0].year
        self.histogram_training_years[train_year] = self.histogram_training_years[train_year] + 1
        train_location = self.model.get_env().get_attr("loc")[0]
        self.histogram_training_locations[train_location] = self.histogram_training_locations[train_location] + 1

        '''Evaluate episodes with learned policy and log it in tensorboard'''
        if self.n_calls % self.eval_freq == 0 or self.n_calls == 1:
            if len(set(list(self.histogram_training_years.keys())).symmetric_difference(
                    set(self.train_years))) != 0:
                print(f'{self.n_calls} {list(self.histogram_training_years.keys())} {self.train_years}')
            else:
                print(f'[{self.n_calls}]')
            tensorboard_logdir = self.logger.dir
            model_path = os.path.join(tensorboard_logdir, f'model-{self.n_calls}')
            self.model.save(model_path)
            if not self.env_eval.envs[0].unwrapped.normalize:
                stats_path = os.path.join(tensorboard_logdir, f'env-{self.n_calls}.pkl')
                self.model.get_env().save(stats_path)

            # evaluate model and get rewards and infos
            episode_rewards, episode_infos = evaluate_policy(policy=self.model, env=self.env_eval)

            if self.pcse_model == 2:
                variables = ['action', 'WSO', 'reward',
                             'NLOSSCUM']
                if self.po_features: variables.append('measure')
                cumulative = ['action', 'reward']
            else:
                variables = ['action', 'WSO', 'reward']
                if self.po_features: variables.append('measure')
                cumulative = ['action', 'reward']

            '''logic for measure graph'''
            if 'measure' in variables:
                if not self.env_eval.envs[0].unwrapped.measure_all:
                    variables, cumulative = self.replace_measure_variable(variables, cumulative)
                    episode_infos = get_measure_graphs(episode_infos)
                else:
                    episode_infos = get_measure_graph(episode_infos)

            for i, variable in enumerate(variables):
                n_timepoints = len(episode_infos[0][variable])
                n_episodes = len(episode_infos)
                episode_results = np.empty((n_episodes, n_timepoints))
                episode_summary = np.empty(n_episodes)
                for e in range(n_episodes):
                    if isinstance(episode_infos[e][variable], dict):
                        _, y = zip(*episode_infos[e][variable].items())
                    else:
                        y = episode_infos[e][variable]
                    if variable in cumulative: y = np.cumsum(y)
                    episode_results[e, :] = y
                    episode_summary[e] = y[-1]
                variable_mean = np.mean(episode_summary, axis=0)
                self.logger.record(f'train/{variable}', variable_mean)

            fig, ax = plt.subplots()
            ax.bar(range(len(self.histogram_training_years)), list(self.histogram_training_years.values()),
                   align='center')
            ax.set_xticks(range(len(self.histogram_training_years)), minor=False)
            ax.set_xticklabels(list(self.histogram_training_years.keys()), fontdict=None, minor=False, rotation=90)
            self.logger.record(f'figures/training-years', Figure(fig, close=True))

            fig, ax = plt.subplots()
            ax.bar(range(len(self.histogram_training_locations)), list(self.histogram_training_locations.values()),
                   align='center')
            ax.set_xticks(range(len(self.histogram_training_locations)), minor=False)
            ax.set_xticklabels(list(self.histogram_training_locations.keys()), fontdict=None, minor=False)
            self.logger.record(f'figures/training-locations', Figure(fig, close=True))

            reward, fertilizer, result_model, WSO, NUE, profit, init_no3, init_nh4 = {}, {}, {}, {}, {}, {}, {}, {}
            log_training = self.get_do_log_training()

            env_pcse_evaluation = self.env_eval
            # if env_pcse_evaluation.normalize:
            #     env_pcse_evaluation = DummyVecEnv([lambda: env_pcse_evaluation])
            # else:
            #     env_pcse_evaluation = VecNormalize(DummyVecEnv([lambda: env_pcse_evaluation]),
            #                                        norm_obs=True, norm_reward=True,
            #                                        clip_obs=10., clip_reward=50., gamma=1)
            env_pcse_evaluation.training = False
            n_year_loc = 0

            total_eval = len(self.get_years(log_training)*len(self.get_locations(log_training)))
            print("evaluating environment with learned policy...")
            years_bar = tqdm(self.get_years(log_training))
            for iy, year in enumerate(years_bar, 1):
                for il, test_location in enumerate(self.get_locations(log_training), 1):
                    if not self.check_year_combination(year, test_location):
                        continue
                    years_bar.set_description(f'Evaluating {year}, {str(test_location): <{10}} | '
                                              f'{str(il+(len(self.get_locations(log_training))*iy)): <{3}}/{total_eval}')
                    env_pcse_evaluation.env_method('overwrite_year', year)
                    env_pcse_evaluation.env_method('overwrite_location', test_location)
                    env_pcse_evaluation.reset()
                    if not self.env_eval.envs[0].unwrapped.normalize:
                        sync_envs_normalization(self.model.get_env(), env_pcse_evaluation)
                    episode_rewards, episode_infos = evaluate_policy(policy=self.model, env=env_pcse_evaluation)
                    my_key = (year, test_location)
                    reward[my_key] = episode_rewards[0].item()
                    # years_bar.set_description(f'Reward {year}, {str(test_location): <{12}}: {reward[my_key]}')
                    if self.po_features:
                        episode_infos = get_measure_graphs(episode_infos)
                    fertilizer[my_key] = sum(episode_infos[0]['fertilizer'].values())
                    WSO[my_key] = list(episode_infos[0]['WSO'].values())[-1]
                    profit[my_key] = list(episode_infos[0]['profit'].values())[-1]
                    NUE[my_key] = self.get_nue(episode_infos)
                    # if self.env_eval.envs[0].unwrapped.random_init:
                        # init_no3[my_key] = episode_infos[0]['init_n']['no3']
                        # init_nh4[my_key] = episode_infos[0]['init_n']['nh4']
                    # self.logger.record(f'eval/reward-{my_key}', reward[my_key])
                    # self.logger.record(f'eval/nitrogen-{my_key}', fertilizer[my_key])
                    result_model[my_key] = episode_infos
                    n_year_loc = 0 if log_training else n_year_loc + 1
            else:
                avg_rew = mean([x for x in reward.values()])
                avg_nue = mean([x for x in NUE.values()])
                avg_profit = mean([x for x in profit.values()])
                avg_wso = mean([x for x in WSO.values()])
                print(f'Evaluation step {self.num_timesteps}\n'
                      f'Avg. reward: {avg_rew:.4f}\n'
                      f'Avg. profit: {avg_profit:.4f}\n'
                      f'Avg. NUE: {avg_nue:.4f}\n'
                      f'Avg. WSO: {avg_wso:.4f}\n')

            for test_location in list(set(self.test_locations)):
                test_keys = [(a, test_location) for a in self.test_years]
                self.logger.record(f'eval/NUE-average-test-{test_location}', compute_average(NUE, test_keys))
                self.logger.record(f'eval/NUE-median-test-{test_location}', compute_median(NUE, test_keys))
                self.logger.record(f'eval/reward-average-test-{test_location}', compute_average(reward, test_keys))
                self.logger.record(f'eval/nitrogen-average-test-{test_location}',
                                   compute_average(fertilizer, test_keys))
                self.logger.record(f'eval/profit-average-test-{test_location}',
                                   compute_average(profit, test_keys))
                self.logger.record(f'eval/WSO-average-test-{test_location}', compute_average(WSO, test_keys))
                self.logger.record(f'eval/reward-median-test-{test_location}', compute_median(reward, test_keys))
                self.logger.record(f'eval/nitrogen-median-test-{test_location}', compute_median(fertilizer, test_keys))
                self.logger.record(f'eval/WSO-median-test-{test_location}', compute_median(WSO, test_keys))
                self.logger.record(f'eval/profit-median-test-{test_location}', compute_median(profit, test_keys))

            if log_training:
                train_keys = [(a, b) for a in self.train_years for b in self.train_locations]
                self.logger.record(f'eval/reward-average-train', compute_average(reward, train_keys))
                self.logger.record(f'eval/nitrogen-average-train', compute_average(fertilizer, train_keys))
                self.logger.record(f'eval/NUE-average-train', compute_average(NUE, train_keys))
                self.logger.record(f'eval/NUE-median-train', compute_median(NUE, train_keys))
                self.logger.record(f'eval/profit-average-train', compute_average(profit, train_keys))
                self.logger.record(f'eval/reward-median-train', compute_median(reward, train_keys))
                self.logger.record(f'eval/nitrogen-median-train', compute_median(fertilizer, train_keys))
                self.logger.record(f'eval/profit-median-train', compute_median(profit, train_keys))

            self.logger.record(f'eval/NUE-median-all', compute_median(NUE))
            self.logger.record(f'eval/NUE-average-all', compute_average(NUE))
            self.logger.record(f'eval/reward-average-all', compute_average(reward))
            self.logger.record(f'eval/nitrogen-average-all', compute_average(fertilizer))
            self.logger.record(f'eval/WSO-average-all', compute_average(WSO))
            self.logger.record(f'eval/profit-average-all', compute_average(profit))
            self.logger.record(f'eval/reward-median-all', compute_median(reward))
            self.logger.record(f'eval/nitrogen-median-all', compute_median(fertilizer))
            self.logger.record(f'eval/WSO-median-all', compute_median(WSO))
            self.logger.record(f'eval/profit-median-all', compute_median(profit))

            if self.pcse_model:
                variables = ['DVS', 'action', 'WSO', 'reward',
                             'fertilizer', 'val', 'IDWST', 'prob_measure',
                             'NLOSSCUM', 'WC', 'Ndemand', 'NAVAIL', 'NuptakeTotal',
                             'SM', 'TAGP', 'LAI', 'NO3', 'NH4']
                if self.env_eval.envs[0].unwrapped.reward_function in ['NUE', 'HAR', 'END', 'ENY']:
                    variables.remove('reward')
                if self.po_features:
                    variables.append('measure')
                    # for p in self.po_features:
                    #     variables.append(p)
                if self.env_eval.envs[0].unwrapped.reward_function == 'ANE': variables.append('moving_ANE')
            else:
                variables = ['action', 'WSO', 'reward', 'TNSOIL', 'val']
                if self.po_features: variables.append('measure')

            if 'measure' in variables and not self.env_eval.envs[0].unwrapped.measure_all:
                variables = self.replace_measure_variable(variables)
                for variable in self.env_eval.envs[0].unwrapped.po_features:  # TODO make tidier
                    variable = 'prob_' + variable
                    variables += [variable]

            keys_figure = [(a, b) for a in self.test_years for b in self.test_locations]
            results_figure = {filter_key: result_model[filter_key] for filter_key in keys_figure}

            # pickle info for creating figures
            dir_log = self.logger.get_dir()
            with open(os.path.join(dir_log, f'infos_{self.num_timesteps}.pkl'), 'wb') as f:
                pickle.dump(results_figure, f)

            # if using comet, log pickle file and model as asset
            if self.comet_experiment:
                # need to check if always true
                model_num = self.num_timesteps / self.n_envs if self.multiprocess else self.num_timesteps
                list_dir = os.listdir(dir_log)
                model_files = [(file, int(file.split('-')[1].split('.')[0])) for file in list_dir if file.startswith('model-') and file.endswith("zip")]
                max_model_file = max(model_files, key=lambda x: x[1])
                latest_model_step = max_model_file[1]
                self.comet_experiment.log_asset(file_data=os.path.join(dir_log, f'infos_{self.num_timesteps}.pkl'),
                                                step=self.num_timesteps,
                                                file_name=f'infos_{self.num_timesteps}')
                self.comet_experiment.log_asset(file_data=os.path.join(dir_log, f'env-{latest_model_step}.pkl'),
                                                step=self.num_timesteps,
                                                file_name=f'env-{latest_model_step}')
                self.comet_experiment.log_model(self.comet_experiment.get_name(),
                                                os.path.join(dir_log, f'model-{latest_model_step}.zip'),
                                                file_name=f'model_{latest_model_step}')

            # create variable plot
            for i, variable in enumerate(variables):
                if variable not in results_figure[list(results_figure.keys())[0]][0].keys():
                    continue
                plot_individual = False
                if plot_individual:
                    fig, ax = plt.subplots()
                    plot_variable(results_figure, variable=variable, ax=ax, ylim=get_ylim_dict(n_year_loc)[variable])
                    self.logger.record(f'figures/{variable}', Figure(fig, close=True))
                    plt.close()

                fig, ax = plt.subplots()
                plot_variable(results_figure, variable=variable, ax=ax, ylim=get_ylim_dict(n_year_loc)[variable],
                              plot_average=True, pcse_env=self.pcse_model)
                if variable.startswith('measure'):
                    self.logger.record(f'figures/sum-{variable}', Figure(fig, close=True))
                else:
                    self.logger.record(f'figures/med-{variable}', Figure(fig, close=True))
                plt.close()

            self.logger.dump(step=self.num_timesteps)

        return True


class CometCallback(EvalCallback):
    def __init__(self, *args, comet_experiment, save_path, **kwargs):
        super(CometCallback, self).__init__(*args, **kwargs)
        self.comet_experiment = comet_experiment
        self.save_path = save_path

    def _on_step(self):
        # Call the original _on_step method
        super(CometCallback, self)._on_step()

        # Your custom Comet.ml logging logic
        if self.n_calls % self.eval_freq == 0 or self.n_calls == 1:
            # Log the .pkl file as an artifact to Comet.ml
            self.comet_experiment.log_asset(file_path=self.save_path, file_name="CustomEnvironmentData.pkl")

        return True


def determine_and_log_optimum(log_dir, env_train: Union[gym.Env, VecEnv],
                              train_years=defaults.get_default_train_years(),
                              test_years=defaults.get_default_test_years(),
                              train_locations=defaults.get_default_location(),
                              test_locations=defaults.get_default_location(),
                              n_steps=250000):
    """
    Run optimizer to find action that maximizes return value. Log to tensorboard.
    Wrapper around FindOptimum().

    :param log_dir: Tensorboard dir
    :param env_train: Base environment to find optimum for
    :param train_years: Optimum is determined on these years
    :param test_years: Used for logging
    :param train_locations: Optimum is determined on these locations
    :param test_locations: Used for logging
    :param n_steps: Used for tensorboard logging
    """

    print(f'find optimum for {train_years}')
    train_locations = [train_locations] if isinstance(train_locations, tuple) else train_locations
    test_locations = [test_locations] if isinstance(test_locations, tuple) else test_locations
    costs_nitrogen = env_train.get_attr("costs_nitrogen")

    optimizer_train = FindOptimum(env_train, train_years)
    optimum_train = optimizer_train.optimize_start_dump()
    optimum_train_path_tb = os.path.join(log_dir, f"Optimum-Ncosts-{costs_nitrogen}-train")
    optimum_train_writer = SummaryWriter(log_dir=optimum_train_path_tb)
    optimum_test_path_tb = os.path.join(log_dir, f"Optimum-Ncosts-{costs_nitrogen}-test")
    optimum_test_writer = SummaryWriter(log_dir=optimum_test_path_tb)

    reward_train, fertilizer_train = {}, {}
    reward_test, fertilizer_test = {}, {}

    for year in list(set(test_years + train_years)):
        for location in list(set(test_locations + train_locations)):
            my_key = (year, location)
            env_test = env_train
            env_test.env_method('overwrite_year', year)
            env_test.env_method('overwrite_location', location)
            env_test.reset()
            optimum_train_rewards, optimum_train_results = evaluate_policy('start-dump', env_test, amount=optimum_train)
            reward_train[my_key] = optimum_train_rewards[0].item()
            fertilizer_train[my_key] = sum(optimum_train_results[0]['action'].values())
            print(f'optimum-train: {my_key} {fertilizer_train[my_key]} {reward_train[my_key]}')
            for step in [0, n_steps]:
                optimum_train_writer.add_scalar(f'eval/reward-{my_key}', reward_train[my_key], step)
                optimum_train_writer.add_scalar(f'eval/nitrogen-{my_key}', fertilizer_train[my_key], step)
            optimum_train_writer.flush()

            print(f'find optimum-test for {my_key}')
            optimizer_test = FindOptimum(env_test)
            optimum_test = optimizer_test.optimize_start_dump()
            optimum_test_rewards, optimum_test_results = evaluate_policy('start-dump', env_test, amount=optimum_test)
            reward_test[my_key] = optimum_test_rewards[0].item()
            fertilizer_test[my_key] = sum(optimum_test_results[0]['action'].values())
            for step in [0, n_steps]:
                optimum_test_writer.add_scalar(f'eval/reward-{my_key}', reward_test[my_key], step)
                optimum_test_writer.add_scalar(f'eval/nitrogen-{my_key}', fertilizer_test[my_key], step)
            optimum_test_writer.flush()

    for location in list(set(test_locations)):
        test_keys = [(a, location) for a in test_years]
        train_keys = [(a, location) for a in train_years]
        for step in [0, n_steps]:
            optimum_test_writer.add_scalar(f'eval/reward-average-test-{location}',
                                           compute_average(reward_test, test_keys), step)
            optimum_test_writer.add_scalar(f'eval/nitrogen-average-test-{location}',
                                           compute_average(fertilizer_test, test_keys), step)
            optimum_test_writer.add_scalar(f'eval/reward-average-train-{location}',
                                           compute_average(reward_test, train_keys), step)
            optimum_test_writer.add_scalar(f'eval/nitrogen-average-train-{location}',
                                           compute_average(fertilizer_test, train_keys), step)

            optimum_train_writer.add_scalar(f'eval/reward-average-test-{location}',
                                            compute_average(reward_train, test_keys), step)
            optimum_train_writer.add_scalar(f'eval/nitrogen-average-test-{location}',
                                            compute_average(fertilizer_train, test_keys),
                                            step)
            optimum_train_writer.add_scalar(f'eval/reward-average-train-{location}',
                                            compute_average(reward_train, train_keys), step)
            optimum_train_writer.add_scalar(f'eval/nitrogen-average-train-{location}',
                                            compute_average(fertilizer_train, train_keys),
                                            step)

    optimum_train_writer.flush()
    optimum_test_writer.flush()
