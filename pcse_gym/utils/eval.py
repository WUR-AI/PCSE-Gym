import os
import datetime
import pandas as pd
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from scipy.optimize import minimize_scalar
from bisect import bisect_left
from typing import Union
from tqdm import tqdm
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from sb3_contrib import RecurrentPPO
import pcse_gym.utils.defaults as defaults
from pcse_gym.utils.process_pcse_output import get_dict_lintul_wofost


def compute_median(results_dict: dict, filter_list=None):
    if filter_list is None:
        filter_list = list(results_dict.keys())
    filtered_results = [results_dict[f] for f in filter_list if f in results_dict.keys()]
    return np.median(filtered_results)


def get_cumulative_variables():
    return ['fertilizer', 'reward']


def get_yield_variable(pcse_model):
    """Storage organ variable name in episode info (WOFOST80 uses TWSO, LINTUL/SNOMIN use WSO)."""
    return "TWSO" if pcse_model == 1 else "WSO"


def get_train_log_variables(pcse_model, po_features=False):
    variables = ["action", get_yield_variable(pcse_model), "reward"]
    if po_features:
        variables.append("measure")
    return variables, ["action", "reward"]


def get_eval_log_variables(pcse_model, po_features=False):
    if pcse_model == 1:
        variables = ["action", "TWSO", "reward", "NAVAIL", "NuptakeTotal", "fertilizer", "val"]
    elif pcse_model == 2:
        variables = ["action", "WSO", "reward", "NLOSSCUM", "NuptakeTotal", "fertilizer", "val"]
    else:
        variables = ["action", "WSO", "reward", "TNSOIL", "val"]
    if po_features:
        variables.append("measure")
    return variables


def get_ylim_dict():
    def def_value():
        return None

    ylim = defaultdict(def_value)
    ylim['WSO'] = [0, 1000]
    ylim['TWSO'] = [0, 10000]
    return ylim


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


def get_titles():
    def def_value(): return ("", "")

    return_dict = defaultdict(def_value)
    return_dict["DVS"] = ("Development stage", "-")
    return_dict["TGROWTH"] = ("Total biomass (above and below ground)", "g/m2")
    return_dict["LAI"] = ("Leaf area Index", "-")
    return_dict["NUPTT"] = ("Total nitrogen uptake", "gN/m2")
    return_dict["TRAN"] = ("Transpiration", "mm/day")
    return_dict["TIRRIG"] = ("Total irrigation", "mm")
    return_dict["TNSOIL"] = ("Total soil inorganic nitrogen", "gN/m2")
    return_dict["TRAIN"] = ("Total rainfall", "mm")
    return_dict["TRANRF"] = ("Transpiration reduction factor", "-")
    return_dict["TRUNOF"] = ("Total runoff", "mm")
    return_dict["TAGBM"] = ("Total aboveground biomass", "g/m2")
    return_dict["TTRAN"] = ("Total transpiration", "mm")
    return_dict["WC"] = ("Soil water content", "m3/m3")
    return_dict["WLVD"] = ("Weight dead leaves", "g/m2")
    return_dict["WLVG"] = ("Weight green leaves", "g/m2")
    return_dict["WRT"] = ("Weight roots", "g/m2")
    return_dict["WSO"] = ("Weight storage organs", "g/m2")
    return_dict["TWSO"] = ("Weight storage organs", "kg/ha")
    return_dict["WST"] = ("Weight stems", "g/m2")
    return_dict["TGROWTHr"] = ("Growth rate", "g/m2/day")
    return_dict["NRF"] = ("Nitrogen reduction factor", "-")
    return_dict["GRF"] = ("Growth reduction factor", "-")

    return_dict["DVS"] = ("Development stage", "-")
    return_dict["TAGP"] = ("Total above-ground Production", "kg/ha")
    return_dict["LAI"] = ("Leaf area Index", "-")
    return_dict["RNuptake"] = ("Total nitrogen uptake", "kgN/ha")
    return_dict["TRA"] = ("Transpiration", "cm/day")
    return_dict["NAVAIL"] = ("Total soil inorganic nitrogen", "kgN/ha")
    return_dict["SM"] = ("Volumatric soul moisture content", "-")
    return_dict["RFTRA"] = ("Transpiration reduction factor", "-")
    return_dict["TRUNOF"] = ("Total runoff", "mm")
    return_dict["TAGBM"] = ("Total aboveground biomass", "kg/ha")
    return_dict["TTRAN"] = ("Total transpiration", "mm")
    return_dict["WC"] = ("Soil water content", "m3/m3")
    return_dict["Ndemand"] = ("Total N demand of crop", "kgN/ha")
    return_dict["NuptakeTotal"] = ("Total N uptake of crop", "kgN/ha/d")
    return_dict["FERT_N_SUPPLY"] = ("Total N supplied by actions", "kgN/ha")

    return_dict["fertilizer"] = ("Nitrogen application", "kg/ha")
    return_dict["TMIN"] = ("Minimum temperature", "°C")
    return_dict["TMAX"] = ("Maximum temperature", "°C")
    return_dict["IRRAD"] = ("Incoming global radiation", "J/m2/day")
    return_dict["RAIN"] = ("Daily rainfall", "cm/day")

    return return_dict


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


def plot_variable(results_dict, variable='reward', cumulative_variables=get_cumulative_variables(), ax=None, ylim=None,
                  put_legend=True, plot_average=False):
    titles = get_titles()
    xmax = 0
    for label, results in results_dict.items():
        x, y = zip(*results[0][variable].items())
        x = ([i.timetuple().tm_yday for i in x])
        if variable in cumulative_variables: y = np.cumsum(y)
        if max(x) > xmax: xmax = max(x)
        if not plot_average:
            ax.step(x, y, label=label, where='post')

    if plot_average:
        plot_df = pd.concat([pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
                            .rename(lambda i: i.timetuple().tm_yday) for label, results in results_dict.items()],
                            axis=1)
        if variable in cumulative_variables: plot_df = plot_df.apply(np.cumsum, axis=0)
        plot_df.ffill(axis=0, inplace=True)
        ax.step(plot_df.index, plot_df.median(axis=1), 'k-', where='post')
        ax.fill_between(plot_df.index, plot_df.quantile(0.25, axis=1), plot_df.quantile(0.75, axis=1), step='post')

    ax.axhline(y=0, color='lightgrey', zorder=1)
    ax.margins(x=0)

    from matplotlib.ticker import FixedLocator
    ax.xaxis.set_minor_locator(FixedLocator(range(0, xmax, 7)))
    ax.xaxis.grid(True, which='minor')
    ax.tick_params(axis='x', which='minor', grid_alpha=0.7, colors=ax.get_figure().get_facecolor(), grid_ls=":")

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    month_days = [0, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    extra_month = next(x[0] for x in enumerate(month_days) if x[1] >= xmax)
    month_days = month_days[0:extra_month + 1]
    months = months[0:extra_month + 1]
    ax.set_xticks(month_days)
    ax.set_xticklabels(months)

    name, unit = titles[variable]
    ax.set_title(f"{variable} - {name}")
    if variable in cumulative_variables:
        ax.set_title(f"{variable} (cumulative) - {name}")
    ax.set_ylabel(f"[{unit}]")
    if ylim is not None:
        ax.set_ylim(ylim)
    if put_legend:
        ax.legend()
    else:
        ax.legend()
        ax.get_legend().set_visible(False)
    return ax


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
    return sum(filtered_results) / len(filtered_results)


def to_scalar(x):
    """Convert numpy arrays / 0-d or 1-d tensors to a Python float."""
    return float(np.asarray(x).ravel()[0])


def _standard_practice_action(env, amount=1):
    """Return action for standard practice baseline (3 split applications per season)."""
    try:
        sb3_env = env.get_attr("sb3_env")[0]
        if getattr(sb3_env, "_uses_snomin", False):
            from pcse_gym.utils.nitrogen_helpers import get_standard_practice_action_index
            return [get_standard_practice_action_index(sb3_env.nitrogen_levels)]
    except (AttributeError, IndexError, TypeError):
        pass
    return [amount * 3]


def _standard_practice_fert_dates(env):
    """Spring split N dates (Feb–Apr) in the harvest year of a winter-cereal campaign."""
    try:
        sb3_env = env.get_attr("sb3_env")[0]
        application_year = sb3_env.agmt.crop_end_date.year
    except (AttributeError, IndexError, TypeError):
        application_year = env.get_attr("date")[0].year
    return [
        datetime.date(application_year, 2, 24),
        datetime.date(application_year, 3, 26),
        datetime.date(application_year, 4, 29),
    ]


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
        training = policy.get_env().training
        policy.get_env().training = False
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    action_space = env.action_space
    if isinstance(action_space, list):
        action_space = action_space[0]

    if isinstance(policy, base_class.BaseAlgorithm):
        assert (policy.action_space == action_space)

    episode_rewards, episode_infos = [], []
    for i in range(n_eval_episodes):
        if isinstance(policy, base_class.BaseAlgorithm):
            sync_envs_normalization(policy.get_env(), env)
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        terminated, truncated, state, lstm_state = False, False, None, None
        episode_reward = 0.0
        episode_length = 0
        fert_dates = _standard_practice_fert_dates(env)
        action = [amount * 0]
        infos_this_episode = []
        prob, val = None, None

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
                if isinstance(policy, RecurrentPPO):
                    action, lstm_state = policy.predict(obs, state=lstm_state, deterministic=deterministic)
                if isinstance(policy, DQN):
                    action = policy.predict(obs, deterministic=deterministic)

            # SB3 VecEnvs don't follow the gymnasium step API, this is a quick fix.
            # see: https://github.com/DLR-RM/stable-baselines3/blob/master/docs/guide/vec_envs.rst
            # TODO: add check on function signature
            obs, rew, terminated, info = env.step(action)
            truncated = info[0].pop("TimeLimit.truncated")

            reward = env.get_original_reward()

            if prob:
                action_date = list(info[0]['action'].keys())[0]
                info[0]['prob'] = {action_date: prob}
                info[0]['dvs'] = {action_date: info[0]['DVS'][action_date]}
            if val:
                val_date = list(info[0]['action'].keys())[0]
                info[0]['val'] = {val_date: val}

            action = [amount * 0]
            if policy in ['standard-practice', 'standard-practise']:
                date = env.get_attr("date")[0]
                for fert_date in fert_dates:
                    if fert_date <= date < fert_date + datetime.timedelta(days=7):
                        action = _standard_practice_action(env, amount)
            if policy == 'no-nitrogen':
                action = [0]
            episode_reward += to_scalar(reward)
            episode_length += 1
            infos_this_episode.append(info[0])
        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        episode_rewards.append(to_scalar(episode_reward))
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
                total_reward = total_reward + to_scalar(reward)
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


def get_cumulative_variables():
    return ['fertilizer', 'reward']


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

    def __init__(self, env_eval=None, train_years=defaults.get_default_train_years(), test_years=defaults.get_default_test_years(),
                 train_locations=defaults.get_default_location(), test_locations=defaults.get_default_location(),
                 n_eval_episodes=1, eval_freq=1000, pcse_model=1, seed=0, **kwargs):
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
        self.po_features = kwargs.get('po_features')

        def def_value(): return 0

        self.histogram_training_years = defaultdict(def_value)

        def def_value(): return 0

        self.histogram_training_locations = defaultdict(def_value)

    def get_locations(self, log_training=False):
        if log_training:
            locations = list(set(self.test_locations + self.train_locations))
        else:
            locations = list(set(self.test_locations))
        return locations

    def get_years(self, log_training=False):
        if log_training:
            years = list(set(self.test_years + self.train_years))
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

    def get_measure_graphs(self, episode_infos):
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
            stats_path = os.path.join(tensorboard_logdir, f'env-{self.n_calls}.pkl')
            self.model.get_env().save(stats_path)
            episode_rewards, episode_infos = evaluate_policy(policy=self.model, env=self.model.get_env())
            variables, cumulative = get_train_log_variables(self.pcse_model, self.po_features)

            '''logic for measure graph'''
            if 'measure' in variables:
                variables, cumulative = self.replace_measure_variable(variables, cumulative)
                episode_infos = self.get_measure_graphs(episode_infos)

            for i, variable in enumerate(variables):
                if variable not in episode_infos[0]:
                    continue
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

            reward, fertilizer, result_model = {}, {}, {}
            log_training = self.get_do_log_training()

            env_pcse_evaluation = self.env_eval
            env_pcse_evaluation = VecNormalize(DummyVecEnv([lambda: env_pcse_evaluation]),
                                               norm_obs=True, norm_reward=True,
                                               clip_obs=10., clip_reward=50., gamma=1)
            env_pcse_evaluation.training = False

            total_eval = len(self.get_years(log_training)) * len(self.get_locations(log_training))
            print("evaluating environment with learned policy...")
            years_bar = tqdm(self.get_years(log_training))
            for iy, year in enumerate(years_bar, 1):
                for il, test_location in enumerate(self.get_locations(log_training), 1):
                    years_bar.set_description(f'Evaluating {year}, {str(test_location): <{11}} | '
                                              f'{str(il + (len(self.get_locations(log_training)) * iy)): <{3}}/{total_eval}')
                    env_pcse_evaluation.env_method('overwrite_year', year)
                    env_pcse_evaluation.env_method('overwrite_location', test_location)
                    env_pcse_evaluation.reset()
                    sync_envs_normalization(self.model.get_env(), env_pcse_evaluation)
                    episode_rewards, episode_infos = evaluate_policy(policy=self.model, env=env_pcse_evaluation)
                    my_key = (year, test_location)
                    reward[my_key] = to_scalar(episode_rewards[0])
                    if self.po_features:
                        episode_infos = self.get_measure_graphs(episode_infos)
                    fertilizer[my_key] = sum(episode_infos[0]['fertilizer'].values())
                    self.logger.record(f'eval/reward-{my_key}', reward[my_key])
                    self.logger.record(f'eval/nitrogen-{my_key}', fertilizer[my_key])
                    result_model[my_key] = episode_infos
            else:
                avg_rew = mean([x for x in reward.values()])
                print(f'Evaluation step {self.num_timesteps} | Avg. reward: {avg_rew:.4f}')

            for test_location in list(set(self.test_locations)):
                test_keys = [(a, test_location) for a in self.test_years]
                self.logger.record(f'eval/reward-average-test-{test_location}', compute_average(reward, test_keys))
                self.logger.record(f'eval/nitrogen-average-test-{test_location}',
                                   compute_average(fertilizer, test_keys))
                self.logger.record(f'eval/reward-median-test-{test_location}', compute_median(reward, test_keys))
                self.logger.record(f'eval/nitrogen-median-test-{test_location}', compute_median(fertilizer, test_keys))

            if log_training:
                train_keys = [(a, b) for a in self.train_years for b in self.train_locations]
                self.logger.record(f'eval/reward-average-train', compute_average(reward, train_keys))
                self.logger.record(f'eval/nitrogen-average-train', compute_average(fertilizer, train_keys))
                self.logger.record(f'eval/reward-median-train', compute_median(reward, train_keys))
                self.logger.record(f'eval/nitrogen-median-train', compute_median(fertilizer, train_keys))

            variables = get_eval_log_variables(self.pcse_model, self.po_features)

            keys_figure = [(a, b) for a in self.test_years for b in self.test_locations]
            results_figure = {filter_key: result_model[filter_key] for filter_key in keys_figure}

            for i, variable in enumerate(variables):
                if variable not in results_figure[list(results_figure.keys())[0]][0].keys():
                    continue
                plot_individual = False
                if plot_individual:
                    fig, ax = plt.subplots()
                    plot_variable(results_figure, variable=variable, ax=ax, ylim=get_ylim_dict()[variable])
                    self.logger.record(f'figures/{variable}', Figure(fig, close=True))
                    plt.close()

                fig, ax = plt.subplots()
                plot_variable(results_figure, variable=variable, ax=ax, ylim=get_ylim_dict()[variable],
                              plot_average=True)
                self.logger.record(f'figures/avg-{variable}', Figure(fig, close=True))
                plt.close()
            self.logger.dump(step=self.num_timesteps)

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
