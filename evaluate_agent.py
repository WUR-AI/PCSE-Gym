import os
import argparse
import pickle
from statistics import mean, median

import numpy as np
import gymnasium as gym
import lib_programname
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import datetime

from torch.utils.tensorboard import SummaryWriter

from pcse_gym.initialize_envs import initialize_env as init_env
from pcse_gym.envs.winterwheat import WinterWheat, WinterWheatRay
import pcse_gym.utils.defaults as defaults
import pcse_gym.utils.eval as eval
from pcse_gym.utils.nitrogen_helpers import get_standard_practices, treatments_list
from pcse_gym.envs.constraints import ActionConstrainer

path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[0]

evaluate_dir = os.path.join(rootdir, "tensorboard_logs", "evaluation_runs")


def get_po_features(pcse_env=1):
    if pcse_env:
        po_features = ['TAGP', 'LAI', 'NAVAIL', 'SM', 'NuptakeTotal']
    else:
        po_features = ['TGROWTH', 'LAI', 'TNSOIL', 'NUPTT', 'TRAIN']
    return po_features


def get_action_space(nitrogen_levels=7, po_features=[]):
    if po_features:
        a_shape = [nitrogen_levels] + [2] * len(po_features)
        space_return = gym.spaces.MultiDiscrete(a_shape)
    else:
        space_return = gym.spaces.Discrete(nitrogen_levels)
    return space_return


def measure_history_histogram(data, year, location, crop_var, axes):
    # Extract data for the given location
    if isinstance(location, tuple):
        location = str(location)
    loc_data = data.get(location, {})

    # Extract the values and dates for the given year and variable
    dates = []
    values = []
    for date_str, var_data in loc_data.items():
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        if date_obj.year == year and crop_var in var_data:
            dates.append(date_obj)
            values.append(var_data[crop_var])

    # Plotting
    axes.bar(dates, values, width=8, align='center')
    axes.set_title(f"Histogram for {crop_var} in {location}, {year}")
    axes.set_xlabel("Date")
    axes.set_ylabel("Measure?")
    axes.set_ylim([0, 1.1])  # Since values are only 0 and 1
    axes.grid(axis='y')


def evaluate_policy(policy, env, n_eval_episodes=1, framework='sb3'):
    episode_rewards, episode_infos = [], []
    for i in range(n_eval_episodes):
        episode_length = 0
        episode_reward = 0
        obs = env.reset()
        if framework == 'rllib':
            state = policy.get_initial_state()
        else:
            state = None
        terminated, truncated, prev_action, prev_reward, info = False, False, None, None, None
        infos_this_episode = []

        while not terminated or truncated:
            action, state, _ = policy.compute_single_action(obs=obs, state=state, prev_action=prev_action,
                                                            prev_reward=prev_reward, info=info)
            obs, reward, terminated, truncated, info = env.step(action)

            prev_action, prev_reward = action, reward
            episode_reward += reward
            episode_length += 1
            infos_this_episode.append(info)
        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        episode_rewards.append(episode_reward)
        episode_infos.append(episode_info)
    return episode_rewards, episode_infos


def evaluate_treatment(policy, env, n_eval_episodes=1):
    episode_rewards, episode_infos = [], []
    for i in range(n_eval_episodes):
        episode_length = 0
        episode_reward = 0
        terminated, truncated, prev_action, prev_reward, info = False, False, None, None, None
        infos_this_episode = []
        fert_dates, fert_amounts = get_standard_practices(policy, env.sb3_env.agmt.get_end_date.year)

        while not terminated or truncated:
            date = env.date
            action = 0
            for amount, fert_date in enumerate(fert_dates):
                if fert_date < date <= fert_date + datetime.timedelta(7):
                    action = fert_amounts[amount] / 10
                    print(f"fertilized {action} at {fert_date}")
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            infos_this_episode.append(info)
        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        episode_rewards.append(episode_reward)
        episode_infos.append(episode_info)
    return episode_rewards, episode_infos


def get_demeter_policy(location, year, soil=None, init_n=None, constrained=True):
    if constrained:
        if soil is not None:
            if 'fast' in soil:
                with open(os.path.join(rootdir, "demeter_results", 'constrained_all_fast_soil',
                                       f"{location[0]}-{location[1]}-{year}.csv"), "r") as file:
                    df_demeter = pd.read_csv(file, header=0)
                return list(df_demeter[str(year)])
            elif 'slo' in soil:
                with open(os.path.join(rootdir, "demeter_results", 'constrained_all_slow_soil',
                                       f"{location[0]}-{location[1]}-{year}.csv"), "r") as file:
                    df_demeter = pd.read_csv(file, header=0)
                return list(df_demeter[str(year)])
        if init_n is not None:
            if 'low' in init_n:
                with open(os.path.join(rootdir, "demeter_results", 'constrained_all_low_n',
                                       f"{location[0]}-{location[1]}-{year}.csv"), "r") as file:
                    df_demeter = pd.read_csv(file, header=0)
                return list(df_demeter[str(year)])
            elif 'high' in init_n:
                with open(os.path.join(rootdir, "demeter_results", 'constrained_all_high_n',
                                       f"{location[0]}-{location[1]}-{year}.csv"), "r") as file:
                    df_demeter = pd.read_csv(file, header=0)
                return list(df_demeter[str(year)])
        with open(os.path.join(rootdir, "demeter_results", 'constrained_all', f"{location[0]}-{location[1]}-{year}.csv"), "r") as file:
            df_demeter = pd.read_csv(file, header=0)
        return list(df_demeter[str(year)])
    with open(os.path.join(rootdir, "demeter_results", f"{location[0]}-{location[1]}-{year}.csv"), "r") as file:
        df_demeter = pd.read_csv(file, header=0)
    return list(df_demeter[str(year)])


def evaluate_demeter(env, n_eval_episodes=1, constrained=True, soil=None, init_n=None):
    episode_rewards, episode_infos = [], []
    for i in range(n_eval_episodes):
        episode_reward = 0
        terminated, truncated, prev_action, prev_reward, info = False, False, None, None, None
        infos_this_episode = []
        fert_amounts = get_demeter_policy(env.loc, env.sb3_env.agmt.get_end_date.year, soil=soil, init_n=init_n, constrained=constrained)
        week = 0
        while not terminated or truncated:
            action = 0
            if constrained:
                if 5 <= week < 30:
                    if fert_amounts[week-5] > 0.0:
                        action = fert_amounts[week-5]
                        print(f"fertilized {action} at week {week}")
            else:
                if fert_amounts[week] > 0.0:
                    action = fert_amounts[week]
                    print(f"fertilized {action} at week {week}")
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            week += 1
            infos_this_episode.append(info)

        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        episode_rewards.append(episode_reward)
        episode_infos.append(episode_info)
    return episode_rewards, episode_infos

def low_n_init() -> dict:
    n_dict = {'NH4I': [0.06181797, 0.33827534, 0.12490669, 0.06109704, 0.06373443, 0.09213926, 0.00802928],
              'NO3I': [2.19391276, 0.33791361, 0.44317362, 0.08596978, 0.65234924, 0.07112439, 0.4655566], }
    return n_dict


def high_n_init() -> dict:
    n_dict = {'NH4I': [3.15162898, 3.87070563, 1.37766539, 1.17044373, 1.58129878, 0.66062145, 0.18763604],
              'NO3I': [7.42620968, 20.4380931, 29.73569722, 11.2684839, 7.24005868, 7.94257558, 8.94888184], }
    return n_dict


def mid_n_init() -> dict:
    n_dict = {'NH4I': [1.7436090131474202, 1.0330637061144692, 1.411524101062196, 0.7287729016659076, 0.24795527588005897, 0.06906348892304397, 0.7491498279555974],
              'NO3I': [7.09833746582459, 6.546983849224111, 10.087794000121118, 6.314227315758486, 0.4043760804573563, 0.48318121856756163, 2.969550520289376], }
    return n_dict


def select_init_n_scenario(starting_n):
    if starting_n == 'high':
        return high_n_init()
    elif starting_n == 'low':
        return low_n_init()
    elif starting_n == 'mid':
        return mid_n_init()
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--step", type=int, default=250000)
    parser.add_argument("-c", "--costs_nitrogen", type=float, default=10.0, help="Costs for nitrogen")
    parser.add_argument("-a", "--agent", type=str, default="PPO", help="RL agent or other policies. PPO,"
                                                                       "RPPO, DQN, standard-practise, or Treatments")
    parser.add_argument("-e", "--environment", type=int, default=2)
    parser.add_argument("-r", "--reward", type=str, default="DEF", help="Reward function. DEF, DEP, GRO, or ANE")
    parser.add_argument("-b", "--n_budget", type=int, default=0, help="Nitrogen budget. kg/ha")
    parser.add_argument("--action_limit", type=int, default=0, help="Limit fertilization frequency."
                                                                    "Recommended 4 times")
    parser.add_argument("--random-weather", action='store_true', dest='random_weather')
    parser.add_argument("-m", "--measure", action='store_true')
    parser.add_argument("--no-measure", action='store_false', dest='measure')
    parser.add_argument("-l", "--location", type=str, default="NL", help="NL or LT.")
    parser.add_argument("-y", "--year", type=int, default=None, help="year to evaluate agent")
    parser.add_argument("--variable-recovery-rate", action='store_true', dest='vrr')
    parser.add_argument("--noisy-measure", action='store_true', dest='noisy_measure')
    parser.add_argument("--no-weather", action='store_true', dest='no_weather')
    parser.add_argument("--random_feature", action='store_true', dest='random_feature')
    parser.add_argument("--init-n", type=str, default=None, dest='init_n')
    parser.add_argument("--vision", type=str, default=None, dest='vision')
    parser.add_argument("-v", "--visualize", action='store_true', default=False, dest='visualize')
    parser.add_argument('--experiment', type=str, default='main_model', help="Experiment name")
    parser.add_argument('--soil', type=str, default=None)
    parser.set_defaults(measure=False, vrr=False, noisy_measure=False, framework='sb3', no_weather=False,
                        random_feature=False, random_weather=False)
    args = parser.parse_args()

    framework_path = "WOFOST_experiments"
    if not args.measure and args.noisy_measure:
        parser.error("noisy measure should be used with measure")
    pcse_model_name = "LINTUL" if not args.environment else "WOFOST"
    pcse_model = args.environment

    init_n = args.init_n
    soil = args.soil

    if args.agent == 'demeter':
        eval_locations = [(52.57, 5.63)]
        if args.year is not None:
            if isinstance(args.year, int):
                eval_year = [args.year]
            else:
                eval_year = args.year
        else:
            eval_year = [*range(1983, 2022)]
    else:
        if args.location == "NL":
            """The Netherlands"""
            eval_locations = [(52, 5.5)]  #, (51.5, 5), (52.5, 5.5)]
        elif args.location == "PAGV":
            eval_locations = [(52.57, 5.63)]
        elif args.location == "LT":
            """Lithuania"""
            eval_locations = [(55.0, 23.5), (55.0, 24.0), (55.5, 23.5)]
        else:
            parser.error("--location arg should be either LT or NL")
        if args.year is not None:
            if isinstance(args.year, int):
                eval_year = [args.year]
            else:
                eval_year = args.year
        else:
            # eval_year = [year for year in [*range(1990, 2024)] if year % 2 == 0]
            if args.location in ["NL", 'LT']:
                eval_year = [*range(1990, 2024)]
            elif args.location == "PAGV":
                eval_year = [*range(1983, 2022)]
    crop_features = defaults.get_default_crop_features(pcse_env=args.environment, vision=args.vision)
    weather_features = defaults.get_default_weather_features()
    action_features = defaults.get_default_action_features(True)

    kwargs = {'args_vrr': args.vrr, 'action_limit': args.action_limit, 'noisy_measure': args.noisy_measure,
              'n_budget': args.n_budget, 'framework': args.framework, 'no_weather': args.no_weather,
              'random_weather': args.random_weather}

    if soil is not None:
        # import yaml
        # config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pcse_gym', 'envs', 'configs')
        # if soil == "fast":
        kwargs['soil'] = soil
        # if soil == "slow":
        #     kwargs['soil_params'] = yaml.safe_load(open(os.path.join(config_dir, 'soil', 'EC6-fine_soil.yaml')))


    if not args.measure:
        action_spaces = gym.spaces.Discrete(9)
    else:
        if args.environment:
            po_features = ['TAGP', 'LAI', 'NAVAIL', 'NuptakeTotal', 'SM']
            if args.random_feature:
                po_features.append('random')
            if 'random' in po_features:
                crop_features.append('random')
        else:
            po_features = ['TGROWTH', 'LAI', 'TNSOIL', 'NUPTT', 'TRAIN']
        kwargs['po_features'] = po_features
        kwargs['args_measure'] = True
        if not args.noisy_measure:
            m_shape = 2
        else:
            m_shape = 3
        a_shape = [7] + [m_shape] * len(po_features)
        action_spaces = gym.spaces.MultiDiscrete(a_shape)

    # check paths
    if os.path.isdir(os.path.join(rootdir, "nue_experiments", args.experiment, args.checkpoint_path)):
        checkpoint_folder = os.path.join(rootdir, "nue_experiments", args.experiment, args.checkpoint_path)
    elif os.path.isdir(os.path.join(rootdir, "tensorboard_logs", framework_path, args.checkpoint_path)):
        print(f"{os.path.join(rootdir, 'nue_experiments', args.experiment, args.checkpoint_path)} is not a path.")
        checkpoint_folder = os.path.join(rootdir, "tensorboard_logs", framework_path, args.checkpoint_path)
    else:
        print(f"{os.path.join(rootdir, 'nue_experiments', args.experiment, args.checkpoint_path)} and "
              f"{os.path.isdir(os.path.join(rootdir, 'tensorboard_logs', framework_path, args.checkpoint_path))} is not a path.")
        os.mkdir(os.path.join(rootdir, "tensorboard_logs", framework_path, args.checkpoint_path))
        checkpoint_folder = os.path.join(rootdir, "tensorboard_logs", framework_path, args.checkpoint_path)

    if args.agent in ['PPO', 'RPPO', 'MaskedPPO', 'LagPPO']:
        model_file_to_load = os.listdir(checkpoint_folder)
        model_zip_name = [a for a in model_file_to_load if a.endswith(".zip")][0]
        env_pkl_name = [a for a in model_file_to_load if 'env' in a][0]
        if os.path.isdir(os.path.join(rootdir, "nue_experiments", args.experiment, args.checkpoint_path)):
            checkpoint_path = os.path.join(checkpoint_folder, model_zip_name)
            stats_path = os.path.join(checkpoint_folder, env_pkl_name)
        else:
            checkpoint_path = os.path.join(rootdir, "tensorboard_logs", framework_path, args.checkpoint_path,
                                           model_zip_name)
            stats_path = os.path.join(rootdir, "tensorboard_logs", framework_path, args.checkpoint_path,
                                  env_pkl_name)

    agent = args.agent
    # if args.framework == 'rllib':
    #     raise NotImplementedError
    #     import ray
    #     from ray.rllib.algorithms.algorithm import Algorithm
    #
    #     agent = Algorithm.from_checkpoint(checkpoint_path)
    #     initialize_env(**kwargs)
    #     policy = agent.get_policy()
    #
    #     pass
    if args.framework == 'sb3':
        from stable_baselines3 import PPO
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import VecEnv, VecNormalize, DummyVecEnv, sync_envs_normalization
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib import MaskablePPO as MaskedPPO
        from pcse_gym.agent.ppo_mod import LagrangianPPO as LagPPO

        nitrogen_levels = 9  # 0 - 80 kg/ha
        env = init_env(crop_features=crop_features,
                       costs_nitrogen=args.costs_nitrogen,
                       years=eval_year,
                       locations=eval_locations,
                       reward=args.reward,
                       pcse_env=args.environment,
                       nitrogen_levels=nitrogen_levels,
                       action_features=action_features,
                       **kwargs)
        cust_objects = {"lr_schedule": lambda x: 0.001, "clip_range": lambda x: 0.2,
                        "action_space": action_spaces}
        if args.agent in ['PPO', 'RPPO', 'MaskedPPO', 'LagPPO']:
            env = ActionConstrainer(env, action_limit=args.action_limit, n_budget=args.n_budget)
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(stats_path, env)
            if args.agent == 'RPPO':
                agent = RecurrentPPO.load(checkpoint_path, custom_objects=cust_objects, device='cuda',
                                          print_system_info=True)
            elif args.agent == 'PPO':
                agent = PPO.load(checkpoint_path, custom_objects=cust_objects, device='cuda', print_system_info=True)

            elif args.agent == 'MaskedPPO':
                agent = MaskedPPO.load(checkpoint_path, custom_objects=cust_objects, device='cuda', print_system_info=True)
            elif args.agent == 'LagPPO':
                agent = LagPPO.load(checkpoint_path, custom_objects=cust_objects, device='cuda', print_system_info=True)
        policy = agent

    print(policy.policy if not isinstance(policy, str) else policy)
    evaluate_dir = os.path.join(evaluate_dir, args.checkpoint_path)
    writer = SummaryWriter(log_dir=checkpoint_folder)

    reward, fertilizer, result_model, WSO, NUE, profit, Nsurplus, Nloss, action_idx = {}, {}, {}, {}, {}, {}, {}, {}, {}

    total_eval = len(eval_year) * len(eval_locations)
    print("evaluating environment with learned policy...")
    years_bar = tqdm(eval_year)
    for iy, year in enumerate(years_bar, 1):
        for il, test_location in enumerate(eval_locations, 1):
            years_bar.set_description(f'Evaluating {year}, {str(test_location): <{10}} | '
                                      f'{str(il + (len(eval_locations) * iy)): <{3}}/{total_eval}')
            if args.framework == 'sb3':
                if isinstance(agent, PPO) or isinstance(agent, RecurrentPPO) or isinstance(agent, MaskedPPO):
                    env.env_method('overwrite_year', year)
                    env.env_method('overwrite_location', test_location)
                    env.env_method('reset', options=select_init_n_scenario(init_n))
                    sync_envs_normalization(agent.get_env(), env)
                    episode_rewards, episode_infos = eval.evaluate_policy(policy=policy, env=env)
                elif args.agent == 'demeter':
                    env.overwrite_year(year)
                    env.overwrite_location(test_location)
                    env.reset(options=select_init_n_scenario(init_n))
                    episode_rewards, episode_infos = evaluate_demeter(env=env, soil=soil, init_n=init_n)
                else:
                    env.overwrite_year(year)
                    env.overwrite_location(test_location)
                    env.reset(options=select_init_n_scenario(init_n))
                    episode_rewards, episode_infos = evaluate_treatment(policy=policy, env=env)
            elif args.framework == 'rllib':
                env.overwrite_year(year)
                env.overwrite_location(test_location)
                env.reset()
                episode_rewards, episode_infos = evaluate_policy(policy=policy, env=env, framework=args.framework)
            my_key = (year, test_location)
            reward[my_key] = episode_rewards[0] if isinstance(episode_rewards[0], float) else episode_rewards[0].item()
            WSO[my_key] = list(episode_infos[0]['WSO'].values())[-1]
            profit[my_key] = list(episode_infos[0]['profit'].values())[-1]
            NUE[my_key] = list(episode_infos[0]['NUE'].values())[-1]
            Nsurplus[my_key] = list(episode_infos[0]['Nsurplus'].values())[-1]
            Nloss[my_key] = list(episode_infos[0]['NLOSSCUM'].values())[-1]
            action_idx[my_key] = np.where(np.array(list(episode_infos[0]['action'].values())) > 0)[0]
            if args.framework == 'sb3':
                if isinstance(env, VecEnv):
                    if env.unwrapped.envs[0].unwrapped.po_features:
                        episode_infos = eval.get_measure_graphs(episode_infos)
            elif args.framework == 'rllib':
                if env.po_features:
                    episode_infos = eval.get_measure_graphs(episode_infos)
            fertilizer[my_key] = sum(episode_infos[0]['fertilizer'].values())
            writer.add_scalar(f'eval/reward-{my_key}', reward[my_key])
            writer.add_scalar(f'eval/nitrogen-{my_key}', fertilizer[my_key])
            writer.add_scalar(f'eval/WSO-{my_key}', WSO[my_key])
            writer.add_scalar(f'eval/profit-{my_key}', profit[my_key])
            writer.add_scalar(f'eval/NUE-{my_key}', NUE[my_key])
            writer.add_scalar(f'eval/Nsurplus-{my_key}', Nsurplus[my_key])
            result_model[my_key] = episode_infos
    else:
        avg_rew = eval.means_for_progress_bar(reward)
        avg_nue = eval.means_for_progress_bar(NUE)
        avg_profit = eval.means_for_progress_bar(profit)
        avg_wso = eval.means_for_progress_bar(WSO)
        avg_nsurplus = eval.means_for_progress_bar(Nsurplus)
        med_rew = eval.medians_for_progress_bar(reward)
        med_nue = eval.medians_for_progress_bar(NUE)
        med_profit = eval.medians_for_progress_bar(profit)
        med_wso = eval.medians_for_progress_bar(WSO)
        med_nsurplus = eval.medians_for_progress_bar(Nsurplus)
        nue = [x for x in NUE.values()]
        nsurp = [x for x in Nsurplus.values()]
        pass_nue = [1 if 0.5 <= x <= 0.9 else 0 for x in nue]
        pass_nsurp = [1 if 0 < x <= 40 else 0 for x in nsurp]
        length = len(nue)
        acts = list({item for sublist in list(action_idx.values()) for item in sublist})
        print(f'Within NUE: {sum(pass_nue)}/{length}\n'
              f'Within Nsurplus: {sum(pass_nsurp)}/{length}\n'
              f'Med. reward: {med_rew:.4f}\n'
              f'Med. profit: {med_profit:.4f}\n'
              f'Med. NUE: {med_nue:.4f}\n'
              f'Med. WSO: {med_wso:.4f}\n'
              f'Med. Nsurplus: {med_nsurplus:.4f}\n'
              f'Avg. reward: {avg_rew:.4f}\n'
              f'Avg. profit: {avg_profit:.4f}\n'
              f'Avg. NUE: {avg_nue:.4f}\n'
              f'Avg. WSO: {avg_wso:.4f}\n'
              f'Avg. Nsurplus: {avg_nsurplus:.4f}\n'
              f'Action weeks: {acts}')

    # #measuring history
    # for year in eval_year:
    #     for loc in eval_locations:
    #         for var in env.unwrapped.envs[0].unwrapped.po_features:
    #             fig, ax = plt.subplots(figsize=(12, 6))
    #             measure_history_histogram(data=env.unwrapped.envs[0].unwrapped.measure_features.measure_freq,
    #                                       crop_var=var, location=loc, year=year, axes=ax)
    #             plt.tight_layout()
    #             if not os.path.exists(os.path.join(rootdir, "plots", args.checkpoint_path,)):
    #                 os.makedirs(os.path.join(rootdir, "plots", args.checkpoint_path))
    #             plt.savefig(os.path.join(rootdir, "plots", args.checkpoint_path, f"{var}_{loc}_{year}.jpeg"))
    #             writer.add_figure(f'figures/{var}_{loc}_{year}', fig)
    #             plt.close()

    if pcse_model:
        variables = ['DVS', 'action', 'WSO', 'reward',
                     'fertilizer', 'val', 'IDWST', 'prob_measure',
                     'NLOSSCUM', 'WC', 'Ndemand', 'NAVAIL', 'NuptakeTotal',
                     'SM', 'TAGP', 'LAI', 'NO3', 'NH4']
        if isinstance(env, VecEnv):
            if env.unwrapped.envs[0].unwrapped.po_features: variables.append('measure')
    else:
        variables = ['action', 'WSO', 'reward', 'TNSOIL', 'val']
        if isinstance(env, VecEnv):
            if env.unwrapped.envs[0].unwrapped.po_features: variables.append('measure')

    if 'measure' in variables:
        variables.remove('measure')
        for variable in env.unwrapped.envs[0].unwrapped.po_features:
            variable = 'measure_' + variable
            variables += [variable]

    keys_figure = [(a, b) for a in eval_year for b in eval_locations]
    results_figure = {filter_key: result_model[filter_key] for filter_key in keys_figure}

    print(keys_figure)
    name = args.agent + (str(args.checkpoint_path) if args.agent == 'LagPPO' else '')
    # pickle info for creating figures and evaluation
    if init_n is None and soil is None:
        with open(os.path.join(checkpoint_folder, f'infos_{name}.pkl'), 'wb') as f:
            pickle.dump(results_figure, f)
    elif init_n is not None and soil is None:
        os.makedirs(os.path.join(checkpoint_folder, init_n), exist_ok=True)
        with open(os.path.join(checkpoint_folder, init_n, f'infos_{name}.pkl'), 'wb') as f:
            pickle.dump(results_figure, f)
    elif init_n is None and soil is not None:
        os.makedirs(os.path.join(checkpoint_folder, soil), exist_ok=True)
        with open(os.path.join(checkpoint_folder, soil, f'infos_{name}.pkl'), 'wb') as f:
            pickle.dump(results_figure, f)

    for i, variable in enumerate(variables):
        if variable not in results_figure[list(results_figure.keys())[0]][0].keys():
            continue
        plot_individual = False
        if plot_individual:
            fig, ax = plt.subplots()
            eval.plot_variable(results_figure, variable=variable, ax=ax, ylim=eval.get_ylim_dict()[variable])
            writer.add_figure(f'figures/{variable}', fig)
            plt.close()

        fig, ax = plt.subplots()
        eval.plot_variable(results_figure, variable=variable, ax=ax, ylim=eval.get_ylim_dict()[variable],
                           plot_average=True)
        writer.add_figure(f'figures/avg-{variable}', fig)
        plt.close()
