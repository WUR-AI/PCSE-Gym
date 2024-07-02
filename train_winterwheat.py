import argparse
import lib_programname
import sys
import os.path
from datetime import datetime
import time
import json

from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger
import torch.nn as nn
import torch

import gymnasium.spaces
from gymnasium.envs.registration import register
import gymnasium as gym

from pcse_gym.envs.constraints import ActionConstrainer
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import get_policy_kwargs, get_model_kwargs
from pcse_gym.utils.eval import EvalCallback, determine_and_log_optimum
from pcse_gym.utils.normalization import VecNormalizePO
import pcse_gym.utils.defaults as defaults

path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[0]

if rootdir not in sys.path:
    print(f'insert {os.path.join(rootdir)}')
    sys.path.insert(0, os.path.join(rootdir))

if os.path.join(rootdir, "pcse_gym") not in sys.path:
    sys.path.insert(0, os.path.join(rootdir, "pcse_gym"))


def args_func(parser):
    parser.add_argument("-s", "--seed", type=int, default=0, help="Set seed")
    parser.add_argument("-n", "--nsteps", type=int, default=400000, help="Number of steps")
    parser.add_argument("-c", "--costs-nitrogen", type=float, default=10.0, help="Costs for nitrogen")
    parser.add_argument("-p", "--multiprocess", action='store_true', dest='multiproc',
                        help="Use stable-baselines3 multiprocessing")
    parser.add_argument("--nenvs", type=int, default=4, help="Number of parallel envs")
    parser.add_argument("--eval-freq", type=int, default=20_000, dest='eval_freq')
    parser.add_argument("--no-comet", action='store_false', dest='comet')
    parser.add_argument('-d', "--device", type=str, default="cpu")
    parser.add_argument("-e", "--environment", type=int, default=2,
                        help="Crop growth model. 0 for LINTUL-3, 1 for WOFOST Classic N, 2 for WOFOST SNOMIN N")
    parser.add_argument("-a", "--agent", type=str, default="PPO", help="RL agent. PPO, RPPO, GRU,"
                                                                       "IndRNN, DiffNC, PosMLP, ATM or DQN")
    parser.add_argument("-r", "--reward", type=str, default="DEF",
                        help="Reward function. DEF, DEP, GRO, END, NUE or ANE")
    parser.add_argument("-b", "--n-budget", type=int, default=0, help="Nitrogen budget. kg/ha")
    parser.add_argument("--action_limit", type=int, default=0, help="Limit fertilization frequency."
                                                                    "Recommended 4 times")
    parser.add_argument("-m", "--measure", action='store_true', help="--measure or --no-measure."
                                                                     "Train an agent in a partially"
                                                                     "observable environment that"
                                                                     "decides when to measure"
                                                                     "certain crop features")
    parser.add_argument("-l", "--location", type=str, default="NL", help="location to train the agent. NL or LT.")
    parser.add_argument("--random-init", action='store_true', dest='random_init')
    parser.add_argument("--random-weather", action='store_true', dest='random_weather')
    parser.add_argument("--no-measure", action='store_false', dest='measure')
    parser.add_argument("--noisy-measure", action='store_true', dest='noisy_measure')
    parser.add_argument("--variable-recovery-rate", action='store_true', dest='vrr')
    parser.add_argument("--no-weather", action='store_true', dest='no_weather')
    parser.add_argument("--random-feature", action='store_true', dest='random_feature')
    parser.add_argument("--mask-binary", action='store_true', dest='obs_mask')
    parser.add_argument("--placeholder-0", action='store_const', const=0.0, dest='placeholder_val')
    parser.add_argument("--normalize", action='store_true', dest='normalize')
    parser.add_argument("--cost-measure", type=str, default='real', dest='cost_measure', help='real, no, or same')
    parser.add_argument("--start-type", type=str, default='sowing', dest='start_type', help='sowing or emergence')
    parser.add_argument("--measure-cost-multiplier", type=int, default=1, dest='m_multiplier',
                        help="multiplier for the measuring cost")
    parser.add_argument("--measure-all", action='store_true', dest='measure_all')
    parser.add_argument("--overfit", action='store_true', dest='overfit')
    parser.add_argument("--year", type=int, default=None, dest='year')
    parser.set_defaults(measure=False, vrr=False, noisy_measure=False, framework='sb3',
                        no_weather=False, random_feature=False, obs_mask=False, placeholder_val=-1.11,
                        normalize=False, random_init=False, m_multiplier=1, measure_all=False, random_weather=False,
                        multiproc=False, comet=True, overfit=False)

    args = parser.parse_args()

    return args


def wrapper_vectorized_env(env_pcse_train, flag_po, flag_eval=False, multiproc=False, n_envs=4, normalize=False):
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
    if normalize:
        return DummyVecEnv([lambda: env_pcse_train])
    if flag_po:
        return VecNormalizePO(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                              clip_obs=10., clip_reward=50., gamma=1)
    if multiproc and not flag_eval:
        vec_env = SubprocVecEnv([lambda: env_pcse_train for _ in range(n_envs)])
        return VecNormalize(vec_env)
    else:
        return VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                            clip_obs=10., clip_reward=50., gamma=1)


def get_hyperparams(agent, pcse_env, no_weather, flag_po, mask_binary):
    if agent == 'PPO':
        hyperparams = {'batch_size': 64, 'n_steps': 2048, 'learning_rate': 0.0002, 'ent_coef': 0.0,
                       'clip_range': 0.3,
                       'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5, 'vf_coef': 0.4,
                       'policy_kwargs': {},
                       }
        if not no_weather:
            hyperparams['policy_kwargs'] = get_policy_kwargs(n_crop_features=len(crop_features),
                                                             n_weather_features=len(weather_features),
                                                             n_action_features=len(action_features),
                                                             n_po_features=len(po_features) if flag_po else 0,
                                                             mask_binary=mask_binary)
        hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
        hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh
        hyperparams['policy_kwargs']['ortho_init'] = False
    if agent == 'RPPO':
        hyperparams = {'batch_size': 128, 'n_steps': 2048, 'learning_rate': 0.0002, 'ent_coef': 0.0,
                       'clip_range': 0.2,
                       'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5, 'vf_coef': 0.4,
                       'policy_kwargs': {},
                       }
        if not no_weather:
            hyperparams['policy_kwargs'] = get_policy_kwargs(n_crop_features=len(crop_features),
                                                             n_weather_features=len(weather_features),
                                                             n_action_features=len(action_features),
                                                             n_po_features=len(po_features) if flag_po else 0,
                                                             mask_binary=mask_binary)
        hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
        hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh
        hyperparams['policy_kwargs']['ortho_init'] = False
    if agent == 'A2C':
        hyperparams = {'n_steps': 2048, 'learning_rate': 0.0002, 'ent_coef': 0.0,
                       'gae_lambda': 0.9, 'vf_coef': 0.4,  # 'rms_prop_eps': 1e-5, 'normalize_advantage': True,
                       'policy_kwargs': {},
                       }
        if not no_weather:
            hyperparams['policy_kwargs'] = get_policy_kwargs(n_crop_features=len(crop_features),
                                                             n_weather_features=len(weather_features),
                                                             n_action_features=len(action_features))
        hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
        hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh
        hyperparams['policy_kwargs']['ortho_init'] = False
        # hyperparams['policy_kwargs']['optimizer_class'] = RMSpropTFLike
        # hyperparams['policy_kwargs']['optimizer_kwargs'] = dict(eps=0.00001)
    if agent == 'DQN':
        hyperparams = {'exploration_fraction': 0.3, 'exploration_initial_eps': 1.0,
                       'exploration_final_eps': 0.001,
                       'policy_kwargs': get_policy_kwargs(n_crop_features=len(crop_features),
                                                          n_weather_features=len(weather_features),
                                                          n_action_features=len(action_features))
                       }
        hyperparams['policy_kwargs']['net_arch'] = [256, 256]
        hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh

    return hyperparams


def get_json_config(n_steps, crop_features, weather_features, train_years, test_years, train_locations, test_locations,
                    action_space, pcse_model, agent, reward, seed, costs_nitrogen, kwargs):
    return dict(
        n_steps=n_steps, crop_features=crop_features, weather_features=weather_features,
        train_years=train_years, test_years=test_years, train_locations=train_locations,
        test_locations=test_locations, action_space=action_space, pcse_model=pcse_model, agent=agent, reward=reward,
        seed=seed, costs_nitrogen=costs_nitrogen, kwargs=kwargs
    )


def train(log_dir, n_steps,
          crop_features=defaults.get_wofost_default_crop_features(2),
          weather_features=defaults.get_default_weather_features(),
          action_features=defaults.get_default_action_features(),
          train_years=defaults.get_default_train_years(),
          test_years=defaults.get_default_test_years(),
          train_locations=defaults.get_default_location(),
          test_locations=defaults.get_default_location(),
          action_space=defaults.get_default_action_space(),
          pcse_model=0, agent='PPO', reward=None,
          seed=0, tag="Exp", costs_nitrogen=10.0,
          multiprocess=False, eval_freq=20_000,
          **kwargs):
    """
    Train a PPO agent (Stable Baselines3).

    Parameters
    ----------
    :param log_dir: directory where the (tensorboard) data will be saved
    :param n_steps: int, number of timesteps the agent spends in the environment
    :param crop_features: crop features
    :param weather_features: weather features
    :param action_features: action features
    :param train_years: train years
    :param test_years: test years
    :param train_locations: train locations (latitude,longitude)
    :param test_locations: test locations (latitude,longitude)
    :param action_space: action space
    :param pcse_model: 0 for LINTUL else WOFOST
    :param agent: one of {PPO, RPPO, DQN}
    :param reward: one of {DEF, GRO, or ANE}
    :param seed: random seed
    :param tag: tag for tensorboard and friends
    :param costs_nitrogen: float, penalty for fertilization application
    :param multiprocess:

    """
    if agent == 'DQN':
        assert not kwargs.get('args_measure'), f'cannot use {agent} with measure args'

    pcse_model_name = "LINTUL" if not pcse_model else "WOFOST"

    action_limit = kwargs.get('action_limit', 0)
    flag_po = kwargs.get('po_features', [])
    n_budget = kwargs.get('n_budget', 0)
    framework = kwargs.get('framework', 'sb3')
    no_weather = kwargs.get('no_weather', False)
    normalize = kwargs.get('normalize', False)
    mask_binary = kwargs.get('mask_binary', False)
    loc_code = kwargs.get('loc_code', None)
    random_init = kwargs.get('random_init', False)
    cost_measure = kwargs.get('cost_measure', None)
    measure_all = kwargs.get('measure_all', None)
    n_envs = kwargs.get('n_envs', 4)

    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

    print('Using the StableBaselines3 framework')
    print(f'Train model {pcse_model_name} with {agent} algorithm and seed {seed}. Logdir: {log_dir}')

    hyperparams = get_hyperparams(agent, pcse_model, no_weather, flag_po, mask_binary)

    # TODO register env initialization for robustness
    # register_cropgym_env = register_cropgym_envs()

    env_pcse_train = WinterWheat(crop_features=crop_features, action_features=action_features,
                                 weather_features=weather_features,
                                 costs_nitrogen=costs_nitrogen, years=train_years, locations=train_locations,
                                 action_space=action_space, action_multiplier=1.0, seed=seed,
                                 reward=reward, **get_model_kwargs(pcse_model, train_locations,
                                                                   start_type=kwargs.get('start_type', 'sowing')),
                                 **kwargs)

    env_pcse_train = Monitor(env_pcse_train)

    env_pcse_train = ActionConstrainer(env_pcse_train, action_limit=action_limit, n_budget=n_budget)

    device = kwargs.get('device')
    if device == 'cuda':
        print('CUDA not available... Using CPU!') if not torch.cuda.is_available() else print('using CUDA!')
    else:
        print('Using CPU!')

    if agent == 'PPO':
        env_pcse_train = wrapper_vectorized_env(env_pcse_train, flag_po,
                                                multiproc=multiprocess, normalize=normalize, n_envs=n_envs)
        model = PPO('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                    tensorboard_log=log_dir, device=device)
    elif agent == 'DQN':
        env_pcse_train = wrapper_vectorized_env(env_pcse_train, flag_po,
                                                multiproc=multiprocess, normalize=normalize, n_envs=n_envs)
        model = DQN('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                    tensorboard_log=log_dir, device=device)
    elif agent == 'A2C':
        env_pcse_train = wrapper_vectorized_env(env_pcse_train, flag_po,
                                                multiproc=multiprocess, normalize=normalize, n_envs=n_envs)
        model = A2C('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                    tensorboard_log=log_dir, device=device)
    elif agent == 'RPPO':
        from sb3_contrib import RecurrentPPO
        env_pcse_train = wrapper_vectorized_env(env_pcse_train, flag_po,
                                                multiproc=multiprocess, normalize=normalize, n_envs=n_envs)
        model = RecurrentPPO('MlpLstmPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                             tensorboard_log=log_dir, device=device)

    # wrap comet after VecEnvs
    comet_log = None
    use_comet = kwargs.get('comet', True)
    if use_comet:
        with open(os.path.join(rootdir, 'comet', 'comet_key'), 'r') as f:
            api_key = f.readline()
        comet_log = Experiment(
            api_key=api_key,
            project_name="cropGym_nue_paper",
            workspace="pcse-gym",
            log_code=True,
            log_graph=True,
            auto_metric_logging=True,
            auto_histogram_tensorboard_logging=True
        )
        comet_log.log_code(folder=os.path.join(rootdir, 'pcse_gym'))
        comet_log.log_parameters(hyperparams)

        env_pcse_train = CometLogger(env_pcse_train, comet_log)
        if pcse_model == 0:
            tag_env = "LINTUL3"
        elif pcse_model == 1:
            tag_env = "WOFOST Classic"
        else:
            tag_env = "WOFOST SNOMIN"
        comet_log.add_tags(['sb3', agent, seed, loc_code, reward, tag_env])

        print('Using Comet!')

    compute_baselines = False
    if compute_baselines:
        determine_and_log_optimum(log_dir, env_pcse_train,
                                  train_years=train_years, test_years=test_years,
                                  train_locations=train_locations, test_locations=test_locations,
                                  n_steps=args.nsteps)

    env_pcse_eval = WinterWheat(crop_features=crop_features, action_features=action_features,
                                weather_features=weather_features,
                                costs_nitrogen=costs_nitrogen, years=test_years, locations=test_locations,
                                action_space=action_space, action_multiplier=1.0, reward=reward,
                                **get_model_kwargs(pcse_model, train_locations,
                                                   start_type=kwargs.get('start_type', 'sowing')),
                                **kwargs, seed=seed)
    if action_limit or n_budget > 0:
        env_pcse_eval = ActionConstrainer(env_pcse_eval, action_limit=action_limit, n_budget=n_budget)

    env_pcse_eval = wrapper_vectorized_env(env_pcse_eval, flag_po,
                                           multiproc=multiprocess, normalize=normalize, flag_eval=True)

    if measure_all:
        cost_measure = 'all'
    tb_log_name = f'{tag}-nsteps-{n_steps}-{agent}-{reward}'
    if use_comet:
        comet_log.set_name(f'{tag}-{agent}-{reward}')
        comet_log.add_tag(cost_measure)
    tb_log_name = tb_log_name + '-run'

    start_time = time.time()
    print(f"Time started training: {start_time}")

    model.learn(total_timesteps=n_steps,
                callback=EvalCallback(env_eval=env_pcse_eval, test_years=test_years,
                                      train_years=train_years, train_locations=train_locations,
                                      test_locations=test_locations, seed=seed, pcse_model=pcse_model,
                                      comet_experiment=comet_log, multiprocess=multiprocess, eval_freq=eval_freq,
                                      **kwargs),
                tb_log_name=tb_log_name)

    print(f'Time taken to train {tb_log_name}: {time.time() - start_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = args_func(parser)

    # making sure everything is compatible with the user choices
    if not args.measure and args.noisy_measure:
        parser.error("noisy measure should be used with measure")
    if args.agent not in ['PPO', 'A2C', 'RPPO', 'DQN', 'GRU', 'PosMLP', 'S4D', 'IndRNN', 'DiffNC', 'ATM']:
        parser.error("Invalid agent argument. Please choose PPO, A2C, RPPO, GRU, IndRNN, DiffNC, PosMLP, ATM, DQN")
    if args.agent in ['GRU', 'PosMLP', 'S4D', 'IndRNN', 'DiffNC']:
        args.framework = 'rllib'
    elif args.agent in ['ATM']:
        args.framework = 'ACNO-MDP'
    pcse_model_name = "LINTUL" if not args.environment else "WOFOST"

    # directory where the model is saved
    print(rootdir)
    log_dir = os.path.join(rootdir, 'tensorboard_logs', f'{pcse_model_name}_experiments')
    print(f'train for {args.nsteps} steps with costs_nitrogen={args.costs_nitrogen} (seed={args.seed})')

    # random weather
    # TODO tidy up
    if args.random_weather and not args.overfit:
        train_locations = [(52.57, 5.63), (52.5, 5.5)]
        test_locations = [(52.57, 5.63)]
        train_years = [*range(4000, 5999)]
        test_years = [*range(1983, 2016)]
    elif args.random_weather and args.overfit:
        train_locations = [(52.57, 5.63)]
        test_locations = [(52.57, 5.63)]
        if args.year is not None:
            train_years = [args.year]
            test_years = [args.year]
        else:
            train_years = [1990]
            test_years = [1990]
    else:
        # define training and testing years
        all_years = [*range(1990, 2022)]
        train_years = [year for year in all_years if year % 2 == 1]
        test_years = [year for year in all_years if year % 2 == 0]

        # define training and testing locations
        if args.location == "NL":
            """The Netherlands"""
            train_locations = [(52, 5.5), (51.5, 5), (52.5, 6.0)]
            test_locations = [(52, 5.5), (48, 0)]
        elif args.location == "LT":
            """Lithuania"""
            train_locations = [(55.0, 23.5), (55.0, 24.0), (55.5, 23.5)]
            test_locations = [(46.0, 25.0), (55.0, 23.5)]  # Romania
        else:
            parser.error("--location arg should be either LT or NL")

    # define the crop, weather and (maybe) action features used in training
    crop_features = defaults.get_default_crop_features(pcse_env=args.environment, minimal=False)
    weather_features = defaults.get_default_weather_features()
    action_features = defaults.get_default_action_features()

    tag = f'Seed-{args.seed}'

    # define key word arguments
    kwargs = {'args_vrr': args.vrr, 'action_limit': args.action_limit, 'noisy_measure': args.noisy_measure,
              'n_budget': args.n_budget, 'framework': args.framework, 'no_weather': args.no_weather,
              'mask_binary': args.obs_mask, 'placeholder_val': args.placeholder_val, 'normalize': args.normalize,
              'loc_code': args.location, 'cost_measure': args.cost_measure, 'start_type': args.start_type,
              'random_init': args.random_init, 'm_multiplier': args.m_multiplier, 'measure_all': args.measure_all,
              'random_weather': args.random_weather, 'comet': args.comet, 'n_envs': args.nenvs}

    assert not (args.environment == 2 and args.measure is True), "WOFOST SNOMIN doesn't support AFA-POMDPs yet"
    # define MeasureOrNot environment if specified
    if not args.measure:
        action_spaces = gymnasium.spaces.Discrete(9)  # 9 levels of fertilizing
    else:
        if args.environment == 1:
            po_features = ['TAGP', 'LAI', 'NAVAIL', 'NuptakeTotal', 'SM']
            if args.random_feature:
                po_features.append('random')
                crop_features.append('random')
        else:
            po_features = ['TGROWTH', 'LAI', 'TNSOIL', 'NUPTT', 'TRAIN']
        kwargs['po_features'] = po_features
        kwargs['args_measure'] = True
        if not args.noisy_measure:
            m_shape = 2
        else:
            m_shape = 3
        if args.measure_all:
            a_shape = [7] + [m_shape]
        else:
            a_shape = [7] + [m_shape] * len(po_features)
        action_spaces = gymnasium.spaces.MultiDiscrete(a_shape)

    print(f"Train and test years are {train_years} and {test_years},"
          f"with train location of {train_locations} and test location of {test_locations}")

    train(log_dir, train_years=train_years, test_years=test_years,
          train_locations=train_locations,
          test_locations=test_locations,
          n_steps=args.nsteps, seed=args.seed,
          tag=tag, costs_nitrogen=args.costs_nitrogen,
          crop_features=crop_features,
          weather_features=weather_features,
          action_features=action_features, action_space=action_spaces,
          pcse_model=args.environment, agent=args.agent,
          reward=args.reward, multiprocess=args.multiproc, **kwargs,
          eval_freq=args.eval_freq, device=args.device)
