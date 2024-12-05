import gymnasium as gym

from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import get_model_kwargs, StableBaselinesWrapper
import pcse_gym.utils.defaults as defaults
from pcse_gym.envs.constraints import ActionConstrainer


def get_po_features(pcse_env=1):
    if pcse_env:
        po_features = ['TAGP', 'LAI', 'NAVAIL', 'SM', 'NuptakeTotal']
    else:
        po_features = ['TGROWTH', 'LAI', 'TNSOIL', 'NUPTT', 'TRAIN']
    return po_features


def get_action_space(nitrogen_levels=7, po_features=[], measure_all=False):
    if po_features:
        if not measure_all:
            a_shape = [nitrogen_levels] + [2] * len(po_features)
        else:
            a_shape = [nitrogen_levels] + [2]
        space_return = gym.spaces.MultiDiscrete(a_shape)
    else:
        space_return = gym.spaces.Discrete(nitrogen_levels)
    return space_return


def initialize_env(pcse_env=1, po_features=[],
                   crop_features=defaults.get_default_crop_features(pcse_env=1, vision=None),
                   costs_nitrogen=10, reward='DEF', nitrogen_levels=7, action_multiplier=1.0, add_random=False,
                   years=defaults.get_default_train_years(), locations=defaults.get_default_location(), args_vrr=False,
                   action_limit=0, noisy_measure=False, n_budget=0, no_weather=False, mask_binary=False,
                   placeholder_val=-1.11, normalize=False, loc_code='NL', cost_measure='real', start_type='emergence',
                   random_init=False, m_multiplier=1, measure_all=False, sb3wrapper=False, seed=None):
    if add_random:
        po_features.append('random'), crop_features.append('random')
    action_space = get_action_space(nitrogen_levels=nitrogen_levels, po_features=po_features, measure_all=measure_all)
    kwargs = dict(po_features=po_features, args_measure=po_features is not None, args_vrr=args_vrr,
                  action_limit=action_limit, noisy_measure=noisy_measure, n_budget=n_budget, no_weather=no_weather,
                  mask_binary=mask_binary, placeholder_val=placeholder_val, normalize=normalize, loc_code=loc_code,
                  cost_measure=cost_measure, start_type=start_type, random_init=random_init, m_multiplier=m_multiplier,
                  measure_all=measure_all)
    if not sb3wrapper:
        env_return = WinterWheat(crop_features=crop_features,
                                 costs_nitrogen=costs_nitrogen,
                                 years=years,
                                 locations=locations,
                                 action_space=action_space,
                                 action_multiplier=action_multiplier,
                                 reward=reward,
                                 **get_model_kwargs(pcse_env, locations, start_type=start_type),
                                 **kwargs)
    else:
        env_return = StableBaselinesWrapper(crop_features=crop_features,
                                            costs_nitrogen=costs_nitrogen,
                                            years=years[0],
                                            locations=locations[0],
                                            action_space=action_space,
                                            action_multiplier=action_multiplier, seed=seed, **kwargs)

    return env_return


def initialize_env_po():
    return initialize_env(po_features=get_po_features(), placeholder_val=0.0, pcse_env=1)


def initialize_env_po_noisy():
    return initialize_env(po_features=get_po_features(), noisy_measure=True, pcse_env=1)


def initialize_env_no_baseline():
    return initialize_env(reward='GRO', action_multiplier=2.0, pcse_env=2)


def initialize_env_random():
    return initialize_env(po_features=get_po_features(), add_random=True, pcse_env=1)


def initialize_env_action_limit_no_measure(limit):
    env = initialize_env(action_limit=limit)
    env = ActionConstrainer(env, action_limit=limit)
    return env


def initialize_env_action_limit_measure(limit):
    env = initialize_env(po_features=get_po_features(), action_limit=limit)
    env = ActionConstrainer(env, action_limit=limit)
    return env


def initialize_env_action_limit_budget_no_measure(limit, budget):
    env = initialize_env(action_limit=limit, n_budget=budget)
    env = ActionConstrainer(env, action_limit=limit, n_budget=budget)
    return env


def initialize_env_action_limit_budget_measure(limit, budget):
    env = initialize_env(po_features=get_po_features(), action_limit=limit, n_budget=budget)
    env = ActionConstrainer(env, action_limit=limit, n_budget=budget)
    return env


def initialize_env_reward_dep():
    return initialize_env(reward='DEP', args_vrr=True, pcse_env=2)


def initialize_env_reward_ane():
    return initialize_env(reward='ANE', pcse_env=2)


def initialize_env_measure_po_extend():
    return initialize_env(po_features=get_po_features(), mask_binary=True, no_weather=True)


def initialize_env_measure_po_normalize():
    return initialize_env(po_features=get_po_features(), normalize=True, no_weather=True)


def initialize_env_measure_po_normalize_extend():
    return initialize_env(po_features=get_po_features(), normalize=True, mask_binary=True, no_weather=True,
                          start_type='emergence')


def initialize_env_measure_no_cost():
    return initialize_env(po_features=get_po_features(), cost_measure='no')


def initialize_env_measure_same_cost():
    return initialize_env(po_features=get_po_features(), cost_measure='same')


def initialize_env_sow():
    return initialize_env(start_type='sowing')


def initialize_env_emergence():
    return initialize_env(start_type='emergence')


def initialize_env_random_init():
    return initialize_env(random_init=True, pcse_env=2)


def initialize_env_multiplier():
    return initialize_env(po_features=get_po_features(), m_multiplier=10)


def initialize_env_non_selective():
    return initialize_env(po_features=get_po_features(), measure_all=True)


def initialize_env_end_reward():
    return initialize_env(reward='END', pcse_env=2)


def initialize_env_eny_reward():
    return initialize_env(reward='ENY', pcse_env=2)


def initialize_env_nue_reward():
    return initialize_env(reward='NUE', pcse_env=2)


def initialize_env_nup_reward():
    return initialize_env(reward='NUP', pcse_env=2)


def initialize_env_har_reward():
    return initialize_env(reward='HAR', pcse_env=2)


def initialize_env_dnu_reward():
    return initialize_env(reward='DNU', pcse_env=2)


def initialize_env_fin_reward():
    return initialize_env(reward='FIN', pcse_env=2)
