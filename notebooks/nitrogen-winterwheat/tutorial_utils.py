"""Utilities for the Advanced Machine Learning CropGym tutorial notebook."""

import functools as ft
import itertools
import os
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pcse_gym.utils.defaults as defaults
from pcse_gym.envs.sb3 import get_model_kwargs
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.utils.eval import FindOptimum, evaluate_policy, save_results


def get_repo_root():
    return Path(__file__).resolve().parents[2]


def get_results_dir():
    return Path(__file__).resolve().parent / "results"


def get_config_dir():
    return get_repo_root() / "pcse_gym" / "envs" / "configs"


def make_env(years, locations, nitrogen_levels=3, action_multiplier=2.0):
    return WinterWheat(
        crop_features=defaults.get_default_crop_features(pcse_env=0),
        action_features=defaults.get_default_action_features(),
        weather_features=defaults.get_default_weather_features(),
        costs_nitrogen=10.0,
        years=years,
        locations=locations,
        action_space=gym.spaces.Discrete(nitrogen_levels),
        action_multiplier=action_multiplier,
        reward="DEF",
        **get_model_kwargs(0),
    )


def evaluate_baseline(policy, years, locations, amount=1.0):
    results = {}
    for year in years:
        for location in locations:
            env = make_env(year, location)
            env_vec = VecNormalize(
                DummyVecEnv([lambda: env]),
                norm_reward=True,
                clip_reward=50.0,
                gamma=1,
            )
            env_vec.training = False
            env_vec.norm_reward = True
            _, episode_infos = evaluate_policy(policy, env_vec, amount=amount)
            results[(year, location)] = episode_infos
    return results


def generate_plot_results(
    test_years=None,
    test_locations=None,
    model_path=None,
    stats_path=None,
    output_path=None,
):
    test_years = test_years or defaults.get_default_test_years()
    test_locations = test_locations or [(52, 5.5), (48, 0)]
    repo_root = get_repo_root()
    model_path = model_path or repo_root / "tests" / "model-1"
    stats_path = stats_path or repo_root / "tests" / "model-1.pkl"
    output_path = output_path or get_results_dir() / "results_RL.pickle"

    custom_objects = {
        "lr_schedule": lambda x: 0.0002,
        "clip_range": lambda x: 0.3,
        "action_space": gym.spaces.Discrete(3),
    }
    model = PPO.load(str(model_path), custom_objects=custom_objects, device="cpu")

    plot_results = {}
    for year in test_years:
        for location in test_locations:
            env = make_env(year, location)
            env_vec = DummyVecEnv([lambda: env])
            env_vec = VecNormalize.load(str(stats_path), env_vec)
            env_vec.training = False
            env_vec.norm_reward = True
            _, episode_infos = evaluate_policy(model, env_vec, amount=1)
            plot_results[(year, location)] = episode_infos

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(plot_results, handle)
    return plot_results


def ensure_tutorial_results(test_years=None, test_locations=None):
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = results_dir / "results_RL.pickle"
    if not pickle_path.exists():
        generate_plot_results(test_years=test_years, test_locations=test_locations)

    required_csvs = ["fixed.csv", "upperbound.csv", "model-1.csv"]
    if not all((results_dir / name).exists() for name in required_csvs):
        _generate_csv_results(results_dir, test_years, test_locations)

    return results_dir


def _generate_csv_results(results_dir, test_years=None, test_locations=None):
    test_years = test_years or defaults.get_default_test_years()
    test_locations = test_locations or [(52, 5.5), (48, 0)]

    sp_results = evaluate_baseline("standard-practice", test_years, test_locations)
    save_results(sp_results, results_dir / "fixed.csv")

    env = make_env(test_years[0], test_locations[0])
    env_vec = VecNormalize(
        DummyVecEnv([lambda: env]), norm_reward=True, clip_reward=50.0, gamma=1
    )
    optimum = FindOptimum(env_vec, test_years).optimize_start_dump().item()
    ceres_results = evaluate_baseline("start-dump", test_years, test_locations, amount=optimum)
    save_results(ceres_results, results_dir / "upperbound.csv")

    repo_root = get_repo_root()
    model = PPO.load(
        str(repo_root / "tests" / "model-1"),
        custom_objects={
            "lr_schedule": lambda x: 0.0002,
            "clip_range": lambda x: 0.3,
            "action_space": gym.spaces.Discrete(3),
        },
        device="cpu",
    )
    rl_results = {}
    for year in test_years:
        for location in test_locations:
            env = make_env(year, location)
            env_vec = DummyVecEnv([lambda: env])
            env_vec = VecNormalize.load(str(repo_root / "tests" / "model-1.pkl"), env_vec)
            env_vec.training = False
            env_vec.norm_reward = True
            _, episode_infos = evaluate_policy(model, env_vec, amount=1)
            rl_results[(year, location)] = episode_infos
    save_results(rl_results, results_dir / "model-1.csv")


def read_data(resultsdir=None, csv_models=None):
    resultsdir = Path(resultsdir) if resultsdir else ensure_tutorial_results()
    csv_models = csv_models or {
        "baseline": "fixed.csv",
        "upperbound": "upperbound.csv",
        "model": ["model-1.csv"],
    }

    df_sp = pd.read_csv(resultsdir / csv_models["baseline"])
    df_ceres = pd.read_csv(resultsdir / csv_models["upperbound"])
    df_rl = [pd.read_csv(resultsdir / csv_model) for csv_model in csv_models["model"]]

    dfs = list(itertools.chain([df_sp, df_ceres, *df_rl]))
    suffix = list(itertools.chain(["_SP", "_Ceres", *[f"_RL_{i}" for i in range(len(df_rl))]]))
    duplicate_cols = ["TMIN", "TMAX", "IRRAD", "RAIN", "year", "location"]

    for i, df in enumerate(dfs):
        dfs[i].columns = [
            str(col) if col in duplicate_cols else str(col) + suffix[i] for col in df.columns
        ]
        if i > 0:
            dfs[i].drop(columns=duplicate_cols, inplace=True)
    df_merged = ft.reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="outer"),
        dfs,
    )

    df_merged["rain"] = 10.0 * df_merged["RAIN"]
    filter_col = [col for col in df_merged if col.startswith("WSO")]
    suffices = [c.split("_", 1)[1] for c in filter_col]
    for suffix_name in suffices:
        df_merged[f"fertilizer_{suffix_name}"] = 10.0 * df_merged[f"fertilizer_{suffix_name}"]
        df_merged[f"WSO_{suffix_name}"] = 0.01 * df_merged[f"WSO_{suffix_name}"]
        df_merged[f"nitrogen_{suffix_name}"] = 10.0 * df_merged[f"fertilizer_{suffix_name}"]

    return df_merged
