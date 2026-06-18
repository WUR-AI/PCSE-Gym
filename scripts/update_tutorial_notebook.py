"""Regenerate the Advanced ML CropGym tutorial for the current codebase."""

import json
from pathlib import Path


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    md(
        "# Advanced Machine Learning course: CropGym tutorial\n"
        "## Nitrogen optimization on the winter wheat crop\n"
        "\n"
        "CropGym is a WUR-made reinforcement learning platform that trains an agent to optimize "
        "the usage of nitrogen fertilizer on crops. The environment is built on PCSE and the "
        "LINTUL-3 crop growth model.\n"
        "\n"
        "This notebook can be run locally or on [Google Colab](https://colab.research.google.com/). "
        "Paper results are generated on first run if they are not already present."
    ),
    md("### Install required packages and import required libraries"),
    code(
        "# Colab setup (skip when running locally with poetry install --extras sb-integration)\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "IN_COLAB = 'google.colab' in sys.modules\n"
        "if IN_COLAB:\n"
        "    !rm -fr /content/PCSE-Gym && git clone https://github.com/WUR-AI/PCSE-Gym.git /content/PCSE-Gym\n"
        "    %cd /content/PCSE-Gym\n"
        "    !pip install -q pcse>=6.0.13 gymnasium pyyaml numpy stable-baselines3 sb3-contrib \\\n"
        "        matplotlib torch tensorboard scipy tqdm pandas seaborn scikit-learn\n"
        "    repo_root = Path('/content/PCSE-Gym')\n"
        "else:\n"
        "    repo_root = Path.cwd()\n"
        "    if not (repo_root / 'pcse_gym').exists():\n"
        "        repo_root = repo_root.parent\n"
        "\n"
        "sys.path.insert(0, str(repo_root))\n"
        "sys.path.insert(0, str(repo_root / 'notebooks' / 'nitrogen-winterwheat'))"
    ),
    code(
        "import os\n"
        "import random\n"
        "import matplotlib\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import gymnasium as gym\n"
        "\n"
        "from pcse_gym.envs.sb3 import get_config_dir, get_policy_kwargs, get_model_kwargs\n"
        "from pcse_gym.envs.winterwheat import WinterWheat\n"
        "import pcse_gym.utils.defaults as defaults\n"
        "from pcse_gym.utils.eval import (\n"
        "    EvalCallback, evaluate_policy, get_ylim_dict, identity_line,\n"
        "    plot_variable, report_ci,\n"
        ")\n"
        "from tutorial_utils import ensure_tutorial_results, get_results_dir, read_data\n"
        "\n"
        "all_years = [*range(1990, 2022)]\n"
        "train_years = [year for year in all_years if year % 2 == 1]\n"
        "test_years = [year for year in all_years if year % 2 == 0]\n"
        "train_locations = [(52, 5.5), (51.5, 5), (52.5, 6.0)]\n"
        "test_locations = [(52, 5.5), (48, 0)]\n"
        "location_to_label = {'52;5.5': 'NL', '48;0': 'FR'}\n"
        "colors = {'RL': 'tab:blue', 'SP': 'tab:orange', 'Ceres': 'red'}\n"
        "markers = {'52;5.5': 'o', '48;0': '^'}\n"
        "random.seed(42)\n"
        "\n"
        "font = {'weight': 'bold', 'size': 14}\n"
        "ax = {'titleweight': 'bold', 'titlesize': 14}\n"
        "matplotlib.rc('font', **font)\n"
        "matplotlib.rc('axes', **ax)\n"
        "sns.set_theme(font_scale=1.00)\n"
        "\n"
        "resultsdir = ensure_tutorial_results(test_years=test_years, test_locations=test_locations)\n"
        "data_dir = get_config_dir()"
    ),
    md("## Getting to know the RL Environment"),
    md(
        "To set up a proper agent, one needs to be familiar with the environment of the RL agent. "
        "In this case, the environment is the LINTUL-3 crop growth model, made available through PCSE.\n"
        "\n"
        "Using PCSE, we can look into different elements of the environment, and how a crop model is defined."
    ),
    code(
        "import pcse\n"
        "import yaml\n"
        "from pcse.input import PCSEFileReader\n"
        "from pcse.base import ParameterProvider\n"
        "\n"
        "data_dir = get_config_dir()"
    ),
    md(
        "In the cell below, we define the files used to parameterize the crop model: crop, soil, and site."
    ),
    code(
        "cropfile = os.path.join(data_dir, 'crop', 'lintul3_winterwheat.crop')\n"
        "soilfile = os.path.join(data_dir, 'soil', 'lintul3_springwheat.soil')\n"
        "sitefile = os.path.join(data_dir, 'site', 'lintul3_springwheat.site')\n"
        "\n"
        "crop = PCSEFileReader(cropfile)\n"
        "soil = PCSEFileReader(soilfile)\n"
        "site = PCSEFileReader(sitefile)\n"
        "parameterprovider = ParameterProvider(soildata=soil, cropdata=crop, sitedata=site)"
    ),
    md("Before initializing the model, we obtain weather parameters from the NASA POWER database."),
    code(
        "from pcse.input import NASAPowerWeatherDataProvider\n"
        "weatherdataprovider = NASAPowerWeatherDataProvider(latitude=52, longitude=5)\n"
        "print(weatherdataprovider)"
    ),
    md("We also load agromanagement actions from the PCSE test data."),
    code(
        "from pcse.input import YAMLAgroManagementReader\n"
        "agromanagement_file = os.path.join(\n"
        "    os.path.dirname(pcse.__file__), 'tests', 'test_data', 'lintul3_springwheat.agro'\n"
        ")\n"
        "agromanagement = YAMLAgroManagementReader(agromanagement_file)\n"
        "print(agromanagement)"
    ),
    md("Initialize and run the LINTUL-3 model with the parameters above."),
    code(
        "from pcse.models import LINTUL3\n"
        "lintul = LINTUL3(\n"
        "    parameterprovider=parameterprovider,\n"
        "    weatherdataprovider=weatherdataprovider,\n"
        "    agromanagement=agromanagement,\n"
        ")"
    ),
    code("lintul.run_till_terminate()"),
    code(
        "output = lintul.get_output()\n"
        "df = pd.DataFrame(output).set_index('day')\n"
        "df.tail()"
    ),
    md(
        "Interesting variables include **TNSOIL**, **WSO**, **LAI**, and **NUPTT**. "
        "Plot them below."
    ),
    code(
        "fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))\n"
        "df['TNSOIL'].plot(ax=axes[0][0], title='TNSOIL')\n"
        "df['NUPTT'].plot(ax=axes[0][1], title='NUPTT')\n"
        "df['LAI'].plot(ax=axes[1][0], title='LAI')\n"
        "df['WSO'].plot(ax=axes[1][1], title='WSO')\n"
        "fig.autofmt_xdate()\n"
        "fig.savefig(os.path.join(data_dir, 'lintul3_winterwheat.png'))"
    ),
    md(
        "#### TO DO 1:\n"
        "* Look at the graph of TNSOIL. What causes the spikes in the N availability?\n"
        "* Look at NUPTT. Why does the N uptake seem to crawl at a certain point?\n"
        "* Look at LAI. What do you think causes the LAI to degrade?\n"
        "* Look at WSO. What is approximately the end result of the yield?"
    ),
    md("## The RL agent in CropGym"),
    md(
        "To get familiar with the CropGym setting, we visualize results from "
        "[Kallenberg et al., 2023](https://www.cambridge.org/core/journals/environmental-data-science/article/"
        "nitrogen-management-with-reinforcement-learning-and-crop-growth-models/358749FAFAA4990B1448DAB7F48D641C).\n"
        "\n"
        "The reward compares fertilized yield growth against a zero-nitrogen baseline, minus a cost for fertilizer."
    ),
    md("The following cell plots fertilizer application and reward for test years in the Netherlands."),
    code(
        "import pickle\n"
        "\n"
        "with open(resultsdir / 'results_RL.pickle', 'rb') as handle:\n"
        "    plot_results = pickle.load(handle)\n"
        "\n"
        "%config InlineBackend.figure_format = 'svg'\n"
        "plot_years = test_years\n"
        "subset_keys = [(year, (52, 5.5)) for year in plot_years]\n"
        "results_subset = {str(k): plot_results[k] for k in subset_keys if k in plot_results}\n"
        "plot_variables = ['fertilizer', 'reward']\n"
        "plot_average = True\n"
        "\n"
        "fig, axes = plt.subplots(len(plot_variables), 1, sharex=True, figsize=(5 * len(plot_variables), 5))\n"
        "for i, variable in enumerate(plot_variables):\n"
        "    ax = axes if len(plot_variables) == 1 else axes[i]\n"
        "    plot_variable(\n"
        "        results_subset,\n"
        "        variable=variable,\n"
        "        cumulative_variables=['fertilizer', 'reward', 'IRRAD', 'RAIN'],\n"
        "        ax=ax,\n"
        "        ylim=get_ylim_dict()[variable],\n"
        "        plot_average=plot_average,\n"
        "        put_legend=True,\n"
        "    )"
    ),
    md("#### TO DO 2:\n* In the fertilizer graph, what amount did the agent fertilize in total?\n* How does the reward evolve over the season?"),
    md("## Training an RL agent in CropGym"),
    md("Train DQN and PPO agents using the current `WinterWheat` environment."),
    md("### Training a DQN agent"),
    code(
        "from stable_baselines3 import DQN, PPO\n"
        "from stable_baselines3.common.monitor import Monitor\n"
        "from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize\n"
        "\n"
        "seed = 1\n"
        "nsteps = 10_000\n"
        "costs_nitrogen = 10\n"
        "rootdir = repo_root\n"
        "log_dir = rootdir / 'notebooks' / 'nitrogen-winterwheat' / 'tensorboard_logs' / 'CropGym-Tutorial'\n"
        "log_dir.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "demo_years = [*range(1990, 1993)]\n"
        "demo_train_years = [year for year in demo_years if year % 2 == 1]\n"
        "demo_test_years = [year for year in demo_years if year % 2 == 0]\n"
        "demo_train_locations = [(52, 5.5)]\n"
        "demo_test_locations = [(48, 0)]\n"
        "\n"
        "crop_features = defaults.get_default_crop_features(pcse_env=0)\n"
        "weather_features = defaults.get_default_weather_features()\n"
        "action_features = []\n"
        "tag = f'Seed-{seed}'"
    ),
    code(
        "env_pcse_train = WinterWheat(\n"
        "    crop_features=crop_features,\n"
        "    action_features=action_features,\n"
        "    weather_features=weather_features,\n"
        "    costs_nitrogen=costs_nitrogen,\n"
        "    years=demo_train_years,\n"
        "    locations=demo_train_locations,\n"
        "    action_space=gym.spaces.Discrete(3),\n"
        "    action_multiplier=2.0,\n"
        "    reward='DEF',\n"
        "    **get_model_kwargs(0),\n"
        ")\n"
        "env_pcse_train = Monitor(env_pcse_train)\n"
        "env_pcse_train = VecNormalize(\n"
        "    DummyVecEnv([lambda: env_pcse_train]),\n"
        "    norm_obs=True,\n"
        "    norm_reward=True,\n"
        "    clip_obs=10000.,\n"
        "    clip_reward=50000.,\n"
        "    gamma=0.99,\n"
        ")\n"
        "\n"
        "env_pcse_eval = WinterWheat(\n"
        "    crop_features=crop_features,\n"
        "    action_features=action_features,\n"
        "    weather_features=weather_features,\n"
        "    costs_nitrogen=costs_nitrogen,\n"
        "    years=demo_test_years,\n"
        "    locations=demo_test_locations,\n"
        "    action_space=gym.spaces.Discrete(3),\n"
        "    action_multiplier=2.0,\n"
        "    reward='DEF',\n"
        "    **get_model_kwargs(0),\n"
        ")"
    ),
    code(
        "dqn_model_name = f'{tag}-Ncosts-{costs_nitrogen}-DQN-run'\n"
        "folder = dqn_model_name + '_1'\n"
        "log_dir_tensorboard = log_dir / folder\n"
        "\n"
        "try:\n"
        "    %load_ext tensorboard\n"
        "    %tensorboard --logdir $log_dir_tensorboard\n"
        "except Exception:\n"
        "    print('TensorBoard extension not available; training will still run.')\n"
        "\n"
        "model_dqn = DQN('MlpPolicy', env_pcse_train, gamma=0.99, seed=seed, verbose=0, tensorboard_log=str(log_dir))\n"
        "print(f'train for {nsteps} steps with costs_nitrogen={costs_nitrogen} (seed={seed})')\n"
        "model_dqn.learn(\n"
        "    total_timesteps=nsteps,\n"
        "    callback=EvalCallback(\n"
        "        env_eval=env_pcse_eval,\n"
        "        test_years=demo_test_years,\n"
        "        train_years=demo_train_years,\n"
        "        train_locations=demo_train_locations,\n"
        "        test_locations=demo_test_locations,\n"
        "        eval_freq=nsteps,\n"
        "        pcse_model=0,\n"
        "    ),\n"
        "    tb_log_name=dqn_model_name,\n"
        ")"
    ),
    md("#### TO DO 3:\n* Inspect TensorBoard images for WSO and reward.\n* Compare train vs. test performance."),
    md("### Training a PPO agent"),
    code(
        "hyperparams = {\n"
        "    'batch_size': 64,\n"
        "    'n_steps': 2048,\n"
        "    'learning_rate': 0.0003,\n"
        "    'ent_coef': 0.0,\n"
        "    'clip_range': 0.3,\n"
        "    'n_epochs': 10,\n"
        "    'gae_lambda': 0.95,\n"
        "    'max_grad_norm': 0.5,\n"
        "    'vf_coef': 0.5,\n"
        "    'policy_kwargs': get_policy_kwargs(\n"
        "        n_crop_features=len(crop_features),\n"
        "        n_weather_features=len(weather_features),\n"
        "        n_action_features=len(action_features),\n"
        "    ),\n"
        "}\n"
        "hyperparams['policy_kwargs']['net_arch'] = dict(pi=[128, 128], vf=[128, 128])\n"
        "hyperparams['policy_kwargs']['ortho_init'] = False"
    ),
    code(
        "env_pcse_train = WinterWheat(\n"
        "    crop_features=crop_features,\n"
        "    action_features=action_features,\n"
        "    weather_features=weather_features,\n"
        "    costs_nitrogen=costs_nitrogen,\n"
        "    years=demo_train_years,\n"
        "    locations=demo_train_locations,\n"
        "    action_space=gym.spaces.Discrete(3),\n"
        "    action_multiplier=2.0,\n"
        "    reward='DEF',\n"
        "    **get_model_kwargs(0),\n"
        ")\n"
        "env_pcse_train = Monitor(env_pcse_train)\n"
        "env_pcse_train = VecNormalize(\n"
        "    DummyVecEnv([lambda: env_pcse_train]),\n"
        "    norm_obs=True,\n"
        "    norm_reward=True,\n"
        "    clip_obs=10.,\n"
        "    clip_reward=50.,\n"
        "    gamma=0.99,\n"
        ")"
    ),
    code(
        "ppo_model_name = f'{tag}-Ncosts-{costs_nitrogen}-PPO-run'\n"
        "folder = ppo_model_name + '_1'\n"
        "log_dir_tensorboard = log_dir / folder\n"
        "\n"
        "try:\n"
        "    %reload_ext tensorboard\n"
        "    %tensorboard --logdir $log_dir_tensorboard\n"
        "except Exception:\n"
        "    pass\n"
        "\n"
        "# TODO 4: set nsteps (e.g. 20_000) and train PPO\n"
        "nsteps = 20_000\n"
        "model_ppo = PPO('MlpPolicy', env_pcse_train, gamma=0.99, seed=seed, verbose=0,\n"
        "                tensorboard_log=str(log_dir), **hyperparams)\n"
        "model_ppo.learn(total_timesteps=nsteps, tb_log_name=ppo_model_name)"
    ),
    md("#### TO DO 5:\n* How many times did the agent fertilize?\n* Compare DQN and PPO learning curves."),
    md("### Looking into the Deep RL network used by the RL agent"),
    code("print(model_dqn.policy)"),
    md(
        "#### TO DO 6:\n"
        "* What is the activation function used for the DQN model?\n"
        "* What is the size of the hidden layers?"
    ),
    code("# TODO 6: inspect model_dqn.policy"),
    md("## Additional results from the paper"),
    code(
        "var = 'WSO'\n"
        "df_merged = read_data(resultsdir)\n"
        "df_merged = df_merged[df_merged.year % 2 == 0]\n"
        "\n"
        "random.seed(42)\n"
        "n_boot = 1000\n"
        "name_var_rl = [col for col in df_merged.columns if f'{var}_RL' in col]\n"
        "cols_var_rl = [df_merged.columns.get_loc(col) for col in name_var_rl]\n"
        "\n"
        "for location in ['52;5.5', '48;0']:\n"
        "    df_boot = df_merged.loc[df_merged['location'] == location]\n"
        "    boot_rl, boot_sp, boot_delta, boot_ceres = [], [], [], []\n"
        "    n_observations = len(df_boot.index)\n"
        "    for _ in range(n_boot):\n"
        "        obs = random.choices(range(n_observations), k=n_observations)\n"
        "        seed_cols = random.choices(cols_var_rl, k=n_observations)\n"
        "        var_rl = df_boot.values[obs, seed_cols]\n"
        "        var_ceres = df_boot.values[obs, [df_boot.columns.get_loc(f'{var}_Ceres')] * n_observations]\n"
        "        var_sp = df_boot.values[obs, [df_boot.columns.get_loc(f'{var}_SP')] * n_observations]\n"
        "        boot_ceres.append(np.median(var_ceres))\n"
        "        boot_rl.append(np.median(var_rl))\n"
        "        boot_sp.append(np.median(var_sp))\n"
        "        boot_delta.append(np.median(var_rl - var_sp))\n"
        "    print(f'**{location_to_label[location]}**')\n"
        "    print(f'median_Ceres: {np.median(df_boot[f\"{var}_Ceres\"]):0.2f} {report_ci(boot_ceres)}')\n"
        "    print(f'median_SP: {np.median(df_boot[f\"{var}_SP\"]):0.2f} {report_ci(boot_sp)}')\n"
        "    print(f'median_RL: {np.median(pd.concat([df_boot[col] for col in name_var_rl])):0.2f} {report_ci(boot_rl)}')\n"
        "    print(f'median_RL-SP: {np.median(pd.concat([df_boot[col] - df_boot[f\"{var}_SP\"] for col in name_var_rl])):0.2f} {report_ci(boot_delta, True)}')"
    ),
    code(
        "%config InlineBackend.figure_format = 'svg'\n"
        "from matplotlib.container import ErrorbarContainer\n"
        "from matplotlib.collections import LineCollection\n"
        "import matplotlib.lines as mlines\n"
        "from sklearn.linear_model import Ridge\n"
        "\n"
        "markers = {'52;5.5': 'o'}\n"
        "plots = {0: ('reward_Ceres', 'reward'), 1: ('fertilizer_Ceres', 'fertilizer')}\n"
        "modes = ['Ceres', 'SP', 'RL']\n"
        "df_merged = read_data(resultsdir)\n"
        "df_merged = df_merged[df_merged.year % 2 == 0]\n"
        "\n"
        "fig, axes_scatter = plt.subplots(1, len(plots), figsize=(len(plots) * 8, 8), sharey='col', sharex='col')\n"
        "for p, (x_orig, y_orig) in plots.items():\n"
        "    ax_scatter = axes_scatter if len(plots) == 1 else axes_scatter[p]\n"
        "    title = f'{x_orig}/{y_orig}'\n"
        "    for m in modes:\n"
        "        x = x_orig\n"
        "        y = f'{y_orig}_RL_0' if m == 'RL' else f'{y_orig}_{m}'\n"
        "        if x not in df_merged.columns or y not in df_merged.columns:\n"
        "            continue\n"
        "        for location in markers:\n"
        "            df_scatter = df_merged.loc[df_merged['location'] == location]\n"
        "            ax_scatter.scatter(df_scatter[x], df_scatter[y], c=colors[m], marker=markers[location], alpha=0.5, label=m)\n"
        "    ax_scatter.set_title(title)\n"
        "    ax_scatter.legend()\n"
        "identity_line(ax_scatter)"
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

target = Path(__file__).resolve().parents[1] / "notebooks/tutorials/Advanced_Machine_Learning_CropGym_Tutorial.ipynb"
target.write_text(json.dumps(notebook, indent=2))
print(f"Wrote {target}")
