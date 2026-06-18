"""Generate the CropGym WOFOST SNOMIN tutorial notebook (Colab self-contained)."""

import json
from pathlib import Path

# GitHub default branch is `master` (not `main`). Until the SNOMIN PR is merged,
# point Colab + clone at the feature branch; switch to "master" after merge.
GITHUB_BRANCH = "feature/pcse6-wofost-snomin-tutorial"
COLAB_URL = (
    "https://colab.research.google.com/github/WUR-AI/PCSE-Gym/blob/"
    f"{GITHUB_BRANCH}/notebooks/tutorials/CropGym_WOFOST_SNOMIN_Tutorial.ipynb"
)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text, colab_title=None):
    meta = {}
    if colab_title:
        meta["colab"] = {"name": colab_title}
        # Colab collapsible install cell
        if text.strip().startswith("#@title"):
            meta["cellView"] = "form"
    return {
        "cell_type": "code",
        "metadata": meta,
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        "# CropGym tutorial: nitrogen management with WOFOST SNOMIN\n"
        "\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({COLAB_URL})\n"
        "\n"
        "This notebook is **self-contained for Google Colab**: run cells top-to-bottom. "
        "It installs CropGym, PCSE 6.x, and Stable-Baselines3 automatically.\n"
        "\n"
        "You will:\n"
        "1. Configure a CropGym environment (`pcse_model=2`, WOFOST + SNOMIN)\n"
        "2. Inspect observations (including layered soil N)\n"
        "3. Compare baseline fertilization policies\n"
        "4. Train a short PPO agent\n"
        "\n"
        "Runtime: ~15–30 min on Colab CPU (mostly NASA weather fetch + short PPO training)."
    ),
    md("## 1. Install CropGym (Colab)"),
    code(
        "#@title Install dependencies and clone CropGym { display-mode: \"form\" }\n"
        "import sys\n"
        "import subprocess\n"
        "from pathlib import Path\n"
        "\n"
        "IN_COLAB = 'google.colab' in sys.modules\n"
        "REPO_URL = 'https://github.com/WUR-AI/PCSE-Gym.git'\n"
        f"REPO_BRANCH = '{GITHUB_BRANCH}'\n"
        "repo_root = Path('/content/PCSE-Gym') if IN_COLAB else Path.cwd()\n"
        "\n"
        "if IN_COLAB:\n"
        "    if not repo_root.exists():\n"
        "        subprocess.run(\n"
        "            ['git', 'clone', '--depth', '1', '-b', REPO_BRANCH, REPO_URL, str(repo_root)],\n"
        "            check=True,\n"
        "        )\n"
        "    else:\n"
        "        print(f'Repo already present at {repo_root}')\n"
        "    import os\n"
        "    os.chdir(repo_root)\n"
        "\n"
        "    deps = [\n"
        "        'pcse>=6.0.13', 'gymnasium>=0.29', 'pyyaml', 'numpy',\n"
        "        'stable-baselines3>=2.3', 'sb3-contrib', 'matplotlib', 'torch',\n"
        "        'tensorboard', 'scipy', 'tqdm', 'pandas', 'seaborn',\n"
        "    ]\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip'], check=True)\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *deps], check=True)\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(repo_root)], check=True)\n"
        "else:\n"
        "    candidate = Path.cwd().resolve()\n"
        "    for _ in range(4):\n"
        "        if (candidate / 'pcse_gym').exists():\n"
        "            repo_root = candidate\n"
        "            break\n"
        "        candidate = candidate.parent\n"
        "    print('Local mode — ensure: poetry install --extras sb-integration')\n"
        "\n"
        "if str(repo_root) not in sys.path:\n"
        "    sys.path.insert(0, str(repo_root))\n"
        "\n"
        "# Writable PCSE cache/logs (must be set before importing pcse)\n"
        "import os\n"
        "pcse_home = repo_root / '.pcse_home'\n"
        "pcse_home.mkdir(exist_ok=True)\n"
        "(pcse_home / 'logs').mkdir(exist_ok=True)\n"
        "(pcse_home / 'meteo_cache').mkdir(exist_ok=True)\n"
        "os.environ['HOME'] = str(pcse_home)\n"
        "\n"
        "print('repo_root:', repo_root)\n"
        "print('Python:', sys.version.split()[0])"
    ),
    code(
        "#@title Verify installation { display-mode: \"form\" }\n"
        "from pathlib import Path\n"
        "import importlib\n"
        "import pcse\n"
        "import gymnasium\n"
        "import stable_baselines3\n"
        "\n"
        "from pcse_gym.envs.sb3 import get_config_dir\n"
        "\n"
        "config_dir = Path(get_config_dir())\n"
        "required = [\n"
        "    config_dir / 'Wofost81_NWLP_MLWB_SNOMIN.conf',\n"
        "    config_dir / 'soil' / 'arminda_soil.yaml',\n"
        "    config_dir / 'site' / 'arminda_site.yaml',\n"
        "    config_dir / 'agro' / 'wheat_cropcalendar_snomin.yaml',\n"
        "    config_dir / 'crop' / 'winterwheat.yaml',\n"
        "]\n"
        "missing = [p for p in required if not p.exists()]\n"
        "if missing:\n"
        "    raise FileNotFoundError('Missing SNOMIN config files:\\n' + '\\n'.join(map(str, missing)))\n"
        "\n"
        "print('pcse', getattr(pcse, '__version__', 'installed'))\n"
        "print('gymnasium', gymnasium.__version__)\n"
        "print('stable-baselines3', stable_baselines3.__version__)\n"
        "print('SNOMIN configs OK')"
    ),
    md("## 2. Imports and settings"),
    code(
        "import os\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import gymnasium as gym\n"
        "\n"
        "from pcse_gym.envs.winterwheat import WinterWheat\n"
        "from pcse_gym.envs.sb3 import get_model_kwargs, get_policy_kwargs\n"
        "from pcse_gym.utils.eval import evaluate_policy\n"
        "from pcse_gym.utils.nitrogen_helpers import get_nitrogen_levels\n"
        "import pcse_gym.utils.defaults as defaults\n"
        "\n"
        "PCSE_MODEL = 2  # WOFOST 8.1 + SNOMIN\n"
        "crop_features = defaults.get_snomin_default_crop_features()\n"
        "weather_features = defaults.get_default_weather_features()\n"
        "action_features = defaults.get_default_action_features()\n"
        "n_levels = 5\n"
        "action_space = defaults.get_snomin_action_space(n_levels)\n"
        "nitrogen_levels = get_nitrogen_levels(n_levels)\n"
        "costs_nitrogen = 10.0\n"
        "\n"
        "print('Crop features:', crop_features)\n"
        "print('Nitrogen levels (kg N/ha):', nitrogen_levels)"
    ),
    md(
        "## 3. Create the CropGym environment\n"
        "\n"
        "`WinterWheat` wraps PCSE in a Gymnasium interface. With `pcse_model=2` it uses:\n"
        "- `Wofost81_NWLP_MLWB_SNOMIN.conf`\n"
        "- multi-layer soil files (`arminda_soil.yaml`, `arminda_site.yaml`)\n"
        "- winter wheat variety **Arminda** (`wheat_cropcalendar_snomin.yaml`)\n"
        "\n"
        "The default reward (`DEF`) compares yield growth against a zero-nitrogen baseline, "
        "minus an economic cost for fertilizer."
    ),
    code(
        "def make_env(years, locations, seed=0):\n"
        "    return WinterWheat(\n"
        "        crop_features=crop_features,\n"
        "        action_features=action_features,\n"
        "        weather_features=weather_features,\n"
        "        costs_nitrogen=costs_nitrogen,\n"
        "        years=years,\n"
        "        locations=locations,\n"
        "        action_space=action_space,\n"
        "        reward='DEF',\n"
        "        seed=seed,\n"
        "        n_nitrogen_levels=n_levels,\n"
        "        **get_model_kwargs(PCSE_MODEL),\n"
        "    )\n"
        "\n"
        "demo_year = 2001\n"
        "demo_location = (52, 5.5)\n"
        "env = make_env(demo_year, demo_location)\n"
        "obs, info = env.reset()\n"
        "print('Observation shape:', obs.shape)\n"
        "print('Action space:', env.action_space)\n"
        "print('Observation space:', env.observation_space)"
    ),
    md(
        "## 4. Run one growing season manually\n"
        "\n"
        "Actions are discrete indices mapped to fertilizer rates in **kg N/ha**. "
        "The agent decides weekly (7-day timesteps)."
    ),
    code(
        "env = make_env(demo_year, demo_location, seed=1)\n"
        "obs, _ = env.reset()\n"
        "total_reward = 0.0\n"
        "history = []\n"
        "terminated = truncated = False\n"
        "while not (terminated or truncated):\n"
        "    action = int(env.action_space.sample())\n"
        "    obs, reward, terminated, truncated, info = env.step(action)\n"
        "    total_reward += reward\n"
        "    history.append({\n"
        "        'day': env.date,\n"
        "        'action_idx': action,\n"
        "        'fertilizer_kg': nitrogen_levels[action],\n"
        "        'reward': reward,\n"
        "        'WSO': info.get('WSO', {}).get(env.date, np.nan),\n"
        "        'NLOSSCUM': info.get('NLOSSCUM', {}).get(env.date, np.nan),\n"
        "    })\n"
        "\n"
        "df_run = pd.DataFrame(history)\n"
        "print(f'Total reward: {total_reward:.1f}')\n"
        "df_run.tail()"
    ),
    code(
        "fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)\n"
        "df_run.set_index('day')['WSO'].plot(ax=axes[0], title='Storage organ weight (WSO, kg/ha)')\n"
        "df_run.set_index('day')['NLOSSCUM'].plot(ax=axes[1], title='Cumulative N loss (NLOSSCUM)')\n"
        "df_run.set_index('day')['fertilizer_kg'].plot(ax=axes[2], drawstyle='steps-post', title='Applied N (kg/ha)')\n"
        "fig.autofmt_xdate()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 5. Compare baseline policies"),
    code(
        "from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize\n"
        "\n"
        "def run_policy(policy, year, location):\n"
        "    env = make_env(year, location)\n"
        "    env_vec = VecNormalize(\n"
        "        DummyVecEnv([lambda y=year, loc=location: make_env(y, loc)]),\n"
        "        norm_reward=True, clip_reward=50.0, gamma=1,\n"
        "    )\n"
        "    env_vec.training = False\n"
        "    env_vec.norm_reward = True\n"
        "    rewards, infos = evaluate_policy(policy, env_vec)\n"
        "    return rewards[0], infos[0]\n"
        "\n"
        "year, location = 2000, (52, 5.5)\n"
        "policies = {\n"
        "    'zero-N': 'no-nitrogen',\n"
        "    'standard practice': 'standard-practice',\n"
        "}\n"
        "for name, policy in policies.items():\n"
        "    reward, info = run_policy(policy, year, location)\n"
        "    wso = info['WSO']\n"
        "    final_wso = list(wso.values())[-1]\n"
        "    total_n = sum(info['fertilizer'].values())\n"
        "    print(f'{name:20s} reward={reward:8.1f}  final WSO={final_wso:8.1f} kg/ha  total N={total_n:.1f} kg/ha')"
    ),
    md(
        "## 6. Train a PPO agent (short demo)\n"
        "\n"
        "Training uses CPU by default (stable on free Colab). "
        "Increase `nsteps` for better policies (e.g. 50_000)."
    ),
    code(
        "from stable_baselines3 import PPO\n"
        "from stable_baselines3.common.monitor import Monitor\n"
        "from pcse_gym.utils.eval import EvalCallback\n"
        "\n"
        "train_years = [1999, 2001, 2003]\n"
        "test_years = [2000, 2002]\n"
        "train_locations = [(52, 5.5)]\n"
        "test_locations = [(52, 5.5)]\n"
        "nsteps = 5_000  # Colab demo; use 50_000+ for real experiments\n"
        "seed = 0\n"
        "\n"
        "env_train = make_env(train_years, train_locations, seed=seed)\n"
        "env_train = Monitor(env_train)\n"
        "env_train = VecNormalize(\n"
        "    DummyVecEnv([lambda: make_env(train_years, train_locations, seed=seed)]),\n"
        "    norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0, gamma=1,\n"
        ")\n"
        "env_eval = make_env(test_years, test_locations, seed=seed)\n"
        "\n"
        "hyperparams = {\n"
        "    'batch_size': 64,\n"
        "    'n_steps': 2048,\n"
        "    'learning_rate': 3e-4,\n"
        "    'clip_range': 0.3,\n"
        "    'policy_kwargs': get_policy_kwargs(\n"
        "        n_crop_features=len(crop_features),\n"
        "        n_weather_features=len(weather_features),\n"
        "        n_action_features=len(action_features),\n"
        "    ),\n"
        "    'device': 'cpu',\n"
        "}\n"
        "hyperparams['policy_kwargs']['net_arch'] = dict(pi=[128, 128], vf=[128, 128])\n"
        "\n"
        "log_dir = repo_root / 'notebooks' / 'tutorials' / 'tensorboard_snomin'\n"
        "log_dir.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "model = PPO('MlpPolicy', env_train, gamma=1, seed=seed, verbose=1,\n"
        "            tensorboard_log=str(log_dir), **hyperparams)\n"
        "model.learn(\n"
        "    total_timesteps=nsteps,\n"
        "    callback=EvalCallback(\n"
        "        env_eval=env_eval,\n"
        "        test_years=test_years,\n"
        "        train_years=train_years,\n"
        "        train_locations=train_locations,\n"
        "        test_locations=test_locations,\n"
        "        eval_freq=nsteps,\n"
        "        pcse_model=PCSE_MODEL,\n"
        "    ),\n"
        "    tb_log_name='SNOMIN-PPO-demo',\n"
        ")"
    ),
    code(
        "# Optional: view TensorBoard in Colab\n"
        "try:\n"
        "    get_ipython().run_line_magic('load_ext', 'tensorboard')\n"
        "    get_ipython().run_line_magic('tensorboard', f'--logdir {log_dir}')\n"
        "except Exception as exc:\n"
        "    print('TensorBoard not available:', exc)"
    ),
    md("## 7. Evaluate the trained agent"),
    code(
        "from stable_baselines3.common.vec_env import DummyVecEnv\n"
        "\n"
        "eval_env = DummyVecEnv([lambda: make_env(2002, (52, 5.5))])\n"
        "rewards, infos = evaluate_policy(model, eval_env)\n"
        "info = infos[0]\n"
        "print(f'RL reward: {rewards[0]:.1f}')\n"
        "print(f'Final WSO: {list(info[\"WSO\"].values())[-1]:.1f} kg/ha')\n"
        "print(f'Total N applied: {sum(info[\"fertilizer\"].values()):.1f} kg/ha')\n"
        "print(f'Cumulative N loss: {list(info[\"NLOSSCUM\"].values())[-1]:.1f}')"
    ),
    code(
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "pd.Series(info['WSO']).plot(ax=ax[0], title='WSO under learned policy')\n"
        "pd.Series(info['NLOSSCUM']).plot(ax=ax[1], title='Cumulative N loss')\n"
        "fig.autofmt_xdate()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 8. Next steps\n"
        "\n"
        "- Increase `nsteps` for better policies\n"
        "- Full CLI training: `python train_winterwheat.py --environment 2 --agent PPO --nsteps 50000`\n"
        "- NUE-focused rewards and constrained RL: [NUE_PCSE-Gym](https://github.com/WUR-AI/NUE_PCSE-Gym)\n"
        "- Simpler LINTUL-3 tutorial: `Advanced_Machine_Learning_CropGym_Tutorial.ipynb`"
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "colab": {
            "name": "CropGym_WOFOST_SNOMIN_Tutorial.ipynb",
            "provenance": [],
        },
    },
    "cells": cells,
}

target = Path(__file__).resolve().parents[1] / "notebooks/tutorials/CropGym_WOFOST_SNOMIN_Tutorial.ipynb"
target.write_text(json.dumps(notebook, indent=2))
print(f"Wrote {target}")
