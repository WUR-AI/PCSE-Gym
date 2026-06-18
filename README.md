CropGym is a highly configurable [Python gymnasium](https://gymnasium.farama.org/) environment to conduct Reinforcement Learning (RL) research for crop management. CropGym is built around [PCSE](https://pcse.readthedocs.io/en/stable/), a well established python library that includes implementations of a variety of crop simulation models. Please refer to https://cropgym.ai/ for further information.

## Tutorials

- **LINTUL-3 (nitrogen)**: `notebooks/tutorials/Advanced_Machine_Learning_CropGym_Tutorial.ipynb`
- **WOFOST SNOMIN**: `notebooks/tutorials/CropGym_WOFOST_SNOMIN_Tutorial.ipynb` — detailed soil nitrogen dynamics, inspired by [NUE_PCSE-Gym](https://github.com/WUR-AI/NUE_PCSE-Gym)

Train from the command line:

```bash
poetry install --extras sb-integration
python train_winterwheat.py --environment 2 --agent PPO --nsteps 50000
```

Environment codes: `0` = LINTUL-3, `1` = WOFOST 8.0, `2` = WOFOST 8.1 + SNOMIN.