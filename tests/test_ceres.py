import unittest
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import tests.initialize_env as init_env
from pcse_gym.utils.eval import FindOptimum
import pcse_gym.utils.defaults as defaults


class TestCeres(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env(pcse_env=0, crop_features=defaults.get_default_crop_features(pcse_env=0))
        self.env = VecNormalize(DummyVecEnv([lambda: self.env]), norm_reward=True, clip_reward=50., gamma=1)

    def test_single_year(self):
        ceres_result = FindOptimum(self.env, [1992]).optimize_start_dump().item()
        self.assertAlmostEqual(5.179, ceres_result, 1)

    def test_multiple_years(self):
        ceres_result = FindOptimum(self.env, [1992, 2002]).optimize_start_dump().item()
        self.assertAlmostEqual(5.281, ceres_result, 1)


