import unittest
import numpy as np

import tests.initialize_env as init_env
from pcse_gym.utils.normalization import RunningMeanStdPO, VecNormalizePO, NormalizeMeasureObservations
from stable_baselines3.common.vec_env import DummyVecEnv


class Normalizations(unittest.TestCase):

    def setUp(self):
        self.env = init_env.initialize_env_measure_po_extend()

    def assertListAlmostEqual(self, list1, list2, tol):
        if len(list1) != 1:
            list2 = [list2]
        self.assertEqual(len(list1[0]), len(list2))
        for a, b in zip(list1[0], list2):
            self.assertAlmostEqual(a, b, tol)

    def test_sb3_rms(self):
        rms = RunningMeanStdPO(epsilon=1e-8, shape=(6,), placeholder_value=-1.11)
        observations = np.array([[5, 5, 5, 5, 5, 5]])
        rms.update(observations)
        self.assertListAlmostEqual([[5, 5, 5, 5, 5, 5]], rms.mean.tolist(), 1)

        observations = np.array([[10, -1.11, 10, 10, 10, 10]])
        rms.update(observations)
        self.assertListAlmostEqual([[7.5, 5, 7.5, 7.5, 7.5, 7.5]], rms.mean.tolist(), 1)

        observations = np.array([[20, 10, 20, 20, -1.11, 20]])
        rms.update(observations)
        self.assertListAlmostEqual([[11.67, 6.67, 11.67, 11.67, 7.5, 11.67]], rms.mean.tolist(), 1)

    def test_sb3_rms_masked(self):
        rms = RunningMeanStdPO(epsilon=1e-8, shape=(6 * 2,), placeholder_value=-1.11,
                               extend=True)
        observations = np.array([[5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1]])
        rms.update(observations)
        self.assertListAlmostEqual([[5, 5, 5, 5, 5, 5]], rms.mean.tolist(), 1)

        observations = np.array([[10, -1.11, 10, 10, 10, 10, 1, 0, 1, 1, 1, 1]])
        rms.update(observations)
        self.assertListAlmostEqual([[7.5, 5, 7.5, 7.5, 7.5, 7.5]], rms.mean.tolist(), 1)

        observations = np.array([[20, 10, 20, 20, -1.11, 20, 1, 1, 1, 1, 0, 1]])
        rms.update(observations)
        self.assertListAlmostEqual([[11.67, 6.67, 11.67, 11.67, 7.5, 11.67]], rms.mean.tolist(), 1)

    # TODO: create unit test for SB3 RMS
    # def test_vec_env_masked(self):
    #     vec_env = VecNormalizePO(DummyVecEnv([lambda: self.env]), norm_obs=True, norm_reward=True,
    #                              clip_obs=10., clip_reward=50., gamma=1)
    #     action = np.array([5, 1, 0, 0, 0, 1])
    #
    #     obs, _, _, _ = vec_env.step(action)
    #     norm = vec_env._normalize_obs(obs, vec_env.obs_rms)
    #     # obs, _, _, _ = vec_env.step(action)
    #     # observations = np.array([[5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1]])
    #
    #     # norm = vec_env._normalize_obs(obs, vec_env.obs_rms)
    #
    #     self.assertListAlmostEqual(obs, norm, 1)


class MeasureNormalizations(unittest.TestCase):

    def setUp(self):
        self.env = init_env.initialize_env_measure_po_normalize_extend()

    def assertListAlmostEqual(self, list1, list2, tol):
        if len(list1) != 1:
            list1 = [list1]
        self.assertEqual(len(list1[0]), len(list2))
        for a, b in zip(list1[0], list2):
            self.assertAlmostEqual(a, b, tol)

    def test_norm_obs_winterwheat(self):
        self.env.reset()
        action = np.array([5, 1, 0, 0, 0, 1])
        measure = action[1:]

        obs, _, _, _, _ = self.env.step(action)

        # expected = np.ones((len(obs),))
        # for i, m, in zip(self.env.measure_features.feature_ind, measure):
        #     j = len(obs) // 2 + i
        #     if not m:
        #         expected[i] = 0.0
        #         expected[j] = 0.0
        #     else:
        #         expected[i] = obs[i]
        #         expected[j] = 1

        self.assertListAlmostEqual([-1.23,
                              -0.78,
                              0.0,
                              -0.85,
                              0.0,
                              0.0,
                              1.0,
                              0.0,
                              1.0,
                              0.0,
                              0.0], list(obs), 1)
