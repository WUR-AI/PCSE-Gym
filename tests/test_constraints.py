import unittest
import yaml
import os
import numpy as np

import pcse.util
import tests.initialize_env as init_env
from pcse_gym.utils.normalization import RunningMeanStdPO, VecNormalizePO
from stable_baselines3.common.vec_env import DummyVecEnv


# class TestRecoveryRate(unittest.TestCase):
#     def setUp(self):
#         self.env = init_env.initialize_env_rr()
#
#     def test_rr(self):
#         self.env.reset()
#         action = np.array([0])
#         _, _, _, _, info = self.env.step(action)
#         initial_n = list(info['NAVAIL'].values())[0]
#         action = np.array([2])
#         actual = int(action) * 10 * self.env.sb3_env.recovery_penalty()
#         _, _, _, _, info = self.env.step(action)
#         # key = info['NAVAIL'].keys()[-1]
#         end_n = list(info['NAVAIL'].values())[-1]
#
#         expected = end_n - initial_n
#
#         # assert almost equal due to possibly the crop taking up some nitrogen
#         self.assertAlmostEqual(expected, actual, 1)


class ActionLimit(unittest.TestCase):
    def setUp(self):
        self.env_meas = init_env.initialize_env_action_limit_measure(4)
        self.env_no_meas = init_env.initialize_env_action_limit_no_measure(4)
        self.env_budget = init_env.initialize_env_action_limit_budget_no_measure(5, 180)
        self.env_budget_meas = init_env.initialize_env_action_limit_budget_measure(6, 180)

    def test_limit_measure(self):
        self.env_meas.reset()

        loop = 16
        hist = []
        # for action
        for i in range(loop):
            action = np.array([1, 1, 0, 0, 0, 1])
            check = self.env_meas.action(action)
            hist.append(check)

        actions_hist = [item[0] for item in hist]

        actions_expected = np.zeros(loop, int)
        for a, b in enumerate(actions_expected[:4]):
            actions_expected[a] = 1

        self.assertListEqual(actions_hist, list(actions_expected))

    def test_limit_no_measure(self):
        self.env_no_meas.reset()

        loop = 16
        hist = []
        # for action
        for i in range(loop):
            action = np.array(1)
            check = self.env_no_meas.action(action)
            hist.append(check)

        actions_expected = np.zeros(loop, int)
        for a, b in enumerate(actions_expected[:4]):
            actions_expected[a] = 1

        self.assertListEqual(hist, list(actions_expected))

    def test_limit_budget_no_measure(self):
        self.env_budget.reset()

        actions = [2, 4, 6, 4, 5, 5, 5]
        loop = 7
        hist = []
        for act, i in zip(actions, range(loop)):
            action = np.array([act])
            check = self.env_budget.action(action)
            hist.append(check)

        actions_expected = [2, 4, 6, 4, 2, 0, 0]

        self.assertListEqual(hist, actions_expected)

    def test_limit_budget_measure(self):
        self.env_budget_meas.reset()

        loop = 6
        hist = []
        for i in range(loop):
            action = np.array([5, 1, 0, 0, 0, 1])
            check = self.env_budget_meas.action(action)
            hist.append(check)

        actions_hist = [item[0] for item in hist]

        actions_expected = [5, 5, 5, 3, 0, 0]

        self.assertListEqual(actions_hist, actions_expected)


class TestStartType(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_sow()
        self.env2 = init_env.initialize_env_emergence()

    def test_sow_start(self):
        self.env.reset()

        year = [2012]
        self.env.overwrite_year(year)
        self.env.reset()
        self.assertEqual(year[0] - 1, int(self.env.date.year))

        year = [2016]
        self.env.overwrite_year(year)
        self.env.reset()
        self.assertEqual(year[0] - 1, int(self.env.date.year))

    def test_emergence_start(self):
        self.env2.reset()

        year = [1990]
        self.env2.overwrite_year(year)
        self.env2.reset()
        self.assertEqual(year[0], int(self.env2.date.year))

        year = [2000]
        self.env2.overwrite_year(year)
        self.env2.reset()
        self.assertEqual(year[0], int(self.env2.date.year))


class TestEnvFeatures(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_random_init()

    def test_different_initial_conditions(self):
        dir_root = os.path.dirname(os.path.realpath(__file__))[:-5]
        site_params = yaml.safe_load(open(os.path.join(dir_root,
                                                       'pcse_gym', 'envs', 'configs', 'site', 'arminda_site.yaml')))
        self.assertEqual(site_params, self.env.sb3_env.model.parameterprovider._sitedata)
        self.env.reset()
        self.assertNotEqual(site_params, self.env.sb3_env.model.parameterprovider._sitedata)






