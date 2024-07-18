import unittest
import numpy as np
from math import isclose

import tests.initialize_env as init_env


class Rewards(unittest.TestCase):
    def setUp(self):
        self.dep = init_env.initialize_env_reward_dep()
        self.eny = init_env.initialize_env_eny_reward()
        self.nue = init_env.initialize_env_nue_reward()
        self.nup = init_env.initialize_env_nup_reward()
        self.har = init_env.initialize_env_har_reward()
        self.dnu = init_env.initialize_env_dnu_reward()
        self.fin = init_env.initialize_env_fin_reward()
        self.def1 = init_env.initialize_env(reward='DEF', pcse_env=2)
        self.dne = init_env.initialize_env(reward='DNE', pcse_env=2)

    @staticmethod
    def run_steps_sp(env, year, terminated):
        env.overwrite_year(year)
        env.reset()

        week = 0
        n = 8
        rewards = 0
        while not terminated:
            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = env.step(action)
            rewards += reward
            week += 1

        return rewards

    def test_reward_functions(self):
        rfs = [self.dep, self.nue, self.eny, self.nup,
               self.har, self.dnu, self.fin, self.def1, self.dne]
        expected_rs = [8638.21, -20, 8658.21, 137.42, 1266.20, -47.95, 1359.65, 2291.47, -120.0,]
        for rf_env, expected_r in zip(rfs, expected_rs):
            r = self.run_steps_sp(rf_env, 2002, False)
            print(f"reward {r} and rf {rf_env.reward_function}")
            check_if_close = isclose(r, expected_r, abs_tol=5)
            self.assertTrue(check_if_close)


