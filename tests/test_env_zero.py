import unittest
import numpy as np

import tests.initialize_env as init_env


class ZeroEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.years = [*range(1990, 2022)]
        self.env_sow = init_env.initialize_env(reward="DEF", start_type='sowing')
        self.env_emerge = init_env.initialize_env(reward="DEF", start_type='emergence')
        self.random_init = init_env.initialize_env(reward="DEF", start_type='sowing', random_init=True)

    @staticmethod
    def run_steps(env, year, terminated):
        env.overwrite_year(year)
        env.reset()
        while not terminated:
            _, _, terminated, _, info = env.step(np.array([0]))
        dvs_rl = list(info['DVS'].values())
        dvs_zero = list(env.zero_nitrogen_env_storage.get_result[f'{year}-(52, 5.5)']['DVS'].values())
        print(f'year {year}')

        return dvs_rl, dvs_zero

    def try_years(self, env):
        for year in self.years:
            terminated = False
            dvs_rl, dvs_zero = self.run_steps(env, year, terminated)
            return dvs_rl, dvs_zero

    def test_dvs_zero_env(self):
        """Test whether zero env is referencing the correct year by
        checking if the DVS in the zero env and RL env is the same"""
        dvs_rl, dvs_zero = self.try_years(self.env_sow)
        self.assertListEqual(dvs_rl, dvs_zero)
        dvs_rl, dvs_zero = self.try_years(self.env_emerge)
        self.assertListEqual(dvs_rl, dvs_zero)

    def test_baseline_env(self):
        self.env_sow.overwrite_year(2002)
        self.env_sow.reset()

        week = 0
        terminated = False

        while not terminated:
            # action = np.array([4]) if week == 8 else np.array([0])
            _, _, terminated, _, info = self.env_sow.step(np.array([6]))
            week += 1

        wso_rl = self.env_sow.sb3_env.model.get_output()[-1:][0]['WSO']
        wso_baseline = self.env_sow.baseline_env.model.get_output()[-1:][0]['WSO']
        print(wso_rl, wso_baseline)
        self.assertNotEqual(wso_rl, wso_baseline)