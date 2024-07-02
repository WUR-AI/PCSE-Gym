import unittest
import numpy as np
import datetime
import yaml
import os

import tests.initialize_env as init_env

from pcse.input.nasapower import NASAPowerWeatherDataProvider

from pcse_gym.utils.nitrogen_helpers import (calculate_year_n_deposition,
                                             convert_year_to_n_concentration,
                                             calculate_day_n_deposition,
                                             get_aggregated_n_depo_days,
                                             get_no3_deposition_pcse,
                                             get_nh4_deposition_pcse)
from pcse_gym.utils.weather_utils.weather_functions import generate_date_list
from pcse_gym.envs.common_env import AgroManagementContainer
from tests import initialize_env as init_env
from pcse_gym.utils.nitrogen_helpers import get_deposition_amount, get_disaggregated_deposition

import matplotlib.pyplot as plt


class TestNitrogenUtils(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env(reward='NUE', pcse_env=2, start_type='sowing')

    def test_n_model_deposition(self):
        year = 2000
        self.env.overwrite_year(year)
        self.env.reset()

        wdp = NASAPowerWeatherDataProvider(52.0, 5.5)
        terminated = False

        while not terminated:
            _, _, terminated, _, _ = self.env.step(np.array([0]))

        date_range = generate_date_list(self.env.sb3_env.agmt.crop_start_date, self.env.sb3_env.agmt.crop_end_date)
        total_rain = 0.0
        daily_rain = [wdp(day).RAIN * 10 for day in date_range]
        for rain in daily_rain:
            total_rain += rain
        print(f"total RAIN {total_rain}")

        nh4test, no3test = calculate_year_n_deposition(year, (52.0, 5.5), self.env.sb3_env.agmt, self.env.sb3_env._site_params)

        self.assertAlmostEqual(nh4test, 16.756101936865323, 0)
        self.assertAlmostEqual(no3test, 9.178370307887013, 0)

    def test_n_concentration_conversion(self):
        nh4, no3 = convert_year_to_n_concentration(2000)

        self.assertEqual(nh4, 2.054742670516606)
        self.assertEqual(no3, 1.1753128075355042)

        nh4, no3 = convert_year_to_n_concentration(4098, random_weather=True)

        self.assertEqual(nh4, 2.371330250564566)
        self.assertEqual(no3, 1.4242391655016677)

    def test_n_concentration_conversion_with_agmt(self):
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(root, 'pcse_gym', 'envs', 'configs', 'agro', 'wheat_cropcalendar.yaml'), 'r') as f:
            agro_config = yaml.load(f, Loader=yaml.SafeLoader)
        agmt = AgroManagementContainer(agro_config)

        agmt.campaign_date = datetime.date(1999, 10, 1)
        agmt.crop_start_date = datetime.date(1999, 10, 3)
        agmt.crop_end_date = datetime.date(2000, 8, 20)

        nh4, no3 = convert_year_to_n_concentration(2000, agmt)

        self.assertEqual(nh4, 2.1035838876987443)
        self.assertEqual(no3, 1.2053951985280167)

    def test_day_n_deposition(self):
        year = 2000
        self.env.overwrite_year(year)
        self.env.reset()

        _, _, _, _, info = self.env.step(np.array([0]))
        _, _, _, _, info = self.env.step(np.array([0]))

        rain = list(info['RAIN'].values())[-1]  # mm

        day_nh4_depo, day_no3_depo = calculate_day_n_deposition(rain, self.env.sb3_env._site_params,)

        print(f"day_nh4_depo {day_nh4_depo}")
        print(f"day_no3_depo {day_no3_depo}")

        self.assertAlmostEqual(day_nh4_depo, 0.0014, 1)
        self.assertAlmostEqual(day_no3_depo, 0.0025, 1)

    def test_n_deposition_aggregation(self):
        year = 2000
        wdp = NASAPowerWeatherDataProvider(52.0, 5.5)
        timestep = 7

        daily_dates = generate_date_list(datetime.date(year, 1, 1), datetime.date(year, 1, timestep+1))
        rain = [wdp(day).RAIN * 10 for day in daily_dates]

        site_params = {'NO3ConcR': 2, 'NH4ConcR': 1}

        nh4_week_depo, no3_week_depo = get_aggregated_n_depo_days(timestep, rain, site_params)

        self.assertAlmostEqual(nh4_week_depo, 0.1561, 1)
        self.assertAlmostEqual(no3_week_depo,  0.31239, 1)

    def test_n_depo_from_pcse(self):
        year = 2000
        self.env.overwrite_year(year)
        self.env.reset()

        terminated = False

        while not terminated:
            _, _, terminated, _, _ = self.env.step(np.array([0]))

        no3_depo = get_no3_deposition_pcse(self.env.sb3_env.model.get_output())
        nh4_depo = get_nh4_deposition_pcse(self.env.sb3_env.model.get_output())

        self.assertAlmostEqual(no3_depo, 9.54, 0)
        self.assertAlmostEqual(nh4_depo, 16.66, 0)


class NitrogenUseEfficiency(unittest.TestCase):
    def setUp(self):
        self.nue1 = init_env.initialize_env_nue_reward()
        self.def1 = init_env.initialize_env_reward_dep()

    def test_disaggregate(self):
        start = datetime.date(2002, 1, 1)
        end = datetime.date(2002, 2, 1)

        nh4_depo, no3_depo = get_deposition_amount(2002)

        daily_nh4 = nh4_depo / 365
        daily_no3 = no3_depo / 365

        expected_nh4 = daily_nh4 * 31
        expected_no3 = daily_no3 * 31

        nh4_dis, no3_dis = get_disaggregated_deposition(year=2002, start_date=start, end_date=end)

        self.assertEqual(expected_nh4, nh4_dis)
        self.assertEqual(expected_no3, no3_dis)

    def test_nue_calc_in_other_rf(self, reward_func='DEP'):
        self.env_rew = init_env.initialize_env(reward=reward_func, pcse_env=2)
        self.env_rew.overwrite_year(2002)
        self.env_rew.reset()
        terminated = False

        while not terminated:
            _, _, terminated, _, infos = self.env_rew.step(np.array([1]))

        self.assertAlmostEqual(0.58, max(list(infos['NUE'].values())), 0)

    def test_nue_all_rf(self):
        self.test_nue_calc_in_other_rf('DEF')
        self.test_nue_calc_in_other_rf('HAR')
        self.test_nue_calc_in_other_rf('NUE')
        self.test_nue_calc_in_other_rf('FIN')
        self.test_nue_calc_in_other_rf('NUP')
        self.test_nue_calc_in_other_rf('DNU')
        self.test_nue_calc_in_other_rf('END')
        self.test_nue_calc_in_other_rf('GRO')
