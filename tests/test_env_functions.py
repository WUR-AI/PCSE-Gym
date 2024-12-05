import unittest
import tests.initialize_env as init_env

class TestEnvs(unittest.TestCase):
    def setUp(self) -> None:
        self.env_init = init_env.initialize_env(random_init=True, seed=10, pcse_env=2)

    def test_random_init(self):
        list_nh4, list_no3 = self.env_init.generate_realistic_n()
        print(list_nh4, list_no3)

        sumnh4 = sum(list_nh4)
        sumno3 = sum(list_no3)
        print(sumnh4, sumno3)
        sum_n = sumnh4 + sumno3

        print(sum_n)

        self.assertAlmostEqual(4.999, sum_n, 0)
