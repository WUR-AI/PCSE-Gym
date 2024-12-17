import datetime
import os

import numpy as np
import yaml

import gymnasium as gym

import pcse

"""
    Gymnasium Environment built around the PCSE library for crop simulation
    Gym:  https://github.com/Farama-Foundation/Gymnasium
    PCSE: https://github.com/ajwdewit/pcse
    
    Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
    
"""


# def replace_years(agro_management, years):  # Deprecated; replaced by AgroManagementContainer
#     if not isinstance(years, list):
#         years = [years]
#
#     updated_agro_management = [
#         {k.replace(year=year): v for k, v in agro.items()}
#         for agro, year in zip(agro_management, years)
#     ]
#
#     def replace_year_value(d, year):
#         for k, v in d.items():
#             if isinstance(v, dict):
#                 replace_year_value(v, year)
#             else:
#                 if isinstance(v, datetime.date):
#                     up_dict = {k: v.replace(year=year)}
#                     d.update(up_dict)
#
#     for agro, year in zip(updated_agro_management, years):
#         replace_year_value(agro, year)
#     return updated_agro_management


class AgroManagementContainer:
    """
    Container to modify agromanagement file.
    """

    def __init__(self, agro_management: list):
        self.agro_structure = agro_management
        self.campaign_date: datetime.date = list(agro_management[0].keys())[0]
        self.crop_name: str = agro_management[0][self.campaign_date]["CropCalendar"][
            "crop_name"
        ]
        self.crop_variety: str = agro_management[0][self.campaign_date]["CropCalendar"][
            "variety_name"
        ]
        self.crop_start_date: datetime.date = agro_management[0][self.campaign_date][
            "CropCalendar"
        ]["crop_start_date"]
        self.crop_start_type: str = agro_management[0][self.campaign_date][
            "CropCalendar"
        ]["crop_start_type"]
        self.crop_end_date: datetime.date = agro_management[0][self.campaign_date][
            "CropCalendar"
        ]["crop_end_date"]
        self.crop_end_type: str = agro_management[0][self.campaign_date][
            "CropCalendar"
        ]["crop_end_type"]
        self.max_duration: int = agro_management[0][self.campaign_date]["CropCalendar"][
            "max_duration"
        ]

        self.structure = None
        self.build_structure()

    def build_structure(self):
        self.structure = yaml.load(
            f"""
                    - {self.campaign_date}:
                        CropCalendar:
                            crop_name: {self.crop_name}
                            variety_name: {self.crop_variety}
                            crop_start_date: {self.crop_start_date}
                            crop_start_type: {self.crop_start_type}
                            crop_end_date: {self.crop_end_date}
                            crop_end_type: {self.crop_end_type}
                            max_duration: {self.max_duration}
                        TimedEvents: null
                        StateEvents: null
                """,
            Loader=yaml.SafeLoader,
        )

    def replace_years(self, y):
        """
        Years replaced are the harvest date. Campaign start and sow date starts a year before.
        """
        if isinstance(y, list):
            y = y[0]
        if self.campaign_date.year == self.crop_end_date.year:
            yprev = y
        else:
            yprev = y - 1
        self.campaign_date = self.campaign_date.replace(year=yprev)
        self.crop_start_date = self.crop_start_date.replace(year=yprev)
        self.crop_end_date = self.crop_end_date.replace(year=y)

        self.build_structure()
        return self.structure

    def replace_sow_date(self, year, month, day):
        self.crop_start_date = self.crop_start_date.replace(
            year=year, month=month, day=day
        )

        self.build_structure()
        return self.structure

    def replace_harvest_date(self, year, month, day):
        self.crop_end_date = self.crop_end_date.replace(year=year, month=month, day=day)

        self.build_structure()
        return self.structure

    def replace_start_type(self, start):
        assert start == "sowing" or start == "emergence"
        self.crop_start_type = start

        self.build_structure()
        return self.structure

    def replace_variety_name(self, name="Winter_wheat_102"):
        self.crop_variety = name

        self.build_structure()
        return self.structure

    def start_sowing(self):
        if self.campaign_date.year == self.crop_end_date.year:
            self.campaign_date = datetime.date(self.crop_end_date.year - 1, 10, 1)
            self.crop_start_date = datetime.date(self.crop_end_date.year - 1, 10, 1)

        self.build_structure()

    def start_emergence(self):
        self.campaign_date = datetime.date(self.crop_end_date.year, 1, 1)
        self.crop_start_date = datetime.date(self.crop_end_date.year, 1, 1)

        self.build_structure()

    def get_start_type(self, start_type):
        self.start_emergence() if start_type == "emergence" else self.start_sowing()

    @property
    def get_structure(self):
        return self.structure

    @property
    def get_start_date(self):
        return self.crop_start_date

    @property
    def get_end_date(self):
        return self.crop_end_date


def get_weather_data_provider(location) -> pcse.db.NASAPowerWeatherDataProvider:
    wdp = pcse.db.NASAPowerWeatherDataProvider(*location)
    return wdp


class Engine(pcse.engine.Engine):
    """
    Wraps around the PCSE engine/crop model to set a flag when the simulation has terminated
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flag_terminated = False

    @property
    def terminated(self):
        return self._flag_terminated

    def _terminate_simulation(self, day):
        super()._terminate_simulation(day)
        self._flag_terminated = True


class PCSEEnv(gym.Env):
    """
    Create a new PCSE-Gym environment

    :param model_config: PCSE config file name (must be available in the pcse/conf/ folder inside the pcse library)
    :param agro_config: file name of the yaml file specifying the agro-management configuration
    :param crop_parameters: Can be specified in two ways:
                                - A path to the crop parameter file
                                  Will be read by a `pcse.fileinput.PCSEFileReader`
                                - An object that is directly passed to the `pcse.base.ParameterProvider`
    :param site_parameters: Can be specified in two ways:
                                - A path to the site parameter file
                                  Will be read by a `pcse.fileinput.PCSEFileReader`
                                - An object that is directly passed to the `pcse.base.ParameterProvider`
    :param soil_parameters: Can be specified in two ways:
                                - A path to the soil parameter file
                                  Will be read by a `pcse.fileinput.PCSEFileReader`
                                - An object that is directly passed to the `pcse.base.ParameterProvider`
    :param years: A single year, or list of years to get weather data for. If not set use year from agro_config
    :param location: latitude, longitude to get weather data for
    :param seed: A seed for the random number generators used in PCSE-Gym
    :param timestep: Number of days that are simulated during a single time step
    """

    _PATH_TO_FILE = os.path.dirname(os.path.realpath(__file__))
    _CONFIG_PATH = os.path.join(_PATH_TO_FILE, "configs")

    _DEFAULT_AGRO_FILE = "agromanagement_fertilization.yaml"
    _DEFAULT_CROP_FILE = "lintul3_winterwheat.crop"
    _DEFAULT_SITE_FILE = "lintul3_springwheat.site"
    _DEFAULT_SOIL_FILE = "lintul3_springwheat.soil"

    _DEFAULT_AGRO_FILE_PATH = os.path.join(_CONFIG_PATH, "agro", _DEFAULT_AGRO_FILE)
    _DEFAULT_CROP_FILE_PATH = os.path.join(_CONFIG_PATH, "crop", _DEFAULT_CROP_FILE)
    _DEFAULT_SITE_FILE_PATH = os.path.join(_CONFIG_PATH, "site", _DEFAULT_SITE_FILE)
    _DEFAULT_SOIL_FILE_PATH = os.path.join(_CONFIG_PATH, "soil", _DEFAULT_SOIL_FILE)

    _DEFAULT_CONFIG = "Lintul3.conf"

    def __init__(
        self,
        model_config: str = _DEFAULT_CONFIG,
        agro_config: str = _DEFAULT_AGRO_FILE_PATH,
        crop_parameters=_DEFAULT_CROP_FILE_PATH,
        site_parameters=_DEFAULT_SITE_FILE_PATH,
        soil_parameters=_DEFAULT_SOIL_FILE_PATH,
        years=None,
        location=None,
        seed: int = None,
        timestep: int = 1,
        **kwargs,
    ):

        assert timestep > 0

        # Optionally set the seed
        super().reset(seed=seed)

        # If any parameter files are specified as path, convert them to a suitable object for pcse
        if isinstance(crop_parameters, str):
            crop_parameters = pcse.fileinput.PCSEFileReader(crop_parameters)
        if isinstance(site_parameters, str):
            site_parameters = pcse.fileinput.PCSEFileReader(site_parameters)
        if isinstance(soil_parameters, str):
            soil_parameters = pcse.fileinput.PCSEFileReader(soil_parameters)

        # Set location
        if location is None:
            location = (52, 5.5)
        self._location = location
        self._timestep = timestep

        # Store the crop/soil/site parameters
        self._crop_params = crop_parameters
        self._site_params = site_parameters
        self._soil_params = soil_parameters

        # Store the agro-management config
        with open(agro_config, "r") as f:
            self._agro_management = yaml.load(f, Loader=yaml.SafeLoader)

        # Initialize Agromanagement Container Class
        self.agmt = AgroManagementContainer(self._agro_management)

        if years is not None:
            self._agro_management = self.agmt.replace_years(years)

        # Store the PCSE Engine config
        self._model_config = model_config

        # Get the weather data source
        self._weather_data_provider = get_weather_data_provider(self._location)

        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()

        # Use the config files to extract relevant settings
        model_config = pcse.util.ConfigurationLoader(model_config)
        self._output_variables = (
            model_config.OUTPUT_VARS
        )  # variables given by the PCSE model output
        self._summary_variables = (
            model_config.SUMMARY_OUTPUT_VARS
        )  # Summary variables are given at the end of a run
        self._weather_variables = list(pcse.base.weather.WeatherDataContainer.required)

        # Define Gym observation space
        self.observation_space = self._get_observation_space()
        # Define Gym action space
        self.action_space = self._get_action_space()

    def _init_pcse_model(self, *args, **kwargs) -> Engine:

        # Combine the config files in a single PCSE ParameterProvider object
        self._parameter_provider = pcse.base.ParameterProvider(
            cropdata=self._crop_params,
            sitedata=self._site_params,
            soildata=self._soil_params,
        )
        # Create a PCSE engine / crop growth model
        model = Engine(
            self._parameter_provider,
            self._weather_data_provider,
            self._agro_management,
            config=self._model_config,
        )
        # The model starts with output values for the initial date
        # The initial observation should contain output values for an entire timestep
        # If the timestep > 1, generate the remaining outputs by running the model
        if self._timestep > 1:
            model.run(days=self._timestep - 1)
        return model

    def _get_observation_space(self) -> gym.spaces.Space:
        space = gym.spaces.Dict(
            {
                "crop_model": self._get_observation_space_crop_model(),
                "weather": self._get_observation_space_weather(),
            }
        )
        return space

    def _get_observation_space_weather(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "IRRAD": gym.spaces.Box(0, np.inf, (self._timestep,)),
                "TMIN": gym.spaces.Box(-np.inf, np.inf, (self._timestep,)),
                "TMAX": gym.spaces.Box(-np.inf, np.inf, (self._timestep,)),
                "VAP": gym.spaces.Box(0, np.inf, (self._timestep,)),
                "RAIN": gym.spaces.Box(0, np.inf, (self._timestep,)),
                "E0": gym.spaces.Box(0, np.inf, (self._timestep,)),
                "ES0": gym.spaces.Box(0, np.inf, (self._timestep,)),
                "ET0": gym.spaces.Box(0, np.inf, (self._timestep,)),
                "WIND": gym.spaces.Box(0, np.inf, (self._timestep,)),
            }
        )

    def _get_observation_space_crop_model(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                var: gym.spaces.Box(0, np.inf, shape=(self._timestep,))
                for var in self._output_variables
            }
        )

    def _get_action_space(self) -> gym.spaces.Space:
        space = gym.spaces.Dict(
            {
                "irrigation": gym.spaces.Box(0, np.inf, shape=()),
                "N": gym.spaces.Box(0, np.inf, shape=()),
                "P": gym.spaces.Box(0, np.inf, shape=()),
                "K": gym.spaces.Box(0, np.inf, shape=()),
            }
        )
        return space

    """
    Properties of the crop model config file
    """

    @property
    def output_variables(self) -> list:
        return list(self._output_variables)

    @property
    def summary_variables(self) -> list:
        return list(self._summary_variables)

    @property
    def weather_variables(self):
        return list(self._weather_variables)

    """
    Properties derived from the agro management config:
    """

    @property
    def _campaigns(self) -> dict:
        return self._agro_management[0]

    @property
    def _first_campaign(self) -> dict:
        return self._campaigns[min(self._campaigns.keys())]

    @property
    def _last_campaign(self) -> dict:
        return self._campaigns[max(self._campaigns.keys())]

    @property
    def start_date(self) -> datetime.date:
        return self._model.agromanager.start_date

    @property
    def end_date(self) -> datetime.date:
        return self._model.agromanager.end_date

    """
    Other properties
    """

    @property
    def date(self) -> datetime.date:
        return self._model.day

    """
    Gym functions
    """

    def step(self, action) -> tuple:
        """
        Perform a single step in the Gym environment. The provided action is performed and the environment transitions
        from state s_t to s_t+1. Based on s_t+1 an observation and reward are generated.

        :param action: an action that respects the action space definition as described by `self._get_action_space()`
        :return: a 4-tuple containing
            - an observation that respects the observation space definition as described by `self._get_observation_space()`
            - a scalar reward
            - a boolean flag indicating whether the environment/simulation has ended
            - a dict containing extra info about the environment and state transition
        """

        # Create a dict for storing info
        info = dict()

        # Apply action
        if isinstance(action, np.ndarray):
            action = action[0]
        self._apply_action(action)

        # Run the crop growth model
        self._model.run(days=self._timestep)
        # Get the model output
        output = self._model.get_output()[-self._timestep :]
        info["days"] = [day["day"] for day in output]

        # Construct an observation and reward from the new environment state
        o = self._get_observation(output)
        r = self._get_reward()
        # Check whether the environment has terminated
        done = self._model.terminated
        if done:
            info["output_history"] = self._model.get_output()
            info["summary_output"] = self._model.get_summary_output()
            info["terminal_output"] = self._model.get_terminal_output()
        truncated = False
        terminated = done
        # Return all values
        return o, r, terminated, truncated, info

    def _apply_action(self, action):

        irrigation = action.get("irrigation", 0)
        N = action.get("N", 0)
        P = action.get("P", 0)
        K = action.get("K", 0)

        self._model._send_signal(
            signal=pcse.signals.irrigate,
            amount=irrigation,
            efficiency=0.8,
        )

        self._model._send_signal(
            signal=pcse.signals.apply_npk,
            N_amount=N,
            P_amount=P,
            K_amount=K,
            N_recovery=0.7,
            P_recovery=0.7,
            K_recovery=0.7,
        )

    def _get_observation(self, output) -> dict:
        """
        Generate an observation based on the current environment state

        :param output: the output of the model after the state transition
        :return: an observation. The default implementation returns a dict containing two dicts containing crop model
                 and weather data, respectively
        """

        # Get the datetime objects characterizing the specific days
        days = [day["day"] for day in output]

        # Get the output variables for each of the days
        crop_model_observation = {
            v: [day[v] for day in output] for v in self._output_variables
        }

        # Get the weather data of the passed days
        weather_data = [self._weather_data_provider(day) for day in days]
        # Cast the weather data into a dict
        weather_observation = {
            var: [getattr(weather_data[d], var) for d in range(len(days))]
            for var in self._weather_variables
        }

        o = {
            "crop_model": crop_model_observation,
            "weather": weather_observation,
        }

        return o

    def _get_reward(self, var="TWSO") -> float:
        """
        Generate a reward based on the current environment state

        :param var: the variable extracted from the model output
        :return: a scalar reward. The default implementation gives the increase in yield during the last state transition
                 if the environment is in its initial state, the initial yield is returned
        """

        output = self._model.get_output()
        # var = 'LAI'  # For debugging
        # Consider different cases:
        if len(output) == 0:  # The simulation has not started -> 0 reward
            return 0
        if (
            len(output) <= self._timestep
        ):  # Only one observation is made -> give initial yield as reward
            return output[-1][var] or 0
        else:  # Multiple observations are made -> give difference of yield of the last time steps
            last_index_previous_state = (
                np.ceil(len(output) / self._timestep).astype("int") - 1
            ) * self._timestep - 1
            return (output[-1][var] or 0) - (
                output[last_index_previous_state][var] or 0
            )

    def reset(
        self, *, seed: int = None, return_info: bool = False, options: dict = None
    ):
        """
        Reset the PCSE-Gym environment to its initial state

        :param seed:
        :param return_info: flag indicating whether an info dict should be returned
        :param options: optional dict containing options for reinitialization
        :return: depending on the `return_info` flag, an initial observation is returned or a two-tuple of the initial
                 observation and the info dict
        """

        # Optionally set the seed
        super().reset(seed=seed)

        # Create an info dict
        info = dict()

        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()
        output = self._model.get_output()[-self._timestep :]
        o = self._get_observation(output)
        info["date"] = self.date

        return o, info if return_info else o

    def render(self, mode="human"):
        pass  # Nothing to see here
