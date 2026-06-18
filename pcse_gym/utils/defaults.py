def get_lintul_default_crop_features():
    return ["DVS", "TGROWTH", "LAI", "NUPTT", "TRAN", "TNSOIL", "TRAIN", "TRANRF", "WSO"]


def get_wofost_default_crop_features():
    return ["DVS", "TAGP", "LAI", "NuptakeTotal", "TRA", "NAVAIL", "SM", "RFTRA", "TWSO"]


def get_snomin_default_crop_features():
    """Observation features for WOFOST 8.1 with SNOMIN soil nitrogen dynamics."""
    return [
        "DVS",
        "LAI",
        "WSO",
        "NuptakeTotal",
        "NO3",
        "NH4",
        "NLOSSCUM",
        "NamountSO",
    ]


def get_default_crop_features(pcse_env=1):
    if pcse_env == 0:
        return get_lintul_default_crop_features()
    if pcse_env == 2:
        return get_snomin_default_crop_features()
    return get_wofost_default_crop_features()


def get_default_weather_features():
    return ["IRRAD", "TMIN", "RAIN"]


def get_default_action_features():
    return []


def get_default_location():
    return (52, 5.5)


def get_default_years():
    return [*range(1990, 2022)]


def get_default_train_years():
    return [year for year in get_default_years() if year % 2 == 1]


def get_default_test_years():
    return [year for year in get_default_years() if year % 2 == 0]


def get_default_action_space():
    import gymnasium as gym
    return gym.spaces.Discrete(3)


def get_snomin_action_space(n_levels=5):
    import gymnasium as gym
    return gym.spaces.Discrete(n_levels)


def get_model_name(pcse_env):
    return {0: "LINTUL", 1: "WOFOST", 2: "WOFOST-SNOMIN"}.get(pcse_env, "WOFOST")
