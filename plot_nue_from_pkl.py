import os
import pickle
import lib_programname
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple

from pcse_gym.utils.plotter import plot_nue_template, plot_variable, plot_fertilization_schedule
from pcse_gym.envs.rewards import calculate_nue
from pcse_gym.utils.nitrogen_helpers import input_nue, m2_to_ha


path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[0]

evaluate_dir = os.path.join(rootdir, "tensorboard_logs")


def make_df(rewards, nue, wso, fertilization):
    def combine_dicts(d1, d2, d3, d4):

        ds = [d1, d2, d3, d4]
        d = {}
        for k in d1.keys():
            d[k] = tuple(d[k] for d in ds)
        return d

    def check_location(tup):
        if tup == (52.57, 5.63):
            return 'PAGV'

    combined = combine_dicts(rewards, nue, wso, fertilization)
    model_idx = [k[0] for k in combined.keys()]
    year_idx = [k[1][0] for k in combined.keys()]
    location_idx = [check_location(k[1][1]) for k in combined.keys()]
    reward_values = [v[0] for v in combined.values()]
    nue_values = [v[1] for v in combined.values()]
    wso_values = [v[2] for v in combined.values()]
    fert_values = [v[3] for v in combined.values()]
    df = pd.DataFrame({'model': model_idx,
                       'year': year_idx,
                       'location': location_idx,
                       'NUE': nue_values,
                       'WSO': wso_values,
                       'reward': reward_values,
                       'fertilization': fert_values})
    df.set_index(['model', 'year', 'location'], inplace=True)
    print(df)
    df.to_excel(os.path.join(evaluate_dir, "nue1.xlsx"))
    return df


def convert_latex(df):
    df['WSO'] = df['WSO'].apply(lambda x: x / 1000)
    df.columns = ['NUE [-]', 'WSO [tons/ha]', 'Reward [-]', 'Nitrogen [kg/ha]']
    print(df.to_latex(column_format='lccrrr', float_format='%.2f'))


def main():
    dir_pkls = os.path.join(evaluate_dir, 'for_plots', '1990')
    pkl_list = os.listdir(dir_pkls)

    result_dict = {}
    for pkl in pkl_list:
        with open(os.path.join(dir_pkls, pkl), 'rb') as f:
            infos = pickle.load(f)
            result_dict[pkl[6:-4]] = infos

    print(list(result_dict['RPPO'].values())[0][0].keys())

    fertilization = {}
    no3_depo = {}
    nh4_depo = {}
    NSO = {}
    rewards = {}
    input_n = {}
    WSO_list = {}
    WSO =  {}
    NUE = {}
    fertilization_list= {}
    for name, v in result_dict.items():
        for yearloc, output in v.items():
            print(output[0].keys())
            fertilization_list[(name, yearloc)] = list(output[0]['fertilizer'].values())
            fertilization[(name, yearloc)] = np.cumsum(list(output[0]['fertilizer'].values()))[-1]
            no3_depo[(name, yearloc)] = list(output[0]['RNO3DEPOSTT'].values())[-1] / m2_to_ha
            nh4_depo[(name, yearloc)] = list(output[0]['RNH4DEPOSTT'].values())[-1] / m2_to_ha
            input_n[(name, yearloc)] = input_nue(n_input=fertilization[(name, yearloc)],
                                                 year=yearloc[0],
                                                 no3_depo=no3_depo[(name, yearloc)],
                                                 nh4_depo=nh4_depo[(name, yearloc)])
            NSO[(name, yearloc)] = list(output[0]['NamountSO'].values())[-1]
            WSO_list[(name, yearloc)] = [list(output[0]['WSO'].values())[i] for i in range(0, len(output[0]['WSO'].values())-2, 7)]
            WSO[(name, yearloc)] = list(output[0]['WSO'].values())[-1]
            rewards[(name, yearloc)] = np.cumsum(list(output[0]['reward'].values()))[-1]
            NUE[(name, yearloc)] = list(output[0]['NUE'].values())[-1]

    df = make_df(rewards, NUE, WSO, fertilization)

    convert_latex(df)

    print(fertilization_list)
    plt = plot_nue_template()

    for key, n_in, n_out in zip(input_n.keys(), input_n.values(), NSO.values()):
        plt.scatter(n_in, n_out, label=f"{key[0]} - {key[1][0]}", s=16)
    plt.legend()
    # plt.show()

    plot_fertilization_schedule(fertilization_list, WSO_list)


if __name__ == '__main__':
    main()
