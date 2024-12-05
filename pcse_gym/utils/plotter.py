import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
from mpl_toolkits.mplot3d import Axes3D

import pcse_gym.initialize_envs


def get_cumulative_variables():
    return ['fertilizer', 'reward']


def get_ylim_dict(n=32):
    def def_value():
        return None

    if n == 0:
        n = 32

    ylim = defaultdict(def_value)
    ylim['WSO'] = [0, 10000]
    ylim['TWSO'] = [0, 10000]
    ylim['measure_SM'] = [0, n]
    ylim['measure_TAGP'] = [0, n]
    ylim['measure_random'] = [0, n]
    ylim['measure_LAI'] = [0, n]
    ylim['measure_NuptakeTotal'] = [0, n]
    ylim['measure_NAVAIL'] = [0, n]
    ylim['measure_SM'] = [0, n]
    ylim['measure'] = [0, n]
    ylim['prob_SM'] = [0, 1.0]
    ylim['prob_TAGP'] = [0, 1.0]
    ylim['prob_random'] = [0, 1.0]
    ylim['prob_LAI'] = [0, 1.0]
    ylim['prob_NuptakeTotal'] = [0, 1.0]
    ylim['prob_NAVAIL'] = [0, 1.0]
    ylim['prob_SM'] = [0, 1.0]
    ylim['prob_measure'] = [0, 1.0]
    return ylim


def get_titles():
    def def_value(): return ("", "")

    return_dict = defaultdict(def_value)
    return_dict["DVS"] = ("Development stage", "-")
    return_dict["TGROWTH"] = ("Total biomass (above and below ground)", "g/m2")
    return_dict["LAI"] = ("Leaf area Index", "-")
    return_dict["NUPTT"] = ("Total nitrogen uptake", "gN/m2")
    return_dict["TRAN"] = ("Transpiration", "mm/day")
    return_dict["TIRRIG"] = ("Total irrigation", "mm")
    return_dict["TNSOIL"] = ("Total soil inorganic nitrogen", "gN/m2")
    return_dict["TRAIN"] = ("Total rainfall", "mm")
    return_dict["TRANRF"] = ("Transpiration reduction factor", "-")
    return_dict["TRUNOF"] = ("Total runoff", "mm")
    return_dict["TAGBM"] = ("Total aboveground biomass", "g/m2")
    return_dict["TTRAN"] = ("Total transpiration", "mm")
    return_dict["WC"] = ("Soil water content", "m3/m3")
    return_dict["WLVD"] = ("Weight dead leaves", "g/m2")
    return_dict["WLVG"] = ("Weight green leaves", "g/m2")
    return_dict["WRT"] = ("Weight roots", "g/m2")
    return_dict["WSO"] = ("Weight storage organs", "g/m2")
    return_dict["TWSO"] = ("Weight storage organs", "kg/ha")
    return_dict["WST"] = ("Weight stems", "g/m2")
    return_dict["TGROWTHr"] = ("Growth rate", "g/m2/day")
    return_dict["NRF"] = ("Nitrogen reduction factor", "-")
    return_dict["GRF"] = ("Growth reduction factor", "-")

    return_dict["DVS"] = ("Development stage", "-")
    return_dict["TAGP"] = ("Total above-ground Production", "kg/ha")
    return_dict["LAI"] = ("Leaf area Index", "-")
    return_dict["RNuptake"] = ("Total nitrogen uptake", "kgN/ha")
    return_dict["TRA"] = ("Transpiration", "cm/day")
    return_dict["NAVAIL"] = ("Total soil inorganic nitrogen", "kgN/ha")
    return_dict["SM"] = ("Volumetric soil moisture content", "-")
    return_dict["RFTRA"] = ("Transpiration reduction factor", "-")
    return_dict["TRUNOF"] = ("Total runoff", "mm")
    return_dict["TAGBM"] = ("Total aboveground biomass", "kg/ha")
    return_dict["TTRAN"] = ("Total transpiration", "mm")
    return_dict["WC"] = ("Soil water content", "m3/m3")
    return_dict["Ndemand"] = ("Total N demand of crop", "kgN/ha")
    return_dict["NuptakeTotal"] = ("Total N uptake of crop", "kgN/ha/d")
    return_dict["FERT_N_SUPPLY"] = ("Total N supplied by actions", "kgN/ha")

    return_dict["fertilizer"] = ("Nitrogen application", "kg/ha")
    return_dict["TMIN"] = ("Minimum temperature", "°C")
    return_dict["TMAX"] = ("Maximum temperature", "°C")
    return_dict["IRRAD"] = ("Incoming global radiation", "J/m2/day")
    return_dict["RAIN"] = ("Daily rainfall", "cm/day")
    return_dict["NO3"] = ("Nitrate Levels (NO3)", "kgN/ha")
    return_dict["NH4"] = ("Ammonium Levels (NH4)", "kgN/ha")
    return_dict["NLOSSCUM"] = ("Cumulative N loss", "kgN/ha")

    return return_dict


def plot_year_loc_heatmap(results_dict, variable, year_locs, cumulative_variables=get_cumulative_variables(), ax=None,
                          fig=None, ylim=None,
                          put_legend=True):
    titles = get_titles()
    xmax = 0
    xmin = 9999
    check = []

    # generator for date of year
    def doy_sender(_i):
        gen.send(None)
        return gen.send(_i)

    # structure x using DVS
    for label, results in results_dict.items():
        x, _ = zip(*results[0]['DVS'].items())
        x = ([i.timetuple().tm_yday for i in x])
        inc = all(i < j for i, j in zip(x, x[1:]))
        if not inc:
            x = restructure_x(x)
        if max(x) > xmax: xmax = max(x)
        if min(x) < xmin: xmin = min(x)

    # make heatmap
    if isinstance(variable, list):
        dataframes_list = pd.DataFrame()
        for v in variable:
            gen = doy_generator()
            df = pd.DataFrame.from_dict(results_dict[year_locs][0][v], orient='index', columns=[v])
            df = df.rename(lambda i: doy_sender(i.timetuple().tm_yday))
            dataframes_list = pd.concat([dataframes_list, df], axis=1)

        dataframes_list = dataframes_list.T

        left, right, down, up = dataframes_list.columns[0], dataframes_list.columns[-1], 0, len(dataframes_list.index)
        extent = [left - 6, right + 6, down - 0.5, up - 0.5]

        # plot heatmap
        heatmap = ax.imshow(dataframes_list, aspect='auto', cmap='RdYlBu', extent=extent, origin='lower')

        print(dataframes_list)
        ax.set_yticks(np.arange(0, len(dataframes_list.index)))
        ax.set_yticklabels(dataframes_list.index)
        ax.yaxis.grid(which="minor", color="black", linestyle=':', linewidth=1)

        ax.set_title('measuring actions', fontsize="8.5")


    else:
        gen = doy_generator()  # restart the generator
        df = pd.DataFrame.from_dict(results_dict[year_locs][0][variable], orient='index', columns=[variable])
        df = df.rename(lambda i: doy_sender(i.timetuple().tm_yday))

        ax.step(df.index, df[variable], where='post')

        ax.axhline(y=0, color='lightgrey', zorder=1)
        ax.margins(x=0)

    from matplotlib.ticker import FixedLocator
    ax.xaxis.set_minor_locator(FixedLocator(range(0, xmax, 7)))
    ax.xaxis.grid(True, which='minor')
    ax.tick_params(axis='x', which='minor', grid_alpha=0.7, colors=ax.get_figure().get_facecolor(), grid_ls=":")

    months, month_days = ticks_checker(inc, xmin, xmax)
    ax.set_xticks(month_days)
    ax.set_xticklabels(months)

    if isinstance(variable, list):
        name, unit = "crop variables", ""
        ax.set_title(f"{year_locs} - {name}", fontsize='8.5')
    else:
        name, unit = titles[variable]
        ax.set_title(f"{variable} - {name}", fontsize='8.5')
        ax.set_ylabel(f"[{unit}]")
    if variable in cumulative_variables:
        ax.set_title(f"{variable} (cumulative) - {name}")

    return ax


def restructure_x(day_nums):
    # sanity check
    # if number resets to 1, add subsequent number with previous so on
    offset = 0
    new_num = []
    for i, n in enumerate(day_nums):
        if i > 0 and day_nums[i] < day_nums[i - 1]:
            offset += day_nums[i - 1]
        new_num.append(n + offset)
    return new_num


def month_of_year_ind(day_of_year):
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] * 2
    cumulative_days = 0
    for i, length in enumerate(month_lengths):
        cumulative_days += length
        # sanity check
        # if our day_of_year is less than the cumulative days,
        # we've found our month
        if day_of_year <= cumulative_days:
            return i
    else:
        return None


def ticks_checker(inc_flag, _xmin, xmax):
    mons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    mon_days = [0, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335,
                366, 397, 425, 456, 486, 517, 547, 578, 609, 639, 670, 700]
    if not inc_flag:
        mons = mons * 2
        _extra_month = next(m[0] for m in enumerate(mon_days) if m[1] >= xmax)
        _xmin = month_of_year_ind(_xmin)
        mon_days = mon_days[_xmin:_extra_month + 1]
        mons = mons[_xmin:_extra_month + 1]
    else:
        _extra_month = next(m[0] for m in enumerate(mon_days) if m[1] >= xmax)
        mon_days = mon_days[0:_extra_month + 1]
        mons = mons[0:_extra_month + 1]
    return mons, mon_days


def doy_generator():
    # sanity check
    # starts in Oct (274), resets in Jan (1), continue with offset for rest
    # new dataframe starts in Oct (274)
    last_value = None
    while True:
        current_value = (yield)
        if last_value is None or (last_value < current_value):
            last_value = current_value
        elif last_value > current_value:
            offset = last_value
            current_value += offset
        yield current_value


def plot_variable(results_dict, variable='reward', cumulative_variables=get_cumulative_variables(), ax=None, ylim=None,
                  put_legend=True, plot_average=False, pcse_env=2, plot_heatmap=False):
    titles = get_titles()
    xmax = 0
    xmin = 9999
    check = []

    # generator for date of year
    def doy_sender(_i):
        gen.send(None)
        return gen.send(_i)

    # structure x and y ticks
    for label, results in results_dict.items():
        x, y = zip(*results[0][variable].items())
        x = ([i.timetuple().tm_yday for i in x])
        inc = all(i < j for i, j in zip(x, x[1:]))
        if not inc:
            x = restructure_x(x)
        if variable in cumulative_variables: y = np.cumsum(y)
        if max(x) > xmax: xmax = max(x)
        if min(x) < xmin: xmin = min(x)
        if not plot_average:
            ax.step(x, y, label=label, where='post')

    where = 'post'

    if plot_average:
        # get top soil layer
        if variable in ['NH4', 'NO3'] and pcse_env == 2:
            for k, v in results_dict.items():
                for key in v[0][variable].keys():
                    results_dict[k][0][variable][key] = sum(results_dict[k][0][variable][key])
        elif variable in ['SM', 'WC'] and pcse_env == 2:
            for k, v in results_dict.items():
                for key in v[0][variable].keys():
                    results_dict[k][0][variable][key] = results_dict[k][0][variable][key][0]
        dataframes_list = []
        for label, results in results_dict.items():
            gen = doy_generator()  # restart the generator
            df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
            df = df.rename(lambda i: doy_sender(i.timetuple().tm_yday))
            dataframes_list.append(df)

        plot_df = pd.concat(dataframes_list, axis=1)
        plot_df.sort_index(inplace=True)
        if variable in cumulative_variables: plot_df = plot_df.apply(np.cumsum, axis=0)
        if variable.startswith("measure"):
            plot_df.ffill(inplace=True)
            ax.step(plot_df.index, plot_df.sum(axis=1), 'k-', where=where)
            ax.fill_between(plot_df.index, plot_df.min(axis=1), plot_df.sum(axis=1), color='g', step=where)
        elif variable == 'action':
            plot_df.fillna(0, inplace=True)
            ax.step(plot_df.index, plot_df.median(axis=1), 'k-', where=where)
            ax.fill_between(plot_df.index, plot_df.quantile(0.25, axis=1), plot_df.quantile(0.75, axis=1), step=where)
        else:
            plot_df.ffill(axis=0, inplace=True)
            ax.step(plot_df.index, plot_df.median(axis=1), 'k-', where=where)
            ax.fill_between(plot_df.index, plot_df.quantile(0.25, axis=1), plot_df.quantile(0.75, axis=1), step=where)

    ax.axhline(y=0, color='lightgrey', zorder=1)
    ax.margins(x=0)

    from matplotlib.ticker import FixedLocator
    ax.xaxis.set_minor_locator(FixedLocator(range(0, xmax, 7)))
    ax.xaxis.grid(True, which='minor')
    ax.tick_params(axis='x', which='minor', grid_alpha=0.7, colors=ax.get_figure().get_facecolor(), grid_ls=":")

    months, month_days = ticks_checker(inc, xmin, xmax)
    ax.set_xticks(month_days)
    ax.set_xticklabels(months)

    name, unit = titles[variable]
    ax.set_title(f"{variable} - {name}")
    if variable in cumulative_variables:
        ax.set_title(f"{variable} (cumulative) - {name}")
    ax.set_ylabel(f"[{unit}]")
    if ylim is not None:
        ax.set_ylim(ylim)
    if put_legend:
        ax.legend()
    else:
        ax.legend()
        ax.get_legend().set_visible(False)
    return ax


def plot_var_vs_freq(results_dict, variable='measure_LAI', ax=None, ylim=None,
                     put_legend=True, n_year_loc=32):
    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        dataframes_list.append(df)

    plot_measure = pd.concat(dataframes_list, axis=1)
    plot_measure = plot_measure.sum(axis=1)

    variable = variable.replace("measure_", "")

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        dataframes_list.append(df)

    plot_var = pd.concat(dataframes_list, axis=1)
    plot_var = plot_var.std(ddof=0, axis=1)

    plot_df = pd.concat([plot_measure, plot_var], axis=1)
    plot_df.dropna(inplace=True)
    plot_df = plot_df.rename(columns={0: 'measure', 1: 'variance'})

    ax.scatter(plot_df['measure'], plot_df['variance'], c='green', alpha=0.5, linewidths=1, edgecolors='black')

    ax.set_title(f'measuring actions vs variance for {variable}')

    ax.set_xticks(np.arange(0, n_year_loc + 1))
    ax.set_xlabel(f'measuring frequency across years and locations')

    ax.set_ylabel(f'variance across years and locations')

    return ax


def plot_var_vs_freq_box(results_dict, variable='measure_LAI', ax=None, ylim=None,
                         put_legend=True, n_year_loc=32):
    titles = get_titles()

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        df.index = df.index.map(lambda d: d.strftime('%m-%d'))
        dataframes_list.append(df)

    plot_measure = pd.concat(dataframes_list, axis=1)
    plot_measure = plot_measure.sum(axis=1)

    variable = variable.replace("measure_", "")

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        df.index = df.index.map(lambda d: d.strftime('%m-%d'))
        dataframes_list.append(df)

    plot_var = pd.concat(dataframes_list, axis=1)

    plot_df = pd.concat([plot_measure, plot_var], axis=1)
    # plot_df.dropna(inplace=True)
    plot_df = plot_df.rename(columns={0: 'measure'})

    col1_values = range(int(max(plot_df['measure'])) + 1)

    # Initialize a dictionary to hold boxplot data
    boxplot_data = {i: [] for i in col1_values}

    for col in plot_df.columns[1:]:
        for val in col1_values:
            matched_data = plot_df[plot_df['measure'] == val][col].dropna()
            if not matched_data.empty:
                boxplot_data[val].extend(matched_data)

    bp = ax.boxplot(boxplot_data.values(), patch_artist=True, positions=list(boxplot_data.keys()))

    colors = plt.cm.viridis(np.linspace(0, 1, len(boxplot_data)))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.5)

    ax.grid(True, linestyle='--', alpha=0.7)

    name, unit = titles[variable]
    ax.set_title(f'measuring actions for {variable}')

    ax.set_xticklabels(list(boxplot_data.keys()), rotation=315, fontsize=7)
    ax.set_xlabel(f'measuring frequency')

    ax.set_ylabel(f'{variable} [{unit}] variance across years and locations')

    return ax


def plot_var_vs_freq_scatter(results_dict, variable='measure_LAI', ax=None):
    if variable.startswith('measure_'):
        variable_type = 'm'
    else:
        variable_type = 'p'

    # Function to find the nearest weekly date
    def nearest_weekly_date(date, base_date):
        days_difference = (date - base_date).days
        nearest_weekly_difference = round(days_difference / 7) * 7
        return base_date + pd.Timedelta(days=nearest_weekly_difference)

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        df.index = df.index.map(lambda d: d.strftime('%m-%d'))
        dataframes_list.append(df)

    plot_measure = pd.concat(dataframes_list, axis=1)
    plot_measure = plot_measure.sum(axis=1)

    if variable_type == 'm':
        variable = variable.replace("measure_", "")
    else:
        variable = variable.replace("prob_", "")

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        df.index = df.index.map(lambda d: d.strftime('%m-%d'))
        dataframes_list.append(df)

    plot_var = pd.concat(dataframes_list, axis=1)
    plot_var = plot_var.std(ddof=0, axis=1)

    plot_df = pd.concat([plot_var, plot_measure], axis=1)
    plot_df.dropna(inplace=True)
    plot_df = plot_df.rename(columns={0: 'variance', 1: 'measure'})

    plot_df['Date'] = plot_df.index

    base_year = 2021  # placeholder non-leap year
    base_date = pd.to_datetime(f'{base_year}-10-01')  # Starting date (1st of October)
    plot_df['Date'] = pd.to_datetime(plot_df['Date'] + f'-{base_year}', format='%m-%d-%Y')

    # Apply the function to assign each date to the nearest weekly date
    plot_df['Nearest_Weekly_Date'] = plot_df['Date'].apply(lambda d: nearest_weekly_date(d, base_date))
    result = plot_df.groupby('Nearest_Weekly_Date').agg({'variance': 'mean', 'measure': 'sum'})

    # Convert the index back to MM-DD format
    result.index = result.index.map(lambda d: d.strftime('%m-%d'))
    result = result.sort_values(by=['variance'])

    result['variance'] = (result['variance'] - result['variance'].min()) / (
            result['variance'].max() - result['variance'].min())

    ax.scatter(result['variance'], result['measure'], edgecolor='black', alpha=0.7)

    titles = get_titles()

    ax.set_xticks(np.arange(min(result['variance']), max(result['variance']) + 0.1, 0.1))

    # ax.set_xticklabels(range(0, int(max(result['variance']))), rotation=315, fontsize=8)

    name, unit = titles[variable]

    ax.set_ylim([-1, 33])

    if variable_type == 'm':
        ax.set_title(f'{variable} variance vs measuring frequency', fontsize=10, fontweight='bold', color='green')
        ax.set_ylabel(f'measuring frequency between test years', fontsize=8)
    else:
        ax.set_title(f'{variable} variance vs measuring probability', fontsize=10, fontweight='bold', color='green')
        ax.set_ylabel(f'probability of measuring between test years', fontsize=8)

    ax.set_xlabel(f'Normalized {variable} [{unit}] variance across test years', fontsize=8)

    return ax


def weather_check(nso, nloss, n_up, navail):
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    # get weather analysis for demeter

    def get_k(year):
        return 'demeter', (year, (52.57, 5.63))

    def change_date(year):
        start_date = datetime.date(year - 1, 10, 3)
        end_date = datetime.date(year, 8, 20)

        # Generate weekly date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='W-SUN').date

        return date_range

    dates = change_date(2020)

    with open(os.path.join(os.path.dirname(__file__), 'weather_utils', 'weather_csv', 'weather1983-2021.csv'), 'r') as f:
        weather_data = pd.read_csv(f)

    weather_data['DAY'] = pd.to_datetime(weather_data['DAY'], format='%Y%m%d')
    # weather_data['DAY'] = weather_data['DAY']

    date_set = set(dates)

    # Filter using a lambda function to check membership in date_set
    filtered_weather_data = weather_data[weather_data['DAY'].isin(date_set)]

    fig, ax = plt.subplots(5, 1, figsize=[8, 6])

    ax[0].plot(dates, nso[get_k(2020)], label='NSO')
    ax[1].plot(dates, nloss[get_k(2020)], label='N loss')
    ax[2].plot(dates, n_up[get_k(2020)], label='N uptake')
    ax[3].plot(dates, filtered_weather_data['RAIN'], label='Precipitation')
    ax[4].plot(dates, navail[get_k(2020)], label='N soil availability')

    for i, _ in enumerate(ax):
        ax[i].xaxis.set_major_locator(mdates.MonthLocator())
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[i].legend()

    # plt.legend()
    plt.tight_layout()
    plt.show()

def plot_nue_template(show_graph_labels=False, size=(6, 6), max=300, get_return=True, ax=None) -> plt:
    l = max

    n_in = np.linspace(0, l, l)

    # For NUE = 50%, the output is the same as the input
    n_output_50 = n_in * 0.5

    # For NUE = 90%, the output is 90% of the input
    n_output_90 = n_in * 0.9

    # Desired minimum productivity (N output > 80 kg/ha/yr)
    min_productivity_output = np.full(n_in.shape, 80)

    max_surplus_line = n_in - 40

    max_surplus_line[max_surplus_line < 0] = 0

    if not ax:
        plt.figure(figsize=size)

        plt.plot(n_in, max_surplus_line, 'k:')  #, label='N surplus = 40 kg/ha/yr')

        plt.fill_between(n_in, n_output_50, max_surplus_line, color='yellow',
                         alpha=0.5)  #, label='N surplus > 40 kg/ha/yr')

        plt.plot(n_in, n_output_50, 'k--')  #, label='NUE = 50%')

        plt.plot(n_in, n_output_90, 'k-')  #, label='NUE = 90%')

        # plt.plot(n_in, min_productivity_output, 'k-.', label='Desired minimum productivity (N output > 80 kg/ha/yr)')

        plt.fill_between(n_in, 0, n_output_50, color='red', alpha=0.3)
        # label='NUE very low (NUE < 50%): Risk of inefficient N use')

        plt.fill_between(n_in, n_output_90, l, color='orange', alpha=0.3)
        # label='NUE very high (NUE > 90%): Risk of soil mning')
        plt.xlabel('Nitrogen Input [kgN/ha]')
        plt.ylabel('Nitrogen Output [kgN/ha]')
        if get_return:
            return plt
    else:
        ax.plot(n_in, max_surplus_line, 'k:')  # , label='N surplus = 40 kg/ha/yr')

        ax.fill_between(n_in, n_output_50, max_surplus_line, color='yellow',
                         alpha=0.5)  # , label='N surplus > 40 kg/ha/yr')

        ax.plot(n_in, n_output_50, 'k--')  # , label='NUE = 50%')

        ax.plot(n_in, n_output_90, 'k-')  # , label='NUE = 90%')

        # plt.plot(n_in, min_productivity_output, 'k-.', label='Desired minimum productivity (N output > 80 kg/ha/yr)')

        ax.fill_between(n_in, 0, n_output_50, color='red', alpha=0.3)
        # label='NUE very low (NUE < 50%): Risk of inefficient N use')

        ax.fill_between(n_in, n_output_90, l, color='orange', alpha=0.3)
        # label='NUE very high (NUE > 90%): Risk of soil mning')
        # plt.xlabel('Nitrogen Input [kgN/ha]')
        # plt.ylabel('Nitrogen Output [kgN/ha]')



def plot_years_within_metrics(nue_dict, all_models):
    def subtract_list(a, b):

        return [a_i - b_i for a_i, b_i in zip(a, b)]

    fig, (ax_nue, ax_nsurp) = plt.subplots(2, 1, figsize=(10, 10))

    if all_models:
        ax_nue.bar(nue_dict['categories'], nue_dict['all_nue'], width=0.5, label='Efficient years', color='green')
        ax_nue.bar(nue_dict['categories'],
                   subtract_list([nue_dict['len_years']] * len(nue_dict['all_nue']), nue_dict['all_nue']),
                   bottom=nue_dict['all_nue'], width=0.5,
                   label='Inefficient years', color='orange')
        ax_nue.set_ylabel('Number of years')
        ax_nue.set_xlabel('Agent')
        ax_nue.legend()

        ax_nsurp.bar(nue_dict['categories'], nue_dict['all_n_surplus'], width=0.5,
                     label='Years below 40 kg/ha N surplus',
                     color='green')
        ax_nsurp.bar(nue_dict['categories'],
                     subtract_list([nue_dict['len_years']] * len(nue_dict['all_n_surplus']), nue_dict['all_n_surplus']),
                     bottom=nue_dict['all_n_surplus'], width=0.5,
                     label='Years above 40 kg/ha N surplus', color='orange')
        ax_nsurp.set_ylabel('Number of years')
        ax_nsurp.set_xlabel('Agent')
        ax_nsurp.legend()
    else:
        ax_nue.bar(nue_dict['categories'], nue_dict['nue'], width=0.5, label='Efficient years', color='green')
        ax_nue.bar(nue_dict['categories'], nue_dict['len_years'] - nue_dict['nue'], bottom=nue_dict['nue'], width=0.5,
                   label='Inefficient years', color='orange')
        ax_nue.set_ylabel('Number of years')
        ax_nue.set_xlabel('Agent')
        ax_nue.legend()

        ax_nsurp.bar(nue_dict['categories'], nue_dict['n_surplus'], width=0.5, label='Years below 40 kg/ha N surplus',
                     color='green')
        ax_nsurp.bar(nue_dict['categories'], nue_dict['len_years'] - nue_dict['n_surplus'],
                     bottom=nue_dict['n_surplus'], width=0.5,
                     label='Years above 40 kg/ha N surplus', color='orange')
        ax_nsurp.set_ylabel('Number of years')
        ax_nsurp.set_xlabel('Agent')
        ax_nsurp.legend()

    ax_nue.set_title('Nitrogen Use Efficiency')
    ax_nsurp.set_title('Nitrogen Surplus')
    plt.tight_layout()
    plt.show()


def plot_fertilization_schedule(fertilization_policies, growth, NUE, Nsurplus, dvs, dict_good=None, year=2020):
    import matplotlib.dates as mdates

    def filter_dict(dict_):
        return {key: value for key, value in dict_.items() if
                year in key[1] and key[0] in ['demeter', 'N2-PA', good_model, relative_model]}

    def label_name(k, nue, nsurp, growth):
        if 'LagPPO' in k[0]:
            model_name = 'NUE Agent'
        if 'demeter' in k[0]:
            model_name = 'Demeter'
        if 'N2-PA' in k[0]:
            model_name = 'N2'
        if 'def' in k[0]:
            model_name = 'Relative-yield Agent'

        # name = model_name + '; ' + 'NUE: ' + str(round(nue[k], 2)) + '; ' + 'N Surplus: ' + str(
        #     round(nsurp[k], 1)) + '; ' + 'Yield: ' + str(round((growth[k][-1] / 1000), 1)) + ' Tons/ha'
        name = model_name + '; ' + 'Cumulative applied N: ' + str(round(sum(fertilization_policies[(k[0], (year, (52.57, 5.63)))]), 1)) + 'kg/ha'

        return name

    colors = ['tomato', 'snow', 'darkslategray', 'peru']

    start_date = datetime.date(year-1, 10, 3)
    end_date = datetime.date(year, 8, 20)

    # Generate weekly date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='W-SUN').date

    fig, ax1 = plt.subplots(figsize=(9, 6))

    good_model = 'LagPPO8'  # dict_good['good_model']
    relative_model = 'def4'

    fertilization_policies = filter_dict(fertilization_policies)
    growth = filter_dict(growth)
    NUE = filter_dict(NUE)
    Nsurplus = filter_dict(Nsurplus)
    dvs = filter_dict(dvs)

    offsets = np.linspace(-3, 3, len(fertilization_policies))
    for i, (agent, agent_v) in enumerate(fertilization_policies.items()):
        shifted_dates = date_range + pd.to_timedelta(offsets[i], unit='D')
        ax1.bar(shifted_dates, agent_v, width=3, alpha=.9, label=label_name(agent, NUE, Nsurplus, growth),
                color=colors[i], edgecolor='black')
    ax1.set_ylabel('Fertilizer applications [kg/ha]')
    ax1.yaxis.grid(True, which='major', linestyle='--')

    # Plot fertilization policy as vertical bars
    ax2 = ax1.twinx()

    dvs = dvs[('demeter', (year, (52.57, 5.63)))]
    idx_dvs = next(i for i, value in enumerate(dvs) if value > 1)

    # # # Plot yield data
    # for agent, agent_v in dvs.items():
    #     if len(date_range) < len(agent_v):
    #         agent_v.pop(0)
    #     ax2.plot(date_range, agent_v, alpha=.9)
    # ax2.set_xlabel('Date in growing year')
    # ax2.set_ylabel('Development Stage [-]')
    # ax2.yaxis.grid(False)

    with open(os.path.join(os.path.dirname(__file__), 'weather_utils', 'weather_csv', 'weather2020.csv'), 'r') as f:
        weather_data = pd.read_csv(f)

    weather_data['DAY'] = pd.to_datetime(weather_data['DAY'], format='%Y%m%d')
    # weather_data['DAY'] = weather_data['DAY']

    date_set = set(date_range)

    # Filter using a lambda function to check membership in date_set
    filtered_weather_data = weather_data[weather_data['DAY'].isin(date_set)]


    ax2.plot(date_range, filtered_weather_data['RAIN'], color='tab:green')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.set_xlabel('Day')

    ax1.axvline(date_range[idx_dvs], color='black', linestyle='--', label='Flowering Date', linewidth=1.5)

    ax1.legend(loc='upper left')

    # Set x-axis major ticks format
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax1.set_ylim([0, 100])


    # Set x-axis major ticks format
    # ax2.xaxis.set_major_locator(mdates.MonthLocator())
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # plt.grid(True)
    # plt.title(f'Fertilization Policy in year {str(year)}')
    plt.tight_layout()
    # plt.legend(loc='lower right')
    plt.show()


def plot_nue_boxes(df, dict_good):
    import matplotlib.patches as mpatches

    def make_label_nue(agent, i):
        return dict_good['categories'][i] + ' (' + str(dict_good[agent + '_nue_good']) + '/39)'

    def make_label_nsurp(agent, i):
        return dict_good['categories'][i] + ' (' + str(dict_good[agent + '_nsurplus_good']) + '/39)'

    fig, (ax_nue, ax_nsurp) = plt.subplots(2, 1, figsize=(13, 13))
    colors = ['darkslategray', 'peru', 'snow', 'tomato', 'mediumaquamarine', 'slateblue']
    model_list = dict_good['categories']

    custom_order = dict_good['good_model']
    #
    # index_sorter = {key: i for i, key in enumerate(custom_order)}
    #
    # df = df.sort_index(level='model', axis=0, key=lambda x: x.map(index_sorter))
    df['models'] = pd.Categorical(df.index.get_level_values(0), categories=custom_order, ordered=True)

    # df_filtered = df[df['model'].isin(model_list)]

    ax_nue.axhline(y=0.5, lw=1, color='black')
    ax_nue.axhline(y=0.9, lw=1, color='black')

    ax_nsurp.axhline(y=0.0, lw=1, color='black')
    ax_nsurp.axhline(y=40.0, lw=1, color='black')

    box_nue = df.boxplot(ax=ax_nue, column='nue_value', by='models',
                         grid=False, patch_artist=True, return_type='both',
                         flierprops={'markeredgewidth': 2},)
    box_nsurp = df.boxplot(ax=ax_nsurp, column='nsurplus_value', by='models',
                           grid=False, patch_artist=True, return_type='both',
                           flierprops={'markeredgewidth': 2},)

    plt.suptitle('')
    ax_nue.set_title('')
    ax_nsurp.set_title('')

    ax_nue.set_xticklabels(model_list, rotation=12, size=18)
    ax_nsurp.set_xticklabels(model_list, rotation=12, size=18)

    line_width = 2.2

    for ax in box_nue['nue_value']:
        if isinstance(ax, dict):
            for i, box in enumerate(ax['boxes']):
                box.set_facecolor(colors[i])
                box.set_edgecolor('black')
                box.set_linewidth(line_width)
            for median in ax['medians']:
                median.set_color('black')  # Set the color of the median to black
                median.set_linewidth(line_width + 0.3)
            for caps in ax['caps']:
                caps.set_color('black')
                caps.set_linewidth(line_width)
            for whisk in ax['whiskers']:
                whisk.set_color('black')
                whisk.set_linewidth(line_width)
            for flier in ax['fliers']:
                flier.set_color('black')
                flier.set_linewidth(line_width)

    for ax in box_nsurp['nsurplus_value']:
        if isinstance(ax, dict):
            for i, box in enumerate(ax['boxes']):
                box.set_facecolor(colors[i])
                box.set_edgecolor('black')
                box.set_linewidth(line_width)
            for median in ax['medians']:
                median.set_color('black')  # Set the color of the median to black
                median.set_linewidth(line_width + 0.3)
            for caps in ax['caps']:
                caps.set_color('black')
                caps.set_linewidth(line_width)
            for whisk in ax['whiskers']:
                whisk.set_color('black')
                whisk.set_linewidth(line_width)
            for flier in ax['fliers']:
                flier.set_color('black')
                flier.set_linewidth(line_width)

    # set limits for fill
    ax_nue.set_ylim([0.2, 1.3])
    ax_nsurp.set_ylim([-100, 300])
    # ax_nue.set_xlim([1982, 2022])
    # ax_nsurp.set_xlim([1982, 2022])

    color_fill = 'salmon'
    ax_nue.fill_between(x=ax_nue.get_xlim(), y1=0.9, y2=ax_nue.get_ylim()[1], alpha=0.35, color=color_fill)
    ax_nue.fill_between(x=ax_nue.get_xlim(), y1=0.5, y2=ax_nue.get_ylim()[0], alpha=0.35, color=color_fill)

    ax_nsurp.fill_between(x=ax_nsurp.get_xlim(), y1=40, y2=ax_nsurp.get_ylim()[1], alpha=0.35, color=color_fill)
    ax_nsurp.fill_between(x=ax_nsurp.get_xlim(), y1=0, y2=ax_nsurp.get_ylim()[0], alpha=0.35, color=color_fill)

    legend_patches_nue = [mpatches.Patch(facecolor=color, edgecolor='black', label=make_label_nue(agent, i)) for
                          i, (color, label, agent) in
                          enumerate(zip(colors, model_list, dict_good['good_model']))]
    legend_patches_nsurp = [mpatches.Patch(facecolor=color, edgecolor='black', label=make_label_nsurp(agent, i)) for
                            i, (color, label, agent) in
                            enumerate(zip(colors, model_list, dict_good['good_model']))]

    # ax_nue.set_title('Nitrogen Use Efficiency of each year')
    ax_nue.set_ylabel('Nitrogen Use Efficiency [-]', size=14)
    ax_nue.set_xlabel('')
    ax_nue.tick_params(axis='y', which='major', labelsize=14)
    ax_nue.legend(handles=legend_patches_nue, loc='lower left', fontsize=8.5)
    # ax_nsurp.set_title('Nitrogen Surplus of each year')
    ax_nsurp.set_ylabel('Nitrogen Surplus [kg/ha]', size=14)
    ax_nsurp.set_xlabel('')
    ax_nsurp.tick_params(axis='y', which='major', labelsize=14)
    ax_nsurp.legend(handles=legend_patches_nsurp, loc='upper left', fontsize=8.5)

    ax_nue.set_ylim([0.3, 1.1])
    ax_nsurp.set_ylim([-20, 150])

    plt.tight_layout()
    plt.show()


def plot_nue_scatter(df, dict_good):
    # df.reset_index()

    # def get_label(category):
    #     if category in

    add_lines = True
    fig, (ax_nue, ax_nsurp) = plt.subplots(2, 1, figsize=(13, 13))

    colors = ['darkslategray', 'peru', 'snow', 'tomato', 'mediumaquamarine', 'slateblue']
    for i, agent in enumerate(dict_good['good_model']):
        ax_nue.scatter(df.loc[agent, 'year'], df.loc[agent, 'nue_value'],
                       label=dict_good['categories'][i] + ' (' + str(dict_good[agent + '_nue_good']) + '/39)',
                       c=colors[i], edgecolors='black')
        ax_nsurp.scatter(df.loc[agent, 'year'], df.loc[agent, 'nsurplus_value'],
                         label=dict_good['categories'][i] + ' (' + str(dict_good[agent + '_nsurplus_good']) + '/39)',
                         c=colors[i], edgecolors='black')

        ax_nue.plot(df.loc[agent, 'year'], df.loc[agent, 'nue_value'], c=colors[i], lw=0.3)
        ax_nsurp.plot(df.loc[agent, 'year'], df.loc[agent, 'nsurplus_value'], c=colors[i], lw=0.3)

        ax_nue.set_xticks(df.loc[agent, 'year'])
        ax_nsurp.set_xticks(df.loc[agent, 'year'])

    year_ranges = range(1982, 2022)
    year_ranges_more = range(1981, 2023)

    color_fill = 'salmon'
    ax_nue.fill_between(year_ranges_more, y1=0.9, y2=1.1, alpha=0.35, color=color_fill)
    ax_nue.fill_between(year_ranges_more, y1=0.5, y2=0.3, alpha=0.35, color=color_fill)

    ax_nsurp.fill_between(year_ranges_more, y1=40, y2=200, alpha=0.35, color=color_fill)
    ax_nsurp.fill_between(year_ranges_more, y1=0, y2=-10, alpha=0.35, color=color_fill)

    ax_nue.tick_params(axis='x', which='major', labelsize=9, rotation=45)
    ax_nsurp.tick_params(axis='x', which='major', labelsize=9, rotation=45)

    ax_nue.axhline(y=0.5, lw=1, color='black')
    ax_nue.axhline(y=0.9, lw=1, color='black')

    ax_nsurp.axhline(y=0.0, lw=1, color='black')
    ax_nsurp.axhline(y=40.0, lw=1, color='black')

    # ax_nue.set_title('Nitrogen Use Efficiency of each year')
    ax_nue.set_ylabel('Nitrogen Use Efficiency [-]')
    ax_nue.set_xlabel('Years')
    ax_nue.legend()
    # ax_nsurp.set_title('Nitrogen Surplus of each year')
    ax_nsurp.set_ylabel('Nitrogen Surplus [kg/ha]')
    ax_nsurp.set_xlabel('Years')
    ax_nsurp.legend()

    ax_nue.set_ylim([0.3, 1.1])
    ax_nsurp.set_ylim([-10, 200])
    ax_nue.set_xlim([1982, 2022])
    ax_nsurp.set_xlim([1982, 2022])
    plt.tight_layout()
    plt.show()


def plot_nue_scatter1(df, dict_good):
    # df.reset_index()

    add_lines = True
    fig, (ax_nue, ax_nsurp) = plt.subplots(2, 1, figsize=(10, 10))

    colors = ['darkslategray', 'peru', 'lightseagreen']
    for i, agent in enumerate([dict_good['good_model'], 'N2-PA', 'demeter']):
        ax_nue.scatter(df.loc[agent, 'year'], df.loc[agent, 'nue_value'],
                       label=dict_good['categories'][i] + ' (' + str(dict_good[agent + '_nue_good']) + '/39)',
                       c=colors[i])
        ax_nsurp.scatter(df.loc[agent, 'year'], df.loc[agent, 'nsurplus_value'],
                         label=dict_good['categories'][i] + ' (' + str(dict_good[agent + '_nsurplus_good']) + '/39)',
                         c=colors[i])

        ax_nue.plot(df.loc[agent, 'year'], df.loc[agent, 'nue_value'], c=colors[i], lw=0.3)
        ax_nsurp.plot(df.loc[agent, 'year'], df.loc[agent, 'nsurplus_value'], c=colors[i], lw=0.3)

        ax_nue.set_xticks(df.loc[agent, 'year'])
        ax_nsurp.set_xticks(df.loc[agent, 'year'])

    year_ranges = range(1982, 2022)
    year_ranges_more = range(1981, 2023)

    color_fill = 'salmon'
    ax_nue.fill_between(year_ranges_more, y1=0.9, y2=1.1, alpha=0.3, color=color_fill)
    ax_nue.fill_between(year_ranges_more, y1=0.5, y2=0.3, alpha=0.3, color=color_fill)

    ax_nsurp.fill_between(year_ranges_more, y1=40, y2=200, alpha=0.3, color=color_fill)
    ax_nsurp.fill_between(year_ranges_more, y1=0, y2=-10, alpha=0.3, color=color_fill)

    ax_nue.tick_params(axis='x', which='major', labelsize=9, rotation=45)
    ax_nsurp.tick_params(axis='x', which='major', labelsize=9, rotation=45)

    ax_nue.axhline(y=0.5, lw=1, color='black')
    ax_nue.axhline(y=0.9, lw=1, color='black')

    ax_nsurp.axhline(y=0.0, lw=1, color='black')
    ax_nsurp.axhline(y=40.0, lw=1, color='black')

    ax_nue.set_title('Nitrogen Use Efficiency of each year')
    ax_nue.set_ylabel('Nitrogen Use Efficiency [-]', size=14)
    ax_nue.set_xlabel('Years')
    ax_nue.legend()
    ax_nsurp.set_title('Nitrogen Surplus of each year')
    ax_nsurp.set_ylabel('Nitrogen Surplus [kg/ha]', size=14)
    ax_nsurp.set_xlabel('Years')
    ax_nsurp.legend()

    ax_nue.set_ylim([0.5, 1.1])
    ax_nsurp.set_ylim([-10, 120])
    ax_nue.set_xlim([1982, 2022])
    ax_nsurp.set_xlim([1982, 2022])
    plt.title('')
    plt.tight_layout()
    plt.show()


def plot_heatmap_fertilization(df):
    import matplotlib.dates as mdates

    # start_date = datetime.date(1989, 10, 3)
    # end_date = datetime.date(1990, 8, 20)
    #
    # # Generate weekly date range
    # date_range = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
    #
    # fig, ax1 = plt.subplots(figsize=(12, 6))
    #
    # offsets = np.linspace(-3, 3, len(fertilization_policies))
    # for i, (agent, agent_v) in enumerate(fertilization_policies.items()):
    #     shifted_dates = date_range + pd.to_timedelta(offsets[i], unit='D')
    #     ax1.bar(shifted_dates, agent_v, width=3, alpha=.9, label=agent[0])
    # ax1.set_ylabel('Fertilizer applications [kg/ha]')
    # ax1.yaxis.grid(True, which='major', linestyle='--')
    #
    # # Plot fertilization policy as vertical bars
    # ax2 = ax1.twinx()
    #
    # # Plot yield data
    # for agent, agent_v in growth.items():
    #     if len(date_range) < len(agent_v):
    #         agent_v.pop(0)
    #     ax2.plot(date_range, agent_v, alpha=.9, label=agent[0])
    # ax2.set_xlabel('Date in growing year')
    # ax2.set_ylabel('Aggregated yield growth [kg/ha]')
    # ax2.yaxis.grid(False)
    # ax2.legend(loc='upper left')
    #
    # # Set x-axis major ticks format
    # ax1.xaxis.set_major_locator(mdates.MonthLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    #
    # # plt.grid(True)
    # plt.tight_layout()
    # plt.legend(loc='lower right')
    # plt.show()


def plot_3d_reward_function(env=False):
    import pcse_gym.envs.rewards as rew
    import matplotlib.cm as cm

    def cgm(env, weeks, n_levels):
        env.reset()
        reward = 0
        weeks = int(weeks)
        for _ in range(weeks):
            _, _, term, _, info = env.step(0)
        else:
            _, _, term, _, info = env.step(n_levels * 3)

        while not term:
            _, reward, term, _, info = env.step(0)

        return reward

    cont = rew.Rewards.ContainerNUE(timestep=7)

    # Generate data for plotting

    if env:
        env = pcse_gym.initialize_envs.initialize_env(reward='NUE',
                                                      years=[1984],
                                                      random_weather=True,
                                                      weather_features=["IRRAD", "TMIN", "RAIN"],
                                                      crop_features=["DVS", "TAGP", "LAI", "SM"],
                                                      nitrogen_levels=9,
                                                      pcse_env=2,
                                                      locations=[(52.57, 5.63)],
                                                      action_features=["action_history"])

        week = np.linspace(0, 45, 46)
        n_lev = np.arange(1, 10)

        B, C = np.meshgrid(week, n_lev)
        Z = np.array([[cgm(env, w, n) for w in week] for n in n_lev])
    else:
        b_values = np.linspace(-100, 140, 100)
        c_values = np.linspace(0, 1, 100)
        B, C = np.meshgrid(b_values, c_values)
        Z = np.array([[cont.n_surplus_formula(b, c) for b in b_values] for c in c_values])

    # Create a 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(B, C, Z, cmap='viridis')

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Reward')

    if env:
        ax.set_xlabel('Week of fertilization')
        ax.set_ylabel('Fertilization Level')
        ax.set_zlabel('env(N_surplus, NUE)')
        ax.set_title('3D Plot of NUE reward function in year 1984')
    else:
        ax.set_xlabel('N_surplus')
        ax.set_ylabel('NUE')
        ax.set_zlabel('f(N_surplus, NUE)')
        ax.set_title('3D Plot of NUE reward function')

    plt.show()
