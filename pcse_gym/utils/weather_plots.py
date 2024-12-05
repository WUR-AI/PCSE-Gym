import os
import math
from collections import defaultdict

import lib_programname
import numpy as np

from pcse.input.nasapower import NASAPowerWeatherDataProvider
import matplotlib.pyplot as plt

NL_loc_ext = [(52, 5.5), (51.5, 5), (52.5, 6.0)]
LT_loc_ext = [(55.0, 23.5), (55.0, 24.0), (55.5, 23.5)]

NL_loc = (52, 5.5)
LT_loc = (55.0, 23.5)

wdp = NASAPowerWeatherDataProvider(*NL_loc)

path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[2]


def plot_weather_data(weather_data, variables: list, start_year=None, end_year=None, save=False):
    """
    Plots the specified variables over the given time range.

    Parameters:
    - variables: List of variables to plot (e.g., ['IRRAD', 'TMIN', 'RAIN'])
    - start_year: Start year for the range. If None, will use the first year in the data.
    - end_year: End year for the range. If None, will use the last year in the data.
    """

    filtered_data = filter_years(weather_data, start_year, end_year)

    dates = [entry['DAY'] for entry in filtered_data]

    # Plotting
    plt.figure(figsize=(14, 7))
    for var in variables:
        values = [entry[var] for entry in filtered_data]
        plt.plot(dates, values, label=var)

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(
        f"Daily Change of {', '.join(variables)} from {start_year if start_year else min(dates).year} to {end_year if end_year else max(dates).year}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(rootdir, "plots", f"NASAPower_{variables[0]}_{start_year}_to_{end_year}"))
    else:
        plt.show()


def plot_weather_data_weekly(weather_data, variables: list, start_year=None, end_year=None, aggregation='sum', save=False):
    """
    Plots the specified variables over the given time range with a weekly timestep.

    Parameters:
    - variables: List of variables to plot (e.g., ['IRRAD', 'TMIN', 'RAIN'])
    - start_year: Start year for the range. If None, will use the first year in the data.
    - end_year: End year for the range. If None, will use the last year in the data.
    - aggregation: Method of aggregation ('mean' or 'sum')
    """

    filtered_data = filter_years(weather_data, start_year, end_year)

    # Group by year and week number
    weekly_data = defaultdict(list)
    for entry in filtered_data:
        year, week, _ = entry['DAY'].isocalendar()
        weekly_data[(year, week)].append(entry)

    aggregated_data = {}
    for (year, week), entries in weekly_data.items():
        aggregated_entry = {'DAY': f"{year}-W{week}"}
        for var in variables:
            if aggregation == 'mean':
                aggregated_entry[var] = np.mean([entry[var] for entry in entries])
            elif aggregation == 'sum':
                aggregated_entry[var] = np.sum([entry[var] for entry in entries])
        aggregated_data[(year, week)] = aggregated_entry

    # Extract week labels and values for the specified variables
    weeks = sorted(aggregated_data.keys())
    week_labels = [f"{year}-W{week}" for year, week in weeks]

    # Plotting
    plt.figure(figsize=(21, 14))
    for var in variables:
        values = [aggregated_data[week][var] for week in weeks]
        plt.plot(week_labels, values, label=var)

    plt.xlabel("Week")
    plt.ylabel("Value")
    plt.title(
        f"Weekly {aggregation.capitalize()} of {', '.join(variables)} from {start_year if start_year else min(weeks)[0]} to {end_year if end_year else max(weeks)[0]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ticks = 1 if end_year - start_year < 3 else 52
    plt.xticks(week_labels[::ticks], rotation=45)
    if save:
        plt.savefig(os.path.join(rootdir, "plots", f"NASAPower_{variables}_{start_year}_to_{end_year}"))
    else:
        plt.show()


def plot_yearly_extremes(weather_data, variable: str, start_year=None, end_year=None, save=False):
    """
    Plots the maximum and minimum values of the specified variable for each year.

    Parameters:
    - variable: Variable to plot (e.g., 'IRRAD')
    - start_year: Start year for the range. If None, will use the first year in the data.
    - end_year: End year for the range. If None, will use the last year in the data.
    """

    # Filter the data based on the specified time range
    filtered_data = filter_years(weather_data, start_year, end_year)

    # Group by year
    yearly_data = defaultdict(list)
    for entry in filtered_data:
        year = entry['DAY'].year
        yearly_data[year].append(entry)

    # Find yearly extremes
    years = sorted(yearly_data.keys())
    max_values = [max([entry[variable] for entry in yearly_data[year]]) for year in years]
    min_values = [min([entry[variable] for entry in yearly_data[year]]) for year in years]
    year_labels = [f"{year}" for year in years]

    plt.figure(figsize=(14, 7))
    plt.plot(years, max_values, label=f'Max {variable}', marker='o', linestyle='-')
    plt.plot(years, min_values, label=f'Min {variable}', marker='o', linestyle='-')

    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title(f"Yearly Extremes of {variable}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.xticks(year_labels[::], rotation=45)
    if save:
        plt.savefig(os.path.join(rootdir, "plots", f"NASAPower_extremes_{variable}_{start_year}_to_{end_year}"))
    else:
        plt.show()


def plot_extreme_years_histogram(weather_data, variable, start_year=None, end_year=None, std_dev_bound=1, save=False):
    """
    Plots a histogram of yearly maximum values of the specified variable for a range of years
    and highlights "extreme years".

    Parameters:
    - variable: Variable to analyze (e.g., 'IRRAD')
    - start_year: Start year for the range. If None, will use the first year in the data.
    - end_year: End year for the range. If None, will use the last year in the data.
    - std_dev_bound: Number of standard deviations away from the mean to categorize a year as "extreme"
    """

    # Filter the data based on the specified time range
    filtered_data = filter_years(weather_data, start_year, end_year)

    # Group filtered data by year and calculate the yearly maximums
    yearly_max_values = {}
    for entry in filtered_data:
        year = entry['DAY'].year
        if year not in yearly_max_values:
            yearly_max_values[year] = entry[variable]
        else:
            yearly_max_values[year] = max(yearly_max_values[year], entry[variable])

    # Calculate mean and standard deviation of yearly maximums
    mean_max = sum(yearly_max_values.values()) / len(yearly_max_values)
    std_dev_max = (sum((x - mean_max) ** 2 for x in yearly_max_values.values()) / len(yearly_max_values)) ** 0.5

    # Identify extreme years
    extreme_years = [year for year, value in yearly_max_values.items() if
                     abs(value - mean_max) > std_dev_bound * std_dev_max]

    # Plotting
    plt.figure(figsize=(14, 7))
    # bins = [mean_max - std_dev_bound*2, mean_max - std_dev_bound, mean_max, mean_max + std_dev_bound, mean_max + std_dev_bound*2]
    if variable == 'IRRAD':
        bins = range(int(min(yearly_max_values.values())), int(max(yearly_max_values.values())) + 1,
                     math.ceil(std_dev_max / 10))
    else:
        bins = np.histogram_bin_edges(list(yearly_max_values.values()), bins='auto')
    plt.hist(yearly_max_values.values(), bins=bins, alpha=0.7, label='All Years')
    plt.hist([yearly_max_values[year] for year in extreme_years], bins=bins, alpha=0.7, label='Extreme Years',
             color='red')
    plt.axvline(mean_max, color='blue', linestyle='dashed', linewidth=2, label=f'Mean Max = {mean_max:.2f}')
    plt.axvline(mean_max + std_dev_bound * std_dev_max, color='green', linestyle='dashed', linewidth=2,
                label=f'Mean + {std_dev_bound}*STD')
    plt.axvline(mean_max - std_dev_bound * std_dev_max, color='green', linestyle='dashed', linewidth=2,
                label=f'Mean - {std_dev_bound}*STD')
    plt.text(0.01, 0.5, f"Extreme Years:\n{' '.join(map(str, extreme_years))}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5))
    plt.xlabel(f'Yearly Maximum {variable}')
    plt.ylabel('Number of Years')
    plt.title(
        f'Histogram of Yearly Maximum {variable} from {start_year if start_year else min(yearly_max_values.keys())} to {end_year if end_year else max(yearly_max_values.keys())} (Extreme Years Highlighted)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(rootdir, "plots", f"NASAPower_extreme_hists_{variable}_{start_year}_to_{end_year}"))
    else:
        plt.show()

    return extreme_years


def filter_years(weather_data, start_year, end_year):
    filtered = [
        entry for entry in weather_data
        if (start_year is None or entry['DAY'].year >= start_year) and
           (end_year is None or entry['DAY'].year <= end_year)
    ]
    return filtered


def get_plots(weather_data=wdp):
    w_data = weather_data.export()
    plot_weather_data_weekly(w_data, ['IRRAD'], 1990, 2022, save=True)
    plot_weather_data_weekly(w_data, ['RAIN'], 1990, 2022, save=True)
    plot_weather_data_weekly(w_data, ['TMIN', 'TEMP', 'TMAX'], 1990, 2022, save=True)

    plot_yearly_extremes(w_data, 'IRRAD', 1990, 2022, save=True)
    plot_yearly_extremes(w_data, 'RAIN', 1990, 2022, save=True)
    plot_yearly_extremes(w_data, 'TMIN', 1990, 2022, save=True)
    plot_yearly_extremes(w_data, 'TEMP', 1990, 2022, save=True)

    plot_extreme_years_histogram(w_data, 'IRRAD', 1990, 2022, save=True)
    plot_extreme_years_histogram(w_data, 'RAIN', 1990, 2022, save=True)
    plot_extreme_years_histogram(w_data, 'TMIN', 1990, 2022, save=True)
    plot_extreme_years_histogram(w_data, 'TEMP', 1990, 2022, save=True)



