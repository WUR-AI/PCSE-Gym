import datetime
from math import exp
from statistics import mean

import pcse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_date_list(start_date, end_date):
    """
    Generate a list of datetime.date objects for each day between start_date and end_date inclusive.

    Args:
    start_date (datetime.date): The starting date.
    end_date (datetime.date): The ending date.

    Returns:
    list: A list of datetime.date objects from start_date to end_date.
    """
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += datetime.timedelta(days=1)
    return date_list


def nasapower_to_larswg(coordinates: tuple,
                        site_name: str = None,
                        co2: float = 400.0,
                        first_date: datetime.date = None,
                        last_date: datetime.date = None):
    """
    Lars WG requires weather data from a site for parameterization.
    This function converts weather from NASA Power to a format for LARS WG random weather generation.

    :param site_name: string of site name; used for naming the file
    :param co2: co2 level of location
    :param coordinates: a tuple of LAT and LON
    :param first_date: datetime.date of the first point to get from NASA power
    :param last_date: datetime.date of the last point to get from NASA power
    :return: None

    Files are saved under pcse_gym/utils/weather_utils/data_larswg/
    """
    wdp = pcse.input.NASAPowerWeatherDataProvider(*coordinates)

    if first_date and last_date is not None:
        date_range = generate_date_list(first_date, last_date)
    else:
        date_range = generate_date_list(wdp.first_date, wdp.last_date)

    try:
        tmin = pd.Series([wdp(x).TMIN for x in date_range])
    except pcse.exceptions.WeatherDataProviderError:
        if wdp.missing > 1:
            miss = wdp.missing_days[-1]
            date_range = generate_date_list(wdp.first_date, miss - datetime.timedelta(days=1))
            tmin = pd.Series([wdp(x).TMIN for x in date_range])
        else:
            date_range = generate_date_list(wdp.first_date, wdp.missing_days[0] - datetime.timedelta(days=1))
            tmin = pd.Series([wdp(x).TMIN for x in date_range])
    tmax = pd.Series([wdp(x).TMAX for x in date_range])
    rain = pd.Series([wdp(x).RAIN*10 for x in date_range])  # cm to mm
    irrad = pd.Series([wdp(x).IRRAD/1000000 for x in date_range])  # J to MJ

    y = pd.Series([x.year for x in date_range])
    m = pd.Series([x.month for x in date_range])
    d = pd.Series([x.day for x in date_range])

    weather_dict = pd.concat([y, m, d, tmin, tmax, rain, irrad], axis=1)

    weather_dict.to_csv(os.path.join('data_larswg', f'{str(wdp.latitude)}-{str(wdp.longitude)}.dat'),
                        sep='\t',
                        header=False,
                        index=False)

# don't change format below!
    site_file = f'''[SITE]
{f'{wdp.latitude}-{wdp.longitude}' if site_name is None else site_name}
[LAT, LON and ALT]
{wdp.latitude}	{wdp.longitude}	 {wdp.elevation}
[CO2]
{co2}
[WEATHER FILES]
{wdp.latitude}-{wdp.longitude}.dat
[FORMAT]
YEAR MONTH DAY	MIN MAX RAIN RAD
[END]
    '''

    file = open(os.path.join('data_larswg',
                             f"{f'{str(wdp.latitude)}-{str(wdp.longitude)}' if site_name is None else site_name}.st"), 'w')
    file.write(site_file)


def convert_pcse_file_to_lars_wg(filename: str, site_name: str = None, co2: float = 400.0,):
    wdp = None
    if '.xls' in filename:
        wdp = pcse.input.ExcelWeatherDataProvider(os.path.join('origin_csv', filename))
    if '.csv' in filename:
        wdp = pcse.input.CSVWeatherDataProvider(os.path.join('origin_csv', filename))
    else:
        Exception(f"file {filename} not CSV or XLSX")

    date_range = generate_date_list(wdp.first_date, wdp.last_date)

    try:
        tmin = pd.Series([wdp(x).TMIN for x in date_range])
    except pcse.exceptions.WeatherDataProviderError:
        if wdp.missing > 1:
            miss = wdp.missing_days[-1]
            date_range = generate_date_list(wdp.first_date, miss - datetime.timedelta(days=1))
            tmin = pd.Series([wdp(x).TMIN for x in date_range])
        else:
            date_range = generate_date_list(wdp.first_date, wdp.missing_days[0] - datetime.timedelta(days=1))
            tmin = pd.Series([wdp(x).TMIN for x in date_range])

    tmax = pd.Series([wdp(x).TMAX for x in date_range])
    rain = pd.Series([wdp(x).RAIN * 10 for x in date_range])  # cm to mm
    irrad = pd.Series([wdp(x).IRRAD / 1000000 for x in date_range])  # J to MJ

    y = pd.Series([x.year for x in date_range])
    m = pd.Series([x.month for x in date_range])
    d = pd.Series([x.day for x in date_range])

    weather_dict = pd.concat([y, m, d, tmin, tmax, rain, irrad], axis=1)

    weather_dict.to_csv(os.path.join('data_larswg', f'{wdp.latitude:.2f}-{wdp.longitude:.2f}.dat'),
                        sep='\t',
                        header=False,
                        index=False)

    # don't change format below!
    site_file = f'''[SITE]
{f'{wdp.latitude}-{wdp.longitude}' if site_name is None else site_name}
[LAT, LON and ALT]
{wdp.latitude:.2f}	{wdp.longitude:.2f}	 {wdp.elevation}
[CO2]
{co2}
[WEATHER FILES]
{f'{wdp.latitude:.2f}-{wdp.longitude:.2f}.dat'}
[FORMAT]
YEAR MONTH DAY	MIN MAX RAIN RAD
[END]
    '''

    file = open(os.path.join('data_larswg',
                             f"{f'{wdp.latitude:.2f}-{wdp.longitude:.2f}'}.st"),
                'w')
    file.write(site_file)


def convert_year(wg_df: pd.DataFrame, base_year=3000):

    formatted_year = []

    for index, row in wg_df.iterrows():
        year_index = int(row.year)
        day_of_year = int(row.doy)

        # Compute the new year
        year = base_year + year_index

        # Create date from year and day of year
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)

        # Format date as yyyymmdd
        formatted_date = date.strftime('%Y%m%d')
        formatted_year.append(year)

    return pd.Series(formatted_year)


def convert_date(wg_df: pd.DataFrame):

    formatted_date = []
    for index, row in wg_df.iterrows():
        year = int(row.year)
        day_of_year = int(row.doy)

        # Create date from year and day of year
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)

        # Format date as yyyymmdd
        converted_date = date.strftime('%Y%m%d')
        formatted_date.append(converted_date)

    return pd.Series(formatted_date)


def convert_date_nasa(np_df: pd.DataFrame):
    formatted_date = []
    for index, row in np_df.iterrows():
        day = row.DAY

        date = day.strftime('%Y%m%d')
        formatted_date.append(date)
    series = pd.Series(formatted_date)

    np_df.DAY = series

    return np_df


def convert_leap_year(wg_df: pd.DataFrame):
    """
    FUnction to add leap years to generated LARSWG8.0 weather

    :param wg_df:
    :return:
    """
    ''' Sanity check... has to be done in order
        filter for 29th feb
            get required index and values for values WITHOUT year and doy
            calculate mean for 29th feb
            then now insert year and doy
        shift doy by 1
            get indices of dates after 28th Feb
            shift by index
            then insert correct value (with .5 index)
            then insert the doy
            finally sort and reset index
    '''
    '''29th Feb is the 60th day of year'''
    i28_feb = (is_leap_year(wg_df['year'])) & (wg_df['doy'] == 59)

    """get indices of 28th Feb for each leap year"""
    filtered_wg_df = wg_df[i28_feb]
    indices_28_feb = filtered_wg_df.index
    indices_1_mar = [x + 1 for x in indices_28_feb]

    """get values of 28th Feb and 1st Mar to calculate 29th Feb with average"""
    values_28_feb = wg_df.loc[indices_28_feb, ['TMIN', 'TMAX', 'RAIN', 'IRRAD']]
    values_1_mar = wg_df.loc[indices_1_mar, ['TMIN', 'TMAX', 'RAIN', 'IRRAD']]
    # doy_28_feb = wg_df.loc[indices_28_feb, ['doy']]
    # doy_1_mar = wg_df.loc[indices_1_mar, ['doy']]
    values_1_mar.index = indices_28_feb  # shift index so the next calculation is correct

    '''Calculate value of 29th Feb'''
    values_29_feb = (values_28_feb+values_1_mar)/2

    '''Insert correct doy and year for 29th Feb'''
    values_29_feb.insert(0, 'doy', [60 for _ in values_29_feb.index], True)
    values_29_feb.insert(0, 'year', [x for x in filtered_wg_df.year], True)

    # print(values_29_feb)

    '''Handle duplicate doy; filter then shift doy by 1'''
    idoy = (is_leap_year(wg_df['year'])) & (wg_df['doy'] > 59)
    filtered_wg_df = wg_df[idoy]
    # print(filtered_wg_df[:10])
    values_doy = pd.Series([x + 1 for x in filtered_wg_df.doy])
    values_doy.index = filtered_wg_df.index
    # print(values_doy[:10])

    '''Hacky :) insert the 29th feb index in between 28th and 1st of Match'''
    values_29_feb.index = [x + .5 for x in indices_28_feb]
    wg_df = (pd.concat([wg_df, values_29_feb])).sort_index()

    '''Fix the doy values'''
    wg_df.loc[filtered_wg_df.index, ['doy']] = values_doy
    # print(wg_df[1153:1157])
    wg_df = wg_df.reset_index(drop=True)

    # print(list(wg_df.loc[wg_df.year == 3004, 'doy']))

    ''' check for NaN values'''
    assert not wg_df.isnull().values.any()

    return wg_df


def is_leap_year(year):
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


def get_csv_header(country='NL', lon=5.5, lat=52.5, elev=15.3, anga=0.18, angb=0.55):
    header = f'''## Site Characteristics
Country = '{country}'
Station = 'Wageningen'
Description = 'Random weather generated from LARSWG8.0. Years are labeled from 3001.'
Source = 'LARSWG8.0 and NASA Power'
Contact = 'Hilmy Baja'
Longitude = {lon}; Latitude = {lat}; Elevation = {elev}; AngstromA = {anga}; AngstromB = {angb}; HasSunshine = False
## Daily weather observations (missing values are NaN)'''

    return header


def larswg_to_pcse_csv(site_file: str, fill_missing=False):
    """
    Function to convert larswg generated output into csv readable by PCSE.

    Note that larswg doesn't simulate VAP and WIND. So, the values are instead filled with
    the actual PCSE outputs.

    :param site_file: Name of the sitefile from LARSWG8.0 output
    :return:
    """

    dat_file = site_file[:-3]

    assert os.path.isfile(os.path.join("output_larswg", f'{dat_file}.dat'))

    with open(os.path.join("output_larswg", f"{site_file}"), 'r') as f:
        st_file = f.read()
        lines = st_file.split('\n')[3]
        line = [splits for splits in lines.split("\t") if splits]

    lat = line[0]
    lon = line[1]

    filename = f'{lat}-{lon}_random_weather'

    # load nasa power to fill in
    wdp = pcse.input.NASAPowerWeatherDataProvider(float(lat), float(lon))

    with open(os.path.join("output_larswg", f"{dat_file}.dat"), 'r') as f:
        wg_df = pd.read_csv(f, sep='\t', names=['year', 'doy', 'TMIN', 'TMAX', 'RAIN', 'IRRAD'], index_col=False)

    # convert year formats
    wg_df.year = convert_year(wg_df)

    # add leap year for PCSE year labeling
    wg_df = convert_leap_year(wg_df)

    # convert date to CSV format
    wg_df.year = convert_date(wg_df)
    wg_df = wg_df.rename(columns={'year': 'DAY'})
    wg_df = wg_df.drop('doy', axis=1)

    # reorder IRRAD
    cols = wg_df.columns.tolist()
    cols = [cols[0]] + cols[-1:] + cols[1:4]
    wg_df = wg_df[cols]

    # insert required columns and correct the units
    wg_df.insert(4, 'VAP', [estimate_vapour_pressure(x) for x in wg_df.TMIN], True)
    wg_df.insert(5, 'WIND', [2 for _ in wg_df.TMIN], True)
    wg_df.insert(7, 'SNOWDEPTH', ['NaN' for _ in wg_df.TMIN], True)

    wg_df.IRRAD = wg_df.IRRAD * 1000  # J to kJ
    wg_df.RAIN = wg_df.RAIN  # * 10  # cm to mm

    # create NASA power df
    date_range = generate_date_list(datetime.date(1984,1,1),
                                    datetime.date(2022, 12, 31))
    tmin = pd.Series([wdp(x).TMIN for x in date_range])
    tmax = pd.Series([wdp(x).TMAX for x in date_range])
    rain = pd.Series([wdp(x).RAIN * 10 for x in date_range])  # cm to mm
    irrad = pd.Series([wdp(x).IRRAD / 1000 for x in date_range])  # J to kJ
    vap = pd.Series([wdp(x).VAP / 10 for x in date_range])  # hPA to kPa
    wind = pd.Series([wdp(x).WIND for x in date_range])  # hPA to kPa
    day = pd.Series([wdp(x).DAY for x in date_range])

    np_df = pd.concat([day, irrad, tmin, tmax, vap, wind, rain], axis=1,
                      keys=['DAY', 'IRRAD', 'TMIN', 'TMAX', 'VAP', 'WIND', 'RAIN'])
    np_df.insert(7, 'SNOWDEPTH', ['NaN' for _ in np_df.TMIN], True)
    np_df = convert_date_nasa(np_df)

    # insert nasa power df on top of wg df
    wg_df.index = [x + len(np_df.index) for x in wg_df.index]
    wg_df = pd.concat([np_df, wg_df])

    # get header for csv
    header = get_csv_header('NL', lon=float(lon), lat=float(lat), elev=15.29)

    with open(os.path.join('random_weather_csv', f'{filename}.csv'), 'w', newline='') as file:
        file.write(header + '\n')
        wg_df.to_csv(file, index=False)


def evaluate_wind_vap_derivation():
    """

    :return:
    """
    with open('NASAPOWER.csv', 'r') as f:
        df_np = pd.read_csv(f, sep=',')

    print(df_np.head())

    wdp = pcse.db.NASAPowerWeatherDataProvider(52.0, 5.5)
    date_range = generate_date_list(datetime.date(1984, 1, 1),
                                    datetime.date(2022, 12, 31))



    ori_wind = df_np.WIND
    ori_vap = df_np.VAP

    temp = df_np.TEMP
    supposed_temp = (df_np.TMIN+df_np.TMAX)/2

    print(temp, supposed_temp)

    ea = df_np.TMIN.apply(estimate_vapour_pressure) * 10

    plt.scatter(ea, ori_vap)
    plt.title('NASA power VAP vs VAP estimated from TMIN')
    r2 = r_squared(ori_vap, ea)
    plt.text(4, 20, f'R^2: {r2}')
    plt.show()

    plt.scatter(supposed_temp, temp)
    plt.title('NASA power VAP vs VAP estimated from TMIN')
    r2 = r_squared(temp, supposed_temp)
    plt.text(4, 20, f'R^2: {r2}')
    plt.show()

    reference_eto = []
    for day, tmin, tmax, avrad, vap, wind in zip(df_np.DAY, df_np.TMIN, df_np.TMAX, df_np.IRRAD, df_np.VAP, df_np.WIND):
        day = datetime.datetime.strptime(day, "%Y-%m-%d").date()
        eto = pcse.util.penman_monteith(day, 25.0, 15.29, tmin, tmax, avrad, vap, wind)
        reference_eto.append(eto)
    reference_eto = pd.Series(reference_eto)

    wind_eto = []
    wind_mean = mean([x for x in df_np.WIND])
    for day, tmin, tmax, avrad, vap in zip(df_np.DAY, df_np.TMIN, df_np.TMAX, df_np.IRRAD, df_np.VAP):
        day = datetime.datetime.strptime(day, "%Y-%m-%d").date()
        eto = pcse.util.penman_monteith(day, 25.0, 15.29, tmin, tmax, avrad, vap, 2)
        wind_eto.append(eto)
    wind_eto = pd.Series(wind_eto)

    plt.scatter(wind_eto, reference_eto)
    plt.title('NASA POWER ETO vs 2m ETO')
    r2 = r_squared(reference_eto, wind_eto)
    plt.text(1.5, 5, f'R^2: {r2}')
    plt.show()


def histogram_check():
    with open(os.path.join("output_larswg", f"52.0-5.5WG.dat"), 'r') as f:
        df = pd.read_csv(f, sep='\t', names=['year', 'doy', 'TMIN', 'TMAX', 'RAIN', 'IRRAD'], index_col=False)

        wdp = pcse.db.NASAPowerWeatherDataProvider(52.0, 5.5)
        date_range = generate_date_list(datetime.date(1984, 1, 1),
                                        datetime.date(2022, 12, 31))
        rain = [wdp(x).RAIN for x in date_range]
        vap = [wdp(x).VAP for x in date_range]
        irrad = [wdp(x).IRRAD for x in date_range]
        tmin = [wdp(x).TMIN for x in date_range]
        tmax = [wdp(x).TMAX for x in date_range]

        # print(mean(wind))
        plt.hist(df.RAIN, bins=100)
        plt.title('RAIN LARSWG')
        # plt.ylim((0, 100000))
        plt.show()

        plt.hist(rain, bins=100)
        plt.title('RAIN NASA power')
        # plt.ylim((0, 1000))
        plt.show()

        vap_wg = [estimate_vapour_pressure(x) * 10 for x in df.TMIN]

        # plt.hist(vap, bins=100)
        # plt.title('VAP NASAPOWER')
        # # plt.ylim((0, 1000))
        # plt.show()
        #
        # plt.hist(vap_wg, bins=100)
        # plt.title('VAP LARSWG')
        # # plt.ylim((0, 100000))
        # plt.show()
        #
        # plt.hist(irrad, bins=100)
        # plt.title('IRRAD NASAPOWER')
        # # plt.ylim((0, 1000))
        # plt.show()
        #
        # plt.hist(df.IRRAD, bins=100)
        # plt.title('IRRAD LARSWG')
        # # plt.ylim((0, 100000))
        # plt.show()
        #
        # plt.hist(tmin, bins=100)
        # plt.title('TMIN NASAPOWER')
        # # plt.ylim((0, 1000))
        # plt.show()
        #
        # plt.hist(df.TMIN, bins=100)
        # plt.title('TMIN LARSWG')
        # # plt.ylim((0, 100000))
        # plt.show()
        #
        # plt.hist(tmax, bins=100)
        # plt.title('TMAX NASAPOWER')
        # # plt.ylim((0, 1000))
        # plt.show()

        plt.hist(df.TMAX, bins=100)
        plt.title('TMAX LARSWG')
        # plt.ylim((0, 100000))
        plt.show()


def estimate_vapour_pressure(tmin):
    """
    This is a function to estimate vapour pressure from tmin.
    This assumption holds for non-arid locations.
    For arid locations, this no longer holds and a subtraction of 2-3 deg C is required for tmin.


    Reference:
        Allen et al. (1998). FAO Irrigation and drainage paper No. 56.
    :param tmin:
    :return:
    """
    term = (17.27 * tmin) / (tmin + 237.3)
    ea = 0.611 * exp(term)

    return ea


def r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()
    return 1 - (ss_res/ss_tot)


if __name__ == '__main__':
    # nasapower_to_larswg((52.5, 5.5),
    #                     "52.5-5.5")
    larswg_to_pcse_csv('PAGVWG.st')
    # convert_pcse_csv_to_lars_wg('pagv_weather.xlsx')
    # nasapower_to_larswg((52.0, 5.5), co2=344.85)
    # evaluate_wind_vap_derivation()
    # histogram_check()

