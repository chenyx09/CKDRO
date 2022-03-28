import pytz
from datetime import datetime, timedelta
import re
from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np
import os
from pandas.core.frame import DataFrame
import pdb
import dill


def load_case():
    filenames = [
        "Zone1_702381_32.77_-96.78_2020.csv",
        "Zone2_734232_29.77_-95.38_2020.csv",
        "Zone3_567437_35.21_-101.82_2020.csv",
        "Zone4_565910_33.57_-101.86_2020.csv",
        "Zone5_677404_30.29_-97.74_2020.csv",
        "Zone6_574575_30.17_-101.54_2020.csv",
        "Zone7_686105_27.81_-97.46_2020.csv",
        "Zone8_513033_30.69_-103.86_2020.csv",
    ]

    # Zones = ['WEST', 'GENESE', 'CENTRL', 'NORTH', 'MHK VL',
    #          'CAPITL', 'HUD VL', 'MILLWD', 'DUNWOD', 'N.Y.C.', 'LONGIL']
    df_weather_hourly = pd.DataFrame()  # 11 zones
    columns = [
        "Global Horizontal UV Irradiance (280-400nm)",
        "Wind Speed",
    ]  # for renewable energy prediction
    columns = ["Temperature", "Precipitable Water"]  # for load prediction

    for i, a in enumerate(filenames):
        region_data = pd.read_csv(a, header=[2])
        if i == 0:
            df_weather_hourly["hour"] = region_data["Hour"]
        for col in columns:

            df_weather_hourly["zone" + str(i) + " " + col] = region_data[col]

    # add time info
    time = np.arange(df_weather_hourly.shape[0])

    df_weather_hourly["t"] = time

    # df_weather_hourly.insert(-1, "zone" + str(i) + " " + col, region_data[col])
    # df[10]["Wind Speed"]
    # df[10]["GHI"]
    # df[10]["Wind Direction"]
    # df[10]["Relative Humidity"]
    # df[10]["Temperature"]
    # df[10]["GHI"]

    df_load_hourly = pd.read_excel("Load_8zone_2020.xlsx")

    Zones = ["EAST", "COAST", "NORTH", "NCENT", "SCENT", "WEST", "SOUTH", "FWEST"]

    # Bus  # ZONE
    # 1	EAST
    # 2	COAST
    # 3	NORTH
    # 4	NCENT
    # 5	SCENT
    # 6	WEST
    # 7	SOUTH
    # 8	FWEST

    # newyearday = datetime(2020, 1, 1, 0, 0, 0, tzinfo=est)
    # leapday = datetime(2020, 2, 29, 23, 55, 0, tzinfo=est)
    set8760 = set(np.arange(0, 366 * 24)) - set(range(24 * 58, 24 * 59))
    df_load_hourly = df_load_hourly.loc[set8760, Zones].reset_index(drop=True)

    output_file = "ERCOT_data.pkl"
    with open(output_file, "wb") as f:
        dill.dump([df_weather_hourly, df_load_hourly], f)


if __name__ == "__main__":
    load_case()
