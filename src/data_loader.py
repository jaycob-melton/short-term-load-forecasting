# src/data_loader.py
import pandas as pd

def load_ospd_time_series(file_path='../data/raw/time_series_60min_singleindex.csv'):
    df = pd.read_csv(file_path, parse_dates=['utc_timestamp'])
    df = df[[
        "utc_timestamp",
        "DE_load_actual_entsoe_transparency",
        "DE_load_forecast_entsoe_transparency",
        "DE_solar_generation_actual",
        "DE_solar_profile",
        "DE_wind_generation_actual",
        "DE_wind_profile"
    ]]

    df.rename(columns={
        "utc_timestamp": "datetime",
        "DE_load_actual_entsoe_transparency": "load",
        "DE_load_forecast_entsoe_transparency": "load_forecast",
        "DE_solar_profile": "solar_profile",
        "DE_solar_generation_actual": "solar",
        "DE_wind_profile": "wind_profile",
        "DE_wind_generation_actual": "wind"
    }, inplace=True)
    #df.dropna(inplace=True)
    return df.copy()


def load_ospd_weather_data(file_path="../data/raw/weather_data.csv"):
    df = pd.read_csv(file_path, parse_dates=["utc_timestamp"])
    df = df[[
        "utc_timestamp", 
        "DE_temperature", 
        "DE_radiation_direct_horizontal", 
        "DE_radiation_diffuse_horizontal"
    ]]
    df.rename(columns={
        "utc_timestamp": "datetime",
        "DE_temperature": "temp",
        "DE_radiation_direct_horizontal": "rad_direct",
        "DE_radiation_diffuse_horizontal": "rad_diffuse"
    }, inplace=True)
    
    #df.dropna(inplace=True)
    return df.copy()


def merge_time_and_weather(load_df, weather_df):
    df = pd.merge(load_df, weather_df, on="datetime", how="inner")
    df = df.dropna()
    df = df.set_index("datetime").sort_index()
    return df.copy()


def get_cleaned_data(
    load_path="../data/raw/time_series_60min_singleindex.csv",
    weather_path="../data/raw/weather_data.csv"
):
    load_df = load_ospd_time_series(load_path)
    weather_df = load_ospd_weather_data(weather_path)
    df = merge_time_and_weather(load_df, weather_df)
    df.to_csv("../data/cleaned/cleaned_data.csv")
    return df