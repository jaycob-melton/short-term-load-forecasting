import pandas as pd
import numpy as np
import holidays

def feature_engineering(df):
  # Time features
  df["hour"] = df.index.hour
  df["dayofweek"] = df.index.dayofweek
  df["month"] = df.index.month
  #df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
  #df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
  df = pd.get_dummies(df, columns=["hour"], drop_first=True).astype(int)
  df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
  df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
  df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
  df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
  # df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
  # df = pd.get_dummies(df, columns=["dayofweek", "month"], drop_first=True).astype(int)

  df['date_only'] = df.index.date
  #print(df['date_only'])
  years = df.index.year.unique()
  min_year, max_year = years.min(), years.max()
  de_holidays = holidays.country_holidays("DE", years=range(min_year, max_year+1))
  df['is_holiday'] = df['date_only'].isin(de_holidays).astype(int)
  # Lag features
  df["load_t-1"] = df["load"].shift(1)
  df["load_t-2"] = df["load"].shift(2)
  df["load_t-3"] = df["load"].shift(3)
  df["load_t-4"] = df["load"].shift(4)
  df["load_t-24"] = df["load"].shift(24)
  
  # lag weather since we technically can't use weather at t to predict t
  df["temp_t-1"] = df["temp"].shift(1)
  # df["temp_t-6"] = df["temp"].shift(6)
  
  df["rad_direct_t-1"] = df["rad_direct"].shift(1)
  # df["rad_direct_t-6"] = df["rad_direct"].shift(6)
  
  df["rad_diffuse_t-1"] = df["rad_diffuse"].shift(1)
  # df["rad_diffuse_t-6"] = df["rad_diffuse"].shift(6)

  # Rolling mean
  df["load_rolling_24h"] = df["load"].rolling(24).mean()
  

  # Drop rows with NA values (from shift/roll)
  df = df.dropna()
  y = df['load']
  df = df.drop(columns=['date_only', 'load', 'load_forecast', 'solar', 'solar_profile', 
                        'wind', 'wind_profile', 'temp', 'rad_direct',
                        'rad_diffuse', "dayofweek", "month"])
  features = df.columns
  time_index = df.index
  return df.to_numpy(), y.to_numpy(), features.tolist(), time_index

def feature_engineering_lstm(df):
    # Time features
  df["hour"] = df.index.hour
  df["dayofweek"] = df.index.dayofweek
  df["month"] = df.index.month
  #df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
  #df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
  #df = pd.get_dummies(df, columns=["hour"], drop_first=True).astype(int)
  df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
  df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
  df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
  df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
  df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
  df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
  # df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
  # df = pd.get_dummies(df, columns=["dayofweek", "month"], drop_first=True).astype(int)
  

  df['date_only'] = df.index.date
  #print(df['date_only'])
  years = df.index.year.unique()
  min_year, max_year = years.min(), years.max()
  de_holidays = holidays.country_holidays("DE", years=range(min_year, max_year+1))
  df['is_holiday'] = df['date_only'].isin(de_holidays).astype(int)
  
  # df["load-1"] = df["load"].shift(1)
  # # lag weather since we technically can't use weather at t to predict t
  # df["temp_t-1"] = df["temp"].shift(1)
  # # df["temp_t-6"] = df["temp"].shift(6)
  
  # df["rad_direct_t-1"] = df["rad_direct"].shift(1)
  # # df["rad_direct_t-6"] = df["rad_direct"].shift(6)
  
  # df["rad_diffuse_t-1"] = df["rad_diffuse"].shift(1)
  # # df["rad_diffuse_t-6"] = df["rad_diffuse"].shift(6)

  

  # Drop rows with NA values (from shift/roll)
  df = df.dropna()
  y = df['load']
  df = df.drop(columns=['date_only', 'load', 'load_forecast', 'solar', 'solar_profile', 
                        'wind', 'wind_profile', "dayofweek", "month"])
  features = df.columns
  time_index = df.index
  return df.to_numpy(), y.to_numpy(), features.tolist(), time_index