import pandas as pd

import os

print("Working directory:", os.getcwd())

df = pd.read_csv("data/processed/n02be_daily.csv")

df["datum"] = pd.to_datetime(df["datum"])

df = df.sort_values("datum").reset_index(drop=True)

print(df.head())
print(df.shape)

# Lag features
df["lag_1"] = df["demand"].shift(1)
df["lag_7"] = df["demand"].shift(7)
df["lag_14"] = df["demand"].shift(14)
df["lag_28"] = df["demand"].shift(28)

## rolling features
df["rolling_mean_7"] = (
    df["demand"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

df["rolling_mean_28"] = (
    df["demand"]
    .shift(1)
    .rolling(window=28)
    .mean()
)

## calendar features
df["day_of_week"] = df["datum"].dt.dayofweek
df["month"] = df["datum"].dt.month
df["day_of_year"] = df["datum"].dt.dayofyear

## add is_weekend
df["is_weekend"] = (
    df["datum"].dt.dayofweek >= 5
).astype(int)

print("Missing values before dropna:")
print(df.isna().sum())


df = df.dropna().reset_index(drop=True)


df.to_csv(
    "data/processed/n02be_features.csv",
    index=False
)

print("\nFeature dataset:")
print(df.head())

print("\nShape:", df.shape)