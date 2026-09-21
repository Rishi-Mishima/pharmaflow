import numpy as np
import pandas as pd


def build_features(history: pd.DataFrame, target_date):
    df = history.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 28:
        raise ValueError(
            "At least 28 days of demand history are required"
        )

    target_date = pd.Timestamp(target_date)

    features = {
        "lag_1": df.iloc[-1]["demand"],
        "lag_7": df.iloc[-7]["demand"],
        "lag_14": df.iloc[-14]["demand"],
        "lag_28": df.iloc[-28]["demand"],

        "rolling_mean_7": df["demand"].iloc[-7:].mean(),
        "rolling_mean_28": df["demand"].iloc[-28:].mean(),

        "dow_sin": np.sin(
            2 * np.pi * target_date.dayofweek / 7
        ),
        "dow_cos": np.cos(
            2 * np.pi * target_date.dayofweek / 7
        ),

        "month_sin": np.sin(
            2 * np.pi * target_date.month / 12
        ),
        "month_cos": np.cos(
            2 * np.pi * target_date.month / 12
        ),

        "doy_sin": np.sin(
            2 * np.pi * target_date.dayofyear / 365.25
        ),
        "doy_cos": np.cos(
            2 * np.pi * target_date.dayofyear / 365.25
        ),

        "is_weekend": int(
            target_date.dayofweek >= 5
        ),
    }

    return pd.DataFrame([features])