from pydantic import BaseModel


class PredictionInput(BaseModel):
    lag_1: float
    lag_7: float
    lag_14: float
    lag_28: float

    rolling_mean_7: float
    rolling_mean_28: float

    dow_sin: float
    dow_cos: float

    month_sin: float
    month_cos: float

    doy_sin: float
    doy_cos: float

    is_weekend: int