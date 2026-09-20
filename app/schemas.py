from pydantic import BaseModel
from datetime import date

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

class DrugCreate(BaseModel):
    code: str
    name: str


class DrugResponse(BaseModel):
    id: int
    code: str
    name: str

    model_config = {
        "from_attributes": True
    }

class DemandHistoryResponse(BaseModel):
    id: int
    drug_id: int
    date: date
    demand: float

    model_config = {
        "from_attributes": True
    }