from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model_artifact = joblib.load("models/demand_forecast_lr.pkl")


model = model_artifact["model"]
features = model_artifact["features"]
metrics = model_artifact["metrics"]


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

@app.get("/model/info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "features": features,
        "metrics": metrics
    }


@app.post("/predict")
def predict(data: PredictionInput):
    input_data = pd.DataFrame(
        [data.model_dump()]
    )

    input_data = input_data[features]

    prediction = model.predict(input_data)

    return {
        "prediction": float(prediction[0])
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }