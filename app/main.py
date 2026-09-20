from fastapi import FastAPI
import joblib
import pandas as pd
from app.schemas import PredictionInput
from app.services.forecast_service import make_prediction
from app.model.model_loader import model, features, metrics
app = FastAPI()


@app.get("/model/info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "features": features,
        "metrics": metrics
    }


@app.post("/predict")
def predict(data: PredictionInput):
    prediction = make_prediction(
        data,
        model,
        features
    )

    return {
        "prediction": prediction
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }