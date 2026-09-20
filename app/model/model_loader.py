import joblib


model_artifact = joblib.load("models/demand_forecast_lr.pkl")

model = model_artifact["model"]
features = model_artifact["features"]
metrics = model_artifact["metrics"]