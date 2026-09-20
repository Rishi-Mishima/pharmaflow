# PharmaML

PharmaML is a machine learning project for pharmaceutical demand forecasting using historical demand data.

The project combines time-series feature engineering, model evaluation, and a FastAPI backend for serving predictions through a REST API.

## Features

- Pharmaceutical demand forecasting
- Time-series feature engineering
- Lag and rolling-window features
- Cyclical calendar features
- Linear Regression and XGBoost comparison
- Time-series cross-validation
- REST API built with FastAPI
- Saved model inference using Joblib

## Model

The forecasting model uses historical demand and calendar-based features, including:

- Lag features: 1, 7, 14, and 28 days
- Rolling mean: 7 and 28 days
- Day-of-week seasonality
- Monthly seasonality
- Annual seasonality
- Weekend indicator

Linear Regression was selected as the final model based on time-series cross-validation performance.

### Model Performance

| Metric | Result |
|---|---:|
| Cross-validation MAE | 9.229 |
| Test MAE | 9.456 |
| Test RMSE | 12.299 |
| Test WAPE | 30.86% |

## API

The trained model is served through a FastAPI REST API.

Available endpoints:

- `GET /` — API status
- `GET /health` — health check
- `GET /model/info` — model metadata and evaluation metrics
- `POST /predict` — generate a demand prediction

## Run Locally

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```


### Start the API: 
```bash 
uvicorn app.main:app --reload
```

Open the interactive API documentation: [Port](http://127.0.0.1:8000/docs)

## Project Structure 
```
PharmaML/
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── demand_forecast_lr.pkl
├── src/
│   └── train.py
├── requirements.txt
└── README.md
```

### Planned Improvements
- PostgreSQL for historical demand and forecast persistence
- Redis caching
- Service and repository architecture
- Automated tests with pytest
- Docker containerization
- GitHub Actions CI
- React forecasting dashboard