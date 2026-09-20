

# PharmaML – Pharmaceutical Demand Forecasting

## Overview
A machine learning project for forecasting daily pharmaceutical demand
using historical demand patterns and calendar-based features.

## Project Goals
- Forecast daily pharmaceutical demand
- Compare simple baselines with machine learning models
- Evaluate models using time-aware validation
- Build a reproducible forecasting pipeline

## Dataset
Briefly describe:
- what the dataset contains
- target variable: `demand`
- date variable: `datum`

## Exploratory Data Analysis
Briefly describe:
- demand distribution
- weekly patterns
- seasonal patterns
- unusual zero-demand observations
- demand spikes

## Feature Engineering

### Lag Features
- lag_1
- lag_7
- lag_14
- lag_28

### Rolling Features
- rolling_mean_7
- rolling_mean_28

Rolling features are shifted by one day to prevent target leakage.

### Calendar Features
- day of week
- month
- day of year
- weekend indicator

Cyclical sine/cosine encoding was also evaluated for the
Linear Regression model.

## Models

Three approaches were evaluated:

1. Naive weekly baseline
2. Linear Regression
3. XGBoost

## Validation Strategy

A chronological train/test split was used instead of random splitting
to preserve the temporal structure of the forecasting problem.

Five-fold time-series cross-validation was additionally used for
model comparison.

## Model Comparison

| Model | CV MAE |
|---|---:|
| Linear Regression | 9.229 |
| Tuned XGBoost | 9.559 |

Although XGBoost performed competitively, Linear Regression showed
better average generalization across chronological validation windows.

## Final Model

Linear Regression was selected as the final model.

Holdout performance:

| Metric | Result |
|---|---:|
| MAE | 9.456 |
| RMSE | 12.299 |
| WAPE | 30.86% |

Compared with the naive weekly baseline (MAE = 12.261), the final
model reduced holdout MAE by approximately 22.9%.

## Error Analysis

The model captures the overall demand trend but tends to underestimate
sudden demand spikes.

Several zero-demand observations were also identified and retained
because their underlying cause could not be verified.

## Project Structure

```text
PharmaML/
├── data/
├── models/
│   └── demand_forecast_lr.pkl
├── src/
│   ├── data/
│   │  └── explore_data.py
│   ├── features/
│   │  └── build_features.py
│   └── models/
│       └── train.py
└── README.md