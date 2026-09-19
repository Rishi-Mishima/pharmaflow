import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


df = pd.read_csv(
    "data/processed/n02be_features.csv",
    parse_dates=["datum"]
)

print(df.head())
print(df.shape)

print("\nDate range:")
print(df["datum"].min(), "→", df["datum"].max())


## chronological split
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()


# Verify temporal ordering
assert train_df["datum"].max() < test_df["datum"].min()


print("Train:")
print(train_df["datum"].min(), "→", train_df["datum"].max())
print("Rows:", len(train_df))

print("\nTest:")
print(test_df["datum"].min(), "→", test_df["datum"].max())
print("Rows:", len(test_df))


# Features
FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
    "day_of_year",
]


X_train = train_df[FEATURES]
y_train = train_df["demand"]

X_test = test_df[FEATURES]
y_test = test_df["demand"]

# Naive weekly baseline
baseline_pred = X_test["lag_7"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)


print("\nNaive Weekly Baseline")
print(f"MAE: {baseline_mae:.3f}")


## ---- linear regression ----
# Train Linear Regression
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

# Predict
lr_pred = lr_model.predict(X_test)

# Evaluate
lr_mae = mean_absolute_error(
    y_test,
    lr_pred
)

print("\nLinear Regression")
print(f"MAE: {lr_mae:.3f}")



xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
xgb_pred = xgb_model.predict(X_test)

# Evaluate
xgb_mae = mean_absolute_error(
    y_test,
    xgb_pred
)

print("\nXGBoost")
print(f"MAE: {xgb_mae:.3f}")


print("\nModel Comparison")
print(f"Naive Weekly Baseline: {baseline_mae:.3f}")
print(f"Linear Regression:     {lr_mae:.3f}")
print(f"XGBoost:               {xgb_mae:.3f}")

## ---- plot ----
plt.figure(figsize=(14, 6))

plt.plot(
    test_df["datum"],
    y_test,
    label="Actual",
)

plt.plot(
    test_df["datum"],
    baseline_pred,
    label="Weekly Baseline",
    alpha=0.7,
)

plt.plot(
    test_df["datum"],
    lr_pred,
    label="Linear Regression",
    alpha=0.8,
)

plt.plot(
    test_df["datum"],
    xgb_pred,
    label="XGBoost",
    alpha=0.8,
)

plt.xlabel("Date")
plt.ylabel("Demand")
plt.title("Actual vs Predicted Demand")
plt.legend()

plt.tight_layout()
plt.show()