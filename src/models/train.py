import pandas as pd
from sklearn.metrics import mean_absolute_error


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