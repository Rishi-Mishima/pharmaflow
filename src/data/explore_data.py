import os
import pandas as pd
import matplotlib.pyplot as plt

print(os.getcwd())

df = pd.read_csv("data/raw/sales_daily.csv")

# 转化日期
df["datum"] = pd.to_datetime(df["datum"])

print(df.head())

print("Shape:")
print(df.shape)


print("Columns:")
print(df.columns)

print("\nData Info:")
print(df.info())


# 检查日期顺序
print(df["datum"].head())
print(df["datum"].tail())
print(df["datum"].is_monotonic_increasing)

# check if any dates are missing
print("Duplicate dates: ", df["datum"].duplicated().sum())

# check if any dates missing
date_diff = df["datum"].diff()

print(date_diff.value_counts())

drug_columns = [
    "M01AB", "M01AE", "N02BA", "N02BE",
    "N05B", "N05C", "R03", "R06"
]

print(df[drug_columns].describe())

# 只取出日期和 N02BE
n02be = df[["datum", "N02BE"]]
print(n02be.head())
print(n02be.shape)

n02be = n02be.rename(columns={"N02BE": "demand"})
print(n02be.head())


# plot N02BE 的 daily demand: 时间序列 EDA
plt.plot(n02be["datum"], n02be["demand"])

plt.xlabel("Date")
plt.ylabel("Daily Demand")
plt.title("N02BE Daily Demand")

plt.show()


# Rolling mean
n02be["rolling_mean_30"] = n02be["demand"].rolling(window = 30).mean()

plt.plot(n02be["datum"], n02be["demand"], alpha=0.3, label="Daily Demand")

plt.plot(
    n02be["datum"],
    n02be["rolling_mean_30"],
    label="30-Day Rolling Mean"
)

plt.xlabel("Date")
plt.ylabel("Demand")
plt.title("N02BE Demand Trend")
plt.legend()

plt.show()


# sesonal
monthly_avg = n02be.groupby(n02be["datum"].dt.month)["demand"].mean()
print(monthly_avg)

## every year
n02be["year"] = n02be["datum"].dt.year
n02be["month"] = n02be["datum"].dt.month
print(n02be.head())

year_month_avg = n02be.groupby(["year", "month"])["demand"].mean()

print(year_month_avg)

monthly_by_year = year_month_avg.unstack(level=0)
print(monthly_by_year)


## plot monthly_by_year

monthly_by_year.plot()

plt.xlabel("Month")
plt.ylabel("Average Daily Demand")
plt.title("N02BE Monthly Demand by Year")

plt.xticks(range(1, 13))

plt.show()


## weekly seaonality
# 先创建 weekday 列
n02be["weekday"] = n02be["datum"].dt.day_name()

# 再按照 weekday 分组
weekday_avg = n02be.groupby("weekday")["demand"].mean()

print(weekday_avg)

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_avg = weekday_avg.reindex(weekday_order)

print(weekday_avg)

weekday_avg.plot(kind="bar")

plt.xlabel("Weekday")
plt.ylabel("Average Daily Demand")
plt.title("N02BE Average Demand by Weekday")

plt.xticks(rotation=45)

plt.show()


## demand hist
plt.hist(n02be["demand"], bins=30)

plt.xlabel("Daily Demand")
plt.ylabel("Frequency")
plt.title("Distribution of N02BE Daily Demand")

plt.show()

Q1 = n02be["demand"].quantile(0.25)
Q3 = n02be["demand"].quantile(0.75)

print("Q1:", Q1)
print("Q3:", Q3)

IQR = Q3 - Q1
print("IQR:", IQR)

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

outliers = n02be[
    (n02be["demand"] < lower_bound) |
    (n02be["demand"] > upper_bound)
]

print(outliers[["datum", "demand"]])
print("Number of outliers:", len(outliers))