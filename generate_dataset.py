"""
Creates bike_demand.csv (10,000 rows) with the SAME columns as the Kaggle
"Bike Sharing Demand" dataset. Use it if you cannot download the real one.
The demand follows realistic patterns: rush-hour peaks on working days,
afternoon peaks on weekends, more rides in warm/clear weather, fewer in rain.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2011-01-01", periods=10000, freq="h")
df = pd.DataFrame({"datetime": dates})

month = df.datetime.dt.month
hour = df.datetime.dt.hour
dow = df.datetime.dt.dayofweek

df["season"] = month.map(lambda m: 1 if m in (12, 1, 2) else 2 if m in (3, 4, 5) else 3 if m in (6, 7, 8) else 4)
df["holiday"] = ((dow < 5) & (rng.random(len(df)) < 0.02)).astype(int)
df["workingday"] = ((dow < 5) & (df.holiday == 0)).astype(int)
df["weather"] = rng.choice([1, 2, 3, 4], size=len(df), p=[0.66, 0.26, 0.079, 0.001])

base_temp = 15 + 12 * np.sin((month - 4) / 12 * 2 * np.pi) + 4 * np.sin((hour - 9) / 24 * 2 * np.pi)
df["temp"] = (base_temp + rng.normal(0, 2.5, len(df))).round(2).clip(0, 41)
df["atemp"] = (df.temp + rng.normal(1.5, 1.2, len(df))).round(2)
df["humidity"] = (60 + 12 * (df.weather - 1) + rng.normal(0, 12, len(df))).round().clip(10, 100).astype(int)
df["windspeed"] = np.abs(rng.normal(13, 8, len(df))).round(2).clip(0, 57)

work_curve = 40 + 260 * np.exp(-((hour - 8) ** 2) / 3) + 230 * np.exp(-((hour - 17.5) ** 2) / 4) + 60 * np.exp(-((hour - 13) ** 2) / 10)
off_curve = 15 + 190 * np.exp(-((hour - 14) ** 2) / 14)
night_drop = np.where(hour <= 5, 0.25, 1.0)
demand = np.where(df.workingday == 1, work_curve, off_curve) * night_drop
demand *= (0.5 + 0.03 * df.temp.clip(0, 35))
demand *= {1: 1.0, 2: 0.85, 3: 0.45, 4: 0.2}[1] * df.weather.map({1: 1.0, 2: 0.85, 3: 0.45, 4: 0.2})
demand *= (1 - 0.006 * df.humidity.clip(30, 100) + 0.3)
demand *= (1 + (dates.year - 2011) * 0.0)
demand *= rng.normal(1, 0.18, len(df))
df["count"] = np.maximum(demand.round(), 1).astype(int)
df["registered"] = (df["count"] * np.where(df.workingday == 1, 0.85, 0.65)).round().astype(int)
df["casual"] = df["count"] - df["registered"]

df = df[["datetime", "season", "holiday", "workingday", "weather", "temp", "atemp",
         "humidity", "windspeed", "casual", "registered", "count"]]
df.to_csv("bike_demand.csv", index=False)
print(df.shape)
print(df.head())
