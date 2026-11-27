"""
BIKE / CAB DEMAND PREDICTION
Algorithms: Linear Regression, KNN, Naive Bayes, Decision Tree
Run:  python bike_demand_project.py      (bike_demand.csv must be in the same folder)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # remove this line if you want plots to pop up
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             r2_score, mean_absolute_error, mean_squared_error)

# ---------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("bike_demand.csv", parse_dates=["datetime"])
print("Shape:", df.shape)
print(df.head(), "\n")
print(df.describe().round(2), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")

# ---------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# Pull useful info out of the datetime column.
# 'casual' and 'registered' add up to 'count', so using them would be
# cheating (data leakage). We drop them.
# ---------------------------------------------------------------
df["hour"] = df.datetime.dt.hour
df["month"] = df.datetime.dt.month
df["dayofweek"] = df.datetime.dt.dayofweek
df = df.drop(columns=["datetime", "casual", "registered"])

# Category target for KNN / Naive Bayes / Decision Tree
df["demand_level"] = pd.qcut(df["count"], 3, labels=["Low", "Medium", "High"])
print("Class balance:\n", df.demand_level.value_counts(), "\n")

# ---------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS (graphs for your report)
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.lineplot(data=df, x="hour", y="count", hue="workingday", ax=ax[0], palette="Set1")
ax[0].set_title("Average rides by hour (1 = working day)")
sns.boxplot(data=df, x="weather", y="count", ax=ax[1])
ax[1].set_title("Rides by weather (1 clear ... 4 heavy rain)")
plt.tight_layout(); plt.savefig("01_eda_hour_weather.png", dpi=130); plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df.drop(columns="demand_level").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap"); plt.tight_layout()
plt.savefig("02_correlation.png", dpi=130); plt.close()

# ---------------------------------------------------------------
# STEP 4: TRAIN / TEST SPLIT
# ---------------------------------------------------------------
feature_cols = ["season", "holiday", "workingday", "weather", "temp", "atemp",
                "humidity", "windspeed", "hour", "month", "dayofweek"]
X = df[feature_cols]
y_num = df["count"]
y_cat = df["demand_level"]

X_train, X_test, yn_train, yn_test, yc_train, yc_test = train_test_split(
    X, y_num, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# KNN needs scaled features (distance based). Others don't.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ===============================================================
# PART A: CLASSIFICATION  (Low / Medium / High demand)
# ===============================================================
print("=" * 60, "\nPART A: CLASSIFICATION\n", "=" * 60)

# --- KNN: find the best K ---
k_values = range(1, 31)
k_scores = []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, yc_train)
    k_scores.append(accuracy_score(yc_test, m.predict(X_test_s)))
best_k = k_values[int(np.argmax(k_scores))]
plt.figure(figsize=(7, 4))
plt.plot(k_values, k_scores, marker="o")
plt.axvline(best_k, color="r", ls="--", label=f"best K = {best_k}")
plt.xlabel("K"); plt.ylabel("Accuracy"); plt.title("KNN: choosing K"); plt.legend()
plt.tight_layout(); plt.savefig("03_knn_k_vs_accuracy.png", dpi=130); plt.close()

knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train_s, yc_train)
nb = GaussianNB().fit(X_train, yc_train)

# --- Decision Tree: check depth to avoid overfitting ---
depths = range(1, 16)
tr_acc, te_acc = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, yc_train)
    tr_acc.append(t.score(X_train, yc_train)); te_acc.append(t.score(X_test, yc_test))
best_depth = depths[int(np.argmax(te_acc))]
plt.figure(figsize=(7, 4))
plt.plot(depths, tr_acc, label="Train"); plt.plot(depths, te_acc, label="Test")
plt.axvline(best_depth, color="r", ls="--", label=f"best depth = {best_depth}")
plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.title("Decision Tree: overfitting check")
plt.legend(); plt.tight_layout(); plt.savefig("04_tree_depth.png", dpi=130); plt.close()

dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42).fit(X_train, yc_train)

plt.figure(figsize=(20, 9))
plot_tree(DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, yc_train),
          feature_names=feature_cols, class_names=list(dt.classes_), filled=True, fontsize=9)
plt.title("Decision Tree (top 3 levels)"); plt.tight_layout()
plt.savefig("05_tree_structure.png", dpi=110); plt.close()

imp = pd.Series(dt.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(7, 5)); imp.plot(kind="barh")
plt.title("Feature importance (Decision Tree)"); plt.tight_layout()
plt.savefig("06_feature_importance.png", dpi=130); plt.close()
print("Feature importance:\n", imp.sort_values(ascending=False).round(3), "\n")

clf_models = {"KNN": (knn, X_test_s), "Naive Bayes": (nb, X_test), "Decision Tree": (dt, X_test)}
clf_rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (name, (model, Xt)) in zip(axes, clf_models.items()):
    pred = model.predict(Xt)
    clf_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(yc_test, pred),
        "Precision": precision_score(yc_test, pred, average="weighted"),
        "Recall": recall_score(yc_test, pred, average="weighted"),
        "F1-score": f1_score(yc_test, pred, average="weighted")})
    cm = confusion_matrix(yc_test, pred, labels=["Low", "Medium", "High"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
    ax.set_title(f"{name} confusion matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    print(f"--- {name} ---\n", classification_report(yc_test, pred), "\n")
plt.tight_layout(); plt.savefig("07_confusion_matrices.png", dpi=130); plt.close()

clf_table = pd.DataFrame(clf_rows).set_index("Model").round(3)
print("CLASSIFICATION COMPARISON\n", clf_table, "\n")

# ===============================================================
# PART B: REGRESSION  (predict exact number of rides)
# ===============================================================
print("=" * 60, "\nPART B: REGRESSION\n", "=" * 60)

def reg_metrics(name, y_true, y_pred):
    return {"Model": name, "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))}

reg_rows = []

# 1) Linear Regression on raw features
lr = LinearRegression().fit(X_train, yn_train)
lr_pred = lr.predict(X_test)
reg_rows.append(reg_metrics("Linear Regression (basic)", yn_test, lr_pred))

# 2) Linear Regression with 'hour' treated as a category (one-hot).
#    Demand vs hour is a curve (peaks at 8am and 5pm), not a straight line,
#    so a straight-line model needs help to understand it.
X_enc = pd.get_dummies(X.assign(hour=X.hour.astype(str)), columns=["hour"], dtype=int)
Xe_train, Xe_test = X_enc.loc[X_train.index], X_enc.loc[X_test.index]
lr2 = LinearRegression().fit(Xe_train, yn_train)
lr2_pred = lr2.predict(Xe_test)
reg_rows.append(reg_metrics("Linear Regression (hour one-hot)", yn_test, lr2_pred))

# 3) KNN and Decision Tree as regressors for comparison
knn_r = KNeighborsRegressor(n_neighbors=best_k).fit(X_train_s, yn_train)
reg_rows.append(reg_metrics("KNN Regressor", yn_test, knn_r.predict(X_test_s)))
dt_r = DecisionTreeRegressor(max_depth=best_depth, random_state=42).fit(X_train, yn_train)
dt_pred = dt_r.predict(X_test)
reg_rows.append(reg_metrics("Decision Tree Regressor", yn_test, dt_pred))

reg_table = pd.DataFrame(reg_rows).set_index("Model").round(3)
print("REGRESSION COMPARISON\n", reg_table, "\n")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].scatter(yn_test, lr_pred, alpha=0.3, s=8)
ax[0].plot([0, yn_test.max()], [0, yn_test.max()], "r--")
ax[0].set_title("Linear Reg (basic): actual vs predicted")
ax[1].scatter(yn_test, lr2_pred, alpha=0.3, s=8, color="green")
ax[1].plot([0, yn_test.max()], [0, yn_test.max()], "r--")
ax[1].set_title("Linear Reg (hour one-hot): actual vs predicted")
ax[2].scatter(lr2_pred, yn_test - lr2_pred, alpha=0.3, s=8, color="purple")
ax[2].axhline(0, color="r", ls="--"); ax[2].set_title("Residual plot (improved model)")
for a in ax: a.set_xlabel("Predicted" if a is not ax[0] else "Actual")
plt.tight_layout(); plt.savefig("08_regression_plots.png", dpi=130); plt.close()

print("Linear Regression coefficients (basic model):")
print(pd.Series(lr.coef_, index=feature_cols).round(3), "\n")

# ---------------------------------------------------------------
# STEP 5: SAVE RESULTS + SAMPLE PREDICTION
# ---------------------------------------------------------------
clf_table.to_csv("results_classification.csv"); reg_table.to_csv("results_regression.csv")

sample = pd.DataFrame([{"season": 3, "holiday": 0, "workingday": 1, "weather": 1, "temp": 28,
                        "atemp": 30, "humidity": 55, "windspeed": 10, "hour": 8,
                        "month": 7, "dayofweek": 2}])
print("SAMPLE PREDICTION: Wednesday 8 AM, July, clear weather, 28C")
print("  KNN class          :", knn.predict(scaler.transform(sample))[0])
print("  Naive Bayes class  :", nb.predict(sample)[0])
print("  Decision Tree class:", dt.predict(sample)[0])
print("  Decision Tree rides:", round(dt_r.predict(sample)[0]))
print("\nDone. Graphs and result CSVs are saved in this folder.")
