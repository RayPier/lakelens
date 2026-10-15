
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "processed" / "algae_with_weather.csv",
    parse_dates=["date"]
)

df = df.sort_values(["lake", "date"]).copy()

# Features known at the time of prediction
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["previous_mean"] = df.groupby("lake")[
    "detected_mean_cells_ml"
].shift(1)

df["previous_detection_fraction"] = df.groupby("lake")[
    "detection_fraction"
].shift(1)

df["days_since_previous"] = (
    df.groupby("lake")["date"].diff().dt.days
)

# Next observed satellite record
df["next_date"] = df.groupby("lake")["date"].shift(-1)
df["next_mean"] = df.groupby("lake")[
    "detected_mean_cells_ml"
].shift(-1)

# Only predict exactly seven days ahead
df["days_to_next"] = (
    df["next_date"] - df["date"]
).dt.days

df = df[df["days_to_next"] == 7].copy()

# Experimental elevated-indicator label
df["target"] = (
    df["next_mean"] > 100000
).astype(int)

# One-hot encode lake identity
df = pd.get_dummies(df, columns=["lake"])

features = [
    "detected_mean_cells_ml",
    "detection_fraction",
    "previous_mean",
    "previous_detection_fraction",
    "days_since_previous",
    "month",
    "day_of_year",
    "temp_mean_7d",
    "temp_max_7d",
    "rain_sum_7d",
    "rain_sum_14d",
    "wind_mean_7d",
] + [c for c in df.columns if c.startswith("lake_")]

df = df.dropna(subset=features + ["target"])

# Chronological train/test split
train = df[df["date"] < "2025-01-01"]
test = df[df["date"] >= "2025-01-01"]

X_train = train[features]
y_train = train["target"]
X_test = test[features]
y_test = test["target"]

print("Training samples:", len(train))
print("Testing samples:", len(test))
print("Positive rate train:", round(y_train.mean(), 3))
print("Positive rate test:", round(y_test.mean(), 3))

# Simple persistence baseline:
# next week's indicator equals this week's indicator
baseline_pred = (
    test["detected_mean_cells_ml"] > 100000
).astype(int)

print("\nPersistence baseline:")
print("Balanced accuracy:",
      round(balanced_accuracy_score(y_test, baseline_pred), 3))

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nRandom Forest:")
print("Balanced accuracy:",
      round(balanced_accuracy_score(y_test, pred), 3))

print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred))

print("\nFeature importance:")
print(
    pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False).head(10)
)


print("\n===== PERFORMANCE BY LAKE =====")

for lake_id in ["lew", "grp", "rr", "tex"]:
    lake_column = f"lake_{lake_id}"

    mask = test[lake_column].astype(bool)

    actual = y_test[mask]
    predicted = pred[mask]
    baseline = baseline_pred[mask]

    if len(actual) == 0:
        continue

    print(f"\nLake: {lake_id}")
    print("Samples:", len(actual))
    print("Elevated rate:", round(actual.mean(), 3))

    if actual.nunique() == 2:
        print(
            "RF balanced accuracy:",
            round(balanced_accuracy_score(actual, predicted), 3)
        )
        print(
            "Persistence balanced accuracy:",
            round(balanced_accuracy_score(actual, baseline), 3)
        )
    else:
        print("Only one actual class; balanced accuracy not comparable")

    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))


from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=4
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("\n===== XGBOOST RESULTS =====")

print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, xgb_pred), 3)
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        xgb_pred,
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, xgb_pred))

print("\nFeature importance:")
print(
    pd.Series(
        xgb_model.feature_importances_,
        index=features
    ).sort_values(ascending=False).head(10)
)