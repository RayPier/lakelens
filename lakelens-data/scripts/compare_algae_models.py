
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "processed" / "algae_with_weather.csv",
    parse_dates=["date"]
)

df = df.sort_values(["lake", "date"]).copy()

df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

df["previous_mean"] = (
    df.groupby("lake")["detected_mean_cells_ml"].shift(1)
)

df["previous_detection_fraction"] = (
    df.groupby("lake")["detection_fraction"].shift(1)
)

df["days_since_previous"] = (
    df.groupby("lake")["date"].diff().dt.days
)

df["next_date"] = df.groupby("lake")["date"].shift(-1)
df["next_mean"] = (
    df.groupby("lake")["detected_mean_cells_ml"].shift(-1)
)

df["days_to_next"] = (
    df["next_date"] - df["date"]
).dt.days

# Evaluate only consecutive seven-day observations.
df = df[df["days_to_next"] == 7].copy()

df["target"] = (df["next_mean"] > 100000).astype(int)

df = pd.get_dummies(df, columns=["lake"])

base_features = [
    "detected_mean_cells_ml",
    "detection_fraction",
    "previous_mean",
    "previous_detection_fraction",
    "days_since_previous",
    "month",
    "day_of_year",
] + [c for c in df.columns if c.startswith("lake_")]

weather_features = [
    "temp_mean_7d",
    "temp_max_7d",
    "rain_sum_7d",
    "rain_sum_14d",
    "wind_mean_7d",
]

# Same samples for every comparison
df = df.dropna(
    subset=base_features + weather_features + ["target"]
)

# Split by the target date to prevent training on
# outcomes that occur in the validation/test period.
train = df[df["next_date"] < "2024-01-01"]
val = df[
    (df["date"] >= "2024-01-01")
    & (df["next_date"] < "2025-01-01")
]
test = df[df["date"] >= "2025-01-01"]

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

def evaluate(name, actual, predicted):
    print(f"\n{name}")
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(actual, predicted), 3)
    )
    print(
        "Elevated recall:",
        round(recall_score(actual, predicted, zero_division=0), 3)
    )
    print(
        "Elevated precision:",
        round(precision_score(actual, predicted, zero_division=0), 3)
    )
    print(
        "F1:",
        round(f1_score(actual, predicted, zero_division=0), 3)
    )
    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))


def persistence(data):
    return (
        data["detected_mean_cells_ml"] > 100000
    ).astype(int)


def make_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )


candidates = {
    "Persistence": None,
    "RF without weather": base_features,
    "RF with weather": base_features + weather_features,
}

validation_scores = {}

print("\n===== VALIDATION: 2024 =====")

for name, features in candidates.items():
    if features is None:
        predicted = persistence(val)
    else:
        model = make_model()
        model.fit(train[features], train["target"])
        predicted = model.predict(val[features])

    evaluate(name, val["target"], predicted)

    validation_scores[name] = balanced_accuracy_score(
        val["target"], predicted
    )

winner = max(validation_scores, key=validation_scores.get)

print("\nSelected on 2024 validation:", winner)

# Refit selected model on all pre-2025 examples,
# then evaluate once on the later test period.
pretest = df[df["next_date"] < "2025-01-01"]

print("\n===== FINAL TEST: 2025–2026 =====")

if candidates[winner] is None:
    final_pred = persistence(test)
else:
    model = make_model()
    selected_features = candidates[winner]

    model.fit(
        pretest[selected_features],
        pretest["target"]
    )

    final_pred = model.predict(test[selected_features])

evaluate(winner, test["target"], final_pred)

print("\nFinal test samples by lake:")
for lake_id in ["lew", "grp", "rr", "tex"]:
    mask = test[f"lake_{lake_id}"].astype(bool)
    actual = test.loc[mask, "target"]
    predicted = final_pred[mask.to_numpy()]

    if len(actual) == 0:
        continue

    print(f"\n{lake_id}: {len(actual)} samples")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

    if actual.nunique() == 2:
        print(
            "Balanced accuracy:",
            round(balanced_accuracy_score(actual, predicted), 3)
        )


import joblib
from datetime import datetime, timezone

# Retrain the selected approach on all available
# labeled examples for deployment.
if winner != "Persistence":
    deployment_features = candidates[winner]

    deployment_model = make_model()
    deployment_model.fit(
        df[deployment_features],
        df["target"]
    )

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": deployment_model,
        "features": deployment_features,
        "model_name": winner,
        "target": "next_7d_detected_mean_above_100k",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(df),
    }

    print(
    "Latest training target date:",
    df["next_date"].max().date()
    )

    model_path = model_dir / "algae_model.joblib"
    joblib.dump(artifact, model_path)

    print("\nSaved deployment model:", model_path)
    print("Training samples:", len(df))
    print("Features:", len(deployment_features))
