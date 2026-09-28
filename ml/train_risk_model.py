"""
train_risk_model.py
--------------------
Trains an XGBoost regressor that predicts a 0-1 "disruption risk score"
for a road segment from weather/terrain/history features.

Real historical incident data for NER roads is thin, so this generates a
synthetic-but-realistic labeled dataset from known relationships:
  - heavier rainfall -> higher risk
  - steeper slope -> higher risk
  - looser/erosion-prone soil -> higher risk
  - past incidents on that segment -> higher risk
  - poor road condition -> higher risk
...with noise added, which is a standard and defensible approach for a
hackathon prototype. Swap in real IMD / GSI Bhusanket / SRTM data later
without changing anything downstream (the API contract stays the same).

Run:
    pip install xgboost pandas numpy scikit-learn joblib
    python train_risk_model.py

Produces:
    ../backend/models/risk_model.pkl
    ../backend/models/feature_importance.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit(
        "xgboost is not installed. Run: pip install xgboost\n"
        f"(original error: {e})"
    )

RNG = np.random.default_rng(42)
N_SAMPLES = 4000

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def generate_synthetic_dataset(n=N_SAMPLES) -> pd.DataFrame:
    rainfall_mm = RNG.gamma(shape=2.0, scale=15.0, size=n)                  # 0-150mm typical, long tail
    slope_deg = RNG.uniform(0, 45, size=n)                                  # terrain steepness
    soil_erodibility = RNG.uniform(0, 1, size=n)                            # 0 = stable rock, 1 = loose/erosion-prone
    past_incident_count = RNG.poisson(0.6, size=n)                         # landslides/washouts in last 3 yrs
    road_condition = RNG.integers(1, 6, size=n)                            # 1 = excellent, 5 = very poor
    is_monsoon_season = RNG.integers(0, 2, size=n)

    # True underlying relationship (nonlinear, with interactions)
    raw_risk = (
        0.012 * rainfall_mm
        + 0.02 * slope_deg
        + 1.4 * soil_erodibility
        + 0.35 * past_incident_count
        + 0.18 * road_condition
        + 0.5 * is_monsoon_season
        + 0.015 * rainfall_mm * soil_erodibility          # wet + loose soil compounds risk
        + 0.01 * slope_deg * is_monsoon_season             # steep + monsoon compounds risk
    )
    noise = RNG.normal(0, 0.6, size=n)
    risk_score = 1 / (1 + np.exp(-(raw_risk - 4) / 3 + noise))  # squash to 0-1

    return pd.DataFrame({
        "rainfall_mm": rainfall_mm,
        "slope_deg": slope_deg,
        "soil_erodibility": soil_erodibility,
        "past_incident_count": past_incident_count,
        "road_condition": road_condition,
        "is_monsoon_season": is_monsoon_season,
        "risk_score": risk_score,
    })


def main():
    df = generate_synthetic_dataset()
    feature_cols = [
        "rainfall_mm", "slope_deg", "soil_erodibility",
        "past_incident_count", "road_condition", "is_monsoon_season",
    ]
    X = df[feature_cols]
    y = df["risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R^2:  {r2:.4f}")

    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    print("\nFeature importances (for your judges / explainability slide):")
    for feat, score in importances.items():
        print(f"  {feat:22s} {score:.3f}")

    model_path = os.path.join(MODEL_DIR, "risk_model.pkl")
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
        json.dump(importances, f, indent=2)

    print(f"\nSaved model to: {os.path.abspath(model_path)}")


if __name__ == "__main__":
    main()
