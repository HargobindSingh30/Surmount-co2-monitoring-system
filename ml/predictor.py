"""
Predictive Layer - predict when CO2 will enter HOLD zone (>= 490 ppm abs,
differential >= 90) within the next LOOK_AHEAD_MINUTES minutes.

ML value:
  Rule-based DDC: reacts at 520 ppm (damper opens after breach).
  ML layer:       predicts HOLD zone entry 10 min ahead (damper pre-opens),
                  reducing peak overshoot and recovery time.

Uses TimeSeriesSplit to prevent temporal data leakage.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PREDICT_THRESHOLD  = 90.0   # differential ppm = 490 ppm absolute
LOOK_AHEAD_MINUTES = 10
MODEL_PATH = os.path.join(os.path.dirname(__file__), "breach_predictor.pkl")

FEATURE_COLS = [
    "differential",
    "co2_level",
    "rate_of_change_5m",
    "rate_of_change_15m",
    "rolling_mean_10m",
    "rolling_std_10m",
    "rolling_max_10m",
    "hour_of_day",
    "occupancy",
    "damper_position",
]


def engineer_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    groups = []
    for sid, grp in df.groupby("sensor_id"):
        grp = grp.copy().reset_index(drop=True)
        grp["rate_of_change_5m"]  = grp["differential"].diff(5).fillna(0) / 5.0
        grp["rate_of_change_15m"] = grp["differential"].diff(15).fillna(0) / 15.0
        grp["rolling_mean_10m"]   = grp["differential"].rolling(10, min_periods=1).mean()
        grp["rolling_std_10m"]    = grp["differential"].rolling(10, min_periods=1).std().fillna(0)
        grp["rolling_max_10m"]    = grp["differential"].rolling(10, min_periods=1).max()
        grp["hour_of_day"]        = grp["timestamp"].dt.hour
        groups.append(grp)

    return pd.concat(groups, ignore_index=True)


def create_labels(df, look_ahead=LOOK_AHEAD_MINUTES):
    df = df.reset_index(drop=True)
    labels = {}
    for sid, grp in df.groupby("sensor_id"):
        idx   = grp.index.tolist()
        diffs = grp["differential"].values
        for i, global_i in enumerate(idx):
            future = diffs[i + 1: i + 1 + look_ahead]
            labels[global_i] = int(any(f >= PREDICT_THRESHOLD for f in future))
    return df.index.map(labels).fillna(0).astype(int)


def train(df_raw):
    df = engineer_features(df_raw)
    df["hold_soon"] = create_labels(df)

    pos = df["hold_soon"].sum()
    neg = (df["hold_soon"] == 0).sum()
    print(f"[ML] Labels — class 0: {neg:,}  class 1: {pos:,}  ratio: {pos/(pos+neg):.1%}")

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    X  = df[FEATURE_COLS].values
    y  = df["hold_soon"].values

    # Time-series cross-validation (no leakage)
    tscv     = TimeSeriesSplit(n_splits=5)
    fold_acc = []
    last_report = None

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        acc    = accuracy_score(y[test_idx], y_pred)
        fold_acc.append(acc)
        last_report = classification_report(
            y[test_idx], y_pred, output_dict=True, zero_division=0
        )
        print(f"  Fold {fold}: accuracy={acc:.3f}")

    mean_acc = float(np.mean(fold_acc))
    print(f"[ML] Mean CV accuracy: {mean_acc:.3f}")
    print(classification_report(
        y[test_idx], y_pred, zero_division=0
    ))

    # Retrain on all data for the deployed model
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"[ML] Model saved -> {MODEL_PATH}")

    return {"accuracy": mean_acc, "report": last_report, "fold_scores": fold_acc}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run train() first.")
    return joblib.load(MODEL_PATH)


def predict_proba_breach(df):
    model = load_model()
    feat  = engineer_features(df)
    X     = feat[FEATURE_COLS].fillna(0)
    return model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from simulator.runner import generate_coupled_data
    from datetime import datetime

    print("Generating 30 days of coupled data...")
    raw, _ = generate_coupled_data(days=30, start_time=datetime(2025, 8, 1, 0, 0))
    df = pd.DataFrame(raw)
    metrics = train(df)
    print(f"\nFold scores: {[round(s,3) for s in metrics['fold_scores']]}")
    print(f"Mean CV accuracy: {metrics['accuracy']:.1%}")
