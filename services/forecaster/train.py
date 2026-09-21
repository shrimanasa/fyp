"""
Forecaster training script.

Trains a HistGradientBoostingRegressor to predict one-step-ahead increments
Delta y_t = y_t - y_{t-1} using causal relative features.

Uses dedicated training streams (FORECASTER_TRAIN_SEEDS) completely disjoint
from gate training and test streams to prevent inter-component leakage.
"""
from __future__ import annotations

import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parents[2]))

from services.data.generator import generate_stream
from services.forecaster.features import extract_dataset_from_stream
from services.seeds import FORECASTER_TRAIN_SEEDS

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "forecaster_model.joblib"
META_PATH = ARTIFACT_DIR / "train_meta.txt"

T = 2000
N_SHIFTS = 3


def train(verbose: bool = True) -> tuple[HistGradientBoostingRegressor, dict]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Generating training data across {len(FORECASTER_TRAIN_SEEDS)} streams (seeds {min(FORECASTER_TRAIN_SEEDS)}..{max(FORECASTER_TRAIN_SEEDS)})...")

    X_list, y_list = [], []
    for seed in FORECASTER_TRAIN_SEEDS:
        sample, _ = generate_stream(T=T, n_shifts=N_SHIFTS, seed=seed)
        X, y_delta, _ = extract_dataset_from_stream(sample.latency)
        X_list.append(X)
        y_list.append(y_delta)

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    if verbose:
        print(f"Training HistGradientBoostingRegressor on {len(X_train)} samples with {X_train.shape[1]} features...")

    reg = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=5,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )
    reg.fit(X_train, y_train)

    y_pred_delta = reg.predict(X_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_delta)))
    train_mae = float(mean_absolute_error(y_train, y_pred_delta))

    joblib.dump(reg, MODEL_PATH)

    meta = {
        "n_samples": len(X_train),
        "n_features": X_train.shape[1],
        "train_rmse_delta": train_rmse,
        "train_mae_delta": train_mae,
        "train_seeds": f"{min(FORECASTER_TRAIN_SEEDS)}..{max(FORECASTER_TRAIN_SEEDS)}",
    }

    with open(META_PATH, "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    if verbose:
        print(f"Model saved to {MODEL_PATH}")
        print(f"Train increment RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")

    return reg, meta


if __name__ == "__main__":
    train(verbose=True)
