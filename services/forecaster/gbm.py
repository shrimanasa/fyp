"""
GBM Forecaster for streaming time-series point prediction.

Uses a trained HistGradientBoostingRegressor on causal lag and trend differences
to forecast one-step-ahead values:
    y_hat_t = y_{t-1} + Delta_hat_t

Cold start: For timesteps t < MIN_HISTORY (30 steps), falls back to EWMA.
"""
from __future__ import annotations

import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parents[2]))

from services.forecaster.features import (
    MIN_HISTORY,
    extract_features_at_step,
    extract_dataset_from_stream,
)
from services.forecaster.train import MODEL_PATH, train


class GBMForecaster:
    def __init__(self, model: HistGradientBoostingRegressor | None = None, alpha_ewma_warmup: float = 0.3):
        self.alpha_ewma_warmup = alpha_ewma_warmup
        if model is not None:
            self.model = model
        elif MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model, _ = train(verbose=False)

    def predict_step(self, history: np.ndarray) -> float:
        """
        One-step-ahead forecast at step t given past observations history = stream[:t].
        """
        t = len(history)
        if t == 0:
            return 50.0  # default prior
        if t == 1:
            return float(history[0])
        if t < MIN_HISTORY:
            # EWMA warmup for cold start
            pred = history[0]
            for val in history[1:]:
                pred = self.alpha_ewma_warmup * val + (1.0 - self.alpha_ewma_warmup) * pred
            return float(pred)

        feat = extract_features_at_step(history).reshape(1, -1)
        delta_hat = float(self.model.predict(feat)[0])
        return float(history[-1] + delta_hat)

    def predict_stream(self, stream: np.ndarray) -> np.ndarray:
        """
        Produce strictly causal one-step-ahead forecasts y_hat_0, ..., y_hat_{T-1}.
        Uses vectorized dataset extraction for speed.
        """
        T = len(stream)
        preds = np.zeros(T, dtype=np.float64)
        if T == 0:
            return preds

        # Cold start warmup via EWMA
        preds[0] = stream[0]
        ewma_val = stream[0]
        warmup_end = min(T, MIN_HISTORY)
        for t in range(1, warmup_end):
            preds[t] = ewma_val
            ewma_val = self.alpha_ewma_warmup * stream[t] + (1.0 - self.alpha_ewma_warmup) * ewma_val

        if T > MIN_HISTORY:
            X, _, valid_timesteps = extract_dataset_from_stream(stream)
            delta_hat = self.model.predict(X)
            # y_hat_t = stream[t-1] + delta_hat
            preds[valid_timesteps] = stream[valid_timesteps - 1] + delta_hat

        return preds
