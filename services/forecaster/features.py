"""
Causal feature extraction for time-series point forecasting.

To prevent decision-tree extrapolation failure during distribution shifts,
all features are strictly relative (lag differences, deviations from local
moving averages, and rolling standard deviations). No absolute levels are
used as input features.

At time t, only observations strictly before t (i.e. up to t-1) are used.
"""
from __future__ import annotations

import numpy as np

MIN_HISTORY = 30  # minimum past observations required to compute all rolling features
FEATURE_NAMES = [
    "diff_1",      # y_{t-1} - y_{t-2}
    "diff_2",      # y_{t-1} - y_{t-3}
    "diff_4",      # y_{t-1} - y_{t-5}
    "diff_9",      # y_{t-1} - y_{t-10}
    "dev_sma5",    # y_{t-1} - mean(y_{t-5:t})
    "dev_sma15",   # y_{t-1} - mean(y_{t-15:t})
    "dev_sma30",   # y_{t-1} - mean(y_{t-30:t})
    "std_5",       # std(y_{t-5:t})
    "std_15",      # std(y_{t-15:t})
    "std_30",      # std(y_{t-30:t})
]


def extract_features_at_step(history: np.ndarray) -> np.ndarray:
    """
    Extract feature vector for predicting step t given history y_0, ..., y_{t-1}.
    history must have length >= MIN_HISTORY.
    Returns 1D array of shape (len(FEATURE_NAMES),).
    """
    if len(history) < MIN_HISTORY:
        raise ValueError(f"History length {len(history)} < MIN_HISTORY ({MIN_HISTORY})")

    y_last = history[-1]
    y_m1 = history[-2]
    y_m2 = history[-3]
    y_m4 = history[-5]
    y_m9 = history[-10]

    w5 = history[-5:]
    w15 = history[-15:]
    w30 = history[-30:]

    features = np.array([
        y_last - y_m1,
        y_last - y_m2,
        y_last - y_m4,
        y_last - y_m9,
        y_last - np.mean(w5),
        y_last - np.mean(w15),
        y_last - np.mean(w30),
        float(np.std(w5)),
        float(np.std(w15)),
        float(np.std(w30)),
    ], dtype=np.float64)

    return features


def extract_dataset_from_stream(
    stream: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (X, y_delta, valid_timesteps) from a full 1D stream.

    X: shape (T - MIN_HISTORY, n_features)
    y_delta: shape (T - MIN_HISTORY,), where y_delta[i] = stream[t] - stream[t-1]
    valid_timesteps: array of indices t where forecast is made (range(MIN_HISTORY, T))
    """
    T = len(stream)
    if T <= MIN_HISTORY:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,)), np.empty((0,), dtype=int)

    valid_timesteps = np.arange(MIN_HISTORY, T)
    N = len(valid_timesteps)
    X = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float64)
    y_delta = np.zeros(N, dtype=np.float64)

    for idx, t in enumerate(valid_timesteps):
        history = stream[:t]
        X[idx] = extract_features_at_step(history)
        y_delta[idx] = stream[t] - stream[t - 1]

    return X, y_delta, valid_timesteps
