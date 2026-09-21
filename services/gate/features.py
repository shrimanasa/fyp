"""
Sliding-window feature extractor for drift detection.

For each timestep t, computes summary statistics over three lookback windows
(short=10, medium=30, long=60). Features are designed to capture:
- Level shift (mean change between windows)
- Volatility shift (std change)
- Autocorrelation change (AR-1 proxy)
- Distribution shape (skew, range)

These are computed for a single 1-D metric stream. For a multi-metric gate,
call this per metric and concatenate.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew as scipy_skew


WINDOWS = [10, 30, 60]
FEATURE_NAMES: list[str] = []

# Build canonical feature name list (used by gate/train.py)
for w in WINDOWS:
    for stat in ["mean", "std", "skew", "range", "autocorr_lag1"]:
        FEATURE_NAMES.append(f"w{w}_{stat}")
for w1, w2 in [(10, 30), (30, 60)]:
    FEATURE_NAMES.append(f"mean_diff_w{w1}_w{w2}")
    FEATURE_NAMES.append(f"std_diff_w{w1}_w{w2}")


def _window_stats(x: np.ndarray) -> dict[str, float]:
    """Compute stats for a 1-D window array."""
    n = len(x)
    mean = float(x.mean())
    std = float(x.std()) if n > 1 else 0.0
    sk = float(scipy_skew(x)) if n >= 3 else 0.0
    rng = float(x.max() - x.min()) if n > 1 else 0.0
    # lag-1 autocorrelation (Pearson of x[:-1], x[1:])
    if n >= 3:
        x1, x2 = x[:-1], x[1:]
        denom = x1.std() * x2.std()
        ac = float(np.corrcoef(x1, x2)[0, 1]) if denom > 1e-10 else 0.0
    else:
        ac = 0.0
    return {"mean": mean, "std": std, "skew": sk, "range": rng, "autocorr_lag1": ac}


def extract_features_at_t(stream: np.ndarray, t: int) -> np.ndarray | None:
    """
    Extract feature vector at timestep t from stream[:t+1].
    Returns None if t < max(WINDOWS) - 1 (not enough history).
    """
    min_t = max(WINDOWS)
    if t < min_t - 1:
        return None

    feats: list[float] = []
    stats_per_window: dict[int, dict] = {}
    for w in WINDOWS:
        window = stream[max(0, t - w + 1): t + 1]
        s = _window_stats(window)
        stats_per_window[w] = s
        feats.extend([s["mean"], s["std"], s["skew"], s["range"], s["autocorr_lag1"]])

    # cross-window differences
    for w1, w2 in [(10, 30), (30, 60)]:
        feats.append(stats_per_window[w1]["mean"] - stats_per_window[w2]["mean"])
        feats.append(stats_per_window[w1]["std"] - stats_per_window[w2]["std"])

    return np.array(feats, dtype=np.float32)


def extract_features(stream: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features for all valid timesteps in stream.

    Returns
    -------
    X : shape (n_valid, n_features)
    valid_indices : shape (n_valid,) — timestep indices with full history
    """
    min_t = max(WINDOWS) - 1
    rows = []
    indices = []
    for t in range(min_t, len(stream)):
        fv = extract_features_at_t(stream, t)
        if fv is not None:
            rows.append(fv)
            indices.append(t)
    return np.stack(rows), np.array(indices)
