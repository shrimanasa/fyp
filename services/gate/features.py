"""
Sliding-window feature extractor for drift detection.

To guarantee shift-invariance across streams with arbitrary baseline levels
or random-walking regimes, NO absolute mean levels are used as features.
All features are strictly relative:
- Cross-window mean differences (short vs. medium vs. long)
- Normalized innovation z-scores relative to baseline
- Volatility ratios (std ratios across windows)
- Distribution shape (skewness, relative range)
- Autocorrelation changes (lag-1 autocorrelation)

Lookback windows: short=10, medium=30, long=60.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew as scipy_skew

WINDOWS = [10, 30, 60]

FEATURE_NAMES: list[str] = [
    # Immediate innovations relative to each window
    "dev_from_w10",
    "dev_from_w30",
    "dev_from_w60",
    "zscore_w10",
    "zscore_w30",
    "zscore_w60",
    # Cross-window mean divergence
    "mean_diff_w10_w30",
    "mean_diff_w10_w60",
    "mean_diff_w30_w60",
    "z_diff_w10_w60",
    # Volatility and volatility ratios
    "std_w10",
    "std_w30",
    "std_w60",
    "std_ratio_w10_w60",
    "std_ratio_w10_w30",
    # Distribution shape
    "skew_w10",
    "skew_w30",
    "skew_w60",
    "rel_range_w10",
    "rel_range_w30",
    "rel_range_w60",
    # Dynamics
    "autocorr_w10",
    "autocorr_w30",
    "autocorr_w60",
]


def _window_stats(x: np.ndarray) -> dict[str, float]:
    """Compute relative stats for a 1-D window array."""
    n = len(x)
    mean = float(x.mean())
    std = float(x.std()) if n > 1 else 0.0
    sk = float(scipy_skew(x)) if n >= 3 else 0.0
    rng = float(x.max() - x.min()) if n > 1 else 0.0

    # lag-1 autocorrelation
    if n >= 3 and std > 1e-6:
        dx = x - mean
        var = float(np.dot(dx, dx))
        ac = float(np.dot(dx[:-1], dx[1:]) / var) if var > 1e-10 else 0.0
    else:
        ac = 0.0

    return {"mean": mean, "std": std, "skew": sk, "range": rng, "autocorr": ac}


def extract_features_at_t(stream: np.ndarray, t: int) -> np.ndarray | None:
    """
    Extract shift-invariant feature vector at timestep t from stream[:t+1].
    Returns None if t < max(WINDOWS) - 1 (not enough history).
    """
    min_t = max(WINDOWS)
    if t < min_t - 1:
        return None

    y_t = stream[t]
    stats_per_window: dict[int, dict] = {}
    for w in WINDOWS:
        window = stream[max(0, t - w + 1): t + 1]
        stats_per_window[w] = _window_stats(window)

    s10 = stats_per_window[10]
    s30 = stats_per_window[30]
    s60 = stats_per_window[60]

    eps = 1e-4

    feats = [
        # Immediate innovations
        y_t - s10["mean"],
        y_t - s30["mean"],
        y_t - s60["mean"],
        (y_t - s10["mean"]) / (s10["std"] + eps),
        (y_t - s30["mean"]) / (s30["std"] + eps),
        (y_t - s60["mean"]) / (s60["std"] + eps),
        # Cross-window mean divergence
        s10["mean"] - s30["mean"],
        s10["mean"] - s60["mean"],
        s30["mean"] - s60["mean"],
        (s10["mean"] - s60["mean"]) / (s60["std"] + eps),
        # Volatility
        s10["std"],
        s30["std"],
        s60["std"],
        s10["std"] / (s60["std"] + eps),
        s10["std"] / (s30["std"] + eps),
        # Distribution shape
        s10["skew"],
        s30["skew"],
        s60["skew"],
        s10["range"] / (s10["std"] + eps),
        s30["range"] / (s30["std"] + eps),
        s60["range"] / (s60["std"] + eps),
        # Dynamics
        s10["autocorr"],
        s30["autocorr"],
        s60["autocorr"],
    ]

    return np.array(feats, dtype=np.float32)


def extract_features(stream: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract shift-invariant features for all valid timesteps in stream.
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
