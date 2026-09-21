"""
Gate classifier training.

Trains a HistGradientBoostingClassifier to detect that a distribution shift
has just started (within the [s, s+50) onset window).

Train/test split strategy: held-out by shift schedule, not by timestep.
This avoids data leakage from splitting a continuous stream mid-run —
a model that sees part of a stream segment can trivially "learn" the rest
of the same segment.

Usage
-----
    python -m services.gate.train
"""
from __future__ import annotations

import sys
import os
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Add project root to path so imports work when run as __main__
sys.path.insert(0, str(Path(__file__).parents[2]))

from services.data.generator import generate_stream
from services.gate.features import extract_features
from services.seeds import GATE_TRAIN_SEEDS, HELD_OUT_TEST_SEEDS

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "gate_model.joblib"
META_PATH = ARTIFACT_DIR / "train_meta.txt"

# --- Reproducible generation parameters ---
N_TRAIN_STREAMS = len(GATE_TRAIN_SEEDS)
N_TEST_STREAMS = len(HELD_OUT_TEST_SEEDS)
T = 2000
N_SHIFTS = 3
TRAIN_SEED_OFFSET = GATE_TRAIN_SEEDS[0]
TEST_SEED_OFFSET = HELD_OUT_TEST_SEEDS[0]


def _build_dataset(
    n_streams: int, seed_offset: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n_streams independent streams and extract (X, y) for the latency metric.
    Each stream uses a unique seed so shift schedules are independent.
    """
    X_parts, y_parts = [], []
    for i in range(n_streams):
        sample, schedule = generate_stream(T=T, n_shifts=N_SHIFTS, seed=seed_offset + i)
        X, valid_idx = extract_features(sample.latency)
        y = sample.latency_label[valid_idx]
        X_parts.append(X)
        y_parts.append(y)
    return np.concatenate(X_parts), np.concatenate(y_parts)


def train(verbose: bool = True) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Generating training data...")
    X_train, y_train = _build_dataset(N_TRAIN_STREAMS, TRAIN_SEED_OFFSET)

    if verbose:
        print("Generating test data (held-out shift schedules)...")
    X_test, y_test = _build_dataset(N_TEST_STREAMS, TEST_SEED_OFFSET)

    if verbose:
        pos_rate = y_train.mean()
        print(f"Train: {len(X_train)} samples, {pos_rate*100:.1f}% positive")
        pos_rate_test = y_test.mean()
        print(f"Test:  {len(X_test)} samples, {pos_rate_test*100:.1f}% positive")

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        class_weight="balanced",  # handles positive-class imbalance
    )
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, proba_test)

    if verbose:
        print(f"\nHeld-out AUROC: {auroc:.4f}")

    joblib.dump(clf, MODEL_PATH)
    meta = (
        f"N_TRAIN_STREAMS={N_TRAIN_STREAMS}\n"
        f"N_TEST_STREAMS={N_TEST_STREAMS}\n"
        f"T={T}\n"
        f"N_SHIFTS={N_SHIFTS}\n"
        f"TRAIN_SEED_OFFSET={TRAIN_SEED_OFFSET}\n"
        f"TEST_SEED_OFFSET={TEST_SEED_OFFSET}\n"
        f"AUROC_TEST={auroc:.6f}\n"
    )
    META_PATH.write_text(meta)
    if verbose:
        print(f"Model saved to {MODEL_PATH}")
        print(f"Metadata saved to {META_PATH}")

    return auroc, X_test, y_test, proba_test


if __name__ == "__main__":
    train()
