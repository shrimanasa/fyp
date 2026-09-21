"""
Gate classifier evaluation: produces precision-recall and ROC curves,
and prints a candidate threshold table for Phase 1.

Saves figures to docs/figures/.
"""
from __future__ import annotations

import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parents[2]))

from services.data.generator import generate_stream
from services.gate.features import extract_features
from services.gate.train import N_TEST_STREAMS, TEST_SEED_OFFSET, T, N_SHIFTS, MODEL_PATH

FIGURES_DIR = Path(__file__).parents[2] / "docs" / "figures"


def load_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the same held-out test set used during training."""
    from services.gate.train import _build_dataset
    X_test, y_test = _build_dataset(N_TEST_STREAMS, TEST_SEED_OFFSET)
    clf = joblib.load(MODEL_PATH)
    proba = clf.predict_proba(X_test)[:, 1]
    return X_test, y_test, proba


def find_threshold_max_recall_at_fpr(
    y_true: np.ndarray, proba: np.ndarray, max_fpr: float = 0.05
) -> float:
    """
    Return the threshold that maximises recall subject to FPR <= max_fpr.
    Ties broken by taking the lowest threshold (higher recall preference).
    """
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    eligible = thresholds[fpr <= max_fpr]
    eligible_tpr = tpr[fpr <= max_fpr]
    if len(eligible) == 0:
        return thresholds[-1]
    best_idx = np.argmax(eligible_tpr)
    return float(eligible[best_idx])


def find_threshold_max_f1(
    y_true: np.ndarray, proba: np.ndarray
) -> float:
    """Return the threshold maximising F1."""
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    # Note: precision_recall_curve returns one more point than thresholds
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thresholds[np.argmax(f1s)])


def find_threshold_balanced_precision(
    y_true: np.ndarray, proba: np.ndarray, target_prec: float = 0.50
) -> float:
    """
    Return the threshold where precision first crosses target_prec
    (searching from low threshold to high).
    """
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    # prec increases as threshold increases; find first index >= target_prec
    eligible = thresholds[prec[:-1] >= target_prec]
    if len(eligible) == 0:
        return float(thresholds[-1])
    return float(eligible[0])


def threshold_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def plot_curves(
    y_true: np.ndarray,
    proba: np.ndarray,
    candidates: dict[str, float],
) -> tuple[Path, Path]:
    """Plot ROC and PR curves with candidate threshold markers."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true, proba)
    auroc = roc_auc_score(y_true, proba)
    prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_true, proba)

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, lw=1.5, label=f"ROC (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    colors = ["tab:red", "tab:green", "tab:orange"]
    for (name, thr), color in zip(candidates.items(), colors):
        # find closest threshold in roc_thresh
        idx = np.argmin(np.abs(roc_thresh - thr))
        ax.scatter(fpr_arr[idx], tpr_arr[idx], color=color, zorder=5,
                   s=80, label=f"{name} (thr={thr:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Drift-Gate ROC Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    roc_path = FIGURES_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- PR ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_arr, prec_arr, lw=1.5, label="PR curve")
    for (name, thr), color in zip(candidates.items(), colors):
        idx = np.argmin(np.abs(pr_thresh - thr))
        ax.scatter(rec_arr[idx], prec_arr[idx], color=color, zorder=5,
                   s=80, label=f"{name} (thr={thr:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Drift-Gate Precision-Recall Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    pr_path = FIGURES_DIR / "pr_curve.png"
    fig.savefig(pr_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return roc_path, pr_path


def evaluate_all(verbose: bool = True) -> dict:
    """
    Full evaluation: load held-out test data, compute curves, pick
    candidate thresholds, return results dict.
    """
    _, y_test, proba_test = load_test_data()

    auroc = roc_auc_score(y_test, proba_test)

    # --- three candidate thresholds ---
    thr_a = find_threshold_max_recall_at_fpr(y_test, proba_test, max_fpr=0.05)
    thr_b = find_threshold_max_f1(y_test, proba_test)
    thr_c = find_threshold_balanced_precision(y_test, proba_test, target_prec=0.50)
    thr_default = 0.5

    candidates = {
        "max_recall@FPR≤5%": thr_a,
        "max_F1": thr_b,
        "precision≥50%": thr_c,
    }

    metrics = {
        "auroc": auroc,
        "default_0.5": threshold_metrics(y_test, proba_test, thr_default),
    }
    for name, thr in candidates.items():
        metrics[name] = threshold_metrics(y_test, proba_test, thr)

    roc_path, pr_path = plot_curves(y_test, proba_test, candidates)

    if verbose:
        print(f"\nHeld-out AUROC: {auroc:.4f}")
        print(f"\n{'Candidate':<25} {'Thr':>6} {'Prec':>6} {'Rec':>6} {'FPR':>6} {'F1':>6}")
        print("-" * 60)
        for name in ["default_0.5"] + list(candidates.keys()):
            m = metrics[name]
            print(
                f"{name:<25} {m['threshold']:>6.3f} {m['precision']:>6.3f} "
                f"{m['recall']:>6.3f} {m['fpr']:>6.4f} {m['f1']:>6.3f}"
            )
        print(f"\nFigures: {roc_path}, {pr_path}")

    return metrics
