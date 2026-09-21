"""
verify_gate.py — Sanity check for the gate classifier.

Loads the trained model from services/gate/artifacts/ and evaluates it
on a fresh held-out set. Prints AUROC and the confusion matrix at the
default 0.5 threshold.

Exits with code 1 if:
- The model artifact doesn't exist (not trained yet)
- AUROC < 0.60 (near-random — likely a training or feature bug)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, confusion_matrix

from services.gate.train import MODEL_PATH, _build_dataset, N_TEST_STREAMS, TEST_SEED_OFFSET

print("=" * 60)
print("Gate Classifier Sanity Check")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"[FAIL] Model artifact not found at {MODEL_PATH}")
    print("       Run: python services/gate/train.py")
    sys.exit(1)

clf = joblib.load(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")

X_test, y_test = _build_dataset(N_TEST_STREAMS, TEST_SEED_OFFSET)
proba = clf.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, proba)

print(f"\nHeld-out AUROC: {auroc:.4f}")
print(f"Positive class rate: {y_test.mean()*100:.1f}%")

# Default threshold
y_pred_default = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default, labels=[0, 1]).ravel()
recall_default = tp / (tp + fn + 1e-9)
fpr_default = fp / (fp + tn + 1e-9)
print(f"\nAt default threshold=0.50:")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Recall (onset detection): {recall_default:.4f}")
print(f"  FPR   (false alarm rate): {fpr_default:.4f}")

print("\n--- Invariant checks ---")
ok = True
if auroc < 0.60:
    print(f"[FAIL] AUROC={auroc:.4f} is near-random (< 0.60) — likely a training or feature bug.")
    ok = False
else:
    print(f"[PASS] AUROC={auroc:.4f} shows non-trivial discrimination")

if ok:
    print("\nAll invariants passed.")
    sys.exit(0)
else:
    print("\nInvariant violation — see above.")
    sys.exit(1)
