"""
verify_forecaster.py — Sanity and integrity checks for the base forecaster.

Verifications:
  1. Artifact check: forecaster_model.joblib exists and loads.
  2. Causality check: confirms modifying future values stream[t >= k] does
     NOT alter forecasts y_hat[t < k]. Any difference flags a lookahead bug.
  3. Noise floor / Bayes error invariant check:
     For an AR(1) process with innovation sigma=5.0, sigma is the irreducible
     Bayes error floor. A causal model using only past observations cannot
     achieve RMSE < sigma asymptotically.
     - If RMSE < 4.70: FAILS with lookahead leakage error (peeking at y_t).
     - If RMSE > 5.50: FAILS with underfitting error.
     - PASSES if 4.70 <= RMSE <= 5.50.

Exits with code 1 on invariant violation.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parents[1]))

from services.forecaster.gbm import GBMForecaster
from services.forecaster.features import MIN_HISTORY
from services.forecaster.train import MODEL_PATH
from services.seeds import STATIONARY_SANITY_SEEDS
from services.data.generator import StreamSample, ShiftSchedule

SIGMA_NOISE = 5.0
TOL_LOWER = 4.70  # RMSE below this flags lookahead leakage
TOL_UPPER = 5.50  # RMSE above this flags severe underfitting
T = 2000

print("=" * 65)
print("Base Forecaster Verification (GBM)")
print("=" * 65)

# ── 1. Artifact Verification ──────────────────────────────────────────────────
print(f"\n[1. ARTIFACT CHECK] Checking {MODEL_PATH}...")
if not MODEL_PATH.exists():
    print(f"[FAIL] Forecaster model artifact not found at {MODEL_PATH}")
    sys.exit(1)

forecaster = GBMForecaster()
print(f"[PASS] Model artifact loaded successfully.")

# ── 2. Causality / Lookahead Verification ─────────────────────────────────────
print("\n[2. CAUSALITY CHECK] Verifying zero future lookahead...")
rng = np.random.default_rng(42)
stream_a = rng.normal(50.0, 5.0, size=T)
stream_b = stream_a.copy()
# Drastically alter future timesteps beyond t=1000
stream_b[1000:] += 500.0

preds_a = forecaster.predict_stream(stream_a)
preds_b = forecaster.predict_stream(stream_b)

diff_past = np.max(np.abs(preds_a[:1000] - preds_b[:1000]))
if diff_past > 1e-10:
    print(f"[FAIL] Causality violated! Modifying future stream altered past predictions by {diff_past:.4e}")
    sys.exit(1)
print(f"[PASS] Strictly causal: past forecasts identical under future perturbations (max diff = {diff_past:.1e})")

# ── 3. Theoretical Noise Floor Invariant Check ────────────────────────────────
print(f"\n[3. NOISE FLOOR CHECK] Evaluating stationary RMSE against sigma={SIGMA_NOISE:.2f}...")
print(f"  Theoretical Bayes error floor: {SIGMA_NOISE:.2f}")
print(f"  Valid expected range: [{TOL_LOWER:.2f}, {TOL_UPPER:.2f}]")

rmses, maes = [], []
for seed in STATIONARY_SANITY_SEEDS:
    rng = np.random.default_rng(seed)
    stream = np.zeros(T)
    stream[0] = 50.0
    for t in range(1, T):
        stream[t] = 0.8 * stream[t - 1] + 0.2 * 50.0 + rng.normal(0, SIGMA_NOISE)

    preds = forecaster.predict_stream(stream)
    # Evaluate after warmup
    y_true = stream[MIN_HISTORY:]
    y_pred = preds[MIN_HISTORY:]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmses.append(rmse)
    maes.append(mae)

mean_rmse = float(np.mean(rmses))
mean_mae = float(np.mean(maes))

print(f"  Measured Stationary RMSE: {mean_rmse:.4f} ± {np.std(rmses):.4f} across {len(rmses)} seeds")
print(f"  Measured Stationary MAE:  {mean_mae:.4f} ± {np.std(maes):.4f}")

ok = True
if mean_rmse < TOL_LOWER:
    print(f"[FAIL] RMSE {mean_rmse:.4f} < {TOL_LOWER:.2f} — Suspiciously below Bayes noise floor!")
    print("       This indicates feature lookahead leakage (peeking at y_t).")
    ok = False
elif mean_rmse > TOL_UPPER:
    print(f"[FAIL] RMSE {mean_rmse:.4f} > {TOL_UPPER:.2f} — Underfitting AR(1) autoregression.")
    ok = False
else:
    print(f"[PASS] Stationary RMSE {mean_rmse:.4f} sits properly at noise floor sigma={SIGMA_NOISE:.2f}")

if ok:
    print("\nAll forecaster checks passed.")
    sys.exit(0)
else:
    print("\nInvariant check failed.")
    sys.exit(1)
