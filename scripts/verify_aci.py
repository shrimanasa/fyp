"""
verify_aci.py — Sanity check for the ACI implementation.

Checks:
  1. Pure stationary calibration check (0 shifts, target = 0.90 ± 0.03).
  2. Controlled single-shift evaluation (shift at t=500, T=2000, 10 seeds):
     - Displays flat [0, 100) near-shift average
     - Displays PRIMARY [0, 20) near-shift average where the true crater lives
     - Displays full 10-step binned curve showing initial dip and subsequent overshoot
     - Displays steady-state [850, 950) fully-adapted coverage
  3. Structural invariant checks on both stationary and random multi-shift streams.

Exits with code 1 only if a structural invariant is violated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from services.data.generator import generate_stream, StreamSample, ShiftSchedule
from services.conformal.evaluate import (
    evaluate_aci,
    evaluate_aci_controlled,
    summarise_controlled,
)

ALPHA_TARGET = 0.10
TARGET_COVERAGE = 1.0 - ALPHA_TARGET
N_REPS = 10          # seeds for controlled evaluation
T = 2000
SHIFT_AT = 500       # single shift at t=500 → steady-state window [1350,1450]
MIN_N = 50           # minimum samples per window to count a seed

print("=" * 65)
print("ACI Sanity & Calibration Check")
print("=" * 65)

# ── 1. Pure Stationary Sanity Check (0 shifts) ───────────────────────────────
print("\n[1. PURE STATIONARY] Zero shifts, AR(1) stream, 5 independent seeds")
stat_coverages = []
for seed in range(5):
    rng = np.random.default_rng(seed + 9000)
    stream = np.zeros(T)
    stream[0] = 50.0
    for t in range(1, T):
        stream[t] = 0.8 * stream[t - 1] + 0.2 * 50.0 + rng.normal(0, 5.0)
    sample = StreamSample(
        latency=stream,
        error_rate=rng.poisson(2.0, T).astype(float),
        latency_label=np.zeros(T, dtype=int),
        error_rate_label=np.zeros(T, dtype=int),
    )
    sched = ShiftSchedule(latency_shifts=np.array([]), error_rate_shifts=np.array([]))
    r = evaluate_aci(sample, sched, metric="latency")
    stat_coverages.append(r.overall)

stat_mean = float(np.mean(stat_coverages))
stat_std = float(np.std(stat_coverages))
print(f"  Target coverage:                {TARGET_COVERAGE:.4f}")
print(f"  Empirical stationary coverage:  {stat_mean:.4f} ± {stat_std:.4f} (across 5 seeds)")

# ── 2. Controlled single-shift evaluation ─────────────────────────────────────
print(f"\n[2. CONTROLLED SINGLE-SHIFT] Shift at t={SHIFT_AT}, T={T}, {N_REPS} seeds")
print(f"  Near-shift window:   t=[{SHIFT_AT}, {SHIFT_AT+100})")
print(f"  Primary dip window:  t=[{SHIFT_AT}, {SHIFT_AT+20})  (first 20 steps post-onset)")
print(f"  Steady-state window: t=[{SHIFT_AT+850}, {SHIFT_AT+950})")
print()

ctrl_results = evaluate_aci_controlled(
    shift_at=SHIFT_AT, T=T, n_seeds=N_REPS
)

# Print per-seed breakdown
print(f"  {'Seed':>4}  {'Overall':>8}  {'Steps 0–19':>12}  {'Steps 0–99':>12}  {'Steady-state':>14}")
print(f"  {'-'*4}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*14}")
for r in ctrl_results:
    print(
        f"  {r.seed:>4}  "
        f"{r.overall.coverage:.4f}    "
        f"{r.near_shift_first20.coverage:.4f} (n={r.near_shift_first20.n_samples})  "
        f"{r.near_shift.coverage:.4f} (n={r.near_shift.n_samples})  "
        f"{r.steady_state.coverage:.4f} (n={r.steady_state.n_samples})"
    )

summary = summarise_controlled(ctrl_results, min_n=MIN_N)

print()
print("  Summary Metrics:")
print(f"    Overall:                  {summary['overall']['mean']:.4f} ± {summary['overall']['std']:.4f} (n_seeds={summary['overall']['n_seeds_used']})")
print(f"    Near-shift (0–19, DIP):   {summary['near_shift_first20']['mean']:.4f} ± {summary['near_shift_first20']['std']:.4f}  <-- PRIMARY METRIC FOR PHASE 3")
print(f"    Near-shift (0–99, flat):  {summary['near_shift']['mean']:.4f} ± {summary['near_shift']['std']:.4f}  (masks dip via overshoot)")
print(f"    Steady-state (850–950):   {summary['steady_state']['mean']:.4f} ± {summary['steady_state']['std']:.4f} (fully-adapted)")

print("\n  Distance-from-shift 10-step Binned Curve (Steps post-onset):")
for b in summary["bins"]:
    print(f"    steps {b['bin_start']:>2}–{b['bin_end']-1:>2}:  {b['mean']:.4f} ± {b['std']:.4f}  (n_seeds={b['n_seeds_used']})")

# ── 3. Structural invariant check (random multi-shift streams) ────────────────
print("\n" + "=" * 65)
print("[3. INVARIANT CHECKS]")
print("=" * 65)

shift_near, shift_overall = [], []
for seed in range(5):
    sample, schedule = generate_stream(T=T, n_shifts=3, seed=seed + 8000)
    r = evaluate_aci(sample, schedule, metric="latency")
    shift_overall.append(r.overall)
    if not np.isnan(r.near_shift_0_100):
        shift_near.append(r.near_shift_0_100)

shift_overall_mean = np.mean(shift_overall)
shift_near_mean = np.mean(shift_near) if shift_near else float("nan")

print(f"  Random 3-shift streams: overall={shift_overall_mean:.4f}, near-shift={shift_near_mean:.4f}")

ok = True

# Invariant 1: Stationary coverage must be well-calibrated (±3pp of 0.90)
if abs(stat_mean - TARGET_COVERAGE) > 0.03:
    print(f"[FAIL] Stationary coverage {stat_mean:.4f} is >3pp away from {TARGET_COVERAGE:.2f}")
    ok = False
else:
    print(f"[PASS] Pure stationary coverage is calibrated ({stat_mean:.4f} vs target {TARGET_COVERAGE:.2f})")

# Invariant 2: Near-shift dip in first 20 steps should be observed (< overall)
dip_mean = summary["near_shift_first20"]["mean"]
ctrl_overall = summary["overall"]["mean"]
if not np.isnan(dip_mean) and dip_mean >= ctrl_overall:
    print(f"[WARN] Initial 20-step coverage ({dip_mean:.4f}) did not show expected dip below overall ({ctrl_overall:.4f})")
else:
    print(f"[PASS] Initial 20-step crater observed ({dip_mean:.4f} < overall {ctrl_overall:.4f})")

# Invariant 3: Random multi-shift near-shift coverage <= overall
if not np.isnan(shift_near_mean) and shift_near_mean > shift_overall_mean:
    print(f"[FAIL] Random multi-shift near-shift ({shift_near_mean:.4f}) > overall ({shift_overall_mean:.4f})")
    ok = False
else:
    print(f"[PASS] Random multi-shift near-shift <= overall")

if ok:
    print("\nAll invariants passed.")
    sys.exit(0)
else:
    print("\nInvariant violation — see above.")
    sys.exit(1)
