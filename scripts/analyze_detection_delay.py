"""
analyze_detection_delay.py — Detection delay distribution of the drift gate.

Evaluates how quickly the gate fires following a distribution shift onset,
measuring event-level latency rather than pooled per-timestep recall.

Definitions:
  For a shift event at timestep s:
  - Detection occurs if any t in [s, s + 50) has P(shift)_t >= threshold.
  - Detection delay: Delta = min{t >= s : P(shift)_t >= threshold} - s.
  - Missed event: gate never triggers in [s, s + 50) (Delta = inf).
  - False alarm: gate triggers at t when distance to nearest shift > 50.

Evaluates across 50 held-out test streams (150 shift events).
Uses vectorized feature extraction and batch predict_proba for fast execution.
"""
from __future__ import annotations

import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from services.data.generator import generate_stream
from services.gate.features import extract_features, WINDOWS
from services.gate.train import MODEL_PATH

N_STREAMS = 50
TEST_SEED_START = 1000
T = 2000
N_SHIFTS = 3
ONSET_WINDOW = 50

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.6597, 0.70]


def main():
    print("=" * 65)
    print("Drift-Gate Detection Delay Analysis (Event-Level Latency)")
    print("=" * 65)

    if not MODEL_PATH.exists():
        print(f"Error: model artifact {MODEL_PATH} not found.")
        sys.exit(1)

    clf = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Precomputing probabilities on {N_STREAMS} held-out streams (seeds {TEST_SEED_START}..{TEST_SEED_START + N_STREAMS - 1})...\n")

    min_t = max(WINDOWS) - 1

    # Precompute probability arrays and shift schedules
    precomputed = []
    for i in range(N_STREAMS):
        sample, sched = generate_stream(T=T, n_shifts=N_SHIFTS, seed=TEST_SEED_START + i)
        X, valid_idx = extract_features(sample.latency)
        probs_stream = np.zeros(T, dtype=np.float32)
        if len(X) > 0:
            probs_stream[valid_idx] = clf.predict_proba(X)[:, 1]

        # Precompute stationary mask
        stationary_mask = np.ones(T, dtype=bool)
        stationary_mask[:min_t] = False
        for s in sched.latency_shifts:
            start_excl = max(0, s - 10)
            end_excl = min(T, s + 100)
            stationary_mask[start_excl:end_excl] = False

        valid_shifts = [s for s in sched.latency_shifts if s >= min_t and s + ONSET_WINDOW < T]
        precomputed.append((probs_stream, valid_shifts, stationary_mask))

    print(f"Done precomputing {N_STREAMS} streams. Analyzing threshold latency...\n")

    for tau in THRESHOLDS:
        all_delays = []
        total_shifts = 0
        total_detected = 0
        total_fa = 0
        total_stat_steps = 0

        for probs_stream, valid_shifts, stationary_mask in precomputed:
            triggers = probs_stream >= tau
            total_stat_steps += int(stationary_mask.sum())
            total_fa += int(triggers[stationary_mask].sum())

            for s in valid_shifts:
                total_shifts += 1
                window_triggers = np.where(triggers[s: s + ONSET_WINDOW])[0]
                if len(window_triggers) > 0:
                    delay = int(window_triggers[0])
                    all_delays.append(delay)
                    total_detected += 1

        delays_arr = np.array(all_delays)
        event_recall = total_detected / total_shifts if total_shifts > 0 else 0.0
        fa_per_1000 = (total_fa / total_stat_steps * 1000.0) if total_stat_steps > 0 else 0.0

        print(f"Threshold tau = {tau:.4f}:")
        print(f"  Shift Event Recall: {event_recall:.1%} ({total_detected}/{total_shifts} shifts detected within 50 steps)")
        print(f"  False Alarms:       {fa_per_1000:.2f} per 1,000 stationary steps (total={total_fa})")

        if len(delays_arr) > 0:
            median_d = float(np.median(delays_arr))
            mean_d = float(np.mean(delays_arr))
            p25 = float(np.percentile(delays_arr, 25))
            p75 = float(np.percentile(delays_arr, 75))
            p90 = float(np.percentile(delays_arr, 90))

            pct_0_4 = float(np.mean(delays_arr < 5)) * 100.0
            pct_5_9 = float(np.mean((delays_arr >= 5) & (delays_arr < 10))) * 100.0
            pct_10_19 = float(np.mean((delays_arr >= 10) & (delays_arr < 20))) * 100.0
            pct_ge_20 = float(np.mean(delays_arr >= 20)) * 100.0

            print(f"  Detection Delay (for detected shifts):")
            print(f"    Median: {median_d:.1f} steps | Mean: {mean_d:.2f} ± {np.std(delays_arr):.2f} steps")
            print(f"    Quartiles: 25th={p25:.1f}, 75th={p75:.1f}, 90th={p90:.1f} steps")
            print(f"    Lag distribution:")
            print(f"      Steps  0– 4 (immediate / pre-dip):  {pct_0_4:>5.1f}%")
            print(f"      Steps  5– 9 (early / inside dip):   {pct_5_9:>5.1f}%")
            print(f"      Steps 10–19 (late / post-dip):      {pct_10_19:>5.1f}%")
            print(f"      Steps >= 20 (overshoot window):     {pct_ge_20:>5.1f}%")
        else:
            print("  No shifts detected.")
        print("-" * 55)


if __name__ == "__main__":
    main()
