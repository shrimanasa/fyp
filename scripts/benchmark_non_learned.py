"""
scripts/benchmark_non_learned.py — Non-learned detector benchmarks on validation seeds.

Calibrates:
  1. VolatilityRatioDetector (ratio of short w=5 to long w=50 residual std)
  2. ResidualCUSUMDetector (Page-Hinkley on standardized residuals)
on stationary validation streams to achieve a strict 5 episodes / 1,000 steps false alarm rate
(or ~5% duty cycle with H=20 refractory).

Then evaluates on 100 validation streams:
  - Detection delay distribution (median, 25th, 75th percentiles)
  - End-to-end ACI coverage, in-window width, and Winkler score when wired to Policy C (kappa=2.0)
  - Compares Variance Shifts and Mean Shifts
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from services.conformal.aci import GatedACIPredictor, ACIPredictor
from services.conformal.evaluate import evaluate_aci_controlled
from services.forecaster.gbm import GBMForecaster
from services.gate.baselines import VolatilityRatioDetector, ResidualCUSUMDetector
from services.seeds import VAL_SEEDS

SHIFT_AT = 500
T = 2000
SEEDS = list(VAL_SEEDS)  # 100 validation seeds
HORIZON = 20


def calibrate_thresholds():
    """Calibrate detector thresholds on stationary streams targeting ~5 episodes / 1000 steps."""
    forecaster = GBMForecaster()
    
    # 20 stationary streams for calibration
    all_resids = []
    for s in SEEDS[:20]:
        rng = np.random.default_rng(s)
        stream = np.zeros(T)
        stream[0] = 50.0
        for t in range(1, T):
            stream[t] = 0.8 * stream[t - 1] + 0.2 * 50.0 + rng.normal(0, 5.0)
        y_hat = forecaster.predict_stream(stream)
        all_resids.append(np.abs(stream - y_hat))

    # Grid search for VolatilityRatio threshold
    best_ratio_th = 2.0
    for th in np.arange(1.5, 4.0, 0.1):
        episodes = 0
        total_steps = 0
        for r in all_resids:
            det = VolatilityRatioDetector(w_short=5, w_long=50, threshold=th, horizon=HORIZON, refractory=HORIZON)
            alarms = det.detect(r)
            # count rising edges
            edges = np.sum((alarms[1:] & ~alarms[:-1]))
            episodes += edges
            total_steps += (T - 50)
        ep_rate = (episodes / total_steps) * 1000.0
        if ep_rate <= 5.0:
            best_ratio_th = float(th)
            break

    # Grid search for CUSUM threshold
    best_cusum_h = 10.0
    for h in np.arange(4.0, 20.0, 0.5):
        episodes = 0
        total_steps = 0
        for r in all_resids:
            det = ResidualCUSUMDetector(sigma_0=5.0, allowance_k=0.5, threshold_h=h, horizon=HORIZON, refractory=HORIZON)
            alarms = det.detect(r)
            edges = np.sum((alarms[1:] & ~alarms[:-1]))
            episodes += edges
            total_steps += (T - 50)
        ep_rate = (episodes / total_steps) * 1000.0
        if ep_rate <= 5.0:
            best_cusum_h = float(h)
            break

    return best_ratio_th, best_cusum_h


def evaluate_detector(detector_name: str, detector_factory, shift_type: str):
    forecaster = GBMForecaster()

    delays = []
    total_shifts = len(SEEDS)
    detected = 0
    fa_episodes = 0
    fa_steps = 0
    total_stat_steps = 0

    all_streams = []
    all_alarms = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        stream = np.zeros(T)
        stream[0] = 50.0
        mean_t = 50.0

        if shift_type == "mean":
            for t in range(1, T):
                if t == SHIFT_AT:
                    direction = rng.choice([-1, 1])
                    mean_t += direction * 4.0 * 5.0
                stream[t] = 0.8 * stream[t - 1] + 0.2 * mean_t + rng.normal(0, 5.0)
        elif shift_type == "variance":
            for t in range(1, T):
                sigma_t = 15.0 if t >= SHIFT_AT else 5.0
                stream[t] = 0.8 * stream[t - 1] + 0.2 * mean_t + rng.normal(0, sigma_t)

        y_hat = forecaster.predict_stream(stream)
        residuals = np.abs(stream - y_hat)

        det = detector_factory()
        alarms = det.detect(residuals)
        all_streams.append(stream)
        all_alarms.append(alarms)

        # Check detection within [shift_at, shift_at + 50)
        onset_alarms = np.where(alarms[SHIFT_AT: SHIFT_AT + 50])[0]
        if len(onset_alarms) > 0:
            delay = int(onset_alarms[0])
            delays.append(delay)
            detected += 1

        # False alarms in stationary zone [50, shift_at - 10)
        stat_alarms = alarms[50: SHIFT_AT - 10]
        edges = np.sum(stat_alarms[1:] & ~stat_alarms[:-1])
        fa_episodes += int(edges)
        fa_steps += int(stat_alarms.sum())
        total_stat_steps += len(stat_alarms)

    delays_arr = np.array(delays)
    recall = detected / total_shifts
    ep_rate = (fa_episodes / total_stat_steps) * 1000.0
    duty_cycle = (fa_steps / total_stat_steps) * 100.0
    median_d = float(np.median(delays_arr)) if len(delays_arr) > 0 else float("nan")
    p25 = float(np.percentile(delays_arr, 25)) if len(delays_arr) > 0 else float("nan")
    p75 = float(np.percentile(delays_arr, 75)) if len(delays_arr) > 0 else float("nan")

    print(f"\nDetector: {detector_name} on {shift_type.upper()} SHIFTS")
    print(f"  Shift Recall:       {recall:.1%} ({detected}/{total_shifts})")
    print(f"  Stationary FA Rate: {ep_rate:.2f} episodes / 1,000 steps (Duty Cycle = {duty_cycle:.2f}%)")
    print(f"  Detection Delay:    Median = {median_d:.1f} steps (25th={p25:.1f}, 75th={p75:.1f})")
    if len(delays_arr) > 0:
        d_le_1 = float(np.mean(delays_arr <= 1)) * 100.0
        d_le_2 = float(np.mean(delays_arr <= 2)) * 100.0
        d_le_4 = float(np.mean(delays_arr <= 4)) * 100.0
        print(f"  Lag Distribution:   <=1 step: {d_le_1:.1f}%, <=2 steps: {d_le_2:.1f}%, <=4 steps: {d_le_4:.1f}%")

    # Wire to Gated ACI with Policy C (kappa=2.0 for variance, kappa=0.5 for mean)
    kappa = 2.0 if shift_type == "variance" else 0.5
    res = evaluate_aci_controlled(
        shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm", shift_type=shift_type,
        predictor_factory=lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=kappa, horizon=HORIZON, update_on_raw=True, max_q=100.0),
        alarm_schedule_fn=lambda t_len, s_at: all_alarms[int(np.random.randint(0, len(all_alarms)))]  # placeholder per-seed below
    )
    # Re-evaluate with exact per-seed alarm schedule
    covs = []
    w20s = []
    winks = []
    for idx, seed in enumerate(SEEDS):
        res_single = evaluate_aci_controlled(
            shift_at=SHIFT_AT, T=T, seeds=[seed], forecaster="gbm", shift_type=shift_type,
            predictor_factory=lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=kappa, horizon=HORIZON, update_on_raw=True, max_q=100.0),
            alarm_schedule_fn=lambda t_len, s_at, a=all_alarms[idx]: a
        )
        covs.append(res_single[0].near_shift_first20.coverage)
        w20s.append(res_single[0].near_shift_first20.mean_width)
        winks.append(res_single[0].near_shift_first20.winkler_score)

    mean_cov = float(np.mean(covs))
    mean_w = float(np.mean(w20s))
    mean_wink = float(np.mean(winks))

    print(f"  End-to-End ACI (Policy C kappa={kappa:.1f}):")
    print(f"    Steps 0–19 Coverage: {mean_cov:.4f}")
    print(f"    In-Window Width:     {mean_w:.2f}")
    print(f"    In-Window Winkler:   {mean_wink:.2f}")

    return {
        "name": detector_name, "recall": recall, "ep_rate": ep_rate,
        "median_d": median_d, "cov": mean_cov, "width": mean_w, "winkler": mean_wink
    }


def main():
    print("=" * 75)
    print("CALIBRATING NON-LEARNED DETECTORS (Targeting ~5.0 FA episodes / 1,000 steps)")
    print("=" * 75)
    th_ratio, th_cusum = calibrate_thresholds()
    print(f"Calibrated Volatility Ratio Threshold: tau_R = {th_ratio:.2f}")
    print(f"Calibrated CUSUM Threshold:            h = {th_cusum:.2f}")

    # Evaluate on Variance Shifts
    print("\n" + "=" * 75)
    print("EVALUATING ON VARIANCE SHIFTS (sigma: 5.0 -> 15.0)")
    print("=" * 75)
    evaluate_detector("VolatilityRatio (w=5 vs w=50)",
                      lambda: VolatilityRatioDetector(threshold=th_ratio, horizon=HORIZON, refractory=HORIZON),
                      shift_type="variance")
    evaluate_detector("ResidualCUSUM (Page-Hinkley)",
                      lambda: ResidualCUSUMDetector(threshold_h=th_cusum, horizon=HORIZON, refractory=HORIZON),
                      shift_type="variance")

    # Evaluate on Mean Shifts
    print("\n" + "=" * 75)
    print("EVALUATING ON MEAN SHIFTS (Delta = 4*sigma = 20.0)")
    print("=" * 75)
    evaluate_detector("ResidualCUSUM (Page-Hinkley)",
                      lambda: ResidualCUSUMDetector(threshold_h=th_cusum, horizon=HORIZON, refractory=HORIZON),
                      shift_type="mean")


if __name__ == "__main__":
    main()
