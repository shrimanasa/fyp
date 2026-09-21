"""
run_oracle_experiment.py — Oracle Gate ceiling experiment on validation seeds.

Fires an oracle alarm exactly at t = shift_at + 1 to establish the theoretical
upper bound of what ANY causal detector can achieve, and compares against:
  1. Standard ACI (Gate OFF)
  2. Random-Trigger Control (same alarm rate, random timing)
  3. Always-On Width-Matched Control (uniform inflation matched to same overall width)

Evaluated across:
  - Condition 1: Mean Shifts (Delta = 4*sigma = 20.0, sigma=5.0)
  - Condition 2: Variance Shifts (sigma: 5.0 -> 15.0, 3x jump)

Uses 100 dedicated validation seeds (services.seeds.VAL_SEEDS = range(500, 600)).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from services.conformal.aci import ACIPredictor, GatedACIPredictor
from services.conformal.evaluate import evaluate_aci_controlled, ControlledCoverageResult
from services.seeds import VAL_SEEDS

SHIFT_AT = 500
T = 2000
SEEDS = list(VAL_SEEDS)  # 100 seeds


def oracle_alarm_schedule(horizon: int = 20):
    def _schedule(stream_len: int, shift_step: int) -> np.ndarray:
        alarms = np.zeros(stream_len, dtype=bool)
        start = shift_step + 1
        end = min(stream_len, start + horizon)
        alarms[start:end] = True
        return alarms
    return _schedule


def random_alarm_schedule(horizon: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    def _schedule(stream_len: int, shift_step: int) -> np.ndarray:
        alarms = np.zeros(stream_len, dtype=bool)
        # pick a random stationary start point far from shift
        valid_starts = [t for t in range(100, shift_step - 50) if t + horizon < shift_step]
        if valid_starts:
            start = int(rng.choice(valid_starts))
            alarms[start: start + horizon] = True
        return alarms
    return _schedule


def run_experiment_for_condition(shift_type: str):
    print("\n" + "=" * 95)
    print(f"ORACLE EXPERIMENT: {shift_type.upper()} SHIFTS (100 Validation Seeds {SEEDS[0]}..{SEEDS[-1]})")
    print("=" * 95)

    # 1. Standard ACI (Gate OFF)
    print("Running Baseline: Standard ACI (Gate OFF)...")
    base_res = evaluate_aci_controlled(
        shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm", shift_type=shift_type
    )

    base_first20 = np.array([r.near_shift_first20.coverage for r in base_res])
    base_b0 = np.array([r.near_shift_bins[0].coverage for r in base_res])
    base_overall = np.array([r.overall.coverage for r in base_res])
    base_width = np.array([r.overall.mean_width for r in base_res])

    print(f"  Standard ACI: Overall={np.mean(base_overall):.4f}, Steps 0–9={np.mean(base_b0):.4f}, "
          f"Steps 0–19={np.mean(base_first20):.4f}, Mean Width={np.mean(base_width):.2f}\n")

    policies = [
        ("Policy A: gamma-boost (gamma=0.05, H=20)",
         lambda: GatedACIPredictor(policy="gamma_boost", gamma_boost=0.05, horizon=20, max_q=100.0),
         oracle_alarm_schedule(20)),
        ("Policy B: window-flush (flush at s+1)",
         lambda: GatedACIPredictor(policy="window_flush", horizon=20, max_q=100.0),
         oracle_alarm_schedule(20)),
        ("Policy C: margin-buffer (kappa=0.50, H=20)",
         lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=0.50, horizon=20, max_q=100.0),
         oracle_alarm_schedule(20)),
        ("Policy C: margin-buffer (kappa=1.00, H=20)",
         lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=1.00, horizon=20, max_q=100.0),
         oracle_alarm_schedule(20)),
        ("Policy D: alpha-shift (alpha=0.02, H=20)",
         lambda: GatedACIPredictor(policy="alpha_shift", alpha_boost=0.02, horizon=20, max_q=100.0),
         oracle_alarm_schedule(20)),
    ]

    print(f"{'Method / Policy':<42} | {'Overall Cov':<11} | {'0–9 Cov':<11} | {'0–19 Cov':<11} | {'Mean Width':<10} | {'Δ vs Base':<12}")
    print("-" * 105)
    print(f"{'Standard ACI (Gate OFF)':<42} | {np.mean(base_overall):.4f}     | {np.mean(base_b0):.4f}     | {np.mean(base_first20):.4f}     | {np.mean(base_width):.2f}       | reference")

    results_summary = []

    for name, factory, sched_fn in policies:
        res = evaluate_aci_controlled(
            shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm",
            shift_type=shift_type, predictor_factory=factory, alarm_schedule_fn=sched_fn
        )
        first20 = np.array([r.near_shift_first20.coverage for r in res])
        b0 = np.array([r.near_shift_bins[0].coverage for r in res])
        overall = np.array([r.overall.coverage for r in res])
        width = np.array([r.overall.mean_width for r in res])

        diff_first20 = np.mean(first20 - base_first20)
        mean_w = np.mean(width)

        print(f"{name:<42} | {np.mean(overall):.4f}     | {np.mean(b0):.4f}     | {np.mean(first20):.4f}     | {mean_w:.2f}       | {diff_first20:+.4f}")
        results_summary.append({
            "name": name, "factory": factory, "sched_fn": sched_fn,
            "overall": np.mean(overall), "b0": np.mean(b0), "first20": np.mean(first20),
            "width": mean_w, "diff_first20": diff_first20, "first20_arr": first20
        })

    # Width-Matched Control & Random Control for the best Margin Buffer policy (Policy C kappa=0.50)
    print("\n" + "-" * 105)
    print("SCIENTIFIC CONTROLS (Evaluated on Policy C kappa=0.50):")
    print("-" * 105)

    pol_c = [r for r in results_summary if "kappa=0.50" in r["name"]][0]
    target_width = pol_c["width"]
    base_mean_w = np.mean(base_width)

    # 1. Random Trigger Control
    rand_res = evaluate_aci_controlled(
        shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm",
        shift_type=shift_type,
        predictor_factory=lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=0.50, horizon=20),
        alarm_schedule_fn=random_alarm_schedule(20)
    )
    rand_first20 = np.array([r.near_shift_first20.coverage for r in rand_res])
    rand_overall = np.array([r.overall.coverage for r in rand_res])
    rand_w = np.array([r.overall.mean_width for r in rand_res])

    print(f"{'Random-Trigger Control (stationary alarm)':<42} | {np.mean(rand_overall):.4f}     | "
          f"{np.mean([r.near_shift_bins[0].coverage for r in rand_res]):.4f}     | {np.mean(rand_first20):.4f}     | "
          f"{np.mean(rand_w):.2f}       | {np.mean(rand_first20 - base_first20):+.4f}")

    # 2. Always-On Width-Matched Control
    # Find uniform kappa such that base_width * (1 + kappa) matches target_width
    kappa_matched = (target_width - base_mean_w) / base_mean_w
    matched_res = evaluate_aci_controlled(
        shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm",
        shift_type=shift_type,
        predictor_factory=lambda: ACIPredictor(uniform_margin_kappa=kappa_matched)
    )
    matched_first20 = np.array([r.near_shift_first20.coverage for r in matched_res])
    matched_overall = np.array([r.overall.coverage for r in matched_res])
    matched_w = np.array([r.overall.mean_width for r in matched_res])

    print(f"{f'Always-On Width-Matched (kappa={kappa_matched:.4f})':<42} | {np.mean(matched_overall):.4f}     | "
          f"{np.mean([r.near_shift_bins[0].coverage for r in matched_res]):.4f}     | {np.mean(matched_first20):.4f}     | "
          f"{np.mean(matched_w):.2f}       | {np.mean(matched_first20 - base_first20):+.4f}")

    # Selective net gain
    selective_gain = pol_c["first20"] - np.mean(matched_first20)
    print("=" * 105)
    print(f"DECISION METRIC for {shift_type.upper()}:")
    print(f"  Oracle First-20 Coverage:          {pol_c['first20']:.4f}")
    print(f"  Always-On Width-Matched Coverage:  {np.mean(matched_first20):.4f}")
    print(f"  SELECTIVE NET GAIN (at matched w): {selective_gain:+.4f} ({selective_gain * 100:+.2f} percentage points)")
    print("=" * 105)

    return selective_gain


def main():
    print("Starting Oracle Gate experiment across 100 validation seeds...")
    mean_gain = run_experiment_for_condition("mean")
    var_gain = run_experiment_for_condition("variance")

    print("\n" + "#" * 60)
    print("FINAL EXPERIMENTAL VERDICT & ROADMAP IMPLICATION:")
    print("#" * 60)
    print(f"Mean Shift Selective Gain:     {mean_gain * 100:+.2f} pp")
    print(f"Variance Shift Selective Gain: {var_gain * 100:+.2f} pp")
    print("-" * 60)
    if mean_gain < 0.02:
        print("RESULT: Mean shift oracle gain is UNDER 2.0pp at matched width.")
        print("ACTION: Per protocol, DROP mean shifts as headline. Make VARIANCE SHIFTS the paper's primary claim!")
    else:
        print("RESULT: Mean shift oracle gain >= 2.0pp at matched width.")
    print("#" * 60)


if __name__ == "__main__":
    main()
