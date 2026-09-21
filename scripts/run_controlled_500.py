"""
run_controlled_500.py — High-powered (500 seeds) paired evaluation with bootstrap CIs.

Evaluates EWMA vs. GBM point forecasters inside ACI on 500 independent
controlled single-shift streams (shift fixed at t=500, T=2000).

Computes:
  - Empirical coverage across all windows (Overall, 0–9, 10–19, 0–19, 20–49, 850–950)
  - Mean interval width across all windows
  - Paired per-seed differences (Delta = GBM - EWMA)
  - 95% Bootstrap Confidence Intervals (B=2000 resamples)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from services.conformal.evaluate import evaluate_aci_controlled

N_SEEDS = 500
SEED_START = 7000
SEEDS = list(range(SEED_START, SEED_START + N_SEEDS))
SHIFT_AT = 500
T = 2000
B_BOOTSTRAP = 2000


def bootstrap_ci(diffs: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Compute 95% percentile bootstrap CI for the mean of diffs."""
    rng = np.random.default_rng(42)
    n = len(diffs)
    boot_means = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    print("=" * 65)
    print(f"High-Powered Conformal Baseline Benchmark ({N_SEEDS} Seeds, Paired)")
    print("=" * 65)
    print(f"Running 500 seeds (shift at t={SHIFT_AT}, T={T})...\n")

    # 1. Run EWMA
    print("Evaluating EWMA across 500 seeds...")
    ewma_results = evaluate_aci_controlled(
        shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="ewma"
    )

    # 2. Run GBM
    print("Evaluating GBM across 500 seeds...")
    gbm_results = evaluate_aci_controlled(
        shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm"
    )

    # Extract metrics per seed
    # Windows:
    # - overall
    # - bin 0-9
    # - bin 10-19
    # - first 20 (0-19)
    # - bins 20-49 (overshoot)
    # - steady state (850-950)

    def extract_arrays(res_list):
        overall_c = np.array([r.overall.coverage for r in res_list])
        overall_w = np.array([r.overall.mean_width for r in res_list])

        first20_c = np.array([r.near_shift_first20.coverage for r in res_list])
        first20_w = np.array([r.near_shift_first20.mean_width for r in res_list])

        b0_c = np.array([r.near_shift_bins[0].coverage for r in res_list])
        b0_w = np.array([r.near_shift_bins[0].mean_width for r in res_list])

        b1_c = np.array([r.near_shift_bins[1].coverage for r in res_list])
        b1_w = np.array([r.near_shift_bins[1].mean_width for r in res_list])

        # overshoot bins 20-49 (bins 2, 3, 4)
        overshoot_c = np.array([
            np.mean([r.near_shift_bins[2].coverage, r.near_shift_bins[3].coverage, r.near_shift_bins[4].coverage])
            for r in res_list
        ])
        overshoot_w = np.array([
            np.mean([r.near_shift_bins[2].mean_width, r.near_shift_bins[3].mean_width, r.near_shift_bins[4].mean_width])
            for r in res_list
        ])

        ss_c = np.array([r.steady_state.coverage for r in res_list])
        ss_w = np.array([r.steady_state.mean_width for r in res_list])

        return {
            "overall_c": overall_c, "overall_w": overall_w,
            "b0_c": b0_c, "b0_w": b0_w,
            "b1_c": b1_c, "b1_w": b1_w,
            "first20_c": first20_c, "first20_w": first20_w,
            "overshoot_c": overshoot_c, "overshoot_w": overshoot_w,
            "ss_c": ss_c, "ss_w": ss_w,
        }

    e = extract_arrays(ewma_results)
    g = extract_arrays(gbm_results)

    windows = [
        ("Overall (all steps >= 50)", "overall_c", "overall_w"),
        ("Steps  0– 9 (Immediate drop)", "b0_c", "b0_w"),
        ("Steps 10–19 (Persistent dip)", "b1_c", "b1_w"),
        ("Steps  0–19 (Combined Primary Crater)", "first20_c", "first20_w"),
        ("Steps 20–49 (Overshoot window)", "overshoot_c", "overshoot_w"),
        ("Steps 850–950 (Steady-state adapted)", "ss_c", "ss_w"),
    ]

    print("\n" + "=" * 90)
    print(f"{'Window':<35} | {'EWMA (Coverage ± SE)':<22} | {'GBM (Coverage ± SE)':<22} | {'Paired Δ (95% CI)':<25}")
    print("-" * 90)

    for label, c_key, w_key in windows:
        e_cov = e[c_key]
        g_cov = g[c_key]

        e_mean, e_se = np.mean(e_cov), np.std(e_cov) / np.sqrt(N_SEEDS)
        g_mean, g_se = np.mean(g_cov), np.std(g_cov) / np.sqrt(N_SEEDS)

        diff = g_cov - e_cov
        diff_mean = np.mean(diff)
        diff_ci = bootstrap_ci(diff)

        print(
            f"{label:<35} | "
            f"{e_mean:.4f} ± {e_se:.4f}        | "
            f"{g_mean:.4f} ± {g_se:.4f}        | "
            f"{diff_mean:+.4f} [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]"
        )

    print("-" * 90)
    print("\n" + "=" * 90)
    print(f"{'Window':<35} | {'EWMA Width':<15} | {'GBM Width':<15} | {'Width Δ (GBM − EWMA)':<25}")
    print("-" * 90)
    for label, c_key, w_key in windows:
        e_w = np.mean(e[w_key])
        g_w = np.mean(g[w_key])
        w_diff = g_w - e_w
        pct_change = (w_diff / e_w) * 100.0
        print(f"{label:<35} | {e_w:.2f}           | {g_w:.2f}           | {w_diff:+.2f} ({pct_change:+.1f}%)")
    print("=" * 90)


if __name__ == "__main__":
    main()
