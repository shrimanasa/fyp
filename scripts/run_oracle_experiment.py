"""
run_oracle_experiment.py — Rigorous Oracle Sensitivity & Ceiling Benchmark.

Addresses all peer-review critique requirements:
1. Empty-window artifact fix: Window flush maintains flush_min_window=20 valid calibration scores.
2. Reports In-Window Width (steps 0–19) and Winkler Interval Score alongside overall width.
3. Tests conservation of coverage: compares update_on_served vs. update_on_raw.
4. Outputs full 10-step binned curves across steps 0–100 to inspect the post-horizon deficit payback.
5. Delay-sensitivity sweep: evaluates d in {0, 1, 2, 4, 8} with realistic 5% stationary false-alarm rates.
6. Computes 95% paired bootstrap confidence intervals (B=2000) for all metric differences.

Evaluated across 100 dedicated validation seeds (services.seeds.VAL_SEEDS = range(500, 600)).
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
B_BOOTSTRAP = 2000


def bootstrap_ci(diffs: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Compute 95% percentile bootstrap CI for the mean of paired differences."""
    rng = np.random.default_rng(42)
    n = len(diffs)
    boot_means = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def make_alarm_schedule(delay: int | None, horizon: int = 20, fpr_rate: float = 0.0, seed: int = 42):
    """
    Constructs an alarm schedule:
    - If delay is not None: fires at t = shift_step + delay for horizon steps.
    - If fpr_rate > 0: injects stationary alarms as a renewal process matching target 5% FPR.
    """
    def _schedule(stream_len: int, shift_step: int) -> np.ndarray:
        alarms = np.zeros(stream_len, dtype=bool)
        
        # Shift trigger
        if delay is not None:
            s_start = shift_step + delay
            s_end = min(stream_len, s_start + horizon)
            alarms[s_start:s_end] = True

        # Realistic stationary false alarms (outside shift protection zone)
        if fpr_rate > 0:
            rng = np.random.default_rng(seed + shift_step)
            p_init = fpr_rate / horizon
            t = 50
            while t < stream_len:
                # Avoid contaminating the true shift window [shift_step - 10, shift_step + 100)
                if shift_step - 10 <= t < shift_step + 100:
                    t = shift_step + 100
                    continue
                if rng.uniform() < p_init:
                    end_t = min(stream_len, min(t + horizon, shift_step - 10 if t < shift_step else stream_len))
                    alarms[t:end_t] = True
                    t = end_t
                else:
                    t += 1

        return alarms
    return _schedule


def run_condition_analysis(shift_type: str):
    print("\n" + "=" * 115)
    print(f"RIGOROUS ORACLE BENCHMARK: {shift_type.upper()} SHIFTS (100 Validation Seeds {SEEDS[0]}..{SEEDS[-1]})")
    print("=" * 115)

    # 1. Baseline Standard ACI (Gate OFF)
    print("Evaluating Baseline Standard ACI (Gate OFF)...")
    base_res = evaluate_aci_controlled(
        shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm", shift_type=shift_type
    )

    base_cov = np.array([r.near_shift_first20.coverage for r in base_res])
    base_b0 = np.array([r.near_shift_bins[0].coverage for r in base_res])
    base_w20 = np.array([r.near_shift_first20.mean_width for r in base_res])
    base_wink20 = np.array([r.near_shift_first20.winkler_score for r in base_res])
    base_w_all = np.array([r.overall.mean_width for r in base_res])
    base_overall = np.array([r.overall.coverage for r in base_res])

    print(f"Baseline Standard ACI: Overall Cov={np.mean(base_overall):.4f}, Steps 0–9 Cov={np.mean(base_b0):.4f}, "
          f"Steps 0–19 Cov={np.mean(base_cov):.4f}, In-Window Width={np.mean(base_w20):.2f}, "
          f"In-Window Winkler={np.mean(base_wink20):.2f}, Overall Width={np.mean(base_w_all):.2f}\n")

    # Candidate Policies (at delay d=1, matching the first observable causal step)
    d1_sched = make_alarm_schedule(delay=1, horizon=20, fpr_rate=0.0)

    policies = [
        ("Policy A: gamma-boost (gamma=0.05)",
         lambda: GatedACIPredictor(policy="gamma_boost", gamma_boost=0.05, horizon=20, max_q=100.0)),
        ("Policy B: valid flush (retain min 20)",
         lambda: GatedACIPredictor(policy="window_flush", horizon=20, flush_min_window=20, max_q=100.0)),
        ("Policy C (served update): buffer kappa=0.50",
         lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=0.50, horizon=20, update_on_raw=False, max_q=100.0)),
        ("Policy C (raw update): buffer kappa=0.50",
         lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=0.50, horizon=20, update_on_raw=True, max_q=100.0)),
        ("Policy C (raw update): buffer kappa=2.00 (true 3x ratio)",
         lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=2.00, horizon=20, update_on_raw=True, max_q=100.0)),
        ("Policy D: alpha-shift (alpha=0.02)",
         lambda: GatedACIPredictor(policy="alpha_shift", alpha_boost=0.02, horizon=20, max_q=100.0)),
    ]

    print("-" * 115)
    print(f"{'Method / Policy':<42} | {'Overall':<7} | {'0–19 Cov (Δ vs Base [95% CI])':<32} | {'Width 0–19':<10} | {'Winkler 0–19':<12}")
    print("-" * 115)
    print(f"{'Standard ACI (W=200, Gate OFF)':<42} | {np.mean(base_overall):.4f}  | {np.mean(base_cov):.4f} (reference)                | {np.mean(base_w20):.2f}       | {np.mean(base_wink20):.2f}")

    eval_records = []

    for name, factory in policies:
        res = evaluate_aci_controlled(
            shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm",
            shift_type=shift_type, predictor_factory=factory, alarm_schedule_fn=d1_sched
        )
        cov = np.array([r.near_shift_first20.coverage for r in res])
        w20 = np.array([r.near_shift_first20.mean_width for r in res])
        wink20 = np.array([r.near_shift_first20.winkler_score for r in res])
        ov = np.array([r.overall.coverage for r in res])
        ov_w = np.array([r.overall.mean_width for r in res])

        d_cov = cov - base_cov
        ci_cov = bootstrap_ci(d_cov)

        print(f"{name:<42} | {np.mean(ov):.4f}  | {np.mean(cov):.4f} ({np.mean(d_cov):+.4f} [{ci_cov[0]:+.4f}, {ci_cov[1]:+.4f}]) | {np.mean(w20):.2f}       | {np.mean(wink20):.2f}")
        eval_records.append((name, res, cov, w20, wink20, ov, ov_w))

    # Gate-Free Short-Window ACI Baselines (Is Policy B just short-window ACI?)
    print("-" * 115)
    print("GATE-FREE PARAMETER BASELINES (Varying Calibration Window W):")
    print("-" * 115)
    short_w_baselines = [
        ("Gate-Free ACI (W=20, no gate)", lambda: ACIPredictor(cal_window=20, max_q=100.0)),
        ("Gate-Free ACI (W=30, no gate)", lambda: ACIPredictor(cal_window=30, max_q=100.0)),
        ("Gate-Free ACI (W=50, no gate)", lambda: ACIPredictor(cal_window=50, max_q=100.0)),
        ("Gate-Free ACI (W=100, no gate)", lambda: ACIPredictor(cal_window=100, max_q=100.0)),
    ]
    for name, factory in short_w_baselines:
        res = evaluate_aci_controlled(
            shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm",
            shift_type=shift_type, predictor_factory=factory
        )
        cov = np.array([r.near_shift_first20.coverage for r in res])
        w20 = np.array([r.near_shift_first20.mean_width for r in res])
        wink20 = np.array([r.near_shift_first20.winkler_score for r in res])
        ov = np.array([r.overall.coverage for r in res])
        d_cov = cov - base_cov
        ci_cov = bootstrap_ci(d_cov)
        print(f"{name:<42} | {np.mean(ov):.4f}  | {np.mean(cov):.4f} ({np.mean(d_cov):+.4f} [{ci_cov[0]:+.4f}, {ci_cov[1]:+.4f}]) | {np.mean(w20):.2f}       | {np.mean(wink20):.2f}")

    # Controls for the primary policy: Policy C (raw update, kappa=0.50)
    print("-" * 115)
    print("SCIENTIFIC CONTROLS & BENCHMARKS:")
    print("-" * 115)

    # 1. Random Trigger Control matching 5% stationary alarm rate
    rand_sched = make_alarm_schedule(delay=None, horizon=20, fpr_rate=0.05)
    rand_res = evaluate_aci_controlled(
        shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm", shift_type=shift_type,
        predictor_factory=lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=0.50, horizon=20, update_on_raw=True, max_q=100.0),
        alarm_schedule_fn=rand_sched
    )
    rand_cov = np.array([r.near_shift_first20.coverage for r in rand_res])
    rand_w20 = np.array([r.near_shift_first20.mean_width for r in rand_res])
    rand_wink20 = np.array([r.near_shift_first20.winkler_score for r in rand_res])
    rand_ov = np.array([r.overall.coverage for r in rand_res])
    rand_ov_w = np.array([r.overall.mean_width for r in rand_res])
    d_rand = rand_cov - base_cov
    ci_rand = bootstrap_ci(d_rand)
    print(f"{'Random 5% Stationary Trigger Control':<38} | {np.mean(rand_ov):.4f}  | {np.mean(rand_cov):.4f} ({np.mean(d_rand):+.4f} [{ci_rand[0]:+.4f}, {ci_rand[1]:+.4f}]) | {np.mean(rand_w20):.2f}       | {np.mean(rand_wink20):.2f}")

    # 2. Always-On Width-Matched Control (matching Policy C raw overall width)
    pol_c_rec = [r for r in eval_records if "raw update" in r[0]][0]
    target_overall_w = np.mean(pol_c_rec[6])
    base_mean_w = np.mean(base_w_all)
    kappa_matched = (target_overall_w - base_mean_w) / base_mean_w
    matched_res = evaluate_aci_controlled(
        shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm", shift_type=shift_type,
        predictor_factory=lambda: ACIPredictor(uniform_margin_kappa=kappa_matched, max_q=100.0)
    )
    m_cov = np.array([r.near_shift_first20.coverage for r in matched_res])
    m_w20 = np.array([r.near_shift_first20.mean_width for r in matched_res])
    m_wink20 = np.array([r.near_shift_first20.winkler_score for r in matched_res])
    m_ov = np.array([r.overall.coverage for r in matched_res])
    d_m = m_cov - base_cov
    ci_m = bootstrap_ci(d_m)
    print(f"{f'Always-On Width-Matched (kappa={kappa_matched:.4f})':<38} | {np.mean(m_ov):.4f}  | {np.mean(m_cov):.4f} ({np.mean(d_m):+.4f} [{ci_m[0]:+.4f}, {ci_m[1]:+.4f}]) | {np.mean(m_w20):.2f}       | {np.mean(m_wink20):.2f}")

    # Paired Selective Net Gain (Policy C vs Matched Control)
    sel_diff = pol_c_rec[2] - m_cov
    ci_sel = bootstrap_ci(sel_diff)
    print(f"\n--> SELECTIVE NET GAIN (Policy C vs Width-Matched): {np.mean(sel_diff):+.4f} [95% CI: {ci_sel[0]:+.4f}, {ci_sel[1]:+.4f}]")

    # Full 10-Step Binned Curves across steps 0–100 (Inspecting Deficit Payback)
    print("\n" + "-" * 115)
    print("DISTANCE-FROM-SHIFT BINNED COVERAGE CURVE OVER [0, 100) STEPS POST-ONSET:")
    print("-" * 115)
    print(f"{'Bin':<12} | {'Standard ACI':<12} | {'Pol C (Served Upd)':<18} | {'Pol C (Raw Upd)':<16} | {'Pol B (Valid Flush)':<18}")
    print("-" * 85)

    base_bins = [np.mean([r.near_shift_bins[b].coverage for r in base_res]) for b in range(10)]
    pol_c_serv_res = [r for r in eval_records if "served update" in r[0]][0][1]
    pol_c_serv_bins = [np.mean([r.near_shift_bins[b].coverage for r in pol_c_serv_res]) for b in range(10)]
    pol_c_raw_res = pol_c_rec[1]
    pol_c_raw_bins = [np.mean([r.near_shift_bins[b].coverage for r in pol_c_raw_res]) for b in range(10)]
    pol_b_res = [r for r in eval_records if "valid flush" in r[0]][0][1]
    pol_b_bins = [np.mean([r.near_shift_bins[b].coverage for r in pol_b_res]) for b in range(10)]

    for b in range(10):
        b_label = f"steps {b*10:>2}–{b*10+9:>2}"
        print(f"{b_label:<12} | {base_bins[b]:.4f}       | {pol_c_serv_bins[b]:.4f}             | {pol_c_raw_bins[b]:.4f}           | {pol_b_bins[b]:.4f}")

    # Delay Sensitivity Sweep: d in {0, 1, 2, 4, 8} with 5% Stationary False Alarms
    print("\n" + "-" * 115)
    print(f"DELAY SENSITIVITY SWEEP (Policy C raw update, kappa=0.50 + 5% Stationary False Alarms):")
    print("-" * 115)
    print(f"{'Delay d':<10} | {'Steps 0–9 Cov':<14} | {'Steps 0–19 Cov (Δ vs Base [95% CI])':<36} | {'Width 0–19':<12} | {'Winkler 0–19':<12}")
    print("-" * 90)

    for d in [0, 1, 2, 4, 8]:
        d_sched = make_alarm_schedule(delay=d, horizon=20, fpr_rate=0.05)
        res_d = evaluate_aci_controlled(
            shift_at=SHIFT_AT, T=T, seeds=SEEDS, forecaster="gbm", shift_type=shift_type,
            predictor_factory=lambda: GatedACIPredictor(policy="margin_buffer", margin_kappa=0.50, horizon=20, update_on_raw=True, max_q=100.0),
            alarm_schedule_fn=d_sched
        )
        cov_d = np.array([r.near_shift_first20.coverage for r in res_d])
        b0_d = np.array([r.near_shift_bins[0].coverage for r in res_d])
        w20_d = np.array([r.near_shift_first20.mean_width for r in res_d])
        wink20_d = np.array([r.near_shift_first20.winkler_score for r in res_d])

        diff_d = cov_d - base_cov
        ci_d = bootstrap_ci(diff_d)

        print(f"d = {d:<7} | {np.mean(b0_d):.4f}         | {np.mean(cov_d):.4f} ({np.mean(diff_d):+.4f} [{ci_d[0]:+.4f}, {ci_d[1]:+.4f}])     | {np.mean(w20_d):.2f}         | {np.mean(wink20_d):.2f}")


def main():
    print("===========================================================================================")
    print("RIGOROUS ORACLE CEILING, DEFICIT PAYBACK, AND DELAY SENSITIVITY BENCHMARK")
    print("===========================================================================================")
    run_condition_analysis("mean")
    run_condition_analysis("variance")


if __name__ == "__main__":
    main()
