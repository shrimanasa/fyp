"""
ACI evaluation harness.

Runs ACI over a labelled stream and computes empirical coverage and interval
width broken down by distance from change points. Supports pluggable point
forecasters (EWMA, GBM, or custom callable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

from services.conformal.aci import ACIPredictor
from services.data.generator import StreamSample, ShiftSchedule
from services.seeds import CONTROLLED_EVAL_SEEDS


# EWMA-based point predictor (Phase 1 baseline)
def _ewma_predict(stream: np.ndarray, alpha_ewma: float = 0.3) -> np.ndarray:
    """
    Returns one-step-ahead EWMA forecasts for a 1-D stream.
    Prediction at t is the EWMA of stream[:t].
    """
    T = len(stream)
    preds = np.zeros(T, dtype=np.float64)
    preds[0] = stream[0]
    for t in range(1, T):
        preds[t] = alpha_ewma * stream[t - 1] + (1 - alpha_ewma) * preds[t - 1]
    return preds


def get_point_forecasts(stream: np.ndarray, forecaster: str | Callable) -> np.ndarray:
    """
    Produce one-step-ahead forecasts y_hat_0..y_hat_{T-1} given a forecaster
    identifier ('ewma', 'gbm') or a custom callable.
    """
    if callable(forecaster):
        return forecaster(stream)
    elif forecaster == "ewma":
        return _ewma_predict(stream)
    elif forecaster == "gbm":
        from services.forecaster.gbm import GBMForecaster
        return GBMForecaster().predict_stream(stream)
    else:
        raise ValueError(f"Unknown forecaster '{forecaster}'. Expected 'ewma', 'gbm', or callable.")


@dataclass
class CoverageResult:
    overall: float
    near_shift_0_100: float         # 0–100 steps after a shift onset
    steady_state_post_shift: float  # 850–950 steps after a shift (fully-adapted)
    mean_width: float
    n_total: int
    n_near_shift: int
    n_steady_state: int


def evaluate_aci(
    sample: StreamSample,
    schedule: ShiftSchedule,
    metric: str = "latency",
    forecaster: str | Callable = "ewma",
    alpha_target: float = 0.10,
    gamma: float = 0.005,
    cal_window: int = 200,
    warm_up: int = 50,
) -> CoverageResult:
    """
    Run ACI on a single metric stream and return coverage broken down by
    proximity to shift points.
    """
    stream = sample.latency if metric == "latency" else sample.error_rate
    shifts = schedule.latency_shifts if metric == "latency" else schedule.error_rate_shifts
    T = len(stream)

    predictor = ACIPredictor(alpha_target=alpha_target, gamma=gamma, cal_window=cal_window)
    y_hat_series = get_point_forecasts(stream, forecaster)

    covered = np.zeros(T, dtype=bool)
    widths = np.zeros(T, dtype=np.float64)
    for t in range(T):
        lo, hi = predictor.predict(y_hat_series[t])
        covered[t] = (lo <= stream[t] <= hi)
        widths[t] = hi - lo
        predictor.update(stream[t], y_hat_series[t])

    # --- distance-from-shift arrays ---
    dist_to_shift = _distance_to_nearest_shift(T, shifts)

    # Overall coverage (after warm-up)
    mask_all = np.arange(T) >= warm_up
    overall = covered[mask_all].mean()
    mean_w = widths[mask_all].mean() if mask_all.any() else float("nan")

    # Near-shift: within 100 steps of onset
    mask_near = mask_all & (np.abs(dist_to_shift) <= 100)
    near_cov = covered[mask_near].mean() if mask_near.any() else float("nan")

    # Steady-state post-shift (850–950 steps past shift)
    mask_ss = mask_all & (dist_to_shift >= 850) & (dist_to_shift <= 950)
    ss_cov = covered[mask_ss].mean() if mask_ss.any() else float("nan")

    return CoverageResult(
        overall=float(overall),
        near_shift_0_100=float(near_cov),
        steady_state_post_shift=float(ss_cov),
        mean_width=float(mean_w),
        n_total=int(mask_all.sum()),
        n_near_shift=int(mask_near.sum()),
        n_steady_state=int(mask_ss.sum()),
    )


def _distance_to_nearest_shift(T: int, shifts: np.ndarray) -> np.ndarray:
    if len(shifts) == 0:
        return np.full(T, np.inf)
    t_idx = np.arange(T)
    dists = np.stack([t_idx - s for s in shifts], axis=0)  # (n_shifts, T)
    abs_dists = np.abs(dists)
    nearest_idx = abs_dists.argmin(axis=0)
    return dists[nearest_idx, np.arange(T)]


# ---------------------------------------------------------------------------
# Controlled single-shift coverage evaluation
# ---------------------------------------------------------------------------

@dataclass
class BinnedWindow:
    """Coverage and width in a single distance-from-shift bin."""
    bin_start: int    # steps post-onset (inclusive)
    bin_end: int      # steps post-onset (exclusive)
    coverage: float
    mean_width: float
    n_samples: int

    def __str__(self) -> str:
        if self.n_samples == 0:
            return f"steps {self.bin_start:>3}–{self.bin_end-1:>3}: n/a (0 samples)"
        return (f"steps {self.bin_start:>3}–{self.bin_end-1:>3}: "
                f"{self.coverage:.4f} (w={self.mean_width:.2f}, n={self.n_samples})")


@dataclass
class WindowCoverage:
    """Coverage and interval width for one window, carrying sample count."""
    coverage: float
    mean_width: float
    n_samples: int

    def __str__(self) -> str:
        if self.n_samples == 0:
            return "n/a (0 samples)"
        return f"{self.coverage:.4f} (w={self.mean_width:.2f}, n={self.n_samples})"


@dataclass
class ControlledCoverageResult:
    """
    Coverage result from evaluate_aci_controlled().
    """
    seed: int
    shift_at: int
    overall: WindowCoverage
    near_shift: WindowCoverage           # [shift_at, shift_at + 100) — flat avg
    near_shift_first20: WindowCoverage   # [shift_at, shift_at + 20)  — PRIMARY DIP
    near_shift_bins: list[BinnedWindow]  # 10-step bins over [0, 100)
    steady_state: WindowCoverage         # [shift_at + 850, shift_at + 950)


def evaluate_aci_controlled(
    shift_at: int = 500,
    T: int = 2000,
    seeds: Sequence[int] | None = None,
    n_seeds: int | None = None,
    forecaster: str | Callable = "ewma",
    ar_phi: float = 0.8,
    latency_base: float = 50.0,
    latency_sigma: float = 5.0,
    latency_jump_scale: float = 4.0,
    shift_type: str = "mean",
    variance_scale: float = 3.0,
    alpha_target: float = 0.10,
    gamma: float = 0.005,
    cal_window: int = 200,
    warm_up: int = 50,
    bin_width: int = 10,
    near_window: int = 100,
    predictor_factory: Callable[[], ACIPredictor] | None = None,
    alarm_schedule_fn: Callable[[int, int], np.ndarray] | None = None,
) -> list[ControlledCoverageResult]:
    """
    Controlled single-shift coverage evaluation.
    Guarantees all windows are populated with fixed sample counts.
    Supports mean shifts and variance shifts, with pluggable predictor factories
    and alarm schedules.
    """
    assert shift_at + 950 < T, f"shift_at={shift_at} + 950 >= T={T}"
    assert shift_at > warm_up + 100, f"shift_at={shift_at} too close to warm_up={warm_up}"

    if seeds is not None:
        eval_seeds = list(seeds)
    elif n_seeds is not None:
        eval_seeds = list(CONTROLLED_EVAL_SEEDS)[:n_seeds]
    else:
        eval_seeds = list(CONTROLLED_EVAL_SEEDS)

    results = []
    for seed in eval_seeds:
        rng = np.random.default_rng(seed)

        # Generate single-shift AR(1) latency stream
        stream = np.zeros(T)
        stream[0] = latency_base
        mean_t = latency_base

        if shift_type == "mean":
            for t in range(1, T):
                if t == shift_at:
                    direction = rng.choice([-1, 1])
                    mean_t += direction * latency_jump_scale * latency_sigma
                stream[t] = ar_phi * stream[t - 1] + (1 - ar_phi) * mean_t + rng.normal(0, latency_sigma)
        elif shift_type == "variance":
            for t in range(1, T):
                sigma_t = latency_sigma * variance_scale if t >= shift_at else latency_sigma
                stream[t] = ar_phi * stream[t - 1] + (1 - ar_phi) * mean_t + rng.normal(0, sigma_t)
        else:
            raise ValueError(f"Unknown shift_type '{shift_type}'. Expected 'mean' or 'variance'.")

        # Point forecast
        y_hat = get_point_forecasts(stream, forecaster)

        # Initialize predictor
        if predictor_factory is not None:
            predictor = predictor_factory()
        else:
            predictor = ACIPredictor(alpha_target=alpha_target, gamma=gamma, cal_window=cal_window)

        alarm_flags = alarm_schedule_fn(T, shift_at) if alarm_schedule_fn is not None else None

        covered = np.zeros(T, dtype=bool)
        widths = np.zeros(T, dtype=np.float64)
        for t in range(T):
            if alarm_flags is not None and alarm_flags[t]:
                if hasattr(predictor, "trigger"):
                    predictor.trigger()
            elif hasattr(predictor, "step_tick"):
                predictor.step_tick()

            lo, hi = predictor.predict(y_hat[t])
            covered[t] = (lo <= stream[t] <= hi)
            widths[t] = hi - lo
            predictor.update(stream[t], y_hat[t])

        # Windows
        t_idx = np.arange(T)
        dist = t_idx - shift_at
        mask_warm = t_idx >= warm_up

        def _make_window_cov(mask: np.ndarray) -> WindowCoverage:
            n = int(mask.sum())
            c = float(covered[mask].mean()) if n > 0 else float("nan")
            w = float(widths[mask].mean()) if n > 0 else float("nan")
            return WindowCoverage(coverage=c, mean_width=w, n_samples=n)

        # overall
        overall_cov = _make_window_cov(mask_warm)

        # near-shift: [0, near_window)
        mask_near = mask_warm & (dist >= 0) & (dist < near_window)
        near_cov = _make_window_cov(mask_near)

        # near-shift first 20 steps: [0, 20)
        mask_first20 = mask_warm & (dist >= 0) & (dist < 20)
        first20_cov = _make_window_cov(mask_first20)

        # near-shift 10-step bins
        bins = []
        for b_start in range(0, near_window, bin_width):
            b_end = b_start + bin_width
            mask_b = mask_warm & (dist >= b_start) & (dist < b_end)
            n_b = int(mask_b.sum())
            c_b = float(covered[mask_b].mean()) if n_b > 0 else float("nan")
            w_b = float(widths[mask_b].mean()) if n_b > 0 else float("nan")
            bins.append(BinnedWindow(
                bin_start=b_start,
                bin_end=b_end,
                coverage=c_b,
                mean_width=w_b,
                n_samples=n_b,
            ))

        # steady-state: [850, 950)
        mask_ss = mask_warm & (dist >= 850) & (dist < 950)
        ss_cov = _make_window_cov(mask_ss)

        results.append(ControlledCoverageResult(
            seed=seed,
            shift_at=shift_at,
            overall=overall_cov,
            near_shift=near_cov,
            near_shift_first20=first20_cov,
            near_shift_bins=bins,
            steady_state=ss_cov,
        ))

    return results


def summarise_controlled(
    results: list[ControlledCoverageResult],
    min_n: int = 50,
) -> dict:
    """
    Aggregate controlled results into summary statistics (mean, std, width).
    """
    def _agg(cov_list: list[float], w_list: list[float], ns: list[int], threshold: int | None = None) -> dict:
        t = threshold if threshold is not None else min_n
        eligible = [(c, w, n) for c, w, n in zip(cov_list, w_list, ns) if n >= t and not np.isnan(c)]
        excluded = len(cov_list) - len(eligible)
        if not eligible:
            return {"mean": float("nan"), "std": float("nan"), "mean_width": float("nan"),
                    "n_seeds_used": 0, "n_seeds_excluded": excluded, "min_window_n": 0}
        cs = [c for c, _, _ in eligible]
        ws = [w for _, w, _ in eligible]
        ns_e = [n for _, _, n in eligible]
        return {
            "mean": float(np.mean(cs)),
            "std": float(np.std(cs)),
            "mean_width": float(np.mean(ws)),
            "n_seeds_used": len(eligible),
            "n_seeds_excluded": excluded,
            "min_window_n": min(ns_e),
        }

    overall_c = [r.overall.coverage for r in results]
    overall_w = [r.overall.mean_width for r in results]
    overall_n = [r.overall.n_samples for r in results]

    near_c = [r.near_shift.coverage for r in results]
    near_w = [r.near_shift.mean_width for r in results]
    near_n = [r.near_shift.n_samples for r in results]

    first20_c = [r.near_shift_first20.coverage for r in results]
    first20_w = [r.near_shift_first20.mean_width for r in results]
    first20_n = [r.near_shift_first20.n_samples for r in results]

    ss_c = [r.steady_state.coverage for r in results]
    ss_w = [r.steady_state.mean_width for r in results]
    ss_n = [r.steady_state.n_samples for r in results]

    # aggregate bins
    binned_summary = []
    if results and results[0].near_shift_bins:
        n_bins = len(results[0].near_shift_bins)
        for i in range(n_bins):
            b_start = results[0].near_shift_bins[i].bin_start
            b_end = results[0].near_shift_bins[i].bin_end
            bin_c = [r.near_shift_bins[i].coverage for r in results]
            bin_w = [r.near_shift_bins[i].mean_width for r in results]
            bin_n = [r.near_shift_bins[i].n_samples for r in results]
            agg_bin = _agg(bin_c, bin_w, bin_n, threshold=5)
            binned_summary.append({
                "bin_start": b_start,
                "bin_end": b_end,
                **agg_bin
            })

    return {
        "overall": _agg(overall_c, overall_w, overall_n),
        "near_shift": _agg(near_c, near_w, near_n),
        "near_shift_first20": _agg(first20_c, first20_w, first20_n, threshold=10),
        "steady_state": _agg(ss_c, ss_w, ss_n),
        "bins": binned_summary,
        "n_seeds_total": len(results),
        "shift_at": results[0].shift_at if results else None,
    }
