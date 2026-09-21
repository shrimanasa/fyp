"""
Synthetic infra-metrics stream generator.

Produces two metrics — `latency` and `error_rate` — with independently
scheduled distribution shifts. A shift in `latency` is NOT a change point
for `error_rate` and vice versa. This metric-aware labeling is critical for
the gate classifier: the gate trained on latency features should only be
labeled positive near latency shift times.

Stream model
------------
latency:   AR(1) with phi=0.8, baseline mean=50ms, noise N(0, sigma^2).
           At a latency shift: mean jumps by +/- jump_scale * sigma.
error_rate: Poisson(lambda_t). lambda_t is piecewise-constant; at an
            error_rate shift it multiplies by a random factor in [2, 5].

Shift schedule
--------------
Each run gets an independent random set of shift times per metric.
Shift times are drawn without replacement from [min_gap, T - min_gap],
with a minimum gap between shifts enforced.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass
class ShiftSchedule:
    """Independently drawn shift times per metric."""
    latency_shifts: np.ndarray   # sorted indices in [0, T)
    error_rate_shifts: np.ndarray

    @classmethod
    def random(
        cls,
        T: int,
        n_shifts: int = 3,
        min_gap: int = 150,
        rng: np.random.Generator | None = None,
    ) -> "ShiftSchedule":
        rng = rng or np.random.default_rng()
        def _draw(n: int) -> np.ndarray:
            candidates = np.arange(min_gap, T - min_gap)
            chosen: list[int] = []
            for _ in range(n):
                if len(candidates) == 0:
                    break
                t = rng.choice(candidates)
                chosen.append(int(t))
                # remove within min_gap of the chosen point
                candidates = candidates[np.abs(candidates - t) > min_gap]
            return np.sort(chosen)

        return cls(
            latency_shifts=_draw(n_shifts),
            error_rate_shifts=_draw(n_shifts),
        )


class StreamSample(NamedTuple):
    latency: np.ndarray          # shape (T,)
    error_rate: np.ndarray       # shape (T,)
    latency_label: np.ndarray    # 1 if within lead_steps BEFORE a latency shift
    error_rate_label: np.ndarray


def _shift_labels(T: int, shift_times: np.ndarray, lead_steps: int = 50) -> np.ndarray:
    """
    Returns a binary label array of length T.
    label[t] = 1 iff t is in [shift_time, shift_time + lead_steps).

    Labeling design note: we label the ONSET window [s, s+lead_steps) as
    positive, not the pre-shift window [s-lead_steps, s).

    Reason: with a step-function shift, the pre-shift distribution is
    identical to any other normal window — there is no predictive signal
    before the step arrives. A classifier trained on pre-shift features
    learns nothing (AUROC ≈ 0.5, confirmed empirically).

    Labeling the onset window is still "early detection": ACI's calibration
    window is 200 steps wide, so coverage degrades slowly after onset. The
    gate's goal is to detect the shift within ~50 steps of onset so the
    interval can widen before the coverage crater accumulates — not to
    predict the shift before it starts.

    Window width (50 steps): the short window (10 steps) fills with
    new-distribution data immediately; the medium window (30 steps) fills
    by t = s+30. 50 steps covers the time for medium-window features to
    fully reflect the new regime, where the detection signal is strongest.
    Shorter (e.g. 30) truncates the window before all features are
    informative; longer (e.g. 100) dilutes positives into the post-onset
    settled regime and inflates false negatives at onset.
    """
    labels = np.zeros(T, dtype=int)
    for s in shift_times:
        start = int(s)
        end = min(T, int(s) + lead_steps)
        labels[start:end] = 1
    return labels


def generate_stream(
    T: int = 2000,
    n_shifts: int = 3,
    min_gap: int = 150,
    ar_phi: float = 0.8,
    latency_base: float = 50.0,
    latency_sigma: float = 5.0,
    latency_jump_scale: float = 4.0,
    error_rate_base_lambda: float = 2.0,
    error_rate_jump_factor_range: tuple[float, float] = (2.0, 5.0),
    lead_steps: int = 50,
    seed: int | None = None,
) -> tuple[StreamSample, ShiftSchedule]:
    """
    Generate a synthetic infra-metrics stream with labeled shift points.

    Returns (StreamSample, ShiftSchedule).
    """
    rng = np.random.default_rng(seed)
    schedule = ShiftSchedule.random(T, n_shifts=n_shifts, min_gap=min_gap, rng=rng)

    # --- latency: AR(1) with mean shifts ---
    latency = np.zeros(T)
    latency[0] = latency_base
    mean_t = latency_base
    shift_set_lat = set(schedule.latency_shifts.tolist())
    for t in range(1, T):
        if t in shift_set_lat:
            direction = rng.choice([-1, 1])
            mean_t += direction * latency_jump_scale * latency_sigma
        noise = rng.normal(0, latency_sigma)
        latency[t] = ar_phi * latency[t - 1] + (1 - ar_phi) * mean_t + noise

    # --- error_rate: piecewise Poisson ---
    error_rate = np.zeros(T)
    lam = error_rate_base_lambda
    shift_set_err = set(schedule.error_rate_shifts.tolist())
    for t in range(T):
        if t in shift_set_err:
            factor = rng.uniform(*error_rate_jump_factor_range)
            lam = lam * factor
        error_rate[t] = rng.poisson(lam)

    # --- labels ---
    lat_label = _shift_labels(T, schedule.latency_shifts, lead_steps)
    err_label = _shift_labels(T, schedule.error_rate_shifts, lead_steps)

    return StreamSample(latency, error_rate, lat_label, err_label), schedule
