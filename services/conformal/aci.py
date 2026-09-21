"""
Adaptive Conformal Inference (ACI) — Gibbs & Candès (2021).

Update rule (Gibbs & Candès eq. 3):
    alpha_{t+1} = alpha_t + gamma * (alpha_target - 1{y_t not in C_t})

Key theoretical properties:
1. Long-run coverage guarantee: alpha_t is NOT clipped to [0, 1]. Allowing alpha_t
   to float unbounded allows the pinball loss gradient to integrate out persistent
   miscoverage or overcoverage, guaranteeing asymptotic validity.
2. Finite-sample quantile: For calibration window of size n, uses the conformal
   quantile index k = ceil((n + 1) * (1 - alpha_t)).
   - If k > n: infinite prediction interval (quantile = inf, 100% coverage).
   - If k <= 0: zero-width interval (quantile = 0, 0% coverage).
   - If 1 <= k <= n: k-th smallest nonconformity score.
3. High-efficiency sorting: Maintains a sorted scores array with bisect,
   providing O(1) quantile lookup and eliminating numpy array allocations.
"""
from __future__ import annotations

import bisect
from collections import deque
import numpy as np


class ACIPredictor:
    """
    Streaming adaptive conformal predictor.

    Parameters
    ----------
    alpha_target : float
        Desired miscoverage rate (e.g., 0.10 for 90% coverage).
    gamma : float
        Step-size for the alpha update.
    cal_window : int
        Number of recent nonconformity scores kept in sliding calibration window.
    """

    def __init__(
        self,
        alpha_target: float = 0.10,
        gamma: float = 0.005,
        cal_window: int = 200,
        uniform_margin_kappa: float = 0.0,
        max_q: float | None = None,
    ) -> None:
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.cal_window = cal_window
        self.uniform_margin_kappa = uniform_margin_kappa
        self.max_q = max_q
        self.alpha_t: float = alpha_target
        self._cal_scores: deque[float] = deque()
        self._sorted_scores: list[float] = []
        self._last_interval: tuple[float, float] = (-np.inf, np.inf)

    def predict(self, y_hat: float) -> tuple[float, float]:
        """
        Return conformal prediction interval [y_hat - q, y_hat + q].
        """
        n = len(self._sorted_scores)
        if n == 0:
            q = self.max_q if self.max_q is not None else np.inf
            self._last_interval = (y_hat - q, y_hat + q)
            return self._last_interval

        # Gibbs & Candès finite-sample conformal quantile index (1-based)
        k = int(np.ceil((n + 1) * (1.0 - self.alpha_t)))

        if k > n:
            q = self.max_q if self.max_q is not None else np.inf
        elif k <= 0:
            q = 0.0
        else:
            q = self._sorted_scores[k - 1]

        if self.uniform_margin_kappa > 0.0 and np.isfinite(q):
            q = q * (1.0 + self.uniform_margin_kappa)

        self._last_interval = (y_hat - q, y_hat + q)
        return self._last_interval

    def update(self, y_true: float, y_hat: float) -> None:
        """
        Observe outcome y_true and point prediction y_hat.
        Order of operations (Gibbs & Candès eq. 3):
          1. Compute miscoverage against PRE-UPDATE interval.
          2. Append score to calibration window.
          3. Update alpha_t unclipped.
        """
        # Step 1: miscoverage check against pre-update interval
        lo, hi = self._last_interval
        missed = float(y_true < lo or y_true > hi)

        # Step 2: append score to sliding calibration window
        score = abs(y_true - y_hat)
        if len(self._cal_scores) == self.cal_window:
            old_score = self._cal_scores.popleft()
            idx = bisect.bisect_left(self._sorted_scores, old_score)
            del self._sorted_scores[idx]

        self._cal_scores.append(score)
        bisect.insort(self._sorted_scores, score)

        # Step 3: alpha update without clipping (Gibbs & Candès 2021)
        self.alpha_t += self.gamma * (self.alpha_target - missed)

    @property
    def cal_scores(self) -> deque[float]:
        return self._cal_scores


class GatedACIPredictor(ACIPredictor):
    """
    Adaptive Conformal Predictor with a Drift-Gate Reaction Engine.

    Supports four candidate reaction policies upon alarm:
    - 'none': standard ACI without reaction.
    - 'gamma_boost': temporarily scales learning rate to gamma_boost for H steps.
    - 'window_flush': flushes pre-shift calibration scores on alarm rising edge.
    - 'margin_buffer': multiplies interval half-width by (1 + margin_kappa) during alarm.
    - 'alpha_shift': temporarily uses a tighter miscoverage alpha_boost during alarm.
    """

    def __init__(
        self,
        alpha_target: float = 0.10,
        gamma: float = 0.005,
        cal_window: int = 200,
        policy: str = "none",
        gamma_boost: float = 0.05,
        horizon: int = 20,
        margin_kappa: float = 0.5,
        alpha_boost: float = 0.02,
        uniform_margin_kappa: float = 0.0,
        max_q: float | None = None,
    ) -> None:
        super().__init__(
            alpha_target=alpha_target,
            gamma=gamma,
            cal_window=cal_window,
            uniform_margin_kappa=uniform_margin_kappa,
            max_q=max_q,
        )
        self.policy = policy
        self.gamma_boost = gamma_boost
        self.horizon = horizon
        self.margin_kappa = margin_kappa
        self.alpha_boost = alpha_boost

        self.cooldown: int = 0
        self._prev_alarm: bool = False

    def trigger(self) -> None:
        """Trigger an alarm at current timestep."""
        if not self._prev_alarm and self.policy == "window_flush":
            self._cal_scores.clear()
            self._sorted_scores.clear()
        self.cooldown = self.horizon
        self._prev_alarm = True

    def step_tick(self) -> None:
        """Advance cooldown timer if no trigger occurred at current step."""
        if self.cooldown > 0:
            self.cooldown -= 1
            if self.cooldown == 0:
                self._prev_alarm = False

    def predict(self, y_hat: float) -> tuple[float, float]:
        n = len(self._sorted_scores)
        if n == 0:
            q = self.max_q if self.max_q is not None else np.inf
            self._last_interval = (y_hat - q, y_hat + q)
            return self._last_interval

        eff_alpha = (
            self.alpha_boost
            if (self.policy == "alpha_shift" and self.cooldown > 0)
            else self.alpha_t
        )
        k = int(np.ceil((n + 1) * (1.0 - eff_alpha)))

        if k > n:
            q = self.max_q if self.max_q is not None else np.inf
        elif k <= 0:
            q = 0.0
        else:
            q = self._sorted_scores[k - 1]

        # Policy C: margin inflation during alarm
        if self.policy == "margin_buffer" and self.cooldown > 0 and np.isfinite(q):
            q = q * (1.0 + self.margin_kappa)
        elif self.uniform_margin_kappa > 0.0 and np.isfinite(q):
            q = q * (1.0 + self.uniform_margin_kappa)

        self._last_interval = (y_hat - q, y_hat + q)
        return self._last_interval

    def update(self, y_true: float, y_hat: float) -> None:
        # Step 1: miscoverage check against pre-update interval
        lo, hi = self._last_interval
        missed = float(y_true < lo or y_true > hi)

        # Step 2: append score to sliding calibration window
        score = abs(y_true - y_hat)
        if len(self._cal_scores) == self.cal_window:
            old_score = self._cal_scores.popleft()
            idx = bisect.bisect_left(self._sorted_scores, old_score)
            del self._sorted_scores[idx]

        self._cal_scores.append(score)
        bisect.insort(self._sorted_scores, score)

        # Step 3: alpha update using effective gamma (Policy A: gamma boost)
        eff_gamma = (
            self.gamma_boost
            if (self.policy == "gamma_boost" and self.cooldown > 0)
            else self.gamma
        )
        self.alpha_t += eff_gamma * (self.alpha_target - missed)
