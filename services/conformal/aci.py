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
    ) -> None:
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.cal_window = cal_window

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
            self._last_interval = (-np.inf, np.inf)
            return self._last_interval

        # Gibbs & Candès finite-sample conformal quantile index (1-based)
        k = int(np.ceil((n + 1) * (1.0 - self.alpha_t)))

        if k > n:
            q = np.inf
        elif k <= 0:
            q = 0.0
        else:
            q = self._sorted_scores[k - 1]

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
