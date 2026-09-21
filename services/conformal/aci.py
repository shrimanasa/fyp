"""
Adaptive Conformal Inference (ACI) — Gibbs & Candès (2021).

The core idea: at each step t the coverage level alpha_t is updated based on
whether the previous prediction interval covered the true value. If we missed
(miscoverage = 1), alpha decreases (intervals widen); if we covered
(miscoverage = 0), alpha increases (intervals narrow). The gamma parameter
controls the adaptation speed.

Reference: Gibbs & Candès, "Adaptive Conformal Inference Under Distribution
Shift," NeurIPS 2021.

Update rule (equation (3) in the paper):
    alpha_{t+1} = alpha_t + gamma * (alpha_target - 1{y_t not in C_t})

where C_t is the prediction interval at step t, alpha_target is the desired
miscoverage rate (e.g., 0.10 for 90% coverage), and 1{...} is the
miscoverage indicator.

Nonconformity score: we use |y - y_hat| (absolute residual). The quantile
is estimated from a sliding calibration window of recent scores.
"""
from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ACIState:
    """Mutable state carried across timesteps."""
    alpha_t: float
    cal_scores: deque = field(default_factory=deque)


class ACIPredictor:
    """
    Streaming adaptive conformal predictor.

    Parameters
    ----------
    alpha_target : float
        Desired miscoverage rate (e.g., 0.10 for 90% coverage).
    gamma : float
        Step-size for the alpha update. Larger = faster adaptation,
        noisier intervals. Gibbs & Candès suggest values in [0.005, 0.05].
    cal_window : int
        Number of recent nonconformity scores kept for quantile estimation.
        Older scores are discarded to allow distribution shift adaptation.
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
        self._state = ACIState(
            alpha_t=alpha_target,
            cal_scores=deque(maxlen=cal_window),
        )

    def update(self, y_true: float, y_hat: float) -> None:
        """
        Observe outcome y_true and point prediction y_hat. Update alpha and
        calibration scores. Call this AFTER calling predict() for step t.

        Order matters (Gibbs & Candès eq. 3):
          1. Compute the miscoverage indicator from the CURRENT interval
             (i.e. the same interval predict() returned this step, before
             the new score is added).
          2. Append the new nonconformity score to the calibration set.
          3. Update alpha_t.

        The previous version appended the score first and then recomputed
        the interval — that interval already included the current point's
        own score, making the missed indicator self-inconsistent with the
        coverage recorded by evaluate.py.
        """
        # Step 1: miscoverage check against the PRE-UPDATE interval
        interval = self._predict_interval(y_hat)
        missed = float(y_true < interval[0] or y_true > interval[1])

        # Step 2: add the new score to the calibration window
        score = abs(y_true - y_hat)
        self._state.cal_scores.append(score)

        # Step 3: alpha update (Gibbs & Candès eq. 3)
        self._state.alpha_t = np.clip(
            self._state.alpha_t + self.gamma * (self.alpha_target - missed),
            1e-6,
            1 - 1e-6,
        )

    def predict(self, y_hat: float) -> tuple[float, float]:
        """
        Return a conformal prediction interval for the next observation,
        given the point prediction y_hat.
        """
        return self._predict_interval(y_hat)

    def _predict_interval(self, y_hat: float) -> tuple[float, float]:
        scores = np.array(self._state.cal_scores)
        if len(scores) == 0:
            # cold start: return a wide interval
            return (-np.inf, np.inf)
        # (1 - alpha_t) quantile of calibration scores
        q = np.quantile(scores, 1 - self._state.alpha_t)
        return (y_hat - q, y_hat + q)

    @property
    def alpha_t(self) -> float:
        return self._state.alpha_t
