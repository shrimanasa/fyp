"""
services/gate/baselines.py — Non-learned change detectors for distribution shifts.

Provides fast, exact statistical detectors operating on forecaster residuals:
1. VolatilityRatioDetector: Ratio of short-window to long-window residual standard deviation.
   R_t = std(e_{t-w_s:t}) / (std(e_{t-w_l:t}) + eps)
2. ResidualCUSUMDetector: Page-Hinkley cumulative sum on normalized residual variance.
   S_t = max(0, S_{t-1} + (e_t^2 / sigma_0^2 - 1.0) - k)

Used as foundational non-learned baselines for variance and volatility shift detection.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DetectorResult:
    alarm_stream: np.ndarray       # boolean array of shape (T,)
    triggers: np.ndarray           # boolean array of alarm rising edges (step-level trigger events)
    stat_episode_rate: float       # trigger episodes per 1,000 stationary steps
    stat_duty_cycle: float         # fraction of stationary steps in alarm state
    delays: list[int]              # detection delays for detected shifts
    event_recall: float            # fraction of shifts detected within onset window


class VolatilityRatioDetector:
    """
    Rolling residual standard deviation ratio detector.
    Triggers an alarm when short-window volatility exceeds long-window baseline.
    """

    def __init__(
        self,
        w_short: int = 5,
        w_long: int = 50,
        threshold: float = 2.0,
        horizon: int = 20,
        refractory: int = 20,
    ) -> None:
        self.w_short = w_short
        self.w_long = w_long
        self.threshold = threshold
        self.horizon = horizon
        self.refractory = refractory

    def detect(self, residuals: np.ndarray) -> np.ndarray:
        """
        Run detector over a 1-D residual series e_t = |y_t - y_hat_t|.
        Returns boolean alarm array of shape (T,).
        """
        T = len(residuals)
        alarms = np.zeros(T, dtype=bool)
        triggers = np.zeros(T, dtype=bool)

        cooldown = 0
        refractory_timer = 0
        eps = 1e-5

        for t in range(self.w_long, T):
            if cooldown > 0:
                cooldown -= 1
                alarms[t] = True
            if refractory_timer > 0:
                refractory_timer -= 1

            short_win = residuals[t - self.w_short + 1: t + 1]
            long_win = residuals[t - self.w_long + 1: t + 1]

            s_short = float(np.std(short_win))
            s_long = float(np.std(long_win))

            ratio = s_short / (s_long + eps)

            if ratio >= self.threshold and refractory_timer == 0:
                triggers[t] = True
                alarms[t] = True
                cooldown = self.horizon
                refractory_timer = self.refractory

        return alarms


class ResidualCUSUMDetector:
    """
    Cumulative Sum (CUSUM / Page-Hinkley) detector on squared residuals.
    Detects jumps in innovation variance sigma^2.
    """

    def __init__(
        self,
        sigma_0: float = 5.0,
        allowance_k: float = 0.5,
        threshold_h: float = 8.0,
        horizon: int = 20,
        refractory: int = 20,
    ) -> None:
        self.sigma_0 = sigma_0
        self.allowance_k = allowance_k
        self.threshold_h = threshold_h
        self.horizon = horizon
        self.refractory = refractory

    def detect(self, residuals: np.ndarray) -> np.ndarray:
        """
        Run CUSUM over residual series.
        Returns boolean alarm array of shape (T,).
        """
        T = len(residuals)
        alarms = np.zeros(T, dtype=bool)

        s_t = 0.0
        cooldown = 0
        refractory_timer = 0
        var_0 = self.sigma_0 ** 2

        for t in range(1, T):
            if cooldown > 0:
                cooldown -= 1
                alarms[t] = True
            if refractory_timer > 0:
                refractory_timer -= 1

            # Standardized variance score
            score = (residuals[t] ** 2) / var_0 - 1.0
            s_t = max(0.0, s_t + score - self.allowance_k)

            if s_t >= self.threshold_h and refractory_timer == 0:
                alarms[t] = True
                cooldown = self.horizon
                refractory_timer = self.refractory
                s_t = 0.0  # reset accumulator upon trigger

        return alarms
