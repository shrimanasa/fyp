# Phase 2 — Base Forecaster Ablation: EWMA vs. GBM

> All numbers are reproducible by re-running `python scripts/run_phase2.py`.
> Paired evaluation runs on the exact same seeds from `services.seeds.CONTROLLED_EVAL_SEEDS`.

## Executive Summary

- **Baseline forecaster**: EWMA ($y_t = 0.3 y_{t-1} + 0.7 \hat{y}_{t-1}$)
- **Candidate forecaster**: `HistGradientBoostingRegressor` predicting step increment $\Delta y_t = y_t - y_{t-1}$ from causal lag differences and local volatility
- **Decision: Adopt GBM Forecaster for Phase 3**
- GBM outperforms the EWMA heuristic in point forecast accuracy and provides narrower prediction intervals while maintaining empirical conformal coverage. The increment-prediction formulation avoids tree extrapolation failure across shifts.

---

## 1. Point Prediction Accuracy

*Evaluated on disjoint test streams.*

| Metric | EWMA (Baseline) | GBM (Candidate) | Difference (GBM − EWMA) | Better |
|--------|-----------------|-----------------|-------------------------|--------|
| **Stationary RMSE** | 6.0049 | 5.0499 | -0.9550 | GBM |
| **Stationary MAE** | 4.8046 | 4.0295 | -0.7751 | GBM |
| **Shift-stream RMSE** | 6.1022 | 5.0425 | -1.0597 | GBM |
| **Shift-stream MAE** | 4.8755 | 4.0370 | -0.8385 | GBM |

*Note: For an AR(1) process with innovation $\sigma = 5.0$, $\sigma$ represents the Bayes irreducible error floor. GBM stationary RMSE (5.0499) matches this floor closely without leaking.*

---

## 2. ACI Conformal Coverage & Interval Width (Paired Ablation)

*Evaluated across 10 controlled single-shift streams (shift fixed at $t=500$, $T=2000$).*
*Paired differences are calculated per seed on the exact same noise realizations.*

| Window | EWMA Coverage (Width) | GBM Coverage (Width) | Paired Coverage Diff (GBM − EWMA) | Paired Width Diff (GBM − EWMA) |
|--------|-----------------------|----------------------|-----------------------------------|--------------------------------|
| **Overall** | 0.9008 (20.14) | 0.9005 (16.85) | -0.0003 | -3.30 |
| **Near-shift (0–19 steps, PRIMARY DIP)** | **0.8500 (20.45)** | **0.8900 (16.80)** | **+0.0400 ± 0.0583** | **-3.65 ± 1.4805** |
| Near-shift (0–99 steps, flat average) | 0.9060 (20.07) | 0.9010 (16.64) | -0.0050 | -3.43 |
| Steady-state (850–950 steps) | 0.9270 (20.82) | 0.9070 (17.55) | -0.0200 | -3.27 |

---

## 3. Distance-from-Shift Binned Curve (Steps Post-Onset)

| Distance | EWMA Coverage (Width) | GBM Coverage (Width) | $\Delta$ Coverage | $\Delta$ Width |
|----------|-----------------------|----------------------|--------------------|-----------------|
| Steps  0– 9 | 0.8500 (w=20.38) | 0.9200 (w=16.77) | +0.0700 | -3.61 |
| Steps 10–19 | 0.8500 (w=20.51) | 0.8600 (w=16.82) | +0.0100 | -3.69 |
| Steps 20–29 | 0.9400 (w=20.48) | 0.9200 (w=16.66) | -0.0200 | -3.83 |
| Steps 30–39 | 0.9700 (w=20.07) | 0.9300 (w=16.51) | -0.0400 | -3.56 |
| Steps 40–49 | 0.9500 (w=19.74) | 0.8900 (w=16.45) | -0.0600 | -3.28 |
| Steps 50–59 | 0.9300 (w=19.56) | 0.9100 (w=16.48) | -0.0200 | -3.09 |
| Steps 60–69 | 0.8600 (w=19.85) | 0.8600 (w=16.45) | +0.0000 | -3.40 |
| Steps 70–79 | 0.9200 (w=20.07) | 0.8900 (w=16.68) | -0.0300 | -3.39 |
| Steps 80–89 | 0.8900 (w=20.17) | 0.9100 (w=16.85) | +0.0200 | -3.32 |
| Steps 90–99 | 0.9000 (w=19.82) | 0.9200 (w=16.69) | +0.0200 | -3.13 |

---

## 4. Key Takeaways & Transition to Phase 3

1. **Shift Invariance**: The relative-increment formulation prevents tree models from getting trapped during mean jumps.
2. **Interval Efficiency vs. Coverage**: ACI maintains nominal coverage under both forecasters, but interval width reflects forecaster sharpness.
3. **Primary Metric for Gate in Phase 3**:
   - The initial dip remains at steps 0–19.
   - Phase 3 will test whether coupling the learned drift gate with ACI improves coverage specifically in this 0–19 step window compared to ACI alone.
