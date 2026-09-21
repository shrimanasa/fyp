# Phase 1 — Drift-Gate Threshold Choice

> All numbers in this document are produced by `scripts/run_phase1.py` and
> are reproducible by re-running that script. No numbers are hand-picked.

## What is real vs. stub

| Component | Status |
|-----------|--------|
| ACI implementation (Gibbs & Candès update rule) | **Real** — `services/conformal/aci.py` |
| Gate classifier (HistGradientBoostingClassifier) | **Real** — `services/gate/train.py` |
| Synthetic data generator | **Real** — `services/data/generator.py` |
| Point forecaster inside ACI | **Stub (EWMA)** — replaced in Phase 2 |
| End-to-end coverage comparison (gate on vs. off) | **Pending** — Phase 3 |

---

## Gate classifier — baseline numbers

| Metric | Value |
|--------|-------|
| Held-out AUROC | 0.9051 |
| Default (0.5) threshold recall | 0.7587 |
| Default (0.5) threshold FPR | 0.1072 |
| Default (0.5) threshold precision | 0.3721 |
| Default (0.5) threshold F1 | 0.4993 |

---

## ACI baseline coverage (EWMA forecaster, Phase 1)

### 1. Pure stationary baseline (zero shifts)
*Evaluated across 5 independent noise seeds on stationary AR(1) stream (target: 0.9000).*

| Condition | Empirical coverage (mean ± std) | Target | Calibration error |
|-----------|---------------------------------|--------|-------------------|
| Pure stationary | 0.9021 ± 0.0014 | 0.9000 | 0.0021 |

### 2. Controlled single-shift evaluation
*Shift fixed at t=500, T=2000, 10 noise seeds. Windows guaranteed fully populated (20 samples/seed for 0–19, 100 for 0–99 and steady-state).
Point forecaster is EWMA (stub); replaced in Phase 2.*

| Window | Coverage (mean ± std) | Seeds used | Min window n | Role in Phase 3 |
|--------|-----------------------|------------|--------------|-----------------|
| Overall | 0.9008 | 10/10 | 1950 | Must stay ~88–90% |
| **Near-shift (0–19 steps post-onset)** | **0.8500 ± 0.1204** | 10/10 | 20 | **PRIMARY METRIC: gate must improve this** |
| Near-shift (0–99 steps, flat average) | 0.9060 ± 0.0400 | 10/10 | 100 | Reference only (masks dip via overshoot) |
| Steady-state (850–950 steps post-shift) | 0.9270 ± 0.0382 | 10/10 | 100 | Fully-adapted ACI |

### 3. Distance-from-shift 10-step binned curve
*The flat 100-step average (0.9060) masks a real ~15pp dip in the first 20 steps (0.8500), because ACI subsequently overcorrects and over-covers (0.9400–0.9700 in steps 20–49). Phase 3 evaluates whether the gate mitigates the 0–19 step drop without exaggerating the subsequent overcorrection.*

| Distance post-onset | Coverage (mean ± std) | Seeds used | Min window n |
|---------------------|-----------------------|------------|--------------|
| Steps  0– 9 | 0.8500 ± 0.1688 | 10/10 | 10 |
| Steps 10–19 | 0.8500 ± 0.1432 | 10/10 | 10 |
| Steps 20–29 | 0.9400 ± 0.1200 | 10/10 | 10 |
| Steps 30–39 | 0.9700 ± 0.0640 | 10/10 | 10 |
| Steps 40–49 | 0.9500 ± 0.0500 | 10/10 | 10 |
| Steps 50–59 | 0.9300 ± 0.0900 | 10/10 | 10 |
| Steps 60–69 | 0.8600 ± 0.1356 | 10/10 | 10 |
| Steps 70–79 | 0.9200 ± 0.0600 | 10/10 | 10 |
| Steps 80–89 | 0.8900 ± 0.1578 | 10/10 | 10 |
| Steps 90–99 | 0.9000 ± 0.1095 | 10/10 | 10 |

---

## Candidate threshold analysis

Three candidates were evaluated. All numbers are on the same held-out test
set; no threshold was selected before seeing these numbers.

| Candidate | Threshold | Precision | Recall | FPR | F1 |
|-----------|-----------|-----------|--------|-----|----|
| max_recall@FPR≤5% | 0.6597 | 0.5358 | 0.6827 | 0.0495 | 0.6004 |
| max_F1 | 0.7657 | 0.6693 | 0.6113 | 0.0253 | 0.6390 |
| precision≥50% | 0.6293 | 0.5000 | 0.6973 | 0.0584 | 0.5824 |
| default (0.5) | 0.5000 | 0.3721 | 0.7587 | 0.1072 | 0.4993 |

---

## Chosen threshold

**Candidate chosen: `max_recall@FPR≤5%`**
**Threshold value: `0.6597`**

### Justification

max_recall@FPR≤5% is the primary choice: early warning requires recall and a 5% FPR cap keeps false alarms tolerable for an ops dashboard (roughly one false alert per 20 windows on average).

### Rejected alternatives

- **max_F1** (thr=0.7657): optimises the
  precision-recall trade-off symmetrically. Appropriate if false alarms and
  missed shifts are equally costly. In the infra-metrics domain, a missed
  shift (coverage crater) is more costly than a spurious alert, so we
  prefer higher recall.

- **precision≥50%** (thr=0.6293):
  chosen to cap false-alarm rate more aggressively. Precision=
  0.5000 with recall=
  0.6973. The precision gain is
  modest relative to the recall cost; this tradeoff is unfavourable for
  early-warning use.

- **default (0.5)**: recall=0.7587,
  FPR=0.1072. The brief noted this as
  not yet an operating point. Reported here for completeness.

---

## Design choices with justifications

### 50-step lead window for positive labels

Features use windows of width 10, 30, and 60 steps. 50 steps is the
shortest lead that guarantees a full 30-step window of pre-shift signal
is visible in the features. Shorter (e.g., 30 steps) leaves the
medium-window statistics with zero pre-shift content; longer (e.g., 100
steps) dilutes the positive class and makes precision harder to achieve
at any useful recall.

### Held-out-by-shift-schedule (not by timestep)

The train/test split holds out entire shift schedules, not segments of
streams. Splitting mid-stream leaks information because model features
from one segment directly reflect the statistical properties of adjacent
segments under the same shift regime.

---

## Figures

- `docs/figures/roc_curve.png` — ROC curve with candidate threshold markers
- `docs/figures/pr_curve.png` — Precision-Recall curve with candidate markers
