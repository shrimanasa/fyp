# Conformal Drift-Gate

> **Honest status as of Phase 2:** The gate classifier is trained and
> threshold-selected (Phase 1). The base point forecaster inside ACI has been
> upgraded from EWMA to a trained `HistGradientBoostingRegressor` (Phase 2),
> reducing RMSE from 6.10 to 5.04 and tightening average prediction interval width
> from 20.14 to 16.85 while maintaining 90% coverage (see `docs/PHASE2_ABLATION.md`).
> The end-to-end gate-on vs. gate-off comparison has not been run yet (Phase 3).

## What this system actually does

The gate does **not** predict distribution shifts before they arrive.
With step-function shifts, there is no pre-shift signal — a classifier
trained to fire 50 steps *before* a shift yielded AUROC 0.547 (near-random),
which confirmed this empirically.

What the gate does: **detect that a shift has just started, within ~50 steps
of onset.** The honest description is "during, fast" not "before."

Why that can still help: ACI's calibration window is 200 steps wide. After
a step shift, empirical coverage degrades gradually over ~100–200 steps as
the window fills with stale pre-shift scores. A gate that fires within 50
steps of onset gives ACI time to widen its intervals before the bulk of
the coverage crater accumulates. Whether this actually improves near-shift
coverage is the question Phase 3 answers — it has not been measured yet.

---

## What is real vs. stub

| Component | Status |
|-----------|--------|
| ACI (Gibbs & Candès 2021 update rule) | ✅ Real — `services/conformal/aci.py` |
| Gate classifier (HistGradientBoostingClassifier) | ✅ Real — `services/gate/train.py` |
| Threshold selection | ✅ Done — see `docs/THRESHOLD_CHOICE.md` |
| Synthetic data generator (AR(1) latency, Poisson error_rate) | ✅ Real — `services/data/generator.py` |
| Point forecaster inside ACI | ✅ Real — `services/forecaster/gbm.py` (`docs/PHASE2_ABLATION.md`) |
| End-to-end coverage comparison (gate on vs. gate off) | ⏳ Phase 3 |
| COP baseline | ⏳ Phase 4 |
| Full-stack app (FastAPI + Next.js) | ⏳ Phase 5 |
| Deployment | ⏳ Phase 6 |

---

## Known limitations (as of Phase 1)

- **Shift model is step-function only.** The generator applies permanent
  mean shifts. There is no hold+decay (reversion) dynamic, so the "recovery
  dip" phenomenon described in the original brief is not reproducible with
  this generator. The 850–950 steps post-shift window measures fully-adapted
  steady-state coverage, not a secondary dip.

- **Gate labels are onset-window, not pre-shift.** The positive class is
  defined as `[s, s+50)` — the first 50 steps *after* a shift. See
  `docs/THRESHOLD_CHOICE.md` for the design rationale and the empirical
  failure that led to this choice.

- **Coverage benefit is not yet measured.** `docs/THRESHOLD_CHOICE.md`
  reports gate discrimination (AUROC, precision, recall). Whether operating
  the gate at the chosen threshold actually improves near-shift coverage
  is measured in Phase 3.

---

## Reproducing the Project
```bash
# Install dependencies
pip install numpy scipy scikit-learn matplotlib joblib

# Run from project root:
# 1. Verification checks
python scripts/verify_aci.py          # ACI calibration & structural invariants
python scripts/verify_gate.py         # Gate model discrimination check
python scripts/verify_forecaster.py   # Forecaster causality & noise floor check

# 2. Phase pipelines
python scripts/run_phase1.py          # Phase 1: Gate threshold choice -> docs/THRESHOLD_CHOICE.md
python scripts/run_phase2.py          # Phase 2: Forecaster ablation -> docs/PHASE2_ABLATION.md
```

---

## Project structure

```
conformal-drift-gate/
├── docs/
│   ├── THRESHOLD_CHOICE.md   ← Phase 1 deliverable
│   ├── PHASE2_ABLATION.md    ← Phase 2 deliverable
│   ├── ROADMAP.md
│   ├── PROBLEM.md
│   └── figures/
│       ├── roc_curve.png
│       └── pr_curve.png
├── scripts/
│   ├── run_phase1.py         ← Phase 1 pipeline
│   ├── run_phase2.py         ← Phase 2 ablation pipeline
│   ├── verify_aci.py         ← ACI structural invariant check
│   ├── verify_gate.py        ← Gate model sanity check
│   └── verify_forecaster.py  ← Forecaster causality & noise floor check
├── services/
│   ├── conformal/
│   │   ├── aci.py            ← ACI implementation
│   │   └── evaluate.py       ← Coverage & width evaluation harness
│   ├── data/
│   │   └── generator.py      ← Synthetic infra-metrics stream
│   ├── forecaster/
│   │   ├── features.py       ← Causal shift-invariant feature extractor
│   │   ├── gbm.py            ← GBM point forecaster
│   │   └── train.py          ← Forecaster training pipeline
│   ├── gate/
│   │   ├── features.py       ← Sliding-window feature extractor
│   │   ├── train.py          ← Classifier training
│   │   └── evaluate_gate.py  ← ROC/PR curves, threshold analysis
│   └── seeds.py              ← Centralized seed allocations
└── pyproject.toml
```
