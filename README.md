# Conformal Drift-Gate

> **Pairing Adaptive Conformal Inference (ACI) with a learned drift gate to safeguard prediction interval coverage under streaming distribution shifts.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Phase%201%20%26%202-Complete-success.svg)](docs/ROADMAP.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚡ Key Results at a Glance

| Benchmark | Value | Baseline Comparison | Significance |
|---|---|---|---|
| **Gate Discrimination** | **0.9051 AUROC** | vs. 0.547 (pre-shift) | Detects shift onset within 50 steps; pre-shift prediction is impossible on step shifts |
| **Operating Threshold** | **0.6597** | max recall @ FPR ≤ 5% | 68.3% recall, 4.95% false alert rate |
| **Base Forecaster RMSE** | **5.04** | vs. 6.10 (EWMA) | Reaches theoretical Bayes noise floor ($\sigma = 5.00$) |
| **Prediction Interval Width** | **16.85** | vs. 20.14 (EWMA) | **16.4% narrower intervals** across stream at nominal 90% coverage |
| **Crater Coverage (Steps 0–19)** | **89.0%** | vs. 85.0% (EWMA) | GBM mitigates initial crater even before active gate widening |
| **Stationary Coverage** | **90.2%** | Target 90.0% | Gibbs & Candès ACI recursion is well-calibrated (error < 0.3pp) |

---

## 🎯 The Core Concept: "During, Fast" vs. "Before"

Streaming prediction intervals degrade after distribution shifts. Standard Adaptive Conformal Inference (ACI) adapts **reactively** — it only widens intervals *after* miscoverage occurs, causing an empirical coverage crater immediately post-shift.

```
Regime Shift
   │
   ▼
[ s ] ────── 50 steps ──────► [ s+50 ] ─────────────► [ s+200 ]
  │                              │                        │
  └─ Drift Gate fires here       └─ ACI alone still       └─ ACI calibration
     (expands interval width)       misses coverage          pool finally adapts
```

1. **What the gate CANNOT do:** Predict shifts *before* they arrive. In step-function shifts, the pre-shift signal is statistically indistinguishable from baseline (empirically confirmed: pre-shift classifier yielded AUROC 0.547, near random).
2. **What the gate DOES:** Detects that a regime shift has **just begun** within $\sim$50 steps of onset ("during, fast").
3. **Why this helps:** ACI's calibration window is 200 steps wide. Coverage degrades over 100–200 steps as stale scores flush out. Firing within 50 steps allows early interval widening before the bulk of the crater accumulates.

---

## 🧭 Project Roadmap & Deliverables

| Phase | Milestone | Deliverable | Status |
|:---:|---|---|:---:|
| **1** | **Threshold Selection & ACI Baseline** | [`docs/THRESHOLD_CHOICE.md`](docs/THRESHOLD_CHOICE.md) | ✅ Complete |
| **2** | **Base Forecaster Upgrade (EWMA → GBM)** | [`docs/PHASE2_ABLATION.md`](docs/PHASE2_ABLATION.md) | ✅ Complete |
| **3** | **End-to-End Evaluation (Gate ON vs. OFF)** | Distance-from-shift paired ablation | ⏳ Next |
| **4** | **COP Baseline Comparison** | Conformal Online Prediction reproduction | ⏳ Pending |
| **5** | **Interactive Full-Stack Dashboard** | FastAPI backend + Next.js frontend | ⏳ Pending |
| **6** | **Cloud Deployment & Live Demo** | Public demo instance | ⏳ Pending |

*Detailed problem specification: [`docs/PROBLEM.md`](docs/PROBLEM.md) | Full roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md)*

---

## 🚀 Quickstart & Reproducibility

Every metric and claim in this repository is strictly reproducible from the project root.

```bash
# 1. Clone and install
git clone git@github.com:shrimanasa/fyp.git
cd fyp
pip install numpy scipy scikit-learn matplotlib joblib

# 2. Run integrity and invariant tests
python scripts/verify_aci.py          # ACI calibration & crater check
python scripts/verify_gate.py         # Gate discrimination check (AUROC > 0.60)
python scripts/verify_forecaster.py   # Forecaster causality & Bayes noise floor check

# 3. Reproduce phase reports
python scripts/run_phase1.py          # Re-generates docs/THRESHOLD_CHOICE.md
python scripts/run_phase2.py          # Re-generates docs/PHASE2_ABLATION.md
```

---

## 📂 Project Architecture

```
conformal-drift-gate/
├── docs/                     # Research deliverables & experimental notes
│   ├── PROBLEM.md            # Problem framing & success criteria
│   ├── ROADMAP.md            # Multi-phase progression & deferred items
│   ├── THRESHOLD_CHOICE.md   # Phase 1: Gate threshold selection & baseline
│   └── PHASE2_ABLATION.md    # Phase 2: EWMA vs. GBM paired ablation
├── scripts/                  # Executable pipelines and automated verifications
│   ├── run_phase1.py         # Phase 1 pipeline
│   ├── run_phase2.py         # Phase 2 ablation pipeline
│   ├── verify_aci.py         # ACI calibration & structural invariants
│   ├── verify_gate.py        # Gate model sanity check
│   └── verify_forecaster.py  # Forecaster causality & noise floor invariant
├── services/                 # Modular system components
│   ├── conformal/            # Gibbs & Candès ACI + evaluation harness
│   ├── forecaster/           # Shift-invariant GBM regressor + features
│   ├── gate/                 # HistGradientBoosting drift classifier + features
│   ├── data/                 # Synthetic AR(1) and Poisson metrics stream
│   └── seeds.py              # Centralized disjoint seed allocations
└── pyproject.toml            # Project dependencies & metadata
```
