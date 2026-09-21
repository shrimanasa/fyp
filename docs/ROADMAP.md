# Roadmap

## Phase 1 — Threshold selection ✅ DONE
- Trained gate classifier (HistGradientBoostingClassifier, held-out-by-schedule split)
- Selected threshold at max recall @ FPR ≤ 5% (thr=0.660, recall=68.3%, FPR=4.95%)
- Deliverable: `docs/THRESHOLD_CHOICE.md`
- Notable finding: pre-shift labels unlearnable (AUROC=0.547); corrected to onset-window labels

## Phase 2 — Upgrade base forecaster ✅ DONE
- Replaced EWMA stub with trained `HistGradientBoostingRegressor` (`services/forecaster/gbm.py`)
- Formulated causal relative step-increment prediction ($\Delta y_t = y_t - y_{t-1}$) to prevent tree extrapolation failure during regime shifts
- Executed paired-seed ablation (`docs/PHASE2_ABLATION.md`):
  - Point prediction: RMSE dropped from 6.10 to 5.04 (matching Bayes noise floor $\sigma=5.0$)
  - ACI interval width: tightened by ~16.4% (from 20.14 to 16.85) at nominal 90% coverage
  - Near-shift dip (0–19 steps): improved from 85.0% to 89.0%
- Deliverable: `docs/PHASE2_ABLATION.md` and `scripts/verify_forecaster.py`

## Phase 3 — End-to-end evaluation framework ⏳
- One script: ACI alone vs. ACI + drift gate on held-out shift schedules
- Core plot: empirical coverage vs. distance-from-shift, one line per condition
- Core table: overall, near-shift, false-alarm rate — gate on vs. off
- Success bar stated before running: gate is worth keeping only if near-shift coverage
  improves without dragging overall below ~88–89%

## Phase 4 — COP baseline reproduction ⏳ (hardest)
- Reproduce COP (conformal online prediction under shift, 2026) from paper
- Write `docs/COP_NOTES.md` in own words before any code
- If full reproduction infeasible, ship partial and say so explicitly

## Phase 5 — Full-stack app ⏳
- FastAPI backend: `/predict` endpoint (point prediction, interval, gate score, gate decision)
- Next.js frontend: live chart with shift/gate-trigger markers

## Phase 6 — Deployment ⏳
- Backend: Render/Railway/Fly.io free tier
- Frontend: Vercel
- `docker-compose.yml` for local reproduction

## Phase 7 — Honest write-up ⏳
- `docs/RESULTS.md` + README rewrite
- Hypothesis, baseline numbers, what the gate actually achieved
- Explicitly includes weak results, partial COP reproduction, open questions

---

## Known deferred items

| Item | Deferred to |
|------|------------|
| Hold+decay shift dynamics (recovery dip) | Phase 2 or 3 — generator currently has permanent-step shifts only |
| error_rate gate (currently only latency gate trained) | Phase 3 |
| Second domain (financial time series) | Phase 7+ stretch goal |
