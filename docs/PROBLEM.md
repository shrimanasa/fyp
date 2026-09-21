# Problem Statement

## Core problem

Streaming prediction intervals — intervals that should contain the true value
with a target probability (e.g., 90%) — degrade after a distribution shift.
Standard Conformal Prediction assumes exchangeability; Adaptive Conformal
Inference (ACI, Gibbs & Candès 2021) partially addresses this by adapting the
coverage level over time. But ACI adapts *reactively*: it only widens intervals
after it has already missed coverage. Near a shift, there is a window of poor
coverage before ACI's calibration pool has enough post-shift data to adapt.

## Research question

Can a learned gate — a binary classifier that fires at shift onset — signal
ACI to widen its intervals early enough to reduce the near-shift coverage
crater? And if so, by how much, and at what false-alarm cost?

## Domain

Primary: streaming infra metrics (service latency, error rate). These exhibit
sudden regime changes (deployments, traffic spikes) with no gradual lead-up —
making the "before vs. during" detection distinction particularly relevant.

Stretch goal: financial time series (Phase 7+).

## What the gate can and cannot do

**Cannot:** Predict a shift before it starts. With step-function shifts (the
model used here), the distribution before the shift is identical to any normal
window — there is no predictive signal. AUROC of 0.547 was confirmed
empirically with pre-shift labels.

**Can:** Detect that a shift has started within ~50 steps of onset. ACI's
200-step calibration window means coverage degrades gradually, so a 50-step
detection lead can trigger interval widening before the crater fully forms.

## Success criterion (to be evaluated in Phase 3)

The gate is worth keeping only if near-shift initial-dip coverage (steps 0–19
post-onset, where empirical baseline drops to 0.8500 vs. 0.9000 target)
improves under gate-on vs. gate-off, without dragging overall coverage below
~88–89% or severely exaggerating post-dip overcorrection. A flat 100-step average
is insufficient as a metric because subsequent over-coverage (0.94–0.97) masks the
initial dip. If the gate fails to improve the initial dip, that result is
reported plainly.

## What this is not

- Not an anomaly detector (no unsupervised component).
- Not a causal model of shift mechanisms.
- Not evaluated on real operational data (synthetic only as of Phase 1).
