"""
Centralized seed allocations across all components.

Guarantees mutually disjoint seed ranges to prevent inter-component data
leakage or accidental stream reuse.
"""
from typing import Sequence

# Gate training & test sets
GATE_TRAIN_SEEDS: Sequence[int] = range(0, 40)         # 40 multi-shift streams
HELD_OUT_TEST_SEEDS: Sequence[int] = range(1000, 1010) # 10 multi-shift test streams

# Forecaster training (disjoint from gate training streams)
FORECASTER_TRAIN_SEEDS: Sequence[int] = range(100, 140) # 40 streams (stationary & multi-shift)

# ACI evaluation & benchmarks
CONTROLLED_EVAL_SEEDS: Sequence[int] = range(7000, 7010) # 10 single-shift streams
MULTI_SHIFT_SANITY_SEEDS: Sequence[int] = range(8000, 8005) # 5 multi-shift sanity streams
STATIONARY_SANITY_SEEDS: Sequence[int] = range(9000, 9005)  # 5 stationary sanity streams
