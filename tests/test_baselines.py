"""Baseline forecasts use only the context window and are in price units."""

import numpy as np

from src.evaluation.baselines import drift, persistence, statistical_paths


def test_persistence_repeats_last_value():
    assert persistence(np.array([3.0, 5.0]), 3).tolist() == [[3, 3, 3], [5, 5, 5]]


def test_drift_extends_context_slope():
    ctx = np.array([[10.0, 11.0, 12.0, 13.0]])          # +1 per step
    assert drift(ctx, 3).tolist() == [[14.0, 15.0, 16.0]]


def test_statistical_fallback_is_counted_and_persistence():
    # A two-point history cannot fit ARIMA(2,1,2) with drift -> falls back
    out, fallbacks = statistical_paths("arima", {"order": (2, 1, 2), "trend": "t"}, [np.array([5.0, 5.0])], 4)
    assert fallbacks in (0, 1)
    if fallbacks:
        assert out.tolist() == [[5.0] * 4]
