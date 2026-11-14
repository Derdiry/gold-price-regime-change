"""Directional-accuracy definitions (src/evaluation/metrics.py)."""

import numpy as np

from src.evaluation.metrics import compute_all_metrics, compute_directional_scores


def v1_directional_accuracy(y_true, y_pred):
    """The v1 formula: flatten, then compare signs of consecutive differences."""
    t, p = np.sign(np.diff(np.ravel(y_true))), np.sign(np.diff(np.ravel(y_pred)))
    return (t == p).mean() * 100


def test_h1_matches_v1_formula():
    rng = np.random.default_rng(0)
    y_true = np.round(rng.normal(100, 1, size=(50, 1)))  # rounding creates zero changes
    y_pred = rng.normal(100, 1, size=(50, 1))
    assert np.isclose(compute_all_metrics(y_true, y_pred)["directional_accuracy"],
                      v1_directional_accuracy(y_true, y_pred))


def test_multi_step_does_not_cross_window_boundaries():
    # Two windows of horizon 2. Per-step differences across origins:
    # step 0: 10 -> 12 (up), step 1: 11 -> 13 (up). Predictions match both.
    y_true = np.array([[10.0, 11.0], [12.0, 13.0]])
    y_pred = np.array([[10.0, 50.0], [12.5, 60.0]])
    assert compute_all_metrics(y_true, y_pred)["directional_accuracy"] == 100.0
    # v1 flattened [10, 11, 12, 13] vs [10, 50, 12.5, 60] and also compared
    # 11 -> 12 (up) against 50 -> 12.5 (down) across the window boundary.
    assert v1_directional_accuracy(y_true, y_pred) < 100.0


def test_nonzero_and_ceiling_with_forward_filled_target():
    # True changes: 0, +1, 0, -1  -> 2 of 4 non-zero, ceiling 50%.
    y_true = np.array([5, 5, 6, 6, 5], float)
    # Predicted changes: +, +, -, -  -> correct on both non-zero steps, wrong on zeros.
    y_pred = np.array([5.0, 5.1, 5.2, 5.1, 5.0])
    s = compute_directional_scores(y_true, y_pred)
    assert s["directional_accuracy"] == 50.0
    assert s["directional_accuracy_nonzero"] == 100.0
    assert s["directional_accuracy_max"] == 50.0
    assert s["directional_accuracy_n_nonzero"] == 2


def test_origin_direction_uses_last_observed_value():
    y_last = np.array([100.0, 100.0, 100.0])
    y_true = np.array([[101.0], [100.0], [99.0]])   # up, flat, down
    y_pred = np.array([[100.5], [100.2], [100.1]])  # up, up, up
    s = compute_directional_scores(y_true, y_pred, y_last)
    assert np.isclose(s["da_origin"], 100 / 3)
    assert s["da_origin_nonzero"] == 50.0
    assert np.isclose(s["da_origin_max"], 200 / 3)


def test_persistence_gets_zero_change_credit_only_in_raw_score():
    y_last = np.array([1.0, 1.0, 2.0, 2.0])
    y_true = np.array([[1.0], [2.0], [2.0], [1.0]])
    persistence = y_last.reshape(-1, 1)
    s = compute_directional_scores(y_true, persistence, y_last)
    assert s["da_origin"] == 50.0          # matches the two zero changes
    assert s["da_origin_nonzero"] == 0.0   # never right when the price moves
