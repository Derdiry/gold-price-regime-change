"""
Evaluation metrics for gold price forecasting models.

All metrics take arrays of shape (n_windows, horizon) in original price units
(1-D arrays are treated as horizon 1). Row i is the forecast made at origin i;
column k is the (k+1)-step-ahead value.

Point metrics (over every window and step):
  - MAE:  Mean Absolute Error (primary metric, same units as price)
  - RMSE: Root Mean Squared Error (penalises large misses)
  - MAPE: Mean Absolute Percentage Error (scale-independent, % units)

Directional metrics, two definitions:

  1. ``directional_accuracy`` (the v1 metric). For each step k, compare the
     change between consecutive forecast origins:
         sign(y_true[i+1, k] - y_true[i, k]) == sign(y_pred[i+1, k] - y_pred[i, k])
     For h = 1 this is exactly the v1 formula. For h > 1, v1 flattened the
     (n, h) array first and so also differenced across window boundaries
     (last step of window i against first step of window i+1). v2 differences
     each step column separately.

  2. ``da_origin``: did the forecast get the direction of change from the last
     observed value right?
         sign(y_true[i, k] - y_last[i]) == sign(y_pred[i, k] - y_last[i])
     It needs ``y_last`` (the last context value, in price units).

Why the zero-change variants exist. The Sudan series is irregular and is
forward-filled to daily frequency, so most true changes are exactly zero
(at h = 1 on the v1 test set only 92 of 436 changes are non-zero). np.sign
gives 0 for those, and a continuous forecast never predicts a change of
exactly 0, so every zero-change step counts as wrong. The raw score is then
bounded far below 100% and is not comparable across tracks. For each
definition we therefore also report:

  - ``*_nonzero``: accuracy over only the steps whose true change is non-zero.
    This is the meaningful directional score; 50% is chance.
  - ``*_max``: the highest raw score a forecaster can reach if it never
    predicts an exactly-zero change: the percentage of steps with a non-zero
    true change. A raw score should be read against this ceiling, not
    against 100%.

A forecaster that does predict zero changes (e.g. persistence for
``da_origin``) gets credit for matching zero true changes in the raw score,
which is why the raw score can exceed ``*_max`` for such baselines.
"""

import numpy as np


def _as_2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error, in the units of the target."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error, in the units of the target."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """MAPE in percentage points. Clips near-zero true values to avoid division errors."""
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _direction_scores(true_change: np.ndarray, pred_change: np.ndarray, prefix: str) -> dict:
    """Raw, non-zero-only and maximum-attainable directional accuracy (all in %)."""
    true_dir = np.sign(true_change).ravel()
    pred_dir = np.sign(pred_change).ravel()
    n = true_dir.size
    if n == 0:
        nan = float("nan")
        return {prefix: nan, f"{prefix}_nonzero": nan, f"{prefix}_max": nan, f"{prefix}_n_nonzero": 0}
    nonzero = true_dir != 0
    correct = true_dir == pred_dir
    return {
        prefix: float(correct.mean() * 100),
        f"{prefix}_nonzero": float(correct[nonzero].mean() * 100) if nonzero.any() else float("nan"),
        f"{prefix}_max": float(nonzero.mean() * 100),
        f"{prefix}_n_nonzero": int(nonzero.sum()),
    }


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    v1 directional accuracy (see module docstring, definition 1), in [0, 100].

    Differences consecutive forecast origins separately for each step k.
    Requires at least 2 windows.
    """
    return compute_directional_scores(y_true, y_pred)["directional_accuracy"]


def compute_directional_scores(y_true: np.ndarray, y_pred: np.ndarray, y_last=None) -> dict:
    """
    All directional metrics.

    Always returns ``directional_accuracy`` with its ``_nonzero`` / ``_max`` /
    ``_n_nonzero`` companions; with ``y_last`` (shape (n,)) also the
    ``da_origin`` family.
    """
    y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
    out = _direction_scores(np.diff(y_true, axis=0), np.diff(y_pred, axis=0), "directional_accuracy")
    if y_last is not None:
        last = np.asarray(y_last, dtype=np.float64).reshape(-1, 1)
        out.update(_direction_scores(y_true - last, y_pred - last, "da_origin"))
    return out


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_last=None) -> dict:
    """
    Compute all metrics and return them as a dict.

    y_true, y_pred: (n_windows, horizon) or (n_windows,), original price units.
    y_last: optional (n_windows,) last observed value at each forecast origin;
            enables the ``da_origin`` metrics.
    """
    y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        **compute_directional_scores(y_true, y_pred, y_last),
        "n_windows": int(y_true.shape[0]),
        "horizon": int(y_true.shape[1]),
    }


def format_metrics_table(results: dict[str, dict]) -> str:
    """
    Format a dict of {model_name: metrics_dict} as a markdown table.

    Example:
        results = {
            "lstm": {"mae": 12.3, "rmse": 18.1, "mape": 2.3, "directional_accuracy": 61.2, ...},
            ...
        }
    """
    header = "| Model | MAE | RMSE | MAPE (%) | DA (%) | DA non-zero (%) | DA max (%) |"
    sep = "|-------|-----|------|----------|--------|-----------------|------------|"
    rows = [header, sep]
    for name, m in sorted(results.items(), key=lambda x: x[1].get("mae", float("inf"))):
        rows.append(
            f"| {name} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['mape']:.2f} | "
            f"{m['directional_accuracy']:.1f} | {m.get('directional_accuracy_nonzero', float('nan')):.1f} | "
            f"{m.get('directional_accuracy_max', float('nan')):.1f} |"
        )
    return "\n".join(rows)
