"""
Forecast comparison statistics.

diebold_mariano: tests equal predictive accuracy of two forecasts of the same
targets (Diebold & Mariano 1995) with the Harvey, Leybourne & Newbold (1997)
small-sample correction.

  Loss per forecast origin t: the mean over the h steps of the window of
  L(e) = e² ("squared") or |e| ("absolute"), so d_t = L_model,t − L_ref,t.
  Consecutive h-step windows share h − 1 targets, so d_t is autocorrelated up
  to lag h − 1. Its long-run variance is estimated with the Newey–West
  (Bartlett) estimator with lag h − 1 (weights 1 − k/h); if that estimate is
  not positive, the lag-0 variance is used and flagged.
      DM   = mean(d) / sqrt(S / n)
      DM*  = DM · sqrt((n + 1 − 2h + h(h − 1)/n) / n)
      p    = 2 · P(T_{n−1} > |DM*|)   (two-sided, Student t with n − 1 df)
  DM* < 0 means the first forecast has the lower loss.

  With ``mask`` (n, h) of booleans, only the masked steps count: the loss per
  origin is the mean over its masked steps, origins with no masked step are
  dropped, and the remaining origins keep their time order. Two kept origins
  whose windows overlap are at most h − 1 apart in the kept sequence too, so
  lag h − 1 still covers the overlap.

mase_vs_persistence: MAE of a forecast divided by the MAE of persistence on
the same windows (< 1 = better than persistence).
"""

import numpy as np
from scipy import stats


def _as_2d(a):
    a = np.asarray(a, np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def diebold_mariano(y, f_model, f_ref, h: int, loss: str = "squared", mask=None) -> dict:
    y, f1, f2 = _as_2d(y), _as_2d(f_model), _as_2d(f_ref)
    e1, e2 = y - f1, y - f2
    if loss == "squared":
        l1, l2 = e1 ** 2, e2 ** 2
    elif loss == "absolute":
        l1, l2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError(loss)
    if mask is None:
        d = l1.mean(axis=1) - l2.mean(axis=1)
    else:
        m = _as_2d(mask).astype(bool)
        keep = m.any(axis=1)
        cnt = m[keep].sum(axis=1)
        d = (np.where(m, l1, 0)[keep].sum(axis=1) - np.where(m, l2, 0)[keep].sum(axis=1)) / cnt
    n = len(d)
    dc = d - d.mean()
    gamma = [dc @ dc / n] + [dc[k:] @ dc[:-k] / n for k in range(1, h)]
    s = gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))
    fallback = False
    if s <= 0:
        s, fallback = gamma[0], True
    if s == 0:
        return {"dm": float("nan"), "dm_hln": float("nan"), "p_value": float("nan"), "n": n,
                "mean_loss_diff": float(d.mean()), "loss": loss, "hac_lag": h - 1, "variance_fallback": fallback}
    dm = d.mean() / np.sqrt(s / n)
    dm_hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm_hln), df=n - 1)
    return {"dm": float(dm), "dm_hln": float(dm_hln), "p_value": float(p), "n": n,
            "mean_loss_diff": float(d.mean()), "loss": loss, "hac_lag": h - 1, "variance_fallback": fallback}


def mase_vs_persistence(y, f, f_persistence, mask=None) -> float:
    y, f, fp = _as_2d(y), _as_2d(f), _as_2d(f_persistence)
    m = np.ones(y.shape, bool) if mask is None else _as_2d(mask).astype(bool)
    return float(np.abs(y - f)[m].mean() / np.abs(y - fp)[m].mean())


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
