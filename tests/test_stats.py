"""Diebold–Mariano with HLN correction and Newey–West variance."""

import numpy as np
from scipy import stats

from src.evaluation.stats import diebold_mariano, holm, mase_vs_persistence


def test_h1_equals_corrected_t_test():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    f1, f2 = y + rng.normal(0, 1.0, 200), y + rng.normal(0, 1.2, 200)
    r = diebold_mariano(y, f1, f2, h=1, loss="squared")
    d = (y - f1) ** 2 - (y - f2) ** 2
    dm = d.mean() / np.sqrt(d.var() / len(d))          # population variance, lag 0
    assert np.isclose(r["dm"], dm)
    assert np.isclose(r["dm_hln"], dm * np.sqrt((len(d) - 1) / len(d)))
    assert np.isclose(r["p_value"], 2 * stats.t.sf(abs(r["dm_hln"]), len(d) - 1))


def test_sign_and_symmetry():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(300, 7))
    good, bad = y + rng.normal(0, 0.5, y.shape), y + rng.normal(0, 2.0, y.shape)
    a = diebold_mariano(y, good, bad, h=7, loss="absolute")
    b = diebold_mariano(y, bad, good, h=7, loss="absolute")
    assert a["dm_hln"] < 0 and a["p_value"] < 0.01
    assert np.isclose(a["dm_hln"], -b["dm_hln"]) and np.isclose(a["p_value"], b["p_value"])


def test_identical_forecasts_give_nan_not_significance():
    y = np.arange(10.0)
    r = diebold_mariano(y, y + 1, y + 1, h=1)
    assert np.isnan(r["p_value"])


def test_mase_and_holm():
    y = np.array([1.0, 2.0, 3.0])
    assert mase_vs_persistence(y, y + 1, y + 2) == 0.5
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_mask_scores_only_masked_steps():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(120, 5))
    f1, f2 = y + rng.normal(0, 1, y.shape), y + rng.normal(0, 1, y.shape)
    mask = rng.random(y.shape) < 0.3
    mask[:10] = False                                    # origins with no masked step are dropped
    r = diebold_mariano(y, f1, f2, h=5, loss="absolute", mask=mask)
    keep = mask.any(1)
    d = np.array([np.abs(y - f1)[i][mask[i]].mean() - np.abs(y - f2)[i][mask[i]].mean() for i in np.where(keep)[0]])
    assert r["n"] == keep.sum() and np.isclose(r["mean_loss_diff"], d.mean())
    # unmasked values do not matter
    f1b = f1.copy(); f1b[~mask] += 1e6
    assert np.isclose(diebold_mariano(y, f1b, f2, h=5, loss="absolute", mask=mask)["dm_hln"], r["dm_hln"])
    # an all-true mask equals no mask
    assert np.isclose(diebold_mariano(y, f1, f2, h=5, mask=np.ones(y.shape, bool))["dm_hln"],
                      diebold_mariano(y, f1, f2, h=5)["dm_hln"])
    assert np.isclose(mase_vs_persistence(y, f1, f2, mask), np.abs(y - f1)[mask].mean() / np.abs(y - f2)[mask].mean())
