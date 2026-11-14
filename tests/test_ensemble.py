"""The ensemble is chosen and weighted on validation predictions only."""

import inspect

import numpy as np

from src.models.ensemble import apply_ensemble, fit_validation_ensemble


def test_selection_follows_validation_not_test():
    y_val = np.zeros((20, 1))
    val_preds = {"a": np.full((20, 1), 1.0), "b": np.full((20, 1), 2.0), "c": np.full((20, 1), 5.0)}
    ens = fit_validation_ensemble(val_preds, y_val, top_n=2)
    assert ens["selected"] == ["a", "b"]
    assert np.isclose(sum(ens["weights"]), 1.0) and all(w >= 0 for w in ens["weights"])
    # The function has no access to test data at all.
    assert set(inspect.signature(fit_validation_ensemble).parameters) == {"val_preds", "y_val", "top_n"}


def test_weights_fit_on_validation():
    y_val = np.zeros((10, 1))
    val_preds = {"a": np.full((10, 1), 1.0), "b": np.full((10, 1), -1.0)}
    ens = fit_validation_ensemble(val_preds, y_val, top_n=2)
    assert np.allclose(ens["weights"], [0.5, 0.5], atol=1e-4)
    assert np.allclose(apply_ensemble(val_preds, ens), 0.0, atol=1e-3)
