"""
Ensemble model — weighted average of top-N trained models.

Weights are optimised on the validation set by minimising MAE.
Supports both TF/Keras models and the Chronos wrapper.

Note: train.py does not use this class for the reported metrics. It applies
the same method (top-N by validation MAE, SLSQP weights) directly to stored
prediction arrays; see the "Ensemble" block in train.train_track.
"""

import logging

import numpy as np
from scipy.optimize import minimize

from src.evaluation.metrics import compute_mae

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Weight optimisation
# ─────────────────────────────────────────────────────────────────────────────

def optimise_weights(
    models: list,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_scaler=None,
) -> np.ndarray:
    """
    Find ensemble weights that minimise validation MAE.

    Uses scipy SLSQP: constrained optimisation with weights ∈ [0,1] and sum=1.
    Starts from equal weights and descends toward the optimal combination.

    Args:
        models:         list of fitted model objects with a .predict() method
        X_val:          validation sequences (batch, seq_len, n_features)
        y_val:          validation targets (batch, horizon)
        target_scaler:  if provided, inverse-scale before computing MAE

    Returns:
        np.ndarray of shape (n_models,) — normalised optimal weights
    """
    n = len(models)

    # Get predictions from each model upfront
    all_preds = []
    for i, model in enumerate(models):
        pred = _get_predictions(model, X_val)
        all_preds.append(pred)
        logger.debug(f"Model {i} val pred shape: {pred.shape}")

    all_preds = np.stack(all_preds, axis=0)  # (n_models, batch, horizon)

    def objective(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()
        ensemble = (all_preds * weights[:, None, None]).sum(axis=0)

        y_true = y_val
        y_pred = ensemble

        if target_scaler is not None:
            shape = y_pred.shape
            y_pred = target_scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).reshape(shape)
            y_true = target_scaler.inverse_transform(
                y_val.reshape(-1, 1)
            ).reshape(y_val.shape)

        return compute_mae(y_true.flatten(), y_pred.flatten())

    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

    result = minimize(
        objective, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    weights = np.array(result.x)
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()

    logger.info(f"Optimised ensemble weights: {dict(zip(range(n), weights.round(4)))}")
    logger.info(f"Ensemble val MAE: {result.fun:.4f}")
    return weights


def _get_predictions(model, X: np.ndarray) -> np.ndarray:
    """Dispatch prediction to the right interface."""
    from src.models.chronos_wrapper import ChronosWrapper, ChronosPyfunc

    if isinstance(model, (ChronosWrapper, ChronosPyfunc)):
        return model.predict(X)
    # TF/Keras model
    return model.predict(X, verbose=0)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble model class
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Weighted ensemble of gold price forecasting models.

    Supports any combination of LSTM, GRU, TSMixer, PatchTST, TFT and Chronos.

    Usage:
        ensemble = EnsembleModel(models=[lstm, gru, tsmixer], model_names=[...])
        ensemble.fit_weights(X_val, y_val, target_scaler)
        y_pred = ensemble.predict(X_test)
    """

    def __init__(
        self,
        models: list,
        model_names: list[str],
        weights: np.ndarray = None,
    ):
        assert len(models) == len(model_names), "models and model_names must have same length"
        self.models = models
        self.model_names = model_names
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)

    def fit_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_scaler=None,
    ) -> np.ndarray:
        """Optimise ensemble weights on validation set. Returns weights."""
        self.weights = optimise_weights(
            self.models, X_val, y_val, target_scaler
        )
        return self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted average prediction.

        Returns np.ndarray of shape (batch, horizon).
        """
        all_preds = np.stack(
            [_get_predictions(m, X) for m in self.models], axis=0
        )  # (n_models, batch, horizon)

        ensemble = (all_preds * self.weights[:, None, None]).sum(axis=0)
        return ensemble

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (weighted mean, weighted std) across model predictions.
        Std captures model disagreement — a form of epistemic uncertainty.
        """
        all_preds = np.stack(
            [_get_predictions(m, X) for m in self.models], axis=0
        )  # (n_models, batch, horizon)

        mean = (all_preds * self.weights[:, None, None]).sum(axis=0)
        variance = (
            self.weights[:, None, None] * (all_preds - mean[None, :, :]) ** 2
        ).sum(axis=0)
        std = np.sqrt(variance)
        return mean, std

    def summary(self) -> str:
        lines = ["Ensemble composition:"]
        for name, w in zip(self.model_names, self.weights):
            lines.append(f"  {name:12s} → weight={w:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise weights for MLflow logging."""
        return {
            "model_names": self.model_names,
            "weights": self.weights.tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Top-N selection helper
# ─────────────────────────────────────────────────────────────────────────────

def select_top_n_models(
    val_metrics: dict[str, dict],
    models_dict: dict,
    n: int = 2,
    metric: str = "mae",
) -> tuple[list, list[str]]:
    """
    Select the top-N models by validation metric.

    Args:
        val_metrics:  {model_name: {"mae": ..., "rmse": ..., ...}}
        models_dict:  {model_name: fitted model object}
        n:            number of top models to select
        metric:       metric to rank by (lower is better for mae/rmse/mape)

    Returns:
        (models_list, model_names_list) — ordered best to worst
    """
    ranked = sorted(val_metrics.items(), key=lambda x: x[1].get(metric, float("inf")))
    top = ranked[:n]

    selected_names = [name for name, _ in top]
    selected_models = [models_dict[name] for name in selected_names]

    logger.info(
        f"Top-{n} models by {metric.upper()}: "
        + ", ".join(f"{n}={v[metric]:.4f}" for n, v in top)
    )
    return selected_models, selected_names


# ─────────────────────────────────────────────────────────────────────────────
# Validation-only ensemble on stored predictions (protocol v2, used by train.py)
# ─────────────────────────────────────────────────────────────────────────────

def fit_validation_ensemble(
    val_preds: dict[str, np.ndarray],
    y_val: np.ndarray,
    top_n: int = 2,
) -> dict:
    """
    Choose ensemble members and weights from validation predictions only.

    1. Rank candidates by validation MAE and keep the best ``top_n``.
    2. Fit non-negative weights summing to 1 that minimise validation MAE
       (SLSQP from equal weights, the same optimiser settings as v1).

    No test data enters this function. (In v1 the members were ranked by
    *test* MAE whenever train.py ran with --resume; see the version audit.)

    Args:
        val_preds: {model_name: (n_val, horizon) predictions in price units}
        y_val:     (n_val, horizon) true values in price units

    Returns a JSON-serialisable dict: ``candidates_val_mae``, ``selected``,
    ``weights`` (aligned with ``selected``), ``val_mae``, ``optimiser``.
    """
    if len(val_preds) < 2:
        raise ValueError("Need at least two candidate models")
    val_mae = {name: compute_mae(y_val, p) for name, p in val_preds.items()}
    ranked = sorted(val_mae, key=val_mae.get)
    selected = ranked[:top_n]
    stack = np.stack([val_preds[n] for n in selected], axis=0)

    def objective(w):
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-12)
        return compute_mae(y_val, (stack * w[:, None, None]).sum(axis=0))

    k = len(selected)
    res = minimize(
        objective, np.ones(k) / k, method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    weights = np.clip(res.x, 0, 1)
    weights = weights / weights.sum()
    return {
        "selection_rule": f"top-{top_n} by validation MAE",
        "candidates_val_mae": {n: val_mae[n] for n in ranked},
        "selected": selected,
        "weights": [float(w) for w in weights],
        "val_mae": float(objective(weights)),
        "optimiser": {"method": "SLSQP", "success": bool(res.success), "message": str(res.message)},
    }


def apply_ensemble(preds: dict[str, np.ndarray], ensemble: dict) -> np.ndarray:
    """Weighted average of ``preds`` using the members and weights from fit_validation_ensemble."""
    return sum(w * preds[n] for n, w in zip(ensemble["selected"], ensemble["weights"]))
