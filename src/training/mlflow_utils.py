"""
MLflow utilities for model registration and tracking.

Handles:
  - Experiment setup
  - Logging params, metrics, and artifacts
  - Registering TF/Keras models as mlflow.tensorflow artifacts
  - Wrapping models (including Chronos) as mlflow.pyfunc with
    preprocessing baked in — so the serving platform receives raw
    JSON arrays and gets back predictions in original price units

All models are registered under the naming convention:
  gold-{model_name}-{track}
  e.g. gold-lstm-global, gold-patchtst-sudan, gold-chronos-global
"""

import logging
import os
import pickle
import tempfile
from typing import Optional

import mlflow
import mlflow.tensorflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yaml

from src.utils.provenance import REPO_ROOT, provenance

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_mlflow(cfg: dict) -> None:
    """
    Configure the MLflow tracking URI from config.

    A URI without a scheme (e.g. "mlruns") is resolved against the repository
    root, so the store works from any checkout location and any working directory.
    """
    uri = cfg["mlflow"]["tracking_uri"]
    if "://" not in uri:
        uri = str((REPO_ROOT / uri).resolve())
    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI: {uri}")


def get_or_create_experiment(cfg: dict, track: str) -> str:
    """
    Return the MLflow experiment name for a given track.

    ``mlflow.experiment_override`` (set by ``train.py --smoke``) routes every run
    to a single named experiment so test runs never land in the reported ones.
    """
    override = cfg["mlflow"].get("experiment_override")
    if override:
        return override
    key = f"experiment_{track}"
    return cfg["mlflow"].get(key, f"gold-{track}")


def log_provenance(cfg: dict) -> None:
    """Tag the active run with git commit / dirty flag / protocol and log the full config."""
    mlflow.set_tags({k: str(v) for k, v in provenance(cfg).items()})
    mlflow.log_dict(cfg, "config_resolved.json")


def model_registry_name(cfg: dict, model_name: str, track: str) -> str:
    """Return the registered model name following the naming convention."""
    prefix = cfg["mlflow"]["registered_model_prefix"]
    return f"{prefix}-{model_name}-{track}"


# ─────────────────────────────────────────────────────────────────────────────
# Log training run
# ─────────────────────────────────────────────────────────────────────────────

def log_training_run(
    cfg: dict,
    model_name: str,
    track: str,
    horizon: int,
    model_params: dict,
    history,           # keras.callbacks.History
    eval_metrics: dict,
    extra_params: Optional[dict] = None,
) -> None:
    """
    Log hyperparameters, training curves, and evaluation metrics to MLflow.
    Called from within an active mlflow.start_run() context.

    ``eval_metrics`` keys are logged as given, so callers prefix them with
    the split (``val_mae``, ``test_mae``, ...). ``extra_params`` (e.g. seed,
    effective batch size, parameter count) override the config values.
    """
    # Params
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("track", track)
    mlflow.log_param("horizon", horizon)
    mlflow.log_param("sequence_length", cfg["sequence"]["input_length"])
    mlflow.log_param("batch_size", cfg["training"]["batch_size"])
    mlflow.log_param("learning_rate", cfg["training"]["learning_rate"])
    mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})
    if extra_params:
        mlflow.log_params(extra_params)

    # Training curves
    hist_dict = history.history if hasattr(history, "history") else history
    for epoch, (loss, val_loss) in enumerate(
        zip(hist_dict.get("loss", []), hist_dict.get("val_loss", []))
    ):
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    for epoch, (mae, val_mae) in enumerate(
        zip(hist_dict.get("mae", []), hist_dict.get("val_mae", []))
    ):
        mlflow.log_metric("train_mae", mae, step=epoch)
        mlflow.log_metric("val_mae", val_mae, step=epoch)

    # Final evaluation metrics (keys already carry their split prefix)
    for metric_name, value in eval_metrics.items():
        mlflow.log_metric(metric_name, value)

    logger.info(f"Logged training run: {model_name}/{track}/h{horizon}")


# ─────────────────────────────────────────────────────────────────────────────
# TF/Keras model registration
# ─────────────────────────────────────────────────────────────────────────────

def register_keras_model(
    model,                         # keras.Model
    cfg: dict,
    model_name: str,
    track: str,
    horizon: int,
    feature_scaler,
    target_scaler,
    feature_cols: list[str],
    model_params: dict,
    history,
    eval_metrics: dict,
    extra_params: Optional[dict] = None,
    extra_tags: Optional[dict] = None,
    register: bool = True,
) -> str:
    """
    Register a trained Keras model to MLflow with preprocessing baked in.

    The model is wrapped in a GoldPricePyfunc that handles:
      - Input: flat list of (sequence_length × n_features) floats
      - Reshaping to (1, sequence_length, n_features)
      - Feature scaling
      - Model inference
      - Target inverse-scaling → original price units

    Returns the MLflow run ID.
    """
    setup_mlflow(cfg)
    experiment = get_or_create_experiment(cfg, track)
    registry_name = model_registry_name(cfg, model_name, track)

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{model_name}_{track}_h{horizon}") as run:
        log_provenance(cfg)
        if extra_tags:
            mlflow.set_tags({k: str(v) for k, v in extra_tags.items()})
        log_training_run(cfg, model_name, track, horizon, model_params, history, eval_metrics, extra_params)
        if not register:
            mlflow.set_tag("registered_model", "none")
            logger.info(f"Logged run without model registration (run_id={run.info.run_id})")
            return run.info.run_id

        # Save scalers and metadata to a temp dir, bundle with model
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = _save_artifacts(
                tmp_dir, model, feature_scaler, target_scaler,
                feature_cols, cfg["sequence"]["input_length"], horizon
            )

            pyfunc_model = GoldPricePyfunc(
                model=model,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                feature_cols=feature_cols,
                input_length=cfg["sequence"]["input_length"],
                horizon=horizon,
            )

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pyfunc_model,
                artifacts=artifacts,
                registered_model_name=registry_name,
                pip_requirements=[
                    "tensorflow==2.16.1",
                    "numpy==1.26.4",
                    "scikit-learn==1.5.0",
                ],
            )

        logger.info(f"Registered model: {registry_name} (run_id={run.info.run_id})")
        return run.info.run_id


def _save_artifacts(
    tmp_dir: str,
    model,
    feature_scaler,
    target_scaler,
    feature_cols: list[str],
    input_length: int,
    horizon: int,
) -> dict:
    """Save model artifacts to tmp_dir and return artifact paths dict."""
    # Save Keras model
    model_path = os.path.join(tmp_dir, "keras_model.keras")
    model.save(model_path)

    # Save scalers
    fs_path = os.path.join(tmp_dir, "feature_scaler.pkl")
    ts_path = os.path.join(tmp_dir, "target_scaler.pkl")
    with open(fs_path, "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(ts_path, "wb") as f:
        pickle.dump(target_scaler, f)

    # Save metadata
    meta_path = os.path.join(tmp_dir, "meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump({
            "feature_cols": feature_cols,
            "input_length": input_length,
            "horizon": horizon,
        }, f)

    return {
        "keras_model": model_path,
        "feature_scaler": fs_path,
        "target_scaler": ts_path,
        "meta": meta_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# mlflow.pyfunc wrapper — makes the model serving-platform compatible
# ─────────────────────────────────────────────────────────────────────────────

class GoldPricePyfunc(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for gold price forecasting models.

    Bakes preprocessing (scaling, reshaping) into the model artifact so
    the serving platform can pass raw price arrays without any extra logic.

    Input format expected by predict():
        pd.DataFrame with one column "inputs" — each cell is a flat list of
        (input_length × n_features) floats, OR:
        dict {"inputs": [[...]], "horizon": 7}

    Output:
        pd.DataFrame with column "predictions" (list of floats per row)
    """

    def __init__(
        self,
        model=None,
        feature_scaler=None,
        target_scaler=None,
        feature_cols: Optional[list] = None,
        input_length: int = 60,
        horizon: int = 1,
    ):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.feature_cols = feature_cols or []
        self.input_length = input_length
        self.horizon = horizon

    def load_context(self, context):
        """Called by MLflow at load time. Loads model + scalers from artifacts."""
        import tensorflow as tf

        self.model = tf.keras.models.load_model(context.artifacts["keras_model"])

        with open(context.artifacts["feature_scaler"], "rb") as f:
            self.feature_scaler = pickle.load(f)
        with open(context.artifacts["target_scaler"], "rb") as f:
            self.target_scaler = pickle.load(f)

        with open(context.artifacts["meta"], "r") as f:
            meta = yaml.safe_load(f)
        self.feature_cols = meta["feature_cols"]
        self.input_length = meta["input_length"]
        self.horizon = meta["horizon"]

    def predict(self, context, model_input):
        """
        Run inference. model_input can be:
          - pd.DataFrame with column "inputs" (list of flat float arrays)
          - dict with key "inputs"
        """
        if isinstance(model_input, pd.DataFrame):
            raw_inputs = model_input["inputs"].tolist()
        elif isinstance(model_input, dict):
            raw_inputs = model_input["inputs"]
        else:
            raw_inputs = model_input

        results = []
        n_features = len(self.feature_cols)

        for flat_input in raw_inputs:
            arr = np.array(flat_input, dtype=np.float32)
            arr = arr.reshape(self.input_length, n_features)

            # Scale
            arr_scaled = self.feature_scaler.transform(arr)
            X = arr_scaled.reshape(1, self.input_length, n_features)

            # Predict
            y_scaled = self.model.predict(X, verbose=0)

            # Inverse-scale
            y = self.target_scaler.inverse_transform(
                y_scaled.reshape(-1, 1)
            ).flatten().tolist()
            results.append(y)

        return pd.DataFrame({"predictions": results})


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble registration
# ─────────────────────────────────────────────────────────────────────────────

class EnsemblePyfunc(mlflow.pyfunc.PythonModel):
    """
    Weighted ensemble of GoldPricePyfunc models.
    Weights are optimized on the validation set.
    """

    def __init__(self, models: list, weights: list[float]):
        self.models = models     # list of loaded GoldPricePyfunc instances
        self.weights = np.array(weights) / np.sum(weights)  # normalize

    def predict(self, context, model_input):
        all_preds = []
        for model in self.models:
            preds = model.predict(context, model_input)["predictions"].tolist()
            all_preds.append(preds)

        # Weighted average across models
        # all_preds: (n_models, n_samples, horizon)
        weighted = sum(
            w * np.array(p) for w, p in zip(self.weights, all_preds)
        )
        return pd.DataFrame({"predictions": weighted.tolist()})


def optimize_ensemble_weights(
    models: list,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_scaler=None,
) -> np.ndarray:
    """
    Grid search over ensemble weights to minimize validation MAE.
    Tests all combinations at 0.1 granularity for 2 models.
    Returns normalized weights array.
    """
    from src.evaluation.metrics import compute_mae
    from scipy.optimize import minimize

    n_models = len(models)

    def objective(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()
        preds_all = np.stack([m.predict(X_val, verbose=0) for m in models], axis=0)
        ensemble_pred = (preds_all * weights[:, None, None]).sum(axis=0)

        if target_scaler is not None:
            shape = ensemble_pred.shape
            ensemble_pred = target_scaler.inverse_transform(
                ensemble_pred.reshape(-1, 1)
            ).reshape(shape)
            y_true = target_scaler.inverse_transform(
                y_val.reshape(-1, 1)
            ).reshape(y_val.shape)
        else:
            y_true = y_val

        return compute_mae(y_true.flatten(), ensemble_pred.flatten())

    x0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1}

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
    optimal_weights = result.x / result.x.sum()

    logger.info(f"Optimal ensemble weights: {optimal_weights}")
    return optimal_weights
