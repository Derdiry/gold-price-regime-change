"""
Unified training loop for all TF/Keras models.

Handles:
  - Standard train/val fit with early stopping
  - Sudan fine-tuning (transfer learning from global weights)
  - MC Dropout inference for uncertainty quantification
  - Consistent callback setup (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
  - MLflow run logging (params, metrics, model artifact)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from tensorflow import keras

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks(
    cfg: dict,
    checkpoint_path: Optional[str] = None,
    model_name: str = "model",
    save_weights_only: bool = False,
) -> list:
    """Build standard training callbacks from config."""
    train_cfg = cfg["training"]

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=train_cfg["early_stopping_monitor"],
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=train_cfg["early_stopping_monitor"],
            factor=0.5,
            patience=train_cfg["early_stopping_patience"] // 2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=train_cfg["early_stopping_monitor"],
                save_best_only=True,
                save_weights_only=save_weights_only,
                verbose=0,
            )
        )

    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Core trainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Unified trainer for TF/Keras time series models.

    Example:
        trainer = ModelTrainer(cfg)
        history = trainer.fit(model, X_train, y_train, X_val, y_val, model_name="lstm")
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.train_cfg = cfg["training"]

    def compile(self, model: keras.Model, lr: Optional[float] = None) -> keras.Model:
        """Compile model with optimizer and loss from config."""
        lr = lr or self.train_cfg["learning_rate"]
        optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=self.train_cfg["loss"],
            metrics=self.train_cfg["metrics"],
        )
        return model

    def fit(
        self,
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str = "model",
        checkpoint_dir: str = "logs/checkpoints",
        weights_only: bool = False,
    ) -> keras.callbacks.History:
        """
        Train a compiled model with early stopping and LR scheduling.

        weights_only: save the best checkpoint as ``<model_name>_best.weights.h5``
        for every model (protocol v2). Otherwise only PatchTST, a keras.Model
        subclass without get_config, uses the weights-only format (v1).

        Returns the Keras History object.
        """
        weights_only = weights_only or model_name.startswith("patchtst")
        ext = ".weights.h5" if weights_only else ".keras"
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best{ext}")
        self.last_checkpoint_path = checkpoint_path
        callbacks = build_callbacks(self.cfg, checkpoint_path, model_name, save_weights_only=weights_only)

        logger.info(
            f"Training {model_name}: "
            f"X_train={X_train.shape}, X_val={X_val.shape}, "
            f"batch={self.train_cfg['batch_size']}, max_epochs={self.train_cfg['max_epochs']}"
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.train_cfg["max_epochs"],
            batch_size=self.train_cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        logger.info(
            f"{model_name} training complete — "
            f"best val_mae={min(history.history.get('val_mae', [np.inf])):.4f}"
        )
        return history

    def fit_and_compile(
        self,
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str = "model",
        checkpoint_dir: str = "logs/checkpoints",
        weights_only: bool = False,
    ) -> tuple[keras.Model, keras.callbacks.History]:
        """Compile then fit. Returns (model, history)."""
        model = self.compile(model)
        history = self.fit(model, X_train, y_train, X_val, y_val, model_name, checkpoint_dir, weights_only)
        return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Transfer learning / Sudan fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune_for_sudan(
    global_model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    model_name: str = "model",
) -> tuple[keras.Model, keras.callbacks.History]:
    """
    Fine-tune a globally pre-trained model on Sudan data.

    Strategy:
      1. Copy global model weights
      2. Freeze first N layers (preserve general temporal patterns)
      3. Unfreeze remaining layers + output head (learn Sudan-specific dynamics)
      4. Train with lower learning rate and fewer epochs

    Rationale: the global model may learn generic gold-price dynamics that the
    Sudan model can reuse. train.py only calls this when the global and Sudan
    input shapes match; with the default feature sets (29 global vs 33 Sudan
    features) they do not, so the reported Sudan models were trained from scratch.
    """
    ft_cfg = cfg["training"]["fine_tune"]
    n_freeze = ft_cfg["freeze_layers"]

    # Clone weights into a new model instance (don't modify global model)
    model = keras.models.clone_model(global_model)
    model.set_weights(global_model.get_weights())

    # Freeze first N layers
    for i, layer in enumerate(model.layers[:n_freeze]):
        layer.trainable = False
        logger.debug(f"Frozen layer {i}: {layer.name}")

    trainable_params = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )
    logger.info(
        f"Fine-tuning {model_name} for Sudan: "
        f"frozen_layers={n_freeze}, trainable_params={trainable_params:,}"
    )

    # Recompile with lower LR
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ft_cfg["learning_rate"]),
        loss=cfg["training"]["loss"],
        metrics=cfg["training"]["metrics"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=ft_cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=ft_cfg["max_epochs"],
        batch_size=cfg["training"]["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(
        f"Sudan fine-tune complete — "
        f"best val_mae={min(history.history.get('val_mae', [np.inf])):.4f}"
    )
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout inference (uncertainty quantification)
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_uncertainty(
    model: keras.Model,
    X: np.ndarray,
    n_samples: int = 100,
    target_scaler=None,
) -> dict:
    """
    MC Dropout inference: run N stochastic forward passes to estimate
    prediction uncertainty.

    Requires the model to have Dropout layers that stay active at inference.
    Uses tf.keras backend call with training=True to keep dropout active.

    Returns:
        {
            "mean":  np.ndarray — mean prediction
            "std":   np.ndarray — prediction standard deviation (uncertainty)
            "lower": np.ndarray — 5th percentile (90% interval)
            "upper": np.ndarray — 95th percentile (90% interval)
        }

    If target_scaler is provided, all outputs are inverse-transformed to
    original price units.
    """
    preds = []
    for _ in range(n_samples):
        # training=True keeps dropout active
        pred = model(X, training=True).numpy()
        preds.append(pred)

    preds = np.stack(preds, axis=0)  # (n_samples, batch, horizon)

    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)

    if target_scaler is not None:
        def inv(arr):
            shape = arr.shape
            return target_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(shape)
        mean = inv(mean)
        std = std * (target_scaler.scale_[0] if hasattr(target_scaler, "scale_") else 1.0)
        lower = inv(lower)
        upper = inv(upper)

    return {"mean": mean, "std": std, "lower": lower, "upper": upper}


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward compatible train/predict callables
# ─────────────────────────────────────────────────────────────────────────────

def make_walk_forward_fns(cfg: dict, model_name: str):
    """
    Returns (train_fn, predict_fn) callables for use with WalkForwardValidator.

    train_fn(model, X_train, y_train) → trained model
    predict_fn(model, X) → np.ndarray
    """
    trainer = ModelTrainer(cfg)

    def train_fn(model, X_train, y_train):
        # Use a dummy val split from the end of training data for callbacks
        n_val = max(1, int(len(X_train) * 0.1))
        X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
        X_vl, y_vl = X_train[-n_val:], y_train[-n_val:]

        if not model.optimizer:
            model = trainer.compile(model)

        model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=cfg["training"]["max_epochs"],
            batch_size=cfg["training"]["batch_size"],
            callbacks=build_callbacks(cfg, model_name=model_name),
            verbose=0,
        )
        return model

    def predict_fn(model, X):
        return model.predict(X, verbose=0)

    return train_fn, predict_fn
