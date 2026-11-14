"""
Walk-Forward Validation Framework.

Implements an expanding-window walk-forward validator that:
  1. Starts with an initial training window (70% of data by default)
  2. Trains (or re-uses) a model on the current window
  3. Predicts the next `step_size` days
  4. Expands the window by `step_size` and repeats
  5. Collects out-of-sample predictions and metrics across all folds

It complements the single chronological train/val/test split used in train.py.
Note: v1 code, not used by protocol v2. The v2 rolling-origin evaluation (Phase 4.1,
five folds, one seed) uses src/data/folds.py.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardFold:
    """Result of a single walk-forward fold."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    pred_start: pd.Timestamp
    pred_end: pd.Timestamp
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated result across all walk-forward folds."""
    model_name: str
    track: str
    horizon: int
    folds: list[WalkForwardFold]

    @property
    def all_y_true(self) -> np.ndarray:
        return np.concatenate([f.y_true for f in self.folds])

    @property
    def all_y_pred(self) -> np.ndarray:
        return np.concatenate([f.y_pred for f in self.folds])

    @property
    def aggregate_metrics(self) -> dict:
        return compute_all_metrics(self.all_y_true, self.all_y_pred)

    def per_fold_metrics(self) -> pd.DataFrame:
        rows = []
        for fold in self.folds:
            row = {
                "fold": fold.fold_idx,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "pred_start": fold.pred_start,
                "pred_end": fold.pred_end,
            }
            row.update(fold.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        agg = self.aggregate_metrics
        return (
            f"WalkForward [{self.model_name} | {self.track} | h={self.horizon}] "
            f"n_folds={len(self.folds)} | "
            f"MAE={agg['mae']:.4f} | RMSE={agg['rmse']:.4f} | "
            f"MAPE={agg['mape']:.2f}% | DirAcc={agg['directional_accuracy']:.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validator
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Expanding-window walk-forward validator.

    Usage:
        validator = WalkForwardValidator(cfg)
        result = validator.run(
            df=merged_df,
            feature_cols=feature_cols,
            target_col="price_sdg",
            build_model_fn=lambda: build_lstm(cfg),
            train_fn=lambda model, X, y: model.fit(X, y, ...),
            predict_fn=lambda model, X: model.predict(X),
            model_name="lstm",
            track="sudan",
            horizon=7,
        )
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        wf = cfg["walk_forward"]
        self.initial_train_ratio = wf["initial_train_ratio"]
        self.step_size = wf["step_size"]
        self.min_train_size = wf["min_train_size"]
        self.input_length = cfg["sequence"]["input_length"]

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        build_model_fn: Callable,
        train_fn: Callable,
        predict_fn: Callable,
        model_name: str = "model",
        track: str = "global",
        horizon: int = 1,
        refit_every_fold: bool = True,
        feature_scaler=None,
        target_scaler=None,
    ) -> WalkForwardResult:
        """
        Run expanding-window walk-forward validation.

        Args:
            df:                 Full merged DataFrame (all splits)
            feature_cols:       Columns to use as model input
            target_col:         Column to predict
            build_model_fn:     Callable() → new untrained model instance
            train_fn:           Callable(model, X_train, y_train) → trained model
            predict_fn:         Callable(model, X_pred) → np.ndarray of predictions
            model_name:         Name for logging and result labelling
            track:              "global" or "sudan"
            horizon:            Prediction horizon (days)
            refit_every_fold:   If True, rebuild and retrain from scratch each fold.
                                If False, continue training from previous weights (faster).
            feature_scaler:     Pre-fitted scaler for features (optional; fitted fresh if None)
            target_scaler:      Pre-fitted scaler for target (optional)
        """
        from src.data.processors import build_sequences, apply_scaling, fit_scalers

        n = len(df)
        initial_train_end = int(n * self.initial_train_ratio)
        folds = []
        fold_idx = 0
        model = None

        cursor = initial_train_end

        while cursor + horizon <= n:
            train_df = df.iloc[:cursor]
            pred_end = min(cursor + self.step_size, n)
            pred_df = df.iloc[cursor - self.input_length : pred_end]

            if len(train_df) < self.min_train_size:
                logger.warning(f"Fold {fold_idx}: training window too small ({len(train_df)}), skipping")
                cursor += self.step_size
                continue

            logger.info(
                f"Fold {fold_idx}: train={len(train_df)} rows "
                f"({train_df.index.min().date()} – {train_df.index.max().date()}), "
                f"predicting {cursor} – {pred_end}"
            )

            # Fit scalers on this fold's training data if not provided externally
            if feature_scaler is None:
                f_scaler, t_scaler = fit_scalers(
                    train_df, feature_cols, target_col, track=f"{track}_fold{fold_idx}"
                )
            else:
                f_scaler, t_scaler = feature_scaler, target_scaler

            # Scale
            train_scaled = apply_scaling(train_df, feature_cols, target_col, f_scaler, t_scaler)
            pred_scaled = apply_scaling(pred_df, feature_cols, target_col, f_scaler, t_scaler)

            # Build sequences
            X_train, y_train = build_sequences(
                train_scaled, feature_cols, target_col, self.input_length, horizon
            )
            X_pred, y_true_scaled = build_sequences(
                pred_scaled, feature_cols, target_col, self.input_length, horizon
            )

            if len(X_train) == 0 or len(X_pred) == 0:
                cursor += self.step_size
                fold_idx += 1
                continue

            # Build / refit model
            if refit_every_fold or model is None:
                model = build_model_fn()

            model = train_fn(model, X_train, y_train)

            # Predict
            y_pred_scaled = predict_fn(model, X_pred)

            # Inverse-scale predictions and true values
            if t_scaler is not None:
                y_true = self._inverse_scale(y_true_scaled, t_scaler)
                y_pred = self._inverse_scale(y_pred_scaled, t_scaler)
            else:
                y_true = y_true_scaled
                y_pred = y_pred_scaled

            fold_metrics = compute_all_metrics(y_true.flatten(), y_pred.flatten())

            fold = WalkForwardFold(
                fold_idx=fold_idx,
                train_start=train_df.index.min(),
                train_end=train_df.index.max(),
                pred_start=df.index[cursor],
                pred_end=df.index[pred_end - 1],
                y_true=y_true.flatten(),
                y_pred=y_pred.flatten(),
                metrics=fold_metrics,
            )
            folds.append(fold)
            logger.info(f"Fold {fold_idx} metrics: {fold_metrics}")

            cursor += self.step_size
            fold_idx += 1

        result = WalkForwardResult(
            model_name=model_name,
            track=track,
            horizon=horizon,
            folds=folds,
        )
        logger.info(result.summary())
        return result

    @staticmethod
    def _inverse_scale(y_scaled: np.ndarray, scaler) -> np.ndarray:
        """Inverse-transform predictions back to original price units."""
        original_shape = y_scaled.shape
        flat = y_scaled.reshape(-1, 1)
        unscaled = scaler.inverse_transform(flat)
        return unscaled.reshape(original_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward result persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_walk_forward_result(result: WalkForwardResult, path: str) -> None:
    """Save walk-forward result to CSV."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = []
    for fold in result.folds:
        for i, (yt, yp) in enumerate(zip(fold.y_true, fold.y_pred)):
            rows.append({
                "fold": fold.fold_idx,
                "pred_start": fold.pred_start,
                "pred_end": fold.pred_end,
                "step": i,
                "y_true": yt,
                "y_pred": yp,
                "model": result.model_name,
                "track": result.track,
                "horizon": result.horizon,
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"Saved walk-forward results to {path}")


def load_walk_forward_result(path: str) -> pd.DataFrame:
    """Load a CSV written by save_walk_forward_result."""
    return pd.read_csv(path)
