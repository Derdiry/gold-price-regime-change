"""
Train/val/test windows for one (track, horizon) under protocol v2.

Steps (shared by train.py, the baselines and the split report):
  1. Chronological 70/15/15 split of the daily rows. The row boundaries are
     the same as in v1, so the test period is unchanged.
  2. Scalers fitted on the training rows only.
  3. Row validity (processors.row_validity): rows inside the long Sudan gap,
     or with undefined features, may not appear in any window.
  4. Sliding windows; a window is kept only if all its context and target
     rows are valid. Val/test windows take their context from the end of the
     previous split, as in v1.
  5. Condition (experiment.condition, src/data/transforms.py):
       minmax      v1 scalers fitted on the training rows (steps 2-4 as is)
       log_return, revin
                   windows built on unscaled data, then a per-window
                   pre-transform and MinMax scalers fitted on the training
                   windows (transforms.WindowScaler)
     Windows are identical across conditions; only their values differ.
"""

import logging

import numpy as np
import pandas as pd

from src.data.folds import fold_frames
from src.data.features import get_available_feature_cols
from src.data.processors import (
    VALID_COL,
    apply_scaling,
    build_all_splits,
    chronological_split,
    fit_scalers,
    row_validity,
    save_sequences,
)
from src.data.transforms import CONDITIONS, WindowScaler

logger = logging.getLogger(__name__)


def make_dataset(
    df: pd.DataFrame,
    cfg: dict,
    track: str,
    horizon: int,
    feature_col_list: list,
    target_col: str,
    save: bool = True,
    condition: str | None = None,
) -> tuple[dict, tuple, list, dict]:
    """
    Return (splits, scalers, feature_cols, split_info).

    scalers is (feature_scaler, target_scaler) for minmax and
    (window_scaler, window_scaler) otherwise; use ``to_price`` to map model
    outputs back to price units. splits also holds, per split s:
    yraw_s (price-unit targets), ctxt_s (target context), last_s (P_t).
    """
    condition = condition or cfg.get("experiment", {}).get("condition", "minmax")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}; expected one of {CONDITIONS}")
    feature_cols = get_available_feature_cols(df, feature_col_list)
    logger.info(f"[{track}] Using {len(feature_cols)} features")

    df = df.copy()
    df[VALID_COL] = row_validity(df, feature_cols)

    split_cfg = cfg["split"]
    if split_cfg.get("fold"):   # Phase 4.1 rolling-origin fold (src/data/folds.py)
        train, val, test = fold_frames(df, df[VALID_COL].to_numpy(), split_cfg["train_ratio"],
                                       cfg["sequence"]["input_length"], int(split_cfg["fold"]))
    else:
        train, val, test = chronological_split(df, split_cfg["train_ratio"], split_cfg["val_ratio"])

    feature_scaler, target_scaler = fit_scalers(
        train, feature_cols, target_col,
        save_dir=cfg["paths"]["scalers"] if save else None, track=track,
    )
    if condition == "minmax":
        frames = [apply_scaling(x, feature_cols, target_col, feature_scaler, target_scaler)
                  for x in (train, val, test)]
    else:
        ident = _Identity()
        frames = [apply_scaling(x, feature_cols, target_col, ident, ident) for x in (train, val, test)]

    splits = build_all_splits(
        *frames, feature_cols, target_col,
        input_length=cfg["sequence"]["input_length"], horizon=horizon,
    )
    splits["condition"] = np.array(condition)
    if condition != "minmax":
        ws = WindowScaler(condition, feature_cols).fit(
            splits["X_train"], splits["yraw_train"], splits["ctxt_train"], splits["last_train"])
        for s in ("train", "val", "test"):
            splits[f"X_{s}"] = ws.transform_X(splits[f"X_{s}"], splits[f"last_{s}"])
            splits[f"y_{s}"] = ws.transform_y(splits[f"yraw_{s}"], splits[f"ctxt_{s}"], splits[f"last_{s}"])
        feature_scaler = target_scaler = ws
    if save:
        save_sequences(splits, track, horizon, save_dir=cfg["paths"]["features"])

    split_info = {"track": track, "horizon": horizon, "n_features": len(feature_cols), "condition": condition}
    for name, part in (("train", train), ("val", val), ("test", test)):
        origins = splits[f"origin_{name}"]
        split_info[name] = {
            "rows": len(part),
            "row_range": [str(part.index.min().date()), str(part.index.max().date())],
            "invalid_rows": int((~part[VALID_COL]).sum()),
            "windows": int(len(origins)),
            "windows_dropped": int(splits[f"dropped_{name}"]),
            "first_origin": str(pd.Timestamp(origins[0]).date()) if len(origins) else None,
            "last_origin": str(pd.Timestamp(origins[-1]).date()) if len(origins) else None,
            "target_range": ([str(pd.Timestamp(splits[f"tfirst_{name}"][0]).date()),
                              str(pd.Timestamp(splits[f"tlast_{name}"][-1]).date())] if len(origins) else None),
        }
    return splits, (feature_scaler, target_scaler), feature_cols, split_info


class _Identity:
    """Pass-through scaler so windows can be built on unscaled values."""

    def transform(self, a):
        return a


def to_price(splits: dict, scalers: tuple, y_model: np.ndarray, split: str) -> np.ndarray:
    """Map model outputs for ``split`` to price units (any condition)."""
    condition = str(splits["condition"])
    y_model = np.asarray(y_model)
    if condition == "minmax":
        t_scaler = scalers[1]
        return t_scaler.inverse_transform(y_model.reshape(-1, 1)).reshape(y_model.shape).astype(np.float64)
    return scalers[0].to_price(y_model, splits[f"ctxt_{split}"], splits[f"last_{split}"])
