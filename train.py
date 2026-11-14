"""
Main training entry point for the Gold Price Forecaster (evaluation protocol v2).

For each (track, horizon, seed) it:
  1. Rebuilds the processed data from raw files (src/data/pipeline.py)
  2. Builds train/val/test windows (src/data/dataset.py): chronological
     70/15/15 split, scalers fitted on train, windows touching the long Sudan
     gap or undefined features dropped
  3. Trains LSTM, GRU, TSMixer, PatchTST and TFT with early stopping on
     validation MAE, then reloads the best-validation checkpoint
  4. Scores every model on validation, and on test unless --val-only
  5. Builds the top-2 ensemble from validation predictions only and saves its
     members and weights
  6. Writes one JSON record per model (results/runs/), the ensemble
     (results/ensembles/) and a metrics CSV (results/metrics/), each with
     the git commit and full config, and registers the model in MLflow

Protocol rules enforced here:
  - Model and ensemble selection use validation data only.
  - With --val-only the test windows are never predicted. A later run without
    --val-only and with --resume evaluates the saved best checkpoint on test
    once, without retraining.
  - --resume reuses a finished model only if its saved record has genuine
    validation metrics for the same windows and the same config.

Phase 4 variants routed to their own results/, run-state and MLflow folders
(never registered, never mixed with the main results): --target (4.7),
--fold (4.1), --ablation no_regime (4.4), --conflict-lag (4.3),
--feature-set shared and --transfer-from-global (4.5). Chronos (4.2) is
scripts/run_chronos.py.

Usage:
    python train.py                                   # both tracks, all horizons, seeds from config
    python train.py --track sudan --model lstm --horizon 7 --seeds 1
    python train.py --val-only                        # selection runs: never touch test
    python train.py --resume                          # continue after a crash
    python train.py --smoke --track sudan --model lstm --horizon 1   # 1-epoch plumbing test
"""

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── ensure src/ is importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.acled import conflict_features
from src.data.dataset import make_dataset, to_price
from src.data.features import GLOBAL_FEATURE_COLS, SUDAN_FEATURE_COLS, shared_feature_cols
from src.data.fetchers import load_config
from src.data.pipeline import build_global, build_sudan
from src.evaluation.metrics import compute_all_metrics, format_metrics_table
from src.models.ensemble import apply_ensemble, fit_validation_ensemble
from src.models.gru import build_gru_from_config
from src.models.lstm import build_lstm_from_config
from src.models.patchtst import build_patchtst_from_config
from src.models.tft import build_tft_from_config, extract_feature_importance
from src.models.tsmixer import build_tsmixer_from_config
from src.training.mlflow_utils import get_or_create_experiment, register_keras_model, setup_mlflow
from src.training.trainer import ModelTrainer
from src.utils.provenance import provenance
from src.utils.resources import InsufficientMemory, require_memory
from src.utils.run_logging import setup_run_logging

import mlflow
from tensorflow import keras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

MODELS = ["lstm", "gru", "tsmixer", "patchtst", "tft"]
PROTOCOL = "v2"
REGIME_COLS = ["regime_1", "regime_2"]


# ─────────────────────────────────────────────────────────────────────────────
# Paths, keys and records
# ─────────────────────────────────────────────────────────────────────────────

def run_dirs(cfg: dict) -> dict:
    """Runtime state under paths.runs (gitignored) and reported outputs under paths.results."""
    runs, results = Path(cfg["paths"]["runs"]), Path(cfg["paths"]["results"])
    dirs = {
        "checkpoints": runs / "checkpoints",
        "done": runs / "done",
        "preds": runs / "preds",
        "records": results / "runs",
        "ensembles": results / "ensembles",
        "metrics": results / "metrics",
        "tft_importance": results / "tft_importance",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def setting_key(track: str, horizon: int, condition: str, seed: int) -> str:
    return f"{track}_h{horizon}_{condition}_s{seed}"


def run_key(name: str, track: str, horizon: int, condition: str, seed: int) -> str:
    return f"{name}_{setting_key(track, horizon, condition, seed)}"


def config_sha256(cfg: dict) -> str:
    """
    Hash of the settings that affect a model's results.

    Excluded: paths, MLflow names, the seed list (each run's seed is part of
    its key) and the memory floor.
    """
    relevant = copy.deepcopy({k: v for k, v in cfg.items() if k not in ("paths", "mlflow")})
    relevant.get("experiment", {}).pop("seeds", None)
    relevant.get("training", {}).pop("min_free_memory_gb", None)
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()


def _jsonable(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg, default=str))


def load_reusable_record(path: Path, cfg_hash: str, n_val: int, n_test: int) -> dict | None:
    """
    Return a saved model record only if --resume may reuse it.

    Refused when the record is missing, from another protocol or config, has
    no validation metrics, or was scored on a different number of windows.
    """
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"[RESUME] Unreadable record {path}: {exc}; retraining")
        return None
    problems = []
    if rec.get("protocol") != PROTOCOL:
        problems.append(f"protocol={rec.get('protocol')}")
    if rec.get("config_sha256") != cfg_hash:
        problems.append("config changed")
    val = rec.get("metrics", {}).get("val")
    if not val or val.get("n_windows") != n_val:
        problems.append("no matching validation metrics")
    test = rec.get("metrics", {}).get("test")
    if test is not None and test.get("n_windows") != n_test:
        problems.append("test windows differ")
    if problems:
        logger.warning(f"[RESUME] Refusing to reuse {path.name}: {', '.join(problems)}; retraining")
        return None
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def get_model_builders(cfg: dict) -> dict:
    """Returns {model_name: build_fn(input_shape, horizon) → model}"""
    return {
        "lstm":     lambda shape, h: build_lstm_from_config(cfg, shape, h),
        "gru":      lambda shape, h: build_gru_from_config(cfg, shape, h),
        "tsmixer":  lambda shape, h: build_tsmixer_from_config(cfg, shape, h),
        "patchtst": lambda shape, h: build_patchtst_from_config(cfg, shape, h),
        "tft":      lambda shape, h: build_tft_from_config(cfg, shape, h),
    }


def effective_batch_size(cfg: dict, name: str, track: str) -> int:
    """Batch size actually used: training.batch_size unless overridden per model and track."""
    override = cfg["training"].get("batch_size_overrides", {}).get(name, {})
    return int(override.get(track, cfg["training"]["batch_size"]))


def count_params(model) -> dict:
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    total = int(model.count_params())
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


# ─────────────────────────────────────────────────────────────────────────────
# One model
# ─────────────────────────────────────────────────────────────────────────────

def _to_price(data: dict, y_model: np.ndarray, split: str) -> np.ndarray:
    """Model outputs -> price units for the run's condition (src/data/dataset.to_price)."""
    return to_price(data, data["scalers"], y_model, split)


def _load_best(builder, input_shape, horizon, ckpt: Path, X_probe: np.ndarray):
    """Rebuild a model and load its best-validation weights."""
    model = builder(input_shape, horizon)
    model(X_probe[:1])  # build variables (needed for subclassed models)
    model.load_weights(str(ckpt))
    return model


def _window_arrays(data: dict, splits: tuple) -> dict:
    """Targets, origin prices and origin dates saved next to the predictions, so reports can check window identity."""
    out = {}
    for s in splits:
        out[f"y_{s}"], out[f"last_{s}"] = data[f"y_{s}_true"], data[f"last_{s}"]
        out[f"origin_{s}"] = np.asarray(data[f"origin_{s}"]).astype(str)
    return out


def _load_pretrained(name, builder, input_shape, horizon, seed, condition, data, transfer):
    """
    Phase 4.5: a Sudan model initialised from the global model with the same
    architecture, horizon, condition and seed; the first layer that has
    weights is frozen. Returns (model, transfer record).
    """
    src_key = run_key(name, "global", horizon, condition, seed)
    ckpt = Path(transfer["checkpoints"]) / f"{src_key}_best.weights.h5"
    src_rec = json.loads((Path(transfer["records"]) / f"{src_key}.json").read_text())
    if not ckpt.exists():
        raise FileNotFoundError(f"pretrained checkpoint missing: {ckpt}")
    # Same inputs in the same order: own price first, then the shared indicators.
    assert len(src_rec["feature_cols"]) == len(data["feature_cols"]), "feature count differs from the pretrained model"
    assert src_rec["feature_cols"][1:] == data["feature_cols"][1:], "shared features differ from the pretrained model"
    model = builder(input_shape, horizon)
    model(data["X_val"][:1])
    model.load_weights(str(ckpt))
    frozen = next(layer for layer in model.layers if layer.weights)
    frozen.trainable = False
    logger.info(f"  Transfer from {src_key}: frozen layer '{frozen.name}'")
    return model, {"source_key": src_key, "source_checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                   "source_val_mae": src_rec["metrics"]["val"]["mae"], "source_provenance": src_rec["provenance"],
                   "frozen_layer": frozen.name, "learning_rate": transfer["learning_rate"],
                   "max_epochs": transfer["max_epochs"], "early_stopping_patience": transfer["early_stopping_patience"]}


def train_one(
    name: str, cfg: dict, track: str, horizon: int, seed: int, data: dict,
    dirs: dict, resume: bool, val_only: bool,
) -> tuple[dict | None, dict]:
    """
    Train (or resume) one model. Returns (record, preds) where preds holds
    'val' and, unless val_only, 'test' predictions in price units.
    """
    condition = cfg["experiment"]["condition"]
    key = run_key(name, track, horizon, condition, seed)
    rec_path = dirs["records"] / f"{key}.json"
    pred_path = dirs["preds"] / f"{key}.npz"
    done_flag = dirs["done"] / f"{key}.done"
    ckpt = dirs["checkpoints"] / f"{key}_best.weights.h5"
    cfg_hash = config_sha256(cfg)
    builder = get_model_builders(cfg)[name]
    X_val, X_test = data["X_val"], data["X_test"]
    input_shape = (data["X_train"].shape[1], data["X_train"].shape[2])

    # ── Resume ───────────────────────────────────────────────────────────────
    if resume and done_flag.exists() and pred_path.exists():
        rec = load_reusable_record(rec_path, cfg_hash, len(X_val), len(X_test))
        if rec is not None:
            saved = dict(np.load(pred_path))
            preds = {"val": saved["val_pred"]}
            if rec["metrics"].get("test") is not None:
                preds["test"] = saved["test_pred"]
                logger.info(f"[RESUME] {key}: reusing saved val+test results")
                return rec, preds
            if val_only:
                logger.info(f"[RESUME] {key}: reusing saved val results")
                return rec, preds
            # Validation-only run finished earlier: score its best checkpoint on test once.
            logger.info(f"[RESUME] {key}: first test evaluation of the saved best checkpoint")
            model = _load_best(builder, input_shape, horizon, ckpt, X_val)
            preds["test"] = _to_price(data, model.predict(X_test, verbose=0), "test")
            rec["metrics"]["test"] = compute_all_metrics(data["y_test_true"], preds["test"], data["last_test"])
            rec["test_evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            rec["test_provenance"] = provenance(cfg)
            np.savez_compressed(pred_path, val_pred=preds["val"], test_pred=preds["test"], **_window_arrays(data, ("val", "test")))
            rec_path.write_text(json.dumps(rec, indent=2))
            if rec.get("mlflow_run_id"):
                client = mlflow.MlflowClient()
                for k, v in rec["metrics"]["test"].items():
                    if isinstance(v, (int, float)):
                        client.log_metric(rec["mlflow_run_id"], f"test_{k}", v)
            keras.backend.clear_session()
            return rec, preds

    # ── Train ────────────────────────────────────────────────────────────────
    _close_active_mlflow_run()
    logger.info(f"\n--- Training {key} ---")
    avail = require_memory(cfg["training"]["min_free_memory_gb"], key)
    logger.info(f"  Memory available: {avail:.2f} GiB")

    run_cfg = copy.deepcopy(cfg)
    run_cfg["training"]["batch_size"] = effective_batch_size(cfg, name, track)
    keras.utils.set_random_seed(seed)

    try:
        transfer = cfg["experiment"].get("transfer")
        if transfer:
            model, transfer_info = _load_pretrained(name, builder, input_shape, horizon, seed, condition, data, transfer)
            run_cfg["training"].update(learning_rate=transfer["learning_rate"], max_epochs=transfer["max_epochs"],
                                       early_stopping_patience=transfer["early_stopping_patience"])
        else:
            model, transfer_info = builder(input_shape, horizon), None
        t0 = time.time()
        model, history = ModelTrainer(run_cfg).fit_and_compile(
            model, data["X_train"], data["y_train"], X_val, data["y_val"],
            model_name=key, checkpoint_dir=str(dirs["checkpoints"]), weights_only=True,
        )
        train_seconds = time.time() - t0
        model.load_weights(str(ckpt))  # evaluate exactly the best-validation checkpoint
        n_epochs = len(history.history.get("loss", []))

        preds = {"val": _to_price(data, model.predict(X_val, verbose=0), "val")}
        metrics = {"val": compute_all_metrics(data["y_val_true"], preds["val"], data["last_val"]), "test": None}
        if not val_only:
            preds["test"] = _to_price(data, model.predict(X_test, verbose=0), "test")
            metrics["test"] = compute_all_metrics(data["y_test_true"], preds["test"], data["last_test"])

        np.savez_compressed(pred_path, val_pred=preds["val"],
                            **({"test_pred": preds["test"]} if "test" in preds else {}),
                            **_window_arrays(data, ("val", "test") if "test" in preds else ("val",)))

        rec = {
            "key": key, "protocol": PROTOCOL, "model": name, "track": track, "horizon": horizon,
            "condition": condition, "seed": seed,
            "metrics": metrics,
            "selection_metric": "val_mae",
            "n_params": count_params(model),
            "epochs_trained": n_epochs,
            "max_epochs": run_cfg["training"]["max_epochs"],
            "early_stopping_patience": run_cfg["training"]["early_stopping_patience"],
            "stopped_early": n_epochs < run_cfg["training"]["max_epochs"],
            "sensitivity": cfg["experiment"].get("sensitivity"),
            "variant": cfg["experiment"].get("variant"),
            "target_col": data["target_col"],
            "best_epoch": int(np.argmin(history.history["val_mae"])) + 1,
            "effective_batch_size": run_cfg["training"]["batch_size"],
            "train_seconds": round(train_seconds, 1),
            "seconds_per_epoch": round(train_seconds / max(n_epochs, 1), 2),
            "n_features": input_shape[1],
            "feature_cols": data["feature_cols"],
            "transfer": transfer_info,
            "windows": {s: int(len(data[f"X_{s}"])) for s in ("train", "val", "test")},
            "provenance": provenance(cfg),
            "config_sha256": cfg_hash,
            "config": _jsonable(run_cfg),
        }

        if name == "tft":
            # Variable-selection weights averaged over the validation windows.
            importance = extract_feature_importance(model, X_val, data["feature_cols"])
            (dirs["tft_importance"] / f"{key}.json").write_text(json.dumps(
                {"computed_on": "validation windows", "key": key, "provenance": rec["provenance"],
                 "importance": importance}, indent=2))

        logger.info(
            f"  {key}: val MAE={metrics['val']['mae']:.4f} MAPE={metrics['val']['mape']:.2f}%"
            + (f" | test MAE={metrics['test']['mae']:.4f} MAPE={metrics['test']['mape']:.2f}%" if metrics["test"] else "")
            + f" | params={rec['n_params']['total']:,} | {n_epochs} epochs, {train_seconds:.0f}s"
        )

        flat = {f"{split}_{k}": v for split, m in metrics.items() if m for k, v in m.items()
                if isinstance(v, (int, float))}
        # The pyfunc serving wrapper assumes v1 MinMax scaling, so only MinMax main runs
        # are registered as models; every run is still logged with metrics and provenance.
        register = (condition == "minmax" and not cfg["experiment"].get("sensitivity")
                    and not cfg["experiment"].get("variant"))
        rec["mlflow_run_id"] = register_keras_model(
            model=model, cfg=run_cfg, model_name=name, track=track, horizon=horizon,
            feature_scaler=data["scalers"][0], target_scaler=data["scalers"][1], register=register,
            feature_cols=data["feature_cols"], model_params=cfg["models"].get(name, {}),
            history=history, eval_metrics=flat,
            extra_params={"seed": seed, "condition": condition,
                          "effective_batch_size": run_cfg["training"]["batch_size"],
                          "n_params": rec["n_params"]["total"], "epochs_trained": n_epochs,
                          "val_only": val_only},
            extra_tags={"run_key": key, "condition": condition, "seed": seed,
                        "config_sha256": cfg_hash, "val_only": val_only,
                        "sensitivity": cfg["experiment"].get("sensitivity") or "none",
                        "variant": cfg["experiment"].get("variant") or "none"},
        )
        rec_path.write_text(json.dumps(rec, indent=2))
        done_flag.touch()  # only after record, predictions and MLflow run are saved
        return rec, preds

    except InsufficientMemory:
        raise
    except Exception as exc:
        logger.error(f"  FAILED {key}: {exc}", exc_info=True)
        _close_active_mlflow_run()
        return None, {}
    finally:
        keras.backend.clear_session()


# ─────────────────────────────────────────────────────────────────────────────
# One (track, horizon, seed) setting
# ─────────────────────────────────────────────────────────────────────────────

def _close_active_mlflow_run() -> None:
    """End any accidentally-active MLflow run from a previous crash."""
    active = mlflow.active_run()
    if active:
        logger.warning(f"Found orphaned MLflow run {active.info.run_id} — closing it")
        mlflow.end_run()


def feature_list_for(cfg: dict, track: str, target_col: str, df: pd.DataFrame) -> list:
    """Model inputs for this run: the track's list, or the shared transfer list; minus ablated, plus added features."""
    exp = cfg["experiment"]
    if exp.get("feature_set") == "shared":
        features = shared_feature_cols(target_col)
    else:
        features = list(GLOBAL_FEATURE_COLS if track == "global" else SUDAN_FEATURE_COLS)
    drop = exp.get("drop_features") or []
    missing = [c for c in drop if c not in df.columns or c not in features]
    if missing:
        raise ValueError(f"features to drop are not model inputs: {missing}")
    extra = exp.get("extra_features") or []
    assert all(c in df.columns for c in extra), "added features missing from the data"
    return [c for c in features if c not in drop] + extra


def train_setting(
    df: pd.DataFrame, cfg: dict, track: str, horizon: int, seed: int,
    models: list, resume: bool, val_only: bool,
) -> dict:
    """Train every model for one setting, then the validation-selected ensemble."""
    condition = cfg["experiment"]["condition"]
    skey = setting_key(track, horizon, condition, seed)
    logger.info(f"\n{'=' * 60}\n{skey}\n{'=' * 60}")

    target_col = cfg["data"]["global_target_col"] if track == "global" else cfg["data"]["target_col"]
    target_col = cfg["experiment"].get("target_col") or target_col
    feature_list = feature_list_for(cfg, track, target_col, df)
    splits, scalers, feature_cols, split_info = make_dataset(
        df, cfg, track, horizon, feature_list, target_col, condition=condition)
    data = {**splits, "scalers": scalers, "feature_cols": feature_cols, "target_col": target_col,
            "y_val_true": splits["yraw_val"], "y_test_true": splits["yraw_test"]}
    dirs = run_dirs(cfg)

    setup_mlflow(cfg)
    mlflow.set_experiment(get_or_create_experiment(cfg, track))

    records, val_preds, test_preds = {}, {}, {}
    for name in models:
        rec, preds = train_one(name, cfg, track, horizon, seed, data, dirs, resume, val_only)
        if rec is None:
            continue
        records[name] = rec
        val_preds[name] = preds["val"]
        if "test" in preds:
            test_preds[name] = preds["test"]

    # ── Ensemble: members and weights from validation only ───────────────────
    ens_record = None
    if len(val_preds) >= 2 and set(val_preds) == set(MODELS):
        ens = fit_validation_ensemble(val_preds, data["y_val_true"], cfg["models"]["ensemble"]["top_n"])
        val_ens = apply_ensemble(val_preds, ens)
        ens_metrics = {"val": compute_all_metrics(data["y_val_true"], val_ens, data["last_val"]), "test": None}
        if not val_only and all(n in test_preds for n in ens["selected"]):
            ens_metrics["test"] = compute_all_metrics(
                data["y_test_true"], apply_ensemble(test_preds, ens), data["last_test"])
        ens_record = {
            "key": f"ensemble_{skey}", "protocol": PROTOCOL, "model": "ensemble", "track": track,
            "horizon": horizon, "condition": condition, "seed": seed,
            **ens, "metrics": ens_metrics,
            "member_records": [records[n]["key"] for n in ens["selected"]],
            "provenance": provenance(cfg), "config_sha256": config_sha256(cfg),
        }
        (dirs["ensembles"] / f"{skey}.json").write_text(json.dumps(ens_record, indent=2))
        logger.info(f"  Ensemble ({ens['selection_rule']}): "
                    + ", ".join(f"{n}={w:.4f}" for n, w in zip(ens["selected"], ens["weights"]))
                    + f" | val MAE={ens_metrics['val']['mae']:.4f}"
                    + (f" | test MAE={ens_metrics['test']['mae']:.4f}" if ens_metrics["test"] else ""))
    elif len(val_preds) >= 2:
        logger.info(f"  Ensemble skipped: only {sorted(val_preds)} of {MODELS} available in this run")

    # ── Metrics CSV (one row per model and split) with provenance ────────────
    # Built from every saved record of this setting, so single-model runs add rows
    # instead of overwriting the file.
    saved = {r["model"]: r for r in (json.loads(p.read_text())
                                     for p in sorted(dirs["records"].glob(f"*_{skey}.json")))}
    ens_path = dirs["ensembles"] / f"{skey}.json"
    if ens_record is None and ens_path.exists():
        ens_record = json.loads(ens_path.read_text())
    rows = []
    for name, rec in [*saved.items(), *([("ensemble", ens_record)] if ens_record else [])]:
        for split, m in rec["metrics"].items():
            if m:
                rows.append({"model": name, "split": split, **m})
    if rows:
        prov = provenance(cfg)
        out = pd.DataFrame(rows).assign(track=track, horizon=horizon, condition=condition, seed=seed,
                                        git_commit=prov["git_commit"], git_dirty=prov["git_dirty"],
                                        config_sha256=config_sha256(cfg))
        csv_path = dirs["metrics"] / f"{skey}.csv"
        out.to_csv(csv_path, index=False)
        csv_path.with_suffix(".meta.json").write_text(json.dumps(
            {"provenance": prov, "config_sha256": config_sha256(cfg), "split_info": split_info,
             "config": _jsonable(cfg)}, indent=2, default=str))
        logger.info(f"Saved metrics to {csv_path}")
        for split in ("val", "test"):
            table = {r["model"]: r for r in rows if r["split"] == split}
            if table:
                logger.info(f"\n[{skey}] {split}\n" + format_metrics_table(table))
    return {"records": records, "ensemble": ens_record}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Gold Price Forecaster — Training (protocol v2)")
    parser.add_argument("--track", choices=["global", "sudan", "both"], default="both")
    parser.add_argument("--model", choices=MODELS + ["all"], nargs="+", default=["all"])
    parser.add_argument("--horizon", type=int, default=None,
                        help="Prediction horizon in days (default: all from config)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (default: experiment.seeds from config)")
    parser.add_argument("--val-only", action="store_true",
                        help="Score validation only; the test windows are never predicted")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse finished models whose saved records pass the resume checks")
    parser.add_argument("--refetch-global", action="store_true",
                        help="Re-download Yahoo/FRED data, replacing the raw snapshot (changes results)")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--condition", choices=["minmax", "log_return", "revin"], default=None,
                        help="Target/input transformation (default: experiment.condition)")
    parser.add_argument("--duplicate-rule", choices=["first", "last", "median"], default=None,
                        help="Sensitivity run with another Sudan duplicate-date rule: separate "
                             "results/sensitivity/ folder, MLflow experiment and tags; never reported as main results")
    parser.add_argument("--target", choices=["fx_rate"], default=None,
                        help="Phase 4.7 (Sudan only): forecast SDG/USD instead of the price. Separate "
                             "results/fx_target/ folder, run state and MLflow experiment; price is rebuilt "
                             "by scripts/fx_target_report.py")
    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="Phase 4.1 rolling-origin fold (src/data/folds.py): separate results/rolling/fold<k>/ "
                             "folder, run state and MLflow experiment; never registered")
    parser.add_argument("--ablation", choices=["no_regime"], default=None,
                        help="Phase 4.4 (Sudan only): drop regime_1 and regime_2 from the inputs. Separate "
                             "results/regime/{main,fold<k>}/no_regime/ folder, run state and MLflow experiment")
    parser.add_argument("--conflict-lag", type=int, choices=[1, 2, 3], default=None,
                        help="Phase 4.3 (Sudan only): add log1p ACLED political-violence events and fatalities of "
                             "calendar month t − k (src/data/acled.py). Separate results/acled/{main,fold<k>}/lag<k>/ folder")
    parser.add_argument("--feature-set", choices=["shared"], default=None,
                        help="Phase 4.5: the 18 inputs both tracks share (own price + 17 indicators). "
                             "results/transfer/{pretrain (global), scratch (Sudan)}/")
    parser.add_argument("--transfer-from-global", action="store_true",
                        help="Phase 4.5 (Sudan, with --feature-set shared): start from the matching "
                             "results/transfer/pretrain model, freeze its first layer, fine-tune "
                             "(training.fine_tune). results/transfer/finetune/")
    parser.add_argument("--smoke", action="store_true",
                        help="Plumbing test: 1 epoch, MLflow experiment 'smoke', "
                             "all file outputs under logs/smoke/. Results are never reported.")
    return parser.parse_args()


def apply_duplicate_sensitivity(cfg: dict, rule: str) -> None:
    """Route a duplicate-date sensitivity run away from the main results, runs and experiments."""
    tag = f"duplicate_{rule}"
    cfg["data"]["sudan"]["duplicate_date_rule"] = rule
    cfg["experiment"]["sensitivity"] = tag
    cfg["paths"]["results"] = str(Path(cfg["paths"]["results"]) / "sensitivity" / tag)
    cfg["paths"]["runs"] = str(Path(cfg["paths"]["runs"]) / "sensitivity" / tag)
    cfg["paths"]["processed_data"] = str(Path(cfg["paths"]["runs"]) / "processed")
    cfg["paths"]["scalers"] = str(Path(cfg["paths"]["runs"]) / "scalers")
    cfg["paths"]["features"] = str(Path(cfg["paths"]["runs"]) / "features")
    for k in ("experiment_global", "experiment_sudan"):
        cfg["mlflow"][k] = f"{cfg['mlflow'][k]}-sensitivity"
    logger.info(f"SENSITIVITY RUN ({tag}): outputs under {cfg['paths']['results']}")


def apply_fx_target(cfg: dict, target: str) -> None:
    """Phase 4.7: Sudan models forecast SDG/USD; route every output away from the price results."""
    tag = "fx_target"
    # Only the training target changes; data.target_col (price) still drives the
    # technical indicators, so the input features are identical to the price runs.
    cfg["experiment"]["target_col"] = target
    cfg["experiment"]["variant"] = tag
    cfg["paths"]["results"] = str(Path(cfg["paths"]["results"]) / tag)
    cfg["paths"]["runs"] = str(Path(cfg["paths"]["runs"]) / tag)
    cfg["paths"]["processed_data"] = str(Path(cfg["paths"]["runs"]) / "processed")
    cfg["paths"]["scalers"] = str(Path(cfg["paths"]["runs"]) / "scalers")
    cfg["paths"]["features"] = str(Path(cfg["paths"]["runs"]) / "features")
    cfg["mlflow"]["experiment_sudan"] = f"{cfg['mlflow']['experiment_sudan']}-fx-target"
    logger.info(f"FX TARGET RUN: target={target}, outputs under {cfg['paths']['results']}")


def apply_rolling_fold(cfg: dict, fold: int) -> None:
    """Phase 4.1: train/val/test = rolling-origin fold ``fold``; route every output away from the main results."""
    tag = f"rolling_fold{fold}"
    cfg["split"]["fold"] = fold
    cfg["experiment"]["variant"] = tag
    cfg["paths"]["results"] = str(Path(cfg["paths"]["results"]) / "rolling" / f"fold{fold}")
    cfg["paths"]["runs"] = str(Path(cfg["paths"]["runs"]) / "rolling" / f"fold{fold}")
    cfg["paths"]["processed_data"] = str(Path(cfg["paths"]["runs"]) / "processed")
    cfg["paths"]["scalers"] = str(Path(cfg["paths"]["runs"]) / "scalers")
    cfg["paths"]["features"] = str(Path(cfg["paths"]["runs"]) / "features")
    for k in ("experiment_global", "experiment_sudan"):
        cfg["mlflow"][k] = f"{cfg['mlflow'][k]}-rolling"
    logger.info(f"ROLLING-ORIGIN FOLD {fold}: outputs under {cfg['paths']['results']}")


def apply_variant(cfg: dict, base: tuple, rel: str, tag: str, mlflow_suffix: str) -> None:
    """Route a Phase 4 variant to base results/runs + rel, appending tag to experiment.variant."""
    prev = cfg["experiment"].get("variant")
    cfg["experiment"]["variant"] = f"{prev}_{tag}" if prev else tag
    cfg["paths"]["results"] = str(Path(base[0]) / rel)
    cfg["paths"]["runs"] = str(Path(base[1]) / rel)
    cfg["paths"]["processed_data"] = str(Path(cfg["paths"]["runs"]) / "processed")
    cfg["paths"]["scalers"] = str(Path(cfg["paths"]["runs"]) / "scalers")
    cfg["paths"]["features"] = str(Path(cfg["paths"]["runs"]) / "features")
    for k in ("experiment_global", "experiment_sudan"):
        cfg["mlflow"][k] = f"{cfg['mlflow'][k]}-{mlflow_suffix}"
    logger.info(f"VARIANT {cfg['experiment']['variant']}: outputs under {cfg['paths']['results']}")


def apply_phase4_variants(cfg: dict, args, base: tuple) -> None:
    """--ablation, --conflict-lag, --feature-set and --transfer-from-global (Phase 4.3-4.5)."""
    where = f"fold{args.fold}" if args.fold else "main"
    picked = [x for x in (args.ablation, args.conflict_lag, args.feature_set) if x]
    if len(picked) > 1:
        raise SystemExit("--ablation, --conflict-lag and --feature-set are not combined")
    if picked and args.target:
        raise SystemExit("Phase 4.3-4.5 variants are not combined with --target")
    if args.transfer_from_global and (args.feature_set != "shared" or args.track != "sudan"):
        raise SystemExit("--transfer-from-global needs --track sudan --feature-set shared")
    if (args.ablation or args.conflict_lag) and args.track != "sudan":
        raise SystemExit("--ablation and --conflict-lag are defined for --track sudan only")
    if args.ablation:
        cfg["experiment"]["drop_features"] = list(REGIME_COLS)
        apply_variant(cfg, base, f"regime/{where}/no_regime", "no_regime", "regime")
    if args.conflict_lag:
        cfg["experiment"]["conflict"] = {"lag": args.conflict_lag, "definition": "disorder_type", "rule": "prorata",
                                         "transform": "log1p", "series": ["pv_events", "pv_fatalities"]}
        apply_variant(cfg, base, f"acled/{where}/lag{args.conflict_lag}", f"acled_lag{args.conflict_lag}", "acled")
    if args.feature_set:
        if args.fold or args.track == "both":
            raise SystemExit("--feature-set shared is run per track on the main split")
        cfg["experiment"]["feature_set"] = "shared"
        if args.transfer_from_global:
            ft = cfg["training"]["fine_tune"]
            pre = Path(base[1]) / "transfer/pretrain"
            cfg["experiment"]["transfer"] = {
                "checkpoints": str(pre / "checkpoints"), "records": str(Path(base[0]) / "transfer/pretrain/runs"),
                "learning_rate": ft["learning_rate"], "max_epochs": ft["max_epochs"],
                "early_stopping_patience": ft["early_stopping_patience"], "freeze": "first layer with weights"}
            apply_variant(cfg, base, "transfer/finetune", "transfer_finetune", "transfer")
        else:
            role = "pretrain" if args.track == "global" else "scratch"
            apply_variant(cfg, base, f"transfer/{role}", f"transfer_{role}", "transfer")


def apply_smoke_mode(cfg: dict) -> None:
    """
    Isolate a smoke run from every reported or resumable artefact.

    Redirects processed data, scalers, sequences, run state and results to
    logs/smoke/, sends MLflow runs to the 'smoke' experiment and registry
    prefix, and caps training at one epoch.
    """
    root = Path("logs/smoke")
    cfg["paths"]["scalers"] = str(root / "scalers")
    cfg["paths"]["features"] = str(root / "features")
    cfg["paths"]["processed_data"] = str(root / "processed")
    cfg["paths"]["runs"] = str(root / "v2")
    cfg["paths"]["results"] = str(root / "results")
    cfg["mlflow"]["experiment_override"] = "smoke"
    cfg["mlflow"]["registered_model_prefix"] = "smoke"
    cfg["training"]["max_epochs"] = 1
    if cfg["experiment"].get("transfer"):   # fine-tune the smoke pretrain run, for one epoch
        cfg["experiment"]["transfer"].update(checkpoints=str(root / "v2" / "checkpoints"),
                                             records=str(root / "results" / "runs"), max_epochs=1)
    logger.info("SMOKE MODE: 1 epoch, outputs under logs/smoke/, MLflow experiment 'smoke'")


def main():
    """Build the data, then train every selected (track, horizon, seed)."""
    args = parse_args()
    setup_run_logging("train_smoke" if args.smoke else "train")
    cfg = load_config(args.config)
    base = (cfg["paths"]["results"], cfg["paths"]["runs"])
    if args.condition:
        cfg["experiment"]["condition"] = args.condition
    if args.duplicate_rule and args.duplicate_rule != cfg["data"]["sudan"]["duplicate_date_rule"]:
        apply_duplicate_sensitivity(cfg, args.duplicate_rule)
    if args.target:
        if args.track != "sudan":
            raise SystemExit("--target fx_rate is defined for --track sudan only")
        apply_fx_target(cfg, args.target)
    if args.fold:
        if args.target:
            raise SystemExit("--fold and --target are not combined")
        apply_rolling_fold(cfg, args.fold)
    apply_phase4_variants(cfg, args, base)
    if args.smoke:
        apply_smoke_mode(cfg)

    tracks = ["global", "sudan"] if args.track == "both" else [args.track]
    models = MODELS if "all" in args.model else [m for m in MODELS if m in args.model]
    horizons = cfg["sequence"]["horizons"] if args.horizon is None else [args.horizon]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    logger.info(f"Tracks={tracks} models={models} horizons={horizons} seeds={seeds} "
                f"condition={cfg['experiment']['condition']} val_only={args.val_only}")

    _close_active_mlflow_run()
    require_memory(cfg["training"]["min_free_memory_gb"], "training")

    data = {}
    if "global" in tracks:
        data["global"] = build_global(cfg, refetch=args.refetch_global)
    if "sudan" in tracks:
        data["sudan"] = build_sudan(cfg)
        if args.conflict_lag:
            feats, sha = conflict_features(data["sudan"].index, args.conflict_lag)
            if feats.isna().any().any():
                raise SystemExit("ACLED features undefined on some Sudan dates")
            data["sudan"] = data["sudan"].join(feats)
            cfg["experiment"]["extra_features"] = list(feats.columns)
            cfg["experiment"]["conflict"]["monthly_sha256"] = sha

    for horizon in horizons:
        for track in tracks:
            for seed in seeds:
                train_setting(data[track], cfg, track, horizon, seed, models, args.resume, args.val_only)

    logger.info("TRAINING COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except InsufficientMemory as exc:
        logger.error(str(exc))
        sys.exit(3)
