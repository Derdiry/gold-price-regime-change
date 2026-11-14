"""
Shared pieces of the Phase 4.3-4.5 paired with-vs-without reports
(scripts/regime_report.py, transfer_report.py, acled_report.py).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.stats import diebold_mariano, holm, mase_vs_persistence  # noqa: E402

ALPHA, LOSSES, SEEDS, HORIZONS = 0.05, ["squared", "absolute"], [1, 2, 3], [1, 7, 30]


def require_committed(prereg: Path) -> str:
    """Commit of ``prereg``; exits unless it is tracked and unchanged."""
    rel = str(prereg.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = not subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not (tracked and clean):
        sys.exit(f"{rel} must be committed (and unchanged) before this step.")
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def baseline(pred_dir: Path, h: int) -> dict:
    """Test targets, origin prices and persistence of the windows in ``pred_dir``/baselines."""
    b = np.load(pred_dir / "baselines" / f"sudan_h{h}.npz")
    return {"y": b["y_test"], "last": b["last_test"], "pers": b["persistence_test"]}


def load_run(runs_dir: Path, pred_dir: Path, key: str, ref: dict) -> dict:
    """
    Record and test predictions of one run, after checking that its windows
    are ``ref``'s (saved targets/origin prices when present, and the
    recorded test MAE recomputed on ``ref``'s targets).
    """
    rec = json.loads((runs_dir / f"{key}.json").read_text())
    z = np.load(pred_dir / f"{key}.npz")
    if "y_test" in z:
        assert np.allclose(z["y_test"], ref["y"]) and np.allclose(z["last_test"], ref["last"]), f"{key}: windows differ"
    f = z["test_pred"]
    assert f.shape == ref["y"].shape, key
    assert np.isclose(np.abs(ref["y"] - f).mean(), rec["metrics"]["test"]["mae"], rtol=1e-6), f"{key}: test MAE mismatch"
    return {"rec": rec, "f": f, "val_mae": rec["metrics"]["val"]["mae"]}


def score(ref: dict, f: np.ndarray, h: int) -> dict:
    m = compute_all_metrics(ref["y"], f, ref["last"])
    out = {"mae": m["mae"], "mape": m["mape"], "mase": mase_vs_persistence(ref["y"], f, ref["pers"]), "n": m["n_windows"]}
    out["vs_persistence"] = {loss: diebold_mariano(ref["y"], f, ref["pers"], h, loss) for loss in LOSSES}
    return out


def paired(ref: dict, f_model: np.ndarray, f_reference: np.ndarray, h: int) -> dict:
    """DM of model vs reference (DM* < 0: model has lower loss)."""
    return {loss: diebold_mariano(ref["y"], f_model, f_reference, h, loss) for loss in LOSSES}


def add_holm(tests: list[dict]) -> None:
    """Holm-adjust the seed-level tests of one family and loss, in place (key p_holm)."""
    for t, p in zip(tests, holm([t["p_value"] for t in tests])):
        t["p_holm"] = p


def verdict(dms: list[dict], pkey: str, words: tuple) -> str:
    """words[0] if every seed has DM* < 0 and p < ALPHA, words[1] if every seed has DM* > 0 and p < ALPHA."""
    if all(d["dm_hln"] < 0 and d[pkey] < ALPHA for d in dms):
        return words[0]
    if all(d["dm_hln"] > 0 and d[pkey] < ALPHA for d in dms):
        return words[1]
    return "no consistent difference"


def ms(v, fmt="{:.2f}") -> str:
    v = np.asarray(v, float)
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}" if len(v) > 1 else fmt.format(v[0])


def pfmt(dms: list[dict], key="p_value") -> str:
    """'p1/p2/p3' with the sign of DM* per seed."""
    return "/".join("%.3f%s" % (d[key], "−" if d["dm_hln"] < 0 else "+") for d in dms)
