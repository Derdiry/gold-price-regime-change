"""
SHA-256 of every array a Sudan model sees (inputs, targets, origins, origin
prices, target contexts) for each requested (horizon, condition, fold), built
by the code of the current working directory. Run it at HEAD and inside a git
worktree of an older commit to prove that reused runs saw byte-identical data.

    cd <checkout> && python <repo>/scripts/dataset_fingerprint.py --out f.json \
        --settings 1:log_return 7:log_return 30:revin [--folds 4 5]
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
logging.disable(logging.WARNING)
from src.data.dataset import make_dataset  # noqa: E402
from src.data.features import SUDAN_FEATURE_COLS  # noqa: E402
from src.data.fetchers import load_config  # noqa: E402
from src.data.pipeline import build_sudan  # noqa: E402


def digest(a) -> str:
    a = np.asarray(a)
    a = a.astype(str) if a.dtype.kind in "OMU" else np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", nargs="+", required=True, help="horizon:condition")
    ap.add_argument("--folds", type=int, nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config("configs/model_config.yaml")
    cfg["paths"]["processed_data"] = str(Path(args.out).parent / "processed")   # never touch the checkout's data/
    df = build_sudan(cfg)
    out = {"cwd": str(Path.cwd()), "feature_list": list(SUDAN_FEATURE_COLS), "arrays": {}}
    for fold in [None, *args.folds]:
        for hc in args.settings:
            h, cond = hc.split(":")
            c = json.loads(json.dumps(cfg))
            if fold:
                c["split"]["fold"] = fold
            splits, _, feature_cols, _ = make_dataset(df, c, "sudan", int(h), SUDAN_FEATURE_COLS,
                                                      c["data"]["target_col"], save=False, condition=cond)
            key = f"{'main' if fold is None else f'fold{fold}'}_h{h}_{cond}"
            out["arrays"][key] = {"feature_cols": digest(np.array(feature_cols)),
                                  **{k: digest(v) for k, v in sorted(splits.items()) if k != "condition"}}
            print(key, len(feature_cols), {s: len(splits[f'origin_{s}']) for s in ('train', 'val', 'test')}, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
