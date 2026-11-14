"""
Pass/fail checks for a release tree (a fresh clone or an unzipped code sample).

  secrets          no token/key/password patterns in any text file
  personal paths   no /media/, /home/, user name or e-mail address
  restricted data  no raw Yahoo/FRED/ACLED files, derived global data,
                   checkpoints, forecast arrays, MLflow store, venv or .git
  compile          every .py file compiles
  ensembles        every stored ensemble's members are the top-2 by
                   validation MAE of its candidates (no test information)
  baselines        every results table that reports test accuracy includes
                   persistence
  claims           scripts/check_claims.py passes (evidence files exist,
                   quoted numbers appear in them)

    python scripts/release_checks.py [TREE] [--allow-git]
"""

import argparse
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

TEXT = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".ipynb", ".cfg", ".toml", ".sh", ".gitignore"}
SECRETS = re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|"
                     r"(?i:(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"])")
# The public repository URL (github.com/Derdiry/...) and the GitHub noreply address are allowed.
PERSONAL = re.compile(r"/media/|/home/|(?<!github\.com/)(?<!\+)derdiry|mod\.derdiry|C:\\\\Users", re.I)
FORBIDDEN = [r"^data/raw/(?!(new_Data_SYNTHETIC_sample\.csv|SUDAN_DATA\.md)$)", r"\.xlsx$", r"global_merged\.csv$", r"^data/features/", r"(^|/)logs/",
             r"(^|/)mlruns/", r"(^|/)\.?venv/", r"\.(keras|h5|npz|pt|ckpt)$", r"test_global\.csv$", r"acled_sudan\.csv$",
             r"sudan_monthly_.*\.csv$", r"crosscheck_by_month\.csv$"]


def files(tree: Path, allow_git: bool):
    for p in sorted(tree.rglob("*")):
        rel = p.relative_to(tree).as_posix()
        if p.is_file() and not (allow_git and rel.startswith(".git/")) and "__pycache__" not in rel:
            yield p, rel


def text_of(p: Path) -> str:
    try:
        s = p.read_text(errors="ignore")
    except OSError:
        return ""
    if p.suffix == ".ipynb":   # skip embedded images
        s = re.sub(r'"image/png": "[^"]*"', '""', s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree", nargs="?", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--allow-git", action="store_true", help="skip .git/ (for checking a clone)")
    args = ap.parse_args()
    tree = Path(args.tree).resolve()
    all_files = list(files(tree, args.allow_git))
    texts = [(p, rel) for p, rel in all_files if p.suffix in TEXT or p.name in TEXT]
    results = {}

    results["secrets"] = [f"{rel}: {m.group(0)[:30]}" for p, rel in texts for m in SECRETS.finditer(text_of(p))]
    results["personal paths"] = [f"{rel}: {m.group(0)}" for p, rel in texts if rel != "scripts/release_checks.py"
                                 for m in PERSONAL.finditer(text_of(p))]   # this file holds the patterns
    results["restricted data"] = [rel for _, rel in all_files if any(re.search(f, rel) for f in FORBIDDEN)]
    if not args.allow_git and (tree / ".git").exists():
        results["restricted data"].append(".git/")

    bad = []
    for p, rel in all_files:
        if p.suffix == ".py":
            try:
                py_compile.compile(str(p), doraise=True, cfile=str(tree / ".release_check.pyc"))
            except py_compile.PyCompileError as e:
                bad.append(f"{rel}: {e.msg[:80]}")
    (tree / ".release_check.pyc").unlink(missing_ok=True)
    results["compile"] = bad

    bad = []
    for p in sorted(tree.glob("results/**/ensembles/*.json")):
        e = json.loads(p.read_text())
        if "selected" not in e:
            continue
        cand = e["candidates_val_mae"]
        top2 = sorted(cand, key=cand.get)[:2]
        if "validation" not in e["selection_rule"] or sorted(top2) != sorted(e["selected"]):
            bad.append(p.relative_to(tree).as_posix())
    results["ensembles"] = bad
    n_ens = len(list(tree.glob("results/**/ensembles/*.json")))

    bad, n_tables = [], 0
    skip = re.compile(r"PREREGISTRATION|provenance/|selection/|crosscheck|split_summary|facts\.md|ensembles\.md|comparison|"
                      r"lag_selection|validation_table")
    for p in sorted(tree.glob("results/**/*.md")):
        rel = p.relative_to(tree).as_posix()
        s = p.read_text()
        if skip.search(rel) or not re.search(r"\b(MAE|MASE)\b", s):
            continue
        n_tables += 1
        if "persistence" not in s.lower():
            bad.append(rel)
    results["baselines"] = bad

    claims = subprocess.run([sys.executable, "scripts/check_claims.py"], cwd=tree, capture_output=True, text=True)
    results["claims"] = [] if claims.returncode == 0 else claims.stdout.strip().splitlines()

    notes = {"compile": f"{sum(p.suffix == '.py' for p, _ in all_files)} .py files",
             "ensembles": f"{n_ens} ensemble records", "baselines": f"{n_tables} accuracy tables",
             "claims": claims.stdout.strip().splitlines()[-1] if claims.stdout.strip() else ""}
    failed = 0
    for name, issues in results.items():
        print(f"{'PASS' if not issues else 'FAIL'}  {name:16s} {notes.get(name, '')}")
        for i in issues[:10]:
            print(f"        {i}")
        failed += bool(issues)
    print(f"{len(all_files)} files checked in {tree.name}; {failed} check(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
