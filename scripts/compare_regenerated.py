"""
Compare regenerated results/ files with a committed version, ignoring only
provenance (which commit regenerated them): JSON keys git_commit / git_dirty
anywhere, and the "git_commit: `...`" line of markdown tables. Every other
byte must match.

    python scripts/make_tables.py && python scripts/compare_regenerated.py [--ref HEAD]

Exit 1 on any difference; prints one line per changed file.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_KEYS = {"git_commit", "git_dirty"}
COMMIT_LINE = re.compile(r"^git_commit: `[0-9a-f]+`$", re.M)


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in IGNORED_KEYS}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def normalise(path: str, data: bytes):
    if path.endswith(".json"):
        return strip(json.loads(data))
    if path.endswith(".md"):
        return COMMIT_LINE.sub("git_commit: <commit>", data.decode())
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ref = ap.parse_args().ref
    tracked = subprocess.run(["git", "ls-files", "results"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    changed = subprocess.run(["git", "diff", "--name-only", ref, "--", "results"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    bad = 0
    for path in changed:
        new = ROOT / path
        if not new.exists():
            print(f"MISSING  {path}")
            bad += 1
            continue
        old = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True).stdout
        same = normalise(path, old) == normalise(path, new.read_bytes())
        print(f"{'same*   ' if same else 'DIFFERS '} {path}")
        bad += not same
    print(f"{len(tracked)} tracked results files; {len(changed)} differ in bytes from {ref}; "
          f"{bad} differ beyond provenance (* = provenance only)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
