"""
Check that the documents only quote numbers that exist in generated files.

docs/CLAIMS.md: for every claim row, each backticked evidence path must exist,
and every number in the claim and result cells must appear in that row's
evidence files.
README.md: every number must appear in some tracked results table
(results/**/*.md, *.csv), in docs/implementation_notes.md, or in the short
list of protocol constants below.

Numbers are compared as tokens (thousands separators removed, Unicode minus
read as "-"); text inside backticks is skipped (paths, identifiers).

    python scripts/check_claims.py      # exit 1 on any missing file or number
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")
TICKS = re.compile(r"`[^`]*`")
# Protocol constants and section numbers that README states without a results table behind them
README_CONSTANTS = {"3.10", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "15", "18", "20", "21", "24",
                    "29", "30", "33", "34", "60", "70", "80", "90", "100", "512", "2000", "2022", "2023", "2024",
                    "2015", "2019", "2026", "0.05"}


def tokens(text: str) -> set[str]:
    text = TICKS.sub(" ", text.replace("−", "-"))
    return {t.replace(",", "") for t in NUMBER.findall(text)}


def file_tokens(paths) -> set[str]:
    out = set()
    for p in paths:
        out |= tokens((ROOT / p).read_text().replace("`", " "))
    return out


CLAIMS = ROOT / "docs/CLAIMS.md"
ROW = re.compile(r"^\| (C\d+) \|")


def check_claims() -> int:
    bad = 0
    for line in CLAIMS.read_text().splitlines():
        m = ROW.match(line)
        if not m:
            continue
        _, claim, result, evidence_cell = [c.strip() for c in line.strip().strip("|").split("|")]
        paths = [p for p in re.findall(r"`([^`]+)`", evidence_cell) if "/" in p]
        missing = [p for p in paths if not (ROOT / p).exists()]
        if missing or not paths:
            print(f"{m.group(1)}: evidence file missing: {missing or 'none given'}")
            bad += 1
            continue
        absent = sorted(tokens(claim + " " + result) - file_tokens(paths) - {m.group(1)[1:]})
        if absent:
            print(f"{m.group(1)}: numbers not in {paths}: {absent}")
            bad += 1
    return bad


def check_readme() -> int:
    git = subprocess.run(["git", "ls-files", "results", "docs/implementation_notes.md"], cwd=ROOT, capture_output=True, text=True)
    tracked = git.stdout.split() if git.returncode == 0 and git.stdout.strip() else \
        [p.relative_to(ROOT).as_posix() for p in ROOT.glob("results/**/*") if p.is_file()] + ["docs/implementation_notes.md"]
    corpus = file_tokens([p for p in tracked if p.endswith((".md", ".csv"))])
    absent = sorted(tokens((ROOT / "README.md").read_text()) - corpus - README_CONSTANTS)
    for t in absent:
        print(f"README: {t} not found in any results table")
    return len(absent)


def main():
    n_claims, n_readme = check_claims(), check_readme()
    rows = sum(bool(ROW.match(line)) for line in CLAIMS.read_text().splitlines())
    print(f"docs/CLAIMS.md: {rows} rows, {n_claims} with problems; README.md: {n_readme} untraced numbers")
    sys.exit(1 if n_claims or n_readme else 0)


if __name__ == "__main__":
    main()
