"""
Run provenance: which code and which configuration produced an artefact.

Every MLflow run and every metrics file written under the v2 protocol records
the git commit, whether the working tree had uncommitted changes, and the full
resolved configuration, so any reported number can be traced to its code.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_commit() -> str:
    """Full hash of HEAD, or "unknown" outside a git checkout."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool:
    """
    True if tracked files outside results/ have uncommitted changes (untracked
    files are ignored). results/ holds generated outputs that experiments
    rewrite while they run, so changes there say nothing about the code.
    """
    return bool(_git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results"))


def provenance(cfg: dict) -> dict:
    """Provenance fields to store with a run or a metrics file."""
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "protocol": cfg.get("mlflow", {}).get("protocol", "unknown"),
    }
