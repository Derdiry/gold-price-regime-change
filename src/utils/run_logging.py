"""
Per-run log files.

Every training or evaluation entry point calls ``setup_run_logging`` once.
It mirrors all log records to ``logs/runs/<YYYYmmdd-HHMMSS>_<script>.log``
so information printed only to stdout (ensemble members and weights,
skipped models, failures) is kept with the run.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.provenance import REPO_ROOT, git_commit, git_dirty

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_run_logging(script: str, log_dir: str | Path = "logs/runs") -> Path:
    """Attach a file handler to the root logger and record argv and git state."""
    log_dir = Path(log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{script}.log"

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)

    log = logging.getLogger("run")
    log.info(f"Run log: {path}")
    log.info(f"argv: {' '.join(sys.argv)}")
    log.info(f"git_commit={git_commit()} git_dirty={git_dirty()}")
    return path
