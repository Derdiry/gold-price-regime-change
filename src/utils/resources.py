"""
Memory guard for CPU training on a small machine.

Training is refused (not attempted) when available memory is below the
configured floor, so a long run never starts in a state likely to swap or be
killed. Completed models are protected by done flags, so a refused run can be
resumed later with ``--resume``.
"""

from pathlib import Path


class InsufficientMemory(RuntimeError):
    """Raised when MemAvailable is below the configured floor."""


def available_memory_gb() -> float:
    """MemAvailable from /proc/meminfo in GiB (Linux). Returns inf if unreadable."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return float("inf")


def require_memory(min_gb: float, what: str) -> float:
    """Return available GiB, or raise InsufficientMemory if it is below ``min_gb``."""
    avail = available_memory_gb()
    if avail < min_gb:
        raise InsufficientMemory(
            f"Refusing to start {what}: {avail:.2f} GiB available < {min_gb:.1f} GiB floor. "
            f"Free memory and re-run with --resume."
        )
    return avail
