"""Split-conformal quantile: finite-sample coverage on exchangeable data."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from conformal_intervals import conformal_q  # noqa: E402


def test_rank_and_infinite_case():
    s = np.arange(1, 10, dtype=float).reshape(-1, 1)      # n = 9
    assert conformal_q(s, 0.2)[0] == 8                     # ceil(10 * 0.8) = 8th smallest
    assert np.isinf(conformal_q(s, 0.05)[0])               # ceil(10 * 0.95) = 10 > n


def test_marginal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(2000):
        e = np.abs(rng.standard_t(3, size=(51, 1)))
        covered.append(e[-1, 0] <= conformal_q(e[:-1], 0.1)[0])
    assert 0.88 <= np.mean(covered) <= 0.93
