"""--resume refuses records without genuine, matching validation metrics."""

import json

import pytest

train = pytest.importorskip("train")


def _record(tmp_path, **over):
    rec = {"protocol": "v2", "config_sha256": "abc",
           "metrics": {"val": {"mae": 1.0, "n_windows": 10}, "test": {"mae": 2.0, "n_windows": 5}}}
    rec.update(over)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rec))
    return p


def test_accepts_matching_record(tmp_path):
    assert train.load_reusable_record(_record(tmp_path), "abc", 10, 5) is not None


@pytest.mark.parametrize("over,args", [
    ({"metrics": {"test": {"mae": 2.0, "n_windows": 5}}}, ("abc", 10, 5)),   # v1-style: test only
    ({}, ("abc", 11, 5)),                                                     # different val windows
    ({}, ("other", 10, 5)),                                                   # config changed
    ({"protocol": "v1"}, ("abc", 10, 5)),
])
def test_refuses(tmp_path, over, args):
    assert train.load_reusable_record(_record(tmp_path, **over), *args) is None
