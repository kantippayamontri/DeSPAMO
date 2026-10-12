import json
from pathlib import Path

import numpy as np
import pytest

from tools.clip_plain.tests.helpers import parent_fixture


def test_global_half_is_exact_and_independent():
    from tools.clip_plain.features import global_columns

    values = np.arange(3 * 2048, dtype=np.float32).reshape(3, 2048)
    before = values.copy()
    child = global_columns(values)
    assert child.shape == (3, 1024) and child.flags.c_contiguous
    assert not np.shares_memory(child, values)
    np.testing.assert_array_equal(child, before[:, :1024])


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 1024), np.float32),
        np.zeros((2, 2048), np.float64),
        np.full((2, 2048), np.nan, np.float32),
    ],
)
def test_wrong_parent_layout_width_dtype_or_nan_rejected(bad):
    from tools.clip_plain.features import global_columns

    with pytest.raises(ValueError):
        global_columns(bad)


def prepare(tmp_path, monkeypatch):
    from tools.clip_plain import features as f

    c = parent_fixture(tmp_path)
    monkeypatch.setattr(f, "require_official_storage", lambda *a, **k: {})
    return f, c


def test_verified_resume_is_idempotent(tmp_path, monkeypatch):
    f, c = prepare(tmp_path, monkeypatch)
    result = f.derive_package(c)
    assert result["clips"] == 13
    root = Path(result["root"])
    before = (root / "train/train0.npy").read_bytes()
    assert f.derive_package(c) == result
    assert (root / "train/train0.npy").read_bytes() == before
    child = json.loads((root / "receipts/train/train0.json").read_text())
    assert child["extra"]["parent_feature_sha256"]
    assert child["extra"]["parent_receipt_sha256"]


def test_parent_receipt_or_source_drift_prevents_complete(tmp_path, monkeypatch):
    f, c = prepare(tmp_path, monkeypatch)
    p = Path(c["parent_root"]) / "train/train0.npy"
    np.save(p, np.zeros((20, 2048), np.float32))
    with pytest.raises(ValueError):
        f.derive_package(c)
    assert not list(Path(c["feature_base"]).glob("*/complete"))


def test_interrupted_unreceipted_file_preserved(tmp_path, monkeypatch):
    f, c = prepare(tmp_path, monkeypatch)
    root = f.package_root(c)
    p = root / "train/train0.npy"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"partial")
    with pytest.raises(ValueError):
        f.derive_package(c)
    assert p.read_bytes() == b"partial"


def test_storage_below_projection_plus_50gib_blocks_before_write(tmp_path, monkeypatch):
    f, c = prepare(tmp_path, monkeypatch)

    def fail(*a, **k):
        raise ValueError("reserve")

    monkeypatch.setattr(f, "require_official_storage", fail)
    with pytest.raises(ValueError, match="reserve"):
        f.derive_package(c)
    assert not Path(c["feature_base"]).exists()


def test_complete_inventory_no_extra_ids(tmp_path, monkeypatch):
    f, c = prepare(tmp_path, monkeypatch)
    r = f.derive_package(c)
    np.save(Path(r["root"]) / "train/extra.npy", np.zeros((20, 1024), np.float32))
    with pytest.raises(ValueError):
        f.verify_derived_package(c)


def test_verify_missing_child_does_not_recreate_it(tmp_path, monkeypatch):
    f, c = prepare(tmp_path, monkeypatch)
    r = f.derive_package(c)
    p = Path(r["root"]) / "dev/dev0.npy"
    p.unlink()
    (Path(r["root"]) / "receipts/dev/dev0.json").unlink()
    with pytest.raises(ValueError):
        f.verify_derived_package(c)
    assert not p.exists()


def test_cpu_conversion_records_elapsed_cost_once(tmp_path, monkeypatch):
    f, c = prepare(tmp_path, monkeypatch)
    f.derive_package(c)
    budget = json.loads((Path(c["feature_base"]) / "preparation-budget.json").read_text())
    assert budget["charged_seconds"] > 0 and budget["status"] == "complete"
