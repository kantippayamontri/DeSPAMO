import json

import pytest

from tools.clip_s2.extract import load_inputs, make_identity


def test_reference_grid_rearrangement_preserves_positions():
    import torch

    from tools.clip_s2.extract import reference_rearrange

    tokens = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    grid = reference_rearrange(tokens, "b (h w) c -> b c h w", h=2)
    assert grid.shape == (2, 3, 2, 2)
    assert torch.equal(grid[1, :, 1, 0], tokens[1, 2])
    assert torch.equal(reference_rearrange(grid, "b c h w -> b (h w) c"), tokens)


def test_config_rejects_extra_fields(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"unknown": True}))
    with pytest.raises(ValueError):
        load_inputs(p)


def test_feature_identity_changes_with_processor_or_batch():
    config = dict(
        revision="pinned",
        annotation_root="annotations",
        source_row_map="rows",
        frames_root="frames",
        cache="cache",
        processor_file="processor",
        output_base="out",
    )
    inputs = dict(config=config, bindings={"processor": "sha1"}, source={"code": "sha"})
    binding = dict(model="clip", revision="pinned")
    profile = dict(frame_batch_size=32, tile_batch_size=32)
    first = make_identity(inputs, binding, profile)
    inputs["bindings"]["processor"] = "sha2"
    assert make_identity(inputs, binding, profile) != first
    assert make_identity(inputs, binding, dict(frame_batch_size=16, tile_batch_size=32)) != first


def test_existing_complete_package_republishes_missing_result_pointer(tmp_path, monkeypatch):
    import numpy as np

    import tools.clip_s2.extract as e
    from tools.benchmark.feature_package import finish_package, write_clip

    monkeypatch.setattr(e, "BASE", tmp_path / "out")
    root = e.BASE / "spatial-fixture"
    paths = [tmp_path / "1.png", tmp_path / "2.png"]
    for p in paths:
        p.write_bytes(b"source")
    identity = dict(kind="spatial", width=2048, policy="fixture")
    inv = {"train": {"clip": dict(paths=[str(p) for p in paths], frame_count=2)}}
    write_clip(root, "train", "clip", paths, np.ones((2, 2048), dtype=np.float32), identity, {})
    finish_package(root, identity, inv, 2048)
    result = e.publish_existing(root, identity, inv, "run")
    assert result["root"] == str(root)
    assert json.loads((e.BASE / "package-result.json").read_text())["root"] == str(root)
