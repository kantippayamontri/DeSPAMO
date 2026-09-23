import importlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch


def test_export_cli_uses_layered_config_and_actual_model_state_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    exporter = importlib.import_module("export_model_schema")
    base = tmp_path / "base.yaml"
    base.write_text("model:\n  name: tiny\n  width: 2\n", encoding="utf-8")
    override = tmp_path / "override.yaml"
    override.write_text("model:\n  width: 3\n", encoding="utf-8")
    output = tmp_path / "nested" / "schema.json"
    built = []

    def fake_build_model(config):
        built.append((config.model.name, config.model.width))
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(config.model.width, 1, bias=False, dtype=torch.float16)
        model.register_buffer("steps", torch.tensor(0, dtype=torch.int64))
        return model

    monkeypatch.setattr(exporter, "build_model", fake_build_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_model_schema.py",
            "--config",
            str(base),
            "--config",
            str(override),
            "--output",
            str(output),
        ],
    )

    exporter.main()

    assert built == [("tiny", 3)]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "state_dict": {
            "proj.weight": {"shape": [1, 3], "dtype": "torch.float16"},
            "steps": {"shape": [], "dtype": "torch.int64"},
        }
    }


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hard_link"])
def test_export_cli_rejects_config_output_alias_before_building_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], alias: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    exporter = importlib.import_module("export_model_schema")
    first = tmp_path / "base.yaml"
    first.write_text("model:\n  name: tiny\n", encoding="utf-8")
    second = tmp_path / "override.yaml"
    second.write_text("model:\n  width: 3\n", encoding="utf-8")
    output = tmp_path / "schema.json"
    if alias == "same_path":
        output = second
    elif alias == "symlink":
        try:
            output.symlink_to(second)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
    else:
        try:
            os.link(second, output)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"hard links unavailable: {exc}")
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()

    def unexpected_build(config: object) -> None:
        pytest.fail("model built before config/output alias rejection")

    monkeypatch.setattr(exporter, "build_model", unexpected_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_model_schema.py",
            "--config",
            str(first),
            "--config",
            str(second),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        exporter.main()

    assert exc.value.code == 2
    assert "config and output must be different files" in capsys.readouterr().err
    assert first.read_bytes() == first_bytes
    assert second.read_bytes() == second_bytes
    assert output.read_bytes() == second_bytes
