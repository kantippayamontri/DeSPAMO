import json

import cv2
import how2sign_probe.cli as cli
import how2sign_probe.detector as detector
import how2sign_probe.frames as frames
import numpy as np
import pytest
from how2sign_probe.labels import MOUTH_FIELDS
from how2sign_probe.report import label_rows


def _csv(tmp_path):
    path = tmp_path / "train.csv"
    columns = (
        "VIDEO_ID",
        "VIDEO_NAME",
        "SENTENCE_ID",
        "SENTENCE_NAME",
        "START_REALIGNED",
        "END_REALIGNED",
        "SENTENCE",
    )
    lines = ["\t".join(columns)]
    for video in ("vidA", "vidB"):
        for i in range(12):
            lines.append(
                "\t".join(
                    (
                        video,
                        f"{video}-5-rgb_front",
                        f"{video}_{i}",
                        f"{video}_{i}-5-rgb_front",
                        "0",
                        "1",
                        f"s{i}",
                    )
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def fake_cv(monkeypatch):
    frame = np.full((720, 1280, 3), 9, dtype=np.uint8)
    monkeypatch.setattr(frames, "frame_count", lambda path: 40)
    monkeypatch.setattr(frames, "read_frame", lambda path, index: frame.copy())
    monkeypatch.setattr(
        cli,
        "read_frame",
        lambda path, index: frame.copy(),
    )
    monkeypatch.setattr(cli, "frame_count", lambda path: 40)

    def fake_detect(image):
        from how2sign_probe.detector import Region

        return [
            Region("hand", "LEFT", (100, 100, 200, 200), 0.9),
            Region("hand", "RIGHT", (400, 300, 520, 420), 0.8),
            Region("mouth", "MOUTH", (600, 80, 700, 140), 0.7),
        ]

    monkeypatch.setattr(cli, "detect_regions", fake_detect)
    return frame


def test_full_probe_stages(tmp_path, fake_cv, monkeypatch, capsys):
    root = tmp_path / "probe"
    csv_path = _csv(tmp_path)
    monkeypatch.setattr(cli, "build_provenance", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(detector, "MIN_SCORE", 0.2)

    cli.main(
        [
            "--root",
            str(root),
            "sample",
            "--csv",
            str(csv_path),
            "--total",
            "4",
            "--seed",
            "0",
        ]
    )
    sample = json.loads((root / "sample.json").read_text())
    assert len(sample["clips"]) == 4

    monkeypatch.setattr(cli, "MIN_DETECTOR_FRAMES", 1)
    with pytest.raises(SystemExit) as first_exit:
        cli.main(["--root", str(root), "detect", "--video-root", str(tmp_path)])
    assert first_exit.value.code == 0
    regions = json.loads((root / "regions.json").read_text())
    assert len(regions) == 4 * 5
    assert all(len(items) == 3 for items in regions.values())
    crops = list((root / "crops").glob("*.png"))
    assert len(crops) == 4 * 5 * 3
    assert (root / "detect_progress.json").exists()

    # interrupted-run resume: already-complete input must not re-decode
    with pytest.raises(SystemExit) as resume_exit:
        cli.main(["--root", str(root), "detect", "--video-root", str(tmp_path)])
    assert resume_exit.value.code == 0
    assert len(json.loads((root / "regions.json").read_text())) == 4 * 5
    assert (root / "detector_report.json").exists()

    valid_mouth = json.dumps(
        {
            "openness": "open",
            "lip_configuration": "rounded",
            "teeth_or_tongue": "tongue",
            "cheek_state": "neutral",
            "visibility": "clear",
        }
    )
    valid_hand = json.dumps(
        {
            "handshape_class": "cupped",
            "finger_spread": "spread",
            "palm_direction": "down",
            "thumb_position": "tucked",
            "visibility": "clear",
        }
    )

    def fake(prompt_text, image_b64, model, port, kind, timeout=900):
        return (valid_hand if kind == "hand" else valid_mouth), {"eval_count": 5}

    import how2sign_probe.labeling as labeling

    monkeypatch.setattr(labeling, "request_label", fake)
    cli.main(["--root", str(root), "label", "--ports", "11434"])
    journals = cli._load_journals(root)
    assert len(journals) == 4 * 5 * 3
    assert all(entry["status"] == "done" for entry in journals.values())

    cli.main(["--root", str(root), "report"])
    report = json.loads((root / "label_report.json").read_text())
    assert report["counts"]["done"] == 60
    assert report["hand"]["handshape_class"]["distinct"] == 1
    assert report["mouth"]["openness"]["top1_share"] == 1.0
    hand_rows = label_rows(journals, "hand")
    assert {row["side"] for row in hand_rows} == {"LEFT", "RIGHT"}
    assert set(MOUTH_FIELDS) <= set(label_rows(journals, "mouth")[0])


def _phoenix_fixtures(tmp_path):
    import io
    import tarfile

    ann = tmp_path / "train_info_ml.npy"
    clips = {
        "A-1": "Signer01",
        "B-2": "Signer02",
        "D-4": "Signer04",
        "S3": "Signer03",
        "S7": "Signer07",
    }
    raw = {"prefix": "x"}
    serials = {}
    for position, (clip, signer) in enumerate(clips.items()):
        raw[position] = {"fileid": clip, "signer": signer, "num_frames": "9"}
        serials[clip] = [f"images{i:04d}" for i in range(9)]
    import numpy as np

    np.save(ann, raw)
    tar = tmp_path / "release.tar.gz"
    prefix = "root/PHOENIX-2014-T/features/fullFrame-210x260px/train"
    with tarfile.open(tar, "w:gz") as archive:
        for clip, names in serials.items():
            for serial in names:
                data = cv2.imencode(
                    ".png", np.full((260, 210, 3), len(clip) + int(serial[-1]), np.uint8)
                )[1].tobytes()
                info = tarfile.TarInfo(f"{prefix}/{clip}/{serial}.png")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return ann, tar


def test_phoenix_stages(tmp_path, monkeypatch, capsys):
    import json as _json

    ann, tar = _phoenix_fixtures(tmp_path)
    root = tmp_path / "pprobe"
    monkeypatch.setattr(cli, "package_code_sha256", lambda: "deadbeef")
    monkeypatch.setattr(cli, "sha256_file", lambda path: "0" * 64)
    monkeypatch.setattr(cli, "ollama_model_digest", lambda model, port=11434: "digest")
    cli.main(
        [
            "--root",
            str(root),
            "sample-phoenix",
            "--annotations",
            str(ann),
            "--tar",
            str(tar),
            "--total",
            "3",
            "--seed",
            "0",
        ]
    )
    sample = _json.loads((root / "sample.json").read_text())
    assert sample["source"] == "phoenix-tar"
    assert len(sample["clips"]) == 3
    assert {c["stratum"] for c in sample["clips"]} <= {"Signer01", "Signer02", "Signer04"}

    cli.main(["--root", str(root), "extract", "--tar", str(tar)])
    plan = _json.loads((root / "extract_plan.json").read_text())
    assert all(len(serials) == 5 for serials in plan.values())
    assert sum(len(p.stem) for p in (root / "frames").glob("*/*.png")) == 5 * len(plan) * len(
        "images0000"
    )

    monkeypatch.setattr(cli, "MIN_DETECTOR_FRAMES", 1)

    def fake_detect(image):
        from how2sign_probe.detector import Region

        assert image.shape[0] == 260 and image.shape[1] == 210
        return [
            Region("hand", "LEFT", (10, 10, 60, 60), 0.9),
            Region("hand", "RIGHT", (10, 100, 60, 150), 0.8),
            Region("mouth", "MOUTH", (80, 20, 130, 60), 0.7),
        ]

    monkeypatch.setattr(cli, "detect_regions", fake_detect)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--root", str(root), "detect"])
    assert exit_info.value.code == 0
    report = _json.loads((root / "detector_report.json").read_text())
    assert report["gate"]["pass"] is True
    assert report["stats"]["frames"] == 15


def _phoenix_dir(tmp_path):
    root = tmp_path / "fullFrame"
    clips = {
        "A-1": "Signer01",
        "B-2": "Signer02",
        "D-4": "Signer04",
        "S3": "Signer03",
        "S7": "Signer07",
    }
    raw = {"prefix": str(root) + "/"}
    for position, (clip, signer) in enumerate(clips.items()):
        directory = root / "train" / clip
        directory.mkdir(parents=True)
        for i in range(1, 10):
            cv2.imwrite(
                str(directory / f"images{i:04d}.png"),
                np.full((260, 210, 3), position + 3, np.uint8),
            )
        raw[position] = {
            "fileid": clip,
            "folder": f"train/{clip}/*.png",
            "signer": signer,
            "num_frames": "9",
            "text": "x",
        }
    import numpy as _np

    ann = tmp_path / "train_info_ml.npy"
    _np.save(ann, raw)
    return ann, root


def test_phoenix_dir_stages(tmp_path, monkeypatch):
    import json as _json

    ann, root = _phoenix_dir(tmp_path)
    probe = tmp_path / "dprobe"
    monkeypatch.setattr(cli, "package_code_sha256", lambda: "cafe")
    monkeypatch.setattr(cli, "sha256_file", lambda path: "1" * 64)
    monkeypatch.setattr(cli, "ollama_model_digest", lambda model, port=11434: "d")
    cli.main(
        [
            "--root",
            str(probe),
            "sample-phoenix",
            "--annotations",
            str(ann),
            "--frames-root",
            str(root),
            "--total",
            "3",
            "--seed",
            "0",
        ]
    )
    sample = _json.loads((probe / "sample.json").read_text())
    assert sample["source"] == "phoenix-dir"
    assert sample["frames_root"] == str(root)
    assert {c["stratum"] for c in sample["clips"]} <= {"Signer01", "Signer02", "Signer04"}

    monkeypatch.setattr(cli, "MIN_DETECTOR_FRAMES", 1)

    def fake_detect(image):
        from how2sign_probe.detector import Region

        assert image.shape[0] == 260
        return [
            Region("hand", "LEFT", (10, 10, 60, 60), 0.9),
            Region("hand", "RIGHT", (10, 100, 60, 150), 0.8),
            Region("mouth", "MOUTH", (80, 20, 130, 60), 0.7),
        ]

    monkeypatch.setattr(cli, "detect_regions", fake_detect)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--root", str(probe), "detect"])
    assert exit_info.value.code == 0
    report = _json.loads((probe / "detector_report.json").read_text())
    assert report["stats"]["frames"] == 15
    assert not (probe / "extract_plan.json").exists()
    # crops are written without any tar extraction step
    assert len(list((probe / "crops").glob("*.png"))) == 45

    # extract must refuse dir-sourced samples
    with pytest.raises(SystemExit):
        cli.main(["--root", str(probe), "extract", "--tar", str(tmp_path / "none.tar")])


def test_sample_phoenix_include_all_signers(tmp_path, monkeypatch, capsys):
    ann, root = _phoenix_dir(tmp_path)
    probe = tmp_path / "allp"
    monkeypatch.setattr(cli, "package_code_sha256", lambda: "cafe")
    monkeypatch.setattr(cli, "sha256_file", lambda path: "1" * 64)
    monkeypatch.setattr(cli, "ollama_model_digest", lambda model, port=11434: "d")
    cli.main(
        [
            "--root",
            str(probe),
            "sample-phoenix",
            "--annotations",
            str(ann),
            "--frames-root",
            str(root),
            "--total",
            "5",
            "--seed",
            "0",
            "--include-all-signers",
        ]
    )
    sample = json.loads((probe / "sample.json").read_text())
    assert len(sample["clips"]) == 5
    assert {c["stratum"] for c in sample["clips"]} == {
        "Signer01",
        "Signer02",
        "Signer04",
        "Signer03",
        "Signer07",
    }
    provenance = json.loads((probe / "provenance.json").read_text())
    assert provenance["signers_excluded"] == []


def test_sample_phoenix_warns_when_total_exceeds_population(tmp_path, monkeypatch, capsys):
    ann, root = _phoenix_dir(tmp_path)
    probe = tmp_path / "warnp"
    monkeypatch.setattr(cli, "package_code_sha256", lambda: "cafe")
    monkeypatch.setattr(cli, "sha256_file", lambda path: "1" * 64)
    monkeypatch.setattr(cli, "ollama_model_digest", lambda model, port=11434: "d")
    cli.main(
        [
            "--root",
            str(probe),
            "sample-phoenix",
            "--annotations",
            str(ann),
            "--frames-root",
            str(root),
            "--total",
            "50",
            "--seed",
            "0",
        ]
    )
    assert "eligible population is 3" in capsys.readouterr().err


def test_detect_resume_excludes_stale_region_keys(tmp_path, monkeypatch):
    import json as _json

    ann, root = _phoenix_dir(tmp_path)
    probe = tmp_path / "stale"
    monkeypatch.setattr(cli, "package_code_sha256", lambda: "cafe")
    monkeypatch.setattr(cli, "sha256_file", lambda path: "1" * 64)
    monkeypatch.setattr(cli, "ollama_model_digest", lambda model, port=11434: "d")
    cli.main(["--root", str(probe), "sample-phoenix", "--annotations", str(ann),
              "--frames-root", str(root), "--total", "3", "--seed", "0"])
    monkeypatch.setattr(cli, "MIN_DETECTOR_FRAMES", 1)

    def fake_detect(image):
        from how2sign_probe.detector import Region

        return [
            Region("hand", "LEFT", (10, 10, 60, 60), 0.9),
            Region("hand", "RIGHT", (10, 100, 60, 150), 0.8),
            Region("mouth", "MOUTH", (80, 20, 130, 60), 0.7),
        ]

    monkeypatch.setattr(cli, "detect_regions", fake_detect)
    with pytest.raises(SystemExit) as e1:
        cli.main(["--root", str(probe), "detect"])
    assert e1.value.code == 0

    regions = _json.loads((probe / "regions.json").read_text())
    regions["ghost_clip|3"] = [["hand", "LEFT", [0, 0, 50, 50], 0.9]]
    (probe / "regions.json").write_text(_json.dumps(regions))

    with pytest.raises(SystemExit) as e2:
        cli.main(["--root", str(probe), "detect"])
    assert e2.value.code == 0
    report = _json.loads((probe / "detector_report.json").read_text())
    assert report["stats"]["frames"] == 15
    assert "ghost_clip" not in _json.loads((probe / "regions.json").read_text())


def test_detect_workers_parallel_matches_sequential(tmp_path, monkeypatch):
    ann, root = _phoenix_dir(tmp_path)
    monkeypatch.setattr(cli, "package_code_sha256", lambda: "cafe")
    monkeypatch.setattr(cli, "sha256_file", lambda path: "1" * 64)
    monkeypatch.setattr(cli, "ollama_model_digest", lambda model, port=11434: "d")

    def fake_detect(image):
        from how2sign_probe.detector import Region

        return [
            Region("hand", "LEFT", (10, 10, 60, 60), 0.9),
            Region("hand", "RIGHT", (10, 100, 60, 150), 0.8),
            Region("mouth", "MOUTH", (80, 20, 130, 60), 0.7),
        ]

    monkeypatch.setattr(cli, "detect_regions", fake_detect)
    monkeypatch.setattr(cli, "MIN_DETECTOR_FRAMES", 1)
    outputs = []
    for workers in (1, 4):
        probe = tmp_path / f"cmp{workers}"
        cli.main(["--root", str(probe), "sample-phoenix", "--annotations", str(ann),
                  "--frames-root", str(root), "--total", "3", "--seed", "0"])
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["--root", str(probe), "detect", "--workers", str(workers)])
        assert exit_info.value.code == 0
        outputs.append(
            (
                (probe / "regions.json").read_text(),
                sorted(px.name for px in (probe / "crops").glob("*.png")),
            )
        )
    assert outputs[0] == outputs[1]
    assert len(outputs[0][1]) == 45
