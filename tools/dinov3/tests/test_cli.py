import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pytest

from tools.dinov3.cli import main
from tools.dinov3.identity import MODEL, digest


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["cli", *map(str, args)])
    main()


def token_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("UNRELATED=ignored\nHF_TOKEN='hf_private_test'\n")
    monkeypatch.setenv("HF_TOKEN", "")
    return env


def inputs(tmp_path):
    frames = tmp_path / "frames"
    (frames / "train/c").mkdir(parents=True)
    ann = tmp_path / "annotations"
    ann.mkdir()
    records = []
    for split, clip in (("train", "c"), ("dev", "d"), ("test", "t")):
        np.save(ann / f"{split}_info_ml.npy", {0: {"fileid": clip, "num_frames": 5}})
        records.append(
            dict(
                clip_id=clip,
                split=split,
                path=f"{split}/{clip}.npy",
                length=5,
                width=2048,
                dtype="float32",
            )
        )
    manifest = tmp_path / "clip.json"
    manifest.write_text(
        json.dumps(dict(schema_version=1, encoder="clip", expected_dim=2048, records=records))
    )
    return frames, ann, manifest


def extraction_args(tmp_path, env, ann, manifest):
    return [
        "--env-file",
        env,
        "--cache",
        tmp_path / "cache",
        "--output-base",
        tmp_path / "out",
        "--annotation-root",
        ann,
        "--clip-manifest",
        manifest,
        "--one-clip",
        "train",
        "c",
    ]


def test_access_check_fetches_only_config_at_resolved_sha(tmp_path, monkeypatch, capsys):
    env = token_file(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "original-caller-token")
    calls = []

    def resolve(token, revision):
        assert (token, revision) == ("hf_private_test", "candidate")
        return "a" * 40

    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", resolve)
    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download", lambda **kw: calls.append(kw))
    monkeypatch.setattr(
        "tools.dinov3.cli.load_model", lambda *args: pytest.fail("weights downloaded")
    )
    monkeypatch.setattr("tools.dinov3.cli.run", lambda *args, **kw: pytest.fail("pipeline ran"))
    invoke(
        monkeypatch,
        "--env-file",
        env,
        "--revision",
        "candidate",
        "--cache",
        tmp_path,
        "--check-access",
    )
    assert calls == [
        dict(
            repo_id=MODEL,
            filename="config.json",
            revision="a" * 40,
            token="hf_private_test",
            cache_dir=tmp_path,
        )
    ]
    captured = capsys.readouterr()
    assert "a" * 40 in captured.out
    assert "hf_private_test" not in captured.out + captured.err
    assert os.environ["HF_TOKEN"] == "original-caller-token"


@pytest.mark.parametrize("mode", ["--all", "--one-clip"])
def test_extraction_requires_paths_before_access(tmp_path, monkeypatch, mode):
    env = token_file(tmp_path, monkeypatch)
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda *args: pytest.fail("Hub reached"))
    args = ["--env-file", env, "--cache", tmp_path, mode]
    if mode == "--one-clip":
        args += ["train", "c"]
    with pytest.raises(SystemExit):
        invoke(monkeypatch, *args)


def test_full_corpus_requires_confirmation_before_network(tmp_path, monkeypatch, capsys):
    env = token_file(tmp_path, monkeypatch)
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda *args: pytest.fail("Hub reached"))
    with pytest.raises(SystemExit):
        invoke(monkeypatch, "--env-file", env, "--all")
    assert "explicit user authorization" in capsys.readouterr().err


@pytest.mark.parametrize("blocked", ["cache", "output"])
def test_checkout_paths_fail_before_download_or_creation(tmp_path, monkeypatch, blocked):
    checkout = Path(__file__).resolve().parents[3]
    inside = checkout / "not-created-by-extractor"
    env = token_file(tmp_path, monkeypatch)
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda *args: pytest.fail("Hub reached"))
    args = ["--env-file", env]
    if blocked == "cache":
        args += ["--cache", inside, "--check-access"]
    else:
        args += [
            "--cache",
            tmp_path,
            "--output-base",
            inside,
            "--annotation-root",
            tmp_path,
            "--clip-manifest",
            tmp_path,
            "--one-clip",
            "train",
            "c",
        ]
    with pytest.raises(ValueError, match="inside Git checkout"):
        invoke(monkeypatch, *args)
    assert not inside.exists()


def test_one_clip_couples_lock_identity_to_version_root_and_pipeline(tmp_path, monkeypatch, capsys):
    env = token_file(tmp_path, monkeypatch)
    output = tmp_path / "out"
    cache = tmp_path / "cache"
    frame_root, ann, manifest = inputs(tmp_path)
    monkeypatch.setenv("PHOENIX14T_FRAME_ROOT", str(frame_root))
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda token, revision: "a" * 40)
    calls = []
    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download", lambda **kw: calls.append(kw))
    fake_model = object()
    monkeypatch.setattr("tools.dinov3.cli.torch.cuda.is_available", lambda: True)
    cuda_calls = []
    monkeypatch.setattr("tools.dinov3.cli.torch.empty", lambda *a, **kw: cuda_calls.append((a, kw)))

    def fake_load(sha, token, model_cache, device):
        assert (sha, token, model_cache, device) == ("a" * 40, "hf_private_test", cache, "cuda")
        return fake_model

    monkeypatch.setattr("tools.dinov3.cli.load_model", fake_load)

    def fake_run(frames, ann, clip, target, key, metadata, model, device, *, only):
        assert (frames, ann, clip) == (frame_root, tmp_path / "annotations", manifest)
        assert target == output / key
        assert len(key) == 64 and key == digest(metadata)
        assert metadata["model_sha"] == "a" * 40
        assert metadata["lock_sha256"] and metadata["scales"] == [224, 448]
        assert (model, device, only) == (fake_model, "cuda", ("train", "c"))
        assert not (target / ".write-probe.npy").exists()
        return {"complete": False, "completed": 1}

    monkeypatch.setattr("tools.dinov3.cli.run", fake_run)
    invoke(
        monkeypatch,
        "--env-file",
        env,
        "--cache",
        cache,
        "--output-base",
        output,
        "--annotation-root",
        ann,
        "--clip-manifest",
        manifest,
        "--one-clip",
        "train",
        "c",
    )
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "complete": False,
        "completed": 1,
        "encoder_key": result["encoder_key"],
        "model_sha": "a" * 40,
    }
    assert (output / result["encoder_key"]).is_dir()
    assert len(calls) == 1 and calls[0]["filename"] == "config.json"
    assert cuda_calls == [(((1,),), {"device": "cuda"})]


def test_complete_rerun_skips_write_probe_and_model_load(tmp_path, monkeypatch):
    output = tmp_path / "out"
    (output / "key/complete").mkdir(parents=True)
    env = token_file(tmp_path, monkeypatch)
    frames, ann, manifest = inputs(tmp_path)
    monkeypatch.setenv("PHOENIX14T_FRAME_ROOT", str(frames))
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda token, revision: "a" * 40)
    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download", lambda **kw: None)
    monkeypatch.setattr("tools.dinov3.cli.identity", lambda sha, lock: ("key", {"sha": sha}))
    monkeypatch.setattr(
        "tools.dinov3.cli.load_model", lambda *args: pytest.fail("completed version loaded model")
    )
    monkeypatch.setattr(
        "tools.dinov3.cli.torch.cuda.is_available", lambda: pytest.fail("CUDA queried")
    )
    monkeypatch.setattr("tools.dinov3.cli.torch.empty", lambda *a, **kw: pytest.fail("CUDA used"))

    def checked(*args, **kwargs):
        assert args[3] == output / "key" and args[4:8] == ("key", {"sha": "a" * 40}, None, "cpu")
        assert kwargs["only"] == ("train", "c")
        return {"complete": True, "completed": 8257}

    monkeypatch.setattr("tools.dinov3.cli.run", checked)
    invoke(
        monkeypatch,
        "--env-file",
        env,
        "--cache",
        tmp_path / "cache",
        "--output-base",
        output,
        "--annotation-root",
        ann,
        "--clip-manifest",
        manifest,
        "--one-clip",
        "train",
        "c",
    )
    assert not (output / "key/.write-probe.npy").exists()
    (ann / "dev_info_ml.npy").unlink()
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda *a: pytest.fail("Hub queried"))
    with pytest.raises(ValueError, match="missing annotation file"):
        invoke(
            monkeypatch,
            *extraction_args(tmp_path, env, ann, manifest),
        )
    assert not (output / "key/.write-probe.npy").exists()


@pytest.mark.parametrize(
    "missing",
    [
        "frame_env",
        "frame_dir",
        "annotation_dir",
        "train_info_ml.npy",
        "dev_info_ml.npy",
        "test_info_ml.npy",
        "manifest",
        "bad_manifest",
        "clip_id",
        "clip_split",
        "clip_dir",
    ],
)
def test_input_preflight_rejects_before_network_probe_or_model(tmp_path, monkeypatch, missing):
    env = token_file(tmp_path, monkeypatch)
    frames, ann, manifest = inputs(tmp_path)
    monkeypatch.setenv("PHOENIX14T_FRAME_ROOT", str(frames))
    if missing == "frame_env":
        monkeypatch.delenv("PHOENIX14T_FRAME_ROOT")
    elif missing == "frame_dir":
        monkeypatch.setenv("PHOENIX14T_FRAME_ROOT", str(tmp_path / "absent"))
    elif missing == "annotation_dir":
        ann = tmp_path / "absent-annotations"
    elif missing.endswith(".npy"):
        (ann / missing).unlink()
    elif missing == "manifest":
        manifest = tmp_path / "missing.json"
    elif missing == "bad_manifest":
        manifest.write_text("not JSON")
    elif missing == "clip_dir":
        (frames / "train/c").rmdir()
    args = extraction_args(tmp_path, env, ann, manifest)
    if missing == "clip_id":
        args[-1] = "absent"
    if missing == "clip_split":
        args[-2] = "unknown"
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda *a: pytest.fail("revision queried"))
    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download", lambda **kw: pytest.fail("Hub used"))
    monkeypatch.setattr("tools.dinov3.cli.load_model", lambda *a: pytest.fail("model loaded"))
    with pytest.raises((ValueError, FileNotFoundError)):
        invoke(monkeypatch, *args)
    assert not (tmp_path / "out").exists()


def test_full_mode_preflights_inputs_before_hub(tmp_path, monkeypatch):
    env = token_file(tmp_path, monkeypatch)
    frames, ann, manifest = inputs(tmp_path)
    monkeypatch.delenv("PHOENIX14T_FRAME_ROOT", raising=False)
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda *a: pytest.fail("Hub queried"))
    with pytest.raises(ValueError, match="PHOENIX14T_FRAME_ROOT"):
        invoke(
            monkeypatch,
            "--env-file",
            env,
            "--cache",
            tmp_path / "cache",
            "--output-base",
            tmp_path / "out",
            "--annotation-root",
            ann,
            "--clip-manifest",
            manifest,
            "--all",
            "--confirm-full-extraction",
        )
    assert frames.is_dir() and not (tmp_path / "out").exists()


def test_corrupt_frames_reach_pipeline_source_journal(tmp_path, monkeypatch):
    env = token_file(tmp_path, monkeypatch)
    frames, ann, manifest = inputs(tmp_path)
    for index in range(5):
        (frames / "train/c" / f"{index:06}.png").write_bytes(b"broken PNG")
    monkeypatch.setenv("PHOENIX14T_FRAME_ROOT", str(frames))
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda *a: "a" * 40)
    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download", lambda **kw: None)
    monkeypatch.setattr("tools.dinov3.cli.identity", lambda *a: ("key", {"sha": "a" * 40}))
    monkeypatch.setattr("tools.dinov3.cli.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("tools.dinov3.cli.torch.empty", lambda *a, **kw: None)
    monkeypatch.setattr("tools.dinov3.cli.load_model", lambda *a: object())
    with pytest.raises(ValueError, match="bounded extraction incomplete"):
        invoke(monkeypatch, *extraction_args(tmp_path, env, ann, manifest))
    failure = (tmp_path / "out/key/failures.json").read_text()
    assert '"stage": "source"' in failure and "invalid PNG" in failure
    assert "hf_private_test" not in failure
    assert not (tmp_path / "out/key/complete").exists()


@pytest.mark.parametrize("dangling", [False, True])
def test_key_root_symlink_rejected_before_probe_or_model(tmp_path, monkeypatch, dangling):
    env = token_file(tmp_path, monkeypatch)
    frames, ann, manifest = inputs(tmp_path)
    monkeypatch.setenv("PHOENIX14T_FRAME_ROOT", str(frames))
    output = tmp_path / "out"
    output.mkdir()
    redirect = tmp_path / "other-external"
    if not dangling:
        redirect.mkdir()
    (output / "key").symlink_to(redirect, target_is_directory=True)
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda *a: "a" * 40)
    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download", lambda **kw: None)
    monkeypatch.setattr("tools.dinov3.cli.identity", lambda *a: ("key", {"sha": "a" * 40}))
    monkeypatch.setattr("tools.dinov3.cli.atomic_array", lambda *a: pytest.fail("probe written"))
    monkeypatch.setattr("tools.dinov3.cli.load_model", lambda *a: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="key root"):
        invoke(monkeypatch, *extraction_args(tmp_path, env, ann, manifest))
    assert not (redirect / ".writer.lock").exists()
    assert not (redirect / ".write-probe.npy").exists()
    assert redirect.exists() is not dangling


def test_cuda_initialization_failure_is_sanitized_before_model_load(tmp_path, monkeypatch):
    env = token_file(tmp_path, monkeypatch)
    frames, ann, manifest = inputs(tmp_path)
    monkeypatch.setenv("PHOENIX14T_FRAME_ROOT", str(frames))
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda *a: "a" * 40)
    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download", lambda **kw: None)
    monkeypatch.setattr("tools.dinov3.cli.identity", lambda *a: ("key", {"sha": "a" * 40}))
    monkeypatch.setattr("tools.dinov3.cli.torch.cuda.is_available", lambda: True)

    def broken_cuda(*args, **kwargs):
        raise RuntimeError("driver failed with token=hf_private_test")

    monkeypatch.setattr("tools.dinov3.cli.torch.empty", broken_cuda)
    monkeypatch.setattr("tools.dinov3.cli.load_model", lambda *a: pytest.fail("model loaded"))
    with pytest.raises(RuntimeError, match="CUDA initialization failed") as caught:
        invoke(monkeypatch, *extraction_args(tmp_path, env, ann, manifest))
    assert "hf_private_test" not in "".join(traceback.format_exception(caught.value))


def test_access_failure_hides_token_and_upstream_url(tmp_path, monkeypatch, capsys):
    env = token_file(tmp_path, monkeypatch)
    secret_url = "https://invalid.example?token=hf_private_test"
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda token, revision: "a" * 40)

    def broken_download(**kwargs):
        raise RuntimeError(secret_url)

    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download", broken_download)
    with pytest.raises(RuntimeError, match="gated config inaccessible") as caught:
        invoke(monkeypatch, "--env-file", env, "--cache", tmp_path, "--check-access")
    rendered = "".join(traceback.format_exception(caught.value))
    assert "hf_private_test" not in rendered + capsys.readouterr().out
    assert secret_url not in rendered
