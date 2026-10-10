"""CPU fixtures exercise actual newly written bytes, never downloaded cache history."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from despamo.appearance.provenance import atomic_json, digest, file_hash


def api():
    return importlib.import_module("tools.motion.records"), importlib.import_module(
        "tools.motion.package"
    )


def test_consumer_validation_matches_producer_without_rehashing_large_identity(tmp_path,
                                                                            monkeypatch):
    from despamo.data import motion_validation as fast

    _, package, root, identity, spatial, annotation = fixture(tmp_path, count=25)
    expected = package.publish_motion_package(root, identity, spatial, annotation)
    original = fast._clip
    keys = []

    def reconstruct(*args):
        keys.append(args[4])
        return original(*args)

    monkeypatch.setattr(fast, "_clip", reconstruct)
    assert fast.validate_motion_package(root, spatial) == expected
    assert keys == [digest(identity)]


@pytest.mark.parametrize("changed", ["source", "array", "receipt", "marker"])
def test_optimized_consumer_still_rejects_changed_bytes(tmp_path, changed):
    from despamo.data.motion_validation import validate_motion_package

    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    package.publish_motion_package(root, identity, spatial, annotation)
    paths = {"source": tmp_path / "sources/00.png", "array": root / "train/c.npy",
             "receipt": root / "receipts/train/c.json",
             "marker": root / "complete/completion.json"}
    with paths[changed].open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        validate_motion_package(root, spatial)


def test_same_transaction_reuses_verified_admission_but_detects_drift(tmp_path, monkeypatch):
    from despamo.data import motion_validation as fast

    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    package.publish_motion_package(root, identity, spatial, annotation)
    admitted = fast.validate_motion_package(root, spatial)
    monkeypatch.setattr(fast, "validate_motion_package",
                        lambda *a, **kw: pytest.fail("duplicate full byte scan"))
    assert fast.reuse_motion_admission(admitted, root, spatial) is admitted
    (tmp_path / "sources/00.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        fast.reuse_motion_admission(admitted, root, spatial)


def test_plain_descriptor_never_claims_reusable_admission(tmp_path, monkeypatch):
    from despamo.data import motion_validation as fast

    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    plain = package.publish_motion_package(root, identity, spatial, annotation)
    monkeypatch.setattr(fast, "validate_motion_package", lambda *a, **kw: "full-scan")
    assert fast.reuse_motion_admission(plain, root, spatial) == "full-scan"


def test_windows_preserve_actual_input_order():
    records, _ = api()
    short = records.window_rows(10)
    assert short == ({"row": 0, "input_source_indices": list(range(10)) + [9] * 6,
                      "real_start": 0, "real_end": 9, "centre": 7.5},)
    rows = records.window_rows(25)
    assert [r["input_source_indices"] for r in rows] == [list(range(16)), list(range(8, 24))]
    assert [r["centre"] for r in rows] == [7.5, 15.5]
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            records.window_rows(count)


def test_source_bytes_match_spatial_receipt(tmp_path):
    records, _ = api()
    paths = (Path("train/c/00.png"), Path("train/c/01.png"))
    hashes = ("a" * 64, "b" * 64)
    assert records.source_identity(paths, hashes) == digest(
        [("train/c/00.png", hashes[0]), ("train/c/01.png", hashes[1])]
    )
    assert records.source_identity(paths[::-1], hashes[::-1]) != records.source_identity(
        paths, hashes
    )


def fixture(tmp_path, count=10, absolute_sources=False, environment_extra=None):
    records, package = api()
    sources = tmp_path / "sources"
    sources.mkdir()
    rows = []
    for index in range(count):
        path = sources / f"{index:02}.png"
        path.write_bytes(bytes([index]))
        rows.append({"index": index, "path": str(path) if absolute_sources else path.name,
                     "sha256": file_hash(path)})
    spatial = {"encoder_key": "spatial", "clips": {"c": {
        "frame_count": count, "paths": [r["path"] for r in rows],
        "source_indices": list(range(count)), "feature_hash": "f" * 64,
        "source_hash": records.source_identity(tuple(Path(r["path"]) for r in rows),
                                                tuple(r["sha256"] for r in rows)),
    }}}
    annotation = tmp_path / "annotation.json"
    atomic_json(annotation, {"physical_split": "train", "clip_ids": ["c"]})
    frame_path = tmp_path / "frames.json"
    atomic_json(frame_path, spatial)
    evidence = {}
    for name in ("extractor", "helpers", "lock"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name)
        evidence[name] = {"path": str(path), "sha256": file_hash(path)}
    identity = records.make_motion_identity(
        {"frame_root": str(sources), "frame_rows": {"path": str(frame_path),
         "sha256": file_hash(frame_path)}, "annotation": {"path": str(annotation),
         "sha256": file_hash(annotation)}},
        dict(records.MODEL), dict(records.PROCESSOR),
        {"lock": evidence.pop("lock"), **(environment_extra or {})},
        evidence, {"device": "cpu-fixture"},
    )
    root = tmp_path / f"motion-features-{digest(identity)}"
    (root / "train").mkdir(parents=True)
    array = root / "train/c.npy"
    np.save(array, np.ones((1 if count < 16 else 2, 1024), dtype=np.float32))
    receipt = records.make_clip_receipt("c", spatial["clips"]["c"], rows,
                                        records.window_rows(count), array, identity)
    atomic_json(root / "receipts/train/c.json", receipt)
    atomic_json(root / "generating-run.json", {
        "schema_version": 1, "origin": "new_extraction_v1", "featurekey": digest(identity),
        "authorization": {"extraction_cap_seconds": 18000, "training_cap_seconds": 64800,
                          "cap_transfer": False},
        "code": identity["code"], "clips": {"c": file_hash(root / "receipts/train/c.json")},
    })
    return records, package, root, identity, spatial, annotation


def test_complete_marker_requires_all_bytes(tmp_path):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    descriptor = package.publish_motion_package(root, identity, spatial, annotation)
    assert descriptor == package.validate_motion_package(root, spatial)
    assert descriptor["clip_ids"] == ["c"]
    assert descriptor["package_policy"] == "despamo-motion-features-v1"
    assert str(root / "complete/completion.json") in descriptor["bindings"]
    assert str(root / "train/c.npy") in descriptor["bindings"]
    assert len(descriptor["bindings"]) == 24  # 10 sources + 5 inputs + 2 clips + run + 6 complete
    timing = json.loads(Path(descriptor["timing_path"]).read_text())
    assert timing["clips"]["c"]["windows"] == [[0, 9]]
    assert timing["provenance"]["origin"] == "new_extraction_v1"


@pytest.mark.parametrize("target", ["complete/completion.json", "train/c.npy",
                                    "receipts/train/c.json"])
def test_missing_bytes_rejected(tmp_path, target):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    package.publish_motion_package(root, identity, spatial, annotation)
    (root / target).unlink()
    with pytest.raises(ValueError):
        package.validate_motion_package(root, spatial)


@pytest.mark.parametrize("mutation", ["permuted", "repeats", "foreign", "width", "dtype",
                                      "nonfinite", "source", "ledger"])
def test_bad_generation_never_publishes_completion(tmp_path, mutation):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    path = root / "receipts/train/c.json"
    receipt = json.loads(path.read_text())
    if mutation == "permuted":
        receipt["windows"][0]["input_source_indices"][0:2] = [1, 0]
    elif mutation == "repeats":
        receipt["source_indices"] += [9] * 6
    elif mutation == "foreign":
        receipt["clip_id"] = "foreign"
    elif mutation in {"width", "dtype", "nonfinite"}:
        value = np.ones((1, 1000 if mutation == "width" else 1024),
                        dtype=np.float64 if mutation == "dtype" else np.float32)
        if mutation == "nonfinite":
            value[0, 0] = np.nan
        np.save(root / "train/c.npy", value)
        receipt["array"]["sha256"] = file_hash(root / "train/c.npy")
    elif mutation == "source":
        (tmp_path / "sources/00.png").write_bytes(b"changed")
    elif mutation == "ledger":
        run = json.loads((root / "generating-run.json").read_text())
        run["origin"] = "old_downloaded_cache"
        atomic_json(root / "generating-run.json", run)
    atomic_json(path, receipt)
    run = json.loads((root / "generating-run.json").read_text())
    run["clips"]["c"] = file_hash(path)
    atomic_json(root / "generating-run.json", run)
    with pytest.raises(ValueError):
        package.publish_motion_package(root, identity, spatial, annotation)
    assert not (root / "complete/completion.json").exists()


@pytest.mark.parametrize("target", ["identity.json", "manifest.json", "motion-timing.json",
                                    "extraction-provenance.json", "source-window-records.json"])
def test_tampered_artifacts_rejected(tmp_path, target):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    package.publish_motion_package(root, identity, spatial, annotation)
    (root / "complete" / target).write_text("{}")
    with pytest.raises(ValueError):
        package.validate_motion_package(root, spatial, verify_features=False)


def test_completion_rehash_does_not_admit_foreign_ids(tmp_path):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    package.publish_motion_package(root, identity, spatial, annotation)
    marker = root / "complete/completion.json"
    value = json.loads(marker.read_text())
    value["clip_ids"] = ["foreign"]
    atomic_json(marker, value)
    with pytest.raises(ValueError):
        package.validate_motion_package(root, spatial)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "foreignroot"])
def test_aliases_and_foreign_roots_rejected(tmp_path, kind):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    if kind == "foreignroot":
        root = root.rename(tmp_path / "foreign")
    else:
        path = root / "train/c.npy"
        other = tmp_path / "other.npy"
        path.rename(other)
        if kind == "symlink":
            path.symlink_to(other)
        else:
            path.hardlink_to(other)
    with pytest.raises(ValueError):
        package.publish_motion_package(root, identity, spatial, annotation)


def test_runtime_clocks_excluded_and_identity_recipe_enforced(tmp_path):
    records, _, _, identity, _, _ = fixture(tmp_path)
    arguments = [identity[key] for key in
                 ("inputs", "model", "processor", "environment", "code", "hardware")]
    with pytest.raises(ValueError):
        records.make_motion_identity(*[arguments[0] | {"elapsed": 1}, *arguments[1:]])
    with pytest.raises(ValueError):
        records.make_motion_identity(arguments[0], arguments[1] | {"tokens": "pooling"},
                                     *arguments[2:])


@pytest.mark.parametrize("mutation", ["omit", "foreign", "source", "receipt", "code", "lock"])
def test_rehashed_marker_cannot_hide_bad_byte_inventory(tmp_path, mutation):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    package.publish_motion_package(root, identity, spatial, annotation)
    marker_path = root / "complete/completion.json"
    marker = json.loads(marker_path.read_text())
    if mutation == "omit":
        marker["bindings"].pop(str(root / "train/c.npy"))
    elif mutation == "foreign":
        path = tmp_path / "foreign.txt"
        path.write_text("unrelated")
        marker["bindings"][str(path)] = file_hash(path)
    else:
        path = {"source": tmp_path / "sources/00.png",
                "receipt": root / "receipts/train/c.json",
                "code": tmp_path / "extractor.txt", "lock": tmp_path / "lock.txt"}[mutation]
        path.write_text("changed")
        marker["bindings"][str(path)] = file_hash(path)
    atomic_json(marker_path, marker)
    with pytest.raises(ValueError):
        package.validate_motion_package(root, spatial, verify_features=False)


def test_boolean_ordinals_cannot_impersonate_actual_indices(tmp_path):
    records, _, root, identity, spatial, _ = fixture(tmp_path)
    receipt = json.loads((root / "receipts/train/c.json").read_text())
    receipt["source_rows"][0]["index"] = False
    with pytest.raises(ValueError):
        records.make_clip_receipt("c", spatial["clips"]["c"], receipt["source_rows"],
                                  records.window_rows(10), root / "train/c.npy", identity)


def test_dropped_tail_still_binds_all_source_bytes(tmp_path):
    _, package, root, identity, spatial, annotation = fixture(tmp_path, count=25)
    result = package.publish_motion_package(root, identity, spatial, annotation)
    timing = json.loads(Path(result["timing_path"]).read_text())
    assert timing["clips"]["c"]["windows"] == [[0, 15], [8, 23]]
    assert timing["coverage"]["uncovered_frame_count"] == 1
    assert str(tmp_path / "sources/24.png") in result["bindings"]
    (tmp_path / "sources/24.png").write_bytes(b"changed even though uncovered")
    with pytest.raises(ValueError):
        package.validate_motion_package(root, spatial)


def test_mutable_runtime_clocks_do_not_change_content_descriptor(tmp_path):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    result = package.publish_motion_package(root, identity, spatial, annotation)
    atomic_json(root / "runtime-status.json", {"featurekey": digest(identity), "elapsed": 2})
    assert package.validate_motion_package(root, spatial) == result


def test_existing_complete_package_cannot_be_republished(tmp_path):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    result = package.publish_motion_package(root, identity, spatial, annotation)
    with pytest.raises(ValueError):
        package.publish_motion_package(root, identity, spatial, annotation)
    assert package.validate_motion_package(root, spatial) == result


def test_absolute_spatial_paths_preserved_without_rebasing(tmp_path):
    _, package, root, identity, spatial, annotation = fixture(tmp_path, absolute_sources=True)
    descriptor = package.publish_motion_package(root, identity, spatial, annotation)
    receipt = json.loads((root / "receipts/train/c.json").read_text())
    assert receipt["source_rows"][0]["path"] == str(tmp_path / "sources/00.png")
    assert str(tmp_path / "sources/00.png") in descriptor["bindings"]


def test_extra_model_input_evidence_is_not_ignored(tmp_path):
    records, package, root, identity, spatial, annotation = fixture(tmp_path)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"synthetic model input")
    identity["environment"]["model_files"] = [{"path": str(weights), "sha256": file_hash(weights)}]
    new_root = root.rename(tmp_path / f"motion-features-{digest(identity)}")
    receipt_path = new_root / "receipts/train/c.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["featurekey"] = digest(identity)
    atomic_json(receipt_path, receipt)
    run = json.loads((new_root / "generating-run.json").read_text())
    run["featurekey"] = digest(identity)
    run["clips"]["c"] = file_hash(receipt_path)
    atomic_json(new_root / "generating-run.json", run)
    result = package.publish_motion_package(new_root, identity, spatial, annotation)
    assert result["bindings"][str(weights)] == file_hash(weights)
    weights.write_bytes(b"changed model bytes")
    with pytest.raises(ValueError):
        package.validate_motion_package(new_root, spatial)


@pytest.mark.parametrize("constructed", [True, False])
def test_missing_nested_tuple_evidence_rejected_before_complete_publication(tmp_path, constructed):
    missing = {"path": str(tmp_path / "missing-weights.bin"), "sha256": "a" * 64}
    extra = {"model_files": ((missing,),)}
    _, package, root, identity, spatial, annotation = fixture(
        tmp_path, environment_extra=extra
    )
    if not constructed:
        # Admission must also handle a caller replacing canonical lists with tuples.
        identity["environment"]["model_files"] = ((missing,),)
    with pytest.raises(ValueError):
        package.publish_motion_package(root, identity, spatial, annotation)
    assert not (root / "complete/completion.json").exists()
    assert not (root / "complete").exists()


def test_nested_tuple_identity_is_canonical_and_publisher_validator_agree(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"synthetic model bytes")
    evidence = {"path": str(weights), "sha256": file_hash(weights)}
    extra = {"model_files": ((evidence,),)}
    _, package, root, identity, spatial, annotation = fixture(tmp_path, environment_extra=extra)
    assert identity["environment"]["model_files"] == [[evidence]]
    assert identity == json.loads(json.dumps(identity))
    # Returned identity must not retain caller-owned nested containers.
    evidence["sha256"] = "b" * 64
    result = package.publish_motion_package(root, identity, spatial, annotation)
    assert result["identity"] == identity
    assert result["bindings"][str(weights)] == file_hash(weights)
    assert package.validate_motion_package(root, spatial) == result


def test_factual_compute_capability_is_accepted(tmp_path):
    records, _, _, identity, _, _ = fixture(tmp_path)
    result = records.make_motion_identity(
        identity["inputs"], identity["model"], identity["processor"],
        identity["environment"], identity["code"], {"compute_capability": (8, 6)}
    )
    assert result["hardware"] == {"compute_capability": [8, 6]}


@pytest.mark.parametrize("key", ["budget", "elapsed", "cap", "output", "authorization", "auth",
                                 "extraction_cap_seconds", "training_cap_seconds", "cap_transfer",
                                 "elapsed_seconds", "output_directory", "runtime_policy"])
def test_explicit_runtime_policy_keys_remain_excluded(tmp_path, key):
    records, _, _, identity, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        records.make_motion_identity(
            identity["inputs"], identity["model"], identity["processor"],
            identity["environment"], identity["code"], {"nested": [{key: 1}]}
        )


def interrupted_publication(tmp_path, monkeypatch, stage="first"):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    original = package.atomic_json

    def interrupt(path, value):
        if stage == "marker" and path.name == "completion.json":
            raise OSError("simulated interruption before marker")
        original(path, value)
        if stage == "first" and path.name == "identity.json":
            raise OSError("simulated interruption after first complete artifact")

    with monkeypatch.context() as patch:
        patch.setattr(package, "atomic_json", interrupt)
        with pytest.raises(ValueError, match="simulated interruption"):
            package.publish_motion_package(root, identity, spatial, annotation)
    assert not (root / "complete/completion.json").exists()
    return package, root, identity, spatial, annotation


def immutable_snapshot(root):
    return {str(path.relative_to(root)):
            (str(path.readlink()) if path.is_symlink() else path.read_bytes(),
             path.lstat().st_ino, path.lstat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file() or path.is_symlink()}


@pytest.mark.parametrize("stage", ["first", "marker"])
def test_resume_actual_interruption_fills_only_missing_artifacts(tmp_path, monkeypatch, stage):
    package, root, identity, spatial, annotation = interrupted_publication(
        tmp_path, monkeypatch, stage
    )
    before = immutable_snapshot(root)
    with pytest.raises(ValueError):
        package.publish_motion_package(root, identity, spatial, annotation)
    assert immutable_snapshot(root) == before
    result = package.resume_motion_package_publication(root, identity, spatial, annotation)
    assert result == package.validate_motion_package(root, spatial)
    after = immutable_snapshot(root)
    assert {name: after[name] for name in before} == before
    assert result["identity"] == identity


def test_resume_adopts_complete_package_without_rewrite(tmp_path):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    result = package.publish_motion_package(root, identity, spatial, annotation)
    before = immutable_snapshot(root)
    assert package.resume_motion_package_publication(root, identity, spatial, annotation) == result
    assert immutable_snapshot(root) == before
    with pytest.raises(ValueError):
        package.publish_motion_package(root, identity, spatial, annotation)


@pytest.mark.parametrize("mutation", ["foreign", "corrupt", "malformed", "noncanonical",
                                      "conflict", "symlink", "hardlink", "marker"])
def test_resume_invalid_partial_preserves_every_existing_byte(tmp_path, monkeypatch, mutation):
    package, root, identity, spatial, annotation = interrupted_publication(tmp_path, monkeypatch)
    path = root / "complete/identity.json"
    if mutation == "foreign":
        (root / "complete/foreign.json").write_text("foreign")
    elif mutation == "corrupt":
        atomic_json(path, {})
    elif mutation == "malformed":
        path.write_text("{")
    elif mutation == "noncanonical":
        path.write_text(json.dumps(identity))
    elif mutation == "conflict":
        atomic_json(root / "complete/motion-timing.json", {})
    elif mutation in {"symlink", "hardlink"}:
        target = tmp_path / "aliased-identity.json"
        path.rename(target)
        if mutation == "symlink":
            path.symlink_to(target)
        else:
            path.hardlink_to(target)
    else:
        atomic_json(root / "complete/completion.json", {})
    before = immutable_snapshot(root)
    with pytest.raises(ValueError):
        package.resume_motion_package_publication(root, identity, spatial, annotation)
    assert immutable_snapshot(root) == before


@pytest.mark.parametrize("target", ["sources/00.png", "extractor.txt", "lock.txt",
                                   "annotation.json", "ledger"])
def test_resume_rejects_generation_or_evidence_drift_before_writes(tmp_path, monkeypatch, target):
    package, root, identity, spatial, annotation = interrupted_publication(tmp_path, monkeypatch)
    path = root / "generating-run.json" if target == "ledger" else tmp_path / target
    path.write_text("drift")
    before = immutable_snapshot(root)
    with pytest.raises(ValueError):
        package.resume_motion_package_publication(root, identity, spatial, annotation)
    assert immutable_snapshot(root) == before


def test_resume_rejects_aliased_complete_directory(tmp_path, monkeypatch):
    package, root, identity, spatial, annotation = interrupted_publication(tmp_path, monkeypatch)
    complete = root / "complete"
    target = complete.rename(tmp_path / "aliased-complete")
    complete.symlink_to(target, target_is_directory=True)
    before = immutable_snapshot(target)
    with pytest.raises(ValueError):
        package.resume_motion_package_publication(root, identity, spatial, annotation)
    assert complete.is_symlink()
    assert immutable_snapshot(target) == before


@pytest.mark.parametrize("layout", ["absent", "empty"])
def test_resume_missing_all_metadata_from_verified_generation(tmp_path, layout):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    if layout == "empty":
        (root / "complete").mkdir()
    result = package.resume_motion_package_publication(root, identity, spatial, annotation)
    assert result == package.validate_motion_package(root, spatial)


def test_resume_corrupt_completed_package_is_never_repaired(tmp_path):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    package.publish_motion_package(root, identity, spatial, annotation)
    (root / "complete/motion-timing.json").write_text("{}")
    before = immutable_snapshot(root)
    with pytest.raises(ValueError):
        package.resume_motion_package_publication(root, identity, spatial, annotation)
    assert immutable_snapshot(root) == before


@pytest.mark.parametrize("resume", [False, True])
def test_candidate_directory_readback_failure_precedes_marker(tmp_path, monkeypatch, resume):
    if resume:
        package, root, identity, spatial, annotation = interrupted_publication(
            tmp_path, monkeypatch, "marker"
        )
    else:
        _, package, root, identity, spatial, annotation = fixture(tmp_path)
    original = Path.iterdir

    def interrupted_read(path):
        if path == root / "complete":
            raise InterruptedError("candidate directory readback failed")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "iterdir", interrupted_read)
        with pytest.raises(ValueError, match="candidate directory readback failed"):
            function = (package.resume_motion_package_publication if resume
                        else package.publish_motion_package)
            function(root, identity, spatial, annotation)
    assert not (root / "complete/completion.json").exists()
    assert len(list((root / "complete").iterdir())) == 5
    result = package.resume_motion_package_publication(root, identity, spatial, annotation)
    assert result == package.validate_motion_package(root, spatial)


def test_foreign_candidate_entry_never_commits_marker(tmp_path, monkeypatch):
    _, package, root, identity, spatial, annotation = fixture(tmp_path)
    original = package.atomic_json
    guards = []

    def guard():
        guards.append("called")

    def inject_foreign(path, value):
        original(path, value)
        if path.name == "motion-timing.json":
            (root / "complete/foreign.json").write_text("foreign bytes")

    with monkeypatch.context() as patch:
        patch.setattr(package, "atomic_json", inject_foreign)
        with pytest.raises(ValueError, match="foreign complete artifacts"):
            package.publish_motion_package(root, identity, spatial, annotation, before_commit=guard)
    assert guards == []
    assert not (root / "complete/completion.json").exists()
    before = immutable_snapshot(root)
    with pytest.raises(ValueError):
        package.resume_motion_package_publication(root, identity, spatial, annotation)
    assert immutable_snapshot(root) == before


@pytest.mark.parametrize("resume", [False, True])
def test_deadline_callback_failure_preserves_verified_partial_without_marker(
    tmp_path, monkeypatch, resume
):
    if resume:
        package, root, identity, spatial, annotation = interrupted_publication(
            tmp_path, monkeypatch, "marker"
        )
    else:
        _, package, root, identity, spatial, annotation = fixture(tmp_path)
    observed = {}

    def expired():
        assert not (root / "complete/completion.json").exists()
        assert {p.name for p in (root / "complete").iterdir()} == set(package.ARTIFACTS)
        observed.update(immutable_snapshot(root))
        raise TimeoutError("insufficient commit margin")

    function = (package.resume_motion_package_publication if resume
                else package.publish_motion_package)
    with pytest.raises(TimeoutError, match="insufficient commit margin"):
        function(root, identity, spatial, annotation, before_commit=expired)
    assert immutable_snapshot(root) == observed
    result = package.resume_motion_package_publication(root, identity, spatial, annotation)
    assert result == package.validate_motion_package(root, spatial)


@pytest.mark.parametrize("resume", [False, True])
def test_commit_guard_follows_full_candidate_checks_without_postcommit_readmission(
    tmp_path, monkeypatch, resume
):
    if resume:
        package, root, identity, spatial, annotation = interrupted_publication(
            tmp_path, monkeypatch, "marker"
        )
    else:
        _, package, root, identity, spatial, annotation = fixture(tmp_path)
    events = []
    original = package.atomic_json

    def guard():
        events.append("guard")
        assert not (root / "complete/completion.json").exists()
        assert {p.name for p in (root / "complete").iterdir()} == set(package.ARTIFACTS)

    def write(path, value):
        if path.name == "completion.json":
            assert events == ["guard"]
            events.append("commit")
        original(path, value)

    def forbidden_readmission(*args, **kwargs):
        raise InterruptedError("full readmission must not follow commit")

    with monkeypatch.context() as patch:
        patch.setattr(package, "atomic_json", write)
        patch.setattr(package, "validate_motion_package", forbidden_readmission)
        function = (package.resume_motion_package_publication if resume
                    else package.publish_motion_package)
        result = function(root, identity, spatial, annotation, before_commit=guard)
    assert events == ["guard", "commit"]
    assert result["completion_hash"] == file_hash(root / "complete/completion.json")
    assert result["bindings"][result["completion"]] == result["completion_hash"]
    assert result == package.validate_motion_package(root, spatial)
    before = immutable_snapshot(root)

    def must_not_commit():
        raise AssertionError("adoption is not a marker commit")

    assert package.resume_motion_package_publication(
        root, identity, spatial, annotation, before_commit=must_not_commit
    ) == result
    assert immutable_snapshot(root) == before
