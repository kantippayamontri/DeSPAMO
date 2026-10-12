"""One-shot, explicitly authorized diagnostic; never a training recovery runner."""

import argparse
import gc
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import tools.clip_s2_inference as inference
import torch
from tools.benchmark.compact_checkpoint import load_snapshot
from tools.benchmark.feature_package import other_gpu_pids
from tools.clip_s2.data import admit_data, batch_for
from tools.clip_s2.train import BASE, CACHE, code_hashes, load_config, model_files
from tools.clip_s2_inference.evaluation import memory_snapshot, score_dev_cached
from tools.clip_s2_inference.generation import GPU_HOLD, guard_device
from tools.followup.model import build_model
from tools.followup.recipe import PromptPool

from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json, writer

BOOT = "21ceb469-0d6b-494a-b4fa-dd12d94f2931"
TOKEN = "cached-dev-batch1-driver61742-20261011"
OUTPUT = BASE / "diagnostics" / TOKEN
CHECKPOINT = (
    BASE
    / "runs"
    / (
        "translation-reference-seed0-"
        "fe5c22c66305d9e3bea3074792c0f64955b2c01c4b61bf92656448c0f7ae5671"
    )
    / "resume-epoch-2.pt"
)
EXPECTED_CHECKPOINT = "d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9"


def preflight() -> tuple[dict, dict, PromptPool]:
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != BOOT:
        raise RuntimeError("Reviewed boot changed; this approval cannot be reused")
    receipt = json.loads(CHECKPOINT.with_suffix(".json").read_text())
    identity = receipt["identity"]
    if (
        CHECKPOINT.is_symlink()
        or receipt["sha256"] != EXPECTED_CHECKPOINT
        or sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT
        or receipt["bytes"] != CHECKPOINT.stat().st_size
        or receipt["step"] != 1774
    ):
        raise ValueError("Native epoch2 checkpoint binding changed")
    if identity["code_hashes"] != code_hashes():
        raise ValueError("Original source closure changed")
    if identity["base_files"] != model_files():
        raise ValueError("Pinned base-model bytes changed")
    config_path = BASE / "study-inputs.json"
    if sha256_file(config_path) != identity["config_sha256"]:
        raise ValueError("Original study configuration changed")
    config = load_config(config_path)
    admission = admit_data(
        Path(config["spatial_root"]), config["spatial_manifest_sha256"], verify_arrays=False
    )
    if admission["identity"] != identity["data"]:
        raise ValueError("Original data/annotation/feature bindings changed")
    # This validates the actual selected arrays on CPU; no additional dev clip loads.
    batch = batch_for(admission, "dev", admission["ids"]["dev"][:1])
    pool = PromptPool(admission["datasets"]["train"].records, identity["seed"])
    if pool.eval_id != identity["decoder"]["exemplar_id"]:
        raise ValueError("Original train-only inference exemplar changed")
    pool.guard_references(tuple(r["text"] for r in admission["datasets"]["dev"].records))
    info = dict(
        checkpoint=str(CHECKPOINT),
        checkpoint_sha256=EXPECTED_CHECKPOINT,
        checkpoint_step=receipt["step"],
        source_files_verified=len(identity["code_hashes"]),
        clip_ids=list(batch.clip_ids),
        spatial_lengths=batch.spatial_mask.sum(1).tolist(),
        motion_lengths=batch.motion_mask.sum(1).tolist(),
        boot_id=BOOT,
        script_sha256=sha256_file(Path(__file__)),
        inference_source_sha256={
            str(p): sha256_file(p) for p in Path(inference.__file__).parent.glob("*.py")
        },
        diagnostic_only=True,
        max_batches=1,
        optimizer_loaded=False,
        training_resumed=False,
    )
    atomic_json(OUTPUT / "preflight.json", info)
    print(json.dumps(dict(phase="preflight_passed", **info)), flush=True)
    return identity, admission, pool


def run_once() -> None:
    identity, admission, pool = preflight()
    hold = json.loads(GPU_HOLD.read_text())
    scope = hold.get("gpu_validation_scope", {})
    if (
        hold.get("status") != "stopped_by_user"
        or hold.get("automatic_recovery_disabled") is not True
        or scope.get("token") != TOKEN
        or scope.get("boot_id") != BOOT
        or scope.get("max_batches") != 1
        or scope.get("script_sha256") != sha256_file(Path(__file__))
        or scope.get("checkpoint_sha256") != EXPECTED_CHECKPOINT
    ):
        raise RuntimeError("One-batch diagnostic authorization does not match")
    guard_device(torch.device("cuda"), gpu_reviewed=True)
    with writer(OUTPUT):
        marker = OUTPUT / "started.json"
        if marker.exists():
            raise RuntimeError("One-shot diagnostic already claimed; no retry permitted")
        atomic_json(marker, dict(token=TOKEN, boot_id=BOOT, utc=datetime.now(UTC).isoformat()))
        begin = time.perf_counter()
        with (OUTPUT / "events.jsonl").open("x") as log:

            def emit(event: dict) -> None:
                record = {"utc": datetime.now(UTC).isoformat(), **event}
                line = json.dumps(record, sort_keys=True)
                log.write(line + "\n")
                log.flush()
                os.fsync(log.fileno())
                print(line, flush=True)

            try:
                others = other_gpu_pids()
                if others:
                    raise RuntimeError(f"Other GPU compute processes present: {others}")
                emit(dict(phase="model_load_start"))
                model, base_hash = build_model(identity["seed"], identity["arm"], CACHE)
                state = load_snapshot(
                    CHECKPOINT, model, None, identity, base_hash=base_hash, replay=False
                )
                if state["step"] != 1774 or base_hash != identity["frozen_fingerprint"]:
                    raise ValueError("Restored checkpoint/base fingerprint changed")
                del state
                gc.collect()
                training_cache = model.language_model.model.config.use_cache
                model.to("cuda").eval()
                emit(dict(phase="model_loaded", memory=memory_snapshot(torch.device("cuda"))))
                report = score_dev_cached(
                    model,
                    admission,
                    pool,
                    identity["data"]["ordered_ids"]["dev"],
                    max_batches=1,
                    gpu_reviewed=True,
                    on_event=emit,
                )
                if (
                    report["processed_clips"] != 1
                    or report["diagnostic_only"] is not True
                    or report["metrics"] is not None
                    or model.language_model.model.config.use_cache != training_cache
                ):
                    raise ValueError("Diagnostic bounds or training cache config changed")
                report.update(
                    status="completed",
                    total_seconds=time.perf_counter() - begin,
                    checkpoint_sha256=EXPECTED_CHECKPOINT,
                    checkpoint_step=1774,
                    optimizer_loaded=False,
                    training_config_use_cache=training_cache,
                    boot_id=BOOT,
                )
                atomic_json(OUTPUT / "report.json", report)
                emit(dict(phase="diagnostic_complete", processed_clips=1))
            except Exception as error:
                # No CUDA cleanup, RNG restoration, memory query or retry after a fault.
                emit(
                    dict(
                        phase="diagnostic_error", error_type=type(error).__name__, error=str(error)
                    )
                )
                atomic_json(
                    OUTPUT / "failure.json",
                    dict(
                        status="failed_no_retry", error_type=type(error).__name__, error=str(error)
                    ),
                )
                raise
            finally:
                current = json.loads(GPU_HOLD.read_text())
                if current.get("gpu_validation_scope", {}).get("token") == TOKEN:
                    current["gpu_validation_approved"] = False
                    current["gpu_validation_scope"]["consumed"] = True
                    current["gpu_validation_scope"]["finished_utc"] = datetime.now(UTC).isoformat()
                    atomic_json(GPU_HOLD, current)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run"))
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight()
    else:
        run_once()
