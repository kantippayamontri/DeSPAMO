"""One full cached S² dev diagnostic; no training, selection or automatic retry."""

import gc
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from tools.benchmark.compact_checkpoint import load_snapshot
from tools.benchmark.feature_package import other_gpu_pids
from tools.clip_plain.generation import generate_cached_text, memory_snapshot
from tools.clip_s2.data import admit_data, batch_for
from tools.clip_s2.train import BASE, CACHE, code_hashes, load_config, model_files
from tools.clip_s2_inference.generation import guard_device
from tools.dinov3.official_runtime import process_owner, require_official_storage
from tools.followup.model import build_model
from tools.followup.recipe import PromptPool

from despamo.evaluation.metrics import evaluate_translations
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.storage import atomic_json, writer

TOKEN = "clip-s2-cached-full-dev-v1"
OUTPUT = BASE / "diagnostics" / TOKEN
HOLD = BASE / "manual-stop.json"
BOOT = "a25da066-fad8-4652-92f2-4d30dd670f2c"
CHECKPOINT_SHA = "d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9"
DEV_HASH = "0147dcfd2438ee98cf2d485d9a994264794d0e4bb7b3dee9dc03193b10d15a44"


def host_check(start_utc: str) -> dict:
    command = (
        "$ErrorActionPreference='Stop'; Get-WinEvent -ListLog System | Out-Null;"
        f"$start=[DateTime]::Parse('{start_utc}').ToLocalTime();"
        "$events=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$start} "
        "-ErrorAction SilentlyContinue | Where-Object {"
        "($_.ProviderName -eq 'nvlddmkm' -and $_.Id -in @(13,14,153)) -or "
        "($_.ProviderName -in @('Display','Microsoft-Windows-WER-SystemErrorReporting') "
        "-and $_.Level -in @(1,2)) -or "
        "($_.ProviderName -eq 'Microsoft-Windows-Kernel-Power' -and $_.Id -eq 41)});"
        "[ordered]@{Utc=[DateTime]::UtcNow.ToString('o');CFree=(Get-PSDrive C).Free;"
        "BootUtc=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime"
        ".ToUniversalTime().ToString('o');NewFaultCount=$events.Count}|ConvertTo-Json"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        check=True,
        timeout=30,
    )
    record = json.loads(result.stdout.decode("utf-8-sig"))
    if (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() != BOOT
        or record["BootUtc"] != "2026-10-11T08:25:48.5000000Z"
        or record["NewFaultCount"] != 0
        or record["CFree"] < 50 * 1024**3 + 64 * 1024**2
    ):
        raise RuntimeError("Host/boot/storage guard failed; no retry")
    return record


def main() -> None:
    hold = json.loads(HOLD.read_text())
    scope = hold.get("full_dev_gpu_validation_scope", {})
    if (
        hold.get("status") != "stopped_by_user"
        or hold.get("automatic_recovery_disabled") is not True
        or scope.get("approved") is not True
        or scope.get("consumed", False)
        or scope.get("boot_id") != BOOT
        or scope.get("token") != TOKEN
        or scope.get("dev_ids_sha256") != DEV_HASH
        or scope.get("max_generations") != 519
        or scope.get("script_sha256") != sha256_file(Path(__file__))
    ):
        raise RuntimeError("Exact reviewed full-dev diagnostic scope missing")
    config = load_config(BASE / "study-inputs.json")
    checkpoint = next((BASE / "runs").glob("*/resume-epoch-2.pt"))
    receipt = json.loads(checkpoint.with_suffix(".json").read_text())
    identity = receipt["identity"]
    if (
        receipt["sha256"] != CHECKPOINT_SHA
        or sha256_file(checkpoint) != CHECKPOINT_SHA
        or receipt["step"] != 1774
        or identity["code_hashes"] != code_hashes()
        or identity["base_files"] != model_files()
        or sha256_file(BASE / "study-inputs.json") != identity["config_sha256"]
    ):
        raise ValueError("Original source/base/config/checkpoint changed")
    admission = admit_data(
        Path(config["spatial_root"]), config["spatial_manifest_sha256"], verify_arrays=False
    )
    ids = admission["ids"]["dev"]
    if admission["identity"] != identity["data"] or len(ids) != 519 or digest(ids) != DEV_HASH:
        raise ValueError("Original ordered full-dev data identity changed")
    pool = PromptPool(admission["datasets"]["train"].records, identity["seed"])
    pool.guard_references(tuple(r["text"] for r in admission["datasets"]["dev"].records))
    if pool.eval_id != identity["decoder"]["exemplar_id"]:
        raise ValueError("Original train-only exemplar changed")
    host = host_check(scope["events_start_utc"])
    require_official_storage(BASE, 64 * 1024**2)
    guard_device(torch.device("cuda"), gpu_reviewed=True)
    with writer(OUTPUT):
        if (OUTPUT / "started.json").exists():
            raise RuntimeError("One-shot scope already claimed; no retry")
        started = datetime.now(UTC).isoformat()
        atomic_json(
            OUTPUT / "started.json",
            dict(
                scope=scope,
                host=host,
                checkpoint=str(checkpoint),
                owner=process_owner(os.getpid()),
                utc=started,
            ),
        )
        completed = []
        attempted = 0
        elapsed = 0.0
        primary = None
        with (OUTPUT / "events.jsonl").open("x") as handle:

            def emit(event: dict) -> None:
                row = {"utc": datetime.now(UTC).isoformat(), **event}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                if row["phase"] in ("model_load_start", "diagnostic_complete", "diagnostic_error"):
                    print(json.dumps(row), flush=True)

            try:
                if other_gpu_pids():
                    raise RuntimeError("Other GPU compute worker present")
                emit(dict(phase="model_load_start"))
                model, base_hash = build_model(identity["seed"], identity["arm"], CACHE)
                state = load_snapshot(
                    checkpoint, model, None, identity, base_hash=base_hash, replay=False
                )
                del state
                gc.collect()
                model.to("cuda").eval()
                for index, clip in enumerate(ids):
                    host = host_check(scope["events_start_utc"])
                    batch = batch_for(admission, "dev", (clip,))
                    torch.cuda.reset_peak_memory_stats()
                    meta = dict(
                        batch_index=index,
                        clip_id=clip,
                        spatial_frames=int(batch.spatial_mask.sum()),
                        motion_windows=int(batch.motion_mask.sum()),
                    )
                    emit(
                        dict(
                            phase="batch_start",
                            **meta,
                            host=host,
                            memory=memory_snapshot(torch.device("cuda")),
                        )
                    )
                    begin = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        visual, valid = model.visual_adapter(
                            batch.spatial.to("cuda"),
                            batch.spatial_mask.to("cuda"),
                            batch.motion.to("cuda"),
                            batch.motion_mask.to("cuda"),
                        )
                        prompts, exemplars = pool.inference(batch.clip_ids)
                        attempted += 1
                        result = generate_cached_text(
                            model.language_model,
                            visual,
                            valid,
                            prompts,
                            scope=scope,
                            on_event=lambda event, meta=meta: emit({**meta, **event}),
                        )
                    seconds = time.perf_counter() - begin
                    memory = memory_snapshot(torch.device("cuda"))
                    checked = host_check(scope["events_start_utc"])
                    row = dict(
                        **meta,
                        seconds=seconds,
                        memory=memory,
                        host_after=checked,
                        token_ids=result["token_ids"][0],
                        prediction=result["texts"][0],
                        output_length=result["output_lengths"][0],
                        exemplar_ids=[exemplars[0]],
                        reference=batch.texts[0].lower(),
                    )
                    atomic_json(OUTPUT / "clips" / f"{index:04}.json", row)
                    completed.append(row)
                    elapsed += seconds
                    emit(dict(phase="batch_complete", **meta, seconds=seconds, memory=memory))
                    atomic_json(
                        OUTPUT / "progress.json",
                        dict(
                            status="running",
                            completed_clips=len(completed),
                            attempted_generations=attempted,
                            expected_clips=519,
                            last_clip=clip,
                            batch_seconds_sum=elapsed,
                            utc=datetime.now(UTC).isoformat(),
                        ),
                    )
                    if (index + 1) % 25 == 0:
                        print(
                            json.dumps(
                                dict(
                                    completed_clips=index + 1,
                                    last_clip=clip,
                                    batch_seconds_sum=elapsed,
                                )
                            ),
                            flush=True,
                        )
                    del visual, valid, result, batch
                if tuple(r["clip_id"] for r in completed) != ids or attempted != 519:
                    raise ValueError("Incomplete/reordered full-dev generation")
                metrics = evaluate_translations(
                    [r["prediction"] for r in completed], [r["reference"] for r in completed]
                )
                report = dict(
                    status="completed",
                    complete=True,
                    diagnostic_only=True,
                    metrics=metrics,
                    processed_clips=519,
                    attempted_generations=attempted,
                    items=completed,
                    checkpoint_sha256=CHECKPOINT_SHA,
                    checkpoint_step=1774,
                    optimizer_loaded=False,
                    batch_size=1,
                    use_cache=True,
                    beam_size=5,
                    max_length=64,
                    boot_id=BOOT,
                    training_config_use_cache=model.language_model.model.config.use_cache,
                    dev_ids_sha256=DEV_HASH,
                    started_utc=started,
                    finished_utc=datetime.now(UTC).isoformat(),
                    final_host=host_check(scope["events_start_utc"]),
                )
                atomic_json(OUTPUT / "report.json", report)
                atomic_json(
                    OUTPUT / "progress.json",
                    dict(
                        status="completed",
                        completed_clips=519,
                        attempted_generations=attempted,
                        expected_clips=519,
                        utc=datetime.now(UTC).isoformat(),
                    ),
                )
                emit(dict(phase="diagnostic_complete", processed_clips=519, metrics=metrics))
            except BaseException as error:
                primary = error
                # CPU-only failure evidence. Never restore/query/synchronize CUDA or retry.
                try:
                    emit(
                        dict(
                            phase="diagnostic_error",
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                    )
                    failure = dict(
                        status="failed_no_retry",
                        error=str(error),
                        completed_clips=len(completed),
                        attempted_generations=attempted,
                        utc=datetime.now(UTC).isoformat(),
                    )
                    atomic_json(OUTPUT / "failure.json", failure)
                    atomic_json(OUTPUT / "progress.json", failure)
                except Exception as observer:
                    error.add_note(f"Failure evidence write failed: {observer}")
                raise
            finally:
                try:
                    current = json.loads(HOLD.read_text())
                    current["gpu_validation_approved"] = False
                    current["full_dev_gpu_validation_scope"]["consumed"] = True
                    current["full_dev_gpu_validation_scope"]["finished_utc"] = datetime.now(
                        UTC
                    ).isoformat()
                    atomic_json(HOLD, current)
                except Exception as observer:
                    if primary is None:
                        raise
                    primary.add_note(f"Diagnostic permission revoke failed: {observer}")


if __name__ == "__main__":

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal{signum}; no retry")

    signal.signal(signal.SIGTERM, stop)
    main()
