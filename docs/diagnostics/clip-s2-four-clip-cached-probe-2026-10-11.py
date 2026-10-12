"""One approved four-clip S² diagnostic; no training, retries or device changes."""

import gc
import json
import os
import shutil
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
from tools.followup.model import build_model
from tools.followup.recipe import PromptPool

from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json, writer

TOKEN = "clip-s2-cached-four-dev-v1"
OUTPUT = BASE / "diagnostics" / TOKEN
HOLD = BASE / "manual-stop.json"
BOOT = "a25da066-fad8-4652-92f2-4d30dd670f2c"
CLIPS = (
    "11August_2010_Wednesday_tagesschau-2",
    "11December_2009_Friday_tagesschau-3509",
    "30January_2011_Sunday_tagesschau-6345",
    "03November_2010_Wednesday_tagesschau-6506",
)
FRAMES = (42, 16, 105, 272)
CHECKPOINT_SHA = "d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9"


def host_check() -> dict:
    command = (
        "$ErrorActionPreference='Stop'; Get-WinEvent -ListLog System | Out-Null;"
        "$start=[DateTime]::Parse('2026-10-11T19:05:00Z').ToLocalTime();"
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
        or min(record["CFree"], shutil.disk_usage(BASE).free) < 50 * 1024**3 + 64 * 1024**2
    ):
        raise RuntimeError("Host/boot/storage guard failed; no retry")
    return record


def main() -> None:
    # All scope/identity/CPU checks precede the first GPU query or model construction.
    hold = json.loads(HOLD.read_text())
    scope = hold.get("four_clip_gpu_validation_scope", {})
    if (
        hold.get("status") != "stopped_by_user"
        or hold.get("automatic_recovery_disabled") is not True
        or scope.get("approved") is not True
        or scope.get("consumed", False)
        or scope.get("boot_id") != BOOT
        or scope.get("token") != TOKEN
        or scope.get("clip_ids") != list(CLIPS)
        or scope.get("max_generations") != 4
        or scope.get("script_sha256") != sha256_file(Path(__file__))
    ):
        raise RuntimeError("Exact reviewed four-clip GPU scope missing")
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
    if admission["identity"] != identity["data"]:
        raise ValueError("Original data identity changed")
    batches = [batch_for(admission, "dev", (clip,)) for clip in CLIPS]
    if [int(b.spatial_mask.sum()) for b in batches] != list(FRAMES):
        raise ValueError("Declared clip frame lengths changed")
    pool = PromptPool(admission["datasets"]["train"].records, identity["seed"])
    pool.guard_references(tuple(r["text"] for r in admission["datasets"]["dev"].records))
    if pool.eval_id != identity["decoder"]["exemplar_id"]:
        raise ValueError("Original train-only exemplar changed")
    host = host_check()
    guard_device(torch.device("cuda"), gpu_reviewed=True)
    with writer(OUTPUT):
        if (OUTPUT / "started.json").exists():
            raise RuntimeError("One-shot scope already claimed; no retry")
        atomic_json(
            OUTPUT / "started.json", dict(scope=scope, host=host, checkpoint=str(checkpoint))
        )
        completed = []
        with (OUTPUT / "events.jsonl").open("x") as handle:

            def emit(event: dict) -> None:
                row = {"utc": datetime.now(UTC).isoformat(), **event}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                if row["phase"] not in ("decoder_start", "decoder_logits"):
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
                for index, batch in enumerate(batches):
                    host = host_check()
                    torch.cuda.reset_peak_memory_stats()
                    metadata = dict(
                        batch_index=index, clip_id=CLIPS[index], spatial_frames=FRAMES[index]
                    )
                    emit(
                        dict(
                            phase="batch_start",
                            **metadata,
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
                        result = generate_cached_text(
                            model.language_model,
                            visual,
                            valid,
                            prompts,
                            scope=scope,
                            on_event=lambda event, metadata=metadata: emit({**metadata, **event}),
                        )
                    seconds = time.perf_counter() - begin
                    memory = memory_snapshot(torch.device("cuda"))
                    checked = host_check()
                    row = dict(
                        **metadata,
                        seconds=seconds,
                        memory=memory,
                        host_after=checked,
                        token_ids=result["token_ids"],
                        texts=result["texts"],
                        output_lengths=result["output_lengths"],
                        exemplar_ids=exemplars,
                    )
                    atomic_json(OUTPUT / f"clip-{index}.json", row)
                    completed.append(row)
                    emit(dict(phase="batch_complete", **metadata, seconds=seconds, memory=memory))
                    del visual, valid, result
                report = dict(
                    status="completed",
                    diagnostic_only=True,
                    metrics=None,
                    processed_clips=4,
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
                    final_host=host_check(),
                )
                atomic_json(OUTPUT / "report.json", report)
                emit(dict(phase="diagnostic_complete", processed_clips=4))
            except BaseException as error:
                # Never query CUDA, transfer tensors, restore RNG or retry after a fault.
                try:
                    emit(
                        dict(
                            phase="diagnostic_error",
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                    )
                    atomic_json(
                        OUTPUT / "failure.json",
                        dict(error=str(error), processed_clips=len(completed), no_retry=True),
                    )
                except Exception as observer:
                    error.add_note(f"Failure evidence write failed: {observer}")
                raise
            finally:
                current = json.loads(HOLD.read_text())
                current["gpu_validation_approved"] = False
                current["four_clip_gpu_validation_scope"]["consumed"] = True
                current["four_clip_gpu_validation_scope"]["finished_utc"] = datetime.now(
                    UTC
                ).isoformat()
                atomic_json(HOLD, current)


if __name__ == "__main__":
    main()
