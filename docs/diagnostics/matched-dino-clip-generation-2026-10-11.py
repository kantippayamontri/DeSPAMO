"""Two one-shot fresh-process diagnostics; only spatial feature bytes differ."""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
from tools.benchmark import translation_data as dino_data
from tools.benchmark.compact_checkpoint import load_snapshot
from tools.benchmark.feature_package import other_gpu_pids
from tools.clip_s2 import data as clip_data
from tools.clip_s2.train import BASE, CACHE, code_hashes, load_config, model_files
from tools.clip_s2_inference.evaluation import memory_snapshot
from tools.clip_s2_inference.generation import GPU_HOLD, generate_cached_text, guard_device
from tools.followup.model import build_model
from tools.followup.recipe import PromptPool

from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json, writer

BOOT = "27947766-34fb-4d7d-9e19-9c9273210cc6"
TOKEN = "matched-audio-isolation-v1"
OUTPUT = BASE / "diagnostics" / TOKEN
AUDIO = Path("/mnt/b/DeSpaMo/diagnostics/gpu-crashes-2026-10-11") / TOKEN
RUN = (
    BASE
    / "runs"
    / "translation-reference-seed0-fe5c22c66305d9e3bea3074792c0f64955b2c01c4b61bf92656448c0f7ae5671"
)
CHECKPOINT = RUN / "resume-epoch-2.pt"
CHECKPOINT_SHA = "d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9"


def audio_state() -> dict:
    command = (
        "$id='PCI\\VEN_10DE&DEV_22BA&SUBSYS_51021462&REV_A1\\4&2106B648&0&0109';"
        "[ordered]@{ProblemCode=[int](Get-PnpDeviceProperty -InstanceId $id "
        "-KeyName 'DEVPKEY_Device_ProblemCode').Data;"
        "BootUtc=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime"
        ".ToUniversalTime().ToString('o');CFree=(Get-PSDrive C).Free}|ConvertTo-Json"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout.decode("utf-8-sig"))


def prepare() -> tuple[dict, dict, PromptPool, dict]:
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != BOOT:
        raise RuntimeError("Approved boot changed; no repeat permitted")
    receipt = json.loads(CHECKPOINT.with_suffix(".json").read_text())
    identity = receipt["identity"]
    if (
        CHECKPOINT.is_symlink()
        or receipt["sha256"] != CHECKPOINT_SHA
        or receipt["step"] != 1774
        or sha256_file(CHECKPOINT) != CHECKPOINT_SHA
        or identity["code_hashes"] != code_hashes()
        or identity["base_files"] != model_files()
        or identity["runtime"] != {n: version(n) for n in ("torch", "transformers", "peft")}
        or sha256_file(BASE / "study-inputs.json") != identity["config_sha256"]
    ):
        raise ValueError("Original checkpoint/source/base binding changed")
    config = load_config(BASE / "study-inputs.json")
    clip = clip_data.admit_data(
        Path(config["spatial_root"]), config["spatial_manifest_sha256"], verify_arrays=False
    )
    dino = dino_data.admit_data(verify_arrays=False)
    if clip["identity"] != identity["data"] or clip["ids"] != dino["ids"]:
        raise ValueError("Original data identity or paired corpus IDs changed")
    selected = clip["ids"]["dev"][:1]
    batches = {
        "dino": dino_data.batch_for(dino, "dev", selected),
        "clip": clip_data.batch_for(clip, "dev", selected),
    }
    left, right = batches["dino"], batches["clip"]
    if (
        left.clip_ids != right.clip_ids
        or left.texts != right.texts
        or left.spatial.shape != right.spatial.shape
        or left.spatial.dtype != right.spatial.dtype
        or not torch.equal(left.spatial_mask, right.spatial_mask)
        or not torch.equal(left.motion, right.motion)
        or not torch.equal(left.motion_mask, right.motion_mask)
        or not torch.isfinite(left.spatial).all()
        or not torch.isfinite(right.spatial).all()
    ):
        raise ValueError("Paired input shape/text/mask/motion/finiteness mismatch")
    pool = PromptPool(clip["datasets"]["train"].records, identity["seed"])
    pool.guard_references(tuple(r["text"] for r in clip["datasets"]["dev"].records))
    if pool.eval_id != identity["decoder"]["exemplar_id"]:
        raise ValueError("Train-only exemplar changed")
    prompts, _ = pool.inference(selected)
    sources = {}
    for label, root in [("dino", dino_data.SPATIAL), ("clip", Path(config["spatial_root"]))]:
        meta = json.loads((root / "receipts/dev" / f"{selected[0]}.json").read_text())
        sources[label] = meta["source_hash"]
    if len(set(sources.values())) != 1:
        raise ValueError("Spatial feature sets use different source frame bytes")
    info = dict(
        boot_id=BOOT,
        checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_step=1774,
        checkpoint_origin="CLIP seed0 reference; same weights for both feature variants",
        source_files_verified=len(identity["code_hashes"]),
        clip_ids=list(selected),
        spatial_shape=list(right.spatial.shape),
        motion_shape=list(right.motion.shape),
        prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        spatial_source_hash=sources["clip"],
        feature_sha256={
            k: v["feature_hashes"]["spatial"]["dev", selected[0]]
            for k, v in [("dino", dino), ("clip", clip)]
        },
        script_sha256=sha256_file(Path(__file__)),
    )
    atomic_json(OUTPUT / "preflight.json", info)
    print(json.dumps(dict(phase="preflight_passed", **info)), flush=True)
    return identity, batches, pool, info


def run_variant(variant: str) -> None:
    identity, batches, pool, info = prepare()
    hold = json.loads(GPU_HOLD.read_text())
    scope = hold.get("gpu_validation_scope", {})
    if (
        hold.get("status") != "stopped_by_user"
        or hold.get("automatic_recovery_disabled") is not True
        or scope.get("token") != TOKEN
        or scope.get("boot_id") != BOOT
        or scope.get("allowed_variants") != ["dino", "clip"]
        or scope.get("max_generations") != 2
        or scope.get("script_sha256") != info["script_sha256"]
    ):
        raise RuntimeError("Exact paired diagnostic scope missing")
    if variant == "clip":
        prior = json.loads((OUTPUT / "dino/report.json").read_text())
        health = json.loads((OUTPUT / "dino-windows-check.json").read_text())
        if prior["status"] != "completed" or health["new_fault_events"] != 0:
            raise RuntimeError("First variant did not pass; second generation blocked")
        if health["boot_id"] != BOOT:
            raise RuntimeError("Post-first-generation boot changed")
    host = audio_state()
    if (
        host["ProblemCode"] != 22
        or host["CFree"] < 50 * 1024**3 + 64 * 1024**2
        or host["BootUtc"] != "2026-10-11T08:25:48.5000000Z"
    ):
        raise RuntimeError("Audio isolation, host boot or backing-disk reserve check failed")
    if not (AUDIO / "audio-disable.json").is_file():
        raise RuntimeError("Verified audio-isolation receipt missing")
    guard_device(torch.device("cuda"), gpu_reviewed=True)
    destination = OUTPUT / variant
    with writer(destination):
        if (destination / "started.json").exists():
            raise RuntimeError("Variant already claimed; no automatic repeat")
        atomic_json(destination / "started.json", dict(utc=datetime.now(UTC).isoformat(), **info))
        with (destination / "events.jsonl").open("x") as log:

            def emit(event: dict) -> None:
                event = {"utc": datetime.now(UTC).isoformat(), "variant": variant, **event}
                text = json.dumps(event, sort_keys=True)
                log.write(text + "\n")
                log.flush()
                os.fsync(log.fileno())
                print(text, flush=True)

            try:
                if other_gpu_pids():
                    raise RuntimeError("Other GPU compute process present")
                emit(dict(phase="model_load_start"))
                model, base_hash = build_model(identity["seed"], identity["arm"], CACHE)
                state = load_snapshot(
                    CHECKPOINT, model, None, identity, base_hash=base_hash, replay=False
                )
                del state
                gc.collect()
                model.to("cuda").eval()
                batch = batches[variant]
                prompts, exemplars = pool.inference(batch.clip_ids)
                token_ids = []
                original_decode = model.language_model.tokenizer.batch_decode

                def capture_decode(sequences, **kwargs):
                    result = original_decode(sequences, **kwargs)
                    token_ids.extend(sequences.detach().cpu().tolist())
                    return result

                model.language_model.tokenizer.batch_decode = capture_decode
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                emit(dict(phase="batch_start", memory=memory_snapshot(torch.device("cuda"))))
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    visual, valid = model.visual_adapter(
                        batch.spatial.to("cuda"),
                        batch.spatial_mask.to("cuda"),
                        batch.motion.to("cuda"),
                        batch.motion_mask.to("cuda"),
                    )
                    texts = generate_cached_text(
                        model.language_model,
                        visual,
                        valid,
                        prompts,
                        beam_size=5,
                        max_length=64,
                        gpu_reviewed=True,
                        on_start=lambda fields: emit(dict(phase="generation_start", **fields)),
                    )
                memory = memory_snapshot(torch.device("cuda"))
                seconds = time.perf_counter() - started
                if len(token_ids) != 1 or model.language_model.model.config.use_cache is not False:
                    raise ValueError("Generation count or training cache config changed")
                report = dict(
                    status="completed",
                    diagnostic_only=True,
                    metrics=None,
                    variant=variant,
                    **info,
                    seconds=seconds,
                    memory=memory,
                    token_ids=token_ids,
                    texts=texts,
                    exemplar_ids=exemplars,
                    batch_size=1,
                    beam_size=5,
                    max_length=64,
                    use_cache=True,
                    do_sample=False,
                    optimizer_loaded=False,
                    training_config_use_cache=model.language_model.model.config.use_cache,
                    audio_controller_problem_code=host["ProblemCode"],
                )
                atomic_json(destination / "report.json", report)
                emit(dict(phase="batch_complete", seconds=seconds, memory=memory))
            except Exception as error:
                # No CUDA cleanup, device query or retry after an error.
                emit(dict(phase="error", error_type=type(error).__name__, error=str(error)))
                atomic_json(destination / "failure.json", dict(error=str(error), no_retry=True))
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run"))
    parser.add_argument("--variant", choices=("dino", "clip"))
    args = parser.parse_args()
    if args.phase == "run":
        if args.variant is None:
            parser.error("run requires one --variant")
        run_variant(args.variant)
    else:
        prepare()
