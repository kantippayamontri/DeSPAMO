"""Once-only, journaled Signer07 evaluation of dev-selected 16k projectors.

Use one shared external registry for the study. Its advisory lock stays held
through generation and publication. A crash may replay an uncommitted batch,
but never a durably journaled batch or a completed logical evaluation. Keep the
registry and journal: a different registry cannot detect previous evaluations.
The 1800-second cumulative stage cap supplements an external parent timeout.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import signal
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.evaluation.metrics import evaluate_translations
from despamo.evaluation.signer_pilot import choose_checkpoint, pilot_prompts
from despamo.models.residual_projector import enable_residual_projectors
from scripts.evaluate_e3_pilot import DECODER, PROTOCOL_HASH, collate_phoenix, load_e3_views

POLICY = "qwen-schema98-unreviewed-v1"
RUN_POLICY = "signer-pilot-projector-ablation-v1"
EVALUATION_POLICY = "projector-heldout-once-v1"
DEV_STEPS = (5500, 10400, 16000)
MAX_SECONDS = 1800
ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("protocol", "dataset", "text_manifest", "dino_root", "motion_root",
          "motion_manifest", "annotation")


def _json(path: Path) -> dict:
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"missing or invalid evaluation artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"evaluation artifact must be an object: {path}")
    return value


def _decoder_matches(value: object) -> bool:
    return isinstance(value, dict) and digest(value) == digest(DECODER)


def _recipe(identity: dict) -> dict:
    """Normalize historical seed-0 fields without changing checkpoint metadata."""
    projector = identity.get("projector")
    seed = identity.get("seed", 0)
    residual = projector == "residual"
    hidden = identity.get("residual_hidden_dim", 1024 if residual else None)
    residual_seed = identity.get("residual_seed", 1 if residual and seed == 0 else None)
    if (projector not in {"linear", "residual"} or type(seed) is not int or seed not in (0, 1)
            or identity.get("variant") != f"E3_projector_{projector}"
            or type(identity.get("config", {}).get("seed", 0)) is not int
            or identity.get("config", {}).get("seed", 0) != seed
            or (residual and (type(hidden) is not int or hidden != 1024
                             or type(residual_seed) is not int or residual_seed < 0))
            or (not residual and (hidden is not None or residual_seed is not None))):
        raise ValueError("projector variant/seed/hidden-width identity mismatch")
    return {"projector": projector, "seed": seed, "residual_hidden_dim": hidden,
            "residual_seed": residual_seed}


def validate_projector_selection(run: Path, protocol: dict,
                                 dev_ids: tuple[str, ...]) -> dict:
    """Admit only complete 16k ablations selected using exact ordered dev reports."""
    status = _json(run / "run-status.json")
    budget = _json(run / "budget.json")
    selected = _json(run / "selected-checkpoint.json")
    identity = budget.get("identity", {})
    if not isinstance(identity, dict):
        raise ValueError("projector training identity missing")
    _recipe(identity)
    if (run.is_symlink() or status.get("status") != "complete"
            or type(status.get("global_step")) is not int or status["global_step"] != 16000
            or budget.get("status") != "closed" or identity.get("run_policy") != RUN_POLICY
            or type(identity.get("steps")) is not int or identity["steps"] != 16000
            or identity.get("dev_checkpoints") != list(DEV_STEPS)
            or identity.get("protocol_hash") != protocol.get("protocol_hash")
            or identity.get("split_hash") != protocol.get("split_hash")
            or not _decoder_matches(protocol.get("decoder"))
            or identity.get("supervision_policy") != POLICY
            or identity.get("human_review_status") != "not_assessed"
            or identity.get("run_key", run.name) != run.name
            or status.get("run_key", run.name) != run.name):
        raise ValueError("projector complete run/budget/policy/horizon identity mismatch")
    reports = [_json(run / f"dev-{step}.json") for step in DEV_STEPS]
    if any(report.get("checkpoint_step") != step or not _decoder_matches(report.get("decoding"))
           or report.get("split_hash") != protocol["split_hash"]
           or report.get("model_variant") != identity["variant"]
           for step, report in zip(DEV_STEPS, reports, strict=True)):
        raise ValueError("projector dev schedule/decoder/variant mismatch")
    best = choose_checkpoint(reports, dev_ids)
    checkpoint = run / f"step-{selected.get('step')}.ckpt"
    if (selected.get("step") != best["checkpoint_step"]
            or selected.get("checkpoint_hash") != best["checkpoint_hash"]
            or selected.get("metrics") != best["metrics"]
            or selected.get("model_variant") != identity["variant"]
            or selected.get("protocol_hash") != protocol["protocol_hash"]
            or selected.get("adaptation_checkpoint_hash")
            != identity.get("adaptation_checkpoint_hash")
            or selected.get("supervision_policy") != POLICY
            or selected.get("human_review_status") != "not_assessed"
            or checkpoint.is_symlink() or not checkpoint.is_file()
            or file_hash(checkpoint) != selected["checkpoint_hash"]):
        raise ValueError("projector dev-best selected checkpoint/byte hash mismatch")
    return {"checkpoint": checkpoint, "checkpoint_hash": selected["checkpoint_hash"],
            "step": selected["step"], "identity": identity}


def _load_projector_model(selection: dict):
    identity = selection["identity"]
    recipe = _recipe(identity)
    if file_hash(selection["checkpoint"]) != selection["checkpoint_hash"]:
        raise ValueError("projector checkpoint byte hash changed before loading")
    checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=False)
    if (digest(checkpoint.get("run_metadata")) != digest(identity)
            or type(checkpoint.get("global_step")) is not int
            or checkpoint["global_step"] != selection["step"]):
        raise ValueError("projector checkpoint full training metadata/global_step mismatch")
    # Seed agent owns this interface; keep import lazy for preflight and CPU tests.
    from scripts.train_projector_pilot import build_seeded_projector_model

    config = OmegaConf.create(identity["config"])
    if not isinstance(config, DictConfig):
        raise ValueError("projector checkpoint config must be an object")
    model = build_seeded_projector_model(config, identity["steps"], recipe["seed"])
    if recipe["projector"] == "residual":
        enable_residual_projectors(model.visual_adapter, 1024, seed=recipe["residual_seed"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def _paths(args, identity: dict) -> tuple[Path, Path]:
    output, registry = args.output.resolve(), args.registry.resolve()
    sources = [args.projector_run, args.dataset, args.dino_root, args.motion_root,
               args.adapted_root, args.annotation.parent]
    cache = identity.get("config", {}).get("model", {}).get("cache_dir")
    if cache:
        sources.append(Path(cache))
    for raw, resolved in ((args.output, output), (args.registry, registry)):
        if (raw.is_symlink() or (raw.exists() and not raw.is_file())
                or any(resolved.is_relative_to(source.resolve()) for source in sources)
                or resolved in {getattr(args, name).resolve() for name in INPUTS}):
            raise ValueError("held-out output/registry is symlink or lies in source/run/cache")
    if (registry.is_relative_to(ROOT) or output == registry
            or output == Path(str(registry) + ".lock")):
        raise ValueError("held-out registry requires a separate external path")
    if Path(str(registry) + ".lock").is_symlink():
        raise ValueError("held-out registry lock cannot be symlink")
    return output, registry


def _prepare(args) -> tuple[dict, object, tuple[str, ...], dict, Path, Path]:
    if args.mode not in {"preflight", "run"}:
        raise ValueError("held-out mode invalid")
    if type(args.gpu_cap_seconds) is not int or not 0 < args.gpu_cap_seconds <= MAX_SECONDS:
        raise ValueError("held-out stage cap must be between 1 and 1800 seconds")
    original = {name: getattr(args, name) for name in INPUTS}
    adapted = _json(args.adapted_root / "complete/identity.json")
    protocol, views = load_e3_views(original, args.adapted_root, adapted)
    if (protocol.get("protocol_hash") != PROTOCOL_HASH
            or not _decoder_matches(protocol.get("decoder"))
            or protocol.get("supervision_policy") != POLICY
            or protocol.get("human_review_status") != "not_assessed"):
        raise ValueError("held-out frozen protocol/decoder/supervision mismatch")
    group = protocol["split"]["groups"]["test"]
    ids = tuple(group["clip_ids"])
    view = views["test"]
    if (len(ids) != 768 or len(set(ids)) != 768 or group.get("signers") != ["Signer07"]
            or tuple(row["fileid"] for row in view.records) != ids
            or any(row["signer"] != "Signer07" for row in view.records)):
        raise ValueError("held-out requires exact ordered 768 Signer07 clips")
    selection = validate_projector_selection(
        args.projector_run, protocol, tuple(protocol["split"]["groups"]["dev"]["clip_ids"]),
    )
    identity = selection["identity"]
    output, registry = _paths(args, identity)
    config = identity["config"]
    data, model, trainer = (config.get(name, {}) for name in ("data", "model", "trainer"))
    required = {"spatial_root": args.adapted_root,
                "spatial_manifest": args.adapted_root / "complete/manifest.json",
                "motion_root": args.motion_root, "motion_manifest": args.motion_manifest,
                "annotation_root": args.annotation.parent}
    if (data.get("num_workers") != 0 or data.get("batch_size") != 4
            or model.get("use_in_context") is not False or model.get("num_in_context", 0) != 0
            or model.get("warm_up_steps") != 2000 or trainer.get("max_steps") != 16000
            or trainer.get("accumulate_grad_batches") != 1
            or any(not data.get(key) or Path(data[key]).resolve() != path.resolve()
                   for key, path in required.items())):
        raise ValueError("held-out trained model/config/cache path mismatch")
    lineage = {
        "adaptation_checkpoint_hash": adapted["checkpoint_hash"],
        "adapted_encoder_key": digest(adapted),
        "adapted_manifest_hash": file_hash(args.adapted_root / "complete/manifest.json"),
        "adapted_frame_rows_hash": file_hash(args.adapted_root / "complete/frame_rows.json"),
    }
    if any(identity.get(key) != value for key, value in lineage.items()):
        raise ValueError("held-out trained adapted cache/hash lineage mismatch")
    # load_e3_views validates original content, receipts, and every adapted feature.
    # Bind those inputs again so a partial claim cannot drift on a later invocation.
    files = {
        **{name: getattr(args, name) for name in
           ("protocol", "text_manifest", "motion_manifest", "annotation")},
        "dataset_source": args.dataset / "source.json",
        "dataset_version": args.dataset / "version.json",
        "dino_manifest": args.dino_root / "complete/manifest.json",
        "dino_frame_rows": args.dino_root / "complete/frame_rows.json",
        "adapted_identity": args.adapted_root / "complete/identity.json",
        "adapted_manifest": args.adapted_root / "complete/manifest.json",
        "adapted_frame_rows": args.adapted_root / "complete/frame_rows.json",
        "training_budget": args.projector_run / "budget.json",
        "training_status": args.projector_run / "run-status.json",
        "selection": args.projector_run / "selected-checkpoint.json",
        **{f"dev_{step}": args.projector_run / f"dev-{step}.json" for step in DEV_STEPS},
    }
    source = {"paths": {name: str(path.resolve()) for name, path in original.items()},
              "file_hashes": {name: file_hash(path) for name, path in files.items()},
              "protocol_content_hash": digest(protocol)}
    binding = {
        "evaluation_policy": EVALUATION_POLICY, "protocol_hash": PROTOCOL_HASH,
        "split": "test", "split_hash": protocol["split_hash"], "decoding": DECODER,
        "test_clips": 768, "clip_ids_hash": digest(ids),
        "checkpoint_hash": selection["checkpoint_hash"], "checkpoint_step": selection["step"],
        "model_variant": identity["variant"], "training_horizon": 16000,
        "training_identity_hash": digest(identity), "config_hash": digest(config),
        "source_identity": source, "source_identity_hash": digest(source),
        "projector_run": str(args.projector_run.resolve()), **_recipe(identity), **lineage,
        "supervision_policy": POLICY, "human_review_status": "not_assessed",
        "gpu_cap_seconds": args.gpu_cap_seconds,
        "evaluator_hash": file_hash(Path(__file__).resolve()),
    }
    return selection, view, ids, binding, output, registry


def _items(items: object, ids: tuple[str, ...], *, complete: bool = False) -> None:
    if (not isinstance(items, list) or len(items) > len(ids) or len(items) % 4
            or (complete and len(items) != len(ids))
            or any(not isinstance(item, dict) or set(item) != {"clip_id", "prediction", "reference"}
                   or not all(isinstance(item[key], str) for key in item) for item in items)
            or tuple(item["clip_id"] for item in items) != ids[:len(items)]):
        raise ValueError("held-out journal/report missing, duplicate, or reordered items")


def _metrics(metrics: object) -> None:
    required = {"bleu1", "bleu2", "bleu3", "bleu4", "rougeL_precision", "rougeL_recall",
                "rougeL_f1"}
    if (not isinstance(metrics, dict) or set(metrics) != required
            or any(type(value) not in {int, float} or not math.isfinite(value)
                   or not 0 <= value <= (100 if key.startswith("bleu") else 1)
                   for key, value in metrics.items())):
        raise ValueError("held-out complete-corpus metrics invalid")


def _journal_path(registry: Path, checkpoint_hash: str) -> Path:
    return registry.with_name(f"{registry.name}.{checkpoint_hash}.journal.json")


def _registry(path: Path) -> dict:
    value = _json(path) if path.exists() else {
        "evaluation_policy": EVALUATION_POLICY, "evaluations": {},
    }
    if value.get("evaluation_policy") != EVALUATION_POLICY or not isinstance(
        value.get("evaluations"), dict,
    ):
        raise ValueError("held-out registry policy/format mismatch")
    return value


def _read_claim(registry: Path, state: dict, binding: dict, output: Path,
                ids: tuple[str, ...]) -> tuple[dict | None, dict | None]:
    claim = state["evaluations"].get(binding["checkpoint_hash"])
    journal_path = _journal_path(registry, binding["checkpoint_hash"])
    if output == journal_path or output.name.startswith(journal_path.name + "."):
        raise ValueError("held-out output collides with journal/state path")
    for path in (journal_path, Path(str(journal_path) + ".status.json"),
                 Path(str(journal_path) + ".budget.json")):
        if path.is_symlink():
            raise ValueError("held-out journal/status/budget cannot be symlink")
    if claim is None:
        if output.exists() or journal_path.exists() or any(
            entry.get("output_path") == str(output) for entry in state["evaluations"].values()
        ):
            raise ValueError("held-out output/journal already exists without matching claim")
        return None, None
    if (not isinstance(claim, dict) or digest(claim.get("binding")) != digest(binding)
            or claim.get("binding_hash") != digest(binding)
            or claim.get("output_path") != str(output)
            or claim.get("journal_path") != str(journal_path)
            or claim.get("status") not in {"in_progress", "partial", "complete"}):
        raise ValueError("held-out registry split/decoder/source/config/output drift")
    if journal_path.exists():
        journal = _json(journal_path)
        if (journal.get("evaluation_policy") != EVALUATION_POLICY
                or digest(journal.get("binding")) != digest(binding)
                or journal.get("binding_hash") != digest(binding)
                or journal.get("output_path") != str(output)):
            raise ValueError("held-out journal binding/provenance drift")
        _items(journal.get("items"), ids)
        if journal.get("items_hash") != digest(journal["items"]):
            raise ValueError("held-out journal items hash mismatch")
        if journal.get("metrics") is not None:
            _items(journal["items"], ids, complete=True)
            _metrics(journal["metrics"])
    elif claim.get("stage") == "claimed" and claim.get("committed_items") == 0:
        journal = None  # Crash between claiming and creating the empty journal.
    else:
        raise ValueError("held-out claimed journal missing; cannot replay committed batches")
    if claim["status"] == "complete" and journal is None:
        raise ValueError("held-out completed journal missing")
    return claim, journal


def _validate_report(path: Path, claim: dict, journal: dict, binding: dict,
                     ids: tuple[str, ...]) -> dict:
    if (path.is_symlink() or not path.is_file() or not claim.get("report_hash")
            or file_hash(path) != claim["report_hash"]):
        raise ValueError("held-out completed report missing or byte hash mismatch")
    report = _json(path)
    if (report.get("status") != "complete"
            or any(report.get(key) != value for key, value in binding.items())
            or report.get("items") != journal["items"]
            or report.get("metrics") != journal.get("metrics")):
        raise ValueError("held-out report provenance/journal mismatch")
    _items(report.get("items"), ids, complete=True)
    _metrics(report.get("metrics"))
    return report


def _report_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _publish_report(path: Path, value: dict) -> None:
    """Durable same-filesystem publication using link(2), never replace(2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".heldout-", delete=False) as out:
            pending = Path(out.name)
            out.write(_report_bytes(value))
            out.flush()
            os.fsync(out.fileno())
        os.link(pending, path)  # Fails if destination exists, including a dangling symlink.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


class StageCapTimeout(TimeoutError):
    """Cumulative held-out runtime exhausted, distinct from generation failures."""


class _Clock:
    def __init__(self, claim: dict, cap: int):
        self.claim = claim
        self.cap = cap
        spent = claim.get("gpu_seconds", 0.0)
        if type(spent) not in {float, int} or not math.isfinite(spent) or spent < 0:
            raise ValueError("held-out stored runtime budget invalid")
        # An unclean exit has no final monotonic charge. Conservatively charge
        # time since the last durable heartbeat, including downtime.
        if claim.get("status") == "in_progress":
            spent += max(0.0, time.time() - claim["updated_at"])
        self.base = spent
        self.started = time.monotonic()

    def spent(self) -> float:
        return self.base + time.monotonic() - self.started

    def check(self) -> None:
        if self.spent() >= self.cap:
            raise StageCapTimeout(f"held-out stage cap expired during {self.claim['stage']}")


@contextmanager
def _stage_timer(clock: _Clock):
    clock.check()
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    def timeout(signum, frame):
        raise StageCapTimeout(f"held-out stage cap expired during {clock.claim['stage']}")

    signal.signal(signal.SIGALRM, timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(0.001, clock.cap - clock.spent()))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            remaining = max(0.001, old_timer[0] - (time.monotonic() - clock.started))
            signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])


def _save_state(registry: Path, state: dict, claim: dict, clock: _Clock) -> None:
    claim.update(gpu_seconds=clock.spent(), updated_at=time.time())
    journal_path = Path(claim["journal_path"])
    atomic_json(Path(str(journal_path) + ".budget.json"), {
        "status": "closed" if claim["status"] in {"partial", "complete"} else "open",
        "evaluation_policy": EVALUATION_POLICY, "binding_hash": claim["binding_hash"],
        "gpu_seconds": claim["gpu_seconds"], "gpu_cap_seconds": clock.cap,
    })
    atomic_json(Path(str(journal_path) + ".status.json"), {
        key: claim[key] for key in ("status", "stage", "committed_items", "updated_at")
    } | {"error": claim.get("error"), "timeout_kind": claim.get("timeout_kind")})
    atomic_json(registry, state)


def _save_journal(path: Path, journal: dict) -> None:
    journal["items_hash"] = digest(journal["items"])
    atomic_json(path, journal)


def _runtime_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("projector held-out scoring requires CUDA")
    return torch.device("cuda")


def _generate(selection: dict, view, ids: tuple[str, ...], journal: dict,
              journal_path: Path, registry: Path, state: dict, claim: dict, clock: _Clock) -> None:
    model = None
    try:
        claim["stage"] = "model_loading"
        _save_state(registry, state, claim, clock)
        clock.check()
        device = _runtime_device()
        model = _load_projector_model(selection).to(device).eval()
        loader = DataLoader(Subset(view, range(len(journal["items"]), len(ids))), batch_size=4,
                            shuffle=False, num_workers=0, collate_fn=collate_phoenix,
                            generator=torch.Generator().manual_seed(0))
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext())
        claim["stage"] = "generation"
        _save_state(registry, state, claim, clock)
        with torch.inference_mode(), autocast:
            for batch in loader:
                clock.check()
                start = len(journal["items"])
                if batch.clip_ids != ids[start:start + 4] or batch.signers != ("Signer07",) * 4:
                    raise ValueError("held-out deterministic batch IDs/signers mismatch")
                visual, valid = model.visual_adapter(
                    batch.spatial.to(device), batch.spatial_mask.to(device),
                    batch.motion.to(device), batch.motion_mask.to(device),
                )
                # Only visuals and language-only prompts cross the generation boundary.
                predictions = model.language_model.generate_text(
                    visual, valid, pilot_prompts(batch, model.prompt_template),
                    DECODER["mode"], DECODER["beam_size"], DECODER["max_length"],
                )
                if len(predictions) != 4 or any(not isinstance(p, str) for p in predictions):
                    raise ValueError("held-out prediction count/type mismatch")
                new_items = [
                    {"clip_id": clip, "prediction": prediction, "reference": reference.lower()}
                    for clip, prediction, reference in zip(
                        batch.clip_ids, predictions, batch.texts, strict=True,
                    )
                ]
                committed = {**journal, "items": journal["items"] + new_items}
                _save_journal(journal_path, committed)  # Commit precedes the registry heartbeat.
                journal.update(committed)
                claim["committed_items"] = len(journal["items"])
                _save_state(registry, state, claim, clock)
                clock.check()
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate(args) -> dict:
    """Read-only admission, or resume/reuse one checkpoint-bound evaluation."""
    selection, view, ids, binding, output, registry = _prepare(args)
    if args.mode == "preflight":
        state = _registry(registry)
        claim, journal = _read_claim(registry, state, binding, output, ids)
        if claim is not None and (claim["status"] == "complete" or output.exists()):
            if journal is None:
                raise ValueError("held-out completed journal missing")
            _validate_report(output, claim, journal, binding, ids)
        return {"status": "ready", "test_clips": 768, "checkpoint_hash": binding["checkpoint_hash"],
                "checkpoint_step": binding["checkpoint_step"],
                "model_variant": binding["model_variant"],
                "seed": binding["seed"], "clip_ids_hash": binding["clip_ids_hash"],
                "registry_status": claim["status"] if claim else "unclaimed",
                "output": str(output), "registry": str(registry),
                "supervision_policy": POLICY, "human_review_status": "not_assessed"}
    registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(registry) + ".lock")
    if lock_path.is_symlink():
        raise ValueError("held-out registry lock cannot be symlink")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _registry(registry)
        claim, journal = _read_claim(registry, state, binding, output, ids)
        if claim is not None and (claim["status"] == "complete" or output.exists()):
            if journal is None:
                raise ValueError("held-out completed journal missing")
            result = _validate_report(output, claim, journal, binding, ids)
            if claim["status"] != "complete":
                clock = _Clock(claim, args.gpu_cap_seconds)
                claim.update(status="complete", stage="complete", committed_items=768)
                _save_state(registry, state, claim, clock)
            return result
        journal_path = _journal_path(registry, binding["checkpoint_hash"])
        if claim is None:
            claim = {"status": "in_progress", "stage": "claimed", "binding": binding,
                     "binding_hash": digest(binding), "output_path": str(output),
                     "journal_path": str(journal_path), "committed_items": 0,
                     "gpu_seconds": 0.0, "updated_at": time.time()}
            state["evaluations"][binding["checkpoint_hash"]] = claim
            atomic_json(registry, state)
        if journal is None:
            journal = {"evaluation_policy": EVALUATION_POLICY, "binding": binding,
                       "binding_hash": digest(binding), "output_path": str(output),
                       "items": [], "metrics": None}
            _save_journal(journal_path, journal)
        clock = _Clock(claim, args.gpu_cap_seconds)
        claim.update(status="in_progress", error=None, timeout_kind=None,
                     committed_items=len(journal["items"]))
        try:
            with _stage_timer(clock):
                if len(journal["items"]) < len(ids):
                    _generate(selection, view, ids, journal, journal_path, registry, state,
                              claim, clock)
                _items(journal["items"], ids, complete=True)
                claim["stage"] = "metrics"
                _save_state(registry, state, claim, clock)
                if journal.get("metrics") is None:
                    journal["metrics"] = evaluate_translations(
                        [item["prediction"] for item in journal["items"]],
                        [item["reference"] for item in journal["items"]],
                    )
                    _metrics(journal["metrics"])
                    _save_journal(journal_path, journal)
                clock.check()
                result = {**binding, "status": "complete", "items": journal["items"],
                          "metrics": journal["metrics"]}
                claim.update(stage="publication",
                             report_hash=hashlib.sha256(_report_bytes(result)).hexdigest())
                _save_state(registry, state, claim, clock)
                _publish_report(output, result)
                _validate_report(output, claim, journal, binding, ids)
                claim.update(status="complete", stage="complete", committed_items=768)
                _save_state(registry, state, claim, clock)
                return result
        except BaseException as error:
            claim.update(status="partial", error=type(error).__name__,
                         committed_items=len(journal["items"]),
                         timeout_kind=("stage_cap" if isinstance(error, StageCapTimeout)
                                       else "generation_timeout" if isinstance(error, TimeoutError)
                                       else None))
            _save_state(registry, state, claim, clock)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    for name in (*INPUTS, "adapted_root", "projector_run", "output", "registry"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--gpu-cap-seconds", type=int, default=MAX_SECONDS)
    result = evaluate(parser.parse_args())
    keys = ("status", "test_clips", "checkpoint_hash", "checkpoint_step", "model_variant",
            "seed", "registry_status", "output", "metrics")
    print(json.dumps({key: result[key] for key in keys if key in result}, sort_keys=True))


if __name__ == "__main__":
    main()
