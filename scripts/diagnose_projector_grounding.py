"""Inference-only Signer03 diagnostic, conditional on exact-length donor pairs.

Run under the parent's hard watchdog (3600 seconds including startup). No feature
extraction or training occurs. The internal cumulative wall clock starts before
cache/checkpoint validation and reserves 15 seconds for shutdown. Completed
conditions are immutable; interrupted conditions restart, completed ones do not.
Preflight reads sources and existing outputs without creating directories/locks.

Public interfaces:
  inference_source_files(project: Path) -> dict[str, Path], sorted project-relative
    names for all local package Python sources, construction scripts and lockfiles.
  select_donor_pairs(records, spatial_lengths, motion_lengths, count=128, seed=0)
    -> {selected_ids, donor_map, pairs, fingerprints, eligibility, selection_seed,
        donor_map_hash}. IDs sorted; pairs retain shuffled group selection order.
  build_condition_samples(samples_by_id, selected_ids, donor_map, condition)
    -> list[PhoenixSample], with isolated cloned tensors and recipient metadata.
  summarize_grounding(normal, reports_by_condition)
    -> {conditions, subset_clips, interpretation}. Each condition contains metrics,
        metric_deltas, prediction_change_rate, repetition, donor_consistency.
  score_condition(model, samples_by_id, selected_ids, donor_map, condition,
                  split_hash, on_batch_end=None) -> fresh standard dev report.
  diagnose(args) -> readiness object in preflight, final summary object in run.

CLI: --mode preflight|run --output-dir PATH [--gpu-cap-seconds 3600] [--count 128].
Count positive/even, one pair per eligible group. Model seeds fixed 0/1; donor
selection seed fixed 0. Optional source flags default to run_projector_overnight.INPUTS:
--protocol, --dataset, --text-manifest, --dino-root, --motion-root,
--motion-manifest, --annotation, --adapted-root.

Output directory schema (policy projector-dev-grounding-v1):
  provenance.json: {binding, content_hash}; binding contains protocol/split,
    complete selection, per-selected-feature byte hashes, source/code byte hashes,
    decoder/hash, both checkpoint hashes/steps/identity hashes, cap and policies.
  selection.json: donor-selection object above plus content_hash.
  seed{0,1}-{condition}.json: {split, split_hash, items, metrics, decoding,
    condition, status, binding, binding_hash, run_policy, content_hash}.
    items: [{clip_id, prediction, reference, donor_id}]; reference stays recipient's.
    metrics: bleu1..bleu4, rougeL_precision, rougeL_recall, rougeL_f1.
    condition binding: provenance_hash, seed, condition, checkpoint_hash,
      decoder_hash, donor_map_hash.
  state.json: mutable cumulative budget/status/heartbeat/error, binding, stage,
    elapsed_seconds, peak_gpu_bytes. Stable sibling OUTPUT_DIR.lock excludes runs.
  progress.json: mutable partial summary; durable reports determine completed work.
  summary.json: immutable {run_policy, status, binding_hash, selection, seeds,
    elapsed_seconds, peak_gpu_bytes, gpu_cap_seconds, scope, bootstrap, content_hash}.
    seeds['0'|'1']: {run, checkpoint_hash, checkpoint_step, identity_hash,
      completed_conditions, condition_states, grounding}. Grounding uses summary
      helper schema. donor_consistency includes matches/rate, informative_recipients,
      informative_pairs, informative_matches/rate (null with no informative pairs).
    repetition includes unique counts/rates, duplicate rate, most-common count,
      repeated-token fraction/count and whitespace length mean/min/max/list.

Preflight stdout: {status='ready', run_policy, selection, binding_hash, models,
  completed_conditions, output_dir, gpu_cap_seconds}. Run stdout: final summary.
No bootstrap or human semantic ratings. Parent owns hard watchdog, launch and
research interpretation; CPU tests do not validate CUDA generation numerics.
"""

import argparse
import fcntl
import gc
import json
import math
import random
import signal
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.batch import PhoenixSample, collate_phoenix
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import normalize_text
from despamo.evaluation.metrics import evaluate_translations
from despamo.evaluation.signer_pilot import score_pilot_batches
from despamo.training.e3_runtime import load_e3_views
from scripts.evaluate_e3_pilot import DECODER, PROTOCOL_HASH
from scripts.evaluate_projector_pilot import (
    DEV_STEPS,
    POLICY,
    _load_projector_model,
    _publish_report,
    validate_projector_selection,
)
from scripts.run_projector_overnight import BASE, INPUTS, SEED0

RUN_POLICY = "projector-dev-grounding-v1"
CONDITIONS = ("normal", "shuffle_spatial", "shuffle_motion", "shuffle_both")
MAX_SECONDS = 3600
RESERVE_SECONDS = 15
ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    0: SEED0["linear"],
    1: BASE / ("projector-linear-16000-"
               "0319401a7579e5fc6ee3a37748c01a441b2e0b75cf97ce9778f6fc60e1f457da"),
}


def inference_source_files(project: Path) -> dict[str, Path]:
    """Conservative local inference/construction closure, without importing modules.

    Binding every package Python file includes transitive and dynamically selected
    local dependencies; additions/removals also change the source-hash inventory.
    Explicit construction scripts, DINO provenance helpers and dependency lockfiles
    must exist: missing files fail hashing instead of silently weakening provenance.
    """
    files = {path.relative_to(project).as_posix(): path
             for path in (project / "src/despamo").rglob("*.py")}
    required = (
        "scripts/diagnose_projector_grounding.py", "scripts/evaluate_projector_pilot.py",
        "scripts/evaluate_e3_pilot.py", "scripts/train_projector_pilot.py",
        "scripts/train_signer_pilot.py", "scripts/train_e3_pilot.py",
        "scripts/run_projector_overnight.py", "tools/dinov3/storage.py",
        "tools/dinov3/identity.py", "pyproject.toml", "uv.lock",
    )
    files.update({name: project / name for name in required})
    return dict(sorted(files.items()))


def _reference(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", normalize_text(text)).casefold().split())


def select_donor_pairs(
    records: Sequence[dict], spatial_lengths: Mapping[str, int],
    motion_lengths: Mapping[str, int], count: int = 128, seed: int = 0,
) -> dict:
    """Return selected_ids (sorted), reciprocal donor_map, pairs, fingerprints,
    eligibility counts, selection_seed and donor_map_hash. One pair per shuffled
    eligible exact-length group; sorted candidates, random first then distinct text.
    Reference fingerprint normalization: Phoenix punctuation, NFC/casefold/space.
    """
    if type(count) is not int or count <= 0 or count % 2:
        raise ValueError("count must be positive and even")
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    fingerprints = {}
    for row in sorted(records, key=lambda r: r["fileid"]):
        clip = row["fileid"]
        if row["signer"] != "Signer03":
            raise ValueError("donor selection requires Signer03 dev only")
        if clip in fingerprints or not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("duplicate IDs or empty references")
        lengths = (spatial_lengths[clip], motion_lengths[clip])
        if any(type(length) is not int or length <= 0 for length in lengths):
            raise ValueError("invalid exact feature lengths")
        fingerprints[clip] = {"reference": normalize_text(row["text"]),
                              "normalized_reference": _reference(row["text"]),
                              "reference_hash": digest(_reference(row["text"])),
                              "spatial_length": lengths[0], "motion_length": lengths[1]}
        groups[lengths].append(clip)
    eligible = [key for key in sorted(groups)
                if len({fingerprints[c]["normalized_reference"] for c in groups[key]}) >= 2]
    eligible_clips = sum(len(groups[key]) for key in eligible)
    eligibility = {
        "total_clips": len(records), "total_groups": len(groups),
        "eligible_clips": eligible_clips, "eligible_groups": len(eligible),
        "excluded_clips": len(records) - eligible_clips,
        "excluded_groups": len(groups) - len(eligible),
        "singleton_groups": sum(len(clips) == 1 for clips in groups.values()),
        "same_text_groups": sum(len(clips) > 1 and key not in eligible
                                for key, clips in groups.items()),
    }
    if len(eligible) < count // 2:
        raise ValueError("insufficient eligible exact-length groups")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    pairs, donors = [], {}
    for key in eligible[:count // 2]:
        first = rng.choice(groups[key])
        second = rng.choice([c for c in groups[key] if fingerprints[c]["normalized_reference"]
                             != fingerprints[first]["normalized_reference"]])
        pairs.append([first, second])
        donors.update({first: second, second: first})
    ids = sorted(donors)
    return {"selected_ids": ids, "donor_map": {c: donors[c] for c in ids}, "pairs": pairs,
            "fingerprints": {c: fingerprints[c] for c in ids}, "eligibility": eligibility,
            "selection_seed": seed, "donor_map_hash": digest(donors)}


def _donors(ids: Sequence[str], donors: Mapping[str, str]) -> None:
    if (not ids or list(ids) != sorted(set(ids)) or set(ids) != set(donors)
            or any(d == c or d not in donors or donors[d] != c for c, d in donors.items())):
        raise ValueError("donor map must contain sorted unique reciprocal non-self pairs")


def build_condition_samples(
    samples_by_id: Mapping[str, PhoenixSample], selected_ids: Sequence[str],
    donor_map: Mapping[str, str], condition: str,
) -> list[PhoenixSample]:
    """Clone visual tensors, dataclasses.replace recipient metadata, never mutate inputs.
    Both-shuffled streams always come from the same exact-shape donor. Cloning
    protects cached originals even if a downstream adapter modifies its inputs.
    """
    if condition not in CONDITIONS:
        raise ValueError("unknown grounding condition")
    _donors(selected_ids, donor_map)
    result = []
    for clip in selected_ids:
        recipient, donor = samples_by_id[clip], samples_by_id[donor_map[clip]]
        if (recipient.clip_id != clip or donor.clip_id != donor_map[clip]
                or recipient.signer != "Signer03" or donor.signer != "Signer03"
                or _reference(recipient.text) == _reference(donor.text)
                or any(getattr(recipient, stream).shape != getattr(donor, stream).shape
                       for stream in ("spatial", "motion"))):
            raise ValueError("donor identity/reference/exact feature shape mismatch")
        spatial = donor if condition in {"shuffle_spatial", "shuffle_both"} else recipient
        motion = donor if condition in {"shuffle_motion", "shuffle_both"} else recipient
        result.append(replace(recipient, spatial=spatial.spatial.clone(),
                              motion=motion.motion.clone()))
    return result


def score_condition(model, samples_by_id: Mapping[str, PhoenixSample],
                    selected_ids: Sequence[str], donor_map: Mapping[str, str],
                    condition: str, split_hash: str,
                    on_batch_end: Callable[[int], None] | None = None) -> dict:
    """Fresh batch4 subset generation through reference-free score_pilot_batches."""
    samples = build_condition_samples(samples_by_id, selected_ids, donor_map, condition)
    # DataLoader supports list datasets at runtime; torch 2.0's stub requires Dataset.
    loader = DataLoader(cast(Dataset[PhoenixSample], samples),
                        batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_phoenix,
                        generator=torch.Generator().manual_seed(0))
    report = score_pilot_batches(model, loader, tuple(selected_ids), "dev", split_hash,
                                beam_size=5, max_length=64, on_batch_end=on_batch_end)
    report["condition"] = condition
    for item in report["items"]:
        item["donor_id"] = donor_map[item["clip_id"]]
    return report


def _repetition(predictions: list[str]) -> dict:
    lengths = [len(p.split()) for p in predictions]
    repeated = sum(len(p.split()) - len(set(p.split())) for p in predictions)
    counts = Counter(predictions)
    return {"unique_predictions": len(counts),
            "unique_prediction_rate": len(counts) / len(predictions),
            "duplicate_prediction_rate": 1 - len(counts) / len(predictions),
            "most_common_prediction_count": max(counts.values()),
            "repeated_token_fraction": repeated / sum(lengths) if sum(lengths) else 0.0,
            "predictions_with_repeated_tokens": sum(len(p.split()) > len(set(p.split()))
                                                    for p in predictions),
            "mean_whitespace_length": mean(lengths), "min_whitespace_length": min(lengths),
            "max_whitespace_length": max(lengths), "whitespace_lengths": lengths}


def summarize_grounding(normal: dict, reports_by_condition: Mapping[str, dict]) -> dict:
    """Return conditions -> metrics, normal-relative deltas/change rate, repetition,
    donor_consistency (all and informative recipients/pairs). Match rates use exact
    strings; informative means recipient and donor fresh normal predictions differ.
    No semantic ratings or full-dev inference can be inferred from these statistics.
    """
    ids = [item["clip_id"] for item in normal["items"]]
    donors = {item["clip_id"]: item["donor_id"] for item in normal["items"]}
    _donors(ids, donors)
    originals = {item["clip_id"]: item["prediction"] for item in normal["items"]}
    informative = [c for c in ids if originals[c] != originals[donors[c]]]
    result = {}
    for condition, report in reports_by_condition.items():
        if condition not in CONDITIONS or [i["clip_id"] for i in report["items"]] != ids:
            raise ValueError("summary condition/order mismatch")
        if any(i["reference"] != n["reference"] or i["donor_id"] != donors[i["clip_id"]]
               for i, n in zip(report["items"], normal["items"], strict=True)):
            raise ValueError("summary recipient reference/donor mismatch")
        predictions = {i["clip_id"]: i["prediction"] for i in report["items"]}
        matches = sum(predictions[c] == originals[donors[c]] for c in ids)
        info_matches = sum(predictions[c] == originals[donors[c]] for c in informative)
        result[condition] = {
            "metrics": report["metrics"],
            "metric_deltas": {key: value - normal["metrics"][key]
                              for key, value in report["metrics"].items()},
            "prediction_change_rate": sum(predictions[c] != originals[c] for c in ids) / len(ids),
            "repetition": _repetition(list(predictions.values())),
            "donor_consistency": {"matches": matches, "rate": matches / len(ids),
                                  "informative_recipients": len(informative),
                                  "informative_pairs": len(informative) // 2,
                                  "informative_matches": info_matches,
                                  "informative_rate": (info_matches / len(informative)
                                                       if informative else None)},
        }
    return {"conditions": result, "subset_clips": len(ids),
            "interpretation": "Conditional exact-length subset diagnostic; stream mismatch "
            "can alter distributions. Dependence does not locate a projector bottleneck "
            "or prove semantic grounding. BF16/batch context may affect donor consistency."}


def _payload(value: dict) -> dict:
    return {key: val for key, val in value.items() if key != "content_hash"}


def _publish(path: Path, value: dict) -> None:
    value = {**value, "content_hash": digest(_payload(value))}
    if path.is_symlink():
        raise ValueError("artifact cannot be a symlink")
    if path.exists():
        if read_json(path) != value:
            raise ValueError("completed artifact differs; refusing overwrite")
        return
    _publish_report(path, value)


def publish_condition(path: Path, report: dict, binding: dict) -> None:
    """No-clobber durable JSON, exact identical publication is a read-only reuse."""
    _publish(path, {**report, "status": "complete", "binding": binding,
                    "binding_hash": digest(binding), "run_policy": RUN_POLICY})


def read_condition(path: Path, binding: dict, ids: Sequence[str],
                   donors: Mapping[str, str]) -> dict:
    """Validate complete report content/provenance, recipient order and donor tracking."""
    _donors(ids, donors)
    if path.is_symlink() or not path.is_file():
        raise ValueError("completed condition missing or symlink")
    value = read_json(path)
    if (value.get("binding") != binding or value.get("binding_hash") != digest(binding)
            or value.get("run_policy") != RUN_POLICY or value.get("status") != "complete"):
        raise ValueError("condition provenance drift")
    if value.get("content_hash") != digest(_payload(value)):
        raise ValueError("condition content hash mismatch")
    items = value.get("items", [])
    if ([i["clip_id"] for i in items] != list(ids)
            or any(i.get("donor_id") != donors[i["clip_id"]]
                   or not isinstance(i.get("prediction"), str) for i in items)):
        raise ValueError("condition recipient order/donor/prediction mismatch")
    metrics = evaluate_translations([i["prediction"] for i in items],
                                    [i["reference"] for i in items])
    if metrics != value.get("metrics"):
        raise ValueError("condition metrics mismatch")
    return value


def _output_path(args) -> Path:
    raw = args.output_dir
    output = raw.resolve()
    sources = [getattr(args, name.replace("-", "_")) for name in INPUTS]
    sources += list(MODELS.values()) + [args.annotation.parent]
    # Protect model cache before even creating the diagnostic state/lock.
    for run in MODELS.values():
        budget = run / "budget.json"
        if budget.is_file():
            cache = read_json(budget).get("identity", {}).get("config", {}).get(
                "model", {}).get("cache_dir")
            if cache:
                sources.append(Path(cache))
    # Block symlink components as well as the leaf; source links remain read-only.
    if (any(p.is_symlink() for p in (raw, *raw.parents))
            or any(output == p.resolve() or output.is_relative_to(p.resolve())
                   for p in sources) or (output.exists() and not output.is_dir())):
        raise ValueError("output must be a separate non-symlink diagnostic directory")
    for path in (output, output.with_name(output.name + ".lock")):
        if path.is_symlink():
            raise ValueError("output/lock symlink forbidden")
    if output.exists() and any(output.iterdir()) and not (output / "state.json").is_file():
        raise ValueError("nonempty output lacks diagnostic state")
    if output.exists() and any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("output artifact symlinks forbidden")
    return output


def _prepare(args) -> tuple[dict, dict, dict, dict]:
    """Read-only admission. Full source checks; access samples on dev only."""
    adapted = read_json(args.adapted_root / "complete/identity.json")
    original = {name.replace("-", "_"): getattr(args, name.replace("-", "_"))
                for name in INPUTS if name != "adapted-root"}
    protocol, views = load_e3_views(original, args.adapted_root, adapted)
    if (protocol.get("protocol_hash") != PROTOCOL_HASH or protocol.get("decoder") != DECODER
            or protocol.get("supervision_policy") != POLICY
            or protocol.get("human_review_status") != "not_assessed"):
        raise ValueError("frozen protocol/decoder/supervision mismatch")
    group, view = protocol["split"]["groups"]["dev"], views["dev"]
    ids = tuple(group["clip_ids"])
    if (len(ids) != 582 or len(set(ids)) != 582 or group.get("signers") != ["Signer03"]
            or tuple(row["fileid"] for row in view.records) != ids
            or any(row["signer"] != "Signer03" for row in view.records)):
        raise ValueError("requires exact ordered 582 Signer03 dev clips")
    spatial = FeatureManifest.load(args.adapted_root / "complete/manifest.json")
    motion = FeatureManifest.load(args.motion_manifest)
    selection = select_donor_pairs(view.records,
                                  {c: spatial.require("train", c).length for c in ids},
                                  {c: motion.require("train", c).length for c in ids}, args.count)
    samples = {c: view[ids.index(c)] for c in selection["selected_ids"]}
    features = {}
    for clip, sample in samples.items():
        fp = selection["fingerprints"][clip]
        if (sample.text != fp["reference"]
                or sample.clip_id != clip or sample.signer != "Signer03"
                or sample.spatial.shape != (fp["spatial_length"], 2048)
                or sample.motion.shape != (fp["motion_length"], 1024)):
            raise ValueError("selected feature/reference drift")
        features[clip] = {
            "spatial": file_hash(args.adapted_root / spatial.require("train", clip).path),
            "motion": file_hash(args.motion_root / motion.require("train", clip).path),
        }
    models = {}
    files = {name: path for name, path in original.items()
             if name not in {"dataset", "dino_root", "motion_root"}}
    files.update({
        "dataset_source": args.dataset / "source.json",
        "dataset_version": args.dataset / "version.json",
        "dino_manifest": args.dino_root / "complete/manifest.json",
        "dino_rows": args.dino_root / "complete/frame_rows.json",
        "adapted_identity": args.adapted_root / "complete/identity.json",
        "adapted_manifest": args.adapted_root / "complete/manifest.json",
        "adapted_rows": args.adapted_root / "complete/frame_rows.json",
    })
    lineage = {"adaptation_checkpoint_hash": adapted["checkpoint_hash"],
               "adapted_encoder_key": digest(adapted),
               "adapted_manifest_hash": file_hash(files["adapted_manifest"]),
               "adapted_frame_rows_hash": file_hash(files["adapted_rows"])}
    for seed, run in MODELS.items():
        model_selection = validate_projector_selection(run, protocol, ids)
        identity = model_selection["identity"]
        config = identity["config"]
        data, model, trainer = (config.get(key, {}) for key in ("data", "model", "trainer"))
        paths = {"spatial_root": args.adapted_root,
                 "spatial_manifest": files["adapted_manifest"],
                 "motion_root": args.motion_root, "motion_manifest": args.motion_manifest,
                 "annotation_root": args.annotation.parent}
        if (identity.get("projector") != "linear" or identity.get("seed", 0) != seed
                or model_selection["step"] != 16000 or data.get("batch_size") != 4
                or data.get("num_workers") != 0 or model.get("use_in_context") is not False
                or model.get("num_in_context", 0) != 0 or model.get("warm_up_steps") != 2000
                or trainer.get("max_steps") != 16000
                or trainer.get("accumulate_grad_batches") != 1
                or any(not data.get(k) or Path(data[k]).resolve() != p.resolve()
                       for k, p in paths.items())
                or any(identity.get(k) != v for k, v in lineage.items())):
            raise ValueError("linear seed/config/cache lineage mismatch")
        cache = model.get("cache_dir")
        if cache and args.output_dir.resolve().is_relative_to(Path(cache).resolve()):
            raise ValueError("output lies in model cache")
        checkpoint = torch.load(model_selection["checkpoint"], map_location="cpu",
                                weights_only=False)
        if (digest(checkpoint.get("run_metadata")) != digest(identity)
                or type(checkpoint.get("global_step")) is not int
                or checkpoint["global_step"] != 16000
                or not isinstance(checkpoint.get("state_dict"), dict)):
            raise ValueError("checkpoint full metadata/global_step mismatch")
        del checkpoint
        models[seed] = model_selection
        for name in ("budget", "run-status", "selected-checkpoint",
                     *(f"dev-{step}" for step in DEV_STEPS)):
            files[f"seed{seed}/{name}"] = run / f"{name}.json"
    files.update(inference_source_files(ROOT))
    binding = {"run_policy": RUN_POLICY, "split": "dev", "protocol_hash": PROTOCOL_HASH,
               "split_hash": protocol["split_hash"], "protocol_content_hash": digest(protocol),
               "decoding": DECODER, "decoder_hash": digest(DECODER), "batch_size": 4,
               "dev_clips": len(ids), "dev_ids_hash": digest(ids),
               "selection": selection, "selection_hash": digest(selection),
               "donor_map_hash": selection["donor_map_hash"], "features": features,
               "source_paths": {k: str(getattr(args, k.replace("-", "_")).resolve())
                                for k in INPUTS},
               "source_hashes": {k: file_hash(p) for k, p in files.items()},
               "models": {str(seed): {"run": str(MODELS[seed].resolve()),
                                      "checkpoint_hash": s["checkpoint_hash"],
                                      "checkpoint_step": s["step"],
                                      "identity_hash": digest(s["identity"])}
                          for seed, s in models.items()},
               "gpu_cap_seconds": args.gpu_cap_seconds,
               "supervision_policy": POLICY, "human_review_status": "not_assessed"}
    return binding, selection, samples, models


def _condition_binding(binding: dict, seed: int, condition: str) -> dict:
    return {"provenance_hash": digest(binding), "seed": seed, "condition": condition,
            "checkpoint_hash": binding["models"][str(seed)]["checkpoint_hash"],
            "decoder_hash": binding["decoder_hash"], "donor_map_hash": binding["donor_map_hash"]}


def _read_reports(output: Path, binding: dict, selection: dict) -> dict:
    reports = {str(seed): {} for seed in MODELS}
    for seed in MODELS:
        for condition in CONDITIONS:
            path = output / f"seed{seed}-{condition}.json"
            if path.exists():
                report = read_condition(path, _condition_binding(binding, seed, condition),
                                        selection["selected_ids"], selection["donor_map"])
                if (report.get("condition") != condition or report.get("split") != "dev"
                        or report.get("split_hash") != binding["split_hash"]
                        or report.get("decoding") != DECODER
                        or any(i["reference"] != selection["fingerprints"][i["clip_id"]]
                               ["reference"].lower() for i in report["items"])):
                    raise ValueError("completed condition decoder/reference/split drift")
                reports[str(seed)][condition] = report
    return reports


def _summary(binding: dict, selection: dict, reports: dict, state: dict) -> dict:
    complete = all(len(reports[str(seed)]) == len(CONDITIONS) for seed in MODELS)
    seeds = {}
    for seed in MODELS:
        by_condition = reports[str(seed)]
        seeds[str(seed)] = {
            **binding["models"][str(seed)],
            "completed_conditions": list(by_condition),
            "condition_states": {c: "complete" if c in by_condition else "pending"
                                 for c in CONDITIONS},
            "grounding": (summarize_grounding(by_condition["normal"], by_condition)
                          if "normal" in by_condition else None),
        }
    return {"run_policy": RUN_POLICY, "status": "complete" if complete else "partial",
            "binding_hash": digest(binding), "selection": selection, "seeds": seeds,
            "elapsed_seconds": state["elapsed_seconds"],
            "peak_gpu_bytes": state.get("peak_gpu_bytes", 0),
            "gpu_cap_seconds": binding["gpu_cap_seconds"],
            "scope": "Signer03 exact-length conditional subset; no broad bottleneck claim",
            "bootstrap": "not_performed"}


def _validate_existing(output: Path, binding: dict, selection: dict, state: dict | None) -> dict:
    if state and (state.get("run_policy") != RUN_POLICY
                  or state.get("status") not in {"partial", "in_progress", "complete"}):
        raise ValueError("invalid diagnostic state")
    if state and state.get("binding") not in (None, binding):
        raise ValueError("diagnostic source/checkpoint/decoder/map provenance drift")
    reports = _read_reports(output, binding, selection)
    for path, expected in ((output / "provenance.json", {"binding": binding}),
                           (output / "selection.json", selection)):
        if path.exists():
            _publish(path, expected)  # Existing path: comparison only, never write.
        elif state and state.get("binding") is not None:
            raise ValueError("saved diagnostic provenance/selection missing")
    if (output / "summary.json").exists():
        if state is None:
            raise ValueError("summary without state")
        summary = read_json(output / "summary.json")
        expected = _summary(binding, selection, reports, state)
        _publish(output / "summary.json", expected)
        if summary.get("status") != "complete":
            raise ValueError("final summary is incomplete")
    elif state and state.get("status") == "complete":
        raise ValueError("completed summary missing")
    return reports


@contextmanager
def _deadline(seconds: float):
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError("grounding cumulative stage budget exhausted")
    if seconds <= 0:
        raise TimeoutError("grounding cumulative stage budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL,
                             max(0.001, old_timer[0] - (time.monotonic() - started)), old_timer[1])


def diagnose(args) -> dict:
    """CLI implementation. Preflight read-only; run lock spans validation/publication."""
    if args.mode not in {"preflight", "run"}:
        raise ValueError("invalid diagnostic mode")
    if type(args.gpu_cap_seconds) is not int or not 0 < args.gpu_cap_seconds <= MAX_SECONDS:
        raise ValueError("gpu cap must be 1..3600 seconds")
    if type(args.count) is not int or args.count <= 0 or args.count % 2:
        raise ValueError("count must be positive and even")
    output = _output_path(args)
    if args.mode == "preflight":
        binding, selection, _, _ = _prepare(args)
        state = read_json(output / "state.json") if (output / "state.json").exists() else None
        reports = _validate_existing(output, binding, selection, state)
        return {"status": "ready", "run_policy": RUN_POLICY, "selection": selection,
                "binding_hash": digest(binding), "models": binding["models"],
                "completed_conditions": {s: list(r) for s, r in reports.items()},
                "output_dir": str(output), "gpu_cap_seconds": args.gpu_cap_seconds}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _output_path(args)
        output.mkdir(exist_ok=True)
        state_path = output / "state.json"
        state = read_json(state_path) if state_path.exists() else {
            "run_policy": RUN_POLICY, "binding": None, "elapsed_seconds": 0.0,
            "status": "partial", "peak_gpu_bytes": 0,
        }
        spent = state.get("elapsed_seconds")
        if (state.get("run_policy") != RUN_POLICY or not isinstance(spent, (float, int))
                or isinstance(spent, bool)
                or not math.isfinite(spent) or spent < 0):
            raise ValueError("invalid diagnostic state/budget")
        # Identical completed reuse never updates state, even if validation fails.
        if state.get("status") == "complete":
            binding, selection, _, _ = _prepare(args)
            _validate_existing(output, binding, selection, state)
            return read_json(output / "summary.json")
        if state.get("status") == "in_progress":
            spent += max(0.0, time.time() - state["updated_at"])
        started = time.monotonic()
        prepared = None
        if state.get("binding") is not None:
            # Admit a bound partial run before updating even its heartbeat/state.
            # Otherwise source drift would modify artifacts while rejecting reuse.
            with _deadline(args.gpu_cap_seconds - spent - RESERVE_SECONDS):
                prepared = _prepare(args)
                _validate_existing(output, prepared[0], prepared[1], state)
        reports = {str(seed): {} for seed in MODELS}
        binding = selection = None
        model = None
        def heartbeat(count=0):
            state.update(elapsed_seconds=spent + time.monotonic() - started,
                         updated_at=time.time())
            if model is not None:
                state["peak_gpu_bytes"] = max(state["peak_gpu_bytes"],
                                               torch.cuda.max_memory_allocated())
            atomic_json(state_path, state)
            if state["elapsed_seconds"] >= args.gpu_cap_seconds - RESERVE_SECONDS:
                raise TimeoutError("grounding cumulative stage budget exhausted")
        try:
            # Completed reuse is CPU-only but still validates all frozen source bytes.
            remaining = (args.gpu_cap_seconds - spent - RESERVE_SECONDS
                         - (time.monotonic() - started))
            with _deadline(remaining):
                state.update(status="in_progress", stage="validation", error=None)
                heartbeat()
                binding, selection, samples, models = prepared or _prepare(args)
                if state.get("binding") not in (None, binding):
                    raise ValueError("diagnostic source/checkpoint/decoder/map provenance drift")
                if state.get("binding") is not None and any(
                    not (output / name).is_file() for name in ("provenance.json", "selection.json")
                ):
                    raise ValueError("saved diagnostic provenance/selection missing")
                state["binding"] = binding
                _publish(output / "provenance.json", {"binding": binding})
                _publish(output / "selection.json", selection)
                reports = _read_reports(output, binding, selection)
                if (output / "summary.json").exists():
                    summary = read_json(output / "summary.json")
                    # Runtime of CPU-only revalidation is not a new GPU-stage charge.
                    expected = _summary(binding, selection, reports, {
                        "elapsed_seconds": summary["elapsed_seconds"],
                        "peak_gpu_bytes": summary["peak_gpu_bytes"],
                    })
                    _publish(output / "summary.json", expected)
                    state.update(status="complete", elapsed_seconds=summary["elapsed_seconds"])
                    atomic_json(state_path, state)
                    return summary
                heartbeat()
                for seed in MODELS:
                    missing = [c for c in CONDITIONS if c not in reports[str(seed)]]
                    if not missing:
                        continue
                    if not torch.cuda.is_available():
                        raise RuntimeError("grounding run requires CUDA")
                    heartbeat()
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        model = _load_projector_model(models[seed]).to("cuda").eval()
                        heartbeat()
                        for condition in missing:
                            state.update(stage=f"seed{seed}/{condition}")
                            heartbeat()
                            report = score_condition(model, samples, selection["selected_ids"],
                                                     selection["donor_map"], condition,
                                                     binding["split_hash"], heartbeat)
                            publish_condition(output / f"seed{seed}-{condition}.json", report,
                                              _condition_binding(binding, seed, condition))
                            reports[str(seed)][condition] = read_condition(
                                output / f"seed{seed}-{condition}.json",
                                _condition_binding(binding, seed, condition),
                                selection["selected_ids"], selection["donor_map"])
                            heartbeat()
                            atomic_json(output / "progress.json",
                                        _summary(binding, selection, reports, state))
                    finally:
                        if model is not None:
                            state["peak_gpu_bytes"] = max(state["peak_gpu_bytes"],
                                                           torch.cuda.max_memory_allocated())
                        del model
                        model = None
                        gc.collect()
                        torch.cuda.empty_cache()
                heartbeat()
                summary = _summary(binding, selection, reports, state)
                _publish(output / "summary.json", summary)
                state.update(status="complete", stage="complete")
                atomic_json(state_path, state)
                return read_json(output / "summary.json")
        except BaseException as error:
            state.update(status="partial", error=type(error).__name__,
                         elapsed_seconds=spent + time.monotonic() - started,
                         updated_at=time.time())
            atomic_json(state_path, state)
            if binding is not None and selection is not None and state.get("binding") == binding:
                atomic_json(output / "progress.json", _summary(binding, selection, reports, state))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-cap-seconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--count", type=int, default=128)
    for name, path in INPUTS.items():
        parser.add_argument("--" + name, type=Path, default=path)
    print(json.dumps(diagnose(parser.parse_args()), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
