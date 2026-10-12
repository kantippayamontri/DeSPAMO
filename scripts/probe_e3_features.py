"""Versioned matched-frame geometry, signer and handshape diagnostics.

Both encoders are evaluated on the same five source frames per logical train clip.
Use --adapted-root to read verified cached features on CPU; otherwise extract the
adapted subset on CUDA. Missing hand probes produce INCOMPLETE, never PASS.
The historical thresholds remain diagnostic heuristics, not predictors of BLEU.
"""

import argparse
import collections
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from despamo.evaluation.pilot_probe import score_probe, split_probe_clips
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.storage import atomic_json


def read_json(path: Path):
    """Local reader: the extractor env lacks the main stack's pydantic dependency."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

PROTOCOL_HASH = "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7"
DATASET_KEY = "b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a"
PROBE_POLICY = "e3-probe-matched-five-v2"
THRESHOLDS = {"std_ratio_min_exclusive": 0.60, "signer_delta_top1_max": 0.005,
              "hand_delta_top1_min": -0.02}


def _load_sources(args) -> tuple[dict, dict, dict]:
    protocol = read_json(args.protocol)
    source = read_json(args.dataset / "source.json")
    text = read_json(args.text_manifest)
    return protocol, source, text


def _sample_clips(source: dict, train_ids: list[str], per_signer: int,
                  seed: int = 0) -> tuple[str, ...]:
    clips = {c["clip_id"]: c for c in source["clips"]}
    by_signer = collections.defaultdict(list)
    for cid in train_ids:
        c = clips[cid]
        if c.get("signer") not in {"Signer03", "Signer07"}:
            by_signer[c["signer"]].append(cid)
    rng = random.Random(seed)
    sel = []
    for s in sorted(by_signer):
        pool = sorted(by_signer[s])
        k = min(per_signer, len(pool))
        sel.extend(rng.sample(pool, k))
    return tuple(sorted(sel))


def _load_adaptation_model(adapt_run: Path, cache: Path):
    from transformers import AutoModel

    from tools.dinov3.e3_model import attach_late_lora
    from tools.dinov3.e3_train import BASE_SHA, _load_lora_state
    from tools.dinov3.identity import MODEL

    selected = read_json(adapt_run / "selected-checkpoint.json")
    ckpt_path = adapt_run / selected["checkpoint"]
    model = AutoModel.from_pretrained(
        MODEL, revision=BASE_SHA, cache_dir=cache, local_files_only=True,
        use_safetensors=True, trust_remote_code=False,
    ).eval().to("cuda")
    attach_late_lora(model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_lora_state(model, state["model"])
    model.eval()
    return model


def _sampled_features(path: Path, clip: dict) -> np.ndarray:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    count = clip.get("frame_count", len(values))
    indices = [frame["index"] for frame in clip["frames"]]
    if (values.shape != (count, 2048) or values.dtype != np.float32
            or len(indices) != 5 or len(set(indices)) != 5
            or any(type(index) is not int or not 0 <= index < count for index in indices)):
        raise ValueError(f"probe source frame/feature alignment invalid: {clip['clip_id']}")
    sampled = values[indices].copy()
    if not np.isfinite(sampled).all():
        raise ValueError(f"probe sampled features non-finite: {clip['clip_id']}")
    return sampled


def _load_subset_features(clip_ids: tuple[str, ...], args,
                          model) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from tools.dinov3.encoder import extract
    source = read_json(args.dataset / "source.json")
    clips = {c["clip_id"]: c for c in source["clips"]}
    orig, adap = {}, {}
    for cid in clip_ids:
        c = clips[cid]
        orig_path = args.original_dino_root / "train" / f"{cid}.npy"
        orig[cid] = _sampled_features(orig_path, c)
        frame_paths = [(args.frames / f["path"]).resolve() for f in c["frames"]]
        arr, _ = extract(frame_paths, model, "cuda", batch_size=8)
        adap[cid] = arr
    return orig, adap


def _cached_metadata(args, selected: dict, protocol: dict, clip_ids: tuple[str, ...]) -> dict:
    complete = args.adapted_root / "complete"
    identity = read_json(complete / "identity.json")
    original_rows_path = args.original_dino_root / "complete/frame_rows.json"
    original_hash = sha256_file(original_rows_path)
    if (identity.get("checkpoint_hash") != selected["checkpoint_hash"]
            or identity.get("adaptation_run_key") != args.adapt_run.name
            or identity.get("protocol_hash") != protocol["protocol_hash"]
            or identity.get("source_hash") != protocol["source_hash"]
            or identity.get("frame_rows_hash") != original_hash):
        raise ValueError("cached feature checkpoint/protocol/source identity mismatch")
    key = digest(identity)
    manifest_path, rows_path = complete / "manifest.json", complete / "frame_rows.json"
    manifest, rows = read_json(manifest_path), read_json(rows_path)
    records = {record["clip_id"]: record for record in manifest["records"]}
    original_rows = read_json(original_rows_path)["clips"]
    if (manifest.get("schema_version") != 1 or manifest.get("encoder") != key
            or manifest.get("expected_dim") != 2048 or rows.get("encoder_key") != key
            or len(records) != len(manifest["records"])
            or not set(clip_ids) <= records.keys()
            or not set(clip_ids) <= rows.get("clips", {}).keys()
            or not set(clip_ids) <= original_rows.keys()):
        raise ValueError("cached feature manifest/inventory mismatch")
    return {
        "records": {cid: records[cid] for cid in clip_ids},
        "rows": {cid: rows["clips"][cid] for cid in clip_ids},
        "original_rows": {cid: original_rows[cid] for cid in clip_ids},
        "metadata": {"adapted_encoder_key": key, "original_frame_rows_hash": original_hash,
                     "adapted_manifest_hash": sha256_file(manifest_path),
                     "adapted_frame_rows_hash": sha256_file(rows_path)},
    }


def _load_cached_subset_features(clip_ids: tuple[str, ...], args, source: dict,
                                 selected: dict, cached: dict) -> tuple[dict, dict]:
    clips = {clip["clip_id"]: clip for clip in source["clips"]}
    orig, adap = {}, {}
    for cid in clip_ids:
        clip, row = clips[cid], cached["rows"][cid]
        original_row, record = cached["original_rows"][cid], cached["records"][cid]
        count = clip["frame_count"]
        indices = list(range(count))
        receipt = read_json(args.adapted_root / "receipts/train" / f"{cid}.json")
        paths = row.get("paths", [])
        if (row.get("frame_count") != count or original_row.get("frame_count") != count
                or row.get("source_indices") != indices
                or original_row.get("source_indices") != indices
                or receipt.get("source_indices") != indices or len(paths) != count
                or receipt.get("paths") != paths
                or any(Path(paths[frame["index"]]).resolve()
                       != (args.frames / frame["path"]).resolve() for frame in clip["frames"])
                or record != {"clip_id": cid, "split": "train", "path": f"train/{cid}.npy",
                              "length": count, "width": 2048, "dtype": "float32"}):
            raise ValueError(f"cached feature frame alignment mismatch: {cid}")
        original_path = args.original_dino_root / "train" / f"{cid}.npy"
        adapted_path = args.adapted_root / "train" / f"{cid}.npy"
        if (receipt.get("clip_id") != cid
                or receipt.get("checkpoint_hash") != selected["checkpoint_hash"]
                or receipt.get("source_hash") != row.get("source_hash")
                or original_row.get("feature_hash") != sha256_file(original_path)
                or row.get("feature_hash") != sha256_file(adapted_path)
                or receipt.get("feature_hash") != row.get("feature_hash")):
            raise ValueError(f"cached feature/receipt hash mismatch: {cid}")
        orig[cid] = _sampled_features(original_path, clip)
        adap[cid] = _sampled_features(adapted_path, clip)
    return orig, adap


def _std_ratio(orig_feats: dict[str, np.ndarray],
               adap_feats: dict[str, np.ndarray], clip_ids: tuple[str, ...]) -> float:
    # Mean-pool the same five sampled source frames for both encoders.
    xo = np.stack([orig_feats[c].mean(0) for c in clip_ids])
    xa = np.stack([adap_feats[c].mean(0) for c in clip_ids])
    ratios = []
    for lo, hi in ((0, 1024), (1024, 2048)):
        no = xo[:, lo:hi] / (np.linalg.norm(xo[:, lo:hi], axis=1, keepdims=True) + 1e-12)
        na = xa[:, lo:hi] / (np.linalg.norm(xa[:, lo:hi], axis=1, keepdims=True) + 1e-12)
        co = no @ no.T
        ca = na @ na.T
        m = ~np.eye(len(co), dtype=bool)
        so = float(co[m].std())
        sa = float(ca[m].std())
        ratios.append(sa / (so + 1e-12))
    return float(np.mean(ratios))


def _probe_signer(feats: dict[str, np.ndarray], source: dict,
                  clip_ids: tuple[str, ...]) -> dict[str, float]:
    clips = {c["clip_id"]: c for c in source["clips"]}
    labels = [clips[c]["signer"] for c in clip_ids]
    split = split_probe_clips([(c, label) for c, label in zip(clip_ids, labels, strict=True)])
    X = torch.stack([torch.from_numpy(feats[c].mean(0)) for c in clip_ids]).float()
    res = score_probe(X, labels, split, clip_ids)
    return {"top1": res["accuracy"], "balanced": res["balanced_accuracy"],
            "majority": res["majority_baseline"]}


def _probe_handshape(feats: dict[str, np.ndarray], source: dict, text: dict,
                     clip_ids: tuple[str, ...], side: str) -> dict[str, float] | None:
    # side: "left" or "right"
    # Target index for middle frame (frame 2): 4 + 2*3 + (0 if left else 1)
    col = 10 if side == "left" else 11
    recs = {r["clip_id"]: r for r in text["records"]}
    clips = {c["clip_id"]: c for c in source["clips"]}
    vals = {}
    for cid in clip_ids:
        tgt = recs[cid]["targets"][col].get("text")
        if tgt:
            m = re.search(r"finger configuration ([^;]+)", tgt)
            if m:
                vals[cid] = m.group(1).strip()
    keep = [v for v, _ in collections.Counter(vals.values()).most_common(7)]
    sub = tuple(c for c in clip_ids if vals.get(c) in keep)
    counts = collections.Counter(vals[c] for c in sub)
    # A 15% validation partition needs at least seven examples in every class.
    if len(counts) != 7 or any(count < 7 for count in counts.values()):
        return None
    labels = [vals[c] for c in sub]
    split = split_probe_clips([(c, label) for c, label in zip(sub, labels, strict=True)])
    # Middle sampled frame. Original arrays hold every frame and are addressed by the
    # absolute source index; adapted arrays hold only the 5 sampled frames, so there the
    # middle frame is row 2. Pick the convention that matches each array's own length.
    rows = []
    for c in sub:
        arr = feats[c]
        row = 2 if arr.shape[0] == 5 else clips[c]["frames"][2]["index"]
        rows.append(torch.from_numpy(arr[row]))
    X = torch.stack(rows).float()
    res = score_probe(X, labels, split, sub)
    return {"top1": res["accuracy"], "balanced": res["balanced_accuracy"],
            "majority": res["majority_baseline"], "n": len(sub)}


def run_gate(args) -> dict:
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("gate output already exists")
    protocol, source, text = _load_sources(args)
    if protocol.get("protocol_hash") != PROTOCOL_HASH:
        raise ValueError("protocol mismatch")
    selected_meta = read_json(args.adapt_run / "selected-checkpoint.json")
    identity = selected_meta.get("identity", {})
    if identity.get("protocol_hash") != PROTOCOL_HASH:
        raise ValueError("adaptation protocol mismatch")
    train_ids = protocol["split"]["groups"]["train"]["clip_ids"]
    sample_ids = _sample_clips(source, train_ids, args.clips_per_signer)
    clips = {clip["clip_id"]: clip for clip in source["clips"]}
    cached = (_cached_metadata(args, selected_meta, protocol, sample_ids)
              if getattr(args, "adapted_root", None) is not None else None)
    summary = {
        "status": "ready",
        "schema_version": 2,
        "probe_policy": PROBE_POLICY,
        "adaptation_checkpoint_hash": selected_meta["checkpoint_hash"],
        "adaptation_step": selected_meta.get("step"),
        "clips": len(sample_ids),
        "clips_per_signer": args.clips_per_signer,
        "feature_mode": "cached" if cached else "extract",
        "supervision_policy": identity.get("supervision_policy"),
        "human_review_status": identity.get("human_review_status"),
    }
    if cached:
        summary.update(cached["metadata"])
    if args.mode == "preflight":
        return summary

    if cached:
        orig, adap = _load_cached_subset_features(sample_ids, args, source, selected_meta, cached)
    else:
        model = _load_adaptation_model(args.adapt_run, args.cache)
        orig, adap = _load_subset_features(sample_ids, args, model)
    ratio = _std_ratio(orig, adap, sample_ids)
    sig_orig = _probe_signer(orig, source, sample_ids)
    sig_adap = _probe_signer(adap, source, sample_ids)
    hand = {}
    for side in ("left", "right"):
        ho = _probe_handshape(orig, source, text, sample_ids, side)
        ha = _probe_handshape(adap, source, text, sample_ids, side)
        if ho and ha:
            hand[side] = {"status": "available", "original": ho["top1"], "adapted": ha["top1"],
                          "delta": ha["top1"] - ho["top1"],
                          "original_metrics": ho, "adapted_metrics": ha}
        else:
            hand[side] = {"status": "unavailable", "reason":
                          "Insufficient labels for seven supported handshape classes."}
    signer_delta = sig_adap["top1"] - sig_orig["top1"]
    if any(hand[side]["status"] != "available" for side in ("left", "right")):
        passed, gate_status = None, "INCOMPLETE"
    else:
        passed = (ratio > THRESHOLDS["std_ratio_min_exclusive"]
                  and signer_delta <= THRESHOLDS["signer_delta_top1_max"]
                  and all(h["delta"] >= THRESHOLDS["hand_delta_top1_min"] for h in hand.values()))
        gate_status = "PASS" if passed else "FAIL"

    report = {
        **summary,
        "status": "complete",
        "adaptation_run": args.adapt_run.name,
        "adaptation_checkpoint_hash": selected_meta["checkpoint_hash"],
        "adaptation_step": selected_meta.get("step"),
        "clips": len(sample_ids),
        "geometry": {"std_ratio": ratio},
        "signer_probe": {
            "original": sig_orig,
            "adapted": sig_adap,
            "delta_top1": signer_delta,
        },
        "handshape_probe": hand,
        "gate_status": gate_status,
        "gate_pass": passed,
        "thresholds": dict(THRESHOLDS),
        "protocol_hash": protocol["protocol_hash"],
        "source_hash": digest(source),
        "text_manifest_digest": digest(text),
        "sampling": {"frames_per_clip": 5, "pooling": "matched-five-frame-mean", "seed": 0,
                     "clip_ids_hash": digest(sample_ids),
                     "frame_indices_hash": digest([
                         [cid, [frame["index"] for frame in clips[cid]["frames"]]]
                         for cid in sample_ids
                     ])},
        "code_hashes": {"scripts/probe_e3_features.py": sha256_file(Path(__file__)),
                        "src/despamo/evaluation/pilot_probe.py": sha256_file(
                            Path(__file__).resolve().parents[1]
                            / "src/despamo/evaluation/pilot_probe.py")},
    }
    atomic_json(args.output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "run"), required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--text-manifest", type=Path, required=True)
    p.add_argument("--original-dino-root", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--adapt-run", type=Path, required=True)
    p.add_argument("--adapted-root", type=Path,
                   help="Verified complete adapted feature cache; avoids CUDA extraction")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clips-per-signer", type=int, default=85)
    args = p.parse_args()
    print(json.dumps(run_gate(args), indent=2))


if __name__ == "__main__":
    main()
