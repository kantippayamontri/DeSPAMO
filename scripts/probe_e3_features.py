"""Subset probe gate: check geometry, signer leakage, and handshape retention

This script extracts adapted features for a small deterministic sample of logical
train clips (~600 by default), avoiding the full 9.8-hour extraction cost. It
computes the three gate metrics that determine whether a new adaptation variant
deserves a full run:
  1. `std_ratio`: must stay near 1.0 (cone collapse measured 0.12).
  2. `signer_top1`: should drop relative to original DINO (measured 0.989 vs 0.979).
  3. `handshape_top1`: must hold or improve over original (measured +0.07).
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
from tools.dinov3.storage import atomic_json


def read_json(path: Path):
    """Local reader: the extractor env lacks the main stack's pydantic dependency."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

PROTOCOL_HASH = "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7"
DATASET_KEY = "b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a"


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


def _load_subset_features(clip_ids: tuple[str, ...], args,
                          model) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from tools.dinov3.encoder import extract
    source = read_json(args.dataset / "source.json")
    clips = {c["clip_id"]: c for c in source["clips"]}
    orig, adap = {}, {}
    for cid in clip_ids:
        c = clips[cid]
        orig_path = args.original_dino_root / "train" / f"{cid}.npy"
        orig[cid] = np.load(orig_path, allow_pickle=False)
        frame_paths = [(args.frames / f["path"]).resolve() for f in c["frames"]]
        arr, _ = extract(frame_paths, model, "cuda", batch_size=8)
        adap[cid] = arr
    return orig, adap


def _std_ratio(orig_feats: dict[str, np.ndarray],
               adap_feats: dict[str, np.ndarray], clip_ids: tuple[str, ...]) -> float:
    # Use middle sampled frame per clip so frames are aligned across clips
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
    if len(set(vals[c] for c in sub)) != 7 or len(sub) < 35:
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
    summary = {
        "status": "ready",
        "adaptation_checkpoint_hash": selected_meta["checkpoint_hash"],
        "adaptation_step": selected_meta.get("step"),
        "clips": len(sample_ids),
        "clips_per_signer": args.clips_per_signer,
    }
    if args.mode == "preflight":
        return summary

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
            hand[side] = {"original": ho["top1"], "adapted": ha["top1"],
                          "delta": ha["top1"] - ho["top1"]}

    report = {
        "status": "complete",
        "adaptation_run": args.adapt_run.name,
        "adaptation_checkpoint_hash": selected_meta["checkpoint_hash"],
        "adaptation_step": selected_meta.get("step"),
        "clips": len(sample_ids),
        "geometry": {"std_ratio": ratio},
        "signer_probe": {
            "original": sig_orig,
            "adapted": sig_adap,
            "delta_top1": sig_adap["top1"] - sig_orig["top1"],
        },
        "handshape_probe": hand,
        "gate_pass": (
            ratio > 0.60
            and (sig_adap["top1"] - sig_orig["top1"]) <= 0.005
            and all(h["delta"] >= -0.02 for h in hand.values())
        ),
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clips-per-signer", type=int, default=85)
    args = p.parse_args()
    print(json.dumps(run_gate(args), indent=2))


if __name__ == "__main__":
    main()
