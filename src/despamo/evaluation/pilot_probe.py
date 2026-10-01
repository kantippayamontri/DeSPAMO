"""Diagnostic closed-set signer probe on frozen pilot projector outputs."""

import hashlib
import math
import random
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from despamo.data.phoenix14t import _load_feature


def split_probe_clips(clips: list[tuple[str, str]]) -> dict[str, tuple[str, ...]]:
    """Make deterministic clip-disjoint partitions inside seven train signers."""
    groups: dict[str, list[str]] = defaultdict(list)
    for clip_id, signer in clips:
        if not clip_id or not signer:
            raise ValueError("missing probe clip ID or signer")
        groups[signer].append(clip_id)
    if len(groups) != 7 or len({clip_id for clip_id, _ in clips}) != len(clips):
        raise ValueError("probe requires unique clips from seven signers")
    parts: dict[str, list[str]] = {name: [] for name in ("fit", "validation", "test")}
    for signer, ids in sorted(groups.items()):
        ids.sort()
        seed = int.from_bytes(hashlib.sha256(f"0:{signer}".encode()).digest()[:8], "big")
        random.Random(seed).shuffle(ids)
        fit, validation = math.floor(0.70 * len(ids)), math.floor(0.15 * len(ids))
        if min(fit, validation, len(ids) - fit - validation) < 1:
            raise ValueError("probe requires every signer in every partition")
        parts["fit"].extend(ids[:fit])
        parts["validation"].extend(ids[fit : fit + validation])
        parts["test"].extend(ids[fit + validation :])
    return {name: tuple(sorted(ids)) for name, ids in parts.items()}


def pool_spatial(view, expected_ids: tuple[str, ...]) -> torch.Tensor:
    """Mean-pool complete physical-train DINO frames, aligned to expected IDs."""
    inventory = {record["fileid"] for record in view.records}
    if (not expected_ids or len(inventory) != len(view.records)
            or len(expected_ids) != len(set(expected_ids)) or set(expected_ids) != inventory):
        raise ValueError("probe clip inventory mismatch")
    rows = []
    for clip_id in expected_ids:
        record = view.spatial_manifest.require("train", clip_id)
        if record.width != 2048 or record.length < 1:
            raise ValueError("probe spatial width or length mismatch")
        frames = _load_feature(view.spatial_root, record, "spatial")
        if not torch.isfinite(frames).all():
            raise ValueError("non-finite spatial frames")
        rows.append(frames.mean(dim=0))
    return torch.stack(rows)


def project_pooled(pooled: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply a checkpoint's spatial projector after frame pooling."""
    if (pooled.ndim != 2 or weight.ndim != 2 or bias.ndim != 1
            or weight.shape[1] != pooled.shape[1] or bias.shape[0] != weight.shape[0]
            or not all(torch.isfinite(x).all() for x in (pooled, weight, bias))):
        raise ValueError("invalid projector shape or non-finite values")
    return F.linear(pooled.float(), weight.float(), bias.float())


def score_probe(
    features: torch.Tensor,
    labels: list[str],
    split: dict[str, tuple[str, ...]],
    clip_ids: tuple[str, ...],
) -> dict:
    """Fit fresh deterministic linear probes; score held-out probe clips once."""
    if (features.ndim != 2 or len(features) != len(labels) or len(labels) != len(clip_ids)
            or len(set(clip_ids)) != len(clip_ids) or set(split) != {"fit", "validation", "test"}
            or any(not ids for ids in split.values())
            or set().union(*(set(ids) for ids in split.values())) != set(clip_ids)
            or sum(map(len, split.values())) != len(clip_ids)
            or not torch.isfinite(features).all()):
        raise ValueError("invalid probe feature/partition inventory")
    names = sorted(set(labels))
    if len(names) != 7:
        raise ValueError("probe needs seven train signers")
    mapping = {name: i for i, name in enumerate(names)}
    positions = {clip_id: i for i, clip_id in enumerate(clip_ids)}
    rows = {part: torch.tensor([positions[clip_id] for clip_id in split[part]])
            for part in ("fit", "validation", "test")}
    targets = torch.tensor([mapping[label] for label in labels], dtype=torch.long)
    for indices in rows.values():
        if len(indices) == 0 or set(targets[indices].tolist()) != set(range(7)):
            raise ValueError("probe partition is not seven-class closed-set")
    features = features.float().cpu()
    mean = features[rows["fit"]].mean(dim=0)
    std = features[rows["fit"]].std(dim=0, unbiased=False)
    scaled = (features - mean) / torch.where(std > 0, std, 1.)
    if not torch.isfinite(scaled).all():
        raise ValueError("non-finite normalized probe features")
    x_fit, x_val, x_test = (scaled[rows[name]] for name in ("fit", "validation", "test"))
    y_fit, y_val, y_test = (targets[rows[name]] for name in ("fit", "validation", "test"))

    def accuracy(predicted: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
        hits = predicted == actual
        balanced = sum(hits[actual == label].float().mean().item() for label in range(7)) / 7
        return hits.float().mean().item(), balanced

    best = None
    for decay in (0., 0.0001, 0.01):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = nn.Linear(features.shape[1], 7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=decay)
        for epoch in range(1, 201):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x_fit), y_fit)
            if not torch.isfinite(loss):
                raise ValueError("non-finite probe loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_score = accuracy(model(x_val).argmax(dim=1), y_val)[1]
            key = (val_score, -decay, -epoch)
            if best is None or key > best[0]:
                best = (key, {k: v.detach().clone() for k, v in model.state_dict().items()},
                        decay, epoch)
    assert best is not None
    model.load_state_dict(best[1])
    with torch.no_grad():
        top1, balanced = accuracy(model(x_test).argmax(dim=1), y_test)
    counts = {name: int((y_test == i).sum()) for i, name in enumerate(names)}
    return {
        "accuracy": top1,
        "balanced_accuracy": balanced,
        "majority_baseline": max(counts.values()) / len(y_test),
        "counts": counts,
        "weight_decay": best[2],
        "epoch": best[3],
    }
