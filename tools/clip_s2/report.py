"""Dev-only reference comparison; stopped predecessor's missing seed is explicit."""

import argparse
import json
from pathlib import Path

from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json


def paired_summary(clip: dict, dino: dict) -> dict:
    if set(clip) != {0, 1}:
        raise ValueError("two completed CLIP reference seeds required")
    changes = {
        str(seed): clip[seed]["bleu4"] - dino[seed]["bleu4"]
        for seed in sorted(clip)
        if seed in dino
    }
    return dict(
        clip_mean_bleu4=sum(r["bleu4"] for r in clip.values()) / 2,
        paired_seed_changes=changes,
        unpaired_seeds=sorted(set(clip) - set(dino)),
        two_seed_paired_mean_change=sum(changes.values()) / 2 if len(changes) == 2 else None,
    )


def completed(root: Path) -> dict:
    result = {}
    for p in (root / "runs").glob("*/complete.json"):
        d = json.loads(p.read_text())
        selected = json.loads((p.parent / "selected-checkpoint.json").read_text())
        i = selected["identity"]
        if i["arm"] != "reference":
            continue
        if i["seed"] in result or d["selected"] != selected:
            raise ValueError("duplicate reference or changed completed selection")
        if sha256_file(p.parent / selected["checkpoint"]) != selected["checkpoint_hash"]:
            raise ValueError("selected weights changed")
        result[i["seed"]] = dict(
            metrics=selected["metrics"],
            identity=i,
            selected_epoch=selected["epoch"],
            stopped_epoch=d["stopped_epoch"],
            stopped_updates=d["stopped_updates"],
            run_root=str(p.parent),
        )
    return result


def compare_dev(clip_root: Path, dino_root: Path) -> dict:
    clip, dino = completed(clip_root), completed(dino_root)
    for seed, c in clip.items():
        if seed not in dino:
            continue
        a, b = c["identity"], dino[seed]["identity"]
        for key in ("recipe", "decoder", "runtime", "common_initialization", "frozen_fingerprint"):
            if a[key] != b[key]:
                raise ValueError(f"unmatched reference condition: {key}")
        if a["data"]["ordered_ids"] != b["data"]["ordered_ids"]:
            raise ValueError("ordered split IDs differ")
        for p, h in b["data"]["bindings"].items():
            if p.endswith("_info_ml.npy") and a["data"]["bindings"].get(p) != h:
                raise ValueError("annotation/caption identity differs")
        for p, h in b["code_hashes"].items():
            if "/src/" in p and a["code_hashes"].get(p) != h:
                raise ValueError("downstream source changed")
    summary = paired_summary(
        {s: r["metrics"] for s, r in clip.items()}, {s: r["metrics"] for s, r in dino.items()}
    )
    return dict(
        status="complete",
        split="dev",
        test_scored=False,
        summary=summary,
        clip_runs=clip,
        dino_runs=dino,
        interpretation=(
            "Feature pipeline changes encoder, adaptation status and extraction together; "
            "no adaptation-only or significance claim."
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip-root", type=Path, required=True)
    p.add_argument("--dino-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    atomic_json(a.output, compare_dev(a.clip_root, a.dino_root))


if __name__ == "__main__":
    main()
