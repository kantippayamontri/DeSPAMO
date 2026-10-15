"""CPU-only admission for the independently identified paper reproduction."""

import json
from pathlib import Path

from tools.dinov3.identity import digest, sha256_file
from tools.spamo_repro.data import admit_data
from tools.spamo_repro.recipe import optimizer_updates_per_epoch

POLICY = "original-spamo-paper40-released-features-seed0-v1"
REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
_APPROVED = {
    "policy": POLICY,
    "run_root": f"/home/kan/datasets/despamo/{POLICY}",
    "checkpoint_root": f"/mnt/b/DeSpaMo/checkpoints-v1/{POLICY}",
    "control_root": "/home/kan/datasets/despamo/checkpoint-storage-v1",
    "original_paused_run": (
        "/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v3/runs/"
        "translation-reference-seed0-fe5c22c66305d9e3bea3074792c0f64955b2c01c4b61bf92656448c0f7ae5671"
    ),
    "author_commit": "aee7f8b6ea8201b710690adcce6f46260dd11bba",
    "model": {
        "name": "google/flan-t5-xl",
        "revision": REVISION,
        "cache_dir": "/home/kan/datasets/spamo/hf_cache",
        "max_text_length": 512,
        "max_frame_len": 512,
        "spatial_crop_mode": "full",
        "spatial_dim": 2048,
        "motion_dim": 1024,
        "adapter_dim": 768,
        "language_dim": 2048,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "prompt": "Translate the given sentence into {}.",
        "use_in_context": True,
        "num_in_context": 3,
        "vt_pooling": "legacy_mean",
        "vt_weight": 1.0,
        "warm_up_steps": 4000,
    },
    "data": {
        "annotation_root": "/home/kan/Research/SpaMo/preprocess/Phoenix14T",
        "spatial_root": "/home/kan/datasets/spamo/features/vit_feat_Phoenix14T",
        "motion_root": "/home/kan/datasets/spamo/features/mae_feat_Phoenix14T",
        "spatial_manifest": "/home/kan/datasets/spamo/features/manifests/phoenix14t_spatial.json",
        "motion_manifest": "/home/kan/datasets/spamo/features/manifests/phoenix14t_motion.json",
        "batch_size": 8,
        "num_workers": 0,
    },
    "optimizer": {"learning_rate": 0.0001, "weight_decay": 0.01},
    "recipe": {
        "epochs": 40,
        "batch_size": 8,
        "accumulation": 2,
        "vt_warmup_steps": 4000,
        "lr_warmup_steps": 10000,
        "peak_lr": 0.0001,
        "min_lr": 0.00005,
        "dev_every": 2,
    },
    "seed": 0,
    "comparison": {"enabled": True},
}


UPSTREAM_POLICY = "original-spamo-upstream-released-recipe-seed0-v1"
# Recovered from released spamo.ckpt Lightning state: lr 6e-4, 222000-update
# cosine horizon (500 epochs x 444), warm_up_steps 0, EarlyStopping patience 50.
_UPSTREAM = json.loads(json.dumps(_APPROVED))
_UPSTREAM.update(
    policy=UPSTREAM_POLICY,
    run_root=f"/home/kan/datasets/despamo/{UPSTREAM_POLICY}",
    checkpoint_root=f"/mnt/b/DeSpaMo/checkpoints-v1/{UPSTREAM_POLICY}",
    optimizer={"learning_rate": 0.0006, "weight_decay": 0.01},
    recipe={
        "epochs": 500,
        "max_epochs": 500,
        "batch_size": 8,
        "accumulation": 2,
        "learning_rate": 0.0006,
        "lr_schedule": "upstream_cosine_10pct_warmup_zero_floor",
        "vt_warmup_steps": 0,
        "dev_every": 2,
        "patience_checks": 50,
        "selection": "dev_bleu4_max",
    },
)
_UPSTREAM["model"]["warm_up_steps"] = 0

OWN_CLIP_POLICY = "clip-s2-own-features-spamo-ckpt-recipe-200ep-seed0-v1"
_OWN_SPATIAL = (
    "/home/kan/datasets/despamo/official-phoenix14t-clip-s2-features-v1/"
    "spatial-f0ea5e0b10af4a796375bcfd935f69c9988caec9e7ddb6c27de0e0c35a098488"
)
_OWN_MOTION = (
    "/home/kan/datasets/despamo/official-phoenix14t-features-v1/"
    "motion-1123ff445677bf310ac0b8c9b45cac4f690b9102f5dad1960aa2fef04937e82d"
)
# User-approved: spamo.ckpt recipe, own CLIP+S2/VideoMAE features, 200 epochs,
# 22,200-update linear warmup then cosine to 0 at 88,800 updates.
_OWN_CLIP = json.loads(json.dumps(_UPSTREAM))
_OWN_CLIP.update(
    policy=OWN_CLIP_POLICY,
    run_root=f"/home/kan/datasets/despamo/{OWN_CLIP_POLICY}",
    checkpoint_root=f"/mnt/b/DeSpaMo/checkpoints-v1/{OWN_CLIP_POLICY}",
)
_OWN_CLIP["data"].update(
    spatial_root=_OWN_SPATIAL,
    motion_root=_OWN_MOTION,
    spatial_manifest=f"{_OWN_SPATIAL}/complete/manifest.json",
    motion_manifest=f"{_OWN_MOTION}/complete/manifest.json",
)
_OWN_CLIP["recipe"].update(
    epochs=200,
    max_epochs=200,
    lr_schedule="cosine_explicit_warmup_zero_floor",
    lr_warmup_updates=22200,
)
_ADMITTED = {POLICY: _APPROVED, UPSTREAM_POLICY: _UPSTREAM, OWN_CLIP_POLICY: _OWN_CLIP}


def validate_config(config: dict) -> int:
    """Reject undeclared recipe, initialization, path and test-selection changes."""
    try:
        canonical = json.dumps(config, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Exact approved configuration required") from exc
    expected = _ADMITTED.get(str(config.get("policy"))) if isinstance(config, dict) else None
    if expected is None or canonical != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Exact approved configuration required")
    return recipe_for(config)["total_updates"]


def recipe_for(config: dict) -> dict:
    per_epoch = optimizer_updates_per_epoch(7096, 8, 2)
    if config["policy"] == POLICY:
        return {
            "updates_per_epoch": per_epoch,
            "max_epochs": 40,
            "total_updates": per_epoch * 40,
            "learning_rate": 1e-4,
            "lr_schedule": "paper_warm10000_cosine_floor",
            "vt_warmup_steps": 4000,
            "dev_every": 2,
            "patience_checks": None,
        }
    r = config["recipe"]
    total = per_epoch * r["max_epochs"]
    return {
        "updates_per_epoch": per_epoch,
        "max_epochs": r["max_epochs"],
        "total_updates": total,
        "lr_warmup_updates": r.get("lr_warmup_updates", int(total * 0.1)),
        "learning_rate": r["learning_rate"],
        "lr_schedule": r["lr_schedule"],
        "vt_warmup_steps": r["vt_warmup_steps"],
        "dev_every": r["dev_every"],
        "patience_checks": r["patience_checks"],
    }


def prepare(config: dict) -> dict:
    """Bind actual local source/data/pretrained bytes without constructing a model.

    This produces provenance evidence, not GPU or storage admission. Dataset
    objects remain outside the serializable experiment identity.
    """
    total = validate_config(config)
    root = Path(__file__).resolve().parents[2]
    source_paths = sorted((root / "src").rglob("*.py"))
    for namespace in ("spamo_repro", "dinov3", "benchmark", "followup"):
        source_paths.extend(sorted((root / "tools" / namespace).rglob("*.py")))
    source_paths += [
        root / "tools/spamo_repro/config.json",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    if any(path.is_symlink() or not path.is_file() for path in source_paths):
        raise ValueError("Executed source must be local regular files")
    sources = {str(path.relative_to(root)): sha256_file(path) for path in source_paths}
    cache = Path(config["model"]["cache_dir"]) / "models--google--flan-t5-xl"
    snapshot = cache / "snapshots" / REVISION
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    required = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json",
        "tokenizer.json",
        *index["weight_map"].values(),
    }
    assets = {}
    for name in sorted(required):
        path = snapshot / name
        if (
            Path(name).name != name
            or not path.is_file()
            or not path.resolve().is_relative_to(cache.resolve())
        ):
            raise ValueError("Pinned pretrained assets required")
        assets[str(path)] = sha256_file(path)
    data = admit_data(config)
    identity = {
        "schema_version": 1,
        "config": json.loads(json.dumps(config)),
        "optimizer_updates": total,
        "microbatches_per_epoch": 887,
        "optimizer_updates_per_epoch": 444,
        "sources": sources,
        "assets": assets,
        "annotations_and_manifests": data["bindings"],
        "ordered_ids": data["ordered_ids"],
        "feature_hashes": data["feature_hashes"],
        "assumptions": {
            "seed_batch_accumulation": "author-code-derived, not proven paper-run values",
            "decoder": "cached batch1/beam5/max64 deterministic clean-dev protocol",
            "text_limit": "512 tokens, original port configuration; parity audit still required",
            "accumulation_tail": (
                "flush last microbatch with divisor2; VT negatives remain per-microbatch"
            ),
            "initialization": "fresh pretrained base and fresh trainable parameters",
            "test": "annotation identity only; separate scoring authorization required",
        },
    }
    return {"identity": identity, "identity_digest": digest(identity), "data": data}
