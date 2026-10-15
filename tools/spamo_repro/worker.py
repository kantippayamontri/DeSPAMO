"""One-shot GPU child: train to a dev boundary, or optimizer-free cached dev."""

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from despamo.data.batch import collate_phoenix
from despamo.evaluation.metrics import evaluate_translations
from despamo.losses.vt_align import VTAlignLoss
from despamo.models.flan_t5 import FlanT5Backbone
from tools.benchmark.compact_checkpoint import frozen_fingerprint, mutable_keys
from tools.clip_plain.generation import generate_cached_text
from tools.dinov3.identity import digest
from tools.dinov3.storage import atomic_json
from tools.followup.recipe import PromptPool
from tools.spamo_repro.model import (
    build_optimizer,
    build_visual_adapter,
    configure_training,
    paper_losses,
)
from tools.spamo_repro.policy import POLICY, REVISION, recipe_for, validate_config
from tools.spamo_repro.state import capture_state, publish_state, restore_state
from tools.spamo_repro.train import accumulated_update, epoch_groups

TOTAL = 17760
PER_EPOCH = 444


class PaperModel(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        torch.manual_seed(config["seed"])
        self.visual_adapter = build_visual_adapter(config)
        self.language_model = FlanT5Backbone.from_pretrained(
            "google/flan-t5-xl",
            config["model"]["cache_dir"],
            512,
            "lora",
            16,
            32,
            0.1,
            revision=REVISION,
        )
        self.vt_align = VTAlignLoss()
        self.rng = random.Random(config["seed"])


def training_icl(config: dict) -> str:
    """Upstream-recipe runs (no VT-only warmup) use in-batch derangement ICL."""
    return "pool" if config["policy"] == POLICY else "derangement"


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def admission(config: dict, root: Path) -> tuple[dict, dict]:
    from tools.spamo_repro.data import admit_data

    data = admit_data(config)
    identity = {
        "policy": config["policy"],
        "config_digest": digest(config),
        "ordered_ids": data["ordered_ids"],
        "feature_digest": digest({k: digest(v) for k, v in data["feature_hashes"].items()}),
        "bindings_digest": digest(data["bindings"]),
    }
    saved = root / "identity.json"
    if saved.exists():
        if json.loads(saved.read_text()) != identity:
            raise ValueError("Run identity drift")
    else:
        atomic_json(saved, identity)
    return data, identity


def prompt_pool(data: dict) -> PromptPool:
    return PromptPool(list(data["datasets"]["train"].records), 0)


def check_prompt_lengths(model: PaperModel, pool: PromptPool) -> dict:
    """Author tokenizes prompts with truncation at FLAN model_max_length=512; record it."""
    tokenizer = model.language_model.tokenizer
    if tokenizer.model_max_length != 512 or model.language_model.max_text_length != 512:
        raise ValueError("Author prompt truncation limit 512 required")
    lengths = [len(tokenizer.encode(pool.render(row))) for row in pool.rows]
    return {
        "longest": max(lengths),
        "rows_over_512": sum(n > 512 for n in lengths),
        "rows": len(lengths),
    }


def batch_for(data: dict, split: str, ids: tuple[str, ...], device: torch.device):
    dataset, lookup = data["datasets"][split], data["lookup"][split]
    batch = collate_phoenix([dataset[lookup[c]] for c in ids])
    if batch.clip_ids != ids:
        raise ValueError("Batch order mismatch")
    return replace(
        batch,
        spatial=batch.spatial.to(device),
        spatial_mask=batch.spatial_mask.to(device),
        motion=batch.motion.to(device),
        motion_mask=batch.motion_mask.to(device),
    )


def receipts(root: Path) -> list[dict]:
    paths = sorted(
        (root / "resumes").glob("resume-update-*.json"), key=lambda p: int(p.stem.split("-")[-1])
    )
    return [json.loads(p.read_text()) for p in paths]


def train(config: dict, root: Path, epoch_target: int, *, max_groups: int | None) -> dict:
    data, identity = admission(config, root)
    recipe = recipe_for(config)
    icl = training_icl(config)
    device = torch.device("cuda")
    model = PaperModel(config)
    pool = prompt_pool(data)
    longest = check_prompt_lengths(model, pool)
    model.to(device)
    configure_training(model.language_model.model)
    model.train()
    base_hash = frozen_fingerprint(model)
    optimizer = build_optimizer(model)
    ckpt_root = Path(config["checkpoint_root"]) if max_groups is None else root / "probe-payload"
    previous = receipts(ckpt_root)
    if previous:
        state = restore_state(previous[-1], model, optimizer, identity, base_hash)
        done = state["step"]
    else:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        done = 0
    if max_groups is not None:
        if previous:
            raise ValueError("Probe is one-shot")
        # Probe crosses the VT-only/joint boundary to measure CE+VT memory.
        done = max(recipe["vt_warmup_steps"], 1) - 1
    elif done % PER_EPOCH:
        raise ValueError("Resumes publish only at epoch boundaries")
    start_epoch = 1 if max_groups is not None else done // PER_EPOCH + 1
    if max_groups is None and start_epoch > epoch_target:
        raise ValueError("Training target already completed")
    ce_window: list[float] = []
    log = (root / "train-log.jsonl").open("a")
    for epoch in range(start_epoch, epoch_target + 1):
        started = time.time()
        groups = epoch_groups(data["ids"]["train"], epoch, seed=config["seed"])
        for index, group in enumerate(groups[:max_groups]):
            batches = [batch_for(data, "train", ids, device) for ids in group]
            done += 1
            result = accumulated_update(
                model,
                optimizer,
                batches,
                done,
                recipe["total_updates"],
                lambda m, b, joint: paper_losses(m, b, joint, pool, icl=icl),
                recipe=recipe,
            )
            if result["CE"] is not None:
                ce_window = (ce_window + [result["CE"]])[-200:]
            if done % 50 == 0 or max_groups is not None:
                log.write(
                    json.dumps({"epoch": epoch, "group": index, **result, "time": time.time()})
                    + "\n"
                )
                log.flush()
        state = capture_state(
            model,
            optimizer,
            identity,
            done,
            {"epoch": epoch, "groups_completed": min(len(groups), max_groups or 444)},
            base_hash=base_hash,
        )
        receipt = publish_state(
            root / "control", ckpt_root / "resumes" / f"resume-update-{done}.pt", state
        )
        log.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "published": receipt["path"],
                    "sha256": receipt["sha256"],
                    "seconds": time.time() - started,
                }
            )
            + "\n"
        )
        log.flush()
        if max_groups is not None:
            break
    out = {
        "status": "complete",
        "epoch": epoch_target,
        "completed_updates": done,
        "probe": max_groups is not None,
        "prompt_lengths": longest,
        "ce_last200_mean": None if not ce_window else sum(ce_window) / len(ce_window),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    name = "probe-train.json" if max_groups is not None else f"train-epoch-{epoch_target}.json"
    atomic_json(root / name, out)
    return out


def dev(config: dict, root: Path, epoch: int, *, max_clips: int | None, approval: dict) -> dict:
    data, identity = admission(config, root)
    device = torch.device("cuda")
    model = PaperModel(config)
    pool = prompt_pool(data)
    pool.guard_references(tuple(r["text"] for r in data["datasets"]["dev"].records))
    model.to(device)
    model.eval()
    base_hash = frozen_fingerprint(model)
    ckpt_root = Path(config["checkpoint_root"]) if max_clips is None else root / "probe-payload"
    wanted = [
        r
        for r in receipts(ckpt_root)
        if r["step"] == (epoch * PER_EPOCH if max_clips is None else r["step"])
    ]
    if not wanted:
        raise ValueError("Missing dev checkpoint")
    receipt = wanted[-1]
    # Weights only: no optimizer object exists in this process.
    state = torch.load(receipt["path"], map_location="cpu", weights_only=False)
    from tools.checkpoint_storage.inventory import file_hash

    if (
        file_hash(Path(receipt["path"])) != receipt["sha256"]
        or state["identity"] != identity
        or state["frozen_fingerprint"] != base_hash
        or set(state["weights"]) != mutable_keys(model)
    ):
        raise ValueError("Dev checkpoint identity/hash mismatch")
    model.load_state_dict(state["weights"], strict=False)
    del state
    events = {"starts": 0, "logits": 0}

    def observe(event: dict) -> None:
        if event["phase"] == "decoder_start":
            if event["decoder_length"] != 1 or event["beam_rows"] != 5:
                raise ValueError("Cached decoder evidence mismatch")
            events["starts"] += 1
        elif event["phase"] == "decoder_logits":
            if event["finite"] is not True or event["shape"][0] != 5:
                raise ValueError("Invalid raw decoder logits")
            events["logits"] += 1

    scope = {"approved": True, "boot_id": approval["boot_id"], "consumed": False}
    items = []
    for clip in data["ids"]["dev"][:max_clips]:
        batch = batch_for(data, "dev", (clip,), device)
        prompts, exemplars = pool.inference(batch.clip_ids)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            visual, mask = model.visual_adapter(
                batch.spatial, batch.spatial_mask, batch.motion, batch.motion_mask
            )
            result = generate_cached_text(
                model.language_model, visual, mask, prompts, on_event=observe, scope=scope
            )
        items.append(
            {
                "clip_id": clip,
                "prediction": result["texts"][0],
                "reference": batch.texts[0].lower(),
                "token_ids": result["token_ids"][0],
                "exemplar_id": exemplars[0],
            }
        )
    if events["starts"] != events["logits"] or not events["starts"]:
        raise ValueError("Decoder evidence incomplete")
    metrics = evaluate_translations(
        [i["prediction"] for i in items], [i["reference"] for i in items]
    )
    report = {
        "epoch": epoch,
        "complete": max_clips is None,
        "clips": len(items),
        "checkpoint_sha256": receipt["sha256"],
        "completed_updates": receipt["step"],
        "metrics": metrics,
        "decoder_steps": events["starts"],
        "exemplar_id": pool.eval_id,
        "decoding": {
            "batch_size": 1,
            "beam_size": 5,
            "max_length": 64,
            "use_cache": True,
            "do_sample": False,
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "items": items,
    }
    (root / "dev").mkdir(exist_ok=True)
    name = "probe-dev.json" if max_clips is not None else f"dev-epoch-{epoch}.json"
    atomic_json(root / "dev" / name, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["train", "dev"], required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--probe", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    approval = json.loads((args.root / "approval.json").read_text())
    if approval.get("approved") is not True or approval.get("boot_id") != boot_id():
        raise RuntimeError("Boot-bound approval required")
    if (args.root / "manual-stop.json").exists():
        raise RuntimeError("Hold active")
    if args.stage == "train":
        train(config, args.root, args.epoch, max_groups=args.probe)
    else:
        dev(config, args.root, args.epoch, max_clips=args.probe, approval=approval)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
