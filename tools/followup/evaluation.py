"""Full-corpus train-only-ICL dev scoring and frozen once-only test journals."""

import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from despamo.evaluation.metrics import evaluate_translations
from tools.benchmark.compact_checkpoint import load_snapshot
from tools.benchmark.translation_data import admit_data, batch_for
from tools.benchmark.translation_model import cuda_batch, optimizer_for
from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json, writer
from tools.followup.model import build_model
from tools.followup.recipe import PromptPool


def score_batches(
    model, loader, expected_ids: tuple[str, ...], pool: PromptPool, split: str, split_hash: str
) -> dict:
    if split not in ("dev", "test") or not expected_ids or not split_hash:
        raise ValueError("valid evaluation corpus required")
    model.eval()
    device = next(model.parameters()).device
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    items = []
    with torch.inference_mode(), context:
        for batch in loader:
            pool.guard_references(batch.texts)
            prompts, exemplars = pool.inference(batch.clip_ids)
            visual, valid = model.visual_adapter(
                batch.spatial.to(device),
                batch.spatial_mask.to(device),
                batch.motion.to(device),
                batch.motion_mask.to(device),
            )
            predictions = model.language_model.generate_text(
                visual, valid, prompts, "deterministic", 5, 64
            )
            items.extend(
                dict(clip_id=i, prediction=p, reference=r.lower(), exemplar_ids=[e])
                for i, p, r, e in zip(
                    batch.clip_ids, predictions, batch.texts, exemplars, strict=True
                )
            )
    if tuple(i["clip_id"] for i in items) != expected_ids:
        raise ValueError("complete ordered corpus required; order mismatch")
    metrics = evaluate_translations(
        [i["prediction"] for i in items], [i["reference"] for i in items]
    )
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError("non-finite corpus metrics")
    return dict(
        split=split,
        split_hash=split_hash,
        items=items,
        metrics=metrics,
        decoding=dict(
            mode="deterministic",
            beam_size=5,
            max_length=64,
            in_context=True,
            num_in_context=3,
            exemplar_id=pool.eval_id,
        ),
    )


def score_run(root: Path) -> dict:
    from tools.followup.train import BASE, CACHE, code_hashes, model_files

    selected = json.loads((root / "selected-checkpoint.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    frozen = json.loads((BASE / "controller/frozen-selections.json").read_text())
    expected = next((r for r in frozen if r["run_root"] == str(root)), None)
    if expected is None or expected["selection_hash"] != sha256_file(
        root / "selected-checkpoint.json"
    ):
        raise ValueError("test requires exact frozen selection")
    identity = selected["identity"]
    if (
        complete["selected"] != selected
        or identity["code_hashes"] != code_hashes()
        or identity["base_files"] != model_files()
    ):
        raise ValueError("completed selected source/base identity changed")
    admission = admit_data()
    if admission["identity"] != identity["data"]:
        raise ValueError("test data identity changed")
    pool = PromptPool(admission["datasets"]["train"].records, identity["seed"])
    if pool.eval_id != identity["decoder"]["exemplar_id"]:
        raise ValueError("inference exemplar identity changed")
    with writer(root / "official-test"):
        target, claim = root / "official-test/report.json", root / "official-test/claim.json"
        binding = dict(
            selection_hash=expected["selection_hash"],
            checkpoint_hash=selected["checkpoint_hash"],
            identity=identity,
        )
        if claim.exists():
            old = json.loads(claim.read_text())
            if old["binding"] != binding:
                raise ValueError("once-only test claim conflict")
            if old["status"] == "complete":
                if sha256_file(target) != old["report_sha256"]:
                    raise ValueError("completed test report changed")
                return json.loads(target.read_text())
        else:
            atomic_json(claim, dict(binding=binding, status="in_progress"))
        model, base_hash = build_model(identity["seed"], identity["arm"], CACHE)
        optimizer = optimizer_for(model)
        load_snapshot(
            root / selected["checkpoint"],
            model,
            optimizer,
            identity,
            base_hash=base_hash,
            replay=False,
        )
        model.to("cuda").eval()
        ids, items = admission["ids"]["test"], []
        for offset in range(0, len(ids), 4):
            path = root / "official-test" / f"batch-{offset}.json"
            batch_ids = ids[offset : offset + 4]
            batch = batch_for(admission, "test", batch_ids)
            if path.exists():
                rows = json.loads(path.read_text())
                if (
                    tuple(r["clip_id"] for r in rows) != batch_ids
                    or [r["reference"] for r in rows] != [t.lower() for t in batch.texts]
                    or any(r["exemplar_ids"] != [pool.eval_id] for r in rows)
                ):
                    raise ValueError("test journal order/reference/provenance changed")
            else:
                report = score_batches(
                    model,
                    [cuda_batch(batch)],
                    batch_ids,
                    pool,
                    "test",
                    identity["data"]["ordered_ids"]["test"],
                )
                rows = report["items"]
                atomic_json(path, rows)
            items.extend(rows)
        if tuple(r["clip_id"] for r in items) != ids:
            raise ValueError("complete642test corpus required")
        report = dict(
            split="test",
            items=items,
            selection=selected,
            metrics=evaluate_translations(
                [r["prediction"] for r in items], [r["reference"] for r in items]
            ),
            decoding=identity["decoder"],
        )
        atomic_json(target, report)
        atomic_json(
            claim, dict(binding=binding, status="complete", report_sha256=sha256_file(target))
        )
        return report
