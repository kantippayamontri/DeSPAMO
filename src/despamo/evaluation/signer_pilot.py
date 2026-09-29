"""Reference-free prompts and complete logical-split pilot evaluation."""

import math
from collections.abc import Callable
from contextlib import nullcontext

import torch

from despamo.evaluation.metrics import evaluate_translations


def pilot_prompts(batch, template: str) -> list[str]:
    """Never place any held-out translation/reference in generation context."""
    if not isinstance(template, str) or template.count("{}") != 1:
        raise ValueError("pilot translation prompt needs one language placeholder")
    return [template.format("German") for _ in batch.clip_ids]


def score_pilot_batches(
    model,
    loader,
    expected_ids: tuple[str, ...],
    split_name: str,
    split_hash: str,
    *,
    beam_size: int = 5,
    max_length: int = 64,
    on_batch_end: Callable[[int], None] | None = None,
) -> dict:
    if (
        split_name not in {"dev", "test"}
        or not split_hash
        or not expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or beam_size < 1
        or max_length < 1
    ):
        raise ValueError("invalid pilot evaluation split/clip IDs or decoding")
    model.eval()
    first = next(model.parameters(), torch.empty(0))
    device = first.device
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    items = []
    with torch.inference_mode(), autocast:
        for batch in loader:
            visual, valid = model.visual_adapter(
                batch.spatial.to(device),
                batch.spatial_mask.to(device),
                batch.motion.to(device),
                batch.motion_mask.to(device),
            )
            prompts = pilot_prompts(batch, model.prompt_template)
            predictions = model.language_model.generate_text(
                visual, valid, prompts, "deterministic", beam_size, max_length
            )
            if len(predictions) != len(batch.clip_ids):
                raise ValueError("pilot predictions/clip count mismatch")
            items.extend(
                {"clip_id": clip_id, "prediction": prediction, "reference": reference.lower()}
                for clip_id, prediction, reference in zip(
                    batch.clip_ids, predictions, batch.texts, strict=True
                )
            )
            if on_batch_end is not None:
                on_batch_end(len(batch.clip_ids))
    if tuple(item["clip_id"] for item in items) != expected_ids:
        raise ValueError("missing, duplicate or reordered pilot clip IDs")
    metrics = evaluate_translations(
        [item["prediction"] for item in items], [item["reference"] for item in items]
    )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("non-finite complete-corpus pilot metrics")
    return {
        "split": split_name,
        "split_hash": split_hash,
        "items": items,
        "metrics": metrics,
        "decoding": {
            "mode": "deterministic",
            "beam_size": beam_size,
            "max_length": max_length,
            "in_context": False,
        },
    }


def choose_checkpoint(reports: list[dict], expected_ids: tuple[str, ...]) -> dict:
    if not reports or not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("empty dev reports or duplicate expected clips")
    hashes = {report.get("split_hash") for report in reports}
    steps = [report.get("checkpoint_step") for report in reports]
    for report in reports:
        items = report.get("items") or []
        bleu = report.get("metrics", {}).get("bleu4")
        if (
            report.get("split") != "dev"
            or tuple(item["clip_id"] for item in items) != expected_ids
            or not report.get("checkpoint_hash")
            or type(report.get("checkpoint_step")) is not int
            or report["checkpoint_step"] <= 0
            or not isinstance(bleu, (int, float))
            or not math.isfinite(bleu)
        ):
            raise ValueError("incomplete or invalid dev checkpoint report")
    if len(hashes) != 1 or None in hashes or len(set(steps)) != len(steps):
        raise ValueError("dev checkpoint reports have mixed split or duplicate step")
    return min(reports, key=lambda report: (-report["metrics"]["bleu4"], report["checkpoint_step"]))
