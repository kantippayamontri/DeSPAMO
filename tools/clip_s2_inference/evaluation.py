"""Opt-in batch-one dev evaluation; bounded probes cannot become selection reports."""

import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime

import torch

from despamo.evaluation.metrics import evaluate_translations
from tools.clip_s2.data import eval_batches
from tools.clip_s2_inference.generation import generate_cached_text, guard_device


def memory_snapshot(device: torch.device) -> dict:
    if device.type == "cpu":
        return dict(
            allocated_bytes=None,
            reserved_bytes=None,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
        )
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def score_dev_cached(
    model,
    admission: dict,
    pool,
    split_hash: str,
    *,
    max_batches: int | None = None,
    gpu_reviewed: bool = False,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    expected = admission["ids"]["dev"]
    if not expected or len(set(expected)) != len(expected) or not split_hash:
        raise ValueError("nonempty unique dev corpus and split hash required")
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("bounded probe batch count must be positive")
    device = next(model.parameters()).device
    guard_device(device, gpu_reviewed=gpu_reviewed)
    # Original raw annotations, rather than punctuation-normalized batch text,
    # establish reference-disjoint precommitted training exemplar before any generation.
    pool.guard_references(tuple(r["text"] for r in admission["datasets"]["dev"].records))
    items, telemetry = [], []
    modes = [(module, module.training) for module in model.modules()]

    def emit(phase: str, base: dict, **fields):
        event = dict(base, phase=phase, utc=datetime.now(UTC).isoformat(), **fields)
        telemetry.append(event)
        if on_event is not None:
            on_event(event)

    try:
        model.eval()
        limit = len(expected) if max_batches is None else min(max_batches, len(expected))
        # Slice admitted IDs first: iterator never reads an unnecessary next batch.
        bounded = dict(admission, ids={**admission["ids"], "dev": expected[:limit]})
        for index, batch in enumerate(eval_batches(bounded, "dev", batch_size=1)):
            if tuple(batch.clip_ids) != expected[index : index + 1]:
                raise ValueError("dev batch order differs from admitted IDs")
            base = dict(
                batch_index=index,
                clip_ids=list(batch.clip_ids),
                device=str(device),
                spatial_lengths=batch.spatial_mask.sum(1).tolist(),
                motion_lengths=batch.motion_mask.sum(1).tolist(),
            )
            begin = time.perf_counter()
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                emit("batch_start", base, memory=memory_snapshot(device))
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                # Per-batch autocast scope bounds cache lifetime across the corpus.
                with torch.inference_mode(), context:
                    visual, valid = model.visual_adapter(
                        batch.spatial.to(device),
                        batch.spatial_mask.to(device),
                        batch.motion.to(device),
                        batch.motion_mask.to(device),
                    )
                    prompts, exemplars = pool.inference(batch.clip_ids)

                    def generation_start(metadata, base=base):
                        emit("generation_start", base, memory=memory_snapshot(device), **metadata)

                    predictions = generate_cached_text(
                        model.language_model,
                        visual,
                        valid,
                        prompts,
                        beam_size=5,
                        max_length=64,
                        gpu_reviewed=gpu_reviewed,
                        on_start=generation_start,
                    )
                items.extend(
                    dict(clip_id=i, prediction=p, reference=r.lower(), exemplar_ids=[e])
                    for i, p, r, e in zip(
                        batch.clip_ids, predictions, batch.texts, exemplars, strict=True
                    )
                )
                emit(
                    "batch_complete",
                    base,
                    seconds=time.perf_counter() - begin,
                    memory=memory_snapshot(device),
                )
            except Exception as error:
                # Do not query CUDA memory, synchronize, or restore CUDA RNG after a fault.
                try:
                    emit(
                        "batch_error",
                        base,
                        seconds=time.perf_counter() - begin,
                        error_type=type(error).__name__,
                        error=str(error),
                        memory=dict(
                            allocated_bytes=None,
                            reserved_bytes=None,
                            peak_allocated_bytes=None,
                            peak_reserved_bytes=None,
                        ),
                    )
                except Exception as observer_error:
                    error.add_note(f"Error telemetry failed: {observer_error}")
                raise
    finally:
        for module, flag in modes:
            module.training = flag
    processed = tuple(row["clip_id"] for row in items)
    if processed != expected[:limit]:
        raise ValueError("incomplete, duplicate or reordered requested dev predictions")
    complete = max_batches is None and processed == expected
    metrics = None
    if complete:
        metrics = evaluate_translations(
            [r["prediction"] for r in items], [r["reference"] for r in items]
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("non-finite dev metrics")
    return dict(
        split="dev",
        split_hash=split_hash,
        complete=complete,
        diagnostic_only=not complete,
        processed_clips=len(items),
        expected_clips=len(expected),
        items=items,
        metrics=metrics,
        telemetry=telemetry,
        decoding=dict(
            mode="deterministic",
            beam_size=5,
            max_length=64,
            batch_size=1,
            use_cache=True,
            in_context=True,
            num_in_context=3,
            exemplar_id=pool.eval_id,
        ),
    )
