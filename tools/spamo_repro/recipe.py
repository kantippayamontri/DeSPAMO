import math


def lr_at_update(update: int, total: int) -> float:
    if (
        type(update) is not int
        or type(total) is not int
        or total <= 10000
        or not 0 <= update <= total
    ):
        raise ValueError("Exact paper optimizer-update schedule required")
    if update <= 10000:
        return 1e-4 * update / 10000
    progress = (update - 10000) / (total - 10000)
    return 5e-5 + 5e-5 * 0.5 * (1 + math.cos(math.pi * progress))


def optimizer_updates_per_epoch(clips: int, batch: int, accumulation: int) -> int:
    if any(type(v) is not int or v <= 0 for v in [clips, batch, accumulation]):
        raise ValueError("Positive physical batch/accumulation counts required")
    return math.ceil(math.ceil(clips / batch) / accumulation)


def joint_update(update: int) -> bool:
    if type(update) is not int or update <= 0:
        raise ValueError("Positive optimizer-update counter required")
    return update > 4000


def _upstream_factor(step: int, total: int, warmup: int | None = None) -> float:
    """transformers.get_cosine_schedule_with_warmup lambda (default warmup int(0.1*total))."""
    warmup = int(total * 0.1) if warmup is None else warmup
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def lr_for(update: int, recipe: dict) -> float:
    """LR used by one-based optimizer update `update` under the declared recipe."""
    total = recipe["total_updates"]
    if type(update) is not int or not 1 <= update <= total:
        raise ValueError("Optimizer update outside declared horizon")
    if recipe["lr_schedule"] == "paper_warm10000_cosine_floor":
        return lr_at_update(update, total)
    if recipe["lr_schedule"] == "upstream_cosine_10pct_warmup_zero_floor":
        # Lightning step-interval LambdaLR: update k uses lambda(k - 1).
        return recipe["learning_rate"] * _upstream_factor(update - 1, total)
    if recipe["lr_schedule"] == "cosine_explicit_warmup_zero_floor":
        return recipe["learning_rate"] * _upstream_factor(
            update - 1, total, recipe["lr_warmup_updates"]
        )
    raise ValueError("Unknown LR schedule")


def joint_for(update: int, recipe: dict) -> bool:
    """Upstream checks global_step <= warm_up_steps before each step (0-based)."""
    if type(update) is not int or update <= 0:
        raise ValueError("Positive optimizer-update counter required")
    warmup = recipe["vt_warmup_steps"]
    # Upstream warm_up_steps=0 still makes global_step 0 (first update) VT-only.
    return update > (1 if warmup == 0 else warmup)
