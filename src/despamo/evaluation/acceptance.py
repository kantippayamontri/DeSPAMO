import math


def validate_baseline_result(item_count: int, metrics: dict[str, float]) -> None:
    if item_count != 642:
        raise ValueError(f"expected 642 test items, got {item_count}")
    for key, label, target, tolerance in (
        ("bleu4", "BLEU-4", 25.08, 1.0),
        ("rougeL_f1", "ROUGE-L F1", 0.4698, 0.01),
    ):
        value = metrics.get(key)
        if (
            value is None
            or not math.isfinite(value)
            or not target - tolerance <= value <= target + tolerance
        ):
            raise ValueError(f"{label} outside tolerance: {value}")
