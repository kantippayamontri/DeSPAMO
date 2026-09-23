import math

import pytest

from despamo.evaluation.metrics import evaluate_translations


def test_exact_predictions_score_perfect_rouge() -> None:
    metrics = evaluate_translations(["es bleibt windig"], ["es bleibt windig"])

    assert metrics["rougeL_f1"] == 1.0
    assert metrics["bleu1"] == 100.0


def test_bleu_orders_use_percent_scale_and_rouge_uses_fraction() -> None:
    metrics = evaluate_translations(["one two three four"], ["one two three four"])

    assert set(metrics) == {
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_f1",
    }
    assert [metrics[f"bleu{order}"] for order in range(1, 5)] == [100.0] * 4
    assert all(metrics[name] == 1.0 for name in ("rougeL_precision", "rougeL_recall", "rougeL_f1"))


def test_partial_translation_reports_distinct_bleu_orders_and_rouge() -> None:
    metrics = evaluate_translations(["one two five four"], ["one two three four"])

    assert metrics["bleu1"] == pytest.approx(75.0)
    assert metrics["bleu2"] == pytest.approx(50.0)
    assert metrics["rougeL_precision"] == pytest.approx(0.75)
    assert metrics["rougeL_recall"] == pytest.approx(0.75)
    assert metrics["rougeL_f1"] == pytest.approx(0.75)
    assert all(math.isfinite(score) for score in metrics.values())


def test_rouge_scores_average_per_clip() -> None:
    metrics = evaluate_translations(
        ["one two three four", "eins"], ["one two five four", "eins bleibt"]
    )

    assert metrics["rougeL_precision"] == pytest.approx((0.75 + 1.0) / 2)
    assert metrics["rougeL_recall"] == pytest.approx((0.75 + 0.5) / 2)
    assert metrics["rougeL_f1"] == pytest.approx((0.75 + 2 / 3) / 2)


@pytest.mark.parametrize(
    ("predictions", "references"),
    [([], []), (["one"], []), ([], ["one"]), (["one", "two"], ["one"])],
)
def test_evaluation_rejects_empty_or_misaligned_batches(
    predictions: list[str], references: list[str]
) -> None:
    with pytest.raises(ValueError, match="equal non-zero length"):
        evaluate_translations(predictions, references)
