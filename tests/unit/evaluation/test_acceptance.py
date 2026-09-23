import pytest

from despamo.evaluation.acceptance import validate_baseline_result


@pytest.mark.parametrize(
    ("bleu4", "rouge_f1"),
    [(25.08, 0.4698), (24.08, 0.4598), (26.08, 0.4798), (24.5, 0.465)],
)
def test_acceptance_includes_documented_tolerance(bleu4: float, rouge_f1: float) -> None:
    validate_baseline_result(642, {"bleu4": bleu4, "rougeL_f1": rouge_f1})


@pytest.mark.parametrize("count", [0, 641, 643])
def test_acceptance_rejects_incomplete_split(count: int) -> None:
    with pytest.raises(ValueError, match="expected 642 test items"):
        validate_baseline_result(count, {"bleu4": 25.08, "rougeL_f1": 0.4698})


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ({"bleu4": 24.079, "rougeL_f1": 0.4698}, "BLEU-4 outside tolerance"),
        ({"bleu4": 26.081, "rougeL_f1": 0.4698}, "BLEU-4 outside tolerance"),
        ({"bleu4": 25.08, "rougeL_f1": 0.4597}, "ROUGE-L F1 outside tolerance"),
        ({"bleu4": 25.08, "rougeL_f1": 0.4799}, "ROUGE-L F1 outside tolerance"),
        ({"bleu4": float("nan"), "rougeL_f1": 0.4698}, "BLEU-4"),
        ({"bleu4": 25.08, "rougeL_f1": float("inf")}, "ROUGE-L F1"),
        ({"rougeL_f1": 0.4698}, "BLEU-4"),
        ({"bleu4": 25.08}, "ROUGE-L F1"),
    ],
)
def test_acceptance_rejects_invalid_metrics(metrics: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_baseline_result(642, metrics)
