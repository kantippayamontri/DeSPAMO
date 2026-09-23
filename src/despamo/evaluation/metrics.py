import math

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


def evaluate_translations(
    predictions: list[str], references: list[str], tokenizer: str = "13a"
) -> dict[str, float]:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must have equal non-zero length")
    metrics = {
        f"bleu{order}": BLEU(max_ngram_order=order, tokenize=tokenizer)
        .corpus_score(predictions, [references])
        .score
        for order in range(1, 5)
    }
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"] for ref, pred in zip(references, predictions, strict=True)
    ]
    metrics["rougeL_precision"] = sum(score.precision for score in scores) / len(scores)
    metrics["rougeL_recall"] = sum(score.recall for score in scores) / len(scores)
    metrics["rougeL_f1"] = sum(score.fmeasure for score in scores) / len(scores)
    if not all(math.isfinite(score) for score in metrics.values()):
        raise ValueError("translation metrics must be finite")
    for order in range(1, 5):
        key = f"bleu{order}"
        metrics[key] = min(metrics[key], 100.0)
    return metrics
