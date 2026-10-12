"""Runtime integrity guards for transient in-process model corruption."""

from tools.benchmark.compact_checkpoint import frozen_fingerprint

# Uniform over FLAN's 32k vocabulary is ln(32128) ~= 10.38; healthy runs stay < 5.
COLLAPSE_CE = 8.0
COLLAPSE_WINDOW = 200


class RecoverableFault(RuntimeError):
    """Live process state is corrupt; durable checkpoints remain valid for replay."""


def assert_base_intact(model, base_hash: str) -> None:
    if frozen_fingerprint(model) != base_hash:
        raise RecoverableFault("frozen base weights changed in memory")


def check_ce_collapse(history: list[float], step: int, recipe: dict) -> None:
    if step <= recipe["vt_warmup_steps"] + COLLAPSE_WINDOW or len(history) < COLLAPSE_WINDOW:
        return
    recent = history[-COLLAPSE_WINDOW:]
    if sum(recent) / len(recent) >= COLLAPSE_CE:
        raise RecoverableFault("translation CE collapse near uniform")


def check_dev_report(report: dict, epoch: int, recipe: dict) -> None:
    """After joint training starts, one identical prediction for the whole corpus is corrupt."""
    if epoch * recipe["steps_per_epoch"] <= recipe["vt_warmup_steps"]:
        return
    predictions = {item["prediction"] for item in report["items"]}
    if len(predictions) == 1 and len(report["items"]) > 1:
        raise RecoverableFault("identical dev prediction for every clip")
