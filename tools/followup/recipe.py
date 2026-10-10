"""Approved schedule, report-derived stopping and train-only ICL provenance."""

import math
import random

TEMPLATE = "Translate the given sentence into German."


def schedule(batch_size: int) -> dict:
    if batch_size != 8:
        raise ValueError("approved physical batch8 required")
    per = 7096 // batch_size
    return dict(
        batch_size=batch_size,
        steps_per_epoch=per,
        total_steps=200 * per,
        epochs=200,
        dev_epochs=list(range(2, 201, 2)),
        vt_warmup_steps=4000,
        lr_warmup_steps=10000,
        vt_pooling="legacy_mean",
        accumulation=1,
        min_epochs=40,
        patience=10,
        min_delta=0.10,
        prompt_max_tokens=1024,
        activation_checkpointing=True,
    )


def group_lr(k: int, recipe: dict, group: str) -> float:
    if type(k) is not int or not 1 <= k <= recipe["total_steps"]:
        raise ValueError("one-based update outside declared horizon")
    if group not in ("adapter", "flan"):
        raise ValueError("unknown optimizer group")
    peak, floor = (3e-4, 5e-5) if group == "adapter" else (5e-5, 1e-5)
    warm = recipe["lr_warmup_steps"]
    if k <= warm:
        return peak * k / warm
    progress = (k - warm) / (recipe["total_steps"] - warm)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def stop_state(reports: list[dict], recipe: dict) -> dict:
    rows = sorted(reports, key=lambda r: r["epoch"])
    if [r["epoch"] for r in rows] != recipe["dev_epochs"][: len(rows)]:
        raise ValueError("contiguous declared dev reports required")
    best, anchor, bad = None, None, 0
    for r in rows:
        score = r["metrics"]["bleu4"]
        if not math.isfinite(score):
            raise ValueError("finite dev BLEU4 required")
        if best is None or score > best["metrics"]["bleu4"]:
            best = r
        if r["epoch"] * recipe["steps_per_epoch"] <= recipe["vt_warmup_steps"]:
            continue
        if anchor is None or score - anchor["metrics"]["bleu4"] >= recipe["min_delta"] - 1e-12:
            anchor, bad = r, 0
        else:
            bad += 1
        if r["epoch"] >= recipe["min_epochs"] and bad >= recipe["patience"]:
            if r is not rows[-1]:
                raise ValueError("reports continued after stopping decision")
            return dict(
                stop=True,
                reason="patience",
                best_epoch=best["epoch"],
                anchor_epoch=anchor["epoch"],
                bad_checks=bad,
                epoch=r["epoch"],
            )
    epoch = rows[-1]["epoch"] if rows else 0
    return dict(
        stop=epoch == recipe["epochs"],
        reason="ceiling" if epoch == recipe["epochs"] else None,
        best_epoch=best["epoch"] if best else None,
        anchor_epoch=anchor["epoch"] if anchor else None,
        bad_checks=bad,
        epoch=epoch,
    )


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


class PromptPool:
    """Three multilingual pairs from one train clip; inference never uses batch text."""

    def __init__(self, records: list[dict], seed: int):
        self.rows = sorted(records, key=lambda r: r["fileid"])
        self.by_id = {r["fileid"]: r for r in self.rows}
        if len(self.by_id) != len(self.rows) or len(self.rows) < 2:
            raise ValueError("unique training exemplar IDs required")
        if any(
            not isinstance(r.get(k), str) or not r[k].strip()
            for r in self.rows
            for k in ("text", "en_text", "fr_text", "es_text")
        ):
            raise ValueError("complete training multilingual targets required")
        self.eval_id = random.Random(9000 + seed).choice(self.rows)["fileid"]

    def render(self, row: dict) -> str:
        pairs = " ".join(f"{row[k]}={row['text']}" for k in ("en_text", "fr_text", "es_text"))
        return f"{TEMPLATE} {pairs}"

    def training(
        self, clip_ids: tuple[str, ...], rng: random.Random
    ) -> tuple[list[str], list[str]]:
        refs = {normalized(self.by_id[i]["text"]) for i in clip_ids}
        eligible = [r for r in self.rows if normalized(r["text"]) not in refs]
        if not eligible:
            raise ValueError("no reference-disjoint training exemplar")
        selected = [rng.choice(eligible) for _ in clip_ids]
        return [self.render(r) for r in selected], [r["fileid"] for r in selected]

    def inference(self, clip_ids: tuple[str, ...]) -> tuple[list[str], list[str]]:
        if self.eval_id in clip_ids:
            raise ValueError("inference exemplar clip overlap")
        return (
            [self.render(self.by_id[self.eval_id])] * len(clip_ids),
            [self.eval_id] * len(clip_ids),
        )

    def guard_references(self, references: tuple[str, ...]) -> None:
        if normalized(self.by_id[self.eval_id]["text"]) in {normalized(t) for t in references}:
            raise ValueError("precommitted train exemplar reference overlap; no reselection")
