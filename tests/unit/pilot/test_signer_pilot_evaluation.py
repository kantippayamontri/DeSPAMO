from dataclasses import replace

import pytest
import torch
from torch import nn

from despamo.data.batch import PhoenixSample, collate_phoenix
from despamo.evaluation.signer_pilot import choose_checkpoint, pilot_prompts, score_pilot_batches


def batch(references=("eins.", "zwei."), en=("one", "two")):
    samples = [
        PhoenixSample(
            f"dev-{i}", "Signer03", text, "GLOSS", en[i], f"fr-{i}", f"es-{i}",
            torch.ones(20, 2) * (i + 1), torch.ones(3, 2),
        )
        for i, text in enumerate(references)
    ]
    return collate_phoenix(samples)


def test_heldout_references_cannot_affect_pilot_prompts():
    original = batch()
    changed = replace(original, texts=("private-a", "private-b"),
                      en_texts=("changed", "changed"),
                      fr_texts=("changed", "changed"),
                      es_texts=("changed", "changed"))
    assert pilot_prompts(original, "Translate into {}.") == ["Translate into German."] * 2
    assert pilot_prompts(changed, "Translate into {}.") == pilot_prompts(original, "Translate into {}.")


class FakeAdapter(nn.Module):
    def forward(self, spatial, spatial_mask, motion, motion_mask):
        return spatial, spatial_mask


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate_text(self, visual, mask, prompts, mode, beam_size, max_length):
        self.calls.append((tuple(prompts), mode, beam_size, max_length))
        return [f"translated-{int(visual[i, 0, 0])}." for i in range(len(prompts))]


class FakeModel(nn.Module):
    prompt_template = "Translate into {}."

    def __init__(self):
        super().__init__()
        self.visual_adapter = FakeAdapter()
        self.language_model = FakeDecoder()


def test_complete_corpus_scores_once_and_never_prompts_with_reference(monkeypatch):
    from despamo.evaluation import signer_pilot

    calls = []

    def score(predictions, references):
        calls.append((predictions[:], references[:]))
        return {"bleu4": 4.5, "rougeL_f1": 0.12}

    monkeypatch.setattr(signer_pilot, "evaluate_translations", score)
    model = FakeModel()
    source = batch()
    result = score_pilot_batches(model, [source], ("dev-0", "dev-1"), "dev", "split-hash")
    assert len(calls) == 1 and calls[0] == (["translated-1.", "translated-2."], ["eins.", "zwei."])
    assert result["metrics"]["bleu4"] == 4.5 and len(result["items"]) == 2
    assert model.language_model.calls == [
        (("Translate into German.", "Translate into German."), "deterministic", 5, 64)
    ]
    changed = replace(source, texts=("changed-a", "changed-b"),
                      en_texts=("private", "private"))
    after = score_pilot_batches(model, [changed], ("dev-0", "dev-1"), "dev", "split-hash")
    assert [item["prediction"] for item in after["items"]] == [
        item["prediction"] for item in result["items"]
    ]
    assert model.language_model.calls[-1][0] == model.language_model.calls[0][0]


def test_dev_accounting_hook_runs_after_every_batch_and_failure_prevents_partial_score(monkeypatch):
    from despamo.evaluation import signer_pilot

    scores, seen = [], []
    monkeypatch.setattr(
        signer_pilot,
        "evaluate_translations",
        lambda predictions, references: scores.append(len(predictions)) or {"bleu4": 1.0},
    )

    def account(count):
        seen.append(count)
        if len(seen) == 2:
            raise RuntimeError("six-hour budget")

    samples = [batch(references=(f"text-{i}.",), en=(f"en-{i}",)) for i in range(2)]
    samples[1] = replace(samples[1], clip_ids=("dev-1",))
    with pytest.raises(RuntimeError, match="budget"):
        score_pilot_batches(
            FakeModel(), samples, ("dev-0", "dev-1"), "dev", "split-hash", on_batch_end=account
        )
    assert seen == [1, 1] and scores == []


@pytest.mark.parametrize("ids", [("dev-1", "dev-0"), ("dev-0",), ("dev-0", "dev-0")])
def test_missing_reordered_or_duplicate_clip_ids_rejected(ids):
    with pytest.raises(ValueError, match="clip"):
        score_pilot_batches(FakeModel(), [batch()], ids, "dev", "split-hash")


def test_checkpoint_choice_is_full_dev_bleu_with_earliest_step_tie():
    ids = ("dev-0", "dev-1")

    def report(step, bleu):
        return {
            "split": "dev", "split_hash": "same", "checkpoint_step": step,
            "checkpoint_hash": str(step),
            "items": [{"clip_id": clip_id} for clip_id in ids],
            "metrics": {"bleu4": bleu, "rougeL_f1": 0.1},
        }

    reports = [report(100, 4.5), report(30, 4.5), report(60, 3.0)]
    assert choose_checkpoint(reports, ids)["checkpoint_step"] == 30
    for broken in (
        [reports[0], {**reports[1], "items": [{"clip_id": "dev-0"}]}],
        [reports[0], {**reports[1], "split_hash": "different"}],
        [reports[0], {**reports[1], "metrics": {"bleu4": float("nan")}}],
    ):
        with pytest.raises(ValueError, match="dev|checkpoint"):
            choose_checkpoint(broken, ids)


@pytest.mark.parametrize("split_name,count", [("dev", 582), ("test", 768)])
def test_complete_pilot_split_scores_once_without_official_642_assumption(
    split_name, count, monkeypatch
):
    from despamo.evaluation import signer_pilot

    calls = []

    def score(predictions, references):
        calls.append((len(predictions), len(references)))
        return {"bleu4": 4.5, "rougeL_f1": 0.1}

    monkeypatch.setattr(signer_pilot, "evaluate_translations", score)
    samples = [
        PhoenixSample(
            f"{split_name}-{index:04}", "Signer03" if split_name == "dev" else "Signer07",
            "target text.", "g", "en", "fr", "es", torch.ones(20, 2), torch.ones(3, 2),
        )
        for index in range(count)
    ]
    batches = [collate_phoenix(samples[start : start + 16]) for start in range(0, count, 16)]
    ids = tuple(item.clip_id for item in samples)
    report = score_pilot_batches(FakeModel(), batches, ids, split_name, "frozen")
    assert calls == [(count, count)] and len(report["items"]) == count
    assert report["items"][0]["clip_id"] == ids[0]
    assert report["items"][-1]["clip_id"] == ids[-1]
