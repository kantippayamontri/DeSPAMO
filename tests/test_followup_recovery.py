from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.followup.evaluation import score_batches
from tools.followup.recipe import PromptPool


def pool():
    return PromptPool(
        [
            dict(
                fileid=str(i),
                text=f"german{i}",
                en_text=f"en{i}",
                fr_text=f"fr{i}",
                es_text=f"es{i}",
            )
            for i in range(8)
        ],
        0,
    )


class Visual(nn.Module):
    def forward(self, spatial, spatial_mask, motion, motion_mask):
        return spatial, spatial_mask


class Language(nn.Module):
    def generate_text(self, visual, valid, prompts, mode, beam, length):
        self.prompts = prompts
        return ["a valid german sentence"] * len(prompts)


def model():
    m = nn.Module()
    m.anchor = nn.Parameter(torch.zeros(1))
    m.visual_adapter = Visual()
    m.language_model = Language()
    return m


def batch(texts=("some held out sentence",)):
    n = len(texts)
    return SimpleNamespace(
        clip_ids=tuple(f"dev{i}" for i in range(n)),
        texts=texts,
        spatial=torch.zeros(n, 2, 4),
        spatial_mask=torch.ones(n, 2, dtype=torch.bool),
        motion=torch.zeros(n, 1, 2),
        motion_mask=torch.ones(n, 1, dtype=torch.bool),
    )


def test_evaluation_uses_only_precommitted_train_partner_and_exact_ids():
    m, p, b = model(), pool(), batch()
    report = score_batches(m, [b], b.clip_ids, p, "dev", "split_hash")
    assert report["items"][0]["exemplar_ids"] == [p.eval_id]
    assert "some held out sentence" not in m.language_model.prompts[0]
    assert report["decoding"]["in_context"] and report["decoding"]["num_in_context"] == 3
    with pytest.raises(ValueError, match="order"):
        score_batches(m, [b], ("wrong",), p, "dev", "split_hash")


def test_evaluation_rejects_reference_overlap_before_generation():
    m, p = model(), pool()
    b = batch((p.by_id[p.eval_id]["text"],))
    with pytest.raises(ValueError, match="overlap"):
        score_batches(m, [b], b.clip_ids, p, "dev", "split_hash")
    assert not hasattr(m.language_model, "prompts")


def test_archived_candidate_recovers_missing_receipt(tmp_path, monkeypatch):
    import tools.followup.train as t
    from tools.benchmark.compact_checkpoint import save_snapshot

    root = tmp_path / "run"
    root.mkdir()
    archive = tmp_path / "archive"
    monkeypatch.setattr(t, "ARCHIVE", archive)
    state = dict(
        identity={"policy": "test"},
        step=1774,
        frozen_fingerprint="base",
        weights={"weight": torch.ones(1)},
    )
    native = root / "candidate-epoch-2.pt"
    save_snapshot(native, state)
    dest = archive / root.name / "model-epoch-2.pt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(native.read_bytes())
    assert t.candidate_to_archive(root, state, 2) == dest
    assert dest.with_suffix(".json").is_file()
    dest.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        t.candidate_to_archive(root, state, 2)


def test_completion_published_before_ledger_close(tmp_path):
    from tools.followup.train import publish_completion

    class Ledger:
        def close(self, status):
            assert (tmp_path / "complete.json").exists()
            assert status == "complete"

    publish_completion(tmp_path, Ledger(), {"status": "complete", "stopped_updates": 35480})
