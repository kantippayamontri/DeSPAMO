import random
from types import SimpleNamespace

import torch

from despamo.losses.vt_align import VTAlignLoss
from tools.followup.recipe import PromptPool
from tools.spamo_repro.model import paper_losses


class Language(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Embedding(8, 4)
        self.base.requires_grad_(False)
        self.lora = torch.nn.Parameter(torch.ones(4))
        self.prompts = []

    def target_embeddings(self, texts):
        return self.base(torch.tensor([[1, 2], [3, 4]])), torch.ones(2, 2, dtype=torch.bool)

    def translation_loss(self, visual, mask, prompts, targets):
        self.prompts = prompts
        return (visual.mean(dim=(0, 1)) * self.lora).square().mean()


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, spatial, spatial_mask, motion, motion_mask):
        return self.linear(spatial), spatial_mask


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_adapter = Visual()
        self.language_model = Language()
        self.vt_align = VTAlignLoss()
        self.rng = random.Random(0)


def setup_model():
    torch.manual_seed(42)
    model = Model()
    batch = SimpleNamespace(
        spatial=torch.randn(2, 3, 4),
        spatial_mask=torch.ones(2, 3, dtype=torch.bool),
        motion=torch.randn(2, 3, 4),
        motion_mask=torch.ones(2, 3, dtype=torch.bool),
        texts=("one.", "two."),
        clip_ids=("one", "two"),
    )
    pool = PromptPool(
        [
            dict(fileid=name, text=name + ".", en_text="en", fr_text="fr", es_text="es")
            for name in ["one", "two", "three", "four"]
        ],
        0,
    )
    return model, batch, pool


def test_vt_only_has_adapter_temperature_gradients_but_no_lora_or_prompt_rng_change():
    model, batch, pool = setup_model()
    before = model.rng.getstate()
    vt, ce = paper_losses(model, batch, False, pool)
    assert ce is None
    vt.backward()
    assert model.visual_adapter.linear.weight.grad is not None
    assert model.vt_align.logit_scale.grad is not None
    assert model.language_model.lora.grad is None
    assert model.language_model.base.weight.grad is None
    assert model.rng.getstate() == before
    assert not model.language_model.prompts


def test_joint_loss_trains_lora_with_reference_disjoint_train_only_multilingual_prompt():
    model, batch, pool = setup_model()
    vt, ce = paper_losses(model, batch, True, pool)
    assert ce is not None
    (vt + ce).backward()
    assert model.language_model.lora.grad is not None
    assert model.language_model.base.weight.grad is None
    for prompt in model.language_model.prompts:
        assert "Translate the given sentence into German." in prompt
        assert "en=" in prompt and "fr=" in prompt and "es=" in prompt
        assert "one." not in prompt and "two." not in prompt


def test_original_padded_mean_symmetric_contrastive_loss_is_preserved():
    model, batch, pool = setup_model()
    vt, _ = paper_losses(model, batch, False, pool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        visual, _ = model.visual_adapter(
            batch.spatial, batch.spatial_mask, batch.motion, batch.motion_mask
        )
        text, _ = model.language_model.target_embeddings(batch.texts)
        logits = (
            torch.nn.functional.normalize(text.float().mean(1), dim=-1)
            @ torch.nn.functional.normalize(visual.float().mean(1), dim=-1).T
        )
        logits = logits * model.vt_align.logit_scale.exp()
        labels = torch.arange(2)
        expected = (
            torch.nn.functional.cross_entropy(logits, labels)
            + torch.nn.functional.cross_entropy(logits.T, labels)
        ) / 2
    torch.testing.assert_close(vt, expected)
