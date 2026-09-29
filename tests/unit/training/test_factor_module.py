import pytest
import torch

from despamo.appearance.schema import FACTORS, STABLE
from despamo.data.batch import PhoenixSample, collate_phoenix
from despamo.data.factors import FactorBatch
from despamo.losses.vt_align import VTAlignLoss
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.baseline_module import SpaMoBaselineModule
from despamo.training.factor_module import DualPathModule, ramp
from tests.unit.training.test_baseline_module import FakeLanguageModel


def base_model():
    return SpaMoBaselineModule(
        visual_adapter=SpaMoVisualAdapter(4, 2, 8, 2),
        language_model=FakeLanguageModel(),
        vt_align=VTAlignLoss(initial_logit_scale=0.0),
        prompt_template="Translate into {}.",
        use_in_context=False,
        num_in_context=1,
        vt_pooling="masked_mean",
        vt_weight=0.5,
        warm_up_steps=0,
        learning_rate=1e-4,
        weight_decay=0.01,
        seed=0,
    )


def mixed_batch():
    samples = [
        PhoenixSample(f"clip-{i}", "s", f"text-{i}", "g", "en", "es", "fr",
                      torch.randn(20, 4), torch.randn(3, 2))
        for i in range(3)
    ]
    base = collate_phoenix(samples)
    vectors, masks, labels = {}, {}, {}
    for factor in FACTORS:
        shape = (3,) if factor in STABLE else (3, 5)
        vectors[factor] = torch.randn(*shape, 4)
        vectors[factor][-1] = 0
        masks[factor] = torch.ones(shape, dtype=torch.bool)
        masks[factor][-1] = False
        labels[factor] = tuple(f"{i}-{factor}" for i in range(int(masks[factor].numel())))
    return FactorBatch(base, torch.tensor([[1, 3, 5, 7, 9]] * 3), vectors, masks, labels)


@pytest.mark.parametrize(
    "step,expected", [(0, None), (1, 0.0), (6, 0.5), (11, 1.0), (99, 1.0)]
)
def test_post_warmup_auxiliary_ramp(step, expected):
    assert ramp(step, warm=0, total=100) == expected


def test_factor_losses_only_train_after_warmup_and_val_stays_baseline():
    torch.manual_seed(4)
    base = base_model()
    model = DualPathModule(base, list(FACTORS), 4, total_steps=100,
                           provenance={"supervision_policy": "qwen-schema98-unreviewed-v1",
                                       "human_review_status": "not_assessed"})
    batch = mixed_batch()
    warmup = model.compute_losses_at_step(batch, 0)
    assert "factor/biometric" not in warmup
    train = model.compute_losses_at_step(batch, 6)
    assert all(f"factor/{factor}" in train for factor in FACTORS)
    assert all(torch.isfinite(value).all() for value in train.values())
    assert train["combined_loss"] > warmup["combined_loss"]
    train["combined_loss"].backward()
    assert model.visual_adapter.spatial_projector.weight.grad is not None
    assert all(model.auxiliary.heads[factor].weight.grad is not None for factor in FACTORS)
    assert set(model.compute_losses(batch.base)) == {"loss", "contra_loss", "combined_loss"}


def test_checkpoint_requires_exact_unreviewed_identity_and_preserves_rng():
    model = DualPathModule(base_model(), list(FACTORS), 4, total_steps=100,
                           provenance={"supervision_policy": "qwen-schema98-unreviewed-v1",
                                       "human_review_status": "not_assessed",
                                       "records_hash": "a" * 64})
    model.run_metadata = {"config": {"factors": "unreviewed"}, "seed": 0}
    saved = {}
    model.on_save_checkpoint(saved)
    assert saved["factor_provenance"]["records_hash"] == "a" * 64
    model.on_load_checkpoint(saved)
    changed = {**saved, "factor_provenance": {**saved["factor_provenance"],
                                              "human_review_status": "completed"}}
    with pytest.raises(ValueError, match="factor provenance"):
        model.on_load_checkpoint(changed)


def test_nested_batch_transfer_preserves_metadata():
    model = DualPathModule(base_model(), list(FACTORS), 4, 100,
                           {"supervision_policy": "qwen-schema98-unreviewed-v1",
                            "human_review_status": "not_assessed"})
    batch = mixed_batch()
    moved = model.transfer_batch_to_device(batch, torch.device("cpu"), 0)
    assert moved is not batch and moved.base.clip_ids is batch.base.clip_ids
    assert moved.labels is batch.labels
    assert moved.rows.shape == (3, 5)
    assert moved.masks["mouth"].dtype == torch.bool
