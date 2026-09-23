import pytest
import torch

from despamo.data.batch import PhoenixSample, collate_phoenix


def _sample(clip_id: str, spatial_length: int, motion_length: int) -> PhoenixSample:
    return PhoenixSample(
        clip_id,
        f"signer-{clip_id}",
        f"text-{clip_id}.",
        f"GLOSS-{clip_id}",
        f"english-{clip_id}",
        f"spanish-{clip_id}",
        f"french-{clip_id}",
        torch.ones(spatial_length, 4),
        torch.ones(motion_length, 3),
    )


def test_collate_pads_independent_modalities_and_builds_boolean_masks() -> None:
    batch = collate_phoenix([_sample("a", 2, 4), _sample("b", 4, 1)])

    assert batch.clip_ids == ("a", "b")
    assert batch.signers == ("signer-a", "signer-b")
    assert batch.texts == ("text-a.", "text-b.")
    assert batch.glosses == ("GLOSS-a", "GLOSS-b")
    assert batch.en_texts == ("english-a", "english-b")
    assert batch.es_texts == ("spanish-a", "spanish-b")
    assert batch.fr_texts == ("french-a", "french-b")
    assert batch.spatial.shape == (2, 4, 4)
    assert batch.motion.shape == (2, 4, 3)
    assert batch.spatial_mask.dtype == batch.motion_mask.dtype == torch.bool
    assert batch.spatial_mask.tolist() == [[True, True, False, False], [True] * 4]
    assert batch.motion_mask.tolist() == [[True] * 4, [True, False, False, False]]
    assert torch.all(batch.spatial[0, 2:] == 0)
    assert torch.all(batch.motion[1, 1:] == 0)


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_phoenix([])


@pytest.mark.parametrize("modality", ["spatial", "motion"])
def test_collate_rejects_empty_feature_sequence(modality: str) -> None:
    sample = _sample("a", 0 if modality == "spatial" else 2, 0 if modality == "motion" else 2)

    with pytest.raises(ValueError, match=rf"{modality}.*a"):
        collate_phoenix([sample])
