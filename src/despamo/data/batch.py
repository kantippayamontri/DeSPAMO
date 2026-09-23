from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class PhoenixSample:
    clip_id: str
    signer: str
    text: str
    gloss: str
    en_text: str
    es_text: str
    fr_text: str
    spatial: torch.Tensor
    motion: torch.Tensor


@dataclass(frozen=True)
class PhoenixBatch:
    clip_ids: tuple[str, ...]
    signers: tuple[str, ...]
    texts: tuple[str, ...]
    glosses: tuple[str, ...]
    en_texts: tuple[str, ...]
    es_texts: tuple[str, ...]
    fr_texts: tuple[str, ...]
    spatial: torch.Tensor
    spatial_mask: torch.Tensor
    motion: torch.Tensor
    motion_mask: torch.Tensor


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]


def collate_phoenix(samples: Sequence[PhoenixSample]) -> PhoenixBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    for sample in samples:
        for modality in ("spatial", "motion"):
            if getattr(sample, modality).shape[0] == 0:
                raise ValueError(f"empty {modality} sequence for {sample.clip_id}")
    spatial_lengths = torch.tensor(
        [sample.spatial.shape[0] for sample in samples], device=samples[0].spatial.device
    )
    motion_lengths = torch.tensor(
        [sample.motion.shape[0] for sample in samples], device=samples[0].motion.device
    )
    spatial = pad_sequence([sample.spatial for sample in samples], batch_first=True)
    motion = pad_sequence([sample.motion for sample in samples], batch_first=True)
    return PhoenixBatch(
        clip_ids=tuple(sample.clip_id for sample in samples),
        signers=tuple(sample.signer for sample in samples),
        texts=tuple(sample.text for sample in samples),
        glosses=tuple(sample.gloss for sample in samples),
        en_texts=tuple(sample.en_text for sample in samples),
        es_texts=tuple(sample.es_text for sample in samples),
        fr_texts=tuple(sample.fr_text for sample in samples),
        spatial=spatial,
        spatial_mask=lengths_to_mask(spatial_lengths, spatial.shape[1]),
        motion=motion,
        motion_mask=lengths_to_mask(motion_lengths, motion.shape[1]),
    )
