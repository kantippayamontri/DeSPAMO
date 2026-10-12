"""SpaMo-equivalent final hidden CLS at224 and mean of448-scale tile CLS."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPImageProcessor, CLIPVisionModel

from tools.dinov3.identity import sha256_file

MODEL = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


def load_encoder(cache: Path, processor_file: Path, device: str):
    config = json.loads(processor_file.read_text())
    expected = dict(
        crop_size=224,
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        feature_extractor_type="CLIPFeatureExtractor",
        resample=3,
        size=224,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    if config != expected:
        raise ValueError("pinned CLIP processor configuration mismatch")
    model = cast(
        CLIPVisionModel,
        CLIPVisionModel.from_pretrained(
            MODEL,
            cache_dir=str(cache),
            revision=REVISION,
            local_files_only=True,
            torch_dtype=torch.float32,
        ),
    )
    if (
        model.config.hidden_size != 1024
        or model.config.image_size != 224
        or model.config.patch_size != 14
    ):
        raise ValueError("CLIP vision configuration mismatch")
    model.requires_grad_(False).eval()
    torch.nn.Module.to(model, device=torch.device(device))
    processor = cast(CLIPImageProcessor, CLIPImageProcessor.from_dict(config))
    snapshot = cache / "models--openai--clip-vit-large-patch14/snapshots" / REVISION
    binding = dict(
        model=MODEL,
        revision=REVISION,
        processor_sha256=sha256_file(processor_file),
        model_files={p.name: sha256_file(p) for p in snapshot.iterdir() if p.is_file()},
        processor=processor.to_dict(),
        tokens="last_hidden_state_cls_no_postnorm",
        scales=[224, 448],
        tile_order="row_major_tile_then_frame",
        dtype="float32",
    )
    return model, processor, binding


@torch.inference_mode()
def s2_cls(pixels: torch.Tensor, model, tile_batch_size: int) -> torch.Tensor:
    if (
        pixels.ndim != 4
        or pixels.shape[1:] != (3, 224, 224)
        or not len(pixels)
        or pixels.dtype != torch.float32
        or type(tile_batch_size) is not int
        or tile_batch_size < 1
    ):
        raise ValueError("nonempty FP32 Bx3x224x224 pixels and positive tile batch required")
    if not torch.isfinite(pixels).all():
        raise ValueError("non-finite input pixels")

    def encode(images):
        cls = model(pixel_values=images).last_hidden_state[:, 0]
        if cls.shape != (len(images), 1024) or not torch.isfinite(cls).all():
            raise ValueError("invalid raw CLIP CLS output")
        return cls.float()

    # Scale1 interpolation is also present in original S2 wrapper (identity size).
    global_cls = encode(F.interpolate(pixels, size=224, mode="bicubic", align_corners=False))
    large = F.interpolate(pixels, size=448, mode="bicubic", align_corners=False, antialias=False)
    tiles = torch.cat(
        [
            large[:, :, i * 224 : (i + 1) * 224, j * 224 : (j + 1) * 224]
            for i in range(2)
            for j in range(2)
        ],
        dim=0,
    )
    local_cls = torch.cat(
        [encode(tiles[k : k + tile_batch_size]) for k in range(0, len(tiles), tile_batch_size)]
    )
    local_cls = local_cls.reshape(4, len(pixels), 1024).mean(0)
    return torch.cat([global_cls, local_cls], dim=-1)


def extract_frames(
    paths: list[Path], model, processor, frame_batch_size: int, tile_batch_size: int
) -> np.ndarray:
    if not paths or frame_batch_size < 1:
        raise ValueError("nonempty source frames and positive frame batch required")
    device = next(model.parameters(), torch.empty(0)).device
    values = []
    for offset in range(0, len(paths), frame_batch_size):
        images = []
        for path in paths[offset : offset + frame_batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images, return_tensors="pt").pixel_values.to(device, dtype=torch.float32)
        values.append(s2_cls(pixels, model, tile_batch_size).cpu().numpy())
    result = np.concatenate(values).astype(np.float32, copy=False)
    if result.shape != (len(paths), 2048) or not np.isfinite(result).all():
        raise ValueError("invalid extracted spatial feature shape/dtype")
    return result
