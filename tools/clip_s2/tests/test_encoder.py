from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from tools.clip_s2.encoder import extract_frames, s2_cls


class PixelEncoder(nn.Module):
    def forward(self, pixel_values):
        # One scalar per input image; raw CLS differs deliberately from pooler.
        cls = pixel_values.mean((1, 2, 3))[:, None].repeat(1, 1024)
        return SimpleNamespace(last_hidden_state=cls[:, None], pooler_output=cls + 100)


def test_tiles_stay_associated_with_frames_and_raw_cls_is_used():
    pixels = torch.stack([torch.ones(3, 224, 224), torch.ones(3, 224, 224) * 7])
    result = s2_cls(pixels, PixelEncoder(), 3)
    assert result.shape == (2, 2048)
    torch.testing.assert_close(result[0], torch.ones(2048), rtol=0, atol=1e-6)
    torch.testing.assert_close(result[1], torch.ones(2048) * 7, rtol=0, atol=1e-6)


def test_invalid_shapes_and_nonfinite_cls_fail():
    with pytest.raises(ValueError):
        s2_cls(torch.zeros(2, 3, 448, 448), PixelEncoder(), 2)
    with pytest.raises(ValueError):
        s2_cls(torch.full((1, 3, 224, 224), float("nan")), PixelEncoder(), 2)


def test_extract_keeps_tail_frame_and_order(tmp_path):
    paths = []
    for i in (10, 20, 30):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (224, 224), (i, i, i)).save(p)
        paths.append(p)

    def processor(images, return_tensors):
        return SimpleNamespace(
            pixel_values=torch.stack(
                [torch.tensor(np.array(im)).permute(2, 0, 1).float() for im in images]
            )
        )

    values = extract_frames(paths, PixelEncoder(), processor, 2, 3)
    assert values.shape == (3, 2048) and values.dtype == np.float32
    np.testing.assert_allclose(values[:, 0], [10, 20, 30])
