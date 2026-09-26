import os
from pathlib import Path

import numpy as np
import pytest

from tools.dinov3.encoder import extract, load_model
from tools.dinov3.identity import read_token, resolve_sha


@pytest.mark.integration
@pytest.mark.gpu
def test_one_gated_frame_cls_is_real_and_finite():
    if os.environ.get("DESPAMO_RUN_DINO_LIVE") != "1":
        pytest.skip("opt in after gate and pinned CUDA lock")
    token = read_token(Path(os.environ["DESPAMO_DINO_ENV_FILE"]))
    sha = os.environ["DESPAMO_DINO_SHA"]
    assert resolve_sha(token, sha) == sha
    model = load_model(sha, token, Path(os.environ["DESPAMO_DINO_CACHE"]), "cuda")
    image = Path(os.environ["PHOENIX14T_FRAME_ROOT"]) / os.environ["DESPAMO_DINO_FRAME"]
    feature, resolution = extract([image], model, "cuda")
    assert feature.shape == (1, 2048) and feature.dtype == np.float32
    assert np.isfinite(feature).all() and resolution == [[210, 260]]
