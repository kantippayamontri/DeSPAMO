from pathlib import Path

import numpy as np
import torch
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from requests.exceptions import RequestException
from transformers import AutoModel

from tools.dinov3.frames import preprocess
from tools.dinov3.identity import MODEL, SHA

FATAL_REASONS = frozenset(
    {
        "DINOv3 CUDA out of memory; stop extraction",
        "DINOv3 device transfer failed",
        "DINOv3 model forward failed",
        "CLS width/shape mismatch",
        "non-finite CLS",
        "invalid extracted feature shape/dtype",
    }
)


def load_model(sha: str, token: str, cache: Path, device: str) -> object:
    if not SHA.fullmatch(sha):
        raise ValueError("immutable model SHA required")
    try:
        model = (
            AutoModel.from_pretrained(
                MODEL,
                revision=sha,
                token=token,
                cache_dir=cache,
                trust_remote_code=False,
                use_safetensors=True,
            )
            .eval()
            .to(device)
        )
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else None
        if status in (401, 403):
            raise RuntimeError("DINOv3 model access denied; check license and HF_TOKEN") from None
        if status == 404:
            raise RuntimeError("DINOv3 Hub rejected model revision") from None
        if status is not None and 500 <= status < 600:
            raise RuntimeError("DINOv3 Hub unavailable; retry later") from None
        raise RuntimeError("DINOv3 Hub request rejected") from None
    except RequestException:
        raise RuntimeError("DINOv3 Hub request failed; check connectivity and retry") from None
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("DINOv3 CUDA out of memory; stop extraction") from None
    except Exception:
        raise RuntimeError(
            "DINOv3 model weights incompatible; check revision and runtime"
        ) from None
    if (
        model.config.hidden_size != 1024
        or model.config.patch_size != 16
        or model.config.num_register_tokens != 4
    ):
        raise ValueError("DINOv3 model configuration mismatch")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract(
    paths: list[Path], model: object, device: str, *, batch_size: int = 1
) -> tuple[np.ndarray, list[list[int]]]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not paths:
        raise ValueError("at least one source frame required")
    output, resolutions = [], []
    for start in range(0, len(paths), batch_size):
        group = paths[start : start + batch_size]
        for path in group:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    resolutions.append([image.width, image.height])
            except (OSError, ValueError, SyntaxError) as error:
                raise ValueError(f"invalid PNG {path}: {error}") from error
        scales = []
        for size in (224, 448):
            pixels = torch.stack([preprocess(path, size) for path in group])
            try:
                pixels = pixels.to(device)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError("DINOv3 CUDA out of memory; stop extraction") from None
            except Exception:
                raise RuntimeError("DINOv3 device transfer failed") from None
            try:
                with (
                    torch.inference_mode(),
                    torch.backends.cudnn.flags(
                        enabled=torch.backends.cudnn.enabled,
                        benchmark=torch.backends.cudnn.benchmark,
                        deterministic=torch.backends.cudnn.deterministic,
                        benchmark_limit=(
                            torch.backends.cudnn.benchmark_limit
                            if torch.backends.cudnn.benchmark_limit is not None
                            else 10
                        ),
                        allow_tf32=False,
                    ),
                ):
                    states = model(pixel_values=pixels).last_hidden_state
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError("DINOv3 CUDA out of memory; stop extraction") from None
            except Exception:
                raise RuntimeError("DINOv3 model forward failed") from None
            if (
                not isinstance(states, torch.Tensor)
                or states.ndim != 3
                or states.shape[0] != len(group)
                or states.shape[1] < 1
                or states.shape[2] != 1024
            ):
                raise RuntimeError("CLS width/shape mismatch")
            cls = states[:, 0, :].detach().float().cpu().numpy()
            if not np.isfinite(cls).all():
                raise RuntimeError("non-finite CLS")
            scales.append(cls)
        output.append(np.concatenate(scales, axis=1).astype(np.float32, copy=False))
    result = np.concatenate(output, axis=0)
    if result.shape != (len(paths), 2048) or result.dtype != np.float32:
        raise RuntimeError("invalid extracted feature shape/dtype")
    return result, resolutions
