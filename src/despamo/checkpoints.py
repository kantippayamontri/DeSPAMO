from collections.abc import Mapping

import torch
from torch import nn

PREFIXES = (
    ("spatio_proj.", "visual_adapter.spatial_projector."),
    ("spatiotemp_proj.", "visual_adapter.motion_projector."),
    ("fusion_proj.", "visual_adapter.multimodal_projector."),
    ("temporal_encoder.", "visual_adapter.temporal_encoder."),
    ("t5_model.", "language_model.model."),
)


def convert_spamo_state_dict(source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if key == "logit_scale":
            target = "vt_align.logit_scale"
        else:
            matches = [(old, new) for old, new in PREFIXES if key.startswith(old)]
            if len(matches) != 1:
                raise KeyError(f"unmapped SpaMo key: {key}")
            old, new = matches[0]
            target = new + key[len(old) :]
        if target in converted:
            raise KeyError(f"checkpoint mapping collision: {target}")
        converted[target] = value
    return converted


def schema_from_model(model: nn.Module) -> dict[str, dict[str, object]]:
    return {
        key: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for key, tensor in model.state_dict().items()
    }


def validate_converted_state_dict(
    converted: Mapping[str, torch.Tensor], schema: Mapping[str, Mapping[str, object]]
) -> None:
    if not isinstance(schema, Mapping) or not schema:
        raise ValueError("invalid target schema: expected nonempty state_dict mapping")
    for key, entry in schema.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(entry, Mapping)
            or set(entry) != {"shape", "dtype"}
            or not isinstance(entry["shape"], list)
            or any(type(dim) is not int or dim < 0 for dim in entry["shape"])
            or not isinstance(entry["dtype"], str)
            or not entry["dtype"].startswith("torch.")
            or not isinstance(
                getattr(torch, entry["dtype"].removeprefix("torch."), None), torch.dtype
            )
        ):
            raise ValueError(f"invalid target schema entry: {key!r}")

    missing = sorted(schema.keys() - converted.keys())
    extra = sorted(converted.keys() - schema.keys())
    shape = [
        f"{key}: expected {schema[key]['shape']}, got {list(converted[key].shape)}"
        for key in sorted(schema.keys() & converted.keys())
        if list(converted[key].shape) != schema[key]["shape"]
    ]
    dtype = [
        f"{key}: expected {schema[key]['dtype']}, got {converted[key].dtype}"
        for key in sorted(schema.keys() & converted.keys())
        if str(converted[key].dtype) != schema[key]["dtype"]
    ]
    errors = []
    for label, items in (
        ("missing keys", missing),
        ("extra keys", extra),
        ("shape mismatches", shape),
        ("dtype mismatches", dtype),
    ):
        if items:
            errors.append(f"{label} ({len(items)}): {', '.join(items[:3])}")
    if errors:
        raise ValueError("checkpoint schema mismatch: " + "; ".join(errors))
