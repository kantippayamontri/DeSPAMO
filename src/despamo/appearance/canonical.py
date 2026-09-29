"""Visible, bounded factor text targets; explicit lexical leakage filter."""

import re

from despamo.appearance.schema import ARTICULATORS, STABLE, Factor, Record

GLOBAL = re.compile(
    r"\b(identity|named|ethnicity|race|nationality|skin tone|skin color|diagnosis|disability|"
    r"sign gloss|translation|subtitle|caption|readable text|screen contents)\b", re.I
)
NUISANCE = re.compile(r"\b(handshape|gesture|mouth|pose|motion|gaze|signing)\b", re.I)
ARTICULATOR = re.compile(r"\b(jewelry|scar|skin|age|gender|smiling|angry|running)\b", re.I)


def clean(value: str) -> str:
    return " ".join(value.replace("_", " ").lower().split())


def canonical(name: str, value: Factor) -> str | None:
    if value.visibility not in {"clear", "partial"}:
        return None
    parts = []
    for field, item in value.model_dump(exclude={"visibility"}).items():
        if item in (None, [], "uncertain"):
            continue
        rendered = (
            ", ".join(sorted({clean(text) for text in item}))
            if isinstance(item, list)
            else clean(item)
        )
        parts.append(f"{clean(field)} {rendered}")
    return f"{clean(name)}: " + "; ".join(parts) if parts else None


def targets(record: Record) -> list[dict]:
    result = [
        dict(factor=name, frame_index=-1, text=canonical(name, getattr(record.stable, name)))
        for name in STABLE
    ]
    result.extend(
        dict(
            factor=name,
            frame_index=frame.source_frame_index,
            text=canonical(name, getattr(frame, name)),
        )
        for frame in record.frames
        for name in ARTICULATORS
    )
    return result


def validate_response(raw: str, clip: dict) -> Record:
    record = Record.model_validate_json(raw)
    if record.clip_id != clip["clip_id"]:
        raise ValueError("clip_id mismatch")
    if [frame.source_frame_index for frame in record.frames] != [
        frame["index"] for frame in clip["frames"]
    ]:
        raise ValueError("source-frame alignment mismatch")
    if [frame.normalized_position for frame in record.frames] != [
        frame["position"] for frame in clip["frames"]
    ]:
        raise ValueError("source-frame position mismatch")
    groups = [(f"stable.{name}", name, getattr(record.stable, name)) for name in STABLE]
    groups.extend(
        (f"frames.{i}.{name}", name, getattr(frame, name))
        for i, frame in enumerate(record.frames)
        for name in ARTICULATORS
    )
    for prefix, factor, obj in groups:
        for field, value in obj.model_dump(exclude={"visibility"}).items():
            text = " ".join(value) if isinstance(value, list) else str(value or "")
            local = NUISANCE if factor in STABLE else ARTICULATOR
            if GLOBAL.search(text) or local.search(text):
                raise ValueError(f"forbidden content at {prefix}.{field}")
    return record
