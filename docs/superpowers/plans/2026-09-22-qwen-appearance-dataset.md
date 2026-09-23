# Qwen Appearance Dataset Implementation Plan

> **SUPERSEDED - DO NOT EXECUTE.** Historical three-factor draft, including unfinished self-review edits. Approved schema-2 replacements: [Seven-factor dataset](2026-09-23-seven-factor-dataset.md) and [Dual-path training](2026-09-23-dual-path-training.md). Their contracts override every code block below.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an audited, resumable PHOENIX14T dataset of separate clothing, hair, and background descriptions from local Ollama `qwen3-vl:8b`, plus frozen CLIP text features for later GRL training.

**Architecture:** Sample five frames per clip, request schema-constrained JSON from local Ollama, validate and canonicalize each factor, then cache factor-specific CLIP text embeddings. Preserve raw responses and immutable provenance so prompt or model changes always create a new dataset version.

**Tech Stack:** Existing DeSpaMo Python 3.11 environment, Ollama Python client, local `qwen3-vl:8b`, Pydantic 2, Transformers 4.32.0, frozen `openai/clip-vit-large-patch14` text projection, NumPy, pytest.

## Global Constraints

- Do not execute this plan until PHOENIX14T baseline acceptance and DINOv3 comparison milestones pass.
- Qwen describes clothing, hair, and background only.
- Qwen must not describe pose, motion, handshape, gesture, arm position, facial expression, mouth movement, gaze, activity, readable text, age, gender, ethnicity, body build, or height.
- Use frames nearest 10%, 30%, 50%, 70%, and 90% of each clip.
- Use one multimodal Qwen request per clip, not one request per frame.
- Use `visibility` values `clear`, `partial`, `not_visible`, or `uncertain`.
- Unknown scalar values are JSON `null`; unknown lists are empty lists.
- Retain raw responses, validation errors, retry counts, model digest, prompt version/hash/text, endpoint identifier, generation options, sampled frame indices, and paths.
- Retry schema-invalid output at most two times. Connection failures stop the run.
- Cache key is `(clip_id, model_digest, prompt_hash)`.
- Canonical strings, not raw Qwen prose, are encoded.
- Frozen text encoder is `openai/clip-vit-large-patch14`; resolve and record immutable Hugging Face revision.
- Generated descriptions and embeddings stay outside Git.
- Do not commit unless user explicitly requests commits. Commit snippets below are checkpoints, not authorization.

## File Map

- `configs/appearance/qwen3_vl.yaml`: local Ollama, sampling, schema, retry, and output settings.
- `src/despamo/appearance/schema.py`: strict Pydantic factor records.
- `src/despamo/appearance/frames.py`: annotation traversal and deterministic frame sampling.
- `src/despamo/appearance/canonical.py`: deterministic factor strings and forbidden-content checks.
- `src/despamo/appearance/prompt.py`: versioned Qwen extraction and correction prompts.
- `src/despamo/appearance/ollama_client.py`: local model lookup, digest capture, and multimodal request.
- `src/despamo/appearance/store.py`: atomic resumable raw and validated record storage.
- `src/despamo/appearance/pipeline.py`: validation retries and per-clip orchestration.
- `src/despamo/appearance/text_features.py`: frozen CLIP factor embeddings and manifests.
- `src/despamo/appearance/audit.py`: stratified audit sheet and threshold summary.
- `scripts/generate_appearance_descriptions.py`: resumable generation CLI.
- `scripts/encode_appearance_text.py`: factor embedding CLI.
- `scripts/build_appearance_audit.py`: 100-clip audit sheet CLI.
- `scripts/summarize_appearance_audit.py`: quality-gate CLI.
- `tests/unit/appearance`: fast schema, sampling, canonicalization, retry, storage, embedding, and audit tests.
- `tests/integration/appearance`: opt-in local Ollama and real PHOENIX tests.

---

### Task 1: Appearance Dependencies And Configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Create: `configs/appearance/qwen3_vl.yaml`
- Create: `src/despamo/appearance/__init__.py`
- Create: `tests/unit/appearance/test_config.py`

**Interfaces:**
- Consumes: `load_config` from baseline plan.
- Produces: resolved `config.appearance` values with explicit model, host, paths, positions, and retry count.

- [ ] **Step 1: Write failing appearance-config test**

```python
# tests/unit/appearance/test_config.py
from pathlib import Path

from despamo.config import load_config


def test_qwen_config_fixes_model_and_sampling(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DESPAMO_APPEARANCE_ROOT", str(tmp_path / "appearance"))
    monkeypatch.setenv("PHOENIX14T_FRAME_ROOT", str(tmp_path / "frames"))

    config = load_config([Path("configs/appearance/qwen3_vl.yaml")])

    assert config.appearance.model == "qwen3-vl:8b"
    assert config.appearance.frame_positions == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert config.appearance.max_validation_retries == 2
    assert config.appearance.output_root == str(tmp_path / "appearance")
```

- [ ] **Step 2: Verify test fails**

Run: `uv run pytest tests/unit/appearance/test_config.py -v`

Expected: FAIL because appearance config does not exist.

- [ ] **Step 3: Add direct dependencies**

Add these entries to `project.dependencies` in `pyproject.toml`, preserving existing dependencies:

```toml
  "ollama>=0.6,<1",
  "pydantic>=2.10,<3",
```

- [ ] **Step 4: Add appearance config**

```yaml
# configs/appearance/qwen3_vl.yaml
appearance:
  model: qwen3-vl:8b
  ollama_host: http://127.0.0.1:11434
  annotation_root: ${oc.env:PHOENIX14T_ANNOTATION_ROOT}
  frame_root: ${oc.env:PHOENIX14T_FRAME_ROOT}
  output_root: ${oc.env:DESPAMO_APPEARANCE_ROOT}
  frame_positions: [0.1, 0.3, 0.5, 0.7, 0.9]
  max_validation_retries: 2
  temperature: 0.0
  keep_alive: 10m
  schema_version: 1
  prompt_version: qwen3-vl-appearance-v1
  text_encoder: openai/clip-vit-large-patch14
  text_encoder_revision: main
```

Append to `.env.example`:

```dotenv
PHOENIX14T_FRAME_ROOT=
DESPAMO_APPEARANCE_ROOT=
DESPAMO_APPEARANCE_FIXTURES=
DESPAMO_APPEARANCE_DATASET_KEY=
```

- [ ] **Step 5: Lock dependencies and run test**

Run: `uv lock && uv sync --group dev && uv run pytest tests/unit/appearance/test_config.py -v`

Expected: dependency lock succeeds and test reports `1 passed`.

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add pyproject.toml uv.lock .env.example configs/appearance/qwen3_vl.yaml src/despamo/appearance/__init__.py tests/unit/appearance/test_config.py
git commit -m "build: configure appearance extraction"
```

---

### Task 2: Deterministic PHOENIX Frame Sampling

**Files:**
- Create: `src/despamo/appearance/frames.py`
- Create: `tests/unit/appearance/test_frames.py`

**Interfaces:**
- Produces: `ClipFrames` with clip ID, signer, split, total frame count, sampled indices/paths, and any frame error.
- Produces: `sample_frame_indices(frame_count, positions) -> tuple[int, ...]`.
- Produces: `sample_frame_paths(paths, positions) -> tuple[Path, ...]`.
- Produces: `iter_phoenix_clips(annotation_path, frame_root, split, positions) -> Iterator[ClipFrames]`.

- [ ] **Step 1: Write failing sampling tests**

```python
# tests/unit/appearance/test_frames.py
from pathlib import Path

import numpy as np
import pytest

from despamo.appearance.frames import (
    iter_phoenix_clips,
    sample_frame_indices,
    sample_frame_paths,
)


def test_sample_frame_paths_uses_normalized_positions() -> None:
    paths = [Path(f"frame-{index:02}.png") for index in range(11)]
    sampled = sample_frame_paths(paths, (0.1, 0.3, 0.5, 0.7, 0.9))
    assert [path.name for path in sampled] == [
        "frame-01.png",
        "frame-03.png",
        "frame-05.png",
        "frame-07.png",
        "frame-09.png",
    ]
    assert sample_frame_indices(11, (0.1, 0.3, 0.5, 0.7, 0.9)) == (1, 3, 5, 7, 9)


def test_sample_frame_paths_rejects_too_few_frames() -> None:
    with pytest.raises(ValueError, match="at least five frames"):
        sample_frame_paths([Path("one.png")], (0.1, 0.3, 0.5, 0.7, 0.9))


def test_iter_phoenix_clips_reads_integer_records(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames" / "train" / "clip-1"
    frame_dir.mkdir(parents=True)
    for index in range(10):
        (frame_dir / f"{index:03}.png").write_bytes(b"fixture")
    annotation = tmp_path / "train_info_ml.npy"
    np.save(
        annotation,
        {
            "prefix": "ignored",
            0: {
                "fileid": "clip-1",
                "folder": "train/clip-1/*.png",
                "signer": "Signer01",
            },
        },
    )

    clips = list(
        iter_phoenix_clips(
            annotation,
            tmp_path / "frames",
            "train",
            (0.1, 0.3, 0.5, 0.7, 0.9),
        )
    )

    assert len(clips) == 1
    assert clips[0].clip_id == "clip-1"
    assert clips[0].signer == "Signer01"
    assert clips[0].sampled_indices == (1, 3, 5, 6, 8)
    assert len(clips[0].sampled_paths) == 5


def test_iter_phoenix_clips_yields_missing_frame_failure(tmp_path: Path) -> None:
    annotation = tmp_path / "train_info_ml.npy"
    np.save(
        annotation,
        {
            0: {
                "fileid": "missing-clip",
                "folder": "train/missing-clip/*.png",
                "signer": "Signer01",
            },
        },
    )
    clips = list(
        iter_phoenix_clips(
            annotation,
            tmp_path / "frames",
            "train",
            (0.1, 0.3, 0.5, 0.7, 0.9),
        )
    )
    assert clips[0].sampled_paths == ()
    assert clips[0].frame_error == "no frames for missing-clip: train/missing-clip/*.png"
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/unit/appearance/test_frames.py -v`

Expected: FAIL because frame-sampling module does not exist.

- [ ] **Step 3: Implement deterministic sampling**

```python
# src/despamo/appearance/frames.py
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class ClipFrames:
    clip_id: str
    signer: str
    split: str
    frame_count: int
    sampled_indices: tuple[int, ...]
    sampled_paths: tuple[Path, ...]
    frame_error: str | None = None


def sample_frame_indices(
    frame_count: int, positions: Sequence[float]
) -> tuple[int, ...]:
    if frame_count < len(positions):
        raise ValueError(f"clip needs at least five frames, found {frame_count}")
    if any(position < 0 or position > 1 for position in positions):
        raise ValueError(f"frame positions must be in [0, 1]: {positions}")
    indices = tuple(int((frame_count - 1) * position + 0.5) for position in positions)
    if len(set(indices)) != len(indices):
        raise ValueError(f"frame positions produced duplicate indices: {indices}")
    return indices


def sample_frame_paths(
    paths: Sequence[Path], positions: Sequence[float]
) -> tuple[Path, ...]:
    ordered = tuple(sorted(paths))
    indices = sample_frame_indices(len(ordered), positions)
    return tuple(ordered[index] for index in indices)


def iter_phoenix_clips(
    annotation_path: Path,
    frame_root: Path,
    split: str,
    positions: Sequence[float],
) -> Iterator[ClipFrames]:
    raw = np.load(annotation_path, allow_pickle=True).item()
    for key in sorted(key for key in raw if isinstance(key, int)):
        item = raw[key]
        frame_paths = tuple(
            sorted(Path(path) for path in glob.glob(str(frame_root / item["folder"])))
        )
        if not frame_paths:
            yield ClipFrames(
                clip_id=item["fileid"],
                signer=item["signer"],
                split=split,
                frame_count=0,
                sampled_indices=(),
                sampled_paths=(),
                frame_error=f"no frames for {item['fileid']}: {item['folder']}",
            )
            continue
        try:
            indices = sample_frame_indices(len(frame_paths), positions)
        except ValueError as error:
            yield ClipFrames(
                clip_id=item["fileid"],
                signer=item["signer"],
                split=split,
                frame_count=len(frame_paths),
                sampled_indices=(),
                sampled_paths=(),
                frame_error=str(error),
            )
            continue
        yield ClipFrames(
            clip_id=item["fileid"],
            signer=item["signer"],
            split=split,
            frame_count=len(frame_paths),
            sampled_indices=indices,
            sampled_paths=tuple(frame_paths[index] for index in indices),
        )
```

- [ ] **Step 4: Run frame tests**

Run: `uv run pytest tests/unit/appearance/test_frames.py -v`

Expected: `4 passed`.

- [ ] **Step 5: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/appearance/frames.py tests/unit/appearance/test_frames.py
git commit -m "feat: sample PHOENIX appearance frames"
```

---

### Task 3: Strict Appearance Schema And Canonical Text

**Files:**
- Create: `src/despamo/appearance/schema.py`
- Create: `src/despamo/appearance/canonical.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/appearance/__init__.py`
- Create: `tests/unit/appearance/helpers.py`
- Create: `tests/unit/appearance/test_schema.py`
- Create: `tests/unit/appearance/test_canonical.py`

**Interfaces:**
- Produces: `AppearanceRecord.model_validate_json(raw) -> AppearanceRecord`.
- Produces: `validate_content(record) -> None`.
- Produces: `canonical_descriptions(record) -> dict[Factor, str | None]`.

- [ ] **Step 1: Write failing schema tests**

Create empty `tests/__init__.py`, `tests/unit/__init__.py`, and `tests/unit/appearance/__init__.py` package markers so shared test helpers resolve consistently.

```python
# tests/unit/appearance/helpers.py
def valid_payload() -> dict:
    return {
        "schema_version": 1,
        "clip_id": "clip-1",
        "clothing": {
            "upper_garment": "long-sleeve shirt",
            "upper_color": "black",
            "lower_garment": None,
            "lower_color": None,
            "pattern": "solid",
            "accessories": [],
            "visibility": "clear",
        },
        "hair": {
            "color": "dark brown",
            "length": "short",
            "style": "straight",
            "visibility": "partial",
        },
        "background": {
            "scene": "indoor studio",
            "dominant_colors": ["blue", "gray"],
            "static_objects": ["plain backdrop"],
            "lighting": "uniform",
            "visibility": "clear",
        },
    }
```

```python
# tests/unit/appearance/test_schema.py
import json

import pytest
from pydantic import ValidationError

from despamo.appearance.schema import AppearanceRecord
from tests.unit.appearance.helpers import valid_payload


def test_schema_accepts_expected_factors() -> None:
    record = AppearanceRecord.model_validate_json(json.dumps(valid_payload()))
    assert record.clothing.upper_color == "black"
    assert record.hair.visibility == "partial"


def test_schema_rejects_biometric_field() -> None:
    payload = valid_payload()
    payload["gender"] = "female"
    with pytest.raises(ValidationError):
        AppearanceRecord.model_validate(payload)


def test_schema_rejects_unknown_visibility() -> None:
    payload = valid_payload()
    payload["hair"]["visibility"] = "mostly"
    with pytest.raises(ValidationError):
        AppearanceRecord.model_validate(payload)


def test_schema_rejects_type_coercion() -> None:
    payload = valid_payload()
    payload["clothing"]["upper_color"] = 7
    with pytest.raises(ValidationError):
        AppearanceRecord.model_validate(payload)


def test_schema_rejects_blank_description() -> None:
    payload = valid_payload()
    payload["hair"]["style"] = "   "
    with pytest.raises(ValidationError):
        AppearanceRecord.model_validate(payload)
```

- [ ] **Step 2: Implement strict Pydantic models**

```python
# src/despamo/appearance/schema.py
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, StringConstraints


Visibility: TypeAlias = Literal["clear", "partial", "not_visible", "uncertain"]
Factor: TypeAlias = Literal["clothing", "hair", "background"]
SCHEMA_VERSION = 1
DescriptionText: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClothingDescription(StrictModel):
    upper_garment: DescriptionText | None
    upper_color: DescriptionText | None
    lower_garment: DescriptionText | None
    lower_color: DescriptionText | None
    pattern: DescriptionText | None
    accessories: list[DescriptionText]
    visibility: Visibility


class HairDescription(StrictModel):
    color: DescriptionText | None
    length: DescriptionText | None
    style: DescriptionText | None
    visibility: Visibility


class BackgroundDescription(StrictModel):
    scene: DescriptionText | None
    dominant_colors: list[DescriptionText]
    static_objects: list[DescriptionText]
    lighting: DescriptionText | None
    visibility: Visibility


class AppearanceRecord(StrictModel):
    schema_version: Literal[1]
    clip_id: DescriptionText
    clothing: ClothingDescription
    hair: HairDescription
    background: BackgroundDescription
```

- [ ] **Step 3: Write failing canonicalization tests**

```python
# tests/unit/appearance/test_canonical.py
import pytest

from despamo.appearance.canonical import canonical_descriptions, validate_content
from despamo.appearance.schema import AppearanceRecord
from tests.unit.appearance.helpers import valid_payload


def test_canonical_descriptions_have_fixed_field_order() -> None:
    record = AppearanceRecord.model_validate(valid_payload())
    descriptions = canonical_descriptions(record)
    assert descriptions["clothing"] == (
        "clothing: upper garment long-sleeve shirt; upper color black; "
        "pattern solid; accessories none"
    )
    assert descriptions["hair"] == "hair: color dark brown; length short; style straight"


def test_uncertain_factor_has_no_canonical_text() -> None:
    payload = valid_payload()
    payload["hair"]["visibility"] = "uncertain"
    record = AppearanceRecord.model_validate(payload)
    assert canonical_descriptions(record)["hair"] is None


def test_forbidden_linguistic_content_is_rejected() -> None:
    payload = valid_payload()
    payload["background"]["static_objects"] = ["subtitle saying rain tomorrow"]
    record = AppearanceRecord.model_validate(payload)
    with pytest.raises(ValueError, match="forbidden appearance content"):
        validate_content(record)


def test_biometric_value_is_rejected() -> None:
    payload = valid_payload()
    payload["hair"]["style"] = "female hairstyle"
    record = AppearanceRecord.model_validate(payload)
    with pytest.raises(ValueError, match="forbidden appearance content"):
        validate_content(record)


def test_visible_factor_requires_an_attribute() -> None:
    payload = valid_payload()
    payload["hair"].update({"color": None, "length": None, "style": None})
    record = AppearanceRecord.model_validate(payload)
    with pytest.raises(ValueError, match="visible hair has no attributes"):
        validate_content(record)


def test_not_visible_factor_rejects_attributes() -> None:
    payload = valid_payload()
    payload["hair"]["visibility"] = "not_visible"
    record = AppearanceRecord.model_validate(payload)
    with pytest.raises(ValueError, match="not-visible hair contains attributes"):
        validate_content(record)
```

- [ ] **Step 4: Implement content validation and canonicalization**

```python
# src/despamo/appearance/canonical.py
import re

from despamo.appearance.schema import AppearanceRecord, Factor


FORBIDDEN_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(hands?|fingers?|handshape|gestures?|signing|arms?|pose|posture)\b",
        r"\b(facial expressions?|mouth movements?|lip movements?|gaze|head movements?)\b",
        r"\b(motion|body movement|activities|activity|actions?|subtitles?|captions?)\b",
        r"\b(readable text|screen contents?|ages?|gender|sex|ethnicity|race)\b",
        r"\b(male|female|men|women|man|woman|boys?|girls?|body build|height)\b",
    )
)


def _clean(value: str) -> str:
    return " ".join(value.strip().lower().split())


def validate_content(record: AppearanceRecord) -> None:
    values: list[str] = []
    for name, factor in (
        ("clothing", record.clothing),
        ("hair", record.hair),
        ("background", record.background),
    ):
        attributes = factor.model_dump(exclude={"visibility"}).values()
        has_attributes = any(value not in (None, []) for value in attributes)
        if factor.visibility in {"clear", "partial"} and not has_attributes:
            raise ValueError(f"visible {name} has no attributes")
        if factor.visibility == "not_visible" and has_attributes:
            raise ValueError(f"not-visible {name} contains attributes")
        for value in factor.model_dump(exclude={"visibility"}).values():
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, list):
                values.extend(str(item) for item in value)
    content = " | ".join(values)
    if any(pattern.search(content) for pattern in FORBIDDEN_PATTERNS):
        raise ValueError(f"forbidden appearance content: {content}")


def _parts(label: str, fields: list[tuple[str, str | list[str] | None]]) -> str:
    parts = []
    for name, value in fields:
        if isinstance(value, list):
            rendered = ", ".join(_clean(item) for item in value) if value else "none"
        elif value is None:
            continue
        else:
            rendered = _clean(value)
        parts.append(f"{name} {rendered}")
    return f"{label}: " + "; ".join(parts)


def canonical_descriptions(record: AppearanceRecord) -> dict[Factor, str | None]:
    validate_content(record)
    output: dict[Factor, str | None] = {
        "clothing": None,
        "hair": None,
        "background": None,
    }
    if record.clothing.visibility in {"clear", "partial"}:
        output["clothing"] = _parts(
            "clothing",
            [
                ("upper garment", record.clothing.upper_garment),
                ("upper color", record.clothing.upper_color),
                ("lower garment", record.clothing.lower_garment),
                ("lower color", record.clothing.lower_color),
                ("pattern", record.clothing.pattern),
                ("accessories", record.clothing.accessories),
            ],
        )
    if record.hair.visibility in {"clear", "partial"}:
        output["hair"] = _parts(
            "hair",
            [
                ("color", record.hair.color),
                ("length", record.hair.length),
                ("style", record.hair.style),
            ],
        )
    if record.background.visibility in {"clear", "partial"}:
        output["background"] = _parts(
            "background",
            [
                ("scene", record.background.scene),
                ("dominant colors", record.background.dominant_colors),
                ("static objects", record.background.static_objects),
                ("lighting", record.background.lighting),
            ],
        )
    return output
```

- [ ] **Step 5: Run schema and canonicalization tests**

Run: `uv run pytest tests/unit/appearance/test_schema.py tests/unit/appearance/test_canonical.py -v`

Expected: `11 passed`.

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/appearance/schema.py src/despamo/appearance/canonical.py tests/unit/appearance
git commit -m "feat: define appearance description schema"
```

---

### Task 4: Versioned Prompt And Ollama Boundary

**Files:**
- Create: `src/despamo/appearance/prompt.py`
- Create: `src/despamo/appearance/ollama_client.py`
- Create: `tests/unit/appearance/test_prompt.py`
- Create: `tests/unit/appearance/test_ollama_client.py`

**Interfaces:**
- Produces: `prompt_hash() -> str` and `build_prompt(clip_id, correction_errors=()) -> str`.
- Produces: `OllamaModelRef(tag: str, digest: str)`.
- Produces: `OllamaAppearanceClient.generate(frame_paths, clip_id, correction_errors=()) -> str`.

- [ ] **Step 1: Write failing prompt tests**

```python
# tests/unit/appearance/test_prompt.py
from despamo.appearance.prompt import PROMPT_VERSION, build_prompt, prompt_hash


def test_prompt_is_versioned_and_forbids_linguistic_cues() -> None:
    prompt = build_prompt("clip-1")
    assert PROMPT_VERSION == "qwen3-vl-appearance-v1"
    assert "Do not describe handshape" in prompt
    assert 'Set clip_id exactly to "clip-1"' in prompt
    assert len(prompt_hash()) == 64


def test_correction_prompt_contains_validation_errors() -> None:
    prompt = build_prompt("clip-1", ("hair.visibility has invalid value",))
    assert "hair.visibility has invalid value" in prompt
```

- [ ] **Step 2: Implement immutable prompt text and hash**

```python
# src/despamo/appearance/prompt.py
import hashlib
import json

from despamo.appearance.schema import AppearanceRecord


PROMPT_VERSION = "qwen3-vl-appearance-v1"
BASE_PROMPT = """Analyze all five frames as one sign-language video clip.
Return only JSON matching the supplied schema.
Describe three stable visual factors: clothing, hair, and background.
Use null for unknown scalar fields and [] for unknown list fields.
Use visibility only from: clear, partial, not_visible, uncertain.
Do not guess attributes that are not visible.
Do not describe handshape, gesture, signing, arm position, pose, posture,
body movement, facial expression, mouth movement, gaze, head movement,
activity, readable text, subtitles, age, gender, ethnicity, body build, or height.
For background, describe only scene type, dominant colors, static objects,
and lighting. Ignore text and screen contents.
"""


def build_prompt(clip_id: str, correction_errors: tuple[str, ...] = ()) -> str:
    prompt = BASE_PROMPT + f"\nSet clip_id exactly to {json.dumps(clip_id)}."
    if not correction_errors:
        return prompt
    rendered = "\n".join(f"- {error}" for error in correction_errors)
    return prompt + "\nCorrect these validation errors from the previous response:\n" + rendered


def prompt_hash() -> str:
    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "prompt": BASE_PROMPT,
            "schema": AppearanceRecord.model_json_schema(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
```

- [ ] **Step 3: Write failing Ollama boundary tests with fake client**

```python
# tests/unit/appearance/test_ollama_client.py
from pathlib import Path
from types import SimpleNamespace

import pytest

from despamo.appearance.ollama_client import OllamaAppearanceClient


class FakeClient:
    def list(self):
        model = SimpleNamespace(model="qwen3-vl:8b", digest="sha256-model")
        return SimpleNamespace(models=[model])

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(message=SimpleNamespace(content='{"schema_version": 1}'))


def test_client_captures_digest_and_sends_five_images() -> None:
    fake = FakeClient()
    client = OllamaAppearanceClient(
        fake,
        endpoint="http://127.0.0.1:11434",
        model="qwen3-vl:8b",
        temperature=0.0,
        keep_alive="10m",
    )
    paths = tuple(Path(f"frame-{index}.png") for index in range(5))

    raw = client.generate(paths, "clip-1")

    assert raw == '{"schema_version": 1}'
    assert client.model_ref.digest == "sha256-model"
    assert client.endpoint == "http://127.0.0.1:11434"
    assert fake.kwargs["messages"][0]["images"] == [str(path) for path in paths]
    assert fake.kwargs["options"] == {"temperature": 0.0}


def test_client_rejects_missing_model() -> None:
    fake = FakeClient()
    fake.list = lambda: SimpleNamespace(models=[])
    with pytest.raises(RuntimeError, match="qwen3-vl:8b is not installed"):
        OllamaAppearanceClient(
            fake, "http://127.0.0.1:11434", "qwen3-vl:8b", 0.0, "10m"
        )


def test_client_rejects_endpoint_credentials() -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        OllamaAppearanceClient(
            FakeClient(),
            "http://user:secret@127.0.0.1:11434",
            "qwen3-vl:8b",
            0.0,
            "10m",
        )
```

- [ ] **Step 4: Implement Ollama client boundary**

```python
# src/despamo/appearance/ollama_client.py
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit

from ollama import Client

from despamo.appearance.prompt import build_prompt
from despamo.appearance.schema import AppearanceRecord


@dataclass(frozen=True)
class OllamaModelRef:
    tag: str
    digest: str


class OllamaAppearanceClient:
    def __init__(
        self,
        client: Client,
        endpoint: str,
        model: str,
        temperature: float,
        keep_alive: str,
    ) -> None:
        parsed_endpoint = urlsplit(endpoint)
        if parsed_endpoint.username is not None or parsed_endpoint.password is not None:
            raise ValueError("Ollama endpoint must not contain credentials")
        models = {item.model: item for item in client.list().models}
        if model not in models:
            raise RuntimeError(f"{model} is not installed in local Ollama")
        digest = getattr(models[model], "digest", None)
        if not digest:
            raise RuntimeError(f"Ollama did not return a digest for {model}")
        self.client = client
        self.endpoint = endpoint
        self.model_ref = OllamaModelRef(model, digest)
        self.temperature = temperature
        self.keep_alive = keep_alive

    def generate(
        self,
        frame_paths: Sequence[Path],
        clip_id: str,
        correction_errors: tuple[str, ...] = (),
    ) -> str:
        if len(frame_paths) != 5:
            raise ValueError(f"expected five frame paths, got {len(frame_paths)}")
        response = self.client.chat(
            model=self.model_ref.tag,
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(clip_id, correction_errors),
                    "images": [str(path) for path in frame_paths],
                }
            ],
            format=AppearanceRecord.model_json_schema(),
            options={"temperature": self.temperature},
            keep_alive=self.keep_alive,
        )
        return response.message.content
```

- [ ] **Step 5: Run prompt and client tests**

Run: `uv run pytest tests/unit/appearance/test_prompt.py tests/unit/appearance/test_ollama_client.py -v`

Expected: `5 passed` without contacting Ollama.

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/appearance/prompt.py src/despamo/appearance/ollama_client.py tests/unit/appearance
git commit -m "feat: add structured Ollama appearance client"
```

---

### Task 5: Atomic Resumable Appearance Store

**Files:**
- Create: `src/despamo/appearance/store.py`
- Create: `tests/unit/appearance/test_store.py`

**Interfaces:**
- Produces: `GenerationEnvelope` containing provenance, frame count, every raw response, validated record, errors, retries, and status.
- Produces: `AppearanceStore.dataset_key(model_digest, prompt_hash) -> str` for version isolation.
- Produces: `AppearanceStore.cache_key(clip_id, model_digest, prompt_hash) -> str`.
- Produces: atomic `save`, `load`, `contains`, and version-filtered `iter_dataset` operations.

- [ ] **Step 1: Write failing store tests**

```python
# tests/unit/appearance/test_store.py
from pathlib import Path

from despamo.appearance.store import AppearanceStore, GenerationEnvelope


def test_store_round_trip_and_resume(tmp_path: Path) -> None:
    store = AppearanceStore(tmp_path)
    dataset_key = store.dataset_key("digest", "prompt")
    envelope = GenerationEnvelope(
        dataset_key=dataset_key,
        cache_key=store.cache_key("clip-1", "digest", "prompt"),
        clip_id="clip-1",
        split="train",
        signer="Signer01",
        endpoint="http://127.0.0.1:11434",
        model_tag="qwen3-vl:8b",
        model_digest="digest",
        temperature=0.0,
        keep_alive="10m",
        prompt_version="qwen3-vl-appearance-v1",
        prompt_hash="prompt",
        prompt_text="prompt text",
        frame_count=20,
        sampled_frame_indices=(2, 6, 10, 13, 17),
        frame_paths=("a.png", "b.png", "c.png", "d.png", "e.png"),
        raw_responses=("not json", "{}"),
        validated_record=None,
        validation_errors=("missing clothing",),
        retries=2,
        status="failed",
    )
    store.save(envelope)
    assert store.load("train", envelope.cache_key) == envelope
    assert store.contains("train", envelope.cache_key)
    assert envelope.dataset_key == store.dataset_key("digest", "prompt")
    assert list(store.iter_dataset("train", dataset_key)) == [envelope]


def test_cache_key_changes_with_prompt() -> None:
    first = AppearanceStore.cache_key("clip-1", "digest", "prompt-a")
    second = AppearanceStore.cache_key("clip-1", "digest", "prompt-b")
    assert first != second
```

- [ ] **Step 2: Implement envelope and atomic store**

```python
# src/despamo/appearance/store.py
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


Status = Literal["valid", "failed"]


@dataclass(frozen=True)
class GenerationEnvelope:
    dataset_key: str
    cache_key: str
    clip_id: str
    split: str
    signer: str
    endpoint: str
    model_tag: str
    model_digest: str
    temperature: float
    keep_alive: str
    prompt_version: str
    prompt_hash: str
    prompt_text: str
    frame_count: int
    sampled_frame_indices: tuple[int, ...]
    frame_paths: tuple[str, ...]
    raw_responses: tuple[str, ...]
    validated_record: dict | None
    validation_errors: tuple[str, ...]
    retries: int
    status: Status


class AppearanceStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def cache_key(clip_id: str, model_digest: str, prompt_hash: str) -> str:
        payload = f"{clip_id}\n{model_digest}\n{prompt_hash}".encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def dataset_key(model_digest: str, prompt_hash: str) -> str:
        payload = f"{model_digest}\n{prompt_hash}".encode()
        return hashlib.sha256(payload).hexdigest()

    def path(self, split: str, cache_key: str) -> Path:
        return self.root / "records" / split / f"{cache_key}.json"

    def save(self, envelope: GenerationEnvelope) -> None:
        path = self.path(envelope.split, envelope.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(envelope), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    def load(self, split: str, cache_key: str) -> GenerationEnvelope:
        payload = json.loads(self.path(split, cache_key).read_text())
        payload["sampled_frame_indices"] = tuple(payload["sampled_frame_indices"])
        payload["frame_paths"] = tuple(payload["frame_paths"])
        payload["raw_responses"] = tuple(payload["raw_responses"])
        payload["validation_errors"] = tuple(payload["validation_errors"])
        envelope = GenerationEnvelope(**payload)
        expected_dataset = self.dataset_key(
            envelope.model_digest, envelope.prompt_hash
        )
        expected_cache = self.cache_key(
            envelope.clip_id, envelope.model_digest, envelope.prompt_hash
        )
        if (
            envelope.dataset_key != expected_dataset
            or envelope.cache_key != expected_cache
            or cache_key != envelope.cache_key
            or split != envelope.split
        ):
            raise ValueError(f"appearance record provenance mismatch: {self.path(split, cache_key)}")
        return envelope

    def contains(self, split: str, cache_key: str) -> bool:
        return self.path(split, cache_key).exists()

    def iter_dataset(
        self, split: str, dataset_key: str
    ) -> Iterator[GenerationEnvelope]:
        if len(dataset_key) != 64 or any(
            character not in "0123456789abcdef" for character in dataset_key
        ):
            raise ValueError("dataset key must be a lowercase SHA-256 hex digest")
        for path in sorted((self.root / "records" / split).glob("*.json")):
            envelope = self.load(split, path.stem)
            if envelope.dataset_key == dataset_key:
                yield envelope
```

- [ ] **Step 3: Run store tests**

Run: `uv run pytest tests/unit/appearance/test_store.py -v`

Expected: `2 passed`.

- [ ] **Step 4: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/appearance/store.py tests/unit/appearance/test_store.py
git commit -m "feat: persist resumable appearance records"
```

---

### Task 6: Validation Retry Pipeline And Generation CLI

**Files:**
- Create: `src/despamo/appearance/pipeline.py`
- Create: `scripts/generate_appearance_descriptions.py`
- Create: `tests/unit/appearance/test_pipeline.py`

**Interfaces:**
- Produces: `generate_clip(client, store, clip, max_retries) -> GenerationEnvelope`.
- CLI supports `--split`, `--limit`, and repeated config files.
- Valid and terminal failed records are skipped for the same cache key, enforcing the retry cap across resumed runs.

- [ ] **Step 1: Write failing pipeline tests**

```python
# tests/unit/appearance/test_pipeline.py
import json
from pathlib import Path

from despamo.appearance.frames import ClipFrames
from despamo.appearance.ollama_client import OllamaModelRef
from despamo.appearance.pipeline import generate_clip
from despamo.appearance.store import AppearanceStore
from despamo.appearance.prompt import prompt_hash
from tests.unit.appearance.helpers import valid_payload


class SequencedClient:
    model_ref = OllamaModelRef("qwen3-vl:8b", "digest")
    endpoint = "http://127.0.0.1:11434"
    temperature = 0.0
    keep_alive = "10m"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.errors = []

    def generate(self, frame_paths, clip_id, correction_errors=()):
        assert clip_id == "clip-1"
        self.errors.append(tuple(correction_errors))
        return self.responses.pop(0)


def clip() -> ClipFrames:
    return ClipFrames(
        "clip-1",
        "Signer01",
        "train",
        20,
        (2, 6, 10, 13, 17),
        tuple(Path(f"frame-{index}.png") for index in range(5)),
    )


def test_pipeline_retries_invalid_json_then_saves_valid(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["clip_id"] = "clip-1"
    client = SequencedClient(["not json", json.dumps(payload)])
    store = AppearanceStore(tmp_path)

    envelope = generate_clip(client, store, clip(), max_retries=2)

    assert envelope.status == "valid"
    assert envelope.retries == 1
    assert envelope.frame_count == 20
    assert envelope.raw_responses == ("not json", json.dumps(payload))
    assert client.errors[1]
    assert store.contains("train", envelope.cache_key)


def test_pipeline_skips_existing_valid_record(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["clip_id"] = "clip-1"
    client = SequencedClient([json.dumps(payload)])
    store = AppearanceStore(tmp_path)
    first = generate_clip(client, store, clip(), max_retries=2)
    second = generate_clip(client, store, clip(), max_retries=2)
    assert second == first
    assert client.responses == []


def test_pipeline_skips_terminal_failed_record(tmp_path: Path) -> None:
    client = SequencedClient(["bad", "still bad", "also bad"])
    store = AppearanceStore(tmp_path)
    first = generate_clip(client, store, clip(), max_retries=2)
    second = generate_clip(client, store, clip(), max_retries=2)
    assert first.status == "failed"
    assert second == first
    assert client.responses == []


def test_pipeline_stores_frame_failure_without_calling_qwen(tmp_path: Path) -> None:
    client = SequencedClient([])
    store = AppearanceStore(tmp_path)
    missing = ClipFrames("missing", "Signer01", "train", 0, (), (), "no frames")
    envelope = generate_clip(client, store, missing, max_retries=2)
    assert envelope.status == "failed"
    assert envelope.validation_errors == ("no frames",)
    assert envelope.raw_responses == ()
```

- [ ] **Step 2: Implement bounded validation retries**

```python
# src/despamo/appearance/pipeline.py
from pydantic import ValidationError

from despamo.appearance.canonical import validate_content
from despamo.appearance.frames import ClipFrames
from despamo.appearance.prompt import PROMPT_VERSION, build_prompt, prompt_hash
from despamo.appearance.schema import AppearanceRecord
from despamo.appearance.store import AppearanceStore, GenerationEnvelope


def generate_clip(
    client,
    store: AppearanceStore,
    clip: ClipFrames,
    max_retries: int,
) -> GenerationEnvelope:
    current_prompt_hash = prompt_hash()
    dataset_key = store.dataset_key(client.model_ref.digest, current_prompt_hash)
    key = store.cache_key(clip.clip_id, client.model_ref.digest, current_prompt_hash)
    if store.contains(clip.split, key):
        return store.load(clip.split, key)
    common = {
        "dataset_key": dataset_key,
        "cache_key": key,
        "clip_id": clip.clip_id,
        "split": clip.split,
        "signer": clip.signer,
        "endpoint": client.endpoint,
        "model_tag": client.model_ref.tag,
        "model_digest": client.model_ref.digest,
        "temperature": client.temperature,
        "keep_alive": client.keep_alive,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": current_prompt_hash,
        "prompt_text": build_prompt(clip.clip_id),
        "frame_count": clip.frame_count,
        "sampled_frame_indices": clip.sampled_indices,
        "frame_paths": tuple(str(path) for path in clip.sampled_paths),
    }
    if clip.frame_error is not None:
        envelope = GenerationEnvelope(
            **common,
            raw_responses=(),
            validated_record=None,
            validation_errors=(clip.frame_error,),
            retries=0,
            status="failed",
        )
        store.save(envelope)
        return envelope
    errors: list[str] = []
    raw_responses: list[str] = []
    for attempt in range(max_retries + 1):
        corrections = (errors[-1],) if errors else ()
        raw = client.generate(clip.sampled_paths, clip.clip_id, corrections)
        raw_responses.append(raw)
        try:
            record = AppearanceRecord.model_validate_json(raw)
            if record.clip_id != clip.clip_id:
                raise ValueError(
                    f"clip_id mismatch: expected {clip.clip_id}, got {record.clip_id}"
                )
            validate_content(record)
        except (ValidationError, ValueError) as error:
            errors.append(str(error))
            continue
        envelope = GenerationEnvelope(
            **common,
            raw_responses=tuple(raw_responses),
            validated_record=record.model_dump(mode="json"),
            validation_errors=tuple(errors),
            retries=attempt,
            status="valid",
        )
        store.save(envelope)
        return envelope
    envelope = GenerationEnvelope(
        **common,
        raw_responses=tuple(raw_responses),
        validated_record=None,
        validation_errors=tuple(errors),
        retries=max_retries,
        status="failed",
    )
    store.save(envelope)
    return envelope
```

- [ ] **Step 3: Implement generation CLI**

```python
# scripts/generate_appearance_descriptions.py
import argparse
from pathlib import Path

from ollama import Client

from despamo.appearance.frames import iter_phoenix_clips
from despamo.appearance.ollama_client import OllamaAppearanceClient
from despamo.appearance.pipeline import generate_clip
from despamo.appearance.prompt import PROMPT_VERSION, prompt_hash
from despamo.appearance.schema import SCHEMA_VERSION
from despamo.appearance.store import AppearanceStore
from despamo.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config).appearance
    if config.prompt_version != PROMPT_VERSION:
        raise ValueError(
            f"config prompt version {config.prompt_version} != code {PROMPT_VERSION}"
        )
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"config schema version {config.schema_version} != code {SCHEMA_VERSION}"
        )
    annotation = Path(config.annotation_root) / f"{args.split}_info_ml.npy"
    client = OllamaAppearanceClient(
        Client(host=config.ollama_host),
        config.ollama_host,
        config.model,
        config.temperature,
        config.keep_alive,
    )
    store = AppearanceStore(Path(config.output_root))
    dataset_key = store.dataset_key(client.model_ref.digest, prompt_hash())
    print(f"dataset_key={dataset_key}")
    counts = {"valid": 0, "failed": 0}
    clips = iter_phoenix_clips(
        annotation,
        Path(config.frame_root),
        args.split,
        tuple(config.frame_positions),
    )
    for index, clip in enumerate(clips):
        if args.limit is not None and index >= args.limit:
            break
        envelope = generate_clip(
            client, store, clip, config.max_validation_retries
        )
        counts[envelope.status] += 1
        print(f"{clip.clip_id}: {envelope.status} retries={envelope.retries}")
    print(counts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run pipeline tests**

Run: `uv run pytest tests/unit/appearance/test_pipeline.py -v`

Expected: `4 passed` without contacting Ollama.

- [ ] **Step 5: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/appearance/pipeline.py scripts/generate_appearance_descriptions.py tests/unit/appearance/test_pipeline.py
git commit -m "feat: generate resumable appearance records"
```

---

### Task 7: Frozen CLIP Factor Text Features

**Files:**
- Create: `src/despamo/appearance/text_features.py`
- Create: `scripts/encode_appearance_text.py`
- Create: `tests/unit/appearance/test_text_features.py`

**Interfaces:**
- Produces: `AppearanceTextEncoder.encode(record) -> dict[Factor, ndarray | None]`.
- Produces: one compressed NPZ per clip and one manifest under its immutable dataset key.
- Records resolved Hugging Face commit SHA, canonical text, feature width, dtype, schema version, and source-record hash.

- [ ] **Step 1: Write failing text-feature test with fake encoder**

```python
# tests/unit/appearance/test_text_features.py
from pathlib import Path

import numpy as np
import torch

from despamo.appearance.canonical import canonical_descriptions
from despamo.appearance.schema import AppearanceRecord
from despamo.appearance.text_features import AppearanceTextEncoder, save_feature_file
from tests.unit.appearance.helpers import valid_payload


class FakeTokenizer:
    def __call__(self, texts, padding, truncation, return_tensors):
        return {
            "input_ids": torch.ones(len(texts), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
        }


class FakeTextModel(torch.nn.Module):
    config = type("Config", (), {"projection_dim": 4})()

    def forward(self, **tokens):
        rows = tokens["input_ids"].shape[0]
        values = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
        return type("Output", (), {"text_embeds": values})()


def test_encoder_returns_one_vector_per_visible_factor() -> None:
    encoder = AppearanceTextEncoder(FakeTokenizer(), FakeTextModel(), "revision-sha")
    record = AppearanceRecord.model_validate(valid_payload())
    features = encoder.encode(record)
    assert features["clothing"].shape == (4,)
    assert features["hair"].shape == (4,)
    assert features["background"].shape == (4,)
    assert all(array.dtype == np.float32 for array in features.values() if array is not None)


def test_feature_file_preserves_factor_masks(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["hair"]["visibility"] = "uncertain"
    record = AppearanceRecord.model_validate(payload)
    encoder = AppearanceTextEncoder(FakeTokenizer(), FakeTextModel(), "revision-sha")
    features = encoder.encode(record)
    output = tmp_path / "clip.npz"
    save_feature_file(output, features, canonical_descriptions(record))

    with np.load(output, allow_pickle=False) as stored:
        assert stored["clothing_valid"].item() is True
        assert stored["hair_valid"].item() is False
        assert stored["hair_feature"].shape == (0,)
```

- [ ] **Step 2: Implement injectable text encoder**

```python
# src/despamo/appearance/text_features.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import model_info
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from despamo.appearance.canonical import canonical_descriptions
from despamo.appearance.schema import AppearanceRecord, Factor


FACTORS: tuple[Factor, ...] = ("clothing", "hair", "background")


class AppearanceTextEncoder:
    def __init__(self, tokenizer, model, revision: str) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.revision = revision
        self.width = int(model.config.projection_dim)

    @classmethod
    def from_pretrained(
        cls, model_name: str, requested_revision: str, cache_dir: str
    ) -> "AppearanceTextEncoder":
        resolved_revision = model_info(model_name, revision=requested_revision).sha
        if resolved_revision is None:
            raise RuntimeError(f"Hugging Face did not resolve a revision for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=resolved_revision, cache_dir=cache_dir
        )
        model = CLIPTextModelWithProjection.from_pretrained(
            model_name, revision=resolved_revision, cache_dir=cache_dir
        ).eval()
        model.requires_grad_(False)
        return cls(tokenizer, model, resolved_revision)

    @torch.no_grad()
    def encode(self, record: AppearanceRecord) -> dict[Factor, np.ndarray | None]:
        descriptions = canonical_descriptions(record)
        valid = [(factor, descriptions[factor]) for factor in FACTORS if descriptions[factor]]
        output: dict[Factor, np.ndarray | None] = {factor: None for factor in FACTORS}
        if not valid:
            return output
        texts = [text for _, text in valid]
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device if any(True for _ in self.model.parameters()) else torch.device("cpu")
        tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
        features = self.model(**tokens).text_embeds.float().cpu().numpy()
        if features.shape != (len(valid), self.width):
            raise ValueError(
                f"unexpected text feature shape {features.shape}; "
                f"expected {(len(valid), self.width)}"
            )
        for (factor, _), feature in zip(valid, features):
            output[factor] = feature.astype(np.float32, copy=False)
        return output


def source_record_hash(record: dict) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def save_feature_file(
    path: Path,
    features: dict[Factor, np.ndarray | None],
    canonical: dict[Factor, str | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for factor in FACTORS:
        arrays[f"{factor}_valid"] = np.array(features[factor] is not None)
        arrays[f"{factor}_text"] = np.array(canonical[factor] or "")
        arrays[f"{factor}_feature"] = (
            features[factor] if features[factor] is not None else np.empty((0,), dtype=np.float32)
        )
    np.savez_compressed(path, **arrays)
```

- [ ] **Step 3: Remove parameter-iterator ambiguity**

Replace the device line in `encode` with a focused helper so fake and real models both work:

```python
def module_device(model) -> torch.device:
    parameter = next(iter(model.parameters()), None)
    return parameter.device if parameter is not None else torch.device("cpu")
```

Then use:

```python
        device = module_device(self.model)
```

- [ ] **Step 4: Implement embedding CLI and manifest**

```python
# scripts/encode_appearance_text.py
import argparse
import json
from pathlib import Path

from despamo.appearance.canonical import canonical_descriptions
from despamo.appearance.schema import AppearanceRecord
from despamo.appearance.store import AppearanceStore
from despamo.appearance.text_features import (
    AppearanceTextEncoder,
    save_feature_file,
    source_record_hash,
)
from despamo.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config).appearance
    root = Path(config.output_root)
    store = AppearanceStore(root)
    encoder = AppearanceTextEncoder.from_pretrained(
        config.text_encoder, config.text_encoder_revision, args.cache_dir
    )
    records = []
    for envelope in store.iter_dataset(args.split, args.dataset_key):
        if envelope.status != "valid" or envelope.validated_record is None:
            continue
        record = AppearanceRecord.model_validate(envelope.validated_record)
        features = encoder.encode(record)
        canonical = canonical_descriptions(record)
        output = (
            root
            / "text_features"
            / args.dataset_key
            / args.split
            / f"{envelope.cache_key}.npz"
        )
        save_feature_file(output, features, canonical)
        widths = {
            factor: int(feature.shape[0]) if feature is not None else 0
            for factor, feature in features.items()
        }
        records.append(
            {
                "clip_id": envelope.clip_id,
                "cache_key": envelope.cache_key,
                "path": str(output.relative_to(root)),
                "model": config.text_encoder,
                "revision": encoder.revision,
                "tokenizer": config.text_encoder,
                "tokenizer_revision": encoder.revision,
                "schema_version": record.schema_version,
                "source_record_hash": source_record_hash(envelope.validated_record),
                "width": encoder.width,
                "widths": widths,
                "dtype": "float32",
                "canonical": canonical,
            }
        )
    manifest = root / "text_features" / args.dataset_key / f"{args.split}_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"encoded {len(records)} clips into {manifest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run text-feature tests**

Run: `uv run pytest tests/unit/appearance/test_text_features.py -v`

Expected: `2 passed` without model download.

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/appearance/text_features.py scripts/encode_appearance_text.py tests/unit/appearance/test_text_features.py
git commit -m "feat: encode appearance factor text"
```

---

### Task 8: Stratified Manual Audit And Quality Gate

**Files:**
- Create: `src/despamo/appearance/audit.py`
- Create: `scripts/build_appearance_audit.py`
- Create: `scripts/summarize_appearance_audit.py`
- Create: `tests/unit/appearance/test_audit.py`

**Interfaces:**
- Produces: deterministic 100-record audit selection across signers and full sequence-length quartiles.
- Produces: CSV fields for visible-factor correctness, visibility correctness, forbidden-cue leakage, biometric leakage, and hallucination.
- Computes schema validity over the complete annotation split, not only the valid audit sample.
- Quality gate: schema validity >=99%, each visible-factor correctness >=85%, forbidden-cue leakage <=5%, biometric leakage ==0.

- [ ] **Step 1: Write failing audit summary test**

```python
# tests/unit/appearance/test_audit.py
import pytest

from despamo.appearance.audit import (
    select_audit_records,
    summarize_audit,
    validate_review_rows,
)
from despamo.appearance.store import GenerationEnvelope


def audit_row() -> dict[str, str]:
    return {
        "clothing_visible": "1",
        "clothing_correct": "1",
        "hair_visible": "1",
        "hair_correct": "1",
        "background_visible": "1",
        "background_correct": "1",
        "visibility_correct": "1",
        "forbidden_cue": "0",
        "biometric_leakage": "0",
        "hallucination": "0",
    }


def test_audit_summary_passes_thresholds() -> None:
    rows = [audit_row() for _ in range(100)]
    summary = summarize_audit(rows, total_records=1000, valid_records=995)
    assert summary["passed"] is True
    assert summary["schema_valid"] == 0.995


def test_audit_summary_rejects_biometric_leakage() -> None:
    rows = [audit_row() for _ in range(100)]
    rows[0]["biometric_leakage"] = "1"
    summary = summarize_audit(rows, total_records=100, valid_records=100)
    assert summary["passed"] is False


def test_review_validation_rejects_correctness_for_invisible_factor() -> None:
    row = audit_row()
    row["hair_visible"] = "0"
    with pytest.raises(ValueError, match="hair_correct"):
        validate_review_rows([row])


def make_envelope(index: int) -> GenerationEnvelope:
    return GenerationEnvelope(
        dataset_key="dataset-key",
        cache_key=f"key-{index}",
        clip_id=f"clip-{index}",
        split="train",
        signer=f"Signer{index % 3:02}",
        endpoint="http://127.0.0.1:11434",
        model_tag="qwen3-vl:8b",
        model_digest="digest",
        temperature=0.0,
        keep_alive="10m",
        prompt_version="qwen3-vl-appearance-v1",
        prompt_hash="prompt",
        prompt_text="prompt text",
        frame_count=(index % 4 + 1) * 10,
        sampled_frame_indices=(1, 3, 5, 7, 9),
        frame_paths=("a", "b", "c", "d", "e"),
        raw_responses=("{}",),
        validated_record={},
        validation_errors=(),
        retries=0,
        status="valid",
    )


def test_audit_selection_is_deterministic_and_stratified() -> None:
    records = [make_envelope(index) for index in range(120)]
    first = select_audit_records(records, count=100, seed=7)
    second = select_audit_records(records, count=100, seed=7)
    assert [record.cache_key for record in first] == [record.cache_key for record in second]
    assert {record.signer for record in first} == {"Signer00", "Signer01", "Signer02"}
    assert {record.frame_count for record in first} == {10, 20, 30, 40}
```

- [ ] **Step 2: Implement quality summary**

```python
# src/despamo/appearance/audit.py
from collections.abc import Sequence


FACTORS = ("clothing", "hair", "background")
ALWAYS_BINARY_FIELDS = (
    "visibility_correct",
    "forbidden_cue",
    "biometric_leakage",
    "hallucination",
)


def validate_review_rows(rows: Sequence[dict[str, str]]) -> None:
    for line_number, row in enumerate(rows, start=2):
        for factor in FACTORS:
            visible_field = f"{factor}_visible"
            correct_field = f"{factor}_correct"
            if row.get(visible_field) not in {"0", "1"}:
                raise ValueError(f"line {line_number}: {visible_field} must be 0 or 1")
            expected = {"0", "1"} if row[visible_field] == "1" else {""}
            if row.get(correct_field) not in expected:
                raise ValueError(
                    f"line {line_number}: {correct_field} must be 0/1 when visible "
                    "and empty otherwise"
                )
        for field in ALWAYS_BINARY_FIELDS:
            if row.get(field) not in {"0", "1"}:
                raise ValueError(f"line {line_number}: {field} must be 0 or 1")


def _rate(rows: Sequence[dict[str, str]], field: str) -> float:
    return sum(int(row[field]) for row in rows) / len(rows)


def _visible_factor_rate(rows: Sequence[dict[str, str]], factor: str) -> float:
    visible = [row for row in rows if row[f"{factor}_visible"] == "1"]
    if not visible:
        raise ValueError(f"audit has no visible {factor} records")
    return _rate(visible, f"{factor}_correct")


def summarize_audit(
    rows: Sequence[dict[str, str]], total_records: int, valid_records: int
) -> dict[str, float | bool]:
    if len(rows) != 100:
        raise ValueError(f"audit requires exactly 100 rows, got {len(rows)}")
    if total_records <= 0 or not 0 <= valid_records <= total_records:
        raise ValueError(
            f"invalid dataset counts: total={total_records}, valid={valid_records}"
        )
    summary: dict[str, float | bool] = {
        "schema_valid": valid_records / total_records,
        "clothing_correct": _visible_factor_rate(rows, "clothing"),
        "hair_correct": _visible_factor_rate(rows, "hair"),
        "background_correct": _visible_factor_rate(rows, "background"),
        "visibility_correct": _rate(rows, "visibility_correct"),
        "forbidden_cue": _rate(rows, "forbidden_cue"),
        "biometric_leakage": _rate(rows, "biometric_leakage"),
        "hallucination": _rate(rows, "hallucination"),
    }
    passed = (
        summary["schema_valid"] >= 0.99
        and summary["clothing_correct"] >= 0.85
        and summary["hair_correct"] >= 0.85
        and summary["background_correct"] >= 0.85
        and summary["forbidden_cue"] <= 0.05
        and summary["biometric_leakage"] == 0
    )
    summary["passed"] = passed
    return summary
```

- [ ] **Step 3: Implement deterministic stratified selector**

Add to `src/despamo/appearance/audit.py`:

```python
import random
from collections import defaultdict

from despamo.appearance.store import GenerationEnvelope


def select_audit_records(
    records: Sequence[GenerationEnvelope], count: int = 100, seed: int = 0
) -> list[GenerationEnvelope]:
    valid = [record for record in records if record.status == "valid"]
    if len(valid) < count:
        raise ValueError(f"need at least {count} valid records, found {len(valid)}")
    lengths = sorted(record.frame_count for record in valid)
    boundaries = [lengths[len(lengths) * index // 4] for index in (1, 2, 3)]

    def bucket(record: GenerationEnvelope) -> int:
        length = record.frame_count
        return sum(length >= boundary for boundary in boundaries)

    groups: dict[tuple[str, int], list[GenerationEnvelope]] = defaultdict(list)
    for record in valid:
        groups[(record.signer, bucket(record))].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            raise ValueError("stratified groups exhausted before audit target")
    return selected
```

- [ ] **Step 4: Implement audit CLIs**

```python
# scripts/build_appearance_audit.py
import argparse
import csv
from pathlib import Path

from despamo.appearance.audit import select_audit_records
from despamo.appearance.canonical import canonical_descriptions
from despamo.appearance.schema import AppearanceRecord
from despamo.appearance.store import AppearanceStore
from despamo.config import load_config


REVIEW_FIELDS = [
    "clothing_visible",
    "clothing_correct",
    "hair_visible",
    "hair_correct",
    "background_visible",
    "background_correct",
    "visibility_correct",
    "forbidden_cue",
    "biometric_leakage",
    "hallucination",
    "review_notes",
]
FIELDS = [
    "clip_id",
    "signer",
    "frame_count",
    "sampled_frame_indices",
    "frame_paths",
    "record_path",
    "clothing_description",
    "hair_description",
    "background_description",
    *REVIEW_FIELDS,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config).appearance
    root = Path(config.output_root)
    store = AppearanceStore(root)
    records = list(store.iter_dataset(args.split, args.dataset_key))
    selected = select_audit_records(records, args.count, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record in selected:
            appearance = AppearanceRecord.model_validate(record.validated_record)
            descriptions = canonical_descriptions(appearance)
            row = {
                "clip_id": record.clip_id,
                "signer": record.signer,
                "frame_count": record.frame_count,
                "sampled_frame_indices": " | ".join(
                    str(index) for index in record.sampled_frame_indices
                ),
                "frame_paths": " | ".join(record.frame_paths),
                "record_path": str(
                    store.path(record.split, record.cache_key).relative_to(root)
                ),
                "clothing_description": descriptions["clothing"] or "",
                "hair_description": descriptions["hair"] or "",
                "background_description": descriptions["background"] or "",
            }
            row.update({field: "" for field in REVIEW_FIELDS})
            writer.writerow(row)


if __name__ == "__main__":
    main()
```

```python
# scripts/summarize_appearance_audit.py
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from despamo.appearance.audit import summarize_audit, validate_review_rows
from despamo.appearance.store import AppearanceStore
from despamo.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config).appearance
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_review_rows(rows)

    annotation = Path(config.annotation_root) / f"{args.split}_info_ml.npy"
    raw_annotations = np.load(annotation, allow_pickle=True).item()
    expected_total = sum(isinstance(key, int) for key in raw_annotations)
    root = Path(config.output_root)
    store = AppearanceStore(root)
    records = list(store.iter_dataset(args.split, args.dataset_key))
    if len(records) != expected_total:
        raise ValueError(
            f"generation incomplete: expected {expected_total} records, "
            f"found {len(records)}"
        )
    valid_records = sum(record.status == "valid" for record in records)
    summary = summarize_audit(rows, expected_total, valid_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run audit tests**

Run: `uv run pytest tests/unit/appearance/test_audit.py tests/unit/appearance/test_store.py -v`

Expected: audit and updated storage tests pass.

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/appearance scripts/build_appearance_audit.py scripts/summarize_appearance_audit.py tests/unit/appearance
git commit -m "feat: audit appearance descriptions"
```

---

### Task 9: Live Integration And Dataset Gate

**Files:**
- Create: `tests/integration/appearance/test_local_ollama.py`
- Modify: `README.md`

**Interfaces:**
- Verifies installed local model digest, five-image request, schema-valid result, resumable storage, CLIP feature cache, and audit gate.
- Produces training-split appearance records and text-feature manifest outside Git.

- [ ] **Step 1: Add opt-in local Ollama integration test**

```python
# tests/integration/appearance/test_local_ollama.py
import os
from pathlib import Path

import pytest
from ollama import Client

from despamo.appearance.ollama_client import OllamaAppearanceClient
from despamo.appearance.schema import AppearanceRecord


@pytest.mark.integration
def test_local_qwen_accepts_five_images() -> None:
    fixture_dir = os.environ.get("DESPAMO_APPEARANCE_FIXTURES")
    if fixture_dir is None:
        pytest.skip("DESPAMO_APPEARANCE_FIXTURES is not configured")
    paths = tuple(sorted(Path(fixture_dir).glob("*.png")))
    if len(paths) != 5:
        pytest.skip("fixture directory must contain exactly five PNG files")
    client = OllamaAppearanceClient(
        Client(host="http://127.0.0.1:11434"),
        "http://127.0.0.1:11434",
        "qwen3-vl:8b",
        0.0,
        "10m",
    )
    record = AppearanceRecord.model_validate_json(
        client.generate(paths, "integration-fixture")
    )
    assert record.schema_version == 1
    assert record.clip_id == "integration-fixture"
```

- [ ] **Step 2: Run one live Qwen request**

Configure `DESPAMO_APPEARANCE_FIXTURES` in `.env` to a directory containing exactly five non-sensitive PNG fixtures. Then run:

```bash
set -a
source .env
set +a
uv run pytest tests/integration/appearance/test_local_ollama.py -v
```

Expected: `1 passed`; model digest is available; response validates against schema.

- [ ] **Step 3: Run one real PHOENIX clip end to end**

Run:

```bash
set -a
source .env
set +a
uv run python scripts/generate_appearance_descriptions.py --config configs/appearance/qwen3_vl.yaml --split train --limit 1
```

Expected: one valid record or one explicit failed record with every raw response and validation error; rerunning skips either terminal result for the same cache key.

Record the printed `dataset_key` as `DESPAMO_APPEARANCE_DATASET_KEY` in `.env`. Prompt or model changes print a different key and must use separate downstream artifacts.

- [ ] **Step 4: Generate training records resumably**

Run same command without `--limit`.

Expected: all 7,096 training clips have either valid or explicit failed envelopes; interrupted runs resume without regenerating terminal cache keys.

- [ ] **Step 5: Build and complete 100-clip audit**

Run:

```bash
uv run python scripts/build_appearance_audit.py --config configs/appearance/qwen3_vl.yaml --split train --dataset-key "$DESPAMO_APPEARANCE_DATASET_KEY" --count 100 --seed 0 --output "$DESPAMO_APPEARANCE_ROOT/audit/$DESPAMO_APPEARANCE_DATASET_KEY/train-100.csv"
uv run python scripts/summarize_appearance_audit.py --config configs/appearance/qwen3_vl.yaml --split train --dataset-key "$DESPAMO_APPEARANCE_DATASET_KEY" --input "$DESPAMO_APPEARANCE_ROOT/audit/$DESPAMO_APPEARANCE_DATASET_KEY/train-100.csv" --output "$DESPAMO_APPEARANCE_ROOT/audit/$DESPAMO_APPEARANCE_DATASET_KEY/train-100-summary.json"
```

Expected: summary passes schema validity, per-factor correctness, forbidden-cue, and zero-biometric-leakage thresholds. If it fails, revise prompt version and regenerate under a new prompt hash before continuing.

- [ ] **Step 6: Encode frozen CLIP text features**

Run:

```bash
uv run python scripts/encode_appearance_text.py --config configs/appearance/qwen3_vl.yaml --split train --dataset-key "$DESPAMO_APPEARANCE_DATASET_KEY" --cache-dir "$DESPAMO_HF_CACHE"
```

Expected: every valid training envelope has one NPZ file and one manifest row; each visible factor has a non-empty float32 vector with common width and immutable encoder revision.

- [ ] **Step 7: Run final verification**

Run:

```bash
uv run ruff check src scripts tests
uv run pytest tests/unit/appearance -v
git status --short
git diff --check
```

Expected: lint clean; all appearance unit tests pass; generated data remains outside Git; no whitespace errors.

- [ ] **Step 8: Document measured dataset statistics**

Add an `Appearance Supervision Data` section to `README.md` containing:

- Qwen model tag and digest.
- Appearance dataset key.
- Prompt version and hash.
- Clip counts by valid and failed status.
- Visibility counts per factor.
- Audit metrics and gate result.
- CLIP text model and resolved revision.
- Manifest paths and exact generation commands.

Report observed values only after generation and audit finish.

- [ ] **Step 9: Checkpoint commit if explicitly authorized**

```bash
git add README.md tests/integration/appearance
git commit -m "test: verify Qwen appearance dataset"
```

---

## Appearance Dataset Completion Gate

- [ ] Baseline and DINOv3 prerequisite milestones passed before execution.
- [ ] Local `qwen3-vl:8b` digest captured.
- [ ] Immutable dataset key isolates one model digest and prompt hash.
- [ ] Five-frame sampling deterministic and traceable.
- [ ] Every generation attempt stored as valid or explicit failure.
- [ ] Valid and terminal failed cache keys resume without regeneration.
- [ ] Structured records contain only clothing, hair, and background factors.
- [ ] Manual 100-clip audit passes all quality thresholds.
- [ ] Every valid clip has factor-specific frozen CLIP text features.
- [ ] Text-feature manifest records immutable encoder revision and source hash.
- [ ] Unit and live integration tests pass.
- [ ] README records measured statistics and exact commands.

## Deferred Follow-Up

After this gate passes, create the separate three-head GRL implementation plan covering:

1. Multi-positive factor losses and visibility masks.
2. Independent GRL heads and coefficient ramp.
3. Single-factor and combined training variants.
4. Fresh signer and factor probes.
5. Three-seed effect tables and background-retention gate.
