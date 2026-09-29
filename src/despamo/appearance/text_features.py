"""Cache revision-pinned, frozen CLIP text targets for all expected clips."""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from despamo.appearance.canonical import targets
from despamo.appearance.generation import load_records
from despamo.appearance.provenance import atomic_json, digest, file_hash, load_dataset, read_json
from despamo.appearance.schema import ARTICULATORS, STABLE, Record

MODEL = "openai/clip-vit-large-patch14"


class TextEncoder:
    def __init__(self, tokenizer, model, revision: str):
        self.tokenizer = tokenizer
        self.model = model.eval().requires_grad_(False)
        self.revision = revision
        self.width = model.config.projection_dim

    @classmethod
    def load(cls, cache_dir: str, revision: str = "main"):
        from huggingface_hub import model_info
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        sha = model_info(MODEL, revision=revision).sha
        if not sha:
            raise ValueError("unresolved encoder revision")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=sha, cache_dir=cache_dir)
        model = CLIPTextModelWithProjection.from_pretrained(
            MODEL, revision=sha, cache_dir=cache_dir
        )
        return cls(tokenizer, model, sha)

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.width), dtype=np.float32)
        tokens = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.model.config.max_position_embeddings:
            raise ValueError("CLIP token overflow: shorten schema/prompt and regenerate version")
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        output = self.model(**{key: value.to(device) for key, value in tokens.items()}).text_embeds
        if output.shape != (len(texts), self.width) or not torch.isfinite(output).all():
            raise ValueError("invalid text feature shape or values")
        return output.float().cpu().numpy()


def _masked_targets(clip: dict) -> list[dict]:
    return [
        dict(factor=factor, frame_index=-1, text=None) for factor in STABLE
    ] + [
        dict(factor=factor, frame_index=frame["index"], text=None)
        for frame in clip["frames"] for factor in ARTICULATORS
    ]


def encode_dataset(dataset: Path, encoder: TextEncoder) -> Path:
    meta = dict(
        model=MODEL, revision=encoder.revision, tokenizer=MODEL,
        tokenizer_revision=encoder.revision, schema_version=2,
        width=encoder.width, dtype="float32", canonical_version=2,
    )
    root = dataset / "text" / digest(meta)
    root.mkdir(parents=True, exist_ok=True)
    sources, _ = load_dataset(dataset)
    source_by_id = {clip["clip_id"]: clip for clip in sources["clips"]}
    rows = []
    for journal in load_records(dataset):
        clip_id = journal["clip_id"]
        target_list = (
            targets(Record.model_validate(journal["record"]))
            if journal["status"] == "valid"
            else _masked_targets(source_by_id[clip_id])
        )
        if len(target_list) != 19:
            raise ValueError(f"{clip_id}: expected nineteen factor targets")
        mask = np.array([target["text"] is not None for target in target_list], dtype=np.bool_)
        values = encoder.encode(
            [target["text"] for target in target_list if target["text"] is not None]
        )
        if values.shape != (int(mask.sum()), encoder.width) or values.dtype != np.float32:
            raise ValueError(f"{clip_id}: text embedding width/dtype mismatch")
        vectors = np.zeros((19, encoder.width), dtype=np.float32)
        vectors[mask] = values
        path = root / f"{digest(clip_id)}.npz"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, vectors=vectors, valid=mask)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                if file_hash(path) != file_hash(temporary):
                    raise ValueError(f"{clip_id}: cached text features differ from encoder version")
            else:
                os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        rows.append(dict(
            clip_id=clip_id, record_hash=digest(journal), path=path.name,
            file_hash=file_hash(path), targets=target_list,
        ))
    manifest = root / "manifest.json"
    content = dict(dataset_key=dataset.name, encoder_key=root.name, metadata=meta, records=rows)
    if manifest.exists() and read_json(manifest) != content:
        raise ValueError("text target manifest differs from encoder version")
    atomic_json(manifest, content)
    return manifest
