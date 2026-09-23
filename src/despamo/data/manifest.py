from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np


def _supported_float_dtype(dtype: str | np.dtype) -> bool:
    try:
        parsed = np.dtype(dtype)
    except (TypeError, ValueError):
        return False
    return (
        parsed.kind == "f"
        and parsed.itemsize in {2, 4, 8}
        and (not isinstance(dtype, str) or str(parsed) == dtype)
    )


@dataclass(frozen=True)
class FeatureRecord:
    clip_id: str
    split: str
    path: str
    length: int
    width: int
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or self.split in {"", ".", ".."} or "/" in self.split:
            raise ValueError(f"invalid split: {self.split!r}")
        if not isinstance(self.clip_id, str) or not self.clip_id or "/" in self.clip_id:
            raise ValueError(f"invalid clip_id: {self.clip_id!r}")
        if self.path != f"{self.split}/{self.clip_id}.npy":
            raise ValueError(f"invalid feature path: {self.path!r}")
        if type(self.length) is not int or self.length <= 0:
            raise ValueError(f"invalid feature length: {self.length!r}")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError(f"invalid feature width: {self.width!r}")
        if not isinstance(self.dtype, str) or not _supported_float_dtype(self.dtype):
            raise ValueError(f"unsupported feature dtype: {self.dtype!r}: {self.path}")


@dataclass(frozen=True)
class FeatureManifest:
    schema_version: int
    encoder: str
    expected_dim: int
    records: tuple[FeatureRecord, ...]
    _lookup: Mapping[tuple[str, str], FeatureRecord] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.encoder, str) or not self.encoder.strip():
            raise ValueError("encoder must be non-empty")
        if type(self.expected_dim) is not int or self.expected_dim <= 0:
            raise ValueError(f"expected_dim must be positive: {self.expected_dim!r}")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("records must be a non-empty tuple")
        lookup: dict[tuple[str, str], FeatureRecord] = {}
        for record in self.records:
            if not isinstance(record, FeatureRecord):
                raise ValueError(f"invalid feature record: {record!r}")
            if record.width != self.expected_dim:
                raise ValueError(
                    f"expected width {self.expected_dim}, got {record.width}: {record.path}"
                )
            key = (record.split, record.clip_id)
            if key in lookup:
                raise ValueError(f"duplicate feature: {record.split}/{record.clip_id}")
            lookup[key] = record
        object.__setattr__(self, "_lookup", MappingProxyType(lookup))

    def __reduce__(
        self,
    ) -> tuple[type[FeatureManifest], tuple[int, str, int, tuple[FeatureRecord, ...]]]:
        return (type(self), (self.schema_version, self.encoder, self.expected_dim, self.records))

    def require(self, split: str, clip_id: str) -> FeatureRecord:
        try:
            return self._lookup[(split, clip_id)]
        except KeyError as exc:
            raise KeyError(f"expected one feature for {split}/{clip_id}, found 0") from exc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "encoder": self.encoder,
            "expected_dim": self.expected_dim,
            "records": [asdict(record) for record in self.records],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> FeatureManifest:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("invalid feature manifest: expected JSON object")
        records_data = payload.get("records")
        if not isinstance(records_data, list):
            raise ValueError("invalid feature manifest records: expected list")
        records: list[FeatureRecord] = []
        for item in records_data:
            if not isinstance(item, dict):
                raise ValueError(f"invalid feature record: {item!r}")
            try:
                records.append(FeatureRecord(**item))
            except TypeError as exc:
                raise ValueError(f"invalid feature record: {exc}") from exc
        try:
            return cls(
                records=tuple(records), **{k: v for k, v in payload.items() if k != "records"}
            )
        except TypeError as exc:
            raise ValueError(f"invalid feature manifest: {exc}") from exc


def index_feature_tree(root: Path, expected_dim: int, encoder: str) -> FeatureManifest:
    if type(expected_dim) is not int or expected_dim <= 0:
        raise ValueError(f"expected_dim must be positive: {expected_dim!r}")
    if not isinstance(encoder, str) or not encoder.strip():
        raise ValueError("encoder must be non-empty")

    records: list[FeatureRecord] = []
    for path in sorted(root.glob("*/*.npy")):
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"invalid feature file: {path}") from exc
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(f"feature must be non-empty rank 2: {path}")
        if array.shape[1] != expected_dim:
            raise ValueError(f"expected width {expected_dim}, got {array.shape[1]}: {path}")
        if not _supported_float_dtype(array.dtype):
            raise ValueError(f"unsupported feature dtype: {array.dtype}: {path}")
        records.append(
            FeatureRecord(
                clip_id=path.stem,
                split=path.parent.name,
                path=path.relative_to(root).as_posix(),
                length=int(array.shape[0]),
                width=int(array.shape[1]),
                dtype=str(array.dtype),
            )
        )
    if not records:
        raise ValueError(f"no .npy feature files found under {root}")
    return FeatureManifest(1, encoder, expected_dim, tuple(records))
