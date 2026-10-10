"""CPU-only timing receipt preparation from supplied, independently assessed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from despamo.appearance.provenance import atomic_json, file_hash
from despamo.data.motion_timing import FILE_KEYS, build_motion_timing, validate_motion_timing


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_output(output: Path, inputs: list[Path]) -> None:
    for path in (output, *output.parents):
        if path.is_symlink():
            raise ValueError(f"output symlink forbidden: {path}")
    for path in inputs:
        if output.resolve() == path.resolve() or (output.exists() and output.samefile(path)):
            raise ValueError(f"output aliases immutable input: {path}")
    if output.exists():
        raise ValueError(f"output already exists: {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in sorted(FILE_KEYS | {"provenance", "output"}):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args(argv)
    paths = {name: getattr(args, name) for name in FILE_KEYS}
    try:
        _safe_output(args.output, [*paths.values(), args.provenance])
        provenance = _read(args.provenance)
        frames = _read(paths["frame_rows"])
        manifest = _read(paths["motion_manifest"])
        timing = build_motion_timing(frames, manifest, provenance)
        # Both byte-level hashes and canonical payload identities must match.
        actual = {name: file_hash(path) for name, path in paths.items()}
        if actual != provenance["file_sha256"]:
            raise ValueError("input/extractor/helper byte hash mismatch")
        evidence_paths = [Path(item["path"]) for item in provenance["evidence"]]
        _safe_output(args.output, [*paths.values(), args.provenance, *evidence_paths])
        for item, path in zip(provenance["evidence"], evidence_paths, strict=True):
            if file_hash(path) != item["sha256"]:
                raise ValueError(f"chronology evidence byte hash mismatch: {path}")
        timing["provenance_sha256"] = file_hash(args.provenance)
        validate_motion_timing(timing, frames, manifest)
        atomic_json(args.output, timing)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"motion timing rejected: {exc}\n")
    print(f"Wrote {args.output}: {len(timing['clips'])} timing-validated clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
