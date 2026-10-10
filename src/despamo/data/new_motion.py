"""Fresh motion views and compact anchors for independently closed package graphs.

References are metadata, never proof: revalidation reconstructs all source, window,
receipt, array, generator and environment bytes via the producer's validator.
"""

import re
from pathlib import Path

from despamo.appearance.provenance import digest, file_hash, read_json
from despamo.data.manifest import FeatureManifest
from despamo.data.motion_validation import reuse_motion_admission, validate_motion_package
from despamo.data.phoenix14t import Phoenix14T
from despamo.data.signer_split import PhoenixTrainView
from tools.motion.package import POLICY
from tools.motion.records import regular_path

GRAPH_POLICY = "despamo-motion-package-graph-ref-v1"
REFERENCE_FIELDS = (
    "package_policy", "root", "identity_hash", "manifest_path", "manifest_hash",
    "timing_path", "timing_hash", "annotation_path", "annotation_hash",
    "spatial_frame_rows_path", "spatial_frame_rows_hash", "completion", "completion_hash",
)


def compact_motion_binding(package: dict) -> dict:
    """Constant-size graph reference, excluding the 827k-entry bindings inventory."""
    result = {"schema_version": 1, "graph_policy": GRAPH_POLICY,
              **{name: package[name] for name in REFERENCE_FIELDS}}
    validate_motion_reference(result)
    return result


def validate_motion_reference(reference: dict) -> None:
    """Structural mode guard only; callers must close the graph at boundaries."""
    if (not isinstance(reference, dict)
            or set(reference) != {*REFERENCE_FIELDS, "schema_version", "graph_policy"}
            or type(reference["schema_version"]) is not int or reference["schema_version"] != 1
            or reference["graph_policy"] != GRAPH_POLICY or reference["package_policy"] != POLICY):
        raise ValueError("new motion graph reference schema/policy mismatch")
    for name in REFERENCE_FIELDS:
        value = reference[name]
        if not isinstance(value, str):
            raise ValueError("new motion reference requires string paths/hashes")
        if name.endswith("hash") and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("new motion reference hash malformed")
        if (name.endswith("path") or name in ("root", "completion")) and (
                not Path(value).is_absolute() or Path(value) != Path(value).resolve()):
            raise ValueError("new motion reference path noncanonical")
    root = Path(reference["root"])
    if (root.name != f"motion-features-{reference['identity_hash']}"
            or reference["completion"] != str(root / "complete/completion.json")
            or reference["manifest_path"] != str(root / "complete/manifest.json")
            or reference["timing_path"] != str(root / "complete/motion-timing.json")):
        raise ValueError("new motion reference package path mismatch")


def revalidate_motion_binding(reference: dict) -> dict:
    """Close the entire graph even if completion bytes have not changed."""
    validate_motion_reference(reference)
    frames = read_json(regular_path(Path(reference["spatial_frame_rows_path"])))
    package = validate_motion_package(Path(reference["root"]), frames, verify_features=True)
    if compact_motion_binding(package) != reference:
        raise ValueError("new motion immutable graph reference changed")
    return package


def load_new_motion_views(original_views: dict, package: dict, root: Path) -> dict:
    """Keep original parents untouched; use pinned annotations and new motion bytes."""
    if not original_views:
        raise ValueError("new motion requires original views")
    first = next(iter(original_views.values())).parent
    frame_path = first.spatial_root / "complete/frame_rows.json"
    frames = read_json(regular_path(frame_path))
    actual = reuse_motion_admission(package, root, frames)
    if ((actual is not package and digest(actual) != digest(package))
            or actual["spatial_frame_rows_path"] != str(frame_path)):
        raise ValueError("new motion descriptor/source differs from independent admission")
    motion = FeatureManifest.load(Path(actual["manifest_path"]))
    parents, result, ids = {}, {}, []
    for name, view in original_views.items():
        parent = view.parent
        if (parent.split != "train" or parent.spatial_crop_mode != "full"
                or parent.spatial_root != first.spatial_root
                or parent.spatial_manifest != first.spatial_manifest):
            raise ValueError("new motion original spatial parent mismatch")
        if id(parent) not in parents:
            fresh = Phoenix14T(Path(actual["annotation_path"]), "train", parent.spatial_root,
                               root, parent.spatial_manifest, motion,
                               parent.max_frame_len, parent.spatial_crop_mode)
            if digest(fresh.records) != digest(parent.records):
                raise ValueError("new motion annotation records/IDs/signers/text mismatch")
            for row in fresh.records:
                clip = row["fileid"]
                frame = frames["clips"][clip]
                record = parent.spatial_manifest.require("train", clip)
                if (record.length != frame["frame_count"]
                        or frame["source_indices"] != list(range(record.length))
                        or file_hash(regular_path(parent.spatial_root / record.path))
                        != frame["feature_hash"]):
                    raise ValueError("new motion full spatial source/feature rows mismatch")
            parents[id(parent)] = fresh
        clip_ids = tuple(row["fileid"] for row in view.records)
        fresh_view = PhoenixTrainView(parents[id(parent)], clip_ids)
        if digest(fresh_view.records) != digest(view.records):
            raise ValueError("new motion logical view records mismatch")
        ids.extend(clip_ids)
        result[name] = fresh_view
    if len(set(ids)) != len(ids) or sorted(ids) != actual["clip_ids"]:
        raise ValueError("new motion logical/physical clip inventory mismatch")
    return result
