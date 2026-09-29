"""Freeze one signer-disjoint E1/E2/E3 comparison protocol, separate from legacy runs."""

from despamo.appearance.provenance import digest

CLIP_MODEL = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
SUPERVISION_POLICY = "qwen-schema98-unreviewed-v1"
EXPECTED = {
    "train": (5746, 5641, 105),
    "dev": (582, 571, 11),
    "test": (768, 765, 3),
}


def validate_live_counts(split: dict) -> None:
    for name, (total, valid, failed) in EXPECTED.items():
        group = split["groups"][name]
        if (
            len(group["clip_ids"]) != total
            or group["valid"] != valid
            or group["failed"] != failed
        ):
            raise ValueError(f"frozen {name} signer-pilot count mismatch")
    train = split["groups"]["train"]
    if train["valid"] * 100 < len(train["clip_ids"]) * 98:
        raise ValueError("98% training schema validity required")


def bind_protocol(
    split: dict,
    dataset_key: str,
    source_hash: str,
    version_hash: str,
    records_hash: str,
    text_manifest_hash: str,
    text_metadata: dict,
    frame_rows_hash: str,
    spatial_manifest_hash: str,
    motion_manifest_hash: str,
    annotation_hash: str,
) -> dict:
    if (
        text_metadata.get("model") != CLIP_MODEL
        or text_metadata.get("revision") != CLIP_REVISION
        or text_metadata.get("tokenizer_revision") != CLIP_REVISION
        or text_metadata.get("schema_version") != 2
        or text_metadata.get("width") != 768
    ):
        raise ValueError("wrong frozen CLIP text encoder/metadata")
    if not all(isinstance(value, str) and value for value in (
        dataset_key, source_hash, version_hash, records_hash, text_manifest_hash,
        frame_rows_hash, spatial_manifest_hash, motion_manifest_hash, annotation_hash,
    )):
        raise ValueError("missing signer-pilot input identity")
    protocol = {
        "version": "signer-pilot-seed0-v1",
        "seed": 0,
        "gpu_hour_ceiling": 24,
        "max_bleu4_drop_points": 0.5,
        "dataset_key": dataset_key,
        "supervision_policy": SUPERVISION_POLICY,
        "human_review_status": "not_assessed",
        "source_hash": source_hash,
        "version_hash": version_hash,
        "records_hash": records_hash,
        "text_manifest_hash": text_manifest_hash,
        "text_metadata": text_metadata,
        "frame_rows_hash": frame_rows_hash,
        "spatial_manifest_hash": spatial_manifest_hash,
        "motion_manifest_hash": motion_manifest_hash,
        "annotation_hash": annotation_hash,
        "split_hash": split["split_hash"],
        "split": split,
        "decoder": {"mode": "deterministic", "beam_size": 5,
                    "max_length": 64, "in_context": False},
        "variants": ["E1_frozen", "E2_projector", "E3_dino_lora"],
        "spaMo_budget": "profile_then_freeze_common_tier",
    }
    return {**protocol, "protocol_hash": digest(protocol)}
