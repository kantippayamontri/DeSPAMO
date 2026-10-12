"""Small complete source joins for merged-target tests."""

from copy import deepcopy

from despamo.appearance.provenance import digest

REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
FACTORS = ("biometric", "clothing", "hair", "background")
INDICES = [5, 16, 26, 36, 47]


def inputs():
    old, crop, clips, rows = [], [], [], {}
    for name, signer in (("a", "Signer03"), ("b", "Signer07")):
        paths = [f"train/{name}/images{i+1:04}.png" for i in range(53)]
        targets = [dict(factor=f, frame_index=-1, text=f"old {f}") for f in FACTORS]
        targets += [dict(factor=f, frame_index=i, text=f"old {f} {i}")
                    for i in INDICES for f in ("left_hand", "right_hand", "mouth")]
        old.append(dict(clip_id=name, targets=targets, path=f"{name}.npz",
                        file_hash="0" * 64, record_hash=digest(targets)))
        fresh = [dict(factor=f, frame_index=-1, text=None,
                      mask_reason="stable_not_extracted") for f in FACTORS]
        fresh += [dict(factor=f, frame_index=i, text=f"new {f} {i}",
                       mask_reason=None, bbox=[1, 2, 3, 4], job_key="j", journal_sha256="1"*64)
                  for i in INDICES for f in ("left_hand", "right_hand", "mouth")]
        crop.append(dict(clip_id=name, signer=signer, split="train", num_frames=53,
                         frame_indices=INDICES.copy(), frame_paths=[paths[i] for i in INDICES],
                         normalized_positions=[.1, .3, .5, .7, .9], targets=fresh))
        clips.append(dict(clip_id=name, signer=signer, frame_count=53,
                          frames=[dict(index=i, path=paths[i], sha256="2"*64) for i in INDICES]))
        rows[name] = dict(frame_count=53, source_indices=list(range(53)), paths=paths,
                          sampled_images={str(i): "2"*64 for i in INDICES})
    old_manifest = dict(metadata=dict(model="openai/clip-vit-large-patch14", revision=REVISION,
                                     tokenizer_revision=REVISION, width=768, dtype="float32"),
                        records=old)
    crop_export = dict(metadata=dict(format_version="qwen-crop-text-targets-v1",
                                    canonical_version="crop-closed-v1", split="train",
                                    clip_count=2, target_slots=19,
                                    human_review_status="not_assessed",
                                    records_sha256=digest(crop)), records=crop)
    return old_manifest, crop_export, dict(clips=clips), dict(clips=rows), ("a", "b")


def refreshed(value):
    value = deepcopy(value)
    value[1]["metadata"]["records_sha256"] = digest(value[1]["records"])
    return value
