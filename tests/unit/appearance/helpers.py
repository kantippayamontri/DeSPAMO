from copy import deepcopy


def payload():
    fields = {
        "biometric": dict(
            apparent_age_band="young_adult",
            gender_presentation="uncertain",
            apparent_height="uncertain",
            apparent_build="uncertain",
        ),
        "clothing": dict(
            upper_garment="shirt", upper_color="black", lower_garment=None,
            lower_color=None, pattern=None, accessories=[],
        ),
        "hair": dict(color="brown", length="short", style=None),
        "background": dict(scene="studio", dominant_colors=[], static_objects=[], lighting=None),
        "left_hand": dict(
            finger_configuration="open fingers", palm_orientation=None,
            body_relative_location="torso", contact="none",
        ),
        "right_hand": dict(
            finger_configuration="closed fingers", palm_orientation=None,
            body_relative_location="torso", contact="none",
        ),
        "mouth": dict(
            openness="closed", lip_configuration="neutral", teeth_or_tongue_visibility="none",
        ),
    }
    positions = (0.1, 0.3, 0.5, 0.7, 0.9)
    indices = (1, 3, 5, 7, 9)
    return {
        "schema_version": 2,
        "clip_id": "clip-1",
        "stable": {factor: {**fields[factor], "visibility": "clear"}
                   for factor in ("biometric", "clothing", "hair", "background")},
        "frames": [
            dict(normalized_position=position, source_frame_index=index,
                 **{factor: {**fields[factor], "visibility": "clear"}
                    for factor in ("left_hand", "right_hand", "mouth")})
            for position, index in zip(positions, indices, strict=True)
        ],
    }


def clip():
    return dict(
        clip_id="clip-1", signer="s1", frame_count=11, error=None,
        frames=[dict(position=position, index=index, path=f"{index}.png", sha256="fixture")
                for position, index in zip((0.1, 0.3, 0.5, 0.7, 0.9),
                                           (1, 3, 5, 7, 9), strict=True)],
    )


def prepared_dataset(tmp_path, count=1001):
    from despamo.appearance.generation import record_path
    from despamo.appearance.provenance import atomic_json, create_dataset

    clips = []
    for i in range(count):
        item = deepcopy(clip())
        item.update(clip_id=f"clip-{i:04}", signer=f"s{i % 4}")
        clips.append(item)
    dataset = create_dataset(tmp_path, dict(split="train", clips=clips), {"fixture": True})
    for i, item in enumerate(clips):
        record = payload()
        record["clip_id"] = item["clip_id"]
        failed = i == count - 1
        journal = dict(
            dataset_key=dataset.name, clip_id=item["clip_id"],
            status="failed" if failed else "valid", record=None if failed else record,
            attempts=[], error="synthetic failure" if failed else None,
        )
        atomic_json(record_path(dataset, item["clip_id"]), journal)
    return dataset
