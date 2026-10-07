import pytest
from how2sign_probe.labels import (
    HAND_FIELDS,
    MOUTH_FIELDS,
    build_prompt,
    parse_label,
    version_manifest,
)


def test_prompts_embed_spec_closed_vocabularies():
    prompt = build_prompt("hand", "LEFT")
    assert "This is the signer's LEFT hand" in prompt
    assert "handshape_class: flat_open | flat_closed" in prompt
    assert "not_visible" in prompt
    mouth = build_prompt("mouth", "MOUTH")
    assert "cheek_state: neutral | puffed" in mouth
    assert "Do NOT describe gaze, head movement, or facial identity." in mouth
    with pytest.raises(ValueError):
        build_prompt("eye", "LEFT")
    with pytest.raises(ValueError):
        build_prompt("hand", "LEFTY")


def test_closed_field_sets():
    assert HAND_FIELDS["handshape_class"].count("uncertain") == 1
    assert "not_visible" in HAND_FIELDS["visibility"]
    assert "teeth_both" in MOUTH_FIELDS["teeth_or_tongue"]


def test_parse_label_accepts_valid_json():
    raw = (
        '{"handshape_class":"fist","finger_spread":"together","palm_direction":"up",'
        '"thumb_position":"extended","visibility":"clear"}'
    )
    got = parse_label("hand", raw)
    assert got["handshape_class"] == "fist"


def test_parse_label_strips_fences_and_prose():
    raw = (
        'Sure! ```json\n{"openness":"open","lip_configuration":"neutral",'
        '"teeth_or_tongue":"tongue","cheek_state":"neutral","visibility":"clear"}'
        "\n``` hope that helps"
    )
    got = parse_label("mouth", raw)
    assert got["openness"] == "open"


@pytest.mark.parametrize(
    "kind,raw,error",
    [
        (
            "hand",
            '{"handshape_class":"pointy","finger_spread":"spread","palm_direction":"up","thumb_position":"tucked","visibility":"clear"}',
            "handshape_class",
        ),
        ("hand", '{"handshape_class":"fist"}', "missing"),
        (
            "hand",
            '{"openness":"open","lip_configuration":"neutral","teeth_or_tongue":"none","cheek_state":"neutral","visibility":"clear"}',
            "unknown",
        ),
        ("mouth", "not json at all", "json"),
    ],
)
def test_parse_label_rejects_invalid(kind, raw, error):
    with pytest.raises(ValueError, match=error):
        parse_label(kind, raw)


def test_all_uncertain_visible_crop_is_accepted_and_masked_downstream():
    raw = (
        '{"handshape_class":"uncertain","finger_spread":"uncertain",'
        '"palm_direction":"uncertain","thumb_position":"uncertain","visibility":"clear"}'
    )
    got = parse_label("hand", raw)
    assert got["handshape_class"] == "uncertain"


def test_not_visible_forces_uncertain_only():
    raw = (
        '{"handshape_class":"uncertain","finger_spread":"uncertain",'
        '"palm_direction":"uncertain","thumb_position":"uncertain",'
        '"visibility":"not_visible"}'
    )
    got = parse_label("hand", raw)
    assert got["visibility"] == "not_visible"
    raw_bad = raw.replace('"visibility":"not_visible"', '"visibility":"not_visible"').replace(
        '"handshape_class":"uncertain"', '"handshape_class":"fist"'
    )
    with pytest.raises(ValueError, match="not_visible"):
        parse_label("hand", raw_bad)


def test_version_manifest_pins_everything():
    manifest = version_manifest(
        model="qwen3-vl:8b",
        model_digest="sha256:abc",
        code_sha256="deadbeef",
    )
    assert manifest["prompt_version"] == "h2s-crop-closed-v1"
    assert manifest["model"] == "qwen3-vl:8b"
    assert manifest["model_digest"] == "sha256:abc"
    assert manifest["options"] == {"temperature": 0.0, "seed": 0, "num_ctx": 4096, "num_gpu": 99}
    assert set(manifest["schema_hash"]) >= {"hand", "mouth"}
    assert len(manifest["prompt_hash"]) == 2
