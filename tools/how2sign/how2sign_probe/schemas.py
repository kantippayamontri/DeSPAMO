"""Closed vocabularies for crop labels, mirroring the Stage-1 prompts exactly."""

import json

HAND_FIELDS = {
    "handshape_class": (
        "flat_open",
        "flat_closed",
        "fist",
        "index_point",
        "two_fingers",
        "three_fingers",
        "four_fingers",
        "curved_claw",
        "pinch_thumb_index",
        "cupped",
        "thumb_out",
        "other",
        "uncertain",
    ),
    "finger_spread": ("spread", "together", "mixed", "not_applicable", "uncertain"),
    "palm_direction": (
        "toward_signer",
        "away_from_signer",
        "up",
        "down",
        "toward_other_hand",
        "sideways",
        "uncertain",
    ),
    "thumb_position": ("tucked", "extended", "across_palm", "opposed", "uncertain"),
    "visibility": ("clear", "partial", "not_visible", "uncertain"),
}

MOUTH_FIELDS = {
    "openness": ("closed", "slightly_open", "open", "wide_open", "uncertain"),
    "lip_configuration": (
        "neutral",
        "rounded",
        "spread_wide",
        "pursed",
        "protruded",
        "corners_down",
        "corners_up",
        "asymmetric",
        "other",
        "uncertain",
    ),
    "teeth_or_tongue": ("none", "teeth_upper", "teeth_both", "tongue", "both", "uncertain"),
    "cheek_state": ("neutral", "puffed", "sucked_in", "uncertain"),
    "visibility": ("clear", "partial", "not_visible", "uncertain"),
}


def _schema(fields: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "enum": list(values)} for name, values in fields.items()
        },
        "required": list(fields),
        "additionalProperties": False,
    }


HAND_SCHEMA = _schema(HAND_FIELDS)
MOUTH_SCHEMA = _schema(MOUTH_FIELDS)


def schema_json(schema: dict) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))
