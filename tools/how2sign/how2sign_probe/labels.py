"""Closed-vocabulary crop prompts and strict parsers from the 2026-10-01 design.

Prompts reproduce the design document's Stage-1 text verbatim except for the
``{LEFT|RIGHT}`` slot, which the probe fills with the anatomical side.
"""

import hashlib
import json
import re

from how2sign_probe.schemas import HAND_FIELDS, HAND_SCHEMA, MOUTH_FIELDS, MOUTH_SCHEMA, schema_json

PROMPT_VERSION = "h2s-crop-closed-v1"

HAND_PROMPT = """You are shown a CROPPED, MAGNIFIED image of ONE hand of a sign language \
signer. The crop may be blurry because the source video is low resolution.

This is the signer's {SIDE} hand (their own anatomical side, NOT \
the side of the image).

Describe ONLY the visible physical form of this hand. Choose exactly one \
option from each list. Use "uncertain" whenever the crop is too blurry \
or occluded to tell.

handshape_class: flat_open | flat_closed | fist | index_point | two_fingers
               | three_fingers | four_fingers | curved_claw | pinch_thumb_index
               | cupped | thumb_out | other | uncertain

finger_spread: spread | together | mixed | not_applicable | uncertain

palm_direction: toward_signer | away_from_signer | up | down
              | toward_other_hand | sideways | uncertain

thumb_position: tucked | extended | across_palm | opposed | uncertain

visibility: clear | partial | not_visible | uncertain

Rules:
- Describe form only. Do NOT name the sign, guess the word or meaning,
  or describe motion, speed, or direction of movement.
- Do NOT mention skin tone, jewelry, scars, nails, or anything identifying
  the person.
- If you cannot see a hand in this crop, set visibility=not_visible and
  every other field to uncertain.

Return ONLY this JSON:
{"handshape_class":"...","finger_spread":"...","palm_direction":"...",
 "thumb_position":"...","visibility":"..."}"""

MOUTH_PROMPT = """You are shown a CROPPED, MAGNIFIED image of the mouth region of a sign \
language signer. The crop may be blurry.

Describe ONLY the visible mouth shape. Choose exactly one option per list.
Use "uncertain" when the crop is too blurry or occluded.

openness: closed | slightly_open | open | wide_open | uncertain

lip_configuration: neutral | rounded | spread_wide | pursed | protruded
                 | corners_down | corners_up | asymmetric | other | uncertain

teeth_or_tongue: none | teeth_upper | teeth_both | tongue | both | uncertain

cheek_state: neutral | puffed | sucked_in | uncertain

visibility: clear | partial | not_visible | uncertain

Rules:
- Form only. Do NOT guess the spoken word, mouthing, emotion, or meaning.
- Do NOT describe gaze, head movement, or facial identity.
- No identifying detail (facial hair, moles, makeup, skin tone).

Return ONLY this JSON:
{"openness":"...","lip_configuration":"...","teeth_or_tongue":"...",
 "cheek_state":"...","visibility":"..."}"""

PROMPTS = {
    "hand": (HAND_PROMPT, ("LEFT", "RIGHT")),
    "mouth": (MOUTH_PROMPT, ("MOUTH",)),
}

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(kind: str, side: str) -> str:
    if kind not in PROMPTS:
        raise ValueError(f"unknown crop kind: {kind}")
    template, sides = PROMPTS[kind]
    if side not in sides:
        raise ValueError(f"side {side!r} invalid for {kind}")
    return template.replace("{SIDE}", side)


def _extract_json(raw: str) -> dict:
    text = str(raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no json found in model reply")
    return json.loads(match.group(0))


def parse_label(kind: str, raw: str) -> dict:
    if kind not in PROMPTS:
        raise ValueError(f"unknown crop kind: {kind}")
    fields = HAND_FIELDS if kind == "hand" else MOUTH_FIELDS
    try:
        payload = _extract_json(raw)
    except json.JSONDecodeError:
        raise ValueError("invalid json object in model reply") from None
    if not isinstance(payload, dict):
        raise ValueError("model reply is not a JSON object")
    if set(payload) != set(fields):
        unknown = set(payload) - set(fields)
        missing = set(fields) - set(payload)
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        raise ValueError(f"missing fields: {sorted(missing)}")
    for name, allowed in fields.items():
        if payload[name] not in allowed:
            raise ValueError(f"field {name} value not in closed vocabulary")
    if payload["visibility"] == "not_visible":
        for name, value in payload.items():
            if name != "visibility" and value != "uncertain":
                raise ValueError("not_visible forces every other field uncertain")
    return dict(payload)


def _sha256(value) -> str:
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 4096, "num_gpu": 99}


def version_manifest(model: str, model_digest: str, code_sha256: str) -> dict:
    return {
        "model": model,
        "model_digest": model_digest,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": {
            "hand": _sha256(PROMPTS["hand"][0]),
            "mouth": _sha256(PROMPTS["mouth"][0]),
        },
        "schema_hash": {
            "hand": _sha256(schema_json(HAND_SCHEMA)),
            "mouth": _sha256(schema_json(MOUTH_SCHEMA)),
        },
        "options": dict(OPTIONS),
        "code_sha256": code_sha256,
    }
