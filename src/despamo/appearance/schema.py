"""Strict, visible-only schema for seven appearance factors."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
STABLE = ("biometric", "clothing", "hair", "background")
ARTICULATORS = ("left_hand", "right_hand", "mouth")
FACTORS = STABLE + ARTICULATORS
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
Visibility = Literal["clear", "partial", "not_visible", "uncertain"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Factor(Strict):
    visibility: Visibility

    @model_validator(mode="after")
    def check_visibility(self):
        known = any(
            value not in (None, [], "uncertain")
            for value in self.model_dump(exclude={"visibility"}).values()
        )
        if self.visibility == "not_visible" and known:
            raise ValueError("not_visible factor contains attributes")
        if self.visibility in {"clear", "partial"} and not known:
            raise ValueError("visible factor has no known attributes")
        return self


class Biometric(Factor):
    apparent_age_band: (
        Literal["young_adult", "middle_aged_adult", "older_adult", "uncertain"] | None
    )
    gender_presentation: Literal["masculine", "feminine", "androgynous", "uncertain"] | None
    apparent_height: Literal["short", "average", "tall", "uncertain"] | None
    apparent_build: Literal["slim", "average", "broad", "uncertain"] | None


class Clothing(Factor):
    upper_garment: Text | None
    upper_color: Text | None
    lower_garment: Text | None
    lower_color: Text | None
    pattern: Text | None
    accessories: list[Text]


class Hair(Factor):
    color: Text | None
    length: Text | None
    style: Text | None


class Background(Factor):
    scene: Text | None
    dominant_colors: list[Text]
    static_objects: list[Text]
    lighting: Text | None


class Hand(Factor):
    finger_configuration: Text | None
    palm_orientation: Text | None
    body_relative_location: Text | None
    contact: Literal["none", "body", "other_hand", "object", "uncertain"] | None


class Mouth(Factor):
    openness: Literal["closed", "slightly_open", "open", "wide_open", "uncertain"] | None
    lip_configuration: (
        Literal["neutral", "rounded", "spread", "pursed", "other", "uncertain"] | None
    )
    teeth_or_tongue_visibility: Literal["none", "teeth", "tongue", "both", "uncertain"] | None


class Stable(Strict):
    biometric: Biometric
    clothing: Clothing
    hair: Hair
    background: Background


class Frame(Strict):
    normalized_position: float
    source_frame_index: Annotated[int, Field(ge=0)]
    left_hand: Hand
    right_hand: Hand
    mouth: Mouth


class Record(Strict):
    schema_version: Literal[2]
    clip_id: Text
    stable: Stable
    frames: Annotated[list[Frame], Field(min_length=5, max_length=5)]

    @model_validator(mode="after")
    def ordered_frames(self):
        if tuple(frame.normalized_position for frame in self.frames) != POSITIONS:
            raise ValueError("frame positions/order mismatch")
        indices = [frame.source_frame_index for frame in self.frames]
        if indices != sorted(set(indices)):
            raise ValueError("source indices must be distinct and increasing")
        return self
