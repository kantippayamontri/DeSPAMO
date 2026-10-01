# Cropped Articulator Supervision Design

## Purpose and scope

Investigate whether region-cropped, magnified articulator supervision produces more
usable hand and mouth targets than the current whole-frame Qwen3-VL extraction. The
hypothesis has two independent parts, and this design keeps them separable because
they are separately testable and have very different costs:

1. **Resolution**: PHOENIX frames are 210x260. A signing hand occupies roughly 57x57
   pixels and a mouth roughly 40x40. Cropping and upscaling the region to the
   encoder's 224 input gives about 3.9x magnification for hands and 5.5x for mouths.
2. **Label space**: the current hand targets are free text and therefore nearly unique
   per row, which prevents contrastive positives from ever grouping.

This design covers a bounded feasibility probe and the full re-extraction it would
justify. It does **not** authorize the full job, does not modify any frozen dataset,
and does not alter the running E1/E2/E3 comparison.

## Measured motivation

Label granularity over the frozen train manifest (5 sampled frames per clip):

| Factor | distinct values | rows | top-1 share | singletons |
|---|---:|---:|---:|---:|
| left_hand | **3,787** | 34,782 | 0.024 | 1,363 |
| right_hand | **3,613** | 34,511 | 0.026 | 1,246 |
| mouth | **53** | 34,885 | **0.344** | 6 |

The two articulator families fail in opposite directions, and the cause is a schema
decision rather than a model limitation. In `src/despamo/appearance/schema.py`:

- `Mouth.openness`, `lip_configuration`, `teeth_or_tongue_visibility` are `Literal`
  closed sets, which is why mouth has only 53 distinct canonical strings.
- `Hand.finger_configuration`, `palm_orientation`, `body_relative_location` are
  `Text = Annotated[str, StringConstraints(..., max_length=80)]`, i.e. free prose.

Free-text hand fields are why 1,363 hand strings occur exactly once. For the
contrastive articulator heads this is worse than coarse labelling: `_contrast` in
`tools/dinov3/e3_model.py` requires at least two positions sharing a label before it
returns a non-zero loss, so near-unique strings degenerate the objective toward
instance discrimination over noisy prose. Finer free-text descriptions would make
this worse, not better. Mouth has the opposite problem: 34.4% of rows carry a single
value, so it is genuinely too coarse and is the factor that magnification should help
most directly.

Downstream, the measured articulator probes are weak in absolute terms (left/right
handshape 7-way accuracy 0.430/0.304 on adapted features, 0.360/0.226 on original),
which is consistent with 57x57 pixel evidence but does not by itself prove that
cropping is the remedy.

## Separable experiments

The two parts must not be bundled, because a positive result from a bundled change
cannot be attributed:

- **Experiment L (label space only)**: constrain hand fields to closed vocabularies,
  re-extract from the **same whole frames**. Cheap, no detector, isolates whether
  grouping alone fixes the articulator objective.
- **Experiment C (crop + label space)**: add detector-driven crops on top of L.
  Expensive, and only justified if the probe shows the detector is reliable and the
  crops reduce the `uncertain` rate rather than inflating it.

Experiment L is the cheaper and more likely source of the measurable gain, since it
addresses the mechanism (`_contrast` needs grouping) rather than the input quality.

## Stage 1: bounded feasibility probe

No training, no frozen-artifact writes, strictly additive outputs under a new path.

**Detector reliability is the first unknown and gates everything else.** No detector
is currently installed in either environment (`mediapipe`, `ultralytics`,
`insightface`, `cv2`, `mtcnn` all absent); only `qwen3-vl:8b` is present in Ollama.
The probe therefore begins by measuring detection, not by generating captions.

Sample roughly 200 logical train clips, stratified across the seven training signers,
excluding Signer03 and Signer07 entirely. For each sampled clip use the same five
frozen frame indices already recorded in `source.json`, so crops align exactly with
existing targets.

Report, before any VLM call:

- hand detection rate (expect two hands per frame, but signing frequently moves a
  hand out of frame, so a miss is not automatically an error)
- mouth/face detection rate
- crop bounding-box size distribution in source pixels, and the resulting upscale
  factor to 224
- fraction of crops whose source region is smaller than a stated floor, since a crop
  below roughly 24x24 source pixels carries little recoverable detail

If detection is unreliable at 210x260, Experiment C stops here and only Experiment L
proceeds. This outcome is a legitimate and useful result, not a failure.

Then, for the detected crops only, query the pinned `qwen3-vl:8b` with the prompts
below and report:

- `uncertain` rate per field, which is the key feasibility signal: magnified but
  still unreadable crops will show a high rate
- distinct-value count and top-1 share per field, compared against the 3,787/53
  baselines above
- fresh 7-way handshape probe accuracy on the **existing** frozen DINO features using
  the new labels, which isolates label quality from any encoder change

## Stage 1 prompts

Both prompts use closed vocabularies for every field, state the anatomical-side
convention explicitly because a crop destroys body context, and retain the existing
privacy and no-meaning constraints from `PROMPT` in
`src/despamo/appearance/generation.py`. An explicit `not_visible` escape is mandatory:
hands leave frame constantly, and without it the model will hallucinate a handshape.

Hand crop prompt:

```
You are shown a CROPPED, MAGNIFIED image of ONE hand of a sign language
signer. The crop may be blurry because the source video is low resolution.

This is the signer's {LEFT|RIGHT} hand (their own anatomical side, NOT
the side of the image).

Describe ONLY the visible physical form of this hand. Choose exactly one
option from each list. Use "uncertain" whenever the crop is too blurry
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
 "thumb_position":"...","visibility":"..."}
```

Mouth crop prompt:

```
You are shown a CROPPED, MAGNIFIED image of the mouth region of a sign
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
 "cheek_state":"...","visibility":"..."}
```

The 12-way handshape inventory is a proposal, not a linguistic standard. A
HamNoSys-derived or ASL-handshape inventory would be more defensible and should be
reviewed before any full run; the probe's purpose is to establish whether a closed
set of roughly this size is recoverable from these crops at all.

## Stage 2: full re-extraction, only if the probe justifies it

Scope if authorized: 7,096 clips x 5 frames x 3 regions (two hands, one mouth) is
about 106,000 crops and the same number of VLM calls, plus a detector pass over
35,480 frames. This is a materially larger job than the original extraction and its
cost must be measured from the probe's observed per-call latency, not estimated.

Consequences that must be accepted explicitly before starting:

- A new schema version with closed hand vocabularies, so `schema_hash` changes.
- A new prompt, so `prompt_hash` and `PROMPT_VERSION` change.
- Therefore a **new frozen dataset key**, a new `records_hash` and
  `text_manifest_hash`, and a new frozen protocol with a new `protocol_hash`.
- E1, E2, and E3 would all require re-running against the new protocol for any fair
  comparison. The existing 4,000-step and 8,000-step results remain valid only
  against the current dataset key and must not be mixed with new-key results in one
  table.
- All existing artifacts stay immutable; nothing under the current dataset key,
  feature versions, or run directories is modified or deleted.

Masking contract is unchanged in meaning: a region whose `visibility` is
`not_visible` or whose fields are all `uncertain` yields no canonical target and is
masked, while the clip keeps its translation target. The canonical renderer in
`src/despamo/appearance/canonical.py` already drops `uncertain` and non-visible
factors, so closed vocabularies flow through it without special handling.

Labels remain LLM-generated and unreviewed. Every new run, checkpoint, and result
stays `qwen-schema98-unreviewed-v1` with `human_review_status=not_assessed`, and
schema validity still does not demonstrate visual correctness.

## Success criteria

The probe justifies Stage 2 only if all of the following hold together:

- detector reaches a stated, pre-declared reliability on 210x260 frames
- per-field `uncertain` rate stays below a pre-declared ceiling, so magnified crops
  are actually readable
- hand label top-1 share rises materially above 0.024 and singleton count falls far
  below 1,363, demonstrating that positives can group
- fresh 7-way handshape probe accuracy on **unchanged** frozen DINO features improves
  over the current 0.360/0.226 baseline, isolating label quality from encoder effects

If only the label-space criteria pass and detection is unreliable, run Experiment L
alone, which needs no detector and no crops.

## Risks

- The detector is an unverified new dependency at this resolution and is the single
  largest unknown; it is measured first for that reason.
- Magnification cannot create detail that the 210x260 source never captured, so a
  high `uncertain` rate is a plausible outcome and would correctly stop the work.
- A 12-way closed handshape set may be too coarse to carry lexical content, or too
  fine to group; either shows up in top-1 share and probe accuracy.
- Bundling crop and label changes would make a positive result unattributable, which
  is why the experiments are kept separate.
- Stage 2 invalidates the current comparison baseline, which is a real scientific
  cost and not merely a compute cost.
- Improved probe accuracy is not improved translation. Only a matched end-to-end run
  against the new protocol could establish that, and single-seed deltas below about
  0.3 BLEU-4 will not be reported as improvements.
