# Seven-Factor Dual-Path Supervision Design

**Created:** 2026-09-22

**Revised:** 2026-09-23

**Status:** Written spec approved; second self-review completed

**Depends on:** Verified PHOENIX14T baseline and DINOv3 spatial-feature milestone

## 1. Purpose

Create reproducible supervision for seven visual factors using local Ollama `qwen3-vl:8b`:

- Suppress four signer or context nuisances: biometrics, clothing, hair, and background.
- Preserve three linguistic articulators: anatomical left hand, anatomical right hand, and mouth posture.

The four nuisance factors use independent gradient-reversal branches. The three articulator factors use independent positive-alignment branches without gradient reversal. Controlled ablations measure every factor separately before grouped or full training.

## 2. DIFFER Adaptation

DIFFER positively aligns biometric captions because person re-identification requires identity information and adversarially removes non-biometric nuisances. DeSpaMo has a different objective: sign translation should preserve linguistic content while reducing signer dependence.

DeSpaMo therefore adapts the concept rather than copying it:

| Factor | Granularity | Training role | Reason |
|---|---|---|---|
| Biometrics | Clip | GRL suppression | Apparent signer traits can reveal identity. |
| Clothing | Clip | GRL suppression | Clothing can correlate with signer or recording. |
| Hair | Clip | GRL suppression | Hairstyle can correlate with signer. |
| Background | Clip | GRL suppression | Scene can correlate with signer or split. |
| Left hand | Sampled frame | Positive alignment | Hand configuration carries lexical content. |
| Right hand | Sampled frame | Positive alignment | Hand configuration carries lexical content. |
| Mouth posture | Sampled frame | Positive alignment | Mouthings and non-manual cues can carry linguistic content. |

No articulator target passes through a GRL. No biometric target receives positive alignment into the shared translation representation.

## 3. Scope

This design covers:

- Deterministic frame sampling and temporal correspondence.
- Structured Qwen3-VL output and validation for seven factors.
- Canonical factor text and frozen CLIP text embeddings.
- Four independent nuisance GRL heads.
- Three independent positive articulator heads.
- Sensitive-metadata handling, caption audits, probes, and controlled ablations.

This design does not cover baseline SpaMo reproduction, DINOv3 extraction, or final multi-dataset evaluation. Those remain separate milestones.

## 4. Factor Boundaries

### 4.1 Stable nuisance factors

- Biometrics: apparent age band, gender presentation, height category, and build category.
- Clothing: garments, garment colors, patterns, and visible accessories.
- Hair: hair color, length, and style.
- Background: scene type, dominant colors, static objects, and lighting.

Biometric fields describe apparent visual presentation, not verified personal attributes. They must exclude identity, name, ethnicity, race, nationality, skin tone, disability, and health status. Biometric output uses controlled fields rather than free prose.

### 4.2 Per-frame articulator factors

- Left hand: finger configuration, palm orientation, body-relative location, contact state, and visibility.
- Right hand: same fields as left hand.
- Mouth posture: openness, lip configuration, teeth or tongue visibility, and overall visibility.

`left` and `right` always mean the signer's anatomical sides, never viewer-image sides. The prompt repeats this convention and the audit scores side assignment separately.

Articulator fields describe visible form only. They must not contain a sign gloss, inferred word, sentence meaning, action label, motion trajectory, signing speed, facial emotion, gaze, or head movement. Skin color, hand size, scars, jewelry, and other signer-identifying details are also excluded from articulator fields.

Controlled values are fixed by schema version:

- `apparent_age_band`: `young_adult`, `middle_aged_adult`, `older_adult`, `uncertain`.
- `gender_presentation`: `masculine`, `feminine`, `androgynous`, `uncertain`.
- `apparent_height`: `short`, `average`, `tall`, `uncertain`.
- `apparent_build`: `slim`, `average`, `broad`, `uncertain`.
- Hand `contact`: `none`, `body`, `other_hand`, `object`, `uncertain`.
- Mouth `openness`: `closed`, `slightly_open`, `open`, `wide_open`, `uncertain`.
- Mouth `lip_configuration`: `neutral`, `rounded`, `spread`, `pursed`, `other`, `uncertain`.
- Mouth `teeth_or_tongue_visibility`: `none`, `teeth`, `tongue`, `both`, `uncertain`.

Hand finger configuration, palm orientation, and body-relative location remain short structured text because a fixed sign-language handshape taxonomy is outside this project's scope.

### 4.3 Globally forbidden content

- Person identity or name.
- Ethnicity, race, nationality, or skin tone.
- Readable text, subtitles, signs, or screen contents.
- Inferred translation, sign gloss, action, or activity.
- Unsupported biometric claims.
- Medical, disability, or health inference.

## 5. Frame Sampling

Each clip uses frames nearest normalized timestamps 10%, 30%, 50%, 70%, and 90%. One Qwen request receives all five frames.

The response contains:

- One stable clip-level record for biometrics, clothing, hair, and background.
- Five ordered frame records for left hand, right hand, and mouth posture.

Every frame record stores its normalized position and zero-based source-frame ordinal in the sorted source sequence. Trusted provenance stores paths and file hashes outside Qwen output. Use round-half-up, `int((N - 1) * position + 0.5)`, and require five distinct indices. Missing frames produce an explicit failed clip record; the sampler never silently substitutes an untracked frame.

DINOv3 extraction must supply one source-index-to-feature-row mapping per clip, tied to its feature-file hash. Caption source indices are looked up in that mapping, never assumed equal to feature rows. Require every requested index exactly once and reject padding or absent rows. The initial factor stage uses full, unflipped spatial sequences; temporal subsampling or horizontal flips require synchronized target transforms in a separate change.

## 6. Structured Output

Schema version 2 has the following illustrative single-frame excerpt. This excerpt is not a valid complete generation response: the validator requires exactly five frame objects at the requested indices.

```json
{
  "schema_version": 2,
  "clip_id": "example-clip",
  "stable": {
    "biometric": {
      "apparent_age_band": "young_adult",
      "gender_presentation": "masculine",
      "apparent_height": "uncertain",
      "apparent_build": "average",
      "visibility": "partial"
    },
    "clothing": {
      "upper_garment": "long-sleeve shirt",
      "upper_color": "black",
      "lower_garment": null,
      "lower_color": null,
      "pattern": "solid",
      "accessories": [],
      "visibility": "clear"
    },
    "hair": {
      "color": "dark brown",
      "length": "short",
      "style": "straight",
      "visibility": "partial"
    },
    "background": {
      "scene": "indoor studio",
      "dominant_colors": ["blue", "gray"],
      "static_objects": ["plain backdrop"],
      "lighting": "uniform",
      "visibility": "clear"
    }
  },
  "frames": [
    {
      "normalized_position": 0.1,
      "source_frame_index": 12,
      "left_hand": {
        "finger_configuration": "open fingers",
        "palm_orientation": "toward signer",
        "body_relative_location": "upper torso",
        "contact": "none",
        "visibility": "clear"
      },
      "right_hand": {
        "finger_configuration": null,
        "palm_orientation": null,
        "body_relative_location": null,
        "contact": null,
        "visibility": "not_visible"
      },
      "mouth": {
        "openness": "slightly_open",
        "lip_configuration": "rounded",
        "teeth_or_tongue_visibility": "none",
        "visibility": "clear"
      }
    }
  ]
}
```

All five frame objects are required and must appear in sampled-frame order.

Allowed visibility values:

- `clear`
- `partial`
- `not_visible`
- `uncertain`

Rules:

- Unknown open-text scalar values use JSON `null`, never guessed values.
- Empty collections use `[]`.
- Controlled biometric and mouth categories include an explicit `uncertain` value.
- Nullable controlled fields use `null` when not visible; `uncertain` means visible but indeterminate. Both are omitted from canonical text. An all-unknown factor is masked, including a biometric factor with four `uncertain` fields.
- `not_visible` fields contain no descriptive attributes.
- `clear` and `partial` fields contain at least one descriptive attribute.
- `not_visible` and `uncertain` targets are excluded from that factor's loss.
- `partial` targets remain eligible but are reported separately.
- Numeric confidence is not requested because VLM self-confidence is not calibrated.

## 7. Generation, Provenance, And Versioning

Generation uses a configurable local Ollama endpoint with model tag `qwen3-vl:8b`. Every terminal clip record stores:

- Ollama endpoint identifier without credentials.
- Model tag and immutable model digest.
- Prompt text, prompt version, and prompt hash.
- JSON schema version and schema hash.
- Sampling positions and sampling hash.
- Source split and immutable sampled-frame manifest hash.
- Generation options.
- Source-frame indices and paths.
- Raw response from every attempt.
- Validated structured record.
- Validation errors, retry count, and terminal status.

Validation sequence:

1. Parse response as JSON.
2. Reject unknown fields or factors.
3. Require exact clip ID, frame count, frame order, normalized positions, and source indices.
4. Validate field types, controlled categories, and visibility consistency.
5. Reject forbidden content; anatomical-side correctness is enforced by prompt wording and measured by audit.
6. Retry invalid responses at most twice with validation errors in the correction prompt.
7. Store unresolved failures and mask all unavailable targets.

Version identifiers are deterministic:

```text
dataset_key = sha256(model_digest, prompt_hash, schema_hash, sampling_hash,
                     source_split, sampled_frame_manifest_hash, generation_options)
cache_key   = sha256(dataset_key, clip_id)
```

The sampled-frame manifest contains ordered clip IDs, source-frame indices, and source-image SHA-256 values. Existing terminal records are not regenerated for the same cache key. Prompt, model, schema, sampling, source split, or source-image changes create a new dataset key. Downstream audit and text-feature jobs require an explicit dataset key so old and new versions cannot mix.

Hash canonical JSON with sorted keys; exclude machine-specific absolute roots. Include temperature and seed in generation options. Serialize one writer per dataset. Persist each attempt before another request; an interrupted in-flight attempt consumes its reserved slot. Never exceed three requests per clip/version across restarts. Transport failures stop generation and leave an explicit resumable interrupted attempt, never a successful empty record. Retain raw response before validation. Endpoint provenance strips credentials, query strings, and fragments.

## 8. Canonical Descriptions And Text Features

Validated fields become deterministic factor strings. Raw Qwen prose is never embedded directly.

Examples:

```text
biometric: apparent age band young adult; gender presentation masculine; apparent build average
clothing: upper garment long-sleeve shirt; upper color black; pattern solid
hair: color dark brown; length short; style straight
background: scene indoor studio; dominant colors blue, gray; static objects plain backdrop; lighting uniform
left hand: finger configuration open fingers; palm orientation toward signer; location upper torso; contact none
mouth: openness slightly open; lip configuration rounded; teeth or tongue visibility none
```

Field order, separators, lowercase normalization, category rendering, and null omission are fixed by schema version.

Empty lists mean unknown and are omitted, never converted to an asserted absence. Explicit controlled `none` values (for example, hand contact) remain valid observations. `not_visible` and `uncertain` factors have no canonical text or embedding; the absent right hand above is masked. Sort and deduplicate unordered color/object/accessory lists. Canonicalization normalizes formatting, not semantic synonyms. Reject CLIP token-length overflow instead of silently truncating a target.

Canonical descriptions are encoded offline with frozen `openai/clip-vit-large-patch14` text features. Stable factors produce four embeddings per clip. Articulators produce three embeddings per valid sampled frame. Stored metadata includes dataset key, clip ID, source-frame index where applicable, factor, canonical text, text encoder and tokenizer revisions, width, dtype, schema version, and source-record hash.

Ollama and the CLIP text encoder are absent from model inference.

## 9. Model Architecture

```text
projected DINOv3 frame features
  |-- masked clip mean --+-- GRL -> biometric head -> loss
  |                     +-- GRL -> clothing head -> loss
  |                     +-- GRL -> hair head -> loss
  |                     +-- GRL -> background head -> loss
  |-- sampled rows -----+-- left-hand head -> loss
  |                     +-- right-hand head -> loss
  |                     +-- mouth head -> loss
  +-- fusion with projected VideoMAE -> temporal adapter -> Flan-T5-XL
```

The four nuisance heads consume the masked temporal mean. The three articulator heads consume the projected feature at each sampled source-frame index. All seven heads are independent and map into the frozen CLIP text-feature width.

Only nuisance branches contain GRLs. Articulator branches propagate ordinary positive gradients into the shared spatial projector. Head outputs never become Flan-T5 inputs.

Training checkpoints contain all seven heads for reproducibility. Explicit inference export removes every auxiliary head and GRL, records source-checkpoint hash, and verifies strict loading of every shared translation tensor.

## 10. Losses And Schedule

```text
L_total = L_translation + lambda_vt * L_vt
        + lambda_bio * L_bio(GRL(z_clip))
        + lambda_cloth * L_cloth(GRL(z_clip))
        + lambda_hair * L_hair(GRL(z_clip))
        + lambda_bg * L_bg(GRL(z_clip))
        + lambda_left * L_left(z_frame)
        + lambda_right * L_right(z_frame)
        + lambda_mouth * L_mouth(z_frame)
```

Each factor uses symmetric multi-positive contrastive loss:

- Samples with identical canonical factor strings are positives.
- Other valid factor strings are negatives.
- Invalid, `not_visible`, and `uncertain` targets are masked.
- A factor loss is skipped when a batch has fewer than two valid targets.
- Also skip when fewer than two distinct canonical targets remain; an all-positive contrastive batch has no discrimination signal.
- Clip-level and frame-level targets use separate masks and denominators.

Use cosine logits with fixed temperature 0.07 and a uniform positive distribution over exact canonical matches in both directions. Nuisance losses operate on B clip rows; articulator losses operate on valid B*5 frame rows. Initial configured weights are 1.0. Initial implementation uses one device, without cross-rank negatives; multi-device behavior is a separately verified extension.

Schedule:

1. Complete VT-Align warm-up.
2. Enable all heads with nuisance GRL coefficient 0 so nuisance heads can learn targets without changing the shared projector.
3. Ramp nuisance GRL coefficient from 0 to 1 over the first 10% of joint-training optimizer steps.
4. Ramp articulator loss weights from 0 to configured values over the same interval.
5. Tune factor weights only after single-factor runs expose scales and stability.

Match baseline warm-up boundary exactly: `global_step <= warm_up_steps` is warm-up. The first joint optimizer step has ramp 0; use `ceil(0.1 * joint_optimizer_steps)` as ramp duration. Resume schedules from Lightning optimizer-step state, not minibatch count. A disabled head receives no loss or parameter update.

All seven losses remain positive. GRL alone reverses nuisance gradients into the shared spatial projector.

## 11. Quality, Safety, And Audit

Biometric captions are inferred appearance metadata, not ground truth. They remain outside Git with generated research data. Captions do not contain names or person-identity labels. Existing opaque dataset signer keys remain in private manifests for stratification and signer probes; reports contain aggregate statistics.

Before factor training:

- Compute post-retry schema-valid rate over the complete training split.
- Audit 100 valid clips stratified across signers and sequence-length quartiles for the four stable factors.
- Audit 100 valid frame entries stratified across signers, sequence-length quartiles, and five sampled positions for the three articulator factors.

Entry criteria:

- At least 99% schema-valid terminal records.
- At least 85% correct visible descriptions for each factor.
- At least 90% correct visibility labels.
- At least 95% correct anatomical left/right assignment.
- At most 5% unsupported biometric inference.
- Zero identity, ethnicity, race, nationality, skin-tone, medical, sign-gloss, or readable-text leakage.
- At least 1,000 valid training targets per stable factor and per articulator factor; lower coverage disables that factor rather than weakening masks.

Biometric review asks whether a field is visibly supported, not whether it is objectively true. If a factor fails, revise prompt or schema and regenerate that dataset version under a new dataset key. Do not patch generated records manually.

Audit identities and source-record hashes must match a deterministic, immutable selection manifest; reject duplicate, missing, foreign-version, or stale rows. Human reviewers mark actual visibility independently of model visibility. A visible cue omitted by the model counts as incorrect, preventing abstention from inflating accuracy. Null denominators fail the gate. Count biometric unsupported inference per audited biometric record and side correctness per auditable anatomical-hand assignment. Lexical validation is only a filter; it cannot establish semantic correctness or zero leakage. Both manual audits are required.

The coverage floor is measured over unique clip/factor targets or unique clip/frame/factor targets. Report coverage and distinct canonical counts. Low coverage disables the affected factor; it does not authorize guessed labels or silently relabel a six-factor experiment as the full model. Dataset-wide schema/leakage failures block all factor training.

## 12. Effect Measurement

Use identical data, optimizer, DINOv3 features, and three-seed set for 11 variants:

| Variant | Active supervision |
|---|---|
| DINO baseline | None |
| Biometrics only | Biometric GRL |
| Clothing only | Clothing GRL |
| Hair only | Hair GRL |
| Background only | Background GRL |
| Left hand only | Left-hand positive alignment |
| Right hand only | Right-hand positive alignment |
| Mouth only | Mouth positive alignment |
| All nuisances | Four GRL factors |
| All articulators | Three positive factors |
| Full | All seven factors |

Report mean and standard deviation for:

- BLEU-4 and ROUGE-L.
- Seen-to-unseen signer BLEU-4 drop.
- Fresh signer-probe top-1 accuracy on frozen shared spatial representations.
- Fresh factor-probe Recall@1, Recall@5, and matched-pair cosine similarity for all seven factors.

Fresh probes are trained after DeSpaMo training. Adversarial training heads are never reused as probes.

First run fresh probes on the frozen DINO baseline as the pre-training probe gate: finite fits, nonempty held-out partitions, and complete aggregate artifacts. No accuracy threshold is invented. Factor probes are fresh linear projections; signer probes are fresh linear classifiers. Fit only on training clips, select hyperparameters on a fixed validation partition, and evaluate on disjoint held-out clips. Never split frames from the same clip across partitions. A closed-set signer classifier requires the same signer classes in its training and evaluation sets; unseen-signer translation is a separate evaluation. Use one versioned split manifest and one unique-canonical retrieval gallery shared by all variants. Report Recall@min(5, gallery_size) with gallery size, and log unavailable metrics instead of fabricating zero.

Use seeds 0, 1, and 2. Prespecify Recall@1 for factor retention and signer top-1 for identity leakage; compare paired seed means in BLEU-4 points, not percentages. ROUGE-L keeps baseline fraction units. The unseen-signer BLEU gap is unavailable until a signer-disjoint translation split is supplied; the official PHOENIX split alone is not proof of unseen-signer evaluation. Experiment code prepares 33 runs but launching them requires explicit experiment authorization.

For the matched factor experiment, all 11 variants start from the same accepted shared-only DINO checkpoint, with seed-controlled data order and identical additional optimizer-step budgets. Retrain the zero-auxiliary control for each seed; do not substitute the earlier DINO milestone score. Bind the pre-training probe gate to that common initial checkpoint hash.

Desired directions:

- Nuisance GRL: lower matching nuisance-probe and signer-probe performance.
- Articulator alignment: higher matching articulator-probe performance without increased signer leakage.
- Every factor: non-negative or small translation effect.

Retention criteria:

- Keep a nuisance factor only when matching factor leakage or signer leakage decreases and mean BLEU-4 loss is at most 0.5 points.
- Keep an articulator factor only when its fresh probe improves, signer leakage does not increase, and mean BLEU-4 loss is at most 0.5 points.
- Compare full model against both DINO baseline and all-nuisance model to isolate articulator recovery.

## 13. Failure Handling

- Ollama unavailable: stop without creating empty successful records.
- Invalid JSON or schema mismatch: retain every raw response and retry at most twice.
- Source-frame mismatch: reject response; never align to a different frame silently.
- Missing sampled frame: store explicit clip failure.
- Forbidden sensitive or linguistic content: reject response and include exact field path in correction prompt.
- Text-feature mismatch: fail on dataset key, clip ID, frame index, factor, width, model revision, or schema version.
- Empty valid subset: skip only that factor loss and log valid count.
- Non-finite factor loss: stop training and report factor, clip IDs, and frame indices.
- Failed quality gate: reject dataset key before training.
- Inference export mismatch: reject export if any shared tensor is missing, unexpected, or shape-incompatible.

## 14. Testing

Unit tests cover:

- Frame-index selection and source-index round trip.
- Version-2 JSON parsing, strict types, field rejection, and five-frame ordering.
- Anatomical-side convention in prompt and canonical output.
- Visibility consistency and frame-level masks.
- Forbidden sensitive and linguistic content.
- Canonical ordering and normalization for all seven factors.
- Dataset and cache key changes for model, prompt, schema, and sampling changes.
- Version-isolated record iteration and text manifests.
- Multi-positive target construction at clip and frame granularity.
- Negative shared-projector gradient for each nuisance GRL.
- Positive shared-projector gradient for each articulator head.
- Factor-loss masking, skip behavior, and independent enablement.

Integration tests cover:

- One local Ollama request with five fixture frames.
- Raw-to-validated-to-canonical pipeline for stable and per-frame factors.
- CLIP text-feature cache round trip under one dataset key.
- One batch forward/backward for every single factor, grouped factors, and full model.
- Exact sampled-frame feature alignment.
- Checkpoint round trip and strict inference export without auxiliary modules.

## 15. Implementation Boundary

Execution starts only after:

1. PHOENIX14T baseline acceptance passes.
2. DINOv3 extraction and baseline comparison pass.

Implementation remains two sequential plans:

1. Qwen3-VL seven-factor dataset, audit, and frozen CLIP text features.
2. Four GRL heads, three positive articulator heads, probes, and 11-variant effect experiments.

The existing three-factor Qwen plan is historical and must not be executed. Its replacements are `docs/superpowers/plans/2026-09-23-seven-factor-dataset.md` and `docs/superpowers/plans/2026-09-23-dual-path-training.md`.
