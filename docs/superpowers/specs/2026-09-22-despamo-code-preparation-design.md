# DeSpaMo Code Preparation Design

**Date:** 2026-09-22

**Revised:** 2026-09-23 (approved seven-factor design and second self-review)

**Status:** Approved

**Project:** Signer-Invariant Gloss-Free Sign Language Translation

## 1. Purpose

Build DeSpaMo as a standalone, reproducible research codebase that first reproduces the SpaMo PHOENIX14T baseline and then adds DINOv3 spatial features and DIFFER-inspired dual-path factor supervision as independently testable changes.

First working milestone:

1. Run a bounded PHOENIX14T training smoke test.
2. Evaluate the released SpaMo checkpoint over all 642 PHOENIX14T test clips.
3. Record BLEU and ROUGE results with the exact configuration and environment.

## 2. Source Projects

The source repositories are references, not runtime dependencies:

- SpaMo: `/home/kan/Research/SpaMo`
- DIFFER: `/home/kan/Research/DIFFER`

DeSpaMo will port only required behavior. It will not import source modules through `SPAMO_PROJECT_PATH` or `DIFFER_PROJECT_PATH`. Those environment variables document provenance and may support development-only comparison tests.

## 3. Goals

- Reproduce SpaMo architecture and released-checkpoint evaluation before changing research variables.
- Preserve SpaMo's Flan-T5-XL and LoRA setup for PHOENIX14T.
- Make CLIP and DINOv3 offline spatial features interchangeable through configuration.
- Add factor supervision after a trainable spatial projection so nuisance gradient reversal can change the translation representation while positive articulator alignment preserves linguistic cues.
- Keep translation, VT-Align, four nuisance objectives, and three articulator objectives independently configurable for controlled ablations.
- Fail early on corrupt data, incompatible feature dimensions, and partial checkpoint loading.
- Record enough metadata to reproduce every reported experiment.

## 4. Non-Goals

- Reusing the full DIFFER person re-identification model.
- Treating pose or motion as nuisance information.
- Training DINOv3 end to end in the first implementation.
- Supporting every SpaMo dataset before the PHOENIX14T baseline is verified.
- Building a general framework before concrete experiment requirements exist.
- Reproducing known upstream command-line and silent-data-loading defects.

## 5. Package Layout

```text
DeSpaMo/
|-- configs/
|   |-- data/
|   |-- model/
|   |-- experiment/
|   `-- local.example.yaml
|-- scripts/
|   |-- extract_spatial_features.py
|   |-- extract_motion_features.py
|   |-- convert_spamo_checkpoint.py
|   |-- train.py
|   |-- evaluate.py
|   `-- build_appearance_features.py
|-- src/despamo/
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- losses/
|   |-- training/
|   |-- evaluation/
|   `-- utils/
`-- tests/
    |-- unit/
    |-- parity/
    `-- integration/
```

Responsibilities:

- `data`: annotations, feature stores, batching, masks, and clip-level metadata.
- `features`: offline CLIP, DINOv3, and VideoMAE extraction.
- `models`: visual projectors, temporal adapter, Flan-T5 wrapper, GRLs, and factor heads.
- `losses`: translation, VT-Align, nuisance contrastive, and articulator contrastive losses.
- `training`: Lightning module, optimizer, scheduler, checkpoint metadata, and logging.
- `evaluation`: deterministic generation, compatibility generation, BLEU, and ROUGE.

Python 3.11 and `uv` locking will be used initially. SpaMo pins PyTorch 2.0.1, which is not a safe baseline target for the current Python 3.12 scaffold. Dependencies will be locked rather than copied blindly from either source repository.

## 6. Baseline Architecture

PHOENIX14T baseline must retain SpaMo's model choices:

- Spatial encoder: offline CLIP ViT-L/14 with S2 scales 1 and 2.
- Spatial feature shape: `(T, 2048)`.
- Motion encoder: offline VideoMAE-L/16.
- Motion feature shape: `(M, 1024)`.
- Spatial projector: `2048 -> 768`.
- Motion projector: `1024 -> 768`.
- Fusion: concatenate valid spatial and motion tokens along the temporal dimension.
- Temporal adapter: `Conv1d(K5) -> MaxPool(P2) -> Conv1d(K5) -> MaxPool(P2)`.
- Multimodal projector: `Linear -> GELU -> Linear`, producing 2048-dimensional tokens.
- Language model: `google/flan-t5-xl`.
- LLM adaptation: LoRA on attention `q` and `v`, rank 16, alpha 32, dropout 0.1.
- Prompt: `Translate the given sentence into {language}.`

The baseline is a compatibility target. Architectural corrections become explicit experiment settings rather than silent changes.

## 7. DeSpaMo Extension

The first DeSpaMo model changes the spatial feature source and adds dual-path auxiliary supervision. Its spatial encoder is DINOv3 ViT-L/16 using the CLS representation and S2 scales 1 and 2, yielding the same 2048 input width as the CLIP baseline:

```text
DINOv3 frame features -> trainable spatial projector -> projected frame features
                         |                                      |
                         |                              masked temporal pool
                         |                                      |
                         |                       four nuisance GRL heads:
                         |                 biometrics, clothing, hair, background
                         |
                         +-> sampled frame features -> three positive heads:
                                                      left hand, right hand, mouth
                         +------------------------------+
                                                        |
VideoMAE motion projection -----------------------------+
                                                        v
                               temporal concatenation -> TCN
                                                        |
                                      multimodal projector -> Flan-T5-XL
```

All seven factor heads are auxiliary and exist only during training. They do not produce visual tokens consumed by Flan-T5. Gradient reversal makes the shared trainable spatial projection less predictive of apparent biometrics, clothing, hair, and background. Ordinary positive alignment makes sampled frame representations predictive of anatomical left-hand, anatomical right-hand, and mouth posture cues.

Stable nuisance supervision uses separate structured biometric, clothing, hair, and background descriptions. Biometric output is limited to controlled apparent age-band, gender-presentation, height, and build fields; identity, ethnicity, race, nationality, and skin tone are forbidden. Per-frame articulator supervision describes visible hand configuration and mouth posture without inferring gloss, translation, activity, facial emotion, gaze, or motion.

Initial total loss:

```text
L_total = L_translation + lambda_vt * L_vt
        + lambda_bio * L_bio
        + lambda_cloth * L_cloth
        + lambda_hair * L_hair
        + lambda_bg * L_bg
        + lambda_left * L_left
        + lambda_right * L_right
        + lambda_mouth * L_mouth
```

Each nuisance GRL reverses only its factor loss gradient. Articulator losses propagate normal positive gradients. All loss weights remain positive; nuisance losses must not also be negated.

## 8. Data Flow

### Baseline

1. Read PHOENIX14T annotation by clip ID.
2. Resolve CLIP and VideoMAE feature files through manifests.
3. Validate dtype, rank, feature width, non-empty length, and unique clip ID.
4. Batch variable-length tensors and construct masks.
5. Project and fuse visual sequences.
6. Compute VT-Align and Flan-T5 translation losses.
7. Generate translations and calculate SacreBLEU and ROUGE-L.

### DINOv3

1. Extract per-frame DINOv3 features offline.
2. Save feature arrays under an encoder-versioned directory.
3. Write a manifest containing model identifier, revision, preprocessing settings, pooling method, output width, frame count, source clip ID, feature-file hash, and source-frame-index-to-feature-row mapping.
4. Select DINOv3 through experiment configuration without changing dataset or model code.

### Seven-Factor Supervision

1. Generate clip-level biometric, clothing, hair, and background descriptions plus per-frame left-hand, right-hand, and mouth descriptions from five sampled frames using local Ollama model `qwen3-vl:8b`.
2. Store raw captions separately from encoded text features.
3. Record Ollama model digest, prompt/schema/sampling hashes, source split, sampled-image manifest hash, frame indices, and generation settings under an immutable dataset key.
4. Join stable embeddings by clip ID and articulator embeddings by clip ID plus source-frame index.
5. Pool projected frame features for stable nuisance factors.
6. Apply four independent `GRL -> factor projector -> multi-positive contrastive loss` branches.
7. Apply three independent positive `factor projector -> multi-positive contrastive loss` branches at sampled frame positions.

Caption generation runs through a configurable local Ollama endpoint and requires schema-validated stable and per-frame factor objects. Failed or invalid responses receive at most two retries and are otherwise masked. Biometric captions are sensitive inferred metadata, remain outside Git, and appear only as aggregate statistics in reports. Full schema, training, audit, and ablation requirements are defined in `docs/superpowers/specs/2026-09-22-appearance-supervision-design.md`.

## 9. Compatibility Boundaries

Two behavior categories require explicit modes:

- `vt_pooling: legacy_mean` reproduces upstream unmasked mean pooling for training parity.
- `vt_pooling: masked_mean` is the corrected DeSpaMo setting.
- `generation: upstream` preserves beam search with sampling for checkpoint comparison.
- `generation: deterministic` uses beam search without sampling for reportable experiments.

Published DeSpaMo experiments use corrected mask-aware pooling and deterministic generation. Upstream modes exist only for baseline comparison and are labeled in output metadata.

## 10. Configuration And Reproducibility

- Tracked configs contain model and experiment settings but no machine-specific absolute paths.
- Local dataset, cache, and checkpoint paths come from an ignored local config or environment variables.
- Each run stores merged configuration, Git revision, package lock hash, random seed, CUDA/PyTorch versions, feature manifest hashes, checkpoint source, and decoding settings.
- External features, datasets, captions, and checkpoints remain outside Git.
- Checkpoint loading is strict by default. Any intentional key migration must be explicit and logged.
- Released SpaMo weights are converted once into DeSpaMo's namespace. Conversion validates every expected tensor name and shape, rejects extras unless explicitly allowlisted, and stores source-checkpoint SHA-256 in output metadata.

## 11. Failure Handling

- Missing or empty feature files raise a clip-specific error instead of returning empty tensors.
- Width mismatches report expected and observed dimensions before model execution.
- Duplicate or unmatched clip IDs fail manifest validation.
- Invalid sequence lengths fail before entering the TCN.
- Feature extraction is resumable and writes failed clips with reasons to a manifest.
- Corrupt videos are reported, not silently skipped.
- Non-finite losses stop training and log component losses and clip IDs.
- Evaluation refuses partial checkpoint loads and incomplete expected test splits.

## 12. Testing Strategy

### Unit Tests

- Feature-manifest parsing and validation.
- Padding masks and masked temporal pooling.
- Spatial and motion projector shapes.
- TCN output-length calculation.
- Symmetric VT-Align loss.
- GRL forward identity and negative upstream gradient.
- Nuisance head receives ordinary gradients while shared spatial projection receives reversed gradients.
- Articulator head and shared spatial projection both receive ordinary positive gradients.
- Per-frame articulator targets align with exact sampled source-frame features.
- Loss weighting and disabled-loss behavior.

### Parity Tests

- Copy SpaMo projector and TCN weights into DeSpaMo modules.
- Feed identical synthetic variable-length inputs.
- Require matching outputs and sequence lengths within numeric tolerance.
- Verify Flan-T5 input construction and target masking against SpaMo behavior.
- Convert the released SpaMo checkpoint and verify that every required model tensor is mapped exactly once.

### Integration Tests

- Load one real PHOENIX14T batch.
- Run forward and backward passes with finite losses.
- Save and reload a checkpoint, then reproduce outputs.
- Run bounded train and validation smoke tests.
- Evaluate a small deterministic subset twice and require identical generations.

## 13. Baseline Acceptance

Baseline milestone is complete when:

- Environment installs from the lock file.
- Unit, parity, and integration tests pass.
- PHOENIX14T smoke training performs at least one optimizer step and writes a reloadable checkpoint.
- Released SpaMo checkpoint evaluates all 642 PHOENIX14T test clips.
- BLEU-4 and ROUGE-L fall within 1.0 absolute point of the documented released-checkpoint results: BLEU-4 25.08 and ROUGE-L 46.98.
- Evaluation artifact records both upstream-compatible and deterministic decoding results.

Full baseline retraining is not required before beginning the DINOv3 stage.

## 14. Research Sequence

1. Establish repository hygiene, environment lock, configuration system, and test harness.
2. Port and verify PHOENIX14T data loading and metrics.
3. Port SpaMo visual adapter, Flan-T5-XL integration, LoRA, and VT-Align.
4. Complete baseline smoke training and released-checkpoint evaluation.
5. Add versioned DINOv3 offline feature extraction.
6. Run CLIP-versus-DINOv3 baseline under otherwise identical settings.
7. Build versioned Qwen3-VL seven-factor descriptions plus frozen CLIP text features.
8. Add four independent nuisance GRL heads and three positive articulator heads after the spatial projector.
9. Run single-factor, grouped-factor, loss-weight, and GRL-schedule ablations.
10. Add signer-disjoint evaluation and representation leakage probes when signer metadata is available.
11. Extend to CSL-Daily, How2Sign, and FLEURS-ASL only after PHOENIX14T stages are stable.

This document is the master design, not one oversized implementation unit. The first implementation plan covers steps 1-4 through PHOENIX14T baseline acceptance. DINOv3 extraction/comparison, seven-factor caption generation, auxiliary factor training, and signer-generalization evaluation each receive a separate implementation plan after the preceding milestone passes.

## 15. Key Design Rationale

- Minimal port avoids inheriting unrelated SpaMo and DIFFER defects.
- Exact Flan-T5-XL baseline keeps encoder and disentanglement comparisons scientifically meaningful.
- Offline frozen visual features control GPU cost and isolate representation changes.
- GRL after a trainable projector works with offline features; attaching it only to frozen DINOv3 output would not change DINOv3.
- Independent heads make all seven factor effects measurable; single-factor runs precede grouped and full training to control adversarial instability.
- Separating nuisance GRL branches from positive articulator branches avoids suppressing hand and mouth information needed for translation.
- Explicit compatibility modes separate reproduction from corrected research behavior.
- PHOENIX14T-first vertical slices provide runnable checkpoints after every major change.
