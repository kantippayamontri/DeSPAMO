# Official PHOENIX14T Full-Split Feature Preparation Implementation Plan

> **For agentic workers:** Use `executing-plans` natively with TDD/self-review.
> User instructed "do next steps" after epoch40 adaptation and archival completed.
> No subagents, commits, original-artifact deletion or implicit held-out scoring.

**Goal:** Publish provenance-bound full-video spatial/motion features for the
approved official conventional benchmark:7096train/519dev/642test clips.

**Architecture:** Keep completed adaptation and old generators unchanged. Add
bounded all-split producer modules in the retained isolated worktree. DINO uses
epoch40 LoRA and existing full-frame FP32 extraction. Motion reuses byte-verified
training features from the documented package and extracts missing dev/test with
the same frozen low-level VideoMAE producer in its dedicated environment.

**Tech Stack:** Existing dedicated DINO/motion uv environments; NumPy/Torch;
native atomic writes/locks; existing image/model preprocessing and feature schema.

**Spec:** This plan records the already approved six-run conventional-benchmark
data contract. Official splits are exact, clip-disjoint, signer-overlapping;
adaptation used training only. Epoch40 is predeclared, never test-selected.

## Global Constraints

- Epoch40 native run:
  `official-phoenix14t-dino-adaptation-v1/runs/official-dino-adapt-bfea762ea2f71c244c9f3e2ef5f6be1a0f10b48bde2a5c6f05dcc8ba0faf5f0d`.
- Frames `/home/kan/datasets/despamo/phoenix-frames-210x260px`; source row-map from
  original DINO root `features/1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d/complete/frame_rows.json`.
- Official `{train,dev,test}_info_ml.npy` in `/home/kan/Research/SpaMo/preprocess/Phoenix14T`.
- Spatial FP32[frames,2048], concatenated224/448 CLS with epoch40 LoRA, full frames.
- Motion FP32[windows,1024], VideoMAE revision12da269a02d3e1fbbb7011e610e591fea8061dca,
  window16/stride8/full tails dropped, same RGB/bilinear/crop/normalize recipe.
- Native output under `/home/kan/datasets/despamo/official-phoenix14t-features-v1`.
  Check Linux and Windows C: space plus50GiB reserve before/through publication.
- Raw arrays projected spatial7,764,017,152bytes + motion436,772,864bytes. Include
  receipt/source JSON and in-flight copies in admission; no duplicated frozen models.
- Data/runtime/model/source identities and per-clip receipts required; resume only
  validates and reuses exact completed clips. Source drift or corrupt cache fails.
- Production GPU modes explicit authorization, unique lock/owner, elapsed accounting,
  no concurrent GPU work. No time-forecast veto; no modifications to frozen producers.

## Review Focus

1. Standard split accidentally reuses signer-pilot partitions: exact IDs/counts and
   integer annotation records, split paths, disjoint IDs validated.
2. Stale selected model or base mismatch: epoch40 receipt/tensor shape and pinned
   base revision/hash validated before loading.
3. Feature/source drift on resume: array and source/receipt byte hashes checked;
   publication completes only after full inventory verification.
4. Reused motion features falsely relabeled as freshly generated: preserve old
   parent completion/source provenance; copied bytes equal originals.
5. Missing B:/storage or partial publication: native production guard, projection
   and50GiB reserve, atomic complete marker last; no original deletion.

---

### Task 1: Shared all-split package contract

**Create:** `tools/benchmark/feature_package.py`,
`tools/benchmark/tests/test_feature_package.py`.

**Interfaces:** `official_inventory(annotation_root: Path, row_map: Path) -> dict`;
`write_clip(root: Path, split: str, clip: str, paths: list[Path], values: np.ndarray,
identity: dict, extra: dict) -> dict`; `verify_clip(...) -> dict`;
`finish_package(root: Path, identity: dict, inventory: dict, width: int) -> dict`.

- [ ] Tests first:exact split membership, full-source ordering, float32 widths,
  regular arrays/receipts, source/feature hash drift, existing-cache immutability,
  partial publication and FeatureManifest-compatible record keys.
- [ ] Run RED using dedicated DINO interpreter; implement minimal stdlib/NumPy
  contract and native atomic publication; run GREEN and static checks.

### Task 2: Epoch40 DINO producer and actual smoke

**Create:** `tools/benchmark/official_spatial.py`,
`tools/benchmark/tests/test_official_spatial.py`.

- [ ] Test epoch40 receipt validation and rejectwrong/frozen/partial model;
  reuse unchanged encoder/scales and strict16 LoRA tensors; source-array pairing.
- [ ] Implement preflight/smoke/run/verify CLI, separate smoke IDs, owned native
  ledger/lock;32 frames per inference batch (not24training clips), FP32.
- [ ] Real GPU smoke over ordinary/long clips; publish only requested full production
  inventory, separate smoke receipt. Verify matching scalar hashes and memory fit.
- [ ] Launch all8257clips detached; verify actual clip counter advances. Future
  complete metadata includes records/frame_rows/timing-ready sourceindices.

### Task 3: Motion coverage and sequential orchestration

**Create:** `tools/benchmark/official_motion.py`,
`tools/benchmark/prepare_features.py`, focused tests in `tools/benchmark/tests/`.

- [ ] Verify old training package with its authoritative consumer before copying;
  per-clip sourcehashes/windows/arrayhash preserved and matched to new spatial rows.
- [ ] Extract only officialdev/test using unchanged
  `tools.motion.extractor.load_encoder/extract_motion_clip` and batch8windows.
  Separate environment/source/model identities from reused training provenance.
- [ ] Add CPU tests for copied-parent bytes and missing/incorrectwindow support;
  real GPU smoke/parity, then verified manifest-last publication.
- [ ] Run orchestrator sequentially:DINO then motion, no simultaneous GPU jobs,
  immutable request/sourceidentity, child owner receipts, durable stage reports.

### Next translation stage

Approved configurations remain reference/fixed/learned x seeds0/1, selecting on
official519dev and final642test scoring after selection is frozen. Its matched
training schedule and batch profile must be recorded in a follow-up training plan.
Compact trainable-only checkpoints plus pinned frozen FLAN/model manifests are
needed: old six-run full-model checkpoint storage exceeds remaining C: headroom.
Prove compact replay/selection equivalence before using this serialization change.
Do not silently remove old/native artifacts or mount a new disk.
