# Merged v1/v2 target preparation — 2026-10-07

## Result

CPU-only preparation Tasks 1–2 completed while the existing fusion study was live.
The user explicitly authorized an isolated Git worktree after inspection showed
the controller hashes **all files beneath `src/`**, including newly added files.
No source files were added to the active checkout's `src/`; no dependencies,
training sources, previous datasets, caption records, or held-out registries changed.

Implementation worktree (uncommitted, not merged):
`.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`
on branch `work/official-merged-targets`, base `ebb0f6b`.

Completed manifest:
`/home/kan/datasets/despamo/official-phoenix14t-dino-adaptation-v1/targets/662471b90eabac7b37044c201541406c084986b646e15e2e119dd667635c7452/manifest.json`

Manifest SHA-256:
`ebb5e1000a00eedb18243e364e1423925db0096d835819dd617a480752c3f554`.

Merged document key:
`f37d1bfcb8cc5d957a892c77222b0b002450dc08cfdca0f07f91241ddcda626c`.

The output contains 7,096 per-clip NPZ files, shape `[19,768]` float32 vectors
and `[19]` boolean masks, sealed parent JSON documents, sealed producer source
files/runtime fingerprints, the merged target document, manifest and completion
receipt. NPZ files total **220,257,125 bytes**. No DINO adaptation was launched.

## Supervision policy and coverage

Policy: `qwen-stable-v1-crop-articulators-v2-unreviewed-v1`.
Human review: `not_assessed`. Availability is not visual correctness.

| Factor | Valid slots | Masked slots |
| --- | ---: | ---: |
| Biometric | 6,977 | 119 |
| Clothing | 6,977 | 119 |
| Hair | 6,977 | 119 |
| Background | 6,977 | 119 |
| Left hand | 24,305 | 11,175 |
| Right hand | 27,618 | 7,862 |
| Mouth | 34,732 | 748 |

Stable target rows were copied from the old pinned CLIP cache without re-encoding.
Articulator text comes exclusively from
`artifacts/pheonix_new_version/phoenix14t_qwen_text_targets_v2.json`;
masked v2 slots remain zero with no old-description backfill. Every original
training clip is retained. All 119 old whole-clip failures have at least one valid
new articulator target while keeping their four unavailable stable targets masked.

Source exports retain asserted crop/job/journal provenance. Their byte and records
digests, clip/signer/frame joins, and source/teacher row-map contracts were checked;
the preparation does not claim independent validation of unavailable producer
journal files or human visual ratings.

Encoding used main `.venv`, pinned CLIP revision
`32bd64288804d66eefd0ccbe215aa642df71cc41`, CPU/FP32, batches of 64 sorted
unique articulator strings, offline model loading, hidden CUDA devices and
`OMP_NUM_THREADS=2`. Sealed producer hashes participate in cache identity, while
cache verification reads its sealed producer metadata so later consumer changes
do not invalidate earlier versions.

## Verification

- Watched merge/encoder/CLI RED failures before implementation; added RED->GREEN
  regressions for frame hashes, CPU device enforcement, annotation signers,
  producer snapshots, consumer-version compatibility and fabricated review status.
- Final focused suite: **45 passed**, one upstream Hugging Face deprecation warning.
- Ruff passed; Pyright reported zero errors; `git diff --check` passed.
- Real preflight admitted all 7,096 clips and source text NPZ hashes/masks.
- Real CPU encoding exited zero and published the completed version.
- Independent full-output verification checked every NPZ/receipt, every copied
  stable row byte-for-byte against v1, every v2 mask, all zero-valued masked rows,
  sealed source/producer hashes, and the manifest digest.
- At `2026-10-07T11:03:54Z`, the active fusion controller remained running and
  its complete source closure matched the original checkpoint identity through
  the existing time-waiver compatibility check. This is a snapshot, not live status.

The broader worktree CPU-hidden unit run reported **664 passed, 1 skipped, and
4 failures** in old fake-scoring tests that call CUDA availability guards:

- `test_e3_heldout_run_writes_once_with_checkpoint_provenance`
- `test_run_writes_ordered_paired_reports_once`
- `test_pair_rejects_reference_drift_without_summary`
- `test_resume_keeps_verified_e1_score_and_only_scores_e2`

Root cause was reproduced as the intentionally hidden CUDA devices. Both affected
files were rerun with normal availability and their fully fake model/scoring
fixtures: **7 passed**. No real held-out scoring or extra GPU workload was run;
no unrelated source changes were made to hide those environmental failures.

## Execution rulings and handoff

1. Scope is Tasks 1–2 only; GPU/data-loader/checkpoint/archival training Tasks 3–7
   remain deferred. Cost: the DINO runner is not implemented by this preparation.
2. Isolate code because even a new active-checkout source file changes live
   admission. Cost: integrate the reviewed code only after final study publication.
3. User-confirmed per-slot merge overrides the linked worktree's older whole-clip
   rule for this new version only. Cost: separate policy and version metadata.
4. Keep sealed producer code/runtime fingerprints with the cache and verify old
   versions against their own metadata. Cost: small additional provenance files.
5. Preserve native self-review/no-delegation and uncommitted workspace. Cost:
   no independent reviewer; worktree and plan ledger must remain available.

Review covered identity/frame joins, independent masks, copied vector bytes,
offline CPU inference, path confinement, partial/idempotent publication, sealed
inputs, consumer compatibility and historical source safety. No known important
findings remain in this scope. The next implementation work is official DINO
data admission and approved independent schedules, after live-source/GPU ownership
constraints are clear. Do not integrate, commit, delete, or launch automatically.
