# Official DINO data and schedules — 2026-10-07

## Scope and implementation

User authorized the next CPU-safe batch, Tasks 3–4 of the
[adaptation plan](superpowers/plans/2026-10-07-official-phoenix14t-dino-adaptation.md).
Work remains uncommitted in the isolated branch `work/official-merged-targets` at
`.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`.
No files were integrated into the active training checkout, no dependencies
changed, and no DINO model, GPU smoke, adaptation job or held-out scoring launched.

New files in that worktree:

- `tools/dinov3/official_data.py`: official split/clip/signer admission, sealed
  merged-cache verification, teacher/frame/target joins and CPU batch loading.
- `tools/dinov3/official_schedule.py`: independent one-based head/LoRA/GRL clocks,
  epoch/checkpoint arithmetic and validated replayable schedule state.
- `tools/dinov3/tests/test_official_data.py`.
- `tools/dinov3/tests/test_official_schedule.py`.

## Task 3: real-data admission and batch preflight

Executed in the existing dedicated DINO runtime with hidden CUDA devices,
`HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2` and no bytecode writes.

- Exact integer-record counts: train **7,096**, dev **519**, test **642**.
- All nine signers represented; official clip inventories disjoint.
- Train Signer03/07 admitted by clip membership, not the old signer exclusion.
- Source and teacher row-map hashes, all 7,096 teacher/target files, and all five
  sampled source PNG hashes per training clip verified during admission.
- Merged cache policy, model/tokenizer revisions, parent/producer snapshots,
  manifest/document/completion hashes and target/frame joins verified using only
  stdlib/NumPy/Torch. **Pydantic was not imported.**
- Nineteen metadata bindings recorded; selected clip descriptors are sealed and
  checked per load. Metadata signatures reject changes without repeatedly hashing
  the large full JSON closure on each training batch.

Prepared manifest:
`/home/kan/datasets/despamo/official-phoenix14t-dino-adaptation-v1/targets/662471b90eabac7b37044c201541406c084986b646e15e2e119dd667635c7452/manifest.json`.

Manifest SHA-256:
`ebb5e1000a00eedb18243e364e1423925db0096d835819dd617a480752c3f554`.

Ordered train-ID digest:
`ccc66d2160406b254f20f1706e5463cbdac3f5582b96727563556da9e7466556`.

Clip-descriptor digest:
`9ce3315b00fe0199cba3871a453414a59d970aec9dfcc8f11caa1a8d44f18828`.

The physical preflight loaded a 16-clip batch including Signer03, Signer07 and a
previous whole-clip failure with independent new articulator supervision:

| Field | Shape | Type/device |
| --- | --- | --- |
| images224 | `[16,5,3,224,224]` | float32 / CPU |
| images448 | `[16,5,3,448,448]` | float32 / CPU |
| teacher | `[16,5,2048]` | float32 / CPU |
| vectors | `[16,19,768]` | float32 / CPU |
| masks | `[16,19]` | boolean / CPU |

That batch had 72 masked slots. All masks/zero rows were validated; clips were
not dropped or given old articulator backfill. This is data preparation, not a
model-training or visual-correctness result.

## Task 4: exact approved schedules

40 total epochs include five calibration and 35 joint epochs. Drop the final
eight training clips per shuffled epoch at batch 16; complete batches per epoch
are 443. Learning-rate multipliers apply before one-based update k.

| Boundary | Updates/behavior |
| --- | --- |
| End head calibration | k=2,215: head multiplier 1, LoRA/GRL 0 |
| First joint update | k=2,216: LoRA/GRL `1/886`; head starts cosine decay |
| End LoRA warmup/GRL ramp | k=3,101: LoRA/GRL 1 |
| Final horizon | k=17,720: both LR multipliers 0; GRL 1 |
| Milestones | 4,430 / 8,860 / 13,290 / 17,720 |

The final zero-LR update is intentional and matches the documented recipe.
Schedule state records completed/applied counts and next update; terminal state
has no next update. Invalid integers, float/bool counters, missing parameters and
off-by-one replay state are rejected. Smaller CPU-test schedules are parameterized
helpers; the deferred production CLI must enforce the approved recipe.

## Tests and self-review

- Watched 16 data tests fail for absent implementation, then pass.
- Watched 22 schedule tests fail for absent implementation, then pass.
- Added RED-to-GREEN regressions for a metadata mutation during sealing and
  numerically equal float replay counters that would otherwise bypass dict equality.
- Dedicated full DINO suite: **231 passed, 1 skipped**, CPU-only.
- Ruff passed; Pyright reported **zero errors**; both checkout diff checks passed.
- Existing legacy E3 data/schedule tests remain unchanged and pass; the old
  signer-pilot policy was not loosened.
- Self-review inspected exact inventory membership, frame/teacher/target hashes,
  mask independence, metadata/per-row drift checks, runtime separation, schedule
  arithmetic and replay boundaries. No known important open finding in this scope.
- Complete active fusion source closure was rechecked and matched its checkpoint
  identity through the existing waiver mapping; the live controller still reported
  running at the last check. Live state must be rechecked before integration/launch.

## Handoff

Tasks 1–4 are complete in the isolated workspace. Next is Task 5: durable
milestone exports and exact optimizer/sampler/RNG resume recovery. Task 6 must
apply these clocks and provide owned accounting, preflight and an explicitly
authorized GPU smoke before any adaptation launch. Task 7 is verified cold
archival to B:. No commits, merges, original-artifact deletion or launch are
implicit in completing this batch.
