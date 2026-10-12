# Time-aligned fusion progress — 2026-10-05

**IMPLEMENTED / PENDING REAL PROFILE; EXPERIMENT NOT RUN**

Latest extension verification2026-10-06: automatic extraction5h→training18h already
approved. Combined F1/F2/F3 fix wave: final affected463passed/4warnings/97.45s;
single main full run1614passed/27warnings/197.57s. One-line latest package-attempt
lookup and fixture extension followed full run; final focused/static checks cover
frozen bytes. Dedicated126passed/1warning evidence retained; extractor/package/
environment sources unchanged. Configured Ruff and both targeted Pyright checks
pass. Historical CPU preflight74.85s and disk/GPU observations remain snapshots
from before source changes. **NOT RUN; scoped final re-review pending**.
Authoritative [extension status](../../motion-fusion-pipeline-status-2026-10-05.md)
and [final-fix report](../../../.superpowers/sdd/2026-10-05-motion-extraction-autotrain/final-fix-report.md).
Earlier verification and original-cache chronology discussion below are historical.

Task5 software verification/documentation: **DONE_WITH_CONCERNS**; final-fix wave
resolves all final-review code/static/whitespace findings. Scientific
execution remains incomplete. This status supersedes the implementation plan's
historical “No implementation has begun” statement without changing that plan.

## References

- [Approved design](../specs/2026-10-05-time-aligned-fusion-design.md)
- [Implementation plan](2026-10-05-time-aligned-fusion.md), Task5
- [Task5 report](../../../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-5-report.md)
- [Results/admission status](../../time-aligned-fusion-results-2026-10-05.md)
- [Whole-review manifest](../../../.superpowers/sdd/2026-10-05-time-aligned-fusion/whole-fusion-review-manifest.md)
- Task reports: [1](../../../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-1-report.md),
  [2](../../../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-2-report.md),
  [3](../../../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-3-report.md),
  [4](../../../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-4-report.md).

## Implemented and freshly verified

- Task1: evidence-bound timing builder, CPU receipt CLI and full-source runtime
  validation. Evidence identity/consistency is checked; historical truth is not
  manufactured by the schema.
- Task2: reference, aligned fixed-gate and aligned learned-gate adapters;
  source-index interpolation, real-window support, padding isolation, zero-init
  correction, common-weight/RNG parity and gradient coverage.
- Task3: opt-in runner, two-seed initialization/profile interfaces, frozen recipe,
  two-stage gradient smoke, dev selection, full-state resume and immutable
  initialization/telemetry/checkpoint-size bindings. Telemetry fixes reviewed PASS.
- Task4: sequential six-run controller, neutral watchdog/source bindings,
  conservative accounting, storage admission, completion/recovery validation and
  paired summaries. Partial recovery validates existing scheduled dev artifacts
  before GPU dispatch even without a selection receipt; scoped fix reviewed.
- Final-fix fresh focused suites: **311 passed, 4 warnings, 41.86s**. Full unit suite
  run once in final-fix wave: **1358 passed, 27 warnings, 135.01s**. These include CUDA software tests,
  not real-data study admission or production profiling.
- Configured Ruff passes. Exact requested nine-file Pyright now reports
  **0 errors, 0 warnings, 0 informations** after reproducing and fixing all eight
  fusion-addition narrowing diagnostics. Initialization order/RNG behavior retained.
- Final summary rechecks complete collected source inventories for all six runs,
  including retained resume checkpoints, and admitted timing/profile/protocol raw
  hashes before publication. Controller and both seed locks remain held. Same-size
  drift, missing/foreign inventories and wrong hashes reject with incomplete status.
- Read-only parser checks cover all 20 valid controller arm/seed/mode combinations.
  Actual missing timing path rejects before root/profile/model creation.
- `git diff --check` and explicit nine-new-file whitespace scan pass; controller
  test trailing space removed.

## Unresolved authentic chronology gate

No real `motion-timing.json` exists under the approved external study root;
that root also does not exist. Required proof is an original historical extraction
receipt or an independently supported, explicitly reconstructed attestation that
establishes actual released window parameters/order, motion feature row order,
and source-frame ordinals shared by the 256x256 motion and 210x260 spatial frames.
Bind extractor/helper, annotation, motion manifest, spatial frame-row map and
external evidence bytes to that proof.

Matching archive contents, matching feature/window counts and today's extractor
defaults do not establish historical chronology. A checksum proves identity, not
the truth of a claimed extraction convention. No such attestation was invented
and no real timing map was published.

## Approved design, still prospective

RTX4090; three arms × seeds 0/1 = six fresh runs. Plain spatial MLP backbone,
existing cosine DINO and VideoMAE caches, original TCN, FLAN-T5-XL LoRA and VT-Align.
16000 updates, batch4, BF16, no accumulation; VT-only warmup2000 and LR warmup1600;
AdamW adapter/VT LR3e-4 and FLAN LR5e-5, betas0.9/0.98, eps1e-8, decay0.01,
clip1, cosine-to-zero. Full Signer03 dev at5500/10400/16000; BLEU-4 selection,
earlier-step tie break, beam5/max64/noICL. No Signer07 scoring.

Ceiling: **64800 charged GPU-stage seconds / 18 hours**, 32400 per seed and 10800
per arm/run, including initialization/smokes/dev stages conservatively. Approval
does not establish feasibility. Existing Qwen policy remains
`qwen-schema98-unreviewed-v1`, `human_review_status=not_assessed`.

Fresh read-only hardware/disk observations: RTX4090, capability8.9, 24564MiB,
driver610.62; Linux available467086688256 bytes (435.008GiB), Windows C:
available305242980352 bytes (284.280GiB). Capacity is a snapshot, not runtime
admission or measured model memory. The 5.6GiB checkpoint estimate remains an
estimate until owned checkpoints exist.

## Remaining actual-run checklist

- [ ] Resolve authentic chronology and produce the byte-bound CPU timing receipt.
- [ ] Produce real common-initialization attestations for both seeds.
- [ ] Run controller-accounted RTX4090 BF16 ordinary/long real-data smokes for all
  arms; verify finite joint losses, two-stage branch gradients, state restoration,
  actual output-token parity and effective optimizer recipe.
- [ ] Measure real memory/step-time/checkpoint size under accounting; establish
  prospective six-run feasibility within the unchanged cap and storage reserve.
- [ ] Admit and execute the six fresh runs through the authorized controller.
- [ ] Validate six selections and all18 dev reports with exact582-reference order;
  report both seeds' trajectories, contrasts, diagnostics and charged total.
- [ ] Add a protocol completion/result entry only after actual completion.

Original Task5 made documentation additions only; final-fix wave changes five new
fusion source/test files and these two status documents. Prior source, recipes, reports, selected/
final models, registry and historical protocol results remain preserved. Whole
review uses an explicitly constrained hash manifest, not a truncated full patch;
controller must rule on that fallback. Historical full task packages remain
unchanged. Authoritative overlay: [final-fix report](../../../.superpowers/sdd/2026-10-05-time-aligned-fusion/final-fix-report.md)
and [delta patch](../../../.superpowers/sdd/2026-10-05-time-aligned-fusion/final-fix-review.patch).
No scientific result is available yet.
