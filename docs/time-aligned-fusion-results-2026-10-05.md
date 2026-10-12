# Time-aligned fusion results/admission status — 2026-10-05

**IMPLEMENTED / PENDING REAL PROFILE; EXPERIMENT NOT RUN**

## Latest extraction-to-training verification — 2026-10-06

New contemporaneous extraction and automatic5h→18h training are already approved.
Tasks1–4 extension gates passed. Combined F1/F2/F3 fix wave: affected463passed/
4warnings/97.45s; single main full run1614passed/27warnings/197.57s. Final one-line
latest package-attempt lookup and fixture extension followed full run; final
focused/static checks cover frozen bytes. Dedicated126passed/1warning/30.84s
evidence retained with unchanged extractor/package/environment bytes. Configured
Ruff and targeted main/extractor Pyright pass. **NOT RUN; scoped final re-review
pending**. No independent final PASS claimed.

Historical Task5 CPU preflight passed in74.85s before this source change, identity
`73adfb8cd6df37d233eed29608234c8a0ac2299d8ef503021be780ece05c54b2`;
7096clips/827354frames/93232windows and actual Task3 producer actor gate pass.
Linux425.117GiB/Windows272.266GiB meet current225.956GiB pipeline estimate;
RTX4090 idle. No pipeline/bootstrap/new-cache/new-training root exists. No real
GPU profile/extraction/training was launched and no new model result exists.
See [current status/operator command](motion-fusion-pipeline-status-2026-10-05.md)
and [final-fix evidence](../.superpowers/sdd/2026-10-05-motion-extraction-autotrain/final-fix-report.md).

The following original-cache verification/history remains historical evidence.
Its authentic chronology blocker remains unresolved; new-motion policy replaces
that input only after an independently validated contemporaneous package exists.

This document records software verification and missing experiment admission.
It contains no fusion model results, real-data timing receipt, production runtime
profile, measured model peaks or measured production token/parameter totals.

## Evidence and status

| Item | Current evidence |
|---|---|
| Task1–4 software | Implemented; independent reviews recorded PASS after scoped telemetry/recovery fixes |
| Final whole-change review | All findings addressed; scoped re-review found no new Critical/Important issues |
| Fresh focused tests | Final-fix: 311 passed, 4 dependency warnings, 41.86s |
| Fresh full unit suite | Final-fix: 1358 passed, 27 warnings, 135.01s; run once with WSL CUDA library path |
| Configured Ruff | Pass |
| Requested targeted Pyright | Exact nine files: 0 errors, 0 warnings, 0 informations |
| Final-summary integrity | Complete six-run source inventories and admitted shared raw hashes rechecked under held controller/seed locks |
| Whitespace | Explicit nine-file scan and tracked git diff check pass |
| Real timing admission | Blocked; authentic historical chronology missing |
| Common initialization on real inputs | Not produced |
| Real-data GPU smoke/profile/admission | Not run |
| Six fusion runs and dev trajectories | Not run; no scores or contrasts available |
| Actual production peaks, masks/tokens, parameter inventories, checkpoint sizes | Not measured/published |
| Overall scientific conclusion | None; no architecture winner or signer-invariance claim |

Full-suite warning categories are installed Lightning/pkg_resources/TypedStorage/
TorchMetrics deprecations and existing CPU-trainer, dataloader, resume and validation
configuration warnings. Warning count matches the supplied pre-fusion baseline;
the baseline was not rerun. CUDA unit checks establish software behavior only.

## Exact admission blocker

User chose authentic original extraction records. Follow-up of remaining local
project/recovery/SpaMo/dataset metadata found no original extraction receipt;
[scope and required external inputs](original-motion-records-check-2026-10-05.md).
User subsequently approved new documented motion extraction and requested
automatic training afterward. The revised
[extraction-to-training design](superpowers/specs/2026-10-05-motion-extraction-autotrain-design.md)
specifies separate5hour/18hour caps and authenticated new-cache handoff; written
design/implementation-plan review precedes that extension. No assumed alignment
or new-cache generation has occurred yet. Original downloaded-cache admission
remains blocked; new contemporaneous extraction records would replace that input
under a distinct new-motion policy.

The approved path
`/home/kan/datasets/despamo/signer-pilot/time-aligned-fusion-study-v1/motion-timing.json`
does not exist; neither does its study root. A fresh read-only controller call
with authorization and that path fails with `regular fusion identity file
required` before study-root/profile/model creation.

Need an authentic historical extraction receipt or independently supported
reconstruction, explicitly labelled reconstructed when applicable. It must
establish the actual released 16-frame/stride8 window convention and ordering,
motion-feature row ordering, and source-frame ordinals linking 256x256 motion
frames with 210x260 spatial frames, with required input/source/evidence byte
bindings. Current helper defaults, archive equality and count agreement cannot
alone prove these facts. Software accepts supplied evidence consistently; it
does not certify historical truth merely because hashes/schema match.

## Approved experiment, not measured results

Approved RTX4090 study: fresh reference, aligned-fixed and aligned-learned runs
for each of seeds0/1, using the same plain spatial MLP/backbone and frozen inputs.
Fixed gate0.5; learned scalar sigmoid gate initially0.5; correction output starts
at zero. Keep original motion tokens, TCN and matched masks/token path.

Matched recipe:16000 updates, batch4, BF16, no accumulation, VT-only warmup2000,
LR warmup1600, adapter/VT LR3e-4, FLAN LoRA LR5e-5, AdamW0.9/0.98,
eps1e-8, decay0.01, clip1 and cosine-to-zero. Select on full582-clip Signer03 dev
at5500/10400/16000 using highest BLEU-4/earlier-step ties, beam5/max64/noICL.
The split counts5746/582/768 are frozen protocol identities, not new experimental
observations. Signer07 is excluded from this study's selection/scoring.

Approved ceiling64800 charged GPU-stage seconds (18h), per-seed32400 and
per-run10800, includes admission/smokes/dev conservatively. Remaining steps:
chronology receipt, real two-seed initialization, controller-accounted ordinary/
long-sequence BF16 smokes for all arms, real profiling/storage admission, six-run
execution and validated report collection. No cap/recipe relaxation is assumed.
Qwen supervision stays `qwen-schema98-unreviewed-v1` with
`human_review_status=not_assessed`.

Once executed, report both seeds' complete dev trajectories, paired BLEU1–4/
ROUGE-L contrasts, repetition/uniqueness, gate/coverage diagnostics, measured
parameters/tokens, memory, checkpoint sizes and charged runtime. Aligned-fixed
versus reference combines timing and added-capacity effects; learned versus fixed
tests incremental gating. Neither dev gains nor software tests prove causal
motion grounding or broad signer invariance.

## Concerns and preserved history

- Final-fix wave reproduced stale complete-summary publication in eight RED
  cases, then fixed publication-time integrity and all eight Pyright diagnostics;
  controller test trailing space removed. Fresh regressions cover all six rows,
  same-size checkpoint/receipt/report/resume drift, missing/foreign inventories,
  wrong hashes and timing/profile/protocol drift, plus valid publication/accounting.
- Final publication retains both seed locks through summary/state writes.
  Historical Task5 report and full review packages remain pre-fix baselines;
  [final-fix report](../.superpowers/sdd/2026-10-05-time-aligned-fusion/final-fix-report.md)
  binds old/new source hashes and authoritative
  [delta patch](../.superpowers/sdd/2026-10-05-time-aligned-fusion/final-fix-review.patch).
- Whole-review fallback is an explicit hash manifest because four exact full
   task diffs already total293602 bytes; it is not a full assembled patch and
   was accepted by the controller's recorded host-constraint ruling. Complete
   task diffs plus final delta remain available; none were omitted or truncated.
- Existing cleanup receipt records preservation of selected/final models and
  reports/metadata. Fresh checks find all23 protected checkpoint paths, all18
  selected checkpoint paths, and ten registry report-byte bindings intact.
  Checkpoint content was not rehashed/deserialized in Task5; fresh stat and
  metadata fingerprints are documented in the report. No historical scoring
  was rerun or protocol completion entry added.

## Rulings made during execution

1. **Current main checkout:** user explicitly selected option1; agents were
   limited to named fusion additions. Cost if wrong: overlapping uncommitted work.
2. **Native artifact writes:** replaced shell-written briefs/review packages with
   native file tools to follow host policy. Cost if wrong: additional bookkeeping.
3. **Uncommitted review and retained ledger:** commits were unauthorized, so
   task-local addition diffs replaced commit ranges and scratch records remain.
   Cost if wrong: less convenient Git recovery, not loss of execution records.
4. **Available agent roles:** Task API has no model-selection argument; used
   general/reviewer roles without unsupported parameters. Cost if wrong: higher
   context cost than ideal tier-specific selection.
5. **Software continued, training gated:** finished independent software while
   unresolved timing chronology blocked production admission. Cost if wrong:
   software may await additional provenance before it can be used.
6. **Telemetry producer gap repaired:** extended runner receipts with actual
   parameter/mask/peak/size measurements required by spec, then controller
   consumed them. Cost if wrong: additional receipt code and tests.
7. **Whole-review package split:** accepted hash-bound four complete task diffs
   plus documentation, later final delta, rather than a322924-byte native-write
   request carrying roughly80k tokens. Cost if wrong: reviewer navigates multiple
   complete packages rather than one; no evidence was truncated.

## References

- [Design](superpowers/specs/2026-10-05-time-aligned-fusion-design.md)
- [Plan](superpowers/plans/2026-10-05-time-aligned-fusion.md), Task5
- [Progress and incomplete execution checklist](superpowers/plans/2026-10-05-time-aligned-fusion-progress.md)
- [Task5 verification report](../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-5-report.md)
- [Whole-review manifest](../.superpowers/sdd/2026-10-05-time-aligned-fusion/whole-fusion-review-manifest.md)
- Task reports: [1](../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-1-report.md),
  [2](../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-2-report.md),
  [3](../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-3-report.md),
  [4](../.superpowers/sdd/2026-10-05-time-aligned-fusion/task-4-report.md).
