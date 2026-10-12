# Official DINO adaptation design/plan self-review — 2026-10-07

Scope: native self-review of the documented, user-approved adaptation design and
implementation plan. This is a document review, not an implementation or training
verification report.

## Artifacts

- [Design](superpowers/specs/2026-10-07-official-phoenix14t-dino-adaptation-design.md).
- [Implementation plan](superpowers/plans/2026-10-07-official-phoenix14t-dino-adaptation.md).

## Evidence inspected

- Crop export and sidecar under `artifacts/pheonix_new_version/`:
  raw SHA-256 `73948b9784e23acac61f51cf46d0aec4259f8402faf10be97df40aa8a7e606dc`;
  canonical records SHA-256
  `8db97d81133a3163bdcc2e2efdb434cd006693f8b2f9dc63c074fac7e3e9b24c`.
- Export contains 7,096 unique training records, each with 19 slots. Existing
  sampling formula/order is compatible. Valid left/right/mouth counts are
  24,305/27,618/34,732. Human review is not assessed.
- Previous text manifest identifies pinned CLIP model/tokenizer revision
  `32bd64288804d66eefd0ccbe215aa642df71cc41` and width 768.
- `e3_data.py` rejects Signer03/07, and `e3_train.py` binds 5,746 old logical
  train clips. Neither is an official-split loader.
- `e3_model.py` exposes seven factor heads, nuisance GRL, and detached
  reference/pattern/spread losses. `e3_train.py` uses shared LR warmup and a
  10%-of-joint-duration GRL ramp; neither implements the newly approved clocks.
- Dedicated DINO runtime/lock exist and pin torch 2.5.1+cu121 and transformers
  4.56.2. Main runtime is used separately for CLIP preparation.
- At the review's initial live check, the existing fusion controller remained
  running. Only documentation was written; its three time-control source hashes
  still matched the fixed waiver record.

Complete joins against all old source/teacher/vector bytes and tokenizer limits
remain implementation/preflight checks; the export audit does not replace them.
Producer journal/source hashes are retained as assertions unless their referenced
files are also available for independent verification. No visual correctness or
signer-invariance result is claimed.

## Findings addressed

| Finding | Resolution |
| --- | --- |
| Legacy encoder-only cache identity would collide with changed targets. | New key binds source digests, merge/frame policy and encoding contract; old cache is untouched. |
| Mutable workspace export could change after target preparation. | Snapshot parent JSON bytes under the new version's `inputs/`; bind original/snapshot paths and verify copied hashes. |
| Old failure policy could erase valid v2 articulators or backfill missing crops. | Exact per-slot policy: stable rows/masks copied from v1, articulators exclusively from v2, null rows remain zero. |
| Old shared warmup would start LoRA near head peak; old GRL ramp would last 3.5 epochs. | Separate clocks: C=2,215 and W=886; apply rates before one-based updates; pin boundaries at 2,216/3,101/17,720. |
| A post-copy native archive receipt could change recursive source inventory on retry. | Seal `completion.json.payload_files` after closed ledger; exclude locks/scratch/archive bookkeeping; bind completion file hash separately. |
| NTFS directory fsync may differ from native output guarantees. | Record supported durability level, re-open/hash all archive bytes, retain native originals and a fsynced native archive receipt. No Linux writer on B:. |
| Exact production recipes conflict with small replay tests. | Parameterized CPU-tested update/schedule helpers, strict production CLI, and no public bypass or fixture/smoke resume admission. |
| Old and new budgets/roots could be confused. | New official policy/identity/root and measured accounting; no reuse of the existing fusion waiver or historical pilot limits. |

## Requirement-to-task coverage

| Approved requirement | Plan coverage |
| --- | --- |
| New left/right/mouth plus previous biometric/clothing/hair/background | Tasks 1–2 |
| No articulator backfill; old failed clips retain independent new targets | Tasks 1–3 |
| Pinned 19x768 vectors, unchanged old stable vectors, new dataset identity | Tasks 1–2 |
| Official train-only 7,096 clips and nine signers | Task 3 |
| 40 total epochs, five calibration, two joint warmup/ramp epochs | Tasks 4/6 |
| Seven factor losses plus all three teacher terms, relational weight 0.10 | Task 6 |
| Ten-epoch exports and latest two full resumes | Task 5 |
| Exact RNG/sampler/optimizer/schedule replay and elapsed accounting | Tasks 5–6 |
| Linux-native working outputs and both Linux/Windows space checks | Task 6 |
| Verified B: archive, immutable historical identities, no automatic deletion | Task 7 |
| Separate runtimes, immutable live study, offline pinned models | Global constraints and Tasks 2/3/6 |
| Feature extraction and six translation runs are later scope | Design goal and execution handoff |

## Review conclusion

No remaining known design-blocking omissions in this scope. The seven tasks have
exact files, interfaces, RED/GREEN commands, boundary assertions, and execution
handoff. Tests cover each of the five Review Focus hazards. Function/property names
are consistent across adjacent tasks. No placeholder requirements remain.

Document checks covered links, code fences, recipe constants, schedule arithmetic,
input digests, whitespace, and `git diff --check`. Implementation test commands
are explicitly future work; no implementation test or GPU smoke was run for this
documentation-only request. There were no commits or training launches.
