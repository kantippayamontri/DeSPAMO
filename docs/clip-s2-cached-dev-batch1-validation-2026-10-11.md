# One cached batch1 dev diagnostic — 2026-10-11

**Result: one bounded diagnostic completed on Game Ready driver617.42.**
This is not a full-dev stability result, training recovery or proof that the
original Windows GPU-reset fault is fixed.

## Authorization and prerequisites

User explicitly authorized one dev batch, batch1/beam5/cacheTrue, after the
driver610.62→617.42 update and Windows restart. Windows driver registration was
rechecked as32.0.16.1742. Windows boot remains2026-10-11T07:11:43.5000000Z and
WSL boot21ceb469-0d6b-494a-b4fa-dd12d94f2931.

- Retained native epoch2 checkpoint,1774updates, verified SHA256
  `d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.
- All84 original source hashes, pinned FLAN file hashes, study configuration,
  annotation/feature identities and train-only exemplar matched original identity.
- Original loader verified the selected clip's actual spatial/motion array hashes.
- Original inference exemplar:`12February_2010_Friday_tagesschau-122`.
-17 existing inference-helper CPU tests passed in3.69s before GPU work.
- Native Linux free256.43GiB, Windows C:89.12GiB approximately, B:626.16GiB
  approximately at preflight; sufficient for the small diagnostic artifacts.
- One-shot marker and writer lock prevent reuse/retry of this diagnostic namespace.
- Training/controller stop remained active. Only scoped GPU diagnostic permission
  was temporarily added; script revoked it after completion.

## Observed execution

| Item | Observation |
| --- | --- |
| Clip |11August_2010_Wednesday_tagesschau-2, first admitted official dev clip |
| Input lengths |42 spatial frames;4 motion windows |
| FLAN joint tensor |[1,154,2048];154 valid visual/prompt tokens |
| Generation |Batch1,beam5,effective5 beam sequences,max64,deterministic,cacheTrue |
| Model load start |07:23:58.318612UTC |
| Generation start |07:24:26.658971UTC |
| Batch complete |07:24:29.019802UTC |
| Whole batch elapsed |4.453312158s, including adapter/prompt/generation work |
| Model-load-to-diagnostic elapsed |30.791892591s, excluding preceding CPU preflight |
| Peak allocated tensor memory |6435653120bytes =5.993669GiB |
| Peak allocator-reserved memory |6606028800bytes =6.152344GiB |
| Optimizer state on GPU |Not loaded; fresh inference-only process |
| Training cache configuration |use_cacheFalse retained; inference override only |
| Translation metrics |None; diagnostic_onlyTrue,completeFalse,1/519 clips processed |
| Post-probe Windows event inspection |07:25:10.3969468UTC;0 matching critical/error events since boot |

The event query covered System-log levels1/2 for nvlddmkm, Display,
Microsoft-Windows-WER-SystemErrorReporting and Kernel-Power event41. No host
reboot occurred. This log scope/time window is not exhaustive hardware-health
evidence and cannot rule out delayed or intermittent failures.

## Interpretation

This small cached batch completed without a CUDA error or observed host reset,
with peak allocator memory well below the RTX4090's nominal24GB capacity. Memory
numbers are PyTorch allocated/reserved values, not measured total host GPU usage.

This does **not** measure the original in-training dev peak: optimizer state and
training buffers were absent, batch size reduced4→1, cacheFalse→True, and driver
changed610.62→617.42. No uncached/batch4 comparison was authorized or performed.
Consequently this combined mitigation smoke cannot identify which change matters.
The selected short clip does not cover long sequences or the later failure locus.

Checkpoint1774updates is still inside4000-update VT-only warmup. Generated text
is retained in the native diagnostic report, but this smoke does not assess
translation quality or publish a dev selection/score. No official test scored.

## Artifacts and containment

Native diagnostic namespace, under the same held v3 study root:

`diagnostics/cached-dev-batch1-driver61742-20261011/`

Contains `preflight.json`, `started.json`, fsynced `events.jsonl`, `report.json`
and diagnostic writer lock. The diagnostic did not overwrite old dev reports,
checkpoint selections, budgets or original source/data identities.

`report.json` SHA256:
`6f015bb1c2eef18223342b69f33623ed2e0bd976336155c33a134629ad2df688`.

Standalone invocation:
`docs/diagnostics/clip-s2-cached-dev-batch1-2026-10-11.py`; SHA256
`f5c6504ca098f6cfe6fbc0103d7aee9849952c7948db2bf8e9a30b0f3c409e72`.
No checkpoint-bound module was edited. Ruff check/format and git diff whitespace
checks passed for this invocation. No source compatibility bridge implemented.

Manual stop is still`stopped_by_user`, automatic recovery disabled,
`gpu_validation_approved=false`, diagnostic scope consumed. No controller,
watchdog, full-dev evaluation or training was started. Additional GPU diagnostics
need separate scope approval; next proposed stage is a small fixed dev subset
with per-batch durable timing/memory logs, not automatic training recovery.
