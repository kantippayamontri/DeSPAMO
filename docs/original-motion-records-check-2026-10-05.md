# Original motion extraction records — focused follow-up

Date: 2026-10-05. **Actual original receipt found: no, in the surveyed local metadata.**
Historical row-to-frame chronology remains blocked for the authentic-record route.
This finding does not establish that original records never existed elsewhere.

## Scope and method

Read-only investigation; this report is the sole authored artifact. Used bounded
filename inventories and exact content anchors (`mo_feat_p14t`,
`mae_feat_Phoenix14T`, `features.zip`, `phoenix14t_motion`,
`mae_extract_feature`, `overlap_size`, extraction receipt, motion row order).
Project text searches were limited to depth five and files below 2 MB, excluding
Git internals, environments, tests, tools, papers, generated caches and linked
directories. Explicitly opened the recovery target separately. Dataset metadata
inventory excluded model cache/HF cache; NPY, checkpoint and archive contents
were not read. No persistent index was created, archive hashes compared, source
history searched, or new extraction run performed.

Baseline evidence was read from
`.superpowers/sdd/2026-10-05-time-aligned-fusion/task-1-report.md`
and `docs/project-handoff.md`. Its archive/cache equality, current frame-order
checks and source conventions were not repeated as historical evidence.

## Newly surveyed roots and relevant files

| Root | Checked metadata and result |
|---|---|
| `/home/kan/Research/DeSpaMo-e1-recovery` | Bounded scan of 89 eligible text files, including root documents, configs, scripts, source and available `docs/` plans/specs. Exact release anchors appeared only in `configs/data/phoenix14t.yaml` and `docs/superpowers/plans/2026-09-22-phoenix14t-baseline.md`. The latter's line 558 gives a cache-indexing command. Available DINO extraction plans concern spatial features. No original motion extraction log, packaging receipt or frame-window sidecar surfaced. |
| `/home/kan/Research/SpaMo` | Bounded scan of 29 eligible text files; filename inventory of root, scripts, preprocess and docs. Relevant metadata: `CODEBASE_REPRODUCE.md`, `VENV_REPRODUCE.md`, `configs/local_paths.yaml`, `configs/finetune.yaml`; remaining doc is `docs/superpowers/specs/2026-09-18-how2sign-pipeline-design.md`. Download/unpack instructions in `CODEBASE_REPRODUCE.md:23–69` identify the external release and local destination, not its historical extraction. Previously assessed README/extractor/helper conventions add no run receipt. `dataset/` contains loader code, not a linked original-frame receipt directory. |
| `/home/kan/datasets/spamo` | Outside `hf_cache`, only three text metadata files found: `features/manifests/phoenix14t_motion.json`, `features/manifests/phoenix14t_spatial.json`, `ckpt/despamo-target-schema.json`. Filename inventory found no other file types besides JSON, ZIP, NPY and checkpoint files. Both manifests record only `clip_id`, `dtype`, `length`, `path`, `split`, `width`; neither records row windows, ordered source frames, extraction invocation or packaging lineage. Target schema contains only `state_dict`. |
| `/home/kan/Research/DeSpaMo` | Bounded scan of 163 eligible text files, plus focused project provenance metadata. Inspected `scripts/index_features.py`, relevant definitions in `src/despamo/data/manifest.py`, all six artifact JSONs listed below, `.superpowers/sdd/training-provenance-report.md`, and storage cleanup/recovery JSONs. No original motion extraction/index-to-source/copy receipt surfaced. |

Project artifact JSONs checked structurally:

- `artifacts/phoenix14t_smoke/run_metadata.json`
- `artifacts/phoenix14t_smoke_retry1/run_metadata.json`
- `artifacts/phoenix14t_smoke_retry2/run_metadata.json`
- `artifacts/phoenix14t_baseline/upstream.json`
- `artifacts/phoenix14t_baseline/deterministic-1.json`
- `artifacts/phoenix14t_baseline/deterministic-2.json`

Other project receipt candidates checked: `docs/storage-cleanup-candidates-2026-10-03.json`,
`docs/storage-cleanup-receipt-2026-10-03.json`,
`docs/storage-cleanup-fusion-manifest-2026-10-05.json`,
`docs/storage-cleanup-fusion-receipt-2026-10-05.json`, and
`docs/storage-recovery-2026-10-05.json`. These describe later checkpoint cleanup
and experiment recovery, not original motion extraction.

## Genuine evidence found versus missing

Newly checked run metadata binds downstream consumers to motion cache paths and
manifest hashes; baseline metadata also binds checkpoint conversion provenance.
The index script validates cache shape/count and writes a feature manifest.
These are genuine cache-consumption/indexing records, not evidence of which
historical source window produced released row `j`. Download instructions are
instructions, not an actual extraction run log. No new evidence established
historical window parameters, row chronology or 256x256-to-210x260 frame identity.

## Exact external inputs still required

For the user's chosen authentic-original-record route, obtain the publisher's
original extraction log/receipt **and any accompanying source-order and packaging
records**, tied to the released `mo_feat_p14t.zip` identified in the prior report:

1. Actual generating run identity: command/resolved parameters and extractor,
   helper, annotation and model versions/identities; window size, stride/overlap,
   tail handling and short-clip padding actually used.
2. Clip-specific mapping from released row `j` to ordered original frame paths,
   indices or timestamps, with source identities sufficient to connect historical
   256x256 motion frames to current 210x260 spatial chronology for all 7,096
   physical-train clips. A contemporaneous ordered frame manifest plus run log
   may establish this without a literal per-row table, if it unambiguously
   determines each row's window.
3. Original output-to-release packaging/renaming lineage, including removal of
   overlap filename suffixes and any row reordering/postprocessing, bound to
   released file/archive identities.

A generic script, default, matching count or download receipt is insufficient.
Unless these records can be supplied from an unsearched original local copy,
the user must request them from the external publisher. No assumption-based
attestation or real-data timing artifact was created.
