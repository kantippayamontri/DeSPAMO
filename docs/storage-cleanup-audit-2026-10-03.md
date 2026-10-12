# Storage cleanup audit before residual-projector experiments

Snapshot: 2026-10-03, approximately 15:14 +08:00. This is a proposed cleanup,
not a deletion receipt. The user approved the residual-MLP projector direction
with a longer-training control and requested a storage audit before starting.

## Current disk situation

| Filesystem | Available | Utilization |
|---|---:|---:|
| WSL Linux `/` (`/dev/sdd`) | approximately 339 GiB | 65% |
| Windows `C:` (`/mnt/c`) | approximately 25 GiB | 98% |

The Windows host remains the tighter constraint. Freeing Linux files increases
free space inside the WSL filesystem; it does not guarantee that Windows free
space immediately increases. Recheck host storage after cleanup and determine
whether WSL virtual-disk space reclamation is needed before a long run. Linux
`df` alone is not sufficient evidence that the previous crash cause is resolved.

No local Python training/extraction jobs were found during the audit. GPU was
idle at 2 MiB. The other server's description extraction is outside this audit.

## Recommended cleanup: 107.39 GiB

All candidates are older resume snapshots in completed runs under
`/home/kan/datasets/despamo/signer-pilot/`. Selected checkpoints were verified
against their recorded SHA-256 hashes for every candidate run. Latest recovery
snapshots and all dev-selection `step-*.ckpt` files are retained.

| Candidate group | Checkpoints | Associated receipts | Allocated space |
|---|---:|---:|---:|
| Cosine SpaMo run: `resume-500.ckpt` through `resume-7500.ckpt`, every 500 steps | 15 | 0 | 82.64 GiB |
| Constant-LR SpaMo run: `resume-7000.ckpt`, `resume-7500.ckpt` | 2 | 0 | 11.04 GiB |
| Original long DINO adaptation: steps 500 through 35500, every 500 | 71 | 71 | 8.85 GiB |
| Constant-LR 8975-step DINO adaptation: steps 500 through 8500, every 500 | 17 | 17 | 2.12 GiB |
| Cosine 8975-step DINO adaptation: steps 500 through 8500, every 500 | 17 | 17 | 2.12 GiB |
| Five 718-step short adaptations: each `resume-500.pt` and its receipt | 5 | 5 | 0.62 GiB |
| **Total** | **127** | **110** | **107.39 GiB** |

Total: **237 explicitly identified files**, occupying **115,309,707,264 bytes**.
Sizes use allocated blocks and candidate files have a single hard link. No
directory-level removal or broad wildcard deletion is proposed.

The exact run keys, steps, preserved filenames, metadata fingerprints and
verified selected hashes are in
[the cleanup manifest](storage-cleanup-candidates-2026-10-03.json).
Each listed step expands only to `resume-<step>.ckpt` or `resume-<step>.pt` in
that specific run. Only the listed DINO snapshot steps also include their
corresponding `resume-<step>.json` receipts.

Pruning loses rollback/reinspection of those intermediate training states.
It retains completed results, selected models, and the latest saved recovery
point for each completed run. Before deletion, recheck run activity, metadata
fingerprints, retained checkpoint hashes and newly added references; stop if
the audit snapshot no longer matches.

## Protected files and directories

- Every `step-*.ckpt`, including selected E1/E2/E3 checkpoints and earlier
  dev-selection checkpoints.
- Translation `resume-8000.ckpt` in the two completed E3 candidate runs.
- DINO selected/latest `resume-35900.pt`, both `resume-8975.pt` checkpoints,
  all selected `resume-718.pt` checkpoints, and their receipts.
- All run budgets, selection receipts, protocol files, dev/Signer07 reports
  and probe reports.
- All DINO/motion feature caches, raw frames, appearance descriptions,
  dataset/text manifests, and model caches required for the next experiment.
- Both incomplete E1 run directories. The older E1 run has two snapshots
  totaling 10.97 GiB; the retry has one approximately 5.48-GiB snapshot. Their
  unfinished status makes them separate recovery decisions.
- `artifacts/phoenix14t_smoke_retry2/lightning_logs/version_0/checkpoints/epoch=0-step=35.ckpt`
  (5.47 GiB). It is explicitly referenced by the project README.
- The `external-data` and `recovery-worktree` links and their targets. These
  are navigation links, not duplicate artifact copies.
- Existing source, papers, Git history and uncommitted user work.

## Why deleting project environments is a poor first choice

The largest in-checkout directories are `.venv` (4.40 GiB), `tools`
(4.97 GiB, mainly its extractor environment), and `artifacts` (5.47 GiB,
mainly the README-referenced smoke checkpoint).

Most files in the in-checkout environments are hard-linked, so removing an
environment does not necessarily recover its displayed size. The independent
temporary extractor environment occupies approximately 4.96 GiB but is the
currently documented pinned extractor environment. Snapshot pruning offers
much more space without disrupting those runtime paths.

The pilot directory occupies approximately 271.66 GiB. Existing feature caches
cost hours to regenerate and are only about 6.58 GiB each; the redundant full
translation snapshots are the high-value target.

## Storage policy for the next experiment

The current downstream callback saves a full checkpoint every 500 steps and
keeps every snapshot. An 8000-step run generated 16 snapshots totaling
approximately 88.16 GiB, in addition to three dev-selection checkpoints.

The new residual-projector experiment specification should include bounded
retention for newly generated resume snapshots, for example the latest two,
while retaining all dev-selection checkpoints and an atomic-write reserve.
Checkpoint retention and host-disk admission checks should be settled before
launching the approved downstream experiments; otherwise cleanup alone merely
postpones the same storage growth.

## Audit scope

The inventory did not follow project symlinks when counting in-checkout files;
linked pilot storage was inspected separately. It checked 198 reference
sources across project documentation/configuration/source and local run
metadata/feature identities. DINO intermediate receipt hashes were checked
against those references. Another 189 sources in the user-owned `main.py`
and the known recovery/signer-probe worktrees were checked for candidate paths
and DINO checkpoint hashes; no additional candidate dependencies were found.
The two completed translation run identities have
no resume-specific lineage references. Final selected checkpoint hashes were
verified in all ten candidate runs. Incomplete runs, referenced smoke output
and immutable dataset/cache storage were excluded from the proposal.

## Approved cleanup completed

After explicit user approval, the exact manifest was revalidated and executed
at 2026-10-03 15:42 +08:00. All 237 approved files were removed under exclusive
run locks. Selected checkpoint SHA-256 hashes matched before and after pruning;
retained checkpoint/receipt fingerprints and budget/selection hashes were
unchanged. See [the execution receipt](storage-cleanup-receipt-2026-10-03.json).

Allocated files removed: **107.390533 GiB**. Measured Linux available space
increased from **338.547 GiB** to **445.938 GiB**. Windows `C:` available space
remained approximately **24.489 GiB**, so the host-side storage constraint is
still unresolved. The original manifest remains the historical proposal;
the linked receipt records its user-approved execution.
