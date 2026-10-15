# Ordinary-B: experiment checkpoint storage and migration

## 1. Status, intent and boundaries

Written design for user-requested checkpoint relocation. User explicitly includes
older DINO/fusion/pilot experiments and wants ordinary files on B:, without a new
ext4 virtual disk. Code publication is already committed, merged and pushed as
`3a5bacfcb0f1830884a2677a9983dcd2ea637890` on `feat/global-local-rvlf`.

User selected **Approve written design** after reviewing this file. This approval
permits implementation planning, not checkpoint movement or GPU cutover.
This document creates no product code,
checkpoint movement, deletion, operator hold or new GPU scope. After approval,
write implementation plan; implement natively only after that plan is reviewed.
No commit/push of these new documents is inferred from prior publication task.

Success: all admitted remaining experiment checkpoint bundles are preserved on
ordinary B: storage; consumers validate the same bytes/identities through an
explicit relocation record; active reference training resumes the same lineage
after a reviewed healthy handoff; new checkpoint payloads publish on B:.
All future published full resumes remain preserved on B: even when only latest2
are used as working resume pointers. No training-duration limit/reset is added.

Checkpoint relocation reduces C: use. It cannot guarantee free space against
unrelated Windows/WSL writers. Enforce50GiB reserve for operations this system
admits, with explicit storage stop/hold handling when observed capacity is low.

Six-run CLIP fusion orchestration and new global/local model/loss are separate
follow-up work. This design changes storage/continuation, not model or recipe.

## 2. Evidence and immutable running state

Inspection2026-10-12: `/mnt/b/DeSpaMo` reports filesystem `9p` (Windows drive).
Existing `tools/dinov3/storage.py::writer` calls Linux-native `verify_location`;
its `_atomic` unconditionally directory-fsyncs. Individual snapshot/receipt
symlinks are rejected by `tools/clip_s2_safe/{policy,state}.py`. A direct move plus
file symlinks therefore does not satisfy active-run admission/publication.

Existing `tools/dinov3/official_archive.py` already supports ordinary B: copies:
file flush/fsync, same-directory temporary→final rename, rehash verification;
its JSON publisher records whether directory fsync is supported. Reuse these
principles without editing current checkpoint-bound helpers.

Read-only inventory found126 candidate payload files/~369GiB allocated under
experiment families. Counts are discovery evidence, not approved deletion list:
Torch feature tensors, missing sidecars and unreceipted legacy payloads require
classification before migration. Already-pruned resumes cannot be reconstructed
or claimed preserved unless an existing exact archive is found.

Active run at12:39:31UTC: parent193877, stage15966→17740, progress16600/durable15966.
Never pin cutover to that observation; discover latest healthy state at handoff.
Current source/recipe remains frozen under binding8298146c/93 source files.

Original study:
`/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v3`.
Original run basename:
`translation-reference-seed0-fe5c22c66305d9e3bea3074792c0f64955b2c01c4b61bf92656448c0f7ae5671`.
Current runtime is that run's `safe-runtime-v1/`; original budget is run `budget.json`.
Original epoch2 anchor remains `resume-epoch-2.pt`, SHA
`d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.
Existing B: candidate archive is
`/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-clip-s2-spamo200-v3/`
followed by run basename and `safe-runtime-v1/`.

## 3. Scope and layout

Discovery source base: `/home/kan/datasets/despamo`.
Included experiment families:
- `comparison-2026-09-26`, `comparison-smokes-2026-09-26`;
- `official-phoenix14t-dino-adaptation-v1`;
- `official-phoenix14t-translation-clip-s2-spamo200-v1`, `-v2`, `-v3`;
- `official-phoenix14t-translation-spamo200-v1`;
- `official-phoenix14t-translation-v1`;
- `signer-pilot`, `unreviewed-factor-100step`, `unreviewed-factor-smoke`.

Include experimental full resumes, model-only candidates, selected weights,
trained encoder exports and admitted profile-reload snapshots, with every required
receipt/runtime/guard/adoption sidecar. Preserve completion/selection/test-claim
and budget records as evidence; small authority metadata may remain on C:.
Exclude raw frames/features, annotation/target data, pretrained model caches,
installed environments, source checkouts, papers and unrelated user files.

Ordinary payload root: `/mnt/b/DeSpaMo/checkpoints-v1` (`B:\DeSpaMo\checkpoints-v1`).
Default migrated path: `<payload-root>/<source-relative-path-from-despamo>`.
Example active full resume:
`<payload-root>/official-phoenix14t-translation-clip-s2-spamo200-v3/runs/`
`<run-basename>/safe-runtime-v1/checkpoints/production/resume-step-<step>.pt`.
New active payloads use the same family/run prefix and
`safe-runtime-b-v1/checkpoints/production/` or `safe-runtime-b-v1/selected/`.

Native control root: `/home/kan/datasets/despamo/checkpoint-storage-v1`.
It owns manifest versions, operation journal, leases and committed registry head.
New current-run metadata: original run's `safe-runtime-b-v1/` on C:.
Existing metadata/migration/source archives/old journals stay as historical evidence.
Reuse an existing exact B: copy through a registered actual destination only after
checking complete bundle hashes; never delete/overwrite a differing destination.

## 4. Components and source isolation

New namespace `tools/checkpoint_storage/`:
- `inventory.py`: discover/classify checkpoint bundles and derive deterministic
  destination names; no deserialization of untrusted/unclassified payloads.
- `publish.py`: file-flush/rehash/same-directory B: publication and committed
  bundle manifests; record truthful durability tier.
- `registry.py`: native single-writer leases/journal/immutable registry versions.
- `resolver.py`: resolve logical original paths to registered B: files and validate
  original byte/receipt/identity evidence.
- `migration.py`: dry-run, inactive copy/verify/remove and rollback orchestration.
- `handoff.py`: reviewed operator hold, child/parent exit and same-run continuation
  evidence; no blind PID kill or automatic retry.
- `cli.py`: explicit inventory/copy/verify/remove/restore/handoff commands.

New `tools/clip_s2_b/` continuation adapter owns policy/state/accounting/train/dev/
supervisor integration where old helpers hardcode C: paths/native snapshot policy.
Files are `policy.py`, `state.py`, `accounting.py`, `train.py`, `evaluate.py`,
`study.py` and `config.json`, with CPU tests in that namespace's `tests/`.
It composes unchanged model/update/sampler/stopping/cached-generation primitives.
Do not global-monkeypatch old modules, weaken native filesystem whitelist, rewrite
old config hashes, change source bytes in live worktree or relabel old reports.

Implementation uses isolated workspace and existing main/motion/DINO environments.
No dependency changes. For continuation, pin unchanged75 shared training sources,
original FLAN/data/initialization/recipe/evaluation signatures and new storage code.

## 5. Bundle, identity and relocation contracts

Each inventory entry binds: stable artifact ID, logical original absolute path,
actual source path, destination, experiment/run/arm/seed/step when known, layout
class, byte count/SHA256 for every bundle member, original identity and receipt
digests, inode/link observations, owner/lock observations and classification.
Legacy unreceipted checkpoint may be retained as byte-verified archival evidence;
it never gains fabricated original provenance or automatic runnable admission.

Resolver accepts only committed registry entries and regular non-symlink target
files contained within configured B: root or explicitly registered existing
archive roots. Recheck size/hash/receipt and reject path traversal, symlink escape,
Windows case-fold collisions/reserved names, wrong seed/arm/step, missing sidecars
and differing target bytes. Existing origin JSON/report/test-claim files keep their
original bytes and logical paths; relocation mapping is new supplemental evidence.

Consumers that reject moved logical paths must use relocation-aware validation
or verified explicit restore to original native location. Do not promise arbitrary
old launchers work through file symlinks; retired controllers remain held.

## 6. Publication, removal and interrupted operations

One operation writer, native authority leases, sequential copying by default.
For each bundle:
1. Snapshot source inventory/stat/hash and prove no active owner, writer lease,
   publication temporary file or source mutation.
2. Check B: capacity; copy to unique temporary files in final B: directory.
3. Flush/fsync each file, rename within B:, rehash all destination files.
4. Publish bundle manifest last and register verified bundle in native journal.
5. Recheck source unchanged and final destination complete before any removal.
6. Remove only approved C: bundle members; record removal result per member.

States: `identified → copied → verified → registered → native_removed`.
Any interruption before registration leaves source authoritative. An interruption
during removal leaves committed B: bundle available and records partial native
cleanup for idempotent review. Never remove data from a hash-mismatched/partial
bundle, overwrite foreign archive, erase orphan evidence or restart GPU work.

B: durability tier is file-flush/rehashed-copy/atomic-rename unless verified
directory flushing is supported. Unsupported directory-fsync is explicitly
recorded only for B: publication; do not suppress arbitrary errors or claim Linux
directory durability. On restart, revalidate every consumed bundle; missing B:
drive/marker/bytes fails closed even if native registry once said verified.

## 7. Healthy active-run cutover and budget continuity

Inactive experiments migrate first after actual owner/lock checks. Active v3
checkpoint/selection files are not moved while current workers own them.

Existing supervisor has no clean pause CLI. Use an explicit reviewed operator
hold at its native worker boundary: current admitted child finishes normally,
parent validates result/publishes accounting/selection, next worker admission
fails before CUDA access, and parent exits held. Preserve operator request and
all actual result/exit events. Distinguish planned hold from device failure; if
current child fails or no healthy durable checkpoint exists, cutover stops.
The handoff command requires exact run/token/owner identities and explicit
`authorize_handoff` input; preparing inventory/design never places that hold.
If the original run has already completed normally, validate completion instead
and perform storage migration without inventing another training continuation.

After exact parent/child PID/boot/start-tick exit checks and locks release, reconcile
dead attempt immediately under current native accounting rules. Retain all elapsed
charges, completed/durable/probe counters and replay history. Require clean
checkpoint boundary with completed==durable; no migration-induced update replay.
Any already-started device fault remains primary and is not masked as storage pause.

Capture sealed handoff containing latest full resume/candidate, sampler/update
position, optimizer groups/moments, prompt/global RNG schema, last200 CE guard,
all admitted dev reports/selected pointers, stopping state, old binding/source
proof, budget/head/last committed intention and consumed prior scope.

New `safe-runtime-b-v1` binding references that handoff and relocation manifest.
Existing tensor identity stays unchanged. Read old reports against their archived
original bindings and newly mapped physical payloads, preserving report bytes.
Append an explicit storage-continuation transaction to original authoritative
budget; change runtime/authority pointer only, continue ledger sequence/previous
head proof and retain historical native journal. No new zero-cost budget or
replacement training lineage. New scopes are unique and boot/source/run-bound.

CPU-audit relocated checkpoint first. After reviewed admission, fresh B:-aware
training process restores exact state once; next dev uses fresh optimizer-free
batch1/cacheTrue/beam5/max64 process after actual train exit. Preserve original
max200/min40/dev2/patience10/minDelta.10, LR/VT warmup, model/data/ICL and losses.
All new full resumes/candidates/selections publish on B: and remain retained there.
Latest2 controls working resume pointers, not destruction of sole B: full resumes.

## 8. Capacity policy and WSL reclamation

Reserve50GiB on C: for native authority metadata/logs, and50GiB on B: plus actual
pending bundle/temp/publication demand. Compute distinct bytes by physical volume;
do not charge B: payloads as C: allocations. Reuse exact existing copies only when
their inventory is verified. Measure staging, checkpoint layout and metadata
growth; do not reuse a VT-only size as universal joint/full-model payload forecast.

Check capacities before each bundle and worker admission; B:-aware workers monitor
capacity at safe operation boundaries. Low space produces explicit storage hold,
no next GPU update/worker or speculative cleanup. Save only in healthy explicitly
supported pause path; never make CUDA/RNG/save calls after a device error.
External disk activity can cross reserve between observations; do not promise
absolute free-space guarantees or kill arbitrary Windows processes.

After verified inactive/native removals, approved online `fstrim /` may reclaim
unused sparse WSL blocks. Measure Linux and Windows C:/B: before/after; trim bytes
are not reported as actual Windows recovery. No WSL shutdown/compaction while GPU
work is active. No new virtual disk, filesystem-format operation, TDR/driver/device
changes or relocation of raw datasets/model caches.

## 9. Rollback

Before native removal: stop operation; source still available, retain partial B:
files and journal for review. After removal: verify committed B: bundle, ensure no
current writer/GPU owner, check C: restore demand+50GiB reserve, restore exact bytes
and original sidecars atomically, and publish explicit rollback record.

For failed B: continuation: preserve primary failure, B: payloads and all new
charges/updates. Do not restore old budget bytes or blindly launch old controller.
If native rollback is possible, restore newest admitted full resume from B: and
latest ledger state under new reviewed continuation scope, with truthful storage
binding. If B: is unavailable and native copies are gone, hold for restoration;
no fake successful rollback or unverified fallback checkpoint.

## 10. Acceptance and test matrix

- Discovery excludes raw features/pretrained caches; layout classification and
  existing archive reuse retain original identities, including legacy unreceipted
  distinction and once-only held-out registries.
- Traversal/case-fold/reserved-name/symlink and duplicate-artifact rejection.
- Byte-exact copy for full resumes/candidates/selected weights/encoder exports,
  sidecars and old receipt/report files; source mutation blocks deletion.
- Simulated copy/flush/rename/manifest/journal/removal interruption and real small
  ordinary-B: file probes; no native deletion on incomplete verification.
- Native locks reject second writer; active/stale-boot/reused PID cases behave
  correctly; no GPU construction during inventory/CPU migration verification.
- Relocation-aware checkpoint loads preserve all tensors, optimizer moments/groups,
  sampler, prompt/global RNG and CE guard. Cross-run/seed/arm/layout mismatch rejects.
- Operator-boundary hold/actual exit/reconciliation preserve counters, costs,
  reports/selection and old source bindings; no duplicate child or automatic retry.
- CPU continuation equals uninterrupted tiny-model lifecycle through joint phase;
  a separately admitted bounded real B:-backed train→exit→cached-dev handoff verifies
  new backend before uninterrupted long-run re-admission.
- Correct C:/B: reserve accounting and exact rollback; dependency/driver/model/data
  bytes unchanged. Six-run/global-local behavior is not claimed implemented.

## 11. Implementation handoff

After user reviews this written design, implementation plan decomposes into
inventory/registry, B: publisher/resolver, inactive migration/rollback, B:-aware
continuation and healthy handoff, then verification/operational rollout tasks.
Use native implementation/self-review, focused TDD and existing interpreters.
No time estimates, subagents, automatic dependency updates or unrequested commits.
