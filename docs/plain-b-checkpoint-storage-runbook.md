# Ordinary B: checkpoint storage

Implementation namespace `tools/checkpoint_storage/` uses native C: operation
leases/journal and ordinary B: file-flush/rehashed bundle publication. No virtual
disk or dependency upgrade. `tools/clip_s2_b/` is a separately admitted same-run
reference Seed0 continuation adapter; old frozen runtime is not modified.

## Verification and scope

Before rollout, run storage/B-adapter CPU tests, old safe/inference/plain regression
tests, scoped Ruff/format/Pyright and native review. Real synthetic B: probe must
verify copy/flush/rename/hash/registry/native removal/exact restore. The synthetic
probe in `checkpoint-storage-v1/probes/ordinary-b-v1/result.json` passed with actual
directory-fsync supported; no user checkpoint was touched by that probe.

User-approved scope covers remaining older DINO/fusion/pilot and current CLIP+S²
experiment checkpoint bundles. Raw data/features/pretrained caches/environments
are excluded. Inventory discoveries are not automatically runnable provenance.
Original metadata/report bytes and source paths remain logical identities.

## CLI

From isolated implementation checkout with existing main interpreter:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/kan/Research/DeSpaMo/.venv/bin/python -B -m tools.checkpoint_storage.cli inventory --inventory /home/kan/datasets/despamo/checkpoint-storage-v1/inventory.json
```

Inspect classified bundles and actual source owner/locks. `copy` publishes and
registers; `remove-native --authorize-remove-native` additionally rechecks current
activity/source/destination and removes approved native members. Active entries
are blocked; refresh inventory after a completed handoff. `verify --artifact-id`
rehashes registered B: bundles. `restore --artifact-id --authorize-restore` requires
native restore demand+50GiB reserve and no live owner; it restores exact original
logical paths without resetting budgets.

## Current-run handoff

`request-handoff --source-config <original-config> --authorize-handoff` places an
explicit next-worker barrier. Existing child finishes; next admission fails before
CUDA, old parent exits held and scope is consumed. `seal-handoff` requires exact
dead owners, healthy operator-held state, original CPU accounting reconciliation
and completed==durable. A real device fault blocks migration/cutover.

After sealed current bundles migrate, create separately bound B: configuration and
storage-continuation journal transaction. Original C: budget/identity/costs/counters
remain authority. Old scopes stay retired. Validate relocated mutable weights,
optimizer/sampler/RNG/200CE and original dev selections before a newly reviewed
bounded B-backed train→actual-exit→cached-dev probe and continuous continuation.
No automatic adoption of diagnostic update or reset/restart-from-zero.

Payload root `/mnt/b/DeSpaMo/checkpoints-v1`; control root
`/home/kan/datasets/despamo/checkpoint-storage-v1`. Enforce50GiB reserve on both
backing drives and actual per-volume staging demands. Preserve all future full
resumes/candidates/selections on B:; latest2 controls working pointers.

## Rollback and limitations

Unregistered/partial B: data never permits native deletion. Committed registry
rehashes before load/removal; source mutation/foreign target/held writer rejects.
Interrupted cleanup retains verified B: bundle and per-member native removal
record. Restore only when C: capacity permits. Missing B: or torn budget bridge
holds for review; no unverified checkpoint fallback or original budget overwrite.
CLI/storage operations do not restart retired controllers, score test or introduce
new model/recipe. Six-run orchestration remains separate work.

## 2026-10-13 read-allocation recovery

Copy-only rollout stopped with ENOMEM while rehashing a B: destination. Kernel
evidence showed a 9p read-request allocation failure. A bounded unbuffered16KiB
read hashed the complete5,925,496,109-byte failed payload and matched its original
inventory SHA256. Both16KiB and4MiB prefix probes passed, so transient allocation
pressure remains relevant; smaller reads are mitigation, not a kernel-cause cure.

New storage hashing and copying use16KiB chunks. Read errors still propagate;
there is no automatic retry or bypass of flush/hash/manifest/registration checks.
The failed bundle subsequently registered successfully. Remaining inactive copies
are copy-only, with native originals retained until reviewed cleanup admission.
Receipt: `docs/plain-b-checkpoint-storage-read-allocation-validation-2026-10-13.json`.
