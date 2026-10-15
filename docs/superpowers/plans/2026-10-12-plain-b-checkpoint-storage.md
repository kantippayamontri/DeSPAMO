# Ordinary-B: Checkpoint Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `executing-plans` for native task-by-task implementation. Steps use checkboxes. User preference overrides subagent/commit/workspace-deletion defaults.

**Goal:** Preserve all admitted remaining experiment checkpoints as ordinary B: files, with verified relocation, original accounting and healthy same-run continuation.

**Architecture:** Native C: metadata owns leases/journals; ordinary B: folders own immutable checkpoint bundles. New storage/resolver modules and a separately admitted CLIP+S² B-aware adapter compose unchanged model/update/sampler/stopping/generation primitives. Current frozen runtime is not edited; cutover uses reviewed healthy worker-boundary hold and original-budget continuation.

**Tech Stack:** Existing Python3.11/main torch2.0.1+cu117/Transformers4.32/PEFT0.7.1, standard library, existing separate motion/DINO interpreters, existing uv locks. No dependency installs/upgrades.

**Spec:** `docs/superpowers/specs/2026-10-12-plain-b-checkpoint-storage-design.md`, written design explicitly approved by user.

## Global Constraints

- Ordinary B: files only; no virtual disk, filesystem format or WSL shutdown/compaction while GPU work is active.
- Payload root `/mnt/b/DeSpaMo/checkpoints-v1`; native control root `/home/kan/datasets/despamo/checkpoint-storage-v1`.
- Included older experiment families and excluded data/cache categories are exactly spec §3.
- Current original run/identity/budget/charges/optimizer/sampler/RNG/CE guard and old reports stay truthful; no zero-cost replacement run or relabelled report.
- All future full resumes/candidates/selections retained on B:. Latest2 is working pointer policy, not deletion of sole B: payload copies.
- Verify complete bundles and unchanged source before native removal; retain partial/orphan evidence and rollback records.
- Reject symlink/traversal/case-fold/reserved-name/identity drift; do not weaken shared native filesystem whitelist or bypass scopes.
- C: and B: reserves50GiB each, plus actual volume-specific pending demand. No absolute guarantee against unrelated writers.
- Inventory and CPU implementation do not place operator holds or access GPU. Real handoff needs exact run/token/owner approval.
- No changes to active retained/main source, environments, model/data bytes or old migration receipts; no extra seeds/test/retries/driver/TDR/device actions.
- Native implementation/self-review, no time estimates, no unrequested commits/pushes/workspace deletion.

## Review Focus

1. NTFS/9p rename succeeds but a later sidecar/registry write fails: source remains until complete bundle registration; restart rehashes B:.
2. Source mutates, PID is reused or old scope advances during copy: removal/handoff rejects and no second worker starts.
3. Historical report embeds removed C: path: original report bytes stay unchanged; registered resolver validates matching original binding and B: payload.
4. Torn budget continuation or missing B: drive: preserve charged/counter high-water, fail closed and require reviewed rollback/reconciliation.
5. Full future resumes have larger joint optimizer state: use actual bytes and preserve all B: copies, not VT-only estimates or sole-copy pruning.

## Workspace, commands and evidence

- After plan review, verify existing isolated workspace or create `work/plain-b-checkpoint-storage` from pushed `feat/global-local-rvlf` baseline3a5bacf. Copy approved spec/plan exactly if new worktree lacks uncommitted documents.
- Ledger identity: `# SDD ledger — plan: docs/superpowers/plans/2026-10-12-plain-b-checkpoint-storage.md` under that plan's ignored SDD directory. Record interface/ruling/test evidence per task.
- Run CPU commands from implementation checkout with `CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1`.
- `PY=/home/kan/Research/DeSpaMo/.venv/bin/python`; tests use `$PY -B -m pytest ... -q`. No generic environment install.
- Preserve original93-source hash map and owner/scope status before/after code work. Never copy current runtime state into a fresh authority.

## File Map

- `tools/checkpoint_storage/{__init__,inventory,publish,registry,resolver,migration,handoff,cli}.py`: generic bundle discovery, B: publication and reviewed migration.
- `tools/checkpoint_storage/tests/{conftest,test_inventory,test_publish,test_registry,test_resolver,test_migration,test_handoff,test_cli}.py`.
- `tools/clip_s2_b/{__init__,policy,state,accounting,train,evaluate,study}.py`, `config.json`, `pyrightconfig.json`: current reference Seed0 B-aware continuation.
- `tools/clip_s2_b/tests/{conftest,helpers,cpu_child,test_policy,test_state,test_accounting,test_train,test_evaluate,test_study,test_lifecycle}.py`.
- `docs/plain-b-checkpoint-storage-runbook.md`, `docs/plain-b-checkpoint-storage-implementation-2026-10-12.json`: verified instructions/result, not premature migration claims.

No existing live-bound product file modification is planned. Six-run runner stays separate.

### Task1: Inventory, bundle schema and destination validation

**Files:** create storage package, `inventory.py`, `tests/conftest.py`, `test_inventory.py`.

**Interfaces:**
- Define `ArtifactMember` with role/logical path/source path/destination path/bytes/SHA256 and stat observations.
- Define `ArtifactBundle` with artifact ID/family/run/layout/original identity digest/members/classification/activity evidence; legacy unreceipted entries are archival-only.
- `discover_bundles(source_base: Path, payload_root: Path) -> list[ArtifactBundle]`.
- `validate_destination(bundle: ArtifactBundle, payload_root: Path) -> None`.

- [ ] Write `test_inventory_classifies_compact_lightning_encoder_and_legacy` with independent fixtures: full resume+receipt/runtime/guard, selected/candidate, encoder export, unreceipted legacy; assert all members included and only unreceipted fixture is non-runnable.
- [ ] Write `test_inventory_excludes_frames_features_pretrained_cache_and_live_copy` and `test_destination_rejects_traversal_symlink_case_collision_reserved_name`; assert non-checkpoint data excluded and conflicts reject.
- [ ] Run `$PY -B -m pytest tools/checkpoint_storage/tests/test_inventory.py -q`; expect missing-module/interface failures before implementation.
- [ ] Implement spec §3 family allowlist, byte/stat inventory, deterministic source-relative destinations and complete bundle classification. Do not unpickle arbitrary discovery candidates.
- [ ] Re-run Task1 tests; expect all pass. Record exact discovered-vs-admitted distinction; no real files moved.

### Task2: Verified B: publication and native registry

**Files:** create `publish.py`, `registry.py`, `test_publish.py`, `test_registry.py`.

**Consumes:** Task1 bundle/members and destination validation.
**Produces:**
- `publish_bundle(bundle: ArtifactBundle, control_root: Path) -> dict` containing complete B: member inventory/manifest hash/durability tier.
- `register_verified(control_root: Path, bundle: ArtifactBundle, publication: dict) -> dict` returns committed relocation entry.
- `read_committed(control_root: Path, artifact_id: str) -> dict` verifies registry head/entry.
- `require_capacity(control_root: Path, payload_root: Path, c_demand: int, b_demand: int) -> dict` checks separate50GiB reserves.

- [ ] Write `test_copy_flush_rename_rehash_manifest_then_registration` asserting source still exists after successful registration and destination bytes/receipts equal source.
- [ ] Write failure cases for copy/file-fsync/rename/manifest/registry journal; assert no registered incomplete bundle and no source removal. Unsupported directory-fsync EINVAL/ENOTSUP/EACCES records weaker tier; EIO propagates.
- [ ] Write `test_existing_matching_B_bundle_reused_foreign_bundle_rejected`, `test_second_registry_writer_rejected`, `test_restart_rehashes_missing_or_changed_B_file`, `test_capacity_charges_payloads_to_B_only`.
- [ ] Run `$PY -B -m pytest tools/checkpoint_storage/tests/test_publish.py tools/checkpoint_storage/tests/test_registry.py -q`; expect missing interfaces.
- [ ] Implement unique same-B-directory temporary publication, file flush/fsync/rehash and manifest-last completion. Native writer/intention/head sequence owns registry; reuse existing archive principles without monkeypatching shared helpers.
- [ ] Re-run Task2 tests; expect pass. Do not claim unsupported directory durability or power-loss-proof B: admission.

### Task3: Resolver, inactive migration, restore and explicit CLI

**Files:** create `resolver.py`, `migration.py`, `cli.py`, corresponding tests.

**Consumes:** Task1 bundles, Task2 committed entries/capacity/publication.
**Produces:**
- `resolve_checkpoint(logical_path: Path, control_root: Path) -> Path`: committed regular B: file with original hash/identity/sidecars validated.
- `migrate_inactive(bundle: ArtifactBundle, control_root: Path, *, authorize_remove: bool) -> dict`.
- `restore_bundle(artifact_id: str, control_root: Path, *, authorize_restore: bool) -> dict`.
- `main(argv: list[str] | None = None) -> None` exposes inventory/copy/verify/remove-native/restore modes, with explicit source/control/payload roots and operation authorization.

- [ ] Write `test_original_report_bytes_and_logical_paths_preserved`, `test_resolver_rejects_wrong_seed_arm_step_or_missing_sidecar`, `test_legacy_archive_does_not_gain_runnable_provenance`.
- [ ] Write `test_source_mutation_and_active_or_reused_owner_prevent_removal`, `test_registered_B_bundle_survives_partial_native_cleanup`, `test_restore_is_byte_exact_and_preserves_metadata`, `test_cli_inventory_does_not_move_or_remove_payloads`.
- [ ] Run `$PY -B -m pytest tools/checkpoint_storage/tests/test_resolver.py tools/checkpoint_storage/tests/test_migration.py tools/checkpoint_storage/tests/test_cli.py -q`; expect missing interfaces.
- [ ] Implement spec state machine `identified/copied/verified/registered/native_removed`, source rechecks and per-member cleanup proof. Validate current filesystem/source under lease; active owner remains active until exact exit/lock proof.
- [ ] Restore requires available verified B: bundle, no owner/writer and C: demand+reserve. Preserve registry and original report/test-claim bytes; never rewrite path-bearing receipts to pretend same producer.
- [ ] Re-run Task3 tests; expect pass. A missing B: drive produces hold, never unverified fallback.

### Task4: B-aware identity, checkpoint state and original-budget bridge

**Files:** create `tools/clip_s2_b/{__init__,policy,state,accounting}.py`, config/typing and policy/state/accounting tests.

**Consumes:** Task2 registry, Task3 resolver and sealed handoff dict schema defined here.
**Produces:**
- `make_continuation_binding(config: dict, handoff: dict, control_root: Path) -> dict`.
- `admit_checkpoint(logical_path: Path, binding: dict, control_root: Path) -> dict`.
- `publish_checkpoint(snapshot: dict, guard: dict, binding: dict, control_root: Path) -> dict` writes full/candidate bundles on B:.
- `commit_continuation(config: dict, binding: dict, handoff: dict) -> dict` appends explicit original-budget storage-continuation transaction.

Handoff schema includes original run/identity/binding/source proof, latest full resume/candidate digests, all old report/selection/stopping digests, original budget/head/last-intention proof, completed==durable/probe counts, charged floor/replay history, exact dead owners and consumed old scope.

- [ ] Write `test_relocated_checkpoint_restores_optimizer_sampler_rng_and_200CE`, `test_old_reports_admitted_against_original_binding_without_relabelling`, `test_original_anchor_relocation_is_registered_not_symlink_bypass`.
- [ ] Write `test_continuation_preserves_original_budget_identity_costs_counts_and_sequence`, `test_torn_bridge_fails_closed_with_before_after_proof`, `test_cross_run_binding_and_unadmitted_paths_rejected`.
- [ ] Run `$PY -B -m pytest tools/clip_s2_b/tests/test_policy.py tools/clip_s2_b/tests/test_state.py tools/clip_s2_b/tests/test_accounting.py -q`; expect missing modules/interfaces.
- [ ] Implement separately versioned policy/binding referencing old immutable bindings and relocation manifest. Original tensor identity stays exact; original C: budget remains authority; historical journal survives and new sequence references previous head.
- [ ] Implement B: tensor serialization through Task2 publisher, including original receipt and runtime/CE sidecars. No shared native whitelist modification; old scope stays consumed/held.
- [ ] Re-run Task4 tests; expect pass. Existing full-model construction/CUDA RNG operations are excluded from CPU audit.

### Task5: B-backed train/dev lifecycle

**Files:** create `clip_s2_b/{train,evaluate,study}.py`, CPU child/helpers and train/evaluate/study/lifecycle tests.

**Consumes:** Task4 binding/state/budget bridge, unchanged common model/update/sampler/stopping/cached generation.
**Produces:**
- `run_train_job(config: dict, job: dict, approval: Path) -> dict`, `run_dev_job(config: dict, job: dict, approval: Path) -> dict`.
- `run(config: dict, approval: Path, *, check_only: bool = False) -> dict` owns sequential B-aware continuation.

- [ ] Write real CPU subprocess lifecycle `test_B_cutover_matches_uninterrupted_joint_updates`: assert same weights/optimizer/sampler/prompt RNG/200CE, cost floor and production counter; no current reference reinitialization.
- [ ] Write `test_train_exit_precedes_optimizer_free_cached_dev`, `test_B_reports_preserve_beam5_max64_order_and_recomputed_metrics`, `test_future_full_resumes_preserved_on_B_with_latest2_working_pointers`, `test_low_space_and_publication_fault_do_not_start_next_worker`.
- [ ] Run `$PY -B -m pytest tools/clip_s2_b/tests/test_train.py tools/clip_s2_b/tests/test_evaluate.py tools/clip_s2_b/tests/test_study.py tools/clip_s2_b/tests/test_lifecycle.py -q`; expect missing components.
- [ ] Implement copied/composed sequencing adapters with original Seed0 reference recipe, original model/update and restore-once behavior; old reports validated via original bindings/resolver. No six-arm feature added.
- [ ] On low capacity, use explicit healthy pause contract; after device error perform no CUDA/RNG/save cleanup. Publish complete B: bundles before native authority advance.
- [ ] Re-run Task5 tests; expect pass. Scope caps and provenance reject duplicates/cross-run state; no automatic retry.

### Task6: Healthy operator handoff and authority seal

**Files:** create `tools/checkpoint_storage/handoff.py`, `tests/test_handoff.py`; extend storage CLI with request/seal-handoff modes.

**Consumes:** Task4 handoff schema and native/B-aware authority contracts; Task5 actual-exit sequencing evidence.
**Produces:**
- `request_handoff(source_config: Path, control_root: Path, *, authorize_handoff: bool) -> dict` returns exact operator-boundary hold request evidence.
- `seal_handoff(source_config: Path, control_root: Path) -> dict` returns Task4 handoff schema only after healthy result/owner exit and reviewed original-ledger reconciliation.

- [ ] Write real CPU subprocess tests `test_boundary_hold_waits_for_actual_child_exit_and_records_operator_reason`, `test_child_fault_blocks_migration_and_preserves_primary_error`, `test_completed_source_run_does_not_restart`, `test_wrong_boot_reused_PID_or_live_lease_blocks_seal`.
- [ ] Write `test_operator_reconciliation_charges_once_and_never_resets_counter` and `test_inventory_or_missing_authorization_cannot_place_hold`; assert same original budget identity/floor and no early payload deletion.
- [ ] Run `$PY -B -m pytest tools/checkpoint_storage/tests/test_handoff.py -q`; expect missing interfaces/CLI modes.
- [ ] Implement exact-owner hold through existing native worker admission, not source patch/blanket kill. Keep truthful held/interrupted outcome and consume old scope. Reconcile immediately after exact exits/locks release, before copying.
- [ ] Seal current full checkpoint/reports/stopping/source/head proof and require completed==durable. Actual device errors or torn unreviewed transactions stop cutover; completed original run is archived without new training.
- [ ] Re-run handoff/CLI tests; expect pass. No real operator hold is executed by these fixture tests.

### Task7: Integrated CPU/static verification and operator runbook

**Files:** create runbook/implementation receipt; extend isolated fixture tests only where integration reveals missing contract coverage.

- [ ] Run `$PY -B -m pytest tools/checkpoint_storage/tests tools/clip_s2_b/tests tools/clip_s2_safe/tests tools/clip_s2_inference/tests tools/clip_plain/tests -q`; expect zero failures.
- [ ] Run `/home/kan/Research/DeSpaMo/.venv/bin/ruff check tools/checkpoint_storage tools/clip_s2_b` and `ruff format --check` on those namespaces; expect clean.
- [ ] Run `/home/kan/.local/bin/pyright --pythonpath /home/kan/Research/DeSpaMo/.venv/bin/python -p tools/clip_s2_b/pyrightconfig.json`; expect0errors/0warnings, including storage imports.
- [ ] Perform native separate review pass against all five Review Focus cases and spec sections; each valid correctness finding gets focused RED→GREEN fix and relevant rerun. No speculative unrelated refactor.
- [ ] Verify all original93 live source hashes and existing environment/asset pins still match. Inspect intended git status/diff/check; no commit unless separately requested.
- [ ] Write runbook with exact CLI semantics, permission boundaries, rollback and a truthful CPU-only implementation receipt. Stop before real B: probes/moves if their scope is not admitted.

### Task8: Controlled real-storage rollout after reviewed implementation

**Consumes:** Tasks1–7 green interfaces/runbook; current native state discovered fresh, never hardcoded to initial16600/15966 observation.

- [ ] Obtain/record exact rollout approval covering ordinary B: file probes, reviewed inactive bundles/native cleanup, operator hold and same-run continuation; derive remaining max200/early-stop caps from original authority.
- [ ] Verify B: actual filesystem/path/capacity and native C: metadata root. Run small real file copy/flush/rename/rehash/registry/load/remove/restore probes in dedicated operation directory; preserve evidence, no production bundle yet.
- [ ] Run `inventory` with source `/home/kan/datasets/despamo`, payload `/mnt/b/DeSpaMo/checkpoints-v1`, control `/home/kan/datasets/despamo/checkpoint-storage-v1`; publish classified manifest and inspect every admitted group, existing archive reuse and source activity proof.
- [ ] Migrate admitted inactive bundles sequentially. Verify complete B: bundle/hash/receipt proof before each native cleanup; journal each operation and retain rollback path.
- [ ] Request current-run healthy operator hold with exact token/owner identities; wait current child/result/accounting/selection then actual parent/child exits and lock release. If actual device fault occurs, stop cutover for review.
- [ ] Seal/reconcile original authority immediately, then migrate active current-run checkpoint bundles and original anchors/selected weights. Re-admit storage binding without budget/source/tensor identity reset.
- [ ] CPU-audit full relocated state and old report/selection/registry bindings. Require completed==durable and preserved307 initialized optimizer state semantics/current200CE, not stale checkpoint assumptions.
- [ ] Run separately admitted bounded B-backed real train→checkpoint→actual-exit→cached-dev diagnostic. Provisional observations/costs stay journalled; do not silently adopt probe or replace production dev selection.
- [ ] After successful evidence, admit continuous B-aware reference Seed0 scope from exact production durable checkpoint under original recipe. Verify actual owned live progress and new B: checkpoint placement; current native scope stays retired.
- [ ] Apply approved online trim after native removals; measure Linux/C:/B: before/after, validate50GiB reserve and publish per-bundle migration/continuation receipts. No WSL shutdown/compaction during GPU work.
- [ ] Declare only observed rollout completion: all admitted remaining checkpoints preserved and B-aware continuation live/finished as actual state says. Unclassified/skipped/blocked items listed explicitly; no fabricated369GiB recovery or six-run implementation claim.

## Coverage and execution handoff

Spec §§1–3 → Task1/8; §4 → all component tasks; §§5–6 → Tasks1–3;
§7 → Tasks4–6/8; §8 → Tasks2/5/8; §9 → Tasks3–6/8;
§10 → all tests/Task7; §11 → native execution after plan review.
Review Focus1→Task2,2→Task3/6,3→Task3/4,4→Task4/6,5→Task2/5.

Native execution method is already user preference. This plan must be reviewed
before implementation. Spec approval alone did not authorize product-code changes,
operator hold or checkpoint cleanup. No new plan commit/push is inferred.
