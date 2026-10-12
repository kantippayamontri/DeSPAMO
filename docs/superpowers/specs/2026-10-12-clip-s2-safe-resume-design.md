# Retained CLIP+S² checkpoint with isolated safe dev processes

## Status and intent

User-approved written design for the "do it" request after successful full519-clip
cached dev validation. Goal: prepare a CPU-tested integration that can later
continue retained S² training without the legacy in-process evaluator. No GPU
handoff test or training restart is authorized by this design work. Implementation
plan: `docs/superpowers/plans/2026-10-12-clip-s2-safe-resume.md`; plan review
was subsequently approved for native CPU implementation. Verified implementation
receipt: `docs/clip-s2-safe-resume-implementation-2026-10-12.json`. Production
preparation and GPU/training validation remain separately gated.

Success means exact retained-state loading, provenance-bound new evaluation,
actual train-process exit before dev, monotonic cost accounting, and fail-stop
error handling. It does not mean the Windows/GPU fault is diagnosed or fixed.

## Current evidence and immutable anchor

Original study:
`/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v3`.

Original run:
`runs/translation-reference-seed0-fe5c22c66305d9e3bea3074792c0f64955b2c01c4b61bf92656448c0f7ae5671`.

Anchor: native `resume-epoch-2.pt`,1774updates, SHA256
`d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.
Its original identity, schema1 snapshot/receipt, trainable weights, persistent
buffers, AdamW, sampler and prompt/global RNG are retained. Original budget is
interrupted/finalized,1774updates,497.5533489369991s, SHA256
`9e2a3bf8d122f1b91947700b3618418b8c8289db6bc91f1ff889e30ff1b5ddf7`.
All84 recorded source files still match. No production dev reports or selections
exist for this v3 run. Native epoch1/2 resumes must remain intact.

Current manual-stop SHA256:
`9b0f3dfebfbb67cbd7e4644ca6d3f67aa6c75dc34214adfc83b209023faeb370`.
Status remains stopped_by_user, automatic recovery disabled, GPU diagnostic
permission false. All earlier diagnostic scopes are consumed.

The full519 cached diagnostic passed, including31887 finite raw-logit checks,
peak6.029GiB allocated/6.604GiB reserved, with GPU audio enabled. It used no
optimizer and did not exercise further training or joint CE. See
`docs/clip-s2-cached-full-dev-validation-2026-10-11.md`/JSON.

## Recommended architecture

Add a sibling `tools/clip_s2_safe/` in the retained adaptation worktree. Compose
unchanged original S² model/data/update/checkpoint primitives with proven cached
generation and process sequencing. Keep variant-specific identity, accounting
and report admission adapters rather than changing plain-CLIP's1024 policies.

This avoids modifying the84 checkpoint-bound files and avoids broadening the
already-tested plain-CLIP experiment. Do not patch old hashes, clear the old
controller's stop, reset a budget, slice checkpoint weights, change feature
representation, create a fresh experiment root or initialize a replacement model.

Reuse:
- `tools/clip_s2/data.py`: full2048 spatial package and unchanged motion.
- `tools/followup/model.py`: original model factory and update kernel.
- `tools/benchmark/translation_model.py`: same optimizer construction/order.
- `tools/benchmark/compact_checkpoint.py`: strict original snapshot overlays.
- `tools/dinov3/official_sampler.py`: original sampler state validation.
- `tools/clip_plain/generation.py`: dimension-agnostic cached/finite-logit core.
- Pure stopping/guard/storage/owner primitives, not old controller or score_dev.

New code owns S² runtime binding, jobs, report validation and parent accounting.
No import-time global monkeypatch, policy-constant mutation or identity bypass.

### Runtime-owned namespaces

Use a new directory **inside the same original run**:
`safe-runtime-v1/` containing immutable migration/runtime bindings, attempts,
new full snapshots, candidates, safe dev reports, selection pointers and runtime
state/hold. Matching archive subdirectory is under the original B: study/run.

Original epoch1/2 files and old identity/report/controller artifacts are unchanged.
The old controller cannot discover safe-runtime snapshots as its legacy resumes.
The legacy study manual-stop remains present so the old runner remains blocked.

No production directory is created during CPU implementation/testing. Temporary
fixtures exercise migration/publication and ownership contracts.

## Checkpoint and runtime compatibility

Retain exact original checkpoint identity for tensor compatibility. Add immutable
**runtime binding sidecar**, separate from that identity, with:

- Original identity digest, anchor checkpoint/receipt and initial budget hashes.
- New runtime-source closure and configuration hashes, existing dependencies and
  package versions, pinned base-model hashes and frozen fingerprint.
- S² representation/model variant and original feature/annotation/order bindings.
- Evaluation policy ID `clip-s2-cached-b1-beam5-v1`, batch1/cacheTrue/beam5/max64,
  deterministic decoding, original train-only exemplar and telemetry schema.
- Storage/accounting/selection contract versions and no-automatic-retry policy.

Only the explicitly allowlisted original epoch2 anchor may enter without a safe
runtime sidecar. Every subsequently produced snapshot needs both original strict
receipt verification and the matching runtime sidecar; source/config drift fails.
Never claim that the original cacheFalse evaluator source produced the new report.

Load weights on CPU with the original identity and base fingerprint. Training
creates model/device/optimizer first, then restores AdamW, sampler, Python/NumPy/
Torch CPU/CUDA and prompt RNG **once**, after successful initialization. Validate
snapshot step==sampler.batches_yielded and absolute learning-rate position. Restore
persistent buffers, not just trainable parameters. Evaluation overlays model
weights only and never constructs/restores optimizer or global/CUDA RNG.

Raw CUDA RNG bytes can be checked on CPU but their native device restoration
requires separately approved GPU handoff validation; CPU success is not that proof.

## Training and process sequencing

Preserve seed0 reference recipe:2048→1024→768 spatial MLP, unchanged motion/
temporal/2048 connector, pinned FLAN-T5-XL and Q/V LoRA16/32/dropout0.1. Train
batch8/accumulation1/drop-last,887updates/epoch,max200,min40,dev every2,patience10,
minDelta0.10; VT4000/LRwarmup10000,legacy_mean VT, original optimizer/LRs/clipping,
training cacheFalse and gradient checkpointing. No warmup or update-index reset.

Initial production runtime state at1774 is **awaiting epoch2 dev**. The prior full
diagnostic remains evidence, not a production selection: do not automatically
promote/relabel it. A later approved production dev worker writes a report under
the new evaluation binding before training step1775 is permitted.

```text
verified anchor1774 -> pending production dev2 -> admit/select/stop
                                           -> train chunk -> durable handoff
                                           -> actual train child exit
                                           -> optimizer-free cached dev
                                           -> admit/select/stop -> next train chunk
any error -> held + permission consumed; no retry
```

Normal training chunks end at next declared dev boundary. Publish full resumable
state each epoch and model-only candidate at dev boundary, with atomic/fsynced
receipts and runtime sidecars. Parent waits/reaps actual child exit and verifies
owned locks/handoff before creating inference child. Inference has its own
per-batch BF16 autocast and never shares live training RNG or CUDA context.

Retain original frozen-base, finite-loss/gradient and CE-collapse checks. Persist
the last200 eligible CE observations and their absolute last-update position in
runtime-owned guard-state sidecars at durable checkpoints; fresh workers must
not reset the collapse window. The1774-update anchor has no joint CE history and
may initialize that window empty. Bind sidecars to the exact snapshot/runtime
hashes. Post-warmup identical-dev-output check also remains fail-stop. Any guard
exception, including old RecoverableFault types, stops rather than enabling retry.

Expose a separately scope-bound bounded handoff-validation job for future GPU
review: receipt declares exact start checkpoint, maximum update count and clip
limits. Validation artifacts/results are diagnostic, never selection. Executed
validation updates/costs are retained in the journal; adoption of its checkpoint
as production continuation requires explicit reviewed commit of that handoff.
No automatic validation training or scope expansion during implementation.

Current integration targets retained seed0 only. Do not initialize Seed1, claim
two-seed completion/paired averages, or score test. A future Seed1 scope is separate.

## Budget, ownership, storage and recovery

The original run's budget stays authoritative: initial497.5533489369991s and
1774updates are not copied to a new zero-cost budget. On future approved execution,
the CPU parent alone charges new stage setup/training/dev and failed normal
attempts; identity remains original, with additional runtime-binding reference.
Append runtime attempt/progress journal before work and retain initial raw budget
bytes as migration evidence. No parent/worker double writer or double charging.

CPU implementation must leave original budget/hold/checkpoint bytes unchanged.
Production budget updates happen only inside explicit GPU-execution scope.
Preserve failed update high-water evidence; a manually reviewed durable replay may
lower current step to saved sampler position but never lower charges. Backwards
wall time rejects reconciliation and preserves earlier accounting.

Use exact PID/boot/start-ticks/command identity and nonblocking same-run ownership
locks. Require an explicit external approval bound to runtime/anchor/hold/driver
review, current boot, allowed stages, maximum updates/clips and unique token.
Child tickets bind exact live parent and stage file. Validate before any CUDA/
NVML query, model construction or transfer. No source modifications while active.

Keep old study stop unchanged and add runtime-local fail-stop gate. No code can
manufacture approval, use diagnostic permission as training permission or auto
clear either hold. No watchdog/recovery daemon; dead/old-boot jobs require manual
reconciliation and a newly reviewed scope.

Check native and Windows backing space: projected remaining demand plus50GiB
reserve, including native staging/resume retention and B: dev candidates. Require
hash-verified atomic publication before optional explicitly permitted pruning of
new-owned resumes. Keep latest2 new full resumes, all dev candidates and actual
best weights. Never delete original epoch1/2 or previous study artifacts.

## Dev reports, selection and failure behavior

Batch1/cacheTrue/beam5/max64; original official519 IDs/order/references/ICL.
Fsync clip/decoder-start evidence before compute; finite valid visual tokens/raw
decoder logits, shapes/lengths/dtypes/strides, timing and allocator peaks. No large
GPU tensor histories. Core helper tag may be retained as implementation metadata;
report's explicit S² policy/binding is authoritative, never plain-CLIP policy.

Complete reports bind original checkpoint identity/hash, runtime/evaluation/data
hashes, source closure, exact corpus/order and train-only exemplar. Recompute
metrics from stored predictions; reject missing/reordered/duplicate clips,
stale candidate/telemetry or different-policy reports. Bounded reports remain
diagnostic/null metrics even if the limit happens to cover the corpus.

Selection/stopping live under safe runtime namespace, never legacy
`dev-epoch-*.json` or selection pointers. Use actual maximum dev BLEU4 for best
checkpoint, earlier ties, separate significant-improvement anchor for patience,
original minimum/VT-warmup eligibility. No mixed-policy report selection. Full
diagnostic validation receipt is prerequisite evidence, not a selection shortcut.

First fault stops worker/parent sequencing. Preserve primary error even when log,
accounting or hold-publication fails; secondary failures attach notes. No CUDA
RNG restore, memory/sync/empty_cache, tensor transfer or checkpoint save after
device failure. Boolean mode/hook cleanup and CPU-only evidence are allowed.
No next GPU child or automatic retry. Retain unreceipted/partial artifacts and
block implicit recovery/overwriting.

## CPU acceptance and future execution gates

CPU tests use actual tiny T5/PEFT,2048-input adapter, AdamW, original sampler,
file-backed data and real subprocesses. Prove complete state restoration and
interrupted-versus-continuous next batches/prompts/updates; reference-disjoint
evaluation must not alter training RNG. Add injected observer/device/clock/
partial-publication/owner-reuse faults, exact source/data/policy rejection,
separate full-vs-bounded reports and no-launch/no-CUDA gates. Test CE guard-window
restoration across worker boundaries and rejection of stale guard-state sidecars.

Inspect native anchor tensor/optimizer/sampler schema and receipts on CPU without
instantiating full FLAN or restoring CUDA RNG. Preserve original84source, native
checkpoint/budget/hold hashes; plain-CLIP runtime remains separate/unchanged.
Run focused CPU tests, Ruff/Pyright and native self-review; no new dependencies,
commits, subagents, extraction, Windows settings or GPU work.

After written design review, create a native task-by-task implementation plan.
After CPU integration passes, separately review bounded training/checkpoint/
exit/dev handoff before production recovery. No long training launch is implied
by "do it" for CPU integration, by earlier519 dev permission, or by this document.

## Design review decisions

Review must confirm: same-root/seed0 anchor recovery rather than fresh training;
separate runtime sidecars/namespaces; original authoritative accounting;
no automatic promotion of diagnostic metrics; actual process-exit barrier;
unchanged old runner/hold and separate subsequent GPU authorization.
