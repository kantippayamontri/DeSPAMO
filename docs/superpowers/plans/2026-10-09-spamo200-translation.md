# SpaMo-style 200-epoch Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native implementation/self-review follows existing user preference; no commits or delegated agents.

**Goal:** Stop the old study safely and launch six separately versioned max200epoch runs with the approved SpaMo-style recipe.

**Architecture:** Add `tools/followup/` to the retained isolated worktree; preserve every old benchmark and main `src/` byte. Reuse verified data, compact checkpoints, model construction, sampler and accounting. New training/evaluation/controller code owns new paths, prompting, schedule and early stopping.

**Tech Stack:** Existing main Python/torch2.0.1/transformers4.32/PEFT environment; no dependency changes.

**Spec:** `docs/spamo-style-translation-followup.md`, including latest user instruction to switch now.

## Global Constraints

- New root `/home/kan/datasets/despamo/official-phoenix14t-translation-spamo200-v1`; B archive with same suffix.
- Seeds0/1 × reference/aligned_fixed/aligned_learned, sequential; fresh initialization.
- Physical batch8, no accumulation;887updates/epoch;177400update ceiling.
- VT-only4000updates; ordinary padded mean; joint CE+VT thereafter.
- Linear LRwarmup10000updates; cosine adapter3e-4→5e-5 / FLAN5e-5→1e-5 at ceiling.
- Full519dev every2epochs; min40; patience10eligible checks; significant gain0.10absolute BLEU4; keep actual best, earlier ties.
- Three multilingual pairs from one reference-disjoint training partner, matching released English/French/Spanish=German formatting. Training partner sampled with checkpointed prompt RNG. Precommitted per-seed evaluation partner comes from train only, independent of held-out inputs; reject reference overlap rather than selecting using held-out references.
- Preserve all multilingual pairs without silent prompt truncation. Profile longest visual/target sequences with longest training exemplar. Activation checkpointing may be used to retain physical batch8 if memory requires it, bound into identity.
- Native latest2full resumes, selected best, all actual dev candidates hash-archived to B. Stop state derived from durable ordered dev reports for replay correctness.
- Freeze six selections before once-only642test; no old-study test scoring. Native/C:/B: demand plus50GiB reserve checked.

## Review Focus

- Inference prompts must not consume dev/test references; verify precommitted partner origin and fail overlap guard.
- Prompt tokenizer limit must not silently discard multilingual pairs.
- Small BLEU gains accumulate relative to significant anchor; actual best and anchor differ.
- Recovery after a dev report publication must reconstruct same stopping decision, without additional optimizer updates after stop.
- SIGTERM/manual stop must disable restart; no duplicate GPU owners.

### Task1: Recipe, prompting and update kernel

**Files:** Create retained-worktree `tools/followup/{__init__,recipe,model}.py`, `tests/test_followup_recipe.py`.

**Interfaces:** `schedule(batch_size)`, `group_lr(step,recipe,group)`, `stop_state(reports,recipe)`, `PromptPool(records,seed)`, `build_model(seed,arm,cache)`, `update(model,optimizer,batch,step,recipe,pool)`.

- [ ] Test exact schedule boundaries, finite scores, VT-only exclusion, min40, cumulative subthreshold gains and tie selection.
- [ ] Test training partner excludes all batch references; fixed inference partner selection ignores held-out references; pool origin and RNG replay.
- [ ] Implement using existing base model, legacy_mean VT pooling and explicit group LRs; keep three pairs and their clip IDs.
- [ ] Run focused tests RED→GREEN and Ruff.

### Task2: Training, evaluation and recovery

**Files:** Create retained-worktree `tools/followup/{train,evaluation}.py`, `tests/test_followup_recovery.py`.

**Interfaces:** `train.run(seed,arm,batch_size,mode)`, `evaluation.score_batches(model,admission,pool,split)`, `evaluation.score_run(root)`.

- [ ] Test evaluation provenance/reference guards, restored modes/RNG and exact corpus order; stop replay and candidate publication recovery.
- [ ] Implement fresh identity/source closure, epoch resume before dev, verified archive and old selection/pruning helpers; derive patience from reports.
- [ ] Profile real batch8 ordinary/long/worst-prompt joint updates and compact fresh-model reload/generation equality. If needed, enable tested activation checkpointing and repeat accepted profile before source freeze.
- [ ] Run focused tests, configured typing, real profile and inspect diff.

### Task3: Controller, monitoring and launch

**Files:** Create retained-worktree `tools/followup/study.py`; update main `AGENTS.md` and runbook.

**Interfaces:** sequential owned stages, fixed six-selection freeze, once-only test journal and summary, controller observation/recovery CLI.

- [ ] Test six selections, failed/owned child guards and manual-stop behavior.
- [ ] Implement single locked controller, graceful owned shutdown and20minute monitor; only verified same-source/same-root recovery.
- [ ] Verify old source hashes and all old owners exited, current storage, new data bindings, accepted profile and clean focused checks.
- [ ] Launch detached controller/watchdog; verify actual owners, new seed0reference startup and unchanged177400update ceiling/4000VT/10000LR settings.
- [ ] Record launch paths/PIDs/checks and report actual status.
