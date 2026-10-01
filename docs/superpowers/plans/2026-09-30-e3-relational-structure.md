# E3 Relational-Structure Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Project policy forbids commits, merges, and subagent dispatch without explicit request.

**Goal:** Test separately whether E3's small held-out gain was undertraining, and whether its DIFFER objective underperformed because adapted feature geometry collapsed rather than because DINOv3 lost linguistic information.

**Architecture:** Parameterize the frozen SpaMo step tier so E1 and E3 are compared at an equal 8,000-step budget on existing features. Add two relational preservation terms that constrain pairwise feature structure against detached frozen-DINO teachers, raise adaptation batch to 16, and gate every new adaptation variant on a ~600-clip probe before paying for full extraction.

**Tech Stack:** Python 3.11; `tools/dinov3` locked env (torch 2.5.1+cu121, transformers 4.56.2) for adaptation/extraction; main env (torch 2.0.1, Lightning 1.9.5) for SpaMo; pytest, Ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-29-e3-complete-pilot-code-design.md` (this plan extends it; the frozen-input, masking, and held-out contracts there remain binding).

## Measured Evidence

| Measurement | Original DINO | Adapted E3 |
|---|---:|---:|
| Left / right handshape probe (7-way) | 0.360 / 0.226 | **0.430 / 0.304** |
| Signer top-1 (linear, mean-pooled) | 0.979 | **0.989** |
| Nearest-centroid signer accuracy | 0.938 | 0.973 |
| Same-vs-other centroid margin | +0.0439 | **+0.0035** |
| Mean pairwise cosine (std) | 0.770 (0.094) | **0.977 (0.009)** |
| `std_ratio` of pairwise cosines vs teacher | 1.00 | **0.12** |

Articulator information improved; the nuisance GRL collapsed all features into a narrow cone without reducing signer decodability. DIFFER preserves metric structure through its ReID identity loss; this pilot has no equivalent scaffold.

Candidate loss discrimination, measured on the real collapsed features versus perfect preservation: Term A (`1 - correlation`) reads **0.3523 vs 0.0042**; naive pairwise MSE reads only **0.0266**, roughly 90x smaller than the ~2.3 factor losses and therefore too weak to matter at weight 0.2.

## Global Constraints

- Frozen protocol hash `0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7`; dataset key `b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a`; DINO base revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`.
- Logical train 5,746 clips / seven signers; Signer03 dev 582; Signer07 held-out test 768. Feature paths remain `train/<clip_id>.npy` for all 7,096.
- Step tier 8,000 with warmup 2,000; full-dev evaluations at **3,500 / 5,600 / 8,000**; deterministic beam 5, `max_length` 64, reference-free prompts.
- Adaptation physical batch **16**; SpaMo physical batch **4** (`pilot_config` admits only 2 or 4, and E1/E2 ran at 4).
- **No balanced batch sampling.** Batch 16 already removes every dead nuisance step (biometric 18.5% -> 0.1%).
- Relational Term A and Term B weights start at **0.2**; the existing `scale_reference_loss` is retained, giving three preservation terms; the seven factor heads are unchanged.
- Declared cumulative GPU ceiling is raised to **40 hours**, superseding the original 24-hour figure, which stays recorded as history. Per-stage caps are enforced in code; the cumulative ceiling is documentation.
- Label every new run, checkpoint, and result `qwen-schema98-unreviewed-v1` with `human_review_status=not_assessed`.
- Signer07 is never used for checkpoint or variant selection, and is scored at most once per selected checkpoint. Existing E1/E2/E3 artifacts are immutable.
- No commits, merges, or pushes. No full extraction or SpaMo run without its explicit authorization flag and cap.

## Review Focus

- Requesting an unsupported step tier (for example 15,000 or 5,000) must be rejected before any GPU work rather than silently coerced to 4,000 (Task 1 test).
- An E3 SpaMo run whose step tier differs from its matched E1 run must be rejected by the config match instead of comparing unequal budgets (Task 1 test).
- A uniform shrink of all pairwise distances must be penalized: Term A alone is scale-invariant and tolerates it, so Term B must catch it (Task 3 test).
- Resuming adaptation at batch 16 must not replay or skip clips, and must not rewind the sampler/step/spend relationship (Task 4 test).
- The subset probe gate must refuse features whose adaptation checkpoint or frozen source identity does not match the variant being judged (Task 5 test).

---

### Task 1: Parameterize the SpaMo step tier

**Files:**
- Modify: `scripts/train_e3_pilot.py` (10 literal `4000` sites), `scripts/retrain_e1_authorized.py` (7 literal `4000` sites)
- Test: `tests/unit/pilot/test_e3_train_cli.py`, new `tests/unit/pilot/test_step_tier.py`

**Interfaces:**
- Consumes: `checkpoint_steps(steps, warmup)` and `pilot_config(..., steps, physical_batch, output)`, which already admit `steps in {4000, 6000, 8000}` with warmup map `{4000: 1000, 6000: 2000, 8000: 2000}`.
- Produces: `--steps` CLI option (default 4000) on both runners; `require_matched_e3_config(e1_identity, e3_config, adapted_root)` additionally requires equal `trainer.max_steps`; `select_e3_checkpoint(reports, dev_ids, steps)` validating the tier's own three dev steps.
- Do **not** edit `scripts/train_signer_pilot.py` or any file in E1-v3's `checkpoint_bound_code_hashes`; `checkpoint_steps` and `FullDevCheckpoints` already accept these as parameters.

- [ ] **Step 1: Write failing tests** for: 8,000 admitted with dev steps `(3500, 5600, 8000)`; 4,000 still admitted with `(1750, 2800, 4000)`; 5,000 and 15,000 rejected; E3-vs-E1 config match failing when only `max_steps` differs; identity recording the tier.
- [ ] **Step 2: Run** `PYTHONPATH=src:. .venv/bin/python -m pytest tests/unit/pilot/test_step_tier.py tests/unit/pilot/test_e3_train_cli.py -q`; expect failures naming the missing `steps` parameter.
- [ ] **Step 3: Implement** the `--steps` option and thread it through both runners, replacing every literal `4000` in model construction, `PilotDataModule`, `FullDevCheckpoints`, completion checks, dev-report reading, and identity.
- [ ] **Step 4: Run** the same tests; expect pass. Then `pytest tests/unit -q` and `ruff check scripts src tests tools/dinov3`.

### Task 2: Stage 0b fairness runs at 8,000 steps

**Files:** No source change. Produces new external run directories under `/home/kan/datasets/despamo/signer-pilot/`.

**Interfaces:** Consumes Task 1's `--steps`. E1 runs first; E3 then matches against E1's stored config, so these are strictly sequential. Per-run GPU cap 6 hours (`E1Ledger` ceiling). Existing 4,000-step runs are untouched; new run keys derive from the changed identity.

- [ ] **Step 1: Preflight E1@8000** and confirm the printed tier, dev steps, and a new run key.
- [ ] **Step 2: Run E1@8000** with its authorization flag; expect `status: complete`, `global_step: 8000`, closed budget, and three dev reports.
- [ ] **Step 3: Preflight E3@8000** against the completed E1@8000 run; expect the config match to pass.
- [ ] **Step 4: Run E3@8000**; expect completion and a selected checkpoint chosen by highest full-dev BLEU-4.
- [ ] **Step 5: Verify** both runs: 582 identical ordered dev clip IDs and references per report, recomputed checkpoint hashes, closed budgets. Record measured GPU hours. Do **not** score Signer07 yet.

### Task 3: Relational structure loss

**Files:**
- Modify: `tools/dinov3/e3_model.py`
- Test: `tools/dinov3/tests/test_e3_relational.py`

**Interfaces:**
- Produces: `relational_structure_loss(adapted: Tensor, teacher: Tensor) -> dict[str, Tensor]` returning keys `pattern` (Term A) and `spread` (Term B). Inputs are the same `[N, 2048]` tensors `scale_reference_loss` already receives; teachers are detached; each scale half (`0:1024`, `1024:2048`) is computed separately and averaged.
- Term A: standardize off-diagonal pairwise cosines of student and teacher, then `1 - mean(z_student * z_teacher)`.
- Term B: `(log(std_student) - log(std_teacher))**2` over the same off-diagonal cosines.
- Consumed by Task 4.

- [ ] **Step 1: Write failing tests**: both terms near zero when `adapted` equals `teacher`; Term A elevated for a structure-scrambling permutation; **Term B elevated for a uniform shrink toward the mean while Term A stays near zero**; teacher receives no gradient; student does; finite values with fewer than three rows.
- [ ] **Step 2: Run** `uv run --offline --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_e3_relational.py -q`; expect import failure.
- [ ] **Step 3: Implement** `relational_structure_loss` with an explicit off-diagonal mask and epsilon-guarded standard deviations.
- [ ] **Step 4: Run** the same tests; expect pass.

### Task 4: Wire relational terms and batch 16 into adaptation

**Files:**
- Modify: `tools/dinov3/e3_train.py`
- Test: `tools/dinov3/tests/test_e3_train.py`

**Interfaces:**
- Consumes Task 3's `relational_structure_loss`.
- Produces: `--relational-weight` (default `0.2`, applied to both terms) and `--batch-size` (default 4, admitting 4, 8, 16) on the adaptation CLI; `step_e3(..., relational_weight)` adding `relational_weight * (pattern + spread)` to `combined_loss` alongside the retained `scale_reference_loss`; identity recording `batch_size`, `relational_weight`, and both term names.
- `E3BatchSampler(ids, *, seed, batch_size)` must accept 4, 8, or 16 while preserving the invariant that `batches_yielded` equals completed optimizer steps and that no clip repeats inside one batch.

- [ ] **Step 1: Write failing tests**: sampler yields unique clips per batch at 16 and survives save/restore mid-epoch; relational terms appear in the loss dict and change `combined_loss`; calibration steps still leave DINO gradients absent; identity records batch size and weight; batch sizes outside {4, 8, 16} rejected.
- [ ] **Step 2: Run** `uv run --offline --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_e3_train.py -q`; expect failures.
- [ ] **Step 3: Implement** the options and loss wiring; keep `scale_reference_loss` unchanged and additive.
- [ ] **Step 4: Run** targeted tests, then the whole `tools/dinov3/tests` suite; expect pass.
- [ ] **Step 5: Bounded real smoke** of one optimizer step at batch 16 on three real train clips; confirm finite losses, non-zero gradients in all eight LoRA up-projections, and peak VRAM below 13 GiB. Measured reference: batch 16 peaked at 11.24 GiB and 1.421 s/step.

### Task 5: Subset probe gate

**Files:**
- Create: `scripts/probe_e3_features.py`
- Test: `tests/unit/evaluation/test_probe_e3_features.py`

**Interfaces:** CLI `--mode preflight|run --adapt-run PATH --original-dino-root PATH --frames PATH --cache PATH --clips-per-signer N --output PATH`. Extracts adapted features for a deterministic per-signer sample of logical train clips only, then reports `std_ratio`, signer top-1 and balanced accuracy, and left/right handshape probe accuracy, each for original and adapted features. Reuses `pool_spatial`, `split_probe_clips`, `score_probe` from `src/despamo/evaluation/pilot_probe.py`. Writes one aggregate JSON with no clip-level predictions. A new script is required because `scripts/probe_e3_pilot.py` demands a complete 7,096-clip manifest and a finished SpaMo run.

- [ ] **Step 1: Write failing tests**: deterministic sample excludes Signer03 and Signer07 and covers all seven train signers; mismatched adaptation checkpoint or frozen source identity rejected before extraction; existing output refused; report contains the three gate metrics and no per-clip predictions.
- [ ] **Step 2: Run** `PYTHONPATH=src:. .venv/bin/python -m pytest tests/unit/evaluation/test_probe_e3_features.py -q`; expect import failure.
- [ ] **Step 3: Implement** the script, reusing the existing extraction helper for a bounded clip subset.
- [ ] **Step 4: Run** targeted tests plus `ruff check`; expect pass.
- [ ] **Step 5: Validate against the completed E3 adaptation**, whose expected signature is `std_ratio` near 0.12, signer top-1 near 0.99, and handshape accuracy above the original. This confirms the gate reproduces the known failure before it is trusted to judge new variants.

### Task 6: Gated relational weight sweep

**Files:** No source change. Produces three short adaptation runs and three gate reports.

**Interfaces:** Consumes Tasks 3, 4, 5. Each variant is a short adaptation (2 to 3 epochs at batch 16, calibration held at one fifth of total steps) with its own explicit cap, followed immediately by the subset gate.

- [ ] **Step 1: Run** three short adaptations at relational weight 0.1, 0.2, 0.4.
- [ ] **Step 2: Gate each** with `scripts/probe_e3_features.py`.
- [ ] **Step 3: Select** the variant with `std_ratio` nearest 1.0 **and** signer top-1 below the original baseline **and** handshape accuracy at or above original. Record all three reports, including failures. If no variant satisfies all three, stop and report that the relational fix did not work rather than proceeding to Stage 4.

### Task 7: Full run of the selected variant and final scoring

**Files:** No source change. Produces one adaptation run, one feature version, one SpaMo run, one Signer07 report; updates `docs/signer-pilot-protocol.md` and `docs/e3-code-runbook.md` after results exist.

**Interfaces:** Consumes Task 6's selected weight and Task 1's `--steps 8000`.

- [ ] **Step 1: Full adaptation** at the selected weight, batch 16, with an explicit cap.
- [ ] **Step 2: Full extraction** of all 7,096 physical train clips with an explicit cap; require the complete manifest, row map, and receipt gate.
- [ ] **Step 3: SpaMo@8000** on the new feature version; select by full Signer03 dev BLEU-4.
- [ ] **Step 4: Score Signer07 once** for the selected checkpoint; verify 768 identical ordered IDs, identical references, and identical decoder settings against the existing E1/E2/E3 reports.
- [ ] **Step 5: Update docs** with measured numbers, the 40-hour ceiling, actual cumulative spend, and the honesty bound that single-seed deltas below about 0.3 BLEU-4 are not improvements. Run `pytest tests/unit -q`, the `tools/dinov3` suite, and `ruff check scripts src tests tools/dinov3`. Inspect `git status` and `git diff`. No commit.

## Success Criteria

- Task 2: if E3@8000 minus E1@8000 on Signer07 exceeds +0.3 BLEU-4, undertraining was a genuine factor at 4,000 steps.
- Task 6: the gate requires `std_ratio` near 1.0, reduced signer top-1, and retained handshape accuracy, together.
- Task 7: report the measured Signer07 delta plainly; a single-seed change below about 0.3 BLEU-4 is not reported as an improvement.

## Risks

- Relational terms pull toward frozen DINO while the nuisance GRL pushes away. Over-weighted, E3 degenerates toward E1 and disentanglement disappears; the sweep exists to locate that boundary.
- Probe partitions hold roughly 94 to 721 clips, so treat magnitudes as indicative and directions as reliable.
- 8,000 steps over 5,746 clips with 23.5M trainable parameters raises overfitting risk; the three-point dev selection is the guard.
- Preserving structure also preserves signer structure. The gate can pass on geometry while translation stays flat; that outcome must be reported as-is.
