# DeSpaMo Project Instructions

## Global/local experiment branch

- Branch: `feat/global-local-rvlf`. Read `docs/global-local-server-handoff.md`,
  `docs/superpowers/specs/2026-10-10-global-local-rvlf-design.md`, then
  `docs/superpowers/plans/2026-10-10-global-local-rvlf.md` before implementation.
- This branch publishes tested baseline source and an approved architecture for
  implementation on another server. The global/local model and fine-grained loss
  are not implemented yet. Do not label baseline tests as new-method results.
- Selected extractors now use frozen `google/siglip2-so400m-patch14-384`, revision
  `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`, for global and all local regions.
  Read `docs/superpowers/specs/2026-10-10-siglip2-global-local-amendment.md` first.
  Global two-scale384/768 pooled views produce2304channels; single384left-hand/
  right-hand/mouth native attention-pooled vectors produce1152each. SigLIP has
  no CLS token; this fixed-resolution checkpoint declares `model_type=siglip`.
  Existing VideoMAE motion stays1024; mouth replaces full face. No skeleton,
  Qwen or vision-encoder adaptation in this stage.
- Hierarchical local fusion first, then global/local fusion in768channels, one
  shared2048channel connector into FLAN-T5-XL. One RVLF-inspired fine-grained
  visual/text contrastive loss on fused output, plus translation CE.
- Use explicit source-frame support for temporal alignment. Existing global
  adapter concatenates spatial/motion tokens; equal sequence lengths or token row
  numbers do not establish matching times. Invalid local evidence must contribute
  zero after projections, temporal encoding and alignment; exact global fallback.
- Compare global-only, fixed local residual and confidence-aware local residual
  using the same local encoder, loss, masks, decoder and recipe. Train/development
  data only for optimization/selection; no new automatic test scoring.
- Existing `tools/followup/` and `tools/benchmark/` are historical same-recipe
  source references with original-host paths. New portable `tools/global_local/`
  entrypoints must not launch those legacy controllers on another server.
- Use existing uv locks and separate motion/DINO environments if those tools are
  needed. SigLIP extraction needs its own supported, locked newer Transformers
  environment; main4.32translation runtime stays unchanged. CPU verification:
  `PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=""
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 uv run --locked pytest tests/unit -q`.
- Local original-host six-run training continues separately. Do not edit its
  retained worktree/main source, model/feature bytes, locks or environment.
- Preserve all unrelated work and checkpoint provenance. No artifact deletion,
  secret/model/dataset publication, merge or extra training without appropriate
  user scope. Commit/push only intended branch files when explicitly requested.

## Qwen Appearance Supervision (2026-09-28 Decision)

- The user does **not require human verification** of LLM-generated appearance descriptions before exploratory DeSpaMo training. Use the versioned `qwen-schema98-unreviewed-v1` policy instead of the older mandatory-human training gate.
- Current Qwen train dataset key: `b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a`. It has 6,977 schema-valid and 119 failed clips of 7,096 (98.323%). The original 99% rule failed; the newer schema floor is 98%.
- Unreviewed training verifies complete dataset/source/model/encoder provenance and the 98% schema floor. Preserve all 7,096 translation clips; all 19 appearance targets for each failed clip remain masked and zero-valued. Schema validity does not prove visual correctness.
- Label such runs, checkpoints, and results `qwen-schema98-unreviewed-v1` with `human_review_status=not_assessed`. The existing 700-row optional `audit/review.csv` is blank: do not fabricate ratings or write an audit pass.
- Keep historical original-99%-failure artifacts unchanged. The audited route remains available if a human later reviews it; it is not a prerequisite for the unreviewed route.
- Start with the cheapest meaningful data-alignment/gradient smoke. Long training or ablations require separate explicit authorization. Do not commit, push, merge, delete prior artifacts, or change frozen caption records without explicit request.
