# Overnight projector study progress

Plan: `2026-10-04-projector-overnight-robustness.md`.
User selected full paired-seed robustness and authorized at most 36000 seconds
of additional GPU-stage wall time, with no further questions.

## Implementation and verification

- Once-only evaluator: checkpoint-bound registry, deterministic batch journaling,
  strict architecture/state reconstruction, dev-selected admission, no-clobber output.
- Seed support: seed-0 path retained; seed-1 common initialization attested on CPU,
  matching data/dropout/prompt seed, residual branch seed 2, same declared 16k recipe.
- Error analysis: 72 tests; frozen DEV-based German lexical rules, supported/unknown
  parsing categories, matched report checks, exploratory broadcast-cluster BLEU bootstrap.
- Controller: external GNU watchdog, parent-interruption cleanup, uncapped elapsed
  charges, global/stage caps, identity-safe recovery/adoption, completed-report revalidation.
- Final integrated verification: **874 passed, 4 skipped**; full Ruff and targeted
  Pyright clean. Integration fixtures corrected for package-module caching and
  `/proc` process disappearance; no training/evaluation behavior weakened.

## Review rulings

Accepted and regression-fixed controller findings: detached-child/deadline cleanup,
timeout grace accounting, finished-training adoption and completed-evaluation integrity.
Ruling: retain checkpoint-first full-metadata validation — measured system RAM
62 GiB, 48 GiB available; the review's limited-RAM loading-order assumption does
not apply to this host. This costs additional peak CPU memory but preserves early
metadata rejection; monitor real resource use.
No broad statistical-significance, signer-invariance or visual-correctness claim
is authorized by one/two seeds or lexical diagnostics.

## Active execution

- Tmux: `projector-overnight-robustness`.
- Study root: `/home/kan/datasets/despamo/signer-pilot/projector-overnight-study-v1`.
- Controller state: `study-state.json` under study root.
- Launch log: `/home/kan/datasets/despamo/signer-pilot/projector-overnight-robustness-launch.log`.
- Authoritative once-only registry:
  `/home/kan/datasets/despamo/signer-pilot/projector-heldout-registry-v1.json`.
- Verified initial scoring: seed-0 linear checkpoint
  `8cde697bf31ba67f0f24ddcb229d7f39829eb3d3cafc9c3f03b8171cf90fef4a`,
  116/768 clips journaled, GPU stage running. Seed-0 residual checkpoint is
  `2c218623c1d725e733ac8fedc7db82a0f2e22ce6f12ab2777fa86d91946a46ea`.

Proceed through all stages without new prompts. Completed study will publish
`summary.json`, `seed0-comparison.json`, `seed1-comparison.json`, and
`future-description-baseline.json`; then add a concise human-readable project report.
New description artifacts are still external and not inputs to this study.

## Primary held-out outcomes and replication progress

Both seed-0 registry claims completed exactly 768 clips. Linear BLEU-4
**4.784803**, residual BLEU-4 **5.915115**, delta **+1.130312**; ROUGE-L F1
0.197896 versus 0.220409. Read-only preliminary paired analysis gives conditional
broadcast-cluster bootstrap interval approximately [0.7256, 1.5401] using 69 inferred
broadcast groups, not a training-seed significance claim. Repeated-4gram outputs:
271 linear versus 213 residual; unique predictions 543 versus 660.

Seed-1 common initialization attestation succeeded on CPU:
`769848a2d06f6f02b38264a3258616a1f4ea343aeca9c7e61cfaea458625d87c`.
Profile hash `78468f995c1bd4393348869e160cd82e150eaf716c29f8cbdd5b72e60d703f47`.
Linear run:
`projector-linear-16000-0319401a7579e5fc6ee3a37748c01a441b2e0b75cf97ce9778f6fc60e1f457da`.
Observed at 9620 steps, first dev BLEU-4 4.555734 at 5500, workflow within budget.
Recipe stays fixed regardless of these held-out results. Controller remains active.

## Final completion

Controller completed with all stages successful and 13936.07 conservative
GPU-stage seconds spent (3.8711 hours, authorized cap10). Four authoritative
held-out claims are complete, each with 768 committed clips.

Seed1 residual run:
`projector-residual-16000-86f337958a8244e9538b3f36d7f6186859982f20fa1edd4ac71f614b8a54f353`.
Seed1 linear dev BLEU-4 5.715049 / test4.991212; residual dev4.583490 / test3.849948.
Primary seed0 gain +1.130312 reversed at seed1 (-1.141264). Two-seed mean
BLEU-4 4.888008 linear versus4.882532 residual; no robust aggregate improvement.

Final report: `docs/projector-heldout-robustness-2026-10-04.md`.
Repository recipe: `configs/experiment/projector_residual16_provisional.yaml`.
Runtime baseline JSON is in study root; status `provisional_mixed_robustness`,
seed0 default by predeclared rule, linear16 reference/comparator retained.
New description dataset remains external; no new-caption training claimed.
