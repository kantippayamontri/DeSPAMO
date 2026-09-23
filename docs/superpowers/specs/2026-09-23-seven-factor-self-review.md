# Seven-Factor Specification And Plan Self-Review

**Date:** 2026-09-23

**Scope:** Approved seven-factor design, master-design consistency, dataset plan, training plan, and planned baseline interfaces. Inline self-review; no subagent review or implementation execution.

## Findings And Resolutions

| Finding | Resolution |
|---|---|
| Empty unknown lists became asserted absence in canonical text. | Omit unknown lists/null/uncertain values; only explicit controlled `none` is an observation. |
| All-unknown biometric objects could become meaningless contrastive targets. | Visibility validator rejects clear/partial objects without known attributes; uncertain objects are masked. |
| Single-frame JSON example contradicted five-frame validation. | Label it an illustrative excerpt; implementation fixture contains five ordered frames. |
| Diagram suggested background branch consumed other nuisance losses. | Redraw all four nuisance branches independently from masked clip mean. |
| Sampling round convention and source-to-feature correspondence were unspecified. | Round half up; require a hash-bound DINO source-index-to-row manifest; reject missing/duplicate/padded matches. |
| Version identity omitted generation settings and source changes. | Include canonical source-manifest hash, split, sampling/schema/prompt hashes, model digest, temperature, and seed. |
| Terminal-only retry persistence allowed repeated requests after crashes. | Reserve every attempt before the network call, persist raw response before validation, and consume interrupted slots; cap three requests across resumes. |
| Factor summaries could mix different encoder revisions or overwrite their artifacts. | Separate dataset and encoder identities; encoder metadata and files are hash checked at load time. |
| Masked/failed captions could drop translation samples. | Feature manifest covers every clip; failed clips retain base translation tensors and carry all-false auxiliary masks. |
| Schema validation was being conflated with semantic correctness. | Lexical checks are a filter; two human audits validate meaning, visibility, anatomical sides, and unsupported fields. |
| Biometric privacy wording conflicted with signer-stratified sampling/probes. | Names and inferred identity labels are absent from captions; private opaque dataset signer keys remain for experimental grouping. |
| Audit accuracy could be inflated by abstentions or stale/duplicate rows. | Human visibility determines correctness denominator; selection/record/review identities are bound, and duplicate/stale rows fail. |
| Contrastive loss with no distinct negatives was underspecified. | Skip fewer than two valid targets or fewer than two distinct strings; fixed cosine temperature and symmetric uniform positive targets. |
| Zero-weight positive heads could still change through AdamW weight decay. | Do not call articulator heads at zero ramp; disabled heads have gradients disabled. |
| Schedule could depend on batch count or resume state incorrectly. | Match baseline warm-up boundary; use optimizer global step and a fixed joint-step budget; save prompt RNG and provenance. |
| Baseline probes and post-training probes lacked split and gallery contracts. | Deterministic clip-disjoint closed-set signer split, fresh linear models, fixed unique-canonical gallery, explicit missing metrics. |
| Existing DINO score was not a matched control for further factor training. | All 11 variants, including zero-auxiliary controls, start from one shared DINO checkpoint and receive identical additional budgets. |
| Baseline evaluation expects checkpoint `metadata`; draft export lacked that key. | Export writes `state_dict` plus compatible `metadata`, with source checkpoint hash and factor provenance. |
| Report assembly could pair unrelated translation/probe checkpoints. | Collector checks source hashes, seed, enabled factors, model policy, and artifact provenance before computing comparison identity. |
| Official PHOENIX splits were not sufficient evidence of unseen-signer evaluation. | Keep unseen-signer gap unavailable until a separate signer-disjoint translation split is provided. |
| Historical three-factor plan still appeared executable. | Add prominent superseded banner linking both replacement plans; preserve historical content. |

## Requirement Coverage

| Specification section | Dataset plan | Training plan |
|---|---|---|
| 1-4: purpose, DIFFER adaptation, scope, factor boundaries | Task 1 schema and scoped canonicalization | Task 2 gradient routing |
| 5: frame sampling and correspondence | Task 2 source manifest | Entry contract and Task 1 row lookup |
| 6: structured output | Task 1 strict schema and fixtures | Task 1 validates cached target identities |
| 7: generation/provenance/versioning | Tasks 2-3; Task 6 writer lock | Tasks 1, 3-4 provenance/resume checks |
| 8: canonical descriptions/text features | Tasks 1 and 4 | Task 1 verified cached tensors |
| 9: architecture | Immutable offline targets | Tasks 2-4 adapter/heads/factory; Task 5 export |
| 10: losses/schedule | Target masks and canonical strings | Tasks 2-3 contrast, signs, ramp |
| 11: quality/audit | Task 5 two-level audit; Task 6 manual gate | Tasks 1 and 4 enforce gate |
| 12: effects/probes | Fixed target identity | Tasks 6-7 probes, 33-run matrix, collector, paired report |
| 13: failure handling | Tasks 1-6 | Tasks 1-7 |
| 14: testing | Offline tests and opt-in live request | Gradient, mapping, parity, schedule, export, probes, report tests |
| 15: implementation boundary | Prerequisites and completion gate | Dataset/baseline probe gate and explicit experiment authorization |

## Verification Evidence

Validation copied fenced source blocks into a disposable directory under `/tmp/opencode`, leaving application scaffold untouched.

- Parsed 37 complete Python source blocks using Python-3.11 grammar rules.
- Checked internal `despamo.*` imports against explicit file blocks in the two new plans and prerequisite baseline plan.
- Ran 48 offline tests: schema, ordering, retries/restart, canonical masks, source manifests, audit rates/identity, frozen-text boundary, cached-target join with a failed clip, frame lookup, all seven gradient signs, grouped backward passes, export, probe splitting/retrieval, collector and 33-run reports.
- Independently exercised optimizer-step ramp boundaries and adapter translation/checkpoint-name parity.
- Pyflakes (`ruff check --select F`) passed on extracted source blocks.
- Placeholder/fence scans and new-document whitespace checks passed.

The available sandbox used Python 3.13.13, PyTorch 2.13.0+cu130, and Pydantic 2.13.2 on CPU. These checks validate plan snippets; they do not certify the future Python-3.11/PyTorch-2.0.1 locked environment. Full Lightning/Flan-T5 integration, real Qwen inference, dataset audit, GPU training, and empirical factor benefits remain execution-stage checks.

No application source files, datasets, or model weights were created in DeSpaMo. No commits or research runs were performed.

## Execution Order

1. Existing PHOENIX14T baseline plan and acceptance.
2. DINOv3 extraction/comparison milestone, including source-row manifest.
3. `docs/superpowers/plans/2026-09-23-seven-factor-dataset.md` (six tasks).
4. `docs/superpowers/plans/2026-09-23-dual-path-training.md` (seven tasks).

Large generation and 33 training jobs require their own explicit execution authorization; producing this plan does not launch them.
