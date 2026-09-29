# DeSpaMo Project Instructions

## Qwen Appearance Supervision (2026-09-28 Decision)

- The user does **not require human verification** of LLM-generated appearance descriptions before exploratory DeSpaMo training. Use the versioned `qwen-schema98-unreviewed-v1` policy instead of the older mandatory-human training gate.
- Current Qwen train dataset key: `b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a`. It has 6,977 schema-valid and 119 failed clips of 7,096 (98.323%). The original 99% rule failed; the newer schema floor is 98%.
- Unreviewed training verifies complete dataset/source/model/encoder provenance and the 98% schema floor. Preserve all 7,096 translation clips; all 19 appearance targets for each failed clip remain masked and zero-valued. Schema validity does not prove visual correctness.
- Label such runs, checkpoints, and results `qwen-schema98-unreviewed-v1` with `human_review_status=not_assessed`. The existing 700-row optional `audit/review.csv` is blank: do not fabricate ratings or write an audit pass.
- Keep historical original-99%-failure artifacts unchanged. The audited route remains available if a human later reviews it; it is not a prerequisite for the unreviewed route.
- Start with the cheapest meaningful data-alignment/gradient smoke. Long training or ablations require separate explicit authorization. Do not commit, push, merge, delete prior artifacts, or change frozen caption records without explicit request.
