# Fixed-checkpoint three-arm Signer07 comparison

User requested evaluation of the dev-selected models before a generalization
claim. Scope: all six completed reference/plain-MLP/Q-Former checkpoints, seeds0/1,
exact ordered768Signer07, no new training/extraction or test-based tuning.

- Source study: `three-arm-connector-study-v1/summary.json`, complete six runs.
- Reference and plain MLP select16000; Q-Former seed0 selects16000 and seed1
  selects10400. These selections are fixed from full Signer03 dev reports.
- Identical deterministic beam5/max64/noICL, BF16, physical evaluation batch4.
- Use authoritative existing `projector-heldout-registry-v1.json` under external
  pilot storage. Unique checkpoint claims/journals, completed reuse, no duplicate
  logical scoring; interrupted uncommitted batch may replay under the same claim.
- Each scoring child cap1800seconds; conservative overall watchdog budget10800
  seconds, including CPU admission/model setup/validation. Sequential GPU jobs.
- Publish under a new `three-arm-connector-heldout-v1` output root, preserving old
  four registry entries and original evaluator bytes. New adapter binds source
  closure and reconstructs exact architecture with strict full checkpoint state.
- Compare per-seed BLEU1-4/ROUGE-L differences versus fresh reference, repeated
  fourgrams, prediction diversity, between-seed variation and exploratory paired
  broadcast-cluster intervals. Intervals conditional on frozen outputs, not a
  broad significance/identity-invariance claim. Analyze once all reports verified.

## Execution

- [x] Connector-aware held-out admission/reconstruction and once-only adapter tests.
- [x] Overall-budget controller, completion validation, fixed manifest/output binding.
- [x] Verification/review and six real-data preflights.
- [x] Score each dev-selected model once logically; no selection/recipe changes.
- [x] Validate identical ordered references and record final generalization findings.

No additional seeds/signers, checkpoint cleanup, commits or pushes authorized.

## Completion

All six fixed checkpoints complete, each768Signer07 clips. Controller charge1.1045
GPU-stage hours below3. Shared registry has10complete claims, including4preserved
older evaluations. Final verification1048tests passed,4skipped; lint/types clean.
Review controller issue reproduced/fixed: exact source pair and checkpoint admission
must precede scoring, not be detected after claiming an unintended model.

Reference test BLEU-4 4.360015/4.513259; plainMLP4.637257/5.698437;
QFormer5.150172/4.655053. Mean paired gains+0.731210/+0.465976.
Both positive in both seeds, but conditional interval includeszero for plainMLP
seed0 andQFormer seed1. Report promising transfer, not broad robustness/significance.
Full [held-out report](../../three-arm-connector-heldout-2026-10-05.md).
