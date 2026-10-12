# Three-arm connector execution ledger

Plan: `2026-10-04-three-arm-connector-study.md`.
User approved exact arms, small Q-Former defaults and two matched seeds; up to
64800 additional GPU-stage seconds, dev-only. No Signer07 scoring or commits.

Preflight reasoning: existing motion stays linear in all arms; plain MLP spatial-only;
Q-Former after TCN with temporalpositions. Fixed query32/layers2/width768/heads12/
FF3072/dropout0.1 declared before inference. Baseline output hook must preserve
original keys and forward behavior. CUDA/kernel behavior checked with smoke.

## Implementation and review

Model tests verified baseline output/state-key parity, spatial-only plain MLP,
masked query outputs/gradients, temporal ordering and strict reload. Source archive
preserving the prior seed1 projector closure:
`projector-original-source-d8e53e3ba38a2a32841e46e200bbf5c138ed1af057d791d86196118a4f8e7e2f.tar.gz`;
archive SHA256 `b7df47cab622d6ddc73e99eca538e90785f8783ea200d52b217caadc9536079f`.

Real-input preflight reproduced the historical generic-config LR0.0006, distinct
from the effective adapter/flan0.0003/0.00005 groups. Corrected false preflight
guard; effective optimizer parameters/groups/hyperparameters now verified.

Independent review fixes RED→GREEN: terminalstep16000 evaluation-only recovery,
all three exact dev-report byte hashes in selection, and controller/watchdog
source closure. Added BLEU1-4/ROUGE paired deltas. Remaining minor fixture concern
about full resume-state coverage is limited; actual complete checkpoints are
validated by own metadata/state and latest-snapshot recovery during runtime.

Final verification: **1002 tests passed,4 skipped**, full Ruff and targeted
Pyright clean. User approval18hours enforced as two nine-hour seed phases with
uncharged unused allowance not transferred, plus perrun10800 deadline. Initialization,
admission smokes and validation are charged conservatively; no prior artifacts pruned.

## Launch

Tmux `three-arm-connectors`; persistent launch log:
`/home/kan/datasets/despamo/signer-pilot/three-arm-connector-study-v1-launch.log`.
Study root `/home/kan/datasets/despamo/signer-pilot/three-arm-connector-study-v1`.
Controller topstate `state.json`; each seed's `study-state.json` records exact stage
wall seconds, run roots and hard watchdog children. Six jobs ordered reference,
plain_mlp,qformer for seed0 then seed1; dev-only reports.

Both common initialization hashes attested before launch. All three real four-clip
joint CE+VT admission smokes passed without an optimizer update or state mutation:
reference7.18GiB, plainMLP7.19GiB, QFormer7.14GiB peak. QFormer learned-query and
attention gradient norms2.291688/41.068566 are nonzero.
First run:
`connector-reference-seed0-16000-c9af7cf6b8aff72ecf8f8e88cd04ddfa6c56811b55072def47d5fbaeacfcf825`.
Observed active after admission; results pending full dev schedule.

## Storage crash and recovery (2026-10-05)

Windows C reached about3GiB free while Linux still had156GiB. WSL restarted during
the seed1 plain-MLP terminal resume write. All three dev reports and the full
step16000 checkpoint remained intact. User explicitly authorized analysis and
safe deletion before continuation.

Removed20 redundant resume checkpoints from11 completed runs plus the2.46GiB
failed pending snapshot: **113.508GiB** recovered inside Linux. All step checkpoints,
reports, data/caption/feature/model caches and partial-run recovery snapshots were
preserved. Retained checkpoint hashes/fingerprints verified. Online fstrim reported
225.4GiB discarded, but Windows free space remained about3.3GiB.

Seed1 plain MLP finalized using existing complete checkpoint/dev reports, without
optimizer update or new generation: selected step16000, dev BLEU-4 **7.090514**.
Controller cross-boot recovery and adoption reconciled conservatively; no budget
reset. Seed1 accounting17734.91seconds, remaining14665.09seconds.

Five of six runs complete. Remaining QFormer seed1 is **paused for Windows host
space**, not relaunched into the same exhaustion condition. Resume script
`scripts/resume_connector_study.sh` requires50GiB available on Linux and Windows
C before invoking the unchanged source-bound controller. Once host space is reclaimed,
it will reuse verified completed work and launch the last run. Exact cleanup and
recovery receipt: `docs/storage-recovery-2026-10-05.json`.

## Post-compaction completion

User compacted disk: verified Linux270GiB / Windows120GiB available, all64 source
hashes unchanged. Resumed guarded controller, which reused/adopted all completed
runs. Seed1 QFormer:
`connector-qformer-seed1-16000-c099b17106f2a4d14724e4ecf5969826aed573e77d7bf12775242533d87e901b`.
Real joint gradient smoke passed, including learned-query/attention norms
2.061449/37.241084; no optimizer update in smoke. Runtime reached16000 and selected
step10400 with dev BLEU-4 **5.901238**; finalstep16000 BLEU-4 was5.794842.

All six validated selections complete. Seed0 reference/plainMLP/QFormer
4.541238/5.197683/5.695729; seed1 4.571809/7.090514/5.901238.
Means4.556524/6.144099/5.798483; paired mean gains+1.587575 and+1.241960.
Overall conservative charge12.1521hours, within18, including interruption accounting.
All source/model/report hashes verified; no Signer07 generation or budget reset.
Report: `docs/three-arm-connector-results-2026-10-05.md`.

Reference reruns differ from historical linear scores despite matched verified
inputs, weights and recipe. Independent read-only diagnosis established no
explaining recipe discrepancy; backend nondeterminism remains possible, not proven.
Compare against this study's fresh references and disclose absolute rerun variability.
