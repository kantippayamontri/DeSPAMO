# DeSpaMo Project Handoff

Open OpenCode from `/home/kan/Research/DeSpaMo`. This directory is local Git
`main`; baseline and verified E1/E2 signer-pilot code/tests/reports are here.
The [signer protocol](signer-pilot-protocol.md) and project `AGENTS.md` define
the active data, masking and held-out evaluation contracts.

## Project-Local Navigation

| Link in project root | Existing target | Status |
|---|---|---|
| `external-data` | `/home/kan/datasets/despamo` | ~170 GiB of datasets, frozen features, model cache and run artifacts; intentionally outside Git. |
| `recovery-worktree` | `/home/kan/Research/DeSpaMo-e1-recovery` | Incomplete, unmerged appearance/InternVL/E3 recovery and related tests. Preserve separately. |

These are Git-ignored symbolic links, **not physical copies**. Do not move or
delete their targets on the assumption everything resides inside Git.
Recorded configurations/checkpoints bind original absolute paths. Deleting
files under `/tmp` does not affect these links or the merged E1/E2 files.

## Frozen Pilot And Results

- Protocol hash: `0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7`.
  Logical train: 5,746 clips from seven signers; Signer03 dev: 582; Signer07
  held-out test: 768. All physically originate from PHOENIX `train/`.
- E1 frozen DINO, fresh SpaMo/FLAN LoRA: seed 0, physical batch 4, 4,000
  steps, 1,000 VT-only warm-up; best full-dev checkpoint step 4,000,
  BLEU-4 **3.695629**. See [E1 report](e1-v3-wsl-retraining.md).
- E2 same shared initialization plus seven projector factor heads: best
  step-4,000 full-dev BLEU-4 **3.028302**, difference from E1 **-0.667327**.
  Train-only Qwen factor join: 5,641 valid, 105 failed clips fully masked
  for appearance (translations retained). Human review status is
  `not_assessed`. See [E2 report](e2-wsl-factor-run.md).
- E1/E2 train/dev DataLoaders use `num_workers=0` on WSL. Their periodic
  `resume-*.ckpt` files were deleted after completion to recover 88.46 GiB.
  Selected `step-4000.ckpt`, earlier dev-selection checkpoints, dev reports
  and closed budgets remain under `external-data/signer-pilot/`.
- E3 adaptation, signer probe and Signer07 translation scoring have **not**
  run. Original three-method 20-GPU-hour feasibility gate failed; its receipt
  remains under `external-data/signer-pilot/profile-gate-b4.json`.

## Recovery Boundary And Verification

The former `/tmp/opencode/despamo-dinov3` worktree vanished after a WSL
shutdown. E1/E2 code was recovered byte-for-byte against checkpoint-bound
SHA-256 identities, tested and merged into `main`. Earlier unrelated
appearance/InternVL/E3 work was **not** merged: the `recovery-worktree` link
exposes what was recovered, but some prior research documentation and edits
were not fully restored. Treat it as an archive requiring review, not an
equivalent clean branch. Main's user-owned `main.py`, `papers/` and
`proposal.md` were left untouched.

Full main verification after the E1/E2 merge: `591 passed` in `tests/unit`,
plus `ruff check scripts src tests tools/dinov3`. Do not interpret the
released SpaMo checkpoint's ~25 BLEU-4 on the official 642-clip test as
directly comparable to fresh E1/E2 on the logical Signer03 dev subset.
