# Official PHOENIX14T DINO adaptation runbook

## Active run and frozen execution workspace

Authorized40-epoch adaptation launched on2026-10-07. Run:
`/home/kan/datasets/despamo/official-phoenix14t-dino-adaptation-v1/runs/official-dino-adapt-bfea762ea2f71c244c9f3e2ef5f6be1a0f10b48bde2a5c6f05dcc8ba0faf5f0d`.

Training log:
`/home/kan/datasets/despamo/official-phoenix14t-dino-adaptation-v1/adaptation-training.log`.

Execute/source workspace:
`/home/kan/Research/DeSpaMo/.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`.
Branch `work/official-merged-targets`; uncommitted and not merged. Retain this
worktree and plan ledger. **Do not change/add DINO Python files, config, lock or
environment while the run is live.** All top-level DINO source files and runtime
identity are checkpoint-bound; documentation changes outside that closure are safe.

Interpreter: `/home/kan/Research/DeSpaMo/tools/dinov3/.venv/bin/python`.
Environment: `PYTHONPATH=.`, `LD_LIBRARY_PATH=/usr/lib/wsl/lib`,
`HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`,
`CUDA_VISIBLE_DEVICES=0`. The main environment is only for text preparation;
the motion-extractor environment is not used by adaptation.

## Approved recipe

One fresh seed0 encoder, official train7,096clips/all nine signers. Pinned
DINOv3 ViT-L/16 with rank8/alpha8 Q/V LoRA in blocks20–23, five frames/clip,
224/448 resolutions, BF16 encoder and **explicitly FP32 factor/geometry losses**.
Batch24, AdamW betas.9/.999, eps1e-8, decay.05; peak LoRA1e-5/heads3e-5;
gradient norms clipped separately at1.0. Seven factor losses plus teacher cosine,
relational pattern and spread (weight.10) remain as approved. Human review is
`not_assessed`; merged v1 stable/v2 articulators are independent masked slots.

| Boundary | Optimizer updates |
| --- | ---: |
| Total40epochs | 11,800 |
| End five-epoch head calibration | 1,475 |
| First joint update | 1,476 |
| End two-joint-epoch LoRA warmup / GRL ramp | 2,065 |
| Epoch10 model/resume | 2,950 |
| Epoch20 model/resume | 5,900 |
| Epoch30 model/resume | 8,850 |
| Epoch40 model/resume and selection | 11,800 |

Head LR warms during calibration then cosines. LoRA LR is0 until unfreezing,
warms over590updates, then cosines. GRL is0 through calibration, ramps to1 over
590joint updates. Final LR multipliers are0 by deliberate declared convention.
No wall-time training stop or forecast veto. Integrity, ownership, finite-value
and storage failures remain real stop conditions.

## Monitoring

Read `budget.json` for owner PID/boot/start ticks, completed updates, charged time,
phase and status; match actual `/proc/<pid>/cmdline`/start identity before action.
Log prints all losses, gradients, group LRs and GRL every25updates. Ledger is
updated at most10seconds between normal batches. Calibration should show zero
LoRA gradient norm/LR and GRL, with nonzero head learning rate and gradients.
Inspect `nvidia-smi` for actual compute owner. Do not start duplicate jobs.

At `2026-10-07T17:08:32Z`, PID592617 matched the durable owner, ledger recorded
**28updates**, phase`train`, status`running`; GPU8265MiB/100%. This is a snapshot.

Follow-up at `2026-10-07T17:10:09Z`: **64/11,800 updates**, status`running`,
charged235.334522s including retained smoke/setup time. PID/boot/start ticks and
command matched; all native/sealed producer hashes remained exact. Step50 logs
showed finite factor losses, nonzero head gradients/LR, and zero LoRA LR/GRL as
expected during calibration. GPU remained8265MiB/100%.

## Checkpoints, recovery and retention

Model files `model-epoch-{10,20,30,40}.pt` contain LoRA/factor heads and pinned
identity, not duplicate frozen base weights. Full `resume-epoch-N.pt` includes
optimizer, group clock state, sampler and all RNG states. SHA-256 receipts and
`milestone-epoch-N.json` publish atomically/fsynced before old owned resumes prune.
Retain all four model exports and latest two full resumes. Epoch40 is predeclared
selection; no official-test-based reselection or scoring is authorized here.

If the process stops, first verify it is gone and source/data bytes still match.
The same `--mode run` command automatically uses latest complete owned resume;
an explicit `--resume-checkpoint` must also be newest/owned. Before epoch10,
restart initialization within the same root/identity, keeping prior charged time.
An incomplete/corrupt milestone transaction fails closed rather than silently
overwriting/pruning it; inspect receipts before resolving such a failure.
Do not erase state, choose another output root or edit clocks to reset charges.

From the frozen worktree, recovery command after ownership inspection:

```bash
PYTHONPATH=. LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 /home/kan/Research/DeSpaMo/tools/dinov3/.venv/bin/python -B -m tools.dinov3.official_adapt --config configs/adaptation/phoenix14t_dino_merged40.json --mode run --authorize-adaptation
```

## Storage and automatic B: archive

Actual smoke serialization: compact model44,607,548bytes; full resume134,149,374
bytes; conservative retained-plus-scratch projection759,635,236bytes. Admission
and each completed epoch require projected demand plus **50GiB reserve** on both
Linux and Windows C: backing disk. No input video/cache duplicates are written.

When training finishes, close ledger and publish immutable `completion.json`
payload inventory plus selected epoch40 receipt. Then automatically copy/verify
to `/mnt/b/DeSpaMo/archives/official-phoenix14t-dino-adaptation-v1/<run-key>`.
Require actual B: mount and free space plus50GiB reserve. Archive is a verified
cold copy on NTFS, with exact file inventory/hashes and recorded directory-fsync
support; native receipt lives in `archive-receipts/verified.json`. No native
original is deleted. Failure to archive preserves completed training and writes
`archive-receipts/failure.json` for later retry.

Archival copies sealed payload plus completion file, not later mutable native
bookkeeping. Every future admission rehashes bytes; a marker alone is insufficient.
For training recovery, restore to exact native paths and verify receipts. Future
feature extraction may bind the archived epoch40 model under a new input contract.
Freeing C: VHDX bytes or deleting historical checkpoints needs separate cleanup
authorization and offline compaction after all active work ends.

## Verification and review record

Native implementation/self-review; no subagents or commits. Dedicated full suite:
**276passed/1skipped**, Ruff and Pyright passed. Regressions cover durable exact
checkpoint replay, all-ten loss weights, calibration no LoRA/weight-decay updates,
FP32 factor projection under outer BF16, finite gradients, source/data admission,
strict config, PID/accounting recovery, smoke-charge inheritance, and B: archive
integrity/unsupported-directory-fsync behavior.

Real CPU admission verified every train frame/teacher/target join. Real GPU smoke
verified calibration freeze, joint gradients, all30AdamW state tensors, native
checkpoint save/load and a subsequent CUDA optimizer step. Smoke receipts/time
are retained and inherited once in production. No production weights came from
smoke. Future model performance and final completion are not claimed by these
checks; use live state and epoch40 receipt when available.
