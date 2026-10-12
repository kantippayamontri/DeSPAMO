# Residual projector and longer-training control

## Execution basis

The user approved the normalized residual-MLP projector direction and a longer
linear-projector training control, then explicitly requested launching now while
deferring Windows disk compaction. These concrete implementation choices are
recorded before execution. Existing unrelated working-tree changes are preserved.
No commit, merge, or push is authorized.

## Question and experimental arms

Does a more expressive downstream projector improve dev translation when the
adapted visual features are fixed, or does the current adapter primarily need
more optimization?

| Arm | Projector | Total steps | VT-only warmup | LR warmup | Dev checkpoints | Wall-time cap |
|---|---|---:|---:|---:|---|---:|
| Historical reference | Linear | 8000 | 2000 | 800 | 3500, 5600, 8000 | completed |
| Residual experiment | Linear + normalized MLP | 8000 | 2000 | 800 | 3500, 5600, 8000 | 10800 s |
| Longer control | Linear | 16000 | 2000 | 1600 | 5500, 10400, 16000 | 14400 s |

The longer control declares a doubled scheduler horizon and the existing 10%
LR warmup convention; it is not an extension of an exhausted 8000-step scheduler.
Both new models start fresh. Their physical batch is 4, seed 0, no accumulation,
masked VT pooling, VT weight 1, adapter LR 3e-4, FLAN LoRA LR 5e-5, AdamW betas
(0.9, 0.98), weight decay 0.01, and cosine decay to zero.

## Fixed inputs

Use the completed cosine DINO features
`e3-features-7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e`.
Historical baseline is
`e3-only-0402717f5ef91cca106caa881b689ce1eba75a3a87976302c99c654bea189690`,
whose selected dev BLEU-4 is 4.709851. Verify its completion, selection and hashes.
Preserve the frozen logical split: 5746 train, 582 Signer03 dev, 768 Signer07 test.
The other server's new descriptions are not inputs to this cached-feature ablation.
Keep `qwen-schema98-unreviewed-v1` and `human_review_status=not_assessed`.

## Residual mapping

Wrap each existing spatial/motion linear projection with
`base(x) + Linear(1024, 768)(GELU(Linear(input_width, 1024)(LayerNorm(x))))`.
Input widths are 2048 and 1024. Zero-initialize the final residual linear weight
and bias, retaining the existing base module and its parameter values. Initialize
new branches with seed 1 inside a CPU/CUDA RNG fork; restore the caller's RNG.
This adds 4,728,320 trainable parameters and initially reproduces the baseline
mapping. Temporal fusion, TCN, cross-modal output MLP and FLAN LoRA remain the
same experiment components. The linear arm uses the original modules unchanged.

## Runner and provenance

Use a new `train_projector_pilot.py` entrypoint and
`signer-pilot-projector-ablation-v1` run policy. Do not relax the frozen E1/E2/E3
step tiers or reuse their completed output directories. Bind input identities,
baseline selection, architecture, horizon, caps, retention and full source-code
closure into a new run key. Verify the published base initialization hash before
attaching residual branches; record the variant initialization hash separately.
Select each arm's checkpoint on its complete full-dev schedule, using deterministic
beam 5 / max length 64 and no in-context examples. This launch scores dev only.

## Storage and recovery

Save full resume snapshots every 500 steps, retaining the latest two only in each
new run. Prune older snapshots only after the new snapshot was atomically published;
a failed save must preserve earlier recovery states. Preserve every dev-selection
checkpoint and report. Require 50 GiB Linux free space for a run and record the
user's accepted Windows-space constraint. Never prune another run or prior artifacts.
Persist budgets, run status and gradient-smoke receipts; resume must verify identity,
latest checkpoint, optimizer, sampler, RNG and scheduler continuity.

## Verification and acceptance

- Initial residual outputs and original base parameters match the linear baseline.
- Additional initialization does not advance caller RNG.
- Both residual branches receive gradients; early hidden weights learn after the
  zero-initialized output weights move. Masked inputs receive zero gradients.
- Strict checkpoint round-trip reconstructs the selected architecture.
- A real four-clip joint CE+VT gradient smoke succeeds before training, with no
  optimizer update. Restore all RNG and model buffers and clear smoke gradients.
- Retention keeps only two resumes, preserves dev checkpoints, and never prunes
  after a failed checkpoint write.
- New schedules are explicit; frozen tier validation remains unchanged.
- Full tests/lint, targeted types, reviewer and real-data preflight pass before launch.
- Run stages sequentially on the 16-GiB RTX 4080 SUPER; report observed progress.

Translation improvement is judged by dev BLEU and prediction diagnostics, not
the appearance probe's unvalidated heuristics. A Signer07 evaluation requires a
dev-selected checkpoint and a separate once-only report.
