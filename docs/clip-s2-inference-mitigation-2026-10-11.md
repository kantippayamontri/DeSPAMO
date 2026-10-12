# Isolated inference mitigation — CPU prepared, GPU held

User approved explicit inference caching, evaluation microbatch1, CPU tests and
staged validation after Windows crash-dump/host mitigation review. This implements
the CPU preparation phase. It does not clear the GPU hold, change host drivers,
start a controller, or establish that VIDEO_TDR_FAILURE is fixed.

## Isolation and interfaces

Code resides in retained isolated worktree
`.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`:

- `tools/clip_s2_inference/generation.py`:
  `generate_cached_text(language, visual, visual_mask, prompts, *, beam_size=5,
  max_length=64, gpu_reviewed=False, on_start=None)`.
- `tools/clip_s2_inference/evaluation.py`:
  `score_dev_cached(model, admission, pool, split_hash, *, max_batches=None,
  gpu_reviewed=False, on_event=None)`.
- `tools/clip_s2_inference/tests/`: tiny real T5, real PEFT/LoRA and actual file-
  backed feature/dataset CPU regression tests. No network weights or GPU work.

New sibling namespace is excluded from the old runner's own-module source glob.
Every recorded v3 source hash still matches. Main `src/`, `tools/clip_s2/` and
inherited helpers were not edited; epoch1/2resumes remain native and hash-verified.
Existing unrelated main `visual_adapter.py` changes were preserved.

These components are opt-in and **not wired into the legacy training controller**.
Do not run the old controller expecting mitigation to be active. GPU work is
blocked by default in these helpers too: CUDA requires explicit `gpu_reviewed=True`
and, while the v3hold receipt exists, explicit `gpu_validation_approved=True` there.
Neither approval flag nor hold release was set during this work. Caller must check
the hold before constructing/loading anything onto CUDA, not just at generation.

## Implemented behavior

Cached generation reuses unchanged `FlanT5Backbone._joint_inputs`, then calls T5/
PEFT generation with **use_cacheTrue**, deterministic beam5/output64 defaults.
Training config stays use_cacheFalse and gradient checkpointing stays enabled.
Temporary eval mode restores every module's original flag (including mixed modes)
on success or failure. No model parameters/optimizer state are changed by helper.
No CUDA RNG restoration is attempted after a failure; primary exceptions propagate.
Valid visual tokens must be finite; masked NaN padding is excluded normally.

Dev evaluator explicitly uses old verified data loader with **batch_size1**. It
keeps official dev order, existing train-only precommitted ICL bank, full source
features, BF16CUDA autocast when later approved, and existing metrics. Raw dev
annotations are checked for reference overlap before any generation; no held-out
answer is supplied as a prompt. Autocast scope is per batch, not entire corpus.

Bounded `max_batches` selects the first fixed admitted clips without loading an
extra next batch. It always returns diagnostic_onlyTrue/metricsNone, even if bound
happens to cover all clips. Unexpected short/missing/reordered loader output fails.
Full metrics are available only with unbounded complete requested corpus. These
results lack legacy checkpoint-selection bindings and must not overwrite old
`dev-epoch-*.json` or trigger selection without a reviewed evaluation-policy bridge.

Telemetry events: batch_start, generation_start, batch_complete and batch_error.
They include UTC, batch index/clip IDs, spatial/motion/joint lengths, joint shape,
effective beam count, caching/decoding settings, elapsed seconds, allocated/
reserved/peak GPU memory after host review. CPU memory fields are null; CPU tests
do not query CUDA. Failure telemetry makes no CUDA memory/synchronization calls.
An observer's logging failure cannot mask the original device error. No retry loop,
watchdog or automatic GPU launch is included. Operator callback should flush each
event before generation so the last batch survives a host crash.

## Verification

TDD RED→GREEN observed for new generation/evaluation modules, cache/unchanged
configuration, batch1/ordered complete dev, bounded diagnostics, reference-overlap
rejection, failure restoration and unexpected-short-loader regression.

- 17new CPU tests pass; real cached T5/PEFT decoder receives one newest token each
  step (batch4/beam5 gives20beam rows), with same small-model output as uncached.
- 34new/nearby retained-source tests pass after final review changes.
- Ruff/format and targeted Pyright pass for new runtime modules.
- Configured main suite ran in main-only import context, CUDA_VISIBLE_DEVICES
  empty: **1625passed,4skipped,5failed**. Full-suite pass is not claimed.

Failures diagnosed, not altered to conceal them:

| Test | Observed reason |
| --- | --- |
| `tests/unit/evaluation/test_e3_heldout_cli.py::test_e3_heldout_run_writes_once_with_checkpoint_provenance` | Existing CLI requires CUDA; disabled for host-crash hold. |
| `tests/unit/evaluation/test_pilot_heldout_cli.py::test_run_writes_ordered_paired_reports_once` | Existing CLI requires CUDA. |
| `tests/unit/evaluation/test_pilot_heldout_cli.py::test_pair_rejects_reference_drift_without_summary` | Existing CUDA prerequisite is reached before intended reference check. |
| `tests/unit/evaluation/test_pilot_heldout_cli.py::test_resume_keeps_verified_e1_score_and_only_scores_e2` | Existing CLI requires CUDA. |
| `tests/unit/pilot/test_new_motion_fusion.py::test_final_summary_revalidates_full_graph_even_with_unchanged_marker` | Source-hash fixture lacks `scripts/run_fusion_study.py` key; unrelated main script/test not modified. |

No existing GPU test was enabled merely to obtain a green suite. No source/
checkpoint/runtime migration or training restart performed. Receipt:
[clip-s2-inference-mitigation-2026-10-11.json](clip-s2-inference-mitigation-2026-10-11.json).

CPU-only reproduction from retained worktree:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/kan/Research/DeSpaMo/src:.:/home/kan/Research/DeSpaMo OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_s2_inference/tests -q
```

## Dump-analysis update

Readable exports verified and official WinDbg installed with user approval.
Both local kernel-dump analyses confirm nvlddmkm.sys/0x116_IMAGE_nvlddmkm.sys and
matching timeout-reset stack; see [WinDbg report](windbg-gpu-crash-analysis-2026-10-11.md).
Dump-access stage below is now complete. Host-driver mitigation and bounded GPU
validation remain pending; inference helpers were not activated and hold remains.

## Remaining gated steps

1. Local Admin WinDbg analysis of both protected dumps from
   [host-reboot investigation](host-reboot-diagnosis-2026-10-11.md): `!analyze -v`,
   MODULE_NAME/IMAGE_NAME/FAILURE_BUCKET_ID/stack. Dump read access remains blocked
   from this session; no upload or automatic elevated action attempted.
2. Review driver/WSL/runtime mitigation with user; record permitted bounded
   validation separately from automatic-training permission. Preserve hold/history.
3. Only after that review, one cached batch1dev probe, then a small fixed subset
   with flushed timing/memory logs. Any driver reset/CUDA unknown error ends the
   attempt without retries or CUDA cleanup. Caching increases KV storage; no GPU
   memory or stability guarantee follows from tiny CPU tests.
4. Full dev evaluation in a fresh single validation process, with training GPU
   process absent, after bounded checks pass. Never load two full models on GPU
   simultaneously. No test scoring or stochastic/random-crop protocol changes.
5. Resume **retained epoch2weights/optimizer/sampler/RNG and elapsed charges**
   through an explicit compatible runner/evaluator binding. Old checkpoint load
   continues to use its exact original identity; new cache/batch policy and source
   hashes join new evaluation/migration receipt. Do not rewrite old identity/budget,
   drop charges, or create a fresh-root budget to bypass compatibility checks.
   Training batch8/recipe stays unchanged. No bridge implementation is claimed yet.

## Native self-review

Reviewed state restoration on nested/mixed module modes, cache override without
config mutation, true PEFT delegation, per-batch autocast lifetime, telemetry
callback failure, device-fault cleanup, all-split reference provenance, subset/full
distinction and old source closure isolation. Final review found missing strict
requested-corpus completion; added short-loader RED→GREEN regression and reran
34focused checks. Remaining limitation is protected host evidence/GPU stability,
not a claimed solved kernel bug. No commits or delegated agents.
