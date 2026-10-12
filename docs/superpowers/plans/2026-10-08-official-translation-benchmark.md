# Official PHOENIX14T Translation Benchmark Implementation Plan

> Use `executing-plans` natively with TDD/self-review. User approved six runs,
>40epochs, every2epochdev, highestBLEU4/earlier-tie selection, no early/time stop,
> then instructed "do it" after feature preparation completed. No Q-Former.

**Goal:** Launch matched conventional benchmark on official7096train/519dev/642test.
**Architecture:** Reuse approved plain spatial MLP and reference/fixed/learned
fusion plus FLAN-T5-XL LoRA. Standalone PyTorch update loop permits compact,
exact replay without saving frozen base weights in every Lightning checkpoint.
Pinned base/source/data bytes remain immutable identity; all mutable buffers,
trainable tensors, optimizer, sampling and RNG state are retained.
**Tech Stack:** Existing main `.venv` torch2.0.1/transformers4.32/PEFT, source
overlay in retained isolated worktree; no dependencies/old artifacts changed.

## Approved contract and execution rulings

- Standard splits, full spatial frames, same window16/stride8 motion alignment.
  Features are published epoch40DINO spatial and mixed-origin verified VideoMAE.
- Models:reference/aligned_fixed/aligned_learned x seeds0/1, sequential GPU jobs.
- Fixed40epochs; full dev every2epochs; best devBLEU4 with earlier tie; official
  test once logically after all six selections freeze. No Signer07 pilot scoring.
- Profile physical batches4/8 with ordinary/long batches and initialized AdamW;
  pick8 only with memory margin and successful generation/replay, otherwise4.
  Same selected batch across all arms/seeds. Drop-last train; no accumulation.
- Retain prior matched recipe: BF16, visual/VT LR3e-4, FLAN-LoRA5e-5,
  AdamW betas.9/.98, eps1e-8, decay.01, clipping1.0,10% LR warmup/cosinezero;
  VT-only first2000updates, then translationCE+VT(weight1). Model LoRA16/32/.1.
  Decoder beam5/max64/German-only task prompt/noICL, masked VT pooling.
- Compact checkpoints include every trainable parameter and persistent buffer;
  reject missing/extra keys or different frozen base/source identity. Exact
  optimizer/sampler/global Python/NumPy/Torch/CUDA and prompt RNG recovery tested.
- All20dev candidate models per run preserved as byte-verified cold B: checkpoints;
  native selected best and latest2full resumes retained. Only temporary current
  candidate files and superseded owned resumes may prune after durable publication.
  No old/original artifact deletion or WSL shutdown/compaction.
- Working root `/home/kan/datasets/despamo/official-phoenix14t-translation-v1`;
  B: checkpoint archive `/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-v1`.
  Linux/C:/B: projection and50GiB reserves checked. No walltime cap/forecast veto.

## Tasks

1. `tools/benchmark/compact_checkpoint.py` and tests:compact capture/load, strict
   frozen identity/mutable keys, buffers, AdamW and RNG replay; native atomic
   receipts and verified B: candidate copy. Watch RED then GREEN.
2. `translation_data.py`:admit exact manifests/source pairs, array hashes/lengths,
   distinct official IDs, train-only sampler and full dev/test loaders. No image
   extraction or dataset redefinition; retain original captions/normalization.
3. `translation_model.py`:same-backbone seeded model, loss/warmup/groups/schedule;
   tests calibration/joint gradients, shared initialization and replay.
4. `official_translation.py`:isolated profiles, fresh runs, owned ledger/source
   drift/storage checks,20dev epochs/selection, latest2resumes, finalized output;
   `run_translation_study.py`:sequential jobs, frozen selections, once-only test
   journal and summary. Test bad checkpoints/ownership/selection/finalization.
5. Real batch profiles and compact reload/generation equivalence; full targeted
   tests/static/native self-review; choose one batch, seal source/config identity,
   launch all six detached and verify actual optimizer advancement.

## Review focus

Frozen tensors accidentally optimized/omitted mutable buffers; replay changes
dropout/sampler after dev evaluation; test-based reselection; batch/padding long
clip OOM; corrupted archives or storage fallback; original study identities and
registries remain untouched. Metrics use existing SacreBLEU13a and ROUGE-L
implementation (ROUGE-L fractions, not percent), limitations explicitly reported.
