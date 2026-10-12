# CLIP+S² safe-generation CPU recheck

User requested keeping CLIP+S² available and testing it too after plain-CLIP
implementation. This follow-up tests existing S² code and compatibility with the
new safe generation core; it does not change the frozen S² runner or release GPU
hold. Plain-CLIP implementation remains separate and byte-identical.

## Verification

-47 tests passed in main runtime: original S² loader, training/config/ownership,
  guards/report/controller tests plus existing opt-in cached inference tests.
-7 encoder/package tests passed in the separate motion-extractor interpreter,
  which already contains Pillow. No dependencies or import paths were mixed.
-3 additional CPU compatibility tests passed using2048-wide inputs, actual
  original visual-adapter structure, and a small real T5/PEFT-LoRA model:
  both1024 spatial halves influence projection; output width2048 is accepted by
  cached beam5 generation; decoder receives one token per step; raw logits remain
  finite; training cacheFalse remains unchanged; injected primary error propagates
  without CUDA RNG/memory/synchronization cleanup.
- New test Ruff check/format and both worktrees' diff whitespace checks passed.
- Original84 S² source files, epoch2 snapshot/receipt, budget and manual-stop
  hashes remain unchanged.1774updates/497.5533489369991s retained.
- Plain runtime digest still
  `bc3a3c2f6ac27d149165435b034a1c6bad7ef550eb3f2d32925035e2148a1827`.

New compatibility test is in retained worktree:
`tools/clip_s2_inference/tests/test_safe_generation_2048.py`, SHA256
`98db7e700462e52c56b22da42c6040cde8d35b0b2cd44d4b6fb465410c1b94d8`.

Commands from retained worktree, with CUDA_VISIBLE_DEVICES empty, existing
src/runtime/main import overlay, OMP_NUM_THREADS2,HF_HUB_OFFLINE1:

```text
main .venv/bin/python -B -m pytest tools/clip_s2_inference/tests tools/clip_s2/tests --ignore=tools/clip_s2/tests/test_encoder.py --ignore=tools/clip_s2/tests/test_package.py -q
tools/motion/.venv/bin/python -B -m pytest tools/clip_s2/tests/test_encoder.py tools/clip_s2/tests/test_package.py -q
main .venv/bin/python -B -m pytest tools/clip_s2_inference/tests/test_safe_generation_2048.py -q
```

First command was run before adding the3 follow-up tests. Total unique checks in
these runs:57passed. Final formatted main-runtime rerun passed all50 checks;
the separate7 extractor checks had already passed. The small compatibility T5 uses d_model2048 but only one
encoder/decoder layer, vocabulary32 and d_ff64; it is not pretrained FLAN-T5-XL.
No native FLAN GPU-generation, long-clip stability or translation quality claim.

## What remains pending

The safe generation core is dimension-compatible with S² outputs. The new
plain-CLIP data/model/state policies are intentionally1024-only and cannot admit
old S² snapshots. Existing `tools/clip_s2_inference` helpers are still opt-in;
old `tools/clip_s2.study` remains unsafe/held. Restoring a production S² study
with fresh train/dev processes needs its own checkpoint-compatible integration,
not editing the old source hash identity or launching the old controller again.

Earlier authorized short cached S² GPU probes passed, but their approvals are
consumed. Current request did not specify a new GPU clip/count/device scope.
Next proposed GPU diagnostic is a separately reviewed fixed short/long-clip
subset with batch1/cacheTrue/beam5/max64, fresh optimizer-free process, durable
telemetry and no retry/training or Windows-device changes. No GPU test has been
started in this follow-up.
