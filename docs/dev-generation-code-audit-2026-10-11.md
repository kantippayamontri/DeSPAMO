# Original dev-generation code audit — CPU only

**One confirmed error-cleanup defect and one confirmed generation inefficiency;
initial GPU/root cause remains unresolved.** No GPU test, training restart or
checkpoint-bound code modification occurred in this audit.

## Trace and exact failure surface

Original CLIP dev path:

```text
tools/clip_s2/train.py:145 score_dev
  -> tools/clip_s2/data.py:150 eval_batches(batch_size=4)
  -> tools/followup/evaluation.py:20 score_batches
  -> visual_adapter(spatial, spatial_mask, motion, motion_mask)
  -> src/despamo/models/flan_t5.py:140 generate_text
  -> PEFT0.7.1 / Transformers4.32.0 generate
  -> beam_search -> torch.topk at generation/utils.py:3057
```

Latest recovery's **first** CUDA illegal-instruction exception surfaced at
`torch.topk`, not in spatial feature loading or adapter indexing. CUDA is
asynchronous: this identifies the reporting API, not necessarily the faulty
GPU kernel. The older unknown-error trace surfaced elsewhere; do not conflate
those two Python stacks.

Installed runtime verified without CUDA initialization: torch2.0.1+cu117,
compiled CUDA11.7, torch git revision
`e9ebda29d87ce0916ab08c06ab26fd3766a870e5`.

## 1. Confirmed unsafe cleanup after a CUDA failure

`tools/clip_s2/train.py:174-178` unconditionally calls `restore_rng(rng)` from
`finally`. `tools/benchmark/compact_checkpoint.py:48-53` then calls
`torch.cuda.set_rng_state_all` when the saved state contains CUDA RNG data.

If generation has already faulted, this touches the failed CUDA context again.
The real failed-recovery log contains a second illegal-instruction exception
from this restoration, replacing the first exception as the final propagated
error. Python retains the first error in exception context, so its evidence is
not completely erased, but final error attribution can be misleading.

A CPU-only fault injection into **actual `score_dev`** reproduced that behavior:

```text
Injected primary: PRIMARY simulated generation fault
Propagated error: SECONDARY simulated CUDA RNG fault
Primary context: PRIMARY simulated generation fault
```

Initial fixture lacked a sampler stub and failed before the intended path. That
fixture was corrected; the focused injection then passed and was rerun after
cleanup of the diagnostic script. No real CUDA error was induced.

**This is a secondary-error handling defect, not proof of the initial GPU crash
cause.** DINO's `tools/followup/train.py:143-147` shares the same cleanup pattern.
Future compatible evaluator should abort a CUDA-faulted process without CUDA
RNG/memory/synchronization cleanup. The opt-in cached helper already avoids it;
old controller has not been migrated to that helper.

## 2. Confirmed inherited uncached decoder work

`tools/followup/model.py:19-23` enables training gradient checkpointing and sets
`transformer.config.use_cache=False`. `generate_text:153-162` supplies no
inference cache override, so dev inherits false. `model.eval()` does disable
training-only checkpoint execution, but it does not reset that config value.

Installed T5 code explicitly uses checkpointing only when both
`gradient_checkpointing` and `training` are true. With cache absent,
`prepare_inputs_for_generation` passes the full growing decoder prefix. With
past keys/values, it sends only the newest decoder token.

Tiny **real installed T5 BF16 CPU** probe, batch4/beam5/vocabulary32128,
forced8-token output:

| Setting | Decoder lengths seen | Decoder-token positions | Last-logit slice |
| --- | --- | ---: | --- |
| cacheFalse |1,2,3,4,5,6,7 |560 |BF16,non-contiguous |
| cacheTrue |1,1,1,1,1,1,1 |140 |BF16,contiguous |

Generated IDs were identical. This proves redundant decoder work and a layout
change, not native-model GPU timing, memory reduction or stability. Non-contiguous
logits are supported tensors; this is not by itself an indexing bug. Caching
also adds KV storage, so it is not automatically a lower-VRAM policy.

This inefficiency is shared by successful DINO evaluation and failing CLIP
evaluation. It is not sufficient explanation for their differing outcomes.

## 3. Full-dev CPU spatial/adapter sanity checks

All **519 dev clips /130 original batch4 groups** were loaded from each package
and processed with its corresponding saved **seed0 epoch2/1774-update adapter**.
The original adapter structure/weights, temporal BatchNorm state, eval mode and
CPU BF16 autocast were used. This is adapter-only validation, not full FLAN
generation and not GPU numerical equivalence.

| CPU measurement | CLIP+S² /CLIP adapter | DINO /DINO adapter |
| --- | ---: | ---: |
| Non-finite input batches |0 |0 |
| Non-finite valid visual-output batches |0 |0 |
| Invalid/empty output masks |0 |0 |
| Maximum spatial absolute value |7.039008 |2.429468 |
| Maximum valid visual absolute value |6.5625 |3.71875 |
| Maximum spatial frames |272 |272 |
| Visual token range |1–73 |1–73 |
| Fixed inference prompt tokens |146 |146 |
| Maximum joint visual/prompt tokens |219 |219 |
| Maximum prompt token ID |30355 |30355 |

Pinned FLAN vocabulary is32128, so checked prompt token IDs are in range.
Spatial tensors are floating embeddings, not token IDs used to index vocabulary.
No malformed shape, empty temporal output, obvious spatial overflow or mask
misalignment was found on CPU. CLIP values are larger, but the two adapter
checkpoints also differ; these norms are not an isolated causal comparison.
Full FLAN/GPU hidden states and logits can still require separate validation.

Longest-joint batch IDs:

```text
29January_2010_Friday_tagesschau-6489
29January_2010_Friday_tagesschau-6490
03November_2010_Wednesday_tagesschau-6505
03November_2010_Wednesday_tagesschau-6506
```

## 4. The550>512 warning is not demonstrated dev-input overflow

Setup scans the full training ICL bank with `tokenizer.encode` in
`validate_prompt_lengths`; tokenizer's nominal model_max_length is512 while
approved prompt limit is1024. Actual seed0 dev uses one fixed146-token prompt,
and full audited joint length is at most219.

T5 uses relative position buckets rather than a512-entry absolute position
embedding table. A separate tiny installed T5 encoder accepted600 embedding
tokens with finite output. This does not certify long-context translation
quality, but rejects treating that generic tokenizer warning as proof of a
512-bound indexing failure in these dev batches.

## 5. Beam/top-k dimensions are valid; exact CUDA path still unproven

The installed CPU probe confirms:

```text
next_token_logits: [20,32128], BF16
candidate scores into topk: [4,160640], FP32
requested k: 10
```

Thus k is within dimension size, and top-k scores are already FP32 after beam
score addition. A claim that the failing top-k necessarily operated on BF16 is
incorrect. Finite synthetic decoder logits produced valid outputs in both cache
modes; this is not a reproduction of the GPU fault.

Upstream torch2.0.1 CUDA sources were inspected as well. The full-sort heuristic
is gated by `disable_sort_for_topk()` and CUB scan-by-key support; do not infer
an actual full-sort path from tensor dimensions alone. The gather-top-k source
supports both single/multiblock implementations and makes input contiguous.
Without an actual GPU trace, no particular CUB/top-k/attention kernel or library
defect has been identified. A previous asynchronous attention/GEMM error can
surface at the later top-k launch.

## 6. Missing observability and weaker memory-lifetime candidates

Original scorer wraps **whole corpus** in one BF16 autocast scope and retains
training model/optimizer. No per-batch clip/timing/peak-memory or pre-top-k
numerical telemetry is flushed. Dev report is published only after completion.
Consequently latest failing batch and decoding step cannot be identified from
that log. This is a major evidence gap.

No obvious per-batch GPU-tensor accumulation was found: prediction storage is
CPU strings, and default generation does not collect output scores/attentions/
hidden states. Whole-corpus autocast can retain parameter casts until scope exit,
but unbounded cache growth/leak was not established. Different padded lengths,
beam lifetimes and in-process allocator state remain workload hypotheses, not
proven causes. Missing original GPU peak data prevents a VRAM conclusion.

## Next diagnostic/recovery recommendation

Use the isolated cached evaluator through an explicit compatible bridge, with
batch1, per-batch autocast and durable clip/input/decoder telemetry. Preserve
original checkpoint identities and elapsed accounting. Any later GPU debugging
needs new scope; do not rerun crashing batch4/cacheFalse merely to add logs.

If kernel-specific reproduction is separately reviewed, inspect decoder logits
before log-softmax/top-k and record decoder length/dtype/stride. Distinguish NaN
or positive infinity from legitimate masked negative-infinity beam scores. A
small shape-matched top-k-only test could narrow backend hypotheses without a
full FLAN model, but CUDA asynchrony and host-crash history still require a
reviewed mitigation/scope. None was run by this audit.

## Artifacts/verification

Standalone CPU analysis:
`docs/diagnostics/dev-generation-cpu-audit-2026-10-11.py`, SHA256
`cdf9c4746e03efb5b54f39fe288c6da44e4b6884108ac59ef76ce07f7d79dfc0`.
Native same-root namespace:`diagnostics/dev-generation-code-audit-2026-10-11/`.
Contains `features.json`, `decoder.json`, `cleanup.json`. CPU probes assert no
CUDA initialization; both checkpoint payload hashes verified against receipts.
Ruff check/format passed. No checkpoint-bound implementation fixed or edited.
Hold and revoked GPU permission retained; no training/test reports published.

Version-specific backend references:

- https://raw.githubusercontent.com/pytorch/pytorch/v2.0.1/aten/src/ATen/native/cuda/TensorTopK.cpp
- https://raw.githubusercontent.com/pytorch/pytorch/v2.0.1/aten/src/ATen/native/cuda/TensorTopK.cu
- https://raw.githubusercontent.com/pytorch/pytorch/v2.0.1/aten/src/ATen/cuda/cub_definitions.cuh
