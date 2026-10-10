# Global/Local Experiment: Other-Server Handoff

## Start here

Branch: `feat/global-local-rvlf` in `kantippayamontri/DeSPAMO`.

```bash
git clone --branch feat/global-local-rvlf --single-branch git@github.com:kantippayamontri/DeSPAMO.git
```

Run later commands from cloned repository. If a checkout already exists, first
preserve its unrelated work, fetch this branch and create a separate worktree.
The original server continues its independent six-run study; no shared output
root, remote process control or edits to its checkpoint-bound source are needed.

Read these in order:

1. [SigLIP 2 encoder amendment](superpowers/specs/2026-10-10-siglip2-global-local-amendment.md)
2. [Architecture design](superpowers/specs/2026-10-10-global-local-rvlf-design.md)
3. [Superpowers implementation plan](superpowers/plans/2026-10-10-global-local-rvlf.md)

The earlier [CLIP/S² plan](superpowers/specs/2026-10-09-frozen-clip-s2-reference-design.md)
is retained as historical/optional comparison material, not the selected encoder
contract for this separate-server project. Current original-host DINO training
continues under its existing identity and features.

## What this branch contains

- Current spatial/motion adapter, plain MLP/fixed/learned fusion helpers and tests.
- Compact checkpoint, deterministic sampler,200epoch schedule/early-stop, ICL,
  translation kernel and provenance/storage source needed as implementation references.
- Motion extraction/provenance code with its separate uv lock.
- Approved architecture, implementation tasks and this handoff.

The new global/local adapter, regional SigLIP extractor, portable runner and
fine-grained loss are **not implemented yet**. Baseline tests verify published
source completeness, not new-method performance. Historical `tools/followup/`,
`tools/benchmark/` and Windows-aware storage helpers retain original-host paths;
do not launch their controllers here. Implement new `tools/global_local/`
config/runtime/entrypoints as the plan describes.

## Approved architecture

```text
Global SigLIP 2 tiled2304 + VideoMAE1024 -> global temporal -> G768 ---+
                                                                    |
Left-hand SigLIP1152 --+                                             |
Right-hand SigLIP1152 -+-> masked local MLP -> temporal -> L768 -------+
Mouth SigLIP1152 ------+                                             |
                                                                    v
                                                    support-aligned global/local fusion
                                                                    |
                                                      shared MLP768 ->2048
                                                                    |
                                         +--------------------------+-----------+
                                         |                                      |
                              one fine-grained VT-Align                   FLAN-T5-XL
                              against German token embeddings              translation
```

- SigLIP vision backbone and native attention-pooling head stay frozen; same
  weights serve all streams. Global384base plus four384tiles at768scale export
  2304channels; local single384crops export1152each. No CLS token/first-patch
  substitution. Mouth replaces full face; left/right identity is anatomical.
- Local regions combine first, global/local combine second. Fixed residual is
  the control; confidence-aware global/local gate is its matched extension.
- Masks remove missing evidence after biases and temporal/alignment processing;
  all missing returns exact global path. Confidence is an input, not proof of
  semantic relevance. Local per-region confidence gating is a later ablation.
- Global outputs are not automatically uniform frame timestamps. Derive receptive
  source support before aligning local outputs; never match token row numbers.
- RVLF-style fine-grained scores compare all visual/text segment pairs before
  aggregation; one fused contrastive objective plus CE, no auxiliary branch losses.
- FLAN raw frozen token embeddings differ from RVLF's contextual text encoder.
  Decoder Qwen3.5 and GRPO belong to later research stages, not this branch.

## Environment setup

Use Python3.11 and existing lock, not another package manager or guessed upgrades:

```bash
uv sync --locked
```

Main translation environment pins torch2.0.1, transformers4.32.0, PEFT0.7.1 and
NumPy1.26.4. Keep this runtime/lock and FLAN decoder unchanged. Transformers4.32
does not support the selected SigLIP checkpoint; source's old CLIP loaders are
not valid SigLIP extractors. DINO/motion tools have separate environments.

Task3 creates `tools/global_local/vision/` with its own uv project/lock and pinned
Torch/Transformers/Pillow stack. Supported starting candidate is Torch2.5.1,
Transformers4.56.2, Pillow11.3.0 and NumPy1.26.4; verify CUDA compatibility and
exact checkpoint pooling/processing before accepting features. Do not upgrade
main or original-host environments and do not mix import paths. New vision
project is an implementation task, not an already present directory. Provider
software may need another explicitly pinned tracker environment.
For motion tool work only, use its existing configuration:

```bash
uv sync --project tools/motion --locked
```

Do not mix that environment's imports into main training. Tracker software may
need another explicitly pinned preparation environment; no detector implementation
or checkpoint is assumed installed. Choose and verify provider through Task2.

CPU baseline verification from this checkout:

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 uv run --locked pytest tests/test_followup_recipe.py tests/test_followup_recovery.py tests/test_followup_study.py tests/unit/models tests/unit/losses tests/unit/data/test_motion_timing.py tests/unit/data/test_new_motion_package.py tools/benchmark/tests -q
```

The command uses synthetic fixtures; no model download/GPU/full training is
required. Legacy dependencies emit known deprecation warnings. Broader main unit
verification is `PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
OMP_NUM_THREADS=2 uv run --locked pytest tests/unit -q`.

Where Pyright is available, targeted baseline typing is:
`pyright --project pyright.global-local.json --pythonpath .venv/bin/python`.
Ruff checks must not skip the ignored worktree path during original-host review:
use `uv run --locked ruff check --no-respect-gitignore` with the intended paths.

Publication verification on original host:848main unit tests passed/4skipped;
313focused source tests passed/3skipped (these sets overlap; do not sum counts).
Four legacy fake-scoring CLI tests initially assumed CUDA; their isolated test
fixtures now simulate availability while production scoring guards stay unchanged.
No real held-out data was scored by these synthetic tests. GPU integration checks
and new global/local method tests remain for destination implementation.

## Data/model preparation outside Git

Git does not carry frames, annotations, model weights, feature arrays, checkpoints,
ZIP archives, environment secrets or server-specific output state. Prepare them
on destination's native storage and provide explicit local paths to new runner.

Required inputs:

| Input | Contract |
| --- | --- |
| Official annotations | train7096/dev519/test642 `*_info_ml.npy`, multilingual train text included |
| Original frames | ordered per-clip source PNGs,947756total, pinned frame IDs/content hashes |
| SigLIP 2 snapshot | `google/siglip2-so400m-patch14-384`,e8e487298228002f3d8a82e0cd5c8ea9c567f57f |
| SigLIP image processor | pinned fixed-resolution384metadata, RGB conversion explicit, mean/std0.5, native attention-pool readout |
| FLAN snapshot/tokenizer | `google/flan-t5-xl`,7d6315df2c2fb742f0f5b556879d730926ca9001 |
| Motion cache |1024width,16frame/stride8, original encoder/array/window/source provenance |
| Regional observations | left_hand/right_hand/mouth boxes, confidence, validity, tracks, original-frame indices and detector identity |
| Global/local caches | global2304/local3×1152per frame, masks/confidence/source receipts, independently verified publication |

Transfer model/cache/data bytes separately or regenerate with declared provenance.
Verify hashes/counts before loading. Do not copy original-host absolute-path
receipts and silently claim paths are unchanged: publish a verified new binding
while retaining origin receipts/array SHA. Do not transfer paused-controller state
to gain another budget or accidentally restart the old study.

Downloading public pinned model/processor snapshots is a separate preparation
step; only set `HF_HUB_OFFLINE=1` once required assets are complete. Never put a
Hugging Face token/private key into a tracked config or handoff.

## First experiments after implementation

1. Pooled versus fine-grained alignment on same global-only SigLIP/tiled baseline,
   following approved roadmap; keep everything else fixed.
2. With fine-grained alignment fixed: M0 global-only, M1 combined local fixed
   residual, M2 same local branch with confidence-aware global/local gate.
3. Matched seeds0/1, same global/FLAN initialization, crops/cache/decoder/schedule,
   dev selection and decoding. Profile memory and review marked defaults before launch.

Reference schedule: batch8/accumulation1, VT4000/LRwarmup10000,
adapter3e-4→5e-5/FLAN5e-5→1e-5, max200/min40/dev every2/patience10/minDelta0.10,
train-only3language-pair ICL, prompt1024/beam5/output64. Preserve actual-best
checkpoint separately from significant-improvement anchor. New loss masks padding.
No new automatic test scoring or test-guided model/region/reward tuning.

Each stage needs accepted profile, data/source identity, storage reserve and
owned checkpoint/recovery behavior. Original-server schedule remains untouched;
separate-server development can proceed independently. New-method BLEU and
robustness results do not exist at this publication stage.
