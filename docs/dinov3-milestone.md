# DINOv3 Milestone: Extraction And Fixed-Budget Comparison

## Status And Scope

Run label: `2026-09-26`. The extraction and controlled-comparison engineering
milestone is complete. All six training runs and full-test evaluations finished;
the report's provenance checks passed. Prediction quality remains limited by
substantial generic-output collapse; the cause has not been established.

Final source review found no significant merge blocker. Before finalization,
review findings were fixed with regression tests: cuDNN backend preservation,
DINO feature-content provenance through evaluation/reporting, AdamW moment and
scheduler-state validation, and complete report protocol enforcement.

Seven-factor data generation and dual-path supervision are subsequent stages
requiring permission. Large generation jobs, human caption audits, and controlled
training ablations have their own execution gates.

## Verified Corpus

| Split | Clips | Frame rows |
|---|---:|---:|
| Train | 7,096 | 827,354 |
| Dev | 519 | 55,775 |
| Test | 642 | 64,627 |
| **Total** | **8,257** | **947,756** |

- Model: `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- Immutable model revision: `ea8dc2863c51be0a264bab82070e3e8836b02d51`.
- Encoder key: `1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d`.
- Spatial manifest SHA-256: `558fbcb9bb1336c26586da2ade9706b94a713fd8c2228b717edbae57bd9e8cf1`.
- Frame-row map SHA-256: `538516cc3e7cee868207bda87d959818f197cf40f1235c4c5a363f2f8ee7a7bb`.
- Frozen RGB CLS features: Pillow bicubic square 224 and 448, ImageNet
  normalization, concatenate 224 then 448 into finite float32 `[T, 2048]`.
- Every source PNG contributes exactly one row; no flip, temporal subsampling,
  or 512-frame crop. Batch size 8; cuDNN TF32 disabled during forward while
  preserving other cuDNN flags. An eight-frame comparison against stored
  per-frame features had maximum absolute difference `3.725e-6`.
- `complete/manifest.json` and `complete/frame_rows.json` were published together
  after final validation; `failures.json` is `{}`. Per-clip receipts bind source
  hashes, ordered paths, sampled images, feature hashes, and encoder key.
- Raw Windows-hosted frames were copied to a Linux-local snapshot; the complete
  checksum dry run matched the source. The source includes one regular non-PNG
  profiler file; it is preserved by the snapshot and excluded from frame counts.

Extractor environment: Python 3.11, Torch 2.5.1+cu121, Transformers 4.56.2,
Pillow 11.3.0, NumPy 1.26.4, Hugging Face Hub 0.35.3. Its lock lives under
`tools/dinov3/`. Baseline translation dependencies remain Torch 2.0.1 and
Transformers 4.32.0; root `pyproject.toml` and `uv.lock` were unchanged.

## Matched Comparison Results

All runs used fresh trainable initialization, identical paired seeds, the pinned
Flan-T5-XL revision `7d6315df2c2fb742f0f5b556879d730926ca9001`, LoRA rank 16,
alpha 32, dropout 0.1, batch size 2, gradient accumulation 2, bf16, AdamW learning
rate `6e-4`, and weight decay 0.01. VideoMAE motion features, annotations, prompt,
optimizer schedule, full spatial sequences, and masked VT pooling were matched.
Only spatial source changed. Each run stopped at 1,000 optimizer steps and was
evaluated on all 642 test clips with beam size 5 and `do_sample=False`.

| Seed | CLIP BLEU-4 | DINO BLEU-4 | Delta | CLIP ROUGE-L F1 | DINO ROUGE-L F1 |
|---|---:|---:|---:|---:|---:|
| 0 | 4.462224 | 5.166133 | +0.703909 | 0.113755 | 0.143915 |
| 1 | 4.333058 | 4.305413 | −0.027645 | 0.115739 | 0.119094 |
| 2 | 4.279750 | 4.203875 | −0.075875 | 0.115958 | 0.115253 |

| Metric | CLIP mean ± sample std | DINO mean ± sample std | Paired delta mean ± sample std |
|---|---:|---:|---:|
| BLEU-4 | 4.358344 ± 0.093828 | 4.558473 ± 0.528691 | +0.200130 ± 0.436952 |
| ROUGE-L F1 | 0.115151 ± 0.001214 | 0.126087 ± 0.015558 | +0.010936 ± 0.016771 |

There is no robust improvement claim: DINO's BLEU-4 is lower in two seeds, and
the paired variability exceeds the mean difference. This is a 1,000-step
fixed-budget experiment, not a comparison of converged models. The released
SpaMo checkpoint and its approximately 25 BLEU-4 result are separate evidence.

## Prediction Audit And Interpretation

All 3,852 predictions were checked for emptiness, normalized exact duplicates,
length, and repeated trigrams. BLEU-1 through BLEU-4 and ROUGE-L metrics were
recomputed from every artifact and matched stored values within `1e-12`.
All six checkpoint-file SHA-256 values matched the evaluated artifacts.
Qualitative inspection used fixed test positions 0, 128, 256, 384, 512, and 641
for each run. References contain 630 distinct strings among 642 clips.

| Source | Seed | Unique normalized predictions | Most frequent sentence count | Empty predictions |
|---|---:|---:|---:|---:|
| CLIP | 0 | 38 | 146 / 642 | 0 |
| DINO | 0 | 124 | 210 / 642 | 0 |
| CLIP | 1 | 19 | 418 / 642 | 0 |
| DINO | 1 | 32 | 186 / 642 | 0 |
| CLIP | 2 | 22 | 337 / 642 | 0 |
| DINO | 2 | 4 | 627 / 642 | 0 |

For example, `05May_2011_Thursday_heute-3747` has reference
“dazu weht nur ein leichter südwind.” DINO seed 2 instead predicts
“und nun die wettervorhersage für morgen sonntag den dreiundzwanzigsten november.”
That same weather-introduction sentence occurs on 627 inputs (97.7%) in the run.
DINO seed 0 sometimes produces more varied weather descriptions, but also
within-sentence repetition: at test position 384, “im süden und südosten regnet
es” repeats rather than matching the reference's forecast.

The decoder is configured with a 64-token maximum, not a ten-word output limit;
the common ten-word outputs are repeated generated sentences. No decoding or
training settings were changed after inspecting the test results. The audit
does not identify the cause of collapse or show that longer training alone
would resolve it. Any follow-up tuning should use training/dev evidence and a
separately agreed protocol, rather than optimizing against these test outputs.

The software/data milestone can be integrated while retaining this limitation.
Before interpreting dual-path experiments, establish an adequate translation
baseline and matched controls; the short-budget checkpoints are not evidence
that signer/articulator factors have been disentangled.

## Artifact Locations And Provenance

On the experiment host:

```text
/home/kan/datasets/despamo/
  phoenix-frames-210x260px/          Linux-local source snapshot
  hf_cache/                        gated DINO model cache
  features/<encoder_key>/           version, receipts, features, complete manifests
  comparison-smokes-2026-09-26/      paired two-step smoke checkpoints
  comparison-2026-09-26/
    {clip,dino}/seed-{0,1,2}/
      checkpoints/final.ckpt        full Lightning checkpoint
      run_metadata.json
      test.json                    642 predictions, references, metrics, metadata
    logs/                          preflight, training, evaluation, report logs
    comparison-report.json
  comparison-six-runs.log           sequential job summary
```

Report SHA-256: `ea068ce728bc0d274a76109f8d3723235ffd26ee49981312301656c9f38f37ef`.
Comparison code/config SHA-256:
`e9c7563f539d85c6989e76b033ce80acf17d9aacc15d799dd40ffd7d07da97a9`.

Experiments ran from the uncommitted DINO worktree at baseline commit
`7e0499199a9bb277794392d550d90f5129151ea2`, so original artifacts correctly record
`git_dirty=true`. Their content-based code hash covers baseline dependencies,
source, scripts, and configs. Committing this milestone does not rewrite that
historical provenance. Fresh reproductions will have a different Git revision;
do not mix old and new run artifacts in one six-run report.

### Finalization Verification

- Root suite: **504 passed**, including real CLIP/DINO data consumers, released
  checkpoint schema parity, upstream temporal-adapter parity, and GPU batch transfer.
- Isolated extractor suite: **151 passed**, one opt-in gated GPU test deselected.
  Existing full extraction and bounded GPU parity checks provide separate live evidence.
- Full comparison preflight again verified both sources at 7,096 / 519 / 642 clips.
- Ruff, both lock checks, Git whitespace checks, and unchanged baseline dependency
  diff passed. All six checkpoint hashes and all stored metric values were rechecked.

## Reproduction Runbook

Run from the repository root. Existing corpus/report inspection requires no GPU.
Extraction and training commands below are separate expensive operations, not
part of ordinary verification. Use fresh external output directories for new
training/evaluation runs; preserve recorded artifacts.

### Environment

These are the paths used on the experiment host; substitute paths on other hosts.
`DESPAMO_DINO_ENV_FILE` points to the ignored token file, never shell-source it.

```bash
export PHOENIX14T_ANNOTATION_ROOT=/home/kan/Research/SpaMo/preprocess/Phoenix14T
export PHOENIX14T_FRAME_ROOT=/home/kan/datasets/despamo/phoenix-frames-210x260px
export DESPAMO_FEATURE_ROOT=/home/kan/datasets/spamo/features
export DESPAMO_HF_CACHE=/home/kan/datasets/spamo/hf_cache
export DESPAMO_DINO_CACHE=/home/kan/datasets/despamo/hf_cache
export DESPAMO_DINO_ENV_FILE=/home/kan/Research/DeSpaMo/.env
export DESPAMO_DINO_SHA=ea8dc2863c51be0a264bab82070e3e8836b02d51
export DINO_OUTPUT=/home/kan/datasets/despamo/features
export DINO_KEY=1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d
export DESPAMO_DINO_ROOT="$DINO_OUTPUT/$DINO_KEY"
export DINO_ROOT="$DESPAMO_DINO_ROOT"
# Required for CUDA library discovery on this WSL host:
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
```

### Extraction (after gated model access and explicit full-run authorization)

```bash
uv run --project tools/dinov3 --locked python -m tools.dinov3.cli \
  --env-file "$DESPAMO_DINO_ENV_FILE" --revision "$DESPAMO_DINO_SHA" \
  --cache "$DESPAMO_DINO_CACHE" --check-access

uv run --project tools/dinov3 --locked python -m tools.dinov3.cli \
  --env-file "$DESPAMO_DINO_ENV_FILE" --revision "$DESPAMO_DINO_SHA" \
  --cache "$DESPAMO_DINO_CACHE" --output-base "$DINO_OUTPUT" \
  --annotation-root "$PHOENIX14T_ANNOTATION_ROOT" \
  --clip-manifest "$DESPAMO_FEATURE_ROOT/manifests/phoenix14t_spatial.json" \
  --all --confirm-full-extraction
```

Interrupted extraction resumes only validated clips. An already complete version
is revalidated read-only, including raw frames; that is much more expensive than
the consumer and comparison checks below. Never bypass a failed receipt/hash
check by overwriting the recorded version.

### Consumer And Comparison Preflight

```bash
DESPAMO_RUN_DINO_BASELINE=1 uv run --locked python -m pytest \
  tests/integration/test_dinov3_baseline.py -q
COMPARISON_RUN_DIR=/home/kan/datasets/despamo/preflight \
  uv run --locked python scripts/compare_encoders.py preflight
```

The consumer must report one passed test, not a skip. Preflight checks both
complete spatial corpora against annotations and validates DINO row/content
hashes; expect 7,096 / 519 / 642 clips for each source.

### Paired Smokes And Six-Run Experiment

Set `COMPARISON_OUTPUT` to a new external directory, for example
`/home/kan/datasets/despamo/comparison-reproduction-01`. The shared layers and
source overlays must appear in the same order during training and evaluation.
This Bash helper only constructs those arguments:

```bash
export COMPARISON_OUTPUT=/home/kan/datasets/despamo/comparison-reproduction-01
common=(
  --config configs/data/phoenix14t.yaml
  --config configs/model/spamo_flan_t5_xl.yaml
  --config configs/experiment/phoenix14t_baseline.yaml
  --config configs/experiment/phoenix14t_encoder_comparison.yaml
)
overlay_for() {
  if [ "$1" = clip ]; then
    overlay=configs/experiment/phoenix14t_clip_control.yaml
  else
    overlay=configs/experiment/phoenix14t_dinov3.yaml
  fi
}
```

After bounded-smoke authorization, run one two-step smoke per source. Step zero
uses VT-only backward; step one includes translation backward. Both observed
smoke checkpoints loaded strictly and contained 288 LoRA optimizer states.

```bash
set -e
for source in clip dino; do
  overlay_for "$source"
  export COMPARISON_RUN_DIR="$COMPARISON_OUTPUT/smoke/$source/seed-0"
  uv run --locked python scripts/train.py "${common[@]}" --config "$overlay" \
    seed=0 comparison.smoke=true trainer.max_steps=2
done
```

After separate six-run authorization and successful smoke checkpoint checks:

```bash
set -e
for seed in 0 1 2; do
  for source in clip dino; do
    overlay_for "$source"
    export COMPARISON_RUN_DIR="$COMPARISON_OUTPUT/$source/seed-$seed"
    uv run --locked python scripts/train.py "${common[@]}" --config "$overlay" seed="$seed"
  done
done
for seed in 0 1 2; do
  for source in clip dino; do
    overlay_for "$source"
    export COMPARISON_RUN_DIR="$COMPARISON_OUTPUT/$source/seed-$seed"
    uv run --locked python scripts/evaluate.py "${common[@]}" --config "$overlay" \
      --checkpoint "$COMPARISON_RUN_DIR/checkpoints/final.ckpt" \
      --generation deterministic --output "$COMPARISON_RUN_DIR/test.json" seed="$seed"
  done
done
uv run --locked python scripts/compare_encoders.py report \
  --results-root "$COMPARISON_OUTPUT" --output "$COMPARISON_OUTPUT/comparison-report.json"
```

Do not resume smoke checkpoints into the full comparison. Keep code/config/lock
and data unchanged across runs; provenance checks intentionally reject drift.
For strict smoke reload checks, see the [comparison execution plan](superpowers/plans/2026-09-24-dinov3-comparison.md#task-4-gate-execution-and-reproducible-commands-documentation-for-later-authorized-session).

### Read-Only Report Audit And Offline Tests

```bash
uv run --locked python -c '
import json
from collections import Counter
from pathlib import Path
from despamo.comparison import summarize
root = Path("/home/kan/datasets/despamo/comparison-2026-09-26")
report = summarize(root)
assert report == json.loads((root / "comparison-report.json").read_text())
for run in report["runs"]:
    items = json.loads(Path(run["result_path"]).read_text())["items"]
    counts = Counter(" ".join(item["prediction"].lower().split()) for item in items)
    print(run["source"], run["seed"], len(counts), counts.most_common(1))
print("six-run report matches artifacts")
'
uv run --locked python -m pytest tests -m 'not gpu' -q
uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests -m 'not gpu' -q
uv run --locked ruff check scripts src tests tools/dinov3
uv lock --check
uv lock --project tools/dinov3 --check
```

Optional data/model tests may skip when their environment variables are absent;
those skips are not live verification. Recorded live evidence includes completed
extraction, the consumer test, source preflight, paired smokes with strict reload,
six exact-budget checkpoints, and six full evaluations. Full checkpoint hashes
and metrics were independently rechecked during milestone finalization.
