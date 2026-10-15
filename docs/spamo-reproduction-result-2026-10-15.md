# SpaMo reproduction result — 2026-10-15

## Claim

Training original SpaMo with the settings recovered from the released `spamo.ckpt`
(not the recipe stated in the paper) reproduces the paper's PHOENIX14T test result:

**clean test BLEU-4 24.59** (dev-selected epoch 62) vs paper **24.32**.

One seed (0). Checkpoint selected on dev only; test scored once after training stopped.

## Results (642 official test clips, clean protocol)

| System | Selection | Dev BLEU-4 | Test BLEU-1 | Test BLEU-4 | Test ROUGE-L F1 |
|---|---|---:|---:|---:|---:|
| Paper (NAACL 2025, Table 1) | test (author config) | — | 49.80 | 24.32 | 46.57 |
| Released checkpoint, clean protocol | author (test) | 23.34 | — | 25.27 | 0.473 |
| **Ours, spamo.ckpt recipe, epoch 62** | **dev** | **24.56** | **50.66** | **24.59** | **0.464** |
| Ours, spamo.ckpt recipe, epoch 84 (diagnostic, last epoch) | none | 23.95 | 50.19 | 23.74 | 0.457 |
| Ours, paper-stated recipe, epoch 36 | dev | 20.35 | 46.33 | 20.10 | 0.420 |

Clean protocol: fixed training-set exemplar `12February_2010_Friday_tagesschau-122`,
batch 1, beam 5, max 64, no sampling, cached decoding. Metrics recomputed independently
from saved predictions (match exactly). 621/642 unique predictions, 0 empty (epoch 62).

## Training (spamo.ckpt recipe)

- Released original-linear SpaMo, released CLIP+S² (2048) and VideoMAE (1024) features.
- FLAN-T5-XL rev 7d6315df, LoRA q/v 16/32/0.1, BF16, seed 0.
- AdamW 0.9/0.98, eps 1e-8, wd 0.01, clip 1; batch 8 × accumulation 2 (444 updates/epoch).
- LR peak 6e-4, cosine with 10% warmup over 222,000 updates (500-epoch horizon), floor 0.
- No VT-only warmup (CE+VT from update 2); training ICL = in-batch derangement.
- Dev every 2 epochs; early stopping patience 50 checks.
- Stopped by operator after epoch 84 (37,296 updates, ~20.4 h), user approved after dev
  plateau (11 checks ≥ 23.0 since epoch 60; best epoch 62, 11 checks without improvement).
  Early stopping had not fired. Record: `operator-stop.json`.

## Root cause of earlier failure

The paper states peak LR 1e-4, 10K warmup, floor 5e-5, 40 epochs, ~4K VT-only warmup.
The released checkpoint's own state shows LR 6e-4, 22,200-update warmup over a 500-epoch
cosine, floor 0, 96 epochs trained; author code uses no VT-only warmup. Training with the
paper-stated recipe reached only 20.10 test BLEU-4. Settings export:
`docs/spamo-ckpt-training-settings.json`.

## Limits

- Single seed; PHOENIX14T seed variance is roughly ±1 BLEU-4.
- Training stopped manually at epoch 84 of a 500-epoch schedule; a later epoch might
  select differently.
- Author selected on test; we selected on dev (stricter). Epoch 84 diagnostic shows
  ~0.9 BLEU-4 test sensitivity to checkpoint choice.
- Physical batch 8 / accumulation 2 inferred from checkpoint counters (author YAML says 2).
- Paper decoding samples (top_p 0.9); we decode deterministically.

## Artifacts

- Run: `/home/kan/datasets/despamo/original-spamo-upstream-released-recipe-seed0-v1/`
  (`state.json`, `dev/`, `test/test-epoch-{62,84}.json`, `operator-stop.json`).
- Checkpoints: `/mnt/b/DeSpaMo/checkpoints-v1/original-spamo-upstream-released-recipe-seed0-v1/resumes/`
  (epoch 62 = `resume-update-27528.pt`, epoch 84 = `resume-update-37296.pt`, hashes verified).
- Code: `.superpowers/sdd/2026-10-13-original-spamo-paper-reproduction/worktree/tools/spamo_repro/`.
