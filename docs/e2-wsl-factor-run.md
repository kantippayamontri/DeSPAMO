# E2 Projector-Factor Signer Pilot: Bounded Run

**Status:** Complete, 4,000 optimizer steps. Separate external run:
`/home/kan/datasets/despamo/signer-pilot/e2-only-64c971377519c5290411e13cbb0a7213ed3fc0104067db4e1858f37a74fbaac0/`.
No Signer07 test evaluation or signer probe was run.

The frozen signer protocol is
`0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7`.
Train: 5,746 seven-signer clips; full dev: 582 Signer03 clips; held-out
test: 768 Signer07 clips. E2 and fresh E1 use seed 0, original frozen DINO,
same fresh SpaMo/FLAN LoRA shared initialization hash
`9e3a02201ca44b7ee1a1d62da3eba1308a91ee914cab983036dc6505aa8a4d88`,
physical batch 4, no accumulation, 1,000 VT-only warm-up steps, 4,000
total steps, reference-free German prompts and deterministic beam-5 decoding.
WSL train/dev DataLoaders use `num_workers=0`.

E2 adds seven projector-level factor heads. CLIP targets use only the frozen
logical train subset: 5,641 schema-valid clips and 105 failed-caption clips
with all appearance targets false/zero; failed clips still contribute
translation training. Policy: `qwen-schema98-unreviewed-v1`,
`human_review_status=not_assessed`. No human appearance audit is claimed.

| Optimizer step | E1 Signer03 dev BLEU-4 | E2 Signer03 dev BLEU-4 | E2 minus E1 |
|---:|---:|---:|---:|
| 1,750 | 1.715840 | 0.157374 | -1.558466 |
| 2,800 | 3.598264 | 2.404917 | -1.193347 |
| **4,000** | **3.695629** | **3.028302** | **-0.667327** |

E2 selected its step-4,000 checkpoint by maximum full-dev BLEU-4;
ROUGE-L F1 was `0.127308`. Selected checkpoint SHA-256:
`7d73af1b8c3a190dd6b9f9725f7ebb01e63e3258837bb322e73494fb1b65575e`.
For all three steps, dev reports covered identical ordered 582 clip IDs and
references as E1. All corresponding checkpoint file SHA-256 values were
independently checked. E2's last dev had 262 distinct predictions; the
most frequent occurred 72 times.

`run-status.json` says `complete`; `budget.json` is closed at **2,624.82
GPU-seconds (0.729 h), including the 179.84-second E2 profile**. Overall
recorded pilot spend including previous E1 attempts and E1-v3 is
**8,760.78 seconds (2.434 h)**, below 24 hours. E2's six-hour cap was
respected. After completion, Windows C: had 44 GiB free.

The -0.667 dev BLEU-4 gap is an unfavorable screening signal, **not** the
predeclared 0.5-point held-out Signer07 gate: no Signer07 translations or
seven-class signer-probe accuracy were evaluated. Neither improvement nor
signer invariance has been established from this single-seed result.
