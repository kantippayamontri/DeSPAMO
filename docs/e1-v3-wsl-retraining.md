# E1-Only WSL Retraining Result

**Status:** Completed 4,000 optimizer steps. Run identity:
`signer-pilot-e1-fresh-retry-v3`. Original partial runs remain external and
untouched. This run used a separately authorized six-GPU-hour cap, with the
previous 4,019.05 recorded GPU-seconds counted against the original
24-GPU-hour overall ceiling. Unclean shutdown downtime was not represented
as GPU residency in this new versioned run.

Frozen signer protocol:
`0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7`.
Train: 5,746 seven-signer clips from physical PHOENIX `train/`; dev: all
582 Signer03 clips; held-out Signer07: 768 clips, **not scored**.

Fresh seed-0 SpaMo/FLAN-T5-XL LoRA, original frozen DINO features, physical
batch 4, no accumulation, WSL `num_workers=0` for train and dev loaders.
VT-only warm-up: 1,000 steps. Prompt is reference-free; decoding is
deterministic beam 5, maximum 64 tokens.

| Complete dev checkpoint | Corpus BLEU-4 | ROUGE-L F1 |
|---:|---:|---:|
| 1,750 | 1.715840 | 0.098241 |
| 2,800 | 3.598264 | 0.145504 |
| **4,000 (selected)** | **3.695629** | **0.148572** |

Each report has exactly 582 Signer03 clip IDs; checkpoint SHA-256 values
were independently recomputed. Selected checkpoint SHA-256:
`04ce223fd1ae00362a7ceaa1e78c1a814c825a8b9551483d8ac2a332631c7755`.
Final dev has 369 distinct predictions; most common one occurs 27 times.
One seed and one held-out signer remain pilot design limitations; dev BLEU
does not establish Signer07 translation quality or signer invariance.

External run root:
`/home/kan/datasets/despamo/signer-pilot/e1-v3-79d14f8b66093483f683102f06bf28a23253d82d056ca03be0771725a170b533/`.
Source-of-truth `run-status.json` says `complete`, 4,000 steps, **2,116.91
GPU-seconds (0.588 h)** for v3; overall recorded pilot spend including the
prior failed runs is **6,135.96 seconds (1.704 h)**. `budget.json` is closed.
Both limits (21,600 seconds for this run; 86,400 overall) were respected.
After completion, Windows C: had 102 GiB available, Linux volume 279 GiB.
No E2/E3 training or final Signer07 evaluation was launched.
