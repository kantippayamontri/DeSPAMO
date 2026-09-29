# Signer-Disjoint E1/E2 Pilot

Frozen external protocol:
`/home/kan/datasets/despamo/signer-pilot/protocol/0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7/protocol.json`.
Its split hash is
`a95fa8f2710ebca33d6565a26f0f076817d839ecc611bc43648a48761fd6527b`.

| Role | Signers | Clips | Qwen valid | Qwen failed |
|---|---|---:|---:|---:|
| Train | 01, 02, 04, 05, 06, 08, 09 | 5,746 | 5,641 | 105 |
| Dev | Signer03 | 582 | 571 | 11 |
| Held-out test | Signer07 | 768 | 765 | 3 |

All three logical groups originate from original physical PHOENIX `train/`.
Training uses only seven train signers; failed Qwen captions keep translation
targets but have all appearance factors masked. Dev references are used only
for checkpoint selection. Generation prompts contain no held-out references.
Signer07 remains unscored, and official PHOENIX dev/test remain untouched.

Fresh matched seed-0 E1 (frozen original DINO, no factor heads) and E2
(same DINO plus seven projector-level factor heads) each trained 4,000
optimizer steps, physical batch 4, VT-only warm-up 1,000, WSL
`num_workers=0`. Both start with shared trainable tensor hash
`9e3a02201ca44b7ee1a1d62da3eba1308a91ee914cab983036dc6505aa8a4d88`.
Full 582-clip dev evaluation uses deterministic beam 5 at joint-phase
steps 1,750, 2,800 and 4,000, choosing highest corpus BLEU-4 (earliest
step on a tie). See [E1 result](e1-v3-wsl-retraining.md) and
[E2 result](e2-wsl-factor-run.md) for checkpoint hashes and measured scores.

The original three-way E1/E2/E3 20-GPU-hour admission gate failed on a
measured estimate; its external receipt is
`/home/kan/datasets/despamo/signer-pilot/profile-gate-b4.json`. E1/E2 were
subsequently authorized as separate bounded runs. Full E3 adaptation,
signer-probe evaluation, and Signer07 held-out translation tests have not
run. Dev scores alone cannot establish reduced signer information.
