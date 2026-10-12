# Three-arm spatial projection / query connector study

## Approval and goal

User approved reference linear projection, plain spatial MLP and a small Q-Former
replacing the final visual-to-language output MLP, with VT-Align retained. User
selected two seeds and authorized up to **64800 seconds (18 GPU-stage hours)**
for six fresh sequential 16000-step training/dev-selection runs. No Signer07
generation, data regeneration, prior-artifact deletion, commits or pushes.

## Fixed recipe

Use cached cosine DINO features `7122b42e...` and original VideoMAE motion. Logical
train5746/dev582/test768; only Signer03 dev is scored. Seed0 and seed1 each begin
from the same attested shared base initialization across all three arms. The
model's discarded parameters are initialized before replacements, and new modules
use a forked CPU/CUDA RNG (seed100+training_seed) so common weights/RNG are preserved.
Sampler and prompt/dropout RNG follow training seed. Record common and arm hashes.

All arms: 16000 steps, batch4, no accumulation, VT-only warmup2000, LR warmup1600,
cosine to zero, adapter LR3e-4, FLAN LoRA LR5e-5, AdamW betas0.9/0.98, eps1e-8,
weight decay0.01, gradient clip1, BF16. Fixed decoder beam5/length64/noICL.
Complete full-dev evaluations at5500/10400/16000 choose best BLEU-4. Resume
retention latest2 plus every dev-selection model, atomic file/directory fsync.
Each run cap10800seconds (3h); preflight/init/smoke/setup/validation count
conservatively toward overall budget. Stop incomplete on cap/integrity failure.
Labels remain `qwen-schema98-unreviewed-v1` / `human_review_status=not_assessed`.

## Architectures

### Reference

Spatial Linear2048→768; motion Linear1024→768; concatenate valid sequences along
time; original K5/P2/K5/P2 TCN; output Linear768→2048/GELU/Linear2048→2048.
Existing module keys/outputs must remain identical for reference.

### Plain spatial MLP

Replace only spatial Linear with Linear2048→1024/GELU/Linear1024→768.
No LayerNorm, skip connection or dropout in this MLP: the requested ablation is
the two-layer nonlinear mapping. Motion, TCN and final output MLP unchanged.

### Small Q-Former connector

Keep spatial/motion Linear and original TCN. Replace the output MLP with a query
connector:32 learned queries,768width,2pre-normalized blocks,12heads,FFN3072,
dropout0.1, final LayerNorm and Linear768→2048. Each block performs query
self-attention, cross-attention to valid TCN outputs, and FFN with residuals.
Add fixed sinusoidal temporal positions to TCN outputs before cross-attention;
normalize visual memory and zero padded memory, masking padding with MHA's
`key_padding_mask=~valid`. Reject empty/invalid visual sequences.

This is a from-scratch lightweight Q-Former-style connector, not the pretrained
BLIP-2 text/query model. It attends to temporal frame-summary features, not image
patches. All32 outputs are valid visual tokens. VT-Align pools these projected
query outputs against the fixed target embedding layer. Decoder loss/VT weight
are unchanged. Fixed32tokens can expand short sequences and compress long ones;
that architectural/token-budget change must be reported, not attributed purely
to parameter count. Position encoding is part of this predeclared arm.

## Implementation and identities

Opt-in `connector_adapter.py` constructs an experimental subclass preserving
baseline fusion/TCN and mask semantics. Hook in base visual_adapter only to allow
a protected encoded-sequence→token method; the default output path and keys remain
unchanged and parity/regression checked. New `train_connector_study.py` entrypoint
and policy `signer-pilot-three-arm-connectors-v1` preserve old frozen runner behavior.
Use existing seeded model/data, validated original inputs, recovery budget and
dev scoring utilities without relaxing seed/profile policies for earlier runs.

Common initialization attestation from verified CPU base model for both seeds.
Bind exact sources (all local dependency closure), lockfile, baseline checkpoint,
data/features/schema, architecture constants, seed/caps/retention and attestation
into new run keys. Preflight read-only. Tests prove no inference/optimizer branch
silently omits new modules. Joint CE+VT smoke restores parameters/buffers/RNG;
spatial/motion/connector/FLAN trainable groups finite with nonzero aggregate norms.

## Runtime and storage

Controller sequentially runs seed0 reference/plain_mlp/qformer then seed1 same
arms, with overall64800 cap and per-run10800 child watchdog deadlines. Completed
reports validated before adoption; interrupted runs resume latest owned checkpoint.
No overlapping GPU jobs. Existing snapshots/features/test reports are preserved.
Six completed runs can retain approximately160–200GiB including dev models;
check50GiB free reserve before each stage, respecting host compaction caveat.
Profile/all parameter hashes/scoring bytes validated before experiments.

## Outcome criteria

Report paired dev BLEU1-4/ROUGE-L differences versus reference within each seed,
repetition/uniqueness, params/tokens/peakGPU/runtime and complete dev trajectories.
No architecture winner based on one favorable seed; report reversals/variance.
No test-set scoring automatically follows this study. Separate full feature/new
caption improvements from downstream architecture changes.
