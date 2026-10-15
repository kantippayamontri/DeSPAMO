# Original SpaMo paper-recipe reproduction

## Approval and scientific objective

User selected the paper recipe, prioritized it over the existing CLIP+S² run,
then approved the proposed protocol and requested training. Execute natively;
preserve workspaces and unrelated work; no commits/pushes or dependency changes.

Goal: train original SpaMo from pretrained components, select on official dev,
and ultimately assess the reported PHOENIX14T test BLEU4=24.32. This is a
paper-recipe reproduction with explicit code-derived assumptions, not a claim
that the released checkpoint's exact training lineage is known.

Existing released-checkpoint clean-dev reference is23.338556169739107 BLEU4.
Reuse that completed reference; do not repeat its consumed GPU evaluation.

## Existing experiment preservation

Current CLIP+S² was operator-held at69186updates/epoch78 train completion.
Parent/children exited, scope consumed, original ledger CPU-reconciled/sealed;
87335.31002131935charged seconds and1historical probe update retained.
Epoch78 dev unscored. Preserve original checkpoints, source hashes, budget,
reports/selection, historical error/hold receipts and supplemental launcher.
Never use these weights to initialize the reproduction or reset that budget.

## Reference provenance and model

- Author repository: `https://github.com/eddie-euijun-hwang/SpaMo`, inspected
  commit `aee7f8b6ea8201b710690adcce6f46260dd11bba`.
- Paper: `https://aclanthology.org/2025.naacl-long.197.pdf`, §4.1/AppendixA.3.
- Pin source used for execution, documenting any port/adapter and parity checks.
- Original linear spatial2048→768, linear motion1024→768, sequence concatenation,
  TCN K5/P2/K5/P2, connector768→2048→2048 with GELU.
- Pretrained FLAN-T5-XL revision7d6315df2c2fb742f0f5b556879d730926ca9001,
  frozen base with Q/V LoRA rank16/alpha32/dropout0.1; fresh trainable initialization.
- No nonlinear spatial MLP, new fusion, DINO substitution or extra objective.

## Data and objective

Original released arrays:
`/home/kan/datasets/spamo/features/vit_feat_Phoenix14T` and
`/home/kan/datasets/spamo/features/mae_feat_Phoenix14T`, existing manifests under
`/home/kan/datasets/spamo/features/manifests/` and original multilingual annotations
`/home/kan/Research/SpaMo/preprocess/Phoenix14T/`.

Verify7096train/519dev/642test IDs/order/disjointness and train/dev array hashes,
shape/dtype/finiteness, spatial2048/motion1024, maxframes512. Test annotations may
be identity-bound; no test generation or checkpoint selection during training.
Do not silently substitute current re-extracted arrays.

VT uses original padded-sequence mean and original contrastive formulation;
first4000actual optimizer updates train SA with VT-only, then CE+VT(weight1).
Preserve original train minibatch reference-disjoint multilingual ICL behavior.
Inference exemplars come only from training, following paperA.2. Reject evaluation
reference exposure even if an upstream/local evaluation helper supplies it.

## Approved training recipe

- Seed0 (upstream launcher default; not a proven paper run seed).
- Physical batch8, accumulation2 (current author YAML; paper physical batch unstated).
  Per-microbatch VT negatives remain8; accumulation does not create16 negatives.
-40complete epochs, dev every2epochs, no early/time-based stop. Retain best dev.
- AdamW betas0.9/0.98, epsilon1e-8, decay0.01, gradient norm clip1.0.
- One peak LR1e-4 for trainable parameters,10000optimizer-update linear warmup,
  cosine decay ending5e-5; do not use old differential LR or10%-warmup/zero-floor.
- Counters distinguish minibatches, optimizer updates, epochs and dev clips;
  incomplete accumulation flush follows declared framework behavior.
- BF16, existing locked main runtime; activation checkpointing allowed as recorded
  execution policy only after RNG/gradient verification. Training cacheFalse.
- Pin tokenizer/prompt limits and decoder options in resolved execution config;
  retain original semantics and document paper-unspecified choices before claim.

## Execution, accounting and storage

Isolated plan-scoped checkout, new experiment identity/root independent of current
study; fresh optimizer/sampler/RNG. Main `.venv`, offline HF cache, OMP2, fixed
existing device. No environment/driver/TDR/device changes.

New native control root under `/home/kan/datasets/despamo/`; immutable full
resumes/candidates/selections on ordinary `/mnt/b/DeSpaMo/checkpoints-v1/`.
Use verified storage primitives; no weakening native filesystem whitelist or
individual checkpoint symlinks. Source/data/config/base pins and actual bytes
bound;50GiB reserve on each backing volume plus distinct projected demand.
No silent sole-copy pruning or resetting costs after a failure/replay.

CPU parent owns claim/ledger, sequential training to dev boundary, actual child
exit, then fresh optimizer-free cached batch1 dev. Cached beam5/max64/deterministic
clean-dev reference protocol; prompt train-only. All raw logits finite, decoder
cached length1/five beams; full coverage/metrics independently recomputed.
Generation policy is an explicit paper-unspecified choice matching the completed
clean-dev reference, not an attestation of the author's stochastic code path.

No duplicate GPU worker, automatic retry or CUDA/RNG/save cleanup after device
error. Preserve primary failure and all charges. Cheap CPU verification first,
then separately scoped bounded train/checkpoint/exit/dev admission before40epochs.

## Selection, test and completion

Save all20dev candidate weights and best selection, plus complete full resumes.
Dev519selects best actual BLEU4; epochs/references/features cannot be tuned on test.
After40epochs, freeze recipe/selection, then once-only official642test evaluation
under separately recorded test authorization. Report exact scores/gap versus24.32
and remaining assumptions; one seed alone does not establish universal failure or
causal improvement. A successful launch is not completed reproduction.

## Required evidence

CPU optimizer-update schedule/accumulation/VT boundary tests; original architecture/
loss/prompt parity; full-state resume and primary-error tests; B capacity/publication
checks; source/host/owner admission; bounded real training/state publication and
cached dev; actual owned long-run progress. No unverified training claim.
