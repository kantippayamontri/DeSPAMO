# SpaMo-style translation follow-up

User approved a separately versioned translation follow-up. Latest instruction
supersedes the previous requirement to finish all six40epoch runs and score their
test first: stop that study and start this200epoch study now. The old study stopped
with all threeSeed0 runs complete, Seed1reference interrupted at2123updates and
epoch2resume retained. Its watchdog/controller/worker exited; official test remains
unscored. Stop receipt: old study `manual-stop.json`. All prior bytes remain intact.

Original approved scope: investigate
a separately versioned translation recipe matching SpaMo's prompting, VT-only
warmup, LR decay endpoint and VT pooling. Preserve the current recipe, all
checkpoints/dev reports, frozen selections and official-test claims as reference.
Run the three existing arms across seeds0/1 from fresh initialization in the new
study. Retain DINO whole-image features and motion packages; CLIP/S² experiments
are separate from this launch.

## Requested differences

| Setting | Current benchmark | SpaMo-style follow-up |
| --- | --- | --- |
| Prompting | German instruction only; no ICL | Instruction plus multilingual source=German training exemplars; released config uses three. |
| VT-only warmup | 2,000 optimizer updates | 4,000 optimizer updates, as PHOENIX14T paper reports. |
| LR decay endpoint | Cosine to zero | Approved adapter/VT5e-5 and FLAN-LoRA1e-5 minima; group-specific SpaMo-inspired variant. |
| VT pooling | Masked mean | Ordinary padded-sequence mean, as released implementation computes. |

ICL exemplars must come exclusively from the official training split, must be
reference-disjoint from the current example, and must have explicit IDs/provenance.
No current dev/test translations or their multilingual variants enter generation
prompts. The paper describes train-set exemplars at inference; the local released
batch-derangement implementation is not automatically that data-provenance policy.

The request names these four changes. Other settings (model/feature sources,
plainMLP fusion arms, physical batch, learning-rate peaks, generation,
selection) are not implicitly changed. Whether to run a bundled four-change
comparison or individual ablations, and how many follow-up runs, will be specified
before execution. A bundled comparison cannot attribute gain to one changed knob.

## LR endpoint comparison and approved decision

SpaMo AppendixA.3 reports peak1e-4 and minimum5e-5; local released scheduler
uses10%warmup and cosine to zero. Matching published endpoint therefore means
changing our scheduler; matching released scheduler would leave endpoint unchanged.

Our separate group peaks are adapter/VT3e-4 and FLAN-LoRA5e-5:

- Absolute paper minimum5e-5: adapter decays to5e-5; FLAN reaches5e-5 after warmup
  and remains constant thereafter because its peak equals the floor.
- Relative50%-of-peak floor: adapter ends1.5e-4 and FLAN ends2.5e-5. This matches
  the paper's ratio but not its absolute stated minimum.

User approved the group-specific decision below instead of either common-floor
interpretation. It does not reproduce all SpaMo hyperparameters while leaving our
group peaks unchanged. The future epoch-cap amendment and approved early-stop
policy are recorded below; current running benchmark is unaffected.

### Approved group-specific floor for the200epoch follow-up

User approved: retain current peaks, use adapter/VT
minimum **5e-5** and FLAN-LoRA minimum **1e-5**. Both group rates decay after
warmup and remain nonzero near the declared horizon. This is a conservative
starting recipe, not an empirically optimized or exact SpaMo schedule. The adapter
floor equals the paper's stated minimum; FLAN has a lower floor because its peak
is already5e-5. A common absolute5e-5 would make FLAN constant after warmup;
relative50% would leave adapter at1.5e-4 throughout late training. These are
different experiments, not interchangeable "same SpaMo" settings.

Use explicit group interpolation after warmup:
`lr(k) = floor + 0.5*(peak-floor)*(1+cos(pi*(k-10000)/(T-10000)))`,
with T=177400at batch8/max200epochs. Warmup remains linear from zero to each peak
over10000updates. Early stopping may finish before either rate reaches its floor;
do not compress/reset the schedule based on observed dev scores.

## Evidence and handoff

### Approved future epoch cap and early stopping

User subsequently requested a **maximum200epochs** for this follow-up. This
supersedes the previous future epoch horizon only; current six-run40epoch/no-stop
benchmark remains unchanged. User approved the follow-up recommendation after
reviewing continued fixed-gate gains near40epochs. The following early-stop values
are approved for that separate study, not yet implemented in running code:

- Full official-dev corpus BLEU4, maximize; evaluate every2epochs.
- Minimum40epochs before stopping is permitted.
- Patience10eligible dev checks =20epochs without significant improvement.
- Significant improvement threshold0.10absolute BLEU4 points (not percent).
- Ignore VT-only evaluations for the patience counter; begin at first dev
  evaluation after translationCE updates have started. Track/reset thereafter,
  but allow stop only at/afterepoch40. Patience can accumulate beforeepoch40.
- Separate best actual checkpoint from significant-improvement anchor. Always
  preserve highest actual dev BLEU4, including gains below0.10; only thresholded
  gains reset patience. Ties select earlier epoch. Restore actual best for testing.
- Stop at200 even if improving; stop earlier only when minimum/patience conditions
  both hold. No test metrics, wall-clock duration or training-loss-only stop.

Example:last significant improvement at60, none at62..80, stop at80 and restore
the highest actual dev checkpoint, which need not be60. Smaller gains accumulate
relative to the last significant anchor; compare before increment/stop so a
qualifying gain on the tenth check prevents stopping.

User subsequently set future LR warmup to **10000optimizer updates**. At batch8,
this is approximately11.274epochs,5.637%of the177400update ceiling. It is distinct
from the4000update VT-only phase (approximately4.510epochs). During early joint
translation the group LRs are still warming; this follows the explicit schedule.
This supersedes any automatic10%/20epoch warmup in the future recipe.
Approved cosine horizon is the200epoch ceiling, with the group minima specified
above; do not reset it at an early-stop decision. Identical
early-stop rules across models/seeds can still produce different training exposure;
report actual stopped epochs/updates and compute alongside translation metrics.

Sources: original SpaMo §§3.6/4.1, AppendixA.2/A.3; local
`/home/kan/Research/SpaMo/configs/finetune.yaml`, `spamo/t5_slt.py` pooling and
scheduler; current retained-worktree `tools/benchmark/translation_model.py`.

This is a future experiment decision, not a patch to running checkpoint-bound
code. Future implementation needs a separate identity, train-exemplar leakage
tests, scheduler/group boundary tests, pooling padding behavior tests and matched
replay checks. Keep prior artifacts and claims immutable; select by dev only.
