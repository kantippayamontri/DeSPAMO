# Validation optimization and training recovery

User requested fixing the timed-out validation and resuming. Self-review and
regressions preceded the restart; no time forecast or increased cap was used.

## Root cause and implementation

Completed generator identity is5MB. Original validation recomputed its JSON/hash
for everyclip and read each source image twice. EveryPNG also repeated full
ancestor/symlink traversal. Initial common initialization took2196.5seconds;
firstpreflight was interrupted by its300second watchdog before optimization.

New `despamo.data.motion_validation` is read-only. It validates full original
identity/evidence once, checks identical receipt/window/source/finitearray facts,
hashes actualsource/array bytes once, and constructs the same862188-entry byte
inventory and unchanged completionhash. Source reads use O_NOFOLLOW with FD
identity/metadata checks; shared ancestor checks are reused in one scan, and
file/ancestor signatures are rechecked before return. No global mutable cache.

An in-process admission token can be reused only within the same preparation
transaction after descriptor/spatial fingerprints and all metadata signatures
are rechecked. Plain JSON descriptors always require full independent scanning.
Data drift, malformed receipts, foreign files, aliases, wrongdtype/shape and
nonfinite arrays still reject. Generator sources/cached features are untouched.

Actual full7096clip validation passed; subsequent same-transaction recheck passed.
Completion remains `9d2893e10c38bc291eaf49dd28a92c25a34f263baa02f3fcab67abcae7ba4507`.
Consumer/producer descriptor equivalence and source/array/receipt/marker tampering
are regression-tested. Tests also cover token reuse and byte drift rejection.

## Recovery and verification

New training namespace preserves the old interrupted controller/profile/logs.
Hash-bound recovery independently reconstructs2864.1890981140023s prior charge;
seed0/reference inherits this work within original32400phase/10800run bounds.
Total18hour allowance is not reset; outer watchdog covers remainingglobal time.
No prior training checkpoints existed; both seeded starting positions are
reinitialized and attested under updated consumer source bindings.

Verification: RED5validationtests, RED2transaction-reusetests, RED1inheritedphase
test before implementation. Fullmain1624passed/27existingwarnings; finalaffected
301passed/4existingwarnings. Ruff and targetedPyright0errors/0warnings; whitespace
check clean. No commits or prior artifact deletion. Generator code evidence still
matches all original hashes, and all7recovery-bound old artifacts remain unchanged.

## Actual resumed work

Recovered common initialization and previously failing preflight completed.
Real reference GPU smoke passed, then Seed0reference entered optimizer training.
Last verifiedoptimizerstep1000/16000, phase=train,status=running. This includes
the declared VT-only initial warmup; translation phase follows existing recipe.
Remaining reference/fixed/learned runs continue sequentially without user launch
prompts. Training results/checkpointselection and signer generalization remain
unmeasured; no testtranslation scoring is automatically scheduled.
