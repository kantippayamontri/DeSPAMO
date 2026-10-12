# Experiment-code publication verification — 2026-10-12

## Scope

User explicitly requested commit and merge into `feat/global-local-rvlf`, then
selected **All experiment code**. Publication snapshots pending experiment source,
configs, tests and relevant documentation from main and retained adaptation
worktree into a separate publication worktree based on the existing target branch.
Papers, proposal drafts, data, checkpoints, environments and ignored operational
approval files are excluded. Existing SigLIP2 global/local design is retained;
its model/loss, six-run CLIP fusion orchestration and direct-B: migration remain
unimplemented.

Original-host CLIP+S² training source/environment bytes are not edited by this
publication. Main/runtime worktree branches and their uncommitted working files
remain independent. No GPU experiment or checkpoint operation is part of this task.

## Verification

Existing target baseline:66passed/1skip for fusion, compact checkpoint and recipe.
Publication tests use existing interpreters without dependency installation:

| Environment | Test groups | Result |
| --- | --- | --- |
| Main `.venv`, CUDA hidden | `tests`, safe/inference/plain/benchmark tests, CLIP training/data/controller/guards; extraction-specific files separated |1902passed/8skip |
| DINOv3 `.venv`, CUDA hidden | `tools/dinov3/tests`, official spatial admission tests |278passed/1skip |
| Motion `.venv`, CUDA hidden | `tools/motion/tests`, CLIP encoder/inventory/package tests |141passed |

Total across disjoint final groups:2321passed/9skip. Existing dependency warnings
remain. Scoped Ruff checks and safe-runtime Pyright: passed/0errors/0warnings.

Initial mixed-environment run failed16 tests:13 missing worktree-local motion
interpreter paths,2 DINOv3 imports unsupported by main Transformers4.32,1 known
source-key fixture error. Correct interpreter split resolved environment failures.
The publication checkout has an ignored test-only motion environment view: unique
regular interpreter copy and existing motion package directory; original installed
environments are not modified or committed.

Two test-only corrections in publication snapshot:
- Summary source-drift fixture supplies actual required controller-source key,
  so it reaches and verifies changed frame inventory instead of failing with KeyError.
- Existing CPU cleanup test uses its bounded actual-lock-acquisition helper,
  avoiding immediate reacquisition race after helper-process cleanup while still
  failing if foreign lock persists. Production code is unchanged by these fixes.

Historical runners still require original-host assets/path bindings. These CPU
checks establish source publication behavior, not new global/local method results
or generic permission to start old controllers on another server.
