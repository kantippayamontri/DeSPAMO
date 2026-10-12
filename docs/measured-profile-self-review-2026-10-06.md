# Measured smoke admission: implementation and self-review

User requested direct self-review/implementation without estimating time.

Changes: extraction smoke profile now schema3 with actual parity/samples/source
observation hashes. Full-run admission checks measured evidence and actual remaining
hard deadline; duration extrapolation and its veto were removed. Existing failure,
cap, ownership, data and automatic-training guards remain.

Versioned recovery leaves old runtime metadata and clip receipts intact. A fixed
hash-bound recovery record permits only the stopped forecast rejection with zero
training spend. It independently reconstructs max prior producer/outer charges,
checks old artifact hashes and quiescent process receipts, and seeds new ledgers
at216.61064642599376seconds. Root and record paths are fixed; duplicate launches
or changed source/credit require owned resume rather than a fresh allowance.

Self-review caught and fixed propagation of the recovery binding into the
preparation child and a failed coordinator's retained accounting anchor. Recovery
does not erase that anchor or old records. Default non-recovery mode retains its
ownership contract. Verified inherited counter floors on restart; source data,
caption policies, precision and model selection are unchanged.

Verification: initial5smoke/clock tests failed before implementation; recovery
test failed on missing helper. Finalmain1616passed/27existingwarnings216.56seconds;
extract132passed/1upstreamwarning30.67seconds. Ruffpass, both Pyright0errors/
0warnings, gitdiffcheckpass. No commits or external review agents for this change.

Actualrestart passed CPU preflight, real7clip smoke/parity, caller clock admission,
GPU/disk checks and entered full extraction. Last verified123real clip receipts,
new profile has no projected_remaining_seconds, inherited charge matches exactly,
and all11prior artifact hashes remain unchanged. Training is queued, not complete.
See [live status](motion-fusion-pipeline-status-2026-10-05.md) for runtime paths.
