# Three-arm Connector Study Implementation Plan

> **For agentic workers:** User approved two-seed execution with18GPU-stage-hour cap. Use executing-plans/TDD; independent implementation units may delegate without shared file writes. No commits/pushes.

**Goal:** Compare reference/plain spatial MLP/small Q-Former at16k across seeds0/1.
**Architecture:** Opt-in connector adapter and new versioned runner/controller;
preserve common initial weights, inputs, VT-Align, LR schedule and decoder.
**Spec:** `docs/superpowers/specs/2026-10-04-three-arm-connector-study.md`.
**Tech stack:** Existing torch2.0.1/Lightning1.9.5/transformers4.32, no new dependencies.

## Tasks

- [x] Implement models and default-preserving adapter hook with CPU/GPU-mask/gradient
  and strict round-trip tests. Spatial-only plain MLP; Q-Former replaces output only.
- [x] Implement common initialization attestations, architecture-sensitive identities,
  new runner, state-neutral smoke, correct optimizer inclusion, durable resume/dev
  selection with two-snapshot retention. Refusal paths never modify other runs.
- [x] Implement overall64800 budgeted sequential six-run controller with10800caps,
  safe shutdown/recovery and completion validators; no automatic held-out scoring.
- [x] Full verification, independent review, real-data preflights and gradient smokes.
- [x] Launch under persistent tmux; verify first live progress and completion/budget
  tracking. Stop/report if declared cap or immutable-input integrity prevents progress.
- [x] Produce dev-only paired two-seed comparison, resource/variance and limitations.

## Review focus

- Baseline adapter keys/output and RNG are unchanged despite opt-in replacements.
- Empty/padded sequences cannot contaminate learned queries or return invalid masks.
- Frozen-base initialization attestations bind all arms to identical common weights.
- New projector/connector receive optimizer gradients after VT and joint losses.
- Source hashes, cap, horizon, architecture, initialization and retention bind resume.
- Default old policies remain frozen; this study receives new run roots and reports.
- Corpus metrics on complete ordereddev selection only; no Signer07 or training/test mix.

## Execution ledger

Approved count6, seeds0/1, budget64800; arm details in spec. Root workspace has
unrelated user work and uncommitted previous implementations; preserve throughout.
Record RED/GREEN evidence, review fixes, preflight/smoke identities and run keys in
`docs/superpowers/plans/2026-10-04-three-arm-connector-progress.md`.

## Completion after storage recovery

All six runs and bound full-dev selections verified. Charged controller budget
12.1521hours, below18; per-seed charges5.3742/6.7779hours below9 each.
Reference mean4.556524; plainMLP6.144099; QFormer5.798483 dev BLEU-4.
Both alternatives beat the fresh reference in both seeds, but reference rerun
variability and n=2 limit broader claims. No Signer07 inference performed.
Full [results and limitations](../../three-arm-connector-results-2026-10-05.md).
