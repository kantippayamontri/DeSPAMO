# Storage audit before time-aligned fusion

Snapshot: 2026-10-05. Analysis only; no files deleted, no trim/compaction or
training launched. Fusion design and its proposed compute budget remain pending.

## Finding

Completed-run checkpoint pruning can reclaim **194.42 GiB** inside Linux:

- **A: 4 redundant resume snapshots, 22.433052 GiB.** Recommended first.
- **B: 31 non-selected, non-terminal dev checkpoints, 171.985077 GiB.**
  Optional archival trade-off requiring explicit authorization: lose those
  intermediate model states, retain their dev results, selected models and final
  recovery states. Previous retention policy protected all dev checkpoints;
  approving B would explicitly relax it for only the files listed below.

No selected checkpoint or terminal checkpoint is proposed for deletion. The
selected Q-Former seed1 step10400 is protected, alongside its final step16000.
These are not blanket permissions to remove any checkpoint matching a wildcard.

## Current capacity and usage

| Location | Allocated size / free space |
|---|---:|
| Linux filesystem available | 240.65 GiB free |
| Windows C: available | 83.76 GiB free |
| Project checkout, without following links | 14.89 GiB used |
| External DeSpaMo data | 479.05 GiB used |
| External DeSpaMo signer-pilot subset | 359.35 GiB used |
| External SpaMo data/model/features | 55.08 GiB used |
| Recovery worktree, primarily hard-linked environment | 4.40 GiB apparent used |

The `external-data` link points to `/home/kan/datasets/despamo`; it is not a
second physical copy. `recovery-worktree` likewise exposes an existing archive.
No project symlinks were followed when measuring checkout size.

GPU was idle (0%,2MiB); no matching local training/evaluation processes found.
The old incomplete E1 retry still says budget `running`; that stale label is
not proof of a live process or a completed run. Its recovery files are protected.

## Fusion capacity estimate

Current plain-MLP full checkpoint is 5.533219 GiB. Allow **5.6 GiB** per fusion
checkpoint for its small new modules and optimizer state; confirm with smoke.

Six runs retaining three dev checkpoints plus two resumes each, with one
atomic-write temporary checkpoint at peak:

    (6 * (3 + 2) + 1) * 5.6 = 173.6 GiB
    plus 50 GiB reserve = 223.6 GiB desired starting headroom

This excludes a large new feature cache because the experiment reuses existing
features. Small logs/timing manifests fit within reserve; unexpected larger
checkpoint sizes require recalculating admission.

Linux currently meets this conservative estimate narrowly. Windows does not
meet the conservative fresh-growth estimate. Existing unallocated VHDX blocks
may absorb some writes, but that cannot be established from Linux free space.
Deleting Linux files does not automatically increase Windows free space.

If all A+B files are removed, Linux free space should rise by approximately
194.42GiB to435.07GiB. If those blocks are subsequently reclaimed from the WSL
virtual disk, Windows could recover a similar amount; that is an estimate,
not a promise. Recheck both filesystems after offline host-side compaction.
Do not launch solely because Linux reports enough space.

An alternative future retention policy would prune redundant resumes after
each completed run while retaining all three dev models, reducing peak study
storage to about117.6GiB plus50GiB reserve. That policy is not currently approved
and does not remove the need to check Windows headroom.

## Exact candidate inventory

All run directories below are under
`/home/kan/datasets/despamo/signer-pilot/`. Each filename is relative to its
specific directory. All35 candidates have one hard link; totals use allocated
blocks, not only logical file size.

| Run directory | A: redundant resumes | B: intermediate dev models | Protected selected/final models | A+B GiB |
|---|---|---|---|---:|
| `connector-plain_mlp-seed0-16000-61f0c94264aae21fd25435bd7c954701e78c8f24c808b8f22032fa62929fe768` | — | `step-10400.ckpt`, `step-5500.ckpt` | `step-16000.ckpt` | 11.066 |
| `connector-plain_mlp-seed1-16000-45ff441985aa7cd1f5007b316fe7646c560ad41cfa4d4c756d3629c1dfa31cad` | `resume-15500.ckpt`, `resume-15000.ckpt` | `step-10400.ckpt`, `step-5500.ckpt` | `step-16000.ckpt` | 22.133 |
| `connector-qformer-seed0-16000-8cec060038e93a6b99242cfcc838abb306e89ffc702f5686158195378bb1ec2c` | — | `step-10400.ckpt`, `step-5500.ckpt` | `step-16000.ckpt` | 11.367 |
| `connector-qformer-seed1-16000-c099b17106f2a4d14724e4ecf5969826aed573e77d7bf12775242533d87e901b` | `resume-16000.ckpt`, `resume-15500.ckpt` | `step-5500.ckpt` | `step-10400.ckpt`, `step-16000.ckpt` | 17.050 |
| `connector-reference-seed0-16000-c9af7cf6b8aff72ecf8f8e88cd04ddfa6c56811b55072def47d5fbaeacfcf825` | — | `step-10400.ckpt`, `step-5500.ckpt` | `step-16000.ckpt` | 11.037 |
| `connector-reference-seed1-16000-74d3aee182b240283f3a0b1643369378f60adcc3b2c6e9a8bef4d4b6ebc71cd7` | — | `step-10400.ckpt`, `step-5500.ckpt` | `step-16000.ckpt` | 11.037 |
| `e1-v3-483c1500d1bc292990f65cd79a96a40899bc3843f92186e1f41243386c444fe8` | — | `step-3500.ckpt` | `step-5600.ckpt`, `step-8000.ckpt` | 5.519 |
| `e1-v3-79d14f8b66093483f683102f06bf28a23253d82d056ca03be0771725a170b533` | — | `step-1750.ckpt`, `step-2800.ckpt` | `step-4000.ckpt` | 11.037 |
| `e2-only-64c971377519c5290411e13cbb0a7213ed3fc0104067db4e1858f37a74fbaac0` | — | `step-1750.ckpt`, `step-2800.ckpt` | `step-4000.ckpt` | 11.130 |
| `e3-only-0402717f5ef91cca106caa881b689ce1eba75a3a87976302c99c654bea189690` | — | `step-3500.ckpt`, `step-5600.ckpt` | `step-8000.ckpt` | 11.037 |
| `e3-only-06d529839542ed33257cfc6a63d955945168cc6a05931222083efd113495f1bc` | — | `step-1750.ckpt`, `step-2800.ckpt` | `step-4000.ckpt` | 11.037 |
| `e3-only-927e19028e8e790e01452fc1097d4786075e485964c49ef0ed1d9c221525ed26` | — | `step-3500.ckpt` | `step-5600.ckpt`, `step-8000.ckpt` | 5.519 |
| `e3-only-d1d697be1ee8e847679c73eb4632b80eb27418a0f182b32bdb1387d691e60f08` | — | `step-3500.ckpt` | `step-5600.ckpt`, `step-8000.ckpt` | 5.519 |
| `projector-linear-16000-0319401a7579e5fc6ee3a37748c01a441b2e0b75cf97ce9778f6fc60e1f457da` | — | `step-10400.ckpt`, `step-5500.ckpt` | `step-16000.ckpt` | 11.037 |
| `projector-linear-16000-eb0a5bef8210afc3d894f4ac57338f8cd5e0f01b6b4d74a8a38cb285b4f5f313` | — | `step-10400.ckpt`, `step-5500.ckpt` | `step-16000.ckpt` | 11.037 |
| `projector-residual-16000-86f337958a8244e9538b3f36d7f6186859982f20fa1edd4ac71f614b8a54f353` | — | `step-10400.ckpt`, `step-5500.ckpt` | `step-16000.ckpt` | 11.143 |
| `projector-residual-16000-f8deb3d4b807d59f3e0df2e6e6253ec069fb01d1a3365c36d4c257422d87526d` | — | `step-10400.ckpt`, `step-5500.ckpt` | `step-16000.ckpt` | 11.143 |
| `projector-residual-8000-def98a1f1fd934780ba7e18fe39e221c4d0a839e669cfa072d43931729dce325` | — | `step-3500.ckpt` | `step-5600.ckpt`, `step-8000.ckpt` | 5.571 |

## Evidence and limitations

- All18 selected checkpoint files were freshly streamed through SHA-256 and
  matched their selection receipts. All18 runs have complete run-status records.
- Ten authoritative held-out report byte hashes match the shared registry;
  each claim is complete with768committed items, and its model hash appears in
  a selected-checkpoint receipt. These selected models are protected.
- The terminal checkpoint in each run exists. Before execution, also verify
  retained terminal files against dev-checkpoint receipts and confirm usable
  recovery payloads. The audit did not deserialize every terminal optimizer state.
- Exact absolute candidate paths were searched across22015 bounded text files
  in the project, recovery tree and pilot storage. No literal matches found.
  This does not rule out dynamic paths, hash references, relative paths or
  external dependencies. B files are intentionally historical dev-model states,
  not evidence-free duplicates; keep their associated reports unchanged.
- Candidate hashes/inodes and metadata fingerprints must be frozen and
  revalidated in a deletion manifest before executing an approved cleanup.
  This report is an inventory, not a ready-to-run deletion script or receipt.

## Other large files: preserve or investigate separately

| Artifact | Size | Reason not included in recommended cleanup |
|---|---:|---|
| Checkout README-referenced smoke checkpoint `artifacts/phoenix14t_smoke_retry2/lightning_logs/version_0/checkpoints/epoch=0-step=35.ckpt` | 5.47 GiB | Published local reference; removing it breaks that artifact path |
| Two incomplete E1 pilot directories | 16.45 GiB total | Their only recovery snapshots; separate abandonment decision |
| `comparison-2026-09-26` | 33.11 GiB | Six final models from an earlier comparison; not redundant resumes |
| `comparison-smokes-2026-09-26` | 11.04 GiB | Historical smoke models, not yet dependency-audited for deletion |
| `unreviewed-factor-100step` | 11.13 GiB | Phase50 and final100 models demonstrate resume behavior; receipts reference both |
| `/home/kan/datasets/spamo/features.zip` and `features/spa_feat_p14t.zip` | 19.30 GiB total | Archive redundancy/content integrity not verified; separately validate before pruning |
| `internvl3_hf_cache` | 14.81 GiB | Reusable caption-model cache; future need and redownload not assessed |
| Spatial/motion feature caches, raw frames, model initialization caches | Various | Inputs and provenance for fusion/earlier studies; expensive to regenerate |

Deleting environments is a poor first choice: checkout `.venv` displays4.40GiB
but only0.064GiB is single-linked; `tools` displays4.97GiB but only0.055GiB is
single-linked. Most package bytes remain through shared hard links. Preserve
working environments, recovery source, papers, captions and uncommitted work.

## Recommendation

For minimal archival loss, approve A only, then assess host compaction and free
space. For enough conservative headroom without relying on VHDX hole reuse,
approve A+B after accepting loss of31intermediate model states, reclaim WSL
virtual-disk space, and require measured starting capacity of roughly224GiB
on both filesystems. Selected models, final recovery states, reports, source
archives and experiment inputs remain available.
