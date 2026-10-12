# Driver617.42 recovery failed; new host bugcheck — 2026-10-11

**GPU hold restored. No additional GPU work or automatic retry.** The user's
single same-root training-resume attempt restored epoch2 state, failed during
original dev generation, and was followed by another confirmed Windows reboot.
Driver update alone did not make this workload stable.

## Scope and retained state

User requested "I think just try to resume training" after driver610.62→617.42
and one successful cached batch1 diagnostic. The resumed path was explicitly
announced as the unchanged original recipe: train batch8; dev batch4/beam5,
cacheFalse. Cached batch1 helpers were not integrated. No migration bridge or
checkpoint-bound source change was performed.

Original v3 root:
`/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v3`.

Native authorization/audit directory:
`recovery/driver61742-original-recipe-2026-10-11/`.

- All84 original source hashes and accepted full-data profile bindings matched.
- Original checkpoint/identity restored with weights, AdamW, sampler, global RNG
  and prompt RNG through the unmodified loader. Budget phase`durable_replay` at
  08:20:17 confirms the loader and sampler checks completed.
- Native epoch2 checkpoint remains SHA256
  `d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.
- Original charge floor360.6147923s retained; interrupted attempt ended at
  497.553348937s. No charges reset.
- Completed optimizer updates remain1774. No new training update, dev report,
  selection, official test score or Seed1 launch occurred.
- Historical manual-stop bytes were moved intact to
  `manual-stop-before-resume.json`, SHA256
  `e147b12d6d2fdfe17aec8523b2b8a965f96fbdf8a09a6930640ca887cae0fa57`.
- One-shot recovery marker/lock prevented automatic retries. Original controller
  ran under PID12748 and worker12796 in old boot21ceb469-0d6b-494a-b4fa-dd12d94f2931.

## Fault and containment timeline

1. Original epoch2 dev replay began after native restoration; no new training
   updates were attempted before completing that pending dev evaluation.
2. NVIDIA events begin08:21:12.2470573UTC. A persisted Event13 XML payload says:
   `Graphics SM Warp Exception on (GPC 0, TPC 1, SM 0): Illegal Instruction Encoding`.
3. Worker raised`CUDA error: an illegal instruction was encountered` during
   Transformers beam generation. Original cleanup also raised the same CUDA
   error while restoring CUDA RNG. These are asynchronous error surfaces, not
   identification of the initiating CUDA kernel.
4. Live Windows query at08:21:40.0025266UTC returned384 Event13,6 Event14 and
   2 Event153 records. No additional CUDA/NVML queries were made after this fault.
5. Exact owned controller received SIGTERM. Worker was already not alive under
   its exact PID/boot/start-ticks identity. Controller stopped; wrapper restored
   manual-stop and no-retry policy at08:22:11.070482UTC.
6. User reported another restart. Read-only post-reboot inspection confirms
   Windows boot08:25:48.5000000UTC and new WSL boot
   `27947766-34fb-4d7d-9e19-9c9273210cc6`.
7. Windows WER System event1001 at08:25:58.8166074UTC reports **0x133**, parameters
   `(1, 0x1e00, 0xfffff8008e3c53c8, 0)`, new dump
   `C:\WINDOWS\Minidump\101126-10500-01.dmp`. Kernel-Power41 was recorded at
   08:25:52.1115875UTC.

Post-reboot persisted NVIDIA query returned301 Event13 records; it did not return
the same Event14/153 set seen in the live query. The discrepancy is not resolved;
do not substitute one count for the other. Pre-reboot query is preserved in local
tool evidence `/home/kan/.local/share/opencode/tool-output/tool_12a0da6850012colsDpeZMN0Cl`,
SHA256`264d97079d8eb0b654650b715aaf2b05cc8b5cad13e0788a6f8343e9bf21cea9`.

## Meaning and limits

This new bugcheck is **DPC_WATCHDOG_VIOLATION (0x133)**, different from the earlier
two0x116 VIDEO_TDR_FAILURE dumps. Microsoft documents parameter1=1 as cumulative
excess time at DISPATCH_LEVEL or above. New dump analysis is required to identify
the component responsible; do not call this another symbolicated nvlddmkm0x116
stack or assert a specific driver/hardware/WSL fault from event proximity alone.

NVIDIA's SM exception and CUDA illegal-instruction error strengthen evidence of
a device execution fault in the generation interval. Neither proves VRAM OOM,
tensor corruption, a faulty GPU, or a particular kernel/compiler defect. The
original failing dev path has no new per-batch peak-memory measurement. The
earlier5.99GiB cached batch1 success does not cover this batch4/cacheFalse path
with optimizer state resident.

## Current state and next evidence

- Manual-stop=`stopped_by_user`; automatic recovery disabled; diagnostic GPU
  permission false. Historical recovery authorization is consumed and bound to
  the old boot; it must not authorize a new attempt.
- No CLIP training/controller processes found after reboot. Native controller
  status=`stopped_by_user`, budget=`interrupted/finalized`. Seed0 stage's stale
  `running` label has an old-boot owner and is not a live worker.
- Source/checkpoint bytes remain intact; no new root, runtime/driver/TDR changes,
  automatic retries or further GPU diagnostics performed.
- **Analysis update:** the new dump has now been exported/hash-verified and
  analyzed locally. Bucket=`0x133_ISR_HDAudBus!HdaController::Isr`, with network
  receive context; not the earlier nvlddmkm0x116 reset stack. Cumulative watchdog
  sampling does not prove audio caused the preceding GPU fault. See
  `windbg-dpc-crash-analysis-2026-10-11.md`/JSON. GPU hold stays active; review the
  host audio/interrupt lead before any later GPU comparison or training recovery.

Administrator PowerShell copy, performed by the user if needed:

```powershell
Copy-Item 'C:\Windows\Minidump\101126-10500-01.dmp' 'B:\DeSpaMo\diagnostics\gpu-crashes-2026-10-11\101126-10500-01.dmp'
```

Microsoft reference:
https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x133-dpc-watchdog-violation
