# New 0x133 dump: audio ISR bucket, GPU hold retained

**New dump analyzed locally.** This analysis concerns only
`101126-10500-01.dmp`, not the earlier two0x116 dumps. It identifies an audio ISR
at the cumulative watchdog stop, but does not establish the initiating cause of
the preceding CUDA illegal-instruction failure or a safe GPU configuration.

## Export and analysis evidence

User's "do it" approved local dump-copy/analysis step. A Windows UAC-elevated
PowerShell process copied only this protected dump to the existing B: evidence
directory. The one-file helper checked source/copy SHA256 equality and refused
to overwrite a differing destination. No ACL, driver, registry or runtime change.

| Artifact | Verified value |
| --- | --- |
| Dump copy |B:\DeSpaMo\diagnostics\gpu-crashes-2026-10-11\101126-10500-01.dmp |
| Dump size |5865292bytes |
| Dump SHA256 |ac6a3fa6490cae662200d9af098a44cf31aba5d3a815d0d735a726ed8fd0ad8f |
| Export completed |2026-10-11T13:36:30.2323616Z |
| Debugger |Existing Microsoft.WinDbg1.2610.1001.0; bundled kd.exe engine10.0.29661.1004 |
| Public symbols |Microsoft symbol server; existing local symbols cache |
| nt image/PDB |ntkrnlmp.pdb loaded; image version10.0.26100.9457 |
| Dump crash time |2026-10-11T08:25:26.111Z; debugger displayed UTC+8 |

Independent PAGEDU64 header read confirms0x133 and arguments
`(1, 0x1e00, 0xfffff8008e3c53c8, 0)`, matching Windows WER event1001.

Both primary and detail kd.exe sessions exited0. No WRONG_SYMBOLS failure bucket
was produced. `analysis-symbols-10500.txt` SHA256:
`956c25431bd229854556eb1762149e0d8740f5c18becd5625decdf5f38e0bf38`.
`analysis-details-10500.txt` SHA256:
`45fde2d0a9a24eea0ed75f058cac4111903f60d5142bbd897ce0c5c63e28911e`.

## Primary WinDbg findings

```text
BUGCHECK_CODE: 133
BUGCHECK_P1: 1
DPC_TIMEOUT_TYPE: DPC_QUEUE_EXECUTION_TIMEOUT_EXCEEDED
PROCESS_NAME: System
SYMBOL_NAME: HDAudBus!HdaController::Isr+19a
MODULE_NAME: HDAudBus
IMAGE_NAME: HDAudBus.sys
IMAGE_VERSION: 10.0.26100.9444
FAILURE_BUCKET_ID: 0x133_ISR_HDAudBus!HdaController::Isr
```

Relevant stop/interrupt stack:

```text
nt!KeBugCheckEx
nt!KeAccumulateTicks
nt!KiUpdateRunTime
nt!KeClockInterruptNotify
nt!KiCallInterruptServiceRoutine
nt!KiInterruptDispatchNoLockNoEtw
HDAudBus!HdaController::Isr+0x19a
nt!KiCallInterruptServiceRoutine
nt!KiInterruptDispatch
tcpip!WfpInTransportShimV4Discard
tcpip!UdpInspectReceiveDiscard
tcpip!UdpReceiveDatagrams
ndis!NdisMIndicateReceiveNetBufferLists
rt640x64+0x104226
```

NVIDIA `nvlddmkm` is loaded in the dump, but **has no frame in this primary
stack**. Earlier0x116 dumps bucketed nvlddmkm's failed graphics reset; this new
dump does not reproduce that stack/bucket.

`!dpcs` lists a queued `HDAudBus!HdaController::CodecDpc` on CPU0 along with
other queued work, including AMD/WDF/system DPCs. A queued DPC alone is not proof
it caused the timeout. `!blackboxpnp` records a USB device with problem-code24;
no causal relation to this crash was established.

## Important attribution limits

Microsoft explicitly cautions that with **parameter1=1**, execution may stop
outside the offending code because excessive high-IRQL time is cumulative.
Therefore the audio ISR bucket is a strong investigation lead, not proof that
HDAudBus, an audio endpoint or Realtek NIC is the original culprit. The network
receive stack can be interrupted context rather than an independent defect.

This is a mini kernel triage dump, not a full kernel/complete-memory dump. WinDbg
warned about unavailable dump extension/XState data. Public nt symbols matched,
but the documented `DPC_WATCHDOG_GLOBAL_TRIAGE_BLOCK` type was unavailable in
these symbols. Private NVIDIA/Realtek internals were not resolved. No complete
watchdog execution profile or CUDA kernel identity is claimed.

The dump was taken at08:25:26UTC, after NVIDIA SM Illegal Instruction Encoding
events around08:21:12 and restoration of training hold at08:22:11. A lingering
device/interrupt state after the GPU fault is possible; an independent audio or
network problem is also possible. This ordering alone cannot distinguish them.
No evidence here proves VRAM exhaustion or a defect in S² feature extraction.

## Read-only device inventory

Current registered device metadata includes:

- Microsoft High Definition Audio Controller, PCI vendor10DE/device22BA,
  hdaudbus.inf/version10.0.26100.9444: NVIDIA-vendor PCI audio controller.
- NVIDIA High Definition Audio endpoint, version1.4.6.3, oem42.inf.
- AMD High Definition Audio endpoint, version10.0.1.42.
- USB Audio2.0 device, Microsoft version10.0.26100.9457.
- Realtek Gaming2.5GbE Family Controller, version10.79.50.1003, oem36.inf.

The observed NVIDIA PCI audio controller is a concrete host-isolation candidate,
but current inventory does not conclusively bind the sampled ISR object in the
dump to that device. None of these devices was disabled or reconfigured.

## Recommendation and retained hold

1. Review audio/interrupt lead before another GPU reproduction. Determine whether
   user relies on RTX4090 HDMI/DisplayPort audio. If unused, a separately approved,
   reversible isolation of the specific NVIDIA PCI audio controller could test
   that hypothesis. Do not blindly disable all sound devices, networking or GPU.
2. If a host-isolation mitigation is reviewed and approved, propose a bounded
   matched DINO/CLIP GPU comparison: same checkpoint/clip/motion/prompt, batch1,
   beam5/cacheTrue, sequential fresh processes, durable telemetry, no retries.
   Such a reduced-load test still cannot reproduce or explain original
   batch4/cacheFalse behavior by itself.
3. This dump analysis does not release the GPU hold or authorize that comparison.
   No training, CUDA/NVML diagnostic, DINO/CLIP generation or test scoring launched.
   Manual-stop remains active, GPU diagnostic permission false, no CLIP worker
   found,1774updates and497.553348937s inherited training charges retained.

Microsoft reference:
https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x133-dpc-watchdog-violation
