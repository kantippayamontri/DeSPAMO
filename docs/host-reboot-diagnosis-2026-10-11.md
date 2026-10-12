# Host restart investigation — 2026-10-11

## Summary

Windows confirms actual graphics-related blue screens: bugcheck
**0x116 VIDEO_TDR_FAILURE**, driver **nvlddmkm.sys** named in event1019. Windows
failed to reset the graphics driver after GPU timeout. `AutoReboot=1` explains
why the user sees a restart rather than a persistent blue-screen display.
The dev-generation workload is a strong suspected trigger. The precise kernel,
driver defect or hardware contribution is not established from Python traces.

All training, diagnostic and watchdog processes were absent after the latest
restart. Published `...clip-s2-spamo200-v3/manual-stop.json` to block automatic
retries. Do not launch another GPU reproduction, controller, watchdog or new root
while this hold applies. Features, model bytes and epoch1/2resumes are retained.
This investigation used host/kernel logs, file reads and CPU-only diagnostics;
no GPU compute, driver install, registry adjustment, WSL shutdown or package upgrade.

Evidence receipt: [host-reboot-diagnosis-2026-10-11.json](host-reboot-diagnosis-2026-10-11.json).

## Timeline (UTC)

| Time | Evidence |
| --- | --- |
| 03:42:22 | v2 durable budget entered epoch2 dev generation. |
| 03:45:11 | NVIDIA driver event14, followed by repeated event153. |
| 03:47:33 | Windows event1001 reports bugcheck0x116; event1019 names nvlddmkm.sys. |
| 04:07:18 | v3 epoch2 resume published, before dev generation. |
| 04:08:49 | NVIDIA driver event14, followed by repeated event153. |
| 04:08:57 | v3 worker exited from CUDA unknown error during dev; epoch2checkpoint retained. |
| 04:17:15 | NVIDIA driver event14 during the previous full-dev diagnostic. |
| 04:18:53 | Windows reports another bugcheck0x116/nvlddmkm.sys. |

Bugcheck events are reported after reboot; their timestamps are not exact fault
onset. Driver events and recorded stage times support association with generation.
Both bugchecks have parameter3 `0xC000009A` (`STATUS_INSUFFICIENT_RESOURCES`).
This is a failed-driver-operation status, not proof that PyTorch exhausted VRAM.

Protected dumps:
- `C:\WINDOWS\Minidump\101126-13859-01.dmp`
- `C:\WINDOWS\Minidump\101126-9468-01.dmp`

Read access to those dumps and WER report files is denied from this session;
no Windows debugger was found in PATH. Exact crash stack/failure bucket requires
local Admin WinDbg analysis. Do not attribute proprietary event14hex payloads to
a specific CUDA kernel without a decoder or dump evidence.

At06:02UTC access was rechecked: current Windows token not elevated, both dump
reads denied, and WinDbg/SDK debugger absent at checked locations. Prepared
[Admin export/handoff](diagnostics/gpu-crash-admin-handoff-2026-10-11.md) and
`docs/diagnostics/export-gpu-crash-evidence.ps1`. Windows PowerShell parser and
CheckOnly passed; non-elevated execution fails before any file creation. It has
not exported dumps or analyzed them yet. User must run it from Admin PowerShell
and obtain local WinDbg analysis. GPU hold remains active.

**06:12UTC update:** user completed export. Both B: copies are readable; byte
counts and SHA256match manifest. Direct PAGEDU64header decoding independently
confirms bugcheck0x116and all four parameters from Windows events. Export includes
26System events. Original protected paths remain restricted, but dump-read blocker
is resolved by verified copies. WinDbg/cdbstill absent; WinGet available. No
symbolicated stack/failure-bucket analysis or GPU work performed yet.

## Completed local WinDbg analysis

User approved official debugger installation after verified export. BundledKD
with correct Microsoft public symbols analyzed both dumps locally; matching
MODULE_NAME=nvlddmkm, IMAGE_NAME=nvlddmkm.sys,
FAILURE_BUCKET_ID=0x116_IMAGE_nvlddmkm.sys, PROCESS_NAME=System and driver owner
offset nvlddmkm+1958210. Both stacks show dxgkrnladapter reset followed by
TdrBugcheckOnTimeout and ntKeBugCheckEx. Source records and hashes in
[WinDbg report](windbg-gpu-crash-analysis-2026-10-11.md)/JSON.

This resolves dump-access/debugger blocker, confirms reset-failure mechanism,
and leaves exact initiating CUDA kernel/driver defect/hardware contribution
unresolved. No training/GPU validation or driver/runtime/TDR change performed.
GPU hold stays in effect pending actual host-mitigation review.

## Code review: confirmed inefficiency, not proven BSOD cause

`tools/followup/model.py:19-23` enables transformer activation checkpointing and
sets training model config `use_cache=False`. This is appropriate for training.
`src/despamo/models/flan_t5.py:153-162` does not override that flag for generation.
Installed transformers4.32T5 only truncates decoder input to its last token when
`past_key_values` exists (`modeling_t5.py:1814-1815`). Cache disabled means the full
decoder prefix is repeatedly processed. Validation uses four clips and five beams,
so decoder operates on20sequences, with output ceiling64.

CPU-only reproduction using installed T5 implementation, batch4/beam5/max12:

| Setting | Decoder prefix lengths | Total decoder-token positions |
| --- | --- | ---: |
| Current inherited cache disable | 1,2,...,11 | 1320 |
| Explicit inference cache enable | 1,1,...,1 | 220 |

Generated IDs were identical in this small model; training config remained false.
This demonstrates unnecessary recomputation. It does not measure full FLAN GPU
kernel duration, prove timeout threshold crossing or validate a host-crash fix.
Caching also increases KV storage: do not enable it on full-size GPU workload
blindly, especially given the driver resource-failure status.

Reproduction (CPU-only; no public weights downloaded):

```bash
/home/kan/Research/DeSpaMo/.venv/bin/python -B docs/diagnostics/2026-10-11-t5-cache-cpu-probe.py
```

The Python exception occurred at beam scorer `next_token.item()`, a synchronization
point where an earlier asynchronous CUDA fault can surface. Subsequent CUDA RNG
restoration also failed because the CUDA context was already unhealthy. Neither
stack location proves the faulting kernel or an EOS-token code bug. No third-party
`beam_search.py` edit is justified from that traceback.

Previous diagnostic's missing `LD_LIBRARY_PATH` produced a separate libcuda/cuDNN
loader failure. The actual controller already supplies `/usr/lib/wsl/lib`; that
diagnostic mistake does not explain the logged Windows VIDEO_TDR_FAILURE.

## Runtime and remaining uncertainty

Affected stack: Windows10.0.26200.9457, WSL2.7.10.0/kernel6.18.33.2, NVIDIA610.62,
RTX4090, torch2.0.1+cu117 (CUDA11.7), transformers4.32/PEFT0.7.1. Versions alone do
not prove incompatibility. Current WSL kernel also logs a field-spanning memcpy
warning in dxgvmbus.c:3095 during GPU queue wait; its connection to BSOD is unknown.
No GPU-temperature/power/VRAM time series was collected at the failure, so do not
rule out hardware instability or declare out-of-memory from a post-reboot snapshot.

Earlier diagnosis asserted in-process corruption as the established cause. Saved
checkpoints working in fresh processes establishes that live behavior differed;
it does not prove a particular frozen tensor changed. Windows crash events now
provide stronger evidence for GPU driver/context failures. Fresh roots and model
integrity guards cannot prevent a Windows kernel-level TDR failure. Stop further
relaunches until mitigation is reviewed.

## Why the successful DINO run matters

Read-only differential check against completed DINO200reference seed0 confirms
equal recipe, decoder/exemplar settings, torch/transformers/PEFT versions, frozen
FLAN fingerprint/base files, common initialization and ordered split IDs. Shared
downstream source hashes match across57files. All8257spatial arrays have equal
sequence length,2048width and FP32dtype between DINO and CLIP. Both translation
runs consume cached spatial features; neither spatial encoder is resident in the
translation worker. Thus "CLIP encoder is heavier" does not explain these failures.

DINO completed96epochs using the same cache-disabled generation code. Cache
disable is a shared inefficiency, not a demonstrated differentiating cause.
Feature values/learned adapters can alter generated tokens and EOS behavior;
custom runner/guard operations and GPU allocation lifetime also differ. Windows
driver/device state at the old DINO run was not fully captured. These are candidate
differences, not established explanations for BSOD.

Early DINO outputs were also long (epoch2mean144.8characters versus CLIPv1mean94.2).
Decoded-string length is not raw decoder-token/step count, and CLIPv1outputs after
failure are unreliable as workload evidence. Do not assert "CLIP generated longer
than DINO" or infer a specific kernel runtime from those strings. Exact differential
cause remains unresolved; current GPU hold stays in effect.

## Original SpaMo source comparison

Inspected `/home/kan/Research/SpaMo` without launching its model/GPU runner.
Relevant tracked source/config files have no working-tree modifications. Local
history contains reproduction changes, so checked initial492a24e,6d7aaf7 and HEAD.

- `spamo/t5_slt.py:156-160` loads BF16T5; no explicit cache disable or gradient
  checkpointing enable appears in initial or current model code. The default
  FLAN config enables cache. CPU tiny-T5/PEFT test confirms LoRA wrapper retains
  use_cacheTrue and does not itself enable checkpointing.
- `spamo/t5_slt.py:535-542` generates with beam5/max64, **do_sampleTrue/top_p0.9**.
  Our generation is deterministic beam search with inherited cacheFalse.
- `configs/finetune.yaml:12` limits spatial frames512; `get_inputs:363-367`
  randomly crops beyond that limit, including validation. Motion sequence remains
  full length. Our official translation preserves all spatial/motion frames.
- `dataset/datamodule.py:29-55` uses one batch setting for training/validation/test.
  **Initial checked-in config batch8**, accumulation2; **local reproduction
  change reduced batch to2**. The earlier claim that2was the released initial
  default was incorrect. Paper does not establish exact runtime launch overrides.
- `max_txt_len64` governs generated output. Tokenizer constructor `max_length64`
  alone does not set prompt cap: CPU check with installed4.32API/cached FLAN
  tokenizer shows model_max_length512 and512-token actual truncation. Our explicit
  prompt cap1024preserves measured569-token longest complete ICL prompt.
- Evaluation is standard Lightning validation_step; no special Windows TDR,
  driver-reset recovery or CUDA host-reboot workaround found in source.

Original cached generation and spatial cap are genuine workload differences;
neither establishes why our host resets. Initial original evalbatch8means40beam
sequences versus ours20, so original source defaults cannot simply be described
as a smaller generation workload. Current local batch2would use10beam sequences.
BF16precision alone is not a new difference; relevant pinned Python versions also
match our stack. Original paper A100results do not validate our Windows/WSL/4090
driver stack.

Do not blindly copy random validation cropping, stochastic decoding or current
batch-derived ICL: original `get_inputs` builds examples from current batch text
and deranges them, which can expose held-out translations during validation.
Current local YAML also uses mode:test for validation. Our train-only ICL and
official-dev selection protocol should remain explicit. CPU-only inference-cache
tests and separately reviewed microbatch controls are the safer comparison scope;
GPU hold remains unchanged.

## Next evidence and bounded mitigation

1. Open both dumps in local Admin WinDbg, run `!analyze -v`, capture MODULE_NAME,
   IMAGE_NAME, FAILURE_BUCKET_ID and stack. No dump upload performed or required.
2. Review NVIDIA/WSL support and driver-failure details. Driver/runtime changes are
   separate host actions; retain existing baseline environment and source identities.
3. Prepare a separate, checkpoint-bound inference path with cache explicitly
   controlled and evaluation microbatch1, preserving beam5/output64. CPU output
   tests first; any later GPU check must be supervised and bounded after hold review.
4. Record batch index/IDs, joint lengths, latency, memory and device health before
   prolonged evaluation. A one-batch smoke or training-memory profile alone does
   not validate long full-dev generation stability.

Do not automatically classify repeated GPU unknown errors/host resets as safe
retryable failures. Do not disable or raise Windows TDR limits as an unverified
workaround. No decoder, precision, LR, features or test scoring changes applied.

## Authoritative references

- [Microsoft: VIDEO_TDR_FAILURE](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x116---video-tdr-failure): failed driver reset after timeout; parameter2responsible driver, parameter3last-operation status.
- [Microsoft: WDDM timeout detection](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery): default2second GPU completion/preemption timeout, not a2second limit on whole Python batches.
- [Microsoft NTSTATUS reference](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/596a1078-e883-4972-9bbc-49e60bebca55): C000009A insufficient resources.
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html): WSL GPU/debugging and resource constraints.
