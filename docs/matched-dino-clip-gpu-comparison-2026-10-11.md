# Matched GPU feature comparison with temporary audio isolation

**Both authorized generation cases completed. Audio restored; GPU/training hold
retained.** This was a two-case diagnostic, not a training restart or a stability
certification. No additional generation/retry was performed.

## Approved scope and host isolation

User confirmed motherboard-jack audio and approved temporary isolation of the
NVIDIA PCI audio controller plus two matched GPU generations. Exact device:
`PCI\VEN_10DE&DEV_22BA&SUBSYS_51021462&REV_A1\4&2106B648&0&0109`, class System.
Windows UAC-elevated PowerShell disabled only that controller. Problem code
changed0→22 at14:04:11.8651616UTC, and both GPU invocations checked code22 before
model construction/transfer. GPU display adapters and motherboard Realtek
USB2.0 audio remained StatusOK. No NIC, motherboard audio or display device disabled.

After both cases, diagnostic permission was revoked and the same controller was
re-enabled. Verified ProblemCode0/StatusOK at14:10:41.7367095UTC. Final read-only
inspection at14:11:40.8398619UTC again confirms code0 and other audio/display OK.

Host remained driver617.42, Windows boot08:25:48.5000000UTC, WSL boot
`27947766-34fb-4d7d-9e19-9c9273210cc6`. Event checks after isolation, after first
case, and after second case/restoration found no new matching NVIDIA13/14/153,
critical/error Display/WER crash events or Kernel-Power41 records. This is a
bounded event scope/time window, not proof of absence of every hardware fault.

## What was matched

Both fresh Python processes loaded **the same CLIP-reference seed0 epoch2
checkpoint,1774updates**. Checkpoint SHA256:
`d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.

Only cached spatial feature bytes changed:

| Invariant | Value |
| --- | --- |
| Official dev clip |11August_2010_Wednesday_tagesschau-2 |
| Spatial input |[1,42,2048],FP32, matching masks/source frames |
| Motion input |[1,4,1024],identical tensors/masks |
| Joint visual/prompt input |[1,154,2048],154 valid tokens |
| Inference exemplar |12February_2010_Friday_tagesschau-122,train-only |
| Prompt SHA256 |fe6b531391e50f2330652a5252769ebbae8aebc3bf9071fe6fdc7575888c0201 |
| Decoder |Batch1,beam5,max64,deterministic,cacheTrue |
| Precision |Same BF16 base and CUDA BF16 autocast |
| Optimizer |Absent in both inference-only processes |
| Training config |use_cacheFalse retained |

CPU preflight verified all84 original source hashes, pinned base/runtime/config,
checkpoint receipt, selected array hashes/finiteness, identical corpus IDs and
source-frame hash, identical motion/text/masks and precommitted train-only prompt.
Existing17 inference-helper CPU tests passed in2.21s before the GPU tests.

DINO features passed through the **CLIP-trained adapter** intentionally to hold
weights fixed. This is an out-of-distribution feature swap, not comparison of
each encoder's own trained translation model or a fair translation benchmark.
Neither visual encoder nor S² image extraction ran in these GPU processes.

## Results

| Spatial feature set | Dev batch seconds | Peak allocated | Peak reserved | Output token IDs | CUDA error |
| --- | ---: | ---: | ---: | ---: | --- |
| DINO |4.508655 |5.993669GiB |6.152344GiB |52 |None |
| CLIP+S² |3.718771 |5.993669GiB |6.152344GiB |48 |None |

Allocated peak6435653120bytes and reserved peak6606028800bytes were identical.
These are PyTorch allocator measurements, not total Windows GPU-memory peaks.
Token lengths include decoder-start/special tokens. No BLEU/test metrics computed.
Timing includes adapter/prompt/generation work and can include process-order/JIT
warmup effects; do not infer an encoder speed advantage from these two timings.

Generated token IDs and decoded texts differ. DINO variant begins Spanish
`la nevada lo deja gradualmente después del domingo...`; CLIP variant begins
French`les chutes de neige le quittent progressivement après dimanche...`.
Full IDs/text are retained in native case reports. Checkpoint is still inside
4000-update VT-only warmup; outputs are not a trained-translation quality claim.

## Interpretation

Changing feature values **did change decoder output**, but did not produce a
CUDA fault or divergent peak allocator memory on this selected short clip under
batch1/cacheTrue/audio-isolated inference. CLIP+S² is therefore not universally
failing even on this GPU/driver; this does not rule out feature-dependent faults
on different clips, model states or decoding shapes.

This does not reproduce original batch4/cacheFalse in-training generation, which
had optimizer state resident and crashed on617.42. It also does **not** establish
audio isolation as a repair: an earlier cached batch1 CLIP probe already succeeded
with audio enabled. No matched audio-on/audio-off contrast or original workload
reproduction was authorized. DINO versus CLIP+S² packages differ in encoder and
spatial extraction recipe, so this is not an S²-only ablation.

## Artifacts and final state

Same retained study root, diagnostic namespace:
`diagnostics/matched-audio-isolation-v1/{dino,clip}/`.
Each contains one-shot `started.json`, fsynced `events.jsonl`, `report.json` and
writer lock. First-case Windows check gates the second case. Source scripts:
`docs/diagnostics/matched-dino-clip-generation-2026-10-11.py` and
`docs/diagnostics/isolate-nvidia-audio-2026-10-11.ps1`.

B: evidence namespace:
`B:\DeSpaMo\diagnostics\gpu-crashes-2026-10-11\matched-audio-isolation-v1`;
contains authorization and audio-before/disable/restore receipts.

- DINO report SHA256:`96ee0dc8a9f7746d5784d36f92ae2fb8091e4d6125464bccfc30c06f5c355104`.
- CLIP report SHA256:`8cfb716355e1564ff2f0557f0e4a698a090c08eee504c2b31a867e6bf895d311`.
- Python invocation SHA256:`6b49c34a396b91f5ec7d5ad02da47aa2bd647b01789bd4a20c38bbcb13f33a76`.
- Audio helper SHA256:`519bf4c7459a378acb79adf754643e24b66f63bc07bb80bdb337ea2ab6d3eed9`.

Manual stop remains`stopped_by_user`; automatic recovery disabled; GPU diagnostic
permission false; scoped two-generation approval consumed. Training remains at
1774updates with497.553348937s charges; native checkpoint/source identities intact.
Diagnostic timings are retained separately; no training-budget reset, new study
root, dev selection, Seed1 or official test scoring. No cached-evaluator migration
bridge implemented or training/controller/watchdog restarted.

Recommended next review: a small fixed longer-clip dev subset through cached
batch1 evaluation before full-dev or training recovery. Additional device/GPU
work needs a newly reviewed scope; current one-shot namespace must not be reused.
