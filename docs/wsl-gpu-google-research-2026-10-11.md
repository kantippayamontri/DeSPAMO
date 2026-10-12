# Google-first WSL GPU crash research — 2026-10-11

Status: research only; no driver/WSL/runtime change or GPU validation authorized.
This follows `host-stack-mitigation-review-2026-10-11.md` and adds more specific
external incident reports. These reports do not establish our root cause.

## Search execution

Google AI Mode searches completed successfully with browser GPU acceleration,
GPU compositing, WebGL, Vulkan and accelerated canvas disabled. The browser used
an existing Chromium binary through a temporary process-local adapter in
`/tmp/opencode/google-cpu-search/sitecustomize.py`. Missing NSS/NSPR libraries
were downloaded and extracted into that temporary directory, not installed into
the system or project environments. No persistent skill configuration changed.

Raw results:

- `/home/kan/.claude/skills/google-ai-mode/results/2026-10-11_15-01-25_WSL2_RTX_4090_PyTorch_CUDA_unknown_error.md`
- `/home/kan/.claude/skills/google-ai-mode/results/2026-10-11_15-02-04_WSL2_CUDA_GPU_crashes_nvlddmkm_VIDEO_TDR.md`

Searches covered WSL2, RTX4090, PyTorch CUDA unknown errors, nvlddmkm and
VIDEO_TDR_FAILURE, driver fixes, memory pressure and GPU scheduling. Google's
source extractor returned zero usable source-list entries. Its prose contains
unsubstantiated claims labeled "confirmed"; those labels are not accepted as
evidence. Follow-up GitHub API searches and the original NVIDIA forum JSON were
used to verify reports independently.

## Closest verified public report

Microsoft WSL issue **41224**, open at inspection, reports:

- NVIDIA Windows driver **610.62**, WSL **2.7.10.0**, kernel **6.18.33.2-2**,
  Windows11 25H2: close host-stack match to our incident.
- CUDA unknown error, NVIDIA events14/153 and GPU loss/TDR; retained dump
  `0x116_IMAGE_nvlddmkm.sys`, `nvlddmkm+0x1958210`, Arg3
  `STATUS_INSUFFICIENT_RESOURCES`: close failure-signature match.
- **RTX5090**, qfloat8 model block transfers/quantization, and additional
  PCIe SRAM/ECC event payloads: important differences from our RTX4090/T5 dev
  generation. Matching driver and reset stack are not proof of matching cause.
- The reporter explicitly treats earlier `make_resident -12` failures as possible
  precursor state, not proof of direct causation.

The same reporter's original NVIDIA forum thread describes failure intervals
with observed memory below capacity and no recorded CUDA OOM. Those samples
are not exact failure-instant peak measurements. This is evidence that similar
host reset failures need not imply VRAM exhaustion, not evidence about our peak.

### Reported mitigation and strength of evidence

On August14 the reporter says NVIDIA requested a **610.62 → 610.74** host-driver
update. The previously failing two-process model-load sequence then completed
without GPU loss, TDR or new target Windows events. The author expressly limits
the result to a bounded retest, not proof that an intermittent fault is fixed.

On September9 the reporter quotes private NVIDIA bug6546168 correspondence:
"This largely confirms our assumption that the issue is fixed in driver version
610.74." This is the reporter's account of vendor correspondence, not an
independently available public NVIDIA engineering confirmation or RTX4090 fix.

A second person in the NVIDIA thread reports continuing failures on Studio
**616.56** after a clean installation/DDU. Disabling HAGS reportedly stopped
Windows reboots but did **not** stop GPU loss. They later report that Game Ready
**616.64** resolved their GPU-loss problem. Their explanation that a fix was
exclusive to a driver channel is speculation; no public release-note verification
was performed. Studio branding and a higher version alone are not fix evidence.

## Memory-pressure report

WSL issue **40732**, open at inspection, alleges repeated CUDA OOM retries can
end in a host BSOD on a different RTX5090 laptop/driver610.47 stack. This is a
user report, not an established GPU-PV corruption mechanism. Our logs have not
established repeated explicit OOMs or peak VRAM exhaustion. Keep VRAM pressure
as a live hypothesis, avoid retrying after GPU errors, and measure allocation,
reservation and host-total usage during any separately approved bounded probe.

## Implication for our next action

The 610.62 report makes a **verified Windows-host NVIDIA driver replacement** a
more specific lead than generic WSL updating or arbitrary TDR changes. First
verify an official RTX4090/Windows11 installer, supported hardware, exact version,
signature/release notes and rollback route. Do not install 610.74 or 616.64 simply
because a different GPU's owner reported improvement; those versions are leads,
not selected installation targets or a guaranteed repair.

Maintain the v3 hold. Driver installation and Windows restart need explicit
authorization; later cached batch1 dev validation needs its separate approval
gate. WSL stable updates remain a separate maintenance candidate. HAGS changes,
DDU, clock/power changes and hardware replacement are not justified automatically
by these reports. Leave TDR registry defaults unchanged.

## Verified source links

- https://github.com/microsoft/WSL/issues/41224
- https://github.com/microsoft/WSL/issues/41224#issuecomment-5290247082
- https://github.com/microsoft/WSL/issues/41224#issuecomment-5594705438
- https://forums.developer.nvidia.com/t/bug-report-rtx-5090-gpu-lost-0x116-tdr-and-failed-warm-reboot-after-pcie-sram-ecc-events-windows-11-610-62/378873
- https://forums.developer.nvidia.com/t/378873/5
- https://forums.developer.nvidia.com/t/378873/6
- https://forums.developer.nvidia.com/t/378873/7
- https://github.com/microsoft/WSL/issues/40732
- https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
