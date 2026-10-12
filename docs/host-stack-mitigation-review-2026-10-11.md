# Host-stack mitigation review — 2026-10-11

Inspection: 2026-10-11T06:48:43Z. Status: review complete; host changes and GPU
validation remain pending. This review does not establish a crash fix.

## Evidence

Both symbolicated minidumps identify `0x116_IMAGE_nvlddmkm.sys` and the same
Windows GPU-reset failure stack. The initiating CUDA kernel and the underlying
driver/hardware fault remain unresolved. See
`windbg-gpu-crash-analysis-2026-10-11.md` and its JSON receipt.

Read-only PowerShell CIM/registry queries and WSL version/list commands confirm:

| Component | Current observation |
| --- | --- |
| Windows | Microsoft Windows 11 Home, 25H2, build 26200.9457 |
| WSL | 2.7.10.0; kernel 6.18.33.2-2; WSLg 1.0.73.2 |
| Distribution | Ubuntu-26.04 running under WSL2; docker-desktop stopped |
| NVIDIA GPU | RTX 4090, signed NVIDIA driver 32.0.16.1062 (610.62), INF oem39.inf |
| Additional display adapter | AMD Radeon 780M, signed driver 32.0.21036.18, INF oem27.inf |
| TDR overrides | TdrDelay, TdrDdiDelay and TdrLevel values absent |
| Hardware scheduling override | HwSchMode value absent; effective scheduling state not established |
| Insider applicability fields | BranchName, ContentType and Ring absent; not proof of historical enrollment |
| Linux boot identity | c81d680c-fc38-4494-b0ea-a91e30b524c8 |

The Windows registry `ProductName` says Windows 10 Home, but CIM reports Windows
11 Home and Microsoft's release table places build 26200 in Windows 11 25H2.
Use the CIM/build evidence rather than the stale product-name string.

Microsoft's release table identifies the installed 26200.9457 revision as the
2026-09-14 general-availability out-of-band update. A later 26200.9550 revision
is listed; no matching GPU fix was established by this review. A Windows feature
upgrade is not required merely to make WSL CUDA available.

Microsoft's official WSL release API lists **3.0.1**, published
2026-09-29T17:18:57Z, with `prerelease=false`. Installed 2.7.10 is also a stable
release. Notes for stable 2.7.11/12/13/14 and 3.0.1 contain no explicit
GPU/CUDA/kernel/crash fix matching this incident. Updating is a maintenance
candidate, not a proven NVIDIA reset repair.

## Vendor guidance and limits

- NVIDIA recommends a current Windows production driver for WSL CUDA. Windows
  supplies WSL's CUDA driver interface; do not install a Linux display driver
  inside this distribution.
- NVIDIA's compatibility table explicitly permits newer drivers to run CUDA
  11.x applications through backward compatibility. The CUDA 11.7/driver 610.62
  version difference alone does **not** establish an unsupported combination.
  This is not a certification of every PyTorch/kernel/host combination.
- NVIDIA describes both Game Ready and Studio drivers as WHQL-certified; Studio
  focuses on creative-application testing. Studio branding is not evidence of
  a fix for this CUDA workload. CIM confirms signature status, not which
  installer channel supplied the existing driver.
- Microsoft's TDR registry guidance is for driver-development testing and
  explicitly advises end users against manipulating those keys. Leave timeout
  and reset behavior at their existing settings.
- No specific replacement NVIDIA driver version, documented 610.62 regression,
  or known-good rollback target was verified. An exploratory driver-service
  query with unverified product IDs returned no download; it is not driver
  availability evidence. No installer was downloaded or executed.
- Research used direct HTTP and read-only commands. Browser-backed Google AI
  search was not executed during the GPU hold.

## Proposed sequence, awaiting explicit host-change authorization

1. Prepare a Windows-host NVIDIA production-driver refresh: verify an official
   RTX 4090 / Windows 11 package, exact version, release notes and signature;
   record the existing package and a rollback route before installation.
   Prefer a verified current production package over an arbitrary downgrade.
   The package target is not yet selected. This is a mitigation hypothesis,
   not a diagnosed driver regression.
2. Apply only that approved driver change, then have the user complete the
   required Windows restart. Recheck versions and crash/event state before
   considering validation. Driver installation/restart may disconnect this
   WSL session; do not initiate either implicitly.
3. If a separate WSL update is chosen, use the stable update path
   `wsl.exe --update`, without `--pre-release`. Apply it as a separate recorded
   host change, not bundled with driver, Windows, PyTorch and TDR changes.
   Follow Microsoft's required restart guidance and verify the installed
   version rather than assuming that the Store delivers the GitHub tag.
4. Preserve the v3 manual stop. Host changes alone do not approve GPU work.
   Any later bounded cache-enabled batch-1 dev probe requires an explicitly
   reviewed call and the held receipt's `gpu_validation_approved=true` gate.
   Existing inference helpers are CPU-tested, opt-in and not wired into the
   old runner. A successful smoke would not prove long-run stability.
5. If mitigation and bounded validation succeed, prepare the retained epoch-2
   checkpoint migration bridge separately. Retain weights, optimizer/sampler/RNG
   state, original source/data identities and elapsed charges. Training recovery
   is not authorized by this review.

The user must approve the driver-change scope before package preparation proceeds
to installation. Hardware clock/power modifications and hardware stability have
not been established; do not assume stock settings or change them speculatively.

## Containment rechecked

The v3 `manual-stop.json` remains `stopped_by_user`, automatic recovery disabled,
last durable epoch 2 / 1774 updates, test unscored. A process scan found no matching
CLIP study/train, inference or benchmark-watchdog Python process. No GPU compute,
host setting change, package update, controller restart or source-bound runtime
edit occurred in this review.

## Sources

- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys
- https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x116---video-tdr-failure
- https://www.nvidia.com/en-us/drivers/
- https://learn.microsoft.com/en-us/windows/release-health/windows11-release-information
- https://api.github.com/repos/microsoft/WSL/releases/latest
- https://api.github.com/repos/microsoft/WSL/releases?per_page=40
- https://github.com/microsoft/WSL/releases/tag/3.0.1
- https://learn.microsoft.com/en-us/windows/wsl/basic-commands
