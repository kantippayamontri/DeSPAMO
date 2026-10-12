# Local WinDbg analysis — both dumps agree

User completed Admin evidence export and approved Microsoft's official WinDbg
installation. Installed Microsoft.WinDbg1.2610.1001.0 via documented WinGet package,
installer hash verified. Used bundled amd64KD console debugger through native
PowerShell, CPU-only. Microsoft public symbols downloaded; dumps stayed local.

## Verified results

| Field | Both exported dumps |
| --- | --- |
| Bugcheck | VIDEO_TDR_FAILURE0x116 |
| MODULE_NAME | nvlddmkm |
| IMAGE_NAME | nvlddmkm.sys |
| FAILURE_BUCKET_ID | 0x116_IMAGE_nvlddmkm.sys |
| PROCESS_NAME | System |
| SYMBOL_NAME / driver owner offset | nvlddmkm+1958210 |
| Last-operation NTSTATUS | C000009A |

Both stacks show:

```text
Windows timeout reset worker
    -> TdrResetFromTimeout
    -> DXGADAPTER::Reset
    -> ADAPTER_RENDER::Reset
    -> TdrBugcheckOnTimeout
    -> KeBugCheckEx (blue screen)
```

This confirms Windows graphics recovery failing against NVIDIA driver after GPU
timeout. System is the kernel reset worker; it does not establish that Python
workload was irrelevant. Driver owner offset is not an identified faulty CUDA
instruction. Exact driver defect versus hardware contribution and offending
CUDA kernel cannot be deduced from this stack alone. C000009Ainsufficient resources
does not prove PyTorch VRAM exhaustion.

Current driver file32.0.16.1062corresponds to NVIDIA610.62. Dump module timestamp
shows June12,2026 and image size0x0736D000. NVIDIA private symbols unavailable;
Microsoft kernel symbols successfully loaded for nt/dxgkrnl.

## Local analysis artifacts

Under `B:\DeSpaMo\diagnostics\gpu-crashes-2026-10-11`:

- `analysis-symbols-13859.txt`, final valid analysis, SHA256
  `0bad06ad0f46a1cf390cb3bf5d0f99d655d9623c893df5d6687367f2878f054f`.
- `analysis-symbols-9468.txt`, final valid analysis, SHA256
  `5515e5e3468f8eec716d775cc6f585be4b820d3abd00466af75e477d5f5902ec`.
- `debugger-commands.txt`, Microsoft symbol cache `symbols/`, exported dumps and
  original manifest/System events retained.

Initial `analysis-*.txt` used malformed symbol-command parsing and produced
WRONG_SYMBOLS/nt_wrong_symbols placeholders. Discard those findings. Retried with
explicit `-y srv*<cache>*https://msdl.microsoft.com/download/symbols` and separate
`-cf` command file. Both final logs have valid reset stacks and no WRONG_SYMBOLS.
WindowsApp binaries require native PowerShell execution rather than direct WSL
filesystem launch. ANSI/cp1252stdout decoding was handled without changing dumps.

Receipt: [windbg-gpu-crash-analysis-2026-10-11.json](windbg-gpu-crash-analysis-2026-10-11.json).

## Decision

GPU hold remains active. Installing debugger and analyzing dumps is evidence
collection, not remediation. No driver/runtime/TDR changes, CUDA validation,
training or watchdog restart performed. Before bounded generation validation,
review supported NVIDIA driver/WSL stability and the separately CPU-tested cached
batch1path with user. Host driver replacement/rollback requires separate explicit
authorization; do not silently change drivers or increase TDR timeouts. Any later
reset/CUDA unknown error ends the attempt without retries, preserving epoch2state.

Related: [host timeline/code audit](host-reboot-diagnosis-2026-10-11.md),
[prepared inference mitigation](clip-s2-inference-mitigation-2026-10-11.md).
