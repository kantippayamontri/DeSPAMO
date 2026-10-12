# Admin crash-dump handoff

Current Linux/OpenCode session is not Windows-elevated. Both Windows minidumps
still deny read access; neither SDK debugger nor WinDbg app is installed at the
checked locations. The next step requires a Windows Administrator session.
This is a permission/debugger blocker, not approval to resume GPU work.

## Export local evidence

Open **Windows PowerShell as Administrator** and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "\\wsl.localhost\Ubuntu-26.04\home\kan\Research\DeSpaMo\docs\diagnostics\export-gpu-crash-evidence.ps1"
```

ExecutionPolicy Bypass is scoped to that child process; script changes no registry,
drivers or system execution policy. It copies only the two named protected dumps
to `B:\DeSpaMo\diagnostics\gpu-crashes-2026-10-11`, verifies SHA256, and writes
limited System crash events plus runtime metadata. Originals and existing unequal
copies are preserved. No dump upload, CUDA job or training restart is performed.

## Analyze with WinDbg

Install/open WinDbg using [Microsoft installation instructions](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/).
Open each copied dump via File > Open dump file. Symbol downloads may be needed;
keep the dumps local. In debugger command window run:

```text
.symfix
.reload
!analyze -v
lmvm nvlddmkm
```

Copy the analysis fields MODULE_NAME, IMAGE_NAME, FAILURE_BUCKET_ID and STACK_TEXT,
plus driver information from `lmvm`, or save a local debugger log. For example,
before those commands:

```text
.logopen B:\DeSpaMo\diagnostics\gpu-crashes-2026-10-11\analysis-9468.txt
```

Afterward run `.logclose`; use a distinct file for the other dump. Return to agent
after export/analysis so access and actual findings can be checked. Evidence
availability does not automatically clear GPU hold or authorize training.

Reference: [Microsoft kernel-dump analysis](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/analyzing-a-kernel-mode-dump-file-with-windbg).
