# Finish Windows space reclamation before fusion

Deletion completed:35approved files,194.42GiB. Online filesystem trim completed:
306.4GiB discarded. Current Linux free435.07GiB; Windows C: free83.63GiB.
Trim did not shrink the host virtual disk. Fusion storage admission remains
blocked on conservative Windows headroom, not on Linux space.

The Ubuntu-26.04 registry entry was queried read-only and identifies this disk:

`C:\Users\kanti\AppData\Local\wsl\{a83d506b-8a90-4175-a372-d34e90737c84}\ext4.vhdx`

## Run from a separate Windows terminal

Save current work and close WSL terminals/apps that could restart the distro.
The shutdown below terminates all WSL distributions, including this OpenCode
session. Disk compaction requires the VHDX to be detached or read-only.

Open **PowerShell as Administrator in Windows**, not inside WSL:

```powershell
wsl --shutdown
diskpart
```

At the `DISKPART>` prompt, run these commands individually:

```text
select vdisk file="C:\Users\kanti\AppData\Local\wsl\{a83d506b-8a90-4175-a372-d34e90737c84}\ext4.vhdx"
detail vdisk
compact vdisk
exit
```

Confirm the selected disk path before `compact vdisk`. Do not format, delete,
resize, or convert the disk. If it reports that the disk is in use, stop and
close the app restarting WSL rather than attempting compaction while mounted.

Then check Windows free space before reopening Ubuntu:

```powershell
Get-PSDrive C | Select-Object Name, Used, Free
wsl -d Ubuntu-26.04
```

After restarting, reopen OpenCode in `/home/kan/Research/DeSpaMo` and remeasure
both filesystems. Desired conservative starting headroom is approximately224GiB
each, based on173.6GiB estimated study peak plus50GiB reserve. Actual reclamation
is not guaranteed to equal deletion size. Training/design approval is separate.

## References and completed receipts

- [Microsoft: compact vdisk](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact-vdisk)
- [Microsoft: WSL shutdown](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#shutdown)
- [Deletion receipt](storage-cleanup-fusion-receipt-2026-10-05.json)
- [Hash-bound deletion manifest](storage-cleanup-fusion-manifest-2026-10-05.json)
- [Original candidate audit](storage-audit-fusion-2026-10-05.md)
