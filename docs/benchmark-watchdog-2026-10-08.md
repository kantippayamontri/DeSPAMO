# Detached benchmark watchdog — 2026-10-08

User authorized20-minute monitoring and error recovery without further approval.
Implementation: `scripts/watch_official_benchmark.py`, outside the active source
closure. It does not modify benchmark code, data, parameters, checkpoints or claims.

## Active monitor

- PID763322, detached with `nohup`, unique `.watch.lock` owned.
- Interval **1,200seconds** using `time.sleep(1200)`; exits after study completes.
- Root `/home/kan/datasets/despamo/official-phoenix14t-translation-v1/watchdog`.
- Latest check/owner/next time: `status.json`.
- Durable append-only observations/actions: `history.jsonl`.
- Restart output: `recovery-controller.log`.
- Launcher output: study parent `watchdog-launcher.log`.

First health check found original controller/seed0-reference worker running with
matching PID/boot/start ticks/commands. Step8571, epoch9.663/40, phase`train`.
Dev BLEU4 epochs2/4/6/8:0.007336/4.307309/6.485516/7.267520. Intermediate scores,
not final benchmark/test results. Next check recorded
`2026-10-08T15:32:00Z`. Monitor process was alive in `hrtimer_nanosleep`.
Live state may have advanced; this is a snapshot.

## Automatic recovery scope

Same controller command, same retained worktree/main interpreter and required
runtime environment. Never change physicalbatch8,40epochs, source identities,
checkpoint selection or test claims. Never stop a live worker because of elapsed
time; dev generation can legitimately hold the same optimizer count.

Recovery is eligible only if original controller and every owned worker have
exited, no PID-reuse mismatch exists, and either controller disappeared or a
known transient failure was recorded (interrupt/signal15, connection/broken-pipe,
I/O error, CUDA memory/driver failure). Before restarting:

1. Verify stored recipe8/40/35480 and20dev epochs.
2. Verify locked runtime versions and all frozen code/base-model/data-binding hashes.
3. Verify retained resume hashes/sizes/identity/step accounting.
4. Require idle GPU and free native controller lock; recheck ownership after hashing.
5. Check Linux/Windows C:/B: space against conservative56GiB threshold.

Same-root native resume logic restores checkpoint/sampler/RNG and preserves
elapsed/replayed charges. Existing completion/test claims remain authoritative.
Total automatic restarts bounded to3; repeated/unknown failures flag `needs_agent`
or retry-limit status rather than burning GPU time indefinitely. Source/integrity
drift, corrupt checkpoints and storage shortage are not bypassed; no cleanup,
batch reduction, new root, test reselection or speculative code edit occurs.

If controller exits while a matching worker remains alive, record
`orphan_worker_active` and wait. Do not kill it or start a duplicate. Controller
can later resume bookkeeping using the existing complete run.

## Limit of unattended shell monitoring

This is a persistent scripted watchdog, **not a continuously running coding AI**.
It can collect health and execute the verified recovery path above after this
conversation returns. It cannot reason about and patch arbitrary new programming
bugs while no coding-agent session is active. Those are recorded as `needs_agent`
with diagnostic error/log context. The user's approval permits the next active
agent to diagnose/fix such issues without asking again, within frozen study scope;
it does not justify silently changing model/recipe/data or bypassing integrity.

Monitoring observations are local files; there is no push notification/chat
callback promised. On return, read `status.json/history.jsonl` before action.
The monitor itself is not a boot-persistent system service; WSL/host shutdown
stops it. Do not shut down/compact WSL during active training.

## Checks

Six watchdog tests watched RED→GREEN: live controller never restarts, complete
study stops, orphan worker waits, reusedPID blocks, transient interruption
eligible only after exit, unknowncode failure flags agent. Ruff/Pyright pass.
Detached process PID/start ticks/command,20-minute interval/next time and first
health record verified. Active benchmark's complete source closure still matches.
No existing controller/worker was stopped or restarted during setup.
