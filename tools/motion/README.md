# Dedicated motion extractor

Task2 fix3 distinguishes pending terminal candidates from admitted success, with
closing/global checks after watchdog wait and all final cleanup. Earlier profile
evidence/inventory fixes remain unchanged. Verification uses
CPU fixtures and small real processes; real GPU profile/extraction remains
reserved for Task5 review and execution.

## Private pinned environment

From the checkout root:

```sh
uv lock --project tools/motion --python 3.11
uv sync --project tools/motion --locked --python 3.11 --link-mode copy
uv pip check --python tools/motion/.venv/bin/python
PYTHONPATH=src:. tools/motion/.venv/bin/python -m pytest tools/motion/tests/test_motion_extractor.py -q
```

`tools/motion/.venv` is independent of the training environment. Installed files
are private copies, not cache hardlinks. Python3.11; distribution torch2.0.1
(runtime2.0.1+cu117/CUDA11.7), transformers4.32.0, numpy1.26.4, pillow11.3.0,
safetensors0.8.0, huggingface-hub0.36.2, pytest8.4.2. Pydantic2.13.5 is also
required by Task1's neutral provenance/schema imports. Full cross-platform
dependency resolution is in this directory's `uv.lock`; Windows-only colorama
is not an active Linux dependency.

## Public APIs

```python
load_encoder(snapshot: Path) -> tuple[object, object]
extract_motion_clip(paths: tuple[Path, ...], model: object, processor: object,
                    *, batch_size: int = 16
                    ) -> tuple[np.ndarray, list[dict], tuple[dict, ...]]
run_motion_extraction(args) -> dict
```

`load_encoder` loads cached safetensors on CPU, checks missing/mismatched encoder
weights, config24layers/1024width/16frames and all FP32 parameters, freezes and
sets eval. The charged worker moves it to CUDA. No downloads, remote code,
fine-tuning, optimizer, BF16, autocast or TF32.

Snapshot revision: `12da269a02d3e1fbbb7011e610e591fea8061dca` under the approved
`MCG-NJU/videomae-large` cache. This snapshot contains no preprocessor config:
the explicit processor is constructed locally. Its resolved configuration and
actual copied package source/native/metadata files are pinned as evidence.
Snapshot symlinks resolve to regular cache blobs; model loading reads those same
snapshot bytes and rechecks them against admitted evidence.

Source images remain their recorded210x260 pixels. Each frame's single byte
buffer supplies both SHA256 and RGB decoding. The processor uses bilinear
shortestedge224, centrecrop224, rescale1/255, mean/std0.5. Windows16/stride8,
ascending full windows, drop tails, repeat last frame only for short input.
NumPy output becomes `torch.from_numpy`; parity uses standard processor tensors
at B1/8/16. Extraction selects exactly `hidden_states[-1][:,0]`, before final
normalization; outputs must be finite float32 `[windows,1024]`.

Sources contain exactly `index/path/sha256`. Window rows contain exactly
`row/input_source_indices/real_start/real_end/centre`. Input ordinals are captured
from actual ordered image objects at each model invocation, then associated with
the corresponding returned output rows. Dropped-tail source bytes remain
inventoried. `generate_clip` preserves the spatial map's path spelling and compares
the consumed aggregate source hash before writing arrays/receipts. No old arrays
are converted into new generating evidence.

## CLI and result contract for Tasks3/4

```sh
PYTHONPATH=src:. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
tools/motion/.venv/bin/python scripts/extract_motion_features.py \
  --mode preflight \
  --adapted-root /home/kan/datasets/despamo/signer-pilot/e3-features-7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e \
  --annotation /home/kan/Research/SpaMo/preprocess/Phoenix14T/train_info_ml.npy \
  --model-snapshot /home/kan/datasets/spamo/hf_cache/models--MCG-NJU--videomae-large/snapshots/12da269a02d3e1fbbb7011e610e591fea8061dca \
  --output-base /home/kan/datasets/despamo/signer-pilot \
  --environment-lock /home/kan/Research/DeSpaMo/tools/motion/uv.lock \
  --gpu-cap-seconds 18000
```

`preflight` is CPU-only and publishes no root or runtime/package artifacts. It
enforces7096 physical IDs/827354 frames/93232 windows and logical5746/582/768,
using the legacy annotation's integer-key records. There is no production count
bypass. `profile` and `run` additionally require
`--authorize-motion-extraction`. Subsequent stages/recovery require `--resume`.
Run requires a byte-verified profile from the same cumulative clock.

`run_motion_extraction` returns exactly:

| Key | Value |
| --- | --- |
| `schema_version`, `policy` | `1`, `motion-extraction-result-v1` |
| `mode` | `preflight`, `profile`, `run` |
| `status` | `ready`, `profiled`, `complete`, `failed`, `capped` |
| `featurekey`, `identity` | verified content key/identity; nullable before binding |
| `root` | verified external feature root; nullable before binding |
| `bootstrap_root`, `bootstrapkey` | fixed early root and immutable launch digest |
| `binding_path`, `input_observation_path` | immutable late binding / first CPU observation |
| `budget_path`, `status_path` | single bootstrap ledger and status snapshot |
| `profile_path` | feature-root profile path; nullable before binding |
| `spent_seconds`, `remaining_seconds` | actual uncapped spend / nonnegative remaining cap |
| `authorization` | extraction18000/training64800/cap_transferFalse |
| `package` | Task1 descriptor or `None` |
| `profile` | profile record or `None` |
| `error` | typed failure string or `None` |
| `admission_seconds` | extra bounded read-only admission time for closed packages; otherwise0 |
| `terminal_admission` | bounded API-return proof for caller/CLI persistence; otherwiseNone |

Authorization/path/launch-owner conflicts can raise before returning this dictionary.
Once clock ownership begins, preparation failures return failed/capped with durable
uncapped spend, including before a feature key exists. CPU preflight returns no
bootstrap/clock/binding/status paths and writes nothing. CLI emits
one JSON result on normal return; exit0 only for ready/profiled/complete, exit1
for failed/capped. Task4 must check status, clock and independent package
validation; exit0 or a marker alone is not an extraction handoff.

Task1 descriptor travels unchanged to Task3: package_policy/root/identity/
identity_hash, manifest_path/hash, timing_path/hash, annotation_path/hash,
spatial_frame_rows_path/hash, completion/completion_hash, clip_ids and bindings.
All bindings are independently checked by `validate_motion_package`.

## Durable ownership, clocks and recovery

One fixed `output-base/motion-extraction-bootstrap-v1` owns launch `owner.json`,
`.writer.lock`, `budget.json`, `status.json`, `input-observation.json`,
`binding.json`, and preparation request/result/error receipts. Bootstrap location
is not content/code/path-hash keyed: changed inputs, paths, code, interpreter or
hardware cannot mint a new allowance. Every resume retains the same18000s ledger.
Unknown bootstrap files and conflicting launch owners reject before ledger/status
writes. Feature roots own arrays/receipts/publication metadata and their own
writer/owner; they never get a second budget.
A pilot-wide `.motion-extraction-gpu.lock` serializes different content roots.
Locks are inherited by an independent Python watchdog and its worker, so caller
SIGKILL does not release GPU ownership prematurely. The EOF launch gate opens
only after watchdog PID/startticks/PGID/command and boot identity are persisted.
No live owned process is adopted, no unowned process is signalled. Host `timeout`
is uutils, so the watchdog does not depend on GNU timeout.

Runtime clock policy `motion-extraction-clock-v2`: schema_version, bootstrapkey,
featurekey(initiallyNone), binding(initiallyNone), authorization, spent_seconds,
status, active, child, stages, watchdogs; successful profile adds profile_sha256.
Active includes mode, phase(preparing/extracting), started_boot_id, boot_id,
monotonic, wall, owner, started_spent and preparation_seconds after binding.
Stages retain phase/boot/owner/child/start/end spend and preparation seconds.
Watchdogs retain phase, boot, PID/startticks/PGID/command and observed spend/exit
or interrupted reconciliation. Same-boot elapsed uses
monotonic time; reboot uses conservative wall elapsed. Rollback grants no new
allowance. Actual spend is never clamped to18000. A30second reserve precedes the
cooperative deadline; the separate watchdog bounds the worker and kills its
owned process group. Preparation, hashing, model loading, profile/run, writes and
validation share the same interval; binding never restarts or retrocharges it.
Status files are provisional snapshots: require admitted terminal state and charged
upper bound, not exact equality with the earlier status snapshot's spend.
An independent parent watchdog covers preparation, child supervision and parent
closure. Successful closure reserves a conservative5s IO bound within the existing
cap/margin. Budget `spent_semantics=observed_work_plus_conservative_closing_bound`
means charged spend is an upper bound, not invented exact last-fsync timing.
Every check refreshes after blocking persist; late terminal writes cannot succeed.
Failure spend remains an observed uncapped lower bound; persisted observation
anchors allow conservative tail/reboot reconciliation without refund/reset.

Durable budget stays `terminal_pending`, with active closing anchor and separate
`terminal_candidate{status,spent_upper_bound}` through seal/wait/cleanup. It is
chargeable and can transition failed/capped; it never early-returns as complete.
`terminal_close` binds nonce, immutable candidate/receipt/admission paths, failure
path, boot/start/spend/upper bound and closing/global deadlines. Candidate snapshot
contains pending clock plus raw pending-budget hash; watchdog receipt binds actual
candidate, pending budget and status hashes. Parent checks BOTH deadlines after
acknowledgment, watchdog exit, cleanup/FD release and immediately before API return.
Late wait/cleanup discards package and persists anchored uncapped failure spend.

Successful API result includes `terminal_admission`, policy
`motion-terminal-admission-v1`: status/keys, candidate/receipt/budget hashes,
conservative charged upper bound, observed closing spend and parent return-check
time. It is not persisted before the final return check. CLI calls
`record_terminal_admission(result)` only AFTER extractor return, writing immutable
`terminal-admission-<nonce>.json` under exclusive ownership and exact proof checks.
Task4's outer watchdog must cover this persistence, stdout delivery and consumer
IO: they are tail BEYOND extractor return, with no invented exact tail duration.
Direct Python callers must likewise record returned admission before resume.

Terminal admission requires finite, non-boolean receipt and parent-check monotonic
timestamps, with `0 <= closing_start <= receipt_observation <= parent_checked_monotonic
< min(closing_deadline, parent_deadline)` under the same boot/proof bindings.
Negative, pre-start and pre-receipt checks reject. Chronological consistency alone
does not authenticate timestamps; Task4's outer watchdog still covers CLI proof
persistence, stdout delivery and consumer IO.

Pending candidate without caller admission is interrupted, not complete; owned
resume reconciles its durable anchor. Admitted state is an immutable projection
from pending budget + sealed candidate + caller admission, yielding effective
profiled/complete and conservative spend without rewriting those closed bytes.
Already-admitted complete adoption remains separately read-only. Failure during
caller admission or CLI delivery is not a successful pipeline handoff.
Closed extraction adoption performs independently supervised CPU validation within
its remaining allowance, records admission_seconds in the returned result, and
does not rewrite the completed extraction ledger/generating records or launch GPU.

Array and receipt commits fsync data and parent directories. An owned array
without a receipt can be recomputed. Completed clips are byte/source/recipe
validated before reuse. Unknown files, aliases and crash temp debris reject;
they are retained for explicit inspection. `generating-run.json` is immutable
actual generating evidence, separate from mutable clock/status files. Task1's
public recovery API fills only verified missing publication metadata, under the
same exclusive writer lock; valid completed packages are adopted read-only.
Both publisher and recovery receive a read-only before_commit guard requiring
same boot and at least30s remaining. Full candidate validation/guard failures leave
no marker. Atomic commit or marker-hash IO can fail after visibility; that stage
is failed/capped and never becomes successful handoff from marker existence.

Profile samples deterministic length quantiles including the longest sequence.
It records actual read/hash/decode/processor/inference/array-write/receipt costs,
consumed bytes/frames, input frames/windows/batches, array sizes, host RSS and GPU
peak, model-load seconds and B1/8/16 parity. Conservative remaining-work and
whole-validation projections must fit remaining allowance. Both Linux and
Windows must retain projected demand plus50GiB before GPU work.

Profile schema_version2 uses immutable `profile-samples/<clip_id>.json`
observations committed immediately after each freshly generated sample. They bind
featurekey, raw receipt hash, source hash, array hash, frame/window counts, actual
observed total/inference seconds, invocation metrics and observer/boot identity.
All samples are revalidated against current receipts/sources/features on retry.
Report `samples` contains every valid prior/fresh observation; `attempt_measurements`
separately records reuse/validation costs, and `observation_sha256` binds raw files.
Conservative rate is max(prior conservative rate,1.25×max actual observed seconds
per window). Fast reuse reads never become inference timings. A committed feature/
receipt without its durable inference observation fails closed; no automatic
re-probe or rewriting of valid feature/receipt is performed. Both observation
files and conservative prior max survive interruption and mixed retries.

Task4 admission requires successful bounded result, matching immutable bootstrap
launch/binding/observation, validated admitted clock projection with spend<=18000,
successful parent exit and outer cap check including delivery/admission tail, and independent
Task1 package validation. Neither a provisional status, child exit0 nor visible
content marker is sufficient. See appended Task2 fix3 report for exact schemas,
RED/GREEN and ownership/recovery evidence.
