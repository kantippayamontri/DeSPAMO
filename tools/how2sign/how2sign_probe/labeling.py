"""Restart-safe crop labeling through local Ollama replicas.

One journal file per crop; every request is reserved before the network call,
so an interrupted run resumes without duplicating VLM work.
"""

import base64
import dataclasses
import hashlib
import json
import time
import urllib.request

import cv2

from how2sign_probe.labels import build_prompt, parse_label

MAX_ATTEMPTS = 3
ENDPOINT = "http://127.0.0.1:{port}/api/chat"
# In raw mode ollama only renders ``{role}\n{content}``; the two
# newlines below complete an empty  block so the pinned
# qwen3-vl:8b answers directly instead of generating ~1,200 hidden thinking
# tokens per crop (measured 1,195 eval tokens vs 34, same strict-vocabulary
# outputs). Requesting ``think: False`` alongside raw mode makes the server
# re-open a think block after our skip, so the flag must stay absent.
THINK_SKIP = "\n\n"


@dataclasses.dataclass(frozen=True)
class Job:
    key: str
    kind: str
    side: str
    clip: str
    frame_index: int
    region_json: str

    @property
    def identity(self):
        return (self.kind, self.side, self.clip, self.frame_index, self.region_json)


def job_key(kind, side, clip, frame_index, region_json) -> str:
    return hashlib.sha256(
        json.dumps([kind, side, clip, frame_index, region_json], sort_keys=True).encode()
    ).hexdigest()


def make_job(kind, side, clip, frame_index, region_json) -> Job:
    return Job(
        key=job_key(kind, side, clip, frame_index, region_json),
        kind=kind,
        side=side,
        clip=clip,
        frame_index=frame_index,
        region_json=region_json,
    )


def journal_path(root, job: Job):
    return root / "jobs" / job.key[:2] / f"{job.key}.json"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".partial-" + path.name)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
    import os

    os.replace(tmp, path)


def load_journal(root, job: Job):
    path = journal_path(root, job)
    if not path.exists():
        return dict(status="pending", attempts=[])
    return json.loads(path.read_text(encoding="utf-8"))


def encode_crop(crop_bgr) -> str:
    ok, png = cv2.imencode(".png", crop_bgr)
    if not ok:
        raise ValueError("png encode failed")
    return base64.b64encode(png.tobytes()).decode()


def request_label(prompt_text, image_b64, model, port, kind, timeout=900):
    payload = {
        "model": model,
        "raw": True,
        "messages": [
            {"role": "user", "content": prompt_text, "images": [image_b64]},
            {"role": "assistant", "content": THINK_SKIP},
        ],
        "options": {
            "temperature": 0.0,
            "seed": 0,
            "num_ctx": 4096,
            "num_gpu": 99,
            "num_predict": 2500,
        },
        "keep_alive": "30m",
    }
    url = ENDPOINT.format(port=port)
    parts, last = [], {}
    with urllib.request.urlopen(url, json.dumps(payload).encode(), timeout=timeout) as stream:
        for line in stream:
            chunk = json.loads(line)
            last = chunk
            parts.append(chunk.get("message", {}).get("content", ""))
    return "".join(parts), last


def run_job(root, job: Job, crops_dir, model: str, port: int, sleeper=time.sleep):
    state = load_journal(root, job)
    state["identity"] = json.dumps(list(job.identity))
    if state["status"] == "failed":
        state.update(status="pending", attempts=[], error=None)
    if state["status"] == "done":
        atomic_json(journal_path(root, job), state)
        return state
    prompt_text = build_prompt(job.kind, job.side)
    try:
        image_b64 = encode_crop(cv2.imread(str(crops_dir / f"{job.key}.png")))
    except Exception as error:  # noqa: BLE001 - journal missing/corrupt crop instead of killing the pool
        state.update(
            status="failed",
            attempts=[
                {
                    "raw": None,
                    "error": f"crop read failed: {error}",
                    "latency_s": None,
                }
            ],
            error="crop unavailable",
        )
        atomic_json(journal_path(root, job), state)
        return state
    while state["status"] != "done" and len(state["attempts"]) < MAX_ATTEMPTS:
        attempt = {"raw": None, "error": "interrupted", "latency_s": None}
        state["attempts"].append(attempt)
        state["status"] = "pending"
        atomic_json(journal_path(root, job), state)
        started = time.time()
        try:
            raw, meta = request_label(prompt_text, image_b64, model, port, job.kind)
        except Exception as error:  # noqa: BLE001 - journal every transport failure
            attempt.update(
                raw=None, error=f"request: {error}", latency_s=round(time.time() - started, 1)
            )
            atomic_json(journal_path(root, job), state)
            sleeper(2.0)
            continue
        attempt["latency_s"] = round(time.time() - started, 1)
        attempt["eval_count"] = meta.get("eval_count")
        try:
            parsed = parse_label(job.kind, raw)
        except ValueError as error:
            attempt.update(raw=raw[:4000], error=f"parse: {error}")
            atomic_json(journal_path(root, job), state)
            continue
        attempt.update(raw=raw[:4000], error=None)
        state.update(status="done", label=parsed, attempts=state["attempts"])
        atomic_json(journal_path(root, job), state)
    if state["status"] != "done":
        state["status"] = "failed"
        atomic_json(journal_path(root, job), state)
    return state
