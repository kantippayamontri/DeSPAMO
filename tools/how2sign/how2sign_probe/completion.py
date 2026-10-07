"""Completion gate based on expected jobs, not on a driver's exit marker alone."""

import argparse
import json
from pathlib import Path

from how2sign_probe.cli import _jobs_from_regions
from how2sign_probe.labeling import journal_path
from how2sign_probe.labels import parse_label


def validate_completion(root):
    root = Path(root)
    if not json.loads((root / "detector_report.json").read_text())["gate"]["pass"]:
        raise ValueError("detector gate failed")
    jobs = _jobs_from_regions(root)
    if not jobs:
        raise ValueError("no expected jobs")
    done = failed = unfinished = 0
    for job in jobs:
        path = journal_path(root, job)
        if not path.exists():
            unfinished += 1
            continue
        state = json.loads(path.read_text())
        if json.loads(state["identity"]) != list(job.identity):
            raise ValueError(f"journal identity mismatch: {job.key}")
        if state["status"] == "done":
            parse_label(job.kind, json.dumps(state["label"]))
            done += 1
        elif state["status"] == "failed":
            failed += 1
        else:
            unfinished += 1
    if unfinished:
        raise ValueError(f"unfinished jobs: {unfinished}")
    if done / len(jobs) < 0.99 or failed > max(5, len(jobs) // 200):
        raise ValueError(f"too many failed labels: {failed}/{len(jobs)}")
    return dict(expected_jobs=len(jobs), done=done, failed=failed, unfinished=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_completion(args.root)))


if __name__ == "__main__":
    main()
