"""CLI for the Stage-1 How2Sign crop probe.

Stages are separate commands writing into one probe root so each can be rerun,
inspected, or resumed independently. Nothing outside the root is created.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from how2sign_probe.detector import Region, detect_regions
from how2sign_probe.frames import frame_count, read_frame, sample_indices
from how2sign_probe.labeling import make_job, run_job
from how2sign_probe.phoenix import EXCLUDED_SIGNERS
from how2sign_probe.phoenix import (
    extract_members as phoenix_extract_members,
)
from how2sign_probe.phoenix import (
    index_frames as phoenix_index_frames,
)
from how2sign_probe.phoenix import index_frames_dir as phoenix_index_frames_dir
from how2sign_probe.phoenix import (
    load_clips as phoenix_load_clips,
)
from how2sign_probe.phoenix import load_dir_clips as phoenix_load_dir_clips
from how2sign_probe.phoenix import read_frame as phoenix_read_frame
from how2sign_probe.phoenix import verify_counts as phoenix_verify_counts
from how2sign_probe.probe import (
    detector_gate,
    enumerate_jobs,
    region_stats,
    write_crop,
)
from how2sign_probe.provenance import (
    build_provenance,
    ollama_model_digest,
    package_code_sha256,
    sha256_file,
)
from how2sign_probe.report import (
    BASELINE_DISTINCT,
    BASELINE_TOP1_SHARE,
    aggregate_labels,
    label_rows,
    latency_stats,
)
from how2sign_probe.sampling import clip_rows, sample_clips
from how2sign_probe.verdict import verdict

MIN_DETECTOR_FRAMES = 100


def _safe_model_digest(model: str, port: int = 11434):
    try:
        return ollama_model_digest(model, port)
    except Exception:  # noqa: BLE001 - provenance must not block sampling when Ollama is down
        return None


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".partial-" + path.name)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=1)
    import os

    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_sample(args):
    root = Path(args.root)
    rows = clip_rows(Path(args.csv))
    picked = sample_clips(rows, total=args.total, seed=args.seed)
    provenance = build_provenance(Path(args.csv), args.model, args.seed, args.total)
    atomic_json(root / "provenance.json", provenance)
    atomic_json(
        root / "sample.json",
        {
            "clips": [
                {
                    "clip": row["clip"],
                    "stratum": row["stratum"],
                    "sentence": row.get("SENTENCE", ""),
                }
                for row in picked
            ]
        },
    )
    print(f"sampled {len(picked)} clips -> {root / 'sample.json'}")


def cmd_sample_phoenix(args):
    root = Path(args.root)
    if bool(args.tar) == bool(args.frames_root):
        raise SystemExit("provide exactly one of --tar / --frames-root")
    excluded = () if args.include_all_signers else EXCLUDED_SIGNERS
    if args.frames_root:
        rows = phoenix_load_dir_clips(
            Path(args.annotations), Path(args.frames_root), split=args.split, excluded=excluded
        )
    else:
        rows = phoenix_load_clips(Path(args.annotations), excluded=excluded)
    if args.total > len(rows):
        print(
            f"WARNING: requested {args.total} clips but eligible population is "
            f"{len(rows)}; sampling {len(rows)} instead",
            file=sys.stderr,
        )
    picked = sample_clips(rows, total=args.total, seed=args.seed)
    atomic_json(
        root / "provenance.json",
        {
            "code_sha256": package_code_sha256(),
            "annotations": str(args.annotations),
            "annotations_sha256": sha256_file(Path(args.annotations)),
            "tar": str(args.tar) if args.tar else None,
            "tar_size_bytes": Path(args.tar).stat().st_size if args.tar else None,
            "frames_root": str(args.frames_root) if args.frames_root else None,
            "split": args.split if args.frames_root else "train",
            "model": args.model,
            "model_digest": _safe_model_digest(args.model),
            "seed": args.seed,
            "requested_clips": args.total,
            "signers_excluded": sorted(excluded),
            "deviations": {
                "frozen_indices": (
                    "No PHOENIX appearance source.json exists on this host; the five frame "
                    "positions use the same 0.1-0.9 recipe as the How2Sign probe."
                ),
                "review_status": (
                    "labels are LLM-generated and unreviewed; human_review_status=not_assessed"
                ),
            },
        },
    )
    atomic_json(
        root / "sample.json",
        {
            "source": "phoenix-dir" if args.frames_root else "phoenix-tar",
            "annotations": str(args.annotations),
            "tar": str(args.tar) if args.tar else None,
            "frames_root": str(args.frames_root) if args.frames_root else None,
            "split": args.split if args.frames_root else "train",
            "clips": [
                {"clip": row["clip"], "stratum": row["stratum"], "num_frames": row["num_frames"]}
                for row in picked
            ],
        },
    )
    print(f"sampled {len(picked)} clips -> {root / 'sample.json'}")


def cmd_extract(args):
    root = Path(args.root)
    sample = read_json(root / "sample.json")
    if sample.get("source") != "phoenix-tar":
        raise SystemExit("extract requires a phoenix-tar sample.json")
    clips = {entry["clip"]: entry["num_frames"] for entry in sample["clips"]}
    frames = phoenix_index_frames(Path(args.tar), set(clips))
    phoenix_verify_counts(frames, clips)
    plan = {
        entry["clip"]: sorted({serials[i] for i in sample_indices(entry["num_frames"])})
        for entry, serials in ((e, sorted(frames[e["clip"]])) for e in sample["clips"])
    }
    member_lookup = {clip: frames[clip] for clip in frames}
    atomic_json(root / "extract_plan.json", plan)
    members = [member_lookup[clip][serial] for clip, serials in plan.items() for serial in serials]
    phoenix_extract_members(Path(args.tar), members, root / "frames")
    for clip, serials in plan.items():
        present = sorted(path.stem for path in (root / "frames" / clip).glob("*.png"))
        if present != serials:
            raise SystemExit(f"{clip}: extracted {present} != planned {serials}")
    print(f"extracted {len(members)} frames -> {root / 'frames'}")


def _detect_one(entry, source, root, video_dir, frame_index_dir):
    """Detect one clip; returns (clip, {index: items}, skip_reason_or_None)."""
    try:
        if source in ("phoenix-tar", "phoenix-dir"):
            indices = sample_indices(entry["num_frames"])
            if source == "phoenix-dir":
                frame_index = frame_index_dir[entry["clip"]]

                def read_one(index, _fi=frame_index):
                    return cv2.imread(str(_fi.abs_path(_fi[index])))
            else:
                positions = {index: pos for pos, index in enumerate(indices)}

                def read_one(index, _root=root, _clip=entry["clip"], _pos=positions):
                    return phoenix_read_frame(_root / "frames", _clip, _pos[index])
        else:
            path = video_dir / f"{entry['clip']}.mp4"
            indices = sample_indices(frame_count(path))

            def read_one(index, _path=path):
                return read_frame(_path, index)
    except ValueError as error:
        return entry["clip"], None, str(error)
    items = {}
    for index in indices:
        try:
            frame = read_one(index)
            regions = detect_regions(frame)
            items[index] = [
                [region.kind, region.side, list(region.box), round(region.score, 3)]
                for region in regions
            ]
            for region in regions:
                job = make_job(
                    region.kind, region.side, entry["clip"], index, json.dumps(list(region.box))
                )
                write_crop(frame, region, root / "crops", job.key)
        except (ValueError, cv2.error) as error:
            print(
                f"detect {entry['clip']} frame {index} failed: {error}; "
                f"recording empty regions",
                file=sys.stderr,
            )
            items[index] = []
    return entry["clip"], items, None


def cmd_detect(args):
    root = Path(args.root)
    full_sample = read_json(root / "sample.json")
    source = full_sample.get("source", "mp4")
    sample = full_sample["clips"]
    video_dir = Path(args.video_root) if args.video_root else None
    if source not in ("phoenix-tar", "phoenix-dir") and video_dir is None:
        raise SystemExit("detect: mp4 source requires --video-root")
    frame_index_dir = {}
    if source == "phoenix-dir":
        frame_index_dir = phoenix_index_frames_dir(
            [
                dict(clip=e["clip"], stratum=e["stratum"], num_frames=e["num_frames"])
                for e in sample
            ],
            Path(full_sample["frames_root"]),
            split=full_sample.get("split", "train"),
        )
    regions_file = root / "regions.json"
    progress_file = root / "detect_progress.json"
    sample_clips = {entry["clip"] for entry in sample}
    regions_by_frame = {}
    if regions_file.exists():
        for key, items in read_json(regions_file).items():
            clip, index = key.split("|")
            if clip in sample_clips:
                regions_by_frame[(clip, int(index))] = items
        done_clips = {clip for (clip, _index) in regions_by_frame}
        if progress_file.exists():
            prior_skipped = set(read_json(progress_file).get("skipped", [])) & sample_clips
            with_regions = {clip for (clip, _index) in regions_by_frame}
            done_clips |= prior_skipped - with_regions
        print(f"resume: {len(done_clips)} clips already detected", flush=True)
    else:
        done_clips = set()
    started = time.time()
    skipped = []
    remaining = [entry for entry in sample if entry["clip"] not in done_clips]
    workers = max(1, args.workers)
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def merge(clip, items, reason):
        nonlocal regions_by_frame, skipped
        if reason is not None:
            print(f"skip {clip}: {reason}", file=sys.stderr)
            skipped.append(clip)
        else:
            for index, frame_items in items.items():
                regions_by_frame[(clip, index)] = frame_items

    try:
        if pool is None:
            results = (_detect_one(e, source, root, video_dir, frame_index_dir) for e in remaining)
        else:
            results = pool.map(
                lambda e: _detect_one(e, source, root, video_dir, frame_index_dir), remaining
            )
        for position, (clip, items, reason) in enumerate(results, 1):
            merge(clip, items, reason)
            done_clips.add(clip)
            if position % 10 == 0 or position == len(remaining):
                atomic_json(
                    progress_file, {"clips": sorted(done_clips), "skipped": sorted(skipped)}
                )
                atomic_json(
                    regions_file,
                    {
                        f"{c}|{i}": v
                        for (c, i), v in sorted(regions_by_frame.items())
                    },
                )
            if position % 20 == 0 or position == len(remaining):
                print(
                    f"detected {position}/{len(remaining)} clips ({time.time() - started:.0f}s)",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()
    plain = {
        key: [Region(kind, side, tuple(box), score) for kind, side, box, score in items]
        for key, items in regions_by_frame.items()
    }
    stats = region_stats(plain)
    gate = detector_gate(stats)
    gate["enough_frames"] = stats["frames"] >= MIN_DETECTOR_FRAMES
    gate["pass"] = gate["pass"] and gate["enough_frames"]
    gate["skipped_clips"] = sorted(skipped)
    atomic_json(
        root / "detector_report.json",
        {"stats": stats, "gate": gate, "elapsed_s": round(time.time() - started, 1)},
    )
    print(json.dumps({"stats": stats, "gate": gate}, indent=1))
    sys.exit(0 if gate["pass"] else 2)


def _jobs_from_regions(root: Path):
    sample = read_json(root / "sample.json")["clips"]
    raw = read_json(root / "regions.json")
    regions = {}
    frames_by_clip = {}
    for key, items in raw.items():
        clip, index = key.split("|")
        index = int(index)
        regions[(clip, index)] = [
            Region(item[0], item[1], tuple(item[2]), item[3]) for item in items
        ]
        frames_by_clip.setdefault(clip, []).append(index)
    clips = [
        {"clip": entry["clip"], "frames": sorted(frames_by_clip.get(entry["clip"], []))}
        for entry in sample
    ]
    return enumerate_jobs(clips, regions)


def _load_journals(root: Path):
    journals = {}
    for path in sorted((root / "jobs").glob("*/*.json")):
        if path.name.startswith(".partial-"):
            continue
        state = json.loads(path.read_text())
        key = path.stem
        meta = json.loads(state["identity"])
        journals[key] = dict(state, kind=meta[0], side=meta[1], clip=meta[2], frame_index=meta[3])
    return journals


def cmd_label(args):
    root = Path(args.root)
    jobs = _jobs_from_regions(root)
    if args.limit:
        jobs = jobs[: args.limit]
    ports = list(args.ports)
    print(f"{len(jobs)} crop jobs across {len(ports)} Ollama ports", flush=True)
    done = failed = 0
    started = time.time()

    def worker(packed):
        slot, job = packed
        return run_job(root, job, root / "crops", args.model, ports[slot % len(ports)])

    with ThreadPoolExecutor(max_workers=len(ports) * args.workers_per_port) as pool:
        for index, state in enumerate(pool.map(worker, enumerate(jobs)), 1):
            if state["status"] == "done":
                done += 1
            else:
                failed += 1
            if index % 25 == 0 or index == len(jobs):
                rate = index / (time.time() - started)
                eta = (len(jobs) - index) / rate / 60 if rate else 0
                print(
                    f"labeled {index}/{len(jobs)} done={done} failed={failed} "
                    f"rate={rate * 60:.1f}/min eta={eta:.0f}min",
                    flush=True,
                )
    print(f"label stage finished: done={done} failed={failed}", flush=True)
    if failed > max(5, len(jobs) // 200):
        raise SystemExit(1)


def cmd_report(args):
    root = Path(args.root)
    journals = _load_journals(root)
    hand = label_rows(journals, "hand")
    mouth = label_rows(journals, "mouth")
    stats = {
        "code_sha256": package_code_sha256(),
        "hand": aggregate_labels(hand),
        "mouth": aggregate_labels(mouth),
        "latency": latency_stats(journals),
        "baselines": {"top1_share": BASELINE_TOP1_SHARE, "distinct": BASELINE_DISTINCT},
        "counts": {
            "jobs": len(journals),
            "done": sum(1 for entry in journals.values() if entry["status"] == "done"),
            "failed": sum(1 for entry in journals.values() if entry["status"] == "failed"),
        },
    }
    detector_file = root / "detector_report.json"
    if detector_file.exists():
        stats["verdict"] = verdict(read_json(detector_file), stats)
    atomic_json(root / "label_report.json", stats)
    print(json.dumps(stats, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="h2s-probe")
    parser.add_argument("--root", default="artifacts/how2sign_probe_v1", type=Path)
    stages = parser.add_subparsers(dest="stage", required=True)

    sample = stages.add_parser("sample")
    sample.add_argument("--csv", required=True, type=Path)
    sample.add_argument("--total", type=int, default=200)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--model", default="qwen3-vl:8b")

    phoenix_sample = stages.add_parser("sample-phoenix")
    phoenix_sample.add_argument("--annotations", required=True, type=Path)
    phoenix_sample.add_argument("--tar", type=Path)
    phoenix_sample.add_argument("--frames-root", type=Path)
    phoenix_sample.add_argument("--split", default="train", choices=("train", "dev", "test"))
    phoenix_sample.add_argument(
        "--include-all-signers",
        action="store_true",
        help="label Signer03/Signer07 clips too (full-dataset runs); default excludes them",
    )
    phoenix_sample.add_argument("--total", type=int, default=200)
    phoenix_sample.add_argument("--seed", type=int, default=0)
    phoenix_sample.add_argument("--model", default="qwen3-vl:8b")

    extract = stages.add_parser("extract")
    extract.add_argument("--tar", required=True, type=Path)

    detect = stages.add_parser("detect")
    detect.add_argument("--video-root", type=Path)
    detect.add_argument("--workers", type=int, default=1)

    label = stages.add_parser("label")
    label.add_argument("--model", default="qwen3-vl:8b")
    label.add_argument("--ports", type=int, nargs="+", default=[11434])
    label.add_argument("--workers-per-port", type=int, default=1)
    label.add_argument("--limit", type=int, default=0)

    stages.add_parser("report")

    args = parser.parse_args(argv)
    {
        "sample": cmd_sample,
        "sample-phoenix": cmd_sample_phoenix,
        "extract": cmd_extract,
        "detect": cmd_detect,
        "label": cmd_label,
        "report": cmd_report,
    }[args.stage](args)


if __name__ == "__main__":
    main()
