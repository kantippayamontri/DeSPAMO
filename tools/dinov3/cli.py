import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from tools.dinov3.encoder import load_model
from tools.dinov3.frames import SPLITS, annotations
from tools.dinov3.identity import MODEL, identity, read_token, resolve_sha
from tools.dinov3.pipeline import run
from tools.dinov3.storage import atomic_array, verify_location, writer


def _preflight_inputs(
    annotation_root: Path, clip_manifest: Path, only: tuple[str, str] | None
) -> Path:
    frame_value = os.environ.get("PHOENIX14T_FRAME_ROOT")
    if not frame_value:
        raise ValueError("PHOENIX14T_FRAME_ROOT required for extraction")
    frame_root = Path(frame_value)
    if not frame_root.is_dir():
        raise ValueError("PHOENIX14T_FRAME_ROOT must be an existing directory")
    if not annotation_root.is_dir():
        raise ValueError("annotation root must be an existing directory")
    for filename in SPLITS.values():
        if not (annotation_root / filename).is_file():
            raise ValueError(f"missing annotation file: {filename}")
    if not clip_manifest.is_file():
        raise ValueError("CLIP manifest must be an existing file")
    clips = annotations(annotation_root, clip_manifest)
    if only is not None:
        split, clip_id = only
        if split not in clips or clip_id not in clips[split]:
            raise ValueError(
                f"selected clip missing from annotations/CLIP manifest: {split}/{clip_id}"
            )
        if not (frame_root / split / clip_id).is_dir():
            raise ValueError(f"{split}/{clip_id}: missing source directory")
    return frame_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output-base", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--clip-manifest", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-access", action="store_true")
    mode.add_argument("--one-clip", nargs=2, metavar=("SPLIT", "CLIP_ID"))
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--confirm-full-extraction", action="store_true")
    args = parser.parse_args()
    if args.all and not args.confirm_full_extraction:
        parser.error("--all requires --confirm-full-extraction and explicit user authorization")
    if not args.check_access and not all(
        (args.cache, args.output_base, args.annotation_root, args.clip_manifest)
    ):
        parser.error("extraction requires cache, output-base, annotation-root, clip-manifest")
    if args.check_access and args.cache is None:
        parser.error("access check requires --cache on external filesystem")

    checkout = Path(__file__).resolve().parents[2]
    verify_location(args.cache, checkout)
    if args.output_base is not None:
        verify_location(args.output_base, checkout)
    only = tuple(args.one_clip) if args.one_clip else None
    frame_root = (
        None
        if args.check_access
        else _preflight_inputs(args.annotation_root, args.clip_manifest, only)
    )
    token = read_token(args.env_file)
    sha = resolve_sha(token, args.revision)
    try:
        hf_hub_download(
            repo_id=MODEL, filename="config.json", revision=sha, token=token, cache_dir=args.cache
        )
    except Exception:
        raise RuntimeError(
            "DINOv3 gated config inaccessible; check model license and HF_TOKEN"
        ) from None
    if args.check_access:
        print(f"DINOv3 config accessible at revision {sha}")
        return

    key, metadata = identity(sha, Path(__file__).resolve().parent / "uv.lock")
    target = args.output_base / key
    if target.is_symlink() or target.resolve(strict=False).parent != args.output_base.resolve(
        strict=False
    ):
        raise ValueError(
            "encoder key root must be an immediate child of output base, not a symlink"
        )
    if not (target / "complete").exists():
        with writer(target):
            probe = target / ".write-probe.npy"
            try:
                atomic_array(probe, np.zeros((5, 2048), dtype=np.float32))
            finally:
                probe.unlink(missing_ok=True)
    if (target / "complete").exists():
        model, device = None, "cpu"
    else:
        try:
            available = torch.cuda.is_available()
        except Exception:
            raise RuntimeError("DINOv3 CUDA initialization failed; check driver/runtime") from None
        if not available:
            raise RuntimeError("live DINOv3 extraction requires CUDA")
        try:
            torch.empty((1,), device="cuda")
        except Exception:
            raise RuntimeError("DINOv3 CUDA initialization failed; check driver/runtime") from None
        model, device = load_model(sha, token, args.cache, "cuda"), "cuda"

    stats = run(
        frame_root,
        args.annotation_root,
        args.clip_manifest,
        target,
        key,
        metadata,
        model,
        device,
        only=only,
    )
    print(json.dumps(dict(encoder_key=key, model_sha=sha, **stats), sort_keys=True))


if __name__ == "__main__":
    main()
