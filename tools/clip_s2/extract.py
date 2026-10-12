"""Owned FP32 CLIP/S2 extraction, parity/profile evidence and immutable packages."""

import argparse
import ast
import copy
import json
import math
import signal
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.backends import cuda, cudnn
from transformers import AutoTokenizer

from tools.benchmark.feature_package import (
    finish_package,
    other_gpu_pids,
    source_rows,
    verify_clip,
    verify_package,
    write_clip,
)
from tools.clip_s2.encoder import REVISION, extract_frames, load_encoder, s2_cls
from tools.clip_s2.inventory import inventory
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_runtime import OfficialLedger, require_official_storage
from tools.dinov3.storage import atomic_json, writer

BASE = Path("/home/kan/datasets/despamo/official-phoenix14t-clip-s2-features-v1")
PROJECT = Path("/home/kan/Research/DeSpaMo")
FIELDS = {
    "annotation_root",
    "source_row_map",
    "frames_root",
    "cache",
    "processor_file",
    "output_base",
    "accepted_profile",
    "revision",
}
REFERENCE = Path("/home/kan/Research/SpaMo/utils/s2wrapper.py")


def reference_rearrange(x, pattern, **kwargs):
    """Two exact tensor permutations used by parity-only SpaMo patch-grid code."""
    if pattern == "b (h w) c -> b c h w":
        b, _, c = x.shape
        return x.reshape(b, kwargs["h"], -1, c).permute(0, 3, 1, 2)
    if pattern == "b c h w -> b (h w) c":
        b, c, _, _ = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, -1, c)
    raise ValueError("unsupported reference tensor permutation")


def reference_forward():
    # Execute original function bodies, excluding optional einops import/sys.path side effects.
    tree = ast.parse(REFERENCE.read_text())
    definitions = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]
    )
    namespace: dict[str, Any] = dict(
        torch=torch, F=torch.nn.functional, math=math, rearrange=reference_rearrange
    )
    exec(compile(definitions, str(REFERENCE), "exec"), namespace)
    return namespace["forward"]


def source_hashes() -> dict:
    here = Path(__file__).resolve().parent
    paths = [here / n for n in ("__init__.py", "encoder.py", "inventory.py", "extract.py")]
    paths += [here.parent / "benchmark/feature_package.py"]
    paths += [REFERENCE]
    paths += [
        here.parent / "dinov3" / n for n in ("identity.py", "storage.py", "official_runtime.py")
    ]
    paths += [PROJECT / "tools/motion" / n for n in ("pyproject.toml", "uv.lock")]
    return {str(p): sha256_file(p) for p in paths}


def load_inputs(path: Path) -> dict:
    config = json.loads(path.read_text())
    if (
        set(config) != FIELDS
        or config["revision"] != REVISION
        or config["output_base"] != str(BASE)
    ):
        raise ValueError("exact pinned extraction input fields/paths required")
    bindings = {str(path): sha256_file(path)}
    for key in ("source_row_map", "processor_file"):
        p = Path(config[key])
        bindings[str(p)] = sha256_file(p)
    for split in ("train", "dev", "test"):
        p = Path(config["annotation_root"]) / f"{split}_info_ml.npy"
        bindings[str(p)] = sha256_file(p)
    return dict(config=config, bindings=bindings, source=source_hashes())


def make_identity(inputs: dict, encoder_binding: dict, profile: dict) -> dict:
    # Profile-path binding is stored separately; only profile decisions affect feature math.
    return dict(
        policy="official-frozen-clip-s2-spatial-v1",
        kind="spatial",
        width=2048,
        inputs=copy.deepcopy(inputs),
        encoder=copy.deepcopy(encoder_binding),
        frame_batch_size=profile["frame_batch_size"],
        tile_batch_size=profile["tile_batch_size"],
        precision="float32_no_autocast_no_tf32",
        runtime={n: version(n) for n in ("torch", "transformers", "numpy", "pillow")},
    )


def choose_smoke(inv: dict, cache: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-xl",
        cache_dir=str(cache),
        revision="7d6315df2c2fb742f0f5b556879d730926ca9001",
        local_files_only=True,
    )
    import numpy as np

    config = Path("/home/kan/Research/SpaMo/preprocess/Phoenix14T")
    raw = np.load(config / "train_info_ml.npy", allow_pickle=True).item()
    records = [raw[k] for k in sorted(k for k in raw if type(k) is int)]
    ids = [r["fileid"] for r in records[:8]]
    ids += sorted(inv["train"], key=lambda i: inv["train"][i]["frame_count"], reverse=True)[:8]
    ids += [
        r["fileid"]
        for r in sorted(records, key=lambda r: len(tokenizer.encode(r["text"])), reverse=True)[:8]
    ]
    dev = sorted(inv["dev"], key=lambda i: inv["dev"][i]["frame_count"], reverse=True)[:4]
    return dict(
        train={i: inv["train"][i] for i in dict.fromkeys(ids)},
        dev={i: inv["dev"][i] for i in dev},
        test={i: inv["test"][i] for i in list(inv["test"])[:1]},
    )


def publish_existing(root: Path, identity: dict, chosen: dict, mode: str) -> dict:
    result = verify_package(root, identity, chosen, 2048)
    result.update(root=str(root), status="complete", purpose=identity.get("purpose", "official"))
    budget = root / "budget.json"
    if budget.exists() and json.loads(budget.read_text())["status"] != "complete":
        ledger = OfficialLedger(root, identity)
        ledger.charge(result["clips"], "complete_publication_replay", replay=True)
        ledger.close("complete")
    atomic_json(BASE / ("smoke-result.json" if mode == "smoke" else "package-result.json"), result)
    return result


def run_extraction(mode: str, config_path: Path) -> dict:
    inputs = load_inputs(config_path)
    c = inputs["config"]
    inv = inventory(Path(c["annotation_root"]), Path(c["source_row_map"]), Path(c["frames_root"]))
    frames = sum(r["frame_count"] for clips in inv.values() for r in clips.values())
    require_official_storage(BASE, frames * 2048 * 4 + 9 * 1024**3)
    if mode == "preflight":
        return dict(status="ready", frames=frames, clips=8257)
    if other_gpu_pids():
        raise RuntimeError("GPU already owned")
    cuda.matmul.allow_tf32 = False
    cudnn.allow_tf32 = False
    model, processor, binding = load_encoder(Path(c["cache"]), Path(c["processor_file"]), "cuda")
    if mode == "profile":
        with writer(BASE / "profiles"):
            longest = max(inv["train"].values(), key=lambda r: r["frame_count"])
            paths = [Path(p) for p in longest["paths"][:33]]
            pixels = processor(
                [Image.open(p).convert("RGB") for p in paths[:2]], return_tensors="pt"
            ).pixel_values.to("cuda")
            original_forward = reference_forward()
            with torch.inference_mode():
                original = original_forward(
                    lambda x: model(x, output_hidden_states=True).hidden_states[-1],
                    pixels,
                    scales=[1, 2],
                    num_prefix_token=1,
                )[:, 0]
                ours = s2_cls(pixels, model, 3)
                torch.testing.assert_close(ours, original, rtol=1e-4, atol=1e-5)
                raw = model(pixels).last_hidden_state[:, 0]
                torch.testing.assert_close(ours[:, :1024], raw, rtol=1e-4, atol=1e-5)
                if torch.allclose(raw, model(pixels).pooler_output):
                    raise ValueError("pooler/hidden CLS parity probe is not discriminating")
            attempts = []
            accepted = None
            for batch in (32, 16, 8):
                torch.cuda.reset_peak_memory_stats()
                begin = time.monotonic()
                try:
                    values = extract_frames(paths, model, processor, batch, batch)
                    torch.cuda.synchronize()
                    row: dict = dict(
                        frame_batch_size=batch,
                        tile_batch_size=batch,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        gpu_free_bytes=torch.cuda.mem_get_info()[0],
                        frames=len(values),
                        seconds=time.monotonic() - begin,
                        status="complete",
                    )
                    attempts.append(row)
                    if (
                        row["peak_reserved_bytes"] <= 21 * 1024**3
                        and row["gpu_free_bytes"] >= 2 * 1024**3
                    ):
                        accepted = row
                        break
                except RuntimeError as error:
                    if "out of memory" not in str(error).lower():
                        raise
                    attempts.append(dict(frame_batch_size=batch, status="oom"))
                    torch.cuda.empty_cache()
            if accepted is None:
                raise ValueError("no profiled frame batch satisfies GPU margin")
            report = dict(
                status="complete",
                accepted=accepted,
                attempts=attempts,
                inputs=inputs,
                encoder=binding,
                reference_sha256=sha256_file(REFERENCE),
                parity=True,
                tolerance=dict(rtol=1e-4, atol=1e-5),
            )
            atomic_json(BASE / "profiles/accepted.json", report)
            return report
    profile_path = Path(c["accepted_profile"])
    profile = json.loads(profile_path.read_text())
    if (
        profile["status"] != "complete"
        or not profile["parity"]
        or profile["encoder"] != binding
        or profile["inputs"]["source"] != source_hashes()
    ):
        raise ValueError("accepted extraction profile/source/encoder changed")
    identity = make_identity(inputs, binding, profile["accepted"])
    identity["profile_sha256"] = sha256_file(profile_path)
    if mode == "smoke":
        identity["purpose"] = "bounded_profile_smoke"
    key = digest(identity)
    root = BASE / (f"spatial-smoke-{key}" if mode == "smoke" else f"spatial-{key}")
    chosen = choose_smoke(inv, Path(c["cache"])) if mode == "smoke" else inv
    with writer(root):
        if (root / "complete").exists():
            return publish_existing(root, identity, chosen, mode)
        atomic_json(root / "identity.json", identity)
        ledger = OfficialLedger(root, identity)
        ledger.charge(0, "verified_clip_replay", replay=True)
        try:
            done = 0
            completed_frames = 0
            for split, clips in chosen.items():
                for clip, row in clips.items():
                    paths = [Path(p) for p in row["paths"]]
                    if (root / "receipts" / split / f"{clip}.json").exists():
                        receipt = verify_clip(root, split, clip, paths, identity)
                    else:
                        observed = source_rows(paths)
                        values = extract_frames(
                            paths,
                            model,
                            processor,
                            profile["accepted"]["frame_batch_size"],
                            profile["accepted"]["tile_batch_size"],
                        )
                        receipt = write_clip(
                            root,
                            split,
                            clip,
                            paths,
                            values,
                            identity,
                            dict(observed_source_rows=observed),
                        )
                    if receipt["source_hash"] != row["expected_source_hash"]:
                        raise ValueError("CLIP/DINO original frame source identity mismatch")
                    done += 1
                    completed_frames += row["frame_count"]
                    if done % 25 == 0 or mode == "smoke":
                        ledger.charge(done, "extract")
                        print(
                            json.dumps(dict(split=split, clip=clip, clips=done, total=8257)),
                            flush=True,
                        )
                    if done % 100 == 0:
                        if source_hashes() != inputs["source"] or any(
                            sha256_file(Path(p)) != h for p, h in inputs["bindings"].items()
                        ):
                            raise ValueError("extraction source/input drift")
                        require_official_storage(
                            BASE, max(0, frames - completed_frames) * 2048 * 4 + 9 * 1024**3
                        )
            result = finish_package(root, identity, chosen, 2048)
            ledger.close("complete")
            result.update(
                root=str(root), status="complete", purpose=identity.get("purpose", "official")
            )
            atomic_json(
                BASE / ("smoke-result.json" if mode == "smoke" else "package-result.json"), result
            )
            return result
        except BaseException:
            ledger.close("interrupted")
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=("preflight", "profile", "smoke", "run", "verify"), required=True
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--authorize-extraction", action="store_true")
    a = p.parse_args()
    if a.mode != "preflight" and not a.authorize_extraction:
        raise ValueError("explicit extraction authorization required")

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal{signum}")

    signal.signal(signal.SIGTERM, stop)
    if a.mode == "verify":
        result = json.loads((BASE / "package-result.json").read_text())
        root = Path(result["root"])
        identity = json.loads((root / "identity.json").read_text())
        c = load_inputs(a.config)["config"]
        inv = inventory(
            Path(c["annotation_root"]), Path(c["source_row_map"]), Path(c["frames_root"])
        )
        print(json.dumps(verify_package(root, identity, inv, 2048)), flush=True)
    else:
        print(json.dumps(run_extraction(a.mode, a.config)), flush=True)


if __name__ == "__main__":
    main()
