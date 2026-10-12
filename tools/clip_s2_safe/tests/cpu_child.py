"""Test-only actual CPU worker harness. No production mode or CUDA discovery."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.data.phoenix14t import Phoenix14T
from tools.benchmark.compact_checkpoint import capture_state, save_snapshot
from tools.clip_s2_inference.tests.helpers import language
from tools.clip_s2_inference.tests.test_evaluation import Visual, fixture
from tools.clip_s2_safe import train
from tools.clip_s2_safe.evaluate import score_dev, validate_report
from tools.clip_s2_safe.state import load_weights, read_binding, restore_training
from tools.clip_s2_safe.tests.helpers import original_fixture
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.storage import atomic_json
from tools.followup.recipe import PromptPool


def model():
    m = nn.Module()
    m.visual_adapter = Visual()
    m.language_model = language()
    m.rng = random.Random(7)
    return m


def lifecycle_fixture(tmp_path):
    c, i, *_ = original_fixture(tmp_path / "original")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _, a, _ = fixture(inputs)
    rows = {
        n: dict(
            fileid=f"train{n}",
            signer="Signer01",
            text=f"german {n}",
            gloss="G",
            en_text="en",
            fr_text="fr",
            es_text="es",
        )
        for n in range(16)
    }
    np.save(inputs / "train.npy", rows)
    for kind, width, length in [("spatial", 2048, 5), ("motion", 1024, 1)]:
        records = list(
            a["datasets"]["dev"].spatial_manifest.records
            if kind == "spatial"
            else a["datasets"]["dev"].motion_manifest.records
        )
        for n in range(16):
            clip = f"train{n}"
            p = a["roots"][kind] / "train" / f"{clip}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, np.ones((length, width), np.float32))
            records.append(
                FeatureRecord(clip, "train", f"train/{clip}.npy", length, width, "float32")
            )
        FeatureManifest(1, "fixture", width, tuple(records)).save(
            a["roots"][kind] / "manifest.json"
        )
    c.update(test_inputs=str(inputs), anchor_step=4, expected_dev_clips=3)
    r = i["recipe"]
    r.update(steps_per_epoch=2, total_steps=12, vt_warmup_steps=4)
    c["recipe"] = r
    i["data"] = dict(bindings=i["data"]["bindings"], ordered_ids=dict(dev=digest(a["ids"]["dev"])))
    pool = PromptPool(list(rows.values()), 0)
    i["decoder"]["exemplar_id"] = pool.eval_id
    c["anchor"] = str(Path(c["run_root"]) / "legacy-full-epoch2.pt")
    m = model()
    opt = torch.optim.AdamW(m.parameters(), lr=0.001)
    sampler = OfficialBatchSampler(tuple(r["fileid"] for r in rows.values()), seed=0, batch_size=8)
    for _ in range(4):
        sampler.next_ids()
    save_snapshot(
        Path(c["anchor"]), capture_state(m, opt, i, 4, sampler.state_dict(), base_hash="frozen")
    )
    c["anchor_sha256"] = sha256_file(Path(c["anchor"]))
    c["anchor_receipt_sha256"] = sha256_file(Path(c["anchor"]).with_suffix(".json"))
    budget = Path(c["run_root"]) / "budget.json"
    original = json.loads(budget.read_text())
    original.update(identity=i, completed_updates=4)
    atomic_json(budget, original)
    c["initial_budget_sha256"] = sha256_file(budget)
    atomic_json(
        Path(c["full_diagnostic"]),
        dict(
            status="completed",
            processed_clips=3,
            checkpoint_sha256=c["anchor_sha256"],
            checkpoint_step=4,
            dev_ids_sha256=i["data"]["ordered_ids"]["dev"],
            batch_size=1,
            use_cache=True,
            beam_size=5,
            max_length=64,
            optimizer_loaded=False,
            diagnostic_only=True,
        ),
    )
    c["full_diagnostic_sha256"] = sha256_file(Path(c["full_diagnostic"]))
    return c


def load_inputs(c):
    root = Path(c["test_inputs"])
    roots = dict(spatial=root / "spatial", motion=root / "motion")
    manifests = {k: FeatureManifest.load(p / "manifest.json") for k, p in roots.items()}
    datasets = {}
    ids = {}
    lookup = {}
    hashes = {k: {} for k in roots}
    for split in ["train", "dev"]:
        ds = Phoenix14T(
            root / f"{split}.npy",
            split,
            roots["spatial"],
            roots["motion"],
            manifests["spatial"],
            manifests["motion"],
        )
        datasets[split] = ds
        ids[split] = tuple(r["fileid"] for r in ds.records)
        lookup[split] = {v: n for n, v in enumerate(ids[split])}
        for kind, p in roots.items():
            for clip in ids[split]:
                hashes[kind][split, clip] = sha256_file(p / split / f"{clip}.npy")
    identity = json.loads(Path(c["anchor"]).with_suffix(".json").read_text())["identity"]["data"]
    return dict(
        roots=roots,
        datasets=datasets,
        ids=ids,
        lookup=lookup,
        feature_hashes=hashes,
        identity=identity,
        bindings=identity["bindings"],
    )


def main():
    assert not torch.cuda.is_initialized()
    torch.cuda.is_available = lambda: False
    stage = sys.argv[1]
    c = json.loads(Path(sys.argv[2]).read_text())
    j = json.loads(Path(sys.argv[3]).read_text())
    b = read_binding(c)
    a = load_inputs(c)
    m = model()
    state = load_weights(Path(j["checkpoint"]), m, b, base_hash="frozen")
    pool = PromptPool(a["datasets"]["train"].records, 0)
    if stage == "train":
        opt = torch.optim.AdamW(m.parameters(), lr=0.001)
        sampler = OfficialBatchSampler(a["ids"]["train"], seed=0, batch_size=8)
        restore_training(state, m, opt, sampler, device=torch.device("cpu"))

        def update(model, opt, batch, k, recipe, pool):
            pool.training(batch.clip_ids, model.rng)
            opt.zero_grad()
            loss = (
                model.language_model.model.encoder.embed_tokens.weight.square().mean()
                * torch.rand(())
            )
            loss.backward()
            opt.step()
            return dict(translation_loss=float(loss.detach()))

        train.update = update
        train.check_guard = lambda *a: None
        result = train.train_chunk(
            m,
            opt,
            sampler,
            a,
            pool,
            b,
            state["safe_guard"],
            j,
            device=torch.device("cpu"),
            on_progress=lambda e: atomic_json(Path(j["progress_path"]), e),
        )
    else:
        report = score_dev(m, a, pool, j, b, on_event=lambda e: None)
        validate_report(report, j, b, a)
        path = Path(c["runtime_root"]) / "dev" / f"epoch-{j['epoch']}.json"
        atomic_json(path, report)
        result = dict(
            status="complete",
            stage="dev",
            mode=j["mode"],
            completed_updates=j["target_step"],
            durable_updates=j["target_step"],
            completed_clips=len(report["items"]),
            report=str(path),
            report_sha256=sha256_file(path),
            diagnostic_only=False,
        )
    atomic_json(Path(j["result_path"]), result)
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
