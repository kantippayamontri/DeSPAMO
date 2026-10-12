import json
import random
from importlib.metadata import version
from pathlib import Path

import torch

from tools.benchmark.compact_checkpoint import capture_state, save_snapshot
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.storage import atomic_json


def tiny():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = torch.nn.Linear(2, 1)
    model.rng = random.Random(17)
    model.register_buffer("persistent", torch.tensor([3.0]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    sampler = OfficialBatchSampler(tuple(f"train{i}" for i in range(8)), seed=0, batch_size=8)
    sampler.next_ids()
    sampler.next_ids()
    return model, optimizer, sampler


def original_fixture(tmp_path):
    from tools.clip_s2_safe import policy

    tmp_path.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(policy.__file__).with_name("config.json").read_text())
    study = tmp_path / "study"
    run = study / "runs/original"
    config.update(
        study_root=str(study),
        run_root=str(run),
        runtime_root=str(run / "safe-runtime-v1"),
        anchor=str(run / "resume-epoch-2.pt"),
        study_config=str(study / "study-inputs.json"),
        archive_root=str(tmp_path / "archive/original/safe-runtime-v1"),
        cache=str(tmp_path / "cache"),
        full_diagnostic=str(study / "diagnostic.json"),
        anchor_step=2,
        expected_source_files=1,
        expected_dev_clips=2,
    )
    source = tmp_path / "old-source.py"
    source.write_text("# immutable original kernel\n")
    data = tmp_path / "data.json"
    data.write_text("{}")
    atomic_json(Path(config["study_config"]), {})
    identity = dict(
        policy="official-frozen-clip-s2-reference-spamo200-v1",
        seed=0,
        arm="reference",
        code_hashes={str(source): sha256_file(source)},
        base_files={},
        config_sha256=sha256_file(Path(config["study_config"])),
        frozen_fingerprint="frozen",
        runtime={n: version(n) for n in ["torch", "transformers", "peft"]},
        data=dict(
            bindings={str(data): sha256_file(data)}, ordered_ids=dict(dev=digest(("dev0", "dev1")))
        ),
        decoder=dict(
            exemplar_id="train0",
            beam=5,
            max_length=64,
            in_context=True,
            num_in_context=3,
            prompt_max_tokens=1024,
        ),
        recipe=dict(
            batch_size=8,
            accumulation=1,
            steps_per_epoch=1,
            total_steps=6,
            epochs=6,
            dev_epochs=[2, 4, 6],
            vt_warmup_steps=4,
            lr_warmup_steps=10000,
            vt_pooling="legacy_mean",
            min_epochs=4,
            patience=10,
            min_delta=0.10,
        ),
    )
    config["recipe"] = identity["recipe"]
    model, optimizer, sampler = tiny()
    state = capture_state(model, optimizer, identity, 2, sampler.state_dict(), base_hash="frozen")
    save_snapshot(Path(config["anchor"]), state)
    atomic_json(
        run / "budget.json",
        dict(
            identity=identity,
            status="interrupted",
            phase="finalized",
            completed_updates=2,
            charged_seconds=497.5533489369991,
            owner=dict(pid=1, boot_id="old", start_ticks="0", command=["old"]),
            duration_limit_seconds=None,
        ),
    )
    atomic_json(
        study / "manual-stop.json",
        dict(
            status="stopped_by_user",
            automatic_recovery_disabled=True,
            gpu_validation_approved=False,
        ),
    )
    config.update(
        anchor_sha256=sha256_file(Path(config["anchor"])),
        anchor_receipt_sha256=sha256_file(Path(config["anchor"]).with_suffix(".json")),
        initial_budget_sha256=sha256_file(run / "budget.json"),
        initial_hold_sha256=sha256_file(study / "manual-stop.json"),
        frozen_fingerprint="frozen",
    )
    atomic_json(
        Path(config["full_diagnostic"]),
        dict(
            status="completed",
            processed_clips=2,
            checkpoint_sha256=config["anchor_sha256"],
            checkpoint_step=2,
            dev_ids_sha256=identity["data"]["ordered_ids"]["dev"],
            batch_size=1,
            use_cache=True,
            beam_size=5,
            max_length=64,
            optimizer_loaded=False,
            diagnostic_only=True,
        ),
    )
    config["full_diagnostic_sha256"] = sha256_file(Path(config["full_diagnostic"]))
    return config, identity, model, optimizer, sampler
