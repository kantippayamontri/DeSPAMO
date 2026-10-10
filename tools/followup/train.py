"""Owned max200epoch training with compact resumes and report-derived early stopping."""

import argparse
import gc
import json
import signal
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch

from tools.benchmark.compact_checkpoint import (
    capture_rng,
    capture_state,
    common_fingerprint,
    load_snapshot,
    restore_rng,
    save_snapshot,
)
from tools.benchmark.feature_package import other_gpu_pids
from tools.benchmark.official_translation import (
    CACHE,
    model_files,
    prune_resumes,
    reconcile_selection,
)
from tools.benchmark.official_translation import PROJECT as PROJECT
from tools.benchmark.official_translation import (
    code_hashes as old_code_hashes,
)
from tools.benchmark.translation_data import admit_data, batch_for, eval_batches
from tools.benchmark.translation_model import FLAN_REVISION, cuda_batch, optimizer_for
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_archive import _archive_json, _copy_file, require_archive_storage
from tools.dinov3.official_runtime import OfficialLedger, require_official_storage
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.storage import atomic_json, writer
from tools.followup.evaluation import score_batches
from tools.followup.model import build_model, update, validate_prompt_lengths
from tools.followup.recipe import PromptPool, normalized, schedule, stop_state

BASE = Path("/home/kan/datasets/despamo/official-phoenix14t-translation-spamo200-v1")
ARCHIVE = Path("/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-spamo200-v1")


def code_hashes() -> dict:
    result = old_code_hashes()
    here = Path(__file__).resolve().parent
    extra = list(here.glob("*.py")) + [
        here.parent / "benchmark/feature_package.py",
        here.parent / "dinov3/identity.py",
        here.parent / "dinov3/official_archive.py",
    ]
    result.update({str(p): sha256_file(p) for p in extra})
    return result


def candidate_to_archive(root: Path, state: dict, epoch: int) -> Path:
    dest, scratch = (
        ARCHIVE / root.name / f"model-epoch-{epoch}.pt",
        root / f"candidate-epoch-{epoch}.pt",
    )
    if dest.exists():
        if not dest.with_suffix(".json").exists():
            native = json.loads(scratch.with_suffix(".json").read_text())
            if (
                native["identity"] != state["identity"]
                or native["sha256"] != sha256_file(scratch)
                or native["sha256"] != sha256_file(dest)
            ):
                raise ValueError("unreceipted archive differs from owned native candidate")
            _archive_json(dest.with_suffix(".json"), native)
        receipt = json.loads(dest.with_suffix(".json").read_text())
        if receipt["identity"] != state["identity"] or receipt["sha256"] != sha256_file(dest):
            raise ValueError("archived candidate bytes/identity changed")
        return dest
    require_archive_storage(ARCHIVE, 512 * 1024**2)
    if scratch.exists():
        prior = torch.load(scratch, map_location="cpu", weights_only=False)
        if (
            prior["identity"] != state["identity"]
            or prior["step"] != state["step"]
            or any(not torch.equal(prior["weights"][n], v) for n, v in state["weights"].items())
        ):
            raise ValueError("native candidate differs from durable replay")
    else:
        save_snapshot(scratch, state)
    receipt = json.loads(scratch.with_suffix(".json").read_text())
    _copy_file(scratch, dest)
    if receipt["sha256"] != sha256_file(dest):
        raise ValueError("candidate archive verification failed")
    _archive_json(dest.with_suffix(".json"), receipt)
    scratch.unlink()
    scratch.with_suffix(".json").unlink()
    return dest


def decision(root: Path, identity: dict, recipe: dict) -> dict:
    reports = [json.loads(p.read_text()) for p in root.glob("dev-epoch-*.json")]
    if any(r.get("run_identity_hash") != digest(identity) for r in reports):
        raise ValueError("dev report identity changed")
    state = stop_state(reports, recipe)
    atomic_json(root / "stopping.json", state)
    return state


def publish_completion(root: Path, ledger, result: dict) -> None:
    atomic_json(root / "complete.json", result)
    ledger.close("complete")


def score_dev(model, optimizer, admission, pool, identity, root, epoch, base_hash, sampler, recipe):
    target = root / f"dev-epoch-{epoch}.json"
    if target.exists():
        reconcile_selection(root, identity)
        return json.loads(target.read_text())
    state = capture_state(
        model,
        optimizer,
        identity,
        epoch * recipe["steps_per_epoch"],
        sampler.state_dict(),
        base_hash=base_hash,
        full=False,
    )
    candidate = candidate_to_archive(root, state, epoch)
    modes, rng, prompt_rng = (
        [(m, m.training) for m in model.modules()],
        capture_rng(),
        model.rng.getstate(),
    )
    try:
        report = score_batches(
            model,
            eval_batches(admission, "dev", 4),
            admission["ids"]["dev"],
            pool,
            "dev",
            identity["data"]["ordered_ids"]["dev"],
        )
    finally:
        for m, flag in modes:
            m.training = flag
        restore_rng(rng)
        model.rng.setstate(prompt_rng)
    report.update(
        epoch=epoch,
        checkpoint_step=epoch * recipe["steps_per_epoch"],
        checkpoint_path=str(candidate),
        checkpoint_hash=sha256_file(candidate),
        run_identity_hash=digest(identity),
    )
    atomic_json(target, report)
    best = reconcile_selection(root, identity)
    print(
        json.dumps(
            dict(dev_epoch=epoch, bleu4=report["metrics"]["bleu4"], selected_epoch=best["epoch"])
        ),
        flush=True,
    )
    return report


def profile(admission: dict, batch_size: int, started: float) -> dict:
    recipe, source = schedule(batch_size), code_hashes()
    pool = PromptPool(admission["datasets"]["train"].records, 0)
    model, base_hash = build_model(0, "aligned_learned", CACHE)
    lengths = validate_prompt_lengths(model, pool)
    optimizer = optimizer_for(model)
    model.to("cuda").train()
    ids = admission["ids"]["train"]
    long_visual = sorted(
        ids,
        key=lambda i: admission["datasets"]["train"].spatial_manifest.require("train", i).length,
        reverse=True,
    )
    long_text = sorted(
        ids,
        key=lambda i: len(model.language_model.tokenizer.encode(pool.by_id[i]["text"])),
        reverse=True,
    )
    root = BASE / "profiles" / f"batch{batch_size}-{digest(source)[:12]}"
    with writer(root):
        identity = dict(
            purpose="followup-profile",
            recipe=recipe,
            source=source,
            data=admission["identity"],
            frozen_fingerprint=base_hash,
        )
        results = {}
        for label, chosen in (
            ("ordinary", ids[:batch_size]),
            ("long", long_visual[:batch_size]),
            ("long_target", long_text[:batch_size]),
        ):
            batch = cuda_batch(batch_for(admission, "train", tuple(chosen)))
            refs = {normalized(t) for t in batch.texts}
            worst = max(
                (r for r in pool.rows if normalized(r["text"]) not in refs),
                key=lambda r: len(model.language_model.tokenizer.encode(pool.render(r))),
            )
            prompts = [pool.render(worst)] * batch_size
            torch.cuda.reset_peak_memory_stats()
            durations = []
            values: dict = {}
            for j in range(4):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                values = update(
                    model,
                    optimizer,
                    batch,
                    10000 + j,
                    recipe,
                    pool,
                    force_joint=True,
                    prompt_override=prompts,
                )
                torch.cuda.synchronize()
                if j:
                    durations.append(time.perf_counter() - begin)
            count = sum(
                bool(optimizer.state.get(p)) for g in optimizer.param_groups for p in g["params"]
            )
            expected = sum(len(g["params"]) for g in optimizer.param_groups)
            if count != expected:
                raise ValueError("profile AdamW states not fully initialized")
            results[label] = dict(
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                seconds=durations,
                clips_per_second=batch_size / statistics.mean(durations),
                optimizer_initialized=count,
                expected_optimizer_parameters=expected,
                exemplar_id=worst["fileid"],
                loss=values,
            )
        model.eval()
        dev_ids = sorted(
            admission["ids"]["dev"],
            key=lambda i: admission["datasets"]["dev"].spatial_manifest.require("dev", i).length,
            reverse=True,
        )
        small = cuda_batch(batch_for(admission, "dev", tuple(dev_ids[:4])))

        def predict():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                visual, mask = model.visual_adapter(
                    small.spatial, small.spatial_mask, small.motion, small.motion_mask
                )
                return model.language_model.generate_text(
                    visual, mask, pool.inference(small.clip_ids)[0], "deterministic", 5, 64
                )

        expected_text = predict()
        state = capture_state(
            model, optimizer, identity, 12, {"profile": True}, base_hash=base_hash
        )
        path = root / "reload.pt"
        receipt = save_snapshot(path, state)
        del state, optimizer, model, batch
        gc.collect()
        torch.cuda.empty_cache()
        model, fresh_hash = build_model(0, "aligned_learned", CACHE)
        optimizer = optimizer_for(model)
        model.to("cuda").eval()
        if fresh_hash != base_hash:
            raise ValueError("fresh base fingerprint changed")
        load_snapshot(path, model, optimizer, identity, base_hash=base_hash)
        if predict() != expected_text:
            raise ValueError("fresh-model compact generation reload differs")
        result: dict = dict(
            status="complete",
            identity=identity,
            results=results,
            batch_size=batch_size,
            generation_equal=True,
            fresh_base_reload=True,
            snapshot_bytes=receipt["bytes"],
            gpu_free_bytes=torch.cuda.mem_get_info()[0],
            prompt_lengths=lengths,
            charged_seconds=time.monotonic() - started,
        )
        atomic_json(root / "profile.json", result)
        if (
            result["gpu_free_bytes"] < 2 * 1024**3
            or max(r["peak_reserved_bytes"] for r in results.values()) > 21 * 1024**3
        ):
            raise ValueError("batch8 ICL profile lacks required GPU margin")
        atomic_json(
            BASE / "profiles/accepted.json",
            dict(path=str(root / "profile.json"), sha256=sha256_file(root / "profile.json")),
        )
        print(json.dumps(result), flush=True)
        return result


def run(seed: int, arm: str, batch_size: int, mode: str) -> dict:
    started = time.monotonic()
    admission = admit_data()
    if other_gpu_pids():
        raise RuntimeError("GPU already owned")
    require_official_storage(BASE, 6 * 1024**3)
    require_archive_storage(ARCHIVE, 60 * 1024**3)
    if mode == "profile":
        return profile(admission, batch_size, started)
    recipe = schedule(batch_size)
    pool = PromptPool(admission["datasets"]["train"].records, seed)
    for split in ("dev", "test"):
        pool.guard_references(tuple(r["text"] for r in admission["datasets"][split].records))
    model, base_hash = build_model(seed, arm, CACHE)
    prompt_lengths = validate_prompt_lengths(model, pool)
    common = common_fingerprint(model)
    init = BASE / "common-initialization" / f"seed{seed}.json"
    with writer(init.parent):
        expected = dict(seed=seed, common_mutable_fingerprint=common, frozen_fingerprint=base_hash)
        if init.exists() and json.loads(init.read_text()) != expected:
            raise ValueError("common initialization changed")
        if not init.exists():
            atomic_json(init, expected)
    identity = dict(
        policy="official-spamo-style-200-v1",
        seed=seed,
        arm=arm,
        recipe=recipe,
        decoder=dict(
            beam=5,
            max_length=64,
            in_context=True,
            num_in_context=3,
            exemplar_id=pool.eval_id,
            prompt_max_tokens=1024,
        ),
        data=admission["identity"],
        code_hashes=code_hashes(),
        frozen_fingerprint=base_hash,
        flan_revision=FLAN_REVISION,
        common_initialization=common,
        base_files=model_files(),
        runtime={n: version(n) for n in ("torch", "transformers", "peft")},
        prompt_lengths=prompt_lengths,
    )
    root = BASE / "runs" / f"translation-{arm}-seed{seed}-{digest(identity)}"
    optimizer = optimizer_for(model)
    model.to("cuda").train()
    with writer(root):
        if (root / "complete.json").exists():
            complete = json.loads((root / "complete.json").read_text())
            if complete["selected"] != json.loads((root / "selected-checkpoint.json").read_text()):
                raise ValueError("completed selection changed")
            budget = json.loads((root / "budget.json").read_text())
            if budget["status"] != "complete":
                final_ledger = OfficialLedger(root, identity)
                if final_ledger.completed != complete["stopped_updates"]:
                    raise ValueError("completed update accounting changed")
                final_ledger.close("complete")
            return complete
        atomic_json(root / "identity.json", identity)
        profile_charge = 0.0
        if seed == 0 and arm == "reference":
            profile_charge = sum(
                json.loads(p.read_text()).get("charged_seconds", 0.0)
                for p in (BASE / "profiles").glob("*/profile.json")
            )
        ledger = OfficialLedger(
            root, identity, initial_seconds=profile_charge, setup_seconds=time.monotonic() - started
        )
        sampler = OfficialBatchSampler(admission["ids"]["train"], seed=seed, batch_size=batch_size)
        try:
            resumes = sorted(
                root.glob("resume-epoch-*.pt"), key=lambda p: int(p.stem.split("-")[-1])
            )
            if resumes:
                restored = load_snapshot(
                    resumes[-1], model, optimizer, identity, base_hash=base_hash
                )
                if (
                    restored["step"] > ledger.completed
                    or restored["sampler"]["batches_yielded"] != restored["step"]
                    or restored["step"] % recipe["steps_per_epoch"]
                ):
                    raise ValueError("resume step/sampler/ledger mismatch")
                sampler.load_state_dict(restored["sampler"])
            ledger.charge(sampler.batches_yielded, "durable_replay", replay=True)
            epoch_done = sampler.batches_yielded // recipe["steps_per_epoch"]
            if epoch_done and epoch_done % 2 == 0:
                score_dev(
                    model,
                    optimizer,
                    admission,
                    pool,
                    identity,
                    root,
                    epoch_done,
                    base_hash,
                    sampler,
                    recipe,
                )
            stopping = decision(root, identity, recipe)
            last = time.monotonic()
            if not stopping["stop"]:
                for k in range(sampler.batches_yielded + 1, recipe["total_steps"] + 1):
                    batch = cuda_batch(batch_for(admission, "train", sampler.next_ids()))
                    values = update(model, optimizer, batch, k, recipe, pool)
                    ledger.completed = k
                    if k % 25 == 0:
                        print(
                            json.dumps(dict(epoch=k / recipe["steps_per_epoch"], **values)),
                            flush=True,
                        )
                    if time.monotonic() - last >= 10:
                        ledger.charge(k, "train")
                        last = time.monotonic()
                    if k % recipe["steps_per_epoch"] == 0:
                        epoch = k // recipe["steps_per_epoch"]
                        if code_hashes() != identity["code_hashes"] or any(
                            sha256_file(Path(p)) != sha for p, sha in admission["bindings"].items()
                        ):
                            raise ValueError("frozen source/data metadata changed")
                        require_official_storage(BASE, 6 * 1024**3)
                        ledger.charge(k, "resume_publication")
                        state = capture_state(
                            model, optimizer, identity, k, sampler.state_dict(), base_hash=base_hash
                        )
                        save_snapshot(root / f"resume-epoch-{epoch}.pt", state)
                        del state
                        prune_resumes(root)
                        if epoch % 2 == 0:
                            ledger.charge(k, "dev_generation")
                            score_dev(
                                model,
                                optimizer,
                                admission,
                                pool,
                                identity,
                                root,
                                epoch,
                                base_hash,
                                sampler,
                                recipe,
                            )
                            ledger.charge(k, "dev_complete")
                            stopping = decision(root, identity, recipe)
                            if stopping["stop"]:
                                break
            if not stopping["stop"]:
                raise ValueError("training exited without stopping decision")
            best = reconcile_selection(root, identity)
            selected = json.loads((root / "selected-checkpoint.json").read_text())
            if selected["epoch"] != best["epoch"]:
                raise ValueError("actual best selection mismatch")
            result = dict(
                status="complete",
                seed=seed,
                arm=arm,
                run_root=str(root),
                selected=selected,
                stopped_epoch=stopping["epoch"],
                stopped_updates=ledger.completed,
                stopping=stopping,
                dev_reports={p.name: sha256_file(p) for p in root.glob("dev-epoch-*.json")},
            )
            publish_completion(root, ledger, result)
            print(
                json.dumps(
                    dict(
                        status="complete",
                        run_root=str(root),
                        selected_epoch=best["epoch"],
                        stopping=stopping,
                    )
                ),
                flush=True,
            )
            return result
        except BaseException:
            ledger.close("interrupted")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("profile", "run"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(8,), default=8)
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--arm", choices=("reference", "aligned_fixed", "aligned_learned"), default="reference"
    )
    parser.add_argument("--authorize-translation", action="store_true")
    args = parser.parse_args(argv)
    if not args.authorize_translation:
        raise ValueError("explicit follow-up authorization required")

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal{signum}")

    signal.signal(signal.SIGTERM, stop)
    run(args.seed, args.arm, args.batch_size, args.mode)


if __name__ == "__main__":
    main()
