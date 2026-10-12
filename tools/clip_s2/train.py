"""Same FLAN/reference update kernel with explicit CLIP roots and dev-only lifecycle."""

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
from tools.benchmark.translation_model import cuda_batch, optimizer_for
from tools.clip_s2.data import admit_data, batch_for, eval_batches
from tools.clip_s2.guards import (
    COLLAPSE_WINDOW,
    assert_base_intact,
    check_ce_collapse,
    check_dev_report,
)
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_archive import _archive_json, _copy_file, require_archive_storage
from tools.dinov3.official_runtime import OfficialLedger, require_official_storage
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.storage import atomic_json, writer
from tools.followup.evaluation import score_batches
from tools.followup.model import build_model, update, validate_prompt_lengths
from tools.followup.recipe import PromptPool, normalized, schedule
from tools.followup.train import code_hashes as inherited_hashes
from tools.followup.train import decision, publish_completion

PROJECT = Path("/home/kan/Research/DeSpaMo")
# v1 collapsed (in-process corruption); v2 interrupted by host reboot and needed a
# reboot-safe owner fix. Both preserved untouched; v3 runs the identical recipe.
BASE = Path("/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v3")
ARCHIVE = Path("/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-clip-s2-spamo200-v3")
PREDECESSOR = Path("/home/kan/datasets/despamo/official-phoenix14t-translation-spamo200-v1")
FIELDS = {
    "spatial_root",
    "spatial_manifest_sha256",
    "feature_identity_sha256",
    "predecessor_root",
    "study_root",
    "archive_root",
    "seeds",
    "arm",
    "test_scoring",
}


def load_config(path: Path) -> dict:
    c = json.loads(path.read_text())
    if (
        set(c) != FIELDS
        or c["seeds"] != [0, 1]
        or c["arm"] != "reference"
        or c["test_scoring"] is not False
        or c["study_root"] != str(BASE)
        or c["archive_root"] != str(ARCHIVE)
        or c["predecessor_root"] != str(PREDECESSOR)
    ):
        raise ValueError("exact dev-only two-reference study inputs required")
    if (
        sha256_file(Path(c["spatial_root"]) / "complete/identity.json")
        != c["feature_identity_sha256"]
    ):
        raise ValueError("CLIP feature identity bytes changed")
    return c


def code_hashes() -> dict:
    values = inherited_hashes()
    here = Path(__file__).resolve().parent
    values.update({str(p): sha256_file(p) for p in here.glob("*.py")})
    return values


def verify_initialization(path: Path, seed: int, common: str, base_hash: str) -> bool:
    expected = dict(seed=seed, common_mutable_fingerprint=common, frozen_fingerprint=base_hash)
    if json.loads(path.read_text()) != expected:
        raise ValueError("matched reference initialization changed")
    return True


def candidate_to_archive(root: Path, archive: Path, state: dict, epoch: int) -> Path:
    dest = archive / root.name / f"model-epoch-{epoch}.pt"
    scratch = root / f"candidate-epoch-{epoch}.pt"
    if dest.exists():
        if not dest.with_suffix(".json").exists():
            native = json.loads(scratch.with_suffix(".json").read_text())
            if (
                native["identity"] != state["identity"]
                or native["sha256"] != sha256_file(scratch)
                or native["sha256"] != sha256_file(dest)
            ):
                raise ValueError("pending candidate copy differs from native source")
            _archive_json(dest.with_suffix(".json"), native)
        r = json.loads(dest.with_suffix(".json").read_text())
        if r["identity"] != state["identity"] or r["sha256"] != sha256_file(dest):
            raise ValueError("archived candidate identity/hash changed")
        return dest
    require_archive_storage(archive, 512 * 1024**2)
    if scratch.exists():
        prior = torch.load(scratch, map_location="cpu", weights_only=False)
        if (
            prior["identity"] != state["identity"]
            or prior["step"] != state["step"]
            or any(not torch.equal(prior["weights"][n], v) for n, v in state["weights"].items())
        ):
            raise ValueError("pending native candidate differs from replay")
    else:
        save_snapshot(scratch, state)
    receipt = json.loads(scratch.with_suffix(".json").read_text())
    _copy_file(scratch, dest)
    if sha256_file(dest) != receipt["sha256"]:
        raise ValueError("candidate archive hash verification failed")
    _archive_json(dest.with_suffix(".json"), receipt)
    scratch.unlink()
    scratch.with_suffix(".json").unlink()
    return dest


def epoch_guard(model, base_hash: str, ce_history: list[float], step: int, recipe: dict) -> None:
    """Reject corrupt live state before it can reach a durable resume or dev report."""
    assert_base_intact(model, base_hash)
    check_ce_collapse(ce_history, step, recipe)


def score_dev(model, optimizer, admission, pool, identity, root, epoch, base_hash, sampler):
    target = root / f"dev-epoch-{epoch}.json"
    if target.exists():
        reconcile_selection(root, identity)
        return json.loads(target.read_text())
    state = capture_state(
        model,
        optimizer,
        identity,
        epoch * 887,
        sampler.state_dict(),
        base_hash=base_hash,
        full=False,
    )
    candidate = candidate_to_archive(root, ARCHIVE, state, epoch)
    flags, rng, prompt_rng = (
        [(m, m.training) for m in model.modules()],
        capture_rng(),
        model.rng.getstate(),
    )
    try:
        report = score_batches(
            model,
            eval_batches(admission, "dev"),
            admission["ids"]["dev"],
            pool,
            "dev",
            identity["data"]["ordered_ids"]["dev"],
        )
    finally:
        for m, flag in flags:
            m.training = flag
        restore_rng(rng)
        model.rng.setstate(prompt_rng)
    check_dev_report(report, epoch, schedule(8))
    report.update(
        epoch=epoch,
        checkpoint_step=epoch * 887,
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


def profile(admission: dict, config_path: Path, started: float) -> dict:
    pool = PromptPool(admission["datasets"]["train"].records, 0)
    model, base_hash = build_model(0, "reference", CACHE)
    lengths = validate_prompt_lengths(model, pool)
    optimizer = optimizer_for(model)
    model.to("cuda").train()
    ids = admission["ids"]["train"]
    long_visual = sorted(
        ids,
        key=lambda i: admission["datasets"]["train"].spatial_manifest.require("train", i).length,
        reverse=True,
    )
    long_target = sorted(
        ids,
        key=lambda i: len(model.language_model.tokenizer.encode(pool.by_id[i]["text"])),
        reverse=True,
    )
    identity = dict(
        purpose="profile",
        source=code_hashes(),
        config_sha256=sha256_file(config_path),
        data=admission["identity"],
        recipe=schedule(8),
        frozen_fingerprint=base_hash,
        bounded=admission["profile_only"],
    )
    root = BASE / "profiles" / digest(identity)
    with writer(root):
        results = {}
        for label, chosen in (
            ("ordinary", ids[:8]),
            ("long", long_visual[:8]),
            ("long_target", long_target[:8]),
        ):
            batch = cuda_batch(batch_for(admission, "train", tuple(chosen)))
            refs = {normalized(pool.by_id[i]["text"]) for i in batch.clip_ids}
            worst = max(
                (r for r in pool.rows if normalized(r["text"]) not in refs),
                key=lambda r: len(model.language_model.tokenizer.encode(pool.render(r))),
            )
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
                    schedule(8),
                    pool,
                    force_joint=True,
                    prompt_override=[pool.render(worst)] * len(chosen),
                )
                torch.cuda.synchronize()
                if j:
                    durations.append(time.perf_counter() - begin)
            initialized = sum(
                bool(optimizer.state.get(p)) for g in optimizer.param_groups for p in g["params"]
            )
            expected = sum(len(g["params"]) for g in optimizer.param_groups)
            if initialized != expected:
                raise ValueError("profile optimizer state initialization failed")
            results[label] = dict(
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                clips_per_second=len(chosen) / statistics.mean(durations),
                optimizer_initialized=initialized,
                expected_optimizer_parameters=expected,
                loss=values,
                exemplar_id=worst["fileid"],
            )
        model.eval()
        dev_ids = sorted(
            admission["ids"]["dev"],
            key=lambda i: admission["datasets"]["dev"].spatial_manifest.require("dev", i).length,
            reverse=True,
        )[:4]
        small = cuda_batch(batch_for(admission, "dev", tuple(dev_ids)))

        def predict():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                visual, mask = model.visual_adapter(
                    small.spatial, small.spatial_mask, small.motion, small.motion_mask
                )
                return model.language_model.generate_text(
                    visual, mask, pool.inference(small.clip_ids)[0], "deterministic", 5, 64
                )

        path = root / "reload.pt"
        if path.exists():
            load_snapshot(path, model, optimizer, identity, base_hash=base_hash)
        else:
            state = capture_state(
                model, optimizer, identity, 12, {"profile": True}, base_hash=base_hash
            )
            save_snapshot(path, state)
            del state
        predictions = predict()
        del model, optimizer, batch
        gc.collect()
        torch.cuda.empty_cache()
        model, fresh_hash = build_model(0, "reference", CACHE)
        if fresh_hash != base_hash:
            raise ValueError("fresh model base fingerprint changed")
        optimizer = optimizer_for(model)
        model.to("cuda").eval()
        load_snapshot(path, model, optimizer, identity, base_hash=base_hash)
        if predict() != predictions:
            raise ValueError("compact generation reload differs")
        report: dict = dict(
            status="complete",
            identity=identity,
            results=results,
            prompt_lengths=lengths,
            generation_equal=True,
            fresh_base_reload=True,
            gpu_free_bytes=torch.cuda.mem_get_info()[0],
            charged_seconds=time.monotonic() - started,
        )
        if (
            report["gpu_free_bytes"] < 2 * 1024**3
            or max(r["peak_reserved_bytes"] for r in results.values()) > 21 * 1024**3
        ):
            raise ValueError("CLIP batch8 profile lacks GPU headroom")
        atomic_json(root / "profile.json", report)
        pointer = "smoke-accepted.json" if admission["profile_only"] else "accepted.json"
        atomic_json(
            BASE / "profiles" / pointer,
            dict(path=str(root / "profile.json"), sha256=sha256_file(root / "profile.json")),
        )
        return report


def run(seed: int, config_path: Path, mode: str) -> dict:
    started = time.monotonic()
    c = load_config(config_path)
    if seed not in (0, 1):
        raise ValueError("approved reference seeds0/1 only")
    if (BASE / "manual-stop.json").exists():
        raise ValueError("new study intentionally stopped")
    admission = admit_data(
        Path(c["spatial_root"]),
        c["spatial_manifest_sha256"],
        profile_only=(mode == "smoke-profile"),
    )
    if other_gpu_pids():
        raise RuntimeError("GPU already owned")
    require_official_storage(BASE, 6 * 1024**3)
    require_archive_storage(ARCHIVE, 40 * 1024**3)
    if mode in ("profile", "smoke-profile"):
        return profile(admission, config_path, started)
    accepted = json.loads((BASE / "profiles/accepted.json").read_text())
    profile_path = Path(accepted["path"])
    if sha256_file(profile_path) != accepted["sha256"]:
        raise ValueError("accepted training profile bytes changed")
    evidence = json.loads(profile_path.read_text())
    if (
        evidence["status"] != "complete"
        or not evidence["generation_equal"]
        or evidence["identity"]["bounded"]
        or evidence["identity"]["source"] != code_hashes()
        or evidence["identity"]["config_sha256"] != sha256_file(config_path)
        or evidence["identity"]["data"] != admission["identity"]
        or evidence["identity"]["recipe"] != schedule(8)
    ):
        raise ValueError("complete accepted CLIP training profile required")
    pool = PromptPool(admission["datasets"]["train"].records, seed)
    for split in ("dev", "test"):
        pool.guard_references(tuple(r["text"] for r in admission["datasets"][split].records))
    model, base_hash = build_model(seed, "reference", CACHE)
    lengths = validate_prompt_lengths(model, pool)
    common = common_fingerprint(model)
    baseline_init = PREDECESSOR / "common-initialization" / f"seed{seed}.json"
    if not baseline_init.exists():
        baseline_init = (
            Path(
                "/home/kan/datasets/despamo/official-phoenix14t-translation-v1/common-initialization"
            )
            / f"seed{seed}.json"
        )
    verify_initialization(baseline_init, seed, common, base_hash)
    recipe = schedule(8)
    identity = dict(
        policy="official-frozen-clip-s2-reference-spamo200-v1",
        seed=seed,
        arm="reference",
        recipe=recipe,
        data=admission["identity"],
        code_hashes=code_hashes(),
        config_sha256=sha256_file(config_path),
        base_files=model_files(),
        frozen_fingerprint=base_hash,
        common_initialization=common,
        decoder=dict(
            beam=5,
            max_length=64,
            in_context=True,
            num_in_context=3,
            exemplar_id=pool.eval_id,
            prompt_max_tokens=1024,
        ),
        runtime={n: version(n) for n in ("torch", "transformers", "peft")},
        prompt_lengths=lengths,
    )
    root = BASE / "runs" / f"translation-reference-seed{seed}-{digest(identity)}"
    optimizer = optimizer_for(model)
    model.to("cuda").train()
    with writer(root):
        init = dict(seed=seed, common_mutable_fingerprint=common, frozen_fingerprint=base_hash)
        atomic_json(root / "initialization.json", init)
        if (root / "complete.json").exists():
            result = json.loads((root / "complete.json").read_text())
            if result["selected"] != json.loads((root / "selected-checkpoint.json").read_text()):
                raise ValueError("completed selection changed")
            if json.loads((root / "budget.json").read_text())["status"] != "complete":
                ledger = OfficialLedger(root, identity)
                if ledger.completed != result["stopped_updates"]:
                    raise ValueError("completed update accounting changed")
                ledger.close("complete")
            return result
        atomic_json(root / "identity.json", identity)
        profile_charge = (
            sum(
                json.loads(p.read_text()).get("charged_seconds", 0.0)
                for p in (BASE / "profiles").glob("*/profile.json")
            )
            if seed == 0
            else 0.0
        )
        ledger = OfficialLedger(
            root, identity, initial_seconds=profile_charge, setup_seconds=time.monotonic() - started
        )
        sampler = OfficialBatchSampler(admission["ids"]["train"], seed=seed, batch_size=8)
        try:
            resumes = sorted(
                root.glob("resume-epoch-*.pt"), key=lambda p: int(p.stem.split("-")[-1])
            )
            if resumes:
                state = load_snapshot(resumes[-1], model, optimizer, identity, base_hash=base_hash)
                if (
                    state["step"] > ledger.completed
                    or state["step"] != state["sampler"]["batches_yielded"]
                    or state["step"] % 887
                ):
                    raise ValueError("resume/ledger/sampler step mismatch")
                sampler.load_state_dict(state["sampler"])
            ledger.charge(sampler.batches_yielded, "durable_replay", replay=True)
            epoch = sampler.batches_yielded // 887
            if epoch and epoch % 2 == 0:
                score_dev(
                    model, optimizer, admission, pool, identity, root, epoch, base_hash, sampler
                )
            stopping = decision(root, identity, recipe)
            last = time.monotonic()
            ce_history: list[float] = []
            if not stopping["stop"]:
                for k in range(sampler.batches_yielded + 1, recipe["total_steps"] + 1):
                    values = update(
                        model,
                        optimizer,
                        cuda_batch(batch_for(admission, "train", sampler.next_ids())),
                        k,
                        recipe,
                        pool,
                    )
                    ledger.completed = k
                    if values["translation_loss"] is not None:
                        ce_history.append(values["translation_loss"])
                        del ce_history[:-COLLAPSE_WINDOW]
                    if k % 25 == 0:
                        print(json.dumps(dict(epoch=k / 887, **values)), flush=True)
                    if time.monotonic() - last >= 10:
                        ledger.charge(k, "train")
                        last = time.monotonic()
                    if k % 887 == 0:
                        epoch = k // 887
                        if (
                            code_hashes() != identity["code_hashes"]
                            or sha256_file(config_path) != identity["config_sha256"]
                            or any(
                                sha256_file(Path(p)) != h for p, h in admission["bindings"].items()
                            )
                        ):
                            raise ValueError("training source/data/config drift")
                        epoch_guard(model, base_hash, ce_history, k, recipe)
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
                            )
                            ledger.charge(k, "dev_complete")
                            stopping = decision(root, identity, recipe)
                            if stopping["stop"]:
                                break
            if not stopping["stop"]:
                raise ValueError("training exited without stopping decision")
            reconcile_selection(root, identity)
            selected = json.loads((root / "selected-checkpoint.json").read_text())
            result = dict(
                status="complete",
                seed=seed,
                arm="reference",
                run_root=str(root),
                selected=selected,
                stopped_epoch=stopping["epoch"],
                stopped_updates=ledger.completed,
                stopping=stopping,
                dev_reports={p.name: sha256_file(p) for p in root.glob("dev-epoch-*.json")},
            )
            publish_completion(root, ledger, result)
            return result
        except BaseException:
            ledger.close("interrupted")
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("run", "profile", "smoke-profile"), required=True)
    p.add_argument("--seed", type=int, choices=(0, 1), default=0)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--authorize-translation", action="store_true")
    a = p.parse_args()
    if not a.authorize_translation:
        raise ValueError("explicit translation authorization required")

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal{signum}")

    signal.signal(signal.SIGTERM, stop)
    print(json.dumps(run(a.seed, a.config, a.mode)), flush=True)


if __name__ == "__main__":
    main()
