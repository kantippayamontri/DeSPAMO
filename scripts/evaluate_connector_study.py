"""Once-only Signer07 scoring for six frozen, dev-selected connector models.

The unmodified projector evaluator owns locking, journals, caps and publication.
Only an isolated module instance receives connector admission/recipe/model hooks.
Use /home/kan/datasets/despamo/signer-pilot/projector-heldout-registry-v1.json;
keep this shared registry and its journals.
Parent controller must count CPU admission in its 1800-second child watchdog and
10800-second overall sequential budget. No training or selection tuning occurs here.
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import torch
from omegaconf import DictConfig, OmegaConf

from despamo.appearance.provenance import digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_POLICY = "projector-heldout-once-v1"
CONNECTOR_POLICY = "three-arm-connector-heldout-v1"
RUN_POLICY = "signer-pilot-three-arm-connectors-v1"
INPUTS = ("protocol", "dataset", "text_manifest", "dino_root", "motion_root",
          "motion_manifest", "annotation")
MAX_SECONDS = 1800


def inference_code_hashes() -> dict[str, str]:
    """Full current closure, including new adapter absent from historical inventory."""
    from scripts.train_connector_study import connector_code_hashes

    return connector_code_hashes(ROOT)


def _recipe(identity: dict) -> dict:
    from scripts.train_connector_study import RECIPE, arm_configs

    architectures = arm_configs()
    arm, seed = identity.get("arm"), identity.get("seed")
    if (arm not in architectures or type(seed) is not int or seed not in (0, 1)
            or identity.get("run_policy") != RUN_POLICY
            or identity.get("variant") != f"E3_connector_{arm}"
            or type(identity.get("module_seed")) is not int
            or identity["module_seed"] != 100 + seed
            or type(identity.get("config", {}).get("seed")) is not int
            or identity["config"]["seed"] != seed
            or digest(identity.get("architecture")) != digest(architectures[arm])
            or identity.get("architecture_schema_hash") != digest(architectures)
            or digest(identity.get("recipe")) != digest(RECIPE)
            or type(identity.get("steps")) is not int or identity["steps"] != 16000
            or identity.get("supervision_policy") != "qwen-schema98-unreviewed-v1"
            or identity.get("human_review_status") != "not_assessed"
            or any(not isinstance(identity.get(key), str)
                   or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None
                   for key in ("initial_base_hash", "initialization_profile_hash"))):
        raise ValueError("connector arm/seed/architecture/recipe/policy identity mismatch")
    current = inference_code_hashes()
    recorded = identity.get("code_hashes")
    required = {"src/despamo/appearance/schema.py", "src/despamo/models/connector_adapter.py",
                "src/despamo/training/pilot_factory.py", "scripts/train_connector_study.py",
                "scripts/evaluate_projector_pilot.py", "uv.lock"}
    if (not isinstance(recorded, dict) or not required.issubset(recorded)
            or any(name not in current or current[name] != value
                   for name, value in recorded.items())
            or identity.get("schema_hash") != current["src/despamo/appearance/schema.py"]):
        raise ValueError("connector bound source/schema changed before held-out admission")
    return {
        "connector_policy": CONNECTOR_POLICY, "arm": arm, "seed": seed,
        "architecture": architectures[arm], "module_seed": 100 + seed,
        "connector_adapter_hash": current["scripts/evaluate_connector_study.py"],
        "inference_code_hashes": current, "inference_dependencies_digest": digest(current),
    }


def validate_connector_selection(run: Path, protocol: dict, dev_ids: tuple[str, ...]) -> dict:
    """Use public strict three-report byte/metadata/declaration admission on CPU."""
    from scripts.train_connector_study import validate_connector_selection as validate

    identity = read_json(run / "budget.json").get("identity", {})
    recipe = _recipe(identity)
    selected = validate(run, protocol, dev_ids)
    expected = 10400 if (recipe["arm"], recipe["seed"]) == ("qformer", 1) else 16000
    if selected["step"] != expected:
        raise ValueError("connector frozen six-model dev-selected step mismatch")
    selected["inference_dependencies_digest"] = recipe["inference_dependencies_digest"]
    return selected


def _load_connector_model(selection: dict):
    """Attest original common initialization, reconstruct exact arm, then strict load."""
    from scripts.train_connector_study import build_connector_model

    identity = selection["identity"]
    recipe = _recipe(identity)
    if recipe["inference_dependencies_digest"] != selection["inference_dependencies_digest"]:
        raise ValueError("connector inference source drift before model loading")
    if file_hash(selection["checkpoint"]) != selection["checkpoint_hash"]:
        raise ValueError("connector checkpoint byte hash changed before loading")
    checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=False)
    if (not isinstance(checkpoint, dict)
            or digest(checkpoint.get("run_metadata")) != digest(identity)
            or digest(checkpoint.get("pilot_identity")) != digest(identity)
            or type(checkpoint.get("global_step")) is not int
            or checkpoint["global_step"] != selection["step"]):
        raise ValueError("connector checkpoint full training/pilot metadata/global_step mismatch")
    config = OmegaConf.create(identity["config"])
    if not isinstance(config, DictConfig):
        raise ValueError("connector checkpoint config must be an object")
    model = build_connector_model(config, recipe["arm"], recipe["seed"],
                                  identity["initial_base_hash"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if digest(inference_code_hashes()) != selection["inference_dependencies_digest"]:
        raise ValueError("connector inference source drift during model loading")
    return model


def _new_engine() -> ModuleType:
    """Never import-and-monkeypatch the canonical projector module's globals."""
    path = ROOT / "scripts/evaluate_projector_pilot.py"
    spec = importlib.util.spec_from_file_location("scripts._connector_heldout_engine", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load held-out engine: {path}")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    vars(engine).update(_recipe=_recipe,
                        validate_projector_selection=validate_connector_selection,
                        _load_projector_model=_load_connector_model)
    return engine


def evaluate(args) -> dict:
    """Preflight or resume/reuse one frozen connector checkpoint's logical score."""
    engine = _new_engine()
    # Only the private engine namespace uses its historical argument name.
    private = argparse.Namespace(**vars(args))
    private.projector_run = args.connector_run
    selection, view, ids, binding, output, registry = engine._prepare(private)
    if (digest(inference_code_hashes()) != binding["inference_dependencies_digest"]
            or selection["inference_dependencies_digest"]
            != binding["inference_dependencies_digest"]):
        raise ValueError("connector inference source drift during admission")
    binding["connector_run"] = str(args.connector_run.resolve())
    # Reuse preparation result; no second scan or checkpoint selection inside scorer.
    prepare = engine._prepare
    vars(engine)["_prepare"] = lambda _args: (selection, view, ids, binding, output, registry)
    try:
        result = engine.evaluate(private)
    finally:
        vars(engine)["_prepare"] = prepare
    if args.mode == "preflight":
        result.update({key: binding[key] for key in (
            "connector_policy", "arm", "architecture", "module_seed", "training_horizon",
            "connector_adapter_hash", "inference_dependencies_digest", "connector_run",
        )})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    for name in (*INPUTS, "adapted_root", "connector_run", "output", "registry"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--gpu-cap-seconds", type=int, default=MAX_SECONDS)
    result = evaluate(parser.parse_args())
    keys = ("status", "test_clips", "checkpoint_hash", "checkpoint_step", "model_variant",
            "arm", "seed", "module_seed", "connector_policy", "registry_status", "output",
            "metrics")
    print(json.dumps({key: result[key] for key in keys if key in result}, sort_keys=True))


if __name__ == "__main__":
    main()
