import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch

from despamo.checkpoints import convert_spamo_state_dict, validate_converted_state_dict
from despamo.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-schema", type=Path, required=True)
    parser.add_argument(
        "--trusted-source",
        action="store_true",
        help="allow unrestricted pickle loading of a checkpoint from a trusted source",
    )
    args = parser.parse_args()
    if args.input == args.output or (
        args.input.exists() and args.output.exists() and args.input.samefile(args.output)
    ):
        parser.error("input and output must be different files")
    if args.target_schema == args.output or (
        args.target_schema.exists()
        and args.output.exists()
        and args.target_schema.samefile(args.output)
    ):
        parser.error("target schema and output must be different files")
    try:
        source = torch.load(args.input, map_location="cpu", weights_only=not args.trusted_source)
    except pickle.UnpicklingError as exc:
        if args.trusted_source:
            raise
        parser.error(
            f"safe checkpoint load failed: {exc}; use --trusted-source only for trusted files"
        )
    converted = convert_spamo_state_dict(source["state_dict"])
    schema_bytes = args.target_schema.read_bytes()
    try:
        schema = json.loads(schema_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(f"invalid target schema JSON: {exc}")
    if not isinstance(schema, dict) or set(schema) != {"state_dict"}:
        parser.error("invalid target schema: expected state_dict mapping")
    try:
        validate_converted_state_dict(converted, schema["state_dict"])
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": converted,
            "metadata": {
                "source_path": str(args.input),
                "source_sha256": sha256_file(args.input),
                "target_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
                "source_tensor_count": len(source["state_dict"]),
                "target_tensor_count": len(converted),
            },
        },
        args.output,
    )
    print(f"converted {len(converted)} tensors into {args.output}")


if __name__ == "__main__":
    main()
