import argparse
import json
from pathlib import Path

from despamo.checkpoints import schema_from_model
from despamo.config import load_config
from despamo.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export target model state-dict schema")
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for config_path in args.config:
        if config_path == args.output or (
            config_path.exists() and args.output.exists() and config_path.samefile(args.output)
        ):
            parser.error("config and output must be different files")
    schema = {"state_dict": schema_from_model(build_model(load_config(args.config)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"exported {len(schema['state_dict'])} tensors into {args.output}")


if __name__ == "__main__":
    main()
