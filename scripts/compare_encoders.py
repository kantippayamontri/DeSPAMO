import argparse
import json
import os
import tempfile
from pathlib import Path

from despamo.comparison import load_pair, preflight_sources, summarize, validate_pair_configs
from despamo.utils.output_policy import require_ignored_git_output

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled PHOENIX14T spatial-source comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    report = commands.add_parser("report")
    report.add_argument("--results-root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        base = [
            ROOT / "configs/data/phoenix14t.yaml",
            ROOT / "configs/model/spamo_flan_t5_xl.yaml",
            ROOT / "configs/experiment/phoenix14t_baseline.yaml",
            ROOT / "configs/experiment/phoenix14t_encoder_comparison.yaml",
        ]
        clip, dino = load_pair(base)
        validate_pair_configs(clip, dino)
        print(json.dumps(preflight_sources(clip), sort_keys=True))
        return
    require_ignored_git_output(args.output, ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require_ignored_git_output(args.output.parent, ROOT)
    if args.output.exists():
        raise ValueError("comparison report exists; select a new output path")
    report_data = summarize(args.results_root)
    contents = json.dumps(report_data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}-",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, args.output)
        except FileExistsError as exc:
            raise ValueError("comparison report exists; select a new output path") from exc
        try:
            directory = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            if args.output.samefile(temporary_path):
                args.output.unlink()
            raise
    finally:
        temporary_path.unlink(missing_ok=True)
    print(json.dumps(report_data["paired_delta"], sort_keys=True))


if __name__ == "__main__":
    main()
