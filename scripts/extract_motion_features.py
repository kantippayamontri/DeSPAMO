"""Dedicated-environment entry point for bounded offline motion extraction."""

import json

from tools.motion.extractor import parse_args, record_terminal_admission, run_motion_extraction


def main() -> int:
    result = run_motion_extraction(parse_args())
    # Post-API admission persistence and stdout are Task4-supervised delivery tail.
    if result["status"] in {"profiled", "complete"}:
        record_terminal_admission(result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"ready", "profiled", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
