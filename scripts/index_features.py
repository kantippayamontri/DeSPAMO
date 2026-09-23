import argparse
from collections import Counter
from pathlib import Path

from despamo.data.manifest import index_feature_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--expected-count", action="append", default=[], metavar="SPLIT=COUNT")
    args = parser.parse_args()
    expected_counts: dict[str, int] = {}
    for entry in args.expected_count:
        split, separator, count = entry.partition("=")
        if (
            not separator
            or not split
            or "/" in split
            or not count.isdecimal()
            or split in expected_counts
        ):
            parser.error(f"invalid --expected-count: {entry!r}")
        expected_counts[split] = int(count)
    manifest = index_feature_tree(args.root, args.expected_dim, args.encoder)
    if expected_counts:
        actual_counts = Counter(record.split for record in manifest.records)
        for split, expected in expected_counts.items():
            actual = actual_counts[split]
            if actual != expected:
                parser.error(f"feature counts mismatch: {split}: expected {expected}, got {actual}")
        for split in sorted(actual_counts.keys() - expected_counts.keys()):
            parser.error(
                f"feature counts mismatch: {split}: expected 0, got {actual_counts[split]}"
            )
    manifest.save(args.output)
    print(f"indexed {len(manifest.records)} features into {args.output}")


if __name__ == "__main__":
    main()
