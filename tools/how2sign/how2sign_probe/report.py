"""Aggregation for probe label journals: uncertain rate, grouping, latency."""

BASELINE_TOP1_SHARE = {"left_hand": 0.024, "right_hand": 0.026, "mouth": 0.344}
BASELINE_DISTINCT = {"left_hand": 3787, "right_hand": 3613, "mouth": 53}


def label_rows(journals: dict, kind: str) -> list[dict]:
    rows = []
    for entry in journals.values():
        if entry.get("status") != "done" or not entry.get("label"):
            continue
        if kind and entry.get("kind") != kind:
            continue
        rows.append(dict(entry["label"], side=entry.get("side", "")))
    return rows


def _share(counts: dict) -> float:
    total = sum(counts.values())
    return round(max(counts.values()) / total, 4) if total else None


def aggregate_labels(rows: list[dict]) -> dict:
    fields = sorted({key for row in rows for key in row if key != "side"})
    stats = {}
    for field in fields:
        values = [row[field] for row in rows if field in row]
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        stats[field] = {
            "rows": len(values),
            "distinct": len(counts),
            "top1_share": _share(counts),
            "singletons": sum(1 for count in counts.values() if count == 1),
            "uncertain_rate": round(sum(v == "uncertain" for v in values) / len(values), 4)
            if values
            else None,
            "not_visible_rate": round(sum(v == "not_visible" for v in values) / len(values), 4)
            if values
            else None,
        }
    sides = {}
    for side in sorted({row.get("side", "") for row in rows}):
        side_rows = [row for row in rows if row.get("side", "") == side]
        key = (
            "handshape_class" if "handshape_class" in (side_rows[0] if side_rows else {}) else None
        )
        if key:
            sides[side] = {
                "rows": len(side_rows),
                "distinct": len({row[key] for row in side_rows}),
                "top1_share": _share(
                    {
                        v: sum(1 for row in side_rows if row[key] == v)
                        for v in {row[key] for row in side_rows}
                    }
                ),
            }
    stats["_sides"] = sides
    return stats


def latency_stats(journals: dict) -> dict:
    attempts = [attempt for entry in journals.values() for attempt in entry.get("attempts", [])]
    latencies = [a["latency_s"] for a in attempts if a.get("latency_s") is not None]
    if not latencies:
        return {"calls": 0, "mean_s": None, "max_s": None, "total_s": 0.0}
    return {
        "calls": len(attempts),
        "mean_s": round(sum(latencies) / len(latencies), 1),
        "max_s": round(max(latencies), 1),
        "total_s": round(sum(latencies), 1),
    }
