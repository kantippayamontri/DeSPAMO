import json

from how2sign_probe.report import (
    BASELINE_TOP1_SHARE,
    aggregate_labels,
    label_rows,
    latency_stats,
)

HAND = json.dumps(
    {
        "handshape_class": "fist",
        "finger_spread": "together",
        "palm_direction": "up",
        "thumb_position": "extended",
        "visibility": "clear",
    }
)


def _journal(side, raw=HAND, status="done", label=None, latency=30.0, attempts=1):
    parsed = json.loads(raw) if label is None and status == "done" else label
    return {
        "status": status,
        "kind": "hand",
        "attempts": [{"latency_s": latency, "error": None}] * attempts,
        "label": parsed,
        "side": side,
    }


def test_label_rows_skips_failed():
    journals = {"k1": _journal("LEFT"), "k2": {"status": "failed", "attempts": [], "side": "RIGHT"}}
    rows = label_rows(journals, "hand")
    assert len(rows) == 1
    assert rows[0]["handshape_class"] == "fist"
    assert rows[0]["side"] == "LEFT"


def test_uncertain_rates_and_grouping():
    rows = [
        {"handshape_class": "fist", "visibility": "clear", "finger_spread": "together"},
        {"handshape_class": "fist", "visibility": "clear", "finger_spread": "uncertain"},
        {"handshape_class": "uncertain", "visibility": "clear", "finger_spread": "spread"},
    ]
    stats = aggregate_labels(rows)
    assert stats["handshape_class"]["uncertain_rate"] == round(1 / 3, 4)
    assert stats["handshape_class"]["distinct"] == 2
    assert stats["handshape_class"]["top1_share"] == round(2 / 3, 4)
    assert stats["handshape_class"]["singletons"] == 1


def test_baseline_constants_recorded():
    assert BASELINE_TOP1_SHARE["left_hand"] == 0.024
    assert BASELINE_TOP1_SHARE["mouth"] == 0.344


def test_latency_stats():
    journals = {
        "a": _journal("LEFT", latency=10.0),
        "b": _journal("RIGHT", latency=30.0),
        "c": _journal("MOUTH", status="failed"),
    }
    stats = latency_stats(journals)
    assert stats["calls"] == 3
    assert stats["mean_s"] == round((10.0 + 30.0 + 30.0) / 3, 1)
    assert stats["max_s"] == 30.0
