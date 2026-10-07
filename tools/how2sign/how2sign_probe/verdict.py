"""Verdict against the design's success criteria, using pre-declared thresholds.

Thresholds were declared before any label existed (see README "Stage-1
criteria") so the outcome cannot be tuned after the fact:

- detector: the implemented ``detector_gate`` (hand_rate >= 0.65,
  two_hand_rate >= 0.50, mouth_rate >= 0.70, below_floor <= 0.20).
- readability: uncertain rate of ``handshape_class`` <= 0.50 AND of mouth
  ``openness`` <= 0.50.
- grouping: per-side top-1 share >= 0.10 (a 12-way closed set is uniform at
  ~0.083, so this requires real concentration) and distinct values >= 3
  (below that the "closed set" is degenerate, not grouped).
- DINO handshape probe: not assessable on this host (no How2Sign DINO corpus,
  model gated); recorded as N/A rather than silently passed.
"""

HANDSHAPE_UNCERTAIN_CEILING = 0.50
MOUTH_UNCERTAIN_CEILING = 0.50
TOP1_FLOOR = 0.10
DISTINCT_FLOOR = 3
LABELED_FRACTION_FLOOR = 0.99
BASELINE_TOP1 = {"left_hand": 0.024, "right_hand": 0.026}


def _field(stats: dict, field: str) -> dict:
    return stats.get(field, {})


def verdict(detector_report: dict, label_report: dict) -> dict:
    gate = detector_report["gate"]
    hand = label_report["hand"]
    mouth = label_report["mouth"]
    hs = _field(hand, "handshape_class")
    op = _field(mouth, "openness")
    sides = hand.get("_sides", {})
    reasons = []
    counts = label_report.get("counts", {})
    jobs, done = counts.get("jobs", 0), counts.get("done", 0)
    labeled_fraction = done / jobs if jobs else None
    if labeled_fraction is None or labeled_fraction < LABELED_FRACTION_FLOOR:
        reasons.append(f"labeled fraction {labeled_fraction} below {LABELED_FRACTION_FLOOR}")
    if not gate["pass"]:
        reasons.append(f"detector gate failed: {gate['reasons']}")
    if hs.get("uncertain_rate") is None or hs["uncertain_rate"] > HANDSHAPE_UNCERTAIN_CEILING:
        reasons.append(
            f"handshape uncertain_rate {hs.get('uncertain_rate')} > {HANDSHAPE_UNCERTAIN_CEILING}"
        )
    if op.get("uncertain_rate") is None or op["uncertain_rate"] > MOUTH_UNCERTAIN_CEILING:
        reasons.append(
            f"mouth openness uncertain_rate {op.get('uncertain_rate')} > {MOUTH_UNCERTAIN_CEILING}"
        )
    grouping = {}
    for side, info in sorted(sides.items()):
        ok = (
            info.get("top1_share") is not None
            and info["top1_share"] >= TOP1_FLOOR
            and info.get("distinct", 0) >= DISTINCT_FLOOR
        )
        grouping[side] = {**info, "pass": ok}
        if not ok:
            reasons.append(
                f"hand side {side}: top1_share {info.get('top1_share')} "
                f"distinct {info.get('distinct')}"
            )
    if not grouping:
        reasons.append("no per-side handshape statistics")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "thresholds": {
            "handshape_uncertain_ceiling": HANDSHAPE_UNCERTAIN_CEILING,
            "mouth_uncertain_ceiling": MOUTH_UNCERTAIN_CEILING,
            "top1_floor": TOP1_FLOOR,
            "distinct_floor": DISTINCT_FLOOR,
            "labeled_fraction_floor": LABELED_FRACTION_FLOOR,
        },
        "observed": {
            "detector": detector_report["stats"],
            "labeled_fraction": labeled_fraction,
            "handshape_uncertain_rate": hs.get("uncertain_rate"),
            "mouth_openness_uncertain_rate": op.get("uncertain_rate"),
            "per_side_handshape": grouping,
            "phoenix_baseline_top1": BASELINE_TOP1,
        },
        "dino_handshape_probe": {
            "status": "not_assessable",
            "reason": "no How2Sign DINO corpus on this host and DINOv3 is gated",
        },
    }
