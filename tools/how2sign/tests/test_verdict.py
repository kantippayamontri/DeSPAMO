import pytest
from how2sign_probe.verdict import verdict


def _detector(pass_=True):
    return {
        "gate": {"pass": pass_, "reasons": [] if pass_ else ["hand_rate too low"]},
        "stats": {"hand_rate": 0.88},
    }


def _labels(hs_uncertain=0.20, op_uncertain=0.30, top1=0.30, distinct=8):
    sides = {
        side: {"rows": 100, "distinct": distinct, "top1_share": top1} for side in ("LEFT", "RIGHT")
    }
    return {
        "hand": {
            "handshape_class": {
                "uncertain_rate": hs_uncertain,
                "distinct": distinct,
                "top1_share": top1,
            },
            "_sides": sides,
        },
        "mouth": {"openness": {"uncertain_rate": op_uncertain}},
        "counts": {"jobs": 1000, "done": 1000, "failed": 0},
    }


def test_verdict_passes_on_healthy_probe():
    got = verdict(_detector(), _labels())
    assert got["pass"] is True, got["reasons"]
    assert got["dino_handshape_probe"]["status"] == "not_assessable"


def test_verdict_fails_when_detector_fails():
    got = verdict(_detector(pass_=False), _labels())
    assert got["pass"] is False
    assert any("detector" in r for r in got["reasons"])


@pytest.mark.parametrize("hs,op,top1", [(0.60, 0.30, 0.30), (0.20, 0.80, 0.30), (0.20, 0.30, 0.05)])
def test_verdict_thresholds(hs, op, top1):
    got = verdict(_detector(), _labels(hs_uncertain=hs, op_uncertain=op, top1=top1))
    assert got["pass"] is False


def test_verdict_requires_per_side_stats():
    labels = _labels()
    labels["hand"]["_sides"] = {}
    got = verdict(_detector(), labels)
    assert got["pass"] is False
    assert any("per-side" in r for r in got["reasons"])


def test_verdict_fails_below_labeled_fraction_floor():
    labels = _labels()
    labels["counts"] = {"jobs": 1000, "done": 940, "failed": 60}
    got = verdict(_detector(), labels)
    assert got["pass"] is False
    assert any("labeled fraction" in r for r in got["reasons"])
    assert got["observed"]["labeled_fraction"] == 0.94


def test_verdict_records_thresholds_and_observations():
    got = verdict(_detector(), _labels())
    assert got["thresholds"]["handshape_uncertain_ceiling"] == 0.50
    assert got["observed"]["handshape_uncertain_rate"] == 0.20
    assert got["observed"]["phoenix_baseline_top1"]["left_hand"] == 0.024
