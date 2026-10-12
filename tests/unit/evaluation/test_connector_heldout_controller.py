import pytest

from scripts import run_connector_heldout as controller
from scripts.run_connector_heldout import summarize_heldout


def report(seed, arm, value):
    return {"seed": seed, "arm": arm, "checkpoint_hash": f"{seed}-{arm}",
            "checkpoint_step": 10400 if (seed, arm) == (1, "qformer") else 16000,
            "split": "test", "split_hash": "split", "decoding": {"beam_size": 5},
            "metrics": {"bleu1": value + 1, "bleu2": value + 0.5,
                        "bleu3": value + 0.2, "bleu4": value, "rougeL_f1": value / 100},
            "items": [{"clip_id": "one", "reference": "fixed ref", "prediction": "one two"},
                      {"clip_id": "two", "reference": "another ref", "prediction": "rain " * 6}]}


def test_summary_uses_all_seeds_and_does_not_choose_by_heldout_score():
    records = [report(s, a, v) for s, values in ((0, (4, 5, 6)), (1, (5, 4, 6)))
               for a, v in zip(("reference", "plain_mlp", "qformer"), values, strict=True)]
    out = summarize_heldout(records)
    assert out["arms"]["plain_mlp"]["bleu4_deltas"] == [1, -1]
    assert out["arms"]["qformer"]["bleu4_deltas"] == [2, 1]
    assert out["arms"]["reference"]["mean_bleu4"] == 4.5
    assert out["results"][0]["repetition"]["repeated_4gram_outputs"] == 1
    assert out["test_based_selection"] is False
    assert next(x for x in out["results"] if x["seed"] == 1 and x["arm"] == "qformer")[
        "selected_step"] == 10400


@pytest.mark.parametrize("drift", ["seed", "reference", "decoder", "split", "order"])
def test_summary_rejects_unmatched_inputs(drift):
    records = [report(s, a, 4) for s in (0, 1) for a in ("reference", "plain_mlp", "qformer")]
    if drift == "seed":
        records[-1]["seed"] = 0
    elif drift == "reference":
        records[-1]["items"][0]["reference"] = "wrong"
    elif drift == "decoder":
        records[-1]["decoding"]["beam_size"] = 3
    elif drift == "split":
        records[-1]["split"] = "dev"
    else:
        records[-1]["items"].reverse()
    with pytest.raises(ValueError):
        summarize_heldout(records)


@pytest.mark.parametrize("drift", ["checkpoint_hash", "checkpoint_step", "seed", "arm"])
def test_preflight_mismatch_refuses_before_scoring_child(tmp_path, drift):
    source = {"status": "complete", "results": [
        {"seed": s, "arm": a, "selected_step": 10400 if (s, a) == (1, "qformer") else 16000,
         "checkpoint_hash": f"{s}-{a}", "run": str(tmp_path / f"{s}-{a}")}
        for s in (0, 1) for a in controller.ARMS
    ]}

    class FakeStudy:
        root = tmp_path
        state = {}
        launches = 0

        def save(self):
            pass

        def cpu_query(self, command):
            result = {"status": "ready", "registry_status": "unclaimed",
                      "checkpoint_hash": "0-reference", "checkpoint_step": 16000,
                      "seed": 0, "arm": "reference"}
            result[drift] = {"checkpoint_hash": "unintended", "checkpoint_step": 5500,
                             "seed": 1, "arm": "plain_mlp"}[drift]
            return result

        def run_stage(self, name, command, cap, completion_validator):
            completion_validator()
            self.launches += 1
            raise AssertionError("Wrong checkpoint must never be scored")

    study = FakeStudy()
    with pytest.raises(ValueError, match="frozen|selection"):
        controller.run(study, source, "fixed-source")
    assert study.launches == 0


def test_source_duplicate_pair_refuses_before_any_stage(tmp_path):
    source = {"status": "complete", "results": [
        {"seed": 0, "arm": "reference", "checkpoint_hash": "same", "run": str(tmp_path),
         "selected_step": 16000} for _ in range(6)
    ]}

    class NoStages:
        root = tmp_path
        state = {}

        def save(self):
            raise AssertionError("Source validation precedes state write")

    with pytest.raises(ValueError, match="six|pairs"):
        controller.run(NoStages(), source, "fixed")
