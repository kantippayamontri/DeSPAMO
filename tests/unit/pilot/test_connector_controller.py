import json
import sys

import pytest

from scripts.run_connector_study import ConnectorPhase, aggregate_connector_results, stage_command


def _result(seed, arm, score):
    return {"seed": seed, "arm": arm, "metrics": {"bleu4": score, "rougeL_f1": score / 100},
            "selected_step": 16000, "run_key": f"{arm}-{seed}"}


def test_aggregate_reports_both_matched_seeds_without_one_seed_promotion():
    values = [_result(0, "reference", 5), _result(0, "plain_mlp", 6),
              _result(0, "qformer", 4), _result(1, "reference", 5.5),
              _result(1, "plain_mlp", 4.5), _result(1, "qformer", 6)]
    summary = aggregate_connector_results(values)
    assert summary["arms"]["plain_mlp"]["paired_bleu4_deltas"] == [1, -1]
    assert summary["arms"]["plain_mlp"]["mean_paired_bleu4_delta"] == 0
    assert summary["arms"]["qformer"]["paired_bleu4_deltas"] == [-1, 0.5]
    assert summary["comparison_scope"] == "dev_only"
    assert summary["signer07_scored"] is False
    assert summary["arms"]["plain_mlp"]["metric_deltas"]["rougeL_f1"] == pytest.approx(
        [0.01, -0.01])
    with pytest.raises(ValueError):
        aggregate_connector_results(values[:-1])


def test_phase_budget_split_is_within_approved_eighteen_hours(tmp_path):
    with ConnectorPhase(tmp_path, 0) as phase:
        assert phase.state["budget_seconds"] == 32400
        assert phase.state["connector_seed"] == 0
        assert phase.state["connector_policy"] == "three-arm-connector-controller-v1"
        with pytest.raises(ValueError):
            stage_command("bad", 0, tmp_path / "profile.json", "run")
    with pytest.raises(ValueError):
        ConnectorPhase(tmp_path, 1)


def test_phase_reuses_completed_stage_without_child_rerun(tmp_path):
    marker = tmp_path / "count"
    command = [sys.executable, "-c",
               f"from pathlib import Path;p=Path({str(marker)!r});p.write_text('one')"]
    with ConnectorPhase(tmp_path, 0) as phase:
        phase.run_stage("short", command, 10)
        before = phase.state["spent_seconds"]
        phase.run_stage("short", command, 10)
        assert phase.state["spent_seconds"] == before
    state = json.loads((tmp_path / "study-state.json").read_text())
    assert state["stages"]["short"]["status"] == "complete"
    assert marker.read_text() == "one"
