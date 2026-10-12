"""Independent CPU-only analysis contract; hand-checked lexical expectations."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from sacrebleu.metrics import BLEU


def test_public_interface_available():
    script = Path(__file__).resolve().parents[3] / "scripts/analyze_projector_study.py"
    assert script.is_file(), "projector analysis unit has not been implemented"
    module = importlib.import_module("scripts.analyze_projector_study")
    assert callable(module.compare_projector_reports)


@pytest.fixture
def analysis():
    script = Path(__file__).resolve().parents[3] / "scripts/analyze_projector_study.py"
    assert script.is_file(), "projector analysis unit has not been implemented"
    return importlib.import_module("scripts.analyze_projector_study")


def report(predictions=None, references=None, ids=None, *, arm="linear"):
    references = references or [
        "am montag den dritten oktober im norden minus drei bis fünf grad.",
        "am dienstag in berlin zehn grad und regen.",
        "am mittwoch im süden scheint heute die sonne.",
    ]
    predictions = predictions if predictions is not None else references.copy()
    ids = ids or ["broadcast-a-1", "broadcast-a-2", "broadcast-b-1"]
    return {
        "split": "test",
        "split_hash": "frozen-split",
        "decoding": {
            "mode": "deterministic", "beam_size": 5, "max_length": 64, "in_context": False,
        },
        "items": [
            {"clip_id": clip, "prediction": pred, "reference": ref}
            for clip, pred, ref in zip(ids, predictions, references, strict=True)
        ],
        "metrics": {
            **{f"bleu{n}": min(
                BLEU(max_ngram_order=n).corpus_score(predictions, [references]).score, 100.0,
            )
               for n in range(1, 5)},
            "rougeL_f1": 0.5,
        },
        "checkpoint_hash": f"checkpoint-{arm}",
        "model_variant": f"E3_projector_{arm}",
        "checkpoint_step": 16000,
    }


def compare(analysis, linear, residual, replicates=25):
    return analysis.compare_projector_reports(
        linear, residual, bootstrap_replicates=replicates, seed=13,
    )


def test_matching_reports_delta_and_identical_cluster_bootstrap(analysis):
    linear = report()
    residual = report(arm="residual")
    before = copy.deepcopy((linear, residual))
    result = compare(analysis, linear, residual)
    assert (linear, residual) == before
    assert result["item_count"] == 3
    assert set(result["metric_deltas"]) == {"bleu1", "bleu2", "bleu3", "bleu4", "rougeL_f1"}
    assert all(value == 0 for value in result["metric_deltas"].values())
    bootstrap = result["bleu4_bootstrap"]
    assert bootstrap["unit"] == "broadcast_cluster"
    assert bootstrap["unit_count"] == 2
    assert bootstrap["observed_delta"] == 0
    assert bootstrap["ci95"] == [0, 0]
    assert bootstrap["mean_delta"] == 0


@pytest.mark.parametrize("change", [
    "order", "references", "duplicate", "empty", "split", "split_hash", "decoder",
    "missing_decoder", "decoder_bool", "nan", "inf_extra", "missing_metric", "metric_bool",
    "metric_range", "checkpoint", "step", "variant", "prediction_type", "reference_type",
    "id_type", "seed_mismatch", "seed_missing", "seed_bool", "swapped_arms",
])
def test_rejects_bad_or_mismatched_pairs(analysis, change):
    linear, residual = report(), report(arm="residual")
    if change == "order":
        residual["items"].reverse()
    elif change == "references":
        residual["items"][0]["reference"] += " geändert"
    elif change == "duplicate":
        residual["items"][1]["clip_id"] = residual["items"][0]["clip_id"]
    elif change == "empty":
        residual["items"] = []
    elif change in {"split", "split_hash"}:
        residual[change] = "dev"
    elif change == "decoder":
        residual["decoding"]["beam_size"] = 1
    elif change == "missing_decoder":
        del residual["decoding"]["mode"]
    elif change == "decoder_bool":
        residual["decoding"]["in_context"] = 0
    elif change == "nan":
        residual["metrics"]["bleu4"] = float("nan")
    elif change == "inf_extra":
        residual["metrics"]["rougeL_precision"] = float("inf")
    elif change == "missing_metric":
        del residual["metrics"]["bleu1"]
    elif change == "metric_bool":
        residual["metrics"]["bleu4"] = True
    elif change == "metric_range":
        residual["metrics"]["rougeL_f1"] = 2
    elif change == "checkpoint":
        residual["checkpoint_hash"] = ""
    elif change == "step":
        residual["checkpoint_step"] = True
    elif change == "variant":
        residual["model_variant"] = ""
    elif change in {"prediction_type", "reference_type", "id_type"}:
        field = {"prediction_type": "prediction", "reference_type": "reference",
                 "id_type": "clip_id"}[change]
        residual["items"][0][field] = None
    elif change == "seed_mismatch":
        linear["seed"], residual["seed"] = 0, 1
    elif change == "seed_missing":
        linear["seed"] = 0
    elif change == "seed_bool":
        linear["seed"] = residual["seed"] = False
    elif change == "swapped_arms":
        linear, residual = residual, linear
    with pytest.raises(ValueError):
        compare(analysis, linear, residual)


@pytest.mark.parametrize("replicates,seed", [(0, 0), (-1, 0), (True, 0), (1.5, 0), (2, True)])
def test_rejects_bad_bootstrap_arguments(analysis, replicates, seed):
    with pytest.raises(ValueError):
        analysis.compare_projector_reports(
            report(), report(arm="residual"), bootstrap_replicates=replicates, seed=seed,
        )


def test_repetition_unique_counts_and_whitespace_lengths(analysis):
    predictions = ["eins zwei drei vier eins zwei drei vier", "eins zwei drei vier", ""]
    result = compare(analysis, report(predictions), report(predictions, arm="residual"))
    output = result["output_diagnostics"]["linear"]
    assert output["unique_prediction_count"] == 3
    assert output["empty_prediction_count"] == 1
    assert output["duplicate_4grams"] == {
        "occurrences": 1, "total_4grams": 6, "fraction": 1 / 6,
        "clips_with_duplicates": 1, "clip_rate": 1 / 3,
    }
    assert output["word_lengths"]["prediction"]["total"] == 12
    assert output["word_lengths"]["prediction"]["mean"] == 4
    assert output["word_lengths"]["prediction"]["min"] == 0
    assert output["word_lengths"]["prediction"]["max"] == 8
    assert len(output["repetition_examples"]) == 1
    assert output["repetition_examples"][0]["clip_id"] == "broadcast-a-1"


def test_prediction_uniqueness_and_repetition_are_distinct(analysis):
    predictions = ["eins zwei drei vier", "eins zwei drei vier", "EINS  zwei drei vier"]
    output = compare(analysis, report(predictions), report(predictions, arm="residual"))[
        "output_diagnostics"
    ]["linear"]
    assert output["unique_prediction_count"] == 2
    assert output["normalized_unique_prediction_count"] == 1
    assert output["duplicate_4grams"]["occurrences"] == 0
    assert output["duplicate_4grams"]["total_4grams"] == 3


def test_returned_rules_do_not_mutate_frozen_rule_metadata(analysis):
    linear, residual = report(), report(arm="residual")
    first = compare(analysis, linear, residual)
    second = copy.deepcopy(compare(analysis, linear, residual))
    first["diagnostic_rules"]["source_reports"].clear()
    first["diagnostic_rules"]["example_limits"]["per_status"] = 1000
    assert compare(analysis, linear, residual) == second


def test_extreme_numeric_metric_rejected_as_invalid_report(analysis):
    residual = report(arm="residual")
    residual["metrics"]["bleu4"] = 10**1000
    with pytest.raises(ValueError):
        compare(analysis, report(), residual)


@pytest.mark.parametrize("text,values", [
    ("Am Montag, den dritten Oktober.", ["day:3", "month:10", "weekday:montag"]),
    ("am 3. Oktober", ["day:3", "month:10"]),
    ("am einundzwanzigsten März", ["day:21", "month:3"]),
    ("am 03.10.", ["day:3", "month:10"]),
    ("am Ostermontag den fünften April", ["day:5", "month:4", "weekday:montag"]),
    ("am Sonnabend den dreißigsten Januar", ["day:30", "month:1", "weekday:samstag"]),
    ("am Mittwoch und donnerstags", ["weekday:donnerstag", "weekday:mittwoch"]),
])
def test_german_date_lexical_forms(analysis, text, values):
    parsed = analysis.parse_weather_text(text)["date"]
    assert parsed["values"] == values
    assert parsed["unparseable"] == parsed["unsupported"] == []


@pytest.mark.parametrize("text,values", [
    ("minus drei bis fünf grad", ["range:-3:5"]),
    ("minus drei bis minus fünf grad", ["range:-5:-3"]),
    ("zwischen minus drei und plus fünf grad", ["range:-3:5"]),
    ("-3 bis +5 Grad Celsius", ["range:-3:5"]),
    ("-3–5 °C", ["range:-3:5"]),
    ("plus fünf grad und minus zwei grad", ["value:-2", "value:5"]),
    ("zweiundzwanzig bis einunddreißig grad", ["range:22:31"]),
    ("Am Dienstag den dritten Oktober", []),
    ("am 03.10. um 5 Uhr", []),
    ("wind mit fünf kilometern pro stunde", []),
])
def test_temperature_requires_degree_anchor_not_date_or_wind(analysis, text, values):
    parsed = analysis.parse_weather_text(text)["temperature"]
    assert parsed["values"] == values
    assert parsed["unparseable"] == parsed["unsupported"] == []


def test_unparseable_and_unsupported_cases_not_silently_rated(analysis):
    cases = [
        ("am 32. oktober", "date", "unsupported"),
        ("den blauen januar", "date", "unparseable"),
        ("ungefähr viele grad", "temperature", "unparseable"),
        ("3,5 grad", "temperature", "unsupported"),
        ("temperaturen zwischen drei und fünf", "temperature", "unsupported"),
        ("in atlantis regnet es", "location", "unsupported"),
    ]
    for text, category, field in cases:
        assert analysis.parse_weather_text(text)[category][field], text


def test_location_boundaries_direction_aliases_and_ambiguous_oder(analysis):
    parsed = analysis.parse_weather_text("im Norden und in Berlin an der Ostsee")
    assert parsed["location"]["values"] == ["berlin", "nord", "ostsee"]
    assert analysis.parse_weather_text("Nordosten und nordöstlich")['location']['values'] == [
        "nordost",
    ]
    assert analysis.parse_weather_text("an der Oder regnet es")['location']['values'] == ["oder"]
    assert analysis.parse_weather_text("schnee oder regen mit nordischen wolken")['location'][
        'values'
    ] == []


def test_diagnostic_counts_denominators_and_bounded_examples(analysis):
    references = ["montag", "montag", "montag", "regen", "regen", "regen", "regen"]
    predictions = ["montag", "dienstag", "regen", "freitag", "regen", "den blauen januar",
                   "am 32. oktober"]
    ids = [f"broadcast-a-{n}" for n in range(7)]
    result = compare(analysis, report(predictions, references, ids),
                     report(predictions, references, ids, arm="residual"))
    date = result["lexical_diagnostics"]["linear"]["date"]
    assert date["counts"] == {
        "matched": 1, "mismatched": 1, "missing_prediction": 1, "unmatched_prediction": 1,
        "both_absent": 1, "unparseable": 1, "unsupported": 1,
    }
    assert date["reference_evaluable_count"] == 3
    assert date["reference_error_rate"] == 2 / 3
    assert date["rates"]["unmatched_prediction"] == 1 / 7
    assert date["label"] == "rule-based lexical diagnostic"
    assert date["examples"]["mismatched"][0]["clip_id"] == "broadcast-a-1"
    assert date["examples"]["unparseable"][0]["prediction_parse"]["unparseable"]
    refs = ["montag"] * 12
    preds = ["dienstag " + "wolken " * 100] * 12
    ids = [f"broadcast-b-{n}" for n in range(12)]
    bounded = compare(analysis, report(preds, refs, ids), report(preds, refs, ids, arm="residual"))
    examples = bounded["lexical_diagnostics"]["linear"]["date"]["examples"]["mismatched"]
    assert len(examples) == 5
    assert all(len(example["prediction"]) <= 240 for example in examples)
    assert bounded["lexical_diagnostics"]["linear"]["temperature"]["reference_error_rate"] is None


def test_bootstrap_matches_manual_cluster_resampling_and_is_deterministic(analysis):
    linear = report(["morgen fällt heute sehr viel regen"] * 3)
    residual = report(arm="residual")
    result = compare(analysis, linear, residual, replicates=40)
    assert result == compare(analysis, linear, residual, replicates=40)
    rng = np.random.default_rng(13)
    samples = []
    groups = [[0, 1], [2]]
    scorer = BLEU()
    for _ in range(40):
        indices = [i for group in rng.integers(0, 2, size=2) for i in groups[group]]
        refs = [linear["items"][i]["reference"] for i in indices]
        scores = [scorer.corpus_score([r["items"][i]["prediction"] for i in indices], [refs]).score
                  for r in (linear, residual)]
        samples.append(scores[1] - scores[0])
    bootstrap = result["bleu4_bootstrap"]
    assert bootstrap["ci95"] == pytest.approx(np.quantile(samples, [0.025, 0.975]))
    assert bootstrap["mean_delta"] == pytest.approx(np.mean(samples))
    assert bootstrap["observed_delta"] == pytest.approx(result["metric_deltas"]["bleu4"])
    assert bootstrap["observed_delta"] > 0


def test_mixed_id_formats_fall_back_to_clips_with_limitation(analysis):
    ids = ["broadcast-a-1", "unstructured", "broadcast-b-1"]
    result = compare(analysis, report(ids=ids), report(ids=ids, arm="residual"))
    bootstrap = result["bleu4_bootstrap"]
    assert bootstrap["unit"] == "clip"
    assert bootstrap["unit_count"] == 3
    assert bootstrap["unsupported_id_count"] == 1
    assert bootstrap["fallback_reason"]
    assert result["limitations"]


def test_literal_identical_reports_are_valid_zero_delta_control(analysis):
    control = report()
    result = compare(analysis, control, copy.deepcopy(control))
    assert result["metric_deltas"]["bleu4"] == 0
    assert result["bleu4_bootstrap"]["ci95"] == [0, 0]


def test_mixed_calendar_and_temperature_keep_numbers_separate(analysis):
    parsed = analysis.parse_weather_text("am 3. oktober minus drei grad")
    assert parsed["date"]["values"] == ["day:3", "month:10"]
    assert parsed["temperature"] == {"values": ["value:-3"], "unparseable": [], "unsupported": []}


@pytest.mark.parametrize("text", ["3.5 grad", "3,5 grad", "3.5 °C"])
def test_decimal_temperatures_do_not_become_calendar_dates(analysis, text):
    parsed = analysis.parse_weather_text(text)
    assert parsed["date"] == {"values": [], "unparseable": [], "unsupported": []}
    assert parsed["temperature"]["unsupported"]


@pytest.mark.parametrize("text,category", [
    ("am 3. oktober 2010", "date"),
    ("am 03.10.2010", "date"),
    ("drei grad kälter", "temperature"),
    ("5 °F", "temperature"),
    ("dreihundert grad", "temperature"),
    ("12.5 grad", "temperature"),
])
def test_outside_grammar_is_explicitly_unresolved(analysis, text, category):
    parsed = analysis.parse_weather_text(text)[category]
    assert parsed["unsupported"] or parsed["unparseable"]


def test_decoder_extra_sampling_flags_refused_even_when_equal(analysis):
    linear, residual = report(), report(arm="residual")
    for item in (linear, residual):
        item["decoding"]["do_sample"] = True
    with pytest.raises(ValueError):
        compare(analysis, linear, residual)


def test_cli_refuses_input_changed_during_analysis(analysis, monkeypatch, tmp_path):
    linear, residual, output = [tmp_path / f"{name}.json" for name in ("linear", "residual", "out")]
    linear.write_text(json.dumps(report()), encoding="utf-8")
    residual.write_text(json.dumps(report(arm="residual")), encoding="utf-8")
    original_compare = analysis.compare_projector_reports

    def compare_and_mutate(*args, **kwargs):
        result = original_compare(*args, **kwargs)
        linear.write_text('{"changed": true}', encoding="utf-8")
        return result

    monkeypatch.setattr(analysis, "compare_projector_reports", compare_and_mutate)
    with pytest.raises(ValueError, match="changed"):
        analysis.main(["--linear", str(linear), "--residual", str(residual),
                       "--output", str(output), "--bootstrap-replicates", "2"])
    assert not output.exists()


def test_atomic_publish_loses_race_without_clobbering(analysis, monkeypatch, tmp_path):
    output = tmp_path / "analysis.json"
    original_atomic_json = analysis.atomic_json

    def stage_and_compete(path, value):
        original_atomic_json(path, value)
        output.write_text("competing writer", encoding="utf-8")

    monkeypatch.setattr(analysis, "atomic_json", stage_and_compete)
    with pytest.raises(FileExistsError):
        analysis._publish(output, {"safe": True})
    assert output.read_text(encoding="utf-8") == "competing writer"
    assert not list(tmp_path.glob(".projector-analysis-*"))


def test_cli_atomic_publication_refuses_overwrite_and_invalid_input(analysis, tmp_path):
    linear, residual, output = [tmp_path / f"{name}.json" for name in ("linear", "residual", "out")]
    linear.write_text(json.dumps(report()), encoding="utf-8")
    residual.write_text(json.dumps(report(arm="residual")), encoding="utf-8")
    arguments = ["--linear", str(linear), "--residual", str(residual), "--output", str(output),
                 "--bootstrap-replicates", "12"]
    analysis.main(arguments)
    original = output.read_bytes()
    assert json.loads(original)["bleu4_bootstrap"]["replicates"] == 12
    assert not list(tmp_path.glob(".projector-analysis-*"))
    with pytest.raises(FileExistsError):
        analysis.main(arguments)
    assert output.read_bytes() == original
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        analysis.main([*arguments[:-4], "--output", str(link), "--bootstrap-replicates", "12"])
    bad = report(arm="residual")
    bad["split_hash"] = "different"
    residual.write_text(json.dumps(bad), encoding="utf-8")
    untouched = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        analysis.main([*arguments[:4], "--output", str(untouched)])
    assert not untouched.exists()
