"""CPU-only paired projector-report analysis with frozen lexical diagnostics.

Metric deltas are residual minus linear. BLEU uses evaluation/metrics.py defaults:
case-sensitive, 13a, exp smoothing, no effective-order adjustment. References are
already lowercased by score_pilot_batches; this script never renormalizes BLEU input.

Lexical rules were defined from existing Signer03 DEV references (not predictions)
in linear/residual dev-16000.json, plus closed German calendar/integer grammar.
They are frozen here; no vocabulary is learned from the reports being compared.
They measure lexical agreement, not semantic completeness or visual correctness.
"""

import argparse
import copy
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
import sacrebleu
from sacrebleu.metrics.bleu import BLEU

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json

DECODER = {"mode": "deterministic", "beam_size": 5, "max_length": 64, "in_context": False}
METRICS = ("bleu1", "bleu2", "bleu3", "bleu4", "rougeL_f1")
CATEGORIES = ("date", "location", "temperature")
STATUSES = (
    "matched", "mismatched", "missing_prediction", "unmatched_prediction",
    "both_absent", "unparseable", "unsupported",
)
EXAMPLE_LIMIT = 5
TEXT_LIMIT = 240


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().translate(
        str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "−": "-"})
    )


WEEKDAYS = {day: day for day in (
    "montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag",
)}
WEEKDAYS.update({day + "s": day for day in tuple(WEEKDAYS)})
WEEKDAYS.update({"sonnabend": "samstag", "sonnabends": "samstag",
                 "ostermontag": "montag", "pfingstmontag": "montag"})
MONTHS = {name: n for n, name in enumerate((
    "januar", "februar", "maerz", "april", "mai", "juni", "juli", "august",
    "september", "oktober", "november", "dezember",
), 1)}
MONTHS["jaenner"] = 1
NUMBER_WORDS = {name: n for n, name in enumerate((
    "null", "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun",
    "zehn", "elf", "zwoelf", "dreizehn", "vierzehn", "fuenfzehn", "sechzehn",
    "siebzehn", "achtzehn", "neunzehn",
))}
NUMBER_WORDS["ein"] = 1
TENS = {"zwanzig": 20, "dreissig": 30, "vierzig": 40, "fuenfzig": 50,
        "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90}
NUMBER_WORDS.update(TENS)
for _ten, _value in TENS.items():
    for _unit, _n in (("ein", 1), ("zwei", 2), ("drei", 3), ("vier", 4), ("fuenf", 5),
                     ("sechs", 6), ("sieben", 7), ("acht", 8), ("neun", 9)):
        NUMBER_WORDS[_unit + "und" + _ten] = _value + _n
ORDINALS = {}
for _word, _n in NUMBER_WORDS.items():
    if not 1 <= _n <= 31 or _word == "ein":
        continue
    _root = {1: "erst", 2: "zweit", 3: "dritt", 7: "siebt", 8: "acht"}.get(
        _n, _word + ("t" if _n < 20 else "st"),
    )
    ORDINALS.update({_root + ending: _n for ending in ("e", "en", "em", "er", "es")})
ORDINALS.update({"siebent" + ending: 7 for ending in ("e", "en", "em", "er", "es")})

# Exact token/phrase aliases, not substring matching. Wind directions also count.
LOCATIONS = {}
for _canonical, _noun, _adjective in (
    ("nord", "norden", "noerdlich"), ("sued", "sueden", "suedlich"),
    ("ost", "osten", "oestlich"), ("west", "westen", "westlich"),
    ("nordost", "nordosten", "nordoestlich"), ("nordwest", "nordwesten", "nordwestlich"),
    ("suedost", "suedosten", "suedoestlich"), ("suedwest", "suedwesten", "suedwestlich"),
):
    LOCATIONS.update({_canonical: _canonical, _noun: _canonical,
                      _canonical + "haelfte": _canonical})
    LOCATIONS.update({_adjective + end: _canonical for end in ("", "e", "en", "er", "es", "em")})
for _canonical, _aliases in {
    "deutschland": ("deutschland", "deutschlands"),
    "norddeutschland": ("norddeutschland", "norddeutschlands", "norddeutschen tiefland"),
    "sueddeutschland": ("sueddeutschland", "sueddeutschlands"),
    "westdeutschland": ("westdeutschland", "westdeutschlands"),
    "ostdeutschland": ("ostdeutschland", "ostdeutschlands"),
    "berlin": ("berlin",), "erfurt": ("erfurt",),
    "bayern": ("bayern", "bayerns"), "thueringen": ("thueringen", "thueringens"),
    "sachsen": ("sachsen", "sachsens"), "oberfranken": ("oberfranken",),
    "breisgau": ("breisgau",), "vogtland": ("vogtland",), "lausitz": ("lausitz",),
    "eifel": ("eifel",), "allgaeu": ("allgaeu",), "schwarzwald": ("schwarzwald",),
    "rheinland": ("rheinland",), "rhein": ("rhein",), "oberrhein": ("oberrhein",),
    "niederrhein": ("niederrhein",), "rhein neckar raum": ("rhein neckar raum",),
    "main": ("main", "mains"), "ems": ("ems",), "spree": ("spree",),
    "nordsee": ("nordsee",), "ostsee": ("ostsee",),
    "alpen": ("alpen",), "mittelgebirge": ("mittelgebirge", "mittelgebirgen"),
    "kueste": ("kueste", "kuesten", "kuestenbereich", "kuestennaehe"),
    "mitte": ("mitte",), "biskaya": ("biskaya",),
    "mitteleuropa": ("mitteleuropa",), "nordosteuropa": ("nordosteuropa",),
    "england": ("england", "englands"), "mittelmeerraum": ("mittelmeerraum",),
}.items():
    LOCATIONS.update(dict.fromkeys(_aliases, _canonical))

# Conservative unresolved-place cue: in/im + optional article + unknown word.
# Common non-place weather/time complements must not become city accusations.
NON_PLACE_WORDS = set("""
der die das dem den des ein einer einem einen teile teilen teil region gebiet stadt
nacht naechsten kommenden tagen tag tage morgen abend frueh winter sommer herbst
fruehling juli januar februar maerz april mai juni august september oktober november
dezember luft kalter kalten warmer warmen kuehler kuehlen klarem klaren himmel hoehe
hoehen boeen spitzen gipfeln taelern tal laufe verlauf weiteren moment augenblick
sonne sonnenschein wolken regen schnee nebel hochnebelfeldern hochnebel frost sturm
richtung mittel punkt wechsel vorhersage wetter aussichten gebieten gegenden land
verbreitet stellenweise teilweise gebietsweise ganz vielen einigen allen
""".split()) | set(WEEKDAYS)

RULES = {
    "version": "german-weather-dev-lexical-v1",
    "label": "rule-based lexical diagnostic",
    "basis": "Frozen Signer03 DEV reference vocabulary plus closed German calendar/integer grammar",
    "source_reports": [
        "projector-linear-16000-eb0a5bef8210afc3d894f4ac57338f8cd5e0f01b6b4d74a8a38cb285b4f5f313"
        "/dev-16000.json",
        "projector-residual-16000-f8deb3d4b807d59f3e0df2e6e6253ec069fb01d1a3365c36d4c257422d87526d"
        "/dev-16000.json",
    ],
    "source_split_hash": "a95fa8f2710ebca33d6565a26f0f076817d839ecc611bc43648a48761fd6527b",
    "reference_examples": [
        "morgen freitag den zweiten oktober", "morgen ostermontag den fünften april",
        "zehn bis ein grad", "dreizehn grad im nordosten bis zu zwanzig grad im breisgau",
        "zwischen minus fünf und minus fünfzehn grad", "acht grad in berlin",
        "südlich des mains", "im norddeutschen tiefland",
    ],
    "normalization": "NFC, casefold, ä/ö/ü -> ae/oe/ue, ß -> ss; exact lexical boundaries",
    "date": "Weekday/month/day value sets; German inflected ordinals 1..31, dotted digits, D.M.; "
            "days need den/am or adjacent month. Explicit years unsupported; "
            "no relative-date resolution or year inference.",
    "temperature": "Signed integers -99..99, German cardinals 0..99; immediate grad/°C anchor; "
                   "N bis [zu] N, N-N, zwischen N und N. Range endpoints sorted; signs never "
                   "inherited. Decimals, Fahrenheit, relative degree changes, "
                   "missing units/unanchored numbers unsupported.",
    "location": "Frozen exact aliases; directional halves/adjectives collapsed; Oder requires "
                "an der. Unknown in/im complements flagged unsupported, not inferred as places.",
    "comparison": "Exact equality of unordered deduplicated value sets, not event alignment",
    "status_priority": "unparseable, unsupported, then value-set comparison",
    "rates": "Status rates divide by all clips; reference_error_rate=(mismatched+"
             "missing_prediction)/(matched+mismatched+missing_prediction); null if denominator 0",
    "absence": "both_absent means no recognized cue, not proven correct or category-free",
    "example_limits": {"per_status": EXAMPLE_LIMIT, "text_characters": TEXT_LIMIT,
                       "parse_entries": 12, "parse_entry_characters": 96},
}
RULES["vocabulary_hash"] = digest({
    "weekdays": WEEKDAYS, "months": MONTHS, "numbers": NUMBER_WORDS, "ordinals": ORDINALS,
    "locations": LOCATIONS, "non_place_words": sorted(NON_PLACE_WORDS),
})
LIMITATIONS = [
    "Exploratory within-report paired BLEU4 interval, not a training-seed confidence interval "
    "or significance test. One/two seeds do not establish broad statistical significance.",
    "No signer invariance claim: lexical diagnostics and translation scores are not signer probes.",
    "Broadcast grouping is inferred from the ID prefix before final -digits, "
    "not verified metadata; "
    "clusters are treated as exchangeable. Clip fallback ignores within-broadcast dependence.",
    "Category rates are rule-based lexical diagnostics, "
    "not complete semantic or visual correctness. "
    "Value sets lose order, multiplicity, date pairing and association with times/places/events.",
    "Date rules do not resolve relative dates, years, holiday meaning or "
    "weekday/calendar consistency; temperature rules do not infer omitted units, "
    "sign inheritance or region-specific assignments.",
    "Location vocabulary is incomplete; directions include wind directions. Unknown-place cues can "
    "flag ordinary complements; concepts without recognized cues can fall into both_absent.",
    "Unparseable/unsupported cases are excluded from reference_error_rate, with explicit coverage; "
    "unmatched predictions are counted separately. "
    "No human or appearance-audit ratings are produced.",
    "Reported metric deltas are preserved. Bootstrap BLEU4 is recomputed from text with project "
    "defaults; its agreement with reported BLEU4 is exposed, not assumed.",
    "Checkpoint hashes/variants/steps are reported identities, not checkpoint-file verification. "
    "Matched training budget, initialization and seed provenance require the parent runtime.",
]


def _tokens(text: str) -> list[str]:
    text = re.sub(r"(?<=\d)[-–—](?=[+-]?\d)", " bis ", _fold(text))
    return re.findall(r"[+-]?\d+(?:[.,]\d+)?\.?|[a-z]+|°|[-–—]", text)


def _result(values=(), unparseable=(), unsupported=()) -> dict:
    return {"values": sorted(set(values)), "unparseable": sorted(set(unparseable)),
            "unsupported": sorted(set(unsupported))}


def _integer(token: str) -> int | None:
    if token in NUMBER_WORDS:
        return NUMBER_WORDS[token]
    if re.fullmatch(r"[+-]?\d+", token):
        return int(token) if len(token.lstrip("+-")) <= 3 else None
    return None


def _date(tokens: list[str]) -> tuple[dict, set[int]]:
    values, unparseable, unsupported, used = [], [], [], set()
    for i, token in enumerate(tokens):
        if token in WEEKDAYS:
            values.append(f"weekday:{WEEKDAYS[token]}")
            used.add(i)
        if token in MONTHS:
            values.append(f"month:{MONTHS[token]}")
            used.add(i)
        if re.fullmatch(r"\d{4}\.?", token) and i > 0 and (
            tokens[i - 1] in MONTHS or tokens[i - 1] == "jahr"
            or re.fullmatch(r"\d{1,2}\.\d{1,2}\.?", tokens[i - 1])
        ):
            unsupported.append("calendar_year:" + token)
            used.add(i)
            continue
        numeric_date = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.?", token)
        degree_follows = i + 1 < len(tokens) and tokens[i + 1] in {"grad", "°"}
        calendar_cue = i > 0 and tokens[i - 1] in {"am", "den", "vom", "zum", "datum"}
        if numeric_date and not degree_follows and (token.endswith(".") or calendar_cue):
            day, month = map(int, numeric_date.groups())
            if 1 <= month <= 12 and 1 <= day <= (29 if month == 2 else
                                               30 if month in {4, 6, 9, 11} else 31):
                values.extend((f"day:{day}", f"month:{month}"))
            else:
                unsupported.append("invalid_calendar_date:" + token)
            used.add(i)
            continue
        next_month = i + 1 < len(tokens) and tokens[i + 1] in MONTHS
        previous_cue = i > 0 and tokens[i - 1] in {"den", "am"}
        dotted = bool(re.fullmatch(r"\d{1,2}\.", token))
        ordinal = ORDINALS.get(token)
        if not (next_month or previous_cue and (ordinal is not None or dotted)):
            continue
        day = ordinal if ordinal is not None else _integer(token.rstrip("."))
        if day is not None:
            maximum = 31
            if next_month:
                month = MONTHS[tokens[i + 1]]
                maximum = 29 if month == 2 else 30 if month in {4, 6, 9, 11} else 31
            if 1 <= day <= maximum:
                values.append(f"day:{day}")
            else:
                unsupported.append("invalid_calendar_day:" + token)
            used.add(i)
        elif next_month and previous_cue:
            unparseable.append("calendar_day:" + token)
            used.add(i)
    return _result(values, unparseable, unsupported), used


def _number_ending(tokens: list[str], end: int) -> tuple[int | None, int]:
    if end < 0:
        return None, end
    value = _integer(tokens[end])
    start = end
    if value is not None and end > 0 and tokens[end - 1] in {"minus", "plus"}:
        start -= 1
        value *= -1 if tokens[start] == "minus" else 1
    return value, start


def _temperature(tokens: list[str], date_indices: set[int]) -> dict:
    values, unparseable, unsupported, used = [], [], [], set(date_indices)
    anchors = [i for i, word in enumerate(tokens) if word in {"grad", "°"}]
    for i in anchors:
        following = tokens[i + 1:i + 3]
        if any(word in {"kaelter", "waermer", "kuehler", "hoeher", "niedriger"}
               for word in following):
            unsupported.append("relative_degree_change")
            continue
        if i + 1 < len(tokens) and tokens[i + 1] in {"fahrenheit", "f", "kelvin"}:
            unsupported.append("unsupported_temperature_unit:" + tokens[i + 1])
            continue
        value, start = _number_ending(tokens, i - 1)
        if value is None or i - 1 in date_indices:
            raw = tokens[i - 1] if i else "<missing>"
            field = unsupported if re.search(r"\d", raw) else unparseable
            field.append("degree_value:" + raw)
            used.add(i - 1)
            continue
        low = value
        connector = start - 1
        if connector >= 0 and tokens[connector] == "zu":
            connector -= 1
        is_range = connector >= 0 and tokens[connector] in {"bis", "und"}
        if is_range:
            first, first_start = _number_ending(tokens, connector - 1)
            # 'und' denotes a range only in an explicit 'zwischen N und N'.
            if tokens[connector] == "und" and not (
                first_start > 0 and tokens[first_start - 1] == "zwischen"
            ):
                is_range = False
            elif first is not None:
                low = first
                start = first_start
            else:
                # A leading 'bis fünf grad' is a scalar bound, not a guessed range.
                is_range = False
        if abs(value) > 99 or abs(low) > 99:
            unsupported.append("temperature_outside_integer_grammar:" + " ".join(tokens[start:i]))
        elif is_range:
            values.append(f"range:{min(low, value)}:{max(low, value)}")
        else:
            values.append(f"value:{value}")
        used.update(range(start, i + 1))
    if not anchors and any(word.startswith("temperatur") for word in tokens):
        unsupported.append("temperature_without_degree_anchor")
    if anchors:
        for i, word in enumerate(tokens):
            if i not in used and (_integer(word) is not None or re.search(r"\d", word)):
                unsupported.append("unanchored_number:" + word)
    return _result(values, unparseable, unsupported)


def _location(tokens: list[str]) -> dict:
    values, used = [], set()
    for i in range(len(tokens)):
        if i in used:
            continue
        for size in range(min(3, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i:i + size])
            if phrase in LOCATIONS:
                values.append(LOCATIONS[phrase])
                used.update(range(i, i + size))
                break
        if tokens[i] == "oder" and tokens[max(0, i - 2):i] == ["an", "der"]:
            values.append("oder")
            used.add(i)
    unsupported = []
    for i, word in enumerate(tokens[:-1]):
        if word not in {"in", "im"}:
            continue
        j = i + 1
        if tokens[j] in {"der", "dem", "den", "die", "des"}:
            j += 1
        if j < len(tokens) and j not in used and tokens[j] not in NON_PLACE_WORDS:
            if _integer(tokens[j]) is None and tokens[j] not in ORDINALS:
                unsupported.append("unknown_location_complement:" + tokens[j])
    return _result(values, unsupported=unsupported)


def parse_weather_text(text: str) -> dict:
    """Frozen rule-based lexical values plus explicit unresolved cues by category."""
    if not isinstance(text, str):
        raise ValueError("weather text must be a string")
    tokens = _tokens(text)
    date, date_indices = _date(tokens)
    return {"date": date, "location": _location(tokens),
            "temperature": _temperature(tokens, date_indices)}


def _validate_report(report: dict, arm: str) -> None:
    if not isinstance(report, dict):
        raise ValueError(f"{arm} report must be a dictionary")
    if report.get("split") != "test":
        raise ValueError(f"{arm} report requires test split")
    for key in ("split_hash", "checkpoint_hash", "model_variant"):
        if not isinstance(report.get(key), str) or not report[key].strip():
            raise ValueError(f"{arm} report missing {key}")
    step = report.get("checkpoint_step")
    if type(step) is not int or step < 1:
        raise ValueError(f"{arm} checkpoint_step must be a positive integer")
    decoding = report.get("decoding")
    if not isinstance(decoding, dict) or set(decoding) != set(DECODER) or any(
        type(decoding.get(key)) is not type(value) or decoding[key] != value
        for key, value in DECODER.items()
    ):
        raise ValueError(f"{arm} report requires deterministic beam5/max64/reference-free decoding")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or any(key not in metrics for key in METRICS):
        raise ValueError(f"{arm} report missing metrics")
    for key, value in metrics.items():
        if type(value) not in {int, float} or type(value) is float and not math.isfinite(value):
            raise ValueError(f"{arm} {key} must be finite numeric metric")
        maximum = 100 if key.startswith("bleu") else 1 if key.startswith("rougeL") else None
        if maximum is not None and not 0 <= value <= maximum:
            raise ValueError(f"{arm} {key} outside metric scale")
    items = report.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{arm} items must be a non-empty ordered list")
    for item in items:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) for key in ("clip_id", "prediction", "reference")
        ) or not item["clip_id"].strip() or not item["reference"].strip():
            raise ValueError(f"{arm} invalid clip ID/prediction/reference")
    if len({item["clip_id"] for item in items}) != len(items):
        raise ValueError(f"{arm} duplicate clip IDs")
    if "seed" in report and (type(report["seed"]) is not int or report["seed"] < 0):
        raise ValueError(f"{arm} training seed must be a non-negative integer")


def _example(item: dict) -> dict:
    return {"clip_id": item["clip_id"][:TEXT_LIMIT],
            "prediction": item["prediction"][:TEXT_LIMIT],
            "reference": item["reference"][:TEXT_LIMIT],
            "text_truncated": any(len(item[k]) > TEXT_LIMIT for k in
                                  ("clip_id", "prediction", "reference"))}


def _bounded_parse(parsed: dict) -> dict:
    result: dict = {key: [value[:96] for value in values[:12]] for key, values in parsed.items()}
    result["truncated"] = any(len(values) > 12 or any(len(v) > 96 for v in values)
                              for values in parsed.values())
    return result


def _status(reference: dict, prediction: dict) -> str:
    if reference["unparseable"] or prediction["unparseable"]:
        return "unparseable"
    if reference["unsupported"] or prediction["unsupported"]:
        return "unsupported"
    ref, pred = reference["values"], prediction["values"]
    if not ref:
        return "unmatched_prediction" if pred else "both_absent"
    if not pred:
        return "missing_prediction"
    return "matched" if ref == pred else "mismatched"


def _lexical_diagnostics(items: list[dict], reference_parses: list[dict]) -> dict:
    result = {category: {"label": RULES["label"], "counts": dict.fromkeys(STATUSES, 0),
                         "examples": {status: [] for status in STATUSES if status != "both_absent"},
                         "parse_issue_counts": dict.fromkeys((
                             "reference_unparseable", "prediction_unparseable",
                             "reference_unsupported", "prediction_unsupported",
                         ), 0)} for category in CATEGORIES}
    for item, ref_parses in zip(items, reference_parses, strict=True):
        prediction = parse_weather_text(item["prediction"])
        for category in CATEGORIES:
            ref, pred = ref_parses[category], prediction[category]
            status = _status(ref, pred)
            row = result[category]
            row["counts"][status] += 1
            for side, parsed in (("reference", ref), ("prediction", pred)):
                for issue in ("unparseable", "unsupported"):
                    row["parse_issue_counts"][f"{side}_{issue}"] += bool(parsed[issue])
            if status != "both_absent" and len(row["examples"][status]) < EXAMPLE_LIMIT:
                row["examples"][status].append({**_example(item),
                                                "reference_parse": _bounded_parse(ref),
                                                "prediction_parse": _bounded_parse(pred)})
    for row in result.values():
        counts = row["counts"]
        evaluable = counts["matched"] + counts["mismatched"] + counts["missing_prediction"]
        row.update({"total_count": len(items), "reference_evaluable_count": evaluable,
                    "reference_error_count": counts["mismatched"] + counts["missing_prediction"],
                    "reference_error_rate": (counts["mismatched"] + counts["missing_prediction"])
                    / evaluable if evaluable else None,
                    "rates": {key: count / len(items) for key, count in counts.items()}})
    return result


def _lengths(texts: list[str]) -> dict:
    lengths = [len(text.split()) for text in texts]
    return {"total": sum(lengths), "mean": mean(lengths), "median": median(lengths),
            "min": min(lengths), "max": max(lengths),
            "p95": float(np.quantile(lengths, 0.95))}


def _output_diagnostics(items: list[dict]) -> dict:
    duplicate_count = total = repeated_clips = 0
    examples = []
    predictions = [item["prediction"] for item in items]
    for item in items:
        words = item["prediction"].casefold().split()
        ngrams = Counter(tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3)))
        duplicates = sum(count - 1 for count in ngrams.values())
        duplicate_count += duplicates
        total += sum(ngrams.values())
        repeated_clips += bool(duplicates)
        if duplicates and len(examples) < EXAMPLE_LIMIT:
            examples.append({**_example(item), "duplicate_occurrences": duplicates,
                             "repeated_4grams": [" ".join(gram)[:96] for gram, count in
                                                 ngrams.items() if count > 1][:12]})
    return {"unique_prediction_count": len(set(predictions)),
            "normalized_unique_prediction_count": len({" ".join(p.casefold().split())
                                                        for p in predictions}),
            "empty_prediction_count": sum(not p.strip() for p in predictions),
            "duplicate_4grams": {"occurrences": duplicate_count, "total_4grams": total,
                                  "fraction": duplicate_count / total if total else 0.0,
                                  "clips_with_duplicates": repeated_clips,
                                  "clip_rate": repeated_clips / len(items)},
            "word_lengths": {"prediction": _lengths(predictions),
                             "reference": _lengths([i["reference"] for i in items])},
            "repetition_examples": examples,
            "definitions": "4grams: overlapping casefolded whitespace words; punctuation retained; "
                           "duplicates=count beyond first within each clip. Unique predictions: "
                           "raw exact strings (and casefold/whitespace normalized separately). "
                           "Lengths: raw whitespace word counts, not decoder tokens."}


def _bleu_from_stats(stats: np.ndarray) -> float:
    score = BLEU.compute_bleu(
        correct=stats[2:6].tolist(), total=stats[6:10].tolist(),
        sys_len=int(stats[0]), ref_len=int(stats[1]), smooth_method="exp",
        effective_order=False, max_ngram_order=4,
    ).score
    return min(score, 100.0)  # Same roundoff cap as evaluate_translations.


def _bootstrap(linear: dict, residual: dict, replicates: int, seed: int) -> dict:
    items = linear["items"]
    matches = [re.fullmatch(r"(.+)-[0-9]+", item["clip_id"]) for item in items]
    unsupported = sum(match is None for match in matches)
    groups = {}
    for i, match in enumerate(matches):
        if unsupported:
            key = str(i)
        else:
            assert match is not None
            key = match[1]
        groups.setdefault(key, []).append(i)
    scorer = BLEU(tokenize="13a", smooth_method="exp", effective_order=False)
    arm_stats = []
    signatures = []
    for report in (linear, residual):
        statistics = []
        for item in report["items"]:
            score = scorer.corpus_score([item["prediction"]], [[item["reference"]]])
            statistics.append([score.sys_len, score.ref_len, *score.counts, *score.totals])
        clip_stats = np.asarray(statistics, dtype=np.int64)
        arm_stats.append(np.asarray([clip_stats[indices].sum(axis=0)
                                     for indices in groups.values()], dtype=np.int64))
        signatures.append(str(scorer.get_signature()))
    observed = [_bleu_from_stats(stats.sum(axis=0)) for stats in arm_stats]
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates)
    for i in range(replicates):
        indices = rng.integers(0, len(groups), size=len(groups))
        scores = [_bleu_from_stats(stats[indices].sum(axis=0)) for stats in arm_stats]
        deltas[i] = scores[1] - scores[0]
    sizes = [len(indices) for indices in groups.values()]
    return {"label": "exploratory paired BLEU4 bootstrap", "replicates": replicates,
            "seed": seed, "unit": "clip" if unsupported else "broadcast_cluster",
            "unit_count": len(groups), "unsupported_id_count": unsupported,
            "grouping_rule": "prefix before final -[0-9]+; all-or-none clip fallback",
            "fallback_reason": "Some IDs lack broadcast-clipnumber format" if unsupported else None,
            "cluster_sizes": {"min": min(sizes), "max": max(sizes), "mean": mean(sizes)},
            "observed_bleu4": dict(zip(("linear", "residual"), observed, strict=True)),
            "reported_bleu4_consistent": {
                arm: math.isclose(score, report["metrics"]["bleu4"], abs_tol=1e-6, rel_tol=1e-7)
                for arm, score, report in zip(("linear", "residual"), observed,
                                             (linear, residual), strict=True)},
            "observed_delta": observed[1] - observed[0], "mean_delta": float(deltas.mean()),
            "ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "interval": "paired percentile 2.5/97.5%, numpy linear quantiles; no p-value",
            "resampling": "Equal-probability units with replacement; include all clips in each "
                          "sampled broadcast, same draws for both arms; corpus BLEU sufficient "
                          "statistics summed (not sentence BLEU averages). ROUGE not resampled.",
            "degenerate_unit_count": len(groups) < 2,
            "sacrebleu_version": sacrebleu.__version__, "signature": signatures[0]}


def compare_projector_reports(
    linear: dict, residual: dict, *, bootstrap_replicates: int = 1000, seed: int = 0,
) -> dict:
    """Validate paired test reports; return JSON-safe residual-minus-linear analysis.

    No file/model/GPU work. Optional report.seed is training seed, distinct from
    bootstrap seed. Selected checkpoint steps may differ; budget parity is external.
    """
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    for arm, report in (("linear", linear), ("residual", residual)):
        _validate_report(report, arm)
    if linear["model_variant"] != residual["model_variant"] and (
        "residual" in linear["model_variant"].casefold()
        or "linear" in residual["model_variant"].casefold()
    ):
        raise ValueError("model_variant names the opposite comparison arm")
    for key in ("split", "split_hash", "decoding"):
        if linear[key] != residual[key]:
            raise ValueError(f"paired {key} mismatch")
    if ("seed" in linear) != ("seed" in residual) or linear.get("seed") != residual.get("seed"):
        raise ValueError("paired training seed missing or mismatched")
    paired_keys = [[(item["clip_id"], item["reference"]) for item in r["items"]]
                   for r in (linear, residual)]
    if paired_keys[0] != paired_keys[1]:
        raise ValueError("paired ordered clip IDs/references mismatch")
    reference_parses = [parse_weather_text(item["reference"]) for item in linear["items"]]
    output = {arm: _output_diagnostics(report["items"]) for arm, report in
              (("linear", linear), ("residual", residual))}
    return {
        "schema_version": "projector-study-analysis-v1", "delta_direction": "residual_minus_linear",
        "split": linear["split"], "split_hash": linear["split_hash"],
        "decoding": dict(linear["decoding"]), "item_count": len(linear["items"]),
        "ordered_ids_references_hash": digest(paired_keys[0]),
        "input_report_hashes": {"linear": digest(linear), "residual": digest(residual)},
        "training_seed": linear.get("seed"),
        "training_seed_status": "reported_matched" if "seed" in linear else "not_provided",
        "checkpoints": {arm: {key: report[key] for key in
                              ("checkpoint_hash", "model_variant", "checkpoint_step")}
                        for arm, report in (("linear", linear), ("residual", residual))},
        "metrics": {arm: {key: report["metrics"][key] for key in METRICS}
                    for arm, report in (("linear", linear), ("residual", residual))},
        "metric_deltas": {key: residual["metrics"][key] - linear["metrics"][key]
                          for key in METRICS},
        "output_diagnostics": output,
        "lexical_diagnostics": {arm: _lexical_diagnostics(report["items"], reference_parses)
                                for arm, report in (("linear", linear), ("residual", residual))},
        "diagnostic_rules": {**copy.deepcopy(RULES), "location_aliases": dict(LOCATIONS)},
        "bleu4_bootstrap": _bootstrap(linear, residual, bootstrap_replicates, seed),
        "limitations": list(LIMITATIONS),
    }


def _publish(output: Path, result: dict) -> None:
    """Native atomic JSON staging, then atomic no-clobber same-filesystem link."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".projector-analysis-", dir=output.parent) as directory:
        staged = Path(directory) / "analysis.json"
        atomic_json(staged, result)
        os.link(staged, output)  # Atomic EEXIST refusal, including dangling symlinks/races.
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("linear", "residual", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sources, reports = {}, {}
    for arm, path in (("linear", args.linear), ("residual", args.residual)):
        original_hash = file_hash(path)
        reports[arm] = read_json(path)
        if file_hash(path) != original_hash:
            raise ValueError(f"{arm} input changed while reading")
        sources[arm] = {"path": str(path), "sha256": original_hash}
    result = compare_projector_reports(reports["linear"], reports["residual"],
                                       bootstrap_replicates=args.bootstrap_replicates)
    for arm, path in (("linear", args.linear), ("residual", args.residual)):
        if file_hash(path) != sources[arm]["sha256"]:
            raise ValueError(f"{arm} input changed during analysis")
    result["source_files"] = sources
    _publish(args.output, result)


if __name__ == "__main__":
    main()
