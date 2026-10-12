import importlib

import pytest


def test_annotation_ids_ignore_metadata_and_reject_duplicates():
    module = importlib.import_module("scripts.prepare_official_factor_targets")
    raw = {"metadata": "ignored", 1: {"fileid": "b"}, 0: {"fileid": "a"}}
    assert module.official_ids(raw) == ("a", "b")
    with pytest.raises(ValueError):
        module.official_ids({0: {"fileid": "a"}, 1: {"fileid": "a"}})


def test_annotation_signer_join_rejects_wrong_source():
    module = importlib.import_module("scripts.prepare_official_factor_targets")
    raw = {0: dict(fileid="a", signer="Signer03")}
    source = dict(clips=[dict(clip_id="a", signer="Signer03")])
    module.validate_annotation_join(raw, source)
    source["clips"][0]["signer"] = "Signer07"
    with pytest.raises(ValueError):
        module.validate_annotation_join(raw, source)
