import pytest
from how2sign_probe.sampling import allocate, clip_rows, sample_clips, stratum_of

COLUMNS = (
    "VIDEO_ID",
    "VIDEO_NAME",
    "SENTENCE_ID",
    "SENTENCE_NAME",
    "START_REALIGNED",
    "END_REALIGNED",
    "SENTENCE",
)


def _row(video, sentence, clip):
    return {
        "VIDEO_ID": video,
        "VIDEO_NAME": f"{video}-5-rgb_front",
        "SENTENCE_ID": f"{video}_{sentence}",
        "SENTENCE_NAME": clip,
        "START_REALIGNED": "0.0",
        "END_REALIGNED": "1.0",
        "SENTENCE": "a sentence",
    }


def test_stratum_of_prefers_split_suffix():
    assert stratum_of(_row("vidA", 3, "vidA_3-5-rgb_front")) == "vidA-5"
    assert stratum_of(_row("vidA", 3, "vidA_3-1-rgb_front")) == "vidA-1"
    assert stratum_of(_row("vidA", 3, "vidA-5-rgb_front")) == "vidA-5"
    assert stratum_of(_row("vidA", 3, "weird_name")) == "vidA"


def test_allocate_is_proportional_largest_remainder():
    got = allocate({"a": 100, "b": 100, "c": 400}, 40)
    assert got == {"a": 7, "b": 7, "c": 26}
    assert sum(got.values()) == 40
    got = allocate({"a": 7, "b": 7}, 10)
    assert sum(got.values()) == 10


def test_allocate_never_exceeds_population():
    got = allocate({"a": 1, "b": 1000}, 200)
    assert got["a"] == 1
    assert sum(got.values()) == 200
    assert got["b"] == 199


def test_allocate_partial_when_population_smaller_than_total():
    got = allocate({"a": 2, "b": 3}, 200)
    assert got == {"a": 2, "b": 3}


def _write_rows(tmp_path, videos=("vid1", "vid2", "vid3"), per_video=10):
    lines = ["\t".join(COLUMNS)]
    for video in videos:
        for i in range(per_video):
            lines.append("\t".join(_row(video, i, f"{video}_{i}-5-rgb_front").values()))
    path = tmp_path / "train.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_clip_rows_requires_columns(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("A,B\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="column"):
        clip_rows(path)


def test_clip_rows_rejects_duplicate_clips(tmp_path):
    path = _write_rows(tmp_path)
    rows = path.read_text(encoding="utf-8").splitlines()
    rows.append(rows[1])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        clip_rows(path)


def test_clip_rows_requires_video_and_clip(tmp_path):
    path = _write_rows(tmp_path)
    text = path.read_text(encoding="utf-8").replace("\nvid1\t", "\n\t").replace("\tvid1\t", "\t\t")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="VIDEO_ID"):
        clip_rows(path)


def test_sample_clips_is_deterministic_and_stratified(tmp_path):
    rows = clip_rows(_write_rows(tmp_path))
    first = sample_clips(rows, total=9, seed=0)
    second = sample_clips(rows, total=9, seed=0)
    assert first == second
    assert len(first) == 9
    assert sorted(r["stratum"] for r in first) == ["vid1-5"] * 3 + ["vid2-5"] * 3 + ["vid3-5"] * 3
    other = sample_clips(rows, total=9, seed=1)
    assert {r["clip"] for r in other} != {r["clip"] for r in first}


def test_sample_clips_no_duplicates(tmp_path):
    rows = clip_rows(_write_rows(tmp_path))
    picked = sample_clips(rows, total=15, seed=3)
    clips = [r["clip"] for r in picked]
    assert len(clips) == len(set(clips)) == 15


def test_sample_clips_total_capped_by_population(tmp_path):
    rows = clip_rows(_write_rows(tmp_path, per_video=2))
    picked = sample_clips(rows, total=200, seed=0)
    assert len(picked) == 6
    picked = sample_clips(rows[:-1], total=5, seed=0)
    assert len(picked) == 5
