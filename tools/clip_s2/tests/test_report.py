from tools.clip_s2.report import paired_summary


def test_missing_dino_seed_is_explicitly_unpaired():
    clip = {0: {"bleu4": 22.0}, 1: {"bleu4": 23.0}}
    dino = {0: {"bleu4": 21.0}}
    result = paired_summary(clip, dino)
    assert result["clip_mean_bleu4"] == 22.5
    assert result["paired_seed_changes"] == {"0": 1.0}
    assert result["unpaired_seeds"] == [1]
    assert result["two_seed_paired_mean_change"] is None


def test_two_complete_pairs_have_arithmetic_mean_difference():
    result = paired_summary(
        {0: {"bleu4": 22.0}, 1: {"bleu4": 23.0}}, {0: {"bleu4": 21.0}, 1: {"bleu4": 21.0}}
    )
    assert result["two_seed_paired_mean_change"] == 1.5
