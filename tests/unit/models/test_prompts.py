import random

from despamo.data.batch import PhoenixBatch
from despamo.models.prompts import build_prompts


def make_batch(
    texts: tuple[str, ...] = ("eins.", "zwei."),
    en_texts: tuple[str, ...] = ("one", "two"),
    fr_texts: tuple[str, ...] = ("un", "deux"),
    es_texts: tuple[str, ...] = ("uno", "dos"),
) -> PhoenixBatch:
    size = len(texts)
    return PhoenixBatch(
        clip_ids=tuple(str(i) for i in range(size)),
        signers=("s1",) * size,
        texts=texts,
        glosses=("A",) * size,
        en_texts=en_texts,
        es_texts=es_texts,
        fr_texts=fr_texts,
        spatial=None,
        spatial_mask=None,
        motion=None,
        motion_mask=None,
    )


def test_in_context_prompts_use_other_sample_and_language_order() -> None:
    prompts = build_prompts(
        make_batch(), "Translate the given sentence into {}.", True, 3, random.Random(0)
    )

    assert prompts == [
        "Translate the given sentence into German. two=zwei. deux=zwei. dos=zwei.",
        "Translate the given sentence into German. one=eins. un=eins. uno=eins.",
    ]


def test_in_context_count_limits_languages() -> None:
    prompts = build_prompts(make_batch(), "Into {}", True, 2, random.Random(0))

    assert prompts == ["Into German two=zwei. deux=zwei.", "Into German one=eins. un=eins."]


def test_no_in_context_works_with_single_sample() -> None:
    batch = make_batch(("eins.",), ("one",), ("un",), ("uno",))

    assert build_prompts(batch, "Into {}", False, 3, random.Random(0)) == ["Into German"]


def test_in_context_singleton_returns_plain_prompt_without_self_example() -> None:
    batch = make_batch(("eins.",), ("one",), ("un",), ("uno",))
    rng = random.Random(0)
    initial_state = rng.getstate()

    assert build_prompts(batch, "Into {}", True, 3, rng) == ["Into German"]
    assert rng.getstate() == initial_state


def test_repeated_translations_do_not_prevent_in_context_pairing() -> None:
    batch = make_batch(
        ("same", "same", "same"),
        ("same", "same", "same"),
        ("same", "same", "same"),
        ("same", "same", "same"),
    )

    assert (
        build_prompts(batch, "Into {}", True, 1, random.Random(0)) == ["Into German same=same"] * 3
    )


def test_duplicate_target_texts_still_use_other_sample() -> None:
    batch = make_batch(("same", "same"), ("first", "second"), ("un", "deux"), ("uno", "dos"))

    assert build_prompts(batch, "Into {}", True, 1, random.Random(0)) == [
        "Into German second=same",
        "Into German first=same",
    ]
