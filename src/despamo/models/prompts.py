import random

from despamo.data.batch import PhoenixBatch


def build_prompts(
    batch: PhoenixBatch,
    template: str,
    use_in_context: bool,
    num_in_context: int,
    rng: random.Random,
) -> list[str]:
    prompts = [template.format("German") for _ in batch.texts]
    if not use_in_context or len(batch.texts) == 1:
        return prompts
    if len(batch.texts) < 2:
        raise ValueError("in-context prompting requires batch size >= 2")

    examples = [
        " ".join(
            (
                f"{batch.en_texts[i]}={batch.texts[i]}",
                f"{batch.fr_texts[i]}={batch.texts[i]}",
                f"{batch.es_texts[i]}={batch.texts[i]}",
            )[:num_in_context]
        )
        for i in range(len(batch.texts))
    ]
    order = list(range(len(batch.texts)))
    rng.shuffle(order)
    partners = [0] * len(order)
    for index, other in zip(order, order[1:] + order[:1], strict=True):
        partners[index] = other
    return [f"{prompt} {examples[partners[i]]}" for i, prompt in enumerate(prompts)]
