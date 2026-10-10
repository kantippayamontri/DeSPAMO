"""Official adaptation batching, including batch24 and receipt-safe RNG replay."""

import random

from tools.dinov3.identity import digest


class OfficialBatchSampler:
    def __init__(self, ids: tuple[str, ...], *, seed: int = 0, batch_size: int = 24):
        if (type(batch_size) is not int or batch_size <= 0 or len(ids) < batch_size
                or len(set(ids)) != len(ids) or type(seed) is not int
                or any(not isinstance(i, str) or not i for i in ids)):
            raise ValueError("invalid official sampler IDs/seed/batch size")
        self.ids, self.batch_size = ids, batch_size
        self.rng = random.Random(seed)
        self.order = list(ids)
        self.rng.shuffle(self.order)
        self.cursor = self.batches_yielded = 0

    def next_ids(self) -> tuple[str, ...]:
        if self.cursor + self.batch_size > len(self.order):
            self.order = list(self.ids)
            self.rng.shuffle(self.order)
            self.cursor = 0
        selected = tuple(self.order[self.cursor:self.cursor+self.batch_size])
        self.cursor += self.batch_size
        self.batches_yielded += 1
        return selected

    def state_dict(self) -> dict:
        return dict(schema_version=1, train_ids_hash=digest(self.ids), batch_size=self.batch_size,
                    order=list(self.order), cursor=self.cursor,
                    batches_yielded=self.batches_yielded, rng=self.rng.getstate())

    def load_state_dict(self, state: dict) -> None:
        count = state.get("batches_yielded")
        if (type(count) is not int or count < 0 or type(state.get("cursor")) is not int
                or type(state.get("batch_size")) is not int
                or type(state.get("schema_version")) is not int
                or state["schema_version"] != 1 or state["batch_size"] != self.batch_size
                or state.get("train_ids_hash") != digest(self.ids)
                or len(state.get("order", [])) != len(self.ids)
                or set(state["order"]) != set(self.ids)):
            raise ValueError("official sampler identity/schema/order mismatch")
        per_epoch = len(self.ids) // self.batch_size
        expected_cursor = 0 if count == 0 else ((count - 1) % per_epoch + 1) * self.batch_size
        if state["cursor"] != expected_cursor:
            raise ValueError("official sampler cursor/update mismatch")
        restored = random.Random()
        try:
            restored.setstate(state["rng"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError("official sampler RNG invalid") from error
        self.rng, self.order = restored, list(state["order"])
        self.cursor, self.batches_yielded = state["cursor"], count
