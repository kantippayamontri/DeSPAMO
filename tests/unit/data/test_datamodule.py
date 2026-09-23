from torch.utils.data import Dataset, RandomSampler, SequentialSampler

from despamo.data.datamodule import PhoenixDataModule


class TinyDataset(Dataset):
    def __init__(self, label: str, size: int) -> None:
        self.label = label
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> str:
        return f"{self.label}-{index}"


def test_datamodule_uses_configured_batch_size_and_workers() -> None:
    module = PhoenixDataModule(
        TinyDataset("train", 3),
        TinyDataset("dev", 2),
        TinyDataset("test", 1),
        batch_size=2,
        num_workers=0,
        collate_fn=list,
    )

    train = module.train_dataloader()
    validation = module.val_dataloader()
    test = module.test_dataloader()

    assert all(
        loader.batch_size == 2 and loader.num_workers == 0 for loader in (train, validation, test)
    )
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(validation.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
    assert sorted(item for batch in train for item in batch) == ["train-0", "train-1", "train-2"]
    assert list(validation) == [["dev-0", "dev-1"]]
    assert list(test) == [["test-0"]]
