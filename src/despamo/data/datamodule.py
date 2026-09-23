from collections.abc import Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PhoenixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: Dataset,
        validation: Dataset,
        test: Dataset,
        batch_size: int,
        num_workers: int,
        collate_fn: Callable,
    ) -> None:
        super().__init__()
        self.train_dataset = train
        self.validation_dataset = validation
        self.test_dataset = test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.validation_dataset, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, False)
