import os
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from src.data.components.lmdb_dataset import LmdbDataset


class OE62DataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/OE62/energy_linref_pbe0",
        batch_size: int = 64,
        infer_batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        transforms: Optional[BaseTransform] = None,
    ) -> None:
    
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.transforms = self.hparams.transforms
            
        self.batch_size_per_device = self.hparams.batch_size
        self.infer_batch_size_per_device = self.hparams.infer_batch_size


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
            self.infer_batch_size_per_device = self.hparams.infer_batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = LmdbDataset(os.path.join(self.hparams.data_dir, 'train', 'pbe0_train.lmdb'), transform=self.transforms)
            self.data_val = LmdbDataset(os.path.join(self.hparams.data_dir, 'val', 'pbe0_val.lmdb'), transform=self.transforms)
            self.data_test = LmdbDataset(os.path.join(self.hparams.data_dir, 'test', 'pbe0_test.lmdb'), transform=self.transforms)

    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.infer_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.infer_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
