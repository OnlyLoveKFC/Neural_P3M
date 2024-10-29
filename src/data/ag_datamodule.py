
from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from src.data.components.ag_dataset import AgDataset


class AgDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/ag",
        batch_size: int = 64,
        infer_batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        transforms: Optional[BaseTransform] = None,
        data_seed: int = 42,
    ) -> None:
    
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.transforms = self.hparams.transforms
            
        self.batch_size_per_device = self.hparams.batch_size
        self.infer_batch_size_per_device = self.hparams.infer_batch_size
        self.data_seed = self.hparams.data_seed


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
        self.dataset = AgDataset(root=self.hparams.data_dir, transform=self.transforms)

        self.num_train = 950
        self.num_val = 50
        self.num_test = len(self.dataset) - self.num_train - self.num_val
    
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(self.dataset, [self.num_train, self.num_val, self.num_test], generator=torch.Generator().manual_seed(self.data_seed))

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
        
    def statistics(self):
        dummy_loader = DataLoader(self.data_train, batch_size=self.infer_batch_size_per_device, pin_memory=False, shuffle=False)
        data = tqdm(
            dummy_loader,
            desc="computing mean and std",
        )
        ys = torch.cat([batch.y.clone() for batch in data])
        return ys.mean(dim=0), ys.std(dim=0)