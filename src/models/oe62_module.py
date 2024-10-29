from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.nn.functional import l1_loss

from src.models.components.normalizer import Normalizer


class OE62LitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        normalizer: Optional[Normalizer] = None,
        compile: bool = False,
        hparams: DictConfig = None,
    ) -> None:
        
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.nn = self.hparams.net
        
        # normalizer
        self.normalizer = self.hparams.normalizer

        self.losses = {}
        self.num_samples = {}

    def forward(self, x):
        return self.nn(x)
        
    def on_test_epoch_start(self) -> None:
        self._reset_losses_dict("test")
        
    def on_validation_epoch_start(self) -> None:
        self._reset_losses_dict("val")

    def on_train_epoch_start(self) -> None:
        self._reset_losses_dict("train")

    def _compute_losses(self, y, batch, loss_fn, stage):
        if self.normalizer is not None:
            if stage == "train":
                loss_y = loss_fn(y, self.normalizer.norm(batch.y))
            else:
                loss_y = loss_fn(self.normalizer.denorm(y), batch.y)
        else:
            loss_y = loss_fn(y, batch.y)
        return {"y": loss_y}
    
    def model_step(self, batch, loss_fn_list, stage):
        assert len(loss_fn_list) > 0
        assert self.losses is not None
        with torch.set_grad_enabled(stage == "train"):
            y, _ = self.forward(batch)
            
        if batch.y.ndim == 1:
            batch.y = batch.y.unsqueeze(1)
            
        for loss_fn in loss_fn_list:
            step_losses = self._compute_losses(y, batch, loss_fn, stage)
            loss_name = loss_fn.__name__
            self.losses[stage]["total"][loss_name].append(step_losses["y"].detach() * len(batch.y))
    
        return step_losses["y"]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch, [l1_loss], "train")
        self.num_samples["train"] += len(batch.y)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch, [l1_loss], "val")
        self.num_samples["val"] += len(batch.y)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if not self.trainer.sanity_checking:
            result_dict = {
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            }
            result_dict.update(self._get_mean_loss_dict_for_type("train"))
            result_dict.update(self._get_mean_loss_dict_for_type("val"))
            self.log_dict(result_dict, sync_dist=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch, [l1_loss], "test")
        self.num_samples["test"] += len(batch.y)

    def on_test_epoch_end(self) -> None:
        result_dict = {}
        result_dict.update(self._get_mean_loss_dict_for_type("test"))
        self.log_dict(result_dict, sync_dist=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.nn = torch.compile(self.nn)

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.hparams.lr
        super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_total_l1_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            else:
                return {
                    "optimizer": optimizer, 
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                }
        return {"optimizer": optimizer}
    
    def _get_mean_loss_dict_for_type(self, stage):
        assert self.losses is not None
        mean_losses = {}
        for loss_fn_name in self.losses[stage]["total"].keys():
            mean_losses[stage + "_" + "total" + "_" + loss_fn_name] = torch.stack(
                self.losses[stage]["total"][loss_fn_name]
            ).sum() / self.num_samples[stage]
        return mean_losses
    
    def _reset_losses_dict(self, stage):
        self.losses[stage] = {"total": defaultdict(list)}
        self.num_samples[stage] = 0
