from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torch.nn.functional import l1_loss, mse_loss

from src.models.components.normalizer import Normalizer

class MD22LitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        normalizer: Optional[Normalizer] = None,
        compile: bool = False,
        hparams: Optional[Dict[str, Any]] = None,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.nn = self.hparams.net
        
        # normalizer
        if self.hparams.normalizer is not None:
            self.normalizer = self.hparams.normalizer
            self.normalizer.mean = mean
            self.normalizer.std = std
            print(f"Normalizer mean: {self.normalizer.mean}, dtype: {self.normalizer.mean.dtype}")
            print(f"Normalizer std: {self.normalizer.std}, dtype: {self.normalizer.std.dtype}")
        else:
            self.normalizer = None
            
        self.losses = {}
        self.num_samples = {}
        
        self._reset_ema_dict()

    def forward(self, x):
        return self.nn(x)
        
    def on_test_epoch_start(self) -> None:
        self._reset_losses_dict("test")
        
    def on_validation_epoch_start(self) -> None:
        self._reset_losses_dict("test")
        self._reset_losses_dict("val")

    def on_train_epoch_start(self) -> None:
        self._reset_losses_dict("train")

    def _update_loss_with_ema(self, stage, type, loss_name, loss):
        alpha = getattr(self.hparams.hparams, f"ema_alpha_{type}")
        if stage in ["train", "val"] and alpha < 1:
            ema = (
                self.ema[stage][type][loss_name]
                if loss_name in self.ema[stage][type]
                else loss.detach()
            )
            loss = alpha * loss + (1 - alpha) * ema
            self.ema[stage][type][loss_name] = loss.detach()
        return loss
    
    def _compute_losses(self, y, neg_dy, batch, loss_fn, stage):
        loss_name = loss_fn.__name__
        if self.normalizer is not None:
            if stage == "train":
                loss_neg_dy = loss_fn(neg_dy * self.normalizer.std, batch.neg_dy)
            else:
                loss_neg_dy = loss_fn(neg_dy * self.normalizer.std, batch.neg_dy)
        else:
            loss_neg_dy = loss_fn(neg_dy, batch.neg_dy)
        loss_neg_dy = self._update_loss_with_ema(stage, "neg_dy", loss_name, loss_neg_dy)
        if self.normalizer is not None:
            if stage == "train":
                loss_y = loss_fn(y * self.normalizer.std, (batch.y - self.normalizer.mean).float())
            else:
                loss_y = loss_fn(y * self.normalizer.std, (batch.y - self.normalizer.mean).float())
        else:
            loss_y = loss_fn(y, batch.y)
        loss_y = self._update_loss_with_ema(stage, "y", loss_name, loss_y)
        return {"y": loss_y, "neg_dy": loss_neg_dy}

        
    def model_step(self, batch, loss_fn_list, stage):
        assert len(loss_fn_list) > 0
        assert self.losses is not None
        with torch.set_grad_enabled(stage == "train" or self.nn.regress_forces):
            y, neg_dy = self.forward(batch)
            
        if batch.y.ndim == 1:
            batch.y = batch.y.unsqueeze(1)
        for loss_fn in loss_fn_list:
            step_losses = self._compute_losses(y, neg_dy, batch, loss_fn, stage)

            loss_name = loss_fn.__name__
            if self.hparams.hparams.neg_dy_weight > 0:
                self.losses[stage]["neg_dy"][loss_name].append(
                    step_losses["neg_dy"].detach() * len(batch.y)
                )
            if self.hparams.hparams.y_weight > 0:
                self.losses[stage]["y"][loss_name].append(step_losses["y"].detach() * len(batch.y))
            total_loss = (
                step_losses["y"] * self.hparams.hparams.y_weight
                + step_losses["neg_dy"] * self.hparams.hparams.neg_dy_weight
            )
            self.losses[stage]["total"][loss_name].append(total_loss.detach() * len(batch.y))
            
        return total_loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch, [mse_loss], "train")
        self.num_samples["train"] += len(batch.y)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, *args) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            loss = self.model_step(batch, [l1_loss, mse_loss], "val")
            self.num_samples["val"] += len(batch.y)
            return loss
        else:
            loss = self.model_step(batch, [l1_loss], "test")
            self.num_samples["test"] += len(batch.y)
            return loss


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if not self.trainer.sanity_checking:
            result_dict = {
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            }
            result_dict.update(self._get_mean_loss_dict_for_type("total", "train"))
            result_dict.update(self._get_mean_loss_dict_for_type("y", "train"))
            result_dict.update(self._get_mean_loss_dict_for_type("neg_dy", "train"))
            result_dict.update(self._get_mean_loss_dict_for_type("total", "val"))
            result_dict.update(self._get_mean_loss_dict_for_type("y", "val"))
            result_dict.update(self._get_mean_loss_dict_for_type("neg_dy", "val"))
            self.log_dict(result_dict, sync_dist=True)
            
            if self.num_samples['test'] > 0:
                result_dict.update(self._get_mean_loss_dict_for_type("total", "test"))
                result_dict.update(self._get_mean_loss_dict_for_type("y", "test"))
                result_dict.update(self._get_mean_loss_dict_for_type("neg_dy", "test"))
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
        result_dict.update(self._get_mean_loss_dict_for_type("total", "test"))
        result_dict.update(self._get_mean_loss_dict_for_type("y", "test"))
        result_dict.update(self._get_mean_loss_dict_for_type("neg_dy", "test"))
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
                        "monitor": "val_total_mse_loss",
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
    
    def _get_mean_loss_dict_for_type(self, type, stage):
        # Returns a list with the mean loss for each loss_fn for each stage (train, val, test)
        # Parameters:
        # type: either y, neg_dy or total
        # Returns:
        # A dict with an entry for each stage (train, val, test) with the mean loss for each loss_fn (e.g. mse_loss)
        # The key for each entry is "stage_type_loss_fn"
        assert self.losses is not None
        mean_losses = {}
        for loss_fn_name in self.losses[stage][type].keys():
            mean_losses[stage + "_" + type + "_" + loss_fn_name] = torch.stack(
                self.losses[stage][type][loss_fn_name]
            ).sum() / self.num_samples[stage]
        return mean_losses
    
    def _reset_losses_dict(self, stage):
        self.losses[stage] = {}
        self.num_samples[stage] = 0
        for loss_type in ["total", "y", "neg_dy"]:
            self.losses[stage][loss_type] = defaultdict(list)
            
    def _reset_ema_dict(self):
        self.ema = {}
        for stage in ["train", "val"]:
            self.ema[stage] = {}
            for loss_type in ["y", "neg_dy"]:
                self.ema[stage][loss_type] = {}
