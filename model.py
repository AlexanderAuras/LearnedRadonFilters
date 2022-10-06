import inspect
import itertools
import typing
import warnings

import omegaconf

import pytorch_lightning as pl
import pytorch_lightning.loggers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard

import torchmetrics

from nafn import NAFN
from radon_op import RadonForwardProjection
from utils import export_image_from_vec_batch



class NAFNDissipatingMinimizer(pl.LightningModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config
        self.example_input_array = ??? #Needed for pytorch lightning

        #Setup metrics
        with warnings.catch_warnings():
            self.training_loss_metric     = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_loss_metric       = torchmetrics.MeanMetric(nan_strategy="ignore")



    #HACK Removes metrics from PyTorch Lightning overview
    def named_children(self) -> typing.Iterator[tuple[str, nn.Module]]:
        stack = inspect.stack()
        if stack[2].function == "summarize" and stack[2].filename.endswith("pytorch_lightning/utilities/model_summary/model_summary.py"):
            return filter(lambda x: not x[0].endswith("metric"), super().named_children())
        return super().named_children()



    #Common forward method used by forward, training_step, validation_step and test_step
    def forward_intern(self, x: torch.Tensor) -> torch.Tensor:
        return ???



    #Apply model for n iterations
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_intern(x)



    def configure_optimizers(self):
        return ???
    


    def training_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> torch.Tensor:
        #Reset metrics
        self.training_loss_metric.reset()

        #Forward pass
        x, y = batch
        z = self.forward_intern(x)
        loss = ???(y, z)
        self.training_loss_metric.update(loss.item())

        #Log training metrics after each batch
        if self.logger:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment
            logger.add_scalar("training/loss", self.training_loss_metric.compute().item(), self.global_step)
        return loss



    def validation_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> dict[str,torch.Tensor|None]:
        #Reset metrics
        self.validation_loss_metric.reset()

        #Forward pass
        x, y = batch
        z = self.forward_intern(x)
        self.validation_loss_metric.update(???(y, z))

        #Return data for logging purposes
        return {}



    def validation_epoch_end(self, outputs: list[dict[str, torch.Tensor]]) -> None:
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment

            #Log validation metrics after each epoch
            logger.add_scalar("validation/loss", self.validation_loss_metric.compute().item(), self.global_step)


    def test_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> dict[str,torch.Tensor|list[torch.Tensor]]:
        #Reset metrics
        self.test_loss_metric.reset()

        #Forward pass
        x, y = batch
        z = self.forward_intern(x)
        self.test_loss_metric.update(???(y, z))

        #Return data for logging purposes
        return {}



    def test_epoch_end(self, outputs: list[dict[str,torch.Tensor|list[torch.Tensor]]]) -> None:
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment
            #Log mean test metrics
            logger.add_scalar("test/loss", self.test_loss_metric.compute().item(), 0)