import inspect
from math import ceil
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

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

import radon

import utils



class LearnedFilterModel(pl.LightningModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config
        self.example_input_array = torch.randn((1,1,len(self.config.sino_angles) if self.config.sino_angles != None else 256,len(self.config.sino_positions) if self.config.sino_positions != None else ceil((self.config.img_size*1.41421356237)/2.0)*2+1)) #Needed for pytorch lightning

        self.filter_params = torch.nn.parameter.Parameter(
            torch.zeros((
                len(self.config.sino_angles) if self.config.sino_angles != None else 256,
                int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.img_size*1.41421356237)//2+1
            ))
        )
        self.angles = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_angles), requires_grad=False) if self.config.sino_angles != None else None
        self.positions = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_positions), requires_grad=False) if self.config.sino_positions != None else None
        self.layers = nn.Sequential(
            radon.RadonFilter(lambda sino, params: sino*params, self.filter_params),
            radon.RadonBackward(self.config.img_size, self.angles, self.positions)
        )

        #Setup metrics
        with warnings.catch_warnings():
            self.training_loss_metric   = torchmetrics.MeanMetric(nan_strategy="ignore")
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
        return self.layers(x)



    #Apply model for n iterations
    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        return self.forward_intern(sino)



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer_lr)
    


    def training_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> torch.Tensor:
        #Reset metrics
        self.training_loss_metric.reset()

        #Forward pass
        img, _ = batch
        x = radon.radon_forward(img, torch.tensor(self.config.sino_angles, device=img.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=img.device) if self.config.sino_positions != None else None)
        y = img
        z = self.forward_intern(x)
        loss = F.mse_loss(y, z)
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
        img, _ = batch
        x = radon.radon_forward(img, torch.tensor(self.config.sino_angles, device=img.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=img.device) if self.config.sino_positions != None else None)
        y = img
        z = self.forward_intern(x)
        self.validation_loss_metric.update(F.mse_loss(y, z))

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
        img, _ = batch
        x = radon.radon_forward(img, torch.tensor(self.config.sino_angles, device=img.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=img.device) if self.config.sino_positions != None else None)
        y = img
        z = self.forward_intern(x)
        self.test_loss_metric.update(F.mse_loss(y, z))

        #Return data for logging purposes
        if batch_idx == 0:
            return {"x": x, "y": y, "z": z}
        return {}



    def test_epoch_end(self, outputs: list[dict[str,torch.Tensor|list[torch.Tensor]]]) -> None:
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment
            #Log mean test metrics
            logger.add_scalar("test/loss", self.test_loss_metric.compute().item(), 0)

            figure = plt.figure()
            axes: mpl_toolkits.mplot3d.Axes3D = figure.add_subplot(1, 1, 1, projection="3d")
            plot_x, plot_y = torch.meshgrid(torch.arange(self.filter_params.shape[0]), torch.arange(self.filter_params.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.filter_params.detach().to("cpu"))
            logger.add_figure("parameters/filter_coefficients", figure, 0)
            figure = plt.figure()
            plt.imshow(utils.export_image_from_batch(typing.cast(dict[str,torch.Tensor], outputs[0])["x"].mT)[0])
            plt.colorbar()
            plt.tight_layout()
            logger.add_figure("test/x", figure, 0)
            figure = plt.figure()
            plt.imshow(utils.export_image_from_batch(typing.cast(dict[str,torch.Tensor], outputs[0])["y"])[0])
            plt.colorbar()
            plt.tight_layout()
            logger.add_figure("test/y", figure, 0)
            figure = plt.figure()
            plt.imshow(utils.export_image_from_batch(typing.cast(dict[str,torch.Tensor], outputs[0])["z"])[0])
            plt.colorbar()
            plt.tight_layout()
            logger.add_figure("test/z", figure, 0)