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

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

import radon

from utils import log_3d, log_img



class LearnedFilterModel(pl.LightningModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config
        self.example_input_array = torch.randn((1,1,len(self.config.sino_angles) if self.config.sino_angles != None else 256,len(self.config.sino_positions) if self.config.sino_positions != None else ceil((self.config.dataset.img_size*1.41421356237)/2.0)*2+1)) #Needed for pytorch lightning

        if self.config.model.initialization == "zeros":
            self.filter_params = torch.nn.parameter.Parameter(
                torch.zeros((
                    len(self.config.sino_angles) if self.config.sino_angles != None else 256,
                    int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
                ))
            )
        elif self.config.model.initialization == "ones":
            self.filter_params = torch.nn.parameter.Parameter(
                torch.ones((
                    len(self.config.sino_angles) if self.config.sino_angles != None else 256,
                    int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
                ))
            )
        elif self.config.model.initialization == "randn":
            self.filter_params = torch.nn.parameter.Parameter(
                torch.randn((
                    len(self.config.sino_angles) if self.config.sino_angles != None else 256,
                    int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
                )).abs()
            )
        elif self.config.model.initialization == "rand":
            self.filter_params = torch.nn.parameter.Parameter(
                torch.rand((
                    len(self.config.sino_angles) if self.config.sino_angles != None else 256,
                    int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
                ))
            )
        else:
            raise NotImplementedError()
        self.angles = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_angles), requires_grad=False) if self.config.sino_angles != None else None
        self.positions = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_positions), requires_grad=False) if self.config.sino_positions != None else None
        self.layers = nn.Sequential(
            radon.RadonFilter(lambda sino, params: sino*params, self.filter_params),
            radon.RadonBackward(self.config.dataset.img_size, self.angles, self.positions)
        )

        #Setup metrics
        with warnings.catch_warnings():
            self.training_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_input_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_output_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")



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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optimizer_lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 2.0)
        #return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer
    


    def training_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> torch.Tensor:
        #Reset metrics
        self.training_loss_metric.reset()
        self.training_psnr_metric.reset()
        self.training_ssim_metric.reset()

        #Forward pass
        ground_truth, _ = batch
        sinogram = radon.radon_forward(ground_truth, torch.tensor(self.config.sino_angles, device=ground_truth.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=ground_truth.device) if self.config.sino_positions != None else None)
        noise = self.config.noise_level*torch.randn_like(sinogram)
        noisy_sinogram = sinogram+noise
        reconstruction = self.forward_intern(noisy_sinogram)
        loss = F.mse_loss(reconstruction, ground_truth)
        self.training_loss_metric.update(loss.item())
        self.training_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(reconstruction, ground_truth))
        self.training_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(reconstruction, ground_truth)))

        #Log training metrics after each batch
        if self.logger:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment
            logger.add_scalar("training/loss", self.training_loss_metric.compute().item(), self.global_step)
            logger.add_scalar("training/psnr", self.training_psnr_metric.compute().item(), self.global_step)
            logger.add_scalar("training/ssim", self.training_ssim_metric.compute().item(), self.global_step)
        return loss



    def validation_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> dict[str,torch.Tensor|None]:
        #Reset metrics
        self.validation_loss_metric.reset()
        self.validation_psnr_metric.reset()
        self.validation_ssim_metric.reset()

        #Forward pass
        ground_truth, _ = batch
        sinogram = radon.radon_forward(ground_truth, torch.tensor(self.config.sino_angles, device=ground_truth.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=ground_truth.device) if self.config.sino_positions != None else None)
        noisy_sinogram = sinogram+self.config.noise_level*torch.randn_like(sinogram)
        reconstruction = self.forward_intern(noisy_sinogram)
        self.validation_loss_metric.update(F.mse_loss(reconstruction, ground_truth))
        self.validation_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(reconstruction, ground_truth))
        self.validation_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(reconstruction, ground_truth)))

        #Return data for logging purposes
        if batch_idx == 0:
            return {"sinogram": sinogram, "noisy_sinogram": noisy_sinogram, "ground_truth": ground_truth, "reconstruction": reconstruction}
        else:
            return {}



    def validation_epoch_end(self, outputs: list[dict[str,torch.Tensor|list[torch.Tensor]]]) -> None:
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment

            #Log validation metrics after each epoch
            logger.add_scalar("validation/loss", self.validation_loss_metric.compute().item(), self.global_step)
            logger.add_scalar("validation/psnr", self.validation_psnr_metric.compute().item(), self.global_step)
            logger.add_scalar("validation/ssim", self.validation_ssim_metric.compute().item(), self.global_step)

            #Log filter coefficients
            figure = plt.figure()
            axes: mpl_toolkits.mplot3d.Axes3D = figure.add_subplot(1, 1, 1, projection="3d")
            plot_x, plot_y = torch.meshgrid(torch.arange(self.filter_params.shape[0]), torch.arange(self.filter_params.shape[1]), indexing="ij")
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//min(5, self.filter_params.shape[0])).to(torch.float32).tolist(), list(map(lambda x: f"{x/self.filter_params.shape[0]:3.2f} \u03C0", torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//min(5, self.filter_params.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.filter_params.shape[1], self.filter_params.shape[1]//min(5, self.filter_params.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Filter value")
            axes.set_zlim(0.0, 2.0)
            axes.plot_surface(plot_x, plot_y, self.filter_params.detach().to("cpu"), alpha=1.0)
            logger.add_figure("validation/filter_coefficients", figure, self.global_step)
            log_3d(logger, "validation/filter_coefficients", self.filter_params, self.global_step, 1.0)

            #Log examples
            sinogram = typing.cast(list[dict[str,torch.Tensor]], outputs)[0]["sinogram"][0,0]
            noisy_sinogram = typing.cast(list[dict[str,torch.Tensor]], outputs)[0]["noisy_sinogram"][0,0]
            filtered_sinogram = radon.radon_filter(sinogram.unsqueeze(0).unsqueeze(0), lambda s,p: s*p, self.filter_params)[0,0]
            ground_truth = typing.cast(list[dict[str,torch.Tensor]], outputs)[0]["ground_truth"][0,0]
            reconstruction = typing.cast(list[dict[str,torch.Tensor]], outputs)[0]["reconstruction"][0,0]
            log_img(logger, "validation/sinogram", sinogram.mT, self.global_step)
            log_img(logger, "validation/noisy_sinogram", noisy_sinogram.mT, self.global_step)
            log_img(logger, "validation/filtered_sinogram", filtered_sinogram.mT, self.global_step)
            log_img(logger, "validation/ground_truth", ground_truth, self.global_step)
            log_img(logger, "validation/reconstruction", reconstruction, self.global_step)



    def test_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> dict[str,torch.Tensor|list[torch.Tensor]]:
        #Reset metrics
        self.test_loss_metric.reset()
        self.test_psnr_metric.reset()
        self.test_ssim_metric.reset()

        #Forward pass
        ground_truth, _ = batch
        sinogram = radon.radon_forward(ground_truth, torch.tensor(self.config.sino_angles, device=ground_truth.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=ground_truth.device) if self.config.sino_positions != None else None)
        noisy_sinogram = sinogram+self.config.noise_level*torch.randn_like(sinogram)
        reconstruction = self.forward_intern(noisy_sinogram)
        self.test_loss_metric.update(F.mse_loss(reconstruction, ground_truth))
        self.test_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(reconstruction, ground_truth))
        self.test_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(reconstruction, ground_truth)))
        self.test_input_l2_metric.update(torch.sqrt(torch.sum(ground_truth**2, 3).sum(2)).mean())
        self.test_output_l2_metric.update(torch.sqrt(torch.sum(reconstruction**2, 3).sum(2)).mean())

        #Return data for logging purposes
        if batch_idx < 10:
            return {"sinogram": sinogram, "noisy_sinogram": noisy_sinogram, "ground_truth": ground_truth, "reconstruction": reconstruction}
        else:
            return {}



    def test_epoch_end(self, outputs: list[dict[str,torch.Tensor|list[torch.Tensor]]]) -> None:
        torch.save(self.filter_params, "coefficients.pt")
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment

            #Log mean test metrics
            logger.add_scalar("test/loss", self.test_loss_metric.compute().item(), 0)
            logger.add_scalar("test/psnr", self.test_psnr_metric.compute().item(), 0)
            logger.add_scalar("test/ssim", self.test_ssim_metric.compute().item(), 0)
            logger.add_scalar("test/input_l2", self.test_input_l2_metric.compute().item(), 0)
            logger.add_scalar("test/output_l2", self.test_output_l2_metric.compute().item(), 0)

            #Log filter coefficients
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//min(5, self.filter_params.shape[0])).to(torch.float32).tolist(), list(map(lambda x: f"{x/self.filter_params.shape[0]:3.2f} \u03C0", torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//min(5, self.filter_params.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.filter_params.shape[1], self.filter_params.shape[1]//min(5, self.filter_params.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Filter value")
            axes.set_zlim(0.0, 2.0)
            plot_x, plot_y = torch.meshgrid(torch.arange(self.filter_params.shape[0]), torch.arange(self.filter_params.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.filter_params.detach().to("cpu"), alpha=1.0)
            logger.add_figure("test/filter_coefficients", figure, 0)
            log_3d(logger, "test/filter_coefficients", self.filter_params, 0, 1.0)

            #Log examples
            for i in range(10):
                sinogram = typing.cast(list[dict[str,torch.Tensor]], outputs)[i]["sinogram"][0,0]
                noisy_sinogram = typing.cast(list[dict[str,torch.Tensor]], outputs)[i]["noisy_sinogram"][0,0]
                filtered_sinogram = radon.radon_filter(sinogram.unsqueeze(0).unsqueeze(0), lambda s,p: s*p, self.filter_params)[0,0]
                ground_truth = typing.cast(list[dict[str,torch.Tensor]], outputs)[i]["ground_truth"][0,0]
                reconstruction = typing.cast(list[dict[str,torch.Tensor]], outputs)[i]["reconstruction"][0,0]
                log_img(logger, "test/sinogram", sinogram.mT, i)
                log_img(logger, "test/noisy_sinogram", noisy_sinogram.mT, i)
                log_img(logger, "test/filtered_sinogram", filtered_sinogram.mT, i)
                log_img(logger, "test/ground_truth", ground_truth, i)
                log_img(logger, "test/reconstruction", reconstruction, i)