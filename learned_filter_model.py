import inspect
from math import ceil
from operator import imod
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

from utils import extract_tensor, log_3d, log_img



class LearnedFilterModel(pl.LightningModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config
        self.example_input_array = torch.randn((1,1,len(self.config.sino_angles) if self.config.sino_angles != None else 256,len(self.config.sino_positions) if self.config.sino_positions != None else ceil((self.config.dataset.img_size*1.41421356237)/2.0)*2+1)) #Needed for pytorch lightning

        self.filter_params = torch.nn.parameter.Parameter(
            torch.zeros((
                len(self.config.sino_angles) if self.config.sino_angles != None else 256,
                int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
            ))
        )
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
            self.training_rel_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_rel_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_rel_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")

        self.pi = torch.nn.parameter.Parameter(torch.zeros((
            self.config.training_batch_size,
            1,
            len(self.config.sino_angles) if self.config.sino_angles != None else 256,
            int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
        ), dtype=torch.complex64), requires_grad=False)
        self.delta = torch.nn.parameter.Parameter(torch.zeros((
            self.config.training_batch_size,
            1,
            len(self.config.sino_angles) if self.config.sino_angles != None else 256,
            int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
        ), dtype=torch.complex64), requires_grad=False)
        self.count = 0



    #HACK Removes metrics from PyTorch Lightning overview
    def named_children(self) -> typing.Iterator[tuple[str, nn.Module]]:
        stack = inspect.stack()
        if stack[2].function == "summarize" and stack[2].filename.endswith("pytorch_lightning/utilities/model_summary/model_summary.py"):
            return filter(lambda x: not x[0].endswith("metric"), super().named_children())
        return super().named_children()

    #TODO Gaussian on image
    #TODO Set 0 and pi/2 to ramp or other angle?
    #TODO Rotate everything?
    #TODO Compare to ramp
    #TODO Single ellipse, changes w.r.t. angle of ellipse
    #TODO Paper: Remove zeros



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
        self.training_rel_l2_metric.reset()

        #Forward pass
        img, _ = batch
        sinogram = radon.radon_forward(img, torch.tensor(self.config.sino_angles, device=img.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=img.device) if self.config.sino_positions != None else None)
        noise = self.config.noise_level*torch.randn_like(sinogram)
        noisy_sinogram = sinogram+noise
        self.pi += torch.fft.rfft(sinogram, dim=3, norm="forward")**2
        self.delta += torch.fft.rfft(noise, dim=3, norm="forward")**2
        self.count += 1
        ground_truth = img
        reconstruction = self.forward_intern(noisy_sinogram)
        loss = F.mse_loss(reconstruction, ground_truth)
        self.training_loss_metric.update(loss.item())
        self.training_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(reconstruction, ground_truth))
        self.training_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(reconstruction, ground_truth)))
        self.training_rel_l2_metric.update(torch.sqrt(torch.sum((ground_truth-reconstruction)**2))/torch.sqrt(torch.sum(ground_truth**2)))

        #Log training metrics after each batch
        if self.logger:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment
            logger.add_scalar("training/loss", self.training_loss_metric.compute().item(), self.global_step)
            logger.add_scalar("training/psnr", self.training_psnr_metric.compute().item(), self.global_step)
            logger.add_scalar("training/ssim", self.training_ssim_metric.compute().item(), self.global_step)
            logger.add_scalar("training/rel_l2", self.training_rel_l2_metric.compute().item(), self.global_step)
        return loss



    def validation_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> dict[str,torch.Tensor|None]:
        #Reset metrics
        self.validation_loss_metric.reset()
        self.validation_psnr_metric.reset()
        self.validation_ssim_metric.reset()
        self.validation_rel_l2_metric.reset()

        #Forward pass
        img, _ = batch
        sinogram = radon.radon_forward(img, torch.tensor(self.config.sino_angles, device=img.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=img.device) if self.config.sino_positions != None else None)
        noisy_sinogram = sinogram+self.config.noise_level*torch.randn_like(sinogram)
        ground_truth = img
        reconstruction = self.forward_intern(noisy_sinogram)
        self.validation_loss_metric.update(F.mse_loss(reconstruction, ground_truth))
        self.validation_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(reconstruction, ground_truth))
        self.validation_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(reconstruction, ground_truth)))
        self.validation_rel_l2_metric.update(torch.sqrt(torch.sum((ground_truth-reconstruction)**2))/torch.sqrt(torch.sum(ground_truth**2)))

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
            logger.add_scalar("validation/rel_l2", self.validation_rel_l2_metric.compute().item(), self.global_step)

            #Log filter coefficients
            figure = plt.figure()
            axes: mpl_toolkits.mplot3d.Axes3D = figure.add_subplot(1, 1, 1, projection="3d")
            plot_x, plot_y = torch.meshgrid(torch.arange(self.filter_params.shape[0]), torch.arange(self.filter_params.shape[1]), indexing="ij")
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//5).to(torch.float32).tolist(), list(map(lambda x: f"{x/self.filter_params.shape[0]:3.2f} \u03C0", torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//5).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.filter_params.shape[1], self.filter_params.shape[1]//5).to(torch.float32).tolist())
            axes.set_zlabel("Filter value")
            axes.set_zlim(0.0, 2.0)
            axes.plot_surface(plot_x, plot_y, self.filter_params.detach().to("cpu"), alpha=1.0)
            logger.add_figure("validation/filter_coefficients", figure, self.global_step)
            log_3d(logger, "validation/filter_coefficients", self.filter_params, self.global_step, 1.0)

            #Log examples
            sinogram = extract_tensor(outputs, "sinogram", 0)
            noisy_sinogram = extract_tensor(outputs, "noisy_sinogram", 0)
            filtered_sinogram = radon.radon_filter(sinogram.unsqueeze(0).unsqueeze(0), lambda s,p: s*p, self.filter_params)[0,0]
            ground_truth = extract_tensor(outputs, "ground_truth", 0)
            reconstruction = extract_tensor(outputs, "reconstruction", 0)
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
        self.test_rel_l2_metric.reset()

        #Forward pass
        img, _ = batch
        sinogram = radon.radon_forward(img, torch.tensor(self.config.sino_angles, device=img.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=img.device) if self.config.sino_positions != None else None)
        noisy_sinogram = sinogram+self.config.noise_level*torch.randn_like(sinogram)
        ground_truth = img
        reconstruction = self.forward_intern(noisy_sinogram)
        self.test_loss_metric.update(F.mse_loss(reconstruction, ground_truth))
        self.test_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(reconstruction, ground_truth))
        self.test_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(reconstruction, ground_truth)))
        self.test_rel_l2_metric.update(torch.sqrt(torch.sum((ground_truth-reconstruction)**2))/torch.sqrt(torch.sum(ground_truth**2)))

        #Return data for logging purposes
        if batch_idx < 10:
            return {"sinogram": sinogram, "noisy_sinogram": noisy_sinogram, "ground_truth": ground_truth, "reconstruction": reconstruction}
        else:
            return {}



    def test_epoch_end(self, outputs: list[dict[str,torch.Tensor|list[torch.Tensor]]]) -> None:
        torch.save(self.filter_params, "coefficients.pt")
        torch.save(self.pi.sum(0)[0]/self.count, "pi.pt")
        torch.save(self.delta.sum(0)[0]/self.count, "delta.pt")
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment

            #Log mean test metrics
            logger.add_scalar("test/loss", self.test_loss_metric.compute().item(), 0)
            logger.add_scalar("test/psnr", self.test_psnr_metric.compute().item(), 0)
            logger.add_scalar("test/ssim", self.test_ssim_metric.compute().item(), 0)
            logger.add_scalar("test/rel_l2", self.test_rel_l2_metric.compute().item(), 0)

            #Log filter coefficients
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//5).to(torch.float32).tolist(), list(map(lambda x: f"{x/self.filter_params.shape[0]:3.2f} \u03C0", torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//5).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.filter_params.shape[1], self.filter_params.shape[1]//5).to(torch.float32).tolist())
            axes.set_zlabel("Filter value")
            axes.set_zlim(0.0, 2.0)
            plot_x, plot_y = torch.meshgrid(torch.arange(self.filter_params.shape[0]), torch.arange(self.filter_params.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.filter_params.detach().to("cpu"), alpha=1.0)
            logger.add_figure("test/filter_coefficients", figure, 0)
            log_3d(logger, "test/filter_coefficients", self.filter_params, 0, 1.0)

            #Log examples
            for i in range(10):
                sinogram = extract_tensor(outputs, "sinogram", i)
                noisy_sinogram = extract_tensor(outputs, "noisy_sinogram", i)
                filtered_sinogram = radon.radon_filter(sinogram.unsqueeze(0).unsqueeze(0), lambda s,p: s*p, self.filter_params)[0,0]
                ground_truth = extract_tensor(outputs, "ground_truth", i)
                reconstruction = extract_tensor(outputs, "reconstruction", i)
                log_img(logger, "test/sinogram", sinogram.mT, i)
                log_img(logger, "test/noisy_sinogram", noisy_sinogram.mT, i)
                log_img(logger, "test/filtered_sinogram", filtered_sinogram.mT, i)
                log_img(logger, "test/ground_truth", ground_truth, i)
                log_img(logger, "test/reconstruction", reconstruction, i)