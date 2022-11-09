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

import radon

from utils import log_img

import typing
import sys
check_python_version = (sys.version_info[0] >= 3) and (sys.version_info[1] >= 10)


class AnalyticSVDModel(pl.LightningModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config
        self.example_input_array = torch.randn((1,1,len(self.config.sino_angles) if self.config.sino_angles != None else 256,len(self.config.sino_positions) if self.config.sino_positions != None else ceil((self.config.dataset.img_size*1.41421356237)/2.0)*2+1)) #Needed for pytorch lightning

        self.angles = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_angles), requires_grad=False) if self.config.sino_angles != None else None
        self.positions = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_positions), requires_grad=False) if self.config.sino_positions != None else None
        
        matrix = radon.radon_matrix(torch.zeros(self.config.dataset.img_size, self.config.dataset.img_size), thetas=self.angles, positions=self.positions)
        v, d, ut = torch.linalg.svd(matrix, full_matrices=False)
        self.vt  = torch.nn.parameter.Parameter(v.mT, requires_grad=False)
        self.u = torch.nn.parameter.Parameter(ut.mT, requires_grad=False)
        self.filter_params = torch.nn.parameter.Parameter(d, requires_grad=False)
        self.pi = torch.nn.parameter.Parameter(torch.zeros_like(self.filter_params), requires_grad=False)
        self.delta =  torch.nn.parameter.Parameter(torch.zeros_like(self.filter_params), requires_grad=False)
        self.gamma =  torch.nn.parameter.Parameter(torch.zeros_like(self.filter_params), requires_grad=False)
        self.count = 0


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
        #self.filter_params are the \sigma_n (singular values of the Radon-matrix)
        filter_params = (self.filter_params*self.pi - self.gamma)/(self.filter_params**2*self.pi+self.delta+2*self.filter_params*self.gamma)
        # print(x.shape)
        # print(x.reshape(x.shape[0],-1).shape)
        # print(((self.u@torch.diag(filter_params)@self.vt@x.reshape(x.shape[0],-1).mT).mT).shape)
        # print(torch.reshape((self.u@torch.diag(filter_params)@self.vt@x.reshape(x.shape[0],-1).mT).mT, (x.shape[0],1,self.config.dataset.img_size,self.config.dataset.img_size)).shape)

        return torch.reshape((self.u@torch.diag(filter_params)@self.vt@x.reshape(x.shape[0],-1).mT).mT, (x.shape[0],1,self.config.dataset.img_size,self.config.dataset.img_size))



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

        self.pi += torch.sum((self.u.mT@ground_truth.reshape(ground_truth.shape[0],-1).mT)**2, dim=1)
        self.delta += torch.sum((self.vt@noise.reshape(noise.shape[0],-1).mT)**2, dim=1)
        self.gamma += torch.sum((self.u.mT@ground_truth.reshape(ground_truth.shape[0],-1).mT)*(self.vt@noise.reshape(noise.shape[0],-1).mT), dim = 1)
        self.count += sinogram.shape[0]

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
        return torch.zeros((1), requires_grad=True)



    def validation_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> dict[str,torch.Tensor|None if check_python_version else typing.Union[torch.Tensor, None]]:
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



    def validation_epoch_end(self, outputs: list[dict[str,torch.Tensor|list[torch.Tensor] if check_python_version else typing.Union[torch.Tensor, list[torch.Tensor]]]]) -> None:
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment

            #Log validation metrics after each epoch
            logger.add_scalar("validation/loss", self.validation_loss_metric.compute().item(), self.global_step)
            logger.add_scalar("validation/psnr", self.validation_psnr_metric.compute().item(), self.global_step)
            logger.add_scalar("validation/ssim", self.validation_ssim_metric.compute().item(), self.global_step)

            #Log filter coefficients

            filter_params = (self.filter_params*self.pi - self.gamma)/(self.filter_params**2*self.pi+self.delta+2*self.filter_params*self.gamma)
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Analytic coefficients")
            axes.plot(torch.arange(filter_params.shape[0]), filter_params.detach().to("cpu"))
            logger.add_figure("validation/singular_values", figure, self.global_step)

            #Log examples
            sinogram = typing.cast(list[dict[str,torch.Tensor]], outputs)[0]["sinogram"][0,0]
            noisy_sinogram = typing.cast(list[dict[str,torch.Tensor]], outputs)[0]["noisy_sinogram"][0,0]
            ground_truth = typing.cast(list[dict[str,torch.Tensor]], outputs)[0]["ground_truth"][0,0]
            reconstruction = typing.cast(list[dict[str,torch.Tensor]], outputs)[0]["reconstruction"][0,0]
            log_img(logger, "validation/sinogram", sinogram.mT, self.global_step)
            log_img(logger, "validation/noisy_sinogram", noisy_sinogram.mT, self.global_step)
            log_img(logger, "validation/ground_truth", ground_truth, self.global_step)
            log_img(logger, "validation/reconstruction", reconstruction, self.global_step)



    def test_step(self, batch: tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> dict[str,torch.Tensor|list[torch.Tensor] if check_python_version else typing.Union[torch.Tensor, list[torch.Tensor]]]:
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



    def test_epoch_end(self, outputs: list[dict[str,torch.Tensor|list[torch.Tensor] if check_python_version else typing.Union[torch.Tensor, list[torch.Tensor]]]]) -> None:
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment

            #Log mean test metrics
            logger.add_scalar("test/loss", self.test_loss_metric.compute().item(), 0)
            logger.add_scalar("test/psnr", self.test_psnr_metric.compute().item(), 0)
            logger.add_scalar("test/ssim", self.test_ssim_metric.compute().item(), 0)
            logger.add_scalar("test/input_l2", self.test_input_l2_metric.compute().item(), 0)
            logger.add_scalar("test/output_l2", self.test_output_l2_metric.compute().item(), 0)

            #Log filter coefficients
            torch.save(self.filter_params, "singularvals.pt")
            filter_params = (self.filter_params*self.pi - self.gamma)/(self.filter_params**2*self.pi+self.delta+2*self.filter_params*self.gamma)
            torch.save(filter_params, "coefficients.pt")
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Singular value")
            axes.plot(torch.arange(self.filter_params.shape[0]), self.filter_params.detach().to("cpu"))
            logger.add_figure("test/singular_values", figure, 0)

            #Log pi, delta and gamma
            torch.save(self.pi, "pi.pt")
            torch.save(self.delta, "delta.pt")
            torch.save(self.gamma, "gamma.pt")

            #Log examples
            for i in range(10):
                sinogram = typing.cast(list[dict[str,torch.Tensor]], outputs)[i]["sinogram"][0,0]
                noisy_sinogram = typing.cast(list[dict[str,torch.Tensor]], outputs)[i]["noisy_sinogram"][0,0]
                ground_truth = typing.cast(list[dict[str,torch.Tensor]], outputs)[i]["ground_truth"][0,0]
                reconstruction = typing.cast(list[dict[str,torch.Tensor]], outputs)[i]["reconstruction"][0,0]
                log_img(logger, "test/sinogram", sinogram.mT, i)
                log_img(logger, "test/noisy_sinogram", noisy_sinogram.mT, i)
                log_img(logger, "test/ground_truth", ground_truth, i)
                log_img(logger, "test/reconstruction", reconstruction, i)