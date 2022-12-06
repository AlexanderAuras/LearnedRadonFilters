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

import radon as radon

from utils import log_img



class SVDModel(pl.LightningModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config
        self.example_input_array = torch.randn((1,1,len(self.config.sino_angles) if self.config.sino_angles != None else 256,len(self.config.sino_positions) if self.config.sino_positions != None else ceil((self.config.dataset.img_size*1.41421356237)/2.0)*2+1)) #Needed for pytorch lightning
        self.automatic_optimization = False

        self.angles = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_angles), requires_grad=False) if self.config.sino_angles != None else None
        self.positions = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_positions), requires_grad=False) if self.config.sino_positions != None else None
        
        matrix = radon.radon_matrix(torch.zeros(self.config.dataset.img_size, self.config.dataset.img_size), thetas=self.angles, positions=self.positions)
        v, d, ut = torch.linalg.svd(matrix, full_matrices=False)
        self.vt = torch.nn.parameter.Parameter(v.mT, requires_grad=False)
        torch.save(d, "singular_values.pt")
        self.u = torch.nn.parameter.Parameter(ut.mT, requires_grad=False)
        self.split_filter_params = []
        if self.config.model.initialization == "zeros":
            for i in range(d.shape[0]):
                setattr(self, f"split_filter_params{i}", torch.nn.parameter.Parameter(torch.zeros((1,))))
                self.split_filter_params.append(getattr(self, f"split_filter_params{i}"))
        elif self.config.model.initialization == "ones":
            for i in range(d.shape[0]):
                setattr(self, f"split_filter_params{i}", torch.nn.parameter.Parameter(torch.ones((1,))))
                self.split_filter_params.append(getattr(self, f"split_filter_params{i}"))
        elif self.config.model.initialization == "randn":
            for i in range(d.shape[0]):
                setattr(self, f"split_filter_params{i}", torch.nn.parameter.Parameter(torch.randn((1,)).abs()))
                self.split_filter_params.append(getattr(self, f"split_filter_params{i}"))
        elif self.config.model.initialization == "rand":
            for i in range(d.shape[0]):
                setattr(self, f"split_filter_params{i}", torch.nn.parameter.Parameter(torch.rand((1,))))
                self.split_filter_params.append(getattr(self, f"split_filter_params{i}"))
        elif self.config.model.initialization == "path":
            init_data = torch.load(self.config.model.initialization_path)
            if isinstance(init_data, torch.nn.parameter.Parameter):
                init_data = torch.nn.utils.convert_parameters.parameters_to_vector(init_data).reshape(init_data.shape)
                for i in range(d.shape[0]):
                    setattr(self, f"split_filter_params{i}", torch.nn.parameter.Parameter(torch.full((1,), init_data[i].item())))
                    self.split_filter_params.append(getattr(self, f"split_filter_params{i}"))
        else:
            raise NotImplementedError()
        self.singular_values = torch.nn.parameter.Parameter(d, requires_grad=False)
        self.pi = torch.nn.parameter.Parameter(torch.zeros_like(self.singular_values), requires_grad=False)
        self.delta = torch.nn.parameter.Parameter(torch.zeros_like(self.singular_values), requires_grad=False)
        self.gamma = torch.nn.parameter.Parameter(torch.zeros_like(self.singular_values), requires_grad=False)
        self.count = 0

        #Setup metrics
        with warnings.catch_warnings():
            self.training_learned_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_learned_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_learned_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_learned_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_learned_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_learned_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_input_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_output_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_analytic_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_analytic_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_analytic_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_analytic_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_analytic_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_analytic_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_input_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_output_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")



    #HACK Removes metrics from PyTorch Lightning overview
    def named_children(self) -> typing.Iterator[typing.Tuple[str, nn.Module]]:
        stack = inspect.stack()
        if stack[2].function == "summarize" and stack[2].filename.endswith("pytorch_lightning/utilities/model_summary/model_summary.py"):
            return filter(lambda x: not x[0].endswith("metric"), super().named_children())
        return super().named_children()



    #Common forward method used by forward, training_step, validation_step and test_step
    def forward_learned(self, x: torch.Tensor) -> torch.Tensor:
        all_filter_params = torch.concat(typing.cast(typing.List[torch.Tensor], self.split_filter_params))
        return torch.reshape(self.u@torch.diag(all_filter_params)@self.vt@x.reshape(x.shape[0],-1,1), (x.shape[0],1,self.config.dataset.img_size,self.config.dataset.img_size))
    
    
    
    def forward_analytic(self, x: torch.Tensor) -> torch.Tensor:
        filter_params = (self.singular_values*self.pi-self.gamma)/(self.singular_values**2*self.pi+self.delta+2*self.singular_values*self.gamma)
        return torch.reshape(self.u@torch.diag(filter_params)@self.vt@x.reshape(x.shape[0],-1,1), (x.shape[0],1,self.config.dataset.img_size,self.config.dataset.img_size))



    #Apply model for n iterations
    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        return self.forward_learned(sino)



    def configure_optimizers(self):
        ##############################################################
        #CHANGE LEARNING RATES HERE
        #Example (indexing might be wrong):
        #
        #learning_rates = torch.linspace(0.0, 1.0, len(self.split_filter_params))+1.0e-2 #Linear
        #learning_rates = 1.0/(float(len(self.split_filter_params))-torch.linspace(0.0, 1.0, len(self.split_filter_params))) #Should be similar to 1/σ
        #optimizer = torch.optim.Adam([{"params": params, "lr": lr} for params, lr in zip(self.split_filter_params, learning_rates.tolist())], lr=self.config.optimizer_lr)
        #
        ##############################################################
        optimizer = torch.optim.Adam([{"params": params, "lr": lr} for params, lr in zip(self.split_filter_params, torch.full((len(self.split_filter_params),), self.config.optimizer_lr).tolist())], lr=self.config.optimizer_lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 2.0)
        #return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer
    


    def training_step(self, batch: typing.Tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> None:#torch.Tensor:
        #Reset metrics
        self.training_learned_loss_metric.reset()
        self.training_learned_psnr_metric.reset()
        self.training_learned_ssim_metric.reset()
        self.training_analytic_loss_metric.reset()
        self.training_analytic_psnr_metric.reset()
        self.training_analytic_ssim_metric.reset()

        #Forward pass
        ground_truth, _ = batch
        sinogram = radon.radon_forward(ground_truth, torch.tensor(self.config.sino_angles, device=ground_truth.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=ground_truth.device) if self.config.sino_positions != None else None)
        noise = self.config.noise_level*torch.randn_like(sinogram)
        noisy_sinogram = sinogram+noise
        learned_reconstruction = self.forward_learned(noisy_sinogram)
        learned_loss = F.mse_loss(learned_reconstruction, ground_truth)
        self.training_learned_loss_metric.update(learned_loss.item())
        self.training_learned_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(learned_reconstruction, ground_truth))
        self.training_learned_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(learned_reconstruction, ground_truth)))
        self.pi += torch.sum((self.u.mT@ground_truth.reshape(ground_truth.shape[0],-1).mT)**2, dim=1)
        self.delta += torch.sum((self.vt@noise.reshape(noise.shape[0],-1).mT)**2, dim=1)
        self.gamma += torch.sum((self.u.mT@ground_truth.reshape(ground_truth.shape[0],-1).mT)*(self.vt@noise.reshape(noise.shape[0],-1).mT), dim = 1)
        self.count += sinogram.shape[0]
        analytic_reconstruction = self.forward_analytic(noisy_sinogram)
        analytic_loss = F.mse_loss(analytic_reconstruction, ground_truth)
        self.training_analytic_loss_metric.update(analytic_loss.item())
        self.training_analytic_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(analytic_reconstruction, ground_truth))
        self.training_analytic_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(analytic_reconstruction, ground_truth)))

        #Log training metrics after each batch
        if self.logger:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment
            logger.add_scalar("training/learned_loss", self.training_learned_loss_metric.compute().item(), self.global_step)
            logger.add_scalar("training/learned_psnr", self.training_learned_psnr_metric.compute().item(), self.global_step)
            logger.add_scalar("training/learned_ssim", self.training_learned_ssim_metric.compute().item(), self.global_step)
            logger.add_scalar("training/analytic_loss", self.training_analytic_loss_metric.compute().item(), self.global_step)
            logger.add_scalar("training/analytic_psnr", self.training_analytic_psnr_metric.compute().item(), self.global_step)
            logger.add_scalar("training/analytic_ssim", self.training_analytic_ssim_metric.compute().item(), self.global_step)

        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            [typing.cast(torch.optim.Optimizer, optimizer).zero_grad() for optimizer in optimizers]
        else:
            typing.cast(torch.optim.Optimizer, optimizers).zero_grad()
        self.manual_backward(learned_loss)
        ##############################################################
        #CHANGE GRADIENT HERE
        #Example (indexing might be wrong):
        #
        #for i in range(len(self.split_filter_params)):
        #   self.split_filter_params[i].grad *= 1.0/i
        #
        ##############################################################
        if isinstance(optimizers, list):
            [optimizer.step() for optimizer in optimizers]
        else:
            optimizers.step()



    def validation_step(self, batch: typing.Tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> typing.Dict[str,typing.Union[torch.Tensor,None]]:
        #Reset metrics
        self.validation_learned_loss_metric.reset()
        self.validation_learned_psnr_metric.reset()
        self.validation_learned_ssim_metric.reset()
        self.validation_analytic_loss_metric.reset()
        self.validation_analytic_psnr_metric.reset()
        self.validation_analytic_ssim_metric.reset()

        #Forward pass
        ground_truth, _ = batch
        sinogram = radon.radon_forward(ground_truth, torch.tensor(self.config.sino_angles, device=ground_truth.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=ground_truth.device) if self.config.sino_positions != None else None)
        noisy_sinogram = sinogram+self.config.noise_level*torch.randn_like(sinogram)
        learned_reconstruction = self.forward_learned(noisy_sinogram)
        self.validation_learned_loss_metric.update(F.mse_loss(learned_reconstruction, ground_truth))
        self.validation_learned_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(learned_reconstruction, ground_truth))
        self.validation_learned_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(learned_reconstruction, ground_truth)))
        analytic_reconstruction = self.forward_analytic(noisy_sinogram)
        self.validation_analytic_loss_metric.update(F.mse_loss(analytic_reconstruction, ground_truth))
        self.validation_analytic_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(analytic_reconstruction, ground_truth))
        self.validation_analytic_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(analytic_reconstruction, ground_truth)))

        #Return data for logging purposes
        if batch_idx == 0:
            return {
                "sinogram": sinogram, 
                "noisy_sinogram": noisy_sinogram, 
                "ground_truth": ground_truth, 
                "learned_reconstruction": learned_reconstruction,
                "analytic_reconstruction": analytic_reconstruction
            }
        else:
            return {}



    def validation_epoch_end(self, outputs: typing.List[typing.Dict[str,typing.Union[torch.Tensor,typing.List[torch.Tensor]]]]) -> None:
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment

            #Log validation metrics after each epoch
            logger.add_scalar("validation/learned_loss", self.validation_learned_loss_metric.compute().item(), self.global_step)
            logger.add_scalar("validation/learned_psnr", self.validation_learned_psnr_metric.compute().item(), self.global_step)
            logger.add_scalar("validation/learned_ssim", self.validation_learned_ssim_metric.compute().item(), self.global_step)
            logger.add_scalar("validation/analytic_loss", self.validation_analytic_loss_metric.compute().item(), self.global_step)
            logger.add_scalar("validation/analytic_psnr", self.validation_analytic_psnr_metric.compute().item(), self.global_step)
            logger.add_scalar("validation/analytic_ssim", self.validation_analytic_ssim_metric.compute().item(), self.global_step)

            #Log learned coefficients
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            all_filter_params = torch.concat(typing.cast(typing.List[torch.Tensor], self.split_filter_params))
            axes.plot(torch.arange(all_filter_params.shape[0]), all_filter_params.detach().to("cpu"))
            logger.add_figure("validation/learned_coefficients", figure, self.global_step)

            #Log pi
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            axes.plot(torch.arange(self.pi.shape[0]), self.pi.detach().to("cpu"))
            logger.add_figure("validation/pi", figure, self.global_step)

            #Log delta
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            axes.plot(torch.arange(self.delta.shape[0]), self.delta.detach().to("cpu"))
            logger.add_figure("validation/delta", figure, self.global_step)

            #Log gamma
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            axes.plot(torch.arange(self.gamma.shape[0]), self.gamma.detach().to("cpu"))
            logger.add_figure("validation/gamma", figure, self.global_step)

            #Log analytic coefficients
            filter_params = (self.singular_values*self.pi-self.gamma)/(self.singular_values**2*self.pi+self.delta+2*self.singular_values*self.gamma)
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            axes.plot(torch.arange(filter_params.shape[0]), filter_params.detach().to("cpu"))
            logger.add_figure("validation/analytic_coefficients", figure, self.global_step)

            #Log examples
            sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["sinogram"][0,0]
            noisy_sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["noisy_sinogram"][0,0]
            ground_truth = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["ground_truth"][0,0]
            learned_reconstruction = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["learned_reconstruction"][0,0]
            analytic_reconstruction = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["analytic_reconstruction"][0,0]
            log_img(logger, "validation/sinogram", sinogram.mT, self.global_step)
            log_img(logger, "validation/noisy_sinogram", noisy_sinogram.mT, self.global_step)
            log_img(logger, "validation/ground_truth", ground_truth, self.global_step)
            log_img(logger, "validation/learned_reconstruction", learned_reconstruction, self.global_step)
            log_img(logger, "validation/analytic_reconstruction", analytic_reconstruction, self.global_step)



    def test_step(self, batch: typing.Tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> typing.Dict[str,typing.Union[torch.Tensor,typing.List[torch.Tensor]]]:
        #Reset metrics
        self.test_learned_loss_metric.reset()
        self.test_learned_psnr_metric.reset()
        self.test_learned_ssim_metric.reset()
        self.test_analytic_loss_metric.reset()
        self.test_analytic_psnr_metric.reset()
        self.test_analytic_ssim_metric.reset()

        #Forward pass
        ground_truth, _ = batch
        sinogram = radon.radon_forward(ground_truth, torch.tensor(self.config.sino_angles, device=ground_truth.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=ground_truth.device) if self.config.sino_positions != None else None)
        noisy_sinogram = sinogram+self.config.noise_level*torch.randn_like(sinogram)
        learned_reconstruction = self.forward_learned(noisy_sinogram)
        self.test_learned_loss_metric.update(F.mse_loss(learned_reconstruction, ground_truth))
        self.test_learned_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(learned_reconstruction, ground_truth))
        self.test_learned_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(learned_reconstruction, ground_truth)))
        self.test_learned_input_l2_metric.update(torch.sqrt(torch.sum(ground_truth**2, 3).sum(2)).mean())
        self.test_learned_output_l2_metric.update(torch.sqrt(torch.sum(learned_reconstruction**2, 3).sum(2)).mean())
        analytic_reconstruction = self.forward_analytic(noisy_sinogram)
        self.test_analytic_loss_metric.update(F.mse_loss(analytic_reconstruction, ground_truth))
        self.test_analytic_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(analytic_reconstruction, ground_truth))
        self.test_analytic_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(analytic_reconstruction, ground_truth)))
        self.test_analytic_input_l2_metric.update(torch.sqrt(torch.sum(ground_truth**2, 3).sum(2)).mean())
        self.test_analytic_output_l2_metric.update(torch.sqrt(torch.sum(analytic_reconstruction**2, 3).sum(2)).mean())

        #Return data for logging purposes
        if batch_idx < 10:
            return {
                "sinogram": sinogram, 
                "noisy_sinogram": noisy_sinogram, 
                "ground_truth": ground_truth,
                "learned_reconstruction": learned_reconstruction,
                "analytic_reconstruction": analytic_reconstruction
            }
        else:
            return {}



    def test_epoch_end(self, outputs: typing.List[typing.Dict[str,typing.Union[torch.Tensor,typing.List[torch.Tensor]]]]) -> None:
        all_filter_params = torch.concat(typing.cast(typing.List[torch.Tensor], self.split_filter_params))
        torch.save(torch.nn.utils.convert_parameters.parameters_to_vector(all_filter_params).reshape(all_filter_params.shape), "coefficients.pt")
        torch.save(torch.nn.utils.convert_parameters.parameters_to_vector(self.pi).reshape(self.pi.shape)/max(self.count, 1), "pi.pt")
        torch.save(torch.nn.utils.convert_parameters.parameters_to_vector(self.delta).reshape(self.delta.shape)/max(self.count, 1), "delta.pt")
        torch.save(torch.nn.utils.convert_parameters.parameters_to_vector(self.gamma).reshape(self.gamma.shape)/max(self.count, 1), "gamma.pt")
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment

            #Log mean test metrics
            logger.add_scalar("test/learned_loss", self.test_learned_loss_metric.compute().item(), 0)
            logger.add_scalar("test/learned_psnr", self.test_learned_psnr_metric.compute().item(), 0)
            logger.add_scalar("test/learned_ssim", self.test_learned_ssim_metric.compute().item(), 0)
            logger.add_scalar("test/learned_input_l2", self.test_learned_input_l2_metric.compute().item(), 0)
            logger.add_scalar("test/learned_output_l2", self.test_learned_output_l2_metric.compute().item(), 0)
            logger.add_scalar("test/analytic_loss", self.test_analytic_loss_metric.compute().item(), 0)
            logger.add_scalar("test/analytic_psnr", self.test_analytic_psnr_metric.compute().item(), 0)
            logger.add_scalar("test/analytic_ssim", self.test_analytic_ssim_metric.compute().item(), 0)
            logger.add_scalar("test/analytic_input_l2", self.test_analytic_input_l2_metric.compute().item(), 0)
            logger.add_scalar("test/analytic_output_l2", self.test_analytic_output_l2_metric.compute().item(), 0)

            #Log learned coefficients
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            axes.plot(torch.arange(all_filter_params.shape[0]), all_filter_params.detach().to("cpu"))
            logger.add_figure("test/learned_coefficients", figure, 0)

            #Log pi
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            axes.plot(torch.arange(self.pi.shape[0]), self.pi.detach().to("cpu"))
            logger.add_figure("validation/pi", figure, 0)

            #Log delta
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            axes.plot(torch.arange(self.delta.shape[0]), self.delta.detach().to("cpu"))
            logger.add_figure("test/delta", figure, 0)

            #Log gamma
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            axes.plot(torch.arange(self.gamma.shape[0]), self.gamma.detach().to("cpu"))
            logger.add_figure("test/gamma", figure, 0)

            #Log analytic coefficients
            filter_params = (self.singular_values*self.pi-self.gamma)/(self.singular_values**2*self.pi+self.delta+2*self.singular_values*self.gamma)
            figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1)
            axes.set_xlabel("Index")
            axes.set_ylabel("Coefficient")
            axes.plot(torch.arange(filter_params.shape[0]), filter_params.detach().to("cpu"))
            logger.add_figure("test/analytic_coefficients", figure, 0)

            #Log examples
            for i in range(10):
                sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["sinogram"][0,0]
                noisy_sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["noisy_sinogram"][0,0]
                ground_truth = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["ground_truth"][0,0]
                learned_reconstruction = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["learned_reconstruction"][0,0]
                analytic_reconstruction = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["analytic_reconstruction"][0,0]
                log_img(logger, "test/sinogram", sinogram.mT, i)
                log_img(logger, "test/noisy_sinogram", noisy_sinogram.mT, i)
                log_img(logger, "test/ground_truth", ground_truth, i)
                log_img(logger, "test/learned_reconstruction", learned_reconstruction, i)
                log_img(logger, "test/analytic_reconstruction", analytic_reconstruction, i)