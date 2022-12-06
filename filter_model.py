import inspect
from math import ceil, sqrt
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

import radon as radon

from utils import log_3d, log_img



class FilterModel(pl.LightningModule):
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
        elif self.config.model.initialization == "ramp":
            positions_count = len(self.config.sino_positions) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2)*2+1
            angles_count = len(self.config.sino_angles) if self.config.sino_angles != None else 256
            self.filter_params = torch.nn.parameter.Parameter(
                torch.abs(torch.arange(0, int(positions_count//2+1))).to(torch.float32).repeat(angles_count,1)*2*ceil(sqrt(2.0)/2.0)*positions_count/(positions_count-1)/angles_count*self.config.dataset.img_size*2*ceil(sqrt(2.0)*self.config.dataset.img_size/2.0)/positions_count
            )
        elif self.config.model.initialization == "path":
            init_data = torch.load(self.config.model.initialization_path)
            if isinstance(init_data, torch.nn.parameter.Parameter):
                init_data = torch.nn.utils.convert_parameters.parameters_to_vector(init_data).reshape(init_data.shape)
            self.filter_params = torch.nn.parameter.Parameter(init_data)
        else:
            raise NotImplementedError()
        self.angles = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_angles), requires_grad=False) if self.config.sino_angles != None else None
        self.positions = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_positions), requires_grad=False) if self.config.sino_positions != None else None
        self.layers = nn.Sequential(
            radon.RadonFilter(lambda sino, params: sino*params, self.filter_params),
            radon.RadonBackward(self.config.dataset.img_size, self.angles, self.positions)
        )
        self.pi = torch.nn.parameter.Parameter(torch.zeros((
            len(self.config.sino_angles) if self.config.sino_angles != None else 256,
            int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
        ), dtype=torch.float32), requires_grad=False)
        self.delta = torch.nn.parameter.Parameter(torch.zeros((
            len(self.config.sino_angles) if self.config.sino_angles != None else 256,
            int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
        ), dtype=torch.float32), requires_grad=False)
        self.gamma = torch.nn.parameter.Parameter(torch.zeros((
            len(self.config.sino_angles) if self.config.sino_angles != None else 256,
            int(len(self.config.sino_positions)//2+1) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2.0)+1
        ), dtype=torch.float32), requires_grad=False)
        self.count = 0

        #Setup metrics
        with warnings.catch_warnings():
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
            
        positions_count = len(self.config.sino_positions) if self.config.sino_positions != None else ceil(self.config.dataset.img_size*1.41421356237/2)*2+1
        angles_count = len(self.config.sino_angles) if self.config.sino_angles != None else 256
        self.ramp = torch.nn.parameter.Parameter(
            torch.abs(torch.arange(0, int(positions_count//2+1))).to(torch.float32).repeat(angles_count,1)*2*ceil(sqrt(2.0)/2.0)*positions_count/(positions_count-1)/angles_count*self.config.dataset.img_size*2*ceil(sqrt(2.0)*self.config.dataset.img_size/2.0)/positions_count, requires_grad=False
            )
        self.ramp[:,0] = 0.25
        #TODO Upload coefficients
        #self.ramp = torch.nn.parameter.Parameter(torch.load("/home/kabri/Documents/LearnedRadonFilters/results/fft_high_learned/noise_level=0/coefficients.pt"), requires_grad=False)



    #HACK Removes metrics from PyTorch Lightning overview
    def named_children(self) -> typing.Iterator[typing.Tuple[str, nn.Module]]:
        stack = inspect.stack()
        if stack[2].function == "summarize" and stack[2].filename.endswith("pytorch_lightning/utilities/model_summary/model_summary.py"):
            return filter(lambda x: not x[0].endswith("metric"), super().named_children())
        return super().named_children()



    #Common forward method used by forward, training_step, validation_step and test_step
    def forward_learned(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


    
    def forward_analytic(self, x: torch.Tensor) -> torch.Tensor:
        filter_params = self.ramp*(self.pi-self.gamma)/(self.pi+self.delta+2*self.gamma)
        #positions_count = self.positions.shape[0] if self.positions != None else ceil(sqrt(2.0)*self.config.dataset.img_size/2.0)*2.0+1
        #angle_count = self.angles.shape[0] if self.angles != None else 256
        #filter_params *= 2.0*ceil(sqrt(2.0)/2.0)*positions_count/(positions_count-1)/angle_count#*self.config.dataset.img_size#*2*ceil(sqrt(2.0)*self.config.dataset.img_size/2.0)/positions_count
        filtered_sinogram = radon.radon_filter(x, lambda sino, params: sino*params, filter_params)
        return radon.radon_backward(filtered_sinogram, self.config.dataset.img_size, self.angles, self.positions)



    #Apply model for n iterations
    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        return self.forward_learned(sino)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optimizer_lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 2.0)
        #return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer
    


    def training_step(self, batch: typing.Tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> torch.Tensor:
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
        self.pi += torch.sum(torch.fft.rfft(sinogram, dim=3, norm="forward").abs()**2, dim=0)[0]
        self.delta += torch.sum(torch.fft.rfft(noise, dim=3, norm="forward").abs()**2, dim=0)[0]
        self.gamma += torch.sum(torch.fft.rfft(sinogram, dim = 3, norm = "forward").real*torch.fft.rfft(noise, dim=3, norm="forward").imag + torch.fft.rfft(sinogram, dim = 3, norm = "forward").imag * torch.fft.rfft(noise, dim=3, norm="forward").real, dim = 0)[0]
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
        return learned_loss



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

            #Log learned filter coefficients
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            plot_x, plot_y = torch.meshgrid(torch.arange(self.filter_params.shape[0]), torch.arange(self.filter_params.shape[1]), indexing="ij")
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//min(5, self.filter_params.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/self.filter_params.shape[0]:3.2f} \u03C0", torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//min(5, self.filter_params.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.filter_params.shape[1], self.filter_params.shape[1]//min(5, self.filter_params.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Filter value")
            axes.set_zlim(0.0, 2.0)
            axes.plot_surface(plot_x, plot_y, self.filter_params.detach().to("cpu").numpy(), alpha=1.0)
            logger.add_figure("validation/learned_filter_coefficients", figure, self.global_step)
            log_3d(logger, "validation/learned_filter_coefficients", self.filter_params, self.global_step, 1.0)
            log_img(logger, "validation/_learned_filter_coefficients", self.filter_params.mT, self.global_step, True)

            #Log analytic filter coefficients
            filter_params = self.ramp*(self.pi-self.gamma)/(self.pi+self.delta+2*self.gamma)
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            plot_x, plot_y = torch.meshgrid(torch.arange(filter_params.shape[0]), torch.arange(filter_params.shape[1]), indexing="ij")
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, filter_params.shape[0], filter_params.shape[0]//min(5, filter_params.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/filter_params.shape[0]:3.2f} \u03C0", torch.arange(0, filter_params.shape[0], filter_params.shape[0]//min(5, filter_params.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, filter_params.shape[1], filter_params.shape[1]//min(5, filter_params.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Filter value")
            axes.set_zlim(0.0, 2.0)
            axes.plot_surface(plot_x, plot_y, filter_params.detach().to("cpu").numpy(), alpha=1.0)
            logger.add_figure("validation/analytic_filter_coefficients", figure, self.global_step)
            log_3d(logger, "validation/analytic_filter_coefficients", filter_params, self.global_step, 1.0)
            log_img(logger, "validation/_analytic_filter_coefficients", filter_params.mT, self.global_step, True)


            #Log pi
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.pi.shape[0], self.pi.shape[0]//min(5, self.pi.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/self.pi.shape[0]:3.2f} \u03C0", torch.arange(0, self.pi.shape[0], self.pi.shape[0]//min(5, self.pi.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.pi.shape[1], self.pi.shape[1]//min(5, self.pi.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Pi")
            plot_x, plot_y = torch.meshgrid(torch.arange(self.pi.shape[0]), torch.arange(self.pi.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.pi.detach().to("cpu").numpy()/max(self.count, 1), alpha=1.0)
            logger.add_figure("validation/pi", figure, self.global_step)
            log_3d(logger, "validation/pi", self.pi/max(self.count, 1), self.global_step, 1.0)
            log_img(logger, "validation/_pi", self.pi.mT/max(self.count, 1), self.global_step, True)

            #Log delta
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.delta.shape[0], self.delta.shape[0]//min(5, self.delta.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/self.delta.shape[0]:3.2f} \u03C0", torch.arange(0, self.delta.shape[0], self.delta.shape[0]//min(5, self.delta.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.delta.shape[1], self.delta.shape[1]//min(5, self.delta.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Delta")
            plot_x, plot_y = torch.meshgrid(torch.arange(self.delta.shape[0]), torch.arange(self.delta.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.delta.detach().to("cpu").numpy()/max(self.count, 1), alpha=1.0)
            logger.add_figure("validation/delta", figure, self.global_step)
            log_3d(logger, "validation/delta", self.delta/max(self.count, 1), self.global_step, 1.0)
            log_img(logger, "validation/_delta", self.delta.mT/max(self.count, 1), self.global_step, True)

            #Log gamma
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.gamma.shape[0], self.gamma.shape[0]//min(5, self.gamma.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/self.gamma.shape[0]:3.2f} \u03C0", torch.arange(0, self.gamma.shape[0], self.gamma.shape[0]//min(5, self.gamma.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.gamma.shape[1], self.gamma.shape[1]//min(5, self.gamma.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Gamma")
            plot_x, plot_y = torch.meshgrid(torch.arange(self.gamma.shape[0]), torch.arange(self.gamma.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.gamma.detach().to("cpu").numpy()/max(self.count, 1), alpha=1.0)
            logger.add_figure("validation/gamma", figure, self.global_step)
            log_3d(logger, "validation/gamma", self.gamma/max(self.count, 1), self.global_step, 1.0)
            log_img(logger, "validation/_gamma", self.gamma.mT/max(self.count, 1), self.global_step, True)

            #Log examples
            sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["sinogram"][0,0]
            noisy_sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["noisy_sinogram"][0,0]
            learned_filtered_sinogram = radon.radon_filter(sinogram.unsqueeze(0).unsqueeze(0), lambda s,p: s*p, self.filter_params)[0,0]
            analytic_filtered_sinogram = radon.radon_filter(sinogram.unsqueeze(0).unsqueeze(0), lambda s, p: s*p, filter_params)[0,0]
            ground_truth = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["ground_truth"][0,0]
            learned_reconstruction = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["learned_reconstruction"][0,0]
            analytic_reconstruction = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[0]["analytic_reconstruction"][0,0]
            log_img(logger, "validation/sinogram", sinogram.mT, self.global_step)
            log_img(logger, "validation/noisy_sinogram", noisy_sinogram.mT, self.global_step)
            log_img(logger, "validation/learned_filtered_sinogram", learned_filtered_sinogram.mT, self.global_step)
            log_img(logger, "validation/analytic_filtered_sinogram", analytic_filtered_sinogram.mT, self.global_step)
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
        torch.save(torch.nn.utils.convert_parameters.parameters_to_vector(self.filter_params).reshape(self.filter_params.shape), "coefficients.pt")
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

            #Log learned filter coefficients
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//min(5, self.filter_params.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/self.filter_params.shape[0]:3.2f} \u03C0", torch.arange(0, self.filter_params.shape[0], self.filter_params.shape[0]//min(5, self.filter_params.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.filter_params.shape[1], self.filter_params.shape[1]//min(5, self.filter_params.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Filter value")
            axes.set_zlim(0.0, 2.0)
            plot_x, plot_y = torch.meshgrid(torch.arange(self.filter_params.shape[0]), torch.arange(self.filter_params.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.filter_params.detach().to("cpu").numpy(), alpha=1.0)
            logger.add_figure("test/learned_filter_coefficients", figure, 0)
            log_3d(logger, "test/learned_filter_coefficients", self.filter_params, 0, 1.0)
            log_img(logger, "test/_learned_filter_coefficients", self.filter_params.mT, 0, True)

            #Log pi
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.pi.shape[0], self.pi.shape[0]//min(5, self.pi.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/self.pi.shape[0]:3.2f} \u03C0", torch.arange(0, self.pi.shape[0], self.pi.shape[0]//min(5, self.pi.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.pi.shape[1], self.pi.shape[1]//min(5, self.pi.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Pi")
            plot_x, plot_y = torch.meshgrid(torch.arange(self.pi.shape[0]), torch.arange(self.pi.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.pi.detach().to("cpu").numpy()/max(self.count, 1), alpha=1.0)
            logger.add_figure("test/pi", figure, 0)
            log_3d(logger, "test/pi", self.pi/max(self.count, 1), 0, 1.0)
            log_img(logger, "test/_pi", self.pi.mT/max(self.count, 1), 0, True)

            #Log delta
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.delta.shape[0], self.delta.shape[0]//min(5, self.delta.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/self.delta.shape[0]:3.2f} \u03C0", torch.arange(0, self.delta.shape[0], self.delta.shape[0]//min(5, self.delta.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.delta.shape[1], self.delta.shape[1]//min(5, self.delta.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Delta")
            plot_x, plot_y = torch.meshgrid(torch.arange(self.delta.shape[0]), torch.arange(self.delta.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.delta.detach().to("cpu").numpy()/max(self.count, 1), alpha=1.0)
            logger.add_figure("test/delta", figure, 0)
            log_3d(logger, "test/delta", self.delta/max(self.count, 1), 0, 1.0)
            log_img(logger, "test/_delta", self.delta.mT/max(self.count, 1), 0, True)

            #Log gamma
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, self.gamma.shape[0], self.gamma.shape[0]//min(5, self.gamma.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/self.gamma.shape[0]:3.2f} \u03C0", torch.arange(0, self.gamma.shape[0], self.gamma.shape[0]//min(5, self.gamma.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, self.gamma.shape[1], self.gamma.shape[1]//min(5, self.gamma.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Gamma")
            plot_x, plot_y = torch.meshgrid(torch.arange(self.gamma.shape[0]), torch.arange(self.gamma.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, self.gamma.detach().to("cpu").numpy()/max(self.count, 1), alpha=1.0)
            logger.add_figure("test/gamma", figure, 0)
            log_3d(logger, "test/gamma", self.gamma/max(self.count, 1), 0, 1.0)
            log_img(logger, "test/_gamma", self.gamma.mT/max(self.count, 1), 0, True)

            #Log analytic filter coefficients
            filter_params = self.ramp*(self.pi-self.gamma)/(self.pi+self.delta+2*self.gamma)
            figure = plt.figure()
            axes = typing.cast(mpl_toolkits.mplot3d.Axes3D, figure.add_subplot(1, 1, 1, projection="3d"))
            axes.set_xlabel("Angle")
            axes.set_xticks(torch.arange(0, filter_params.shape[0], filter_params.shape[0]//min(5, filter_params.shape[0])).to(torch.float32).tolist())
            axes.set_xticklabels(list(map(lambda x: f"{x/filter_params.shape[0]:3.2f} \u03C0", torch.arange(0, filter_params.shape[0], filter_params.shape[0]//min(5, filter_params.shape[0])).to(torch.float32).tolist())))
            axes.set_ylabel("Frequency")
            axes.set_yticks(torch.arange(0, filter_params.shape[1], filter_params.shape[1]//min(5, filter_params.shape[1])).to(torch.float32).tolist())
            axes.set_zlabel("Filter value")
            axes.set_zlim(0.0, 2.0)
            plot_x, plot_y = torch.meshgrid(torch.arange(filter_params.shape[0]), torch.arange(filter_params.shape[1]), indexing="ij")
            axes.plot_surface(plot_x, plot_y, filter_params.detach().to("cpu").numpy(), alpha=1.0)
            logger.add_figure("test/analytic_filter_coefficients", figure, 0)
            log_3d(logger, "test/analytic_filter_coefficients", filter_params, 0, 1.0)
            log_img(logger, "test/_analytic_filter_coefficients", filter_params.mT, 0, True)

            #Log examples
            for i in range(10):
                sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["sinogram"][0,0]
                noisy_sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["noisy_sinogram"][0,0]
                learned_filtered_sinogram = radon.radon_filter(sinogram.unsqueeze(0).unsqueeze(0), lambda s,p: s*p, self.filter_params)[0,0]
                analytic_filtered_sinogram = radon.radon_filter(sinogram.unsqueeze(0).unsqueeze(0), lambda s,p: s*p, filter_params)[0,0]
                ground_truth = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["ground_truth"][0,0]
                learned_reconstruction = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["learned_reconstruction"][0,0]
                analytic_reconstruction = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["analytic_reconstruction"][0,0]
                log_img(logger, "test/sinogram", sinogram.mT, i)
                log_img(logger, "test/noisy_sinogram", noisy_sinogram.mT, i)
                log_img(logger, "test/learned_filtered_sinogram", learned_filtered_sinogram.mT, i)
                log_img(logger, "test/analytic_filtered_sinogram", analytic_filtered_sinogram.mT, i)
                log_img(logger, "test/ground_truth", ground_truth, i)
                log_img(logger, "test/learned_reconstruction", learned_reconstruction, i)
                log_img(logger, "test/analytic_reconstruction", analytic_reconstruction, i)