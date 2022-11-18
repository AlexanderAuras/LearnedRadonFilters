#pyright: reportPrivateImportUsage=false, reportGeneralTypeIssues=false
import functools
import logging
import os
import subprocess
import typing
import warnings

import hydra
import hydra.core.hydra_config
import omegaconf

#Register additional resolver for log path
omegaconf.OmegaConf.register_new_resolver("list_to_string", lambda o: functools.reduce(lambda acc, x: acc+", "+x.replace("\"","").replace("/"," "), o, "")[2:])
omegaconf.OmegaConf.register_new_resolver("eval", lambda c: eval(c))

import pytorch_lightning
import pytorch_lightning.accelerators
import pytorch_lightning.callbacks
import pytorch_lightning.loggers
import pytorch_lightning.utilities

import torch
import torch.utils.tensorboard
import torch.version

from mnist_datamodule import MNISTDataModule
from ellipses_datamodule import EllipsesDataModule
from filter_model import FilterModel
from svd_model import SVDModel


#Custom version of pytorch lightnings TensorBoardLogger, to allow manipulation of internal logging settings
class CustomTensorBoardLogger(pytorch_lightning.loggers.TensorBoardLogger):
    #Disables logging of epoch
    @pytorch_lightning.utilities.rank_zero_only
    def log_metrics(self, metrics: typing.Dict[str, typing.Union[torch.Tensor,float]], step: int) -> None:
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)
    
    #Disables creation of hparams.yaml
    @pytorch_lightning.utilities.rank_zero_only
    def save(self) -> None:
        dir_path = self.log_dir
        if not os.path.isdir(dir_path):
            dir_path = self.save_dir



@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(config: omegaconf.DictConfig) -> None:
    #Setup logging
    logger = logging.getLogger(__name__)
    logging.captureWarnings(True)
    logging.getLogger("pytorch_lightning").handlers.append(logger.root.handlers[1]) #Route pytorch lightning logging to hydra logger
    for old_log in os.listdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir):
        if old_log.startswith("events.out.tfevents"):
            os.remove(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, old_log))

    #Append current git commit hash to saved config
    with open(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,".hydra","config.yaml"), "a") as cfg_file:
        cfg_file.write(f"git_project: {subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], cwd=hydra.utils.get_original_cwd()).decode('ascii').strip()}\n")
        cfg_file.write(f"git_branch: {subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=hydra.utils.get_original_cwd()).decode('ascii').strip()}\n")
        cfg_file.write(f"git_commit: {subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=hydra.utils.get_original_cwd()).decode('ascii').strip()}\n")

    #Set num_workers to available CPU count
    if config.num_workers == -1:
        config.num_workers = os.cpu_count()
    
    #Initialize determinism
    if config.deterministic:
        pytorch_lightning.seed_everything(config.seed, workers=True)
        
    if config.dataset.name == "MNIST":
        datamodule = MNISTDataModule(config)
    elif config.dataset.name == "ellipses":
        datamodule = EllipsesDataModule(config)
    else:
        raise NotImplementedError()

    #Create model and load data
    if config.model.name == "filter":
        modelClass = FilterModel
    elif config.model.name == "svd":
        modelClass = SVDModel
    else:
        raise NotImplementedError()
    if config.checkpoint != None:
        model = modelClass.load_from_checkpoint(os.path.abspath(os.path.join("../../" if hydra.core.hydra_config.HydraConfig.get().mode == hydra.types.RunMode.MULTIRUN else "../", config.checkpoint)), config=config)
    else:
        model = modelClass(config)

    #Execute training and testing
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Checkpoint directory .+ exists and is not empty\.") #Needed thanks to the custom logger
        warnings.filterwarnings("ignore", r"The dataloader, ((train)|(val)|(test)) dataloader( \d+)?, does not have many workers which may be a bottleneck\. Consider increasing the value of the `num_workers` argument` \(try \d+ which is the number of cpus on this machine\) in the `DataLoader` init to improve performance\.") #BUG Contradictory warnings with num_workers on cluster and slow loading with LoDoPaB
        trainer = pytorch_lightning.Trainer(
            deterministic=config.deterministic, 
            callbacks=[pytorch_lightning.callbacks.ModelCheckpoint(dirpath=".")], 
            accelerator="gpu" if config.device == "cuda" else None, devices=1,
            max_epochs=config.epochs, 
            logger=CustomTensorBoardLogger(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, None, ""), 
            limit_train_batches=int(config.training_batch_count) if config.training_batch_count != -1 else len(datamodule.train_dataloader()), 
            limit_val_batches=int(config.validation_batch_count) if config.validation_batch_count != -1 else len(datamodule.val_dataloader()), 
            limit_test_batches=int(config.test_batch_count) if config.test_batch_count != -1 else len(datamodule.test_dataloader()))
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
    
    logging.getLogger("pytorch_lightning").handlers.remove(logger.root.handlers[1]) #Stops pytorch lightning from writing to previous runs during multiruns



if __name__ == "__main__":
    main()