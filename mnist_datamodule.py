import omegaconf
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms



class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config



    def train_dataloader(self) -> torch.utils.data.DataLoader:
        training_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.config.dataset.img_size, self.config.dataset.img_size))
        ])
        training_dataset = torchvision.datasets.MNIST("/data/datasets/", train=True, transform=training_transform, download=True)
        _, training_dataset = torch.utils.data.random_split(training_dataset, [int(len(training_dataset)*self.config.validation_split_percent/100.0), len(training_dataset)-int(len(training_dataset)*self.config.validation_split_percent/100.0)])
        return torch.utils.data.DataLoader(training_dataset, drop_last=self.config.drop_last_training_batch, batch_size=self.config.training_batch_size, shuffle=self.config.shuffle_training_data, num_workers=self.config.num_workers)
    


    def val_dataloader(self) -> torch.utils.data.DataLoader:
        validation_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.config.dataset.img_size, self.config.dataset.img_size))
        ])
        validation_dataset = torchvision.datasets.MNIST("/data/datasets/", train=True, transform=validation_transform, download=True)
        validation_dataset, _ = torch.utils.data.random_split(validation_dataset, [int(len(validation_dataset)*self.config.validation_split_percent/100.0), len(validation_dataset)-int(len(validation_dataset)*self.config.validation_split_percent/100.0)])
        return torch.utils.data.DataLoader(validation_dataset, drop_last=self.config.drop_last_validation_batch, batch_size=self.config.validation_batch_size, shuffle=self.config.shuffle_validation_data, num_workers=self.config.num_workers)



    def test_dataloader(self) -> torch.utils.data.DataLoader:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.config.dataset.img_size, self.config.dataset.img_size))
        ])
        test_dataset = torchvision.datasets.MNIST("/data/datasets/", train=False, transform=test_transform, download=True)
        return torch.utils.data.DataLoader(test_dataset, drop_last=self.config.drop_last_test_batch, batch_size=self.config.test_batch_size, shuffle=self.config.shuffle_test_data, num_workers=self.config.num_workers)