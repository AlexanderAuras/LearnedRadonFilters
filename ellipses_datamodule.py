#pyright: reportGeneralTypeIssues=false

from math import atan, cos, sin, tan

import omegaconf
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np



class _EllipsesDataset(torch.utils.data.Dataset):
    def __init__(self, img_count: int, img_size: int, ellipses_count: int, ellipses_size: float, ellipses_size_min: float=1):
        self.img_count = img_count
        self.img_size = img_size
        self.ellipses_count = ellipses_count
        self.ellipses_size = ellipses_size
        self.ellipses_size_min = ellipses_size_min
        self.ellipse_width_aa = np.random.rand(img_count)*max(0.0, self.ellipses_size-self.ellipses_size_min)+self.ellipses_size_min
        self.ellipse_height_aa = np.random.rand(img_count)*max(0.0, self.ellipses_size-self.ellipses_size_min)+self.ellipses_size_min
        self.ellipse_x_raw = np.random.rand(img_count)
        self.ellipse_y_raw = np.random.rand(img_count)
        self.ellipse_angle = np.random.rand(img_count)*360.0

    def __len__(self) -> int:
        return self.img_count

    def __item__(self, idx: int) -> torch.Tensor:
        fig = plt.figure(figsize=(self.img_size,self.img_size), dpi=1)
        ax = fig.add_axes([0.0,0.0,1.0,1.0])
        ellipse_func = lambda w, h, a, t: (w/2.0*cos(t)*cos(a)-h/2.0*sin(t)*sin(a), w/2.0*cos(t)*sin(a)+h/2.0*sin(t)*cos(a))
        for _ in range(self.ellipses_count):
            args = (self.ellipse_width_aa[idx], self.ellipse_height_aa[idx], self.ellipse_angle[idx]/180.0*np.pi)
            t = atan(-self.ellipse_height_aa[idx]*tan(self.ellipse_angle[idx]/180.0*np.pi)/self.ellipse_width_aa[idx])
            ellipse_width = max(ellipse_func(*args, t)[0], ellipse_func(*args, t+np.pi)[0])-min(ellipse_func(*args, t)[0], ellipse_func(*args, t+np.pi)[0])
            t = atan(self.ellipse_height_aa[idx]/(tan(self.ellipse_angle[idx]/180.0*np.pi)*self.ellipse_width_aa[idx]))
            ellipse_height = max(ellipse_func(*args, t)[1], ellipse_func(*args, t+np.pi)[1])-min(ellipse_func(*args, t)[1], ellipse_func(*args, t+np.pi)[1])
            ellipse_x = ellipse_width/2.0+self.ellipse_x_raw[idx]*(self.img_size-ellipse_width)
            ellipse_y = ellipse_height/2.0+self.ellipse_y_raw[idx]*(self.img_size-ellipse_height)
            ellipse = matplotlib.patches.Ellipse(xy=[ellipse_x, ellipse_y], width=self.ellipse_width_aa[idx], height=self.ellipse_height_aa[idx], angle=self.ellipse_angle[idx])
            ax.add_artist(ellipse)
            ellipse.set_clip_box(ax.bbox)
            ellipse.set_alpha(0.1+0.9*np.random.rand())
        ax.axis("off")
        ax.set_xlim(0.0, self.img_size)
        ax.set_ylim(0.0, self.img_size)
        fig.add_axes(ax)
        fig.canvas.draw()
        img = torch.from_numpy(1.0-np.array(fig.canvas.renderer._renderer)[:,:,3])
        plt.close()
        return img.unsqueeze(0)



class EllipsesDataModule(pl.LightningDataModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        training_dataset = _EllipsesDataset((640 if self.training_batch_count == -1 else self.training_batch_count)*self.training_batch_size, self.config.img_size, 100, 5, 5)
        _, training_dataset = torch.utils.data.random_split(training_dataset, [int(len(training_dataset)*self.config.validation_split_percent/100.0), len(training_dataset)-int(len(training_dataset)*self.config.validation_split_percent/100.0)])
        return torch.utils.data.DataLoader(training_dataset, drop_last=self.config.drop_last_training_batch, batch_size=self.config.training_batch_size, shuffle=self.config.shuffle_training_data, num_workers=self.config.num_workers)
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        validation_dataset = _EllipsesDataset((160 if self.validation_batch_count == -1 else self.validation_batch_count)*self.validation_batch_size, self.config.img_size, 100, 5, 5)
        return torch.utils.data.DataLoader(validation_dataset, drop_last=self.config.drop_last_validation_batch, batch_size=self.config.validation_batch_size, shuffle=self.config.shuffle_validation_data, num_workers=self.config.num_workers)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        test_dataset = _EllipsesDataset((200 if self.test_batch_count == -1 else self.test_batch_count)*self.test_batch_size, self.config.img_size, 100, 5, 5)
        return torch.utils.data.DataLoader(test_dataset, drop_last=self.config.drop_last_test_batch, batch_size=self.config.test_batch_size, shuffle=self.config.shuffle_test_data, num_workers=self.config.num_workers)