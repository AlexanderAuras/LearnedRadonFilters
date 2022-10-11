#pyright: reportGeneralTypeIssues=false

from math import atan, cos, sin, tan
import sys

import omegaconf
import pytorch_lightning as pl
import torch
import torch.utils.data
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np



class _EllipsesDataset(torch.utils.data.Dataset):
    def __init__(self, img_count: int, img_size: int, ellipses_count: int, ellipses_size: float, ellipses_size_min: float=1):
        self.img_count = img_count
        self.img_size = img_size
        #self.ellipses_count = ellipses_count
        self.ellipses_count = torch.poisson(torch.full((img_count,), ellipses_count).to(torch.float32)).to(torch.int32)
        self.ellipses_size = ellipses_size
        self.ellipses_size_min = ellipses_size_min
        self.ellipse_width_aa = torch.rand((img_count*ellipses_count,))*max(0.0, self.ellipses_size-self.ellipses_size_min)+self.ellipses_size_min
        self.ellipse_height_aa = torch.rand((img_count*ellipses_count,))*max(0.0, self.ellipses_size-self.ellipses_size_min)+self.ellipses_size_min
        self.ellipse_x_raw = torch.rand((img_count*ellipses_count,))
        self.ellipse_y_raw = torch.rand((img_count*ellipses_count,))
        self.ellipse_angle = torch.rand((img_count*ellipses_count,))*360.0
        self.ellipse_alpha = torch.rand((img_count*ellipses_count,))*0.9+0.1

    def __len__(self) -> int:
        return self.img_count

    def __getitem__(self, idx: int) -> torch.Tensor:
        fig = plt.figure(figsize=(self.img_size,self.img_size), dpi=1)
        ax = fig.add_axes([0.0,0.0,1.0,1.0])
        ellipse_func = lambda w, h, a, t: (w/2.0*cos(t)*cos(a)-h/2.0*sin(t)*sin(a), w/2.0*cos(t)*sin(a)+h/2.0*sin(t)*cos(a))
        for i in range(self.ellipses_count[idx]):
            e_idx = idx*self.ellipses_count[idx]+i
            args = (self.ellipse_width_aa[e_idx].item(), self.ellipse_height_aa[e_idx].item(), self.ellipse_angle[e_idx].item()/180.0*torch.pi)
            t = atan(-self.ellipse_height_aa[e_idx].item()*tan(self.ellipse_angle[e_idx].item()/180.0*torch.pi)/self.ellipse_width_aa[e_idx].item())
            ellipse_width = max(ellipse_func(*args, t)[0], ellipse_func(*args, t+torch.pi)[0])-min(ellipse_func(*args, t)[0], ellipse_func(*args, t+torch.pi)[0])
            t = atan(self.ellipse_height_aa[e_idx].item()/(tan(self.ellipse_angle[e_idx].item()/180.0*torch.pi)*self.ellipse_width_aa[e_idx].item()))
            ellipse_height = max(ellipse_func(*args, t)[1], ellipse_func(*args, t+torch.pi)[1])-min(ellipse_func(*args, t)[1], ellipse_func(*args, t+torch.pi)[1])
            ellipse_x = ellipse_width/2.0+self.ellipse_x_raw[e_idx].item()*(self.img_size-ellipse_width)
            ellipse_y = ellipse_height/2.0+self.ellipse_y_raw[e_idx].item()*(self.img_size-ellipse_height)
            ellipse = matplotlib.patches.Ellipse(xy=[ellipse_x, ellipse_y], width=self.ellipse_width_aa[e_idx].item(), height=self.ellipse_height_aa[e_idx].item(), angle=self.ellipse_angle[e_idx].item())
            ax.add_artist(ellipse)
            ellipse.set_clip_box(ax.bbox)
            ellipse.set_alpha(self.ellipse_alpha[e_idx].item())
            ellipse.set_facecolor("black")
            ellipse.set_edgecolor(None)
            ellipse.set_antialiased(False)
        ax.axis("off")
        ax.set_xlim(0.0, self.img_size)
        ax.set_ylim(0.0, self.img_size)
        fig.add_axes(ax)
        fig.canvas.draw()
        img = torch.from_numpy(np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).copy())
        if sys.platform == "win32":
            img = 1.0-torch.swapaxes(img.reshape(self.img_size,self.img_size,4), 0, 2).to(torch.float32)[3:4]/255.0
        else:
            img = torch.swapaxes(img.reshape(self.img_size,self.img_size,4), 0, 2).to(torch.float32)[0:1]/255.0
        plt.close()
        return img



class EllipsesDataModule(pl.LightningDataModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        training_dataset = _EllipsesDataset((640 if self.config.training_batch_count == -1 else self.config.training_batch_count)*self.config.training_batch_size, self.config.img_size, self.config.ellipse_count, self.config.ellipse_size, self.config.ellipse_size_min)
        return torch.utils.data.DataLoader(training_dataset, drop_last=self.config.drop_last_training_batch, batch_size=self.config.training_batch_size, shuffle=self.config.shuffle_training_data, num_workers=self.config.num_workers)
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        validation_dataset = _EllipsesDataset((160 if self.config.validation_batch_count == -1 else self.config.validation_batch_count)*self.config.validation_batch_size, self.config.img_size, self.config.ellipse_count, self.config.ellipse_size, self.config.ellipse_size_min)
        return torch.utils.data.DataLoader(validation_dataset, drop_last=self.config.drop_last_validation_batch, batch_size=self.config.validation_batch_size, shuffle=self.config.shuffle_validation_data, num_workers=self.config.num_workers)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        test_dataset = _EllipsesDataset((200 if self.config.test_batch_count == -1 else self.config.test_batch_count)*self.config.test_batch_size, self.config.img_size, self.config.ellipse_count, self.config.ellipse_size, self.config.ellipse_size_min)
        return torch.utils.data.DataLoader(test_dataset, drop_last=self.config.drop_last_test_batch, batch_size=self.config.test_batch_size, shuffle=self.config.shuffle_test_data, num_workers=self.config.num_workers)