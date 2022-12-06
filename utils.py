import typing

import torch
import matplotlib
import matplotlib.pyplot as plt



def log_img(logger: typing.Any, tag: str, img: torch.Tensor, step: int, log_color: bool=False) -> None:
    figure = plt.figure()
    img = torch.nan_to_num(img, 1.0)
    if img.max() == img.min():
        img[0,0] += 0.0001
    if log_color:
        plt.imshow(img.detach().to("cpu"), cmap="gray", norm=matplotlib.colors.SymLogNorm(0.1))
    else:
        plt.imshow(img.detach().to("cpu"), cmap="gray")
    plt.colorbar()
    plt.tight_layout()
    logger.add_figure(tag, figure, step)
    plt.close()



def log_3d(logger: typing.Any, tag: str, data: torch.Tensor, step: int, ar: typing.Union[float,None] = None) -> None:
    x_coords, z_coords = torch.meshgrid(torch.cat((torch.tensor([0.0]), torch.arange(data.shape[1]), torch.tensor([data.shape[1]-1]))), torch.cat((torch.tensor([0.0]), torch.arange(data.shape[0]), torch.tensor([data.shape[0]-1]))), indexing="xy")
    x_coords, z_coords = x_coords.flatten(), z_coords.flatten()
    if ar == None:
        ar = data.shape[0]/data.shape[1]
    x_coords = x_coords/data.shape[1]*2.0-1.0
    z_coords = (z_coords/data.shape[0]*2.0-1.0)*ar
    y_coords = data.detach().to("cpu")
    mini = y_coords.min().item()
    y_coords = torch.cat((torch.full((y_coords.shape[0],1), mini),y_coords,torch.full((y_coords.shape[0],1), mini)), dim=1)
    y_coords = torch.cat((torch.full((1,y_coords.shape[1]), mini),y_coords,torch.full((1,y_coords.shape[1]), mini)), dim=0)
    y_coords = y_coords.flatten()
    if y_coords.max()-y_coords.min() != 0.0:
        y_coords = (y_coords-y_coords.min())/(y_coords.max()-y_coords.min())
    vertices = torch.stack((x_coords, y_coords, z_coords), dim=1).unsqueeze(0)

    i = torch.arange((data.shape[1]+1)*(data.shape[0]+1))
    i = torch.floor(i/(data.shape[1]+1))*(data.shape[1]+2)+i%(data.shape[1]+1)
    i = i.reshape(-1,1)
    F = torch.tensor([[data.shape[1]+2,1,0,data.shape[1]+2,data.shape[1]+3,1]])
    indices = torch.flatten(F+i)
    indices = torch.cat((indices, torch.tensor([
        0, data.shape[1]+1, (data.shape[0]+1)*(data.shape[1]+2),
        (data.shape[0]+1)*(data.shape[1]+2), data.shape[1]+1, (data.shape[0]+2)*(data.shape[1]+2)-1
    ])))
    indices = indices.reshape(1,-1,3)

    logger.add_mesh(tag, vertices, None, indices, {}, step)