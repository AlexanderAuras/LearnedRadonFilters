import typing

import torch
import matplotlib.pyplot as plt



def extract_tensor( outputs: list[dict[str,torch.Tensor|list[torch.Tensor]]], name: str, idx: int) -> torch.Tensor:
    return typing.cast(list[dict[str,torch.Tensor]], outputs)[idx][name][0,0]



def log_img( logger: typing.Any, tag: str, img: torch.Tensor, step: int) -> None:
    figure = plt.figure()
    plt.imshow(img.detach().to("cpu"), cmap="gray")
    plt.colorbar()
    plt.tight_layout()
    logger.add_figure(tag, figure, step)



def _hsv_to_rgb( hsv: torch.Tensor) -> torch.Tensor:
    h = torch.floor(hsv[:,0]/60.0)
    f = hsv[:,0]/60.0-h
    p = hsv[:,2]*(1.0-hsv[:,1])
    q = hsv[:,2]*(1.0-hsv[:,1]*f)
    t = hsv[:,2]*(1.0-hsv[:,1]*(1.0-f))
    h = h.unsqueeze(1).repeat(1,3)
    rgb = torch.stack((hsv[:,0],t,p), dim=1)
    rgb[h==1] = torch.stack((q,hsv[:,0],p), dim=1)[h==1]
    rgb[h==2] = torch.stack((p,hsv[:,0],t), dim=1)[h==2]
    rgb[h==3] = torch.stack((p,q,hsv[:,0]), dim=1)[h==3]
    rgb[h==4] = torch.stack((t,p,hsv[:,0]), dim=1)[h==4]
    rgb[h==5] = torch.stack((hsv[:,0],p,q), dim=1)[h==5]
    return rgb



def log_3d(logger: typing.Any, tag: str, data: torch.Tensor, step: int, ar: float|None = None) -> None:
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
    y_coords = (y_coords-y_coords.min())/(y_coords.max()-y_coords.min())
    vertices = torch.stack((x_coords, y_coords, z_coords), dim=1).unsqueeze(0)

    #colors = _hsv_to_rgb(torch.stack((y_coords*360.0, torch.ones_like(y_coords), torch.ones_like(y_coords)), dim=1)).unsqueeze(0)

    #i = torch.arange((data.shape[1]-1)*(data.shape[0]-1))
    #i = torch.floor(i/(data.shape[1]-1))*data.shape[1]+i%(data.shape[1]-1)
    #i = i.reshape(-1,1)
    #F = torch.tensor([[data.shape[1],1,0,data.shape[1],data.shape[1]+1,1]])
    #indices = torch.flatten(F+i).reshape(1,-1,3)

    i = torch.arange((data.shape[1]+1)*(data.shape[0]+1))
    i = torch.floor(i/(data.shape[1]+1))*(data.shape[1]+2)+i%(data.shape[1]+1)
    i = i.reshape(-1,1)
    F = torch.tensor([[data.shape[1]+2,1,0,data.shape[1]+2,data.shape[1]+3,1]])
    indices = torch.flatten(F+i).reshape(1,-1,3)

    logger.add_mesh(tag, vertices, None, indices, {}, step)