import torch

def export_image_from_batch(v: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    img = v.detach().to("cpu")[0,...]
    if normalize and torch.max(img) > torch.min(img):
        return (img-torch.min(img))/(torch.max(img)-torch.min(img))
    else:
        return torch.clamp(img, 0.0, 1.0)