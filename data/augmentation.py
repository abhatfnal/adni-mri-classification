import torch
import torch.nn.functional as F
import pandas as pd
import nibabel as nib
import numpy as np

import torchio as tio
from torch.utils.data import Dataset

def random_crop(
    volume: torch.Tensor,
    min_scale: float = 0.5,
    max_scale: float = 1.0,
) -> torch.Tensor:
    """
    Randomly crop a sub-volume of `volume` and resize it
    back to the original shape using trilinear interpolation.
    Input:  4D torch.Tensor (C, D, H, W) on any device.
    Output: same shape & same device.
    """
    C, D, H, W = volume.shape
    device = volume.device

    # pick random scale
    scale = torch.empty(1, device=device).uniform_(min_scale, max_scale).item()
    new_D = max(1, int(D * scale))
    new_H = max(1, int(H * scale))
    new_W = max(1, int(W * scale))

    # pick random corner
    z0 = torch.randint(0, D - new_D + 1, (), device=device).item()
    y0 = torch.randint(0, H - new_H + 1, (), device=device).item()
    x0 = torch.randint(0, W - new_W + 1, (), device=device).item()

    # crop
    cropped = volume[
        :,
        z0 : z0 + new_D,
        y0 : y0 + new_H,
        x0 : x0 + new_W,
    ]  # shape (C, new_D, new_H, new_W)

    # prepare for interpolation: add batch dim
    cropped = cropped.unsqueeze(0)  # (1, C, new_D, new_H, new_W)

    # resize back to (D, H, W) with trilinear
    resized = F.interpolate(
        cropped,
        size=(D, H, W),
        mode='trilinear',
        align_corners=False,
    )
    return resized.squeeze(0)  # (C, D, H, W) on original device


def build_augmentation(cfg: dict) -> tio.Compose:
    """
    Build a torchio Compose where each transform
    takes & returns a torch.Tensor.
    """
    transforms = []
    for name, params in cfg.items():
        if name == 'random_crop':
            transforms.append(
                tio.Lambda(
                    lambda x: random_crop(
                        x,
                        params.get('min_scale', 0.5),
                        params.get('max_scale', 1.0),
                    ),
                    p=params.get('p', 0.5),
                )
            )
        elif name == 'random_flip':
            transforms.append(
                tio.RandomFlip(axes=('LR',), p=params.get('p', 0.5))
            )
        elif name == 'random_affine':
            transforms.append(
                tio.RandomAffine(
                    scales=params.get('scales',1),
                    degrees=params.get('degrees', 10),
                    translation=params.get('translation',10),
                    p=params.get('p', 0.5),
                )
            )
        else:
            raise ValueError(f"Unknown transform: {name}")
    return tio.Compose(transforms)





# def build_augmentation(cfg: dict) -> tio.Compose:
#     """
#     Construct a TorchIO Compose transform based on a config dictionary.

#     Args:
#         cfg: A dict containing augmentation settings, e.g.
#             {
#               'random_crop': {
#                   'enable': True,
#                   'min_scale': 0.6,
#                   'max_scale': 1.0,
#                   'order': 1
#               },
#               'RandomFlip': {
#                   'enable': True,
#                   'axes': ('LR',),
#                   'p': 0.5
#               },
#               'RandomElasticDeformation': {
#                   'enable': False,
#                   'max_displacement': 5,
#                   'p': 0.3
#               },
#               'RandomAffine': {
#                   'enable': True,
#                   'scales': (0.95, 1.05),
#                   'degrees': 5,
#                   'translation': 5,
#                   'p': 0.3
#               },
#               'RandomNoise': {
#                   'enable': True,
#                   'std': (0, 0.05),
#                   'p': 0.2
#               }
#             }

#     Returns:
#         A `tio.Compose` of the enabled transforms. If no transforms are
#         enabled, returns `None`.
#     """
#     transforms = []

#     # Custom random_crop
#     rc_cfg = cfg.get('random_crop', {})
#     if rc_cfg.get('enable', False):
#         transforms.append(
#             tio.Lambda(
#                 lambda image: random_crop(
#                     image.numpy(),
#                     min_scale=rc_cfg.get('min_scale', 0.5),
#                     max_scale=rc_cfg.get('max_scale', 1.0),
#                     order=rc_cfg.get('order', 1)
#                 ),
#                 lambda image: image
#             )
#         )

#     # TorchIO built-ins
#     if cfg.get('RandomFlip', {}).get('enable', False):
#         ff = cfg['RandomFlip']
#         transforms.append(
#             tio.RandomFlip(
#                 axes=ff.get('axes', ('LR',)),
#                 p=ff.get('p', 0.5)
#             )
#         )

#     if cfg.get('RandomElasticDeformation', {}).get('enable', False):
#         ed = cfg['RandomElasticDeformation']
#         transforms.append(
#             tio.RandomElasticDeformation(
#                 max_displacement=ed.get('max_displacement', 5),
#                 p=ed.get('p', 0.3)
#             )
#         )

#     if cfg.get('RandomAffine', {}).get('enable', False):
#         af = cfg['RandomAffine']
#         transforms.append(
#             tio.RandomAffine(
#                 scales=af.get('scales', (0.95, 1.05)),
#                 degrees=af.get('degrees', 5),
#                 translation=af.get('translation', 5),
#                 p=af.get('p', 0.3)
#             )
#         )

#     if cfg.get('RandomNoise', {}).get('enable', False):
#         rn = cfg['RandomNoise']
#         transforms.append(
#             tio.RandomNoise(
#                 std=rn.get('std', (0, 0.05)),
#                 p=rn.get('p', 0.2)
#             )
#         )

#     if not transforms:
#         return None

#     return tio.Compose(transforms)

# def random_crop(volume: np.ndarray,
#                 min_scale: float = 0.5,
#                 max_scale: float = 1.0,
#                 order: int = 1) -> np.ndarray:
#     """
#     Randomly crop a sub-volume of `volume` and resize it back.
#     """
#     if volume.ndim != 3:
#         raise ValueError("Input volume must be a 3D array")

#     D, H, W = volume.shape
#     scale = np.random.uniform(min_scale, max_scale)
#     new_D = max(1, int(D * scale))
#     new_H = max(1, int(H * scale))
#     new_W = max(1, int(W * scale))

#     z0 = np.random.randint(0, D - new_D + 1)
#     y0 = np.random.randint(0, H - new_H + 1)
#     x0 = np.random.randint(0, W - new_W + 1)

#     cropped = volume[z0:z0 + new_D,
#                      y0:y0 + new_H,
#                      x0:x0 + new_W]

#     zoom_factors = (D / new_D, H / new_H, W / new_W)
#     resized = zoom(cropped, zoom_factors, order=order)
#     return resized


if __name__ == "__main__":
    
    cfg = {
        'random_crop':{
            'min_scale':0.3,
            'max_scale':1.2,
            'order':1,
            'p':0.2,
        }
    }
    
    transform = build_augmentation(cfg)