import numpy as np
from scipy.ndimage import zoom

def random_crop(volume, min_scale = 0.5, max_scale = 1.0, order = 1):
    """
    Randomly crop a sub-volume of `volume` and resize it back.

    Args:
        volume:    Input 3D array of shape (D, H, W).
        min_scale: Minimum fraction of original size to crop.
        max_scale: Maximum fraction of original size to crop.
        order:     The order of the spline interpolation (0=nearest, 1=linear, ...).

    Returns:
        Augmented volume of same shape as input.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D array")

    D, H, W = volume.shape

    # 1) Sample a random scale and compute new crop size
    scale = np.random.uniform(min_scale, max_scale)
    new_D = max(1, int(D * scale))
    new_H = max(1, int(H * scale))
    new_W = max(1, int(W * scale))

    # 2) Randomly choose the top‐left‐front corner of the crop
    z0 = np.random.randint(0, D - new_D + 1)
    y0 = np.random.randint(0, H - new_H + 1)
    x0 = np.random.randint(0, W - new_W + 1)

    cropped = volume[z0 : z0 + new_D,
                     y0 : y0 + new_H,
                     x0 : x0 + new_W]

    # 3) Compute zoom factors to bring crop back to (D, H, W)
    zoom_factors = (D / new_D, H / new_H, W / new_W)

    # 4) Resize with interpolation
    resized = zoom(cropped, zoom_factors, order=order)

    return resized

augmentation_transform = random_crop

# augmentation_transform = tio.Compose([
#     tio.RandomFlip(axes=('LR',), p=0.5),
#     tio.RandomElasticDeformation(max_displacement=5, p=0.3),
#     tio.RandomAffine(
#         scales=(0.95, 1.05),       # ±5% scaling
#         degrees=5,                 # ±5° rotations
#         translation=5,             # ±5 voxels
#         p=0.3,                     # only 30% of the time
#     ),
#     tio.RandomNoise(std=(0, 0.05), p=0.2),  # small Gaussian noise
# ])
