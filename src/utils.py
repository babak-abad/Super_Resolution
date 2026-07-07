"""Shared helpers: device, seeding, metrics, and tensor/image conversion."""

import random

import numpy as np
import torch
from skimage.metrics import structural_similarity


def pick_device():
    """Return the CUDA device when available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Seed Python, NumPy and Torch so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_psnr(pred, target):
    """Peak signal-to-noise ratio (dB) between two [0,1] float arrays."""
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(pred, target):
    """Structural similarity between two [0,1] float HxWxC arrays."""
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return float(
        structural_similarity(target, pred, channel_axis=2, data_range=1.0)
    )


def image_to_tensor(image):
    """HxWxC uint8 image -> CxHxW float tensor in [0,1]."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_image(tensor):
    """CxHxW float tensor in [0,1] -> HxWxC uint8 image."""
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)
