"""Turn a clean high-resolution face into one noisy low-resolution "shot".

A single Albumentations pipeline chains the distortions a real hand-held capture
suffers - small rotation, motion blur, and zoom at capture resolution, then the
downscale to the sensor's low resolution, then sensor noise, salt & pepper, and
JPEG storage compression. Every LR sample draws a fresh random combination.
"""

import os
import sys
from pathlib import Path

import cv2

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def build_degrade_pipeline():
    """Compose the capture -> downscale -> sensor degradation chain."""
    cfg = config.DEGRADE
    prob = cfg["prob"]
    return A.Compose(
        [
            # --- capture-resolution optics / motion ---
            A.Rotate(
                limit=cfg["rotate_limit"],
                border_mode=cv2.BORDER_REFLECT_101,
                p=prob,
            ),
            A.Affine(
                scale=cfg["zoom_scale"],
                border_mode=cv2.BORDER_REFLECT_101,
                p=prob,
            ),
            A.MotionBlur(blur_limit=cfg["motion_blur_limit"], p=prob),
            # --- drop to the sensor's low resolution ---
            A.Resize(
                height=config.LR_SIZE,
                width=config.LR_SIZE,
                interpolation=cv2.INTER_AREA,
            ),
            # --- sensor + storage noise ---
            A.GaussNoise(std_range=cfg["gauss_noise_std"], p=prob),
            A.SaltAndPepper(amount=(0.0, cfg["salt_pepper_amount"]), p=prob),
            A.ImageCompression(quality_range=cfg["jpeg_quality"], p=prob),
        ]
    )


def degrade_hr(hr_image, pipeline=None):
    """Apply the degradation pipeline to an HxWx3 uint8 HR image -> LR image."""
    pipeline = pipeline or build_degrade_pipeline()
    return pipeline(image=hr_image)["image"]


if __name__ == "__main__":
    import numpy as np

    dummy = (np.random.rand(config.HR_SIZE, config.HR_SIZE, 3) * 255).astype("uint8")
    out = degrade_hr(dummy)
    print("input", dummy.shape, "-> output", out.shape, out.dtype)
