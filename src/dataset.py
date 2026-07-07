"""Torch dataset that pairs a degraded low-resolution face with its HR original.

For each HR image the degradation pipeline produces a small noisy LR shot; that
LR is bicubic-upsampled back to the HR size so SRCNN sees an image at the target
resolution and only has to sharpen it. Training samples are degraded randomly
every epoch; validation samples are degraded deterministically (seeded by index)
so metrics are comparable across epochs.
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from degrade import build_degrade_pipeline, degrade_hr
from utils import image_to_tensor


# Seed stride between frames so shot f of face i is reproducible and, crucially,
# identical for every frame count - adding frames only appends new shots.
_FRAME_SEED_STRIDE = 100


def _bicubic_upsample(lr_image):
    """Bring an LR image back up to the HR working size with bicubic interp."""
    return cv2.resize(
        lr_image, (config.HR_SIZE, config.HR_SIZE), interpolation=cv2.INTER_CUBIC
    )


class Sr_Dataset(torch.utils.data.Dataset):
    """Pairs an HR face with either one degraded LR shot (num_frames=1) or a
    stack of num_frames independent shots for multi-frame super-resolution."""

    def __init__(self, hr_paths, augment, num_frames=1):
        self.hr_paths = list(hr_paths)
        self.augment = augment
        self.num_frames = num_frames
        self.pipeline = build_degrade_pipeline()

    def __len__(self):
        return len(self.hr_paths)

    def load_hr(self, index):
        """Return the HR image at index as an HxWx3 uint8 array."""
        image = Image.open(self.hr_paths[index]).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def make_lr(self, index, hr_image):
        """Degrade an HR image to LR, deterministically when not augmenting."""
        if not self.augment:
            random.seed(config.SEED + index)
            np.random.seed(config.SEED + index)
        return degrade_hr(hr_image, pipeline=self.pipeline)

    def make_lr_frames(self, index, hr_image):
        """Return num_frames independent LR shots of one face.

        When not augmenting, frame f of face index is seeded reproducibly and
        independently of the frame count, so the k-th shot is the same whether
        the model fuses 2 frames or 7.
        """
        frames = []
        for frame in range(self.num_frames):
            if not self.augment:
                seed = config.SEED + index * _FRAME_SEED_STRIDE + frame
                random.seed(seed)
                np.random.seed(seed)
            frames.append(degrade_hr(hr_image, pipeline=self.pipeline))
        return frames

    def sample_triplet(self, index):
        """Return (lr, lr_upsampled, hr) uint8 images for visualisation."""
        hr = self.load_hr(index)
        lr = self.make_lr(index, hr)
        return lr, _bicubic_upsample(lr), hr

    def sample_frames(self, index):
        """Return (lr_frames, lr_up_frames, hr) uint8 lists for MFSR visuals."""
        hr = self.load_hr(index)
        frames = self.make_lr_frames(index, hr)
        return frames, [_bicubic_upsample(f) for f in frames], hr

    def __getitem__(self, index):
        hr = self.load_hr(index)
        if self.num_frames == 1:
            lr_up = _bicubic_upsample(self.make_lr(index, hr))
            return image_to_tensor(lr_up), image_to_tensor(hr)
        frames = self.make_lr_frames(index, hr)
        stack = torch.cat(
            [image_to_tensor(_bicubic_upsample(f)) for f in frames], dim=0
        )
        return stack, image_to_tensor(hr)


def list_hr_paths():
    """All HR image paths, shuffled reproducibly."""
    paths = sorted(config.HR_DIR.glob("*.png"))
    rng = random.Random(config.SEED)
    rng.shuffle(paths)
    return paths


def split_paths():
    """Split the HR paths into (train, val) by config.VAL_FRACTION."""
    paths = list_hr_paths()
    if not paths:
        raise FileNotFoundError(
            f"No HR images in {config.HR_DIR}. Run src/download_data.py first."
        )
    val_count = max(1, int(len(paths) * config.VAL_FRACTION))
    return paths[val_count:], paths[:val_count]


def make_loaders(num_frames=1, num_workers=None):
    """Build the train and validation DataLoaders.

    num_frames=1 gives the single-image SRCNN loaders; a larger value stacks
    that many degraded shots per sample for multi-frame training.
    """
    workers = config.NUM_WORKERS if num_workers is None else num_workers
    train_paths, val_paths = split_paths()
    train_set = Sr_Dataset(
        hr_paths=train_paths, augment=True, num_frames=num_frames
    )
    val_set = Sr_Dataset(
        hr_paths=val_paths, augment=False, num_frames=num_frames
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
    )
    return train_loader, val_loader
