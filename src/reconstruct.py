"""Run a trained SRCNN on held-out faces and score it against the bicubic input.

Writes a LR / bicubic / SRCNN / HR comparison montage to results/ and prints the
mean PSNR and SSIM for both the bicubic baseline and SRCNN over the whole
validation split.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dataset import Sr_Dataset, split_paths
from model import Sr_Cnn
from utils import (
    compute_psnr,
    compute_ssim,
    image_to_tensor,
    pick_device,
    tensor_to_image,
)

NUM_MONTAGE_ROWS = 5


def load_model(device):
    checkpoint_path = config.CHECKPOINT_DIR / config.CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Run train.py.")
    model = Sr_Cnn().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def super_resolve(model, lr_up_image, device):
    """LR-upsampled uint8 image -> SRCNN reconstruction uint8 image."""
    tensor = image_to_tensor(lr_up_image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).squeeze(0)
    return tensor_to_image(pred)


def score_dataset(model, dataset, device):
    """Mean (bicubic_psnr, srcnn_psnr, bicubic_ssim, srcnn_ssim) over a set."""
    rows = []
    for index in range(len(dataset)):
        _lr, lr_up, hr = dataset.sample_triplet(index)
        sr = super_resolve(model=model, lr_up_image=lr_up, device=device)
        hr_f, lr_up_f, sr_f = (x / 255.0 for x in (hr, lr_up, sr))
        rows.append(
            (
                compute_psnr(lr_up_f, hr_f),
                compute_psnr(sr_f, hr_f),
                compute_ssim(lr_up_f, hr_f),
                compute_ssim(sr_f, hr_f),
            )
        )
    return np.mean(rows, axis=0)


def save_montage(model, dataset, device, out_path):
    """Save a LR / bicubic / SRCNN / HR comparison grid."""
    rows = min(NUM_MONTAGE_ROWS, len(dataset))
    titles = ["LR input", "Bicubic", "SRCNN", "HR target"]
    fig, axes = plt.subplots(rows, 4, figsize=(9, 2.25 * rows))
    axes = np.atleast_2d(axes)
    for row in range(rows):
        lr, lr_up, hr = dataset.sample_triplet(row)
        sr = super_resolve(model=model, lr_up_image=lr_up, device=device)
        panels = [lr, lr_up, sr, hr]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col in (1, 2):
                psnr = compute_psnr(panel / 255.0, hr / 255.0)
                ax.set_xlabel(f"{psnr:.1f} dB", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    device = pick_device()
    print(f"Device: {device}")
    model = load_model(device=device)

    _train_paths, val_paths = split_paths()
    val_set = Sr_Dataset(hr_paths=val_paths, augment=False)

    bicubic_psnr, srcnn_psnr, bicubic_ssim, srcnn_ssim = score_dataset(
        model=model, dataset=val_set, device=device
    )
    print(f"Validation faces: {len(val_set)}")
    print(f"  Bicubic : PSNR {bicubic_psnr:.2f} dB   SSIM {bicubic_ssim:.4f}")
    print(f"  SRCNN   : PSNR {srcnn_psnr:.2f} dB   SSIM {srcnn_ssim:.4f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    montage_path = config.RESULTS_DIR / "reconstructions.png"
    save_montage(model=model, dataset=val_set, device=device, out_path=montage_path)
    print(f"Montage saved to {montage_path}")


if __name__ == "__main__":
    main()
