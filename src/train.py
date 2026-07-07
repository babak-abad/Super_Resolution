"""Train SRCNN to reconstruct HR faces from degraded LR input.

Saves the best-by-validation-PSNR weights to checkpoints/ and the full training
history to results/metrics.json (used by the article's training-curve figure).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dataset import make_loaders
from model import Sr_Cnn
from utils import compute_psnr, pick_device, set_seed


def _batch_psnr(pred, target):
    """Mean PSNR across a batch of CxHxW tensors."""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return float(
        np.mean([compute_psnr(p, t) for p, t in zip(pred, target)])
    )


def evaluate(model, loader, device):
    """Return mean SRCNN PSNR and mean bicubic-input PSNR over a loader."""
    model.eval()
    srcnn_psnr, bicubic_psnr = [], []
    with torch.no_grad():
        for lr_up, hr in loader:
            lr_up, hr = lr_up.to(device), hr.to(device)
            pred = model(lr_up).clamp(0.0, 1.0)
            srcnn_psnr.append(_batch_psnr(pred=pred, target=hr))
            bicubic_psnr.append(_batch_psnr(pred=lr_up, target=hr))
    return float(np.mean(srcnn_psnr)), float(np.mean(bicubic_psnr))


def train():
    set_seed(seed=config.SEED)
    device = pick_device()
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders()
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = Sr_Cnn().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.CHECKPOINT_DIR / config.CHECKPOINT_NAME

    history = {"train_loss": [], "val_psnr": [], "bicubic_psnr": []}
    best_psnr = -1.0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for lr_up, hr in train_loader:
            lr_up, hr = lr_up.to(device), hr.to(device)
            optimizer.zero_grad()
            loss = criterion(model(lr_up), hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * lr_up.size(0)
        epoch_loss /= len(train_loader.dataset)

        val_psnr, bicubic_psnr = evaluate(
            model=model, loader=val_loader, device=device
        )
        history["train_loss"].append(epoch_loss)
        history["val_psnr"].append(val_psnr)
        history["bicubic_psnr"].append(bicubic_psnr)
        print(
            f"Epoch {epoch:3d}/{config.EPOCHS}  loss {epoch_loss:.5f}  "
            f"val PSNR {val_psnr:.2f} dB  (bicubic {bicubic_psnr:.2f} dB)"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), checkpoint_path)

    metrics = {
        "epochs": config.EPOCHS,
        "scale": config.SCALE,
        "hr_size": config.HR_SIZE,
        "best_val_psnr": best_psnr,
        "final_bicubic_psnr": history["bicubic_psnr"][-1],
        "history": history,
    }
    with open(config.RESULTS_DIR / config.METRICS_NAME, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\nBest val PSNR {best_psnr:.2f} dB  ->  {checkpoint_path}")
    print(f"Metrics saved to {config.RESULTS_DIR / config.METRICS_NAME}")


if __name__ == "__main__":
    train()
