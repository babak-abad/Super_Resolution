"""Train the multi-frame SRCNN for every frame count in config.FRAME_SWEEP.

Each model fuses N bicubic-upsampled LR shots of one face into a single HR
reconstruction. For every N we save the best-by-validation-PSNR weights to
checkpoints/mfsr_x{scale}_n{N}.pt and append the run to results/mfsr_metrics.json,
which the article's MFSR figures read. Two baselines are tracked per epoch: a
single upsampled shot (the SISR starting point) and the plain average of the N
shots (classical frame fusion), so the learned gain is easy to see.
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
from model import Sr_Cnn_Mf
from utils import compute_psnr, pick_device, set_seed


def _batch_psnr(pred, target):
    """Mean PSNR across a batch of CxHxW tensors."""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return float(np.mean([compute_psnr(p, t) for p, t in zip(pred, target)]))


def _frame_average(lr_stack, num_frames):
    """Mean of the N stacked upsampled shots -> a BxCxHxW tensor."""
    b, cn, h, w = lr_stack.shape
    return lr_stack.view(b, num_frames, cn // num_frames, h, w).mean(dim=1)


def evaluate(model, loader, device, num_frames):
    """Return mean PSNR for the model, one bicubic shot, and the frame average."""
    model.eval()
    model_psnr, single_psnr, avg_psnr = [], [], []
    with torch.no_grad():
        for lr_stack, hr in loader:
            lr_stack, hr = lr_stack.to(device), hr.to(device)
            pred = model(lr_stack).clamp(0.0, 1.0)
            model_psnr.append(_batch_psnr(pred=pred, target=hr))
            single_psnr.append(_batch_psnr(pred=lr_stack[:, :3], target=hr))
            avg = _frame_average(lr_stack, num_frames)
            avg_psnr.append(_batch_psnr(pred=avg, target=hr))
    return (
        float(np.mean(model_psnr)),
        float(np.mean(single_psnr)),
        float(np.mean(avg_psnr)),
    )


def train_one(num_frames, device):
    """Train a single MFSR model that fuses num_frames shots; return its run."""
    print(f"\n=== MFSR  N={num_frames} frames ===")
    set_seed(seed=config.SEED)
    train_loader, val_loader = make_loaders(
        num_frames=num_frames, num_workers=config.MFSR_NUM_WORKERS
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = Sr_Cnn_Mf(num_frames=num_frames).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    checkpoint_path = config.CHECKPOINT_DIR / config.mfsr_checkpoint_name(num_frames)

    history = {"train_loss": [], "val_psnr": [], "single_psnr": [], "avg_psnr": []}
    best_psnr = -1.0

    for epoch in range(1, config.MFSR_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for lr_stack, hr in train_loader:
            lr_stack, hr = lr_stack.to(device), hr.to(device)
            optimizer.zero_grad()
            loss = criterion(model(lr_stack), hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * lr_stack.size(0)
        epoch_loss /= len(train_loader.dataset)

        val_psnr, single_psnr, avg_psnr = evaluate(
            model=model, loader=val_loader, device=device, num_frames=num_frames
        )
        history["train_loss"].append(epoch_loss)
        history["val_psnr"].append(val_psnr)
        history["single_psnr"].append(single_psnr)
        history["avg_psnr"].append(avg_psnr)
        print(
            f"Epoch {epoch:3d}/{config.MFSR_EPOCHS}  loss {epoch_loss:.5f}  "
            f"MFSR {val_psnr:.2f} dB  (1 shot {single_psnr:.2f}, "
            f"avg {avg_psnr:.2f})"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), checkpoint_path)

    print(f"N={num_frames}: best val PSNR {best_psnr:.2f} dB -> {checkpoint_path}")
    return {
        "num_frames": num_frames,
        "num_params": n_params,
        "best_val_psnr": best_psnr,
        "final_single_psnr": history["single_psnr"][-1],
        "final_avg_psnr": history["avg_psnr"][-1],
        "history": history,
    }


def train():
    device = pick_device()
    print(f"Device: {device}   frame sweep: {config.FRAME_SWEEP}")
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    runs = {str(n): train_one(num_frames=n, device=device) for n in config.FRAME_SWEEP}
    metrics = {
        "epochs": config.MFSR_EPOCHS,
        "scale": config.SCALE,
        "hr_size": config.HR_SIZE,
        "frame_sweep": config.FRAME_SWEEP,
        "runs": runs,
    }
    out_path = config.RESULTS_DIR / config.MFSR_METRICS_NAME
    with open(out_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"\nMFSR metrics saved to {out_path}")


if __name__ == "__main__":
    train()
