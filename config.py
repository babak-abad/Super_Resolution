"""Central settings for the super-resolution pipeline.

Every tunable value the pipeline uses lives here, so behaviour changes in one
place. To run the whole "lower any image folder and retrain" flow on your own
data, set INPUT_DIR to a folder of images and rerun download_data -> train.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LFW_DIR = DATA_DIR / "lfw"
HR_DIR = DATA_DIR / "hr"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

# Point this at any folder of images to build an SR dataset from your own data.
# When None, the pipeline downloads LFW faces instead.
INPUT_DIR = None

# ---------------------------------------------------------------------------
# Super-resolution task
# ---------------------------------------------------------------------------
HR_SIZE = 128          # high-resolution square face crop (pixels)
SCALE = 4              # downscale factor -> LR is HR_SIZE // SCALE (32 px)
LR_SIZE = HR_SIZE // SCALE

# ---------------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------------
MAX_IMAGES = 2000      # cap how many faces to ingest (keeps a demo run quick)
LFW_MIN_FACES = 10     # LFW: keep people with at least this many photos
VAL_FRACTION = 0.15
SEED = 42

# ---------------------------------------------------------------------------
# Degradation ("one noisy low-res shot") - read by src/degrade.py
# ---------------------------------------------------------------------------
DEGRADE = {
    "rotate_limit": 8,          # +/- degrees of camera rotation
    "motion_blur_limit": 7,     # max motion-blur kernel size (odd)
    "zoom_scale": (0.9, 1.1),   # affine scale jitter = zoom in / out
    "gauss_noise_std": (0.02, 0.12),  # sensor noise std (fraction of 255)
    "salt_pepper_amount": 0.02, # fraction of pixels hit by salt & pepper
    "jpeg_quality": (40, 80),   # JPEG compression quality range
    "prob": 0.9,                # per-transform apply probability
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 0        # 0 is safest on Windows
CHECKPOINT_NAME = f"srcnn_x{SCALE}.pt"
METRICS_NAME = "metrics.json"

# ---------------------------------------------------------------------------
# Multi-frame super-resolution (MFSR) - fuse several noisy shots of one face
# ---------------------------------------------------------------------------
NUM_FRAMES = 5              # LR shots fused per reconstruction (the primary model)
FRAME_SWEEP = [2, 3, 5, 7]  # frame counts trained for the PSNR-vs-frames figure
MFSR_EPOCHS = 30           # enough to converge on this small model
MFSR_NUM_WORKERS = 6       # N degradations/sample -> parallelise the augmentation
MFSR_METRICS_NAME = "mfsr_metrics.json"


def mfsr_checkpoint_name(num_frames):
    """Per-frame-count MFSR checkpoint filename."""
    return f"mfsr_x{SCALE}_n{num_frames}.pt"
