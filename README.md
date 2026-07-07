# Face Super-Resolution — Reconstructing HR faces from noisy low-res shots

Companion code for the article *"Teaching a Small CNN to Undo a Bad Photo:
Face Super-Resolution from Scratch."* It downloads a common face dataset
([LFW](http://vis-www.cs.umass.edu/lfw/)), lowers each face to a small noisy
"shot" (rotation, motion blur, salt & pepper, zoom, sensor noise, JPEG), and
trains the classic **SRCNN** to reconstruct the sharp original. It also trains a
**multi-frame** variant (MFSR) that fuses several noisy shots of one face, and
compares the two. The whole thing is config-driven, so pointing it at any other
image folder re-runs the same "lower and retrain" flow unchanged.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt   # uses pip's cache

# For an NVIDIA GPU, replace the CPU torch with a CUDA build (training auto-uses it):
pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
```

## Run

Run every script from the project root so the relative paths resolve:

```bash
python src/download_data.py     # fetch LFW faces -> 128x128 HR crops in data/hr/
python src/train.py             # train SRCNN -> checkpoints/ + results/metrics.json
python src/reconstruct.py       # score vs bicubic + write a comparison montage
python src/train_mfsr.py        # train the multi-frame models -> results/mfsr_metrics.json
```

`src/download_data.py` downloads and caches LFW on first run; later runs reuse
the cache. `src/train.py` auto-detects a CUDA GPU and falls back to the CPU.

## Use it on your own images

Every tunable lives in [`config.py`](config.py). To build the SR dataset from
your own photos instead of LFW, set `INPUT_DIR` to a folder of images and rerun:

```python
# config.py
INPUT_DIR = r"D:\my_photos"   # any folder tree of .jpg/.png/... images
```

Then `python src/download_data.py` centre-crops and resizes them into `data/hr/`,
and `python src/train.py` retrains on them. The scale factor, HR size, degradation
strengths, and training schedule are all in `config.py`.

## Layout

```
config.py            all tunable settings (paths, scale, degradation, training)
src/                 the pipeline: download -> degrade -> dataset -> model -> train -> reconstruct
                     (train_mfsr.py adds the multi-frame sweep)
data/hr/             generated HR crops (created on first run)
checkpoints/         trained SRCNN + MFSR weights
results/             metrics.json, mfsr_metrics.json + reconstruction montage
```

## Credits

The dataset, libraries, and the SRCNN method are credited in
[RESOURCES.md](RESOURCES.md).
