"""Build the high-resolution image pool the SR dataset is derived from.

By default this downloads LFW colour faces via scikit-learn and writes centre
square crops resized to HR_SIZE into data/hr/. Set config.INPUT_DIR to a folder
of your own images to build the HR pool from those instead - the rest of the
pipeline is unchanged, which is what makes the project reusable.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def center_square_crop(image):
    """Crop the largest centred square from a PIL image."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def prepare_hr(image):
    """Centre-crop to a square and resize to the HR working size."""
    square = center_square_crop(image.convert("RGB"))
    return square.resize((config.HR_SIZE, config.HR_SIZE), Image.BICUBIC)


def _to_uint8(array):
    """sklearn returns floats in [0,1]; normalise to a uint8 RGB array."""
    if array.max() <= 1.5:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def iter_lfw_images():
    """Yield colour PIL faces from LFW (downloaded + cached on first run).

    A tight centred slice of the 250x250 funneled originals is taken at full
    scale so each face is a genuine HR crop, then the images are shuffled so a
    capped run still spans many identities.
    """
    from sklearn.datasets import fetch_lfw_people

    half = config.HR_SIZE // 2
    center = 125  # centre of the 250x250 funneled LFW image
    face_slice = (
        slice(center - half, center + half),
        slice(center - half, center + half),
    )
    people = fetch_lfw_people(
        data_home=str(config.LFW_DIR),
        color=True,
        resize=1.0,
        funneled=True,
        slice_=face_slice,
        min_faces_per_person=config.LFW_MIN_FACES,
        download_if_missing=True,
    )
    images = people.images
    order = np.random.default_rng(config.SEED).permutation(len(images))
    for index in order:
        yield Image.fromarray(_to_uint8(images[index]))


def iter_folder_images(folder):
    """Yield PIL images from every supported file in a folder tree."""
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            yield Image.open(path)


def build_hr_pool():
    """Write HR crops to config.HR_DIR and return how many were written."""
    config.HR_DIR.mkdir(parents=True, exist_ok=True)
    if config.INPUT_DIR:
        source = iter_folder_images(folder=config.INPUT_DIR)
        print(f"Building HR pool from {config.INPUT_DIR}")
    else:
        source = iter_lfw_images()
        print("Building HR pool from LFW faces")

    written = 0
    for image in source:
        if written >= config.MAX_IMAGES:
            break
        hr = prepare_hr(image)
        hr.save(config.HR_DIR / f"face_{written:05d}.png")
        written += 1
        if written % 250 == 0:
            print(f"  {written} images")
    print(f"Wrote {written} HR images to {config.HR_DIR}")
    return written


if __name__ == "__main__":
    build_hr_pool()
