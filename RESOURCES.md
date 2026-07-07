# Resources &amp; Credits

Every external resource this project uses — the face dataset, the software
libraries, and the paper behind the model. If you reuse the data, respect the
license listed for each.

Repository: <https://github.com/babak-abad/Super_Resolution>

---

## Dataset

- **LFW — Labeled Faces in the Wild** — used as the pool of high-resolution
  faces the super-resolution pairs are derived from. Downloaded via
  scikit-learn's `fetch_lfw_people` loader.
  Gary B. Huang, Manu Ramesh, Tamara Berg, Erik Learned-Miller,
  *Labeled Faces in the Wild: A Database for Studying Face Recognition in
  Unconstrained Environments*, University of Massachusetts, Amherst,
  Technical Report 07-49, October 2007.
  <http://vis-www.cs.umass.edu/lfw/>

  License / usage: LFW is distributed for **research/educational use**. The
  photographs were collected from news articles on the web and their individual
  copyrights are held by their respective owners; LFW makes no copyright claim.
  For commercial publication, replace the dataset via `INPUT_DIR` with images
  you have the rights to (the pipeline is dataset-agnostic).

The degraded low-resolution images and reconstructions this project generates
are derivative works of the LFW photographs.

---

## Software &amp; libraries

- **PyTorch** — the SRCNN model, training loop, and tensors.
  Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning
  Library*, NeurIPS 2019, BSD-3-Clause. <https://pytorch.org>
- **Albumentations** — the degradation pipeline (rotation, motion blur, salt &
  pepper, zoom, Gaussian noise, JPEG compression). MIT.
  Buslaev et al., *Albumentations: Fast and Flexible Image Augmentations*,
  Information, 2020. <https://albumentations.ai>
- **OpenCV** — resize / warp backend used by Albumentations. Apache-2.0.
  <https://opencv.org>
- **scikit-learn** — the LFW dataset loader. BSD-3-Clause.
  <https://scikit-learn.org>
- **scikit-image** — SSIM metric. BSD-3-Clause. <https://scikit-image.org>
- **NumPy** — array math. BSD-3-Clause. <https://numpy.org>
- **Pillow** — image I/O. HPND/MIT-CMU. <https://python-pillow.org>
- **Matplotlib** — figures. J. D. Hunter, *Matplotlib: A 2D Graphics
  Environment*, CiSE 2007, PSF-based license. <https://matplotlib.org>

---

## Method &amp; further reading

- **SRCNN** — the three-layer super-resolution CNN trained here.
  Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, *Image Super-Resolution
  Using Deep Convolutional Networks*, IEEE TPAMI 2016 (arXiv 2014).
  <https://arxiv.org/abs/1501.00092>
- **Multi-frame super-resolution (MFSR)** — the multi-shot approach compared
  against single-image SRCNN.
  Farsiu, Robinson, Elad, Milanfar, *Fast and Robust Multiframe Super
  Resolution*, IEEE TIP 2004. <https://doi.org/10.1109/TIP.2004.834669>
- **Handheld Multi-Frame Super-Resolution** — the modern phone-camera
  ("Super Res Zoom") realisation of the idea.
  Wronski et al., ACM TOG (SIGGRAPH) 2019.
  <https://doi.org/10.1145/3306346.3323024>
- **ESPCN** — sub-pixel upsampling, a common next step beyond SRCNN.
  Shi et al., *Real-Time Single Image and Video Super-Resolution Using an
  Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016.
  <https://arxiv.org/abs/1609.05158>
- **PSNR / SSIM** — the reconstruction-quality metrics.
  Wang, Bovik, Sheikh, Simoncelli, *Image Quality Assessment: From Error
  Visibility to Structural Similarity*, IEEE TIP 2004.
  <https://doi.org/10.1109/TIP.2003.819861>
