<div align="center">

<img src="./src/Maya4.png" alt="Maya4 Logo" width="1200" />


### Multi-Level SAR Processing & PyTorch DataLoader

*Unveiling the layers of Synthetic Aperture Radar data from Sentinel-1 missions*

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/sirbastiano/Maya4/actions/workflows/ci.yml/badge.svg?branch=pypi)](https://github.com/sirbastiano/Maya4/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/maya4.svg)](https://pypi.org/project/maya4/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![HF Bucket](https://img.shields.io/badge/🤗%20Hugging%20Face-ESA--philab%2FMaya4-yellow)](https://huggingface.co/buckets/ESA-philab/Maya4)

[Overview](#-overview) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Processing Levels](#-processing-levels) •
[Citation](#-citation)

</div>

---

## 🎯 Overview

Maya4 is a production-ready Python package dedicated to curating and providing **multi-level intermediate SAR representations** from Sentinel-1 acquisitions, spanning the entire processing chain from Level 0 (raw) to Level 1 (focused imagery).

Try it directly in Colab: [Maya4 Demo Notebook](https://colab.research.google.com/drive/1TqG0ldv87-dDlUJyL0qMdlRnu5AzjJj9?usp=sharing).

### The Māyā Philosophy

The name **Maya4** draws inspiration from the *Māyā veil* in philosophy, where reality is hidden behind successive layers—just as radar echoes undergo multiple transformations before forming a final SAR image. Each processing level reveals a different aspect of the electromagnetic interaction with Earth's surface.

### Why Maya4?

- **🎚️ Multi-Level Access**: Complete processing chain from raw echoes to focused imagery
- **🚀 Performance**: Zarr-based storage with intelligent chunk caching and lazy loading
- **🔧 Flexibility**: Access any intermediate representation for research and experimentation
- **☁️ Cloud-Native**: Native Hugging Face bucket integration with lazy metadata and chunk downloads
- **📊 ML-Ready**: PyTorch-compatible dataloaders optimized for pre-training workflows
- **🌍 Geographic-Aware**: Built-in support for location-based clustering and filtering

---

## 🌐 Processing Levels

<img src="./src/intermediates.jpg" alt="Maya4 Steps" width="1200" />


Maya4 exposes the complete SAR processing chain through intermediate signal representations:

| Level | Abbrev. | Description | Purpose / Value |
|-------|---------|-------------|-----------------|
| 📡 **Raw** | `raw` | Unprocessed radar echoes as recorded by Sentinel-1 | Baseline data; enables full custom SAR processing |
| 🎚️ **Range Compressed** | `rc` | Echoes compressed in range via matched filtering | Improved SNR; isolates scatterers along range |
| 🎯 **Range Cell Migration Corrected** | `rcmc` | Motion-compensated with corrected range migration | Preserves geometric fidelity; enables azimuth focusing |
| 🖼️ **Azimuth Compressed** | `az` | Fully focused SAR image in slant-range geometry | Standard Level-1 product; interpretable imagery |

Each level represents a distinct transformation in the SAR focusing pipeline, allowing researchers to:
- **Experiment** with custom processing algorithms
- **Pre-train** deep learning models on intermediate representations
- **Analyze** signal characteristics at different processing stages
- **Develop** novel focusing techniques

---

## ☁️ Bucket Access

Maya4 online loading now targets the public Hugging Face bucket:

- Bucket: [`ESA-philab/Maya4`](https://huggingface.co/buckets/ESA-philab/Maya4)
- Online mode mirrors products locally as `data_dir/<product>.zarr`
- Only the required metadata and Zarr chunks are downloaded on demand
- The current public bucket does not require authentication; `huggingface-cli login` is only needed if a future bucket becomes private or gated

*Data provided by the Copernicus Sentinel-1 mission (ESA)*

---

## ✨ Features

<table>
<tr>
<td width="50%">

### Core Capabilities
- **Multi-Level Data Access**  
  Complete processing chain from raw to focused
  
- **Zarr Backend**  
  Scalable, chunked storage for large SAR products
  
- **Normalization Suite**  
  MinMax, Z-Score, Robust, and Adaptive strategies
  
- **Hugging Face Bucket Integration**  
  Direct loading from `ESA-philab/Maya4`

</td>
<td width="50%">

### Advanced Features
- **Geographic Clustering**  
  Balanced sampling by location distribution
  
- **Positional Encoding**  
  Built-in transformer-compatible embeddings
  
- **Flexible Patch Modes**  
  Rectangular and parabolic extraction
  
- **Lazy Loading**  
  Memory-efficient processing of massive datasets

</td>
</tr>
</table>

---

## 📦 Installation

### Quick Install

```bash
# From PyPI (recommended)
pip install maya4

# Using PDM
pdm install

# Using pip (development)
pip install -e .
```

PyPI releases are published automatically from matching git tags such as `v0.1.3`.

### Environment-Specific Installation

<details>
<summary><b>Jupyter Environment</b></summary>

```bash
pdm install -G jupyter_env
```

Includes Jupyter notebook and lab dependencies for interactive development.
</details>

<details>
<summary><b>Geospatial Features</b></summary>

```bash
pdm install -G geospatial
```

Adds geographic processing tools and coordinate system support.
</details>

<details>
<summary><b>Development Setup</b></summary>

```bash
pdm install -G dev
```

Installs testing, linting, and development utilities.
</details>

<details>
<summary><b>Complete Installation</b></summary>

```bash
pdm install -G :all
```

Installs all optional dependencies for full functionality.
</details>

### Requirements

- Python 3.12+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🛠️ CI & Releases

- Pull requests and pushes to `pypi` run offline CI on Ubuntu with Python 3.12 and 3.13.
- Live Hugging Face bucket smoke tests run in a separate `Network Smoke` workflow on demand and weekly; they are not PR-blocking.
- PyPI publishing is triggered by pushing a matching tag `vX.Y.Z`.

### Maintainer Release Guide

1. Update the version in `pyproject.toml` and `maya4/__init__.py`.
2. Run the offline release checks:

```bash
pdm install -G dev
pdm run pytest -q -m "not network"
pdm build
pdm run python -m twine check dist/*.tar.gz dist/*.whl
```

3. Commit and push the version bump, then create and push the matching tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

4. Ensure PyPI trusted publishing is configured for `sirbastiano/Maya4`. The `publish.yml` workflow verifies the tag/version match and uploads to PyPI automatically.

## 🚀 Quick Start

```python
from maya4 import DEFAULT_BUCKET_ID, SampleFilter, get_sar_dataloader

filters = SampleFilter(
    years=[2025],
    polarizations=["vv"],
    stripmap_modes=[1, 4],
)

loader = get_sar_dataloader(
    data_dir="./data",
    bucket_id=DEFAULT_BUCKET_ID,
    filters=filters,
    level_from="rcmc",
    level_to="az",
    batch_size=1,
    patch_size=(256, 256),
    stride=(256, 256),
    buffer=(0, 0),
    online=True,
    use_balanced_sampling=False,
)

x_batch, y_batch = next(iter(loader))
print(x_batch.shape, y_batch.shape)
```

Notes:

- `online=True` downloads metadata and chunks from the bucket on demand.
- Flat local cache layout is now the default online layout: `./data/<product>.zarr/...`.
- Legacy nested local layouts like `./data/PT1/<product>.zarr` still load in offline mode.

---

## 📖 Citation

If you use Maya4 datasets or tools in your research, please cite:

```bibtex
@software{maya4_2024,
  author       = {Del Prete, Roberto and Maya4 Organization},
  title        = {Maya4: Multi-Level SAR Processing and Intermediate Representations},
  year         = {2024},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/buckets/ESA-philab/Maya4}},
  note         = {Sentinel-1 Stripmap data spanning processing levels from raw to focused imagery}
}
```


---

<div align="center">

Made with ❤️ by the Maya4 Team

</div>
