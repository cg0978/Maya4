"""Basic package import and transform smoke tests."""

import tomllib
from pathlib import Path

import numpy as np

import maya4

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_project_metadata() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_imports():
    project = _load_project_metadata()

    from maya4 import (
        BaseTransformModule,
        ComplexNormalizationModule,
        KPatchSampler,
        NormalizationModule,
        SampleFilter,
        SARDataloader,
        SARTransform,
        SARZarrDataset,
        download_metadata_from_product,
        fetch_chunk_from_bucket_zarr,
        list_base_files_in_bucket,
    )

    assert SARZarrDataset is not None
    assert KPatchSampler is not None
    assert SARDataloader is not None
    assert SARTransform is not None
    assert SampleFilter is not None
    assert BaseTransformModule is not None
    assert NormalizationModule is not None
    assert ComplexNormalizationModule is not None
    assert list_base_files_in_bucket is not None
    assert fetch_chunk_from_bucket_zarr is not None
    assert download_metadata_from_product is not None
    assert maya4.__version__ == project["version"]
    assert maya4.__author__ == "Roberto Del Prete"


def test_repository_version_consistency():
    project = _load_project_metadata()

    assert project["name"] == "maya4"
    assert project["requires-python"] == ">=3.12"
    assert maya4.__version__ == project["version"]


def test_basic_functionality():
    transforms = maya4.SARTransform.create_minmax_normalized_transform(
        normalize=True,
        rc_min=maya4.RC_MIN,
        rc_max=maya4.RC_MAX,
        gt_min=maya4.GT_MIN,
        gt_max=maya4.GT_MAX,
        complex_valued=True,
    )

    test_data = np.array([1000.0 + 1000.0j], dtype=np.complex128)
    normalized = transforms(test_data, "rcmc")

    assert normalized.shape == test_data.shape
