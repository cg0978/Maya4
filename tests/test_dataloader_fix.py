"""Live bucket smoke tests, disabled by default for CI stability."""

from __future__ import annotations

import os

import pytest
import zarr

from maya4 import (
    DEFAULT_BUCKET_ID,
    download_metadata_from_product,
    fetch_chunk_from_bucket_zarr,
    get_sar_dataloader,
    list_base_files_in_bucket,
)

pytestmark = pytest.mark.network


def _network_enabled() -> bool:
    return os.getenv("MAYA4_RUN_NETWORK_TESTS") == "1"


@pytest.mark.skipif(not _network_enabled(), reason="set MAYA4_RUN_NETWORK_TESTS=1 to run live bucket smoke tests")
def test_live_bucket_smoke(tmp_path):
    product = next(name for name in list_base_files_in_bucket(DEFAULT_BUCKET_ID, relative_path=True) if name.endswith(".zarr"))

    download_metadata_from_product(
        zfile_name=product,
        local_dir=tmp_path,
        bucket_id=DEFAULT_BUCKET_ID,
        levels=["rcmc", "az"],
        show_progress=False,
    )
    fetch_chunk_from_bucket_zarr(
        level="rcmc",
        y=0,
        x=0,
        local_dir=tmp_path,
        bucket_id=DEFAULT_BUCKET_ID,
        zarr_archive=product,
        show_progress=False,
    )
    fetch_chunk_from_bucket_zarr(
        level="az",
        y=0,
        x=0,
        local_dir=tmp_path,
        bucket_id=DEFAULT_BUCKET_ID,
        zarr_archive=product,
        show_progress=False,
    )

    rcmc = zarr.open_array(str(tmp_path / product / "rcmc"), mode="r")
    az = zarr.open_array(str(tmp_path / product / "az"), mode="r")
    assert rcmc.shape == az.shape
    assert rcmc.chunks == az.chunks

    loader = get_sar_dataloader(
        data_dir=str(tmp_path),
        bucket_id=DEFAULT_BUCKET_ID,
        level_from="rcmc",
        level_to="az",
        batch_size=1,
        num_workers=0,
        patch_mode="rectangular",
        patch_size=(128, 128),
        buffer=(0, 0),
        stride=(128, 128),
        shuffle_files=False,
        patch_order="chunk",
        complex_valued=True,
        save_samples=False,
        backend="zarr",
        verbose=False,
        samples_per_prod=1,
        cache_size=8,
        online=True,
        max_products=1,
        use_balanced_sampling=False,
    )

    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape[0] == 1
    assert y_batch.shape[0] == 1
