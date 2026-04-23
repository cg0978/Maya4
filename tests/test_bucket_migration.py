from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import maya4.api as api
import maya4.dataloader as dataloader_module
from maya4 import build_local_product_path, get_part_from_filename, parse_product_filename
from maya4.utils import SampleFilter


def test_parse_product_filename_flat_and_nested_paths():
    flat = parse_product_filename("data/s1c-s4-raw-s-vv-20250426t023204-20250426t023237-002058-004414.zarr")
    nested = parse_product_filename("data/PT2/s1a-s3-raw-s-hh-20230331t123129-20230331t123154-047888-05c119.zarr")

    assert flat is not None
    assert flat["satellite"] == "s1c"
    assert flat["stripmap_mode"] == 4
    assert flat["part"] is None

    assert nested is not None
    assert nested["satellite"] == "s1a"
    assert nested["stripmap_mode"] == 3
    assert nested["part"] == "PT2"


def test_part_and_local_path_resolution():
    assert get_part_from_filename("data/PT3/product.zarr") == "PT3"
    assert get_part_from_filename("data/product.zarr") is None
    assert build_local_product_path("data", "product.zarr") == Path("data/product.zarr")
    assert build_local_product_path("data", "product.zarr", "PT1") == Path("data/PT1/product.zarr")


def test_sample_filter_ignores_parts_when_dataframe_has_no_part_values():
    df = pd.DataFrame(
        [
            {"part": None, "acquisition_date": pd.Timestamp("2025-03-28"), "stripmap_mode": 1, "polarization": "vv"},
            {"part": None, "acquisition_date": pd.Timestamp("2025-03-29"), "stripmap_mode": 2, "polarization": "vv"},
        ]
    )
    filtered = SampleFilter(parts=["PT1"])._filter_products(df)
    assert len(filtered) == 2


def test_list_base_files_in_bucket_relative_paths(monkeypatch):
    def fake_list_bucket_tree(bucket_id, prefix=None, recursive=False):
        assert bucket_id == "ESA-philab/Maya4"
        assert recursive is False
        return [
            SimpleNamespace(path="a/product-1.zarr", type="directory"),
            SimpleNamespace(path="a/product-2.zarr", type="directory"),
        ]

    monkeypatch.setattr(api, "list_bucket_tree", fake_list_bucket_tree)
    assert api.list_base_files_in_bucket("ESA-philab/Maya4", prefix="a", relative_path=True) == [
        "product-1.zarr",
        "product-2.zarr",
    ]


def test_download_file_from_bucket(monkeypatch, tmp_path):
    written = {}

    def fake_download_bucket_files(bucket_id, files, raise_on_missing_files=True, token=None):
        assert bucket_id == "ESA-philab/Maya4"
        assert len(files) == 1
        remote_path, local_path = files[0]
        written["remote_path"] = remote_path
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_text("ok")

    monkeypatch.setattr(api, "download_bucket_files", fake_download_bucket_files)
    local_path = api.download_file_from_bucket(
        "ESA-philab/Maya4",
        "product.zarr/zarr.json",
        tmp_path,
        show_progress=False,
    )
    assert written["remote_path"] == "product.zarr/zarr.json"
    assert local_path == tmp_path / "product.zarr" / "zarr.json"
    assert local_path.read_text() == "ok"


def test_list_files_in_bucket_filters_modalities(monkeypatch):
    def fake_list_bucket_tree(bucket_id, prefix=None, recursive=True):
        return [
            SimpleNamespace(path="product.zarr/rcmc/c/0/0", type="file"),
            SimpleNamespace(path="product.zarr/az/c/0/0", type="file"),
            SimpleNamespace(path="product.zarr/raw", type="directory"),
        ]

    monkeypatch.setattr(api, "list_bucket_tree", fake_list_bucket_tree)
    filtered = api.list_files_in_bucket("ESA-philab/Maya4", "product.zarr", ["az"])
    assert [item.path for item in filtered] == ["product.zarr/az/c/0/0"]


def test_get_sar_dataloader_propagates_bucket_id(monkeypatch):
    captured = {}

    class DummyDataset:
        def __init__(self, **kwargs):
            captured["dataset_kwargs"] = kwargs

    class DummySampler:
        def __init__(self, dataset, **kwargs):
            captured["sampler_kwargs"] = kwargs
            self.dataset = dataset

    class DummyLoader:
        def __init__(self, dataset, batch_size, sampler, num_workers, pin_memory, verbose):
            captured["loader_kwargs"] = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "verbose": verbose,
            }

    monkeypatch.setattr(dataloader_module, "SARZarrDataset", DummyDataset)
    monkeypatch.setattr(dataloader_module, "KPatchSampler", DummySampler)
    monkeypatch.setattr(dataloader_module, "SARDataloader", DummyLoader)

    dataloader_module.get_sar_dataloader(
        data_dir="data",
        bucket_id="ESA-philab/Maya4",
        batch_size=4,
        num_workers=0,
        online=True,
        use_balanced_sampling=False,
    )

    assert captured["dataset_kwargs"]["bucket_id"] == "ESA-philab/Maya4"
    assert captured["loader_kwargs"]["batch_size"] == 4
