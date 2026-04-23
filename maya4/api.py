"""
Bucket-native Hugging Face access helpers for Maya4 products.

The Maya4 online flow now targets the public Hugging Face bucket
``ESA-philab/Maya4`` instead of legacy dataset repositories.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import warnings
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

from huggingface_hub import download_bucket_files, list_bucket_tree

from maya4.utils import get_chunk_name_from_coords

DEFAULT_BUCKET_ID = "ESA-philab/Maya4"
LEGACY_MODALITIES = ["raw", "rc", "rcmc", "az"]


try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0)
            self.desc = kwargs.get("desc", "")
            self.n = 0

        def update(self, n):
            self.n += n

        def set_description(self, desc):
            self.desc = desc
            print(f"\r{desc}", end="", flush=True)

        def close(self):
            print()


_download_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


@contextmanager
def file_download_lock(file_path: str):
    """Prevent concurrent downloads of the same remote object."""
    lock_key = hashlib.md5(file_path.encode(), usedforsecurity=False).hexdigest()
    with _locks_lock:
        if lock_key not in _download_locks:
            _download_locks[lock_key] = threading.Lock()
        lock = _download_locks[lock_key]

    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _locks_lock:
            if lock_key in _download_locks and not lock.locked():
                time.sleep(0.001)
                if not lock.locked():
                    _download_locks.pop(lock_key, None)


def create_colored_progress_bar(desc: str, total: Optional[int] = None, color: str = "blue"):
    """Create a lightweight colored progress bar."""
    colors = {
        "blue": "\033[34m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "cyan": "\033[36m",
        "magenta": "\033[35m",
        "reset": "\033[0m",
    }
    color_code = colors.get(color.lower(), colors["blue"])
    reset_code = colors["reset"]
    display_total = total if total is not None else 1

    try:
        return tqdm(
            total=display_total,
            desc=f"{color_code}{desc}{reset_code}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            bar_format=(
                f"{color_code}{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
                "{{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}]"
                f"{{postfix}}{reset_code}"
            ),
            dynamic_ncols=True,
            position=0,
        )
    except Exception:
        class SimpleFallbackProgressBar:
            def __init__(self):
                self.total = display_total
                self.n = 0
                self.desc = desc

            def update(self, n):
                self.n += n

            def set_description(self, new_desc):
                self.desc = new_desc
                print(f"\r{new_desc}", end="", flush=True)

            def close(self):
                print()

        return SimpleFallbackProgressBar()


def _warn_legacy_alias(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"{old_name} is deprecated and bucket-only Maya4 now uses {new_name}.",
        DeprecationWarning,
        stacklevel=2,
    )


def _running_in_dataloader_worker() -> bool:
    try:
        import torch

        return torch.utils.data.get_worker_info() is not None
    except (ImportError, AttributeError):
        return False


def filter_files_by_modalities(files: Iterable[Any], filters: list[str]) -> list[Any]:
    """Filter bucket files by modality keywords."""
    assert isinstance(filters, list), "filters must be a list"
    return [item for item in files if getattr(item, "type", None) == "file" and any(token in item.path for token in filters)]


def download_file_from_bucket(
    bucket_id: str,
    filename: str,
    local_dir: Union[str, os.PathLike],
    show_progress: bool = True,
) -> Path:
    """
    Download a specific file from a Hugging Face bucket with per-object locking.
    """
    if _running_in_dataloader_worker():
        show_progress = False

    local_dir_path = Path(local_dir)
    local_path = local_dir_path / filename
    lock_id = f"{bucket_id}:{filename}"

    if local_path.exists() and local_path.stat().st_size > 0:
        if show_progress:
            print(f'\033[34m\u2713 File "{filename}" already exists locally\033[0m')
        return local_path

    with file_download_lock(lock_id):
        if local_path.exists() and local_path.stat().st_size > 0:
            if show_progress:
                print(f'\033[34m\u2713 File "{filename}" was downloaded by another worker\033[0m')
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        pbar = None
        if show_progress:
            try:
                pbar = create_colored_progress_bar(
                    desc=f"Downloading {os.path.basename(filename)}",
                    color="blue",
                )
                pbar.update(0)
            except Exception:
                print(f"\033[34mDownloading {os.path.basename(filename)}...\033[0m")

        try:
            download_bucket_files(
                bucket_id,
                files=[(filename, local_path)],
                raise_on_missing_files=True,
            )
            if not local_path.exists():
                raise FileNotFoundError(f'File "{local_path}" not found after download.')

            if show_progress and pbar is not None:
                try:
                    file_size = local_path.stat().st_size
                    pbar.total = file_size
                    pbar.update(file_size)
                    pbar.set_description(f"\033[34m\u2713 Downloaded {os.path.basename(filename)}\033[0m")
                    time.sleep(0.1)
                    pbar.close()
                except Exception:
                    print(f"\033[34m\u2713 Downloaded {os.path.basename(filename)}\033[0m")
            elif show_progress:
                print(f"\033[34m\u2713 Downloaded {os.path.basename(filename)}\033[0m")

            return local_path
        except Exception as exc:
            if show_progress and pbar is not None:
                try:
                    pbar.set_description(f"\033[31m\u2717 Failed to download {os.path.basename(filename)}\033[0m")
                    time.sleep(0.1)
                    pbar.close()
                except Exception:
                    pass
            if show_progress:
                print(f"\033[31m\u2717 Failed to download {os.path.basename(filename)}: {exc}\033[0m")
            raise


def list_base_files_in_bucket(bucket_id: str, prefix: str = "", relative_path: bool = False) -> list[str]:
    """List direct children under a bucket prefix."""
    items = list_bucket_tree(bucket_id=bucket_id, prefix=prefix or None, recursive=False)
    names = [item.path for item in items if hasattr(item, "path")]
    if relative_path:
        names = [os.path.basename(name.rstrip("/")) for name in names]
    return names


def list_files_in_bucket(bucket_id: str, prefix: str, filters: list[str]) -> list[Any]:
    """List all files under a bucket prefix and filter them by modality."""
    assert isinstance(filters, list), "filters must be a list"
    assert all(token in LEGACY_MODALITIES for token in filters), f"Each filter must be one of {LEGACY_MODALITIES}"

    items = list_bucket_tree(bucket_id=bucket_id, prefix=prefix, recursive=True)
    files = [item for item in items if getattr(item, "type", None) == "file"]
    return filter_files_by_modalities(files, filters)


def parser():
    import argparse

    arg_parser = argparse.ArgumentParser(description="Download files from a Maya4 Hugging Face bucket.")
    arg_parser.add_argument(
        "--bucket_id",
        type=str,
        default=DEFAULT_BUCKET_ID,
        help=f"Bucket ID (default: {DEFAULT_BUCKET_ID})",
    )
    arg_parser.add_argument(
        "--path_in_bucket",
        type=str,
        default="s1c-s1-raw-s-vv-20250328t052810-20250328t052835-001637-002a0d.zarr",
        help="Path in the bucket to list files from.",
    )
    arg_parser.add_argument(
        "--filters",
        type=str,
        nargs="+",
        default=["rc", "az"],
        help='List of modalities to filter by (default: ["rc", "az"])',
    )
    arg_parser.add_argument("--local_dir", type=str, default=None, help="Local directory to download files to.")
    return arg_parser.parse_args()


def download_wrapper(file_info: Any, bucket_id: str, output_dir: Optional[str], show_progress: bool = True) -> None:
    """Wrapper used by multiprocessing downloads."""
    filename = file_info.path
    local_dir = output_dir or "."
    try:
        download_file_from_bucket(bucket_id, filename, local_dir, show_progress=show_progress)
    except FileNotFoundError as exc:
        if show_progress:
            print(f"\033[31mError downloading {filename}: {exc}\033[0m")
        else:
            print(f"Error: {exc}")


def down(
    bucket_id: str = DEFAULT_BUCKET_ID,
    path_in_bucket: str = "s1c-s1-raw-s-vv-20250328t052810-20250328t052835-001637-002a0d.zarr",
    filters: list[str] = ["rc", "az"],
    output_dir: Optional[str] = None,
    show_progress: bool = True,
    max_workers: Optional[int] = None,
) -> None:
    """Download files from a bucket prefix filtered by modality."""
    files = list_files_in_bucket(bucket_id, path_in_bucket, filters)
    if show_progress:
        print(f"\033[36mFound {len(files)} files matching the filter criteria\033[0m")
    else:
        print(f"Found {len(files)} files matching the filter criteria.")

    if len(files) > 1 and max_workers != 1:
        with Pool(processes=max_workers) as pool:
            pool.starmap(download_wrapper, [(item, bucket_id, output_dir, show_progress) for item in files])
    else:
        for item in files:
            download_wrapper(item, bucket_id, output_dir, show_progress)


def download_metadata(
    bucket_id: str = DEFAULT_BUCKET_ID,
    zarr_archive: str = "s1c-s1-raw-s-vv-20250328t052810-20250328t052835-001637-002a0d.zarr",
    base_dir: str = "",
    local_dir: Optional[Union[str, os.PathLike]] = None,
    download_all: bool = True,
    show_progress: bool = True,
) -> Path:
    """
    Download Zarr metadata files from a Maya4 product in the bucket.
    """
    local_dir_path = Path(local_dir or ".")
    if base_dir:
        base_remote_metadata_path = f"{zarr_archive}/{base_dir}"
        base_local_metadata_path = local_dir_path / zarr_archive / base_dir
    else:
        base_remote_metadata_path = zarr_archive
        base_local_metadata_path = local_dir_path / zarr_archive

    meta_candidates = [".zgroup", ".zarray", "zarr.json"]
    misc_metadata = [".zattrs"]

    files = list_base_files_in_bucket(bucket_id, prefix=base_remote_metadata_path, relative_path=True)
    if show_progress:
        print(f"\033[36mChecking metadata in {base_remote_metadata_path}\033[0m")

    if download_all:
        for meta in misc_metadata:
            meta_path = base_local_metadata_path / meta
            if meta in files and not meta_path.exists():
                download_file_from_bucket(bucket_id, f"{base_remote_metadata_path}/{meta}", local_dir_path, show_progress=show_progress)

    meta_path = None
    for meta in meta_candidates:
        if meta not in files:
            continue
        candidate_path = base_local_metadata_path / meta
        if not candidate_path.exists():
            download_file_from_bucket(bucket_id, f"{base_remote_metadata_path}/{meta}", local_dir_path, show_progress=show_progress)
        meta_path = candidate_path
        break

    if meta_path is not None:
        return meta_path
    raise FileNotFoundError(f"No metadata file (.zarray or zarr.json) found in {base_remote_metadata_path}.")


def download_metadata_from_product(
    zfile_name: str = "s1c-s1-raw-s-vv-20250328t052810-20250328t052835-001637-002a0d.zarr",
    local_dir: Union[str, os.PathLike] = "data",
    bucket_id: str = DEFAULT_BUCKET_ID,
    levels: Optional[List[str]] = None,
    show_progress: bool = True,
) -> os.PathLike:
    """Download root and per-level metadata for a product."""
    levels = levels if levels is not None else ["raw", "rc", "rcmc", "az"]
    meta_file_path = download_metadata(
        bucket_id=bucket_id,
        zarr_archive=zfile_name,
        local_dir=local_dir,
        base_dir="",
        show_progress=show_progress,
    )
    for level in levels:
        meta_file_path = download_metadata(
            bucket_id=bucket_id,
            zarr_archive=zfile_name,
            local_dir=local_dir,
            base_dir=level,
            show_progress=show_progress,
        )
    return meta_file_path


def fetch_chunk_from_bucket_zarr(
    level: str,
    y: int,
    x: int,
    local_dir: Union[str, os.PathLike],
    bucket_id: str = DEFAULT_BUCKET_ID,
    zarr_archive: str = "s1c-s1-raw-s-vv-20250328t052810-20250328t052835-001637-002a0d.zarr",
    show_progress: bool = False,
) -> Path:
    """
    Download only the chunk containing ``(y, x)`` from a product Zarr archive.
    """
    download_metadata(bucket_id=bucket_id, zarr_archive=zarr_archive, local_dir=local_dir, show_progress=show_progress)
    zarray_meta_file = download_metadata(
        bucket_id=bucket_id,
        zarr_archive=zarr_archive,
        base_dir=level,
        local_dir=local_dir,
        show_progress=show_progress,
    )

    with open(zarray_meta_file) as handle:
        zarr_meta = json.load(handle)

    if zarr_meta.get("zarr_format", 2) == 3:
        chunks = zarr_meta["chunk_grid"]["configuration"]["chunk_shape"]
    else:
        chunks = zarr_meta["chunks"]

    chunk_fname = get_chunk_name_from_coords(
        y=y,
        x=x,
        zarr_file_name=zarr_archive,
        level=level,
        chunks=chunks,
        version=zarr_meta.get("zarr_format", 2),
    )
    return download_file_from_bucket(bucket_id, chunk_fname, local_dir, show_progress=show_progress)


def download_file_from_hf(
    repo_id: str,
    filename: str,
    local_dir: Union[str, os.PathLike],
    show_progress: bool = True,
) -> Path:
    _warn_legacy_alias("download_file_from_hf", "download_file_from_bucket")
    return download_file_from_bucket(repo_id, filename, local_dir, show_progress=show_progress)


def list_base_files_in_repo(repo_id: str, path_in_repo: str = "", relative_path: bool = False) -> list[str]:
    _warn_legacy_alias("list_base_files_in_repo", "list_base_files_in_bucket")
    return list_base_files_in_bucket(repo_id, prefix=path_in_repo, relative_path=relative_path)


def list_repos_by_author(author: str) -> list[str]:
    _warn_legacy_alias("list_repos_by_author", "DEFAULT_BUCKET_ID")
    return [DEFAULT_BUCKET_ID]


def list_files_in_repo(repo_id: str, path_in_repo: str, filters: list[str]) -> list[Any]:
    _warn_legacy_alias("list_files_in_repo", "list_files_in_bucket")
    return list_files_in_bucket(repo_id, path_in_repo, filters)


def fetch_chunk_from_hf_zarr(
    level: str,
    y: int,
    x: int,
    local_dir: Union[str, os.PathLike],
    repo_id: str = DEFAULT_BUCKET_ID,
    zarr_archive: str = "s1c-s1-raw-s-vv-20250328t052810-20250328t052835-001637-002a0d.zarr",
    show_progress: bool = False,
) -> Path:
    _warn_legacy_alias("fetch_chunk_from_hf_zarr", "fetch_chunk_from_bucket_zarr")
    return fetch_chunk_from_bucket_zarr(
        level=level,
        y=y,
        x=x,
        local_dir=local_dir,
        bucket_id=repo_id,
        zarr_archive=zarr_archive,
        show_progress=show_progress,
    )


if __name__ == "__main__":
    down()
