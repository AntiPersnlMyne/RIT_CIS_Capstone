"""
Filename: dataloader.py
Description: Image dataloader I/O.
- reads single-file datacubes (.npy)
- reads directories of single-band TIFFs (stacks them into a datacube)
- Reads H5 files
- Returns a NumPy memmap with shape (rows, cols, bands).

--------
Examples
--------
# Load a single file (keep default dtype and raw data)
cube = load_datacube("/path/to/dataset.npy")

# Load a directory of single-band TIFFs, request float64, get min-max normalization and custom output filename
cube = load_datacube("/path/to/band_dir", dtype=np.float64, output_path="/output/my_cube.npy", normalization="minmax")
"""

import os
import h5py
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rasterio.env import Env
from math import ceil, sqrt
import tifffile
import inspect

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "01-31-2025"
__email__ = "mt9485@rit.edu"


# ---------------------------
# Helpers
# ---------------------------


def _find_tiff_files(dirpath: str) -> list[str]:
    """Return a list of paths of found TIFF files from directory"""

    return sorted(str(p) for p in Path(dirpath).glob("*.[tT][iI][fF]*"))


def _dir_to_npy(
    src_path: str,
    dst_path: str,
    dst_dtype: np.dtype,
    normalize: bool,
) -> np.memmap:

    # ----------------------------------------
    # Locate files
    # ----------------------------------------

    # Search for TIFF files
    file_dirs = _find_tiff_files(src_path)

    # ----------------------------------------
    # Open files
    # ----------------------------------------

    open_files_list = []

    try:
        for file_dir in file_dirs:
            open_files_list.append(rasterio.open(file_dir))
    except Exception:
        # Clean up any opened files
        for f in open_files_list:
            f.close()
        # Then raise error
        raise

    # ----------------------------------------
    # Create datacube (np.memmap) object
    # ----------------------------------------

    # Output dimensions from test image
    with rasterio.open(file_dirs[0]) as src:
        rows, cols, num_bands = src.height, src.width, len(open_files_list)

    # Create datacube (np.memmap) object
    dst_datacube = np.lib.format.open_memmap(
        dst_path,
        mode="w+",
        dtype=dst_dtype,
        shape=(rows, cols, num_bands),
    )

    # Calculate results on RAM (temp)
    # Store results on disk (dst)
    temp_datacube = np.empty((num_bands, rows, cols), dtype=dst_dtype, order="C")

    # ----------------------------------------
    # Populate datacube - First Pass
    # ----------------------------------------

    for dst_idx, file in enumerate(
        tqdm(open_files_list, desc="band writing", unit="band", colour="green")
    ):

        # Read band from file, convert to dst_dtype
        band = file.read(1, out_dtype=dst_dtype)

        # Write band to memmap
        temp_datacube[dst_idx] = band[:]

    # ----------------------------------------
    # Normalize datacube - Second Pass
    # ----------------------------------------
    if normalize:
        # Magnitue of pixel vectors
        norms = np.linalg.norm(temp_datacube, axis=0, keepdims=True)

        # Identify where norms are zero to avoid division by zero
        nonzero_mask = norms > 1e-12

        # Perform division only on non-zero elements
        np.divide(temp_datacube, norms, out=temp_datacube, where=nonzero_mask)

    # Close open band files
    for file in open_files_list:
        file.close()

    # Move bands to last axis
    temp_datacube = np.moveaxis(temp_datacube, 0, -1)

    # Move data to memmap
    dst_datacube[:] = temp_datacube[:]
    dst_datacube.flush()

    # Return opened datacube object with read/write permissions
    return np.lib.format.open_memmap(
        dst_path, mode="r", dtype=dst_dtype, shape=(rows, cols, num_bands)
    )


def _tiff_to_npy(
    src_path: str,
    dst_path: str,
    dst_dtype: np.dtype,
    normalize: bool,
) -> np.memmap:
    """
    Write GeoTIFF / BigTIFF file to NumPy array on disk (outpath).

    Returns:
        np.memmap: Reference to datacube object on disk
    """

    # Open file, read only ; automatically closes file
    with rasterio.open(src_path, mode="r") as src:

        # Image dimensions
        rows, cols, bands = src.height, src.width, src.count

        # Define output datacube (np.memmap) object
        dst_datacube = np.lib.format.open_memmap(
            dst_path, mode="w+", dtype=dst_dtype, shape=(rows, cols, bands)
        )

        # Output buffer
        temp_datacube = np.empty((bands, rows, cols), dtype=dst_dtype, order="C")

        # ----------------------------------------
        # Populate datacube
        # ----------------------------------------
        for band_idx in tqdm(
            range(bands), desc="band writing", unit="bands", colour="green"
        ):

            # Read in band; bands are indexed from 1
            band = src.read(band_idx + 1)

            # Write band to datacube
            temp_datacube[band_idx] = band[:]

    # ----------------------------------------
    # Normalize datacube
    # ----------------------------------------
    if normalize:
        # Magnitue of pixel vectors
        norms = np.linalg.norm(temp_datacube, axis=0, keepdims=True)

        # Identify where norms are zero to avoid division by zero
        nonzero_mask = norms > 1e-12

        # Perform division only on non-zero elements
        np.divide(temp_datacube, norms, out=temp_datacube, where=nonzero_mask)

    # Move bands to last axis
    temp_datacube = np.moveaxis(temp_datacube, 0, -1)

    # Move data to memmap
    dst_datacube[:] = temp_datacube[:]
    dst_datacube.flush()

    # Return opened datacube object with read/write permissions
    return np.lib.format.open_memmap(
        dst_path, mode="r", dtype=dst_dtype, shape=(rows, cols, bands)
    )


def _h5_to_npy(
    src_path: str,
    dst_path: str,
    dst_dtype: np.dtype,
    normalize: bool,
) -> np.memmap:
    """
    Write h5 files (from HYPERDOC) to NumPy array on disk (outpath).

    Returns:
        np.memmap: Reference to datacube object on disk
    """

    # Open file, read only
    with h5py.File(src_path, "r") as src:

        # h5py dataset object ; BSQ interleave
        image = src["DataCube"]

        bands, rows, cols = image.shape

        # Define output datacube (np.memmap) object
        dst_datacube = np.lib.format.open_memmap(
            dst_path, mode="w+", dtype=dst_dtype, shape=(rows, cols, bands)
        )

        # Output buffer
        temp_datacube = np.empty((bands, rows, cols), dtype=dst_dtype, order="C")

        # ----------------------------------------
        # Populate datacube - First pass
        # ----------------------------------------
        for band_idx in tqdm(
            range(bands), desc="band writing", unit="bands", colour="green"
        ):

            # Read in band
            band = image[band_idx, :, :]

            # Write band to datacube
            temp_datacube[band_idx] = band[:]

    # ----------------------------------------
    # Normalize datacube - Second pass
    # ----------------------------------------
    if normalize:
        # Magnitue of pixel vectors
        norms = np.linalg.norm(temp_datacube, axis=0, keepdims=True)

        # Identify where norms are zero to avoid division by zero
        nonzero_mask = norms > 1e-12

        # Perform division only on non-zero elements
        np.divide(temp_datacube, norms, out=temp_datacube, where=nonzero_mask)

    # Move bands to last axis
    temp_datacube = np.moveaxis(temp_datacube, 0, -1)

    # Move data to memmap
    dst_datacube[:] = temp_datacube[:]
    dst_datacube.flush()

    # Return opened datacube object with read/write permissions
    return np.lib.format.open_memmap(
        dst_path, mode="r", dtype=dst_dtype, shape=(rows, cols, bands)
    )


# ------------------------------------------------------------
# Loader
# ------------------------------------------------------------
def load_datacube(
    source_path: str,
    output_path: str | None = None,
    dtype: np.dtype = np.float32,
    normalize: bool = False,
    cachemax_mb: int = 1024,
) -> np.memmap:
    """
    Load a datacube from a single file or a directory of single-band TIFFs.

    Args:
        source_path:
            Path to a file (TIFF, h5, npy) OR a directory containing single-band TIFFs.
        output_path:
            Optional path for the on-disk memmap (.npy will be used if omitted).\n
            If provided and exists, it will be overwritten.
        dtype:
            NumPy dtype object to cast the memmap to (e.g. np.float32, np.float64)
        normalize:
            True: per-band min-max scaling to [0,1] (default)\n
            False: no normalization
        cachemax_mb (int, optional):
            GDAL image loading cache in MB. Reduce if program exceeds available RAM.

    Returns:
        A numpy.memmap object with shape (rows, cols, bands) opened in mode 'r' (read).

    Raises:
        ValueError: mismatched shapes or unsupported inputs.
    """
    # ------------------------------------------------------------
    # Path checks
    # ------------------------------------------------------------

    src_path = Path(source_path)
    dst_path = Path(output_path) if output_path is not None else Path("")

    # Check src path existence
    if not src_path.exists():
        raise FileNotFoundError(f"[dataloader] source directory {src_path} does not exist")

    # Check directory exists, create if not
    if dst_path.is_dir():
        os.makedirs(dst_path, exist_ok=True)

    # # Remove existing file at destination
    # elif not dst_path.is_dir() and dst_path.exists():
    #     print("[dataloader] warning: existing file at destination removed")
    #     os.remove(dst_path)

    with Env(GDAL_CACHEMAX=cachemax_mb, NUM_THREADS="ALL_CPUS"):

        # Decide how to read the single file based on suffix
        file_extension = src_path.suffix

        # ============================================================
        # Datacube: load pre-computed datacube
        # ============================================================

        # NumPy array datacube
        if file_extension == ".npy":
            return np.lib.format.open_memmap(
                source_path,
                mode="r",
                dtype=dtype,
            )

        # ============================================================
        # Directory: load all image files from directory
        # ============================================================

        if src_path.is_dir():
            return _dir_to_npy(
                source_path,
                output_path,
                dst_dtype=dtype,
                normalize=normalize,
            )

        # ============================================================
        # File: load a single image file
        # ============================================================

        # TIFF / GeoTIFF / BigTIFF
        if file_extension in (".tif", ".tiff", ".TIF", ".TIFF"):
            return _tiff_to_npy(
                source_path,
                output_path,
                dst_dtype=dtype,
                normalize=normalize,
            )

        # Hierarchical Data Format
        if file_extension in (".h5", "hdf5", ".he5", ".hdf"):
            return _h5_to_npy(
                source_path,
                output_path,
                dst_dtype=dtype,
                normalize=normalize,
            )

        raise ValueError("[dataloader] Invalid source file format")


# ---------------------------
# Save
# ---------------------------
def save_score_map(
    score_map: np.ndarray,
    score_out_path: str | Path,
) -> None:
    """
    Saves score map to image file format `ext` (extension). If `score_map` is populated with multiple images,
    several output images will be created with same basename e.g. `map-0.tif`, `map-1.tif` etc.

    Score maps are saved out as an unsigned 16-bit (uint16) TIFF.

    Args:
        score_map (np.ndarray):
            Output from algorithm; shape `(R, C) or `(R, C, M)`, where M is number
            of target members.
        score_out_path (str or Path):
            path + filename. Filetype converted to `.tiff` automatically.
    """

    # Score map normalization methods
    def _minmax_normalize(x: np.ndarray):
        """Simple min/max normalize converts data range to [0,1]"""
        return (x - x.min()) / (x.max() - x.min())

    def _percentile_normalize(x: np.ndarray, pmin: int = 1, pmax: int = 99):
        """Clips top/bottom 1% of data to avoid outlier norm skew"""
        lo, hi = np.percentile(x, (pmin, pmax))
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo)

    # Remove excess dimensions
    score_map = np.squeeze(score_map)

    # Normalize
    score_map = _percentile_normalize(score_map)  # [0,1]
    score_map = (score_map * 65_535).astype(np.uint16)  # [0, 65_535]

    # Save behavior determines multi-save vs single-save
    shape = score_map.shape

    # Filename attribute
    score_out_path = Path(score_out_path)
    stem = score_out_path.stem

    if not score_map.ndim in (2, 3):
        raise ValueError("[save] score_map must be 2D or 3D array")

    # Single image imwrite
    if score_map.ndim == 2:
        out_path = score_out_path.with_suffix(".tiff")
        tifffile.imwrite(out_path, score_map)
        return

    # Multi-image imwrite
    for idx in range(shape[-1]):
        # e.g. osp_map-0.tiff, osp_map-1.tiff, ...
        out_path = score_out_path.with_stem(stem + f"-{idx}").with_suffix(".tiff")
        tifffile.imwrite(out_path, score_map[:, :, idx])

    return


# ---------------------------
# Display
# ---------------------------
def display_score_map(score_maps: np.ndarray, plot_title: str = "Score Map") -> None:
    """
    Displays algorithm output as Matplotlib figure.

    Args:
        score_map (np.ndarray):
            Algorithm output. May be one score map array (rows, cols), a multi-score array (rows, cols, n_maps)
    """

    # Check if input is 2D (single image) or 3D (multiple images)
    if score_maps.ndim == 2:
        # Single image case
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(score_maps, cmap="gray", vmin=0, vmax=1)
        ax.set_title(plot_title)
        ax.axis("off")  # Turn off axis for cleaner display
        plt.tight_layout()
        plt.show()

    elif score_maps.ndim == 3:

        # Multiple images
        num_images = score_maps.shape[2]  # Get number of images from third dimension

        # Calculate the number of rows and columns for square subplot grid
        # We want the smallest square grid that can contain all images
        num_subplots = num_images
        num_cols = ceil(sqrt(num_subplots))  # Number of columns
        num_rows = ceil(num_subplots / num_cols)  # Number of rows

        # Create figure with subplots
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows)
        )

        # If there's only one row or column, axes might not be an array
        # Make sure axes is always an array for consistent indexing
        if num_rows == 1 and num_cols == 1:
            axes = [axes]
        elif num_rows == 1 or num_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Display each image
        for i in range(num_images):
            # Display image on corresponding subplot
            axes[i].imshow(score_maps[:, :, i], cmap="gray", vmin=0, vmax=1)
            axes[i].set_title(f"{plot_title}: target {i}")
            axes[i].axis("off")  # Turn off axis for cleaner display

        # Turn off any remaining empty subplots
        for i in range(num_images, len(axes)):
            axes[i].axis("off")
            axes[i].set_title("")  # Clear title if any

        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Input array must be 2D or 3D")


# ---------------------------
# Arg parse helper
# ---------------------------
def kwarg_match(func, kwargs) -> dict:
    """
    Return a dict of keyword arguments (kwargs) matching a function (`func`)'s signature, with defaults applied.
    Only applies the kwargs applicable to `func`, and ignores the remaining.

    ### Example
    >>> def area(length, width): return length * width
    >>> volume_kwargs = {
        "height": 2, # unused for area
        "length:  3,
        "width":  5,
        }
    >>> only_area_kwargs = kwarg_unpack(area, volume_kwargs)
    >>> area(**only_area_kwargs)
    >>> 15

    Args:
        func (function_like):
            The function to unpack the arguments to. Returned dict
            can be direectly unpacked to this function.
        kwargs (dict):
            Dictionary of keyword arguments (kwargs) to unpack.

    Returns:
        dict: Keyword arguments matching func's signature.

    """
    # Get function signature / arguments
    sig = inspect.signature(func)

    # Extract kwargs relevant to func
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}

    # Map args and kwargs to the signature
    bound = sig.bind_partial(**filtered)

    # Apply default values, if func has any
    bound.apply_defaults()

    # Return dict of kwargs for function unpacking
    return bound.arguments


# Then in get_targets:
# gui_kwargs = filter_signature_kwargs(target_selection_gui, kwargs)
# coordinates = target_selection_gui(datacube, **gui_kwargs)
