"""
Functions to build a detector pipeline: load, process, save.
The pipeline aims to be modular and customizable. The layout is as follows:

          [ Load datacube ]
                  |
                  |
      [ Is Archimedes dataset? ]
                  |
                  |
     [ Yes ] ------------ [ No ]
        |                   |
        |                   |
[ Crop datacube ]           |
        |                   |
        |___________________|
                  |
                  |
      [ Spectra pre-computed? ]
                  |
                  |
     [ Yes ] ----------- [ No ]
        |                   |
        |                   |
[ Load from file ]    [ Extraction GUI ]
        |                   |
        |                   |
        |           [ Average targets? ]
        |                   |
        |                   |
        |        [ Yes ]---------[ No ]
        |            |              |
        |            |              |
        |       [ average ]         |
        |       [ targets ]         |
        |            |              |
        |            |______________|
        |                   |
        |                   |
        |      [extract spectra from targets ]
        |                   |
        |                   |
        |             [ Save spectra ]
        |                   |
        |___________________|
                    |
                    |
                [ EDA ]
                    |
                    |
        [ Detector processing ]
                    |
                    |
            [ Save score maps ]
                    |
                    |
    [ Does BGP datacube for data exist? ]
                    |
                    |
        [Yes] -------------- [ No ]
          |                     |
          |                     |
 [ Load BGP datacube ]   [ Exit program! ]
          |
          |__________
                     |
                     |
   [ Extract spectra @ previous coordinates ]
                     |
                     |
                  [ EDA ]
                     |
                     |
           [ Detector processing ]
                     |
                     |
             [ Save score maps ]
"""

import numpy as np
from pathlib import Path

from utils.target_selection import (
    extract_spectra,
    save_spectra,
    load_spectra,
    target_selection_gui,
)

from utils.eda import (
    calculate_band_statistics,
    save_band_statistics,
    cov_matrix,
    corr_matrix,
    plot_corr_matrix,
)

from algorithms import (
    gosp,
    osp,
    sam,
    ace,
    pca,
)

from utils.dataloader import (
    save_score_map,
)

NDArray = np.ndarray


def import_and_crop_datacube(
    datacube_path: str,
    datacube_dtype: np.dtype = np.float64,
    row_bounds: tuple | list = (),
    col_bounds: tuple | list = (),
) -> tuple[np.memmap, str]:
    """
    Imports datacube (numpy.memmap). Optonally crops datacube to boundaries
    by percent - (0.0, 1.0] - or by pixel count.

    Cropping by percent starts on opposite ends of the array. E.g.,
    >>> array = np.array( [0,1,2,3,4,5,6,7,8,9] )
    >>> len = np.size(array)
    >>> row_bounds = (0.25, 0.25) # 25% cropped off each end
    >>> r_start, r_end = len * row_bounds[0], len * row_bounds[1]
    >>> array[r_start : -r_end]
    >>> [2 3 4 5 6 7]

    Args:
        datacube_path (str):
            Path to datacube object, including filename. Opens at `numpy.memmap`.
        datacube_dtype (np.dtype, optional):
            Datatype of the datacube object as `numpy.dtype`. Defaults to np.float64.
        row_bounds (tuple | list, optional):
            Clipping boundaries by percentage (`(0,1]`) or by pixel count.
            Left boundary counts from 0, right boundary counts backwards from total rows.
            Coordinates read in as `(row_begin, row_end)`. Defaults to ().
        col_bounds (tuple | list, optional):
            Clipping boundaries by percentage (`(0,1]`) or by pixel count.
            Left pixel boundary counts from 0, right boundary counts backwards from total columns.
            Coordinates read in as `(row_begin, row_end)`. Defaults to ().
    Returns:
        np.memmap: Tuple object. First, opened datacube object in "r" mode. Second, datacube name.
    """
    # ==============================
    # Load
    # ==============================
    # Filenames for output identifiers
    datacube_name = Path(datacube_path).stem

    # Load datacube
    datacube = np.lib.format.open_memmap(
        datacube_path,
        mode="r",
        dtype=datacube_dtype,
    )

    # Get shape
    rows, cols, _ = datacube.shape

    # ==============================
    # Crop
    # ==============================

    # Get crop values
    row_start, row_end = row_bounds
    col_start, col_end = col_bounds

    # Crop rows by percent
    if row_start <= 1.0 and row_end <= 1.0:
        # Set bounds
        row_start = int(row_start * rows)
        row_end = int(row_end * rows)
        # Crop
        datacube = datacube[row_start:-row_end, :, :]
    # Crop rows by pixel count
    else:
        datacube = datacube[row_start:-row_end, :, :]

    # Crop cols by percent
    if col_start <= 1.0 and col_end <= 1.0:
        # Set bounds
        col_start = int(col_start * cols)
        col_end = int(col_end * cols)
        # Crop
        datacube = datacube[row_start:-row_end, :, :]
    # Crop cols by pixel count
    else:
        datacube = datacube[row_start:-row_end, :, :]

    assert issubclass(datacube, np.memmap), "np.NDArray returned instead of np.memmap"
    return datacube, datacube_name


def spectra_selection_pipeline(
    *,
    spectral_lib_path: str,
    datacube: np.memmap = None,
    average_targets: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Extracts spectral library if it already exists. Otherwise, runs target
    selection GUI, returns the spectra, and saves results for future use.

    A "spectral library" is a NumPy zip file (.npz). The contents are 4 NumPy
    arrays, the coordinates of targets and background points, and the spectra
    at those points. Returns target (t) coordinats, then spectral signatures,
    followed by the background (b). i.e.,

    >>> t_coords, t_spectra, b_coords, b_spectra = spectra_selection_pipeline(...)

    Args:
        spectral_lib_dir (str):
            Filepath to spectral library (.npz) file. Returns spectra of this file
            if exists. If not, creates file and populates after selection GUI.
        datacube (np.memmap):
            3D datacube `np.memmap` object, shape (R,C,B).
        average_targets (bool, optional):
            If True, averages all targets together to one spectra. Does **NOT** average
            background points, **ONLY** targets. Defaults to True.
            
    Returns:
        tuple[NDArray, NDArray, NDArray, NDArray]: Spectra coordinate and signature arrays.
    """

    # ------------------------------------------------------------
    # Load existing spectral library
    # ------------------------------------------------------------
    spectral_lib_path = Path(spectral_lib_path)

    if spectral_lib_path.exists() and spectral_lib_path.endswith(".npz"):
        return load_spectra(spectral_lib_path)

    # ------------------------------------------------------------
    # Load GUI
    # ------------------------------------------------------------
    coordinates = target_selection_gui(datacube)
    t_coords, b_coords = coordinates
    
    # ------------------------------------------------------------
    # Extract spectra
    # ------------------------------------------------------------
        # Extract spectral signatures of datacube
    spectra = extract_spectra(coordinates=coordinates, datacube=datacube)
    target_members, background_members = spectra
    
    if average_targets:
        # shape (M,B) -> (1, B)
        target_members = np.average(target_members, axis=0, keepdims=True)

    # ------------------------------------------------------------
    # Save spectra and coordinates for reproducibility
    # ------------------------------------------------------------
    save_spectra(
        dst_path=spectral_lib_path,
        spectra=spectra,
        coordinates=coordinates,
    )
    
    # ------------------------------------------------------------
    # Return 
    # ------------------------------------------------------------
    return t_coords, target_members, b_coords, background_members



if __name__ == "__main__":
    array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    len = np.size(array)
    row_bounds = (0.25, 0.25)  # 25% cropped off each end
    r_start, r_end = int(len * row_bounds[0]), int(len * row_bounds[1])
    print(array[r_start:-r_end])
