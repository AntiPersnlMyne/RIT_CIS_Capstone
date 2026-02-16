"""
Functions to build a detector pipeline: load, process, save.
The pipeline aims to be modular and customizable. The layout is as follows:

          [ Load datacube ]
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

from utils.dataloader import (
    save_score_map,
    load_datacube,
    kwarg_match,
)

from algorithms import (
    gosp,
    osp,
    sam,
    ace,
    pca,
)


NDArray = np.ndarray


def import_datacube(
    source_path: str,
    datacube_out_dir: str | None = None,
    datacube_dtype: np.dtype = np.float64,
    row_bounds: tuple | list = (1.0, 1.0),
    col_bounds: tuple | list = (1.0, 1.0),
    normalize: bool = True,
    cachemax: int = 10_240,
) -> tuple[np.memmap, str]:
    """
    Imports datacube (numpy.memmap).

    Optonally crops datacube to boundaries
    by percent - (0.0, 1.0] - or by pixel count.

    Cropping by percent starts on opposite ends of the array. E.g.,
    >>> array = np.array( [0,1,2,3,4,5,6,7,8,9] )
    >>> len = np.size(array)
    >>> row_bounds = (0.25, 0.25) # 25% cropped off each end
    >>> r_start, r_end = len * row_bounds[0], len * row_bounds[1]
    >>> array[r_start : -r_end]
    >>> [2 3 4 5 6 7]

    Args:
        source_path (str):
            Path to A) datacube object including filename, or B) directory of TIFF files.
            Opens as `numpy.memmap`.
        datacube_out_dir (str or None, optional):
            Saves resulting datacube object to directory. If `source_path` leads to a datacube
            object, this parameter is ignored.
        datacube_dtype (np.dtype, optional):
            Datatype of the datacube object as `numpy.dtype`. Defaults to np.float64.
        row_bounds (tuple | list, optional):
            Clipping boundaries by percentage (`(0,1]`) or by pixel count.
            Left boundary counts from 0, right boundary counts backwards from total rows.
            Coordinates read in as `(row_begin, row_end)`. Defaults to (1.0,1.0) -> no crop.
        col_bounds (tuple | list, optional):
            Clipping boundaries by percentage (`(0,1]`) or by pixel count.
            Left pixel boundary counts from 0, right boundary counts backwards from total columns.
            Coordinates read in as `(row_begin, row_end)`. Defaults to (1.0,1.0) -> no crop.
        normalize (bool, optional):
            If importing TIFF files rather than datacube object, flag to unit-vector normalize each
            data point (pixel) when loading and convering to datacube object. Default is True.
        cachemax (int, optional):
            Preallicated memory in megabytes (mb) for proram to load files. Default is 10240 (10 GB).
    Returns:
        np.memmap: Tuple object. First, opened datacube object in "r" mode. Second, datacube name.
    """
    # ==============================
    # Load
    # ==============================
    # Filenames for output identifiers
    datacube_name = Path(source_path).stem

    # Load datacube
    datacube = load_datacube(
        source_path=source_path,
        output_path=datacube_out_dir,
        dtype=datacube_dtype,
        normalize=normalize,
        cachemax_mb=cachemax,
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
        datacube = datacube[:, col_start:-col_end, :]
    # Crop cols by pixel count
    else:
        datacube = datacube[:, col_start:-col_end, :]

    assert isinstance(datacube, np.memmap), "np.NDArray returned instead of np.memmap"
    return datacube, datacube_name


def get_spectral_lib(
    spectral_lib_path: str,
    datacube: np.memmap = None,
    average_targets: bool = False,
    *,
    coordinates: tuple[NDArray, NDArray] | None = None,
    **kwargs,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Extracts spectral library if it already exists. Otherwise, runs target
    selection GUI, returns the spectra, and saves results for future use.

    A "spectral library" is a NumPy zip file (.npz). The contents are 4 NumPy
    arrays, the coordinates of targets and background points, and the spectra
    at those points. Returns target (t) coordinats, then spectral signatures,
    followed by the background (b). i.e.,

    >>> target_coords, target_spectra, background_coords, background_spectra = get_spectral_lib(...)

    Args:
        spectral_lib_dir (str):
            Filepath to spectral library (.npz) file. Returns spectra of this file
            if exists. If not, requires `datacube` param passed, and creates file.
        datacube (np.memmap, optional):
            3D datacube `np.memmap` object, shape (R,C,B).
        average_targets (bool, optional):
            If True, averages all targets together to one spectra. Does **NOT** average
            background points, **ONLY** targets. Defaults to True.
        coordinates (tuple, optional):
            If provided, uses pre-selected coordinates instead of opening GUI program.
            Useful if processing a new datacube (i.e., bgp datacube) with pre-selected
            coordinates.

    ## Kwargs:
        band_labels (list[str]): (disfunctional) Labels for RGB slider tickmarks
        max_points (int): Maximum points able to be plotted on GUI.
        header_font_size (int): Title text font size
        controls_font_size (int): Dialogue box font size
        label_size (int): Slider label text
        display_scale (int): Ratio of display scale e.g. 8 -> displayed at 1/8 resolution.

    Returns:
        tuple[NDArray, NDArray, NDArray, NDArray]: Spectra signature and coordinate arrays.
    """
    # ------------------------------------------------------------
    # Load existing spectral library
    # ------------------------------------------------------------
    spectral_lib_path = Path(spectral_lib_path)

    if spectral_lib_path.exists():
        return load_spectra(spectral_lib_path)

    # ------------------------------------------------------------
    # Load GUI
    # ------------------------------------------------------------
    # Load kwargs
    gui_kwargs = kwarg_match(target_selection_gui, kwargs)

    # Use existing coordinates
    if coordinates:
        # If just background coords were given, call GUI for target coords
        t_coords, b_coords = coordinates
        if not t_coords.any():
            coordinates = target_selection_gui(datacube, **gui_kwargs)
            t_coords = coordinates[0]

        # Zip coordinates back into variable
        coordinates = (t_coords, b_coords)

    # Extract new target and background coordinates
    else:
        coordinates = target_selection_gui(datacube, **gui_kwargs)

    # ------------------------------------------------------------
    # Extract spectra
    # ------------------------------------------------------------
    # Extract spectral signatures of datacube
    spectra = extract_spectra(coordinates=coordinates, datacube=datacube)
    target_members, background_members = spectra

    # Average targets. If array is empty, prevent DIV0 error
    if average_targets and target_members.any():
        # shape (M,B) -> (1, B)
        target_members = np.average(target_members, axis=0, keepdims=True)

    # Shape spectra back into variable
    spectra = (target_members, background_members)

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
    target_coords, background_coords = coordinates
    return target_coords, target_members, background_coords, background_members


def eda(
    datacube: np.memmap,
    stats_out_dir: str,
    datacube_name: str,
    show_corr_plot: bool = False,
):
    """
    Calculates the band-statistics (notably, kurtosis). Saves as CSV.

    Additionally calculates band correlation matrix.

    Args:
        datacube (np.memmap):
            3D datacube `np.memmap` object, shape (R,C,B).
        stats_out_dir (str):
            Output directory for all statistics: band stats and covariance matrix
        show_corr_plot (bool, optional):
            If provided, displays correlation plot with the title.
            If None, only saves plot to `dst_path`. Defaults to False.
    """
    # ------------------------------------------------------------
    # Output dirctories
    # ------------------------------------------------------------
    # Direectory/folder for current datacube stats
    Path(stats_out_dir).mkdir(exist_ok=True, parents=True)

    # Output: stats_out_dir/stats_<datacubename>.csv
    band_stats_dst_path = Path(stats_out_dir, f"stats_{datacube_name}.csv")

    # Output: dst_dir/corr_<datacubename>.png
    corr_save_path = Path(stats_out_dir, f"corr_{datacube_name}").with_suffix(".png")

    # Check if statistics have already been calculated
    if band_stats_dst_path.exists() and corr_save_path.exists():
        print("Statistics exist, skipping")
        return

    # Convert to str for downstream compatability
    corr_save_path = str(corr_save_path)

    # ------------------------------------------------------------
    # Band statistics (.csv)
    # ------------------------------------------------------------
    save_band_statistics(
        statistics=calculate_band_statistics(datacube),
        dst_path=band_stats_dst_path,
    )

    # ------------------------------------------------------------
    # Band correlation plot
    # ------------------------------------------------------------

    # Compute correlation matrix
    corr_mat = corr_matrix(cov_matrix=cov_matrix(datacube))

    # Plot/Save correlation matrix
    plot_corr_matrix(corr_mat, save_dir=corr_save_path, show_plot=show_corr_plot)


def detector_processing(
    datacube: np.memmap,
    spectra: tuple[NDArray, NDArray],
    datacube_name: str,
    algorithm_out_dir: str,
    chunk_size: int = 500,
    **kwargs,
):
    """
    Processes datacube on all 5 algorithms: OSP, GOSP, SAM, ACE, PCA.

    Args:
        datacube (np.memmap):
            3D datacube object, shape (R,C,B).
        spectra (tuple):
            Tuple of target and background spectra arrays, expected as `(t_spectra, b_spectra)`.
        datacube_name (str):
            Name of datacube (e.g., 177r-172v).
        algorithm_out_dir (str):
            Output directory for score maps.
        chunk_size (int, optional):
            Number of rows to process at once. Defaults to 500.

    ## Keyword Args (**kwargs):
        detect_filename (str):
            Manually override name of saved score map. Change detector prefix for chosen detector.
            Format as: `"osp_filename": "my_scoremap_name"`.
        n_components (int):
            Number of PCs to return from PCA.
        max_targets (int):
            Max number of targets for GOSP algorithm.
        ocpi_threshold (float):
            Correlation ("purity") threshold for GOSP.

    """
    # Ensure output directory exists
    Path(algorithm_out_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Parameter setup
    # ------------------------------

    # Unpack spectra
    target_members, background_members = spectra

    # If empty (i.e. array of NaN), raise error
    # This catches edge case where solely background coordinates are passed during
    # automation script
    assert not np.isnan(target_members).any(), (
        "Must pass valid 'target_members' to perform detector processing."
        f" This error may be intentional if passing array of only background members\nReceived array: {target_members}"
    )

    # Append datacube name to algorithm out directory
    algorithm_out_dir = Path(algorithm_out_dir, datacube_name)

    # Unpack kwargs, use default if not provided
    max_targets = kwargs.pop("max_targets", None)
    n_components = kwargs.pop("n_components", None)
    opci_thresh = kwargs.pop("opci_thresh", 0.7)
    osp_path = kwargs.pop("osp_filename", f"{algorithm_out_dir}_osp.tiff")
    gosp_path = kwargs.pop("gosp_filename", f"{algorithm_out_dir}_gosp.tiff")
    sam_path = kwargs.pop("sam_filename", f"{algorithm_out_dir}_sam.tiff")
    ace_path = kwargs.pop("ace_filename", f"{algorithm_out_dir}_ace.tiff")
    pca_path = kwargs.pop("pca_filename", f"{algorithm_out_dir}_pca.tiff")

    # ------------------------------
    # Detectors
    # ------------------------------

    # ACE
    score_map = ace(datacube, target_members, chunk_size=chunk_size)
    save_score_map(score_map, ace_path)

    # SAM
    score_map = sam(datacube, target_members, chunk_size=chunk_size)
    save_score_map(score_map, sam_path)

    # OSP
    score_map = osp(datacube, target_members, background_members, chunk_size=chunk_size)
    save_score_map(score_map, osp_path)

    # GOSP
    score_map = gosp(
        datacube=datacube,
        chunk_size=chunk_size,
        max_targets=max_targets,
        opci_thresh=opci_thresh,
    )
    save_score_map(score_map, gosp_path)

    # PCA
    score_map = pca(datacube, chunk_size=chunk_size, n_components=n_components)
    save_score_map(score_map, pca_path)


def get_coordinates(
    coordinate_lib_path: str,
    *,
    return_none: bool = False,
) -> tuple[NDArray, NDArray] | None:
    """
    Extracts the coordinates from existing spectral library, if it exists.
    Otherwise, returns None, indicating no existing coordinates found.

    Args:
        coordinate_lib_path (str):
            If this path contains "background" in the filename, assumed to be the background coordinates.
            If this path contains "bgp", assumed to be coordinates for BGP datacube.
        return_none (bool, optional):
            If True, returns None if no coordinates found. Otherwise, raises FileNotFoundError.

    Returns
    -------
        tuple[NDArray, NDArray] | None: Returns `(target_coords, background_coords)`

    Raises
    ------
        FileNotFoundError: No coordinates returned from library.
    """

    # Return None if path not found
    if not Path(coordinate_lib_path).exists():
        raise FileNotFoundError(
            f"No coordinate library found at:\n{coordinate_lib_path}"
        )

    # Extract coordinates from library
    target_coords, _, background_coords, _ = get_spectral_lib(
        spectral_lib_path=str(coordinate_lib_path)
    )

    # Check if no coordinates found
    if not target_coords.any() or background_coords.any():
        if return_none:
            return None
        else:
            raise FileNotFoundError(
                f"No coordinates saved to spectral library:\n{coordinate_lib_path}"
            )

    # Return coordinates from library
    return (target_coords, background_coords)
