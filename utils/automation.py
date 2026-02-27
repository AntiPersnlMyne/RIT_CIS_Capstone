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
import logging
from typing import Literal
from tqdm import tqdm

from utils.target_selection import (
    extract_spectra,
    save_spectral_lib,
    load_spectral_lib,
    target_selection_gui,
)

from utils.eda import (
    # calculate_band_statistics,
    # save_band_statistics,
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

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def _extract_and_save(spectral_lib_path, coords, datacube):
    spectra = extract_spectra(coords, datacube)
    save_spectral_lib(spectral_lib_path, spectra, coordinates=coords)
    return (*coords, *spectra)


def _print_missing(which: str):
    """Logs (which) missing coordinates and informs GUI will be used to select them"""
    logger.info(f"Missing {which}. Loading GUI to select {which} coordinates ...")


def _run_gui_for_missing(datacube, which, kwargs):
    """Runs the"""
    coords = target_selection_gui(datacube, **kwargs)
    if which == "target":
        return coords[0]
    elif which == "background":
        return coords[1]
    else:
        return coords


# ------------------------------------------------------------
# Tests run in paper
# ------------------------------------------------------------


class Detectors:
    def __init__(
        self,
        datacube: np.memmap,
        target_members: NDArray,
        background_members: NDArray,
        algorithm_out_dir: str | Path,
        chunk_size: int,
        opci_thresh: float,
        max_targets: int | None,
        n_components: int | None,
    ):
        self.datacube = datacube
        self.target_members = target_members
        self.background_members = background_members
        self.out_dir = Path(algorithm_out_dir)
        self.chunk_size = chunk_size
        self.n_components = n_components
        self.opci_thresh = opci_thresh
        self.max_targets = max_targets
        self.test_name = None
        self._prog_bar = tqdm(
            total=80,
            colour="#80d3e5",
            desc="Subtests",
            unit="score_map",
        )
        self._prog_bar.disable = True

    def _save(self, score_map: NDArray, basename: str) -> None:
        """
        Saves score_map to {out_dir}/{basename}.tiff
        """

        # Add filename to output path
        path = Path(self.out_dir, self.test_name, basename)

        # Save score map, log procedure success
        try:
            save_score_map(score_map, path)
        except Exception as e:
            logger.error(f"Failed to save score map to '{path}'\nReason: {e}")

    def _run_detector(
        self,
        name: str,
        func: callable,
        args: dict[str, any],
    ) -> None:
        """
        Generic wrapper to run an algorithm, catch errors, and save
        """

        try:
            score_map = func(**args)
            self._save(score_map, f"{name}")
        except Exception as e:
            logger.exception(f"Exception during {name}: {e}")
        finally:
            self._prog_bar.update(1)

    def set_prog_vis(self, is_visibile: bool) -> None:
        self._prog_bar.disable = not is_visibile

    def process_all(self) -> None:
        """
        Runs all detectors with default configuration
        """

        # Common args for each detector
        common_args = {
            "datacube": self.datacube,
            "chunk_size": self.chunk_size,
        }

        # Run ACE
        self._run_detector(
            "ace", ace, {**common_args, "target_members": self.target_members}
        )
        # Run SAM
        self._run_detector(
            "sam", sam, {**common_args, "target_members": self.target_members}
        )
        # Run OSP
        self._run_detector(
            "osp",
            osp,
            {
                **common_args,
                "target_members": self.target_members,
                "background_members": self.background_members,
            },
        )
        # Run GOSP
        self._run_detector(
            "gosp",
            gosp,
            {
                **common_args,
                "max_targets": self.max_targets,
                "opci_thresh": self.opci_thresh,
            },
        )
        # Run PCA
        self._run_detector(
            "pca",
            pca,
            {
                **common_args,
                "n_components": self.n_components,
            },
        )

    def processing_test(
        self,
        average_targets: bool,
        background_subset: Literal["individual", "cluster", "swap"],
        test_name: str,
        *,
        skip_pca: bool = False,
    ) -> None:
        """
        Runs tests for combinations of target averaging and background subspace selection.

        Args:
            average_targets (int):
                Whether to average target signatures.
            background_subset (str):
                Configuration for background subspace.
                - "individual": Tests each of indices [1, 2, 3, 4] separately.
                - "cluster": Tests prefixes [1:4], [1:20], [1:40] of background_members.
                - "swap": Swaps targets and background (uses first 4 background as targets).
            test_name (str):
                Name of the test being run, creates a directory with this name.
            skip_pca (bool):
                Since PCA outputs are identical, prevent repeated calculations. If True, skips PCA.
        """

        # Add test name to object
        self.test_name = test_name
        # Create output directory for that test
        Path(self.out_dir, self.test_name).mkdir(parents=True, exist_ok=True)

        # Create list of how many members to include in subsets
        n_members = []
        match background_subset:
            case "individual":
                n_members = [1, 2, 3, 4]
            case "cluster":
                n_members = [4, 20, 40]
            case "swap":
                n_members = [0]
            case _:
                raise ValueError(f"Invalid background_subset: {background_subset!r}")

        # Average targets
        target_members = (
            np.average(self.target_members, axis=0, keepdims=True)
            if average_targets
            else self.target_members
        )

        # Run for each configuration
        for n in n_members:

            # Determine member set, given test
            match background_subset:
                case "individual":
                    background_members = self.background_members[n]
                case "cluster":
                    background_members = self.background_members[:n]
                case "swap":
                    background_members = target_members
                    target_members = self.background_members[:4]

            # Parameters shared between detectors
            common_args = {
                "datacube": self.datacube,
                "chunk_size": self.chunk_size,
            }

            # ACE
            self._run_detector(
                "ace",
                ace,
                {**common_args, "target_members": target_members},
            )

            # SAM
            self._run_detector(
                "sam",
                sam,
                {**common_args, "target_members": target_members},
            )

            # OSP
            self._run_detector(
                "osp",
                osp,
                {
                    **common_args,
                    "target_members": target_members,
                    "background_members": background_members,
                },
            )

            # GOSP
            self._run_detector(
                "gosp",
                gosp,
                {
                    **common_args,
                    "max_targets": self.max_targets,
                    "opci_thresh": self.opci_thresh,
                },
            )

            # PCA
            if not skip_pca:
                self._run_detector(
                    "pca",
                    pca,
                    {
                        **common_args,
                        "n_components": self.n_components,
                    },
                )


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
        row_start = rows - int(row_start * rows)
        row_end = int(row_end * rows)
        # Crop
        datacube = datacube[row_start:row_end, :, :]
    # Crop rows by pixel count
    else:
        datacube = datacube[row_start:-row_end, :, :]

    # Crop cols by percent
    if col_start <= 1.0 and col_end <= 1.0:
        # Set bounds
        col_start = cols - int(col_start * cols)
        col_end = int(col_end * cols)
        # Crop
        datacube = datacube[:, col_start:col_end, :]
    # Crop cols by pixel count
    else:
        datacube = datacube[:, col_start:-col_end, :]

    # assert isinstance(datacube, np.memmap), "np.NDArray returned instead of np.memmap"
    return datacube, datacube_name


def get_spectral_lib(
    spectral_lib_path: str,
    datacube: np.memmap = ...,
    average_targets: bool = False,
    force_coordinates: bool = False,
    **kwargs,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Extracts spectral library if it already exists. Otherwise, runs target
    selection GUI, returns the spectra, and saves results for future use.

    A "spectral library" is a NumPy zip file (.npz). The contents are 4 NumPy
    arrays, the coordinates of targets and background points, and the spectra
    at those points. Returns target (t) coordinats, then spectral signatures,
    followed by the background (b). i.e.,

    >>> target_coords, background_coords, target_spectra, background_spectra = get_spectral_lib(...)

    Args:
        spectral_lib_dir (str):
            Filepath to spectral library (.npz) file. Returns spectra of this file
            if exists. If not, requires `datacube` param passed, and creates file.
        datacube (np.memmap, optional):
            3D datacube `np.memmap` object, shape (R,C,B).
        average_targets (bool, optional):
            If True, averages all targets together to one spectra. Does **NOT** average
            background points, **ONLY** targets. Defaults to False.
        force_coordinates (bool, optional):
            If True, forces using coordinates from spectral library with datacube
            to extract spectra, even if target_members and background_members exist.

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
    # Enforce Path and .npz suffix for downstream compatibility
    spectral_lib_path = Path(spectral_lib_path).with_suffix(".npz")

    # Check if spectral library exists.
    # If not, run GUI to extract spectra and save.
    exists = spectral_lib_path.exists()

    # No spectral library exists, run GUI to extract both
    if not exists:
        logger.info(
            f"Spectral library not found at '{spectral_lib_path}'. Running GUI for both targets and backgrounds."
        )
        coords = target_selection_gui(datacube, **kwargs)
        return _extract_and_save(spectral_lib_path, coords, datacube)

    targ_coords, targ_spec, back_coords, back_spec = load_spectral_lib(
        spectral_lib_path
    )

    # Force coordinates: check for missing coordinates, not spectra
    if force_coordinates:
        missing_targets = not targ_coords.any()
        missing_backgrounds = not back_coords.any()

        if missing_targets or missing_backgrounds:
            # Supplement missing targets
            if missing_targets:
                _print_missing("target coordinates")
                targ_coords = _run_gui_for_missing(datacube, "target", kwargs)
            # Supplement missing backgrounds
            if missing_backgrounds:
                _print_missing("background coordinates")
                back_coords = _run_gui_for_missing(datacube, "background", kwargs)
            coords = (targ_coords, back_coords)
            # Return spectra extracted from coordinates
            return _extract_and_save(spectral_lib_path, coords, datacube)

        # No missing coordinates ; extract and return
        else:
            coords = (targ_coords, back_coords)
            spectra = extract_spectra(coords, datacube)
            save_spectral_lib(spectral_lib_path, spectra, coordinates=coords)
            return (*coords, *spectra)

    # Not force_coordinates: check for missing spectra
    missing_t = not targ_spec.any()
    missing_b = not back_spec.any()
    if missing_t or missing_b:
        coords = (targ_coords, back_coords)
        if missing_t:
            _print_missing("target spectra")
            targ_coords = _run_gui_for_missing(datacube, "target", kwargs)
            coords = (targ_coords, back_coords)
        if missing_b:
            _print_missing("background spectra")
            back_coords = _run_gui_for_missing(datacube, "background", kwargs)
            coords = (targ_coords, back_coords)
        return _extract_and_save(spectral_lib_path, coords, datacube)

    # Optionally average targets
    if average_targets and targ_spec.any():
        targ_spec = np.average(targ_spec, axis=0, keepdims=True)

    return targ_coords, back_coords, targ_spec, back_spec


def eda(
    datacube: np.memmap,
    stats_out_dir: str,
    datacube_name: str,
    show_corr_plot: bool = False,
):
    """
    Calculates band correlation matrix.

    Args:
        datacube (np.memmap):
            3D datacube `np.memmap` object, shape (R,C,B).
        stats_out_dir (str):
            Output directory for statistics.
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
    # band_stats_dst_path = Path(stats_out_dir, f"stats_{datacube_name}.csv") # uncomment for csv

    # Output: dst_dir/corr_<datacubename>.png
    corr_save_path = Path(stats_out_dir, f"corr_{datacube_name}").with_suffix(".png")

    # Check if statistics have already been calculated
    if corr_save_path.exists():
        logger.info("Statistics exist, skipping")
        return

    # Convert to str for downstream compatability
    corr_save_path = str(corr_save_path)

    # ------------------------------------------------------------
    # Band statistics (.csv)
    # ------------------------------------------------------------
    # save_band_statistics(
    #     statistics=calculate_band_statistics(datacube),
    #     dst_path=band_stats_dst_path,
    # )

    # ------------------------------------------------------------
    # Band correlation plot (.png)
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
    opci_threshold: float,
    chunk_size: int = 500,
    n_components: int | None = None,
    max_targets: int | None = None,
):
    """
    Processes datacube on all 5 algorithms: OSP, GOSP, SAM, ACE, PCA.

    Args:
        datacube (np.memmap):
            3D datacube object, shape (R,C,B).
        spectra (tuple):
            Tuple of target and background spectra arrays, expected as `(t_spectra, b_spectra)`.
        datacube_name (str):
            Name of datacube (e.g., '177r-172v').
        algorithm_out_dir (str):
            Output directory for score maps.
        chunk_size (int, optional):
            Number of rows to process at once. Defaults to 500.
        ocpi_threshold (float):
            Correlation ("purity") threshold for GOSP.
        n_components (int):
            Number of PCs to return from PCA.
        max_targets (int):
            Max number of targets for GOSP algorithm.
    """
    # Ensure output directory exists
    Path(algorithm_out_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Parameter setup
    # ------------------------------

    # Unpack spectra
    target_members, background_members = spectra

    # If any member is invalid (i.e. array contains NaN), raise error
    assert not np.isnan(target_members).any(), (
        "Must pass valid 'target_members' to perform detector processing."
        f"\nReceived array: {target_members}"
    )

    # Append datacube name to algorithm out directory
    algorithm_out_dir = Path(algorithm_out_dir, datacube_name)

    # ------------------------------
    # Detectors Processing - Testsdatacube_name
    # ------------------------------

    # Define detectors object
    detectors = Detectors(
        datacube=datacube,
        target_members=target_members,
        background_members=background_members,
        algorithm_out_dir=algorithm_out_dir,
        chunk_size=chunk_size,
        opci_thresh=opci_threshold,
        max_targets=max_targets,
        n_components=n_components,
    )

    # Enable progress bar
    detectors.set_prog_vis(is_visibile=True)

    # Test 1
    detectors.processing_test(True, "individual", "Test1")

    # Test 2
    detectors.processing_test(False, "individual", "Test2")

    # Test 3
    detectors.processing_test(True, "cluster", "Test3")

    # Test 4
    detectors.processing_test(False, "cluster", "Test4")

    # Test 5
    detectors.processing_test(True, "swap", "Test5")

    # Test 6
    detectors.processing_test(False, "swap", "Test6")


def get_coordinates(
    spectral_lib_path: str, return_none: bool = False
) -> tuple[NDArray, NDArray] | None:
    """
    Extract coordinates from spectral library file.

    Args:
        spectral_lib_path (str): Path to spectral library file
        return_none (bool): If True, returns None when file doesn't exist

    Returns:
        tuple[NDArray, NDArray] | None: Target and background coordinates
    """
    spectral_lib_path = Path(spectral_lib_path).with_suffix(".npz")

    if not spectral_lib_path.exists():
        return None if return_none else (np.empty(0), np.empty(0))

    t_coords, _, b_coords, _ = load_spectral_lib(spectral_lib_path)
    return t_coords, b_coords
