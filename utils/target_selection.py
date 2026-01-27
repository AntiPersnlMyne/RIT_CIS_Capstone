"""
Filename: target_selection.py
Description: Functions for picking target and background spectra
    - target_selection_gui(): GUI to select target and background points
    - extract_spectra(): Extracts spectra at point of datacube
    - save_spectra(): Saves target spectra to disk
    - load_spectra(): Loads a target from disk
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from pathlib import Path
from logging import info

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "12-30-2025"
__email__ = "mt9485@rit.edu"


def target_selection_gui(
    rgb_images: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Displays a window, allowing user to click points on image to return pixel coordinates of desired targets.

    Controls
    --------
    Left click   : add point
    Right click  : undo last point **irrespective** of selection mode
    't'          : target selection mode
    'b'          : background selection mode
    'q' or ESC   : save/quit

    Args:
        images (list[ndarray]):
            Three, 2D grayscale images (ndarray). NOTE: The spectra will NOT be extracted from these images.
            Images must be passed in RGB format `[R,G,B]`.

    Returns:
        tuple[np.ndarray]:
            List of coordinates (row, col) for 1D arrays `targets` and `background` (in that order). shape=(n_coords, 2).
    """

    # ------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------

    if not rgb_images or not all(isinstance(img, np.ndarray) for img in rgb_images):
        raise TypeError("images must be a non-empty list of NumPy arrays")

    if len(rgb_images) != 3:
        raise ValueError("Exactly three 2D arrays [R, G, B] are required")

    if any(img.ndim != 2 for img in rgb_images):
        raise ValueError("Each image must be a 2D array")

    img_rows, img_cols = rgb_images[0].shape
    if not all(img.shape == (img_rows, img_cols) for img in rgb_images):
        raise ValueError("All channels must have identical shapes")

    # ------------------------------------------------------------
    # Compile display image
    # ------------------------------------------------------------

    # Stack images
    rgb_image = np.dstack(rgb_images)

    # ------------------------------------------------------------
    # Output storage
    # ------------------------------------------------------------

    targets_coords = []  # Tuples of targets coords
    backgrounds_coords = []  # Tuples of backgrounds coords
    history = []  # stack of (class_key, artist)
    mode = "targets"  # vs background

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        ncols=2,
        figsize=(30, 20),
        num="Coordinate Extraction Window",
        gridspec_kw={"width_ratios": [4, 1]},
    )
    ax[0].imshow(rgb_image)
    ax[0].set_title("Mode: TARGETS", fontsize=35)  # vs. BACKGROUND
    ax[0].axis("off")

    # Controls text
    controls_text = """
Steps
--------
1) Click on the image to create points
2) save/quit when finished adding points


Description
----------------
target (red) = area to visually enhance
background (blue) = background/clutter

(Tip) Distinct target and background points 
        produce better results


Controls
------------
Left click    : add point
Right click  : undo last point
t                 : target selection mode 
b                : background selection mode
q or ESC     : save/quit"
"""

    # Position the text box in the upper right corner
    ax[1].text(
        # Aligning text location
        -0.4,  
        0.95,
        # Textbox contents
        controls_text,
        transform=ax[1].transAxes,
        fontsize=28,
        # Text alignment
        verticalalignment="top",
        horizontalalignment="left",
        # Box parameters
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax[1].axis("off")
    ax[1].set_title("Controls", fontsize=35)
    
    # Initialize scatter plots 
    targets_scatter = ax[0].scatter([], [], c="red", s=40)
    backgrounds_scatter = ax[0].scatter([], [], c="blue", s=40)

    # ------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------

    def on_key(event):
        nonlocal mode

        # Quit
        if event.key in ("q", "escape"):
            plt.close(fig)

        # Target mode
        elif event.key == "t":
            mode = "targets"
            ax[0].set_title("Mode: TARGETS", fontsize=35)
            fig.canvas.draw_idle()

        # Background mode
        elif event.key == "b":
            mode = "background"
            ax[0].set_title("Mode: BACKGROUND", fontsize=35)
            fig.canvas.draw_idle()

    def on_click(event):

        # Check click is on figure
        if event.inaxes != ax[0]:
            return

        # Undo (right click)
        if event.button == MouseButton.RIGHT:
            if history:

                last = history.pop()

                if last == "targets":
                    targets_coords.pop()
                    targets_scatter.set_offsets(targets_coords)
                else:
                    backgrounds_coords.pop()
                    backgrounds_scatter.set_offsets(backgrounds_coords)

                fig.canvas.draw_idle()

            return

        # Add point (left click)
        if event.button == MouseButton.LEFT:
            row = int(event.ydata)
            col = int(event.xdata)

            if mode == "targets":
                targets_coords.append((col, row))
                targets_scatter.set_offsets(targets_coords)
                history.append("targets")
            else:
                backgrounds_coords.append((col, row))
                backgrounds_scatter.set_offsets(backgrounds_coords)
                history.append("background")

            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    # Return result once user exits GUI
    plt.close("all")

    # Convert lists to NDArrays
    # Swap (col, row) to (row, col)
    targets_temp = [(row, col) for col, row in targets_coords]
    backgrounds_temp = [(row, col) for col, row in targets_coords]
    return (np.array(targets_temp), np.array(backgrounds_temp))


def extract_spectra(
    coordinates: tuple[np.ndarray, np.ndarray],
    datacube: np.memmap,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts spectra of datacube at coordinate. Creates the target and background spectra from coordinates
    selected from `target_selection_gui()`.
    Note: A known crash will occur if user selects no points (IndexError).

    Args:
        coordinates (tuple[np.ndarray]):
            Arrays for coordinates `targets` and `backgrounds` (expected in that order). shape=(n_coords, 2).
        datacube (np.memmap):
            3D datacube of shape (rows, cols, bands).

    Returns:
        tuple[np.ndarray]: List of signatures (spectra) for `targets` and `background` (in that order).
        targets shape = (n_targets, bands). background shape = (n_background, bands)
    """
    # Output dict with arrays of target and background spectras
    targets_coords, backgrounds_coords = coordinates

    try:  # Extract rows and cols from targ coords
        t_rows = targets_coords[:, 0]
        t_cols = targets_coords[:, 1]
    except IndexError:  # No targ coords given
        t_rows = np.empty((0, 0))
        t_cols = np.empty((0, 0))

        info("Empty array for (targets) being saved")

    try:  # Extract rows and cols backgnd coords
        b_rows = backgrounds_coords[:, 0]
        b_cols = backgrounds_coords[:, 1]
    except IndexError:  # No backgnd coords given
        b_rows = np.empty((0, 0))
        b_cols = np.empty((0, 0))

    # Assert coordinat arrays are same length/shape
    assert np.size(t_rows) == np.size(
        t_cols
    ), f"[targ_select] Target array uneven: (rows,cols)=({t_rows, t_cols})"
    assert np.size(b_rows) == np.size(
        b_cols
    ), f"[targ_select] Background array uneven: (rows,cols)=({t_rows, t_cols})"

    try:  # Extract spectra at each coordinate
        targets_spectra = datacube[t_rows, t_cols, :]
    except IndexError:  # Return empty array if no coordinates
        print("No targets found. Returning empty array.")
        targets_spectra = np.empty((0, 0, 0))
    try:  # Extract spectra at each coordinate
        backgrounds_spectra = datacube[b_rows, b_cols, :]
    except IndexError:  # Return empty array if no coordinates
        print("No backgrounds found. Returning empty array.")
        backgrounds_spectra = np.empty((0, 0, 0))

    return (targets_spectra, backgrounds_spectra)


def save_spectra(
    dst_path: str,
    spectra: tuple[np.ndarray],
    filename: str | None = None,
    coordinates: tuple[np.ndarray, np.ndarray] | None = None,
):
    """
    Saves extracted spectra of targets alongside their coordinates. File is always saved as NumPy zip (`.npz`).

    Args:
        dst_path (str):
            Output file directory. NOTE: Do not append a file extension. Will always save as `.npz`
        spectra (tuple[np.ndarray]):
            List of spectra for `targets` and `background`. shape=(n_coords, 1).
        filename (str, optional):
            Name of output file. If None, uses default name "targ_backgnd.npz".
        coordinates (tuple[np.ndarray]):
            List of coordinates for `targets` and `background`. shape=(n_coords, 2).
    """

    # ------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------

    # Check empty paths and empty data
    assert dst_path is not None, "[save_spectra] Destination path cannot be empty"
    assert spectra[0].size != 0, "[save_spectra] Targets cannot be empty"

    # Define .npz suffix
    npz_suffix = Path(".npz")

    # Ensure output directory exists
    dst_path = Path(dst_path)

    # Append filename if given
    if filename is not None:
        filename = Path(filename)
        dst_path = dst_path / filename
    elif filename is None:
        # Directory only, append filename
        if dst_path.name == "/":
            filename = Path("targ_backgnd.npz")
            dst_path = dst_path / filename

    # Add suffix is file doesn't have suffix
    if not dst_path.suffix and dst_path.name:
        dst_path = dst_path.with_name(f"{dst_path.name}{npz_suffix}")

    # Override existing suffixes
    elif dst_path.suffix != ".npz":
        dst_path = dst_path.with_suffix(npz_suffix)

    # Warning for overwrite existing files
    if dst_path.exists():
        info(
            f"[save_spectra] warning: file at {dst_path} already exists, being erased."
        )

    # ------------------------------------------------------------
    # Define output variables
    # ------------------------------------------------------------

    # Create dummy variable to be saved
    if not coordinates:
        target_coords = np.empty(0)
        background_coords = np.empty(0)
    else:
        target_coords, background_coords = coordinates

    target_spectra, background_spectra = spectra

    # Save results to NumPy file
    # Converts Python lists to NumPy array
    print(dst_path)
    np.savez(
        # Output path
        str(dst_path),
        # Targets
        target_coords=target_coords,
        target_spectra=target_spectra,
        # Background
        background_coords=background_coords,
        background_spectra=background_spectra,
    )


def load_spectra(src_path: str) -> tuple[np.ndarray, ...]:
    """
    Loads targets from NumPy zip file (.npz).

    -------
    Example
    -------
    t_coords, t_specs, b_coords, b_specs = load_spectra("arch_172_177.npz")

    Args:
        npz_src_path (str):
            Path to target file

    Returns:
        tuple[np.ndarray,...]: 4 output arrays. If any of the saved arrays were empty, that array will be empty.
    """
    try:
        # Load dictionary-like file
        with np.load(src_path) as file:
            t_coord = file["target_coords"]
            t_specs = file["target_spectra"]
            b_coords = file["background_coords"]
            b_specs = file["background_spectra"]
    except IsADirectoryError:
        raise IsADirectoryError(
            "[load_spectra] Add filename and/or suffix (.npz) to `npz_src_path` to load file"
        )

    return t_coord, t_specs, b_coords, b_specs
