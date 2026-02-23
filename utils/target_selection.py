"""
Filename: target_selection.py
Description: Functions for picking target and background spectra
    - target_selection_gui(): GUI to select target and background points
    - extract_spectra(): Extracts spectra at point of datacube
    - save_spectra(): Saves target spectra to disk
    - load_spectra(): Loads a target from disk
"""

import numpy as np
from pathlib import Path
from logging import info
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Slider
from screeninfo import get_monitors
from dataclasses import dataclass, field

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "02-06-2025"
__email__ = "mt9485@rit.edu"


# ----------------------------
# Constants
# ----------------------------

MODE_TARGET = "targets"
MODE_BACKGROUND = "background"

# ----------------------------
# Helper functions
# ----------------------------


def _clamp_point(row, col, rows, cols):
    row = np.clip(row, 0, rows - 1)
    col = np.clip(col, 0, cols - 1)
    return row, col


def _coords_to_fullres_scale(event, scale):
    row = int(round(event.ydata)) * scale
    col = int(round(event.xdata)) * scale
    return row, col


def _compute_rgb_indices(n_bands: int):
    return (
        int(n_bands * 0.75),
        int(n_bands * 0.5),
        int(n_bands * 0.25),
    )


def _to_display_coords(coords: np.ndarray, count: int, scale: int):
    """Convert full-res (row, col) -> display (x, y)."""
    if count == 0:
        return np.empty((0, 2))
    return coords[:count][:, ::-1] / scale


def _fill_rgb_buffer(buffer, datacube, scale, r, g, b):
    buffer[..., 0] = (datacube[::scale, ::scale, r] * 255).clip(0, 255).astype(np.uint8)
    buffer[..., 1] = (datacube[::scale, ::scale, g] * 255).clip(0, 255).astype(np.uint8)
    buffer[..., 2] = (datacube[::scale, ::scale, b] * 255).clip(0, 255).astype(np.uint8)


@dataclass
class GUIState:
    mode: str = MODE_TARGET
    t_count: int = 0
    b_count: int = 0
    history: list[str] = field(default_factory=list)
    background = None


# ----------------------------
# Public functions
# ----------------------------


def target_selection_gui(
    datacube: np.memmap,
    *,
    display_scale: int = 4,
    max_points: int = 100,
    header_font_size: int = 35,
    controls_font_size: int = 28,
    label_font_size: int = 25,
    tick_font_size: int = 18,
    tick_labels: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Displays a window, allowing user to click points on image to return
    pixel coordinates of desired targets.

    ### Note
    Requires interactive plotting backend. I don't forsee this as an issue, but in the event
    the plot doesn't show / function, check your backend:
    >>> matplotlib.get_backend() # e.g., QtAgg, macosx

    Your backend should be under "Interactive backends"
    [https://matplotlib.org/stable/users/explain/figure/backends.html]

    Args:
        datacube (np.memmap):
            3D datacube object, shape (R,C,B).
        display_scale (int, optional):
            Ratio of display scale e.g. 8 -> displayed at 1/8 resolution. Default is 4.
        max_points (int, optional):
            Maximum points able to be plotted on GUI. Default is 100.
        header_font_size (int, optional):
            Title text font size. Default is 35.
        controls_font_size (int, optional):
            Dialogue box font size. Default is 28.
        label_font_size (int, optional):
            Slider label text. Default is 25.
        tick_font_size (int, optional):
            Slider tickmark size. Default is 18.
        tick_labels (list):
            Slider wavelength labels. If None, labels are "B#" for each B in datacube.
            Currently bunk, doesn't work. Default is None.

    Returns:
        tuple[np.ndarray]:
            List of coordinates (row, col) for 1D arrays `targets` and `background` (in that order). shape=(n_coords, 2).
    """
    # ==========================================================
    # Initial Setup
    # ==========================================================

    monitor = get_monitors()[0]
    px = 1 / plt.rcParams["figure.dpi"]
    figsize = (int(0.9 * monitor.width * px), int(0.9 * monitor.height * px))

    rows, cols, bands = datacube.shape
    red_idx, green_idx, blue_idx = _compute_rgb_indices(bands)

    disp_rows, disp_cols = datacube[::display_scale, ::display_scale, 0].shape

    rgb_display = np.empty((disp_rows, disp_cols, 3), dtype=np.uint8)

    _fill_rgb_buffer(rgb_display, datacube, display_scale, red_idx, green_idx, blue_idx)

    targets_coords = np.zeros((max_points, 2), dtype=np.int32)
    backgrounds_coords = np.zeros((max_points, 2), dtype=np.int32)

    state = GUIState()

    # ==========================================================
    # Figure & Axes
    # ==========================================================

    fig, ax = plt.subplots(
        ncols=2,
        figsize=figsize,
        num="Coordinate Extraction Window",
        gridspec_kw={"width_ratios": [4, 1]},
    )

    img_display = ax[0].imshow(rgb_display, interpolation="nearest")
    ax[0].set_xlim(0, disp_cols)
    ax[0].set_ylim(disp_rows, 0)
    ax[0].axis("off")
    ax[0].set_autoscale_on(False)

    def set_mode_title():
        ax[0].set_title(f"Mode: {state.mode.upper()}", fontsize=header_font_size)

    # ==========================================================
    # Controls Panel
    # ==========================================================

    # Controls text
    controls_text = """
                            Instructions
===========================
1) Adjust the pseudocolor image with the 
    left sliders (visual effect only)
2) Zoom-in ('o') to get a closer look
3) Click on the image to create points
4) Switch between creating targets & 
    background points with 't' and 'b'
5) Save/quit ('q') when finished adding points

                            Description
===========================
Click the image to create target and 
background points. 
Targets (red-dot) become visually enhanced, 
Backgrounds (blue-dot) get suppressed. 

(TIP) Only one target and one background 
point are required.

                              Controls
===========================
Left click    : add point
Right click  : undo last point
t                 : target selection mode 
b                : background selection mode
q or ESC     : save/quit
p or o         : pan/zoom in

(TIP) Unselect zoom tool before 
selecting points
"""

    ax[1].text(
        -0.4,
        0.95,
        controls_text,
        transform=ax[1].transAxes,
        fontsize=controls_font_size,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="lightsteelblue", alpha=0.8),
    )

    ax[1].axis("off")
    ax[1].set_title("Controls", fontsize=header_font_size)

    # ==========================================================
    # Scatter Plots
    # ==========================================================

    targets_scatter = ax[0].scatter([], [], c="red", s=25, animated=True)
    backgrounds_scatter = ax[0].scatter([], [], c="blue", s=25, animated=True)

    def update_scatters():
        targets_scatter.set_offsets(
            _to_display_coords(targets_coords, state.t_count, display_scale)
        )
        backgrounds_scatter.set_offsets(
            _to_display_coords(backgrounds_coords, state.b_count, display_scale)
        )

    # ==========================================================
    # Rendering (Blitting)
    # ==========================================================

    def refresh_background():
        targets_scatter.set_visible(False)
        backgrounds_scatter.set_visible(False)

        fig.canvas.draw()
        state.background = fig.canvas.copy_from_bbox(ax[0].bbox)

        targets_scatter.set_visible(True)
        backgrounds_scatter.set_visible(True)

    def redraw():
        if state.background is None:
            return

        fig.canvas.restore_region(state.background)
        ax[0].draw_artist(targets_scatter)
        ax[0].draw_artist(backgrounds_scatter)
        fig.canvas.blit(ax[0].bbox)
        fig.canvas.flush_events()

    # ==========================================================
    # Point Management
    # ==========================================================

    def add_point(row, col):
        if state.mode == MODE_TARGET and state.t_count < max_points:
            targets_coords[state.t_count] = (row, col)
            state.t_count += 1
            state.history.append(MODE_TARGET)

        elif state.mode == MODE_BACKGROUND and state.b_count < max_points:
            backgrounds_coords[state.b_count] = (row, col)
            state.b_count += 1
            state.history.append(MODE_BACKGROUND)

        update_scatters()

    def undo_last():
        if not state.history:
            return

        last = state.history.pop()

        if last == MODE_TARGET:
            state.t_count -= 1
        else:
            state.b_count -= 1

        update_scatters()

    # ==========================================================
    # Event Handlers
    # ==========================================================

    def on_key(event):
        if event.key in ("q", "escape"):
            plt.close(fig)

        elif event.key == "t":
            state.mode = MODE_TARGET
            set_mode_title()
            refresh_background()
            redraw()

        elif event.key == "b":
            state.mode = MODE_BACKGROUND
            set_mode_title()
            refresh_background()
            redraw()

    def on_click(event):
        if fig.canvas.toolbar.mode != "":
            return

        if event.inaxes != ax[0]:
            return

        if event.xdata is None or event.ydata is None:
            return

        if event.button == MouseButton.RIGHT:
            undo_last()
            redraw()
            return

        if event.button == MouseButton.LEFT:
            row, col = _coords_to_fullres_scale(event, display_scale)
            row, col = _clamp_point(row, col, rows, cols)
            add_point(row, col)
            redraw()

    def on_draw(event):
        if event.canvas != fig.canvas:
            return

        state.background = fig.canvas.copy_from_bbox(ax[0].bbox)
        ax[0].draw_artist(targets_scatter)
        ax[0].draw_artist(backgrounds_scatter)

    # Initialize GUI
    set_mode_title()
    refresh_background()
    redraw()

    # ==========================================================
    # Sliders
    # ==========================================================

    fig.subplots_adjust(left=0.05)

    ax_band_r = fig.add_axes([0.02, 0.25, 0.04, 0.5])
    ax_band_g = fig.add_axes([0.05, 0.25, 0.04, 0.5])
    ax_band_b = fig.add_axes([0.08, 0.25, 0.04, 0.5])

    band_slider_r = Slider(
        ax_band_r,
        label="R",
        valmin=0,
        valmax=bands - 1,
        valinit=red_idx,
        valstep=1,
        orientation="vertical",
        handle_style={"facecolor": "red"},
    )

    band_slider_g = Slider(
        ax_band_g,
        label="G",
        valmin=0,
        valmax=bands - 1,
        valinit=green_idx,
        valstep=1,
        orientation="vertical",
        handle_style={"facecolor": "green"},
    )

    band_slider_b = Slider(
        ax_band_b,
        label="B",
        valmin=0,
        valmax=bands - 1,
        valinit=blue_idx,
        valstep=1,
        orientation="vertical",
        handle_style={"facecolor": "blue"},
    )

    # Set fontsize
    for slider in [band_slider_r, band_slider_g, band_slider_b]:
        slider.label.set_fontsize(label_font_size)
        slider.valtext.set_fontsize(label_font_size)

    def update(_):
        _fill_rgb_buffer(
            rgb_display,
            datacube,
            display_scale,
            int(band_slider_r.val),
            int(band_slider_g.val),
            int(band_slider_b.val),
        )
        img_display.set_data(rgb_display)
        refresh_background()
        redraw()

    band_slider_r.on_changed(update)
    band_slider_g.on_changed(update)
    band_slider_b.on_changed(update)

    # ==========================================================
    # Connect Events
    # ==========================================================

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("draw_event", on_draw)

    plt.show()
    plt.close("all")

    if state.t_count == 0 and state.b_count == 0:
        raise ValueError("No coordinates clicked. Terminating program.")

    return (
        targets_coords[: state.t_count].copy(),
        backgrounds_coords[: state.b_count].copy(),
    )


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

    # Check that user didn't quit without selecting any points
    # Prevent propogating errors
    if not any([np.size(targets_coords), np.size(backgrounds_coords)]):
        raise ValueError("No coordinates clicked. Terminating program")

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
    assert any(a.size for a in spectra), "[save_spectra] Spectra cannot be empty"

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
    print(f"Saving data to: {dst_path}")
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


def load_spectral_lib(src_path: str) -> tuple[np.ndarray, ...]:
    """
    Loads targets from NumPy zip file (.npz).

    -------
    Example
    -------
    target_coords, target_members, background_coords, background_members = load_spectra(...)

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
