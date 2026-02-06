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

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "02-06-2025"
__email__ = "mt9485@rit.edu"


def target_selection_gui(
    datacube: np.memmap,
    *,
    band_labels: list | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Displays a window, allowing user to click points on image to return
    pixel coordinates of desired targets.

    ### Note
    Requires interactive plotting backend. I don't forsee this as an issue, but in the event
    the plot doesn't show / function, check your backend:
    >>> matplotlib.get_backend() # e.g., QtAgg, macosx

    Your backend should be under "Interactive backends" [https://matplotlib.org/stable/users/explain/figure/backends.html]

    Args:
        datacube (np.memmap):
            3D datacube object, shape (R,C,B).
        band_labels (list or None, optional):
            Tick mark labels for color slider. Used to display which wavelength each tick corresponds to.
            Wavelengths in **ascending** order i.e. [400nm -> 700nm].
    
    ## Kwargs:
        max_points (int): Maximum points able to be plotted on GUI.
        header_font_size (int): Title text font size
        controls_font_size (int): Dialogue box font size
        label_size (int): Slider label text
        display_scale (int): Ratio of display scale e.g. 8 -> displayed at 1/8 resolution. 
        
    Returns:
        tuple[np.ndarray]:
            List of coordinates (row, col) for 1D arrays `targets` and `background` (in that order). shape=(n_coords, 2).
    """

    # ------------------------------------------------------------
    # Display parameters
    # ------------------------------------------------------------

    # Downsample factor for display speed
    DISPLAY_SCALE = kwargs.pop("display_scale", 4)

    # Preallocate coordinate arrays
    # Increase to allow more points on screen
    MAX_POINTS = kwargs.pop("max_points", 100)

    # Size of text
    HEADER_FONT_SIZE = kwargs.pop("header_font_size", 35)
    CONTROLS_FONT_SIZE = kwargs.pop("controls_font_size", 28)

    # Slider labels
    LABEL_SIZE = kwargs.pop("label_size", 25)

    # ------------------------------------------------------------
    # Compile initial display image
    # ------------------------------------------------------------

    # Shape
    rows, cols, bands = datacube.shape

    # Artibrary indices @ band quartiles
    red_idx, green_idx, blue_idx = (
        int(bands * 0.75),
        int(bands * 0.5),
        int(bands * 0.25),
    )

    # ------------------------------------------------------------
    # Display downsampling (display only)
    # ------------------------------------------------------------

    disp_rows, disp_cols = datacube[::DISPLAY_SCALE, ::DISPLAY_SCALE, 0].shape

    # ------------------------------------------------------------
    # Preallocate RGB display buffer
    # ------------------------------------------------------------

    # R,G,B images for display ONLY, uint8 for less plotting data
    rgb_display = np.empty(
        (disp_rows, disp_cols, 3),
        dtype=np.uint8,
    )

    def fill_rgb(r, g, b):
        """Fill reusable RGB buffer"""
        rgb_display[..., 0] = (
            (datacube[::DISPLAY_SCALE, ::DISPLAY_SCALE, r] * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        rgb_display[..., 1] = (
            (datacube[::DISPLAY_SCALE, ::DISPLAY_SCALE, g] * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        rgb_display[..., 2] = (
            (datacube[::DISPLAY_SCALE, ::DISPLAY_SCALE, b] * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )

    # Initial image render
    fill_rgb(red_idx, green_idx, blue_idx)

    # ------------------------------------------------------------
    # Output storage
    # ------------------------------------------------------------

    targets_coords = np.empty((MAX_POINTS, 2), dtype=np.int32)
    backgrounds_coords = np.empty((MAX_POINTS, 2), dtype=np.int32)

    t_count = 0
    b_count = 0

    history = []  # stack of class keys
    mode = "targets"  # vs background

    # ------------------------------------------------------------
    # Define Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        ncols=2,
        figsize=(30, 20),
        num="Coordinate Extraction Window",
        gridspec_kw={"width_ratios": [4, 1]},
    )

    # Store image artist
    img_display = ax[0].imshow(
        rgb_display,
        interpolation="nearest",
    )

    ax[0].set_xlim(0, disp_cols)
    ax[0].set_ylim(disp_rows, 0)

    ax[0].set_title("Mode: TARGETS", fontsize=HEADER_FONT_SIZE)
    ax[0].axis("off")
    ax[0].set_autoscale_on(False)

    ax[0].set_title("Mode: TARGET", fontsize=HEADER_FONT_SIZE)
    ax[0].axis("off")
    ax[0].set_autoscale_on(False)

    # Controls text
    controls_text = """
Instructions
--------
1) Adjust the image colors 
   with the left sliders
2) Use the zoom (magnifying glass) to
   get a closer look
3) Click on the image to create points
4) Save/quit when finished adding points

Description
----------------
target (red) = area to visually enhance
background (blue) = background/clutter

! Remember to unselect magnifying tool
when selecting points in zoomed view

Controls
------------
Left click    : add point
Right click  : undo last point
t                 : target selection mode 
b                : background selection mode
q or ESC     : save/quit
"""

    ax[1].text(
        -0.4,
        0.95,
        controls_text,
        transform=ax[1].transAxes,
        fontsize=CONTROLS_FONT_SIZE,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax[1].axis("off")
    ax[1].set_title("Controls", fontsize=HEADER_FONT_SIZE)

    # Initialize scatter plots
    targets_scatter = ax[0].scatter([], [], c="red", s=45, animated=True)
    backgrounds_scatter = ax[0].scatter([], [], c="blue", s=45, animated=True)

    # ------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------

    background = None

    def refresh_background():
        """Capture static background for blitting."""

        nonlocal background
        
        # Hide the scatter points temporarily so they aren't 'baked' into the background
        targets_scatter.set_visible(False)
        backgrounds_scatter.set_visible(False)
        
        # Redraw/refresh background
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax[0].bbox)
        
        # Make points visible again 
        targets_scatter.set_visible(True)
        backgrounds_scatter.set_visible(True)

    def redraw_points():
        """Fast redraw using blitting."""

        if background is None:
            return
        
        # Restore the clean background (the image at current zoom level)
        fig.canvas.restore_region(background)

        # Draw the current coordinates (scaled for display)
        ax[0].draw_artist(targets_scatter)
        ax[0].draw_artist(backgrounds_scatter)

        # Push to screen
        fig.canvas.blit(ax[0].bbox)
        fig.canvas.flush_events()

    def on_key(event):
        """On keypress from keyboard"""
        nonlocal mode

        if event.key in ("q", "escape"):
            plt.close(fig)

        elif event.key == "t":
            mode = "targets"
            ax[0].set_title("Mode: TARGET", fontsize=HEADER_FONT_SIZE)
            refresh_background()
            redraw_points()

        elif event.key == "b":
            mode = "background"
            ax[0].set_title("Mode: BACKGROUND", fontsize=HEADER_FONT_SIZE)
            refresh_background()
            redraw_points()

    def on_click(event):
        nonlocal mode, t_count, b_count
        
        # If the zoom or pan tool is active, don't add a point
        if fig.canvas.toolbar.mode != "":
            return

        # Prevent user from clicking out-of-bounds
        if event.xdata is None or event.ydata is None:
            return

        if event.inaxes != ax[0]:
            return

        # Undo (right click)
        if event.button == MouseButton.RIGHT:

            if history:
                selection_mode = history.pop()
                if selection_mode == "targets":
                    t_count -= 1
                    # FIX: Divide by DISPLAY_SCALE to map back to screen coordinates
                    visual_coords = targets_coords[:t_count][:, ::-1] / DISPLAY_SCALE
                    targets_scatter.set_offsets(visual_coords)

                elif selection_mode == "background":
                    b_count -= 1
                    # FIX: Divide by DISPLAY_SCALE to map back to screen coordinates
                    visual_coords = (
                        backgrounds_coords[:b_count][:, ::-1] / DISPLAY_SCALE
                    )
                    backgrounds_scatter.set_offsets(visual_coords)

                redraw_points()

            return

        # Add point (left click)
        if event.button == MouseButton.LEFT:

            # Map display click -> full resolution
            # Store as (row,col) for downline processing
            row = int(round(event.ydata)) * DISPLAY_SCALE
            col = int(round(event.xdata)) * DISPLAY_SCALE

            # Clamp invalid values
            row = np.clip(row, 0, rows - 1)
            col = np.clip(col, 0, cols - 1)

            if mode == "targets" and t_count < MAX_POINTS:
                # Append clicked location (Full Resolution)
                targets_coords[t_count] = (row, col)
                t_count += 1

                # FIX: Divide by DISPLAY_SCALE for visualization only
                visual_coords = targets_coords[:t_count][:, ::-1] / DISPLAY_SCALE
                targets_scatter.set_offsets(visual_coords)

                history.append("targets")

            elif mode == "background" and b_count < MAX_POINTS:
                # Append clicked location (Full Resolution)
                backgrounds_coords[b_count] = (row, col)
                b_count += 1

                # FIX: Divide by DISPLAY_SCALE for visualization only
                visual_coords = backgrounds_coords[:b_count][:, ::-1] / DISPLAY_SCALE
                backgrounds_scatter.set_offsets(visual_coords)

                history.append("background")

            redraw_points()

    def on_draw(event):
        """
        Capture the background image pixels after a full draw.
        """
        nonlocal background
        # Ensure we are drawing the correct canvas
        if event.canvas != fig.canvas:
            return

        # Capture the image (without the scatter points, as they are animated=True)
        background = fig.canvas.copy_from_bbox(ax[0].bbox)
        
        # Re-draw the scatter points on top of the new background
        ax[0].draw_artist(targets_scatter)
        ax[0].draw_artist(backgrounds_scatter)
        
        return

    def update(val):
        """Update the RGB display based on slider values."""
        nonlocal background

        r_val = int(band_slider_r.val)
        g_val = int(band_slider_g.val)
        b_val = int(band_slider_b.val)

        fill_rgb(r_val, g_val, b_val)
        img_display.set_data(rgb_display)

        refresh_background()
        redraw_points()

    # ------------------------------------------------------------
    # Connect events to figure
    # ------------------------------------------------------------
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("draw_event", on_draw)
    

    # ------------------------------------------------------------
    # Sliders setup
    # ------------------------------------------------------------

    fig.subplots_adjust(left=0.05)

    ax_band_r = fig.add_axes([0.02, 0.25, 0.04, 0.5])
    ax_band_g = fig.add_axes([0.05, 0.25, 0.04, 0.5])
    ax_band_b = fig.add_axes([0.08, 0.25, 0.04, 0.5])

    band_slider_r = Slider(
        ax=ax_band_r,
        label="R",
        valmin=0,
        valmax=bands - 1,
        valinit=red_idx,
        orientation="vertical",
        valstep=1,
    )

    band_slider_g = Slider(
        ax=ax_band_g,
        label="G",
        valmin=0,
        valmax=bands - 1,
        valinit=green_idx,
        orientation="vertical",
        valstep=1,
    )

    band_slider_b = Slider(
        ax=ax_band_b,
        label="B",
        valmin=0,
        valmax=bands - 1,
        valinit=blue_idx,
        orientation="vertical",
        valstep=1,
    )

    # Increase label font size ("R", "G", "B")
    band_slider_r.label.set_size(LABEL_SIZE)
    band_slider_g.label.set_size(LABEL_SIZE)
    band_slider_b.label.set_size(LABEL_SIZE)

    # Increase value font size (the number)
    band_slider_r.valtext.set_size(LABEL_SIZE)
    band_slider_g.valtext.set_size(LABEL_SIZE)
    band_slider_b.valtext.set_size(LABEL_SIZE)

    # Add slider ticks and labels
    for ax_slider in [ax_band_r, ax_band_g, ax_band_b]:
        # Tick location
        tick_locations = [x for x in range(bands)]
        tick_labels = (
            [f"B{n}" for n in range(bands)] if not band_labels else band_labels
        )

        ax_slider.set_yticks(tick_locations)
        ax_slider.set_yticklabels(tick_labels, fontsize=18)

        # Ensure ticks are visible
        ax_slider.tick_params(
            axis="y", length=10, width=2, colors="black", direction="inout"
        )

    # Enable slider visibility
    ax_band_r.xaxis.set_visible(True)
    ax_band_g.xaxis.set_visible(True)
    ax_band_b.xaxis.set_visible(True)

    band_slider_r.on_changed(update)
    band_slider_g.on_changed(update)
    band_slider_b.on_changed(update)

    plt.show()

    # ------------------------------------------------------------
    # Cleanup and Return
    # ------------------------------------------------------------

    plt.close("all")

    targets_final = targets_coords[:t_count]
    backgrounds_final = backgrounds_coords[:b_count]

    return (
        targets_final.copy(),
        backgrounds_final.copy(),
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
