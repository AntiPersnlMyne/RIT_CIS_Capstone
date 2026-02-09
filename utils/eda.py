"""
Filename: eda.py
Description: Memory-safe exploratory data analysis
    - Per-band statistics
    - Display statistics
    - Covariance matrix
    - Correlation matrix
    - Display correlation matrix

-------
Example
-------
datacube = np.load("data/datacubes/archimedes_cubes/Arch_93r_92v_bgp.npy", mmap_mode="r")
cov_matrix = compute_covariance_matrix(datacube, chunk_size=5_000_000)
corr_matrix = correlation_matrix(cov_matrix)
plot_corr_matrix(corr_matrix)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from pathlib import Path
import pandas as pd

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "02-04-2025"
__email__ = "mt9485@rit.edu"


def calculate_band_statistics(datacube: np.memmap) -> pd.DataFrame:
    """
    Calculates a dictionary of band statistics per-band

    ----------
    Statistics
    ----------
    - mean
    - std
    - quartile 1
    - quartile 2 (median)
    - quartile 3
    - kurtosis

    Kurtosis is a way to measure gaussinaity
    - Gaussian: ~0
    - Light-tailed: -2 to 0
    - Heavy-tailed: 0 to +10
    - Peaked: +5 to +10

    Args:
        datacube (np.memmap):
            3D datacube of shape (R, C, B)

    Returns:
        pd.DataFrame: Tabular representation of statistics for each band. Rows=bands, cols=stats.
    """
    # ------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------

    # Extract shape
    R, C, B = datacube.shape

    # ------------------------------------------------------------
    # Calculate statistics for each band
    # ------------------------------------------------------------
    
    # Broadcasting results
    means = np.mean(datacube, axis=(0,1))
    # std = np.std(datacube, axis=(0,1))
    # print("Q1 ...")
    # Q1 = np.percentile(datacube, 25, axis=(0,1))
    # print("Median ...")
    # median = np.median(datacube, axis=(0,1))
    # print("Q3 ...")
    # Q3 = np.percentile(datacube, 75, axis=(0,1))

    # Loop through bands, store in dict index
    all_stats = {}
    from tqdm import tqdm
    for band_idx in tqdm(range(B), unit="band", desc="band stats", colour="red"):
        # Extract entire band
        band = datacube[:, :, band_idx]

        # Store in dictionary
        all_stats[f"band_{band_idx}"] = dict(
            mean=means[band_idx],
            standard_deviation=np.std(band),
            # quartile_1=Q1[band_idx],
            # median=median[band_idx],
            # quartile_3=Q3[band_idx],
            # Fisher kurt is 0 for Gauss distrib
            kurtosis=kurtosis(band, fisher=True, axis=None),
        )

    # ------------------------------------------------------------
    # Convert to Pandas DataFrame
    # ------------------------------------------------------------

    # Convert the dictionary of dictionaries to a DataFrame
    stats_df = pd.DataFrame.from_dict(all_stats, orient="index")

    # Statistical information for all bands in datacube
    return stats_df


def display_band_statistics(
    statistics: pd.DataFrame,
    precision: int = 3,
    font_size: str = "12px",
    show_index: bool = True,
    highlight_max: bool = False,
    highlight_min: bool = False,
) -> None:
    """
    Displays statistics as table with HTML formatting. Will (probably) automatically open in browser.

    Args:
        statistics (pd.DataFrame):
            Tabular representation of statistics for each band. Rows=bands, cols=stats.
            Works directly with output from `calculate_band_statistics()`
        precision (int, optional):
            Floating point precision. Defaults to 3.
        font_size (str, optional):
            Size of text. Defaults to "12px".
        show_index (bool, optional):
            If False, hides the Pandas index (leftmost column). Defaults to True.
        background_color (str, optional):
            Background color of table. Defaults to "lightblue".
        highlight_max (bool, optional):
            Highlight the maximum value in each column of the dataframe in blue. Defaults to False.
        highlight_min (bool, optional):
            Highlights the minimum value in each column of the dataframe in orange. Defaults to False.
    """

    # Check dictionary isn't empty
    assert statistics is not None, "[eda] statistics dictionary is empty"

    # ------------------------------------------------------------
    # Format DataFrame
    # ------------------------------------------------------------

    # Create a copy to avoid modifying the original DataFrame
    styled_df = statistics.copy()

    # Format all numeric columns with specified precision
    numeric_columns = styled_df.select_dtypes(include=["number"]).columns
    if len(numeric_columns) > 0:
        format_dict = {col: f"{{:.{precision}f}}" for col in numeric_columns}
        styled_df = styled_df.style.format(format_dict)

    # Apply basic styling
    styled_df = styled_df.set_properties(
        **{
            "background-color": "#424344",
            "color": "black",
            "font-size": font_size,
            "text-align": "center",
            "padding": "8px",
            "border": "1px solid #ddd",
        }
    )

    # Apply header styling
    styled_df = styled_df.set_properties(
        selector="th",
        **{
            "background-color": "#a6aaad",
            "color": "white",
            "font-weight": "bold",
            "border": "1px solid #ddd",
            "padding": "10px",
        },
    )

    # Apply table styling
    styled_df = styled_df.set_properties(
        selector="table",
        **{
            "border-collapse": "collapse",
            "width": "100%",
            "margin": "25px 0px 25px 0px",
            "font-family": "Arial, sans-serif",
        },
    )

    # Highlight maximum values in each column
    if highlight_max:
        for col in numeric_columns:
            max_val = statistics[col].max()
            styled_df = styled_df.applymap(
                lambda x: "background-color: lightblue" if x == max_val else "",
                subset=[col],
            )

    # Highlight minimum values in each column
    if highlight_min:
        for col in numeric_columns:
            min_val = statistics[col].min()
            styled_df = styled_df.applymap(
                lambda x: "background-color: orange" if x == min_val else "",
                subset=[col],
            )

    # Hide index if requested
    if not show_index:
        styled_df = styled_df.hide_index()

    # ------------------------------------------------------------
    # Display
    # ------------------------------------------------------------

    # Stylized DataFrames cannot be displayed with print
    # Save as HTML and immediately open instead
    from webbrowser import open as op
    from os.path import realpath as rp

    styled_df.to_html("formatted_statistics.html")
    op("file://" + rp("formatted_statistics.html"))


def save_band_statistics(statistics: pd.DataFrame, dst_path: str | Path) -> None:
    """
    Saves tabular statistics data out as a CSV file.

    Args:
        statistics (pd.DataFrame | dict):
            Works directly with output from `calculate_band_statistics()`.
        dst_path (str):
            String or pathlib.Path object. Suffix will be replaced by .csv, if any.
    """
    # ------------------------------------------------------------
    # Verify output path
    # ------------------------------------------------------------

    # Convert to Path object
    dst_path = Path(dst_path)
    suffix = dst_path.suffix
    name = dst_path.name

    # Check if path has a filename
    if not name or name == "/":
        raise ValueError("Output path must include a filename")

    # Correct suffix if needed
    if suffix != ".csv":
        dst_path = dst_path.with_suffix(".csv")

    # ------------------------------------------------------------
    # Export / Save
    # ------------------------------------------------------------

    if isinstance(statistics, pd.DataFrame):
        statistics.to_csv(dst_path)
    else:
        raise ValueError("[save] Incorrect format for statistics data")


def cov_matrix(datacube: np.memmap, chunk_size:int = 1) -> np.ndarray:
    """
    Computes the covariance matrix of a dataset stored in a file.

    Parameters:
        datacube (np.memmap):
            3D datacube of shape (R, C, B).
        chunk_size (int):
            Number of image rows to process at a time. If RAM is available, increase 
            for speed, decrease for memory-efficiency. Must be `chunk_size >= 1`

    Returns:
        np.ndarray: Covariance matrix of shape (B, B).
    """

    # Load the dataset
    R, C, B = datacube.shape
    N = R * C

    # ------------------------------
    # (attempt) vectorized cov
    # ------------------------------
    try:
        return np.cov(datacube.reshape(-1, B), rowvar=False)
    except Exception:
        print("Efficient cov operation failed. Memory-safe fallback ...")
        pass

    # ------------------------------
    # Memory-safe cov
    # ------------------------------

    # Output matrix
    cov = np.zeros((B, B), dtype=np.float64)

    # Mean over n_pixels
    means = datacube.mean(axis=(0, 1))  # (B)

    for r_start in range(0, R, chunk_size):
        # Define chunk bounds
        r_end = min(r_start + chunk_size, R)
        
        # Grab data chunk
        chunk = datacube[r_start:r_end, :, :].reshape(-1, B).astype(np.float64)

        # Center the data
        chunk -= means 

        # Accumulate X.T X
        # (B,N) @ (N,B) -> (B,B)
        cov += chunk.T @ chunk

    # Normalize with ddof=1
    cov /= N - 1

    return cov


def corr_matrix(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the correlation matrix from a given covariance matrix.

    Parameters:
        cov_matrix (np.ndarray): A square covariance matrix of shape (bands, bands).

    Returns:
        np.ndarray: A square correlation matrix of shape (bands, bands).
    """
    # Extract standard deviations from the diagonal of the covariance matrix
    std_devs = np.sqrt(np.diag(cov_matrix))

    # Compute the outer product of standard deviations
    outer_product = np.outer(std_devs, std_devs)

    # Compute the correlation matrix by normalizing the covariance matrix
    corr_matrix = cov_matrix / outer_product

    return corr_matrix


def plot_corr_matrix(
    corr_matrix: np.ndarray,
    save_dir: str | None = None,
    labels: list[str] | None = None,
    title: str = "Correlation Matrix",
    show_plot: bool = True,
) -> None:
    """
    Plots the correlation matrix using Matplotlib.

    Args:
        corr_matrix (np.ndarray):
            The correlation matrix shape (bands, bands)
        save_dir (str or None, optional):
            If path is specified, saves figure to directory. Otherwise, function only plots.
            Accepted extensions (required) are: .png and .pdf.
        labels (list[str], optional):
            List of feature names (for axis labels). Defaults to "B#", # is band number.
        title (str, optional):
            Title of the plot.
        show_plot (bool, optional):
            Whether or not to show the plot. Yes, the plot function can optionally plot. But
            it can also save.
    """
    # Create labels
    if labels is None:
        labels = [f"B{i}" for i in range(corr_matrix.shape[0])]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create a mask to hide the upper triangle
    mask = np.triu(
        np.ones(corr_matrix.shape, dtype=bool), k=1
    )  # k=1 excludes the diagonal
    masked_matrix = np.ma.masked_array(corr_matrix, mask=mask)

    # Plot the heatmap using the masked array
    heatmap = ax.imshow(masked_matrix, cmap="coolwarm", aspect="auto")

    # Add colorbar
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.set_label("Correlation Value")

    # Annotate only the lower triangle and diagonal
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i >= j:
                ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=4,
                )

    # Add labels
    ax.set_xlabel("Bands (B)")
    ax.set_ylabel("Bands (B)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # Set title and layout
    ax.set_title(title)
    fig.tight_layout()

    # Save plot
    if save_dir:
        plt.savefig(save_dir) # human visible: png, jpeg, or pdf
        plt.savefig(Path(save_dir).with_suffix(".eps")) # for LaTeX
        
    # Show plot
    if show_plot:
        plt.show()
