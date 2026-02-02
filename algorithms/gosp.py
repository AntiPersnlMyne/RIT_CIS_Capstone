"""
Filename: gosp.py
Description: Generalized Orthogonal Subspace Projection
    - Implements ATDCA from the paper DOI:10.1109/36.885199 (Chang & Ren, 2000)
    - Implements BGP from same paper.
"""

import numpy as np
import os
from psutil import virtual_memory
from itertools import combinations
from tqdm import tqdm
from pathlib import Path

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "12-31-2025"
__email__ = "mt9485@rit.edu"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _calc_buffer_max(shape: tuple[int, int], dtype: np.dtype) -> int:
    """
    Exact copy of the function in dataloader.py. Exists because "ImportError: attempted
    relative import beyond top-level package"

    Args:
        shape (tuple):
            Shape of band
        dtype (np.dtype):
            NumPy dtype of band

    Returns:
        int: Number of bands, of shape and dtype, capable of being stored in avaiable memory
    """

    # Memory available in bytes
    svmem = virtual_memory()
    mem_free = svmem.available

    # Memory per band
    dtype_size = dtype.itemsize
    pixels = np.prod(shape)
    mem_band = pixels * dtype_size

    # Calculate buffer rate: number of bands before exceeding memory
    buffer_size = int(mem_free // mem_band)

    return buffer_size


def bgp(datacube: np.memmap, dst_path: str, dst_name: str | None = None):
    """
    Generates pairwise combinations of bands. Used to generalize performance of OSP by
    expanding the endmember subspace. In Chang's words: "to fix the band number constraint
    ... extending hyperspectral algorithms into multispectral"
    
    In plain words, OSP works best if the target has a unique spectral signature, and by 
    adding "more bands", there's a greater chance the target will be given a unique signature

    Args:
        datacube (np.memmap):
            3D datacube shape (R,C,B). Data range assumed [0,1].
        dst_path (str):
            Output path of new datacube. Creates new output directory if necessary.
        dst_name (str or None, optional):
            If provided, name appended to dst_path with suffix (.npy). If not provided, assumes
            dst_path includes name and suffix.

    Returns:
        np.memmap: Datacube with original + generated bands in BIP format (R,C,G), where
        G = (B * (B - 1) // 2) + B.
    """

    # =============================
    # Pairwise combinations
    # =============================

    # Cube information
    rows, cols, bands = datacube.shape
    dst_dtype = datacube.dtype

    # Calculate number of output bands
    num_pairwise = bands * (bands - 1) // 2
    dst_bands: int = bands + num_pairwise

    # =============================
    # Output path
    # =============================

    # Create output directory, if necessary
    if Path(dst_path).is_dir():
        os.makedirs(dst_path, exist_ok=True)

    # Add (.npy) extension if doesn't exist
    dst_path_suffix = Path(dst_path).suffix
    if not dst_path_suffix:
        if dst_name:
            # Remove any incompatible file format
            name_base = Path(dst_name).stem
            dst_path = os.path.join(dst_path, name_base + ".npy")
        else:
            raise ValueError(
                "[bgp] output path must be directory or end with 'filename.npy'"
            )

    del dst_path_suffix, name_base

    # =============================
    # Output datacube
    # =============================

    dst_shape = (rows, cols, dst_bands)

    dst_datacube: np.memmap = np.lib.format.open_memmap(
        dst_path, mode="w+", dtype=dst_dtype, shape=dst_shape
    )

    # =============================
    # Temporary array and flushrate
    # =============================

    # total num bands able to be stored on memory
    buffer_max = _calc_buffer_max((rows, cols), dst_dtype)
    buffer_max = int(buffer_max * 0.95)

    # C-order datacube for fast writing
    temp_datacube: np.ndarray = np.empty((rows, cols, buffer_max), dtype=dst_dtype)

    # ----------------------------------------------------------
    # Progress tracking
    # ----------------------------------------------------------

    pbar = tqdm(total=dst_bands, desc="Generation", unit="band", colour="green")

    # Track written band loc in memmap
    out_idx: int = 0

    # ----------------------------------------------------------
    # Original bands
    # ----------------------------------------------------------

    # Beginning of dst is copy of original cube
    dst_datacube[:, :, :bands] = datacube[:, :, :]
    out_idx += bands

    dst_datacube.flush()
    pbar.update(bands)

    # -----------------------------
    # Pairwise products
    # -----------------------------

    buffer_count = 0

    for i, j in combinations(range(bands), 2):

        # Compute product
        gen_band = datacube[:, :, i] * datacube[:, :, j]

        # Populate temp cube
        temp_datacube[:, :, buffer_count] = gen_band
        buffer_count += 1

        pbar.update(1)

        # Flush when buffer is full
        if buffer_count == buffer_max:

            # Populate dst cube
            dst_datacube[:, :, out_idx : out_idx + buffer_count] = temp_datacube[
                :, :, :buffer_count
            ]

            out_idx += buffer_count
            buffer_count = 0

    # Write any remaining bands
    if buffer_count:
        dst_datacube[:, :, out_idx : out_idx + buffer_count] = temp_datacube[
            :, :, :buffer_count
        ]

    # Close the progressbar
    pbar.close()
    print()  # prevent pbar artifact in terminal

    # -----------------------------
    # Return
    # -----------------------------

    # Return opened datacube object with read/write permissions
    return np.lib.format.open_memmap(
        dst_path,
        mode="r+",
        dtype=dst_dtype,
        shape=(rows, cols, dst_bands),
    )


def _opci(projector: np.ndarray, target: np.ndarray) -> float:
    """
    Orthogonal Projection Correlation Index (OPCI) stopping criteria for GOSP. Measures how
    correlated a newly extracted target is with the existing background subspace after
    orthogonal projection.

    Mathematically (Chang & Ren, 2000):
        OPCI(x) = ||P_perp x|| / ||x||

    Note:
        Interpreted as: does the new target really represent new information, or is
        it just another linear combination of what we already have?

    Returns:
        float: Value 0 to 1.
        1 -> novel target (new information),
        0 -> old news (fully explained by background).
    """
    # Numerator: Energy of the target orthogonal to existing background
    projected_target = projector @ target
    projected_target_energy = np.linalg.norm(projected_target)

    # Denom: Total energy of target
    total_energy = np.linalg.norm(target)

    # Handle DIV0 gracefully
    if total_energy < 1e-12:
        return 0.0

    # OPCI is ratio of energies
    return projected_target_energy / total_energy


def gosp(
    datacube: np.memmap,
    opci_thresh: float = 0.09,
    max_targets: int|None = None,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """
    Generalized Orthogonal Subspace Projection (GOSP) target detection
    following Chang & Ren (2000). Revised to be memory safe, and implement
    QR decomposition.

    Automatically extracts spectrally distinct targets using an
    iterative target generation process and produces one OSP
    classification map per extracted target.

    Args:
        datacube (np.memmap):
            3D hyperspectral datacube of shape (R, C, B).
        opci_thresh (float):
            Novelty measure of found targets. \n
            < 0.3: Redundant \n
            0.4-0.7: Marginal \n
            0.7-0.9: Meaningful \n
            > 0.9: Novel
        max_targets (int):
            Maximum number of targets to extract.
        chunk_size (int):
            How much datacube loaded into memory at once.
            Smaller value for less memory usage.
            Greater value for faster processing.

    Returns:
        np.ndarray:
            Stack of GOSP score maps, shape: (R, C, T), where T (n targets)
            is determined dynamically by the stopping criteria.
    """
    # ------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------
    rows, cols, bands = datacube.shape

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    
    # Set no upper threshold on maximum targets to find
    if not max_targets or max_targets <= 0:
        max_targets = np.inf

    # Target matrix (columns = extracted targets, NOT orthonormal yet)
    T = np.empty((bands, 0), dtype=np.float64)

    # Orthogonal projector onto complement of target subspace
    P_perp = np.eye(bands, dtype=np.float64)

    # Stored for TCP
    p_perp_list = []
    target_list = []

    # ------------------------------------------------------------
    # Iterative Target Generation Process (TGP)
    # ------------------------------------------------------------

    while True:

        # ==============================
        # Find max-energy residual pixel
        # ==============================

        max_energy = -np.inf
        max_px = (-1, -1)  # Pixel loc of maximum energy

        for row_begin in range(0, rows, chunk_size):

            row_end = min(row_begin + chunk_size, rows)

            block = datacube[row_begin:row_end]  # (Rchunk, C, B)
            X = block.reshape(-1, bands)  # (Nchunk, B)

            # Residual projection
            Xp = X @ P_perp

            energies = np.linalg.norm(Xp, axis=1)

            idx = np.argmax(energies)

            if energies[idx] > max_energy:
                max_energy = energies[idx]

                # Convert flat index → (row, col)
                local_row = idx // cols
                local_col = idx % cols

                max_px = (row_begin + local_row, local_col)

        # ==============================
        # Candidate target
        # ==============================

        row_maxnrg, col_maxnrg = max_px
        x = datacube[row_maxnrg, col_maxnrg].astype(np.float64)

        # ==============================
        # Stopping criteria
        # ==============================

        if T.shape[1] >= max_targets:
            print("Loop broken due to max targets found")
            break

        opci = _opci(projector=P_perp, target=x)

        print(f"OPCI vs. Thresh: {opci:.4f} > {opci_thresh:.4f}")

        if opci <= opci_thresh:
            break

        # ==============================
        # Accept new target
        # ==============================

        # Store projector (TCP)
        p_perp_list.append(P_perp.copy())

        # Append target
        T = np.column_stack((T, x))

        # Orthonormalize target subspace
        Q, _ = np.linalg.qr(T, mode="reduced")

        # Update projector
        P_perp = np.eye(bands) - Q @ Q.T

        # Store newest orthonormal target
        target_list.append(Q[:, -1].copy())

    # ------------------------------------------------------------
    # Target Classification Process (TCP)
    # ------------------------------------------------------------

    n_targets = len(target_list)

    score_maps = np.empty((rows, cols, n_targets), dtype=np.float32)

    for k, (q, p_perp) in enumerate(zip(target_list, p_perp_list)):

        # Project target
        target_proj = p_perp @ q
        target_norm = np.linalg.norm(target_proj) + 1e-12

        for row_start in range(0, rows, chunk_size):

            row_end = min(row_start + chunk_size, rows)

            block = datacube[row_start:row_end]

            X = block.reshape(-1, bands)

            # Background rejection
            Xp = X @ p_perp

            data_norm = np.linalg.norm(Xp, axis=1) + 1e-12

            score = np.abs(Xp @ target_proj) / (data_norm * target_norm)

            score = np.clip(score, 0.0, 1.0)

            # Restore spatial layout
            score_maps[row_start:row_end, :, k] = score.reshape(
                row_end - row_start, cols
            )

    return score_maps
