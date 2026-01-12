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


def _calc_flush_rate(shape: tuple[int, int], dtype: np.dtype) -> int:
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

    # Calculate flush rate: number of bands before exceeding memory
    flush_rate = int(mem_free // mem_band)  # int floor operation

    return flush_rate


def bgp(datacube: np.memmap, dst_path: str, dst_name: str | None = None):
    """
    Generates pairwise combinations of bands. Used to enhance performance of OSP.

    Args:
        datacube (np.memmap):
            3D datacube shape=(cube.rows, cube.cols, cube.bands). Data range assumed [0,1].
        dst_path (str):
            Output path of new datacube. Creates new output directory if necessary.
        dst_name (str or None, optional):
            If provided, name appended to dst_path with suffix (.npy). If not provided, assumes
            dst_path includes name and suffix.

    Returns:
        np.memmap: Datacube with original + generated bands in BIP format (cube.rows, cube.cols, cube.bands).
    """
    # ----------------------------------------------------------
    # Output setup
    # ----------------------------------------------------------

    # Initialization progress bar
    pbar = tqdm(total=4, desc="Initialization", unit="", colour="blue")

    # =============================
    # Pairwise combinations
    # =============================

    # Cube information
    rows, cols, bands = datacube.shape
    dst_dtype = datacube.dtype

    # Calculate number of output bands
    num_pairwise = bands * (bands - 1) // 2
    dst_bands: int = bands + num_pairwise

    pbar.update(1)

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
    pbar.update(1)

    # =============================
    # Output datacube
    # =============================

    dst_shape = (rows, cols, dst_bands)

    dst_datacube: np.memmap = np.lib.format.open_memmap(
        dst_path, mode="w+", dtype=dst_dtype, shape=dst_shape
    )

    pbar.update(1)

    # =============================
    # Temporary array and flushrate
    # =============================

    # total num bands able to be stored on memory
    flush_rate = _calc_flush_rate((rows, cols), dst_dtype)
    flush_rate = int(flush_rate * 0.85)  # Allot 85% available memory

    # C-order datacube for fast writing
    temp_datacube: np.ndarray = np.empty((rows, cols, flush_rate), dtype=dst_dtype)

    pbar.update(1)

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

        # Normalize
        max_val = np.max(gen_band)
        if max_val != 0:
            gen_band /= max_val

        # Populate temp cube
        temp_datacube[:, :, buffer_count] = gen_band
        buffer_count += 1

        pbar.update(1)

        # Flush when buffer is full
        if buffer_count == flush_rate:

            # Populate dst cube
            dst_datacube[:, :, out_idx : out_idx + buffer_count] = temp_datacube[
                :, :, :buffer_count
            ]
            dst_datacube.flush()

            out_idx += buffer_count
            buffer_count = 0

    # Write any remaining bands
    if buffer_count:
        dst_datacube[:, :, out_idx : out_idx + buffer_count] = temp_datacube[
            :, :, :buffer_count
        ]

    # Close the progressbar
    pbar.close()

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
    max_targets: int = 20,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """
    Generalized Orthogonal Subspace Projection (GOSP) target detection
    following Chang & Ren (2000).

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
            Stack of GOSP score maps with values in [0, 1].
            Shape: (R, C, T), where T (num targets) is
            determined dynamically by the stopping criteria.
            dtype: float32.
    """
    # ------------------------------------------------------------
    # Flatten cube
    # ------------------------------------------------------------
    rows, cols, bands = datacube.shape
    n_pixels = rows * cols
    X = datacube.reshape((n_pixels, bands), copy=False)

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------

    # Target matrix (columns = extracted targets, NOT orthonormal yet)
    T = np.empty((bands, 0), dtype=np.float64)

    # Orthogonal projector onto complement of target subspace
    P_perp = np.eye(bands)

    # prev_max_energy = np.inf

    # Stored for TCP
    p_perp_list = []
    target_list = []

    # ------------------------------------------------------------
    # Iterative Target Generation Process (TGP)
    # ------------------------------------------------------------

    progress = tqdm(desc="TGP", unit=" target", colour="blue")

    while True:

        # ==============================
        # Find max-energy residual pixel
        # ==============================

        max_energy = -np.inf
        max_idx = -1

        for start in range(0, n_pixels, chunk_size):
            stop = min(start + chunk_size, n_pixels)
            chunk = X[start:stop] @ P_perp
            energies = np.linalg.norm(chunk, axis=1)

            idx = np.argmax(energies)
            if energies[idx] > max_energy:
                max_energy = energies[idx]
                max_idx = start + idx

        # ==============================
        # Candidate target
        # ==============================

        x = X[max_idx]

        # ==============================
        # Stopping criteria
        # ==============================

        # if max_energy <= energy_thresh:
        #     info("Loop broken due to max energy threshold")
        #     break

        # rel_drop = (prev_max_energy - max_energy) / prev_max_energy
        # if rel_drop <= relative_drop_thresh:
        #     info("Loop broken due to relative energy drop")
        #     break

        if T.shape[1] >= max_targets:
            print("Loop broken due to max targets found")
            break

        opci = _opci(projector=P_perp, target=x)
        print(f"OPCI: {opci}")
        if opci <= opci_thresh:
            break

        # prev_max_energy = max_energy

        # ==============================
        # Accept new target
        # ==============================

        # Store projector BEFORE update (TCP requirement)
        p_perp_list.append(P_perp.copy())

        # Append raw target vector (column-wise)
        T = np.column_stack((T, x))

        # QR decomposition (Householder)
        # Q has orthonormal columns spanning target subspace
        Q, _ = np.linalg.qr(T, mode="reduced")

        # Update projector
        P_perp = np.eye(bands) - Q @ Q.T

        # Store orthonormalized target direction for TCP
        # (background-orthogonal by construction)
        target_list.append(Q[:, -1].copy())

        progress.update()

    # ------------------------------------------------------------
    # Target Classification Process (TCP)
    # ------------------------------------------------------------

    n_targets = len(target_list)
    score_maps = np.empty((rows, cols, n_targets), dtype=np.float32)

    for idx, (q, p_perp) in tqdm(
        enumerate(zip(target_list, p_perp_list)),
        total=n_targets,
        desc="TCP",
        colour="magenta",
    ):
        # Project target using projector at that iteration
        target_proj = p_perp @ q
        target_norm = np.linalg.norm(target_proj) + 1e-12

        out = np.empty(n_pixels, dtype=np.float32)

        for start in range(0, n_pixels, chunk_size):
            stop = min(start + chunk_size, n_pixels)
            chunk = X[start:stop] @ p_perp

            data_norm = np.linalg.norm(chunk, axis=1) + 1e-12
            score = np.abs(chunk @ target_proj) / (data_norm * target_norm)
            out[start:stop] = np.clip(score, 0, 1)

        score_maps[:, :, idx] = out.reshape(rows, cols)

    return score_maps
