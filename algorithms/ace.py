"""
Filename: ace.py
Description: Returns ACE detection scores
"""

import numpy as np
from scipy.linalg import solve

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "01-5-2025"
__email__ = "mt9485@rit.edu"


def ace(
    datacube: np.memmap,
    target_members: np.ndarray,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    """
    Adaptive Coherence/Cosine Estimator (ACE) algorithm according to
    Kraut & Scharf (1999).

    Args:
        datacube (np.memmap):
            3D image cube shape (R, C, B).
        target_members (np.ndarray):
            Spectral members with shape (M, B) where M is the number of members.
        chunk_size (int, optional):
            Number of pixels to process at once.
            Decrease to reduce memory footprint.
            Increment to increase throughput.

    Returns:
        np.ndarray: 3D array `(R, C, M)` where M is the number of members.
        ACE scores with maximum value of 1 correspond to a perfect match.
    """

    # Return flattened view
    rows, cols, bands = datacube.shape
    n_pixels = rows * cols
    datacube = datacube.reshape((n_pixels, bands), copy=None)

    M = target_members.shape[0]

    # ------------------------------------------------------------
    # Background statistics
    # ------------------------------------------------------------

    # ==============================
    # Background mean
    # ==============================

    mean = np.zeros(bands, dtype=np.float64)

    # chunked sum
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        chunk = datacube[start:end]
        mean += chunk.sum(axis=0)

    # divide to get average sum
    mean /= n_pixels

    # ==============================
    # Covariance matrix
    # ==============================
    cov = np.zeros((bands, bands), dtype=np.float64)

    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)

        # mean center
        chunk = datacube[start:end] - mean

        # cov(X, Y) and var(X)
        cov += chunk.T @ chunk

    # divide (N-1)
    cov /= n_pixels - 1

    # Regularize slightly for numerical stability
    cov.flat[:: bands + 1] += 1e-12

    # ------------------------------------------------------------------
    # Precompute target terms
    # ------------------------------------------------------------------

    t_centered = target_members - mean  # (M, B)
    t_ic = solve(cov, t_centered.T, assume_a="pos").T
    denom_t = np.sqrt(np.sum(t_ic * t_centered, axis=1))  # (M,)

    # Output buffer, later reshaped
    ace_flat = np.empty((n_pixels, M), dtype=np.float64)

    # ------------------------------------------------------------------
    # Compute ACE scores
    # ------------------------------------------------------------------
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)

        chunk = datacube[start:end] - mean  # (N, B)
        chunk_ic = solve(cov, chunk.T, assume_a="pos").T

        denom_chunk = np.sqrt(np.sum(chunk_ic * chunk, axis=1))  # (N,)

        # Numerator
        num = t_ic @ chunk.T  # (M, N)

        ace_flat[start:end, :] = (
            num / (denom_t[:, None] * denom_chunk[None, :] + 1e-12)
        ).T  # shitty fix but I'm lazy

    return np.reshape(ace_flat, (rows, cols, M))
