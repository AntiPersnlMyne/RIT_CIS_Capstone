"""
Filename: ace.py
Description: Returns ACE detection scores
"""

import numpy as np
from scipy.linalg import qr, solve_triangular

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "01-5-2025"
__email__ = "mt9485@rit.edu"


def ace(
    datacube: np.memmap,
    target_members: np.ndarray,
    chunk_size: int = 128,
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
            Reduce if memory is limited.
            Increase if more RAM is available throughput.

    Returns:
        np.ndarray: 3D array `(R, C, M)` where M is the number of members.
        ACE scores with maximum value of 1 correspond to a perfect match.
    """
    
    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    # Return flattened view
    rows, cols, bands = datacube.shape
    n_pixels = rows * cols
        
    # Get the number of members (M)
    M = target_members.shape[0]

    # ------------------------------------------------------------------
    # Background mean
    # ------------------------------------------------------------------

    # NumPy means
    try:
        mean = np.mean(datacube, axis=(0,1)) # shape (bands)
    # Chunked processing means
    except:
        mean = np.zeros(bands, dtype=np.float64)

        for r_begin in range(0, rows, chunk_size):
            r_end = min(r_begin + chunk_size, rows)

            # Read contiguous block from disk
            block = datacube[r_begin:r_end]  # (Rchunk, C, B)

            # Flatten only in RAM
            block_flat = block.reshape(-1, bands)

            # Accumulate sum
            mean += block_flat.sum(axis=0)
            
        # Normalize 
        mean /= n_pixels

    # ------------------------------------------------------------------
    # Covariance Matrix
    # ------------------------------------------------------------------
    
    cov = np.zeros((bands, bands), dtype=np.float64)

    for r_start in range(0, rows, chunk_size):

        r_end = min(r_start + chunk_size, rows)

        block = datacube[r_start:r_end]

        block_flat = block.reshape(-1, bands)

        # Mean center
        X = block_flat - mean

        # Accumulate outer product
        cov += X.T @ X

    # DDOF
    cov /= (n_pixels - 1)
    
    # Help prevent singular matrix
    eps = 1e-10
    cov.flat[:: bands + 1] += eps

    # QR decomposition
    Q, R = qr(cov, mode="economic")

    # ------------------------------------------------------------
    # Helper: Solve cov x = b using QR
    # ------------------------------------------------------------

    def solve_qr(b):
        """
        Solve cov x = b using precomputed QR.

        Args:
            b (ndarray): Right-hand side (B, N)

        Returns:
            x (ndarray): Solution (B, N)
        """

        # Q^T b
        y = Q.T @ b

        # R x = y
        x = solve_triangular(R, y, lower=False)

        return x
    
    # ------------------------------------------------------------
    # Target Preprocessing
    # ------------------------------------------------------------

    # Mean center targets
    t_centered = target_members - mean  # (M, B)

    # Solve Σ x = t^T
    t_ic = solve_qr(t_centered.T).T  # (M, B)

    # Target denominator
    denom_t = np.sqrt(np.sum(t_ic * t_centered, axis=1))

    # ------------------------------------------------------------
    # ACE Scores
    # ------------------------------------------------------------
    
    # Output buffer 
    score_map = np.empty((rows, cols, M), dtype=np.float64)

    for row_begin in range(0, rows, chunk_size):

        row_end = min(row_begin + chunk_size, rows)

        block = datacube[row_begin:row_end]

        block_flat = block.reshape(-1, bands)

        # Mean center
        X = block_flat - mean

        # Σ^-1 X
        X_ic = solve_qr(X.T).T

        # Pixel denominator
        denom_x = np.sqrt(np.sum(X_ic * X, axis=1))

        # Numerator (M x N)
        num = t_ic @ X.T

        # ACE formula
        ace = num / (denom_t[:, None] * denom_x[None, :] + 1e-12)

        # Restore spatial layout
        score_map[row_begin:row_end] = ace.T.reshape(
            row_end - row_begin,
            cols,
            M
        )

    return score_map
