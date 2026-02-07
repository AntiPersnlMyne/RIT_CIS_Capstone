"""
Filename: pca.py
Description: Returns first principal component (PC) band
"""

import numpy as np

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "01-20-2025"
__email__ = "mt9485@rit.edu"


def pca(
    datacube: np.memmap,
    n_components: int | None = 1,
    chunk_size: int = 128,
) -> np.ndarray:
    """
    Computes PCA using covariance eigendecomposition.

    Args:
        datacube (np.memmap):
            3D image cube shape (R, C, B).
        n_components (int or None, optional):
            Number of principal components (N) to return, in descending order of
            explained variance i.e. PC1 is most variance. None returns all PCs.
            Defaults to 1.
        chunk_size (int, optional):
            Number of rows loaded into RAM.
            Increase for throughput,
            Decrease for less RAM usage.
            Defaults to 128.

    Returns:
        np.ndarray: Forward-transformed image data on principal components. Shape (R, C, N)
    """

    assert n_components > 0, "n_components must be greater than 0"
    assert (
        n_components <= datacube.shape[2]
    ), "n_components cannot exceed number of bands"
    
    # Flatten data
    R, C, B = datacube.shape
    n_pixels = R * C
    
    # Set returned components to all if None type specified
    n_components = B if not n_components else n_components
    
    # ------------------------------------------------------------
    # Means
    # ------------------------------------------------------------

    # broadcasting
    try:
        means = np.mean(datacube, axis=(0, 1))

    # chunked
    except:
        means = np.zeros(B, dtype=np.float64)

        for row_start in range(0, R, chunk_size):

            row_end = min(row_start + chunk_size, R)

            block = datacube[row_start:row_end]

            X = block.reshape(-1, B)

            mean += X.sum(axis=0)

        means /= n_pixels

    # ------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------

    cov = np.zeros((B, B), dtype=np.float64)

    for row_begin in range(0, R, chunk_size):

        row_end = min(row_begin + chunk_size, R)

        block = datacube[row_begin:row_end]

        X = block.reshape(-1, B) - means

        cov += X.T @ X

    cov /= n_pixels - 1

    # ------------------------------------------------------------
    # Eigen Decomposition
    # ------------------------------------------------------------

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select components
    V = eigvecs[:, :n_components]

    # ------------------------------------------------------------
    # Orthogonal Projection
    # ------------------------------------------------------------

    pc_image = np.empty((R, C, n_components), dtype=np.float64)

    for row_begin in range(0, R, chunk_size):

        row_end = min(row_begin + chunk_size, R)

        block = datacube[row_begin:row_end]

        X = block.reshape(-1, B) - means

        # PCA scores
        Z = X @ V

        pc_image[row_begin:row_end] = Z.reshape(row_end - row_begin, C, n_components)

    return pc_image
