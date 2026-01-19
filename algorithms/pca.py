"""
Filename: pca.py
Description: Returns first principal component (PC) band
"""

import numpy as np
import dask.array as da

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "01-19-2025"
__email__ = "mt9485@rit.edu"


def pca(
    datacube: np.memmap,
    chunk_size: int = 500_000,
    n_components: int = 1,
) -> np.ndarray:
    """
    Computes PCA using dask-assisted singular value decomposition (SVD).
    Computationally equivalent to NumPy's np.linalg.svd, but memory safe (chunking)

    Args:
        datacube (np.memmap):
            3D image cube shape (R, C, B).
        chunk_size (int, optional):
            Number of pixels (not bytes) loaded into RAM. Increase for throughput,
            decrease for smaller RAM usage. Defaults to 500_000.
        n_components (int, optional):
            Number of principal components to return, in descending order of
            explained variance i.e. PC1 is most variance. Defaults to 1.

    Returns:
        np.ndarray: Forward-transformed image data on principal components. Shape (R, C, n_components).
                    If `n_components=1`, shape = (R, C).
    """
    
    assert n_components > 0, "n_components must be greater than 0"
    assert datacube is not None
    assert datacube.ndim == 3, "datacube must have shape (R,C,B)"

    # Flatten data
    R, C, B = datacube.shape
    datacube = np.reshape(datacube, (R * C, B))  # n_samples x n_features
    n_samples, n_features = datacube.shape

    # Mean center data
    try: # broadcasting
        means = np.mean(datacube, axis=0)
        datacube -= mean
        # for feature_idx in range(n_features):
        #     datacube[:, feature_idx] -= means[feature_idx]
    
    except: # chunked
        for feature_idx in range(n_features):
            mean = np.average(datacube[:, feature_idx])
            datacube[:, feature_idx] -= mean


    # Dask-assisted SVD
    dask_datacube = da.from_array(datacube, chunks=(chunk_size,B))
    U, S, Vt = da.linalg.svd(dask_datacube)

    # Extract first component
    pc_scores = (U[:, :n_components] * S[:n_components]).compute()
    
    # Reshape back to image
    pc_image = pc_scores.reshape(R, C, n_components)
    
    # Flatten (R, C, 1) to (R, C) - convenience step
    return np.squeeze(pc_image)

