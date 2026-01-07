"""
Filename: sam.py
Description: Spectral Angle Mapper algorithm
    - Uses an n-D angle to match pixels to reference spectra
    - Return 2D score map (ndarray) of similarity scores [0,1]
    - 0 = no match
    - 1 = perfect match
"""

import numpy as np

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "12-31-2025"
__email__ = "mt9485@rit.edu"


def sam(
    datacube: np.memmap,
    target_members: np.ndarray,
    chunk_size: int = 128,
):
    """
    Spectral Angle Mapper. Algorithm reworked to utilize chunking,
    to accomodate large datacubes.

    Args:
        data (np.ndarray):
            3D image cube shape (R, C, B).
        target_members (np.ndarray):
            Spectral members with shape (M, B) where M is the number of members.
        chunk_size (int):
            Processing chunk size. Smaller chunk sizes consume less RAM.
            Larger chunks increase process speed.

    Returns:
        np.ndarray: 3D array `(R, C, M)` depending on number of
        targets given. SAM scores with maximum value of
        1 correspond to a perfect match (zero spectral angle).
    """

    # Assertions
    assert datacube.ndim == 3, "data must be 3D image shape (R,C,B)"
    assert target_members.ndim, "members must be 2D array (M,B)"
    assert (
        target_members.shape[1] == datacube.shape[2]
    ), f"Matrix dimensions are not aligned. ({target_members.shape[1]}) != ({datacube.shape[2]})"

    R, C, _ = datacube.shape
    M = target_members.shape[0]

    # Reference spectra norms
    ref_norms = np.linalg.norm(target_members, axis=1)  # (M,)

    # Output array
    score_map = np.empty((R, C, M), dtype=np.float32)

    # Iterate through chunks of rows
    # chunk_rows = end_row - begin_row
    for begin_row in range(0, R, chunk_size):

        # Chunk size
        end_row = min(begin_row + chunk_size, R)

        # Load a chunk
        img_chunk = datacube[begin_row:end_row]

        # Norm of test pixels
        test_norms = np.linalg.norm(img_chunk, axis=2)  # (chunk_R, C)

        # Dot products
        dot_product = np.tensordot(img_chunk, target_members, axes=([2], [1]))

        # Calculate cos(alpha)
        denominator = test_norms[..., np.newaxis] * ref_norms[np.newaxis, np.newaxis, :]
        cos_alpha = dot_product / (denominator + 1e-12)
        cos_alpha = np.divide(dot_product, denominator, where=denominator != 0)

        # Clip floating point errors (in-place)
        np.clip(cos_alpha, -1.0, 1.0, out=cos_alpha)

        # Angle transform
        score_map[begin_row:end_row] = 1.0 - np.arccos(cos_alpha) / (np.pi / 2.0)

    return score_map


if __name__ == "__main__":
    print(
        f"""
    (This file is a module, do not run directly)
    
    Spectral Angle Mapper according to Oshigami, et al.

    Args:
        datacube (np.memmap):
            3D datacube of shape (datacube.rows, datacube.cols, datacube.bands)
        target_spectra (np.ndarray):
            Target spectral signature(s), shape (n_targets, bands)

    Returns:
        np.ndarray: SAM score map, shape (datacube.rows, datacube.cols).
        Maximum value of 1 corresponds to a perfect match (zero spectral angle).
            
    Author: {__author__}
    License: {__license__}
    Contact: {__email__}
    """
    )
