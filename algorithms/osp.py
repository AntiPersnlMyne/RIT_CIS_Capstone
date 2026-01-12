"""
Filename: osp.py
Description: Orthogonal Subspace Projection algorithm
    - Creates orthogonal subspace projector to eliminate the response of non-targets
    - Then matched filter is applied to match the desired target from the data
"""

import numpy as np
import spectral
spectral.ace

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "12-31-2025"
__email__ = "mt9485@rit.edu"


def osp(
    datacube: np.memmap,
    target_members: np.ndarray,
    background_members: np.ndarray,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """
    Performs orthogonal subspace projection (OSP) on a 3D datacube. 
    
    Multiple target members are treated as a joint target subspace, 
    consistent with the original OSP formulation (Chang 1994), and 
    will not "batch process" multiple output images.

    Args:
        datacube (np.memmap):
            3D image cube shape (rows, cols, bands)
        target_members (np.ndarray):
            Spectral members with shape (M, B) where M is the number of target members. 
        background_members (np.ndarray):
            Spectral members of shape (T, B) where T is number of background members
            defining the background subspace.
        chunk_size:
            Number of pixels processed per chunk 

    Returns:
        np.ndarray:
            2D array `(R, C)`. Output will never be `(R, C, M)` by formulation of the algorithm.
            Values normalized [0,1]; after normalization, values are no longer physically meaningful 
            i.e. scores are relative, not absolute.
    """

    rows, cols, bands = datacube.shape
    n_pixels = rows * cols
    
    # --------------------------------------------------
    # Background subspace
    # --------------------------------------------------

    # shape: (B, T)
    B = background_members.T

    # SVD gives an orthonormal basis for span(B)
    # B = U Σ Vᵀ
    U, S, Vh = np.linalg.svd(B, full_matrices=False)
    
    tol = 1e-12
    rb = np.sum(S > tol) # numerical rank
    
    if rb == 0:
        raise ValueError("Background subspace is degenerate.")

    Qb = U[:, :rb]  # (B, rb)

    # Orthogonal complement projector
    # P_B^⊥ = I − Qb Qbᵀ
    P_perp_B = np.eye(bands) - Qb @ Qb.T

    # --------------------------------------------------
    # Target subspace after background rejection
    # --------------------------------------------------
    
    # Shape to 2D array if 1 target member is passed
    if target_members.ndim == 1:
        target_members = target_members[np.newaxis, :]
    elif target_members.ndim != 2:
        raise ValueError("target_members must be 1D or 2D array")
    
    # Shape: (B, M)
    S = target_members.T

    # Remove background from target signatures
    S_perp = P_perp_B @ S
    
    # Orthonormal basis of background-rejected target subspace
    Ut, St, _ = np.linalg.svd(S_perp, full_matrices=False)

    rt = np.sum(St > tol)
    if rt == 0:
        raise ValueError("Target subspace collapses after background rejection.")

    Qt = Ut[:, :rt]

    # Target projector
    P_S = Qt @ Qt.T

    # --------------------------------------------------
    # OSP detection 
    # --------------------------------------------------
    scores = np.empty(n_pixels, dtype=np.float64)

    X = datacube.reshape(n_pixels, bands, order="C")

    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        Xc = X[start:end]

        # Chang's operation: P_S P_B^⊥ x
        Xb = Xc @ P_perp_B
        Xe = Xb @ P_S

        scores[start:end] = np.linalg.norm(Xe, axis=1)

    # --------------------------------------------------
    # Reshape and normalize 
    # --------------------------------------------------
    osp_map = scores.reshape(rows, cols)

    max_val = osp_map.max()
    min_val = osp_map.min()

    if max_val > min_val:
        osp_map = (osp_map - min_val) / (max_val - min_val)

    return osp_map


def batch_osp(datacube:np.ndarray, target_members:np.ndarray, background_members:np.ndarray, chunk_size:int):
    """Return score map (R, C, M) for (M) target_members"""
    
    # Input shape
    R, C, _ = datacube.shape
    
    # Split rows, separate targets
    M = target_members.shape[0]  # (M)
    target_members = np.split(target_members, M, axis=0)  # list[ndarray]
    
    # Remove excess dimension (1,B) -> (B)
    target_members = [np.squeeze(x) for x in target_members]

    # out buffer
    score_map = np.empty((R, C, M))

    for t_idx in range(len(target_members)):
        score_map[:, :, t_idx] = osp(
            datacube, target_members[t_idx], background_members, chunk_size
        )
        
    return score_map

