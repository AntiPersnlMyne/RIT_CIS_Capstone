"""
Filename: pca.py
Description: Returns first principal component (PC) band
"""

import numpy as np
import matplotlib.pyplot as plt

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "01-19-2025"
__email__ = "mt9485@rit.edu"


def pca(
    datacube:np.memmap, 
    chunk_size:int = 1_000_000
) -> np.ndarray:
    
    pass


def pca(X, n_components=2, feature_names=None, plot=False, compare_with_sklearn = False):
    X_mean = np.mean(X, axis=0)
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered)

    components = Vt[:n_components]
    X_proj = X_centered @ components.T
    X_reconstructed = X_proj @ components + X_mean
