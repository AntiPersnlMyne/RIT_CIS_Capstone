import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from tqdm import tqdm
from pathlib import Path
import time
import math
from pprint import pprint
import spectral

"""Future purpose for main: Automatic reading in of HYPERDOC cubes -> feeding to gui -> feeding to all algorithms -> display outputs & save"""

# datacube = np.lib.format.open_memmap(
#     "data/datacubes/93r_92v.npy.npy",
#     mode="w+",
#     dtype=np.float32,
#     shape=(10000, 8160, 66),
# )

# test_target = np.random.random((2, 11))

import numpy as np
import matplotlib.pyplot as plt
import dask.array as da

from numpy.linalg import svd
from scipy.linalg import subspace_angles

# -------------------------------
# USER PARAMETERS
# -------------------------------

MMAP_PATH = "data/datacubes/79v_74r.npy"     # <-- replace
DTYPE = np.float32
RANDOMIZED_N_ITER = 10             # increase for higher accuracy
DASK_CHUNK_SIZE = 100_000
# DASK_CHUNK_SIZE = "8000 MiB"

# -------------------------------
# LOAD MEMORY-MAPPED DATA
# -------------------------------

X = np.lib.format.open_memmap(
    MMAP_PATH,
    mode="r",
    dtype=np.float64,
)

r, c, b = X.shape
X = np.reshape(X, (r*c, b))

# Mean-center (important!)
X_mean = X.mean(axis=0)
Xc = X - X_mean

# -------------------------------
# 1️⃣ TRUE FULL SVD (REFERENCE)
# -------------------------------

# print("Computing full SVD...")
# U_ref, S_ref, Vt_ref = svd(Xc, full_matrices=False)

# -------------------------------
# 3️⃣ DASK OUT-OF-CORE SVD
# -------------------------------

print("Computing Dask SVD...")
Xd = da.from_array(Xc, chunks=DASK_CHUNK_SIZE)
Ud, Sd, Vtd = da.linalg.svd(Xd)

U_dask = Ud[:, :].compute()
S_dask = Sd[:].compute()
Vt_dask = Vtd[:].compute()

# -------------------------------
# PLOTTING
# -------------------------------

plt.figure(figsize=(10, 6))
plt.semilogy(S_dask, ":", label="Dask SVD")
plt.title("Singular Value Spectrum")
plt.xlabel("Component")
plt.ylabel("Singular value (log scale)")
plt.legend()
plt.tight_layout()
plt.show()

# Explained variance
var_dask = S_dask**2 / np.sum(S_dask**2)

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(var_dask), ":", label="Dask SVD")
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.legend()
plt.tight_layout()
plt.show()

