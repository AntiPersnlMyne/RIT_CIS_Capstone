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

datacube = np.lib.format.open_memmap(
    "data/datacubes/archimedes_cubes/Arch_93r_92v.npy.npy",
    mode="w+",
    dtype=np.float32,
    shape=(10000, 8160, 66),
)

test_target = np.random.random((2, 11))
