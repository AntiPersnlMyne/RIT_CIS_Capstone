import numpy as np
from pathlib import Path
import time

"""Future purpose for main: Automatic reading in of HYPERDOC cubes -> feeding to gui -> feeding to all algorithms -> display outputs & save"""

# Find all datacube filepaths
datacube_paths = Path("data/datacubes").glob("*.npy")
datacube_paths = sorted(datacube_paths)

# Iterate through all datacubes
for path in datacube_paths:
    # Load datacube
    datacube = np.lib.format.open_memmap(
        path,
        mode="r",
        dtype=np.float64,
    )
    
    
    



