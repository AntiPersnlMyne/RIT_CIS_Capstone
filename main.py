#!/usr/bin/env python3

"""
Filename: main.py
Author: Gian-Mateo (Mateo) Tifone

Iteratively goes through datacubes,
allows user to select targets and background points,
and saves results to hardcoded destination.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from utils.automation import (
    import_datacube,
    get_spectral_lib,
    eda,
    detector_processing,
)

######################## USER PARAMETRS ########################
# Data paths
datacube_path = "data/datacubes/177r-172v.npy"
spectra_dir = "spectra/"
# data_path = "data/raw_data/"
algorithm_out_dir = "results/score_maps"
spectra_and_coordinate_out_dir = "results/spectra"
statistics_out_dir = "results/statistics"

# Average target signatures
average_targets = True

# Throughput
chunk_size = 500

# Crop bounds - Archimedes
row_bounds = (200, 700) 
col_bounds = (400, 1150)
################################################################

datacube, datacube_name = import_datacube(
    datacube_path,
    datacube_out_dir=None,
    row_bounds=row_bounds,
    col_bounds=col_bounds,
)

plt.imshow(datacube[:,:,:3])
plt.show()
