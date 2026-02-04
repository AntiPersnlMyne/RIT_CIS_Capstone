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
# Input paths
data_path = "data/datacubes/75r-78v.npy"
spectra_lib_path = "spectra/75r-78v.npz"

# Output paths
datacube_out_dir = None
algorithm_out_dir = "results/score_maps"
spectra_and_coordinate_out_dir = "results/spectra"
statistics_out_dir = "results/statistics"

# Algorithm behavior
average_targets = True
save_corr_plot = True

# Throughput
chunk_size = 500

# Crop bounds - Archimedes
row_bounds = (200, 700)
col_bounds = (400, 1150)
################################################################

# Load datacube
datacube, datacube_name = import_datacube(
    data_path,
    datacube_out_dir=datacube_out_dir,
    row_bounds=row_bounds,
    col_bounds=col_bounds,
)

# Get spectra for targets and backgrounds
t_coords, t_spectra, b_coords, b_spectra = get_spectral_lib(
    spectral_lib_path=spectra_lib_path,
    datacube=datacube,
    average_targets=average_targets,
)

eda(
    datacube=datacube,
    stats_out_dir=statistics_out_dir,
    datacube_name=datacube_name,
    show_corr_plot=not save_corr_plot,
)
