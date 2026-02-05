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

################################################################
######################## USER PARAMETRS ########################
################################################################
# Input paths
data_path = "data/datacubes/75r-78v.npy"
spectra_lib_path = "spectra/spectra_75r-78v.npz"

# Output paths
datacube_out_dir = None
detector_out_dir = "results/score_maps"
statistics_out_dir = "results/statistics"

# Algorithm behavior
average_targets = True
save_corr_plot = True

# Throughput
chunk_size = 500

# Crop bounds - Archimedes
row_bounds = (200, 700)
col_bounds = (400, 1150)

# Optional detector arguments
detector_kwargs = {"n_components": 1, "max_targets": None, "opci_threshold": 0.7}
################################################################
################################################################
################################################################

# Load datacube
datacube, datacube_name = import_datacube(
    data_path,
    datacube_out_dir=datacube_out_dir,
    row_bounds=row_bounds,
    col_bounds=col_bounds,
)

# Change behavior if datacube is a bgp datacube
coordinates = None  # Default; previous coordinates do not exist

# If a spectral library exist AND
# Is a BGP datacube AND
# A BGP spectral library does not already exist
if (
    Path(spectra_lib_path).exists()
    and data_path.stem[-3:] == "bgp"
    and not Path(spectra_lib_path).stem[-3:] == "bgp"
):
    # Get pre-existing coordinates
    t_coords, t_spectra, b_coords, b_spectra = get_spectral_lib(
        spectral_lib_path=spectra_lib_path,
        datacube=datacube,
        average_targets=average_targets,
    )

    # Set coordinates to extract bgp spectra
    coordinates = (t_coords, b_coords)

# Get spectra for targets and backgrounds
t_coords, t_spectra, b_coords, b_spectra = get_spectral_lib(
    spectral_lib_path=spectra_lib_path,
    datacube=datacube,
    average_targets=average_targets,
    coordinates=coordinates,
)

# Zip into one variable
spectra = (t_spectra, b_spectra)  

eda(
    datacube=datacube,
    stats_out_dir=statistics_out_dir,
    datacube_name=datacube_name,
    show_corr_plot=not save_corr_plot,
)

detector_processing(
    datacube=datacube,
    spectra=spectra,
    datacube_name=datacube_name,
    algorithm_out_dir=detector_out_dir,
    chunk_size=chunk_size,
    kwargs=detector_kwargs,
)
