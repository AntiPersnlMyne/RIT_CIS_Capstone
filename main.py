#!/usr/bin/env python3

"""
Filename: main.py
Author: Gian-Mateo (Mateo) Tifone

Process:
1) Loads data as either pre-existing datacube object (.npy) or
   from TIFF/H5 files via a passed directory
2) Loads pre-compute spectral library for that datacube otherwise
   runs selection GUI to select points to create a spectral library
3) Performs EDA on the datacube object
4) Processes datacube and spectral library on the 5 detectors of my
   study - SAM, OSP, GOSP, ACE, PCA

The USER PARAMETERS section allows for fine-tuning

NOTE: directories are folders, paths include the filename

=====Inputs=====
- data_name: Datacube name, existing or to-be created
- data_path: Datacube or image directory
- spectral_lib_path: Existing library or where to create new library

===Outputs===
- datacube_out_dir: If loading image files from data_path, must specify
                    where to save resulting datacube object
- detector_out_dir: Directory to save detector image results
- statistics_out_dir: Directory to save EDA results


=====Algorithm Behavior=====
- average_targets: If True, all targets averaged into one target signal. False, processed individually.
- save_corr_plot: If True, saves rather than displays band correlation plot to statistics_out_dir.


=====Throughput=====
- chunk_size: This program is designed to be safe for low-end hardware.
              Smaller chunks reduce RAM usage, larger executes the program faster.

=====Crop Bounds=====
- row/col_bounds: To crop existing datacube, provide percentage (0,1] or pixel number (e.g., 400)

=====Detector Arguments=====
- n_components: Number of principal components for PCA to return
- max_targets: Maximum number of targets for GOSP can find before returning. None = infinity.
- opci_threshold: Purity index for GOSP. Values closer to 1 are "pure" aka. unique (fewer returned targets),
                  values smaller than 1 are "unpure" aka. similar (more returned targets)
"""

from utils.automation import (
    import_datacube,
    get_spectral_lib,
    eda,
    detector_processing,
    get_coordinates,
)

################################################################
######################## USER PARAMETRS ########################
################################################################
# Datacube name
data_name = "177r-172v"

# Input paths
data_path = f"data/datacubes/{data_name}.npy" 
spectral_lib_path = f"spectral_library/spectra_{data_name}.npz"

# Output directories
datacube_out_dir = "data/datacubes"
detector_out_dir = f"results/score_maps/{data_name}"
statistics_out_dir = f"results/statistics/{data_name}"

# Algorithm behavior
average_targets = True
save_corr_plot = True

# Throughput
chunk_size = 2500

# (upper_bound, lower_bound)
row_bounds = (200, 700)
# (left_bound, right_bound)
col_bounds = (400, 1150)

# Optional detector arguments
n_components = 8  # num PCs returned
max_targets = None  # max GOSP targets
opci_threshold = 0.7  # GOSP stopping criteria

# Optional GUI arguments
controls_font_size = 25
header_font_size = 35
################################################################
################################################################
################################################################

print("Importing datacube ...")

# Load datacube
datacube, datacube_name = import_datacube(
    source_path=data_path,
    datacube_out_dir=datacube_out_dir,
    row_bounds=row_bounds,
    col_bounds=col_bounds,
)

print("Getting spectra ...")

# Check if previous coordinates exist 
coordinates = get_coordinates(spectral_lib_path)

# Get spectra for targets and backgrounds
_, target_spectra, _, background_spectra = get_spectral_lib(
    spectral_lib_path=spectral_lib_path,
    datacube=datacube,
    average_targets=average_targets,
    coordinates=coordinates,
    # kwargs
    controls_font_size=controls_font_size,
    header_font_size=header_font_size,
)

print("Band statistics ...")

eda(
    datacube=datacube,
    stats_out_dir=statistics_out_dir,
    datacube_name=datacube_name,
    show_corr_plot=not save_corr_plot,
)

print("Detector processing ...")

detector_processing(
    datacube=datacube,
    spectra=(target_spectra, background_spectra),
    datacube_name=datacube_name,
    algorithm_out_dir=detector_out_dir,
    chunk_size=chunk_size,
    # kwargs
    n_components=n_components,
    max_targets=max_targets,
    opci_threshold=opci_threshold,
)

print("Finished!")