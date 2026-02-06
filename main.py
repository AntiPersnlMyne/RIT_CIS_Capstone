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

=====Inputs=====
- data_path: Datacube or image directory 
- spectral_lib_path: Existing library or where to create new library

---Outputs---
- datacube_out_dir: If loading image files from data_path, must specify
                    where to save resulting datacube object
- detector_out_dir: Directory to save detector image results 
- statistics_out_dir: Directory to save EDA results 


=====Algorithm Behavior=====
- average_targets: If True, all targets averaged into one target signal. False, treated individually.
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

from pathlib import Path

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
spectral_lib_path = "results/spectra/spectra_75r-78v.npz"

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
# (upper_bound, lower_bound)
row_bounds = (200, 700)
# (left_bound, right_bound)
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
    Path(spectral_lib_path).exists()
    and Path(data_path).stem[-3:] == "bgp"
    and not Path(spectral_lib_path).stem[-3:] == "bgp"
):
    print("Getting pre-existing coords ...")
    
    # Get pre-existing coordinates
    t_coords, t_spectra, b_coords, b_spectra = get_spectral_lib(
        spectral_lib_path=spectral_lib_path,
        datacube=datacube,
        average_targets=average_targets,
    )

    # Set coordinates to extract bgp spectra
    coordinates = (t_coords, b_coords)

print("Getting new coords ...")

# Get spectra for targets and backgrounds
t_coords, t_spectra, b_coords, b_spectra = get_spectral_lib(
    spectral_lib_path=spectral_lib_path,
    datacube=datacube,
    average_targets=average_targets,
    coordinates=coordinates,
)

print(t_coords)

# Zip into one variable
spectra = (t_spectra, b_spectra)  

print("EDA ...")

eda(
    datacube=datacube,
    stats_out_dir=statistics_out_dir,
    datacube_name=datacube_name,
    show_corr_plot=not save_corr_plot,
)

print("Detector ...")

detector_processing(
    datacube=datacube,
    spectra=spectra,
    datacube_name=datacube_name,
    algorithm_out_dir=detector_out_dir,
    chunk_size=chunk_size,
    kwargs=detector_kwargs,
)
