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
data_name = "75r-78v"

# Input paths
data_path = f"data/datacubes/{data_name}.npy"
spectral_lib_path = f"spectral_library/spectra_{data_name}.npz"
background_coords_lib_path = f"{spectral_lib_path[:-4]}_background.npz"

# Output directories
datacube_out_dir = "data/datacubes"
detector_out_dir = f"results/score_maps/{data_name}"
statistics_out_dir = f"results/statistics/{data_name}"

# Datacube cropping parameters
row_bounds = (200, 700)  # (upper_bound, lower_bound)
col_bounds = (400, 1150)  # (left_bound, right_bound)
################################################################
################################################################
################################################################


print("Importing datacube ...")

datacube, datacube_name = import_datacube(
    source_path=data_path,
    datacube_out_dir=datacube_out_dir,
    row_bounds=row_bounds,
    col_bounds=col_bounds,
)

print("Loading spectral library...")

_, target_spectra, _, background_spectra = get_spectral_lib(
    spectral_lib_path=spectral_lib_path,
    datacube=datacube,
    average_targets=False,
    coordinates=get_coordinates(background_coords_lib_path, return_none=True),
    # kwargs
    controls_font_size=25,
    header_font_size=35,
)

print("Generating band statistics ...")

eda(
    datacube=datacube,
    stats_out_dir=statistics_out_dir,
    datacube_name=datacube_name,
    show_corr_plot=False,  # saves plot instead
)

print("Detector processing ...")

detector_processing(
    datacube=datacube,
    spectra=(target_spectra, background_spectra),
    datacube_name=datacube_name,
    algorithm_out_dir=detector_out_dir,
    chunk_size=4000,
    # kwargs
    n_components=None,  # None = all
    max_targets=None,  # None = infinity
    opci_threshold=0.00005,  # GOSP stopping criteria
)

print("Program Finished!")
