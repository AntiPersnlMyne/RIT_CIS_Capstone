#!/usr/bin/env python3

"""
Filename: main.py
Author: Gian-Mateo (Mateo) Tifone

Iteratively goes through datacubes,
allows user to select targets and background points,
and saves results to hardcoded destination.
"""

# TODO: Run the stupid thing, debug it till it works
# TODO: Check algorithms to see that chunked processing isn't messing with detector logic

import numpy as np
from pathlib import Path

from utils.target_selection import (
    extract_spectra,
    save_spectra,
    target_selection_gui,
)

from utils.eda import (
    calculate_band_statistics,
    display_band_statistics,
    save_band_statistics,
    cov_matrix,
    corr_matrix,
    plot_corr_matrix,
)

from algorithms import (
    gosp,
    osp,
    sam,
    ace,
    batch_osp,
    pca,
)

from utils.dataloader import (
    save_score_map,
)

######################## USER PARAMETRS ########################
# Data paths
datacube_path = "data/datacubes/75r-78v.npy"  # most frqeuently changed
algorithm_out_dir = "results/score_maps"
spectra_and_coordinate_out_dir = "results/spectra"
statistics_out_dir = "results/statistics"
# Target averaging behavior
average_targets = True
# Throughput parameters
chunk_size = 4_000_000
chunk_size_sam = 500
# Archimedes Palimpsest
red_idx = 9
green_idx = 5
blue_idx = 0
# HYPERDOC
# red_idx = 60
# green_idx = 31
# blue_idx = 0
################################################################


# ==============================
# Import Datacube
# ==============================

# Filenames for output identifiers
datacube_name = Path(datacube_path).stem

# Load datacube
datacube = np.lib.format.open_memmap(
    datacube_path,
    mode="r",
    dtype=np.float64,
)

# ==============================
# Crop Archimedes
# ==============================
# Archimedes includes color checker chart
# Throws off algorithms, crop it out

# Crop by percent
rows, cols, _ = datacube.shape
row_start = int(0.02 * rows)
row_end = int(0.91 * rows)
col_start = int(0.05 * cols)
col_end = int(0.88 * cols)

datacube = datacube[row_start:row_end, col_start:col_end, :]

# ==============================
# Target Extraction GUI
# ==============================

# Load rgb images for GUI
red_img = datacube[:, :, red_idx]
green_img = datacube[:, :, green_idx]
blue_img = datacube[:, :, blue_idx]

# GUI to extract coordinates
coordinates = target_selection_gui(rgb_images=[red_img, green_img, blue_img])

# Extract spectral signatures of datacube
spectra = extract_spectra(coordinates=coordinates, datacube=datacube)
target_members, background_members = spectra

# Average target spectra to one endmember
if average_targets:
    # shape (M,B) -> (1, B)
    target_members = np.average(target_members, axis=0, keepdims=True)

# Save spectra and coordinates for reproducibility
save_spectra(
    dst_path=f"{spectra_and_coordinate_out_dir}/spectra_{datacube_name}",
    spectra=spectra,
    coordinates=coordinates,
)

# ==============================
# Datacube EDA
# ==============================

# Band statistics for entire datacube
statistics = calculate_band_statistics(datacube=datacube)
save_band_statistics(
    statistics=statistics,
    dst_path=f"{statistics_out_dir}/stats_{datacube_name}",
)

# Covaraince
cov_mat = cov_matrix(datacube=datacube)

# Correlation
corr_mat = corr_matrix(cov_matrix=cov_mat)
del cov_mat

# ==============================
# Detector Processing
# ==============================

# ACE
score_map = ace(datacube, target_members, chunk_size=chunk_size)
save_score_map(score_map, f"{algorithm_out_dir}/{datacube_name}_ace.tiff")

# SAM
score_map = sam(datacube, target_members, chunk_size=chunk_size_sam)
save_score_map(score_map, f"{algorithm_out_dir}/{datacube_name}_sam.tiff")

# OSP - targets are combined
score_map = osp(datacube, target_members, background_members, chunk_size=chunk_size)
save_score_map(score_map, f"{algorithm_out_dir}/{datacube_name}_osp.tiff")

# GOSP
score_map = gosp(datacube, chunk_size=chunk_size)
save_score_map(score_map, f"{algorithm_out_dir}/{datacube_name}_gosp.tiff")

# PCA
score_map = pca(datacube)
save_score_map(score_map, f"{algorithm_out_dir}/{datacube_name}_pca.tiff")

# ==============================
# Load BGP Datacube
# ==============================

bgp_datacube_path = "data/datacubes_bgp/" + datacube_name + "_bgp.npy"
bgp_datacube_name = datacube_name + "_bgp"
del datacube # Close old

# bgp_datacube = np.lib.format.open_memmap(
#     bgp_datacube_path,
#     mode="r",
#     dtype=np.float64,
# )

bgp_datacube = np.random.random_sample((100,120,15)) 

# ==============================
# Load New Spectra at Old Coordinates
# ==============================

# Extract spectral signatures of datacube
spectra = extract_spectra(coordinates=coordinates, datacube=bgp_datacube)
target_members, background_members = spectra

# Average target spectra to one endmember
if average_targets:
    # shape (M,B) -> (1, B)
    target_members = np.average(target_members, axis=0, keepdims=True)

# Save spectra and coordinates for reproducibility
save_spectra(
    dst_path=f"{spectra_and_coordinate_out_dir}/spectra_{bgp_datacube_name}",
    spectra=spectra,
    coordinates=coordinates,
)

# ==============================
# Detector Processing
# ==============================

# (BGP) ACE
score_map = ace(bgp_datacube, target_members, chunk_size=chunk_size)
save_score_map(score_map, f"{algorithm_out_dir}/{bgp_datacube_name}_ace.tiff")

# (BGP) SAM
score_map = sam(bgp_datacube, target_members, chunk_size=chunk_size_sam)
save_score_map(score_map, f"{algorithm_out_dir}/{bgp_datacube_name}_sam.tiff")

# (BGP) OSP
score_map = osp(bgp_datacube, target_members, background_members, chunk_size=chunk_size)
save_score_map(score_map, f"{algorithm_out_dir}/{bgp_datacube_name}_osp.tiff")

# (BGP) GOSP
score_map = gosp(bgp_datacube, chunk_size=chunk_size)
save_score_map(score_map, f"{algorithm_out_dir}/{bgp_datacube_name}_gosp.tiff")

# (BGP) PCA
score_map = pca(bgp_datacube)
save_score_map(score_map, f"{algorithm_out_dir}/{bgp_datacube_name}_pca.tiff")

# ==============================
# Datacube EDA
# ==============================

# Band statistics for entire datacube
bgp_statistics = calculate_band_statistics(datacube=bgp_datacube)
save_band_statistics(
    statistics=statistics,
    dst_path=f"{statistics_out_dir}/stats_{bgp_datacube_name}",
)

# Covaraince
cov_mat = cov_matrix(datacube=bgp_datacube)

# Correlation
bgp_corr_mat = corr_matrix(cov_matrix=cov_mat)

# ==============================
# Results Statistics
# ==============================

# Display correlation matrix
plot_corr_matrix(corr_matrix=corr_mat, title=f"{bgp_datacube_name} Correlation Matrix")
plot_corr_matrix(corr_matrix=corr_mat, title=f"{bgp_corr_mat} Correlation Matrix")

# Display stats as HTML
display_band_statistics(
    statistics=statistics,
    highlight_max=True,
    highlight_min=True,
)

# # ACE
# display_score_map(score_map_ace, "ACE Score")

# # SAM
# display_score_map(score_map_sam, "SAM Score")

# # OSP
# display_score_map(score_map_osp, "OSP Score")

# # Batch OSP
# display_score_map(score_map_bosp, "batch-OSP Score")

# # GOSP
# display_score_map(score_map_gosp, "GOSP Score")

# # PCA
# display_score_map(score_map_pca, "PCA Score")

# Close all matplotlib figures
# close("all")
