#!/usr/bin/env python3

"""
Filename: test_eda.py
Description: Statistical properties of datacube bands
    - Calculate and display per-band statistics, outputs console
    - Covariance matrix
    - Calculate and plot correlation matrix
    
--------
Examples
--------

Call the help menu
tests/test_eda.py -h

Plot statistics of f/93v-92r
tests/test_eda.py -i data/datacubes/archimedes_cubes/Arch_93v_92r.npy

"""

# Relative package import workaround
import os, sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

import getopt
from utils.dataloader import load_datacube
from utils.eda import (
    calculate_band_statistics,
    display_band_statistics,
    cov_matrix,
    corr_matrix,
    plot_corr_matrix,
)

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "01-08-2026"
__email__ = "mt9485@rit.edu"


if __name__=="__main__":

    # ---------------------------------------------
    # Commandline parser
    # ---------------------------------------------

    # markers ':' '=' require value
    args = sys.argv[1:]
    options = "hi:"
    long_options = [
        "help",
        "input_dir=",
    ]

    src_dir = None

    try:
        # Parse arguments from command line
        arguments, _ = getopt.getopt(args, options, long_options)

        # Extract (key, value) pairs
        for key, value in arguments:
            if key in ("-h", "--help"):
                print(
                    """
                Test script to calculate statistical analytics:
                1) Calculate mean, std, kurtosis, etc.
                2) Display statistics as DataFrame (table)
                3) Saves DataFrame as HTML in proj. root
                4) Plot correlation matrix 
                
                Note: Statistics auto-saved to root, matrix saved via Matplotlib window.
                
                Options:
                  -h, --help                display this message
                  -i, --input_dir           input directory of datacube file
                """
                )

                # Exit program after help message
                sys.exit()

            elif key in ("-i", "--input_dir"):
                src_dir = value

    except getopt.error as err:
        print(str(err))

    # Check for empty inputs
    assert src_dir, "Input directory cannot be empty."

# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

datacube = load_datacube(source_path=src_dir)

# ------------------------------------------------------------
# Calculate and display statistics
# ------------------------------------------------------------

# Band statistics for entire datacube
statistics = calculate_band_statistics(datacube=datacube)

# Display stats as HTML
display_band_statistics(statistics=statistics, highlight_max=True, highlight_min=True)

# ------------------------------------------------------------
# Cov and Corr matrices
# ------------------------------------------------------------

# Covaraince
cov_mat = cov_matrix(datacube=datacube)

# Correlation
corr_mat = corr_matrix(cov_matrix=cov_mat)

# Display correlation matrix to matplotlib figure
plot_corr_matrix(corr_matrix=corr_mat)
