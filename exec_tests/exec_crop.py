#!/usr/bin/env python3

"""
Filename: test_crop.py
Author: Gian-Mateo (Mateo) Tifone
Description:
Simple script to consistently crop images using set bounds,
to analyze consistent ROIs, and ensure LaTeX figures are
equally created (not eyeballed).

Set the crop bounds inside the "Parameter(s)" section. I'm too lazy
to add a CLI.

Shows the original image (left) and cropped version (right). When
the figure is closed, saves cropped verison automatically.

Always saves cropped image as a TIFF.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tifffile import imread, imwrite
from os import makedirs

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "02-28-2026"
__email__ = "mt9485@rit.edu"

# ------- Parameter(s) -----------
uncropped_image = "results/score_maps/102r-98v_nouv/Test6/sam-0.tiff"
out_dir = "results/figures/102r-98v/Test6"

# Pixels removed off of sides of image
pixels_off_the_top = 4300
pixels_off_the_bottom = 3300
pixels_off_the_left = 500
pixels_off_the_right = 3500
# ------------------------------


if __name__ == "__main__":
    
    # Save cropped and uncropped image with same name
    file_name = Path(uncropped_image).parts[-1]
    
    # Create output path if doesn't exist
    makedirs(out_dir, exist_ok=True)

    # Two plots, side-by-side
    # Left plot: original
    # Right plot: cropped
    fig, ax = plt.subplots(nrows=1, ncols=2)

    # Read in image
    uncropped_image = imread(uncropped_image)

    # Crop image
    cropped_image = uncropped_image[
        pixels_off_the_top:-pixels_off_the_bottom,  # vertical crop
        pixels_off_the_left:-pixels_off_the_right,  # horizontal crop
    ]

    # Append filename and extension to make the file path
    out_path = Path(out_dir, file_name).with_suffix(".tiff")
    
    # Crop and save bounds
    print(f"Saving cropped image to: '{out_path}'")
    imwrite(out_path, cropped_image)
