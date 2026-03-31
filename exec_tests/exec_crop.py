#!/usr/bin/env python3

"""
Filename: exec_crop.py
Author: Gian-Mateo (Mateo) Tifone

Version: 1.0
Description:

Simple script to crop images using set bounds,
to analyze consistent ROIs, and ensure LaTeX figures are
equally created (not eyeballed).

Set the crop bounds inside the "Parameter(s)" section. I'm too tired
to add a CLI.

Saves two cropped version: TIFF (raster, lossless) and EPS (vector).

-------
Example
-------
# Just run the script, no parameters required
python exec_tests/exec_crop.py
"""

# Mainstream packages
import matplotlib.pyplot as plt
from tifffile import imread, imwrite

# ------- Parameter(s) -----------

datacube_name = "102r-98v"
test_name = "test7"

# Image to be cropped (src)
uncropped_image = f"results/figures/{datacube_name}/pseudocolors/{test_name}.tiff"

out_base = f"results/figures/{datacube_name}/cropped/{test_name}"
out_tiff = f"{out_base}.tiff" # Saves TIFF
out_eps = f"{out_base}.eps"   # Saves EPS

# Pixels removed off of sides of image
pixels_off_the_top = 4300
pixels_off_the_bottom = 5000
pixels_off_the_left = 700
pixels_off_the_right = 3650
# ------------------------------


if __name__ == "__main__":
    # Read in image
    uncropped_image = imread(uncropped_image)

    # Crop image
    cropped_image = uncropped_image[
        pixels_off_the_top:-pixels_off_the_bottom,  # vertical crop
        pixels_off_the_left:-pixels_off_the_right,  # horizontal crop
    ]
    
    # Save
    print(f"Saving cropped image(s) to: '{out_base}'")
    
    # TIFF
    imwrite(out_tiff, cropped_image)    
    
    # EPS
    fig, ax = plt.subplots()
    ax.imshow(cropped_image / 65535.0) 
    ax.axis('off')  # no ticks, no frame

    plt.savefig(
        out_eps,
        format="eps",
        bbox_inches='tight',  # removes outer whitespace
        pad_inches=0          # no padding at all
    )
    plt.savefig(out_eps, format="eps")  
