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

Always saves cropped image as a PNG.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from cv2 import imread, imwrite, IMREAD_COLOR_RGB
from numpy import stack

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "02-28-2026"
__email__ = "mt9485@rit.edu"

# ------- Parameter(s) -----------
uncropped_image = "data/raw_data/102r-98v/102r-098v_Arch39r_Sinar_LED365_01_pack8.tif"
out_path = "results/figures/102r-98v_gt"

# Pixels removed off of sides of image
pixels_off_the_top = 4500
pixels_off_the_bottom = 5200
pixels_off_the_left = 800
pixels_off_the_right = 4800
# ------------------------------


if __name__ == "__main__":

    # Two plots, side-by-side
    # Left plot: original
    # Right plot: cropped
    fig, ax = plt.subplots(nrows=1, ncols=2)

    # Read in image
    uncropped_image = imread(uncropped_image, IMREAD_COLOR_RGB)

    # Crop image
    cropped_image = uncropped_image[
        pixels_off_the_top:-pixels_off_the_bottom,  # vertical crop
        pixels_off_the_left:-pixels_off_the_right,  # horizontal crop
    ]

    # Add images to figure
    ax[0].imshow(uncropped_image)
    ax[0].set_title("Original")
    ax[1].imshow(cropped_image)
    ax[0].set_title("Cropped")

    # Plot details
    plt.show()

    # Reverse RGB -> BGR for OpenCV
    cropped_image = stack(
        [cropped_image[..., 2], cropped_image[..., 1], cropped_image[..., 0]], axis=2
    )

    # Crop and save bounds
    print(f"Saving cropped image to: '{out_path}.png'")
    imwrite(Path(out_path).with_suffix(".png"), cropped_image)
