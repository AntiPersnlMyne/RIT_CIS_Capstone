#!/usr/bin/env python3

"""
Filename: test_extract_targets.py
Description: Choosing algorithm targets
- Displays pseudocolor image
- Returns coordinates and signatures of selected point(s)
- Save signatures to file

--------
Examples
--------
# Call help menu
tests/test_extract.py -h

# Define pseudocolor image, as band indices of datacube
tests/test_extract_gui.py -i data/datacubes/archimedes_cubes/Arch_93r_92v.npy -r 7 -g 4 -b 2

# Save results to file
tests/test_extract_gui.py -i data/datacubes/archimedes_cubes/Arch_93r_92v.npy -r 7 -g 4 -b 2 -o results/results.npz

# VNIR mockup from HYPERDOC
tests/test_extract_gui.py -i data/datacubes/hyperdoc_cubes/00008-VNIR-mock-up.npy -o results/hyper8_test.npz -r 60 -g 31 -b 0
"""

# Relative package import workaround
import os, sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

import getopt
from utils.target_selection import (
    extract_spectra,
    save_spectra,
    target_selection_gui,
)
from utils.dataloader import load_datacube  # use np.load if doesn't work

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "12-29-2025"
__email__ = "mt9485@rit.edu"


if __name__ == "__main__":
    # ---------------------------------------------
    # Commandline parser
    # ---------------------------------------------

    # markers ':' '=' require value
    args = sys.argv[1:]
    options = "hi:o:r:g:b:"
    long_options = [
        "help",
        "input_dir=",
        "output_dir=",
        "red=",
        "green=",
        "blue=",
    ]

    src_dir, dst_dir = None, None
    red_idx, green_idx, blue_idx = None, None, None

    try:
        # Parse arguments from command line
        arguments, _ = getopt.getopt(args, options, long_options)

        # Extract (key, value) pairs
        for key, value in arguments:
            if key in ("-h", "--help"):
                print(
                    """
                Test script to extract spectral signatures from a datacube:
                1) Displays pseudocolor image (GUI) 
                2) Click and return target and background points
                3) Optionally save spectral signatures if -o given
                
                Options:
                  -h, --help                display this message
                  -i, --input_dir           input directory of datacube file
                  -o, --output_dir          output directory of spectra
                  -r, --red                 datacube index of red band
                  -g, --green               datacube index of green band
                  -b, --blue                datacube index of blue band
                """
                )

                # Exit program after help message
                sys.exit()

            elif key in ("-i", "--input_dir"):
                src_dir = value

            elif key in ("-o", "--output_dir"):
                dst_dir = value
                print("Saving output to: ", value)

            elif key in ("-r", "--red"):
                red_idx = int(value)

            elif key in ("-g", "--green"):
                green_idx = int(value)

            elif key in ("-b", "--blue"):
                blue_idx = int(value)

    except getopt.error as err:
        print(str(err))

    # Check for empty inputs
    assert src_dir, "Input directory cannot be empty."
    assert all(
        (
            isinstance(red_idx, int),
            isinstance(green_idx, int),
            isinstance(blue_idx, int),
        )
    ), "-r,-g,-b flags cannot be empty."

    # ------------------------------------------------------------
    # Setup GUI
    # ------------------------------------------------------------

    # Load the datacube
    datacube = load_datacube(src_dir)
    print(f"Input datacube shape: {datacube.shape}")

    # Load rgb images for GUI
    red_img = datacube[:, :, red_idx]
    green_img = datacube[:, :, green_idx]
    blue_img = datacube[:, :, blue_idx]

    # ------------------------------------------------------------
    # Run GUI, extract coordinates in image
    # ------------------------------------------------------------

    # GUI to extract coordinates
    coordinates = target_selection_gui(rgb_images=[red_img, green_img, blue_img])

    # ------------------------------------------------------------
    # Extract spectral signature at each coordinate
    # ------------------------------------------------------------

    # Extract spectral signatures of datacube
    spectra = extract_spectra(coordinates=coordinates, datacube=datacube)

    # ------------------------------------------------------------
    # Save spectra and coordinates to disk
    # ------------------------------------------------------------

    if dst_dir:
        save_spectra(dst_path=dst_dir, spectra=spectra, coordinates=coordinates)
