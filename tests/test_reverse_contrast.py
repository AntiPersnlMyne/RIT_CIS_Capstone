#!/usr/bin/env python3

"""
Filename: test_reverse_contrast.py
Author: Gian-Mateo (Mateo) Tifone
Description:
Reverse the contrast of score maps. Effect is converting processed
text to always appear black.

NOTE: Overrides existing file; however calling the function twice 
undoes the operation.

--------
Examples
--------
With positional argument
tests/test_reverse_contrast.py results/score_maps/<map_name>.tiff

With keyword argument
tests/test_reverse_contrast.py -i results/score_maps/<map_name>.tiff
"""

import sys, os
import getopt
from pathlib import Path
import tifffile as tif
from numpy import invert

# Relative package import workaround
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

if __name__ == "__main__":
    # ---------------------------------------------
    # Commandline parser
    # ---------------------------------------------

    # ':' expects value for input and output files
    args = sys.argv[1:]
    options = "hi:"
    long_options = [
        "help",
        "input_file=",
    ]

    # Non-keyword argument
    scoremap_path = sys.argv[1]

    try:
        # Parse arguments from command line
        arguments, _ = getopt.getopt(args, options, long_options)

        # Extract (key, value) pairs
        for key, value in arguments:
            if key in ("-h", "--help"):
                print(
                    """
                Reverses the contrast of an image. Image dtype expected as uint16.
                
                Options:
                  -h, --help                display this message
                  -i, --input_file          score map file 
                """
                )

                # Exit program after help message
                sys.exit()

            if key in ("-i, --input"):
                scoremap_path = value

    except getopt.error as err:
        print(str(err))

    assert Path(
        scoremap_path
    ).exists(), f"Must provide valid input file path. Recieved: {scoremap_path}"

    # ---------------------------------------------
    # Invert and Save
    # ---------------------------------------------
    # Read-in image
    image = tif.imread(scoremap_path)
    assert (
        image.ndim == 2
    ), f"Image must be 2D (i.e. grayscale), got {image.ndim}D instead"

    # Invert image
    inverted_image = invert(image)

    # Save image
    tif.imwrite(scoremap_path, inverted_image)
