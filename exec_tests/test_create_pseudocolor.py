#!/usr/bin/env python3

"""
Filename: test_create_pseudo.py
Author: Gian-Mateo (Mateo) Tifone
Description:
Simple execution script to create a pseudocolor RGB image from
three input images. The three images are three score map TIFF's.

The convention for the paper follows:
Image1: (R,G,B) = (ACE, GOSP, SAM)
Image2: (R,G,B) = (PCA, OSP, Zeros)

Zeros: an "image" of same shape, only zero values. Since Image2
       only has 2 score maps, Zeros is palceholder for blue channel.
"""

import sys, getopt
from numpy import stack
from numpy.typing import NDArray
from pathlib import Path
from cv2 import imwrite
from tifffile import imread

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "02-28-2026"
__email__ = "mt9485@rit.edu"

import sys, getopt
from numpy import stack
from numpy.typing import NDArray
from pathlib import Path
from cv2 import imwrite
from tifffile import imread


def save_pseudocolor(
    out_path: str | Path, red_img: NDArray, green_img: NDArray, blue_img: NDArray
):
    """
    Creates and saves a pseudocolor image to `out_path`.

    Args:
        out_path (str or Path):
            Saved image destination. Must include filename and filetype (e.g. 75r-78v_Test1_Image1.png).
        red_img (NDArray):
            Score map for red channel.
        green_img (NDArray):
            Score map for green channel.
        blue_img (NDArray):
            Score map for blue channel.
    """

    # Image shapes
    rs, gs, bs = red_img.shape, green_img.shape, blue_img.shape
    assert rs == gs == bs, f"Images have unequal shapes: R:{rs} != G:{gs} != B:{bs}"

    # Stack images along channels dimension
    pseudocolor = stack([red_img, green_img, blue_img], axis=2)

    # Save/write-out images
    imwrite(out_path, pseudocolor)


if __name__ == "__main__":

    # ---------------------------------------------
    # Commandline parser
    # ---------------------------------------------

    # markers ':' '=' require value
    args = sys.argv[1:]
    options = "ho:r:g:b:"
    long_options = [
        "help",
        "out_path=" "red_image=",
        "green_image=",
        "blue_image=",
    ]

    out_path = "temp.png"
    red_image_path, green_image_path, blue_image_path = None, None, None

    try:
        # Parse arguments from command line
        arguments, _ = getopt.getopt(args, options, long_options)

        # Extract (key, value) pairs
        for key, value in arguments:
            if key in ("-h", "--help"):
                print(
                    """
                Creates RGB image from paths to input images. 
                
                Options:
                  -h, --help                display this message
                  -o, --out_path            path to save pseudocolor image
                  -r, --red_image           path to red channel image
                  -g, --green_image         path to green channel image
                  -b, --blue_image          path to blue channel image
                """
                )

                # Exit program after help message
                sys.exit()

            elif key in ("-o", "--out_path"):
                out_path = Path(value)

            elif key in ("-r", "--red_image"):
                red_image_path = Path(value)

            elif key in ("-g", "--green_image"):
                green_image_path = Path(value)

            elif key in ("-b", "--blue_image"):
                blue_image_path = Path(value)

        assert all(
            (red_image_path, green_image_path, blue_image_path)
        ), "All 3 channels required to make a pseudocolor image"

    except getopt.error as err:
        print(str(err))

    # ---------------------------------------------
    # Pseudocolor creator / saver
    # ---------------------------------------------

    # Get images from paths using TIFFFile
    red_image, green_image, blue_image = imread(
        red_image_path, green_image_path, blue_image_path
    )

    # Save images out
    save_pseudocolor(out_path, red_image, green_image, blue_image)
