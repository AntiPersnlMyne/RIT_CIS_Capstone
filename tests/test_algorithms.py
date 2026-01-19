#!/usr/bin/env python3

"""
Filename: test_algorithms.py
Description: Runs alrogithm from algorithms folder
    - Imports datacube
    - Runs algorithm
    - Display its score maps

--------
Examples
--------

# Run SAM
test/test_algorithms -a sam -i data/datacubes/archimedes_cubes/Arch_93r_92v_bgp.npy -t results/arch_test.npz

# Run OSP  with batch processing
test/test_algorithms -a osp -i data/datacubes/archimedes_cubes/Arch_93r_92v_bgp.npy -t results/arch_test.npz -O

# Run GOSP with HIGH chunked processing for faster throughput
test/test_algorithms -a gosp -i data/datacubes/archimedes_cubes/Arch_93r_92v_bgp.npy -t results/arch_test.npz -b high
"""

# Mainstream packages
import getopt
import sys
import os

# Relative package import workaround
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

# Environment packages
from utils.dataloader import load_datacube, display_score_map, save_score_map
from utils.target_selection import load_spectra
from algorithms import gosp, osp, sam, ace, batch_osp, pca

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "12-31-2025"
__email__ = "mt9485@rit.edu"

# enumeration for chunk sizes in pixels (N)
batch_enum = {
    "low": 500_000,
    "med": 2_000_000,
    "high": 10_000_000,
    "default": 4_000_000,
}

# Note: chunk_size based on rows (R), not pixels (N)
sam_batch_enum = {
    "low": 128,
    "med": 300,
    "high": 2_000,
    "default": 500,
}


if __name__ == "__main__":

    # ---------------------------------------------
    # Commandline parser
    # ---------------------------------------------

    # markers ':' '=' require value
    args = sys.argv[1:]
    options = "hi:t:o:a:b:O"
    long_options = [
        "help",
        "input_path=",
        "target_path=",
        "output_path=",
        "algorithm=",
        "batch=",
        "osp_multi",
    ]

    # CLI Variables
    datacube_dir, targets_dir, out_dir = None, None, None
    algorithm, targets = None, None
    chunk_size, chunk_size_sam = batch_enum["default"], sam_batch_enum["default"]
    target_split = True

    try:
        # Parse arguments from command line
        arguments, _ = getopt.getopt(args, options, long_options)

        # Extract (key, value) pairs
        for key, value in arguments:
            if key in ("-h", "--help"):
                print(
                    """
                Script to test each of the 4 algorithms
                1) Evaluate data on selected algorithm
                2) Displays algorithm output (score map)
                3) Save score map as image file
                
                Options:
                  -h, --help                display this message
                  -i, --input_path          input path of datacube file
                  -t, --target_path         input path of targets/background file
                  -o, --output_path         output path and filename of score map(s)
                  -a, --algorithm           which algorithm program runs
                  -b, --batch_size          (low, med, high) -> quantity pixels processed
                  -O, --osp_multi           OSP handles multi-targets as multi-output
                """
                )

                # Exit program after help message
                sys.exit()

            elif key in ("-i", "--input_path"):
                datacube_dir = value

            elif key in ("-t", "--target_path"):
                targets_dir = value

            elif key in ("-o", "--output_path"):
                out_dir = value

            elif key in ("-a", "--algorithm"):
                algorithm = value.lower()  # lowercase name

            elif key in ("-b", "--batch_size"):
                # chunk = batch
                chunk_size = batch_enum[value]
                chunk_size_sam = sam_batch_enum[value]

            elif key in ("-O", "--osp_multi"):
                target_split = (
                    True
                    if value in ("True", "true", "TRUE", "T", "t")
                    else (
                        False
                        if value in ("False", "false", "FALSE", "F", "f")
                        else True
                    )
                )

    except getopt.error as err:
        print(str(err))

    # Check for empty inputs
    assert datacube_dir, "Datacube directory cannot be empty."
    assert algorithm, "Choose an algorithm to run e.g., -a ace"

    # ------------------------------
    # Load data
    # ------------------------------

    # Datacube
    datacube = load_datacube(source_path=datacube_dir)

    # Targets and backgrounds
    if targets_dir:
        t_coords, t_members, b_coords, b_members = load_spectra(targets_dir)

    # ------------------------------
    # ACE
    # ------------------------------
    if algorithm == "ace":
        # calculate
        score_map = ace(datacube, t_members, chunk_size=chunk_size)
        # display
        display_score_map(score_map)
        # save
        if out_dir:
            save_score_map(score_map, out_dir, ".tif")

    # ------------------------------
    # SAM
    # ------------------------------
    elif algorithm == "sam":
        # calculate
        score_map = sam(datacube, t_members, chunk_size=chunk_size_sam)
        # display
        display_score_map(score_map)
        # save
        if out_dir:
            save_score_map(score_map, out_dir, ".tif")

    # ------------------------------
    # OSP
    # ------------------------------
    elif algorithm == "osp":

        # Individual list of spectra
        if target_split:
            # calculate
            score_map = batch_osp(datacube, t_members, b_members, chunk_size=chunk_size)
            # display
            display_score_map(score_map)
            # save
            if out_dir:
                save_score_map(score_map, out_dir, ".tif")

        # Singular, multi-target space for OSP
        else:
            # calculate
            score_map = osp(datacube, t_members, b_members, chunk_size=chunk_size)
            # display
            display_score_map(score_map)
            # save
            if out_dir:
                save_score_map(score_map, out_dir, ".tif")

    # ------------------------------
    # GOSP
    # ------------------------------
    elif algorithm == "gosp":
        score_map = gosp(datacube, chunk_size=chunk_size)
        display_score_map(score_map)
        
    elif algorithm == "pca":
        pc_image = pca(datacube)
        display_score_map(pc_image)

    else:
        raise Exception("Choose valid algorithm: ace, sam, gosp, osp")
