#!/usr/bin/env python3

"""
Filename: test_algorithms.py
Author: Gian-Mateo T.
License: GPL-2.0
Version: 1.0
Description: Runs alrogithm from algorithms folder
    - Imports datacube
    - Runs algorithm
    - Display its score maps

--------
Examples
--------

# Run SAM, display results
python exec_tests/test_algorithms.py -a sam -i data/datacubes/93r_92v.npy -t results/arch_test.npz -d

# Run GOSP with HIGH chunked processing for faster throughput, save results
python exec_tests/test_algorithms.py -a gosp -i path/to/<datacube>.npy -b high -o results/
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

# enumeration for chunk sizes in rows (R)
batch_enum = {
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
    options = "hi:t:o:a:b:Od"
    long_options = [
        "help",
        "input_path=",
        "target_path=",
        "output_path=",
        "algorithm=",
        "batch=",
        "osp_multi",
        "display",
    ]

    # CLI Variables
    datacube_dir, targets_dir, out_dir = None, None, None
    algorithm, targets = None, None
    chunk_size = batch_enum["default"]
    target_split = True
    display = False

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
                  -d, --display             display results as matplotlib figure
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

            elif key in ("-d", "--display"):
                # chunk = batch
                display = True

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
        _, target_members, _, background_members = load_spectra(targets_dir)

    # ------------------------------
    # ACE
    # ------------------------------
    if algorithm == "ace":
        # calculate
        score_map = ace(datacube, target_members, chunk_size=chunk_size)
        # display
        if display:
            display_score_map(score_map)
        # save
        if out_dir:
            save_score_map(score_map, out_dir)

    # ------------------------------
    # SAM
    # ------------------------------
    elif algorithm == "sam":
        # calculate
        score_map = sam(datacube, target_members, chunk_size=chunk_size)
        # display
        if display:
            display_score_map(score_map)
        # save
        if out_dir:
            save_score_map(score_map, out_dir)

    # ------------------------------
    # OSP
    # ------------------------------
    elif algorithm == "osp":

        # Individual list of spectra
        if target_split:
            # calculate
            score_map = batch_osp(
                datacube, target_members, background_members, chunk_size=chunk_size
            )
            # display
            if display:
                display_score_map(score_map)
            # save
            if out_dir:
                save_score_map(score_map, out_dir)

        # Singular, multi-target space for OSP
        else:
            # calculate
            score_map = osp(
                datacube, target_members, background_members, chunk_size=chunk_size
            )
            # display
            if display:
                display_score_map(score_map)
            # save
            if out_dir:
                save_score_map(score_map, out_dir)

    # ------------------------------
    # GOSP
    # ------------------------------
    elif algorithm == "gosp":
        # calculate
        score_map = gosp(datacube, chunk_size=chunk_size)
        # display
        if display:
            display_score_map(score_map)
        # save
        if out_dir:
            save_score_map(score_map, out_dir)

    elif algorithm == "pca":
        # calculate
        pc_image = pca(datacube)
        # display
        if display:
            display_score_map(pc_image)
        # save
        if out_dir:
            save_score_map(pc_image, out_dir)

    else:
        raise Exception("Choose valid algorithm: ace, sam, gosp, osp, pca")
