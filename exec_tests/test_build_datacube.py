#!/usr/bin/env python3

"""
Filename: test_build_datacube.py
Author: Gian-Mateo T.

Version: 1.0
Description: Builds datacubes from raw data
- Creates datacube from directory of Arhimedes TIFF files
- Create datacube from individual H5 file

--------
Examples
--------

# Build a datacube from f/93v-92r directory, specify output directory with filename, time execution
python exec_tests/test_build_datacube.py -i data/raw_data/93v_92r/ -o data/datacubes/93v_92r.npy -t

# Build datacube as float64, normalizing output
python exec_tests/test_build_datacube.py -i dir/to/TIFFs -o path/to/<datacube>.npy -d float64 -n
"""

# Mainstream packages
import sys
import os
import getopt
import numpy as np
from pathlib import Path

# Relative package import workaround
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

# Environment packages
from utils.dataloader import load_datacube


def _first_file_extension(directory) -> tuple[str, bool]:
    """
    Returns the suffix of the first file to determine load_datacube behavior.

    Boolean returns True if there is one file

    Args:
        directory (str): Path to input file directory

    Returns:
        tuple[str,bool]: suffix of first file, boolean if only one file in directory
    """
    directory_path = Path(directory)

    # Use generator to avoid iterating over all files
    dir_generator = (entry for entry in directory_path.iterdir() if entry.is_file())
    first_file = next(dir_generator, None)

    # False if fails to read next file in directory
    is_multi_file = True if next(dir_generator, None) else False

    suffix = str(first_file.suffix)

    return suffix, is_multi_file


if __name__ == "__main__":
    # ---------------------------------------------
    # Commandline parser
    # ---------------------------------------------

    # markers ':' '=' require value
    args = sys.argv[1:]
    options = "hi:o:nd:c:"
    long_options = [
        "help",
        "input_dir=",
        "output_dir=",
        "normalize",
        "dtype=",
        "cache_max=",
    ]

    in_dir, out_dir, dtype = None, None, None
    normalize = False  # No normalization
    cache_max = 1024  # 1 GB

    try:
        # Parse arguments from command line
        arguments, _ = getopt.getopt(args, options, long_options)

        # Extract (key, value) pairs
        for key, value in arguments:
            if key in ("-h", "--help"):
                print(
                    """
                Test script to build a datacube from:
                1) Directory of single-band image files
                2) A single multiband image file
                
                A datacube is a memory safe, NumPy array data file (.npy)
                
                Supported file types are TIFF (.tif, .tiff) and H5 (.h5)
                
                Options:
                  -h, --help                display this message
                  -i, --input_dir           input directory of image file(s)
                  -o, --output_dir          output directory of image file
                  -n, --normalize           normalize output to be [0,1]
                  -d, --dtype               output floating-point datatype e.g. float32
                  -c, --cache_max           memory in megabytes program is allowed to utilize
                """
                )

                # Exit program after help message
                sys.exit()

            elif key in ("-i", "--input_dir"):
                in_dir = value

            elif key in ("-o", "--output_dir"):
                out_dir = value
                print("Saving output to: ", value)

            elif key in ("-n", "--normalize"):
                # Apply normalization, otherwise keep data raw (default)
                normalize = "minmax"

            elif key in ("-d", "--dtype"):
                dtype = str(value)

            elif key in ("-c", "--cache_max"):
                cache_max = int(cache_max)

    except getopt.error as err:
        print(str(err))

    assert in_dir, "Input directory cannot be empty."

    # ---------------------------------------------
    # Convert dtype arg into datatype object
    # ---------------------------------------------

    match dtype:
        case "float16":
            dtype = np.float16
        case "float32":
            dtype = np.float32
        case "f4":
            dtype = np.float32
        case "float":
            dtype = np.float32
        case "float64":
            dtype = np.float64
        case "double":
            dtype = np.float64
        case "f8":
            dtype = np.float64
        case _:  # Default
            print("warning: only floating-point dtype accepted. defaulting to float32.")
            dtype = np.float32

    # ---------------------------------------------
    # Construct path to data
    # ---------------------------------------------
    # Join base path with user path
    # NOTE: This assumes test file is run in project directory
    #       and path provided is not absolute path
    data_path = os.path.join(base_dir, in_dir)
    out_path = os.path.join(base_dir, out_dir)

    # ---------------------------------------------
    # Determine case criteria
    # ---------------------------------------------

    extension, is_multi_file = _first_file_extension(data_path)

    # ---------------------------------------------
    # Case 1: Build datacube from TIFFs
    # ---------------------------------------------
    if is_multi_file and extension in (".tif", ".tiff"):

        # Saves datacube (.npy) for future loading
        datacube = load_datacube(
            source_path=data_path,
            output_path=out_path,
            dtype=dtype,
            normalize=normalize,
            cachemax_mb=cache_max,
        )

    # ---------------------------------------------
    # Case 2: Build datacubes from H5 files
    # ---------------------------------------------
    elif is_multi_file and extension in (".h5", "hdf5"):

        h5_paths = Path(data_path).glob("*.h5")

        for src_path in h5_paths:
            # Get name of h5 file
            basename = src_path.stem

            load_datacube(
                source_path=src_path,
                output_path=os.path.join(out_path, basename + ".npy"),
                dtype=dtype,
                normalize=normalize,
                cachemax_mb=cache_max,
            )

    # ---------------------------------------------
    # Case 3: Single multiband TIFF file
    # ---------------------------------------------
    elif extension in (".tiff", ".tif"):

        # Saves datacube (.npy) for future loading
        datacube = load_datacube(
            source_path=data_path,
            output_path=out_path,
            dtype=dtype,
            normalize=normalize,
            cachemax_mb=cache_max,
        )

    else:
        raise Exception("Error: Invalid input directory or file path")
