#!/usr/bin/env python3

"""
Filename: test_bgp.py
Description: Test the band generation function
- Reads datacube (.npy)
- Creates synthetic band combiantions of datacube
- Saves output to new datacube (.npy)

-------
Example
-------

# Input from datacube directory, output new datacube (suffix "_bgp") 
tests/test_bgp.py -i data/datacubes/archimedes_cubes/Arch_93r_92v.npy -o results/
"""

# Mainstream packages
import sys
import os
from pathlib import Path
import getopt


# Relative package import workaround
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

# Environment packages
from algorithms import bgp
from utils.dataloader import load_datacube

__author__ = "Gian-Mateo (Mateo) Tifone"
__license__ = "MIT"
__date__ = "12-23-2025"
__email__ = "mt9485@rit.edu"


if __name__ == "__main__":
    # ---------------------------------------------
    # Commandline parser
    # ---------------------------------------------

    # ':' expects value for input and output files
    args = sys.argv[1:]
    options = "hi:o:n:"
    long_options = [
        "help",
        "input_dir=",
        "output_dir=",
        "name=",
    ]

    out_name = None

    try:
        # Parse arguments from command line
        arguments, _ = getopt.getopt(args, options, long_options)

        # Extract (key, value) pairs
        for key, value in arguments:
            if key in ("-h", "--help"):
                print(
                    """
                Test script to expand a datacube with synthetic, non-linear band combinations
                
                A datacube is a memory safe, NumPy array data file (.npy)
                
                Input supported file type is NumPy array (.npy)
                
                Exports to NumPy array (.npy)
                
                Options:
                  -h, --help                display this message
                  -i, --input_file          input datacube file 
                  -o, --output_dir          output directory for new datacube file
                  -n, --name                specify output file name
                """
                )

                # Exit program after help message
                sys.exit()

            elif key in ("-i", "--input_file"):
                in_dir = value

            elif key in ("-o", "--output_dir"):
                out_dir = value
                print("Saving output to: ", value)

            elif key in ("-n", "--name"):
                out_name = value

    except getopt.error as err:
        print(str(err))

    assert (
        in_dir
    ) is not None, "Input file cannot be empty or blank. See --help for details."

    # ---------------------------------------------
    # Construct path to data
    # ---------------------------------------------

    # Join base path with user path
    # NOTE: This assumes test file is run in project directory
    #       and path provided is not absolute path
    data_path = os.path.join(base_dir, in_dir)
    out_path = os.path.join(base_dir, out_dir)

    # ---------------------------------------------
    # Load datacube and out name
    # ---------------------------------------------

    # Load datacube
    datacube = load_datacube(source_path=data_path)

    # Assign out_name if not given
    # Assumes input file name with "_bgp.npy" suffix
    if not out_name:
        out_name = Path(data_path).stem + "_bgp.npy"

    # ---------------------------------------------
    # Run BGP
    # ---------------------------------------------

    # Saves datacube (.npy) for future loading
    bgp(datacube=datacube, dst_path=out_path, dst_name=out_name)


