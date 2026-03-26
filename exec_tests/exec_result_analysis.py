#!/usr/bin/env python3

"""
File: exec_result_analysis.py
Author: Gian-Mateo T.
License: GPL-2.0
Version: 1.0
Brief:

Statistical analysis of the data collectd from the
6 tests + unsupervised pseudocolors ("Test 7") 
- Bar chart: compares how you actually did, to how you think you did
- Error bar: compares mean + std of confidence scores 

-------
Example
-------

# Just run the script, no parameters required
python exec_tests/result_analysis.py
"""
    
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
from tifffile import imread
from pathlib import Path
from typing import Iterable

# Shorthand
f, F = 0, 0
t, T = 1, 1

# ------------------------------------------------------------
# Justin - Test 1
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test1:
    def __str__():
        return "Test1"
    
    letters_line1 = [
        f, f, f, f, t, f,   
        f, f, t, 
        f, t, 
        f, f, t, t, t, t, f, t, t, f,
    ]

    confidences_line1 = [
        2, 1, 1, 1, 3, 2,
        3, 2, 5, 
        3, 4, 
        4, 3, 5, 5, 5, 5, 3, 3, 5, 1,
    ]

    letters_line2 = [
        f, f, f, f, f, 
        t, t,
        f, f, f,
        f, f, f, f,
        t, t, t, f, t, f,
    ]

    confidences_line2 = [
        1, 1, 1, 5, 5, 
        3, 2, 
        1, 1, 2, 
        1, 3, 3, 4, 4, 4, 3, 1, 5, 1,
    ]

    letters_line3 = [
        t, t, t, t, t, f, f, t, 
        f, t, f, f, t, t,
        t, f, t, t, t, f, f, t, t,
    ]
    
    confidences_line3 = [
        5, 5, 5, 5, 1, 2, 2, 3,
        4, 4, 2, 3, 4, 3,
        3, 4, 5, 5, 3, 2, 2, 4, 3,
    ]
    
    letters_line4 = [
        f, f, t, t, t, t, t, 
        t, t, t, t,
        f, f, f,
        t, t,
        t, f, f, f, f, f, f, t,   
    ]
    
    confidences_line4 = [
        3 ,2 ,4, 4, 5, 4, 3, 
        5, 4, 4, 4, 
        1, 1, 2, 
        2, 3, 
        4, 4, 4, 5, 5, 4, 1, 3,
    ]
    

# ------------------------------------------------------------
# Evie - Test 2
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test2:
    def __str__():
        return "Test2"
    
    letters_line1 = [
        t, t, f, f, f, t,
        f, t, t, 
        t, t,
        t, f, f, t, t, t, t, t, 
        t, f,
    ]
    
    confidences_line1 = [
        1, 1, 1, 1, 2, 4,
        4, 4, 4,
        4, 4, 
        4, 4, 4, 4, 4, 4, 4, 2, 
        4, 1,
    ]
    
    letters_line2 = [
        f, f, f, f, f, 
        t, f,
        f, f, f,
        f, f, f, f, 
        f, f, t, t, t, f,
    ]
    
    confidences_line2 = [
        1, 1, 1, 4, 1, 
        1, 4, 
        1, 1, 1, 
        1, 3, 2, 1, 
        2, 2, 1, 3, 2, 1,
    ]
    
    letters_line3 = [
        t, t, f, t, f, f, t, t, 
        f, f, f, f, f, f,
        f, f, f, t, t, f, t, t, t,
    ]
    
    confidences_line3 = [
        4, 4, 4, 4, 4, 3, 2, 2,
        3, 2, 2, 2, 1, 3,
        1, 4, 4, 1, 4, 4, 4, 2, 1
    ]
    
    letters_line4 = [
        f, f, t, f, t, t, t, 
        t, t, t, t, 
        f, f, f, 
        f, f,
        f, f, f, t, f, f, t, t,
    ]
    
    confidences_line4 = [
        4, 4, 4, 4, 4, 4, 4, 
        4, 4, 4, 4, 
        3, 3, 3, 
        3, 3, 
        2, 4, 4, 2, 4, 2, 1, 1,
    ]

    
# ------------------------------------------------------------
# Luke - Test 3 (v2) 
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test3:
    def __str__():
        return "Test3"
    
    letters_line1 = [
        f, t, t, t, t, t, 
        f, t, t, 
        t, t,
        f, f, t, t, t, t, t, t, 
        f, t,
    ]
    
    confidences_line1 = [
        1, 3, 2, 3, 2, 2, 
        3, 2, 3, 
        4, 3, 
        3, 3, 4, 5, 5, 5, 5, 4, 
        4, 5,
    ]
    
    letters_line2 = [
        f, f, f, f, f, 
        t, t, 
        f, f, f,
        f, t, f, t,
        t, t, t, t, t, f,
    ]
    
    confidences_line2 = [
        1, 1, 1, 2, 1,
        3, 2,
        1, 1, 1,
        1, 2, 2, 2, 
        3, 1, 2, 1, 2, 1,
    ]
    
    letters_line3 = [
        f, f, t, t, f, f, t, f, 
        f, f, f, f, f, f,
        f, f, f, f, f, f, f, f, t,
    ]
    
    confidences_line3 = [
        2, 1, 3, 3, 1, 1, 1, 1,
        1, 3, 2, 1, 2, 3,
        2, 4, 3, 2, 2, 2, 2, 2, 1,
    ]
    
    letters_line4 = [
        f, f, t, t, t, t, t,
        t, t, t, f,
        f, f, f,
        t, t, 
        t, t, f, f, f, t, f, f,
    ]
    
    confidences_line4 = [
        2, 3, 4, 4, 5, 5, 4,
        2, 4, 2, 2,
        1, 1, 1,
        3, 2,
        3, 4, 3, 3, 1, 2, 2, 1,
    ]        


# ------------------------------------------------------------
# Will - Test 4
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test4:
    def __str__():
        return "Test4"
    
    letters_line1 = [
        t, t, t, t, t, t, 
        f, t, t, 
        t, f,
        t, f, t, t, t, t, t, t, 
        t, t,  
    ]
    
    confidences_line1 = [
        4, 5, 4, 5, 4, 4,
        1, 5, 5,
        4, 2,
        4, 2, 3, 5, 5, 5, 3, 5, 
        5, 3,
    ]
    
    letters_line2 = [
        f, f, f, f, f, 
        t, t, 
        f, t, t, 
        t, f, t, t, 
        f, f, t, t, t, f, 
    ]
    
    confidences_line2 = [
        1, 1, 1, 2, 1,
        5, 5, 
        1, 3, 4,
        5, 3, 3, 4, 
        2, 4, 4, 5, 4, 3,
    ]
    
    letters_line3 = [
        t, t, t, t, f, f, t, f,
        f, t, f, f, t, t,
        f, f, f, t, t, f, t, t, f,
    ]
    
    confidences_line3 = [
        5, 4, 4, 5, 1, 2, 4, 2,
        2, 4, 3, 1, 3, 3, 
        2, 1, 1, 5, 5, 2, 2, 2, 1,
    ]
    
    letters_line4 = [
        f, t, t, t, t, t, t, 
        t, t, t, t, 
        f, f, f, 
        t, t,
        t, t, f, t, f, f, t, t,
        
    ]
    
    confidences_line4 = [
        2, 1, 4, 5, 4, 4, 4,
        5, 4, 3, 5,
        3, 3, 1, 
        4, 4, 
        2, 4, 5, 4, 1, 1, 3, 2,
    ]


# ------------------------------------------------------------
# Cooper - Test 5
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test5:
    def __str__():
        return "Test5"
    
    letters_line1 = [
        f, f, f, t, f, f, 
        f, f, t, 
        t,f, 
        f, f, t, t, t, t, f, f,
        f, f, 
    ]
    
    confidences_line1 = [
        1, 2, 2, 1, 1, 1, 
        1, 2, 4, 
        5, 2, 
        1, 2, 3, 4, 4, 5, 2, 3, 
        3, 4,
    ]

    letters_line2 = [
        f, f, f, f, t,
        t, t, 
        f, t, f,
        t, t, f, t, 
        f, f, f, f, t, f,
    ]
    
    confidences_line2 = [
        1, 1, 1, 1, 1, 
        2, 1, 
        1, 2, 1, 
        1, 1, 1, 2, 
        1, 4, 4, 2, 5, 1,
    ]

    letters_line3 = [
        f, f, f, f, f, t, f, f,
        f, f, f, f, f, f,
        f, f, f, t, t, f, f, f, f,
    ]
    
    confidences_line3 = [
        3, 3, 3, 1, 3, 3, 1, 1, 
        3, 4, 2, 2, 4, 2, 
        3, 3, 3, 4, 3, 1, 2, 2, 1,
    ]
    
    letters_line4 = [
        f, f, t, t, t, t, t,
        t, t, f, t,
        f, f, f,
        t, t,
        f, t, t, t, f, f, f, t, 
    ]
    
    confidences_line4 = [
        2, 3, 4, 4, 4, 4, 4,
        3, 4, 5, 4,
        1, 1, 2, 
        3, 3,
        3, 4, 4, 5, 4, 1, 1, 1,
    ]


# ------------------------------------------------------------
# Liam - Test 6
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test6:
    def __str__():
        return "Test6"
    
    letters_line1 = [
        f, f, f, f, f, f,
        f, f, f,
        f, f,
        f, f, f, f, t, t, f, f,
        f, t,
    ]
    
    confidences_line1 = [
        1, 1, 1, 1, 2, 2,
        2, 2, 1,
        2, 3, 
        3, 2, 4, 3, 4, 4, 5, 4, 
        4, 5,
    ]
    
    letters_line2 = [
        f, f, f, f, f,
        f, f,
        f, f, f,
        f, f, f, f,
        f, t, t, f, t, f,
    ]
    
    confidences_line2 = [
        1, 1, 1, 1, 1,
        1, 1,
        1, 1, 1,
        1, 1, 1, 1,
        1, 1, 2, 1, 1, 2,
    ]
    
    letters_line3 = [
        t, f, f, t, f, f, f, f,
        f, f, f, f, t, t,
        f, f, f, f, f, f, f, f, f,
    ]
    
    confidences_line3 = [
        4, 1, 1, 5, 1, 1, 2, 2,
        1, 1, 1, 1, 1, 1,
        2, 2, 1, 2, 1, 1, 1, 1, 1,
    ]
    
    letters_line4 = [
        f, f, t, t, t, t, f, 
        f, t, t, t, 
        f, f, f, 
        t, f,
        f, f, f, t, f, f, t, t,
    ]
    
    confidences_line4 = [
        2, 1, 5, 5, 5, 3, 2, 
        3, 1, 3, 1, 
        3, 4, 3, 
        4, 3, 
        1, 2, 1, 4, 1, 1, 2, 2, 
    ]


# ------------------------------------------------------------
# Elyse - Test 7 (Unsupervised)
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test7:
    def __str__():
        return "Test7"
    
    letters_line1 = [
        f, t, t, t, t, t, 
        f, t, t,
        t, f, f, f, t, t, t, t, f, f, 
        t, f, 
    ]
    
    confidences_line1 = [
        1, 4, 3, 3, 2, 3,
        1, 4, 5, 
        5, 3, 
        3, 3, 5, 5, 5, 5, 3, 3, 
        3, 2,
    ]
    
    letters_line2 = [
        f, f, f, f, f, 
        t, t,
        f, t, t, 
        t, t, t, t, 
        t, t, t, t, t, f,
    ]
    
    confidences_line2 = [
        2, 2, 2, 5, 1, 
        5, 2,
        2, 5, 1,
        3, 3, 3, 4,
        4, 2, 3, 2, 5, 2, 
    ]
    
    letters_line3 = [
        t, t, t, t, t, f, t, f,
        f, t, f, t, t, t, 
        f, f, t, t, t, f, f, f, f, 
    ]
    
    confidences_line3 = [
        3, 4, 4, 4, 2, 3, 5, 3,
        5, 2, 1, 2, 3, 3, 
        3, 5, 4, 5, 5, 5, 1, 3, 3,
    ]
    
    letters_line4 = [
        f, f, t, t, t, t, t, 
        t, t, f, t, 
        f, f, t, 
        t, t,
        t, f, f, t, f, f, f, f,
    ]
    
    confidences_line4 = [
        3, 3, 5, 5, 5, 3, 4, 
        5, 3, 2, 4,
        2, 4, 5, 
        5, 3,
        3, 5, 4, 4, 2, 1, 1, 1,
    ]        
    
    
# ------------------------------------------------------------
# Analysis Plots
# ------------------------------------------------------------
def test_and_line_scores_plot(corrects: np.ndarray, confidences: np.ndarray, save_dir:Path|str) -> None:
    """
    Creates a bar plot for the 7 tests. Plots accuracy (%correct) against confidences. 

    Args:
        corrects (np.ndarray): 2D array, `(n_tests, test_data)`
        confidences (np.ndarray): 2D array `(n_tests, test_data)`
        save_dir (Path|str): Where to save the plot. Saves as TeX-friendly `.eps`.
    """
    figname = "Character_Recognition_Performance"
    
    fig, axes_left = plt.subplots(3, 3, sharex=True, sharey=True, num=figname)
    fig.set_size_inches(10, 8)
    fig.suptitle("Transcription Performance: Per-Test, Per-Line", fontsize=16)
    
    # Left and Right y-label
    axes_left = axes_left.ravel()
    axes_right = np.array([ax.twinx() for ax in axes_left])
    
    left_ax_color = "midnightblue"
    right_ax_color = "tab:red"

    # Set y-limits and tick colors on all axes
    for i, (ax_l, ax_r) in enumerate(zip(axes_left, axes_right)):
        ax_l.set_ylim(0, 1)
        ax_r.set_ylim(0, 1)
        ax_l.tick_params(axis="y", colors=left_ax_color)
        ax_r.tick_params(axis="y", colors=right_ax_color)
        
        # Three leftmost plots keep their left y-axis
        # Formatted by default
        
        # Three rightmost plots keep their right y-axis
        if i not in (2, 5):
            ax_r.set(yticklabels=[])     
            ax_r.tick_params(left=False)  
            
        ax_r.grid(True, linestyle="--", alpha=0.4)
        ax_l.grid(True, linestyle="--", alpha=0.4)

    x_axis = ["Line 1", "Line 2", "Line 3", "Line 4"]
    x = np.arange(len(x_axis))
    bar_width = 0.30

    # Plot bar charts
    n_plots = min(corrects.shape[0], 7)
    for i in range(n_plots):
        ax_l = axes_left[i]
        ax_r = axes_right[i]
        
        # y-axis
        ax_l.bar(x - bar_width / 2, corrects[i, :],    width=bar_width, color=left_ax_color, alpha=0.5)
        ax_r.bar(x + bar_width / 2, confidences[i, :], width=bar_width, color=right_ax_color, alpha=0.5)

        # x-axis
        ax_l.set_xticks(x)
        ax_l.set_xticklabels(x_axis, ha="center")

        # Title
        ax_l.set_title(f"Test {i + 1}", fontsize=14)
    
    # Overrite Test 7's title
    axes_left[6].set_title("Test 7 (Unsupervised)", fontsize=14)

    # Remove/hide unused plots
    for i in range(n_plots, 9):
        axes_left[i].set_visible(False)
        axes_right[i].set_visible(False)

    # Add legend
    legend_handles = [
        Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.5, label="Accuracy"),
        Patch(facecolor="tab:red",  edgecolor="tab:red",  alpha=0.5, label="Mean confidence (normalized)")
    ]
    fig.legend(handles=legend_handles, loc="lower right")
    
    # rect=[left, bottom, right, top]
    fig.tight_layout(rect=[0,0,1.0,0.97]) 

    # Save and show plot
    fig.savefig(f"{save_dir}/{figname}.eps")
    plt.show()


# def confidence_plot(means:np.ndarray, stds:np.ndarray, save_dir:Path|str) -> None:
#     """
#     Plots mean+std confidence for each line, from each test. Creates 2x2 plot.

#     Args:
#         means (np.ndarray): 2D array, `(n_tests, test_data)`
#         stds (np.ndarray): 2D array `(n_tests, test_data)`
#         save_dir (Path|str): Where to save the plot. Saves as TeX-friendly `.eps`.
#     """
#     figname = "Average_Confidence_per_Line"
#     fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=figname)
#     fig.set_size_inches(8, 5)
#     axes = axes.ravel()
    
#     # Title and axis name
#     fig.suptitle("Average Confidence: Per Line", fontsize=16)
#     fig.text(      # y-axis label
#         0.02, 0.5, # x (near left edge), y (center)
#         "Confidence Score",
#         va="center", ha="left",
#         fontsize=14, rotation="vertical"
#     )
    
#     # x-axis
#     n_tests = means.shape[0]
#     x_labels = [f"Test {n + 1}" for n in range(n_tests)]
#     x_axis = np.arange(0, n_tests)
    
#     # Dynamic colors
#     colors = cm.Accent(np.linspace(0, 1, n_tests))
    
#     for i, ax in enumerate(axes):
#         for x, color in zip(x_axis, colors):
#             y = means[x, i]
#             ax.errorbar(
#                 x, y,
#                 yerr=stds[x, i],
#                 fmt="o",
#                 color=color,
#                 ecolor="darkgray",
#                 capsize=2
#             )
        
#         # Labels
#         ax.set_xticks(x_axis)
#         ax.set_xticklabels(x_labels, ha="center")
#         ax.set_title(f"Line {i + 1}", fontsize=14)
        
#         # Format
#         ax.set_ylim([0,5])
#         ax.grid(True, linestyle='--', alpha=0.4)
    
#     fig.tight_layout(rect=(0.05, 0.01, 0.95, 0.95))
    
#     fig.savefig(f"{save_dir}/{figname}.eps")
#     plt.show()
   
    
def pseudocolors_plot(pseudocolor_paths:Iterable[str|Path], uv_path:str|Path, save_dir:Path|str) -> None:
    """
    Displays the (cropped) pseudocolors for each test around the UV (ground truth). 
    Creates 3x3 figure.

    Args:
        pseudocolor_paths (Iterable[str | Path]): Paths to the cropped images, `Test1->Test7`. Assumes TIFFs.
        uv_path (str | Path): Path to cropped UV image. Assumes TIFF.
        save_dir (Path|str): Where to save the plot. Saves as TeX-friendly `.eps`.
    """
    # Read in data
    pseudo_images = [imread(path, sort=True) / 65535.0 for path in pseudocolor_paths]
    uv_image = imread(uv_path) 
  
    # Define figure
    figname = "Test_Pseudocolors"
    fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, num=figname)
    fig.set_size_inches(4, 5)
    # fig.suptitle("Test Pseudocolors", fontsize=16)
    
    axes = axes.ravel()
    fig.subplots_adjust(wspace=-0.05, hspace=0.00)
    
    n_plots = len(pseudocolor_paths) # +1 = UV
    
    # Add images to plot
    for i in range(n_plots):
        ax = axes[i]

        # Turn off tick marks
        ax.set(yticklabels=[], xticklabels=[])     
        ax.tick_params(left=False, bottom=False, top=False, right=False)  
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot and label Test 1-7
        ax.imshow(pseudo_images[i])
        ax.set_title(f"Test {i+1}")
        
    # Plot UV
    axes[n_plots].imshow(uv_image)
    axes[n_plots].set_title(f"Ground Truth")
        
    for i in range(n_plots + 1, len(axes)):
        # Hide unused plot
        axes[i].set_visible(False)
    
    fig.tight_layout(rect=[0, -0.25, 1.0, 1.0])

    # Save and show
    fig.savefig(f"{save_dir}/{figname}.eps", format="eps")
    plt.show()
        

def test_aggregation_plot(tests:list, save_dir:Path|str) -> None:
    """
    Aggregates performance over tests, per line.
    Plots overall accuracy vs mean confidence.

    Saves as: Test_Aggregations.eps
    """
    figname = "Test_Aggregations"
    
    n_tests = len(tests)
    accuracies = np.zeros(n_tests)
    mean_conf  = np.zeros(n_tests)

    for i, test in enumerate(tests):
        # Concatenate all lines
        letters = np.array(
            test.letters_line1 +
            test.letters_line2 +
            test.letters_line3 +
            test.letters_line4
        )
        
        confs = np.array(
            test.confidences_line1 +
            test.confidences_line2 +
            test.confidences_line3 +
            test.confidences_line4
        )
        
        # Compute metrics
        accuracies[i] = np.mean(letters)
        mean_conf[i]  = np.mean(confs) / 5.0  # normalize to [0,1]

    # Plot
    x = np.arange(n_tests)
    width = 0.30

    fig, ax = plt.subplots(num=figname)
    fig.set_size_inches(8, 5)
    
    ax.bar(x - width/2, accuracies, width, label="Accuracy", alpha=0.7, color="tab:blue")
    ax.bar(x + width/2, mean_conf, width, label="Mean Confidence (normalized)", alpha=0.7, color="tab:red")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Test {i+1}" for i in range(n_tests)])
    ax.set_ylim(0, 1)

    ax.set_title("Overall Performance per Test", fontsize=14)
    ax.set_ylabel("Score [%]")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/{figname}.eps", format="eps")
    plt.show()


def line_aggregation_plot(tests: list, save_dir: Path | str) -> None:
    """
    Aggregates performance over lines, per test.
    Plots overall accuracy vs mean confidence.

    Saves as: Line_Aggregations.eps
    """
    figname = "Line_Aggregations"

    n_lines = 4
    accuracies = np.zeros(n_lines)
    mean_conf  = np.zeros(n_lines)

    for i in range(n_lines):
        all_letters = []
        all_confs = []

        for test in tests:
            # Dynamically access line attributes
            letters = getattr(test, f"letters_line{i+1}")
            confs   = getattr(test, f"confidences_line{i+1}")

            all_letters.extend(letters)
            all_confs.extend(confs)

        # Convert to arrays
        all_letters = np.array(all_letters)
        all_confs   = np.array(all_confs)

        # Compute metrics
        accuracies[i] = np.mean(all_letters)
        mean_conf[i]  = np.mean(all_confs) / 5.0  # normalize to [0,1]

    # Plot
    x = np.arange(n_lines)
    width = 0.30

    fig, ax = plt.subplots(num=figname)
    fig.set_size_inches(8, 5)

    ax.bar(x - width/2, accuracies, width, label="Accuracy", alpha=0.7, color="tab:blue")
    ax.bar(x + width/2, mean_conf, width, label="Mean Confidence (normalized)", alpha=0.7, color="tab:red")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Line {i+1}" for i in range(n_lines)])
    ax.set_ylim(0, 1)

    ax.set_title("Overall Performance per Line", fontsize=14)
    ax.set_ylabel("Score [%]")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/{figname}.eps", format="eps")
    plt.show()
    

def calibration_curve_plot(tests:list, save_dir:Path|str) -> None:
    """
    Computes and plots calibration curve:
    accuracy vs confidence level.

    Saves as: Calibration_Curve.eps
    """
    figname = "Calibration_Curve"

    all_letters = []
    all_confs   = []

    # Combine all participants, all lines, all tests
    for test in tests:
        # Aggregate all letters into one line
        all_letters.extend(
            test.letters_line1 +
            test.letters_line2 +
            test.letters_line3 +
            test.letters_line4
        )
        # Aggregate all confidences into one line
        all_confs.extend(
            test.confidences_line1 +
            test.confidences_line2 +
            test.confidences_line3 +
            test.confidences_line4
        )

    all_letters = np.array(all_letters)
    all_confs   = np.array(all_confs)

    # Compute accuracy per confidence level
    conf_levels = np.arange(1, 6)
    accuracies  = []

    # Iterate trials where confidence = # (1, 2, ...)
    for conf_lvl in conf_levels:
        mask = (all_confs == conf_lvl)
        if np.any(mask):
            accuracies.append(np.mean(all_letters[mask]))
        else:
            accuracies.append(np.nan)

    accuracies = np.array(accuracies)

    # Normalize confidence to [0,1]
    conf_norm = conf_levels / 5.0

    # Plot
    fig, ax = plt.subplots(num=figname)
    fig.set_size_inches(6, 6)

    # Empirical curve
    ax.plot(conf_norm, accuracies, marker='o', label="Observed")

    # Ideal calibration
    ax.plot([0,1], [0,1], linestyle="--", label="Perfect Calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Confidence (normalized) [%]")
    ax.set_ylabel("Accuracy [%]")
    ax.set_title("Calibration Curve", fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{save_dir}/{figname}.eps", format="eps")
    plt.show()


if __name__ == "__main__":
    
    # =================================
    # Load Cropped Pseudocolors
    # =================================
    
    # Cropped pseudocolors
    pseudo_dir = "results/figures/102r-98v/cropped"
    pseudocolor_paths = sorted(Path(pseudo_dir).glob("test*.tiff"))
    # Cropped UV
    uv_path = "results/figures/102r-98v/cropped/uv.tiff"
    
    # Save dir
    save_dir = "results/figures/paper_figures"
    
    # =================================
    # Load Test(s) Results
    # =================================
    
    tests:list = [Test1, Test2, Test3, Test4, Test5, Test6, Test7]
    
    # rows = Test#
    # cols = Mean (Std)
    means = np.empty((7, 4), dtype=np.float64)
    stds  = np.empty((7, 4), dtype=np.float64)
    
    # rows = Test#
    # cols = correct/100 (total confidence normalized) :: [0,1]
    corrects  = np.empty((7, 4), dtype=np.float64)
    conf_norm = np.empty((7, 4), dtype=np.float64)
    
    for i, test in enumerate(tests):
        # Line 1
        assert len(test.letters_line1) == len(test.confidences_line1), f"Test {test}"
        
        means[i, 0] = np.mean(conf:=test.confidences_line1)
        stds[i, 0]  = np.std(conf, ddof=1)
        conf_norm[i, 0] = np.sum(conf) / (len(conf) * 5)
        corrects[i, 0]  = np.sum(let:=test.letters_line1) / len(let)
        
        # Line 2
        assert len(test.letters_line2) == len(test.confidences_line2), f"Test {test}"
        
        means[i, 1] = np.mean(conf:=test.confidences_line2)
        stds[i, 1]  = np.std(conf, ddof=1)
        conf_norm[i, 1] = np.sum(conf) / (len(conf) * 5)
        corrects[i, 1]  = np.sum(let:=test.letters_line2) / len(let)
        
        # Line 3
        assert len(test.letters_line3) == len(test.confidences_line3), f"Test {test}"
        
        means[i, 2] = np.mean(conf:=test.confidences_line3)
        stds[i, 2]  = np.std(conf, ddof=1)
        conf_norm[i, 2] = np.sum(conf) / (len(conf) * 5)
        corrects[i, 2]  = np.sum(let:=test.letters_line3) / len(let)
        
        # Line 4
        assert len(test.letters_line4) == len(test.confidences_line4), f"Test {test}"
        
        means[i, 3] = np.mean(conf:=test.confidences_line4)
        stds[i, 3]  = np.std(conf, ddof=1)
        conf_norm[i, 3] = np.sum(conf) / (len(conf) * 5)
        corrects[i, 3]  = np.sum(let:=test.letters_line4) / len(let)
        

    # =================================
    # Analysis
    # =================================
    
    # # Mean + std of confidence scores
    # confidence_plot(means=means, stds=stds, save_dir=save_dir) 
    
    # Cropped pseudocolors around UV
    pseudocolors_plot(pseudocolor_paths=pseudocolor_paths, uv_path=uv_path, save_dir=save_dir)
    
    # Calibration curve
    calibration_curve_plot(tests=tests, save_dir=save_dir)
        
    # Average Scores: per line, per test
    test_and_line_scores_plot(corrects=corrects, confidences=conf_norm, save_dir=save_dir)
    
    # Average Score: across tests, per line
    test_aggregation_plot(tests=tests, save_dir=save_dir)

    # Average Score: across lines, per test
    line_aggregation_plot(tests=tests, save_dir=save_dir)

