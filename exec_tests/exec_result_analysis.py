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
        t, t, t, t, f, f, f, t, 
        f, t, f, f, f, f, 
        f, t, f, t, t, t, f, t, t, t, f, f, f, t, t, 
    ]
    
    confidences_line3 = [
        5, 5, 5, 5, 3, 2, 2, 3, 
        4, 4, 2, 3, 4, 4, 
        2, 3, 4, 3, 3, 4, 5, 5, 3, 2, 2, 4 , 3,
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
    letters_line1 = [
        f, f, f, t, t, f, 
        f, t, t, 
        t, f, 
        f, f, t, t, t, t, f, t, t, f,
    ]
    
    confidences_line1 = [
        2, 2, 1, 2, 4, 1, 
        1, 3, 5,
        4, 1,
        5, 4, 4, 5, 4, 3, 3, 2, 4, 2,
    ]
    
    letters_line2 = [
        f, f, t, f, f, 
        t, t, 
        f, t, f, 
        t, f, f, f, 
        f, t, t, t, t, f,
    ]
    
    confidences_line2 = [
        2, 1, 1, 4, 1,
        4, 2,
        1, 3, 1, 
        4, 2, 2, 3, 
        3, 4, 2, 2, 4, 2,
    ]
    
    letters_line3 = [
        t, t, t, t, f, f, t, f,
        f, f, f, f, t, t,
        f, f, t, t, t, t, t, t, f,
    ]
    
    confidences_line3 = [
        3, 3, 2, 2, 2, 1, 3, 3, 
        2, 2, 1, 1, 2, 3, 
        2, 1, 2, 4, 5, 4, 3, 5, 1,
    ]
    
    letters_line4 = [
        f, f, f, f, t, t, t, 
        t, t, t, t, 
        f, f, f, 
        f, f,
        f, t, t, t, t, f, f, t, t,
    ]
    
    confidences_line4 = [
        1, 3, 2, 3, 4, 5, 1,
        3, 4, 1, 3, 
        2, 1, 3,
        1, 1,
        2, 2, 2, 4, 3, 1, 2, 1,
    ]


# ------------------------------------------------------------
# Cooper - Test 5
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test5:
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
    letters_line1 = [
        f, f, f, f, t, t,
        f, f, f, 
        f, f, 
        f, f, f, f, f, t, f, f,
        t, f,
    ]
    
    confidences_line1 = [
        2, 3, 1, 2, 3, 3, 
        4, 1, 3,
        3, 2,
        1, 1, 2, 1, 1, 5, 5, 5, 
        2, 3,
    ]
    
    letters_line2 = [
        f, f, f, f, t, 
        f, f,
        f, f, f,
        f, f, f, f,
        f, t, f, f, t, f,
    ]
    
    confidences_line2 = [
        1, 1, 1, 1, 2,
        2, 1,
        2, 2, 1,
        1, 1, 1, 1,
        1, 4, 3, 1, 2, 1,
    ]
    
    letters_line3 = [
        t, f, f, f, f, f, f, f,
        f, t, f, f, f, f,
        f, f, f, t, t, f, f, f, f,
    ]
    
    confidences_line3 = [
        4, 2, 4, 1, 1, 1, 2, 2,
        2, 2, 2, 1, 1, 5,
        3, 1, 1, 5, 4, 1, 2, 4, 4,
    ]
    
    letters_line4 = [
        f, f, t, t, t, f, t, 
        t, t, t, t, 
        f, f, f, 
        f, f, 
        t, t, f, f, f, f, f, t,
    ]
    
    confidences_line4 = [
        2, 3, 3, 3, 4, 5, 5,
        3, 5, 5, 4, 
        3, 2, 2, 
        1, 1, 
        1, 4, 4, 4, 4, 5, 3, 5,
    ]


# ------------------------------------------------------------
# Elyse - Test 7 (Unsupervised)
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test7:
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
        f, f, t, t, t, t, f, f, f,
    ]
    
    confidences_line3 = [
        3, 4, 4, 4, 2, 3, 5, 3,
        5, 2, 2, 3, 3,
        1, 1, 5, 4, 5, 5, 5, 3, 3,
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
def bar_plot(corrects: np.ndarray, confidences: np.ndarray, save_dir:Path|str) -> None:
    """
    Creates a bar plot for the 7 tests. Plots %correct against confidences. 

    Args:
        corrects (np.ndarray): 2D array, `(n_tests, test_data)`
        confidences (np.ndarray): 2D array `(n_tests, test_data)`
        save_dir (Path|str): Where to save the plot. Saves as TeX-friendly `.eps`.
    """
    figname = "Character_Recognition_Performance"
    
    fig, axes_left = plt.subplots(3, 3, sharex=True, sharey=True, num=figname)
    fig.set_size_inches(10, 8)
    fig.suptitle("Character Recognition Performance", fontsize=16)
    
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

    # Place a single left-side y-label for the whole figure
    fig.text(
        0.02, 0.5,              # x (near left edge), y (center)
        "% Correct",      # text
        va="center", ha="left",
        color=left_ax_color, fontsize=14, rotation="vertical"
    )

    # Place a single right-side y-label for the whole figure
    fig.text(
        0.98, 0.5, 
        "Total confidence (normalized)",
        va="center", ha="right",
        color=right_ax_color, fontsize=14, rotation="vertical"
    )

    x_axis = ["Line 1", "Line 2", "Line 3", "Line 4"]
    x = np.arange(len(x_axis))
    bar_width = 0.25

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

    # Leave space for edge labels and suptitle
    fig.tight_layout(rect=[0.06, 0.08, 0.93, 0.95]) # left, bottom, right, top
    
    # Save and show plot
    fig.savefig(f"{save_dir}/{figname}.eps")
    plt.show()


def confidence_plot(means:np.ndarray, stds:np.ndarray, save_dir:Path|str) -> None:
    """
    Plots mean+std confidence for each line, from each test. Creates 2x2 plot.

    Args:
        means (np.ndarray): 2D array, `(n_tests, test_data)`
        stds (np.ndarray): 2D array `(n_tests, test_data)`
        save_dir (Path|str): Where to save the plot. Saves as TeX-friendly `.eps`.
    """
    figname = "Average_Confidence_per_Line"
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=figname)
    fig.set_size_inches(8, 5)
    axes = axes.ravel()
    
    # Title and axis name
    fig.suptitle("Average Confidence Score per Line", fontsize=16)
    fig.text(      # y-axis label
        0.02, 0.5, # x (near left edge), y (center)
        "Confidence Score",
        va="center", ha="left",
        fontsize=14, rotation="vertical"
    )
    
    # x-axis
    n_tests = means.shape[0]
    x_labels = [f"Test {n + 1}" for n in range(n_tests)]
    x_axis = np.arange(0, n_tests)
    
    # Dynamic colors
    colors = cm.Accent(np.linspace(0, 1, n_tests))
    
    for i, ax in enumerate(axes):
        for x, color in zip(x_axis, colors):
            y = means[x, i]
            ax.errorbar(
                x, y,
                yerr=stds[x, i],
                fmt="o",
                color=color,
                ecolor="darkgray",
                capsize=2
            )
        
        # Labels
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_labels, ha="center")
        ax.set_title(f"Line {i + 1}", fontsize=14)
        
        # Format
        ax.set_ylim([0,5])
        ax.grid(True, linestyle='--', alpha=0.4)
    
    fig.tight_layout(rect=(0.05, 0.01, 0.95, 0.95))
    
    fig.savefig(f"{save_dir}/{figname}.eps")
    plt.show()
   
    
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
    figname = "Detector_Pseudocolors"
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, num=figname)
    fig.set_size_inches(9, 5)
    fig.suptitle("Detector Pseudocolors", fontsize=16)
    
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.16, hspace=0.15)
    
    # Add images to plot
    for i, ax in enumerate(axes):

        # Turn off tick marks
        ax.set(yticklabels=[], xticklabels=[])     
        ax.tick_params(left=False, bottom=False, top=False, right=False)  
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot and label
        if i < 4: # Test 1-4
            ax.imshow(pseudo_images[i])
            ax.set_title(f"Test {i+1}")
        elif i > 4: # Test 5-7
            ax.imshow(pseudo_images[i-1])
            ax.set_title(f"Test {i}")
        else: # UV
            ax.imshow(uv_image)
            ax.set_title(f"Ultraviolet")
        
        if i == 7: break
        
    # Hide unused plot
    axes[8].set_visible(False)

    # Save and show
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
        means[i, 0] = np.mean(conf:=test.confidences_line1)
        stds[i, 0]  = np.std(conf)
        conf_norm[i, 0] = np.sum(conf) / (len(conf) * 5)
        corrects[i, 0]  = np.sum(let:=test.letters_line1) / len(let)
        
        # Line 2
        means[i, 1] = np.mean(conf:=test.confidences_line2)
        stds[i, 1]  = np.std(conf)
        conf_norm[i, 1] = np.sum(conf) / (len(conf) * 5)
        corrects[i, 1]  = np.sum(let:=test.letters_line1) / len(let)
        
        # Line 3
        means[i, 2] = np.mean(conf:=test.confidences_line3)
        stds[i, 2]  = np.std(conf)
        conf_norm[i, 2] = np.sum(conf) / (len(conf) * 5)
        corrects[i, 2]  = np.sum(let:=test.letters_line1) / len(let)
        
        # Line 4
        means[i, 3] = np.mean(conf:=test.confidences_line4)
        stds[i, 3]  = np.std(conf)
        conf_norm[i, 3] = np.sum(conf) / (len(conf) * 5)
        corrects[i, 3]  = np.sum(let:=test.letters_line1) / len(let)
        

    # =================================
    # Analysis
    # =================================
        
    # %Correct juxtaposed against Confidence
    bar_plot(corrects=corrects, confidences=conf_norm, save_dir=save_dir)
    
    # Mean + std of confidence scores
    confidence_plot(means=means, stds=stds, save_dir=save_dir) 
    
    # Cropped pseudocolors around UV
    pseudocolors_plot(pseudocolor_paths=pseudocolor_paths, uv_path=uv_path, save_dir=save_dir)
    

