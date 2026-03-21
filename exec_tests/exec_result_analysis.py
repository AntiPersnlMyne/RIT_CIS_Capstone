#!/usr/bin/env python3

"""
File: exec_result_analysis.py
Author: Gian-Mateo T.
License: GPL-2.0
Version: 1.0
Brief:
Statistical analysis of the data collectd from the
6 tests + unsupervised pseudocolors 

-------
Example
-------

# Just run the script, no parameters required
python exec_tests/result_analysis.py
"""
    
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Shorthand
f, F = 0, 0
t, T = 1, 1

def bar_plot(corrects: np.ndarray, confidences: np.ndarray) -> None:
    fig, axes_left = plt.subplots(3, 3, sharex=True, sharey=True, num="Character Recognition Performance")
    fig.set_size_inches(10, 10)
    axes_left = axes_left.ravel()
    axes_right = np.array([ax.twinx() for ax in axes_left])

    left_ax_color = "midnightblue"
    right_ax_color = "tab:red"

    fig.suptitle("Character Recognition Performance", fontsize=16)

    # Set y-limits and tick colors on all axes
    for i, (ax_l, ax_r) in enumerate(zip(axes_left, axes_right)):
        ax_l.set_ylim(0, 1)
        ax_r.set_ylim(0, 1)
        ax_l.tick_params(axis="y", colors=left_ax_color)
        ax_r.tick_params(axis="y", colors=right_ax_color)
        
        # # Three leftmost plots keep their left y-axis
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
        color=left_ax_color, fontsize=12, rotation="vertical"
    )

    # Place a single right-side y-label for the whole figure
    fig.text(
        0.98, 0.5, 
        "Total confidence (normalized)",
        va="center", ha="right",
        color=right_ax_color, fontsize=12, rotation="vertical"
    )

    x_axis = ["line 1", "line 2", "line 3", "line 4"]
    x = np.arange(len(x_axis))
    bar_width = 0.25

    n_plots = min(corrects.shape[0], 7)
    for i in range(n_plots):
        ax_l = axes_left[i]
        ax_r = axes_right[i]
        ax_l.bar(x - bar_width / 2, corrects[i, :],    width=bar_width, color=left_ax_color, alpha=0.5)
        ax_r.bar(x + bar_width / 2, confidences[i, :], width=bar_width, color=right_ax_color, alpha=0.5)

        ax_l.set_xticks(x)
        ax_l.set_xticklabels(x_axis, ha="center")

        ax_l.set_title(f"Test {i + 1}", fontsize=10)
        
    axes_left[6].set_title("Test 7 (Unsupervised)", fontsize=10)

    for i in range(n_plots, 9):
        axes_left[i].set_visible(False)
        axes_right[i].set_visible(False)

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])  # leave space for edge labels and suptitle
    plt.show()


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
# Luke - Test 3 ; Luke did a crap job, replace his data
# ------------------------------------------------------------
@dataclass(frozen=True)
class Test3:
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
        

if __name__ == "__main__":
    
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
        
    bar_plot(corrects=corrects, confidences=conf_norm)
    
        



