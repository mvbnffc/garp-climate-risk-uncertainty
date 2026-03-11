"""
Plotting utilities — consistent style across all notebooks.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Colour palette
COLOURS = {
    "beta": "#2166ac",
    "triangular": "#b2182b",
    "uniform": "#7fbc41",
    "data_min": "#fee08b",
    "data_mean": "#66c2a5",
    "data_max": "#3288bd",
    "stage1": "#4393c3",
    "stage2": "#d6604d",
    "threshold": "#878787",
    "grey": "#969696",
}


def set_style():
    """Apply consistent matplotlib style for all project figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })
