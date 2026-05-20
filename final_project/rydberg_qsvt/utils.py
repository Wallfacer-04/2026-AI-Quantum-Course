"""Shared utility functions for plotting and figure management."""

import os
import matplotlib.pyplot as plt


FIGURES_DIR = "figures"


def save_figure(fig, name: str, dpi: int = 300):
    """Save a matplotlib figure to the figures directory.

    Args:
        fig: matplotlib figure object
        name: filename (without extension)
        dpi: resolution
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")
