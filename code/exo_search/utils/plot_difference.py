import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_difference(
    x: np.array,
    y_old: np.array,
    y_new: np.array,
    save_fig: Path = None,
    show_fig: bool = False,
    x_label: str = "Time",
    y_label: str = "Flux",
) -> None:
    """Plots the difference between two sets of y-values.

    Args:
        x (np.array): x-values.
        y_old (np.array): old y-values.
        y_new (np.array): new y-values
        save_fig (Path, optional): filename for saving the figure. Defaults to None.
        show_fig (bool, optional): if True, display the figure. Defaults to False.
        x_label (str, optional): label for the x axis. Defaults to "Time".
        y_label (str, optional): label for the y axis. Defaults to "Flux".
    """
    if save_fig is not None:
        save_fig.parent.mkdir(parents=True, exist_ok=True)

    # Create the figure
    _, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        height_ratios=[3, 1],
    )

    # Plot the difference
    ax[0].vlines(x, y_old, y_new, color="blue", linewidth=0.1, alpha=0.1)
    ax[0].set_title("Difference")
    ax[0].set_ylabel(y_label)

    # Plot the difference value
    ax[1].set_title("Difference value")
    ax[1].scatter(x, y_old - y_new, color="blue", s=0.5, edgecolors="none")
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_label)

    plt.tight_layout()

    # Save the figure
    if save_fig:
        plt.savefig(save_fig)

    # Show the figure
    if show_fig:
        plt.show()

    # Close the figure
    plt.close()
