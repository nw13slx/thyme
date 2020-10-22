import logging
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

from fmeee.routines.parity_plots.setup import tabcolors

def base_line_hist(data, label, filename):
    """
    """

    # for each config, compute the possible shift
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    axs[0].scatter(np.arange(data.shape[0]), data, linewidths=0.5, edgecolors='k')
    axs[0].set_xlabel("Step (a.u.)")
    axs[0].set_ylabel(label)

    axs[1].hist(data, zorder=1, bins=50)
    axs[1].set_xlabel(label)
    axs[1].set_ylabel("Counts")

    fig.tight_layout()
    fig.savefig(f"{filename}.png", dpi=300)
    plt.close()
    del fig
    del axs
