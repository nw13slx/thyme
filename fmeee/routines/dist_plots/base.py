import logging
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

from fmeee.routines.parity_plots.setup import tabcolors

def base_line_hist(data, label, filename, lims=[None, None]):
    """
    """
    if lims[0] is None:
        lims[0] = np.min(data)
    if lims[1] is None:
        lims[1] = np.max(data)
    if lims[0] < np.min(data):
        lins[0] = np.min(data)
    if lims[1] > np.max(data):
        lins[1] = np.max(data)

    outliers1 = len(np.where(data<lims[0])[0])
    outliers2 = len(np.where(data<lims[0])[0])

    # for each config, compute the possible shift
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    axs[0].scatter(np.arange(data.shape[0]), data, linewidths=0.5, edgecolors='k')
    axs[0].set_xlabel("Step (a.u.)")
    axs[0].set_ylabel(label)
    axs[0].set_ylim(lims)

    text = ""
    if outliers1 > 0:
        text += f"{outliers1} samples < {lims[0]}"
    if outliers2 > 0:
        text += f"\n{outliers2} samples > {lims[1]}"

    axs[1].hist(data, range=(lims[0], lims[1]), zorder=1, bins=50, label=text, log=True)
    axs[1].set_xlabel(label)
    axs[1].set_ylabel("log(Counts)")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(f"{filename}.png", dpi=300)
    plt.close()
    del fig
    del axs
