from thyme.routines.dist_plots.base import base_hist
from thyme.routines.parity_plots.setup import tabcolors
import numpy as np
import logging
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def base_parity(
    reference, prediction, prefix, postfix, xlabel, ylabel, shift=[0], scatter_skip=1
):
    """"""
    # for each config, compute the possible shift
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))

    base_parity_ax(
        axs, reference, prediction, prefix, postfix, xlabel, ylabel, shift, scatter_skip
    )

    fig.tight_layout()
    fig.savefig(f"{prefix}{postfix}.png", dpi=300)
    plt.close()
    del fig
    del axs


def base_parity_ax(
    axs,
    reference,
    prediction,
    prefix,
    postfix,
    xlabel,
    ylabel,
    shift=[0],
    scatter_skip=1,
):
    """"""

    if len(shift) > 0:
        de = reference - prediction - shift[0]
    else:
        de = reference - prediction
    mae = np.average(np.abs(de))
    rmse = np.sqrt(np.average(de * de))
    logging.info(f"{prefix:30s} mae {mae:5.2f} rmse {rmse:5.2f}")

    # for each config, compute the possible shift
    sc = axs[0].scatter(
        reference[::scatter_skip],
        prediction[::scatter_skip],
        linewidths=0.5,
        edgecolors="k",
        label=f"mae {mae:.2f} rmse {rmse:.2f}",
    )
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    xlims = axs[0].get_xlim()
    for s in shift:
        axs[0].plot(xlims, xlims - s, "--", zorder=1, label=f"shift {s}")
    if len(shift) == 0:
        xlims = axs[0].get_xlim()
        ylims = axs[0].get_ylim()
        lims = (np.max([xlims[0], ylims[0]]), np.min([xlims[1], ylims[1]]))
        axs[0].plot(lims, lims, "--", zorder=1)
    axs[0].set_title(f"{prefix}")
    axs[0].legend()

    base_hist(
        axs[1], prediction - reference, "Prediction - Reference", lims=[None, None]
    )
    for s in shift:
        axs[1].axvline(x=-s, linestyle="--", color="k", zorder=0)
    if len(shift) == 0:
        axs[1].axvline(x=0, linestyle="--", color="k", zorder=0)
