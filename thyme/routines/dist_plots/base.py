from thyme.routines.parity_plots.setup import tabcolors
import numpy as np
import logging
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def base_line_hist_axs(axs, data, label, legend, filename, lims=[None, None],
                       scatter_skip=100):
    """
    """

    plot_d = data.reshape([-1])

    include_id = set(np.arange(plot_d.shape[0])[::scatter_skip])
    sort_id = np.argsort(plot_d)
    n = (np.min([len(sort_id), 100]))
    include_id.update(sort_id[:n])
    include_id.update(sort_id[-n:])
    include_id = np.array(list(include_id), dtype=int)

    # newlims = np.copy(lims)
    # dmin = np.min(plot_d)
    # dmax = np.max(plot_d)
    # if newlims[0] is None:
    #     newlims[0] = dmin
    # if newlims[1] is None:
    #     newlims[1] = dmax
    # if newlims[0] < dmin:
    #     newlims[0] = dmin
    # if newlims[1] > dmax:
    #     newlims[1] = dmax

    mean = np.average(plot_d)
    std = np.sqrt(np.var(plot_d))
    logging.info(f"filename {filename} mean {mean:6.3f} std {std:6.3f}")

    # for each config, compute the possible shift
    axs[0].scatter(np.arange(plot_d.shape[0])[include_id],
                   plot_d[include_id],
                   s=5, linewidths=0.5, edgecolors='k',
                   label=f"{legend} mean {mean:6.3f} std {std:6.3f}")
    axs[0].set_xlabel("Step (a.u.)")
    axs[0].set_ylabel(label)
    # axs[0].set_ylim(newlims)
    axs[0].legend()

    base_hist(axs[1], plot_d, label, lims)


def base_line_hist(data, label, filename, lims=[None, None], legend="",
                   scatter_skip=100):
    """
    """

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    base_line_hist_axs(axs, data, label, legend,
                       filename, lims, scatter_skip)
    fig.tight_layout()
    fig.savefig(f"{filename}.png", dpi=300)
    plt.close()
    del fig
    del axs


def base_hist(ax, data, label, lims=[None, None], legend=""):
    """
    """

    if len(data) == 0:
        return

    logging.info(f"plotting basic histogram")
    newlims = np.copy(lims)
    dmin = np.min(data)
    dmax = np.max(data)
    logging.info(f"min {dmin} max {dmax}")
    if newlims[0] is None:
        newlims[0] = dmin
    if newlims[1] is None:
        newlims[1] = dmax
    if newlims[1] > dmax:
        newlims[1] = dmax
    logging.info(f"latest range to plot {newlims}")

    outliers1 = len(np.where(data < newlims[0])[0])
    outliers2 = len(np.where(data > newlims[1])[0])

    text = f"{legend} "
    if outliers1 > 0:
        text += f"{outliers1} pts. < {newlims[0]}"
    if outliers2 > 0:
        text += f" {outliers2} pts. > {newlims[1]}"

    ax.hist(data, range=(newlims[0], newlims[1]), zorder=1, bins=50, label=text, log=True,
            alpha=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel("log(Counts)")
    if len(text) > 0:
        ax.legend()
