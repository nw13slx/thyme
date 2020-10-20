import logging
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

from fmeee.routines.parity_plots.setup import tabcolors

def single_plot(energies, pred, prefix, shift=[0]):
    """
    """

    de = energies-pred-shift[0]
    mae = np.average(np.abs(de))
    rmse = np.sqrt(np.average(de*de))
    logging.info(f"{prefix:30s} mae {mae:5.2f} rmse {rmse:5.2f}")

    # for each config, compute the possible shift
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.scatter(energies, pred, zorder=2, linewidths=0.5, edgecolors='k',
               label=f"mae {mae:.2f} rmse {rmse:.2f}")
    ax.set_xlabel("DFT energies (eV)")
    ax.set_ylabel("Predicted energies (eV)")
    xlims = ax.get_xlim()
    for s in [shift[0]]:
        ax.plot(xlims, xlims-s, '--', zorder=1, label=f"shift {s}")
    ax.set_title(f"{prefix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}energy.png", dpi=300)
    plt.close()
    del fig
    del ax

def multiple_plots(trajectories, pred_label='pe'):

    nframes = 0
    reference_tally = 0
    prediction_tally = 0
    for trj in trajectories.alldata.values():

        reference = trj.energies
        prediction = getattr(trj, pred_label)
        reference_tally += np.sum(reference)
        prediction_tally += np.sum(prediction)
        nframes += len(reference)

    universal_shift = (reference_tally - prediction_tally) / nframes

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, trj in enumerate(trajectories.alldata.values()):

        reference = trj.energies
        prediction = getattr(trj, pred_label)
        shift = np.average(reference) - np.average(prediction)

        single_plot(reference, prediction, trj.name, shift=[shift, universal_shift])
        ax.scatter(reference, prediction, zorder=2, c=tabcolors[i], label=trj.name,
                   s=8, linewidths=0.5, edgecolors='k')
        xlims = [np.min(reference), np.max(reference)]
        ax.plot(xlims, xlims-shift, '--', zorder=1, color=tabcolors[i])

    ax.set_xlabel("DFT energies (eV)")
    ax.set_ylabel("Predicted energies (eV)")
    xlims = ax.get_xlim()
    ax.plot(xlims, xlims-universal_shift, '--k', zorder=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"all_energy.png", dpi=300)
