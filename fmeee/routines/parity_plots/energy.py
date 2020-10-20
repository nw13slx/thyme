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
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    axs[0].scatter(energies, pred, linewidths=0.5, edgecolors='k',
               label=f"mae {mae:.2f} rmse {rmse:.2f}")
    axs[0].set_xlabel("DFT energies (eV)")
    axs[0].set_ylabel("Predicted energies (eV)")
    xlims = axs[0].get_xlim()
    for s in [shift[0]]:
        axs[0].plot(xlims, xlims-s, '--', zorder=1, label=f"shift {s}")
    axs[0].set_title(f"{prefix}")
    axs[0].legend()

    axs[1].hist(pred-energies, zorder=1, bins=50,
               label=f"mae {mae:.2f} rmse {rmse:.2f}")
    axs[1].set_xlabel("Predicted energy - DFT energy (eV)")
    axs[1].set_ylabel("Counts")
    axs[1].axvline(x=-shift[0], linestyle='--', color='k', zorder=0)

    fig.tight_layout()
    fig.savefig(f"{prefix}energy.png", dpi=300)
    plt.close()
    del fig
    del axs

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

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    data = []
    for i, trj in enumerate(trajectories.alldata.values()):

        reference = trj.energies
        prediction = getattr(trj, pred_label)
        shift = np.average(reference) - np.average(prediction)

        single_plot(reference, prediction, trj.name, shift=[shift, universal_shift])
        axs[0].scatter(reference, prediction, zorder=2, c=tabcolors[i], label=trj.name,
                   s=8, linewidths=0.5, edgecolors='k')
        data += [prediction-reference]
        xlims = [np.min(reference), np.max(reference)]
        axs[0].plot(xlims, xlims-shift, '--', zorder=1, color=tabcolors[i])
    data = np.hstack(data)
    axs[0].set_xlabel("DFT energies (eV)")
    axs[0].set_ylabel("Predicted energies (eV)")
    xlims = axs[0].get_xlim()
    axs[0].plot(xlims, xlims-universal_shift, '--k', zorder=1)
    axs[0].legend()

    axs[1].hist(data, bins=50)
    axs[1].set_xlabel("Predicted energy - DFT energy (eV)")
    axs[1].set_ylabel("Counts")
    axs[1].axvline(x=-universal_shift, linestyle='--', color='k', zorder=0)

    fig.tight_layout()
    fig.savefig(f"all_energy.png", dpi=300)

