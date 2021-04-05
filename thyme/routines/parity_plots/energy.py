from thyme.routines.parity_plots.setup import tabcolors
from thyme.routines.parity_plots.base import base_parity
import numpy as np
import logging
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def single_plot(energies, pred, prefix, shift=[0]):
    """"""
    base_parity(
        energies,
        pred,
        prefix,
        "energy",
        "DFT energy (eV)",
        "Predicted energy (eV)",
        shift[:1],
    )


def multiple_plots(trajectories, pred_label="pe", prefix=""):

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

        reference = trj.energies.reshape([-1])
        prediction = getattr(trj, pred_label).reshape([-1])
        shift = np.average(reference) - np.average(prediction)

        single_plot(
            reference,
            prediction,
            f"{prefix}{trj.name}",
            shift=[shift, universal_shift],
        )
        axs[0].scatter(
            reference,
            prediction,
            zorder=2,
            c=tabcolors[i % len(tabcolors)],
            label=trj.name,
            s=8,
            linewidths=0.5,
            edgecolors="k",
        )
        data += [prediction - reference]
        xlims = [np.min(reference), np.max(reference)]
        axs[0].plot(
            xlims, xlims - shift, "--", zorder=1, color=tabcolors[i % len(tabcolors)]
        )
    data = np.hstack(data)
    axs[0].set_xlabel("DFT energies (eV)")
    axs[0].set_ylabel("Predicted energies (eV)")
    xlims = axs[0].get_xlim()
    axs[0].plot(xlims, xlims - universal_shift, "--k", zorder=1)
    axs[0].legend()

    axs[1].hist(data, bins=50)
    axs[1].set_xlabel("Predicted energy - DFT energy (eV)")
    axs[1].set_ylabel("Counts")
    axs[1].axvline(x=-universal_shift, linestyle="--", color="k", zorder=0)

    fig.tight_layout()
    fig.savefig(prefix + "all_energy.png", dpi=300)
    del axs
    del fig
