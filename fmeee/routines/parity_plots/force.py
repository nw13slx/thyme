import colorsys
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
plt.switch_backend('agg')
import numpy as np

from fmeee.routines.parity_plots.setup import tabcolors
from fmeee.utils.atomic_symbols import species_to_idgroups

def single_plot(forces, pred, prefix, symbol):
    """
    """

    logging.info(f"plot {prefix}")
    species, idgroups = species_to_idgroups(symbol)

    layer = int(np.ceil(len(species)/2))
    fig, axs = plt.subplots(layer, 2, figsize=(6.8, 2.5*layer))
    fig_hist, axs_hist = plt.subplots(layer, 2, figsize=(6.8, 2.5*layer))
    if layer == 1:
        axsf = axs
        axsf_hist = axs_hist
    else:
        axsf = []
        axsf_hist = []
        for i in range(layer):
            for j in range(2):
                axsf += [axs[i, j]]
                axsf_hist += [axs[i, j]]

    for iele, element in enumerate(species):

        reference = forces[:, idgroups[iele], :]
        prediction = pred[:, idgroups[iele], :]

        df = reference-prediction
        mae = np.average(np.abs(df))
        rmse = np.sqrt(np.average(df*df))

        axsf[iele].scatter(reference, prediction, zorder=2, linewidths=0.5, edgecolors='k',
                           label=f"{element} mae {mae:.2f} rmse {rmse:.2f}",
                           c=tabcolors[iele])
        axsf_hist[iele].hist(prediction-reference, zorder=2, edgecolors='k',
                           label=f"{element} mae {mae:.2f} rmse {rmse:.2f}",
                           c=tabcolors[iele])
        logging.info(f"    {element:2s} mae {mae:5.2f} rmse {rmse:5.2f}")
        xlims = axsf[iele].get_xlim()
        axsf[iele].plot(xlims, xlims, '--', zorder=1, color=tabcolors[iele])
        axsf[iele].legend()
        axsf_hist[iele].axvline(x=0, linestyle='--', color=tabcolors[iele])
        axsf_hist[iele].legend()

        if iele%2 == 0:
            axsf[iele].set_ylabel("Predicted forces (eV)")
            axsf_hist[iele].set_ylabel("Counts")

    axsf[-1].set_xlabel("DFT forces (eV/$\\mathrm{\\AA}$)")
    axsf[-2].set_xlabel("DFT forces (eV/$\\mathrm{\\AA}$)")
    axsf_hist[-1].set_xlabel("Predicted force - DFT force (eV)")
    axsf_hist[-2].set_xlabel("Predicted force - DFT force (eV)")

    fig.tight_layout()
    fig.savefig(f"{prefix}force.png", dpi=300)
    fig_hist.tight_layout()
    fig_hist.savefig(f"{prefix}force_hist.png", dpi=300)
    plt.close()
    del fig
    del axs
    del fig_hist
    del axs_hist

def multiple_plots(trajectories, pred_label='pred'):

    for trj in trajectories.alldata.values():
        single_plot(trj.forces, getattr(trj, pred_label), trj.name, trj.species)

