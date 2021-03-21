from thyme.routines.parity_plots.setup import tabcolors
from thyme.routines.dist_plots.base import base_line_hist
import numpy as np
import logging
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def multiple_plots(trajectories, label="energies", prefix="", xlabel="Energy (eV)"):
    for i, trj in enumerate(trajectories.alldata.values()):
        single_plot(trj, label, prefix, xlabel)


def single_plot_energy(trj, prefix=""):
    base_line_hist(trj.energies, "Energy (eV)", f"{prefix}{trj.name}_energy_dist")


def single_plot(trj, label="energies", prefix="", xlabel="Energy (eV)"):
    item = getattr(trj, label, None)
    if item is not None:
        base_line_hist(item, xlabel, f"{prefix}{trj.name}_{label}_dist")
