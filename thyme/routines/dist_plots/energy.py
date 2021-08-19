from thyme.routines.parity_plots.setup import tabcolors
from thyme.routines.dist_plots.base import base_line_hist
import numpy as np
import logging
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def multiple_plots(trajectories, label="total_energy", prefix="", xlabel="Energy (eV)", norm=False):
    for name, trj in trajectories.alltrjs.items():
        single_plot(trj, label, prefix+"_"+name, xlabel)


def single_plot(trj, label="total_energy", prefix="", xlabel="Energy (eV)", norm=False):
    item = getattr(trj, label, None)
    if item is not None:
        if norm:
            base_line_hist(item/trj.natoms, xlabel, f"{prefix}_{label}_dist")
        else:
            base_line_hist(item, xlabel, f"{prefix}_{label}_dist")
