from thyme.routines.parity_plots.setup import tabcolors
from thyme.routines.dist_plots.base import base_line_hist
import numpy as np
import logging
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def multiple_plots(trajectories, prefix=""):

    for i, trj in enumerate(trajectories.alldata.values()):
        single_plot(trj, prefix)


def single_plot(trj, prefix=""):
    base_line_hist(trj.energies, "Energy (eV)",
                   f"{prefix}{trj.name}_energy_dist")
