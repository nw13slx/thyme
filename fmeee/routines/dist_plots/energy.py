import logging
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

from fmeee.routines.dist_plots.base import base_line_hist
from fmeee.routines.parity_plots.setup import tabcolors

def multiple_plots(trajectories, prefix=""):

    for i, trj in enumerate(trajectories.alldata.values()):
        single_plot(trj, prefix)

def single_plot(trj, prefix=""):
        base_line_hist(trj.energies, "Energy (eV)", f"{prefix}{trj.name}_energy_dist")
