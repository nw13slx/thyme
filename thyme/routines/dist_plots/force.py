from thyme.routines.parity_plots.setup import tabcolors
from thyme.routines.dist_plots.base import base_line_hist
from thyme.trajectory import Trajectory
import numpy as np
import logging
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def multiple_plots(trajectories, prefix=""):
    for i, trj in enumerate(trajectories.alltrjs.values()):
        single_plot(trj, prefix)


def single_plot(trj, prefix=""):
    specs = list(set(list(trj.species)))
    for s in specs:
        f = np.linalg.norm(trj.force, axis=-1)
        ids = np.where(trj.species == s)
        base_line_hist(
            f[:, ids].reshape([-1]),
            "Forces (eV/$\\mathrm{\\AA}$)",
            f"{prefix}{trj.name}_{s}_force_dist",
        )
