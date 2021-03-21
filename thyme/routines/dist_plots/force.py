from thyme.routines.parity_plots.setup import tabcolors
from thyme.routines.dist_plots.base import base_line_hist
from thyme.trajectory import PaddedTrajectory, Trajectory
import numpy as np
import logging
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def multiple_plots(trajectories, prefix=""):
    for i, trj in enumerate(trajectories.alldata.values()):
        single_plot(trj, prefix)


def single_plot(trj, prefix=""):
    if isinstance(trj, PaddedTrajectory):
        specs = set(list(trj.symbols.reshape([-1])))
    elif isinstance(trj, Trajectory):
        specs = set(list(trj.species))
    specs = list(specs)
    for s in ["NA", 0, "0"]:
        if s in specs:
            specs.pop(s)
    for s in specs:
        maxf = np.max(trj.forces, axis=-1)
        if isinstance(trj, PaddedTrajectory):
            ids = np.where(trj.symbols == s)
        elif isinstance(trj, Trajectory):
            ids = np.where(trj.species == s)
        base_line_hist(
            maxf[ids],
            "Forces (eV/$\\mathrm{\\AA}$)",
            f"{prefix}{trj.name}_{s}_force_dist",
        )
