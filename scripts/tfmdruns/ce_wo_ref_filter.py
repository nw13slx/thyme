from thyme.parsers.extxyz import write
from thyme.routines.parity_plots.base import base_parity
from thyme.routines.dist_plots.base import base_line_hist_axs
from thyme.trajectories import Trajectories
from thyme.trajectory import PaddedTrajectory
from thyme.filters.error import sort_by_force

from glob import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    filename=f"plot.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())

plt.switch_backend("agg")


# dictionary = dict(np.load(sys.argv[1], allow_pickle=True))

all_trjs = []
min_length = -1
for filename in glob("result*.npz"):
    trjs = Trajectories.from_file(
        filename, format="padded_mat.npz", preserve_order=True
    )
    all_trjs += [trjs]


for spe_list in all_trjs[0].alltrjs:
    pred = []
    nframe = all_trjs[0].alltrjs[spe_list].pred.shape[0]
    species = np.array(all_trjs[0].alltrjs[spe_list].species, dtype=str)
    for i, trjs in enumerate(all_trjs):
        pred += [trjs.alltrjs[spe_list].pred.reshape([-1])]

    pred = np.vstack(pred)
    pred_var = np.sqrt(np.var(pred, axis=0))

    # take the largest element out
    pred_var = pred_var.reshape([nframe, -1, 3])
    # all_pred_var = np.max(pred_var, axis=-1)

    trj = all_trjs[0].alltrjs[spe_list]
    trj.pred_var = pred_var
    trj.per_frame_attrs += ["pred_var"]
    sort_id, maxf = sort_by_force(trj, "pred_var", "Au")
    trj.include_frames(sort_id[-200:])
    write(f"filter_{spe_list}.xyz", trj)
