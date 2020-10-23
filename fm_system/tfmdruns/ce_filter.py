import logging
logging.basicConfig(filename=f'plot.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
import numpy as np
import sys

from glob import glob

from fmeee.trajectory import PaddedTrajectory
from fmeee.trajectories import Trajectories
from fmeee.routines.dist_plots.base import base_line_hist
from fmeee.routines.parity_plots.base import base_parity

# dictionary = dict(np.load(sys.argv[1], allow_pickle=True))

threshold=0.1
all_trjs = []
min_length = -1
for filename in glob("result*.npz"):
    trjs = Trajectories.from_file(filename, format="padded_mat.npz")
    all_trjs += [trjs]

for spe_list in all_trjs[0].alldata:
    print(spe_list)
    pred = []


    for i, trjs in enumerate(all_trjs):
        pred += [trjs.alldata[spe_list].pred.reshape([-1])]

    nframe = all_trjs[0].alldata[spe_list].pred.shape[0]
    pred = np.vstack(pred)
    pred_var = np.sqrt(np.var(pred, axis=0))
    pred_var = pred_var.reshape([nframe, -1])
    pred_var = np.max(pred_var, axis=1)
    exceed = len(np.where(pred_var>threshold)[0])
    logging.info(f"{exceed} out of {nframe} samples have error > {threshold}")

    sort_id = np.argsort(pred_var)
    all_trjs[0].alldata[spe_list].filter_frames(sort_id[-100:])

all_trjs[0].save("filtered.xyz")
all_trjs[0].save("filtered.poscar")
all_trjs[0].save("filtered_padded_mat.npz")
