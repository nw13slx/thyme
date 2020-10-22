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
        if i == 0:
            ref = trjs.alldata[spe_list].forces.reshape([-1])

    skip = int(np.max([np.ceil(ref.shape[0]/2000), 1]))
    pred = np.vstack(pred)
    pred_var = np.sqrt(np.var(pred, axis=0))
    base_line_hist(pred_var, "std (eV/A)", f"{spe_list}_ce_w_ref", lims=[0, 1],
                   scatter_skip=skip)
