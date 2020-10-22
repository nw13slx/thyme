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

pred = []
ref = []
for i, trjs in enumerate(all_trjs):
    pred += [[]]
for spe_list in all_trjs[0].alldata:
    for i, trjs in enumerate(all_trjs):
        pred[i] += [trjs.alldata[spe_list].pred.reshape([-1])]
        if i == 0:
            ref += [trjs.alldata[spe_list].forces.reshape([-1])]
        zero_counts = len(np.where(pred[i][-1]==0)[0])
        counts = pred[i][-1].shape[0]
        logging.info(f" saving {spe_list} in {counts} and {zero_counts} zero entries")

for i, trjs in enumerate(all_trjs):
    pred[i] = np.hstack( pred[i])
ref = np.hstack(ref)

pred = np.vstack(pred)
pred_var = np.sqrt(np.var(pred, axis=0))
base_line_hist(pred_var, "std (eV/A)", "ce_w_ref", lims=[None, 1])
for i, trjs in enumerate(all_trjs):
    error = pred[i]-ref
    base_line_hist(error, "Error (eV/A)", f"e_w_ref_{i}", lims=[-1, 1])
    base_parity(np.abs(error), pred[i], "parity_e_ce", i, "True Error (eV/A)",
                "Predicted Error (eV/A)", shift=[])
